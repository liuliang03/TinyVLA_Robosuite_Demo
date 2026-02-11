"""
动作空间映射器

将 TinyVLA 模型的输出（7D 末端执行器动作）转换为 robosuite 兼容的动作格式。
支持多种控制器类型的映射，包括完整的 IK 求解器。
"""

from typing import Dict, Tuple, Optional, Union
import numpy as np
from abc import ABC, abstractmethod

# 导入 IK 求解器
try:
    from src.mapping.ik_solver import (
        IKSolverBase,
        OptimizationIKSolver,
        JacobianIKSolver,
        PyBulletIKSolver,
        create_ik_solver,
    )
except ImportError:
    from mapping.ik_solver import (
        IKSolverBase,
        OptimizationIKSolver,
        JacobianIKSolver,
        PyBulletIKSolver,
        create_ik_solver,
    )


class BaseActionMapper(ABC):
    """动作映射器基类"""

    @abstractmethod
    def map(self, vla_action: np.ndarray, obs: Dict) -> np.ndarray:
        """
        将 VLA 动作映射到环境动作

        Args:
            vla_action: VLA 输出的动作向量
            obs: 当前观察

        Returns:
            环境动作向量
        """
        pass


class ActionMapper(BaseActionMapper):
    """
    动作空间映射器（支持完整 IK 求解）

    将 TinyVLA 的 7D 输出 [x, y, z, rx, ry, rz, gripper] 映射到 robosuite 动作空间。

    Attributes:
        controller_type: 控制器类型
        action_dim: 动作维度
        workspace_bounds: 工作空间边界
        action_scale: 动作缩放因子
        ik_solver: IK 求解器（用于关节位置控制）
        use_ik: 是否使用 IK 求解器
    """

    def __init__(
        self,
        controller_type: str = "BASIC",
        action_dim: int = 7,
        workspace_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        action_scale: float = 0.1,
        position_bounds: Tuple[float, float] = (-0.5, 0.5),
        rotation_bounds: Tuple[float, float] = (-1.0, 1.0),
        gripper_open: float = 1.0,
        gripper_close: float = -1.0,
        use_ik: bool = True,
        ik_method: str = "optimization",
        robot_name: str = "Panda",
        ik_config: Optional[Dict] = None,
        verbose: bool = False,
    ):
        """
        初始化动作映射器

        Args:
            controller_type: 控制器类型（BASIC, OSC_POSE, OSC_POSITION, JOINT_POSITION）
            action_dim: 动作维度（通常为 7）
            workspace_bounds: 工作空间边界 {x: (min, max), y: (min, max), z: (min, max)}
            action_scale: 动作缩放因子
            position_bounds: 位置边界（用于归一化）
            rotation_bounds: 旋转边界
            gripper_open: 夹爪打开值
            gripper_close: 夹爪闭合值
            use_ik: 是否使用 IK 求解器
            ik_method: IK 求解方法 ("optimization", "jacobian", "pybullet")
            robot_name: 机器人名称
            ik_config: IK 求解器配置
            verbose: 是否打印调试信息
        """
        self.controller_type = controller_type
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.position_bounds = position_bounds
        self.rotation_bounds = rotation_bounds
        self.gripper_open = gripper_open
        self.gripper_close = gripper_close
        self.use_ik = use_ik
        self.verbose = verbose

        # 设置默认工作空间边界
        if workspace_bounds is None:
            self.workspace_bounds = {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.8),
            }
        else:
            self.workspace_bounds = workspace_bounds

        # 初始化 IK 求解器
        self.ik_solver: Optional[IKSolverBase] = None
        if use_ik and controller_type in ["BASIC", "JOINT_POSITION"]:
            self.ik_solver = create_ik_solver(
                method=ik_method,
                robot_name=robot_name,
                **(ik_config or {}),
            )
            if self.verbose:
                print(f"✓ IK 求解器已初始化: {ik_method}")

    def map(self, vla_action: np.ndarray, obs: Dict) -> np.ndarray:
        """
        将 VLA 动作映射到环境动作

        根据控制器类型选择映射策略：
        - BASIC: 使用关节位置控制（需要 IK）
        - OSC_POSE: 直接使用末端执行器位姿
        - OSC_POSITION: 使用末端执行器位置 + 固定姿态
        - JOINT_POSITION: 直接关节位置控制

        Args:
            vla_action: VLA 输出的 7D 动作 [x, y, z, rx, ry, rz, gripper]
            obs: 当前观察（用于 IK 和状态查询）

        Returns:
            环境动作向量
        """
        if vla_action.shape[0] != self.action_dim:
            raise ValueError(f"期望动作维度 {self.action_dim}，得到 {vla_action.shape[0]}")

        # 根据控制器类型选择映射方法
        if self.controller_type == "BASIC":
            # BASIC 控制器使用关节位置，需要 IK 求解
            return self._map_to_joint_position(vla_action, obs)
        elif self.controller_type == "OSC_POSE":
            # OSC_POSE 直接使用末端执行器位姿
            return self._map_to_osc_pose(vla_action)
        elif self.controller_type == "OSC_POSITION":
            # OSC_POSITION 只使用位置，固定旋转
            return self._map_to_osc_position(vla_action)
        elif self.controller_type == "JOINT_POSITION":
            # 直接关节位置控制
            return self._map_to_joint_position(vla_action, obs)
        else:
            raise ValueError(f"不支持的控制器类型: {self.controller_type}")

    def _map_to_joint_position(self, vla_action: np.ndarray, obs: Dict) -> np.ndarray:
        """
        映射到关节位置（使用完整的 IK 求解）

        使用 IK 求解器将末端执行器位姿转换为关节角度。

        Args:
            vla_action: VLA 输出 [x, y, z, rx, ry, rz, gripper]
            obs: 当前观察

        Returns:
            关节位置动作向量
        """
        # 获取当前关节位置
        if "joint_positions" in obs:
            current_joints = obs["joint_positions"]
        else:
            # 默认 7 自由度
            current_joints = np.zeros(7)

        # VLA 输出通常是相对动作
        # 我们需要计算目标末端执行器位姿

        # 方法 1: 使用 IK 求解器（推荐）
        if self.use_ik and self.ik_solver is not None:
            return self._solve_ik(vla_action, current_joints)

        # 方法 2: 简化映射（备选）
        else:
            return self._simplified_mapping(vla_action, current_joints)

    def _solve_ik(self, vla_action: np.ndarray, current_joints: np.ndarray) -> np.ndarray:
        """
        使用 IK 求解器计算关节角度

        Args:
            vla_action: VLA 动作
            current_joints: 当前关节角度

        Returns:
            目标关节角度
        """
        # 计算当前末端执行器位姿
        current_ee_pose = self.ik_solver.forward_kinematics(current_joints)

        # 计算目标位姿（当前位姿 + VLA 动作）
        target_pos = current_ee_pose[:3] + vla_action[:3] * self.action_scale
        target_rot = current_ee_pose[3:6] + vla_action[3:6] * 0.1  # 旋转缩放较小

        # 限制在工作空间范围内
        target_pos[0] = np.clip(target_pos[0], self.workspace_bounds["x"][0], self.workspace_bounds["x"][1])
        target_pos[1] = np.clip(target_pos[1], self.workspace_bounds["y"][0], self.workspace_bounds["y"][1])
        target_pos[2] = np.clip(target_pos[2], self.workspace_bounds["z"][0], self.workspace_bounds["z"][1])

        # 使用 IK 求解器
        target_joints = self.ik_solver.solve(
            target_pos=target_pos,
            target_rot=target_rot,
            current_joints=current_joints,
        )

        # 添加夹爪动作
        gripper_action = np.clip(vla_action[6], -1.0, 1.0)
        joint_action = np.append(target_joints[:6], gripper_action)  # 前6个关节 + 夹爪

        return joint_action

    def _simplified_mapping(self, vla_action: np.ndarray, current_joints: np.ndarray) -> np.ndarray:
        """
        简化映射（不使用 IK）

        作为 IK 求解失败时的备选方案。

        Args:
            vla_action: VLA 动作
            current_joints: 当前关节角度

        Returns:
            关节动作
        """
        # 位置部分 (x, y, z)
        target_pos = vla_action[:3] * self.action_scale

        # 夹爪动作
        gripper_action = vla_action[6]

        # 简化映射：将末端执行器位移映射为关节偏移
        joint_action = current_joints.copy()

        # 将位置变化映射到主要关节（简化的启发式）
        joint_action[0] += target_pos[0] * 0.5  # 基座旋转
        joint_action[1] += target_pos[1] * 0.5  # 肩部
        joint_action[2] += target_pos[2] * 0.3  # 肘部
        joint_action[3] += (target_pos[0] + target_pos[1]) * 0.2  # 手腕

        # 夹爪直接使用
        if len(joint_action) > 6:
            joint_action[6] = gripper_action
        else:
            joint_action = np.append(joint_action[:6], gripper_action)

        # 限制关节范围
        joint_action = np.clip(joint_action, -1.0, 1.0)

        if self.verbose:
            print("⚠️ 使用简化映射（IK 未启用或失败）")

        return joint_action

    def _map_to_osc_pose(self, vla_action: np.ndarray) -> np.ndarray:
        """
        映射到 OSC_POSE 控制器

        OSC_POSE 直接控制末端执行器的 6D 位姿 (x, y, z, rx, ry, rz) + 夹爪

        Args:
            vla_action: VLA 输出 [x, y, z, rx, ry, rz, gripper]

        Returns:
            OSC_POSE 动作向量 [x, y, z, rx, ry, rz, gripper]
        """
        action = vla_action.copy()

        # 缩放到工作空间
        action[:3] = self._scale_position(action[:3])

        # 旋转角度归一化
        action[3:6] = self._process_rotation(action[3:6])

        # 夹爪动作限制在 [-1, 1]
        action[6] = np.clip(action[6], -1.0, 1.0)

        return action

    def _map_to_osc_position(self, vla_action: np.ndarray) -> np.ndarray:
        """
        映射到 OSC_POSITION 控制器

        OSC_POSITION 只控制 3D 位置，使用固定姿态

        Args:
            vla_action: VLA 输出 [x, y, z, rx, ry, rz, gripper]

        Returns:
            OSC_POSITION 动作向量 [x, y, z, gripper]
        """
        # 提取位置和夹爪
        position = self._scale_position(vla_action[:3])
        gripper = np.clip(vla_action[6], -1.0, 1.0)

        # OSC_POSITION 通常使用 4D 动作 [x, y, z, gripper]
        action = np.concatenate([position, [gripper]])

        return action

    def _scale_position(self, position: np.ndarray) -> np.ndarray:
        """
        缩放位置到工作空间

        Args:
            position: 位置向量 [x, y, z]

        Returns:
            缩放后的位置
        """
        scaled = position.copy()

        # 应用缩放因子
        scaled = scaled * self.action_scale

        # 限制在工作空间范围内
        scaled[0] = np.clip(scaled[0], self.workspace_bounds["x"][0], self.workspace_bounds["x"][1])
        scaled[1] = np.clip(scaled[1], self.workspace_bounds["y"][0], self.workspace_bounds["y"][1])
        scaled[2] = np.clip(scaled[2], self.workspace_bounds["z"][0], self.workspace_bounds["z"][1])

        return scaled

    def _process_rotation(self, rotation: np.ndarray) -> np.ndarray:
        """
        处理旋转角度

        Args:
            rotation: 旋转角度 [rx, ry, rz]

        Returns:
            处理后的旋转角度
        """
        # 归一化到 [-1, 1] 或 [-pi, pi]
        processed = rotation.copy()

        # 限制旋转范围
        processed = np.clip(processed, self.rotation_bounds[0], self.rotation_bounds[1])

        return processed

    def get_ik_stats(self) -> Optional[Dict]:
        """获取 IK 求解统计信息"""
        if self.ik_solver is not None and hasattr(self.ik_solver, 'get_stats'):
            return self.ik_solver.get_stats()
        return None


class MockVLAActionMapper(BaseActionMapper):
    """
    模拟 VLA 动作映射器

    用于测试和开发，生成随机或启发式动作代替真实的 VLA 输出。
    """

    def __init__(
        self,
        action_dim: int = 7,
        strategy: str = "random",
        target_height: float = 0.3,
    ):
        """
        初始化模拟映射器

        Args:
            action_dim: 动作维度
            strategy: 策略类型 ("random", "lift", "center")
            target_height: 目标高度（用于 lift 策略）
        """
        self.action_dim = action_dim
        self.strategy = strategy
        self.target_height = target_height
        self.step_count = 0

    def map(self, vla_action: np.ndarray, obs: Dict) -> np.ndarray:
        """
        生成模拟动作

        Args:
            vla_action: 忽略（模拟器不使用真实 VLA 输出）
            obs: 当前观察

        Returns:
            模拟动作向量
        """
        self.step_count += 1

        if self.strategy == "random":
            return self._random_action()
        elif self.strategy == "lift":
            return self._lift_action(obs)
        elif self.strategy == "center":
            return self._center_action(obs)
        else:
            raise ValueError(f"未知策略: {self.strategy}")

    def _random_action(self) -> np.ndarray:
        """生成随机动作"""
        return np.random.uniform(-0.1, 0.1, self.action_dim)

    def _lift_action(self, obs: Dict) -> np.ndarray:
        """
        生成提升动作（启发式）

        简单的启发式策略：先向下移动，然后抓取，然后向上提升
        """
        # 阶段 1: 向下移动 (步 0-100)
        if self.step_count < 100:
            return np.array([0, 0, -0.05, 0, 0, 0, 0])
        # 阶段 2: 闭合夹爪 (步 100-150)
        elif self.step_count < 150:
            return np.array([0, 0, 0, 0, 0, 0, 1.0])
        # 阶段 3: 向上提升 (步 150+)
        else:
            return np.array([0, 0, 0.05, 0, 0, 0, 1.0])

    def _center_action(self, obs: Dict) -> np.ndarray:
        """
        生成向中心移动的动作
        """
        # 简单的向中心移动策略
        return np.array([0.01, 0.01, 0, 0, 0, 0, 0])


def create_action_mapper(
    controller_type: str = "BASIC",
    use_mock: bool = False,
    use_ik: bool = True,
    ik_method: str = "optimization",
    robot_name: str = "Panda",
    **kwargs
) -> BaseActionMapper:
    """
    创建动作映射器工厂函数

    Args:
        controller_type: 控制器类型
        use_mock: 是否使用模拟映射器（用于测试）
        use_ik: 是否使用 IK 求解器
        ik_method: IK 求解方法
        robot_name: 机器人名称
        **kwargs: 其他参数

    Returns:
        动作映射器实例
    """
    if use_mock:
        return MockVLAActionMapper(**kwargs)
    else:
        return ActionMapper(
            controller_type=controller_type,
            use_ik=use_ik,
            ik_method=ik_method,
            robot_name=robot_name,
            **kwargs
        )
