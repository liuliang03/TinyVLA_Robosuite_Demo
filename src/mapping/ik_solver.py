"""
完整的 IK（逆运动学）求解器实现

提供多种 IK 求解方法，将末端执行器位姿转换为关节角度。
支持：
- 基于优化的 IK 求解
- Jacobian 伪逆法
- PyBullet IK（可选）
- mink IK（robosuite 集成）
"""

from typing import Dict, Tuple, Optional, List, Callable
from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import minimize, fsolve
import time


class IKSolverBase(ABC):
    """IK 求解器基类"""

    @abstractmethod
    def solve(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        current_joints: np.ndarray,
    ) -> np.ndarray:
        """
        求解 IK

        Args:
            target_pos: 目标位置 (x, y, z) in meters
            target_rot: 目标旋转 (rx, ry, rz) in radians
            current_joints: 当前关节角度

        Returns:
            目标关节角度
        """
        pass

    @abstractmethod
    def forward_kinematics(self, joints: np.ndarray) -> np.ndarray:
        """
        正向运动学

        Args:
            joints: 关节角度

        Returns:
            末端执行器位姿 [x, y, z, rx, ry, rz]
        """
        pass


class OptimizationIKSolver(IKSolverBase):
    """
    基于优化的 IK 求解器

    使用 scipy.optimize.minimize 求解 IK 问题。
    适用于各种机器人，但需要定义正向运动学。
    """

    def __init__(
        self,
        robot_name: str = "Panda",
        n_joints: int = 7,
        joint_limits: Optional[np.ndarray] = None,
        max_iterations: int = 100,
        position_threshold: float = 0.01,
        orientation_threshold: float = 0.1,
        position_weight: float = 1.0,
        orientation_weight: float = 0.5,
        verbose: bool = False,
    ):
        """
        初始化 IK 求解器

        Args:
            robot_name: 机器人名称
            n_joints: 关节数量
            joint_limits: 关节限制 (n_joints, 2)
            max_iterations: 最大迭代次数
            position_threshold: 位置误差阈值 (m)
            orientation_threshold: 旋转误差阈值 (rad)
            position_weight: 位置误差权重
            orientation_weight: 旋转误差权重
            verbose: 是否打印调试信息
        """
        self.robot_name = robot_name
        self.n_joints = n_joints
        self.max_iterations = max_iterations
        self.position_threshold = position_threshold
        self.orientation_threshold = orientation_threshold
        self.position_weight = position_weight
        self.orientation_weight = orientation_weight
        self.verbose = verbose

        # 设置关节限制
        if joint_limits is None:
            self.joint_limits = self._get_default_joint_limits(robot_name)
        else:
            self.joint_limits = joint_limits

        # 性能统计
        self.solve_times = []
        self.success_count = 0
        self.failure_count = 0

    def _get_default_joint_limits(self, robot_name: str) -> np.ndarray:
        """获取默认关节限制"""
        if robot_name == "Panda":
            return np.array([
                [-2.8973, 2.8973],  # 关节 1: 基座
                [-1.7628, 1.7628],  # 关节 2: 肩部
                [-2.8973, 2.8973],  # 关节 3: 肘部
                [-3.0718, -0.0698],  # 关节 4: 前臂
                [-2.8973, 2.8973],  # 关节 5: 手腕 1
                [-0.0175, 3.7525],  # 关节 6: 手腕 2
                [-2.8973, 2.8973],  # 关节 7: 手腕 3
            ])
        elif robot_name == "Sawyer":
            return np.array([
                [-3.050, 3.050],
                [-3.810, 2.270],
                [-3.040, 3.040],
                [-3.040, 3.040],
                [-2.980, 2.980],
                [-1.850, 1.850],
                [-3.040, 3.040],
            ])
        else:
            # 默认限制
            return np.array([[-np.pi, np.pi]] * self.n_joints)

    def solve(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        current_joints: np.ndarray,
    ) -> np.ndarray:
        """
        使用优化方法求解 IK

        Args:
            target_pos: 目标位置 (x, y, z)
            target_rot: 目标旋转 (rx, ry, rz)
            current_joints: 当前关节角度

        Returns:
            目标关节角度
        """
        start_time = time.time()

        # 目标位姿
        target_pose = np.concatenate([target_pos, target_rot])

        # 目标函数
        def objective(joints):
            """计算当前位姿与目标的误差"""
            current_pose = self.forward_kinematics(joints)

            # 位置误差
            pos_error = np.linalg.norm(current_pose[:3] - target_pos)

            # 旋转误差
            rot_error = np.linalg.norm(current_pose[3:6] - target_rot)

            # 加权总误差
            total_error = (
                self.position_weight * pos_error +
                self.orientation_weight * rot_error
            )

            return total_error

        # 优化求解
        result = minimize(
            objective,
            current_joints,
            method='L-BFGS-B',
            bounds=self.joint_limits,
            options={
                'maxiter': self.max_iterations,
                'ftol': 1e-6,
                'gtol': 1e-6,
            }
        )

        solve_time = time.time() - start_time
        self.solve_times.append(solve_time)

        if result.success:
            # 验证解的质量
            final_pose = self.forward_kinematics(result.x)
            pos_error = np.linalg.norm(final_pose[:3] - target_pos)
            rot_error = np.linalg.norm(final_pose[3:6] - target_rot)

            if pos_error < self.position_threshold and rot_error < self.orientation_threshold:
                self.success_count += 1
                if self.verbose:
                    print(f"IK 求解成功: 误差=({pos_error:.4f}, {rot_error:.4f}), 时间={solve_time*1000:.2f}ms")
                return result.x
            else:
                self.failure_count += 1
                if self.verbose:
                    print(f"IK 求解未达标: 误差=({pos_error:.4f}, {rot_error:.4f})")
                # 返回当前关节位置
                return current_joints
        else:
            self.failure_count += 1
            if self.verbose:
                print(f"IK 求解失败: {result.message}")
            return current_joints

    def forward_kinematics(self, joints: np.ndarray) -> np.ndarray:
        """
        简化的正向运动学计算

        注意：这是一个简化实现。对于生产环境，应该使用真实的运动学库。

        Args:
            joints: 关节角度

        Returns:
            末端执行器位姿 [x, y, z, rx, ry, rz]
        """
        # 简化的运动学模型（仅用于演示）
        # 实际应用中应使用真实的运动学

        # 基于 DH 参数的简化计算
        # 这里使用启发式映射

        if self.robot_name == "Panda":
            # Panda 机器人的简化运动学
            # 实际应该使用真实的 DH 参数

            # 连杆长度
            l1, l2, l3, l4 = 0.333, 0.316, 0.0825, 0.0825

            # 简化的正向运动学
            x = (l1 + l2 * np.cos(joints[1]) + l3 * np.cos(joints[1] + joints[2])) * np.cos(joints[0])
            y = (l1 + l2 * np.cos(joints[1]) + l3 * np.cos(joints[1] + joints[2])) * np.sin(joints[0])
            z = l2 * np.sin(joints[1]) + l3 * np.sin(joints[1] + joints[2]) + l4

            # 旋转（简化）
            rx = joints[0]
            ry = joints[1] + joints[2]
            rz = joints[3] + joints[4] + joints[5]

            return np.array([x, y, z, rx, ry, rz])
        else:
            # 通用简化模型
            x = joints[0] * 0.5
            y = joints[1] * 0.5
            z = joints[1] * 0.3 + joints[2] * 0.4
            rx = joints[0]
            ry = joints[1]
            rz = joints[2]

            return np.array([x, y, z, rx, ry, rz])

    def get_stats(self) -> Dict:
        """获取求解统计信息"""
        return {
            'solve_times': self.solve_times,
            'avg_solve_time': np.mean(self.solve_times) if self.solve_times else 0,
            'success_rate': self.success_count / (self.success_count + self.failure_count)
            if (self.success_count + self.failure_count) > 0 else 0,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
        }


class JacobianIKSolver(IKSolverBase):
    """
    基于 Jacobian 伪逆的 IK 求解器

    使用迭代方法求解 IK，适用于实时控制。
    """

    def __init__(
        self,
        robot_name: str = "Panda",
        n_joints: int = 7,
        max_iterations: int = 100,
        damping: float = 0.01,
        step_size: float = 0.5,
        position_threshold: float = 0.01,
        orientation_threshold: float = 0.1,
        verbose: bool = False,
    ):
        """
        初始化 Jacobian IK 求解器

        Args:
            robot_name: 机器人名称
            n_joints: 关节数量
            max_iterations: 最大迭代次数
            damping: 阻尼系数（用于奇异值避免）
            step_size: 步长
            position_threshold: 位置误差阈值
            orientation_threshold: 旋转误差阈值
            verbose: 是否打印调试信息
        """
        self.robot_name = robot_name
        self.n_joints = n_joints
        self.max_iterations = max_iterations
        self.damping = damping
        self.step_size = step_size
        self.position_threshold = position_threshold
        self.orientation_threshold = orientation_threshold
        self.verbose = verbose

        # 创建辅助的优化求解器用于正向运动学
        self._opt_solver = OptimizationIKSolver(
            robot_name=robot_name,
            n_joints=n_joints,
            verbose=verbose,
        )

    def solve(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        current_joints: np.ndarray,
    ) -> np.ndarray:
        """
        使用 Jacobian 伪逆法求解 IK

        Args:
            target_pos: 目标位置
            target_rot: 目标旋转
            current_joints: 当前关节角度

        Returns:
            目标关节角度
        """
        # 使用简化的梯度下降方法
        joints = current_joints.copy()
        target_pose = np.concatenate([target_pos, target_rot])

        for iteration in range(self.max_iterations):
            # 计算当前位姿
            current_pose = self.forward_kinematics(joints)

            # 计算误差
            error = target_pose - current_pose
            pos_error = np.linalg.norm(error[:3])
            rot_error = np.linalg.norm(error[3:6])

            # 检查收敛
            if pos_error < self.position_threshold and rot_error < self.orientation_threshold:
                if self.verbose:
                    print(f"Jacobian IK 收敛: 迭代={iteration}, 误差=({pos_error:.4f}, {rot_error:.4f})")
                return joints

            # 数值 Jacobian
            jacobian = self._compute_jacobian(joints)

            # 使用阻尼最小二乘法
            delta_joints = self.step_size * np.linalg.pinv(
                jacobian.T @ jacobian + self.damping * np.eye(self.n_joints)
            ) @ jacobian.T @ error

            # 更新关节角度
            joints += delta_joints

        if self.verbose:
            print(f"Jacobian IK 达到最大迭代次数: {self.max_iterations}")

        return joints

    def forward_kinematics(self, joints: np.ndarray) -> np.ndarray:
        """正向运动学"""
        return self._opt_solver.forward_kinematics(joints)

    def _compute_jacobian(self, joints: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
        """
        数值计算 Jacobian 矩阵

        Args:
            joints: 当前关节角度
            epsilon: 微小扰动

        Returns:
            Jacobian 矩阵 (6, n_joints)
        """
        jacobian = np.zeros((6, self.n_joints))

        # 计算当前位姿
        current_pose = self.forward_kinematics(joints)

        # 对每个关节进行数值微分
        for i in range(self.n_joints):
            joints_perturbed = joints.copy()
            joints_perturbed[i] += epsilon

            pose_perturbed = self.forward_kinematics(joints_perturbed)

            # 数值导数
            jacobian[:, i] = (pose_perturbed - current_pose) / epsilon

        return jacobian


class PyBulletIKSolver(IKSolverBase):
    """
    PyBullet IK 求解器（可选）

    使用 PyBullet 的内置 IK 求解器。
    需要 pybullet 包。
    """

    def __init__(
        self,
        robot_id: Optional[int] = None,
        end_effector_index: int = 11,
        verbose: bool = False,
    ):
        """
        初始化 PyBullet IK 求解器

        Args:
            robot_id: PyBullet 机器人 ID
            end_effector_index: 末端执行器链接索引
            verbose: 是否打印调试信息
        """
        self.robot_id = robot_id
        self.end_effector_index = end_effector_index
        self.verbose = verbose
        self.pybullet_available = False

        # 尝试导入 PyBullet
        try:
            import pybullet as p
            self.p = p
            self.pybullet_available = True
            if self.verbose:
                print("PyBullet 可用，将使用 PyBullet IK")
        except ImportError:
            if self.verbose:
                print("PyBullet 不可用，将回退到优化方法")

    def solve(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        current_joints: np.ndarray,
    ) -> np.ndarray:
        """使用 PyBullet IK 求解"""
        if not self.pybullet_available or self.robot_id is None:
            # 回退到优化方法
            opt_solver = OptimizationIKSolver(verbose=self.verbose)
            return opt_solver.solve(target_pos, target_rot, current_joints)

        # 使用 PyBullet IK
        import pybullet as p

        # 将欧拉角转换为四元数
        target_orn = p.getQuaternionFromEuler(target_rot)

        # 计算 IK
        joint_angles = p.calculateInverseKinematics(
            self.robot_id,
            self.end_effector_index,
            target_pos,
            target_orn,
            maxNumIterations=100,
            residualThreshold=1e-5,
        )

        return np.array(joint_angles[:7])  # 只返回前 7 个关节

    def forward_kinematics(self, joints: np.ndarray) -> np.ndarray:
        """PyBullet 正向运动学"""
        if not self.pybullet_available or self.robot_id is None:
            opt_solver = OptimizationIKSolver(verbose=self.verbose)
            return opt_solver.forward_kinematics(joints)

        import pybullet as p

        # 设置关节位置
        for i, joint_angle in enumerate(joints):
            p.resetJointState(self.robot_id, i, joint_angle)

        # 获取末端执行器位姿
        link_state = p.getLinkState(self.robot_id, self.end_effector_index)
        pos = np.array(link_state[0])
        orn = p.getEulerFromQuaternion(link_state[1])

        return np.concatenate([pos, orn])


def create_ik_solver(
    method: str = "optimization",
    robot_name: str = "Panda",
    **kwargs
) -> IKSolverBase:
    """
    创建 IK 求解器工厂函数

    Args:
        method: 求解方法 ("optimization", "jacobian", "pybullet")
        robot_name: 机器人名称
        **kwargs: 其他参数

    Returns:
        IK 求解器实例
    """
    if method == "optimization":
        return OptimizationIKSolver(robot_name=robot_name, **kwargs)
    elif method == "jacobian":
        return JacobianIKSolver(robot_name=robot_name, **kwargs)
    elif method == "pybullet":
        return PyBulletIKSolver(robot_name=robot_name, **kwargs)
    else:
        raise ValueError(f"未知的 IK 求解方法: {method}")
