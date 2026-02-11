"""
robosuite 环境包装器

封装 robosuite 环境，提供统一的接口用于 TinyVLA 控制。
功能包括：
- 环境创建和配置
- 观察空间管理
- 动作空间管理
- 图像获取
- 渲染管理
"""

from typing import Dict, Tuple, Optional, List, Any, Union
import numpy as np
import robosuite


class RobosuiteEnvironment:
    """
    robosuite 环境包装器

    封装 robosuite 环境，提供简洁统一的接口。

    Attributes:
        env: robosuite 环境实例
        task_name: 任务名称
        robot_name: 机器人名称
        controller_type: 控制器类型
        observation_space: 观察空间规范
        action_space: 动作空间规范
    """

    # 支持的任务列表
    SUPPORTED_TASKS = [
        "Lift",
        "PickPlace",
        "Stack",
        "NutAssembly",
        "Door",
        "TwoArmLift",
        "TwoArmPegInHole",
    ]

    # 支持的机器人列表
    SUPPORTED_ROBOTS = [
        "Panda",
        "Sawyer",
        "Jaco",
        "Kinova3",
        "IIWA",
        "UR5e",
    ]

    # 支持的控制器类型
    SUPPORTED_CONTROLLERS = [
        "BASIC",           # 基础控制器（JOINT_POSITION）
        "WHOLE_BODY_IK",   # 全身逆运动学
    ]

    def __init__(
        self,
        task_name: str = "Lift",
        robot_name: str = "Panda",
        controller_type: str = "BASIC",
        has_renderer: bool = False,
        has_offscreen_renderer: bool = True,
        render_camera: str = "frontview",
        camera_names: Optional[List[str]] = None,
        image_size: Tuple[int, int] = (224, 224),
        control_freq: int = 20,
        horizon: int = 500,
        **kwargs
    ):
        """
        初始化 robosuite 环境

        Args:
            task_name: 任务名称
            robot_name: 机器人名称
            controller_type: 控制器类型（composite controller）
            has_renderer: 是否启用屏幕渲染
            has_offscreen_renderer: 是否启用离屏渲染
            render_camera: 渲染相机名称
            camera_names: 相机名称列表
            image_size: 图像尺寸 (width, height)
            control_freq: 控制频率 (Hz)
            horizon: 最大步数
            **kwargs: 其他传递给 robosuite.make 的参数
        """
        self.task_name = task_name
        self.robot_name = robot_name
        self.controller_type = controller_type
        self.image_size = image_size
        self.camera_names = camera_names or [render_camera]
        self.render_camera = render_camera

        # 验证参数
        self._validate_params()

        # 加载控制器配置
        self.controller_configs = robosuite.load_composite_controller_config(
            controller=controller_type
        )

        # 创建环境
        self.env = robosuite.make(
            env_name=task_name,
            robots=robot_name,
            controller_configs=self.controller_configs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            camera_names=self.camera_names,
            control_freq=control_freq,
            horizon=horizon,
            **kwargs
        )

        # 获取观察和动作空间
        self.observation_spec = self.env.observation_spec()
        self.action_spec = self.env.action_spec

    def _validate_params(self):
        """验证参数"""
        if self.task_name not in self.SUPPORTED_TASKS:
            raise ValueError(f"不支持的任务: {self.task_name}。支持的任务: {self.SUPPORTED_TASKS}")
        if self.robot_name not in self.SUPPORTED_ROBOTS:
            raise ValueError(f"不支持的机器人: {self.robot_name}。支持的机器人: {self.SUPPORTED_ROBOTS}")
        if self.controller_type not in self.SUPPORTED_CONTROLLERS:
            raise ValueError(f"不支持的控制器: {self.controller_type}。支持的控制器: {self.SUPPORTED_CONTROLLERS}")

    def reset(self) -> Dict[str, np.ndarray]:
        """
        重置环境

        Returns:
            观察字典，包含：
            - image: 相机图像
            - robot_state: 机器人状态（关节位置、速度等）
            - object_state: 物体状态（如果有）
            - gripper_state: 夹爪状态
        """
        raw_obs = self.env.reset()
        return self._process_observation(raw_obs)

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """
        执行动作

        Args:
            action: 动作向量，形状为 action_dim

        Returns:
            tuple: (观察, 奖励, 完成, 信息)
        """
        raw_obs, reward, done, info = self.env.step(action)
        obs = self._process_observation(raw_obs)
        return obs, reward, done, info

    def _process_observation(self, raw_obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        处理原始观察，提取关键信息

        Args:
            raw_obs: 原始观察字典

        Returns:
            处理后的观察字典
        """
        obs = {}

        # 获取图像
        obs["image"] = self.get_image()

        # 机器人状态
        robot_keys = [k for k in raw_obs.keys() if k.startswith("robot0_")]
        for key in robot_keys:
            obs[key] = raw_obs[key]

        # 添加方便访问的键
        if "robot0_joint_pos" in raw_obs:
            obs["joint_positions"] = raw_obs["robot0_joint_pos"]
        if "robot0_joint_vel" in raw_obs:
            obs["joint_velocities"] = raw_obs["robot0_joint_vel"]
        if "robot0_gripper_qpos" in raw_obs:
            obs["gripper_state"] = raw_obs["robot0_gripper_qpos"]

        # 物体状态
        object_keys = [k for k in raw_obs.keys() if "object" in k.lower()]
        for key in object_keys:
            obs[key] = raw_obs[key]

        return obs

    def get_image(
        self,
        camera_name: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> np.ndarray:
        """
        获取相机图像

        Args:
            camera_name: 相机名称（默认使用 render_camera）
            width: 图像宽度（默认使用 image_size[0]）
            height: 图像高度（默认使用 image_size[1]）

        Returns:
            图像数组 (H, W, 3)，uint8
        """
        camera = camera_name or self.render_camera
        w = width or self.image_size[0]
        h = height or self.image_size[1]

        # 渲染图像
        image = self.env.sim.render(camera_name=camera, width=w, height=h)

        # MuJoCo 渲染的图像已经是 RGB 格式
        return image

    def get_images(self) -> Dict[str, np.ndarray]:
        """
        获取所有相机的图像

        Returns:
            字典，键为相机名称，值为图像
        """
        images = {}
        for camera in self.camera_names:
            images[camera] = self.get_image(camera_name=camera)
        return images

    def render(self):
        """渲染环境到屏幕"""
        if hasattr(self.env, "render"):
            self.env.render()

    def close(self):
        """关闭环境"""
        if hasattr(self.env, "close"):
            self.env.close()

    def __del__(self):
        """析构函数，确保环境关闭"""
        self.close()

    @property
    def action_dim(self) -> int:
        """获取动作维度"""
        action_low, action_high = self.action_spec
        return len(action_low)

    @property
    def observation_dim(self) -> int:
        """获取观察维度（不包括图像）"""
        # 计算非图像观察的总维度
        dim = 0
        obs = self.reset()
        for key, value in obs.items():
            if key != "image" and isinstance(value, np.ndarray):
                dim += value.size
        return dim

    def get_action_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取动作边界

        Returns:
            (action_low, action_high)
        """
        return self.action_spec

    def sample_action(self) -> np.ndarray:
        """
        采样随机动作

        Returns:
            随机动作向量
        """
        action_low, action_high = self.action_spec
        return np.random.uniform(action_low, action_high)

    def get_robot_state(self) -> Dict[str, np.ndarray]:
        """
        获取机器人状态

        Returns:
            包含关节位置、速度、夹爪状态等的字典
        """
        obs = self._process_observation(self.env._get_observations())
        state = {}
        for key, value in obs.items():
            if key.startswith("robot0_") or key in ["joint_positions", "joint_velocities", "gripper_state"]:
                state[key] = value
        return state

    def get_object_state(self) -> Dict[str, np.ndarray]:
        """
        获取物体状态

        Returns:
            物体位置、速度等信息
        """
        obs = self._process_observation(self.env._get_observations())
        state = {}
        for key, value in obs.items():
            if "object" in key.lower():
                state[key] = value
        return state


class EnvironmentFactory:
    """
    环境工厂类

    用于创建不同配置的环境实例。
    """

    @staticmethod
    def create_lift_env(**kwargs) -> RobosuiteEnvironment:
        """创建 Lift 任务环境"""
        return RobosuiteEnvironment(task_name="Lift", **kwargs)

    @staticmethod
    def create_pickplace_env(**kwargs) -> RobosuiteEnvironment:
        """创建 PickPlace 任务环境"""
        return RobosuiteEnvironment(task_name="PickPlace", **kwargs)

    @staticmethod
    def create_stack_env(**kwargs) -> RobosuiteEnvironment:
        """创建 Stack 任务环境"""
        return RobosuiteEnvironment(task_name="Stack", **kwargs)

    @staticmethod
    def from_config(config: Dict[str, Any]) -> RobosuiteEnvironment:
        """
        从配置字典创建环境

        Args:
            config: 配置字典，包含 task_name, robot_name 等参数

        Returns:
            环境实例
        """
        return RobosuiteEnvironment(**config)


# 便捷函数
def create_env(
    task_name: str = "Lift",
    robot_name: str = "Panda",
    **kwargs
) -> RobosuiteEnvironment:
    """
    便捷的环境创建函数

    Args:
        task_name: 任务名称
        robot_name: 机器人名称
        **kwargs: 其他参数

    Returns:
        环境实例
    """
    return RobosuiteEnvironment(
        task_name=task_name,
        robot_name=robot_name,
        **kwargs
    )
