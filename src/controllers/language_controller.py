"""
语言指令控制器

协调 TinyVLA 模型、动作映射器和 robosuite 环境，
实现从语言指令到机器人动作的完整控制流程。
"""

from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
from dataclasses import dataclass
from enum import Enum

try:
    from src.core.vla_wrapper import BaseVLAWrapper, create_vla_wrapper
    from src.environments.robosuite_env import RobosuiteEnvironment
    from src.mapping.action_mapper import BaseActionMapper, create_action_mapper
    from src.vision.image_preprocessor import ImagePreprocessor
except ImportError:
    # 处理相对导入问题
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.vla_wrapper import BaseVLAWrapper, create_vla_wrapper
    from environments.robosuite_env import RobosuiteEnvironment
    from mapping.action_mapper import BaseActionMapper, create_action_mapper
    from vision.image_preprocessor import ImagePreprocessor


class ControllerState(Enum):
    """控制器状态"""
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class ControlResult:
    """
    控制结果

    Attributes:
        success: 是否成功完成任务
        steps: 执行步数
        final_reward: 最终奖励
        trajectory: 执行轨迹
        state: 最终状态
        message: 状态消息
    """
    success: bool
    steps: int
    final_reward: float
    trajectory: Dict[str, List]
    state: ControllerState
    message: str


class LanguageController:
    """
    语言指令控制器

    高层控制器，协调 VLA 模型、动作映射器和仿真环境。

    Attributes:
        vla_wrapper: VLA 模型封装
        action_mapper: 动作映射器
        env: 仿真环境
        image_preprocessor: 图像预处理器
        max_steps: 最大步数
        state: 当前状态
    """

    def __init__(
        self,
        vla_wrapper: BaseVLAWrapper,
        action_mapper: BaseActionMapper,
        env: RobosuiteEnvironment,
        image_preprocessor: Optional[ImagePreprocessor] = None,
        max_steps: int = 500,
        reward_threshold: float = 1.0,
        verbose: bool = True,
    ):
        """
        初始化语言指令控制器

        Args:
            vla_wrapper: VLA 模型封装
            action_mapper: 动作映射器
            env: 仿真环境
            image_preprocessor: 图像预处理器（可选，默认创建新的）
            max_steps: 最大执行步数
            reward_threshold: 成功奖励阈值
            verbose: 是否打印详细信息
        """
        self.vla = vla_wrapper
        self.mapper = action_mapper
        self.env = env
        self.max_steps = max_steps
        self.reward_threshold = reward_threshold
        self.verbose = verbose

        # 创建图像预处理器
        if image_preprocessor is None:
            self.preprocessor = ImagePreprocessor(
                target_size=(224, 224),
                normalize=True,
            )
        else:
            self.preprocessor = image_preprocessor

        # 状态
        self.state = ControllerState.IDLE
        self.current_step = 0

    def execute_instruction(
        self,
        instruction: str,
        render: bool = False,
        save_trajectory: bool = True,
        callback: Optional[Callable] = None,
    ) -> ControlResult:
        """
        执行语言指令

        Args:
            instruction: 语言指令（如 "抓起红色的方块"）
            render: 是否渲染环境
            save_trajectory: 是否保存轨迹
            callback: 每步回调函数 callback(step, obs, action, reward)

        Returns:
            控制结果
        """
        # 重置环境
        obs = self.env.reset()
        self.state = ControllerState.RUNNING
        self.current_step = 0

        # 初始化轨迹
        trajectory = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "images": [],
            "vlm_outputs": [],
        }

        total_reward = 0.0

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"执行指令: {instruction}")
            print(f"{'='*60}\n")

        # 控制循环
        for step in range(self.max_steps):
            self.current_step = step + 1

            # 1. 获取当前图像
            image = obs["image"]

            # 2. 预处理图像
            processed_image = self.preprocessor.preprocess(image)

            # 3. VLA 推理
            vla_action = self.vla.forward(
                image=image,
                instruction=instruction
            )

            # 4. 动作映射
            env_action = self.mapper.map(vla_action, obs)

            # 5. 执行动作
            obs, reward, done, info = self.env.step(env_action)
            total_reward += reward

            # 6. 记录轨迹
            if save_trajectory:
                trajectory["observations"].append(obs)
                trajectory["actions"].append(env_action)
                trajectory["rewards"].append(reward)
                trajectory["images"].append(image)
                trajectory["vlm_outputs"].append(vla_action)

            # 7. 渲染
            if render:
                self.env.render()

            # 8. 回调
            if callback is not None:
                callback(step, obs, env_action, reward)

            # 9. 打印进度
            if self.verbose and step % 50 == 0:
                print(f"  步数 {step}: 奖励={reward:.4f}, 总奖励={total_reward:.4f}")

            # 10. 检查完成
            if done:
                if total_reward >= self.reward_threshold:
                    self.state = ControllerState.SUCCESS
                    if self.verbose:
                        print(f"\n✓ 任务成功完成! 步数: {step + 1}, 总奖励: {total_reward:.4f}")
                else:
                    self.state = ControllerState.FAILED
                    if self.verbose:
                        print(f"\n✗ 任务未达标。步数: {step + 1}, 总奖励: {total_reward:.4f}")
                break

        # 超时检查
        if self.current_step >= self.max_steps and self.state == ControllerState.RUNNING:
            self.state = ControllerState.TIMEOUT
            if self.verbose:
                print(f"\n⏱️ 达到最大步数 {self.max_steps}，总奖励: {total_reward:.4f}")

        # 构建结果
        result = ControlResult(
            success=(self.state == ControllerState.SUCCESS),
            steps=self.current_step,
            final_reward=total_reward,
            trajectory=trajectory,
            state=self.state,
            message=f"状态: {self.state.value}",
        )

        return result

    def execute_multiple_instructions(
        self,
        instructions: List[str],
        **kwargs
    ) -> List[ControlResult]:
        """
        执行多个语言指令

        Args:
            instructions: 指令列表
            **kwargs: 传递给 execute_instruction 的参数

        Returns:
            结果列表
        """
        results = []

        for i, instruction in enumerate(instructions):
            if self.verbose:
                print(f"\n[{i+1}/{len(instructions)}] 执行指令: {instruction}")

            result = self.execute_instruction(instruction, **kwargs)
            results.append(result)

        # 打印总结
        if self.verbose:
            success_count = sum(1 for r in results if r.success)
            print(f"\n{'='*60}")
            print(f"完成 {success_count}/{len(instructions)} 个指令")
            print(f"{'='*60}\n")

        return results

    def reset(self):
        """重置控制器状态"""
        self.state = ControllerState.IDLE
        self.current_step = 0

    def get_state(self) -> ControllerState:
        """获取当前状态"""
        return self.state


class ControllerBuilder:
    """
    控制器构建器

    用于方便地构建完整的语言指令控制器。
    """

    def __init__(self):
        """初始化构建器"""
        self._vla_wrapper = None
        self._action_mapper = None
        self._env = None
        self._preprocessor = None
        self._max_steps = 500
        self._reward_threshold = 1.0
        self._verbose = True

    def with_vla(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        use_mock: bool = False,
    ) -> "ControllerBuilder":
        """设置 VLA 模型"""
        self._vla_wrapper = create_vla_wrapper(
            model_path=model_path,
            device=device,
            use_mock=use_mock,
        )
        return self

    def with_mapper(
        self,
        controller_type: str = "BASIC",
        use_mock: bool = False,
    ) -> "ControllerBuilder":
        """设置动作映射器"""
        self._action_mapper = create_action_mapper(
            controller_type=controller_type,
            use_mock=use_mock,
        )
        return self

    def with_environment(
        self,
        task_name: str = "Lift",
        robot_name: str = "Panda",
        controller_type: str = "BASIC",
    ) -> "ControllerBuilder":
        """设置仿真环境"""
        from src.environments.robosuite_env import create_env
        self._env = create_env(
            task_name=task_name,
            robot_name=robot_name,
            controller_type=controller_type,
        )
        return self

    def with_preprocessor(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
    ) -> "ControllerBuilder":
        """设置图像预处理器"""
        self._preprocessor = ImagePreprocessor(
            target_size=target_size,
            normalize=normalize,
        )
        return self

    def with_max_steps(self, max_steps: int) -> "ControllerBuilder":
        """设置最大步数"""
        self._max_steps = max_steps
        return self

    def with_verbose(self, verbose: bool) -> "ControllerBuilder":
        """设置详细输出"""
        self._verbose = verbose
        return self

    def build(self) -> LanguageController:
        """
        构建控制器

        Returns:
            语言指令控制器实例
        """
        # 验证必需组件
        if self._vla_wrapper is None:
            raise ValueError("必须设置 VLA 模型（使用 with_vla 方法）")
        if self._action_mapper is None:
            raise ValueError("必须设置动作映射器（使用 with_mapper 方法）")
        if self._env is None:
            raise ValueError("必须设置环境（使用 with_environment 方法）")

        # 创建控制器
        controller = LanguageController(
            vla_wrapper=self._vla_wrapper,
            action_mapper=self._action_mapper,
            env=self._env,
            image_preprocessor=self._preprocessor,
            max_steps=self._max_steps,
            verbose=self._verbose,
        )

        return controller


# 便捷函数
def create_controller(
    task_name: str = "Lift",
    robot_name: str = "Panda",
    model_path: Optional[str] = None,
    device: str = "cpu",
    use_mock_vla: bool = True,
    use_mock_mapper: bool = False,
    **kwargs
) -> LanguageController:
    """
    便捷的控制器创建函数

    Args:
        task_name: 任务名称
        robot_name: 机器人名称
        model_path: VLA 模型路径
        device: 计算设备
        use_mock_vla: 是否使用模拟 VLA 模型
        use_mock_mapper: 是否使用模拟动作映射器
        **kwargs: 其他参数

    Returns:
        语言指令控制器实例
    """
    return (
        ControllerBuilder()
        .with_vla(model_path=model_path, device=device, use_mock=use_mock_vla)
        .with_mapper(controller_type="BASIC", use_mock=use_mock_mapper)
        .with_environment(task_name=task_name, robot_name=robot_name, controller_type="BASIC")
        .with_max_steps(kwargs.get("max_steps", 500))
        .with_verbose(kwargs.get("verbose", True))
        .build()
    )
