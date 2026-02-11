#!/usr/bin/env python3
"""
高级演示：使用 IK 求解器和 TinyVLA 模型

展示：
1. 如何加载真实 TinyVLA 模型
2. 如何使用 IK 求解器进行精确控制
3. 完整的控制流程示例

用法:
    # 使用模拟模式（默认）
    python demos/advanced_demo.py

    # 使用真实模型（需要提供模型路径）
    python demos/advanced_demo.py --model-path /path/to/tinyvla

    # 使用不同的 IK 方法
    python demos/advanced_demo.py --ik-method jacobian

    # 禁用 IK（使用简化映射）
    python demos/advanced_demo.py --no-ik
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np

from src.controllers.language_controller import create_controller
from src.core.vla_wrapper import create_vla_wrapper
from src.mapping.action_mapper import create_action_mapper
from src.environments.robosuite_env import create_env


def print_banner():
    """打印横幅"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║        TinyVLA + robosuite 高级演示系统                    ║
    ║                                                           ║
    ║        支持真实模型加载和完整 IK 求解                      ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def demo_with_real_model(model_path: str, ik_method: str, max_steps: int):
    """
    使用真实 TinyVLA 模型的演示

    Args:
        model_path: 模型路径
        ik_method: IK 方法
        max_steps: 最大步数
    """
    print("\n" + "="*60)
    print("真实模型演示")
    print("="*60)

    print(f"\n配置:")
    print(f"  模型路径: {model_path}")
    print(f"  IK 方法: {ik_method}")
    print(f"  最大步数: {max_steps}")

    # 创建控制器
    print("\n正在初始化控制器...")
    controller = create_controller(
        task_name="Lift",
        robot_name="Panda",
        model_path=model_path,
        use_mock_vla=False,  # 使用真实模型
        use_mock_mapper=False,  # 使用 IK
        max_steps=max_steps,
        verbose=True,
    )

    # 执行指令
    instruction = "Pick up the cube"
    print(f"\n执行指令: {instruction}")

    result = controller.execute_instruction(
        instruction=instruction,
        render=False,
        save_trajectory=True,
    )

    # 打印结果
    print("\n" + "="*60)
    print("执行结果:")
    print("="*60)
    print(f"  状态: {result.state.value}")
    print(f"  成功: {result.success}")
    print(f"  步数: {result.steps}/{max_steps}")
    print(f"  最终奖励: {result.final_reward:.4f}")

    # IK 统计
    ik_stats = controller.mapper.get_ik_stats()
    if ik_stats:
        print("\nIK 求解统计:")
        print(f"  平均求解时间: {ik_stats['avg_solve_time']*1000:.2f} ms")
        print(f"  成功率: {ik_stats['success_rate']*100:.1f}%")
        print(f"  成功次数: {ik_stats['success_count']}")
        print(f"  失败次数: {ik_stats['failure_count']}")

    controller.env.close()


def demo_with_ik_comparison(ik_methods: list, max_steps: int):
    """
    比较不同 IK 方法的性能

    Args:
        ik_methods: IK 方法列表
        max_steps: 最大步数
    """
    print("\n" + "="*60)
    print("IK 方法性能对比")
    print("="*60)

    instruction = "Pick up the cube"

    results = {}

    for ik_method in ik_methods:
        print(f"\n{'='*60}")
        print(f"测试 IK 方法: {ik_method}")
        print(f"{'='*60}")

        try:
            controller = create_controller(
                task_name="Lift",
                robot_name="Panda",
                use_mock_vla=True,
                use_mock_mapper=False,
                ik_method=ik_method,
                max_steps=max_steps,
                verbose=False,
            )

            result = controller.execute_instruction(
                instruction=instruction,
                render=False,
            )

            results[ik_method] = {
                "success": result.success,
                "steps": result.steps,
                "reward": result.final_reward,
            }

            # 获取 IK 统计
            ik_stats = controller.mapper.get_ik_stats()
            if ik_stats:
                results[ik_method]["avg_time"] = ik_stats['avg_solve_time'] * 1000
                results[ik_method]["success_rate"] = ik_stats['success_rate']

            controller.env.close()

            print(f"  成功: {result.success}")
            print(f"  步数: {result.steps}")
            print(f"  奖励: {result.final_reward:.4f}")

        except Exception as e:
            print(f"  ✗ 失败: {e}")
            results[ik_method] = None

    # 打印对比表
    print("\n" + "="*60)
    print("对比结果:")
    print("="*60)
    print(f"{'方法':<15} {'成功':<8} {'步数':<8} {'奖励':<10} {'平均时间':<12} {'成功率':<10}")
    print("-" * 60)

    for method, result in results.items():
        if result is not None:
            print(f"{method:<15} {str(result['success']):<8} {result['steps']:<8} "
                  f"{result['reward']:<10.4f} "
                  f"{result.get('avg_time', 0):<12.2f} "
                  f"{result.get('success_rate', 0)*100:<10.1f}%")
        else:
            print(f"{method:<15} {'失败':<38}")


def demo_ik_vs_no_ik(max_steps: int):
    """
    对比使用 IK 和不使用 IK 的效果

    Args:
        max_steps: 最大步数
    """
    print("\n" + "="*60)
    print("IK vs 简化映射对比")
    print("="*60)

    instruction = "Pick up the red cube"

    # 配置
    configs = [
        ("简化映射 (无 IK)", {"use_mock_mapper": True, "use_ik": False}),
        ("优化 IK", {"use_mock_mapper": False, "use_ik": True, "ik_method": "optimization"}),
        ("Jacobian IK", {"use_mock_mapper": False, "use_ik": True, "ik_method": "jacobian"}),
    ]

    results = {}

    for name, config in configs:
        print(f"\n测试: {name}")
        print("-" * 60)

        try:
            controller = create_controller(
                task_name="Lift",
                robot_name="Panda",
                use_mock_vla=True,
                max_steps=max_steps,
                verbose=False,
                **config,
            )

            result = controller.execute_instruction(
                instruction=instruction,
                render=False,
            )

            results[name] = {
                "success": result.success,
                "steps": result.steps,
                "reward": result.final_reward,
            }

            print(f"  成功: {result.success}")
            print(f"  步数: {result.steps}")
            print(f"  奖励: {result.final_reward:.4f}")

            # IK 统计
            if "IK" in name:
                ik_stats = controller.mapper.get_ik_stats()
                if ik_stats:
                    print(f"  平均时间: {ik_stats['avg_solve_time']*1000:.2f} ms")
                    print(f"  成功率: {ik_stats['success_rate']*100:.1f}%")

            controller.env.close()

        except Exception as e:
            print(f"  ✗ 失败: {e}")
            results[name] = None

    # 对比
    print("\n" + "="*60)
    print("对比结果:")
    print("="*60)
    print(f"{'方法':<20} {'成功':<8} {'步数':<8} {'奖励':<10}")
    print("-" * 60)

    for name, result in results.items():
        if result is not None:
            print(f"{name:<20} {str(result['success']):<8} {result['steps']:<8} "
                  f"{result['reward']:<10.4f}")
        else:
            print(f"{name:<20} {'失败':<28}")


def demo_model_loading_guide():
    """打印模型加载指南"""
    print("\n" + "="*60)
    print("真实 TinyVLA 模型加载指南")
    print("="*60)

    guide = """
1. 下载 TinyVLA 模型权重

   方法 1 - 从 HuggingFace:
   ```bash
   # 安装 huggingface-hub
   pip install huggingface-hub

   # 下载模型
   huggingface-cli download JayceWen/tinyvla
   ```

   方法 2 - 从 GitHub:
   ```bash
   git clone https://github.com/JayceWen/tinyvla
   cd tinyvla
   ```

2. 安装依赖

   ```bash
   pip install transformers>=4.30.0
   pip install torch torchvision
   pip install opencv-python
   ```

3. 使用模型

   ```python
   from src.controllers.language_controller import create_controller

   controller = create_controller(
       task_name="Lift",
       robot_name="Panda",
       model_path="path/to/tinyvla/model",  # 模型路径
       use_mock_vla=False,  # 使用真实模型
       use_mock_mapper=False,  # 使用 IK
       max_steps=500,
   )

   result = controller.execute_instruction(
       instruction="抓起红色的方块",
       render=True,
   )
   ```

4. 配置选项

   - model_path: 模型权重路径
   - device: "cpu" 或 "cuda"
   - dtype: "float32", "float16", "bfloat16"
   - use_diffusion: 是否使用扩散解码
   - diffusion_steps: 扩散步数 (默认 20)

5. IK 求解器配置

   - ik_method: "optimization", "jacobian", "pybullet"
   - max_iterations: 最大迭代次数
   - position_threshold: 位置误差阈值
   - orientation_threshold: 旋转误差阈值

6. 性能优化

   - 使用 GPU: device="cuda"
   - 混合精度: dtype="float16"
   - 模型量化: 启用 INT8 量化
   - 批处理推理
    """
    print(guide)


def main():
    """主函数"""
    print_banner()

    parser = argparse.ArgumentParser(
        description="TinyVLA + robosuite 高级演示",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--model-path", type=str, default=None,
                        help="TinyVLA 模型路径")
    parser.add_argument("--ik-method", type=str, default="optimization",
                        choices=["optimization", "jacobian", "pybullet"],
                        help="IK 求解方法")
    parser.add_argument("--no-ik", action="store_true",
                        help="禁用 IK 求解器（使用简化映射）")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="最大执行步数")
    parser.add_argument("--comparison", action="store_true",
                        help="运行 IK 方法对比")
    parser.add_argument("--ik-vs-no-ik", action="store_true",
                        help="对比 IK 和简化映射")
    parser.add_argument("--guide", action="store_true",
                        help="显示模型加载指南")
    parser.add_argument("--quiet", action="store_true",
                        help="静默模式")

    args = parser.parse_args()

    if args.guide:
        demo_model_loading_guide()
        return

    if args.comparison:
        demo_with_ik_comparison(
            ik_methods=["optimization", "jacobian", "pybullet"],
            max_steps=args.max_steps
        )
    elif args.ik_vs_no_ik:
        demo_ik_vs_no_ik(max_steps=args.max_steps)
    elif args.model_path:
        demo_with_real_model(
            model_path=args.model_path,
            ik_method=args.ik_method,
            max_steps=args.max_steps
        )
    else:
        # 默认演示
        print("\n使用默认配置（模拟模式 + IK）")
        demo_with_real_model(
            model_path=None,  # 使用模拟模式
            ik_method=args.ik_method,
            max_steps=50  # 较短步数用于演示
        )


if __name__ == "__main__":
    main()
