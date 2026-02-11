#!/usr/bin/env python3
"""
TinyVLA + robosuite 最小演示程序

展示使用语言指令控制机械臂抓取物体的完整流程。

用法:
    python demos/minimal_demo.py
    python demos/minimal_demo.py --instruction "抓起红色的方块"
    python demos/minimal_demo.py --task PickPlace --instruction "将方块放到目标位置"
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np

try:
    from src.controllers.language_controller import create_controller, ControllerBuilder
    from src.core.vla_wrapper import create_vla_wrapper
    from src.environments.robosuite_env import create_env
    from src.mapping.action_mapper import create_action_mapper
    from src.vision.image_preprocessor import ImagePreprocessor
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装所有依赖: pip install -r requirements.txt")
    sys.exit(1)


def print_banner():
    """打印横幅"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║        TinyVLA + robosuite 机械臂抓取演示系统             ║
    ║                                                           ║
    ║        使用语言指令控制机器人抓取物体                     ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_demo(
    task_name: str = "Lift",
    instruction: str = "Pick up the cube",
    max_steps: int = 500,
    render: bool = False,
    verbose: bool = True,
    model_path: str = None,
    use_real_model: bool = False,
    offscreen_render: bool = False,
):
    """
    运行演示

    Args:
        task_name: 任务名称
        instruction: 语言指令
        max_steps: 最大步数
        render: 是否渲染
        verbose: 是否显示详细信息
        model_path: 模型路径
        use_real_model: 是否使用真实模型
        offscreen_render: 是否使用 offscreen 渲染
    """
    print_banner()

    # 确定模型路径
    if use_real_model and model_path is None:
        # 尝试从配置读取
        from src.utils.config_loader import load_config
        try:
            config = load_config()
            model_path = config.get_model_path("small")
            if model_path and verbose:
                print(f"✓ 从配置读取模型路径: {model_path}")
        except:
            pass

    if verbose:
        print(f"\n配置信息:")
        print(f"  任务: {task_name}")
        print(f"  指令: {instruction}")
        print(f"  最大步数: {max_steps}")
        print(f"  渲染: {render}")
        if model_path:
            print(f"  模型: {model_path}")
        else:
            print(f"  模型: 模拟模式")
        print()

    # 创建控制器
    if verbose:
        print("正在初始化控制器...")

    controller = create_controller(
        task_name=task_name,
        robot_name="Panda",
        model_path=model_path,
        device="cpu",
        use_mock_vla=(model_path is None),  # 如果没有模型路径，使用模拟模式
        use_mock_mapper=False,  # 使用真实的 IK 映射器
        max_steps=max_steps,
        verbose=verbose,
        offscreen_render=offscreen_render,
    )

    if verbose:
        print("✓ 控制器初始化完成\n")

    # 执行指令
    result = controller.execute_instruction(
        instruction=instruction,
        render=render,
        save_trajectory=True,
    )

    # 打印结果
    print("\n" + "="*60)
    print("执行结果:")
    print("="*60)
    print(f"  状态: {result.state.value}")
    print(f"  成功: {'是' if result.success else '否'}")
    print(f"  步数: {result.steps}/{max_steps}")
    print(f"  最终奖励: {result.final_reward:.4f}")
    print(f"  平均奖励: {result.final_reward/max_steps:.4f}")
    print("="*60)

    # 关闭环境
    controller.env.close()

    return result


def run_interactive_demo():
    """运行交互式演示"""
    print_banner()
    print("\n交互式演示模式")
    print("输入 'quit' 退出\n")

    # 创建控制器（复用）
    controller = create_controller(
        task_name="Lift",
        robot_name="Panda",
        use_mock_vla=True,
        use_mock_mapper=True,
        max_steps=200,
        verbose=True,
    )

    while True:
        try:
            instruction = input("\n请输入指令: ").strip()

            if instruction.lower() in ["quit", "exit", "q"]:
                print("退出演示")
                break

            if not instruction:
                continue

            # 执行指令
            result = controller.execute_instruction(
                instruction=instruction,
                render=False,
            )

            print(f"\n结果: {result.state.value}, 奖励: {result.final_reward:.4f}")

        except KeyboardInterrupt:
            print("\n\n退出演示")
            break
        except Exception as e:
            print(f"\n错误: {e}")

    controller.env.close()


def run_batch_demo():
    """运行批量测试演示"""
    print_banner()
    print("\n批量测试模式")
    print("测试多个不同的指令...\n")

    # 定义测试指令
    instructions = [
        "Pick up the cube",
        "Lift the object",
        "Grab the cube",
    ]

    # 创建控制器
    controller = create_controller(
        task_name="Lift",
        robot_name="Panda",
        use_mock_vla=True,
        use_mock_mapper=True,
        max_steps=200,
        verbose=True,
    )

    # 执行批量测试
    results = controller.execute_multiple_instructions(
        instructions=instructions,
        render=False,
    )

    # 统计结果
    print("\n" + "="*60)
    print("批量测试结果:")
    print("="*60)

    for i, (instruction, result) in enumerate(zip(instructions, results)):
        status = "✓" if result.success else "✗"
        print(f"  {status} {instruction}")
        print(f"      步数: {result.steps}, 奖励: {result.final_reward:.4f}")

    success_count = sum(1 for r in results if r.success)
    print(f"\n  成功率: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    print("="*60)

    controller.env.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="TinyVLA + robosuite 机械臂抓取演示"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Lift",
        choices=["Lift", "PickPlace", "Stack"],
        help="任务名称",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Pick up the cube",
        help="语言指令",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="最大步数",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="启用渲染",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="交互式模式",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="批量测试模式",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="模型路径（留空使用模拟模式）",
    )
    parser.add_argument(
        "--use-real-model",
        action="store_true",
        help="使用真实模型（需要先下载）",
    )
    parser.add_argument(
        "--offscreen-render",
        action="store_true",
        help="使用 offscreen 渲染（无需显示服务器）",
    )

    args = parser.parse_args()

    try:
        if args.interactive:
            run_interactive_demo()
        elif args.batch:
            run_batch_demo()
        else:
            run_demo(
                task_name=args.task,
                instruction=args.instruction,
                max_steps=args.max_steps,
                render=args.render,
                verbose=not args.quiet,
                model_path=args.model_path,
                use_real_model=args.use_real_model,
                offscreen_render=args.offscreen_render,
            )
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
