#!/usr/bin/env python3
"""
测试模型配置和加载

验证：
1. 配置文件加载
2. 模型路径检测
3. 模拟模式运行
4. VLA wrapper 创建
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_config_loader():
    """测试配置加载器"""
    print("=" * 60)
    print("测试 1: 配置加载器")
    print("=" * 60)

    try:
        from src.utils.config_loader import load_config

        config = load_config()

        print(f"\n✓ 配置加载成功")
        print(f"  - 默认模型: {config.get('vlm_backbone.model_name')}")
        print(f"  - 模型 ID: {config.get('vlm_backbone.huggingface_id')}")
        print(f"  - 设备: {config.get_device()}")
        print(f"  - 数据类型: {config.get_dtype()}")
        print(f"  - 缓存目录: {config.get_cache_dir()}")

        # 列出模型
        models = config.list_models()
        print(f"\n  可用模型:")
        for variant, info in models.items():
            print(f"    - {variant}: {info['name']} ({info['params']})")
            print(f"      ID: {info['model_id']}")
            print(f"      路径: {info['local_path']}")

        return True

    except Exception as e:
        print(f"  ✗ 配置加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_path_detection():
    """测试模型路径检测"""
    print("\n" + "=" * 60)
    print("测试 2: 模型路径检测")
    print("=" * 60)

    try:
        from src.utils.config_loader import load_config

        config = load_config()

        # 检查各个变体的模型路径
        variants = ["small", "base", "large"]
        found_models = []

        for variant in variants:
            model_path = config.get_model_path(variant)
            if model_path:
                print(f"  ✓ {variant} 模型存在: {model_path}")
                found_models.append(variant)
            else:
                print(f"  ○ {variant} 模型未找到")

        if found_models:
            print(f"\n  找到 {len(found_models)} 个模型")
            return True
        else:
            print(f"\n  ⚠️ 未找到任何本地模型")
            print(f"  提示: 请参考 models/README.md 下载模型")
            return False

    except Exception as e:
        print(f"  ✗ 路径检测失败: {e}")
        return False


def test_vla_wrapper_mock():
    """测试 VLA wrapper 模拟模式"""
    print("\n" + "=" * 60)
    print("测试 3: VLA Wrapper 模拟模式")
    print("=" * 60)

    try:
        from src.core.vla_wrapper import create_vla_wrapper
        import numpy as np

        # 创建模拟 VLA
        print("\n  创建模拟 VLA wrapper...")
        vla = create_vla_wrapper(use_mock=True)

        # 测试推理
        print("  测试推理...")
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        instruction = "Pick up the cube"

        action = vla.forward(image, instruction)

        print(f"  ✓ 推理成功")
        print(f"  动作形状: {action.shape}")
        print(f"  动作值: {action}")

        return True

    except Exception as e:
        print(f"  ✗ VLA wrapper 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_controller_with_mock():
    """测试控制器使用模拟模型"""
    print("\n" + "=" * 60)
    print("测试 4: 控制器（模拟模式）")
    print("=" * 60)

    try:
        from src.controllers.language_controller import create_controller

        # 创建控制器（使用模拟 VLA）
        print("\n  创建控制器...")
        controller = create_controller(
            task_name="Lift",
            robot_name="Panda",
            use_mock_vla=True,
            use_mock_mapper=True,  # 简化映射
            max_steps=10,
            verbose=False,
        )

        # 执行指令
        print("  执行指令...")
        result = controller.execute_instruction(
            instruction="Pick up the cube",
            render=False,
        )

        print(f"\n  ✓ 控制器测试成功")
        print(f"  状态: {result.state.value}")
        print(f"  步数: {result.steps}")
        print(f"  最终奖励: {result.final_reward:.4f}")

        controller.env.close()

        return True

    except Exception as e:
        print(f"  ✗ 控制器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_real_model_loading():
    """测试真实模型加载（如果可用）"""
    print("\n" + "=" * 60)
    print("测试 5: 真实模型加载（可选）")
    print("=" * 60)

    try:
        from src.utils.config_loader import load_config
        from src.core.vla_wrapper import create_vla_wrapper

        config = load_config()

        # 检查是否有本地模型
        model_path = config.get_model_path("small")

        if not model_path:
            print("  ○ 跳过：未找到本地模型")
            print("  提示: 下载模型后可启用真实模型模式")
            return None

        print(f"\n  尝试加载模型: {model_path}")

        # 尝试创建真实模型
        vla = create_vla_wrapper(
            model_path=model_path,
            use_mock=False,
            device=config.get_device(),
        )

        print("  ✓ 真实模型加载成功!")
        print(f"  模型路径: {model_path}")
        print(f"  设备: {vla.device}")

        return True

    except Exception as e:
        print(f"  ○ 真实模型加载失败: {e}")
        print("  这可能是因为:")
        print("    1. 模型文件不完整")
        print("    2. 缺少依赖库")
        print("    3. 模型格式不兼容")
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("TinyVLA 配置和模型加载测试")
    print("=" * 60)

    results = {}

    # 运行测试
    results["配置加载"] = test_config_loader()
    results["路径检测"] = test_model_path_detection()
    results["VLA模拟模式"] = test_vla_wrapper_mock()
    results["控制器测试"] = test_controller_with_mock()
    results["真实模型加载"] = test_real_model_loading()

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    for test_name, result in results.items():
        if result is True:
            status = "✓ 通过"
        elif result is False:
            status = "✗ 失败"
        else:
            status = "○ 跳过"

        print(f"  {status}: {test_name}")

    print("=" * 60)

    # 统计
    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)

    print(f"\n总计: {passed} 通过, {failed} 失败, {skipped} 跳过")

    if failed == 0 and passed >= 3:
        print("\n✓ 核心功能测试通过！系统可以正常运行（模拟模式）。")
        if skipped > 0:
            print("\n提示: 下载真实模型后可启用完整功能。")
        return 0
    else:
        print("\n⚠️ 部分测试失败，请检查配置。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
