"""
测试 robosuite 环境安装和基本功能
"""

import sys


def test_import_robosuite():
    """测试 robosuite 导入"""
    try:
        import robosuite
        print("✓ robosuite 导入成功")
        print(f"  版本: {robosuite.__version__ if hasattr(robosuite, '__version__') else 'Unknown'}")
        return True
    except ImportError as e:
        print(f"✗ robosuite 导入失败: {e}")
        return False


def test_environment_creation():
    """测试环境创建"""
    try:
        import robosuite

        # 加载控制器配置 - 使用 BASIC composite controller
        # BASIC 控制器内部会使用合适的 part controller（如 JOINT_POSITION）
        controller_configs = robosuite.load_composite_controller_config(controller="BASIC")

        # 测试 Lift 环境
        env = robosuite.make(
            env_name="Lift",
            robots="Panda",
            controller_configs=controller_configs,
            has_renderer=False,
            has_offscreen_renderer=True,
            render_camera="frontview",
        )
        print("✓ Lift 环境创建成功")
        print(f"  观察空间: {env.observation_spec if hasattr(env, 'observation_spec') else 'N/A'}")
        print(f"  动作空间: {env.action_spec if hasattr(env, 'action_spec') else 'N/A'}")
        env.close()

        # 测试 PickPlace 环境
        controller_configs = robosuite.load_composite_controller_config(controller="BASIC")
        env = robosuite.make(
            env_name="PickPlace",
            robots="Panda",
            controller_configs=controller_configs,
            has_renderer=False,
            has_offscreen_renderer=True,
        )
        print("✓ PickPlace 环境创建成功")
        env.close()

        return True
    except Exception as e:
        print(f"✗ 环境创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_reset_step():
    """测试环境重置和执行"""
    try:
        import robosuite
        import numpy as np

        controller_configs = robosuite.load_composite_controller_config(controller="BASIC")
        env = robosuite.make(
            env_name="Lift",
            robots="Panda",
            controller_configs=controller_configs,
            has_renderer=False,
            has_offscreen_renderer=True,
        )

        # 重置环境
        obs = env.reset()
        print("✓ 环境重置成功")
        print(f"  观察键: {list(obs.keys())[:5]}...")

        # 执行随机动作
        action_low, action_high = env.action_spec
        action = np.random.uniform(action_low, action_high)
        obs, reward, done, info = env.step(action)
        print("✓ 动作执行成功")
        print(f"  奖励: {reward:.4f}")
        print(f"  完成: {done}")

        env.close()
        return True
    except Exception as e:
        print(f"✗ 环境执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rendering():
    """测试渲染功能"""
    try:
        import robosuite
        import numpy as np

        controller_configs = robosuite.load_composite_controller_config(controller="BASIC")
        env = robosuite.make(
            env_name="Lift",
            robots="Panda",
            controller_configs=controller_configs,
            has_renderer=False,
            has_offscreen_renderer=True,
            render_camera="frontview",
        )

        obs = env.reset()

        # 获取图像
        image = env.sim.render(camera_name="frontview", width=224, height=224)
        print(f"✓ 图像渲染成功")
        print(f"  图像形状: {image.shape}")
        print(f"  数据类型: {image.dtype}")

        env.close()
        return True
    except Exception as e:
        print(f"✗ 渲染失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("robosuite 环境安装测试")
    print("=" * 60)
    print()

    results = []

    # 测试导入
    print("1. 测试 robosuite 导入...")
    results.append(test_import_robosuite())
    print()

    # 测试环境创建
    print("2. 测试环境创建...")
    results.append(test_environment_creation())
    print()

    # 测试重置和执行
    print("3. 测试环境重置和执行...")
    results.append(test_environment_reset_step())
    print()

    # 测试渲染
    print("4. 测试渲染功能...")
    results.append(test_rendering())
    print()

    # 总结
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"测试结果: {passed}/{total} 通过")
    print("=" * 60)

    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
