#!/usr/bin/env python3
"""
测试 IK 求解器和真实模型集成

验证：
1. IK 求解器功能
2. 动作映射器集成 IK
3. 完整控制流程
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
# matplotlib.pyplot 仅在需要时导入


def test_ik_solver():
    """测试 IK 求解器"""
    from src.mapping.ik_solver import OptimizationIKSolver, JacobianIKSolver, create_ik_solver

    print("="*60)
    print("测试 IK 求解器")
    print("="*60)

    # 创建优化 IK 求解器
    print("\n1. 创建 OptimizationIKSolver...")
    ik_solver = OptimizationIKSolver(
        robot_name="Panda",
        n_joints=7,
        max_iterations=100,
        verbose=True,
    )

    # 测试正向运动学
    print("\n2. 测试正向运动学...")
    test_joints = np.array([0, 0, 0, -1.5, 0, 1.5, 0])
    ee_pose = ik_solver.forward_kinematics(test_joints)
    print(f"   关节角度: {test_joints}")
    print(f"   末端位姿: {ee_pose}")

    # 测试 IK 求解
    print("\n3. 测试 IK 求解...")
    target_pos = np.array([0.3, 0.0, 0.3])
    target_rot = np.array([0, 0, 0])

    solution_joints = ik_solver.solve(
        target_pos=target_pos,
        target_rot=target_rot,
        current_joints=test_joints,
    )

    print(f"   目标位置: {target_pos}")
    print(f"   目标旋转: {target_rot}")
    print(f"   求解关节: {solution_joints}")

    # 验证解的质量
    final_pose = ik_solver.forward_kinematics(solution_joints)
    pos_error = np.linalg.norm(final_pose[:3] - target_pos)
    rot_error = np.linalg.norm(final_pose[3:6] - target_rot)

    print(f"\n   位置误差: {pos_error:.6f} m")
    print(f"   旋转误差: {rot_error:.6f} rad")

    # 获取统计信息
    stats = ik_solver.get_stats()
    print(f"\n4. IK 求解统计:")
    print(f"   平均求解时间: {stats['avg_solve_time']*1000:.2f} ms")
    print(f"   成功率: {stats['success_rate']*100:.1f}%")
    print(f"   成功/失败: {stats['success_count']}/{stats['failure_count']}")

    return ik_solver


def test_action_mapper_with_ik():
    """测试动作映射器集成 IK"""
    from src.mapping.action_mapper import create_action_mapper

    print("\n" + "="*60)
    print("测试动作映射器（带 IK）")
    print("="*60)

    # 创建带 IK 的映射器
    print("\n1. 创建 ActionMapper with IK...")
    mapper = create_action_mapper(
        controller_type="BASIC",
        use_ik=True,
        ik_method="optimization",
        robot_name="Panda",
        verbose=True,
    )

    # 模拟观察
    print("\n2. 模拟观察...")
    obs = {
        "joint_positions": np.array([0, -0.5, 0, -1.5, 0, 1.5, 0]),
        "joint_velocities": np.zeros(7),
        "gripper_state": np.array([0.0, 0.0]),
    }

    # 模拟 VLA 动作
    print("\n3. 模拟 VLA 动作...")
    vla_action = np.array([0.05, 0, -0.05, 0, 0, 0, 1.0])  # 向前、向下、闭合夹爪

    print(f"   VLA 动作: {vla_action}")

    # 映射到环境动作
    print("\n4. 映射到环境动作...")
    env_action = mapper.map(vla_action, obs)

    print(f"   环境动作: {env_action}")

    # 验证动作维度
    print(f"\n5. 验证:")
    print(f"   输入维度: {vla_action.shape[0]}")
    print(f"   输出维度: {env_action.shape[0]}")
    print(f"   动作范围: [{env_action.min():.3f}, {env_action.max():.3f}]")

    # 获取 IK 统计
    ik_stats = mapper.get_ik_stats()
    if ik_stats:
        print(f"\n6. IK 统计:")
        print(f"   平均求解时间: {ik_stats['avg_solve_time']*1000:.2f} ms")
        print(f"   成功率: {ik_stats['success_rate']*100:.1f}%")

    return mapper


def test_vla_model():
    """测试 TinyVLA 模型封装"""
    from src.core.vla_wrapper import create_vla_wrapper

    print("\n" + "="*60)
    print("测试 TinyVLA 模型封装")
    print("="*60)

    # 创建 VLA 模型
    print("\n1. 创建 TinyVLA 模型...")
    vla = create_vla_wrapper(
        model_path=None,  # 使用模拟模式
        device="cpu",
        use_mock=True,
    )

    # 模拟图像和指令
    print("\n2. 测试推理...")
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    instruction = "Pick up the cube"

    print(f"   指令: {instruction}")
    print(f"   图像形状: {image.shape}")

    # 推理
    action = vla.forward(image, instruction)

    print(f"\n3. 输出动作: {action}")
    print(f"   动作类型: {type(action)}")
    print(f"   动作形状: {action.shape}")

    return vla


def test_full_pipeline():
    """测试完整控制流程"""
    from src.controllers.language_controller import create_controller

    print("\n" + "="*60)
    print("测试完整控制流程（带 IK）")
    print("="*60)

    # 创建控制器
    print("\n1. 创建控制器...")
    controller = create_controller(
        task_name="Lift",
        robot_name="Panda",
        use_mock_vla=True,
        use_mock_mapper=False,  # 使用真实的 IK 映射器
        max_steps=10,
        verbose=True,
    )

    # 执行指令
    print("\n2. 执行指令...")
    result = controller.execute_instruction(
        instruction="Pick up the cube",
        render=False,
        save_trajectory=True,
    )

    # 打印结果
    print("\n3. 执行结果:")
    print(f"   状态: {result.state.value}")
    print(f"   成功: {result.success}")
    print(f"   步数: {result.steps}")
    print(f"   最终奖励: {result.final_reward:.4f}")

    # 获取 IK 统计
    ik_stats = controller.mapper.get_ik_stats()
    if ik_stats:
        print(f"\n4. IK 求解统计:")
        print(f"   平均求解时间: {ik_stats['avg_solve_time']*1000:.2f} ms")
        print(f"   成功率: {ik_stats['success_rate']*100:.1f}%")
        print(f"   成功次数: {ik_stats['success_count']}")
        print(f"   失败次数: {ik_stats['failure_count']}")

    # 关闭环境
    controller.env.close()

    return controller


def visualize_ik_results():
    """可视化 IK 求解结果"""
    from src.mapping.ik_solver import OptimizationIKSolver

    print("\n" + "="*60)
    print("可视化 IK 求解结果")
    print("="*60)

    ik_solver = OptimizationIKSolver(robot_name="Panda")

    # 测试多个目标点
    targets = [
        (np.array([0.3, 0.0, 0.3]), np.array([0, 0, 0])),
        (np.array([0.2, 0.2, 0.2]), np.array([0, 0, 0])),
        (np.array([0.0, 0.3, 0.4]), np.array([0, 0, 0])),
    ]

    current_joints = np.array([0, 0, 0, -1.5, 0, 1.5, 0])

    print(f"\n初始关节: {current_joints}")

    for i, (target_pos, target_rot) in enumerate(targets):
        print(f"\n目标 {i+1}: 位置={target_pos}, 旋转={target_rot}")

        solution = ik_solver.solve(target_pos, target_rot, current_joints)
        final_pose = ik_solver.forward_kinematics(solution)

        pos_error = np.linalg.norm(final_pose[:3] - target_pos)
        rot_error = np.linalg.norm(final_pose[3:6] - target_rot)

        print(f"  解: {solution}")
        print(f"  实际位置: {final_pose[:3]}")
        print(f"  误差: pos={pos_error:.6f}, rot={rot_error:.6f}")

        if pos_error < 0.01 and rot_error < 0.1:
            print(f"  ✓ 收敛")
        else:
            print(f"  ✗ 未收敛")


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("IK 求解器和模型集成测试")
    print("="*60)

    try:
        # 测试 IK 求解器
        test_ik_solver()

        # 测试动作映射器
        test_action_mapper_with_ik()

        # 测试 VLA 模型
        test_vla_model()

        # 测试完整流程
        test_full_pipeline()

        # 可视化结果
        visualize_ik_results()

        print("\n" + "="*60)
        print("✓ 所有测试完成!")
        print("="*60)

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
