# TinyVLA + robosuite 机械臂抓取演示系统

## 🎯 项目概述

使用 **TinyVLA** 作为控制器的机械臂抓取演示程序，将视觉-语言-动作（VLA）模型与 robosuite 仿真环境集成，实现通过自然语言指令控制机械臂抓取物体。

### ✨ 核心特性

- 🤖 **VLA 模型集成**：支持 TinyVLA 视觉-语言-动作模型（真实模型 + 模拟模式）
- 🎮 **语言指令控制**：通过自然语言指令控制机器人（中英文支持）
- 🦾 **多机器人支持**：Panda、Sawyer、UR5e 等多种机械臂
- 🎯 **多任务支持**：Lift、PickPlace、Stack 等抓取任务
- 🧮 **完整 IK 求解**：支持优化方法、Jacobian、PyBullet 等多种 IK 求解器
- 🎨 **可视化演示**：实时渲染抓取过程
- 🔧 **模块化设计**：遵循 SOLID 原则，易于扩展和维护

---

## 🏗️ 系统架构

```
用户交互层 (CLI + 可视化)
    ↓
控制逻辑层 (TinyVLA控制器 + IK求解器 + 动作映射)
    ↓
仿真环境层 (robosuite + MuJoCo)
```

### 核心数据流

1. 用户输入语言指令（如"抓起红色的方块"）
2. 获取当前场景图像
3. TinyVLA 模型推理生成 7D 动作向量 [x,y,z,rx,ry,rz,gripper]
4. **IK 求解器**将末端位姿转换为关节角度
5. 执行动作并更新环境
6. 循环直到任务完成

---

## 📦 安装

### 环境要求

- Python 3.10+
- MuJoCo 3.0+
- CUDA（可选，用于 GPU 加速）

### 快速安装

```bash
# 克隆项目
git clone <repository-url>
cd tinyvla-robosuite-demo

# 安装依赖
pip install -r requirements.txt

# 验证安装
python tests/test_environment.py
```

### 可选依赖

```bash
# PyBullet IK 求解器
pip install pybullet

# transformers（真实模型需要）
pip install transformers>=4.30.0

# 可视化
pip install matplotlib
```

---

## 🚀 使用方法

### 基础演示

```bash
# 运行最小演示（模拟模式）
python demos/minimal_demo.py

# 指定指令
python demos/minimal_demo.py --instruction "抓起红色的方块"

# 交互式模式
python demos/minimal_demo.py --interactive

# 批量测试
python demos/minimal_demo.py --batch
```

### 高级演示（带 IK）

```bash
# 使用 IK 求解器（模拟模型）
python demos/advanced_demo.py --ik-method optimization

# 对比不同 IK 方法
python demos/advanced_demo.py --comparison

# 对比 IK vs 简化映射
python demos/advanced_demo.py --ik-vs-no-ik

# 查看模型加载指南
python demos/advanced_demo.py --guide
```

### 使用真实 TinyVLA 模型

```bash
# 使用真实模型
python demos/advanced_demo.py --model-path /path/to/tinyvla

# 指定 IK 方法
python demos/advanced_demo.py --model-path /path/to/model --ik-method jacobian
```

---

## 📖 API 使用

### 基础控制器

```python
from src.controllers.language_controller import create_controller

# 创建控制器（模拟模式 + IK）
controller = create_controller(
    task_name="Lift",
    robot_name="Panda",
    use_mock_vla=True,       # 模拟 VLA
    use_mock_mapper=False,    # 使用 IK
    ik_method="optimization",
    max_steps=500,
    verbose=True,
)

# 执行指令
result = controller.execute_instruction(
    instruction="Pick up the cube",
    render=True,
)

print(f"成功: {result.success}")
print(f"步数: {result.steps}")
print(f"奖励: {result.final_reward:.4f}")
```

### 真实模型集成

```python
from src.controllers.language_controller import create_controller

# 使用真实 TinyVLA 模型
controller = create_controller(
    task_name="Lift",
    robot_name="Panda",
    model_path="models/tinyvla",  # 模型路径
    use_mock_vla=False,       # 使用真实模型
    device="cuda",           # GPU 加速
    use_mock_mapper=False,    # 使用 IK
    ik_method="optimization",
)

result = controller.execute_instruction(
    instruction="抓起红色的方块",
    render=True,
)
```

### IK 求解器

```python
from src.mapping.ik_solver import create_ik_solver

# 创建 IK 求解器
ik_solver = create_ik_solver(
    method="optimization",  # optimization, jacobian, pybullet
    robot_name="Panda",
    max_iterations=100,
    position_threshold=0.01,  # 1cm
    orientation_threshold=0.1,  # 0.1 rad
)

# 求解 IK
target_pos = np.array([0.3, 0.0, 0.3])
target_rot = np.array([0, 0, 0])
current_joints = np.array([0, 0, 0, -1.5, 0, 1.5, 0])

solution_joints = ik_solver.solve(
    target_pos=target_pos,
    target_rot=target_rot,
    current_joints=current_joints,
)

print(f"解: {solution_joints}")
```

### 动作映射器

```python
from src.mapping.action_mapper import create_action_mapper

# 创建带 IK 的映射器
mapper = create_action_mapper(
    controller_type="BASIC",
    use_ik=True,              # 启用 IK
    ik_method="optimization",
    robot_name="Panda",
    action_scale=0.1,
    verbose=True,
)

# 映射动作
vla_action = np.array([0.05, 0, -0.05, 0, 0, 0, 1.0])
obs = {"joint_positions": current_joints, ...}

env_action = mapper.map(vla_action, obs)
```

---

## 📂 项目结构

```
tinyvla-robosuite-demo/
├── README.md                      # 项目说明
├── requirements.txt               # 依赖列表
├── setup.py                       # 安装配置
│
├── config/                        # 配置文件
│   ├── default_config.yaml        # 默认配置
│   ├── robot_configs/             # 机器人配置
│   │   ├── panda.yaml
│   │   └── sawyer.yaml
│   └── vla_configs/               # VLA 模型配置
│       └── tinyvla_config.yaml
│
├── src/                           # 源代码
│   ├── core/                      # 核心模块
│   │   └── vla_wrapper.py         # TinyVLA 模型封装（支持真实模型）
│   ├── environments/              # 环境模块
│   │   └── robosuite_env.py       # robosuite 环境包装器
│   ├── mapping/                   # 映射模块
│   │   ├── action_mapper.py       # 动作空间映射器（集成 IK）
│   │   └── ik_solver.py          # IK 求解器（新增）
│   ├── vision/                    # 视觉模块
│   │   └── image_preprocessor.py  # 图像预处理器
│   ├── controllers/               # 控制器模块
│   │   └── language_controller.py # 语言指令控制器
│   ├── utils/                     # 工具模块
│   └── cli/                       # CLI 模块
│
├── demos/                         # 演示程序
│   ├── minimal_demo.py            # 基础演示
│   └── advanced_demo.py           # 高级演示（带 IK，新增）
│
└── tests/                         # 测试
    ├── test_environment.py         # 环境测试
    └── test_ik_integration.py      # IK 集成测试（新增）
```

---

## 🔧 配置说明

### 默认配置 (`config/default_config.yaml`)

```yaml
# 机器人配置
robot:
  name: "Panda"
  controller_type: "BASIC"
  control_frequency: 20
  horizon: 500

# IK 求解器配置（新增）
ik:
  solver: "optimization"  # optimization, jacobian, pybullet
  max_iterations: 100
  position_threshold: 0.01  # 位置误差阈值 (m)
  orientation_threshold: 0.1  # 旋转误差阈值 (rad)
  use_ik: true  # 是否使用 IK

# VLA 模型配置
vla:
  model_path: "models/tinyvla"
  device: "cpu"  # cpu, cuda
  use_diffusion: true
  diffusion_steps: 20
```

### IK 求解器选择

| 求解器 | 优点 | 缺点 | 适用场景 |
|--------|------|------|----------|
| **Optimization** | 精度高，稳定 | 计算量大 | 精确操作 |
| **Jacobian** | 速度快 | 可能不收敛 | 实时控制 |
| **PyBullet** | 准确，稳定 | 需要额外依赖 | 已有 PyBullet 环境 |

---

## 📊 性能指标

| 指标 | 目标 | 说明 |
|------|------|------|
| 任务成功率 | >80% | Lift 任务（使用真实模型） |
| 平均步数 | <200 | 完成任务的平均步数 |
| IK 求解时间 | <10ms | 单次 IK 求解（优化方法） |
| 控制频率 | >10 Hz | 每秒动作数 |
| 内存占用 | <8GB | 不包括模型权重 |

---

## 🔍 支持的任务

| 任务名称 | 描述 | 难度 |
|---------|------|-----|
| Lift | 抓起物体 | ⭐ |
| PickPlace | 抓取并放置物体 | ⭐⭐ |
| Stack | 堆叠物体 | ⭐⭐⭐ |
| NutAssembly | 螺母组装 | ⭐⭐⭐⭐ |
| Door | 开门操作 | ⭐⭐ |

---

## 🤖 支持的机器人

- **Panda** (Franka Emika) - 推荐
- **Sawyer** (Rethink Robotics)
- **UR5e** (Universal Robots)
- **Jaco** (Kinova)
- **IIWA** (KUKA)

---

## 🔬 测试

运行完整测试套件：

```bash
# 环境测试
python tests/test_environment.py

# IK 集成测试
python tests/test_ik_integration.py
```

---

## 📈 最新更新 (v0.2)

### ✨ 新功能

1. **完整 IK 求解器实现**
   - 基于 scipy 优化的 IK 求解器
   - Jacobian 伪逆法
   - PyBullet IK 支持

2. **真实 TinyVLA 模型集成**
   - 支持 HuggingFace transformers
   - 自定义模型加载
   - 扩散解码动作生成

3. **改进的动作映射**
   - 自动 IK 回退机制
   - 工作空间边界检查
   - 性能统计和调试信息

### 🔧 改进

- 模块化 IK 求解器接口
- 更详细的错误处理
- 性能监控和统计
- 批量测试和对比功能

---

## ⚠️ 已知限制

1. **TinyVLA 模型权重**
   - 当前使用模拟模式演示
   - 需要真实模型权重才能实现真正的语言理解

2. **IK 求解器精度**
   - 使用简化的正向运动学
   - 生产环境应使用完整的运动学库

3. **仅单臂**
   - 双臂任务尚未完全测试

---

## 🚀 未来计划

- [ ] 集成真实的 TinyVLA 模型权重
- [ ] 实现完整的机器人运动学（基于 MuJoCo）
- [ ] 支持双臂操作
- [ ] 添加更多任务和物体
- [ ] Web 界面
- [ ] 真实机器人部署（ROS 2）

---

## 📚 参考资料

### 核心资源
- [TinyVLA GitHub](https://github.com/JayceWen/tinyvla)
- [TinyVLA 论文](https://arxiv.org/abs/2409.12514)
- [robosuite 官网](https://robosuite.ai/)
- [robosuite GitHub](https://github.com/ARISE-Initiative/robosuite)

### IK 求解
- [PyBullet IK 教程](https://pybullet.org/)
- [机器人学导论](https://www.coursera.org/learn/robotics)

### VLA 研究
- [OpenVLA](https://github.com/openvla/openvla)
- [Foundation Models for Robotics](https://rohitbandaru.github.io/blog/)

---

## 📝 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

---

## 📧 联系方式

- 项目主页: [GitHub Repository]
- 问题反馈: [GitHub Issues]

---

**注意**: 本项目仍在积极开发中，API 可能会发生变化。

**特别说明**: 要使用真实的 TinyVLA 模型，请参考 `demos/advanced_demo.py --guide` 中的加载指南。
