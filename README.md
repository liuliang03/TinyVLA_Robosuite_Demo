# TinyVLA + robosuite 机械臂抓取演示系统

使用 **TinyVLA** 作为控制器的机械臂抓取演示程序，将视觉-语言-动作（VLA）模型与 robosuite 仿真环境集成，实现通过自然语言指令控制机械臂抓取物体。

## ⚡ 快速开始

### 当前状态

✅ **核心功能已完成** - 系统可以正常运行模拟模式

要使用真实 TinyVLA 模型，请先下载模型权重：
> 📋 **[模型下载指南](docs/MODEL_DOWNLOAD_GUIDE.md)** - 详细的下载步骤

```bash
# 克隆项目
git clone <repository-url>
cd tinyvla-robosuite-demo

# 安装依赖
pip install -r requirements.txt

# 运行演示（模拟模式）
python demos/minimal_demo.py
```

---

## 项目简介

本项目实现了一个完整的视觉-语言-动作（VLA）控制系统，将 TinyVLA 模型与 robosuite 仿真环境集成，实现通过自然语言指令控制机械臂完成抓取任务。

### 核心功能

- 🤖 **VLA 模型集成**：支持 TinyVLA 视觉-语言-动作模型（真实模型 + 模拟模式）
- 🎮 **语言指令控制**：通过自然语言指令控制机器人（中英文支持）
- 🦾 **多机器人支持**：Panda、Sawyer、UR5e 等多种机械臂
- 🎯 **多任务支持**：Lift、PickPlace、Stack 等抓取任务
- 🧮 **完整 IK 求解**：支持优化方法、Jacobian、PyBullet 等多种 IK 求解器
- 🎨 **可视化演示**：实时渲染抓取过程
- 🔧 **模块化设计**：遵循 SOLID 原则，易于扩展和维护

## 系统架构

```
用户交互层 (CLI + 可视化)
    ↓
控制逻辑层 (TinyVLA控制器 + 动作映射)
    ↓
仿真环境层 (robosuite + MuJoCo)
```

### 核心组件

1. **TinyVLA 模型封装** (`src/core/vla_wrapper.py`)
   - 模型加载和推理
   - 支持模拟模式（用于测试）

2. **robosuite 环境包装器** (`src/environments/robosuite_env.py`)
   - 统一的环境接口
   - 观察和动作空间管理

3. **动作空间映射器** (`src/mapping/action_mapper.py`)
   - VLA 输出到 robosuite 动作的转换
   - 支持多种控制器类型

4. **图像预处理器** (`src/vision/image_preprocessor.py`)
   - 图像尺寸调整、颜色空间转换、归一化

5. **语言指令控制器** (`src/controllers/language_controller.py`)
   - 高层控制逻辑，协调所有组件

## 安装

### 环境要求

- Python 3.10+
- MuJoCo 3.0+
- CUDA（可选，用于 GPU 加速）

### 安装步骤

1. 克隆项目
```bash
git clone <repository-url>
cd tinyvla-robosuite-demo
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 验证安装
```bash
python tests/test_environment.py
```

## 使用方法

### 快速开始

运行最小演示：

```bash
python demos/minimal_demo.py
```

### 自定义指令

```bash
# 指定任务和指令
python demos/minimal_demo.py --task Lift --instruction "抓起红色的方块"

# 启用渲染
python demos/minimal_demo.py --render

# 设置最大步数
python demos/minimal_demo.py --max-steps 1000
```

### 交互式模式

```bash
python demos/minimal_demo.py --interactive
```

### 批量测试

```bash
python demos/minimal_demo.py --batch
```

## 项目结构

```
tinyvla-robosuite-demo/
├── README.md                      # 项目说明
├── requirements.txt               # 依赖列表
├── setup.py                       # 安装配置
├── .gitignore                     # Git 忽略文件
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
│   │   └── vla_wrapper.py         # TinyVLA 模型封装
│   ├── environments/              # 环境模块
│   │   └── robosuite_env.py       # robosuite 环境包装器
│   ├── mapping/                   # 映射模块
│   │   └── action_mapper.py       # 动作空间映射器
│   ├── vision/                    # 视觉模块
│   │   └── image_preprocessor.py  # 图像预处理器
│   ├── controllers/               # 控制器模块
│   │   └── language_controller.py # 语言指令控制器
│   ├── utils/                     # 工具模块
│   └── cli/                       # CLI 模块
│
├── demos/                         # 演示程序
│   ├── minimal_demo.py            # 最小演示
│   ├── lift_demo.py               # Lift 任务演示
│   └── pickplace_demo.py          # PickPlace 演示
│
└── tests/                         # 测试
    ├── test_environment.py         # 环境测试
    └── integration/               # 集成测试
```

## 配置说明

### 默认配置 (`config/default_config.yaml`)

主要配置项：

```yaml
# 机器人配置
robot:
  name: "Panda"              # 机器人类型
  controller_type: "BASIC"    # 控制器类型
  control_frequency: 20       # 控制频率 Hz

# 环境配置
environment:
  task_name: "Lift"          # 任务类型
  image_size: [224, 224]     # 图像尺寸

# VLA 模型配置
vla:
  model_path: "models/tinyvla"  # 模型路径
  device: "cpu"                 # 计算设备

# 动作配置
action:
  action_dim: 7               # 动作维度
  action_scale: 0.1           # 动作缩放
```

## 支持的任务

| 任务名称 | 描述 | 难度 |
|---------|------|-----|
| Lift | 抓起物体 | ⭐ |
| PickPlace | 抓取并放置物体 | ⭐⭐ |
| Stack | 堆叠物体 | ⭐⭐⭐ |
| NutAssembly | 螺母组装 | ⭐⭐⭐⭐ |
| Door | 开门操作 | ⭐⭐ |

## 支持的机器人

- Panda (Franka Emika)
- Sawyer (Rethink Robotics)
- UR5e (Universal Robots)
- Jaco (Kinova)
- IIWA (KUKA)

## 开发指南

### 添加新任务

1. 在 `src/environments/robosuite_env.py` 中添加任务名称
2. 在 `config/` 中创建相应的配置文件
3. 在 `demos/` 中创建演示脚本

### 自定义 VLA 模型

1. 实现 `BaseVLAWrapper` 接口
2. 在 `src/core/vla_wrapper.py` 中添加模型加载逻辑
3. 更新配置文件

### 调试技巧

启用详细日志：

```python
controller = create_controller(verbose=True)
```

保存轨迹：

```python
result = controller.execute_instruction(
    instruction="Pick up the cube",
    save_trajectory=True,
)
# 访问轨迹
trajectory = result.trajectory
```

## 测试

运行单元测试：

```bash
pytest tests/
```

运行集成测试：

```bash
pytest tests/integration/
```

测试覆盖率：

```bash
pytest --cov=src tests/
```

## 性能指标

| 指标 | 目标 | 说明 |
|------|------|-----|
| 任务成功率 | >80% | Lift 任务 |
| 平均步数 | <200 | 完成任务的平均步数 |
| 推理延迟 | <100ms | 单次 VLA 推理时间 |
| 控制频率 | >10 Hz | 每秒动作数 |

## 已知问题

1. **TinyVLA 模型权重** ⚠️
   - 当前使用模拟模式演示
   - 需要真实模型权重才能实现真正的语言理解
   - 详见 [模型下载指南](docs/MODEL_DOWNLOAD_GUIDE.md)

2. **IK 求解器精度**
   - 使用简化的正向运动学
   - 生产环境应使用完整的运动学库

3. **仅支持单臂**
   - 双臂任务尚未完全测试

## 未来计划

- [x] ~~集成真实的 TinyVLA 模型~~（基础设施已完成，待下载模型）
- [x] ~~实现完整的 IK 求解器~~（已完成三种方法）
- [ ] 支持双臂操作
- [ ] 添加更多任务
- [ ] Web 界面
- [ ] 真实机器人部署

## 更多文档

| 文档 | 说明 |
|------|------|
| [模型下载指南](docs/MODEL_DOWNLOAD_GUIDE.md) | 如何下载和配置 TinyVLA 模型 |
| [项目状态](docs/PROJECT_STATUS.md) | 当前开发状态和已知限制 |
| [完整 README](README_UPDATED.md) | 详细的项目文档 |

## 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

## 致谢

- [TinyVLA](https://github.com/JayceWen/tinyvla) - 视觉-语言-动作模型
- [robosuite](https://github.com/ARISE-Initiative/robosuite) - 机器人仿真框架
- [MuJoCo](https://mujoco.org/) - 物理仿真引擎

## 联系方式

- 项目主页: [GitHub Repository]
- 问题反馈: [GitHub Issues]

---

**注意**: 本项目仍在开发中，API 可能会发生变化。
