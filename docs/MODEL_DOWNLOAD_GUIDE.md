# TinyVLA 模型下载和配置指南

## 概述

本指南介绍如何下载和配置 TinyVLA 所需的 Llava-Pythia 模型权重。

## 模型信息

TinyVLA 使用 **Llava-Pythia** 系列模型作为视觉-语言骨干网络：

| 模型 | 参数量 | 用途 | HuggingFace ID |
|------|--------|------|----------------|
| Llava-Pythia-400M | ~400M | TinyVLA-S (推荐) | `lesjie/Llava-Pythia-400M` |
| Llava-Pythia-700M | ~700M | TinyVLA-B | `lesjie/Llava-Pythia-700M` |
| Llava-Pythia-1.3B | ~1.3B | TinyVLA-H | `lesjie/Llava-Pythia-1.3B` |

**模型大小参考**：
- Llava-Pythia-400M: ~1.5GB
- Llava-Pythia-700M: ~2.5GB
- Llava-Pythia-1.3B: ~5GB

---

## 下载方法

### 方法 1: 使用 HuggingFace CLI（推荐）

```bash
# 安装依赖
pip install huggingface-hub

# 下载模型
huggingface-cli download lesjie/Llava-Pythia-400M \
  --local-dir models/llava_pythia/Llava-Pythia-400M

# 或使用镜像站（国内推荐）
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
  lesjie/Llava-Pythia-400M \
  --local-dir models/llava_pythia/Llava-Pythia-400M
```

### 方法 2: 使用 Python Transformers

```python
from transformers import LlavaForConditionalGeneration, AutoProcessor

# 下载并保存模型
model_id = "lesjie/Llava-Pythia-400M"
save_path = "models/llava_pythia/Llava-Pythia-400M"

model = LlavaForConditionalGeneration.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

model.save_pretrained(save_path)
processor.save_pretrained(save_path)
```

### 方法 3: 网页手动下载

1. 访问 HuggingFace 模型页面：
   - https://huggingface.co/lesjie/Llava-Pythia-400M
   - 或使用镜像: https://hf-mirror.com/lesjie/Llava-Pythia-400M

2. 点击 **"Files and versions"** 标签

3. 下载所有文件：
   - `pytorch_model.bin` 或 `model.safetensors`
   - `config.json`
   - `tokenizer_config.json`
   - `preprocessor_config.json`
   - `special_tokens_map.json`
   - `tokenizer.json`

4. 将文件保存到：
   ```
   models/llava_pythia/Llava-Pythia-400M/
   ```

### 方法 4: 使用 Git LFS

```bash
# 安装 git-lfs
apt-get install git-lfs  # Linux
brew install git-lfs     # macOS

# 克隆模型仓库
git lfs install
git clone https://huggingface.co/lesjie/Llava-Pythia-400M \
  models/llava_pythia/Llava-Pythia-400M
```

---

## 配置验证

下载完成后，验证模型文件：

```bash
# 检查文件
ls -lh models/llava_pythia/Llava-Pythia-400M/

# 应包含以下文件：
# - config.json
# - pytorch_model.bin 或 model.safetensors
# - tokenizer_config.json
# - preprocessor_config.json
# - 等
```

运行测试脚本验证：

```bash
python tests/test_config_and_model.py
```

---

## 使用真实模型

下载模型后，在代码中使用：

### 方式 1: 使用配置文件

```yaml
# config/model_config.yaml
vlm_backbone:
  variants:
    small:
      local_path: "models/llava_pythia/Llava-Pythia-400M"
```

```python
from src.controllers.language_controller import create_controller

controller = create_controller(
    task_name="Lift",
    robot_name="Panda",
    use_mock_vla=False,  # 使用真实模型
)
```

### 方式 2: 直接指定路径

```python
from src.controllers.language_controller import create_controller

controller = create_controller(
    task_name="Lift",
    robot_name="Panda",
    model_path="models/llava_pythia/Llava-Pythia-400M",
    use_mock_vla=False,
)

result = controller.execute_instruction("抓起红色的方块", render=True)
```

### 方式 3: 使用演示脚本

```bash
# 高级演示（真实模型）
python demos/advanced_demo.py \
  --model-path models/llava_pythia/Llava-Pythia-400M \
  --ik-method optimization
```

---

## 离线模式配置

如果无法联网下载，可以：

1. 从其他设备下载模型
2. 复制到项目目录
3. 设置离线模式

```bash
# 设置环境变量
export HF_HUB_OFFLINE=1
export TINYVLA_DEFAULT_MODEL="models/llava_pythia/Llava-Pythia-400M"
```

---

## 镜像站配置（中国大陆）

使用 HuggingFace 镜像加速下载：

```bash
# 设置镜像站
export HF_ENDPOINT=https://hf-mirror.com

# 或在代码中
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import LlavaForConditionalGeneration
model = LlavaForConditionalGeneration.from_pretrained("lesjie/Llava-Pythia-400M")
```

---

## 故障排除

### 问题 1: 无法连接到 HuggingFace

**解决方案**：
- 使用镜像站：`export HF_ENDPOINT=https://hf-mirror.com`
- 或手动下载文件

### 问题 2: 模型加载失败

**常见原因**：
- 文件不完整：重新下载
- 缺少依赖：`pip install transformers accelerate`
- 格式不兼容：检查 transformers 版本

### 问题 3: 内存不足

**解决方案**：
- 使用更小的模型（400M）
- 启用量化：`dtype="float16"`
- 使用 CPU 推理：`device="cpu"`

### 问题 4: 下载速度慢

**解决方案**：
- 使用镜像站
- 使用多线程下载工具
- 分批下载文件

---

## 相关资源

- [TinyVLA 官方仓库](https://github.com/liyaxuanliyaxuan/TinyVLA)
- [Llava-Pythia HuggingFace](https://huggingface.co/lesjie)
- [transformers 文档](https://huggingface.co/docs/transformers)
- [HuggingFace 镜像站](https://hf-mirror.com)

---

## 快速命令参考

```bash
# 安装依赖
pip install huggingface-hub transformers torch

# 使用镜像下载
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
  lesjie/Llava-Pythia-400M \
  --local-dir models/llava_pythia/Llava-Pythia-400M

# 验证模型
python tests/test_config_and_model.py

# 运行演示
python demos/advanced_demo.py \
  --model-path models/llava_pythia/Llava-Pythia-400M
```

---

## 支持

如有问题，请：
1. 检查 [故障排除](#故障排除) 部分
2. 查看 GitHub Issues
3. 参考官方文档
