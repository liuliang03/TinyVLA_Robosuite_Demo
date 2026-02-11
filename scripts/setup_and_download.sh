#!/bin/bash
# TinyVLA 模型下载和配置脚本
#
# 此脚本提供了多种方法来获取和配置 TinyVLA 模型权重

set -e

echo "================================================"
echo "  TinyVLA 模型下载和配置脚本"
echo "================================================"
echo ""

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 项目路径
PROJECT_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
MODELS_DIR="${PROJECT_ROOT}/models"
LLAVA_DIR="${MODELS_DIR}/llava_pythia"

# 创建目录
echo -e "${YELLOW}创建模型目录...${NC}"
mkdir -p "${LLAVA_DIR}"
echo "  模型目录: ${MODELS_DIR}"
echo ""

# 检查网络连接
echo -e "${YELLOW}检查网络连接...${NC}"
if ping -c 1 huggingface.co &> /dev/null; then
    echo -e "${GREEN}✓ 网络连接正常${NC}"
else
    echo -e "${RED}✗ 无法连接到 HuggingFace${NC}"
    echo "  请检查网络或使用离线下载方式"
fi
echo ""

# 方法 1: 使用 HuggingFace CLI 下载（推荐）
echo -e "${YELLOW}方法 1: 使用 HuggingFace CLI 下载...${NC}"
if command -v huggingface-cli &> /dev/null; then
    echo "  huggingface-cli 已安装"
    echo ""
    echo "  正在下载 Llava-Pythia-400M..."
    huggingface-cli download lesjie/Llava-Pythia-400M --local-dir "${LLAVA_DIR}/Llava-Pythia-400M" || echo "   下载失败，尝试其他方法..."
else
    echo "  huggingface-cli 未安装"
    echo "  安装: pip install huggingface-hub"
fi
echo ""

# 方法 2: 使用 Python transformers 下载
echo -e "${YELLOW}方法 2: 使用 Python transformers 下载...${NC}"
python3 << 'EOF'
import os
import sys

# 检查 transformers
try:
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    print("  transformers 已安装")
except ImportError:
    print("  需要安装 transformers")
    print("  安装: pip install transformers")
    sys.exit(1)

# 模型列表
models = [
    ("Llava-Pythia-400M", "lesjie/Llava-Pythia-400M"),
    ("Llava-Pythia-700M", "lesjie/Llava-Pythia-700M"),
    ("Llava-Pythia-1.3B", "lesjie/Llava-Pythia-1.3B"),
]

print("\n  可用模型:")
for i, (name, model_id) in enumerate(models, 1):
    print(f"    {i}. {name}: {model_id}")

print("\n  选择要下载的模型（1-3，或 'q' 退出）:")
choice = "1"

# 默认下载第一个模型
name, model_id = models[0]
print(f"\n  正在下载 {name}...")

try:
    # 下载模型
    model = LlavaForConditionalGeneration.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)

    # 保存到本地
    save_path = f"{LLAVA_DIR}/{name}"
    os.makedirs(save_path, exist_ok=True)

    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)

    print(f"  ✓ 模型已保存到: {save_path}")

except Exception as e:
    print(f"  ✗ 下载失败: {e}")
    print("  可能原因:")
    print("    1. 网络连接问题")
    print("    2. 模型 ID 不正确")
    print("    3. HuggingFace 认证问题")
EOF
echo ""

# 方法 3: 手动下载指南
echo -e "${YELLOW}方法 3: 手动下载指南${NC}"
cat << 'EOF'

如果自动下载失败，可以手动下载模型：

1. 访问 HuggingFace 模型页面：
   https://huggingface.co/lesjie/Llava-Pythia-400M

2. 点击 "Files and versions" 标签

3. 下载以下文件：
   - pytorch_model.bin
   - config.json
   - tokenizer_config.json
   - preprocessor_config.json

4. 将文件保存到：
   PROJECT_ROOT/models/llava_pythia/Llava-Pythia-400M/

5. 运行此脚本配置模型

EOF
echo ""

# 方法 4: 使用 Git LFS（如果可用）
echo -e "${YELLOW}方法 4: 使用 Git LFS...${NC}"
if command -v git-lfs &> /dev/null; then
    echo "  git-lfs 已安装"
    echo ""
else
    echo "  git-lfs 未安装"
    echo "  安装: apt-get install git-lfs"
fi
echo ""

# 创建配置文件
echo -e "${YELLOW}创建配置文件...${NC}"
CONFIG_FILE="${PROJECT_ROOT}/config/model_config.yaml"
cat > "${CONFIG_FILE}" << EOF
# TinyVLA 模型配置文件

# VLM Backbone 配置
vlm_backbone:
  model_name: "Llava-Pythia-400M"
  huggingface_id: "lesjie/Llava-Pythia-400M"

  # 模型变体选项
  variants:
    small:
      name: "Llava-Pythia-400M"
      params: "400M"
      model_id: "lesjie/Llava-Pythia-400M"
      local_path: "${LLAVA_DIR}/Llava-Pythia-400M"

    base:
      name: "Llava-Pythia-700M"
      params: "700M"
      model_id: "lesjie/Llava-Pythia-700M"
      local_path: "${LLAVA_DIR}/Llava-Pthia-700M"

    large:
      name: "Llava-Pthia-1.3B"
      params: "1.3B"
      model_id: "lesjie/Llava-Pythia-1.3B"
      local_path: "${LLAVA_DIR}/Llava-Pthia-1.3B"

# 默认使用的模型
default_variant: "small"

# Policy Head 配置
policy_head:
  type: "diffusion"
  hidden_dim: 512
  num_layers: 6
  action_dim: 7

  # 训练配置
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 100

# 推理配置
inference:
  device: "cuda"  # cpu or cuda
  dtype: "float32"  # float32, float16
  use_cache: true
  max_new_tokens: 100

  # 扩散解码配置
  diffusion:
    enabled: true
    num_steps: 20
    scheduler: "ddpm"
    beta_schedule: "linear"

  # 温度采样配置
  temperature: 1.0
  top_k: 50
  top_p: 0.95

# 模型路径配置
paths:
  models_dir: "${MODELS_DIR}"
  cache_dir: "${LLAVA_DIR}"
  checkpoints_dir: "${MODELS_DIR}/checkpoints"
  logs_dir: "${PROJECT_ROOT}/logs"
EOF

echo "  ✓ 配置文件已创建: ${CONFIG_FILE}"
echo ""

# 检查已下载的模型
echo -e "${YELLOW}检查已下载的模型...${NC}"
if [ -d "${LLAVA_DIR}" ]; then
    echo "  已下载的模型:"
    for dir in "${LLAVA_DIR}"/*/; do
        if [ -d "$dir" ]; then
            model_name=$(basename "$dir")
            if [ -f "$dir/pytorch_model.bin" ] || [ -f "$dir/model.safetensors" ]; then
                echo -e "    ${GREEN}✓${NC} ${model_name}"
            else
                echo -e "    ${YELLOW}○${NC} ${model_name} (不完整)"
            fi
        fi
    done
else
    echo "  尚未下载模型"
fi
echo ""

# 创建环境变量配置
echo -e "${YELLOW}创建环境变量配置...${NC}"
ENV_FILE="${PROJECT_ROOT}/.env.local"
cat > "${ENV_FILE}" << EOF
# TinyVLA 环境变量配置

# 模型路径
export TINYVLA_MODELS_DIR="${MODELS_DIR}"
export TINYVLA_CACHE_DIR="${LLAVA_DIR}"
export TINYVLA_DEFAULT_MODEL="lesjie/Llava-Pythia-400M"

# 设备配置
export TINYVLA_DEVICE="cpu"
export TINYVLA_DTYPE="float32"

# IK 配置
export TINYVLA_IK_METHOD="optimization"
export TINYVLA_USE_IK="true"

# 日志配置
export TINYVLA_LOG_LEVEL="INFO"

# HuggingFace 配置
export HF_HOME="${MODELS_DIR}/.cache"
EOF

echo "  ✓ 环境变量已创建: ${ENV_FILE}"
echo ""
echo "  要加载环境变量，运行: source ${ENV_FILE}"
echo ""

# 创建下载脚本
DOWNLOAD_SCRIPT="${PROJECT_ROOT}/scripts/download_model.sh"
mkdir -p "$(dirname "$DOWNLOAD_SCRIPT")"
cat > "${DOWNLOAD_SCRIPT}" << 'EOFSCRIPT'
#!/bin/bash
# TinyVLA 模型下载脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 读取配置
source "${PROJECT_ROOT}/.env.local"

echo "正在下载 TinyVLA 模型..."
echo "模型目录: $TINYVLA_CACHE_DIR"

# 检查 Python 环境
if ! python3 -c "import transformers" 2>/dev/null; then
    echo "错误: 需要安装 transformers"
    echo "安装: pip install transformers"
    exit 1
fi

# 下载模型
python3 << 'PYTHON_CODE'
import os
import sys

# 从环境变量读取模型 ID
model_id = os.environ.get("TINYVLA_DEFAULT_MODEL", "lesjie/Llava-Pythia-400M")
cache_dir = os.environ.get("TINYVLA_CACHE_DIR", "/root/project/project1/models/llava_pythia")

print(f"下载模型: {model_id}")
print(f"保存目录: {cache_dir}")

try:
    from transformers import LlavaForConditionalGeneration, AutoProcessor

    # 下载并保存模型
    model = LlavaForConditionalGeneration.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)

    save_path = os.path.join(cache_dir, model_id.replace("/", "--"))
    os.makedirs(save_path, exist_ok=True)

    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)

    print(f"\n✓ 模型已成功下载到: {save_path}")
    print("\n文件列表:")
    for file in os.listdir(save_path):
        file_path = os.path.join(save_path, file)
        file_size = os.path.getsize(file_path) / (1024*1024)  # MB
        print(f"  - {file} ({file_size:.2f} MB)")

except Exception as e:
    print(f"\n✗ 下载失败: {e}")
    print("\n请检查:")
    print("  1. 网络连接")
    print("  2. HuggingFace 权限")
    print("  3. 磁盘空间")
    sys.exit(1)
PYTHON_CODE
SCRIPT
EOF

chmod +x "${DOWNLOAD_SCRIPT}"
echo "  ✓ 下载脚本已创建: ${DOWNLOAD_SCRIPT}"
echo ""

# 创建 README
MODEL_README="${MODELS_DIR}/README.md"
cat > "${MODEL_README}" << 'EOF'
# TinyVLA 模型目录

本目录包含 TinyVLA 所需的模型权重。

## 目录结构

```
models/
├── llava_pythia/           # VLM Backbone
│   ├── Llava-Pythia-400M/  # TinyVLA-S (推荐)
│   ├── Llava-Pythia-700M/  # TinyVLA-B
│   └── Llava-Pthia-1.3B/  # TinyVLA-H
├── checkpoints/            # 训练检查点
└── .cache/                # HuggingFace 缓存
```

## 模型说明

TinyVLA 使用 **Llava-Pythia** 作为视觉-语言模型骨干网络。

### 可用模型

| 模型 | 参数量 | 用途 | 下载命令 |
|------|--------|------|----------|
| Llava-Pythia-400M | ~400M | TinyVLA-S | `python scripts/download_model.sh` |
| Llava-Pythia-700M | ~700M | TinyVLA-B | `huggingface-cli download lesjie/Llava-Pythia-700M` |
| Llava-Pthia-1.3B | ~1.3B | TinyVLA-H | `huggingface-cli download lesjie/Llava-Pythia-1.3B` |

## 下载方法

### 方法 1: 自动下载脚本

```bash
python scripts/download_model.sh
```

### 方法 2: HuggingFace CLI

```bash
# 安装 CLI
pip install huggingface-hub

# 下载模型
huggingface-cli download lesjie/Llava-Pythia-400M \
  --local-dir models/llava_pythia/Llava-Pythia-400M
```

### 方法 3: Python 代码

```python
from transformers import LlavaForConditionalGeneration

model = LlavaForConditionalGeneration.from_pretrained("lesjie/Llava-Pythia-400M")
model.save_pretrained("models/llava_pythia/Llava-Pythia-400M")
```

### 方法 4: 网页下载

1. 访问: https://huggingface.co/lesjie/Llava-Pythia-400M
2. 点击 "Files and versions"
3. 下载所有文件（pytorch_model.bin, config.json, 等）
4. 解压到本目录

## 使用方法

下载完成后，在代码中引用模型：

```python
from src.controllers.language_controller import create_controller

controller = create_controller(
    model_path="models/llava_pythia/Llava-Pythia-400M",
    use_mock_vla=False,  # 使用真实模型
    use_mock_mapper=False,  # 使用 IK
)

result = controller.execute_instruction("抓起红色的方块")
```

## 模型大小参考

- Llava-Pythia-400M: ~1.5GB
- Llava-Pythia-700M: ~2.5GB
- Llava-Pthia-1.3B: ~5GB

## 相关链接

- [TinyVLA GitHub](https://github.com/JayceWen/tinyvla)
- [TinyVLA 代码仓库](https://github.com/liyaxuanliyaxuan/TinyVLA)
- [Llava-Pythia HuggingFace](https://huggingface.co/lesjie/Llava-Pythia-400M)
EOF

echo "  ✓ 模型 README 已创建: ${MODEL_README}"
echo ""

# 生成测试脚本
TEST_SCRIPT="${PROJECT_ROOT}/tests/test_model_loading.py"
cat > "${TEST_SCRIPT}" << 'EOF'
#!/usr/bin/env python3
"""
测试模型加载
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_model_loading():
    """测试模型加载"""
    print("="*60)
    print("测试 TinyVLA 模型加载")
    print("="*60)

    # 测试 VLM Backbone
    print("\n1. 测试 Llava-Pythia 模型加载...")
    try:
        from transformers import LlavaForConditionalGeneration, AutoProcessor

        model_path = "models/llava_pythia/Llava-Pythia-400M"

        if not Path(model_path).exists():
            print(f"  ✗ 模型不存在: {model_path}")
            print(f"  请先运行: python scripts/download_model.sh")
            return False

        print(f"  正在加载模型: {model_path}")

        model = LlavaForConditionalGeneration.from_pretrained(model_path)
        processor = AutoProcessor.from_pretrained(model_path)

        print(f"  ✓ 模型加载成功!")
        print(f"  模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

        return True

    except Exception as e:
        print(f"  ✗ 模型加载失败: {e}")
        return False

def test_vla_wrapper():
    """测试 VLA Wrapper"""
    print("\n2. 测试 VLA Wrapper...")
    try:
        from src.core.vla_wrapper import create_vla_wrapper

        # 使用真实模型路径
        wrapper = create_vla_wrapper(
            model_path="models/llava_pythia/Llava-Pythia-400M",
            use_mock=False,  # 使用真实模型
        )

        print("  ✓ VLA Wrapper 创建成功")

        # 测试推理
        import numpy as np
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        instruction = "Pick up the cube"

        action = wrapper.forward(image, instruction)

        print(f"  ✓ 推理成功!")
        print(f"  动作形状: {action.shape}")
        print(f"  动作值: {action}")

        return True

    except Exception as e:
        print(f"  ✗ VLA Wrapper 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_pipeline():
    """测试完整流程"""
    print("\n3. 测试完整控制流程...")
    try:
        from src.controllers.language_controller import create_controller

        controller = create_controller(
            model_path="models/llava_pythia/Llava-Pythia-400M",
            use_mock_vla=False,
            use_mock_mapper=False,
            max_steps=5,
        )

        result = controller.execute_instruction(
            instruction="Pick up the cube",
            render=False,
        )

        print(f"  ✓ 控制流程测试成功!")
        print(f"  状态: {result.state.value}")
        print(f"  步数: {result.steps}")

        controller.env.close()

        return True

    except Exception as e:
        print(f"  ✗ 完整流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("\nTinyVLA 模型加载测试")
    print("="*60)

    results = []

    # 测试模型加载
    results.append(test_model_loading())

    # 如果模型加载成功，继续测试其他组件
    if results[0]:
        results.append(test_vla_wrapper())
        results.append(test_full_pipeline())

    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)

    tests = ["模型加载", "VLA Wrapper", "完整流程"]
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {status}: {test}")

    all_passed = all(results)
    print("="*60)

    if all_passed:
        print("✓ 所有测试通过! 模型已正确配置。")
    else:
        print("✗ 部分测试失败，请检查配置。")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
EOF

chmod +x "${TEST_SCRIPT}"
echo "  ✓ 测试脚本已创建: ${TEST_SCRIPT}"
echo ""

# 总结
echo "================================================"
echo "  配置完成!"
echo "================================================"
echo ""
echo "下一步操作:"
echo ""
echo "1. 下载模型（选择一种方法）:"
echo "   方法 A - 运行下载脚本:"
echo "     python scripts/download_model.sh"
echo ""
echo "   方法 B - 手动下载:"
echo "   访问 https://huggingface.co/lesjie/Llava-Pythia-400M/tree/main"
echo "   下载文件到: models/llava_pythia/Llava-Pythia-400M/"
echo ""
echo "2. 测试模型加载:"
echo "   python tests/test_model_loading.py"
echo ""
echo "3. 运行演示:"
echo "   python demos/advanced_demo.py --model-path models/llava_pythia/Llava-Pythia-400M"
echo ""
echo "================================================"
