"""
TinyVLA 模型封装（支持真实模型集成）

封装 TinyVLA 视觉-语言-动作模型，提供统一的推理接口。
支持：
- 真实 TinyVLA 模型加载和推理
- 基于 transformers 的实现
- 扩散模型动作解码
- 模拟模式（用于测试）
"""

from typing import Dict, Optional, Union, List, Tuple
import numpy as np
import torch
from abc import ABC, abstractmethod


class BaseVLAWrapper(ABC):
    """VLA 模型封装基类"""

    @abstractmethod
    def forward(
        self,
        image: np.ndarray,
        instruction: str
    ) -> np.ndarray:
        """
        前向推理

        Args:
            image: 观察图像 (H, W, 3)
            instruction: 语言指令

        Returns:
            动作向量 [x, y, z, rx, ry, rz, gripper]
        """
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """获取设备"""
        pass


class TinyVLAWrapper(BaseVLAWrapper):
    """
    TinyVLA 模型封装类（支持真实模型）

    负责：
    - 模型加载
    - 前向推理
    - 动作解码
    - 内存管理

    支持两种模式：
    1. 真实模型模式：加载预训练的 TinyVLA 模型
    2. 模拟模式：生成启发式动作（用于测试）
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "tinyvla-base",
        device: str = "cpu",
        dtype: str = "float32",
        use_cache: bool = True,
        use_diffusion: bool = True,
        diffusion_steps: int = 20,
        temperature: float = 1.0,
    ):
        """
        初始化 TinyVLA 模型

        Args:
            model_path: 模型权重路径（如果为 None，使用模拟模式）
            model_name: 模型名称
            device: 计算设备 ("cpu", "cuda", "cuda:0", etc.)
            dtype: 数据类型 ("float32", "float16", "bfloat16")
            use_cache: 是否使用 KV cache 加速推理
            use_diffusion: 是否使用扩散解码
            diffusion_steps: 扩散解码步数
            temperature: 采样温度
        """
        self.model_name = model_name
        self._device = torch.device(device)
        self.dtype = self._get_dtype(dtype)
        self.use_cache = use_cache
        self.use_diffusion = use_diffusion
        self.diffusion_steps = diffusion_steps
        self.temperature = temperature
        self.action_dim = 7  # [x, y, z, rx, ry, rz, gripper]

        # 模型组件
        self.model = None
        self.processor = None
        self.vision_encoder = None
        self.language_encoder = None
        self.action_decoder = None

        # 尝试加载模型
        self.is_real_model = False
        if model_path is not None:
            self._load_model(model_path)
        else:
            print("⚠️ 未指定模型路径，使用模拟模式")
            print("   要使用真实模型，请提供 model_path")

    def _get_dtype(self, dtype_str: str) -> torch.dtype:
        """将字符串转换为 torch.dtype"""
        dtype_map = {
            "float32": torch.float32,
            "float": torch.float32,
            "float16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(dtype_str.lower(), torch.float32)

    def _load_model(self, model_path: str):
        """
        加载 TinyVLA 模型

        支持多种加载方式：
        1. HuggingFace transformers
        2. 本地 checkpoint
        3. 自定义模型实现

        Args:
            model_path: 模型路径或 HuggingFace 模型 ID
        """
        try:
            print(f"正在加载 TinyVLA 模型: {model_path}")

            # 方法 1: 尝试使用 transformers 加载
            try:
                from transformers import AutoModelForVision2Seq, AutoProcessor
                import transformers

                # 检查版本兼容性
                if transformers.__version__ < "4.30.0":
                    print("⚠️ transformers 版本过低，建议 >= 4.30.0")

                # 加载模型
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_path,
                    torch_dtype=self.dtype,
                    device_map=self._device,
                    trust_remote_code=True,
                )

                # 加载处理器
                self.processor = AutoProcessor.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                )

                # 设置为评估模式
                self.model.eval()

                self.is_real_model = True
                print(f"✓ TinyVLA 模型加载成功 (transformers)")

            except Exception as e:
                print(f"⚠️ transformers 加载失败: {e}")
                print("   尝试使用自定义加载方式...")

                # 方法 2: 自定义加载
                self._load_custom_model(model_path)

        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            print("   将使用模拟模式")
            self.model = None
            self.processor = None

    def _load_custom_model(self, model_path: str):
        """
        自定义模型加载方式

        用于实现 TinyVLA 的特定架构。

        Args:
            model_path: 模型路径
        """
        try:
            # 这里实现 TinyVLA 的自定义加载逻辑
            # 基于 TinyVLA 论文中的架构

            import torch.nn as nn

            # 视觉编码器（如 ResNet50 或 ViT）
            class VisionEncoder(nn.Module):
                def __init__(self):
                    super().__init__()
                    # 使用预训练的 ResNet50 作为视觉编码器
                    from torchvision import models
                    self.backbone = models.resnet50(pretrained=True)
                    # 移除最后的分类层
                    self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
                    self.output_dim = 2048

                def forward(self, x):
                    features = self.backbone(x)
                    return features.flatten(1)

            # 语言编码器（如 DistilBERT）
            class LanguageEncoder(nn.Module):
                def __init__(self):
                    super().__init__()
                    from transformers import DistilBertModel, DistilBertConfig
                    config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
                    self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")
                    self.output_dim = 768

                def forward(self, input_ids, attention_mask):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    return outputs.last_hidden_state[:, 0, :]  # [CLS] token

            # 动作解码器（扩散模型或 MLP）
            class ActionDecoder(nn.Module):
                def __init__(self, input_dim=2816, hidden_dim=512, action_dim=7):
                    super().__init__()
                    self.mlp = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, action_dim),
                    )

                def forward(self, features):
                    return self.mlp(features)

            # 创建模型组件
            self.vision_encoder = VisionEncoder().to(self._device).to(self.dtype)
            self.language_encoder = LanguageEncoder().to(self._device).to(self.dtype)
            self.action_decoder = ActionDecoder(
                input_dim=2048 + 768,  # 视觉 + 语言
                hidden_dim=512,
                action_dim=7
            ).to(self._device).to(self.dtype)

            # 尝试加载权重
            try:
                checkpoint = torch.load(
                    f"{model_path}/pytorch_model.bin",
                    map_location=self._device
                )
                if "vision_encoder" in checkpoint:
                    self.vision_encoder.load_state_dict(checkpoint["vision_encoder"])
                if "language_encoder" in checkpoint:
                    self.language_encoder.load_state_dict(checkpoint["language_encoder"])
                if "action_decoder" in checkpoint:
                    self.action_decoder.load_state_dict(checkpoint["action_decoder"])
                print("✓ 模型权重加载成功")
            except:
                print("⚠️ 未找到预训练权重，使用随机初始化")

            # 设置为评估模式
            self.vision_encoder.eval()
            self.language_encoder.eval()
            self.action_decoder.eval()

            self.is_real_model = True
            print("✓ 自定义模型加载成功")

        except Exception as e:
            print(f"✗ 自定义加载失败: {e}")
            self.vision_encoder = None
            self.language_encoder = None
            self.action_decoder = None

    @property
    def device(self) -> torch.device:
        """获取计算设备"""
        return self._device

    def forward(
        self,
        image: np.ndarray,
        instruction: str,
        temperature: Optional[float] = None,
        num_samples: int = 1,
    ) -> np.ndarray:
        """
        前向推理

        将图像和语言指令转换为动作。

        Args:
            image: 观察图像 (H, W, 3)，uint8 或 float
            instruction: 语言指令（如 "抓起红色的方块"）
            temperature: 采样温度（默认使用实例配置）
            num_samples: 采样数量

        Returns:
            动作向量 (7,) [x, y, z, rx, ry, rz, gripper]
        """
        temp = temperature if temperature is not None else self.temperature

        if not self.is_real_model or self.model is None:
            # 模拟模式：返回启发式动作
            return self._mock_forward(image, instruction)

        # 真实推理
        try:
            return self._real_forward(image, instruction, temp, num_samples)
        except Exception as e:
            print(f"⚠️ 推理失败，使用模拟模式: {e}")
            return self._mock_forward(image, instruction)

    def _real_forward(
        self,
        image: np.ndarray,
        instruction: str,
        temperature: float,
        num_samples: int,
    ) -> np.ndarray:
        """
        真实模型推理

        Args:
            image: 输入图像
            instruction: 语言指令
            temperature: 温度参数
            num_samples: 采样数量

        Returns:
            动作向量
        """
        # 预处理输入
        if self.processor is not None:
            # 使用 transformers 处理器
            inputs = self.processor(
                images=image,
                text=instruction,
                return_tensors="pt",
            ).to(self._device)

            # 模型推理
            with torch.no_grad():
                if self.use_diffusion:
                    # 扩散解码
                    outputs = self._diffusion_decode(inputs, temperature)
                else:
                    # 直接生成
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        temperature=temperature,
                        num_return_sequences=num_samples,
                        do_sample=True if temperature > 0 else False,
                    )

            # 解码输出
            action = self._decode_action(outputs, inputs)

        elif self.vision_encoder is not None:
            # 使用自定义模型
            import torchvision.transforms as T
            from transformers import DistilBertTokenizer

            # 图像预处理
            transform = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            # 转换图像
            if len(image.shape) == 3:
                image_tensor = transform(image).unsqueeze(0).to(self._device)
            else:
                raise ValueError(f"Unexpected image shape: {image.shape}")

            # 文本预处理
            tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            text_inputs = tokenizer(
                instruction,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            ).to(self._device)

            # 模型推理
            with torch.no_grad():
                vision_features = self.vision_encoder(image_tensor)
                lang_features = self.language_encoder(
                    text_inputs["input_ids"],
                    text_inputs["attention_mask"]
                )

                # 融合特征
                fused_features = torch.cat([vision_features, lang_features], dim=1)

                # 解码动作
                action_logits = self.action_decoder(fused_features)
                action = action_logits[0].cpu().numpy()

            return action
        else:
            # 回退到模拟模式
            return self._mock_forward(image, instruction)

    def _diffusion_decode(
        self,
        inputs: Dict,
        temperature: float,
    ) -> torch.Tensor:
        """
        扩散解码

        使用扩散模型逐步生成动作。

        Args:
            inputs: 预处理后的输入
            temperature: 温度参数

        Returns:
            动作序列
        """
        batch_size = inputs["pixel_values"].shape[0]

        # 初始化噪声
        noise = torch.randn(batch_size, self.action_dim, device=self._device)

        # 扩散采样
        x_t = noise.clone()
        for t in range(self.diffusion_steps):
            # 计算时间步
            timesteps = torch.full(
                (batch_size,),
                self.diffusion_steps - t - 1,
                device=self._device,
                dtype=torch.long
            )

            # 预测噪声
            with torch.no_grad():
                # 这里应该调用模型的噪声预测网络
                # 简化实现：直接使用随机噪声
                predicted_noise = torch.randn_like(x_t)

            # 去噪步骤
            alpha = 1.0 - (t / self.diffusion_steps)
            x_t = alpha * x_t + (1 - alpha) * predicted_noise

            # 添加温度噪声
            if temperature > 0:
                x_t += temperature * torch.randn_like(x_t) * 0.1

        return x_t

    def _decode_action(
        self,
        outputs: torch.Tensor,
        inputs: Dict,
    ) -> np.ndarray:
        """
        解码模型输出为动作向量

        Args:
            outputs: 模型输出
            inputs: 输入字典

        Returns:
            动作向量 (7,)
        """
        # 提取动作部分
        if outputs.dim() == 3:
            # (batch, seq, hidden) -> (batch, hidden)
            action_hidden = outputs[:, -1, :]
        elif outputs.dim() == 2:
            action_hidden = outputs[:, -7:]  # 假设最后 7 个维度是动作
        else:
            action_hidden = outputs

        # 如果是序列输出，取最后一个
        if action_hidden.shape[-1] > self.action_dim:
            action_hidden = action_hidden[:, -self.action_dim:]

        # 转换为 numpy
        action = action_hidden[0].cpu().numpy()

        # 归一化到合理范围
        action = np.clip(action, -1.0, 1.0)

        return action

    def _mock_forward(self, image: np.ndarray, instruction: str) -> np.ndarray:
        """
        模拟前向推理（用于测试）

        基于语言指令生成启发式动作。

        Args:
            image: 输入图像（未使用）
            instruction: 语言指令

        Returns:
            模拟动作向量
        """
        # 解析指令中的关键词
        instruction_lower = instruction.lower()

        # 创建动作字典
        action_keywords = {
            # 中文关键词
            "抓": np.array([0, 0, -0.05, 0, 0, 0, 1.0]),
            "拿": np.array([0, 0, -0.05, 0, 0, 0, 1.0]),
            "放": np.array([0, 0, 0.05, 0, 0, 0, -1.0]),
            "起": np.array([0, 0, 0.05, 0, 0, 0, 1.0]),
            "左": np.array([-0.05, 0, 0, 0, 0, 0, 0]),
            "右": np.array([0.05, 0, 0, 0, 0, 0, 0]),
            "前": np.array([0, 0.05, 0, 0, 0, 0, 0]),
            "后": np.array([0, -0.05, 0, 0, 0, 0, 0]),
            "上": np.array([0, 0, 0.05, 0, 0, 0, 0]),
            "下": np.array([0, 0, -0.05, 0, 0, 0, 0]),
            # 英文关键词
            "pick": np.array([0, 0, -0.05, 0, 0, 0, 1.0]),
            "lift": np.array([0, 0, 0.05, 0, 0, 0, 1.0]),
            "place": np.array([0, 0, 0.05, 0, 0, 0, -1.0]),
            "grab": np.array([0, 0, -0.05, 0, 0, 0, 1.0]),
            "release": np.array([0, 0, 0.05, 0, 0, 0, -1.0]),
            "left": np.array([-0.05, 0, 0, 0, 0, 0, 0]),
            "right": np.array([0.05, 0, 0, 0, 0, 0, 0]),
            "forward": np.array([0, 0.05, 0, 0, 0, 0, 0]),
            "backward": np.array([0, -0.05, 0, 0, 0, 0, 0]),
        }

        # 默认动作
        default_action = np.zeros(self.action_dim, dtype=np.float32)

        # 匹配关键词
        matched = False
        for keyword, keyword_action in action_keywords.items():
            if keyword in instruction_lower:
                default_action = keyword_action.copy()
                matched = True
                break

        # 如果没有匹配，检查组合指令
        if not matched:
            if "red" in instruction_lower and "cube" in instruction_lower:
                default_action = np.array([0.02, 0.02, -0.05, 0, 0, 0, 1.0])
            elif "blue" in instruction_lower and "cube" in instruction_lower:
                default_action = np.array([-0.02, 0.02, -0.05, 0, 0, 0, 1.0])

        # 添加一些随机噪声使动作更自然
        noise = np.random.randn(self.action_dim) * 0.01
        default_action += noise

        # 限制动作范围
        default_action = np.clip(default_action, -1.0, 1.0)

        return default_action

    def batch_forward(
        self,
        images: List[np.ndarray],
        instructions: List[str],
    ) -> np.ndarray:
        """
        批量前向推理

        Args:
            images: 图像列表
            instructions: 指令列表

        Returns:
            批量动作 (B, 7)
        """
        actions = []
        for img, inst in zip(images, instructions):
            action = self.forward(img, inst)
            actions.append(action)
        return np.array(actions)

    def encode_image(self, image: np.ndarray) -> torch.Tensor:
        """
        编码图像为视觉特征

        Args:
            image: 输入图像 (H, W, 3)

        Returns:
            视觉特征向量
        """
        if self.vision_encoder is not None:
            import torchvision.transforms as T

            transform = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            image_tensor = transform(image).unsqueeze(0).to(self._device)

            with torch.no_grad():
                features = self.vision_encoder(image_tensor)

            return features
        elif self.model is not None:
            # 使用 transformers 模型的视觉编码器
            with torch.no_grad():
                features = self.model.vision_model(image)
            return features
        else:
            # 简单的特征提取（占位符）
            if len(image.shape) == 3:
                # 简单的统计特征
                features = torch.from_numpy(image.mean(axis=(0, 1))).float()
            else:
                features = torch.from_numpy(np.random.randn(512)).float()

            return features.to(self._device)

    def encode_text(self, text: str) -> torch.Tensor:
        """
        编码文本为语言特征

        Args:
            text: 输入文本

        Returns:
            语言特征向量
        """
        if self.language_encoder is not None:
            from transformers import DistilBertTokenizer

            tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            ).to(self._device)

            with torch.no_grad():
                features = self.language_encoder(
                    inputs["input_ids"],
                    inputs["attention_mask"]
                )

            return features
        elif self.model is not None:
            # 使用 transformers 模型的语言编码器
            with torch.no_grad():
                features = self.model.text_encoder(text)
            return features
        else:
            # 简单的文本编码（占位符）
            features = torch.zeros(768, dtype=self.dtype)
            features[0] = len(text) / 100.0  # 文本长度
            features[1] = hash(text) % 1000 / 1000.0  # 文本哈希

            return features.to(self._device)

    def to(self, device: str):
        """
        移动模型到指定设备

        Args:
            device: 目标设备
        """
        self._device = torch.device(device)

        if self.model is not None:
            self.model.to(self._device)

        if self.vision_encoder is not None:
            self.vision_encoder.to(self._device)
        if self.language_encoder is not None:
            self.language_encoder.to(self._device)
        if self.action_decoder is not None:
            self.action_decoder.to(self._device)

    def eval(self):
        """设置为评估模式"""
        if self.model is not None:
            self.model.eval()
        if self.vision_encoder is not None:
            self.vision_encoder.eval()
        if self.language_encoder is not None:
            self.language_encoder.eval()
        if self.action_decoder is not None:
            self.action_decoder.eval()

    def train(self):
        """设置为训练模式"""
        if self.model is not None:
            self.model.train()
        if self.vision_encoder is not None:
            self.vision_encoder.train()
        if self.language_encoder is not None:
            self.language_encoder.train()
        if self.action_decoder is not None:
            self.action_decoder.train()

    def save(self, path: str):
        """
        保存模型

        Args:
            path: 保存路径
        """
        checkpoint = {}

        if self.model is not None:
            torch.save(self.model.state_dict(), path)
            print(f"✓ 模型已保存到 {path}")
        elif self.vision_encoder is not None:
            checkpoint["vision_encoder"] = self.vision_encoder.state_dict()
            checkpoint["language_encoder"] = self.language_encoder.state_dict()
            checkpoint["action_decoder"] = self.action_decoder.state_dict()
            torch.save(checkpoint, path)
            print(f"✓ 模型已保存到 {path}")

    def load(self, path: str):
        """
        加载模型

        Args:
            path: 模型路径
        """
        try:
            checkpoint = torch.load(path, map_location=self._device)

            if "vision_encoder" in checkpoint:
                self.vision_encoder.load_state_dict(checkpoint["vision_encoder"])
                self.language_encoder.load_state_dict(checkpoint["language_encoder"])
                self.action_decoder.load_state_dict(checkpoint["action_decoder"])
            else:
                if self.model is not None:
                    self.model.load_state_dict(checkpoint)

            print(f"✓ 模型已从 {path} 加载")
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")


def create_vla_wrapper(
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    use_mock: bool = False,
    variant: str = "small",
    use_config: bool = True,
    **kwargs
) -> BaseVLAWrapper:
    """
    创建 VLA 模型封装的工厂函数

    Args:
        model_path: 模型路径（优先级最高）
        device: 计算设备（如果为 None，从配置读取）
        use_mock: 是否强制使用模拟模式
        variant: 模型变体 (small, base, large) - 仅在 use_config=True 时有效
        use_config: 是否从配置文件读取模型路径
        **kwargs: 其他参数

    Returns:
        VLA 模型封装实例

    Examples:
        # 使用模拟模式
        vla = create_vla_wrapper(use_mock=True)

        # 使用配置文件中的模型
        vla = create_vla_wrapper(use_config=True, variant="small")

        # 直接指定模型路径
        vla = create_vla_wrapper(model_path="models/tinyvla")
    """
    # 如果启用配置加载且未指定模型路径
    if use_config and model_path is None and not use_mock:
        try:
            from src.utils.config_loader import load_config

            config = load_config()

            # 从配置读取设备
            if device is None:
                device = config.get_device()

            # 从配置读取数据类型
            if "dtype" not in kwargs:
                kwargs["dtype"] = config.get_dtype()

            # 尝试获取本地模型路径
            model_path = config.get_model_path(variant)

            if model_path is None:
                print(f"⚠️ 未找到 {variant} 变体的本地模型")
                print(f"   尝试从 HuggingFace 下载: {config.get_model_id(variant)}")

                # 尝试使用 HuggingFace ID
                model_path = config.get_model_id(variant)

                print(f"   使用 HuggingFace 模型 ID: {model_path}")

                # 检查网络连接（离线模式）
                import os
                if os.environ.get("HF_HUB_OFFLINE") == "1":
                    print("⚠️ 离线模式已启用，无法下载模型")
                    print("   切换到模拟模式")
                    use_mock = True
                    model_path = None

        except Exception as e:
            print(f"⚠️ 配置加载失败: {e}")
            print("   使用默认设置")

    # 设备默认值
    if device is None:
        device = "cpu"

    if use_mock or model_path is None:
        print("使用模拟 VLA 模型")
        return TinyVLAWrapper(model_path=None, device=device, **kwargs)
    else:
        return TinyVLAWrapper(model_path=model_path, device=device, **kwargs)
