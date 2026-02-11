"""
图像预处理器

负责将 robosuite 渲染的图像转换为 TinyVLA 模型期望的格式。
功能包括：
- 图像尺寸调整
- 颜色空间转换（BGR → RGB）
- 归一化
- Tensor 转换
"""

from typing import Tuple, Optional, Union
import numpy as np
import cv2
import torch


class ImagePreprocessor:
    """
    图像预处理器

    将 robosuite 渲染的图像（H, W, 3）转换为 TinyVLA 模型输入格式（1, 3, H, W）。

    Attributes:
        target_size: 目标图像尺寸 (width, height)
        normalize: 是否归一化到 [0, 1]
        color_space: 颜色空间 ("RGB", "BGR", "GRAY")
        data_format: 输出数据格式 ("CHW", "HWC")
        mean: 归一化均值（用于 ImageNet 标准化）
        std: 归一化标准差
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        color_space: str = "RGB",
        data_format: str = "CHW",
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """
        初始化图像预处理器

        Args:
            target_size: 目标尺寸 (width, height)
            normalize: 是否归一化
            color_space: 颜色空间 ("RGB", "BGR", "GRAY")
            data_format: 数据格式 ("CHW", "HWC")
            mean: ImageNet 均值（用于标准化）
            std: ImageNet 标准差（用于标准化）
        """
        self.target_size = target_size
        self.normalize = normalize
        self.color_space = color_space.upper()
        self.data_format = data_format.upper()
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

        # 验证参数
        assert self.color_space in ["RGB", "BGR", "GRAY"], f"不支持的颜色空间: {color_space}"
        assert self.data_format in ["CHW", "HWC"], f"不支持的数据格式: {data_format}"

    def preprocess(
        self,
        image: np.ndarray,
        normalize: Optional[bool] = None
    ) -> torch.Tensor:
        """
        预处理图像

        Args:
            image: 输入图像 (H, W, C)，来自 robosuite 渲染器
            normalize: 是否归一化（默认使用实例配置）

        Returns:
            torch.Tensor: 预处理后的图像
                - CHW 格式: (1, 3, H, W)
                - HWC 格式: (1, H, W, 3)
        """
        use_normalize = normalize if normalize is not None else self.normalize

        # 1. 确保图像是 numpy 数组
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        # 2. 调整尺寸
        processed = self._resize(image)

        # 3. 颜色空间转换
        processed = self._convert_color_space(processed)

        # 4. 归一化
        if use_normalize:
            processed = self._normalize(processed)

        # 5. 转换为 Tensor 并添加 batch 维度
        tensor = self._to_tensor(processed)

        return tensor

    def _resize(self, image: np.ndarray) -> np.ndarray:
        """调整图像尺寸"""
        if image.shape[:2][::-1] == self.target_size:
            return image

        # 使用双线性插值调整尺寸
        resized = cv2.resize(
            image,
            self.target_size,
            interpolation=cv2.INTER_LINEAR
        )
        return resized

    def _convert_color_space(self, image: np.ndarray) -> np.ndarray:
        """转换颜色空间"""
        if self.color_space == "RGB":
            # robosuite 输出 RGB，无需转换
            if len(image.shape) == 3 and image.shape[2] == 3:
                # 假设输入已经是 RGB（robosuite 默认）
                return image
        elif self.color_space == "BGR":
            # 转换为 BGR（OpenCV 默认）
            if len(image.shape) == 3 and image.shape[2] == 3:
                return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif self.color_space == "GRAY":
            # 转换为灰度图
            if len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return image

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """
        归一化图像

        支持两种归一化方式：
        1. 简单归一化：像素值 / 255.0 → [0, 1]
        2. ImageNet 标准化：(x/255 - mean) / std
        """
        # 转换为 float32
        normalized = image.astype(np.float32)

        # 简单归一化到 [0, 1]
        normalized = normalized / 255.0

        # ImageNet 标准化（可选）
        if self.mean is not None and self.std is not None:
            if len(normalized.shape) == 3:
                # (H, W, C) 格式
                normalized = (normalized - self.mean) / self.std
            elif len(normalized.shape) == 2:
                # 灰度图 (H, W)
                normalized = (normalized - self.mean[0]) / self.std[0]

        return normalized

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """
        将 numpy 数组转换为 PyTorch Tensor

        Args:
            image: 预处理后的图像 (H, W, C) 或 (H, W)

        Returns:
            torch.Tensor: 添加 batch 维度后的张量
        """
        # 转换为 Tensor
        if len(image.shape) == 3:
            # (H, W, C) → (1, H, W, C) 或 (1, C, H, W)
            tensor = torch.from_numpy(image).float()
        else:
            # (H, W) 灰度图 → (1, 1, H, W) 或 (1, H, W)
            tensor = torch.from_numpy(image).float().unsqueeze(0)

        # 调整维度顺序
        if self.data_format == "CHW":
            if len(tensor.shape) == 4:  # (1, H, W, C)
                tensor = tensor.permute(0, 3, 1, 2)
            # 如果已经是 (1, 1, H, W) 灰度图，无需调整

        # 添加 batch 维度
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)

        return tensor

    def batch_preprocess(self, images: list) -> torch.Tensor:
        """
        批量预处理图像

        Args:
            images: 图像列表

        Returns:
            torch.Tensor: 批量处理后的图像 (B, C, H, W) 或 (B, H, W, C)
        """
        processed = [self.preprocess(img).squeeze(0) for img in images]
        batch = torch.stack(processed, dim=0)
        return batch

    def postprocess(self, tensor: torch.Tensor) -> np.ndarray:
        """
        将 Tensor 转换回 numpy 数组（用于可视化）

        Args:
            tensor: 预处理后的 Tensor

        Returns:
            np.ndarray: 可用于显示的图像 (H, W, 3)，uint8
        """
        # 移除 batch 维度
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0)

        # 转换为 numpy
        array = tensor.cpu().numpy()

        # 调整维度顺序
        if self.data_format == "CHW":
            if len(array.shape) == 3:
                array = array.transpose(1, 2, 0)

        # 反归一化
        if self.normalize:
            if self.mean is not None and self.std is not None:
                array = array * self.std + self.mean
            array = array * 255.0

        # 转换为 uint8
        array = np.clip(array, 0, 255).astype(np.uint8)

        return array


# 创建默认预处理器实例
_default_preprocessor: Optional[ImagePreprocessor] = None


def get_default_preprocessor() -> ImagePreprocessor:
    """获取默认预处理器实例（单例模式）"""
    global _default_preprocessor
    if _default_preprocessor is None:
        _default_preprocessor = ImagePreprocessor()
    return _default_preprocessor


# 便捷函数
def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
) -> torch.Tensor:
    """
    便捷的图像预处理函数

    Args:
        image: 输入图像
        target_size: 目标尺寸
        normalize: 是否归一化

    Returns:
        预处理后的 Tensor
    """
    preprocessor = ImagePreprocessor(target_size=target_size, normalize=normalize)
    return preprocessor.preprocess(image)
