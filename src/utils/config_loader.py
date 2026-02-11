"""
配置加载器

支持从 YAML 文件和环境变量加载配置
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


class ConfigLoader:
    """配置加载器"""

    def __init__(self, config_path: Optional[str] = None, project_root: Optional[str] = None):
        """
        初始化配置加载器

        Args:
            config_path: 配置文件路径
            project_root: 项目根目录
        """
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        else:
            project_root = Path(project_root)

        self.project_root = project_root
        self.config_path = config_path or str(project_root / "config" / "model_config.yaml")
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        config_file = Path(self.config_path)

        if not config_file.exists():
            print(f"⚠️ 配置文件不存在: {config_file}")
            return self._get_default_config()

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✓ 配置文件已加载: {config_file}")
            return config
        except Exception as e:
            print(f"⚠️ 配置文件加载失败: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "vlm_backbone": {
                "model_name": "Llava-Pythia-400M",
                "huggingface_id": "lesjie/Llava-Pythia-400M",
            },
            "inference": {
                "device": "cpu",
                "dtype": "float32",
                "use_cache": True,
            },
            "paths": {
                "models_dir": "models",
                "cache_dir": "models/llava_pythia",
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项（支持嵌套键，如 'inference.device'）

        Args:
            key: 配置键（支持点号分隔的嵌套键）
            default: 默认值

        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_model_path(self, variant: str = "small") -> Optional[str]:
        """
        获取模型路径

        Args:
            variant: 模型变体 (small, base, large)

        Returns:
            模型路径或 None
        """
        variant_config = self.get(f"vlm_backbone.variants.{variant}")
        if variant_config:
            local_path = variant_config.get("local_path")
            if local_path:
                full_path = self.project_root / local_path
                if full_path.exists():
                    return str(full_path)

        return None

    def get_model_id(self, variant: str = "small") -> str:
        """
        获取 HuggingFace 模型 ID

        Args:
            variant: 模型变体

        Returns:
            模型 ID
        """
        return self.get(f"vlm_backbone.variants.{variant}.model_id", "lesjie/Llava-Pythia-400M")

    def get_device(self) -> str:
        """获取推理设备"""
        # 优先从环境变量读取
        device = os.environ.get("TINYVLA_DEVICE")
        if device:
            return device

        return self.get("inference.device", "cpu")

    def get_dtype(self) -> str:
        """获取数据类型"""
        dtype = os.environ.get("TINYVLA_DTYPE")
        if dtype:
            return dtype

        return self.get("inference.dtype", "float32")

    def get_cache_dir(self) -> str:
        """获取缓存目录"""
        cache_dir = os.environ.get("TINYVLA_CACHE_DIR")
        if cache_dir:
            return str(self.project_root / cache_dir)

        cache_dir = self.get("paths.cache_dir", "models/llava_pythia")
        return str(self.project_root / cache_dir)

    def list_models(self) -> Dict[str, Dict[str, str]]:
        """
        列出所有可用模型

        Returns:
            模型信息字典
        """
        variants = self.get("vlm_backbone.variants", {})
        models = {}

        for variant_name, variant_config in variants.items():
            models[variant_name] = {
                "name": variant_config.get("name"),
                "params": variant_config.get("params"),
                "model_id": variant_config.get("model_id"),
                "local_path": variant_config.get("local_path"),
            }

        return models

    def save_env_file(self, env_path: Optional[str] = None):
        """
        保存环境变量配置到 .env 文件

        Args:
            env_path: .env 文件路径
        """
        if env_path is None:
            env_path = self.project_root / ".env.local"

        env_content = f"""# TinyVLA 环境变量配置

# 模型路径
export TINYVLA_MODELS_DIR="{self.get('paths.models_dir', 'models')}"
export TINYVLA_CACHE_DIR="{self.get('paths.cache_dir', 'models/llava_pythia')}"
export TINYVLA_DEFAULT_MODEL="{self.get('vlm_backbone.huggingface_id', 'lesjie/Llava-Pythia-400M')}"

# 设备配置
export TINYVLA_DEVICE="{self.get_device()}"
export TINYVLA_DTYPE="{self.get_dtype()}"

# IK 配置
export TINYVLA_IK_METHOD="{os.environ.get('TINYVLA_IK_METHOD', 'optimization')}"
export TINYVLA_USE_IK="{os.environ.get('TINYVLA_USE_IK', 'true')}"

# 日志配置
export TINYVLA_LOG_LEVEL="{os.environ.get('TINYVLA_LOG_LEVEL', 'INFO')}"
"""

        with open(env_path, 'w', encoding='utf-8') as f:
            f.write(env_content)

        print(f"✓ 环境变量已保存: {env_path}")


def load_config(config_path: Optional[str] = None) -> ConfigLoader:
    """
    加载配置（便捷函数）

    Args:
        config_path: 配置文件路径

    Returns:
        ConfigLoader 实例
    """
    return ConfigLoader(config_path=config_path)
