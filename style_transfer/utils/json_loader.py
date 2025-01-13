"""Json加载器，用于加载json文件"""

import json
import logging
from typing import List
from style_transfer.utils.func_utils import my_not_none


class JsonLoader:
    """Json加载器"""

    def __init__(self, json_path):
        with open(json_path, encoding="utf-8") as json_file:
            self.__json_obj = json.load(json_file)

    def load(self, key):
        """加载json文件中的key对应的值"""
        if key in self.__json_obj:
            return self.__json_obj[key]
        logging.warning("Key %s not found in json file.", key)
        return None

    @my_not_none
    def load_not_none(self, key):
        """加载json文件中的key对应的值，如果为None则抛出异常"""
        return self.load(key)

    @my_not_none
    def load_resize_stragety(self) -> dict:
        """加载调整图像尺寸的策略"""
        stragety = self.load_not_none("resize_stragety")
        if "type" not in stragety:
            raise ValueError("Resize stragety should have type.")
        stragety_type = stragety["type"]
        if "size" not in stragety:
            raise ValueError("Resize stragety should have size.")
        size = stragety["size"]
        if stragety_type == "trivial":
            return {"type": stragety_type, "size": size}
        if stragety_type == "pyramid":
            return {"type": stragety_type, "size": size}
        if stragety_type == "srcnn":
            if "scale" not in stragety:
                raise ValueError("Resize stragety srcnn should have scale.")
            scale = stragety["scale"]
            if "model_path" not in stragety:
                raise ValueError("Resize stragety srcnn should have model_path.")
            model_path = stragety["model_path"]
            return {
                "type": stragety_type,
                "size": size,
                "scale": scale,
                "model_path": model_path,
            }
        raise ValueError("Unsupported stragety")

    @staticmethod
    def __load_feature_extractor_param(json_obj: json) -> dict:
        """加载特征提取器参数"""
        feature_extractor_param = json_obj.get("feature_extractor")
        if "type" not in feature_extractor_param:
            raise ValueError("Feature extractor should have type.")
        feature_extractor_type = feature_extractor_param["type"]
        if feature_extractor_type == "vgg19":
            return {
                "type": feature_extractor_type,
                "content_layers": feature_extractor_param.get("content_layers"),
                "style_layers": feature_extractor_param.get("style_layers"),
            }
        raise ValueError("Unsupported feature extractor type.")

    @my_not_none
    def load_style_transfer_param(self) -> dict:
        """加载风格迁移参数"""
        style_transfer_param = self.load_not_none("model_config")
        if "type" not in style_transfer_param:
            raise ValueError("Style transfer should have type.")
        style_transfer_type = style_transfer_param["type"]
        if style_transfer_type == "gatys":
            return {
                "type": style_transfer_type,
                "content_weight": style_transfer_param.get("content_weight", 1.0),
                "style_weight": style_transfer_param.get("style_weight", [1e4]),
                "feature_extractor": self.__load_feature_extractor_param(
                    style_transfer_param
                ),
                "content_layer_weights": style_transfer_param.get(
                    "content_layer_weights"
                ),
                "style_layer_weights": style_transfer_param.get("style_layer_weights"),
                "decorator": self.__load_decorator_params(style_transfer_param),
                "init_strategy": style_transfer_param.get("init_strategy"),
            }
        raise ValueError("Unsupported style transfer type.")

    @staticmethod
    def __load_decorator_params(json_obj: json) -> List[dict]:
        """加载装饰器参数"""
        decorators = json_obj.get("additional_loss", [])
        return [
            JsonLoader.__load_decorator_param(decorator) for decorator in decorators
        ]

    @staticmethod
    def __load_decorator_param(json_obj: json) -> dict:
        """加载装饰器参数"""
        if "type" not in json_obj:
            raise ValueError("Decorator should have type.")
        decorator_type = json_obj["type"]
        if decorator_type == "lap_loss":
            return {
                "type": decorator_type,
                "pool_size": json_obj.get("pool_size", 3),
                "lap_weight": json_obj.get("lap_weight", 1e-2),
            }
        if decorator_type == "tv_loss":
            return {
                "type": decorator_type,
                "tv_weight": json_obj.get("tv_weight", 1e-2),
            }
        raise ValueError("Unsupported decorator type.")
