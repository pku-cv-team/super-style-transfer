"""Json加载器，用于加载json文件"""

import json
import logging
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
        if stragety_type == "pyrimid":
            return {"type:": stragety_type, "size": size}
        raise ValueError("Unsupported stragety")
