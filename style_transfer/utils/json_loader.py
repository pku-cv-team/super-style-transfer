"""Json加载器，用于加载json文件"""

import json
from style_transfer.utils.func_utils import my_not_none


class JsonLoader:
    """Json加载器"""

    def __init__(self, json_path):
        self.json_path = json_path

    def load(self, key):
        """加载json文件中的key对应的值"""
        with open(self.json_path, encoding="utf-8") as json_file:
            return json.load(json_file)[key]

    @my_not_none
    def load_not_none(self, key):
        """加载json文件中的key对应的值，如果为None则抛出异常"""
        return self.load(key)
