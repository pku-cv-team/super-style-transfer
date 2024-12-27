"""Json加载器，用于加载json文件"""

import json


# pylint: disable=too-few-public-methods
class JsonLoader:
    """Json加载器"""

    def __init__(self, json_path):
        self.json_path = json_path

    def load(self, key):
        """加载json文件中的key对应的值"""
        with open(self.json_path, encoding="utf-8") as json_file:
            return json.load(json_file)[key]
