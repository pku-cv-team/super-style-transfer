"""Json加载器，用于加载json文件"""

import json
import logging


# pylint: disable=too-few-public-methods
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
