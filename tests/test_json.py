"""测试JsonLoader类的功能
本部分测试由生成式人工智能生成
"""

import json
from unittest.mock import mock_open, patch
from style_transfer.utils.json_loader import JsonLoader


def test_load_valid_key():
    """测试加载有效的key"""
    json_data = {"name": "John", "age": 30}
    with patch("builtins.open", mock_open(read_data=json.dumps(json_data))):
        loader = JsonLoader("fake_path.json")
        result = loader.load("name")
        assert result == "John"


def test_load_invalid_key():
    """测试加载无效的key"""
    json_data = {"name": "John", "age": 30}
    with patch("builtins.open", mock_open(read_data=json.dumps(json_data))):
        loader = JsonLoader("fake_path.json")
        assert loader.load("address") is None


def test_load_empty_json():
    """测试加载空的json"""
    json_data = {}
    with patch("builtins.open", mock_open(read_data=json.dumps(json_data))):
        loader = JsonLoader("fake_path.json")
        assert loader.load("name") is None
