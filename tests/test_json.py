# pylint: disable=all

import pytest
import json
from unittest.mock import mock_open, patch
from style_transfer.utils.json_loader import JsonLoader  # 替换为JsonLoader类所在的模块


def test_load_valid_key():
    json_data = {"name": "John", "age": 30}
    with patch("builtins.open", mock_open(read_data=json.dumps(json_data))):
        loader = JsonLoader("fake_path.json")
        result = loader.load("name")
        assert result == "John"


def test_load_invalid_key():
    json_data = {"name": "John", "age": 30}
    with patch("builtins.open", mock_open(read_data=json.dumps(json_data))):
        loader = JsonLoader("fake_path.json")
        with pytest.raises(KeyError):
            loader.load("address")


def test_load_empty_json():
    json_data = {}
    with patch("builtins.open", mock_open(read_data=json.dumps(json_data))):
        loader = JsonLoader("fake_path.json")
        with pytest.raises(KeyError):
            loader.load("name")
