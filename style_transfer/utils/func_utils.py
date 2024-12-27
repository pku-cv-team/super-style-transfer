"""函数工具模块"""

from functools import wraps


def my_not_none(func):
    """如果返回值为None则抛出异常的装饰器"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        value = func(*args, **kwargs)
        if value is None:
            raise ValueError(f"{func.__name__} is None.")
        return value

    return wrapper
