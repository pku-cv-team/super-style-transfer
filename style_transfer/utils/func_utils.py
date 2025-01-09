"""函数工具模块"""

from functools import wraps


def my_check_return_type(expected):
    """检查返回值类型的装饰器"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            value = func(*args, **kwargs)
            if not isinstance(value, expected):
                raise TypeError(
                    f"{func.__name__} return type should be {expected}, but got {type(value)}."
                )
            return value

        return wrapper

    return decorator
