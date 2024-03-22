from functools import wraps
from typing import Any, Callable


def repeat_func(number_of_times: int) -> Any:
    def decorate(func: Callable) -> Any:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> None:
            for _ in range(number_of_times):
                func(*args, **kwargs)

        return wrapper

    return decorate
