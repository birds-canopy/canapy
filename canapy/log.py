import logging

from functools import wraps

logger = logging.getLogger("canapy")


def log(fn_type):
    def decorator(fn):
        @wraps(fn)
        def fn_wrapper(*args, **kwargs):
            logger.info(f"Applying {fn_type} {fn.__name__} on {args, kwargs}.")
            res = fn(*args, **kwargs)
            return res

        return fn_wrapper

    return decorator
