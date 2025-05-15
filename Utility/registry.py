from typing import Callable, Dict, Any
FACTOR_REGISTRY: Dict[str, Dict[str, Any]] = {}
def factor(*tags: str):
    def decorator(fn: Callable):
        FACTOR_REGISTRY[fn.__name__] = {'func': fn, 'tags': set(tags)}
        return fn
    return decorator