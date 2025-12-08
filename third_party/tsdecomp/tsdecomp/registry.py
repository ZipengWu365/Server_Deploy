from typing import Callable, Dict, Any

METHODS: Dict[str, Callable] = {}

def register(name: str):
    """
    Decorator to register a decomposition method.
    """
    def decorator(fn: Callable):
        METHODS[name] = fn
        return fn
    return decorator
