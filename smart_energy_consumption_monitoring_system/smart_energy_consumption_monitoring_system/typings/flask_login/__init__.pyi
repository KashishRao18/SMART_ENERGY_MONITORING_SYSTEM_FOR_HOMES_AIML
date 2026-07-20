from flask import Flask
from typing import Any, Callable, TypeVar

F = TypeVar('F', bound=Callable[..., Any])

def login_required(f: F) -> F:
    """Decorator to require login for a route."""
    return f
