from typing import Any
from typing_extensions import Self

class local:
    def __init__(self, *args: object, **kwargs: object) -> None: ...
    def __copy__(self) -> Self: ...
    def __getattribute__(self, name: str) -> Any: ...
    def __delattr__(self, name: str) -> None: ...
    def __setattr__(self, name: str, value: Any) -> None: ...

__all__ = ["local"]
