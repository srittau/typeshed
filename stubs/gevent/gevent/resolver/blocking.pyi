from gevent._types import _AddrinfoResult, _NameinfoResult, _SockAddr
from gevent.hub import Hub

class Resolver:
    def __init__(self, hub: Hub | None = None) -> None: ...
    def close(self) -> None: ...
    def gethostbyname(self, hostname: str, family: int = 2) -> str: ...
    def gethostbyname_ex(self, hostname: str, family: int = 2) -> tuple[str, list[str], list[str]]: ...
    def getaddrinfo(
        self, host: str, port: int, family: int = 0, socktype: int = 0, proto: int = 0, flags: int = 0
    ) -> _AddrinfoResult: ...
    def gethostbyaddr(self, ip_address: str) -> tuple[str, list[str], list[str]]: ...
    def getnameinfo(self, sockaddr: _SockAddr, flags: int) -> _NameinfoResult: ...

__all__ = ["Resolver"]
