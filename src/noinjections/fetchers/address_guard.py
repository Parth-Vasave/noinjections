"""Keep HTTP fetches on the public internet (SSRF protection).

A scanned page, or a redirect from it, must not make noinjections read ``localhost``, a private
network, or a cloud metadata endpoint (``169.254.169.254``) and put the answer in a report. The
check runs where the socket is opened, on the exact addresses it connects to, so it covers every
redirect hop and a DNS answer can't change between the check and the connection (rebinding).
"""

import ipaddress
import socket
import typing

import httpcore
import httpx

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address

_NAT64 = ipaddress.ip_network("64:ff9b::/96")


class BlockedAddressError(Exception):
    """A connection to a non-public address was refused."""

    def __init__(self, host: str, address: str) -> None:
        detail = address if host == address else f"{host} resolves to {address}"
        super().__init__(f"refused to connect to a non-public address ({detail})")
        self.host = host
        self.address = address


def is_public_address(ip: IPAddress) -> bool:
    """Whether ``ip`` is a globally routable unicast address.

    Loopback, private, link-local, carrier-grade NAT, documentation, reserved, and multicast
    addresses are not public. IPv4 addresses embedded in IPv6 (``::ffff:10.0.0.1``, NAT64
    ``64:ff9b::10.0.0.1``, 6to4 ``2002:a00:1::``) are judged by the IPv4 address.
    """
    if isinstance(ip, ipaddress.IPv6Address):
        # Unwrapped explicitly: Python's own tables for these ranges changed between patch
        # releases (3.11.9 calls 6to4 of 10.0.0.5 global), so the answer must not rely on them.
        if ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        elif ip.sixtofour is not None:
            ip = ip.sixtofour
        elif ip in _NAT64:
            ip = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    return ip.is_global and not ip.is_multicast


class PublicOnlyBackend(httpcore.NetworkBackend):
    """An httpcore network backend that connects only to public addresses.

    The host is resolved here, and if **any** of its addresses is not public the connection is
    refused (a name mixing public and private answers is a rebinding setup). The socket is then
    opened to the checked IP itself, so nothing is resolved again. TLS still verifies the
    certificate against the host name: httpcore passes the origin's name to ``start_tls``.
    """

    def __init__(self) -> None:
        self._inner = httpcore.SyncBackend()

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: typing.Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.NetworkStream:
        addresses = _resolve(host, port)
        for address in addresses:
            if not is_public_address(ipaddress.ip_address(address)):
                raise BlockedAddressError(host, address)
        *fallbacks, last = addresses
        options = list(socket_options or [])
        for address in fallbacks:
            try:
                return self._inner.connect_tcp(address, port, timeout, local_address, options)
            except (httpcore.ConnectError, httpcore.ConnectTimeout):
                continue
        return self._inner.connect_tcp(last, port, timeout, local_address, options)

    def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: typing.Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.NetworkStream:
        raise BlockedAddressError(path, path)

    def sleep(self, seconds: float) -> None:
        self._inner.sleep(seconds)


def public_only_client() -> httpx.Client:
    """An :class:`httpx.Client` whose connections are limited by :class:`PublicOnlyBackend`.

    Passing a ``transport`` turns off httpx's environment proxies (``HTTPS_PROXY`` etc.), which
    is required: through a proxy, the proxy resolves the target and the guard can't see where
    the connection really goes.
    """
    transport = httpx.HTTPTransport()
    # httpx has no option for a custom network backend, so the transport's connection pool is
    # replaced with one that uses the guard (same TLS context and limits as httpx's defaults).
    transport._pool = httpcore.ConnectionPool(
        ssl_context=httpx.create_ssl_context(),
        max_connections=100,
        max_keepalive_connections=20,
        keepalive_expiry=5.0,
        network_backend=PublicOnlyBackend(),
    )
    return httpx.Client(transport=transport)


def _resolve(host: str, port: int) -> list[str]:
    """The IP addresses ``host`` resolves to, deduplicated, in resolver order."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError) as exc:
        raise httpcore.ConnectError(f"cannot resolve {host!r}: {exc}") from exc
    addresses = list(dict.fromkeys(str(info[4][0]) for info in infos))
    if not addresses:
        raise httpcore.ConnectError(f"cannot resolve {host!r}: no addresses")
    return addresses
