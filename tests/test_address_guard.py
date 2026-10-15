import ipaddress
import socket

import pytest

from conftest import LOOPBACK, FakeNet, LocalServer, port_of
from noinjections.fetchers import address_guard
from noinjections.fetchers.address_guard import is_public_address
from noinjections.fetchers.web import WebFetchError, fetch_url


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "127.8.9.10",
        "0.0.0.0",  # noqa: S104 - just an address under test
        "10.0.0.5",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.169.254",  # cloud metadata
        "100.64.0.1",  # carrier-grade NAT
        "192.0.2.1",  # documentation
        "198.18.0.1",  # benchmarking
        "240.0.0.1",  # reserved
        "255.255.255.255",
        "224.0.0.1",  # multicast
        "239.1.1.1",
        "::",
        "::1",
        "fe80::1",
        "fc00::1",
        "fd12:3456::1",
        "2001:db8::1",
        "ff02::1",
        "::ffff:127.0.0.1",  # IPv4-mapped
        "::ffff:169.254.169.254",
        "64:ff9b::a00:5",  # NAT64 of 10.0.0.5
        "2002:a00:5::",  # 6to4 of 10.0.0.5
        "2002:a9fe:a9fe::",  # 6to4 of 169.254.169.254
    ],
)
def test_non_public_addresses(address: str) -> None:
    assert not is_public_address(ipaddress.ip_address(address))


@pytest.mark.parametrize(
    "address",
    [
        "8.8.8.8",
        "93.184.215.14",
        "2606:4700::1111",
        "::ffff:8.8.8.8",
        "64:ff9b::808:808",  # NAT64 of 8.8.8.8
        "2002:808:808::",  # 6to4 of 8.8.8.8
    ],
)
def test_public_addresses(address: str) -> None:
    assert is_public_address(ipaddress.ip_address(address))


def test_connects_to_the_checked_address(local_server: LocalServer, fake_net: FakeNet) -> None:
    fake_net.answers["public.example"] = [LOOPBACK]
    url = f"http://public.example:{port_of(local_server)}/index.html"
    page = fetch_url(url)
    assert "Example Garden Supplies" in page.text
    # The name was never looked up for real: the socket went to the address that was checked.
    assert local_server.requests[-1].headers["Host"] == f"public.example:{port_of(local_server)}"


def test_redirect_to_a_private_host_is_refused(
    local_server: LocalServer, fake_net: FakeNet
) -> None:
    fake_net.answers["intranet.evil.example"] = ["10.0.0.5"]
    url = local_server.redirect("/r", "http://intranet.evil.example/admin")
    with pytest.raises(WebFetchError, match=r"intranet\.evil\.example resolves to 10\.0\.0\.5"):
        fetch_url(url)
    assert [r.path for r in local_server.requests] == ["/r"]


@pytest.mark.parametrize(
    ("location", "address"),
    [
        ("http://169.254.169.254/latest/meta-data/", "169.254.169.254"),
        ("http://[::ffff:10.0.0.5]/", "::ffff:10.0.0.5"),
        ("http://[fd00::1]:8080/", "fd00::1"),
    ],
)
def test_redirect_to_a_private_ip_literal_is_refused(
    local_server: LocalServer, fake_net: FakeNet, location: str, address: str
) -> None:
    url = local_server.redirect("/r", location)
    with pytest.raises(WebFetchError, match=f"non-public address \\({address}\\)"):
        fetch_url(url)


def test_any_private_answer_refuses_the_host(local_server: LocalServer, fake_net: FakeNet) -> None:
    # A name that mixes public and private answers is a DNS-rebinding setup.
    fake_net.answers["mixed.evil.example"] = [LOOPBACK, "10.0.0.5"]
    with pytest.raises(WebFetchError, match=r"mixed\.evil\.example resolves to 10\.0\.0\.5"):
        fetch_url(f"http://mixed.evil.example:{port_of(local_server)}/index.html")
    assert local_server.requests == []


def test_falls_back_to_the_next_address(local_server: LocalServer, fake_net: FakeNet) -> None:
    # The server listens on 127.0.0.1 only, so ::1 is refused and 127.0.0.1 is tried next.
    fake_net.answers["dual.example"] = ["::1", LOOPBACK]
    fake_net.public.add("::1")
    page = fetch_url(f"http://dual.example:{port_of(local_server)}/index.html")
    assert "Example Garden Supplies" in page.text


def test_unresolvable_host_is_a_connect_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> list[object]:
        raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")

    monkeypatch.setattr(address_guard.socket, "getaddrinfo", fail)
    with pytest.raises(WebFetchError, match=r"cannot resolve 'nowhere\.example'"):
        fetch_url("http://nowhere.example/")


def test_proxy_variables_cannot_route_around_the_guard(
    local_server: LocalServer, fake_net: FakeNet, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Through a proxy the guard couldn't see the real destination, so proxies aren't used.
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:9")
    page = fetch_url(local_server.url("/index.html"))
    assert page.status_code == 200
