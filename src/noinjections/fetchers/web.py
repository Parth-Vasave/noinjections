"""Fetch a web page over HTTP(S), without running JavaScript."""

import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from noinjections import __version__
from noinjections.fetchers.address_guard import BlockedAddressError, public_only_client
from noinjections.fetchers.content import ContentRejected, decode_text
from noinjections.targets import TargetError, normalize_url

DEFAULT_TIMEOUT_SECONDS = 30.0
"""Default time budget for a whole fetch: connecting, every redirect, and reading the body."""

DEFAULT_MAX_REDIRECTS = 10

DEFAULT_MAX_PAGE_BYTES = 5 * 1024 * 1024
"""Default cap on a page's body (5 MiB, after decompression). Longer bodies are cut off there."""

USER_AGENT = f"noinjections/{__version__}"

HTML_MEDIA_TYPES = frozenset({"text/html", "application/xhtml+xml"})

_ACCEPT = "text/html,application/xhtml+xml,text/plain;q=0.9,text/*;q=0.8"


class WebFetchError(Exception):
    """The URL could not be fetched, or its response can't be scanned."""

    def __init__(self, url: str, message: str) -> None:
        super().__init__(f"{url}: {message}")
        self.url = url
        self.message = message


@dataclass(frozen=True, slots=True)
class FetchedBytes:
    """A successful response body, read up to a byte cap.

    ``final_url`` is normalised (see :func:`~noinjections.targets.normalize_url`). ``media_type``
    is the lowercased Content-Type without parameters, or ``""`` if the header is missing.
    ``charset`` is the declared charset if Python has a text codec for it, else ``None``.
    ``received`` counts the body bytes read, which can be more than ``len(data)``.
    """

    final_url: str
    status_code: int
    media_type: str
    charset: str | None
    data: bytes
    received: int

    @property
    def truncated(self) -> bool:
        """Whether the body was longer than the cap and ``data`` holds only its start."""
        return self.received > len(self.data)


@dataclass(frozen=True, slots=True)
class WebPage:
    """A fetched page. ``text`` is the raw HTML (or plain text) as served, before any parsing.

    ``url`` is the requested URL and ``final_url`` the one after redirects, both normalised.
    ``truncated`` means the body was longer than the size cap and ``text`` holds only its start.
    """

    url: str
    final_url: str
    status_code: int
    media_type: str
    text: str
    truncated: bool = False

    @property
    def is_html(self) -> bool:
        """Whether the server said the content is HTML."""
        return self.media_type in HTML_MEDIA_TYPES


def fetch_url(
    url: str,
    *,
    max_bytes: int = DEFAULT_MAX_PAGE_BYTES,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    allow_private: bool = False,
    client: httpx.Client | None = None,
) -> WebPage:
    """Fetch ``url`` and return the page's raw text and its final URL.

    HTML and other ``text/*`` content types are accepted; anything else (images, PDFs, JSON)
    raises :class:`WebFetchError`, as do network errors, timeouts, HTTP errors, and redirect
    problems. A body over ``max_bytes`` is read up to the cap and marked ``truncated``. The text
    is decoded with the declared charset, falling back to UTF-8. A response with no Content-Type
    is sniffed and rejected if it looks binary. Only public addresses are contacted unless
    ``allow_private`` is set (see :func:`fetch_bytes`). Raises
    :class:`~noinjections.targets.TargetError` if ``url`` itself is invalid.
    """
    url = normalize_url(url)
    response = fetch_bytes(
        url,
        max_bytes=max_bytes,
        timeout=timeout,
        max_redirects=max_redirects,
        allow_private=allow_private,
        client=client,
        accepts=_is_page_type,
    )
    # A declared text type means no binary sniff: see decode_text.
    encoding = response.charset or ("utf-8" if response.media_type else None)
    try:
        text = decode_text(response.data, truncated=response.truncated, encoding=encoding)
    except ContentRejected as exc:
        raise WebFetchError(url, "response has no Content-Type and looks binary") from exc
    return WebPage(
        url=url,
        final_url=response.final_url,
        status_code=response.status_code,
        media_type=response.media_type,
        text=text,
        truncated=response.truncated,
    )


def fetch_bytes(
    url: str,
    *,
    max_bytes: int,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    allow_private: bool = False,
    client: httpx.Client | None = None,
    accepts: Callable[[str], bool] | None = None,
) -> FetchedBytes:
    """GET ``url`` and read up to ``max_bytes`` of the body; shared by the HTTP-based fetchers.

    Sends the noinjections User-Agent and follows up to ``max_redirects`` redirects. Every hop
    must be a plain ``http(s)`` URL with no credentials, and redirect bodies are never read.
    ``timeout`` bounds the whole fetch; it is checked between body chunks, so a single stalled
    read can overrun it by up to ``timeout`` again. If ``accepts`` is given, it is called with
    the media type before the body is read, and ``False`` raises :class:`WebFetchError`.
    Non-2xx responses raise :class:`WebFetchError`.

    Every connection, including each redirect hop, goes only to public IP addresses (see
    :mod:`~noinjections.fetchers.address_guard`); a hop that resolves to a loopback, private,
    or link-local address raises :class:`WebFetchError`. ``allow_private=True`` lifts that and
    honours proxy environment variables again. A ``client`` passed in is used as-is, with
    whatever transport it has: the guard applies only to the client created here.
    """
    if max_bytes < 1:
        raise ValueError(f"max_bytes must be >= 1, got {max_bytes}")
    if not timeout > 0:
        raise ValueError(f"timeout must be > 0, got {timeout}")
    if max_redirects < 0:
        raise ValueError(f"max_redirects must be >= 0, got {max_redirects}")
    deadline = time.monotonic() + timeout
    own_client = client is None
    if client is not None:
        http = client
    else:
        http = httpx.Client() if allow_private else public_only_client()
    try:
        return _get(http, url, max_bytes, timeout, deadline, max_redirects, accepts)
    except BlockedAddressError as exc:
        raise WebFetchError(url, str(exc)) from exc
    except httpx.TimeoutException as exc:
        raise WebFetchError(url, f"timed out after {timeout:g}s") from exc
    except (httpx.HTTPError, httpx.InvalidURL) as exc:
        raise WebFetchError(url, f"{type(exc).__name__}: {exc}") from exc
    finally:
        if own_client:
            http.close()


def _get(
    client: httpx.Client,
    url: str,
    max_bytes: int,
    timeout: float,
    deadline: float,
    max_redirects: int,
    accepts: Callable[[str], bool] | None,
) -> FetchedBytes:
    # Redirects are followed by hand: httpx would read each redirect's body in full, and this
    # way every hop is checked before it is requested.
    request = client.build_request(
        "GET", url, headers={"User-Agent": USER_AGENT, "Accept": _ACCEPT}, timeout=timeout
    )
    for hop in range(max_redirects + 1):
        _check_deadline(url, timeout, deadline)
        response = client.send(request, stream=True, follow_redirects=False)
        try:
            if not response.has_redirect_location:
                return _read_response(response, url, max_bytes, timeout, deadline, accepts)
            next_request = response.next_request
        finally:
            response.close()
        if next_request is None or hop == max_redirects:
            break
        try:
            normalize_url(str(next_request.url))
        except TargetError as exc:
            raise WebFetchError(url, f"redirected to an unusable URL: {exc}") from exc
        request = next_request
    raise WebFetchError(url, f"more than {max_redirects} redirects")


def _read_response(
    response: httpx.Response,
    url: str,
    max_bytes: int,
    timeout: float,
    deadline: float,
    accepts: Callable[[str], bool] | None,
) -> FetchedBytes:
    if not response.is_success:
        raise WebFetchError(url, f"HTTP {response.status_code} {response.reason_phrase}".strip())
    media_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
    if accepts is not None and not accepts(media_type):
        raise WebFetchError(url, f"unsupported content type {media_type!r}")

    chunks: list[bytes] = []
    received = 0
    for chunk in response.iter_bytes():  # decompressed, so the cap bounds the real size
        chunks.append(chunk)
        received += len(chunk)
        if received > max_bytes:
            break
        _check_deadline(url, timeout, deadline)
    data = b"".join(chunks)[:max_bytes]
    return FetchedBytes(
        final_url=normalize_url(str(response.url)),
        status_code=response.status_code,
        media_type=media_type,
        charset=_text_codec(response.charset_encoding),
        data=data,
        received=received,
    )


def _check_deadline(url: str, timeout: float, deadline: float) -> None:
    if time.monotonic() > deadline:
        raise WebFetchError(url, f"timed out after {timeout:g}s")


def _is_page_type(media_type: str) -> bool:
    return not media_type or media_type in HTML_MEDIA_TYPES or media_type.startswith("text/")


def _text_codec(charset: str | None) -> str | None:
    """``charset`` if Python has a text codec of that name, else ``None``."""
    if not charset:
        return None
    try:
        # Decoding a non-empty input also rejects bytes-to-bytes codecs such as "zlib".
        b" ".decode(charset, errors="replace")
    except LookupError:
        return None
    return charset
