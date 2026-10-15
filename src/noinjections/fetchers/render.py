"""Fetch a web page in headless Chromium and return its DOM after the page's JavaScript has run.

Chromium never fetches anything itself. Every request it makes (the page, scripts, frames, XHRs)
is intercepted and fetched with noinjections's own HTTP client, so the address guard
(:mod:`~noinjections.fetchers.address_guard`), the redirect checks, and the size caps of
:mod:`~noinjections.fetchers.web` apply to all of them. Redirects are followed on the Python side
and never handed to Chromium, which would follow them without going through the interception.
As a backstop, Chromium is pointed at a proxy that refuses every connection, so a request that
escapes interception (a SharedWorker's fetch, a WebSocket) fails instead of reaching the network.

Needs the ``render`` extra and Playwright's Chromium; see :class:`RenderUnavailableError`.
"""

import asyncio
import contextlib
import functools
import secrets
import socket
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib import resources
from typing import TYPE_CHECKING, Any

import httpx

from noinjections.fetchers.content import decode_text
from noinjections.fetchers.web import (
    DEFAULT_MAX_PAGE_BYTES,
    DEFAULT_MAX_REDIRECTS,
    DEFAULT_TIMEOUT_SECONDS,
    HTML_MEDIA_TYPES,
    FetchedBytes,
    WebFetchError,
    WebPage,
    fetch_request,
    guarded_client,
    is_page_type,
    page_from_response,
)
from noinjections.targets import TargetError, normalize_url

if TYPE_CHECKING:
    from playwright.async_api import Browser, Frame, Page, Playwright, Request, Route

DEFAULT_IDLE_TIMEOUT_SECONDS = 10.0
"""Default time to wait for the network to go quiet after the page's HTML has loaded."""

INSTALL_HINT = "pip install 'noinjections[render]' && playwright install chromium"

SKIPPED_RESOURCE_TYPES = frozenset({"image", "media", "font", "eventsource", "ping"})
"""Request types refused without fetching: they carry no page text (images, fonts, beacons) or
never finish (EventSource)."""

_DROPPED_REQUEST_HEADERS = frozenset(
    {
        "accept-encoding",  # httpx asks only for encodings it can decode
        "connection",
        "content-length",
        "host",
        "keep-alive",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)

_DROPPED_RESPONSE_HEADERS = frozenset(
    {
        "connection",
        "content-encoding",  # httpx has already decoded the body
        "content-length",
        "keep-alive",
        "location",  # redirects are followed here, never by Chromium
        "transfer-encoding",
    }
)

_FETCH_THREADS = 8

_MAX_PAGE_LOADS = 3
"""Loads of the top-level page: the first, plus a retry at the final URL after a redirect."""

HIDDEN_TEXT_MIN_OPACITY = 0.1
"""Text whose opacity, multiplied through its ancestors, is below this is hidden."""

HIDDEN_TEXT_MIN_FONT_PX = 4.0
"""Text with a computed font size below this many pixels is hidden."""

HIDDEN_TEXT_MIN_VISIBLE_PX = 2.0
"""Text showing less than this many pixels across or down (after clipping) is hidden."""

HIDDEN_TEXT_MIN_CONTRAST = 1.5
"""Text whose contrast ratio with its background is below this is hidden (WCAG ratio, 1-21)."""

MAX_CHECKED_ELEMENTS = 50_000
"""Elements with text checked for hiding per page; text past this is counted as unchecked."""

_HIDDEN_TEXT_JS = (resources.files(__package__) / "hidden_text.js").read_text(encoding="utf-8")
"""Marks hidden text and serialises the DOM; see the comment at the top of the file."""

_WORLD_NAME = "noinjections"


class RenderUnavailableError(Exception):
    """Pages can't be rendered here: Playwright or its Chromium is missing or won't start."""


def render_url(
    url: str,
    *,
    max_bytes: int = DEFAULT_MAX_PAGE_BYTES,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    idle_timeout: float = DEFAULT_IDLE_TIMEOUT_SECONDS,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    allow_private: bool = False,
) -> WebPage:
    """Load ``url`` in headless Chromium and return the DOM as HTML once its scripts have run.

    The page is loaded, then given up to ``idle_timeout`` seconds for the network to go quiet;
    a page that never goes quiet (polling, long-lived requests) is captured as it is then.
    ``timeout`` bounds the whole render, including starting Chromium. The page and every
    resource are fetched as in :func:`~noinjections.fetchers.web.fetch_url`: public addresses
    only unless ``allow_private`` is set, at most ``max_redirects`` redirects, and at most
    ``max_bytes`` each. A page over the cap is rendered from its first ``max_bytes``, and a
    rendered DOM over the cap is cut there; either way the result is marked ``truncated``.
    Images, fonts, media, and service workers are not loaded.

    A page served as something other than HTML (``text/plain``, ``llms.txt``) is not rendered:
    the result is the same as from ``fetch_url``. Problems with the page itself raise
    :class:`~noinjections.fetchers.web.WebFetchError`; a missing Playwright or Chromium raises
    :class:`RenderUnavailableError`. Raises :class:`~noinjections.targets.TargetError` if
    ``url`` itself is invalid.
    """
    url = normalize_url(url)
    if max_bytes < 1:
        raise ValueError(f"max_bytes must be >= 1, got {max_bytes}")
    if not timeout > 0:
        raise ValueError(f"timeout must be > 0, got {timeout}")
    if idle_timeout < 0:
        raise ValueError(f"idle_timeout must be >= 0, got {idle_timeout}")
    if max_redirects < 0:
        raise ValueError(f"max_redirects must be >= 0, got {max_redirects}")
    try:
        import playwright.async_api  # noqa: F401 - only checking that it is installed
    except ImportError as exc:
        raise RenderUnavailableError(
            f"rendering pages needs Playwright, which is not installed. Install it with: "
            f"{INSTALL_HINT}"
        ) from exc
    deadline = time.monotonic() + timeout
    return asyncio.run(
        _render(
            url,
            allow_private=allow_private,
            max_bytes=max_bytes,
            timeout=timeout,
            deadline=deadline,
            idle_timeout=idle_timeout,
            max_redirects=max_redirects,
        )
    )


async def _render(
    url: str,
    *,
    allow_private: bool,
    max_bytes: int,
    timeout: float,
    deadline: float,
    idle_timeout: float,
    max_redirects: int,
) -> WebPage:
    from playwright.async_api import async_playwright

    session = _Session(
        url,
        guarded_client(allow_private=allow_private),
        max_bytes=max_bytes,
        timeout=timeout,
        deadline=deadline,
        max_redirects=max_redirects,
    )
    try:
        async with async_playwright() as playwright:
            browser = await _launch(playwright, session)
            try:
                return await _load(browser, session, idle_timeout)
            finally:
                await browser.close()
    finally:
        await session.close()


async def _launch(playwright: "Playwright", session: "_Session") -> "Browser":
    from playwright.async_api import Error as PlaywrightError

    try:
        return await playwright.chromium.launch(
            # Anything that escapes interception goes to a proxy that refuses all connections.
            # Chromium sends loopback around proxies unless told not to; Playwright adds that
            # rule too, but an env var turns it off, so it is set here explicitly.
            proxy={"server": session.blackhole_url, "bypass": "<-loopback>"},
            # WebRTC can send UDP around a proxy; this keeps it on the (dead) proxy too.
            args=["--force-webrtc-ip-handling-policy=disable_non_proxied_udp"],
            timeout=session.remaining_ms(),
        )
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc):
            raise RenderUnavailableError(
                "rendering pages needs Playwright's Chromium, which is not installed. "
                "Install it with: playwright install chromium"
            ) from exc
        raise RenderUnavailableError(f"could not start Chromium: {_first_line(exc)}") from exc


async def _load(browser: "Browser", session: "_Session", idle_timeout: float) -> WebPage:
    from playwright.async_api import Error as PlaywrightError
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError

    context = await browser.new_context(service_workers="block", accept_downloads=False)
    await context.route("**/*", session.handle)
    page = await context.new_page()
    session.main_frame = page.main_frame

    target = session.url
    for _ in range(_MAX_PAGE_LOADS):
        session.redirect = None
        failure: PlaywrightError | None = None
        try:
            await page.goto(target, wait_until="domcontentloaded", timeout=session.remaining_ms())
        except PlaywrightError as exc:
            failure = exc
        if session.error is not None:
            raise session.error
        if session.plain is not None:
            return page_from_response(session.url, session.plain)
        if failure is None:
            break
        if session.redirect is None:
            if isinstance(failure, PlaywrightTimeoutError):
                raise WebFetchError(session.url, f"timed out after {session.timeout:g}s")
            raise WebFetchError(session.url, _first_line(failure))
        target = session.redirect  # already fetched: served from the cache this time
    else:
        raise WebFetchError(session.url, "the page kept redirecting")

    idle_ms = min(idle_timeout * 1000, session.remaining_ms())
    if idle_ms > 0:
        # A page that keeps polling never goes idle: take the DOM as it is then.
        with contextlib.suppress(PlaywrightTimeoutError):
            await page.wait_for_load_state("networkidle", timeout=idle_ms)
    snapshot = await _snapshot(page, session)

    # Lone surrogates in JS strings become U+FFFD.
    data = snapshot.html.encode("utf-8", "surrogatepass")
    cut = len(data) > session.max_bytes
    main = session.main
    if main is None:  # can't happen: the page loaded, so the handler handed it to Chromium
        raise WebFetchError(session.url, "the page loaded without being fetched")
    return WebPage(
        url=session.url,
        final_url=_page_url(page.url),
        status_code=main.status_code,
        media_type=main.media_type,
        text=decode_text(data[: session.max_bytes], truncated=cut, encoding="utf-8"),
        truncated=cut or main.truncated,
        hidden_marker=snapshot.marker,
        unchecked_text=snapshot.unchecked,
    )


@dataclass(frozen=True, slots=True)
class _Snapshot:
    html: str
    marker: str
    unchecked: int


async def _snapshot(page: "Page", session: "_Session") -> _Snapshot:
    """The DOM with hidden text marked, serialised as HTML.

    Runs in a fresh isolated world (CDP ``Page.createIsolatedWorld``): it shares the page's DOM
    but not its JavaScript globals, so page scripts can't patch ``getComputedStyle``,
    ``outerHTML``, and the rest to hide text from the scan. Retried if a script navigates while
    the page is being read. The marker attribute name is random per render, so a page can't
    plant marks of its own.
    """
    from playwright.async_api import Error as PlaywrightError

    marker = f"data-noinj-{secrets.token_hex(6)}"
    options = {
        "marker": marker,
        "limit": session.max_bytes,
        "maxElements": MAX_CHECKED_ELEMENTS,
        "minOpacity": HIDDEN_TEXT_MIN_OPACITY,
        "minFontPx": HIDDEN_TEXT_MIN_FONT_PX,
        "minVisiblePx": HIDDEN_TEXT_MIN_VISIBLE_PX,
        "minContrast": HIDDEN_TEXT_MIN_CONTRAST,
    }
    attempts = 3
    while True:
        attempts -= 1
        try:
            reply = await _call_isolated(page, _HIDDEN_TEXT_JS, options)
        except PlaywrightError as exc:
            if not attempts:
                message = f"could not read the page: {_first_line(exc)}"
                raise WebFetchError(session.url, message) from exc
            await page.wait_for_load_state("domcontentloaded", timeout=session.remaining_ms())
            continue
        if "exceptionDetails" in reply:
            details = reply["exceptionDetails"]
            text = details.get("exception", {}).get("description") or details.get("text", "")
            message = f"could not read the page: {_first_line(Exception(text))}"
            raise WebFetchError(session.url, message)
        value = reply["result"]["value"]
        return _Snapshot(value["html"], marker, value["unchecked"])


async def _call_isolated(page: "Page", function: str, argument: object) -> dict[str, Any]:
    """Call ``function(argument)`` in a new isolated world of the page's main frame.

    Returns the CDP ``Runtime.callFunctionOn`` reply: ``result.value``, or ``exceptionDetails``
    if the function threw.
    """
    cdp = await page.context.new_cdp_session(page)
    try:
        tree = await cdp.send("Page.getFrameTree")
        world = await cdp.send(
            "Page.createIsolatedWorld",
            {"frameId": tree["frameTree"]["frame"]["id"], "worldName": _WORLD_NAME},
        )
        reply: dict[str, Any] = await cdp.send(
            "Runtime.callFunctionOn",
            {
                "functionDeclaration": function,
                "executionContextId": world["executionContextId"],
                "arguments": [{"value": argument}],
                "returnByValue": True,
            },
        )
        return reply
    finally:
        with contextlib.suppress(Exception):
            await cdp.detach()


class _Session:
    """Shared state for one render: the HTTP client, the budget, and the top-level page."""

    def __init__(
        self,
        url: str,
        client: httpx.Client,
        *,
        max_bytes: int,
        timeout: float,
        deadline: float,
        max_redirects: int,
    ) -> None:
        self.url = url
        self.client = client
        self.max_bytes = max_bytes
        self.timeout = timeout
        self.deadline = deadline
        self.max_redirects = max_redirects
        self.main_frame: Frame | None = None
        self.main: FetchedBytes | None = None  # the top-level page, once handed to Chromium
        self.plain: FetchedBytes | None = None  # the top-level page, if it is not HTML
        self.redirect: str | None = None  # where the top-level page redirected to
        self.error: WebFetchError | None = None  # why the top-level page could not be loaded
        self._cache: dict[str, FetchedBytes] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._executor = ThreadPoolExecutor(_FETCH_THREADS, thread_name_prefix="noinjections")
        # Bound but never listening, so every connection to it is refused.
        self._blackhole = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._blackhole.bind(("127.0.0.1", 0))
        self.blackhole_url = f"http://127.0.0.1:{self._blackhole.getsockname()[1]}"

    def remaining_ms(self) -> float:
        """Milliseconds left in the budget; at least 1, since Playwright reads 0 as no limit."""
        return max((self.deadline - time.monotonic()) * 1000, 1.0)

    async def handle(self, route: "Route") -> None:
        """Route handler: fetch the request here and hand Chromium the response."""
        from playwright.async_api import Error as PlaywrightError

        task = asyncio.current_task()
        if task is not None:
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        # PlaywrightError here means the page or browser closed while the request was in flight.
        with contextlib.suppress(PlaywrightError):
            await self._handle(route)

    async def _handle(self, route: "Route") -> None:
        request = route.request
        if request.resource_type in SKIPPED_RESOURCE_TYPES:
            await route.abort("blockedbyclient")
            return
        is_page = self.main is None and self._is_top_level(request)
        try:
            url = normalize_url(request.url)
        except TargetError as exc:
            if is_page:
                self.error = WebFetchError(self.url, f"cannot fetch {request.url}: {exc}")
            await route.abort("blockedbyclient")
            return

        cached = self._cache.pop(url, None) if is_page else None
        if cached is not None:
            response = cached
        else:
            fetch = functools.partial(
                fetch_request,
                self.client,
                request.method,
                url,
                headers=_forwarded_headers((await request.all_headers()).items()),
                content=request.post_data_buffer,
                max_bytes=self.max_bytes,
                timeout=self.timeout,
                deadline=self.deadline,
                max_redirects=self.max_redirects,
                # The page must be a scannable page; other resources pass through as served.
                accepts=is_page_type if is_page else None,
                require_success=is_page,
            )
            try:
                response = await asyncio.get_running_loop().run_in_executor(self._executor, fetch)
            except WebFetchError as exc:
                if is_page:
                    self.error = exc
                await route.abort("failed")
                return

        if is_page:
            if response.final_url != url:
                # Chromium must load the page at its final URL, or relative links resolve
                # against the old one. Stop here; _load navigates again to the cached copy.
                self._cache[response.final_url] = response
                self.redirect = response.final_url
                await route.abort("aborted")
                return
            if response.media_type not in HTML_MEDIA_TYPES:
                self.plain = response
                await route.abort("aborted")
                return
            self.main = response
        elif response.truncated:
            await route.abort("failed")  # a cut-off script or stylesheet is of no use
            return
        await route.fulfill(
            status=response.status_code,
            headers=_fulfilled_headers(response.headers),
            body=response.data,
        )

    def _is_top_level(self, request: "Request") -> bool:
        from playwright.async_api import Error as PlaywrightError

        if not request.is_navigation_request():
            return False
        try:
            return request.frame == self.main_frame
        except PlaywrightError:
            return False

    async def close(self) -> None:
        """Release the client and wait for fetches still running, so no thread outlives this."""
        self.client.close()
        if self._tasks:
            await asyncio.wait(self._tasks)
        self._executor.shutdown(wait=False)
        self._blackhole.close()


def _forwarded_headers(headers: Iterable[tuple[str, str]]) -> dict[str, str]:
    """The browser's request headers, minus the ones that belong to its own connection."""
    return {
        name: value
        for name, value in headers
        if not name.startswith(":") and name.lower() not in _DROPPED_REQUEST_HEADERS
    }


def _fulfilled_headers(headers: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Response headers for Chromium: connection-level ones dropped, repeats joined."""
    merged: dict[str, str] = {}
    for name, value in headers:
        key = name.lower()
        if key in _DROPPED_RESPONSE_HEADERS:
            continue
        if key in merged:
            # Playwright splits Set-Cookie on newlines; other repeats join as a list.
            merged[key] += ("\n" if key == "set-cookie" else ", ") + value
        else:
            merged[key] = value
    return merged


def _page_url(url: str) -> str:
    try:
        return normalize_url(url)
    except TargetError:
        return url  # a script navigated to about:blank or a data: URL


def _first_line(exc: BaseException) -> str:
    return str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
