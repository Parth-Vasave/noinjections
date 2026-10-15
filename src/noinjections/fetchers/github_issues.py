"""Fetch the text of a GitHub repository's open issues and pull requests through the REST API."""

import json
import os
import re
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from urllib.parse import quote, urlencode

import httpx

from noinjections.fetchers.web import (
    DEFAULT_MAX_REDIRECTS,
    DEFAULT_TIMEOUT_SECONDS,
    USER_AGENT,
    FetchedBytes,
    WebFetchError,
    fetch_request,
    guarded_client,
)
from noinjections.models import Layer
from noinjections.targets import GitHubRepoTarget

API_ROOT = "https://api.github.com"
API_HOST = "api.github.com"
API_VERSION = "2022-11-28"

TOKEN_ENV = "GITHUB_TOKEN"  # noqa: S105 (the variable's name, not a secret)

DEFAULT_MAX_ITEMS = 500
"""Default cap on fetched text items: issues, PRs, and comments combined."""

DEFAULT_MAX_REQUESTS = 200
"""Default cap on API requests for one fetch, so a filtered listing can't page forever."""

DEFAULT_MAX_RATE_LIMIT_WAIT_SECONDS = 60.0
"""Longest wait for a rate limit to reset before giving up with partial results."""

DEFAULT_MAX_RESPONSE_BYTES = 32 * 1024 * 1024
"""Cap on one API response. A page of 100 issues with maximal bodies stays under it."""

PER_PAGE = 100  # the API's maximum

# Without retry-after or reset headers, GitHub asks clients to wait at least a minute.
_FALLBACK_RATE_LIMIT_WAIT_SECONDS = 60.0

_NEXT_LINK = re.compile(r'<([^>]*)>\s*;\s*rel="next"')
_PULL_NUMBER = re.compile(r"/pulls/(\d+)$")


class GitHubTextKind(StrEnum):
    """What a piece of GitHub text is."""

    ISSUE = "issue"
    PULL_REQUEST = "pull_request"
    ISSUE_COMMENT = "issue_comment"
    PR_COMMENT = "pr_comment"
    """A comment in a PR's conversation tab."""
    REVIEW_COMMENT = "review_comment"
    """An inline comment on a PR's diff."""

    @property
    def layer(self) -> Layer:
        """The :class:`~noinjections.models.Layer` for text of this kind."""
        if self in (GitHubTextKind.ISSUE, GitHubTextKind.ISSUE_COMMENT):
            return Layer.ISSUE
        return Layer.PR


@dataclass(frozen=True, slots=True)
class GitHubTextItem:
    """One issue, PR, or comment.

    ``number`` is the issue or PR number, also for its comments. ``url`` is the ``html_url`` of
    the item itself (a comment's own anchor), for provenance. ``title`` is set for issues and
    PRs only. ``body`` is the raw Markdown as an agent reading the API sees it, so HTML comments
    and other text hidden in GitHub's rendered view are kept. ``path`` and ``line`` locate a
    review comment in the diff (``line`` is ``None`` for an outdated or file-level comment).
    """

    kind: GitHubTextKind
    number: int
    url: str
    author: str | None
    body: str
    title: str = ""
    path: str | None = None
    line: int | None = None

    @property
    def layer(self) -> Layer:
        """The :class:`~noinjections.models.Layer` for this item."""
        return self.kind.layer


@dataclass(frozen=True, slots=True)
class GitHubIssuesResult:
    """The fetched items, most recently updated thread first, each followed by its comments.

    ``capped`` means ``max_items`` was reached and some items were left out. ``warnings`` holds
    non-fatal problems (a rate limit or the request cap cut the fetch short, or one thread's
    comments failed); ``items`` is then partial. ``requests`` counts the API requests sent.
    """

    repo: str
    items: tuple[GitHubTextItem, ...]
    capped: bool
    warnings: tuple[str, ...]
    requests: int


class GitHubApiError(Exception):
    """The GitHub API refused the request or answered with something unusable."""

    def __init__(self, url: str, message: str) -> None:
        super().__init__(f"{url}: {message}")
        self.url = url
        self.message = message


class _Stop(Exception):
    """The fetch can't continue (rate limit, request cap); what was fetched so far is kept."""


@dataclass
class _Thread:
    item: GitHubTextItem
    comments: int
    created_at: str


def github_token_from_env(env: Mapping[str, str] | None = None) -> str | None:
    """The token in ``GITHUB_TOKEN``, or ``None`` if it is unset or blank."""
    token = (os.environ if env is None else env).get(TOKEN_ENV, "").strip()
    return token or None


def fetch_github_issues(
    target: GitHubRepoTarget,
    *,
    token: str | None = None,
    max_items: int = DEFAULT_MAX_ITEMS,
    max_requests: int = DEFAULT_MAX_REQUESTS,
    max_rate_limit_wait: float = DEFAULT_MAX_RATE_LIMIT_WAIT_SECONDS,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    client: httpx.Client | None = None,
) -> GitHubIssuesResult:
    """Fetch the open issues and PRs of ``target``'s repository, with their comments.

    Fetches issue and PR titles and bodies, their conversation comments, and inline review
    comments on open PRs. Review summaries are not fetched. Threads are taken most recently
    updated first, each followed by its comments, until ``max_items`` items are collected.

    ``token`` (see :func:`github_token_from_env`) is sent only to ``api.github.com``; without it
    GitHub allows 60 requests an hour. When a rate limit is hit, the fetch waits for the reset
    if that is at most ``max_rate_limit_wait`` seconds away and retries once; otherwise, or
    after ``max_requests`` requests, it stops and returns what it has with a warning.
    ``timeout`` applies to each request. ``client`` is used as-is (the default client only
    connects to public addresses) and is left open.

    Raises :class:`GitHubApiError` if the repository's issues can't be listed (not found,
    private, bad token, HTTP error, malformed response).
    """
    if max_items < 1:
        raise ValueError(f"max_items must be >= 1, got {max_items}")
    if max_requests < 1:
        raise ValueError(f"max_requests must be >= 1, got {max_requests}")
    if not max_rate_limit_wait >= 0:
        raise ValueError(f"max_rate_limit_wait must be >= 0, got {max_rate_limit_wait}")
    if not timeout > 0:
        raise ValueError(f"timeout must be > 0, got {timeout}")
    if token is not None and not re.fullmatch(r"[\x21-\x7e]+", token):
        raise ValueError("token must be non-empty printable ASCII without spaces")

    own_client = client is None
    http = client if client is not None else guarded_client()
    api = _Api(http, token, max_requests, max_rate_limit_wait, timeout)
    try:
        return _fetch(api, target, max_items)
    finally:
        if own_client:
            http.close()


def _fetch(api: "_Api", target: GitHubRepoTarget, max_items: int) -> GitHubIssuesResult:
    repo_path = f"/repos/{quote(target.owner, safe='')}/{quote(target.repo, safe='')}"
    warnings: list[str] = []
    stopped = False
    capped = False  # set when an item had to be left out
    threads_left_out = False

    # 1. Open issues and PRs (one listing holds both), most recently updated first.
    threads: list[_Thread] = []
    listing = api.pages(
        f"{repo_path}/issues",
        {"state": "open", "sort": "updated", "direction": "desc", "per_page": PER_PAGE},
    )
    try:
        for raw, more in listing:
            threads.append(_parse_thread(raw, api.last_url))
            if len(threads) == max_items:
                threads_left_out = more
                break
    except _Stop as exc:
        warnings.append(str(exc))
        stopped = True

    # 2. Inline review comments on those PRs, from the repo-wide listing. Comments on an open PR
    # were last updated after the PR was opened, so `since` skips most closed-PR history.
    reviews: dict[int, list[tuple[int, GitHubTextItem]]] = {}
    prs = [t for t in threads if t.item.kind is GitHubTextKind.PULL_REQUEST]
    if prs and not stopped:
        wanted = {t.item.number for t in prs}
        since = min(t.created_at for t in prs)
        query = {"sort": "updated", "direction": "desc", "since": since, "per_page": PER_PAGE}
        found = 0
        try:
            for raw, _more in api.pages(f"{repo_path}/pulls/comments", query):
                comment_id, review = _parse_review_comment(raw, api.last_url)
                if review.number in wanted:
                    reviews.setdefault(review.number, []).append((comment_id, review))
                    found += 1
                    if found == max_items:
                        break
        except _Stop as exc:
            warnings.append(str(exc))
            stopped = True
        except GitHubApiError as exc:
            warnings.append(f"review comments skipped: {exc}")

    # 3. Each thread, then its conversation comments, then its review comments.
    items: list[GitHubTextItem] = []

    def add(item: GitHubTextItem) -> bool:
        nonlocal capped
        if len(items) == max_items:
            capped = True
            return False
        items.append(item)
        return True

    for thread in threads:
        if not add(thread.item):
            break
        if thread.comments > 0 and not stopped:
            kind = (
                GitHubTextKind.PR_COMMENT
                if thread.item.kind is GitHubTextKind.PULL_REQUEST
                else GitHubTextKind.ISSUE_COMMENT
            )
            path = f"{repo_path}/issues/{thread.item.number}/comments"
            try:
                for raw, _more in api.pages(path, {"per_page": PER_PAGE}):
                    if not add(_parse_comment(raw, kind, thread.item.number, api.last_url)):
                        break
            except _Stop as exc:
                warnings.append(str(exc))
                stopped = True
            except GitHubApiError as exc:
                warnings.append(f"comments on #{thread.item.number} skipped: {exc}")
        # Oldest first, like conversation comments (ids grow over time).
        for _id, review in sorted(reviews.get(thread.item.number, ()), key=lambda r: r[0]):
            if not add(review):
                break
        if capped:
            break

    return GitHubIssuesResult(
        repo=f"{target.owner}/{target.repo}",
        items=tuple(items),
        capped=capped or threads_left_out,
        warnings=tuple(warnings),
        requests=api.requests,
    )


class _Api:
    """Sends GET requests to the GitHub API, with paging, rate-limit handling, and a request cap."""

    def __init__(
        self,
        client: httpx.Client,
        token: str | None,
        max_requests: int,
        max_rate_limit_wait: float,
        timeout: float,
    ) -> None:
        self.client = client
        self.has_token = token is not None
        self.headers = {
            "User-Agent": USER_AGENT,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": API_VERSION,
        }
        if token is not None:
            self.headers["Authorization"] = f"Bearer {token}"
        self.max_requests = max_requests
        self.max_rate_limit_wait = max_rate_limit_wait
        self.timeout = timeout
        self.requests = 0
        self.last_url = ""

    def pages(self, path: str, query: Mapping[str, object]) -> Iterator[tuple[Any, bool]]:
        """Yield each element of a paged list, with whether more elements may follow it."""
        url: str | None = f"{API_ROOT}{path}?{urlencode(query)}"
        while url is not None:
            page, url = self.get_list(url)
            for index, element in enumerate(page):
                yield element, url is not None or index < len(page) - 1

    def get_list(self, url: str) -> tuple[list[Any], str | None]:
        """GET one page; return its JSON list and the next page's URL, if any."""
        self.last_url = url
        response = self._get(url)
        if response.status_code == 401:
            raise GitHubApiError(url, f"authentication failed: check {TOKEN_ENV}")
        if response.status_code == 404:
            hint = "" if self.has_token else f" (set {TOKEN_ENV} to scan a private repository)"
            message = f"not found: the repository may not exist or be private{hint}"
            raise GitHubApiError(url, message)
        if not 200 <= response.status_code < 300:
            raise GitHubApiError(url, f"HTTP {response.status_code}")
        if not _is_api_url(response.final_url):
            raise GitHubApiError(url, f"redirected away from {API_HOST}")
        if response.truncated:
            raise GitHubApiError(url, f"response is larger than {len(response.data)} bytes")
        try:
            page = json.loads(response.data)
        except ValueError as exc:
            raise GitHubApiError(url, f"response is not valid JSON: {exc}") from exc
        if not isinstance(page, list):
            raise GitHubApiError(url, "expected a JSON list")
        next_url = _next_link(_header(response, "link"))
        if next_url is not None and not _is_api_url(next_url):
            raise GitHubApiError(url, f"next page link points away from {API_HOST}")
        return page, next_url

    def _get(self, url: str) -> FetchedBytes:
        for attempt in range(2):
            if self.requests >= self.max_requests:
                raise _Stop(f"stopped after {self.requests} GitHub API requests (max_requests)")
            self.requests += 1
            try:
                response = fetch_request(
                    self.client,
                    "GET",
                    url,
                    headers=self.headers,
                    max_bytes=DEFAULT_MAX_RESPONSE_BYTES,
                    timeout=self.timeout,
                    deadline=time.monotonic() + self.timeout,
                    max_redirects=DEFAULT_MAX_REDIRECTS,
                    require_success=False,
                )
            except WebFetchError as exc:
                raise GitHubApiError(url, exc.message) from exc
            wait = _rate_limit_wait(response)
            if wait is None:
                return response
            if attempt == 0 and wait <= self.max_rate_limit_wait:
                _sleep(wait)
                continue
            hint = "" if self.has_token else f"; set {TOKEN_ENV} for a higher limit"
            raise _Stop(f"GitHub API rate limit reached (resets in {wait:.0f}s){hint}")
        raise AssertionError("unreachable")  # pragma: no cover


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _rate_limit_wait(response: FetchedBytes) -> float | None:
    """Seconds to wait if ``response`` is a rate-limit refusal, else ``None``.

    Follows GitHub's guidance: ``retry-after`` first, then ``x-ratelimit-reset`` when nothing
    remains, else at least a minute for a secondary limit.
    """
    if response.status_code not in (403, 429):
        return None
    retry_after = _header(response, "retry-after")
    if retry_after is not None and retry_after.strip().isdigit():
        return float(retry_after.strip())
    if _header(response, "x-ratelimit-remaining") == "0":
        reset = _header(response, "x-ratelimit-reset") or ""
        if reset.strip().isdigit():
            return max(int(reset) - time.time(), 0.0) + 1.0
        return _FALLBACK_RATE_LIMIT_WAIT_SECONDS
    if response.status_code == 429 or b"rate limit" in response.data.lower():
        return _FALLBACK_RATE_LIMIT_WAIT_SECONDS
    return None


def _header(response: FetchedBytes, name: str) -> str | None:
    """The last value of header ``name`` (lowercase), or ``None``."""
    values = [value for key, value in response.headers if key == name]
    return values[-1] if values else None


def _next_link(link: str | None) -> str | None:
    if link is None:
        return None
    match = _NEXT_LINK.search(link)
    return match.group(1) if match else None


def _is_api_url(url: str) -> bool:
    """Whether ``url`` is on the GitHub API host over HTTPS, so the token may be sent there."""
    try:
        parsed = httpx.URL(url)
    except httpx.InvalidURL:
        return False
    return (
        parsed.scheme == "https"
        and parsed.host == API_HOST
        and parsed.port in (None, 443)
        and not parsed.userinfo
    )


def _parse_thread(raw: Any, url: str) -> _Thread:
    obj = _object(raw, url)
    kind = GitHubTextKind.PULL_REQUEST if "pull_request" in obj else GitHubTextKind.ISSUE
    item = GitHubTextItem(
        kind=kind,
        number=_int(obj, "number", url),
        url=_str(obj, "html_url", url),
        author=_author(obj),
        title=_str(obj, "title", url),
        body=_optional_str(obj, "body", url),
    )
    comments = _int(obj, "comments", url)
    return _Thread(item, comments=comments, created_at=_str(obj, "created_at", url))


def _parse_comment(raw: Any, kind: GitHubTextKind, number: int, url: str) -> GitHubTextItem:
    obj = _object(raw, url)
    return GitHubTextItem(
        kind=kind,
        number=number,
        url=_str(obj, "html_url", url),
        author=_author(obj),
        body=_optional_str(obj, "body", url),
    )


def _parse_review_comment(raw: Any, url: str) -> tuple[int, GitHubTextItem]:
    """The comment's id (for ordering) and the comment."""
    obj = _object(raw, url)
    match = _PULL_NUMBER.search(_str(obj, "pull_request_url", url))
    if match is None:
        raise GitHubApiError(url, "review comment without a pull request number")
    line = obj.get("line")
    return _int(obj, "id", url), GitHubTextItem(
        kind=GitHubTextKind.REVIEW_COMMENT,
        number=int(match.group(1)),
        url=_str(obj, "html_url", url),
        author=_author(obj),
        body=_optional_str(obj, "body", url),
        path=_str(obj, "path", url),
        line=line if isinstance(line, int) and not isinstance(line, bool) and line >= 1 else None,
    )


def _object(raw: Any, url: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise GitHubApiError(url, "expected a JSON object in the list")
    return raw


def _str(obj: dict[str, Any], key: str, url: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str):
        raise GitHubApiError(url, f"expected a string for {key!r}")
    return value


def _optional_str(obj: dict[str, Any], key: str, url: str) -> str:
    """A string field that may be ``null`` (an empty body), as ``""``."""
    return "" if obj.get(key) is None else _str(obj, key, url)


def _int(obj: dict[str, Any], key: str, url: str) -> int:
    value = obj.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise GitHubApiError(url, f"expected a non-negative integer for {key!r}")
    return value


def _author(obj: dict[str, Any]) -> str | None:
    """The author's login; ``None`` for a deleted account (``user`` is ``null``)."""
    user = obj.get("user")
    login = user.get("login") if isinstance(user, dict) else None
    return login if isinstance(login, str) else None
