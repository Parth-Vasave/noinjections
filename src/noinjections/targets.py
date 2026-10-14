"""Classify a CLI target string as a GitHub repo, a generic web URL, or a local file/directory."""

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

from noinjections.models import SourceKind


class TargetError(ValueError):
    """The target string is not something noinjections can scan."""


@dataclass(frozen=True, slots=True)
class WebTarget:
    """A generic ``http(s)`` URL, normalised."""

    url: str

    @property
    def source(self) -> SourceKind:
        return SourceKind.WEB


@dataclass(frozen=True, slots=True)
class GitHubRepoTarget:
    """A repository on github.com.

    ``ref_path`` is everything after ``/tree/`` in the URL: a branch or tag, possibly followed by
    a sub-path. The two can't be told apart without asking the remote (branch names may contain
    ``/``), so splitting is left to the fetcher.
    """

    owner: str
    repo: str
    ref_path: str | None = None

    @property
    def source(self) -> SourceKind:
        return SourceKind.GITHUB

    @property
    def url(self) -> str:
        """Canonical web URL of the repository."""
        return f"https://github.com/{self.owner}/{self.repo}"

    @property
    def clone_url(self) -> str:
        """HTTPS clone URL."""
        return f"{self.url}.git"


@dataclass(frozen=True, slots=True)
class LocalTarget:
    """An existing local file or directory, as an absolute resolved path."""

    path: Path
    is_dir: bool

    @property
    def source(self) -> SourceKind:
        return SourceKind.LOCAL


Target = WebTarget | GitHubRepoTarget | LocalTarget

_HTTP_SCHEME = re.compile(r"^https?://", re.IGNORECASE)
_ANY_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)
_GITHUB_HOSTS = frozenset({"github.com", "www.github.com"})
_DEFAULT_PORTS = {"http": 80, "https": 443}

# GitHub usernames: alphanumerics and single hyphens, no leading/trailing hyphen, max 39 chars.
_GITHUB_OWNER = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}$")
_GITHUB_REPO = re.compile(r"^[A-Za-z0-9._-]{1,100}$")
# First path segments that are GitHub site pages, not users or orgs.
_GITHUB_RESERVED_OWNERS = frozenset(
    {
        "about",
        "apps",
        "codespaces",
        "collections",
        "contact",
        "customer-stories",
        "dashboard",
        "enterprise",
        "events",
        "explore",
        "features",
        "issues",
        "login",
        "marketplace",
        "new",
        "notifications",
        "organizations",
        "orgs",
        "pricing",
        "pulls",
        "search",
        "security",
        "settings",
        "site",
        "sponsors",
        "stars",
        "team",
        "topics",
        "trending",
    }
)


def detect_target(raw: str) -> Target:
    """Classify and normalise a target string.

    Explicit ``http(s)://`` URLs are parsed as URLs; anything else is tried as a local path, and
    finally as a schemeless ``github.com/owner/repo`` shorthand. Raises :class:`TargetError` if
    the input fits none of these.
    """
    text = raw.strip()
    if not text:
        raise TargetError("target is empty")

    if _HTTP_SCHEME.match(text):
        return _classify_url(text)
    if _ANY_SCHEME.match(text):
        scheme = text.split("://", 1)[0]
        raise TargetError(f"unsupported URL scheme {scheme!r}: only http and https are supported")

    local = _local_target(text)
    if local is not None:
        return local

    if text.lower().startswith(("github.com/", "www.github.com/")):
        target = _classify_url(f"https://{text}")
        if isinstance(target, GitHubRepoTarget):
            return target

    raise TargetError(
        f"no such file or directory: {text!r} (for a URL, include the http:// or https:// prefix)"
    )


def _local_target(text: str) -> LocalTarget | None:
    """Return a target for an existing local path, or ``None`` if nothing exists there."""
    path = Path(text).expanduser()
    try:
        if not path.exists():
            return None
    except (OSError, ValueError):  # e.g. name too long, embedded null byte
        return None
    resolved = path.resolve()
    if resolved.is_dir():
        return LocalTarget(path=resolved, is_dir=True)
    if resolved.is_file():
        return LocalTarget(path=resolved, is_dir=False)
    raise TargetError(f"not a regular file or directory: {text!r}")


def _classify_url(text: str) -> WebTarget | GitHubRepoTarget:
    url = normalize_url(text)
    parts = urlsplit(url)
    if parts.hostname in _GITHUB_HOSTS:
        repo = _github_repo(parts.path)
        if repo is not None:
            return repo
    return WebTarget(url=url)


def normalize_url(text: str) -> str:
    """Normalise an ``http(s)`` URL.

    Lowercases the scheme and host, converts internationalised hosts to punycode, drops the
    default port and the fragment, and turns an empty path into ``/``. Raises
    :class:`TargetError` for malformed URLs and for URLs carrying credentials.
    """
    if any(c.isspace() or not c.isprintable() for c in text):
        raise TargetError(f"URL contains whitespace or control characters: {text!r}")
    parts = urlsplit(text)
    scheme = parts.scheme.lower()
    if scheme not in _DEFAULT_PORTS:
        raise TargetError(
            f"unsupported URL scheme {parts.scheme!r}: only http and https are supported"
        )
    if parts.username is not None or parts.password is not None:
        raise TargetError("URLs with embedded credentials are not supported")
    if not parts.hostname:
        raise TargetError(f"URL has no host: {text!r}")
    try:
        port = parts.port
    except ValueError as exc:
        raise TargetError(f"URL has an invalid port: {text!r}") from exc

    host = _normalize_host(parts.hostname)
    netloc = host if port is None or port == _DEFAULT_PORTS[scheme] else f"{host}:{port}"
    return urlunsplit((scheme, netloc, parts.path or "/", parts.query, ""))


def _normalize_host(hostname: str) -> str:
    """Lowercase, strip a trailing dot, punycode-encode IDNs, and re-bracket IPv6 literals."""
    if ":" in hostname:  # IPv6 literal; urlsplit strips the brackets
        return f"[{hostname.lower()}]"
    host = hostname.rstrip(".").lower()
    if not host:
        raise TargetError("URL has an empty host")
    if not host.isascii():
        try:
            host = host.encode("idna").decode("ascii")
        except UnicodeError as exc:
            raise TargetError(f"invalid internationalised host name: {hostname!r}") from exc
    return host


def _github_repo(path: str) -> GitHubRepoTarget | None:
    """Parse ``/owner/repo[.git][/tree/<ref_path>]``; ``None`` for any other github.com page."""
    segments = [s for s in path.split("/") if s]
    if len(segments) < 2:
        return None
    owner, repo, *rest = segments
    if rest and rest[0] != "tree":
        return None
    if repo.endswith(".git"):
        if rest:  # /owner/repo.git/tree/... is not a real GitHub URL
            return None
        repo = repo.removesuffix(".git")
    if (
        owner.lower() in _GITHUB_RESERVED_OWNERS
        or not _GITHUB_OWNER.match(owner)
        or not _GITHUB_REPO.match(repo)
        or repo in {".", ".."}
    ):
        return None
    ref_path = unquote("/".join(rest[1:])) or None
    return GitHubRepoTarget(owner=owner, repo=repo, ref_path=ref_path)
