from pathlib import Path

import pytest

from noinjections.models import SourceKind
from noinjections.targets import (
    GitHubRepoTarget,
    LocalTarget,
    TargetError,
    WebTarget,
    detect_target,
    normalize_url,
)


@pytest.mark.parametrize(
    ("raw", "owner", "repo", "ref_path"),
    [
        ("https://github.com/octo/hello", "octo", "hello", None),
        ("https://github.com/octo/hello/", "octo", "hello", None),
        ("https://github.com/octo/hello.git", "octo", "hello", None),
        ("http://www.github.com/octo/hello", "octo", "hello", None),
        ("HTTPS://GitHub.com/Octo/Hello-World", "Octo", "Hello-World", None),
        ("https://github.com/octo/hello?tab=readme#install", "octo", "hello", None),
        ("https://github.com/octo/hello/tree/main", "octo", "hello", "main"),
        ("https://github.com/octo/hello/tree/feature/x/src", "octo", "hello", "feature/x/src"),
        ("https://github.com/octo/hello/tree/feature%2Fx", "octo", "hello", "feature/x"),
        ("https://github.com/octo/hello/tree/", "octo", "hello", None),
        ("https://github.com/octo/my.repo_v2", "octo", "my.repo_v2", None),
        ("  https://github.com/octo/hello  ", "octo", "hello", None),
        ("github.com/octo/hello", "octo", "hello", None),
        ("www.github.com/octo/hello.git", "octo", "hello", None),
    ],
)
def test_github_repo_urls(raw: str, owner: str, repo: str, ref_path: str | None) -> None:
    target = detect_target(raw)
    assert target == GitHubRepoTarget(owner=owner, repo=repo, ref_path=ref_path)
    assert target.source is SourceKind.GITHUB
    assert target.url == f"https://github.com/{owner}/{repo}"
    assert target.clone_url == f"https://github.com/{owner}/{repo}.git"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://example.com", "https://example.com/"),
        ("HTTP://Example.COM/Page?q=1#frag", "http://example.com/Page?q=1"),
        ("https://example.com:443/a", "https://example.com/a"),
        ("http://example.com:80/a", "http://example.com/a"),
        ("http://example.com:8080/a", "http://example.com:8080/a"),
        ("https://example.com./a", "https://example.com/a"),
        ("https://bücher.example/", "https://xn--bcher-kva.example/"),
        ("http://[::1]:8000/x", "http://[::1]:8000/x"),
        ("http://127.0.0.1:8000", "http://127.0.0.1:8000/"),
        # github.com pages that are not repositories stay generic URLs
        ("https://github.com/octo", "https://github.com/octo"),
        ("https://github.com/features/copilot", "https://github.com/features/copilot"),
        ("https://github.com/octo/hello/issues/1", "https://github.com/octo/hello/issues/1"),
        (
            "https://github.com/octo/hello/blob/main/a.py",
            "https://github.com/octo/hello/blob/main/a.py",
        ),
        ("https://gist.github.com/octo/abc123", "https://gist.github.com/octo/abc123"),
        (
            "https://raw.githubusercontent.com/o/r/main/x",
            "https://raw.githubusercontent.com/o/r/main/x",
        ),
    ],
)
def test_web_urls(raw: str, expected: str) -> None:
    target = detect_target(raw)
    assert target == WebTarget(url=expected)
    assert target.source is SourceKind.WEB


def test_local_directory(tmp_path: Path) -> None:
    target = detect_target(str(tmp_path))
    assert target == LocalTarget(path=tmp_path.resolve(), is_dir=True)
    assert target.source is SourceKind.LOCAL


@pytest.mark.parametrize("raw", ["README.md", "./README.md", "docs/../README.md"])
def test_relative_local_file(raw: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text("hi")
    monkeypatch.chdir(tmp_path)
    assert detect_target(raw) == LocalTarget(path=(tmp_path / "README.md").resolve(), is_dir=False)


def test_relative_local_dir_dot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert detect_target(".") == LocalTarget(path=tmp_path.resolve(), is_dir=True)


def test_home_expansion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "notes.txt").write_text("hi")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
    assert detect_target("~/notes.txt") == LocalTarget(
        path=(tmp_path / "notes.txt").resolve(), is_dir=False
    )


def test_existing_local_path_wins_over_github_shorthand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_dir = tmp_path / "github.com" / "octo" / "hello"
    repo_dir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    assert detect_target("github.com/octo/hello") == LocalTarget(
        path=repo_dir.resolve(), is_dir=True
    )


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("", "empty"),
        ("   ", "empty"),
        ("ftp://example.com/x", "unsupported URL scheme 'ftp'"),
        ("file:///etc/passwd", "unsupported URL scheme 'file'"),
        ("https://", "no host"),
        ("https://user:secret@example.com/", "credentials"),
        ("https://example.com:99999/", "invalid port"),
        ("https://exa mple.com/", "whitespace"),
        ("https://example.com/\x00", "control characters"),
        ("does-not-exist.md", "no such file or directory"),
        ("example.com/page", "include the http:// or https:// prefix"),
        ("github.com/features/copilot", "no such file or directory"),
        ("bad\x00path", "no such file or directory"),
    ],
)
def test_invalid_targets(
    raw: str, message: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(TargetError, match=message):
        detect_target(raw)


def test_target_error_is_value_error() -> None:
    assert issubclass(TargetError, ValueError)


def test_normalize_url_is_idempotent() -> None:
    once = normalize_url("HTTPS://Bücher.Example:443?x=1#y")
    assert once == "https://xn--bcher-kva.example/?x=1"
    assert normalize_url(once) == once


@pytest.mark.parametrize(
    "path",
    [
        "/octo-/hello",  # trailing hyphen in owner
        "/-octo/hello",  # leading hyphen in owner
        "/oc--to/hello",  # double hyphen in owner
        "/octo/hel%20lo",  # space in repo name
        "/octo/..",
        "/orgs/octo",
        "/octo/hello.git/tree/main",
    ],
)
def test_invalid_github_repo_paths_fall_back_to_web(path: str) -> None:
    target = detect_target(f"https://github.com{path}")
    assert isinstance(target, WebTarget)
