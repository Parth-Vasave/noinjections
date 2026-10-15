import os
import shutil
import subprocess
import sys
import tempfile
import time
from functools import partial
from pathlib import Path

import pytest

from noinjections.fetchers import git_repo, github
from noinjections.fetchers.content import SkipReason
from noinjections.fetchers.git_repo import (
    RepoFetchError,
    RepoFetchResult,
    _GitRunner,
    fetch_git_repo,
)
from noinjections.targets import GitHubRepoTarget

# Every remote is a local repository reached through a file:// URL. Nothing touches the network.

GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(GIT is None, reason="git not installed")
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="needs sh and POSIX processes")

fetch = partial(fetch_git_repo, allow_file_protocol=True)


@pytest.fixture(autouse=True)
def home(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty HOME, so the developer's own git config can't affect the tests."""
    path = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(path / ".config"))
    return path


def git(cwd: Path, *args: str, stdin: bytes | None = None) -> str:
    assert GIT is not None
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    config = [
        "user.name=Test",
        "user.email=test@example.com",
        "commit.gpgsign=false",
        "tag.gpgsign=false",
        "init.defaultBranch=main",
        "core.autocrlf=false",
        "protocol.file.allow=always",
    ]
    argv = [GIT, *(arg for c in config for arg in ("-c", c)), *args]
    proc = subprocess.run(  # noqa: S603
        argv, cwd=cwd, env=env, input=stdin, capture_output=True, check=True
    )
    return proc.stdout.decode().strip()


class Repo:
    """A working repository that tests commit to, published as a bare remote."""

    def __init__(self, root: Path) -> None:
        self.work = root / "work"
        self.work.mkdir(parents=True)
        self.bare = root / "remote.git"
        git(self.work, "init", "--quiet")

    def commit(self, files: dict[str, str | bytes], message: str = "commit") -> str:
        for rel, content in files.items():
            path = self.work / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                path.write_bytes(content)
            else:
                path.write_text(content, encoding="utf-8", newline="")
        git(self.work, "add", "--all", "--force")
        git(self.work, "commit", "--quiet", "--allow-empty", "-m", message)
        return git(self.work, "rev-parse", "HEAD")

    def add_entry(self, path: str, mode: str, sha: str) -> str:
        """Commit a raw tree entry (a symlink or a submodule), without touching the disk."""
        git(self.work, "update-index", "--add", "--cacheinfo", f"{mode},{sha},{path}")
        git(self.work, "commit", "--quiet", "-m", f"add {path}")
        return git(self.work, "rev-parse", "HEAD")

    def blob(self, content: str) -> str:
        return git(self.work, "hash-object", "-w", "--stdin", stdin=content.encode())

    def publish(self) -> str:
        """Mirror the work repo (all branches and tags) to the bare remote; return its URL."""
        if self.bare.exists():
            shutil.rmtree(self.bare)
        git(self.work.parent, "clone", "--quiet", "--bare", str(self.work), str(self.bare))
        return self.bare.as_uri()


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    return Repo(tmp_path / "repo")


def paths(result: RepoFetchResult) -> list[str]:
    return [f.path for f in result.files]


def text_of(result: RepoFetchResult, path: str) -> str:
    return next(f.text for f in result.files if f.path == path)


# --- default branch ---


def test_clones_the_default_branch(repo: Repo) -> None:
    sha = repo.commit(
        {"README.md": "# demo\n", "src/app.py": "print(1)\n", "AGENTS.md": "be nice\n"}
    )
    url = repo.publish()
    result = fetch(url)
    assert result.url == url
    assert result.ref is None
    assert result.commit == sha
    assert result.subdir is None
    assert paths(result) == ["AGENTS.md", "README.md", "src/app.py"]  # agent files first
    assert text_of(result, "src/app.py") == "print(1)\n"
    assert result.skipped == []
    assert result.not_scanned == 0


def test_scans_only_the_tip_commit(repo: Repo) -> None:
    repo.commit({"old.md": "old"})
    (repo.work / "old.md").unlink()
    sha = repo.commit({"new.md": "new"})
    result = fetch(repo.publish())
    assert result.commit == sha
    assert paths(result) == ["new.md"]


def test_committed_but_gitignored_files_are_scanned(repo: Repo) -> None:
    # A hostile repo can't hide a committed file from the scan with its own .gitignore.
    repo.commit({".gitignore": "hidden.md\nsecret/\n", "hidden.md": "x", "secret/a.txt": "y"})
    result = fetch(repo.publish())
    assert paths(result) == [".gitignore", "hidden.md", "secret/a.txt"]


def test_empty_repository(tmp_path: Path) -> None:
    bare = tmp_path / "empty.git"
    git(tmp_path, "init", "--quiet", "--bare", str(bare))
    result = fetch(bare.as_uri())
    assert result == RepoFetchResult(bare.as_uri(), None, None, None, [], [], 0)


def test_file_caps_are_passed_on(repo: Repo) -> None:
    repo.commit({"a.txt": "a" * 10, "b.txt": "b", "c.txt": "c"})
    result = fetch(repo.publish(), max_file_bytes=4, max_files=2)
    assert paths(result) == ["a.txt", "b.txt"]
    assert result.files[0].text == "aaaa"
    assert result.files[0].truncated
    assert result.not_scanned == 1


# --- refs and sub-paths ---


def test_branch_with_a_slash_and_a_sub_path(repo: Repo) -> None:
    repo.commit({"README.md": "main"})
    git(repo.work, "checkout", "--quiet", "-b", "feature/x")
    sha = repo.commit({"docs/guide.md": "guide", "docs/deep/more.md": "more", "src/a.py": "a"})
    git(repo.work, "checkout", "--quiet", "main")
    result = fetch(repo.publish(), "feature/x/docs")
    assert result.ref == "refs/heads/feature/x"
    assert result.subdir == "docs"
    assert result.commit == sha
    assert paths(result) == ["docs/deep/more.md", "docs/guide.md"]


def test_branch_without_a_sub_path(repo: Repo) -> None:
    repo.commit({"README.md": "main"})
    git(repo.work, "checkout", "--quiet", "-b", "dev")
    repo.commit({"dev.md": "dev"})
    git(repo.work, "checkout", "--quiet", "main")
    result = fetch(repo.publish(), "dev/")
    assert result.ref == "refs/heads/dev"
    assert result.subdir is None
    assert paths(result) == ["README.md", "dev.md"]


def test_longest_matching_ref_wins(repo: Repo) -> None:
    # refs/tags/release and refs/heads/release/v2 can coexist (different namespaces).
    repo.commit({"docs/a.md": "tag"})
    git(repo.work, "tag", "release")
    git(repo.work, "checkout", "--quiet", "-b", "release/v2")
    branch_sha = repo.commit({"docs/b.md": "branch"})
    git(repo.work, "checkout", "--quiet", "main")
    url = repo.publish()

    result = fetch(url, "release/v2/docs")
    assert result.ref == "refs/heads/release/v2"
    assert (result.subdir, result.commit) == ("docs", branch_sha)

    result = fetch(url, "release/docs")
    assert (result.ref, result.subdir) == ("refs/tags/release", "docs")
    assert paths(result) == ["docs/a.md"]


def test_branch_beats_a_tag_with_the_same_name(repo: Repo) -> None:
    repo.commit({"a.md": "tag"})
    git(repo.work, "tag", "v1")
    git(repo.work, "checkout", "--quiet", "-b", "v1")
    branch_sha = repo.commit({"b.md": "branch"})
    git(repo.work, "checkout", "--quiet", "main")
    result = fetch(repo.publish(), "v1")
    assert result.ref == "refs/heads/v1"
    assert result.commit == branch_sha


def test_annotated_tag(repo: Repo) -> None:
    sha = repo.commit({"a.md": "v1"})
    git(repo.work, "tag", "-a", "v1.0", "-m", "release")
    repo.commit({"b.md": "later"})
    result = fetch(repo.publish(), "v1.0")
    assert result.ref == "refs/tags/v1.0"
    assert result.commit == sha  # the tag object is peeled to its commit
    assert paths(result) == ["a.md"]


def test_full_commit_sha_with_a_sub_path(repo: Repo) -> None:
    old = repo.commit({"docs/a.md": "old", "x.md": "x"})
    repo.commit({"docs/a.md": "new"})
    result = fetch(repo.publish(), f"{old.upper()}/docs")
    assert result.ref == old
    assert result.commit == old
    assert result.subdir == "docs"
    assert text_of(result, "docs/a.md") == "old"


def test_short_sha_is_refused_with_a_hint(repo: Repo) -> None:
    sha = repo.commit({"a.md": "a"})
    with pytest.raises(RepoFetchError, match="must be given in full"):
        fetch(repo.publish(), sha[:7])


def test_unknown_ref(repo: Repo) -> None:
    repo.commit({"a.md": "a"})
    with pytest.raises(RepoFetchError, match="no branch or tag matches 'nope/docs'"):
        fetch(repo.publish(), "nope/docs")


@pytest.mark.parametrize("ref_path", ["main/../x", "main/./docs", "../main", "/"])
def test_bad_ref_paths(repo: Repo, ref_path: str) -> None:
    repo.commit({"a.md": "a"})
    with pytest.raises(RepoFetchError, match=r"invalid path|empty ref"):
        fetch(repo.publish(), ref_path)


def test_ref_path_with_glob_characters_is_matched_literally(repo: Repo) -> None:
    repo.commit({"docs/[x]*.md": "odd name"})
    result = fetch(repo.publish(), "main/docs/[x]*.md")
    assert paths(result) == ["docs/[x]*.md"]


def test_missing_sub_path(repo: Repo) -> None:
    repo.commit({"docs/a.md": "a"})
    with pytest.raises(RepoFetchError, match="'nope' not found"):
        fetch(repo.publish(), "main/nope")


def test_sub_path_is_matched_literally(repo: Repo) -> None:
    # Without literal pathspecs, ":(glob)*" would match every file.
    repo.commit({"docs/a.md": "a"})
    with pytest.raises(RepoFetchError, match="not found"):
        fetch(repo.publish(), "main/:(glob)*")


def test_sub_path_that_is_a_file(repo: Repo) -> None:
    repo.commit({"docs/AGENTS.md": "agent", "docs/other.md": "other"})
    result = fetch(repo.publish(), "main/docs/AGENTS.md")
    assert result.subdir == "docs/AGENTS.md"
    assert paths(result) == ["docs/AGENTS.md"]


def test_symlinked_sub_path_is_refused(repo: Repo) -> None:
    repo.commit({"a.md": "a"})
    repo.add_entry("docs", "120000", repo.blob("/etc"))
    url = repo.publish()
    with pytest.raises(RepoFetchError, match="'docs' is a symlink"):
        fetch(url, "main/docs")
    with pytest.raises(RepoFetchError, match="not found"):  # git doesn't follow it either
        fetch(url, "main/docs/passwd")


@pytest.mark.skipif(
    sys.platform == "win32", reason="Git for Windows checks out symlinks as plain files"
)
def test_symlinks_inside_the_repo_are_skipped(repo: Repo) -> None:
    repo.commit({"a.md": "a"})
    repo.add_entry("link.md", "120000", repo.blob("/etc/passwd"))
    result = fetch(repo.publish())
    assert paths(result) == ["a.md"]
    assert {s.path: s.reason for s in result.skipped} == {"link.md": SkipReason.SYMLINK}


# --- submodules ---


def test_submodules_are_not_fetched(repo: Repo, tmp_path: Path, home: Path) -> None:
    lib = Repo(tmp_path / "lib")
    lib_sha = lib.commit({"payload.md": "from the submodule"})
    lib_url = lib.publish()
    repo.commit({".gitmodules": f'[submodule "lib"]\n\tpath = vendor/lib\n\turl = {lib_url}\n'})
    repo.add_entry("vendor/lib", "160000", lib_sha)
    # Even a user who asked git to always recurse into submodules gets no submodule content.
    (home / ".gitconfig").write_text("[submodule]\n\trecurse = true\n", encoding="utf-8")
    url = repo.publish()

    result = fetch(url)
    assert paths(result) == [".gitmodules"]
    assert {s.path: s.reason for s in result.skipped} == {"vendor/lib": SkipReason.NOT_REGULAR}
    with pytest.raises(RepoFetchError, match="is a submodule"):
        fetch(url, "main/vendor/lib")


# --- size and time limits ---


def test_checkout_over_the_cap_is_refused(repo: Repo) -> None:
    # 2 MiB of zeros compresses to a tiny pack: only the checkout would be big.
    repo.commit({"bomb.txt": b"\0" * (2 * 1024 * 1024), "a.md": "a"})
    with pytest.raises(
        RepoFetchError, match=r"checkout would be \d+ bytes, over the 1048576 byte cap"
    ):
        fetch(repo.publish(), max_repo_bytes=1024 * 1024)


def test_checkout_cap_counts_only_the_sub_path(repo: Repo) -> None:
    repo.commit({"big/bomb.txt": b"\0" * (2 * 1024 * 1024), "docs/a.md": "a"})
    result = fetch(repo.publish(), "main/docs", max_repo_bytes=1024 * 1024)
    assert paths(result) == ["docs/a.md"]


@pytest.mark.parametrize("poll", [0.001, 60.0])
def test_download_over_the_cap_is_refused(
    repo: Repo, monkeypatch: pytest.MonkeyPatch, poll: float
) -> None:
    # With a 60 s poll, git finishes before the first poll: only the final check can catch it.
    monkeypatch.setattr(git_repo, "_POLL_SECONDS", poll)
    repo.commit({"noise.bin": os.urandom(2 * 1024 * 1024)})  # incompressible
    with pytest.raises(RepoFetchError, match="repository is over the 1048576 byte cap"):
        fetch(repo.publish(), max_repo_bytes=1024 * 1024)


@posix_only
def test_runner_kills_git_when_the_watched_dir_grows(tmp_path: Path) -> None:
    runner = _GitRunner(GIT or "git", "u", tmp_path, time.monotonic() + 60, 60, False)
    filler = "!head -c 2000000 /dev/zero > big && sleep 30"
    start = time.monotonic()
    with pytest.raises(RepoFetchError, match="over the 1000000 byte cap"):
        runner.run("-c", f"alias.fill={filler}", "fill", watch=tmp_path, max_bytes=1_000_000)
    assert time.monotonic() - start < 10


@posix_only
def test_timeout_kills_git_and_its_children(tmp_path: Path) -> None:
    runner = _GitRunner(GIT or "git", "u", tmp_path, time.monotonic() + 0.5, 0.5, False)
    pid_file = tmp_path / "pid"
    stall = f"!sleep 30 & echo $! > {pid_file}; wait"
    start = time.monotonic()
    with pytest.raises(RepoFetchError, match=r"timed out after 0\.5 s"):
        runner.run("-c", f"alias.stall={stall}", "stall")
    assert time.monotonic() - start < 10
    child = int(pid_file.read_text())
    for _ in range(50):  # the orphaned child is reaped by init shortly after the kill
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail("the grandchild process survived the timeout")


def test_deadline_already_passed(repo: Repo, monkeypatch: pytest.MonkeyPatch) -> None:
    repo.commit({"a.md": "a"})
    url = repo.publish()
    clock = iter([0.0, 100.0])  # the deadline is set at 0 + 5 s; the first check sees 100 s
    monkeypatch.setattr(git_repo.time, "monotonic", lambda: next(clock, 100.0))
    with pytest.raises(RepoFetchError, match="timed out after 5 s"):
        fetch(url, timeout=5)


# --- cleanup ---


def test_clone_is_deleted_after_success_and_after_errors(
    repo: Repo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    repo.commit({"a.md": "a", "zeros.bin": b"\0" * 4096})
    url = repo.publish()

    assert paths(fetch(url)) == ["a.md"]  # zeros.bin is skipped as binary
    assert list(scratch.iterdir()) == []
    with pytest.raises(RepoFetchError):
        fetch(url, "main/nope")
    assert list(scratch.iterdir()) == []
    with pytest.raises(RepoFetchError):
        fetch(url, max_repo_bytes=1024)
    assert list(scratch.iterdir()) == []


# --- hardening ---


def test_file_protocol_is_refused_by_default(repo: Repo) -> None:
    repo.commit({"a.md": "a"})
    with pytest.raises(RepoFetchError, match="transport 'file' not allowed"):
        fetch_git_repo(repo.publish())


def test_callers_git_env_is_ignored(
    repo: Repo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sha = repo.commit({"a.md": "a"})
    other = tmp_path / "other"
    other.mkdir()
    git(other, "init", "--quiet")
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(other))
    result = fetch(repo.publish())
    assert result.commit == sha
    assert paths(result) == ["a.md"]


@posix_only
def test_user_hooks_and_templates_never_run(repo: Repo, tmp_path: Path, home: Path) -> None:
    marker = tmp_path / "hook-ran"
    hooks = tmp_path / "user-hooks"
    template_hooks = tmp_path / "template" / "hooks"
    for directory in (hooks, template_hooks):
        directory.mkdir(parents=True)
        for name in ("post-checkout", "reference-transaction", "post-index-change"):
            hook = directory / name
            hook.write_text(f"#!/bin/sh\ntouch {marker}\n", encoding="utf-8")
            hook.chmod(0o755)
    (home / ".gitconfig").write_text(
        f"[core]\n\thooksPath = {hooks}\n[init]\n\ttemplateDir = {template_hooks.parent}\n",
        encoding="utf-8",
    )
    repo.commit({"a.md": "a"})
    url = repo.publish()  # the test's own git calls may run the hooks; reset the marker after
    marker.unlink(missing_ok=True)
    assert paths(fetch(url)) == ["a.md"]
    assert not marker.exists()


@posix_only
def test_lfs_filter_never_runs(repo: Repo, tmp_path: Path, home: Path) -> None:
    pointer = "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 3\n"
    repo.commit({".gitattributes": "*.md filter=lfs\n", "a.md": pointer})
    url = repo.publish()
    marker = tmp_path / "filter-ran"
    (home / ".gitconfig").write_text(
        '[filter "lfs"]\n'
        f"\tsmudge = touch {marker} && cat\n"
        f"\tprocess = touch {marker}\n"
        "\trequired = true\n",
        encoding="utf-8",
    )
    result = fetch(url)
    assert not marker.exists()
    assert text_of(result, "a.md") == pointer  # the pointer file, as committed


def test_committed_bytes_are_kept_despite_autocrlf(repo: Repo, home: Path) -> None:
    repo.commit({"a.md": "one\ntwo\n"})
    url = repo.publish()
    (home / ".gitconfig").write_text("[core]\n\tautocrlf = true\n", encoding="utf-8")
    assert text_of(fetch(url), "a.md") == "one\ntwo\n"


def test_does_not_read_config_from_a_repo_above_the_temp_dir(
    repo: Repo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Put the temp dir inside a hostile repo whose config breaks every fetch.
    hostile = tmp_path / "hostile"
    hostile.mkdir()
    git(hostile, "init", "--quiet")
    git(hostile, "config", "url.file:///nonexistent/.insteadOf", "file://")
    monkeypatch.setattr(tempfile, "tempdir", str(hostile))
    repo.commit({"a.md": "a"})
    assert paths(fetch(repo.publish(), "main")) == ["a.md"]


# --- errors and arguments ---


def test_missing_repository(tmp_path: Path) -> None:
    with pytest.raises(RepoFetchError, match=r"git fetch failed: fatal: .*does not appear"):
        fetch((tmp_path / "nope.git").as_uri())


def test_git_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(git_repo.shutil, "which", lambda _name: None)
    with pytest.raises(RepoFetchError, match="git is not installed"):
        fetch("https://github.com/o/r.git")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"max_repo_bytes": 0}, "max_repo_bytes"), ({"timeout": 0}, "timeout")],
)
def test_invalid_arguments(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        fetch("https://github.com/o/r.git", **kwargs)


@pytest.mark.parametrize(
    "stderr",
    [
        b"remote: Repository not found.\nfatal: repository 'https://github.com/o/r/' not found\n",
        b"fatal: could not read Username for 'https://github.com': terminal prompts disabled\n",
    ],
)
def test_not_found_or_private_message(stderr: bytes) -> None:
    message = git_repo._describe_failure("fetch", stderr)
    assert message == "repository not found, or it is private (no credentials were available)"


def test_failure_message_keeps_fatal_lines_and_is_capped() -> None:
    message = git_repo._describe_failure("fetch", b"hint: x\nfatal: " + b"y" * 1000 + b"\n")
    assert message.startswith("git fetch failed: fatal: yyy")
    assert len(message) == 500
    assert git_repo._describe_failure("ls-tree", b"") == "git ls-tree failed: no error message"


# --- GitHub wrapper ---


def test_fetch_github_repo_clones_over_https_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake(*args: object, **kwargs: object) -> RepoFetchResult:
        calls.append((args, kwargs))
        return RepoFetchResult("u", None, None, None, [], [], 0)

    monkeypatch.setattr(github, "fetch_git_repo", fake)
    github.fetch_github_repo(GitHubRepoTarget("octo", "demo", "main/docs"), max_files=5)
    ((args, kwargs),) = calls
    assert args == ("https://github.com/octo/demo.git", "main/docs")
    assert kwargs["max_files"] == 5
    assert "allow_file_protocol" not in kwargs


@pytest.mark.network
def test_real_github_clone() -> None:
    result = github.fetch_github_repo(GitHubRepoTarget("octocat", "Hello-World", "master"))
    assert result.ref == "refs/heads/master"
    assert result.commit is not None
    assert "README" in paths(result)
