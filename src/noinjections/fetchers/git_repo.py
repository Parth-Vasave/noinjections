"""Shallow-clone a git repository into a temporary directory and collect its text files."""

import contextlib
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path

from noinjections.fetchers.content import (
    DEFAULT_MAX_AGENT_FILE_BYTES,
    DEFAULT_MAX_FILE_BYTES,
    SkippedFile,
)
from noinjections.fetchers.local import DEFAULT_MAX_FILES, LocalFile, fetch_local

DEFAULT_MAX_REPO_BYTES = 512 * 1024 * 1024
"""Default repository size cap (512 MiB), applied to the download and to the checkout."""

DEFAULT_CLONE_TIMEOUT_SECONDS = 120.0
"""Default overall time budget for every git command of one fetch."""

_POLL_SECONDS = 0.1
_MAX_ERROR_CHARS = 500

# A full SHA-1 or SHA-256 object name. GitHub accepts a commit in /tree/<sha>/..., and git can
# fetch a commit by its full name, but not by an abbreviation.
_FULL_SHA = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_SHORT_SHA = re.compile(r"[0-9a-f]{7,39}")
# Characters that can never appear in a ref name (git check-ref-format). Skipping prefixes that
# contain them also keeps ls-remote's glob patterns literal.
_NOT_IN_REF = re.compile(r"[\x00-\x20\x7f~^:?*\[\\]")

_MODE_SYMLINK = "120000"


class RepoFetchError(Exception):
    """The repository could not be cloned, or the requested ref or path doesn't exist."""

    def __init__(self, url: str, message: str) -> None:
        super().__init__(f"{url}: {message}")
        self.url = url
        self.message = message


@dataclass(frozen=True, slots=True)
class RepoFetchResult:
    """The files collected from a cloned repository.

    ``ref`` is the full ref name that was fetched (``refs/heads/<branch>`` or ``refs/tags/<tag>``),
    a commit SHA, or ``None`` for the default branch. ``commit`` is the commit that was scanned,
    or ``None`` for an empty repository. ``subdir`` is the repo-relative path that was scanned,
    or ``None`` for the whole repository. Paths in ``files`` and ``skipped`` are POSIX-style and
    relative to the repository root (not to ``subdir``), ordered as by
    :func:`~noinjections.fetchers.local.fetch_local`.
    """

    url: str
    ref: str | None
    commit: str | None
    subdir: str | None
    files: list[LocalFile]
    skipped: list[SkippedFile]
    not_scanned: int


@dataclass(frozen=True, slots=True)
class _TreeEntry:
    mode: str
    kind: str
    size: int
    path: str


def fetch_git_repo(
    url: str,
    ref_path: str | None = None,
    *,
    max_repo_bytes: int = DEFAULT_MAX_REPO_BYTES,
    timeout: float = DEFAULT_CLONE_TIMEOUT_SECONDS,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_agent_file_bytes: int = DEFAULT_MAX_AGENT_FILE_BYTES,
    max_files: int = DEFAULT_MAX_FILES,
    allow_file_protocol: bool = False,
) -> RepoFetchResult:
    """Shallow-clone ``url`` into a temporary directory and collect its text files.

    ``ref_path`` is ``<ref>[/<sub/path>]`` as in a GitHub ``/tree/`` URL. The ref is the longest
    leading part that names a branch or tag on the remote (a branch wins over a tag), or a full
    commit SHA. Only the sub-path, if any, is checked out and scanned. Without ``ref_path`` the
    default branch is scanned.

    Only ``https`` remotes are allowed, so the repository can't make git reach other transports.
    ``allow_file_protocol`` also allows ``file://``, for tests only. Untrusted content is handled
    defensively: no credential prompts, no submodules, no hooks, and no Git LFS downloads.

    The fetch is refused if the download grows past ``max_repo_bytes`` or if the files to check
    out add up to more than that. ``timeout`` is one budget for every git command. The clone is
    always deleted before returning. The file caps work as in
    :func:`~noinjections.fetchers.local.fetch_local`. Every problem raises
    :class:`RepoFetchError`.
    """
    if max_repo_bytes < 1:
        raise ValueError(f"max_repo_bytes must be >= 1, got {max_repo_bytes}")
    if timeout <= 0:
        raise ValueError(f"timeout must be > 0, got {timeout}")
    git = shutil.which("git")
    if git is None:
        raise RepoFetchError(url, "git is not installed (it is needed to clone repositories)")
    deadline = time.monotonic() + timeout

    with tempfile.TemporaryDirectory(prefix="noinjections-", ignore_cleanup_errors=True) as tmp:
        work = Path(tmp).resolve()
        repo = work / "repo"
        runner = _GitRunner(git, url, work, deadline, timeout, allow_file_protocol)

        ref, subdir = (None, None) if ref_path is None else _resolve(runner, ref_path)
        runner.run("init", "--quiet", "--template=", str(repo))
        try:
            runner.run(
                *("fetch", "--quiet", "--depth", "1", "--no-tags", "--no-recurse-submodules"),
                *("--", url, ref or "HEAD"),
                cwd=repo,
                watch=repo,
                max_bytes=max_repo_bytes,
            )
        except RepoFetchError as exc:
            if ref is None and "couldn't find remote ref HEAD" in exc.message:
                return RepoFetchResult(url, None, None, None, [], [], 0)  # empty repository
            raise
        commit = runner.run("rev-parse", "--verify", "FETCH_HEAD^{commit}", cwd=repo)
        commit_sha = commit.decode("ascii").strip()

        entries = _list_tree(runner, repo, commit_sha, subdir)
        _check_tree(url, entries, subdir, max_repo_bytes)
        runner.run("checkout", "--quiet", commit_sha, "--", subdir or ".", cwd=repo)

        scan_root = repo / subdir if subdir else repo
        # ls-tree already refused a symlinked sub-path; this is a second line of defence.
        if not scan_root.resolve().is_relative_to(repo):
            raise RepoFetchError(url, f"path {subdir!r} resolves outside the repository")
        local = fetch_local(
            scan_root,
            max_file_bytes=max_file_bytes,
            max_agent_file_bytes=max_agent_file_bytes,
            max_files=max_files,
        )
        prefix = local.root.relative_to(repo).as_posix()
        files = [replace(f, path=_join(prefix, f.path)) for f in local.files]
        skipped = [replace(s, path=_join(prefix, s.path)) for s in local.skipped]
        return RepoFetchResult(url, ref, commit_sha, subdir, files, skipped, local.not_scanned)


def _join(prefix: str, path: str) -> str:
    if prefix == ".":
        return path
    return prefix if path == "." else f"{prefix}/{path}"


def _resolve(runner: "_GitRunner", ref_path: str) -> tuple[str, str | None]:
    """Split ``ref_path`` into ``(ref, subdir)`` by asking the remote which refs exist."""
    parts = [p for p in ref_path.split("/") if p]
    if not parts:
        raise RepoFetchError(runner.url, "empty ref")
    if any(p in {".", ".."} for p in parts):
        raise RepoFetchError(runner.url, f"invalid path in {ref_path!r}")
    prefixes = ["/".join(parts[:n]) for n in range(len(parts), 0, -1)]
    names = [p for p in prefixes if not _NOT_IN_REF.search(p)]
    found: set[str] = set()
    if names:
        patterns = [f"refs/{kind}/{name}" for name in names for kind in ("heads", "tags")]
        out = runner.run("ls-remote", "--heads", "--tags", "--", runner.url, *patterns)
        lines = out.decode("utf-8", errors="replace").splitlines()
        found = {line.split("\t", 1)[1] for line in lines if "\t" in line}

    for n, name in zip(range(len(parts), 0, -1), prefixes, strict=True):
        for ref in (f"refs/heads/{name}", f"refs/tags/{name}"):
            if ref in found:
                return ref, "/".join(parts[n:]) or None
    sha = parts[0].lower()
    if _FULL_SHA.fullmatch(sha):
        return sha, "/".join(parts[1:]) or None
    hint = " (commit SHAs must be given in full)" if _SHORT_SHA.fullmatch(sha) else ""
    raise RepoFetchError(runner.url, f"no branch or tag matches {ref_path!r}{hint}")


def _list_tree(
    runner: "_GitRunner", repo: Path, commit: str, subdir: str | None
) -> list[_TreeEntry]:
    """Every blob and submodule under ``subdir`` (or the whole tree), with blob sizes."""
    args = ["ls-tree", "-r", "-l", "-z", commit]
    if subdir:
        args += ["--", subdir]
    out = runner.run(*args, cwd=repo)
    entries = []
    for record in out.split(b"\0"):
        if not record:
            continue
        meta, _, path = record.partition(b"\t")
        mode, kind, _object, size = meta.split()
        entries.append(
            _TreeEntry(
                mode.decode(), kind.decode(), int(size) if size != b"-" else 0, os.fsdecode(path)
            )
        )
    return entries


def _check_tree(
    url: str, entries: list[_TreeEntry], subdir: str | None, max_repo_bytes: int
) -> None:
    """Refuse a missing, symlinked, or submodule sub-path, and a checkout over the size cap."""
    if subdir:
        if not entries:
            raise RepoFetchError(url, f"path {subdir!r} not found in the repository")
        exact = next((e for e in entries if e.path == subdir), None)
        if exact is not None and exact.mode == _MODE_SYMLINK:
            raise RepoFetchError(url, f"path {subdir!r} is a symlink")
        if exact is not None and exact.kind == "commit":
            raise RepoFetchError(url, f"path {subdir!r} is a submodule (not scanned)")
    total = sum(e.size for e in entries)
    if total > max_repo_bytes:
        raise RepoFetchError(
            url, f"checkout would be {total} bytes, over the {max_repo_bytes} byte cap"
        )


class _GitRunner:
    """Runs hardened git commands for one fetch, sharing one deadline.

    Commands run in ``work`` (an empty temporary directory) unless told otherwise.
    """

    def __init__(
        self,
        git: str,
        url: str,
        work: Path,
        deadline: float,
        timeout: float,
        allow_file_protocol: bool,
    ) -> None:
        self.url = url
        self._work = work
        hooks = work / "hooks"  # stays empty: no hook ever runs in the clone
        hooks.mkdir()
        self._deadline = deadline
        self._timeout = timeout
        config = {
            # Only https (plus file:// in tests): the remote can't steer git to other transports.
            "protocol.allow": "never",
            "protocol.https.allow": "always",
            "core.hooksPath": str(hooks),
            "core.fsmonitor": "false",
            # Check out the committed bytes exactly, so sizes match ls-tree and line endings are
            # the same on every platform.
            "core.autocrlf": "false",
            "submodule.recurse": "false",
            "credential.interactive": "false",
            # Never run the Git LFS filter, even if the user's config defines or requires it:
            # checkout keeps the pointer files. (GIT_LFS_SKIP_SMUDGE below only stops git-lfs
            # itself from downloading.)
            "filter.lfs.smudge": "",
            "filter.lfs.process": "",
            "filter.lfs.required": "false",
        }
        if allow_file_protocol:
            config["protocol.file.allow"] = "always"
        self._argv = [git]
        for key, value in config.items():
            self._argv += ["-c", f"{key}={value}"]
        # Drop the caller's GIT_* variables (e.g. GIT_DIR) so they can't redirect git.
        self._env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        self._env.update(
            GIT_TERMINAL_PROMPT="0",  # a private repo fails instead of waiting for a password
            GIT_LFS_SKIP_SMUDGE="1",  # keep LFS pointer files; don't download LFS objects
            GIT_LITERAL_PATHSPECS="1",  # a path from the URL can't use pathspec magic
            GIT_OPTIONAL_LOCKS="0",
            # Never look for a repository above the temporary directory: one there (e.g. the
            # untrusted folder the user ran noinjections from) could carry its own config.
            GIT_CEILING_DIRECTORIES=str(work.parent),
            LC_ALL="C",  # English messages, so errors can be recognised
        )

    def run(
        self, *args: str, cwd: Path | None = None, watch: Path | None = None, max_bytes: int = 0
    ) -> bytes:
        """Run git and return its stdout, or raise :class:`RepoFetchError`.

        With ``watch``, git is killed as soon as that directory grows past ``max_bytes``.
        """
        self._check_deadline()
        proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            [*self._argv, *args],
            cwd=cwd or self._work,
            env=self._env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **_new_process_group(),
        )
        try:
            while True:
                try:
                    out, err = proc.communicate(timeout=_POLL_SECONDS)
                    break
                except subprocess.TimeoutExpired:
                    pass
                self._check_deadline()
                self._check_size(watch, max_bytes)
        finally:
            if proc.returncode is None:
                _kill(proc)
        if proc.returncode != 0:
            raise RepoFetchError(self.url, _describe_failure(args[0], err))
        # Once more at the end: a fast fetch can finish between two polls.
        self._check_size(watch, max_bytes)
        return out

    def _check_size(self, watch: Path | None, max_bytes: int) -> None:
        if watch is not None and _dir_size(watch) > max_bytes:
            raise RepoFetchError(self.url, f"repository is over the {max_bytes} byte cap")

    def _check_deadline(self) -> None:
        if time.monotonic() >= self._deadline:
            raise RepoFetchError(self.url, f"timed out after {self._timeout:g} s")


def _new_process_group() -> dict[str, object]:
    """Popen arguments that put git in its own process group, so its helpers can be killed."""
    if os.name == "posix":
        return {"start_new_session": True}
    return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}


def _kill(proc: subprocess.Popen[bytes]) -> None:
    """Kill git and the helpers it started (remote-https, index-pack), then reap it."""
    if os.name == "posix":
        with contextlib.suppress(OSError):
            os.killpg(proc.pid, signal.SIGKILL)
    else:
        taskkill = shutil.which("taskkill")
        if taskkill is not None:
            subprocess.run(  # noqa: S603 - fixed argv, no shell
                [taskkill, "/F", "/T", "/PID", str(proc.pid)], capture_output=True, check=False
            )
    proc.kill()
    proc.wait()
    for pipe in (proc.stdout, proc.stderr):
        if pipe is not None:
            pipe.close()


def _dir_size(path: Path) -> int:
    """Total size of the files under ``path``. Files may vanish while it runs (git renames)."""
    total = 0
    for dirpath, _dirs, names in os.walk(path):
        for name in names:
            with contextlib.suppress(OSError):
                total += os.lstat(os.path.join(dirpath, name)).st_size
    return total


def _describe_failure(command: str, stderr: bytes) -> str:
    text = stderr.decode("utf-8", errors="replace")
    lowered = text.lower()
    if (
        "repository not found" in lowered
        or "could not read username" in lowered
        or "authentication failed" in lowered
    ):
        return "repository not found, or it is private (no credentials were available)"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    errors = [line for line in lines if line.startswith(("fatal:", "error:"))]
    detail = "; ".join(errors or lines[-1:]) or "no error message"
    return f"git {command} failed: {detail}"[:_MAX_ERROR_CHARS]
