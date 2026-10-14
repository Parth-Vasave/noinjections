"""Collect text files from a local directory (or one local file) for scanning."""

import os
import shutil
import stat
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from noinjections.extractors.agent_files import is_agent_file
from noinjections.fetchers.content import (
    DEFAULT_MAX_AGENT_FILE_BYTES,
    DEFAULT_MAX_FILE_BYTES,
    ContentRejected,
    SkippedFile,
    SkipReason,
    decode_text,
)
from noinjections.fetchers.gitignore import IgnoreRule, RuleSet, is_ignored, parse_gitignore

DEFAULT_MAX_FILES = 10_000
"""Default cap on the number of files scanned."""

GIT_TIMEOUT_SECONDS = 60

SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "bower_components",
        ".venv",
        "venv",
        "dist",
        "__pycache__",
        ".tox",
        ".nox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
)
"""Directory names never scanned, at any depth (dependencies, build output, caches)."""

LOCKFILES = frozenset(
    {
        "package-lock.json",
        "npm-shrinkwrap.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "bun.lock",
        "bun.lockb",
        "deno.lock",
        "uv.lock",
        "poetry.lock",
        "Pipfile.lock",
        "pdm.lock",
        "Cargo.lock",
        "go.sum",
        "Gemfile.lock",
        "composer.lock",
        "mix.lock",
        "pubspec.lock",
        "flake.lock",
        "Package.resolved",
        "packages.lock.json",
        "gradle.lockfile",
    }
)
"""File names never scanned: generated lockfiles are large and full of hashes."""

# Don't follow a final symlink, and don't block if the path became a FIFO after lstat().
_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_BINARY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
)


class Listing(StrEnum):
    """How the files to scan were found."""

    GIT = "git"  # git ls-files: tracked plus untracked-but-not-ignored
    WALK = "walk"  # directory walk with the built-in .gitignore matcher
    FILE = "file"  # a single file was given


@dataclass(frozen=True, slots=True)
class LocalFile:
    """A scannable text file.

    ``path`` is POSIX-style and relative to the scan root. ``size`` is the file's full size in
    bytes. ``truncated`` means ``text`` holds only the start of the file, up to the size cap.
    """

    path: str
    text: str
    size: int
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class LocalFetchResult:
    """The files collected from a local target.

    ``files`` holds agent files first, then everything else, each group sorted by path.
    ``not_scanned`` counts the files left out because ``max_files`` was reached.
    """

    root: Path
    files: list[LocalFile]
    skipped: list[SkippedFile]
    not_scanned: int
    listing: Listing


def fetch_local(
    path: Path,
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_agent_file_bytes: int = DEFAULT_MAX_AGENT_FILE_BYTES,
    max_files: int = DEFAULT_MAX_FILES,
    use_git: bool = True,
) -> LocalFetchResult:
    """Collect the text files to scan under ``path``.

    Inside a git work tree the file list comes from ``git ls-files``, so untracked files are
    included unless git ignores them. Otherwise the directory is walked, honouring
    ``.gitignore`` files with a simple matcher. Either way, :data:`SKIP_DIRS`,
    :data:`LOCKFILES`, symlinks, and binaries are skipped, and at most ``max_files`` files are
    read. Files over ``max_file_bytes`` (``max_agent_file_bytes`` for agent files) are read up to
    the cap and marked ``truncated``. A single file is read as-is, with only the size cap and the
    binary check.
    """
    if max_file_bytes < 1:
        raise ValueError(f"max_file_bytes must be >= 1, got {max_file_bytes}")
    if max_agent_file_bytes < 1:
        raise ValueError(f"max_agent_file_bytes must be >= 1, got {max_agent_file_bytes}")
    # Agent files never get a lower cap than ordinary files.
    caps = (max_file_bytes, max(max_file_bytes, max_agent_file_bytes))
    if max_files < 1:
        raise ValueError(f"max_files must be >= 1, got {max_files}")
    target = path.resolve(strict=True)

    if not target.is_dir():
        item = _read_file(target.parent, target.name, caps)
        files = [item] if isinstance(item, LocalFile) else []
        skipped = [item] if isinstance(item, SkippedFile) else []
        return LocalFetchResult(target.parent, files, skipped, 0, Listing.FILE)

    skipped: list[SkippedFile] = []
    candidates = _git_list(target) if use_git else None
    listing = Listing.GIT
    if candidates is None:
        candidates = _walk(target, skipped)
        listing = Listing.WALK
    candidates = sorted(
        (p for p in candidates if not _excluded(p)),
        key=lambda p: (not is_agent_file(p), p),
    )

    files: list[LocalFile] = []
    not_scanned = 0
    for index, rel in enumerate(candidates):
        if len(files) >= max_files:
            not_scanned = len(candidates) - index
            break
        item = _read_file(target, rel, caps)
        if isinstance(item, LocalFile):
            files.append(item)
        elif item is not None:
            skipped.append(item)
    return LocalFetchResult(target, files, skipped, not_scanned, listing)


def _excluded(rel: str) -> bool:
    """Whether a relative path sits under a :data:`SKIP_DIRS` directory or is a lockfile."""
    *dirs, name = rel.split("/")
    return name in LOCKFILES or not SKIP_DIRS.isdisjoint(dirs)


def _git_list(root: Path) -> list[str] | None:
    """List files with ``git ls-files``, or ``None`` if git is missing or ``root`` isn't a repo.

    Also ``None`` when git lists nothing, e.g. because ``root`` itself is git-ignored: the user
    asked for that folder, so it is walked instead.
    """
    git = shutil.which("git")
    if git is None:
        return None
    # Drop the caller's GIT_* variables (e.g. GIT_DIR) so they can't redirect the listing.
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env["GIT_OPTIONAL_LOCKS"] = "0"
    # The folder may be untrusted: a core.fsmonitor in its .git/config is a command that
    # ls-files would run, so force it off.
    cmd = [
        git,
        "-c",
        "core.fsmonitor=false",
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
    ]
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            cmd, cwd=root, env=env, capture_output=True, timeout=GIT_TIMEOUT_SECONDS, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    # Untracked nested repositories are listed as "dir/"; they are reported as not regular.
    paths = {os.fsdecode(p).rstrip("/") for p in proc.stdout.split(b"\0") if p}
    return list(paths) or None


def _walk(root: Path, skipped: list[SkippedFile]) -> list[str]:
    """Walk ``root`` without following symlinks, pruning :data:`SKIP_DIRS` and ignored paths."""
    files: list[str] = []
    stack: list[tuple[str, list[RuleSet]]] = [("", [])]
    while stack:
        rel_dir, rule_sets = stack.pop()
        dir_path = root / rel_dir
        rules = _load_gitignore(dir_path / ".gitignore")
        if rules:
            rule_sets = [*rule_sets, (rel_dir, rules)]
        try:
            with os.scandir(dir_path) as it:
                entries = list(it)
        except OSError as exc:
            skipped.append(SkippedFile(rel_dir or ".", SkipReason.UNREADABLE, _os_detail(exc)))
            continue
        for entry in entries:
            rel = f"{rel_dir}/{entry.name}" if rel_dir else entry.name
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                is_dir = False
            if is_dir:
                if entry.name not in SKIP_DIRS and not is_ignored(rel, True, rule_sets):
                    stack.append((rel, rule_sets))
            elif not is_ignored(rel, False, rule_sets):
                files.append(rel)
    return files


def _load_gitignore(path: Path) -> tuple[IgnoreRule, ...]:
    try:
        return parse_gitignore(path.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return ()


def _read_file(root: Path, rel: str, caps: tuple[int, int]) -> LocalFile | SkippedFile | None:
    """Read one candidate, up to its size cap: ``caps`` is ``(ordinary, agent file)``.

    Returns ``None`` if the file has vanished (e.g. deleted but still in git's index).
    """
    path = root / rel
    try:
        st = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        return SkippedFile(rel, SkipReason.UNREADABLE, _os_detail(exc))
    if stat.S_ISLNK(st.st_mode):
        return SkippedFile(rel, SkipReason.SYMLINK)
    if stat.S_ISDIR(st.st_mode):
        return SkippedFile(rel, SkipReason.NOT_REGULAR, "directory (submodule or nested repo)")
    if not stat.S_ISREG(st.st_mode):
        return SkippedFile(rel, SkipReason.NOT_REGULAR)
    if not path.parent.resolve().is_relative_to(root):
        return SkippedFile(rel, SkipReason.SYMLINK, "a parent directory is a symlink")
    max_bytes = caps[1] if is_agent_file(rel) else caps[0]
    try:
        data = _read_bytes(path, max_bytes + 1)
        truncated = len(data) > max_bytes
        text = decode_text(data[:max_bytes], truncated=truncated)
    except ContentRejected as exc:
        return SkippedFile(rel, exc.reason, exc.detail)
    except OSError as exc:
        return SkippedFile(rel, SkipReason.UNREADABLE, _os_detail(exc))
    return LocalFile(rel, text, size=max(st.st_size, len(data)), truncated=truncated)


def _read_bytes(path: Path, limit: int) -> bytes:
    """Read at most ``limit`` bytes from a regular file, refusing symlinks and special files."""
    fd = os.open(path, _OPEN_FLAGS)
    with os.fdopen(fd, "rb") as f:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ContentRejected(SkipReason.NOT_REGULAR)
        return f.read(limit)


def _os_detail(exc: OSError) -> str:
    return exc.strerror or str(exc)
