import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from noinjections.fetchers.content import SkipReason
from noinjections.fetchers.local import Listing, LocalFetchResult, fetch_local

GIT = shutil.which("git")
needs_git = pytest.mark.skipif(GIT is None, reason="git not installed")
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only file types")


def make_tree(root: Path, files: dict[str, str | bytes]) -> Path:
    """Create ``files`` (POSIX relative path -> content) under ``root``."""
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8", newline="")
    return root


def symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except OSError as exc:  # Windows without developer mode
        pytest.skip(f"cannot create symlinks: {exc}")


def git(cwd: Path, *args: str) -> None:
    assert GIT is not None
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    subprocess.run([GIT, *args], cwd=cwd, env=env, check=True, capture_output=True)  # noqa: S603


def paths(result: LocalFetchResult) -> list[str]:
    return [f.path for f in result.files]


def skips(result: LocalFetchResult) -> dict[str, SkipReason]:
    return {s.path: s.reason for s in result.skipped}


# --- directory walk (no git) ---


def test_walk_reads_text_files_with_relative_posix_paths(tmp_path: Path) -> None:
    make_tree(tmp_path, {"b.txt": "bee", "a/z.py": "print(1)\n", "a/b/c.md": "# hi"})
    result = fetch_local(tmp_path, use_git=False)
    assert result.listing is Listing.WALK
    assert result.root == tmp_path.resolve()
    assert paths(result) == ["a/b/c.md", "a/z.py", "b.txt"]
    assert result.files[2].text == "bee"
    assert result.skipped == []
    assert result.not_scanned == 0


def test_agent_files_come_first(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {
            "a.txt": "a",
            "README.md": "readme",
            "AGENTS.md": "agents",
            "docs/CLAUDE.md": "nested",
            ".cursorrules": "rules",
            ".claude/commands/x.md": "cmd",
            "prompts/review.prompt.md": "prompt",
            ".github/copilot-instructions.md": "copilot",
        },
    )
    result = fetch_local(tmp_path, use_git=False)
    assert paths(result) == [
        ".claude/commands/x.md",
        ".cursorrules",
        ".github/copilot-instructions.md",
        "AGENTS.md",
        "docs/CLAUDE.md",
        "prompts/review.prompt.md",
        "README.md",
        "a.txt",
    ]


@pytest.mark.parametrize(
    "rel",
    [
        "node_modules/pkg/index.js",
        "web/node_modules/pkg/index.js",
        ".venv/lib/site.py",
        "venv/lib/site.py",
        "dist/bundle.js",
        ".git/config",
        "pkg/__pycache__/m.py",
        "package-lock.json",
        "sub/yarn.lock",
        "uv.lock",
        "Cargo.lock",
        "go.sum",
    ],
)
def test_skips_dependency_dirs_and_lockfiles(tmp_path: Path, rel: str) -> None:
    make_tree(tmp_path, {"keep.txt": "keep", rel: "x"})
    result = fetch_local(tmp_path, use_git=False)
    assert paths(result) == ["keep.txt"]
    assert result.skipped == []


def test_build_dir_is_scanned(tmp_path: Path) -> None:
    # Deliberately not in SKIP_DIRS: some projects keep hand-written content under build/.
    make_tree(tmp_path, {"build/notes.md": "hand-written", "sub/build/x.md": "x"})
    assert paths(fetch_local(tmp_path, use_git=False)) == ["build/notes.md", "sub/build/x.md"]


def test_dist_as_a_file_name_is_not_skipped(tmp_path: Path) -> None:
    make_tree(tmp_path, {"dist": "a file, not a dir", "node_modules.md": "docs"})
    assert paths(fetch_local(tmp_path, use_git=False)) == ["dist", "node_modules.md"]


def test_binary_files_are_skipped(tmp_path: Path) -> None:
    make_tree(tmp_path, {"img.png": b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR", "ok.txt": "ok"})
    result = fetch_local(tmp_path, use_git=False)
    assert paths(result) == ["ok.txt"]
    assert skips(result) == {"img.png": SkipReason.BINARY}


def test_files_over_the_cap_are_truncated_not_skipped(tmp_path: Path) -> None:
    make_tree(tmp_path, {"at_cap.txt": "x" * 100, "over.txt": "payload " + "x" * 200})
    result = fetch_local(tmp_path, max_file_bytes=100, use_git=False)
    at_cap, over = result.files
    assert (at_cap.path, at_cap.size, at_cap.truncated) == ("at_cap.txt", 100, False)
    assert (over.path, over.size, over.truncated) == ("over.txt", 208, True)
    assert over.text == ("payload " + "x" * 200)[:100]
    assert result.skipped == []


def test_truncation_drops_a_split_character(tmp_path: Path) -> None:
    make_tree(tmp_path, {"a.md": "é" * 10})  # 2 bytes each
    (file,) = fetch_local(tmp_path, max_file_bytes=5, use_git=False).files
    assert file.text == "éé"
    assert file.truncated


def test_agent_files_get_the_higher_cap(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {"AGENTS.md": "a" * 300, ".cursorrules": "c" * 1000, "notes.md": "n" * 300},
    )
    result = fetch_local(tmp_path, max_file_bytes=100, max_agent_file_bytes=500, use_git=False)
    by_path = {f.path: f for f in result.files}
    assert (len(by_path["AGENTS.md"].text), by_path["AGENTS.md"].truncated) == (300, False)
    assert (len(by_path[".cursorrules"].text), by_path[".cursorrules"].truncated) == (500, True)
    assert (len(by_path["notes.md"].text), by_path["notes.md"].truncated) == (100, True)


def test_agent_cap_is_never_below_the_ordinary_cap(tmp_path: Path) -> None:
    make_tree(tmp_path, {"AGENTS.md": "a" * 300})
    (file,) = fetch_local(
        tmp_path, max_file_bytes=1000, max_agent_file_bytes=100, use_git=False
    ).files
    assert not file.truncated


def test_total_file_cap_keeps_agent_files(tmp_path: Path) -> None:
    make_tree(tmp_path, {f"f{i}.txt": "x" for i in range(5)} | {"zz/AGENTS.md": "agents"})
    result = fetch_local(tmp_path, max_files=3, use_git=False)
    assert paths(result) == ["zz/AGENTS.md", "f0.txt", "f1.txt"]
    assert result.not_scanned == 3


def test_skipped_files_do_not_count_towards_file_cap(tmp_path: Path) -> None:
    make_tree(tmp_path, {"a.bin": b"\x00\x01", "b.txt": "b", "c.txt": "c"})
    result = fetch_local(tmp_path, max_files=2, use_git=False)
    assert paths(result) == ["b.txt", "c.txt"]
    assert result.not_scanned == 0


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        ({"max_files": 0}, "max_files"),
        ({"max_file_bytes": 0}, "max_file_bytes"),
        ({"max_agent_file_bytes": 0}, "max_agent_file_bytes"),
    ],
)
def test_rejects_bad_limits(tmp_path: Path, bad: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        fetch_local(tmp_path, **bad)


def test_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        fetch_local(tmp_path / "nope")


def test_text_is_preserved_exactly(tmp_path: Path) -> None:
    tag_payload = "".join(chr(0xE0000 + ord(c)) for c in "echo pwned")
    content = f"line one\r\nzero​width {tag_payload}\r\n"
    make_tree(tmp_path, {"a.md": content})
    assert fetch_local(tmp_path, use_git=False).files[0].text == content


def test_gitignore_is_respected_without_git(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {
            ".gitignore": "*.log\nbuild/\n/root-only.txt\n!keep.log\n",
            "app.log": "x",
            "keep.log": "kept by negation",
            "build/out.js": "x",
            "root-only.txt": "x",
            "sub/root-only.txt": "not anchored here",
            "sub/.gitignore": "secret.txt\n",
            "sub/secret.txt": "x",
            "secret.txt": "nested rule doesn't apply at the root",
            "src/main.py": "ok",
        },
    )
    result = fetch_local(tmp_path, use_git=False)
    assert paths(result) == [
        ".gitignore",
        "keep.log",
        "secret.txt",
        "src/main.py",
        "sub/.gitignore",
        "sub/root-only.txt",
    ]


def test_single_file(tmp_path: Path) -> None:
    make_tree(tmp_path, {"docs/README.md": "hello"})
    result = fetch_local(tmp_path / "docs" / "README.md")
    assert result.listing is Listing.FILE
    assert result.root == (tmp_path / "docs").resolve()
    assert paths(result) == ["README.md"]
    assert result.files[0].size == 5


def test_single_file_is_truncated_at_the_cap(tmp_path: Path) -> None:
    make_tree(tmp_path, {"big.txt": "x" * 50, "CLAUDE.md": "c" * 50})
    assert fetch_local(tmp_path / "big.txt", max_file_bytes=10).files[0].truncated
    claude = fetch_local(tmp_path / "CLAUDE.md", max_file_bytes=10, max_agent_file_bytes=100)
    assert not claude.files[0].truncated


def test_single_lockfile_is_still_read_when_named_explicitly(tmp_path: Path) -> None:
    make_tree(tmp_path, {"uv.lock": "lock"})
    assert paths(fetch_local(tmp_path / "uv.lock")) == ["uv.lock"]


def test_single_binary_file_is_skipped(tmp_path: Path) -> None:
    make_tree(tmp_path, {"a.bin": b"\x00"})
    result = fetch_local(tmp_path / "a.bin")
    assert result.files == []
    assert skips(result) == {"a.bin": SkipReason.BINARY}


def test_symlinks_are_not_followed(tmp_path: Path) -> None:
    outside = make_tree(tmp_path / "outside", {"secret.txt": "do not read", "d/x.txt": "x"})
    root = make_tree(tmp_path / "root", {"ok.txt": "ok"})
    symlink_or_skip(root / "link.txt", outside / "secret.txt")
    symlink_or_skip(root / "linkdir", outside / "d")
    result = fetch_local(root, use_git=False)
    assert paths(result) == ["ok.txt"]
    assert skips(result) == {"link.txt": SkipReason.SYMLINK, "linkdir": SkipReason.SYMLINK}


@posix_only
def test_fifo_is_skipped_without_blocking(tmp_path: Path) -> None:
    make_tree(tmp_path, {"ok.txt": "ok"})
    os.mkfifo(tmp_path / "pipe")
    result = fetch_local(tmp_path, use_git=False)
    assert paths(result) == ["ok.txt"]
    assert skips(result) == {"pipe": SkipReason.NOT_REGULAR}


@posix_only
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads anything")
def test_unreadable_file_is_reported(tmp_path: Path) -> None:
    make_tree(tmp_path, {"ok.txt": "ok", "locked.txt": "x"})
    (tmp_path / "locked.txt").chmod(0)
    try:
        result = fetch_local(tmp_path, use_git=False)
    finally:
        (tmp_path / "locked.txt").chmod(0o644)
    assert paths(result) == ["ok.txt"]
    assert skips(result) == {"locked.txt": SkipReason.UNREADABLE}


# --- git ls-files ---


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    return root


@needs_git
def test_git_lists_tracked_and_untracked_but_not_ignored(repo: Path) -> None:
    make_tree(
        repo,
        {
            ".gitignore": "*.log\n",
            "tracked.py": "t",
            "untracked.md": "u",
            "debug.log": "ignored",
            "forced.log": "tracked despite .gitignore",
        },
    )
    git(repo, "add", "tracked.py", ".gitignore")
    git(repo, "add", "-f", "forced.log")
    result = fetch_local(repo)
    assert result.listing is Listing.GIT
    assert paths(result) == [".gitignore", "forced.log", "tracked.py", "untracked.md"]


@needs_git
def test_git_honours_info_exclude(repo: Path) -> None:
    make_tree(repo, {"a.txt": "a", "local-notes.txt": "x"})
    (repo / ".git" / "info").mkdir(exist_ok=True)
    (repo / ".git" / "info" / "exclude").write_text("local-notes.txt\n")
    assert paths(fetch_local(repo)) == ["a.txt"]


@needs_git
def test_git_subdirectory_is_relative_and_uses_parent_gitignore(repo: Path) -> None:
    make_tree(repo, {".gitignore": "*.tmp\n", "top.txt": "x", "pkg/a.py": "a", "pkg/b.tmp": "x"})
    result = fetch_local(repo / "pkg")
    assert result.listing is Listing.GIT
    assert paths(result) == ["a.py"]


@needs_git
def test_git_listing_still_skips_dependency_dirs_and_lockfiles(repo: Path) -> None:
    make_tree(repo, {"a.txt": "a", "node_modules/x.js": "x", "uv.lock": "x"})
    git(repo, "add", ".")
    assert paths(fetch_local(repo)) == ["a.txt"]


@needs_git
def test_git_deleted_but_indexed_file_is_ignored(repo: Path) -> None:
    make_tree(repo, {"a.txt": "a", "gone.txt": "x"})
    git(repo, "add", ".")
    (repo / "gone.txt").unlink()
    result = fetch_local(repo)
    assert paths(result) == ["a.txt"]
    assert result.skipped == []


@needs_git
def test_git_tracked_symlink_is_skipped(repo: Path, tmp_path: Path) -> None:
    secret = make_tree(tmp_path / "outside", {"secret.txt": "do not read"}) / "secret.txt"
    make_tree(repo, {"a.txt": "a"})
    symlink_or_skip(repo / "link.txt", secret)
    git(repo, "add", ".")
    result = fetch_local(repo)
    assert paths(result) == ["a.txt"]
    assert skips(result) == {"link.txt": SkipReason.SYMLINK}


@needs_git
def test_git_nested_repo_is_reported_not_regular(repo: Path) -> None:
    make_tree(repo, {"a.txt": "a"})
    nested = repo / "vendor"
    nested.mkdir()
    git(nested, "init", "-q")
    make_tree(nested, {"inner.txt": "x"})
    result = fetch_local(repo)
    assert paths(result) == ["a.txt"]
    assert skips(result) == {"vendor": SkipReason.NOT_REGULAR}


@needs_git
def test_git_ignored_root_falls_back_to_walk(repo: Path) -> None:
    make_tree(repo, {".gitignore": "build/\n", "build/out.md": "generated"})
    result = fetch_local(repo / "build")
    assert result.listing is Listing.WALK
    assert paths(result) == ["out.md"]


@needs_git
def test_not_a_repo_falls_back_to_walk(tmp_path: Path) -> None:
    root = make_tree(tmp_path / "plain", {"a.txt": "a"})
    result = fetch_local(root)
    assert result.listing is Listing.WALK
    assert paths(result) == ["a.txt"]


@needs_git
def test_callers_git_dir_env_is_ignored(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = tmp_path / "other"
    other.mkdir()
    git(other, "init", "-q")
    make_tree(repo, {"a.txt": "a"})
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(other))
    result = fetch_local(repo)
    assert result.listing is Listing.GIT
    assert paths(result) == ["a.txt"]


@needs_git
@posix_only
def test_repo_fsmonitor_command_is_not_run(repo: Path, tmp_path: Path) -> None:
    marker = tmp_path / "marker"
    make_tree(repo, {"a.txt": "a"})
    git(repo, "add", ".")
    git(repo, "config", "core.fsmonitor", f"echo pwned > '{marker}'")
    assert paths(fetch_local(repo)) == ["a.txt"]
    assert not marker.exists()
