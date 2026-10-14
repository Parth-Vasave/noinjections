import pytest

from noinjections.extractors.agent_files import is_agent_file
from noinjections.fetchers.gitignore import is_ignored, parse_gitignore
from noinjections.globs import translate


def ignored(patterns: str, path: str, is_dir: bool = False) -> bool:
    return is_ignored(path, is_dir, [("", parse_gitignore(patterns))])


@pytest.mark.parametrize(
    ("patterns", "path", "is_dir", "expected"),
    [
        ("*.log", "a.log", False, True),
        ("*.log", "deep/dir/a.log", False, True),
        ("*.log", "a.log.txt", False, False),
        ("foo", "a/b/foo", False, True),
        ("foo", "a/foo", True, True),
        ("build/", "build", True, True),
        ("build/", "build", False, False),  # dir-only pattern, but this is a file
        ("build/", "src/build", True, True),
        ("/root.txt", "root.txt", False, True),
        ("/root.txt", "sub/root.txt", False, False),
        ("doc/*.txt", "doc/a.txt", False, True),
        ("doc/*.txt", "doc/sub/a.txt", False, False),  # "*" doesn't cross "/"
        ("doc/*.txt", "x/doc/a.txt", False, False),  # a middle "/" anchors the pattern
        ("**/cache", "cache", True, True),
        ("**/cache", "a/b/cache", True, True),
        ("logs/**", "logs/a/b.txt", False, True),
        ("a/**/b", "a/b", False, True),
        ("a/**/b", "a/x/y/b", False, True),
        ("file?.txt", "file1.txt", False, True),
        ("file?.txt", "file10.txt", False, False),
        ("[ab].txt", "a.txt", False, True),
        ("[!ab].txt", "a.txt", False, False),
        ("[!ab].txt", "c.txt", False, True),
        ("*.log\n!keep.log", "keep.log", False, False),
        ("!keep.log\n*.log", "keep.log", False, True),  # last match wins
        ("# comment\n\n", "# comment", False, False),
        ("\\#hash", "#hash", False, True),
        ("\\!bang", "!bang", False, True),
        ("trailing   ", "trailing", False, True),
        ("[unclosed", "[unclosed", False, True),
    ],
)
def test_gitignore_patterns(patterns: str, path: str, is_dir: bool, expected: bool) -> None:
    assert ignored(patterns, path, is_dir) is expected


def test_nested_gitignore_is_relative_and_overrides_parent() -> None:
    rule_sets = [("", parse_gitignore("*.md\n")), ("docs", parse_gitignore("!*.md\n/local\n"))]
    assert is_ignored("README.md", False, rule_sets)
    assert not is_ignored("docs/guide.md", False, rule_sets)
    assert is_ignored("docs/local", False, rule_sets)
    assert not is_ignored("local", False, rule_sets)
    assert not is_ignored("docsx/local", False, rule_sets)


def test_translate_double_star_inside_segment_acts_like_star() -> None:
    assert translate("a**b") == "a[^/]*b"


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("AGENTS.md", True),
        ("sub/dir/AGENTS.md", True),
        ("CLAUDE.md", True),
        ("claude.md", True),
        ("CLAUDE.local.md", True),
        (".claude/settings.json", True),
        ("pkg/.claude/commands/x.md", True),
        (".cursorrules", True),
        (".cursor/rules/style.mdc", True),
        (".github/copilot-instructions.md", True),
        (".github/instructions/py.instructions.md", True),
        ("prompts/fix.prompt.md", True),
        ("skills/pdf/SKILL.md", True),
        (".gemini/settings.json", True),
        ("GEMINI.md", True),
        ("llms.txt", True),
        (".well-known/ai-plugin.json", True),
        (".mcp.json", True),
        (".vscode/mcp.json", True),
        ("README.md", False),
        ("NOTAGENTS.md", False),
        ("AGENTS.md.bak", False),
        (".claude", False),
        (".cursor/settings.json", False),
        ("prompt.md", False),
        (".github/workflows/ci.yml", False),
    ],
)
def test_is_agent_file(path: str, expected: bool) -> None:
    assert is_agent_file(path) is expected
