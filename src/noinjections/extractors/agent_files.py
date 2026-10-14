"""Files that AI coding tools load automatically ("agent files").

Minimal for now: the local fetcher only uses this to put agent files first, so a file cap never
drops them. Checkpoint 2.5 makes the list configurable and tags matching chunks.
"""

import re

from noinjections.globs import translate

AGENT_FILE_GLOBS: tuple[str, ...] = (
    "AGENTS.md",
    "AGENT.md",
    "CLAUDE.md",
    "CLAUDE.local.md",
    ".claude/**",
    "SKILL.md",
    "GEMINI.md",
    ".gemini/**",
    ".cursorrules",
    ".cursor/rules/**",
    ".github/copilot-instructions.md",
    ".github/instructions/**",
    "*.prompt.md",
    "llms.txt",
    ".well-known/ai-plugin.json",
    ".mcp.json",
    "mcp.json",
)
"""Gitignore-style globs, matched at any depth and case-insensitively."""

_AGENT_FILE_RE = re.compile(
    "^(?:.*/)?(?:" + "|".join(translate(g) for g in AGENT_FILE_GLOBS) + ")$",
    re.IGNORECASE,
)


def is_agent_file(path: str) -> bool:
    """Whether a POSIX relative ``path`` is a file AI tools load automatically."""
    return _AGENT_FILE_RE.match(path) is not None
