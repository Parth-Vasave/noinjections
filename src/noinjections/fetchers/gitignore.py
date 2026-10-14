"""A small ``.gitignore`` matcher, used when a folder is not inside a git repository."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from noinjections.globs import translate


@dataclass(frozen=True, slots=True)
class IgnoreRule:
    """One parsed ``.gitignore`` line."""

    regex: re.Pattern[str]
    negate: bool
    dir_only: bool


RuleSet = tuple[str, tuple[IgnoreRule, ...]]
"""Rules from one ``.gitignore``, with its directory relative to the scan root (``""`` = root)."""


def parse_gitignore(text: str) -> tuple[IgnoreRule, ...]:
    """Parse the contents of a ``.gitignore`` file.

    Supports comments, ``!`` negation, a trailing ``/`` (directories only), anchoring (a pattern
    containing ``/`` is relative to the ``.gitignore``'s directory, otherwise it matches a name at
    any depth), and the glob syntax of :func:`noinjections.globs.translate`.
    """
    rules: list[IgnoreRule] = []
    for raw in text.splitlines():
        line = raw
        while line.endswith(" ") and not line.endswith("\\ "):
            line = line[:-1]
        if not line or line.startswith("#"):
            continue
        negate = line.startswith("!")
        if negate:
            line = line[1:]
        dir_only = line.endswith("/")
        line = line.rstrip("/")
        if not line:
            continue
        anchored = "/" in line
        body = translate(line.lstrip("/"))
        regex = re.compile(f"^{body}$" if anchored else f"^(?:.*/)?{body}$")
        rules.append(IgnoreRule(regex=regex, negate=negate, dir_only=dir_only))
    return tuple(rules)


def is_ignored(path: str, is_dir: bool, rule_sets: Sequence[RuleSet]) -> bool:
    """Whether ``path`` (POSIX, relative to the scan root) is ignored.

    ``rule_sets`` go from the root down. The last matching rule wins, so deeper ``.gitignore``
    files override shallower ones, as in git.
    """
    ignored = False
    for base, rules in rule_sets:
        if not base:
            sub = path
        elif path.startswith(f"{base}/"):
            sub = path[len(base) + 1 :]
        else:
            continue
        for rule in rules:
            if rule.dir_only and not is_dir:
                continue
            if rule.regex.match(sub):
                ignored = not rule.negate
    return ignored
