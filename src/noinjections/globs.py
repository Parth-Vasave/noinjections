"""Gitignore-style glob patterns, translated to regular expressions."""

import re


def translate(pattern: str) -> str:
    """Translate a gitignore-style glob into a regex fragment, without anchors.

    ``*`` and ``?`` never match ``/``. A whole-segment ``**`` matches any number of directories:
    ``**/x`` matches ``x`` at any depth and ``x/**`` matches everything under ``x``. A ``**`` inside
    a segment acts like ``*``. ``[...]`` is a character class (``[!...]`` negates it), and a
    backslash escapes the next character.
    """
    out: list[str] = []
    i, n = 0, len(pattern)
    while i < n:
        c = pattern[i]
        if c == "*":
            end = i + 2 if pattern.startswith("**", i) else i + 1
            whole_segment = (i == 0 or pattern[i - 1] == "/") and (end == n or pattern[end] == "/")
            if end == i + 2 and whole_segment:
                if end == n:  # trailing "**"
                    out.append(".*")
                    i = end
                else:  # "**/": zero or more directories
                    out.append("(?:.*/)?")
                    i = end + 1
                continue
            out.append("[^/]*")
            i = end
        elif c == "?":
            out.append("[^/]")
            i += 1
        elif c == "[":
            cls, i = _char_class(pattern, i)
            out.append(cls)
        elif c == "\\" and i + 1 < n:
            out.append(re.escape(pattern[i + 1]))
            i += 2
        else:
            out.append(re.escape(c))
            i += 1
    return "".join(out)


def _char_class(pattern: str, start: int) -> tuple[str, int]:
    """Translate the ``[...]`` class at ``start``; an unclosed ``[`` is a literal."""
    j = start + 1
    if j < len(pattern) and pattern[j] in "!^":
        j += 1
    if j < len(pattern) and pattern[j] == "]":  # "]" first in the class is a member
        j += 1
    close = pattern.find("]", j)
    if close == -1:
        return re.escape("["), start + 1
    body = pattern[start + 1 : close]
    negate = body[:1] in ("!", "^")
    if negate:
        body = body[1:]
    body = body.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
    # Classes never match "/", just like "*" and "?".
    return f"(?!/)[{'^' if negate else ''}{body}]", close + 1
