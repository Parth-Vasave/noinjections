"""Checks shared by every fetcher that returns file contents: size caps, binary sniff, decoding."""

import codecs
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath

from noinjections.extractors.agent_files import is_agent_file

DEFAULT_MAX_FILE_BYTES = 1024 * 1024
"""Default per-file size cap (1 MiB). Larger files are scanned up to the cap, then cut off."""

DEFAULT_MAX_AGENT_FILE_BYTES = 10 * 1024 * 1024
"""Default cap for agent files (10 MiB). AI tools load these whole, so they get a higher cap."""

SNIFF_BYTES = 8192
"""How many leading bytes are checked for a null byte (the same heuristic git uses)."""

TEXT_EXTENSIONS = frozenset({".md", ".markdown", ".mdx", ".mdc", ".txt", ".rst", ".adoc"})
"""Prose formats that are always text, so they never go through the binary sniff."""

# Longest BOMs first: the UTF-32-LE BOM starts with the UTF-16-LE one.
_BOMS = (
    (codecs.BOM_UTF32_LE, "utf-32-le"),
    (codecs.BOM_UTF32_BE, "utf-32-be"),
    (codecs.BOM_UTF8, "utf-8"),
    (codecs.BOM_UTF16_LE, "utf-16-le"),
    (codecs.BOM_UTF16_BE, "utf-16-be"),
)


class SkipReason(StrEnum):
    """Why a file was not scanned."""

    BINARY = "binary"
    SYMLINK = "symlink"
    NOT_REGULAR = "not_regular"
    UNREADABLE = "unreadable"


@dataclass(frozen=True, slots=True)
class SkippedFile:
    """A file that was found but not scanned."""

    path: str
    reason: SkipReason
    detail: str = ""


class ContentRejected(Exception):
    """The bytes can't be scanned as text."""

    def __init__(self, reason: SkipReason, detail: str = "") -> None:
        super().__init__(detail or reason.value)
        self.reason = reason
        self.detail = detail


def text_encoding_for(path: str) -> str | None:
    """``"utf-8"`` if ``path`` is text by its name, else ``None`` (to pass to :func:`decode_text`).

    Files with a :data:`TEXT_EXTENSIONS` extension and agent files are text by name. Their
    content then skips the binary sniff, so a null byte planted near the top of an ``AGENTS.md``
    or ``notes.txt`` can't get it skipped as binary.
    """
    if is_agent_file(path) or PurePosixPath(path).suffix.lower() in TEXT_EXTENSIONS:
        return "utf-8"
    return None


def decode_text(data: bytes, *, truncated: bool = False, encoding: str | None = None) -> str:
    """Decode file bytes to text, or raise :class:`ContentRejected` for binary content.

    A UTF-8/16/32 byte-order mark selects that codec and is stripped. Otherwise, if ``encoding``
    is given (the source declared it, e.g. an HTTP ``charset``), the data is decoded with it.
    With neither, a null byte in the first :data:`SNIFF_BYTES` marks the content as binary, and
    the rest is decoded as UTF-8. A declared encoding skips that sniff: the source already says
    the content is text, and one planted null byte must not hide it from the scan. ``encoding``
    must name a text codec Python knows.

    Invalid bytes become U+FFFD instead of rejecting the file, so one stray byte can't hide a
    payload from the scan. Line endings are kept as they are.

    Pass ``truncated=True`` when ``data`` was cut off at a size cap: a character split by the cut
    is dropped instead of becoming U+FFFD.
    """
    for bom, bom_encoding in _BOMS:
        if data.startswith(bom):
            encoding = bom_encoding
            data = data[len(bom) :]
            break
    else:
        if encoding is None:
            if b"\x00" in data[:SNIFF_BYTES]:
                raise ContentRejected(SkipReason.BINARY)
            encoding = "utf-8"
    decoder = codecs.getincrementaldecoder(encoding)(errors="replace")
    return decoder.decode(data, final=not truncated)
