"""Checks shared by every fetcher that returns file contents: size caps, binary sniff, decoding."""

import codecs
from dataclasses import dataclass
from enum import StrEnum

DEFAULT_MAX_FILE_BYTES = 1024 * 1024
"""Default per-file size cap (1 MiB). Larger files are scanned up to the cap, then cut off."""

DEFAULT_MAX_AGENT_FILE_BYTES = 10 * 1024 * 1024
"""Default cap for agent files (10 MiB). AI tools load these whole, so they get a higher cap."""

SNIFF_BYTES = 8192
"""How many leading bytes are checked for a null byte (the same heuristic git uses)."""

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


def decode_text(data: bytes, *, truncated: bool = False) -> str:
    """Decode file bytes to text, or raise :class:`ContentRejected` for binary content.

    A UTF-8/16/32 byte-order mark selects that codec and is stripped. Otherwise a null byte in the
    first :data:`SNIFF_BYTES` marks the content as binary, and the rest is decoded as UTF-8.
    Invalid bytes become U+FFFD instead of rejecting the file, so one stray byte can't hide a
    payload from the scan. Line endings are kept as they are.

    Pass ``truncated=True`` when ``data`` was cut off at a size cap: a character split by the cut
    is dropped instead of becoming U+FFFD.
    """
    encoding = "utf-8"
    for bom, bom_encoding in _BOMS:
        if data.startswith(bom):
            encoding = bom_encoding
            data = data[len(bom) :]
            break
    else:
        if b"\x00" in data[:SNIFF_BYTES]:
            raise ContentRejected(SkipReason.BINARY)
    decoder = codecs.getincrementaldecoder(encoding)(errors="replace")
    return decoder.decode(data, final=not truncated)
