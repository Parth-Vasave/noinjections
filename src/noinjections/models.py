"""Core data models shared by fetchers, extractors, detectors, scoring, and reporters."""

from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

Score = Annotated[float, Field(ge=0.0, le=1.0)]
"""A probability-like score in ``[0, 1]``. NaN and infinities are rejected."""


class _Model(BaseModel):
    """Base for all models: immutable, and unknown fields are ignored on load."""

    model_config = ConfigDict(frozen=True)


class SourceKind(StrEnum):
    """Where the scanned content came from."""

    WEB = "web"
    GITHUB = "github"
    LOCAL = "local"


class Layer(StrEnum):
    """The text layer a chunk was extracted from."""

    VISIBLE_TEXT = "visible_text"
    HTML_COMMENT = "html_comment"
    META = "meta"
    ALT = "alt"
    ARIA = "aria"
    TITLE_ATTR = "title_attr"
    URL = "url"
    NOSCRIPT = "noscript"
    JSON_LD = "json_ld"
    CSS_HIDDEN = "css_hidden"
    MD_COMMENT = "md_comment"
    CODE_COMMENT = "code_comment"
    DOCSTRING = "docstring"
    STRING_LITERAL = "string_literal"
    ISSUE = "issue"
    PR = "pr"
    COMMIT_MSG = "commit_msg"
    AGENT_FILE = "agent_file"
    DECODED_PAYLOAD = "decoded_payload"
    INVISIBLE_UNICODE = "invisible_unicode"


class Severity(StrEnum):
    """Finding severity, derived from a final score via :meth:`from_score`."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @classmethod
    def from_score(cls, score: float) -> "Severity":
        """Map a score in ``[0, 1]`` to a severity. Each threshold is an inclusive lower bound."""
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {score!r}")
        for minimum, severity in _SEVERITY_THRESHOLDS:
            if score >= minimum:
                return severity
        return cls.INFO


# Highest first. HIGH starts at 0.7 to line up with the default ``--threshold``.
_SEVERITY_THRESHOLDS: tuple[tuple[float, Severity], ...] = (
    (0.9, Severity.CRITICAL),
    (0.7, Severity.HIGH),
    (0.5, Severity.MEDIUM),
    (0.3, Severity.LOW),
)


class Provenance(_Model):
    """Exactly where a piece of text came from."""

    source: SourceKind
    path_or_url: str = Field(min_length=1)
    line: int | None = Field(default=None, ge=1, description="1-based line number.")
    column: int | None = Field(default=None, ge=1, description="1-based column number.")
    selector: str | None = Field(default=None, description="CSS selector path, for web content.")
    layer: Layer
    visible: bool = Field(description="Whether a human viewing the content would see this text.")
    hidden_reason: str | None = Field(
        default=None,
        description="How text that would otherwise show is hidden, e.g. 'display:none'.",
    )

    @model_validator(mode="after")
    def _check_hidden_reason(self) -> Self:
        if self.visible and self.hidden_reason is not None:
            raise ValueError("hidden_reason is only allowed when visible is False")
        return self


class TextSegment(_Model):
    """A piece of text from one layer of one source item, as an extractor found it.

    Extractors produce segments; chunking (3.4) splits them into :class:`Chunk` objects.
    ``provenance.line``/``column`` locate the start of the text in the source item.
    """

    text: str = Field(min_length=1)
    provenance: Provenance


class Chunk(_Model):
    """A unit of text handed to detectors.

    ``text`` is the normalised form detectors run on; ``original_text`` is what appeared in the
    source, kept for display. ``char_start``/``char_end`` are offsets into the source item.
    """

    id: str = Field(min_length=1)
    text: str
    original_text: str
    provenance: Provenance
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_span(self) -> Self:
        if self.char_end < self.char_start:
            raise ValueError(f"char_end ({self.char_end}) < char_start ({self.char_start})")
        return self


class DetectorHit(_Model):
    """One detector's verdict on a chunk. Exactly one of ``rule_id`` or ``model`` is set."""

    detector: str = Field(min_length=1)
    rule_id: str | None = None
    model: str | None = None
    score: Score
    categories: list[str] = Field(default_factory=list)
    explanation: str = ""

    @model_validator(mode="after")
    def _check_origin(self) -> Self:
        if (self.rule_id is None) == (self.model is None):
            raise ValueError("exactly one of rule_id or model must be set")
        return self


class Finding(_Model):
    """A chunk with its detector hits and combined score."""

    chunk: Chunk
    hits: list[DetectorHit] = Field(min_length=1)
    final_score: Score
    context_signals: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def severity(self) -> Severity:
        """Severity derived from ``final_score``."""
        return Severity.from_score(self.final_score)


class ScanError(_Model):
    """A non-fatal problem hit while scanning one item."""

    message: str = Field(min_length=1)
    location: str | None = Field(default=None, description="File path or URL, if known.")


class ScanResult(_Model):
    """Everything a scan produced, ready for a reporter."""

    target: str = Field(min_length=1)
    findings: list[Finding] = Field(default_factory=list)
    stats: dict[str, int] = Field(default_factory=dict)
    errors: list[ScanError] = Field(default_factory=list)
    duration: float = Field(ge=0.0, description="Wall-clock scan time in seconds.")
