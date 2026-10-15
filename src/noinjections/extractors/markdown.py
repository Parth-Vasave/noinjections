"""Split a Markdown document into text layers, with source line numbers.

markdown-it-py parses the document (CommonMark, plus the GFM table and strikethrough
extensions) and renders it to HTML, which :func:`~noinjections.extractors.html_layers.extract_html`
splits into layers. Going through HTML means raw HTML in the Markdown is judged as a browser
would judge it: a ``<details>`` opened in one HTML block and closed in another wraps the
Markdown between them, and ``<span style="display:none">`` in a paragraph hides the text up to
its ``</span>``. HTML in code spans and code blocks (a comment, a ``<details>``) is escaped by
the renderer, so it stays visible text.

The HTML is rendered token by token, recording the source line each output line came from (see
:class:`_LineMap`), so segments carry Markdown line numbers. Columns aren't recorded: container
markers (``>``, list indentation) are stripped before rendering, so output columns don't match
the source.
"""

from collections.abc import Callable, Sequence
from functools import cache

from markdown_it import MarkdownIt
from markdown_it.parser_inline import RuleFuncInlineType
from markdown_it.renderer import RendererHTML
from markdown_it.rules_block.state_block import StateBlock
from markdown_it.rules_inline import backtick, image, link
from markdown_it.rules_inline.state_inline import StateInline
from markdown_it.token import Token
from markdown_it.utils import EnvType, OptionsDict

from noinjections.extractors.html_layers import SKIPPED_URLS, extract_html
from noinjections.models import Layer, Provenance, SourceKind, TextSegment

MAX_NESTING = 20
"""Deepest nesting markdown-it parses (a blockquote is one level, a nested list two).

markdown-it skips blocks nested deeper; this extractor keeps their source lines as raw
``visible_text``, so their text is still scanned.
"""

_TOO_DEEP = "noinjections_too_deep"
"""``env`` key: ``(start, end)`` 0-based line ranges markdown-it skipped for being too deep."""

_SOURCE_NEWLINES = "noinjections_source_newlines"
"""``Token.meta`` key: source newlines an inline token covers, where its content doesn't show."""

_ASCII_WS = " \t\n\f\r"
"""Whitespace stripped from segments. NBSP, zero-width, and other Unicode spaces are content."""

_VERBATIM_BLOCKS = frozenset({"code_block", "fence", "html_block"})
"""Block tokens whose output keeps the source's lines one to one."""


def extract_markdown(
    text: str, *, location: str, source: SourceKind = SourceKind.LOCAL
) -> list[TextSegment]:
    """Extract every text layer of a Markdown document as :class:`TextSegment` items.

    ``location`` (a path or URL) and ``source`` go into each segment's provenance. Segments are
    in source order, and ``line`` is the 1-based Markdown line the text starts on.

    Layers:

    * ``md_comment``: HTML comments (``<!-- ... -->``, block or inline), and link reference
      definitions (``[label]: url "title"``, including the ``[//]: # (comment)`` idiom), which
      aren't shown. One segment per definition: the label, then the title. (Where a link uses
      the definition, its title is also the link's ``title_attr``.)
    * ``alt``: image alt text. ``title_attr``: link and image titles (``[x](url "title")``).
    * ``url``: link destinations, image sources, autolinks, and definition URLs, as markdown-it
      normalises them (percent-encoded). An empty URL or ``#`` is skipped.
    * ``css_hidden``: the content of a closed ``<details>`` block (reason ``"closed
      <details>"``), and text hidden by raw HTML as in
      :func:`~noinjections.extractors.html_layers.extract_html` (``hidden``, inline
      ``display:none``, ...).
    * ``visible_text`` and the other HTML layers, for the rendered document and its raw HTML.

    ``selector`` and ``column`` are always ``None``.
    """
    md = _parser()
    env: EnvType = {}
    tokens = md.parse(text, env)
    blocks = [token for token in tokens if token.type != "definition"]
    html, lines = _render(md, blocks, env)

    segments = [
        _relocate(segment, lines)
        for segment in extract_html(html, location=location, source=source)
    ]
    extra: list[TextSegment | None] = []
    for token in tokens:
        if token.type == "definition" and token.map is not None:
            parts = (token.meta.get("label"), token.meta.get("title"))
            label_and_title = "\n".join(str(part).strip(_ASCII_WS) for part in parts if part)
            line = token.map[0] + 1
            extra.append(_segment(label_and_title, Layer.MD_COMMENT, line, location, source))
            url = str(token.meta.get("url") or "")
            if url.strip(_ASCII_WS) not in SKIPPED_URLS:
                extra.append(_segment(url, Layer.URL, line, location, source))
    source_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for start, end in env.get(_TOO_DEEP, []):
        raw = "\n".join(source_lines[start:end])
        extra.append(_segment(raw, Layer.VISIBLE_TEXT, start + 1, location, source))
    segments.extend(segment for segment in extra if segment is not None)
    return sorted(segments, key=lambda segment: segment.provenance.line or 0)


# --- parsing -------------------------------------------------------------------------------


@cache
def _parser() -> MarkdownIt:
    md = MarkdownIt("commonmark", {"maxNesting": MAX_NESTING, "inline_definitions": True})
    md.enable(["table", "strikethrough"])
    for name, rule in (("backticks", backtick), ("link", link), ("image", image)):
        md.inline.ruler.at(name, _counting(rule))
    md.add_render_rule("image", _render_image)
    md.block.tokenize = _guarded_tokenize(md.block.tokenize)  # type: ignore[method-assign]
    return md


def _counting(rule: RuleFuncInlineType) -> RuleFuncInlineType:
    """Wrap an inline rule to record the source newlines its tokens cover.

    The last token the rule pushes (a code span, an image, a link's closing token) gets the
    newlines the rule consumed that the tokens before it don't account for: those in a link's
    destination or reference label, in an image's label, or in a code span (the renderer turns
    them into spaces).
    """

    def counted(state: StateInline, silent: bool) -> bool:
        start, count = state.pos, len(state.tokens)
        if not rule(state, silent):
            return False
        if not silent and len(state.tokens) > count:
            pushed = state.tokens[count:]
            newlines = state.src.count("\n", start, state.pos)
            pushed[-1].meta[_SOURCE_NEWLINES] = newlines - sum(map(_source_newlines, pushed[:-1]))
        return True

    return counted


def _guarded_tokenize(
    tokenize: Callable[[StateBlock, int, int], None],
) -> Callable[[StateBlock, int, int], None]:
    """Wrap the block tokenizer to record the lines it skips at the nesting limit."""

    def guarded(state: StateBlock, start_line: int, end_line: int) -> None:
        if state.level >= state.md.options.maxNesting and start_line < end_line:
            state.env.setdefault(_TOO_DEEP, []).append((start_line, end_line))
        tokenize(state, start_line, end_line)

    return guarded


# --- rendering -----------------------------------------------------------------------------


class _LineMap:
    """The source line of each output line, built as rendered pieces are appended."""

    def __init__(self) -> None:
        self.lines = [1]  # lines[i]: the source line of output line i + 1
        self.line = 1  # the source line the next piece starts on
        self.fresh = True  # the last output line has no content yet, so its line can change

    def add(self, piece: str, newlines: int) -> None:
        """Account for ``piece``, which covers ``newlines`` newlines of the source.

        When the piece has as many newlines as its source, each output line maps to the next
        source line. Otherwise (tags the renderer puts on their own lines, an entity that
        decodes to a newline), the piece maps to the line it starts on, except its last output
        line: what follows the piece continues that line, so it maps to where the piece ends.
        """
        parts = piece.split("\n")
        last = len(parts) - 1
        one_to_one = last == newlines
        end_line = self.line + newlines
        for index, part in enumerate(parts):
            if index:
                if one_to_one:
                    self.line += 1
                elif index == last:
                    self.line = end_line
                self.lines.append(self.line)
                self.fresh = True
            if part and self.fresh:
                self.lines[-1] = self.line
                self.fresh = False
        self.line = end_line


def _render(md: MarkdownIt, tokens: Sequence[Token], env: EnvType) -> tuple[str, list[int]]:
    """Render ``tokens`` to HTML, as ``md.render`` does, and the source line of each output line."""
    renderer, options = md.renderer, md.options
    pieces: list[str] = []
    line_map = _LineMap()
    for index, token in enumerate(tokens):
        if token.map is not None:
            # A fence's content, which starts the output line, is on the line after the fence.
            line_map.line = token.map[0] + 1 + (token.type == "fence")
        if token.type == "inline":
            children = token.children or []
            for child_index, child in enumerate(children):
                if child.type in renderer.rules:
                    piece = renderer.rules[child.type](children, child_index, options, env)
                else:
                    piece = renderer.renderToken(children, child_index, options, env)
                newlines = _source_newlines(child)
                if (missing := newlines - piece.count("\n")) > 0 and piece.endswith(">"):
                    # Put the newlines the output lacks inside the closing tag, where they don't
                    # change the text, so the following text maps to the right line.
                    end = len(piece) - (2 if piece.endswith("/>") else 1)
                    piece = piece[:end] + "\n" * missing + piece[end:]
                line_map.add(piece, newlines)
                pieces.append(piece)
            continue
        if token.type in renderer.rules:
            piece = renderer.rules[token.type](tokens, index, options, env)
        else:
            piece = renderer.renderToken(tokens, index, options, env)
        line_map.add(piece, piece.count("\n") if token.type in _VERBATIM_BLOCKS else 0)
        pieces.append(piece)
    return "".join(pieces), line_map.lines


def _source_newlines(token: Token) -> int:
    """How many newlines of the source an inline token covers."""
    if _SOURCE_NEWLINES in token.meta:
        return int(token.meta[_SOURCE_NEWLINES])
    if token.type in ("softbreak", "hardbreak"):
        return 1
    if token.type == "html_inline":
        return token.content.count("\n")
    if token.type == "link_open":
        return str(token.attrGet("title") or "").count("\n")
    return 0


def _render_image(
    self: RendererHTML, tokens: Sequence[Token], idx: int, options: OptionsDict, env: EnvType
) -> str:
    """Render an image with all the text of its label as ``alt``.

    markdown-it-py's ``alt`` keeps only plain text and soft breaks, so escaped characters,
    entities, code spans, and inline HTML in the label would never be scanned.
    """
    token = tokens[idx]
    token.attrSet("alt", _plain_text(token.children or []))
    return self.renderToken(tokens, idx, options, env)


def _plain_text(tokens: Sequence[Token]) -> str:
    parts: list[str] = []
    for token in tokens:
        if token.type in ("text", "text_special", "code_inline", "html_inline"):
            parts.append(token.content)
        elif token.type in ("softbreak", "hardbreak"):
            parts.append("\n")
        elif token.type == "image":
            parts.append(_plain_text(token.children or []))
    return "".join(parts)


# --- segments ------------------------------------------------------------------------------


def _relocate(segment: TextSegment, lines: list[int]) -> TextSegment:
    """Move a segment of the rendered HTML to its Markdown source line."""
    provenance = segment.provenance
    line = None
    if provenance.line is not None and provenance.line <= len(lines):
        line = lines[provenance.line - 1]
    layer = Layer.MD_COMMENT if provenance.layer is Layer.HTML_COMMENT else provenance.layer
    return segment.model_copy(
        update={
            "provenance": provenance.model_copy(
                update={"line": line, "column": None, "selector": None, "layer": layer}
            )
        }
    )


def _segment(
    text: str, layer: Layer, line: int, location: str, source: SourceKind
) -> TextSegment | None:
    text = text.strip(_ASCII_WS)
    if not text:
        return None
    provenance = Provenance(
        source=source,
        path_or_url=location,
        line=line,
        layer=layer,
        visible=layer is Layer.VISIBLE_TEXT,
    )
    return TextSegment(text=text, provenance=provenance)
