"""Split an HTML page into text layers, each with provenance.

Parsing uses selectolax's lexbor backend, which follows the HTML5 parsing algorithm. The tree it
builds (implied ``html``/``head``/``body``/``tbody``, foster-parented table content, and so on)
is the one a browser builds, so the selectors recorded here match what devtools shows.

lexbor doesn't expose source positions. A stdlib :mod:`html.parser` pass records where each tag,
comment, and text run starts, and lexbor nodes are matched to those positions by content, in
document order (see :class:`_SourceIndex`). Line numbers are therefore approximate: where the two
parsers disagree (malformed markup), a node falls back to its nearest ancestor's position.

Hiding is judged from markup: the ``hidden`` attribute, inline ``style`` declarations,
``<template>``, ``<input type=hidden>``, closed ``<details>``, and elements browsers never
render. For a rendered page, the browser has also judged every element's computed style and
geometry (stylesheets, classes, off-screen positioning, clipping, contrast) and marked the
hidden ones with attributes; see :func:`~noinjections.fetchers.render.render_url`.
``aria-hidden`` is deliberately not treated as hiding: it hides text from screen readers, but
sighted readers still see it.
"""

import json
import re
from bisect import bisect_left
from collections import defaultdict, deque
from contextlib import suppress
from dataclasses import dataclass, field, replace
from html.parser import HTMLParser
from itertools import count
from typing import Literal

from selectolax.lexbor import LexborHTMLParser, LexborNode

from noinjections.models import Layer, Provenance, SourceKind, TextSegment

MIN_LITERAL_CHARS = 20
"""An inline-script string literal needs at least this many characters to be extracted."""

MIN_LITERAL_WORDS = 3
"""... and at least this many words, so identifiers, URLs, and CSS class lists are skipped."""

MAX_TEMPLATE_DEPTH = 8
"""``<template>`` content is re-parsed per nesting level. Deeper content is kept as raw markup."""

MAX_SELECTOR_PARTS = 64
"""Selectors deeper than this keep only their last parts, so deep nesting can't blow up output."""

_ASCII_WS = " \t\n\f\r"
"""HTML whitespace. NBSP, zero-width, and other Unicode spaces are content, so they're kept."""

_WS_RUN = re.compile(r"[ \t\n\f\r]+")
_SPACES = re.compile(" {2,}")
_BREAK_SPACES = re.compile(r" *\n[ \n]*")

ATTRIBUTE_LAYERS: dict[str, Layer] = {
    "alt": Layer.ALT,
    "aria-label": Layer.ARIA,
    "aria-description": Layer.ARIA,
    "title": Layer.TITLE_ATTR,
    "href": Layer.URL,
    "src": Layer.URL,
}
"""Attributes whose values are extracted, on any element."""

SKIPPED_URLS = frozenset({"", "#"})
"""``href``/``src`` values that lead nowhere, so aren't extracted."""

_BLOCK_TAGS = frozenset(
    {
        "address", "article", "aside", "blockquote", "body", "caption", "center", "dd",
        "details", "dialog", "dir", "div", "dl", "dt", "fieldset", "figcaption", "figure",
        "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "head", "header", "hgroup", "hr",
        "html", "legend", "li", "listing", "main", "menu", "nav", "ol", "optgroup", "option",
        "p", "plaintext", "pre", "search", "section", "summary", "table", "tbody", "td",
        "tfoot", "th", "thead", "title", "tr", "ul", "xmp",
    }
)  # fmt: skip
"""Elements a browser lays out as blocks by default. Visible text is split at their edges."""

_INLINE_DISPLAYS = frozenset({"contents", "ruby", "ruby-base", "ruby-text"})
_CSS_WIDE_KEYWORDS = frozenset({"inherit", "initial", "unset", "revert", "revert-layer"})

_UNRENDERED_TAGS = frozenset({"noembed", "noframes", "iframe", "datalist", "rp", "title"})
"""Elements whose text content browsers don't render (only the document ``<title>`` shows)."""

_RENDERED_REMOVED = frozenset({"display:none", "not rendered"})
"""Browser hiding reasons for an element that takes no space, so doesn't split visible text."""

_JS_TYPES = frozenset(
    {
        "", "module", "application/ecmascript", "application/javascript",
        "application/x-ecmascript", "application/x-javascript", "text/ecmascript",
        "text/javascript", "text/javascript1.0", "text/javascript1.1", "text/javascript1.2",
        "text/javascript1.3", "text/javascript1.4", "text/javascript1.5", "text/jscript",
        "text/livescript", "text/x-ecmascript", "text/x-javascript",
    }
)  # fmt: skip
"""``<script type>`` values browsers run as JavaScript (the empty string means no ``type``)."""

_JSON_LD_TYPE = "application/ld+json"

Pos = tuple[int, int]
"""A 1-based (line, column) source position."""


def extract_html(
    html: str,
    *,
    location: str,
    source: SourceKind = SourceKind.WEB,
    hidden_marker: str | None = None,
) -> list[TextSegment]:
    """Extract every text layer of an HTML document as :class:`TextSegment` items.

    ``location`` (a URL or path) and ``source`` go into each segment's provenance.
    ``hidden_marker`` is a rendered page's
    :attr:`~noinjections.fetchers.web.WebPage.hidden_marker`: with it, text the browser marked
    as hidden is ``css_hidden`` too, with the browser's reason (``"off-screen"``,
    ``"low contrast (1.00:1)"``, ...). Where the markup alone already hides text, the markup's
    reason is kept.

    Layers:

    * ``visible_text``: rendered text, one segment per block element. The document ``<title>``
      counts (it shows in the tab).
    * ``css_hidden``: text a human can't see because of markup: the ``hidden`` attribute, inline
      ``display:none``, ``visibility:hidden``, ``font-size:0`` (also via the ``font``
      shorthand), ``opacity:0``, ``<template>`` content, ``<input type=hidden>`` values, the
      content of a closed ``<details>`` (all but its first ``<summary>``), and elements
      browsers don't render (``<iframe>`` fallback, ``<noembed>``, a ``<title>``
      other than the document's, and so on). One segment per hiding element, with the reason in
      ``provenance.hidden_reason``. Hidden text inside a paragraph gets its own segment, and
      the visible text around it stays one segment, as a human reads it.
    * ``html_comment`` (also ``<?...>`` and ``<!...>`` bogus comments), ``meta`` (``content``),
      ``alt``, ``aria`` (``aria-label``, ``aria-description``), ``title_attr``, ``url``
      (``href`` and ``src``, except an empty one or ``#``): one segment each.
    * ``noscript``: the text of each ``<noscript>`` element.
    * ``json_ld``: the string values of each ``application/ld+json`` script, one per line.
    * ``string_literal``: long natural-language string literals in inline JavaScript, with
      escapes such as ``\\u0069`` decoded.

    Only ``visible_text`` segments have ``visible=True``. Each segment's ``selector`` is a CSS
    path such as ``html > body > div:nth-of-type(2) > p`` (for a comment: its parent element),
    and ``line``/``column`` locate where the text, or its element, starts in ``html``.
    """
    return _Extractor(html, location=location, source=source, hidden_marker=hidden_marker).run()


# --- source positions ----------------------------------------------------------------------


_Key = tuple[object, ...]


def _element_key(tag: str, attrs: list[tuple[str, str | None]]) -> _Key:
    first: dict[str, str | None] = {}
    for name, value in attrs:
        first.setdefault(name.lower(), value)  # HTML keeps the first of duplicate attributes
    return ("element", tag.lower(), tuple(sorted(first.items(), key=lambda item: item[0])))


class _SourceIndex(HTMLParser):
    """Start positions of tags, comments, and text runs in the source, keyed by their content.

    :meth:`take` hands out the positions for a key in source order, so the k-th lexbor node
    with a given key gets the k-th position recorded for it.
    """

    def __init__(self, html: str) -> None:
        super().__init__(convert_charrefs=True)
        self._positions: defaultdict[_Key, deque[Pos]] = defaultdict(deque)
        # _markupbase still asserts on some malformed declarations. Keep the positions recorded
        # up to there: they're best-effort anyway.
        with suppress(AssertionError):
            self.feed(html)
            self.close()

    def take(self, key: _Key) -> Pos | None:
        """The next unused position recorded for ``key``, if any."""
        positions = self._positions.get(key)
        return positions.popleft() if positions else None

    def _record(self, key: _Key) -> None:
        line, offset = self.getpos()
        self._positions[key].append((line, offset + 1))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._record(_element_key(tag, attrs))

    def handle_comment(self, data: str) -> None:
        self._record(("comment", data))

    def handle_decl(self, decl: str) -> None:
        self._record(("comment", decl))  # lexbor turns <!x> into a comment "x"

    def handle_pi(self, data: str) -> None:
        self._record(("comment", data))

    def handle_data(self, data: str) -> None:
        if data.strip(_ASCII_WS):
            self._record(("text", data))


def _normalize_newlines(html: str) -> str:
    """Normalise CRLF and lone CR to LF, as the HTML parser does before tokenising."""
    return html.replace("\r\n", "\n").replace("\r", "\n")


def _offset_pos(base: Pos, newlines: list[int], offset: int) -> Pos:
    """The position of ``offset`` in a text that starts at ``base`` and has ``newlines``."""
    before = bisect_left(newlines, offset)
    if before == 0:
        return base[0], base[1] + offset
    return base[0] + before, offset - newlines[before - 1]


# --- selectors -----------------------------------------------------------------------------


_CSS_IDENT_UNSAFE = re.compile(r"[^A-Za-z0-9_\-\u0080-\U0010ffff]")


def _css_ident(name: str) -> str:
    """Escape ``name`` for use as a CSS type selector."""

    def escape(match: re.Match[str]) -> str:
        char = match.group()
        return f"\\{ord(char):x} " if char.isspace() or not char.isprintable() else "\\" + char

    return _CSS_IDENT_UNSAFE.sub(escape, name)


@dataclass(frozen=True, slots=True)
class _Path:
    """A CSS selector path, built lazily from the parent chain."""

    parent: "_Path | None"
    part: str

    def __str__(self) -> str:
        parts: list[str] = []
        path: _Path | None = self
        while path is not None and len(parts) < MAX_SELECTOR_PARTS:
            parts.append(path.part)
            path = path.parent
        if path is not None:
            parts.append("...")
        return " > ".join(reversed(parts))


# --- inline styles -------------------------------------------------------------------------


_CSS_COMMENT = re.compile(r"/\*.*?(?:\*/|\Z)", re.DOTALL)
_CSS_ESCAPE = re.compile(r"\\(?:([0-9a-fA-F]{1,6})[ \t\n\r\f]?|(.))", re.DOTALL)
_IMPORTANT = re.compile(r"![ \t\n\r\f]*important$")
_NUMBER = re.compile(r"([+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?)([a-z%]*)")
_FONT_SIZE_KEYWORDS = frozenset(
    {"xx-small", "x-small", "small", "medium", "large", "x-large", "xx-large", "xxx-large"}
)
_RELATIVE_FONT_UNITS = frozenset({"em", "ex", "ch", "%", "cap", "ic", "lh"})


def _css_unescape(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        hex_digits, char = match.groups()
        if hex_digits is None:
            return char
        code = int(hex_digits, 16)
        return chr(code) if 0 < code <= 0x10FFFF and not 0xD800 <= code <= 0xDFFF else "\ufffd"

    return _CSS_ESCAPE.sub(replace, text)


def _parse_style(style: str | None) -> dict[str, str]:
    """Inline ``style`` declarations as ``{property: value}``, lowercased, after the cascade.

    A later declaration wins unless an earlier one is ``!important``. The ``font`` shorthand
    also sets ``font-size`` when it has a size, so ``font: 0/0 a`` reads as ``font-size: 0``.
    """
    if not style:
        return {}
    declarations: dict[str, str] = {}
    important: set[str] = set()
    for declaration in _CSS_COMMENT.sub(" ", style).split(";"):
        name, colon, value = declaration.partition(":")
        if not colon:
            continue
        name = _css_unescape(name).strip(_ASCII_WS).lower()
        value = _css_unescape(value).strip(_ASCII_WS).lower()
        is_important = (match := _IMPORTANT.search(value)) is not None
        if match is not None:
            value = value[: match.start()].rstrip(_ASCII_WS)
        if not name or not value:
            continue  # invalid declaration: ignored, as a browser does
        settings = [(name, value)]
        if name == "font" and (size := _font_shorthand_size(value)) is not None:
            settings.append(("font-size", size))
        for prop, prop_value in settings:
            if prop in important and not is_important:
                continue
            declarations[prop] = prop_value
            if is_important:
                important.add(prop)
    return declarations


_FontSize = Literal["zero", "absolute", "relative"]


def _font_size_kind(value: str) -> _FontSize | None:
    """Classify a ``font-size`` value. ``None`` means unknown or invalid (the parent's applies)."""
    if value in _FONT_SIZE_KEYWORDS:
        return "absolute"
    if value in ("smaller", "larger"):
        return "relative"
    match = _NUMBER.fullmatch(value)
    if match is None:
        return None
    number, unit = float(match.group(1)), match.group(2)
    if number == 0:
        return "zero"
    if number < 0 or not unit:
        return None  # invalid: negative, or a non-zero number without a unit
    return "relative" if unit in _RELATIVE_FONT_UNITS else "absolute"


def _font_shorthand_size(value: str) -> str | None:
    """The size in a ``font`` shorthand (``bold 12px/1.2 serif`` -> ``12px``), if any."""
    for token in value.split():
        size = token.split("/", 1)[0]
        if _font_size_kind(size) is not None:  # a unitless weight such as 700 is skipped
            return size
    return None


def _is_zero_opacity(value: str) -> bool:
    match = _NUMBER.fullmatch(value)
    if match is None or match.group(2) not in ("", "%"):
        return False
    return float(match.group(1)) <= 0


# --- script content ------------------------------------------------------------------------


_JS_TOKEN = re.compile(
    r"//[^\n]*"
    r"|/\*(?:[^*]|\*(?!/))*(?:\*/|\Z)"
    r"""|"((?:[^"\\\n]|\\[\s\S])*)(?:"|(?=\n)|\Z)"""
    r"""|'((?:[^'\\\n]|\\[\s\S])*)(?:'|(?=\n)|\Z)"""
    r"|`((?:[^`\\]|\\[\s\S])*)(?:`|\Z)"
)
"""Comments (skipped, so an apostrophe in one doesn't open a string) and string literals.

An unterminated comment or literal runs to the end of its line or of the script, so every match
consumes its input and scanning stays linear.
"""

_JS_ESCAPE = re.compile(
    r"\\(?:x([0-9A-Fa-f]{2})|u\{([0-9A-Fa-f]{1,6})\}|u([0-9A-Fa-f]{4})|(\r\n|[\s\S]))"
)
_JS_SIMPLE_ESCAPES = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "b": "\b",
    "f": "\f",
    "v": "\v",
    "0": "\0",
    "\n": "",  # line continuation
    "\r\n": "",
    "\u2028": "",
    "\u2029": "",
}


def _fix_surrogates(text: str) -> str:
    """Join escaped UTF-16 surrogate pairs; a lone surrogate becomes U+FFFD."""
    return text.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")


def _js_unescape(raw: str) -> str:
    """Decode the escapes in a JavaScript string literal's body."""

    def replace(match: re.Match[str]) -> str:
        hex2, braced, hex4, other = match.groups()
        if other is not None:
            return _JS_SIMPLE_ESCAPES.get(other, other)
        code = int(hex2 or braced or hex4, 16)
        return chr(code) if code <= 0x10FFFF else match.group()

    return _fix_surrogates(_JS_ESCAPE.sub(replace, raw))


def _json_ld_text(raw: str) -> str:
    """The string values of a JSON-LD document, one per line (``@context`` skipped).

    Invalid JSON is returned as it is, so its text is still scanned.
    """
    try:
        data = json.loads(raw)
    except (ValueError, RecursionError):
        return raw
    strings: list[str] = []
    stack: list[object] = [data]
    while stack:
        value = stack.pop()
        if isinstance(value, str):
            strings.append(value.strip(_ASCII_WS))
        elif isinstance(value, dict):
            stack.extend(reversed([v for k, v in value.items() if k != "@context"]))
        elif isinstance(value, list):
            stack.extend(reversed(value))
    return _fix_surrogates("\n".join(s for s in strings if s))


def _start_tag_end(serialized: str) -> int:
    """Index just past the start tag of a lexbor-serialised element (values are ``"``-quoted)."""
    in_quotes = False
    for index, char in enumerate(serialized):
        if char == '"':
            in_quotes = not in_quotes
        elif char == ">" and not in_quotes:
            return index + 1
    return len(serialized)


# --- the tree walk -------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Hider:
    """An element that hides the text under it. That text is collected into one segment."""

    key: int  # increases in document order, so an outer hider has a smaller key
    path: _Path
    pos: Pos | None
    layer: Layer
    reason: str | None


@dataclass(frozen=True, slots=True)
class _Ctx:
    """What an element passes down to its children."""

    path: _Path | None
    pos: Pos | None  # this element's source position, else the nearest ancestor's
    block: _Path | None  # nearest block-level element: the container of visible text
    fixed: _Hider | None = None  # hiding no descendant can undo (display:none, opacity:0, ...)
    visibility: _Hider | None = None  # visibility:hidden; a descendant can set it back
    font_zero: _Hider | None = None  # font-size 0; a descendant's absolute size undoes it
    rendered: _Hider | None = None  # the browser's mark on this element's own text, if any
    collapsed: _Hider | None = None  # a closed <details>: hides children but the <summary>
    out_of_flow: bool = False  # inside an element that takes no space (display:none, ...)
    template_depth: int = 0

    @property
    def hider(self) -> _Hider | None:
        """The outermost hider in effect, if any; the markup's before the browser's."""
        hiders = [h for h in (self.fixed, self.visibility, self.font_zero) if h is not None]
        return min(hiders, key=lambda h: h.key) if hiders else self.rendered


@dataclass(slots=True)
class _Run:
    """Text being collected for one segment."""

    path: _Path | None
    pos: Pos | None
    layer: Layer
    reason: str | None
    pieces: list[str] = field(default_factory=list)
    pending_break: bool = False


@dataclass(frozen=True, slots=True)
class _Exit:
    """Marks the end of an element in the walk stack."""

    is_block: bool
    out_of_flow: bool
    hider_keys: tuple[int, ...]


class _Extractor:
    def __init__(
        self,
        html: str,
        *,
        location: str,
        source: SourceKind,
        hidden_marker: str | None,
    ) -> None:
        html = _normalize_newlines(html)
        self.location = location
        self.source = source
        self.marker = hidden_marker.lower() if hidden_marker is not None else None
        self.marked_roots: dict[str, _Hider] = {}  # by mark id; only roots being walked
        self.index = _SourceIndex(html)
        self.tree = LexborHTMLParser(html)
        self.fragments: list[LexborHTMLParser] = []  # keeps re-parsed templates alive
        self.segments: list[TextSegment] = []
        self.visible_run: _Run | None = None
        self.hidden_runs: dict[int, _Run] = {}
        self.keys = count()
        self.seen_document_title = False

    def run(self) -> list[TextSegment]:
        document = self.tree.root.parent if self.tree.root is not None else None
        if document is not None:
            self._walk(_child_list(document), _Ctx(path=None, pos=None, block=None))
        self._flush_visible()
        return self.segments

    def _walk(self, nodes: list[LexborNode], ctx: _Ctx) -> None:
        stack: list[tuple[LexborNode, _Ctx, _Path | None] | _Exit] = []
        self._push_children(stack, nodes, ctx)
        while stack:
            item = stack.pop()
            if isinstance(item, _Exit):
                self._exit(item)
                continue
            node, parent_ctx, path = item
            if node.is_element_node and path is not None:
                self._enter(stack, node, parent_ctx, path)
            elif node.is_text_node:
                self._text(node.text_content or "", parent_ctx)
            elif node.is_comment_node:
                # comment_content is stripped; the serialised form keeps the data as written
                self._comment((node.html or "<!---->")[4:-3], parent_ctx)
            elif node.tag is None and (markup := node.html or "").startswith("<?"):
                # lexbor keeps <?...> as a processing instruction; HTML treats it as a comment
                self._comment(markup[2:].removesuffix(">"), parent_ctx)

    def _push_children(
        self,
        stack: list[tuple[LexborNode, _Ctx, _Path | None] | _Exit],
        nodes: list[LexborNode],
        ctx: _Ctx,
    ) -> None:
        tags = [node.tag if node.is_element_node else None for node in nodes]
        totals: defaultdict[str, int] = defaultdict(int)
        for tag in tags:
            if tag is not None:
                totals[tag] += 1
        summary: int | None = None  # index of the child that stays visible in a closed <details>
        content_ctx = ctx
        if ctx.collapsed is not None:
            summary = next((i for i, tag in enumerate(tags) if tag == "summary"), None)
            content_ctx = replace(ctx, fixed=ctx.collapsed, collapsed=None, out_of_flow=True)
            ctx = replace(ctx, collapsed=None)
        seen: defaultdict[str, int] = defaultdict(int)
        items: list[tuple[LexborNode, _Ctx, _Path | None]] = []
        for index, (node, tag) in enumerate(zip(nodes, tags, strict=True)):
            path = None
            if tag is not None:
                seen[tag] += 1
                part = _css_ident(tag)
                if totals[tag] > 1:
                    part += f":nth-of-type({seen[tag]})"
                path = _Path(ctx.path, part)
            items.append((node, ctx if index == summary else content_ctx, path))
        stack.extend(reversed(items))

    def _enter(
        self,
        stack: list[tuple[LexborNode, _Ctx, _Path | None] | _Exit],
        node: LexborNode,
        parent: _Ctx,
        path: _Path,
    ) -> None:
        tag = (node.tag or "").lower()
        attrs = node.attributes
        pos = self.index.take(_element_key(tag, list(attrs.items()))) or parent.pos
        style = _parse_style(attrs.get("style"))
        ctx, new_hiders, is_block = self._child_ctx(node, tag, attrs, style, parent, path, pos)
        if is_block:  # first, so text before this element is emitted before its attributes
            self._block_boundary(flush_visible=not ctx.out_of_flow)

        for name, layer in ATTRIBUTE_LAYERS.items():
            if (value := attrs.get(name)) is None:
                continue
            if layer is Layer.URL and value.strip(_ASCII_WS) in SKIPPED_URLS:
                continue
            self._emit(value, layer, path, pos)
        if tag == "meta" and (content := attrs.get("content")) is not None:
            self._emit(content, Layer.META, path, pos)
        if tag == "input" and (attrs.get("type") or "").strip(_ASCII_WS).lower() == "hidden":
            self._emit(attrs.get("value") or "", Layer.CSS_HIDDEN, path, pos, "input type=hidden")
        if tag == "script":
            self._script(node, attrs, path, pos)
            return
        if tag == "style":
            return

        stack.append(_Exit(is_block, ctx.out_of_flow, tuple(h.key for h in new_hiders)))
        if tag == "template":
            children = self._template_children(node, ctx, path, pos)
        else:
            children = _child_list(node)
        self._push_children(stack, children, ctx)

    def _child_ctx(
        self,
        node: LexborNode,
        tag: str,
        attrs: dict[str, str | None],
        style: dict[str, str],
        parent: _Ctx,
        path: _Path,
        pos: Pos | None,
    ) -> tuple[_Ctx, list[_Hider], bool]:
        """The context for ``node``'s children, the hiders it adds, and whether it's a block."""
        display = style.get("display")
        new_hiders: list[_Hider] = []

        def hider(layer: Layer, reason: str | None) -> _Hider:
            new = _Hider(next(self.keys), path, pos, layer, reason)
            new_hiders.append(new)
            return new

        is_document_title = (
            tag == "title"
            and not self.seen_document_title
            and node.parent is not None
            and node.parent.tag == "head"
        )
        if is_document_title:
            self.seen_document_title = True

        # Hidden this way, an element takes no space, so it doesn't split the text around it.
        removed: str | None = None
        if tag == "template":
            removed = "template"
        elif "hidden" in attrs and display in (None, "none", "revert", "revert-layer"):
            removed = "hidden attribute"
        elif display == "none":
            removed = "display:none"
        elif tag in _UNRENDERED_TAGS and not is_document_title:
            removed = f"<{tag}> is not rendered"

        fixed = parent.fixed
        if fixed is None:
            if removed is not None:
                fixed = hider(Layer.CSS_HIDDEN, removed)
            elif _is_zero_opacity(style.get("opacity", "")):
                fixed = hider(Layer.CSS_HIDDEN, "opacity:0")
            elif tag == "noscript":
                fixed = hider(Layer.NOSCRIPT, None)

        collapsed = None
        if tag == "details" and "open" not in attrs and fixed is None:
            collapsed = hider(Layer.CSS_HIDDEN, "closed <details>")

        visibility = parent.visibility
        if style.get("visibility") in ("hidden", "collapse"):
            visibility = visibility or hider(Layer.CSS_HIDDEN, "visibility:hidden")
        elif style.get("visibility") == "visible":
            visibility = None

        font_zero = parent.font_zero
        font_size = _font_size_kind(style.get("font-size", ""))
        if font_size == "zero":
            font_zero = font_zero or hider(Layer.CSS_HIDDEN, "font-size:0")
        elif font_size == "absolute":
            font_zero = None

        rendered: _Hider | None = None
        rendered_removed = False
        if self.marker is not None:
            for mark_id, reason in _marks(attrs.get(self.marker + "-root")):
                self.marked_roots[mark_id] = hider(Layer.CSS_HIDDEN, reason)
                rendered_removed = rendered_removed or reason in _RENDERED_REMOVED
            for mark_id, reason in _marks(attrs.get(self.marker))[:1]:
                # The root is an ancestor unless re-parsing the serialised DOM moved this
                # element (script-built markup that isn't valid HTML): then it's its own root.
                rendered = self.marked_roots.get(mark_id) or hider(Layer.CSS_HIDDEN, reason)

        if tag == "template":
            is_block = True  # its content is a separate document: keep it apart from neighbours
        elif display not in (None, "none") and display not in _CSS_WIDE_KEYWORDS:
            first = display.split()[0]
            is_block = not first.startswith("inline") and first not in _INLINE_DISPLAYS
        else:
            is_block = tag in _BLOCK_TAGS

        ctx = _Ctx(
            path=path,
            pos=pos,
            block=path if is_block else parent.block,
            fixed=fixed,
            visibility=visibility,
            font_zero=font_zero,
            rendered=rendered,
            collapsed=collapsed,
            out_of_flow=(
                parent.out_of_flow or removed is not None or rendered_removed or tag == "noscript"
            ),
            template_depth=parent.template_depth + (tag == "template"),
        )
        return ctx, new_hiders, is_block

    def _template_children(
        self, node: LexborNode, ctx: _Ctx, path: _Path, pos: Pos | None
    ) -> list[LexborNode]:
        """Template content lives in a separate fragment, so re-parse its markup to walk it."""
        markup = node.html or ""
        inner = markup[_start_tag_end(markup) :].removesuffix("</template>")
        if not inner.strip(_ASCII_WS):
            return []
        if ctx.template_depth > MAX_TEMPLATE_DEPTH:
            self._emit(inner, Layer.CSS_HIDDEN, path, pos, "template")
            return []
        fragment = LexborHTMLParser("<body>" + inner)
        self.fragments.append(fragment)
        return _child_list(fragment.body) if fragment.body is not None else []

    def _exit(self, item: _Exit) -> None:
        if item.is_block:
            self._block_boundary(flush_visible=not item.out_of_flow)
        for key in item.hider_keys:
            if (run := self.hidden_runs.pop(key, None)) is not None:
                self._flush(run)
        if item.hider_keys and self.marked_roots:
            keys = set(item.hider_keys)
            for mark_id in [i for i, h in self.marked_roots.items() if h.key in keys]:
                del self.marked_roots[mark_id]

    def _text(self, data: str, ctx: _Ctx) -> None:
        if not data.strip(_ASCII_WS):
            if data:
                self._append(" ", ctx, None)
            return
        self._append(_WS_RUN.sub(" ", data), ctx, self.index.take(("text", data)) or ctx.pos)

    def _append(self, text: str, ctx: _Ctx, pos: Pos | None) -> None:
        hider = ctx.hider
        if hider is None:
            if self.visible_run is None:
                if pos is None and text == " ":
                    return  # leading whitespace: nothing to start a run with
                self.visible_run = _Run(ctx.block, pos, Layer.VISIBLE_TEXT, None)
            run = self.visible_run
        else:
            run = self.hidden_runs.get(hider.key)
            if run is None:
                if text == " ":
                    return
                run = _Run(hider.path, pos or hider.pos, hider.layer, hider.reason)
                self.hidden_runs[hider.key] = run
        if run.pending_break and run.pieces:
            run.pieces.append("\n")
        run.pending_break = False
        run.pieces.append(text)
        if run.pos is None:
            run.pos = pos

    def _block_boundary(self, *, flush_visible: bool) -> None:
        """A block element starts or ends: visible text breaks, hidden text gets a newline."""
        if flush_visible:
            self._flush_visible()
        for run in self.hidden_runs.values():
            run.pending_break = True

    def _flush_visible(self) -> None:
        if self.visible_run is not None:
            self._flush(self.visible_run)
            self.visible_run = None

    def _flush(self, run: _Run) -> None:
        text = _BREAK_SPACES.sub("\n", _SPACES.sub(" ", "".join(run.pieces)))
        self._emit(text, run.layer, run.path, run.pos, run.reason)

    def _comment(self, data: str, ctx: _Ctx) -> None:
        pos = self.index.take(("comment", data)) or ctx.pos
        self._emit(data, Layer.HTML_COMMENT, ctx.path, pos)

    def _script(
        self, node: LexborNode, attrs: dict[str, str | None], path: _Path, pos: Pos | None
    ) -> None:
        kind = (attrs.get("type") or "").split(";", 1)[0].strip(_ASCII_WS).lower()
        if kind != _JSON_LD_TYPE and kind not in _JS_TYPES:
            return  # templates and data blocks (text/template, application/json, ...)
        code = node.text(deep=True) or ""
        if not code.strip(_ASCII_WS):
            return
        if kind == _JSON_LD_TYPE:
            self._emit(_json_ld_text(code), Layer.JSON_LD, path, pos)
            return
        base = self.index.take(("text", code)) or pos
        newlines = [m.start() for m in re.finditer("\n", code)]
        for match in _JS_TOKEN.finditer(code):
            raw = next((g for g in match.groups() if g is not None), None)
            if raw is None:
                continue  # a comment
            text = _js_unescape(raw).strip(_ASCII_WS)
            if len(text) < MIN_LITERAL_CHARS or len(text.split()) < MIN_LITERAL_WORDS:
                continue
            literal_pos = None if base is None else _offset_pos(base, newlines, match.start())
            self._emit(text, Layer.STRING_LITERAL, path, literal_pos)

    def _emit(
        self,
        text: str,
        layer: Layer,
        path: _Path | None,
        pos: Pos | None,
        hidden_reason: str | None = None,
    ) -> None:
        text = text.strip(_ASCII_WS)
        if not text:
            return
        line, column = pos if pos is not None else (None, None)
        provenance = Provenance(
            source=self.source,
            path_or_url=self.location,
            line=line,
            column=column,
            selector=str(path) if path is not None else None,
            layer=layer,
            visible=layer is Layer.VISIBLE_TEXT,
            hidden_reason=hidden_reason if layer is Layer.CSS_HIDDEN else None,
        )
        self.segments.append(TextSegment(text=text, provenance=provenance))


def _marks(value: str | None) -> list[tuple[str, str]]:
    """The ``(id, reason)`` pairs of a hidden-text mark: ``"<id> <reason>"``, joined by ``;``."""
    marks = []
    for item in (value or "").split(";"):
        mark_id, _, reason = item.strip(_ASCII_WS).partition(" ")
        if mark_id and reason.strip(_ASCII_WS):
            marks.append((mark_id, reason.strip(_ASCII_WS)))
    return marks


def _child_list(node: LexborNode) -> list[LexborNode]:
    children = []
    child = node.child
    while child is not None:
        children.append(child)
        child = child.next
    return children
