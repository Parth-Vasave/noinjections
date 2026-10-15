/* Marks text a human can't see in the rendered page, then serialises the DOM.
 *
 * Called by noinjections.fetchers.render in an isolated world, so page scripts can't patch the
 * APIs used here, and in one synchronous call, so no page script runs between measuring,
 * marking, and serialising.
 *
 * Every element under <body> with text of its own is checked. The first reason found wins:
 * display:none, closed <details>, content-visibility:hidden, no box at all ("not rendered"),
 * visibility:hidden, near-zero opacity (multiplied through ancestors), tiny font-size, zero
 * size, off-screen (left of or above the page), clipped (by overflow, clip, or clip-path), and
 * low contrast against the background. A hidden element gets `<marker>="<id> <reason>"`, and
 * the element where the hiding starts (its root, an ancestor or the element itself) gets
 * `<marker>-root="<id> <reason>"`, several joined with ";". Text sharing a root is one segment.
 *
 * Returns {html, unchecked}: the serialised document (cut just past `limit` characters) and
 * the number of text nodes left unchecked once `maxElements` elements were checked.
 */
(opts) => {
  const { marker, limit, maxElements, minOpacity, minFontPx, minVisiblePx, minContrast } = opts;

  const WHITESPACE = /^[ \t\n\f\r]*$/;
  // Not walked: no rendered text (script, ...), text the static extractor already judges
  // (title, noscript, template, iframe fallback), or form controls whose text has no box of
  // its own even when visible (select options, textarea).
  const SKIPPED = new Set([
    "script", "style", "noscript", "template", "title", "select", "datalist", "textarea",
    "iframe", "noembed", "noframes",
  ]);
  // Displays that overflow doesn't apply to.
  const NO_OVERFLOW = new Set([
    "inline", "contents", "none", "table-row", "table-row-group", "table-header-group",
    "table-footer-group", "table-column", "table-column-group",
  ]);
  const WHITE = { r: 255, g: 255, b: 255 };

  const styles = new Map();
  const style = (el) => {
    let s = styles.get(el);
    if (s === undefined) styles.set(el, (s = getComputedStyle(el)));
    return s;
  };
  const px = (value) => parseFloat(value) || 0;
  const fmt = (n) => String(Number(n.toPrecision(2)));

  // A value computed from the element's own style and its parent's value, memoised.
  const inherited = (cache, el, step, rootValue) => {
    const chain = [];
    let cur = el;
    while (cur && !cache.has(cur)) {
      chain.push(cur);
      cur = cur.parentElement;
    }
    let value = cur ? cache.get(cur) : rootValue;
    for (let i = chain.length - 1; i >= 0; i--) {
      value = step(chain[i], value);
      cache.set(chain[i], value);
    }
    return value;
  };
  // The outermost ancestor-or-self of `el` for which `test` holds, walking up while it holds.
  const climb = (el, test) => {
    let root = el;
    for (let p = el.parentElement; p && test(p); p = p.parentElement) root = p;
    return root;
  };

  // --- display, visibility, opacity, font size -------------------------------------------

  const noneRoots = new Map();
  const noneRoot = (el) =>
    inherited(noneRoots, el, (e, outer) => outer || (style(e).display === "none" ? e : null), null);

  const byDisplay = (el) => {
    const none = noneRoot(el);
    if (none) return ["display:none", none];
    let host = el;
    while (host && style(host).display === "contents") host = host.parentElement;
    if (!host) return null;
    for (let a = host; a; a = a.parentElement) {
      if (a.localName === "details" && !a.open) {
        const summary = a.querySelector(":scope > summary");
        if (!summary || !summary.contains(host)) return ["closed <details>", a];
      }
      if (style(a).contentVisibility === "hidden") return ["content-visibility:hidden", a];
    }
    // No box: e.g. an element inside a shadow host that has no <slot> for it.
    return host.checkVisibility() ? null : ["not rendered", host];
  };

  const isInvisible = (e) => ["hidden", "collapse"].includes(style(e).visibility);
  const byVisibility = (el) => {
    if (!isInvisible(el)) return null;
    const root = climb(el, isInvisible);
    return ["visibility:" + style(root).visibility, root];
  };

  const opacities = new Map();
  const opacity = (el) => inherited(opacities, el, (e, outer) => outer * px(style(e).opacity), 1);
  const isFaint = (e) => opacity(e) < minOpacity;
  const byOpacity = (el) => {
    if (!isFaint(el)) return null;
    const root = climb(el, isFaint);
    return ["opacity:" + fmt(opacity(root)), root];
  };

  const isTiny = (e) => px(style(e).fontSize) < minFontPx;
  const byFontSize = (el) => {
    if (!isTiny(el)) return null;
    const root = climb(el, isTiny);
    const size = px(style(root).fontSize);
    return ["font-size:" + (size === 0 ? "0" : fmt(size) + "px"), root];
  };

  // --- geometry: zero size, off-screen, clipped -------------------------------------------

  const intersect = (a, b) => ({
    left: Math.max(a.left, b.left),
    top: Math.max(a.top, b.top),
    right: Math.min(a.right, b.right),
    bottom: Math.min(a.bottom, b.bottom),
  });
  const tooSmall = (r) => r.right - r.left < minVisiblePx || r.bottom - r.top < minVisiblePx;
  const EMPTY = { left: 0, top: 0, right: 0, bottom: 0 };

  // Contains absolutely positioned descendants / fixed ones (approximately: no will-change).
  const holdsFixed = (s) =>
    s.transform !== "none" ||
    s.filter !== "none" ||
    s.perspective !== "none" ||
    /paint|layout|strict|content/.test(s.contain);
  const holdsAbsolute = (s) => s.position !== "static" || holdsFixed(s);

  const overflowBox = (a, s) => {
    const clipX = ["hidden", "clip"].includes(s.overflowX);
    const clipY = ["hidden", "clip"].includes(s.overflowY);
    if (!(clipX || clipY) || NO_OVERFLOW.has(s.display)) return null;
    // The root's overflow applies to the viewport, which scrolls; so does the body's, unless
    // the root's overflow is set (then the body clips like any other element).
    const root = document.documentElement;
    if (a === root) return null;
    const rs = style(root);
    if (a === document.body && rs.overflowX === "visible" && rs.overflowY === "visible") {
      return null;
    }
    const r = a.getBoundingClientRect();
    return {
      left: clipX ? r.left + px(s.borderLeftWidth) : -Infinity,
      right: clipX ? r.right - px(s.borderRightWidth) : Infinity,
      top: clipY ? r.top + px(s.borderTopWidth) : -Infinity,
      bottom: clipY ? r.bottom - px(s.borderBottomWidth) : Infinity,
    };
  };

  // The legacy `clip` property, on absolutely positioned elements: rect(top, right, bottom,
  // left) from the border box's top-left corner; `auto` is the border box edge.
  const clipBox = (a, s) => {
    if (!["absolute", "fixed"].includes(s.position)) return null;
    const m = /^rect\((.*)\)$/.exec(s.clip);
    if (!m) return null;
    const v = m[1].split(/\s*,\s*|\s+/);
    if (v.length !== 4) return null;
    const r = a.getBoundingClientRect();
    const at = (value, auto, base) => (value === "auto" ? auto : base + px(value));
    return {
      top: at(v[0], r.top, r.top),
      right: at(v[1], r.right, r.left),
      bottom: at(v[2], r.bottom, r.top),
      left: at(v[3], r.left, r.left),
    };
  };

  // clip-path: inset() and zero-radius circle()/ellipse(). Other shapes are not judged.
  const clipPathBox = (a, s) => {
    const path = s.clipPath;
    if (path === "none") return null;
    if (/^(circle\(0(px|%)?[\s)]|ellipse\((0(px|%)?\s|\S+\s+0(px|%)?[\s)]))/.test(path)) {
      return EMPTY;
    }
    const m = /^inset\(([^)]*)\)/.exec(path);
    if (!m) return null;
    const values = m[1].split(/\s+round\s+/)[0].trim().split(/\s+/);
    if (values.length > 4 || values.some((v) => !/^-?[\d.]+(px|%)?$/.test(v))) return null;
    const [t, r = t, b = t, l = r] = values;
    const box = a.getBoundingClientRect();
    const len = (v, size) => (v.endsWith("%") ? (px(v) / 100) * size : px(v));
    return {
      top: box.top + len(t, box.height),
      right: box.right - len(r, box.width),
      bottom: box.bottom - len(b, box.height),
      left: box.left + len(l, box.width),
    };
  };

  // What clips the text of `el`: [element, box, how] triples, innermost first. Overflow clips only
  // along the containing-block chain (an absolute element escapes a static overflow:hidden
  // parent); clip and clip-path clip everything inside.
  const clipsOf = (el) => {
    const clips = [];
    let mode = "normal";
    for (let a = el; a; a = a.parentElement) {
      const s = style(a);
      const inChain =
        mode === "normal" || (mode === "absolute" ? holdsAbsolute(s) : holdsFixed(s));
      const boxes = [[clipBox(a, s), "clip"], [clipPathBox(a, s), "clip-path"]];
      if (inChain) boxes.push([overflowBox(a, s), "overflow"]);
      for (const [box, how] of boxes) if (box) clips.push([a, box, how]);
      if (inChain) mode = ["absolute", "fixed"].includes(s.position) ? s.position : "normal";
    }
    return clips;
  };

  const range = document.createRange();
  const textRects = (nodes) => {
    const rects = [];
    for (const node of nodes) {
      range.selectNodeContents(node);
      for (const r of range.getClientRects()) rects.push(r);
    }
    return rects;
  };
  // Left of or above the page: unreachable by scrolling.
  const offPage = (r) => r.right + scrollX <= 0 || r.bottom + scrollY <= 0;
  const PAGE = { left: -scrollX, top: -scrollY, right: Infinity, bottom: Infinity };

  const byGeometry = (el, nodes) => {
    const rects = textRects(nodes).filter((r) => !tooSmall(r));
    if (!rects.length) {
      // A box with room for text but none drawn: light-DOM text of a shadow host that has no
      // <slot> for it (the shadow root may be closed, so this can't be checked directly).
      if (!tooSmall(el.getBoundingClientRect())) return ["not rendered", el];
      const root = climb(el, (p) => tooSmall(p.getBoundingClientRect()) && p !== document.body);
      return ["zero size", root];
    }
    const clips = clipsOf(el);
    let clipper = null;
    for (const rect of rects) {
      let r = intersect(rect, PAGE);
      if (tooSmall(r)) continue;
      let cut = null;
      for (const [a, box, how] of clips) {
        r = intersect(r, box);
        if (tooSmall(r)) {
          cut = [a, how];
          break;
        }
      }
      if (!cut) return null; // some of the text shows
      clipper = clipper || cut;
    }
    if (!clipper) {
      const root = climb(el, (p) => offPage(p.getBoundingClientRect()) && p !== document.body);
      return ["off-screen", root];
    }
    return [`clipped (${clipper[1]})`, clipper[0]];
  };

  // --- contrast ----------------------------------------------------------------------------

  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 1;
  const paint = canvas.getContext("2d", { willReadFrequently: true });
  const colors = new Map();
  // Any CSS colour as {r, g, b, a} in sRGB (a in [0, 1]), or null if it can't be read.
  const color = (value) => {
    if (colors.has(value)) return colors.get(value);
    let c = null;
    paint.fillStyle = "#010203";
    paint.fillStyle = value;
    if (paint.fillStyle !== "#010203" || value === "#010203") {
      paint.clearRect(0, 0, 1, 1);
      paint.fillRect(0, 0, 1, 1);
      const [r, g, b, a] = paint.getImageData(0, 0, 1, 1).data;
      c = { r, g, b, a: a / 255 };
    }
    colors.set(value, c);
    return c;
  };
  const over = (front, back) => ({
    r: front.r * front.a + back.r * (1 - front.a),
    g: front.g * front.a + back.g * (1 - front.a),
    b: front.b * front.a + back.b * (1 - front.a),
  });
  const luminance = (c) => {
    const lin = (v) => ((v /= 255) <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4);
    return 0.2126 * lin(c.r) + 0.7152 * lin(c.g) + 0.0722 * lin(c.b);
  };

  // The opaque colour behind an element's text, from its own and its ancestors' background
  // colours over a white canvas; null if an image or gradient is involved.
  const backdrops = new Map();
  const backdrop = (el) =>
    inherited(
      backdrops,
      el,
      (e, behind) => {
        const s = style(e);
        if (s.backgroundImage !== "none") return null;
        const c = color(s.backgroundColor);
        if (!c) return null;
        if (c.a >= 1) return c;
        return behind && over(c, behind);
      },
      WHITE,
    );
  const textClips = new Map();
  const clipsText = (el) =>
    inherited(textClips, el, (e, outer) => outer || /text/.test(style(e).backgroundClip), false);

  const contrast = (el) => {
    const s = style(el);
    if (s.textShadow !== "none" || px(s.getPropertyValue("-webkit-text-stroke-width")) > 0) {
      return null; // the outline or shadow can make the text readable
    }
    if (clipsText(el)) return null; // the text shows the background (gradient text)
    const back = backdrop(el);
    const fill = color(s.getPropertyValue("-webkit-text-fill-color"));
    if (!back || !fill) return null;
    const [hi, lo] = [luminance(over(fill, back)), luminance(back)].sort((x, y) => y - x);
    return (hi + 0.05) / (lo + 0.05);
  };
  const isLowContrast = (e) => {
    const ratio = contrast(e);
    return ratio !== null && ratio < minContrast;
  };
  const byContrast = (el) => {
    if (!isLowContrast(el)) return null;
    const root = climb(el, isLowContrast);
    return [`low contrast (${contrast(root).toFixed(2)}:1)`, root];
  };

  // --- walk, mark, serialise ---------------------------------------------------------------

  const owners = new Map(); // element -> its non-blank text nodes
  let unchecked = 0;
  if (document.body) {
    const walker = document.createTreeWalker(
      document.body,
      NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT,
      (node) =>
        node.nodeType === Node.ELEMENT_NODE && SKIPPED.has(node.localName)
          ? NodeFilter.FILTER_REJECT
          : NodeFilter.FILTER_ACCEPT,
    );
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      if (node.nodeType !== Node.TEXT_NODE || WHITESPACE.test(node.data)) continue;
      const owner = node.parentElement;
      let nodes = owners.get(owner);
      if (nodes === undefined) {
        if (owners.size >= maxElements) {
          unchecked++;
          continue;
        }
        owners.set(owner, (nodes = []));
      }
      nodes.push(node);
    }
  }

  const checks = [byDisplay, byVisibility, byOpacity, byFontSize, byGeometry, byContrast];
  const roots = new Map(); // root element -> Map(reason -> id)
  const marks = [];
  let nextId = 0;
  for (const [el, nodes] of owners) {
    let found = null;
    for (const check of checks) if ((found = check(el, nodes))) break;
    if (!found) continue;
    const [reason, root] = found;
    let ids = roots.get(root);
    if (ids === undefined) roots.set(root, (ids = new Map()));
    let id = ids.get(reason);
    if (id === undefined) ids.set(reason, (id = nextId++));
    marks.push([el, `${id} ${reason}`]);
  }
  // Only now, after every measurement: the new attributes can't change what was measured.
  for (const [el, value] of marks) el.setAttribute(marker, value);
  for (const [root, ids] of roots) {
    const value = [...ids].map(([reason, id]) => `${id} ${reason}`).join(";");
    root.setAttribute(marker + "-root", value);
  }

  const doctype = document.doctype
    ? new XMLSerializer().serializeToString(document.doctype)
    : "";
  const page = doctype + (document.documentElement ? document.documentElement.outerHTML : "");
  return { html: page.length > limit ? page.slice(0, limit + 1) : page, unchecked };
}
