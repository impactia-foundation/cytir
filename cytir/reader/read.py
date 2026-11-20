"""`read(url, read_url=None, max_chars=...)`: the fallback order of the local reader (task 2710).

1. plain fetch of `url` -> real text: done.
2. shell and `read_url` given -> fetch `read_url` -> real text: done.
3. still shell -> render `url` with the selected renderer -> real text: done.
4. no renderer -> return the shell text and name the install hint in `notes`.
"""
from __future__ import annotations

import time

from . import config, fetch as F
from .extract import extract
from .render import Renderer, RenderError, find_renderer
from .shell import is_shell


def _carry_fragment(read_url: str, url: str) -> str:
    """The twin keeps the anchor: reuse the fragment of `url` when `read_url` has none."""
    frag = url.partition("#")[2]
    return read_url if "#" in read_url or not frag else f"{read_url}#{frag}"


def read(url: str, read_url: str | None = None, max_chars: int = config.DEFAULT_MAX_CHARS,
         renderer: Renderer | None | str = "auto", fetcher=F.fetch) -> dict:
    if max_chars < 0:
        raise ValueError(f'max_chars must be >= 0, got {max_chars}')
    max_chars = max(1, min(max_chars, config.MAX_BYTES))
    t0 = time.monotonic()
    notes: list[str] = []
    F.check_url(url)
    if read_url:
        F.check_url(read_url)
    if renderer == "auto":
        renderer = find_renderer()

    def done(final_url, source, text, rname=None, anchored=False):
        if anchored:
            notes.append("trimmed to the #fragment element")
        truncated = len(text) > max_chars
        return {"url": url, "final_url": final_url, "source_used": source, "renderer": rname,
                "text": text[:max_chars], "truncated": truncated,
                "elapsed_ms": int((time.monotonic() - t0) * 1000), "notes": notes}

    page = fetcher(url)
    text, anchored = extract(page.html, url)
    shell_text, shell_url = text, page.final_url
    if not is_shell(page.html, text):
        return done(page.final_url, "url", text, anchored=anchored)
    notes.append("plain fetch returned a JavaScript shell")

    if read_url:
        target = _carry_fragment(read_url, url)
        try:
            alt = fetcher(target)
            t2, a2 = extract(alt.html, target)
            if not is_shell(alt.html, t2):
                return done(alt.final_url, "read_url", t2, anchored=a2)
            notes.append("read_url also returned a shell")
        except (F.FetchError, F.FetchRefused) as e:
            notes.append(f"read_url failed: {e}")

    if renderer is None:
        notes.append(config.INSTALL_HINT)
        return done(shell_url, "url", shell_text)
    try:
        html = renderer.render(url.partition("#")[0])   # obscura looped to 6.5 GB on a #fragment URL; trim happens in extract
    except RenderError as e:
        notes.append(f"render failed: {e}")
        return done(shell_url, "url", shell_text)
    t3, a3 = extract(html, url)
    if is_shell(html, t3):
        notes.append(f"{renderer.name} output still looks like a shell")
    return done(page.final_url, "rendered", t3, rname=renderer.name, anchored=a3)
