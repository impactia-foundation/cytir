"""cytir.ops.cytir_site -- render the four static pages of cytir.org (EN, FR, DE, IT) from one template.

    python -m cytir.ops.cytir_site --write    # render all four pages into webapp/cytir/
    python -m cytir.ops.cytir_site --check    # exit 1 when a committed page differs from its render

The single source is `webapp/cytir/_template.html` (markup, CSS, logos) plus one flat strings file per
language, `webapp/cytir/i18n/<lang>.json` (key -> string; HTML is allowed in a value where the English
has inline markup). The template names a string as `{{key}}`. Five more names are computed here:
`lang`, `url`, `og_locale`, `hreflang_links` and `lang_switch`.

The committed pages are the render, byte for byte: `/index.html` (EN), `/fr/index.html`,
`/de/index.html`, `/it/index.html`. Every page links the shared tree `/assets/...`, so Caddy serves
all four from one root. A key whose name ends in `_attr` lands inside an HTML attribute or a CSS
string and may not contain `"`, `<` or `\\`. Every language file carries exactly the keys of en.json.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

APP = pathlib.Path(__file__).resolve().parent.parent.parent
SITE = APP / "webapp" / "cytir"
TEMPLATE = SITE / "_template.html"
I18N = SITE / "i18n"
BASE = "https://cytir.org"
LANGS = ("en", "fr", "de", "it")           # the first is the default (x-default)
OG_LOCALE = {"en": "en_US", "fr": "fr_CH", "de": "de_CH", "it": "it_CH"}
NATIVE = {"en": "EN", "fr": "FR", "de": "DE", "it": "IT"}
#: names the renderer fills itself; everything else in `{{...}}` must be a key of the strings files
COMPUTED = ("lang", "url", "og_locale", "hreflang_links", "lang_switch")
#: keys the renderer reads itself (not named in the template)
RENDERER_KEYS = ("lang_label_attr",)
SLOT = re.compile(r"\{\{([a-z0-9_]+)\}\}")
ATTR_BAD = re.compile(r'["<\\]')


def page_path(lang: str) -> pathlib.Path:
    return SITE / "index.html" if lang == LANGS[0] else SITE / lang / "index.html"


def page_url(lang: str) -> str:
    return BASE + ("/" if lang == LANGS[0] else f"/{lang}/")


def load_strings(lang: str) -> dict[str, str]:
    return json.loads((I18N / f"{lang}.json").read_text(encoding="utf-8"))


def problems(template: str, strings: dict[str, dict[str, str]]) -> list[str]:
    """Everything that would make the render wrong: missing keys, empty values, unused keys, bad attrs."""
    out: list[str] = []
    ref = strings[LANGS[0]]
    used = set(SLOT.findall(template))
    for lang in LANGS:
        s = strings[lang]
        for k in sorted(set(ref) - set(s)):
            out.append(f"{lang}: missing key {k}")
        for k in sorted(set(s) - set(ref)):
            out.append(f"{lang}: key {k} is not in {LANGS[0]}.json")
        for k, v in s.items():
            if not isinstance(v, str) or not v.strip():
                out.append(f"{lang}: empty value for {k}")
            elif k.endswith("_attr") or k.endswith("_css"):
                if ATTR_BAD.search(v):
                    out.append(f"{lang}: {k} sits in an attribute and may not hold a quote, '<' or backslash")
    for k in sorted(used - set(ref) - set(COMPUTED)):
        out.append(f"template uses {{{{{k}}}}} and no strings file defines it")
    for k in sorted(set(ref) - used - set(RENDERER_KEYS)):
        out.append(f"key {k} is in the strings files and not in the template")
    return out


def hreflang_links() -> str:
    rows = [f'<link rel="alternate" hreflang="{lg}" href="{page_url(lg)}">' for lg in LANGS]
    rows.append(f'<link rel="alternate" hreflang="x-default" href="{page_url(LANGS[0])}">')
    return "\n".join(rows)


def lang_switch(lang: str, label: str) -> str:
    links = []
    for lg in LANGS:
        cur = ' aria-current="page"' if lg == lang else ""
        href = "/" if lg == LANGS[0] else f"/{lg}/"
        links.append(f'<a href="{href}" lang="{lg}" hreflang="{lg}"{cur}>{NATIVE[lg]}</a>')
    sep = '<span class="sep" aria-hidden="true">·</span>'
    return f'<nav class="lang-switch" aria-label="{label}">' + sep.join(links) + "</nav>"


def render(lang: str, template: str | None = None, strings: dict[str, dict[str, str]] | None = None) -> str:
    template = TEMPLATE.read_text(encoding="utf-8") if template is None else template
    strings = {lg: load_strings(lg) for lg in LANGS} if strings is None else strings
    bad = problems(template, strings)
    if bad:
        raise ValueError("cytir_site: " + "; ".join(bad))
    values = dict(strings[lang])
    values.update(lang=lang, url=page_url(lang), og_locale=OG_LOCALE[lang], hreflang_links=hreflang_links(),
                  lang_switch=lang_switch(lang, values["lang_label_attr"]))
    return SLOT.sub(lambda m: values[m.group(1)], template)


def render_all() -> dict[pathlib.Path, str]:
    template = TEMPLATE.read_text(encoding="utf-8")
    strings = {lg: load_strings(lg) for lg in LANGS}
    return {page_path(lg): render(lg, template, strings) for lg in LANGS}


def stale() -> list[pathlib.Path]:
    return [p for p, text in render_all().items() if not p.is_file() or p.read_text(encoding="utf-8") != text]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cytir.ops.cytir_site", description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true", help="render the four pages")
    g.add_argument("--check", action="store_true", help="exit 1 when a committed page is stale")
    a = ap.parse_args(argv)
    try:
        pages = render_all()
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    if a.write:
        for p, text in pages.items():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
            print(f"wrote {p.relative_to(APP)}")
        return 0
    old = [p for p, text in pages.items() if not p.is_file() or p.read_text(encoding="utf-8") != text]
    for p in old:
        print(f"cytir_site: {p.relative_to(APP)} is stale; run: python -m cytir.ops.cytir_site --write", file=sys.stderr)
    return 1 if old else 0


if __name__ == "__main__":
    sys.exit(main())
