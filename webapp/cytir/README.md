# webapp/cytir — the cytir.org site

Four static pages, one per language, rendered from one template and four strings files. No backend.
It is NOT served by the AW api; it is served on its own domain, cytir.org, by Caddy as plain files.

    index.html            EN   https://cytir.org/
    fr/index.html         FR   https://cytir.org/fr/
    de/index.html         DE   https://cytir.org/de/
    it/index.html         IT   https://cytir.org/it/
    assets/               one shared tree, linked as /assets/... from every page
    _template.html        markup, CSS and logos; strings appear as {{key}}
    i18n/{en,fr,de,it}.json   flat key -> string (HTML allowed where the English has inline markup)

The four `index.html` files are generated. Edit the template or a strings file, then:

    python3 -m cytir.ops.cytir_site --write    # render all four pages
    python3 -m cytir.ops.cytir_site --check    # exit 1 when a committed page is stale

`tests/test_cytir_site.py` fails when a committed page differs from its render, when a language lacks a key,
or when a page lacks its own `lang` or one of the four `hreflang` links. A key ending in `_attr` or `_css`
sits in an attribute or a CSS string: no `"`, `<` or backslash. Swiss orthography (ss, never ß).
The "Get the code" link points at `https://github.com/impactia-foundation/cytir` (set 2026-11-19, Tim). The repository is created at the release on 2026-11-20, so the link 404s until then.

Hosting is an additive block in `deploy/Caddy.snippet` (append only, other blocks untouched):

    cytir.org, www.cytir.org { root * /srv/cytir; file_server; encode gzip }

`file_server` serves `/fr/`, `/de/` and `/it/` from their `index.html` and redirects `/fr` to `/fr/`; no Caddy
change is needed for the language pages. The robot-me Caddy needs a read-only bind mount of `/opt/aw/webapp/cytir`
at `/srv/cytir`. DNS for cytir.org is Tim's step. The live Caddy is not edited by the repo.
