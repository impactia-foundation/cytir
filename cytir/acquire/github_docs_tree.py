"""The GitHub docs-tree selector and licence set (task 986), shared by the acquirer and `cyt ops github-admission`.

One definition of "the repo's docs tree" (vendor-content boundary 0.60): the acquirer selects files with it and the
admission roster counts them with it.

Task 1237 folds two legacy drivers in as the render half of the same module (nothing here fetches):
  * `docs_repo_rows` -- an OFFICIAL framework docs repo (cloned, permissive) -> passages under the CANONICAL
    live doc-site URLs (`REPOS`: PyTorch, scikit-learn, TensorFlow, JAX, Keras, LangChain, Ollama, HF
    transformers; the roster rows of task 119). Text keeps API identifiers (autodoc symbol names), code fences
    are capped, not dropped. Was `ingest_docs_repo.py`.
  * `techdocs_rows` -- react.dev and the Rust Book (task 118, `TECHDOC_SOURCES`). Was `ingest_techdocs.py`.
  * `verify_urls` -- the live HTTP-200 gate over passage URLs (the old `ingest_techdocs.py verify`).
The rows are unchanged; `tests/test_aw_github_docs_tree_render.py` pins them against the legacy drivers' output.
"""
from __future__ import annotations

import glob
import json
import os
import re
import time
from typing import Any

from cytir.chunk.generic import chunk_generic_records
from cytir.core.common import (extractive_snippet, low_quality, passage_corpus_record,
                            passage_from_chunk)

ROOT_DOC = re.compile(r"^(README|GETTING[-_ ]?STARTED|OVERVIEW|INTRO)", re.I)

# Full docs-tree mode (task 986, `--tree docs`): the vendor-content boundary 0.60 = only the repo's
# docs tree (docs/, doc/, documentation/ at any depth), README and root *.md. Case-insensitive.
TREE_EXTENSIONS = (".md", ".mdx", ".markdown", ".rst", ".txt")
TREE_DIR = re.compile(r"(^|/)(docs?|documentation)/", re.I)
TREE_SKIP_DIRS = re.compile(
    r"(^|/)(i18n|l10n|locales?|translations?|translated|node_modules|vendor|third[-_]party|_build|"
    r"build|dist|generated|__snapshots__|versioned_docs|"
    r"zh|zh[-_](cn|tw|hans|hant)|ja|jp|ko|kr|ru|es|pt|pt[-_]br|de|fr|it|tr|vi|id|ar|pl|nl|uk|th|hi)/",
    re.I,
)
TREE_SKIP_FILES = re.compile(
    r"(^|/)(CONTRIBUTING|CODE_OF_CONDUCT|LICENSE|LICENCE|SECURITY|NOTICE|AUTHORS|CODEOWNERS)[^/]*$"
    r"|(^|/)(test|tests|examples?|fixtures)/|node_modules", re.I)
CHANGELOG_FILE = re.compile(r"(^|/)(CHANGELOG|CHANGES|HISTORY|RELEASES?|NEWS)[^/]*$", re.I)
TREE_DEFAULT_MAX_FILES = 400
TREE_DEFAULT_MAX_FILE_BYTES = 2_000_000
PERMISSIVE = {
    "MIT", "APACHE-2.0", "BSD-3-CLAUSE", "BSD-2-CLAUSE", "ISC", "MPL-2.0",
    "CC-BY-4.0", "CC-BY-SA-4.0", "CC0-1.0", "UNLICENSE", "0BSD", "ZLIB",
    "BSD-3-CLAUSE-CLEAR", "CC-BY-3.0",
}


def select_docs_tree(nodes: list[dict[str, Any]], permissive: bool, max_files: int,
                     max_file_bytes: int = TREE_DEFAULT_MAX_FILE_BYTES,
                     include_changelog: bool = False,
                     exclude: str | None = None) -> list[dict[str, Any]]:
    """Task 986 `--tree docs`: every doc file in the repo's docs tree + README + root `*.md`/`.rst`/`.txt`,
    under a hard cap. Order is deterministic (root files, then shallow paths, then path) so a cap cuts the
    deepest files first. A link-only (non-permissive) repo still yields its single README, as today."""
    skip = re.compile(exclude) if exclude else None
    picked = []
    for node in nodes:
        path = str(node.get("path", ""))
        if node.get("type") != "blob" or not isinstance(node.get("sha"), str):
            continue
        if not path.lower().endswith(TREE_EXTENSIONS):
            continue
        size = node.get("size")
        if isinstance(size, int) and size > max_file_bytes:
            continue
        if TREE_SKIP_DIRS.search(path) or TREE_SKIP_FILES.search(path):
            continue
        if not include_changelog and CHANGELOG_FILE.search(path):
            continue
        if skip and skip.search(path):
            continue
        if "/" not in path or TREE_DIR.search(path):
            picked.append(node)
    picked.sort(key=lambda n: ("/" in str(n["path"]), str(n["path"]).count("/"), str(n["path"]).lower()))
    if not permissive:
        readme = [n for n in picked if "/" not in str(n["path"])
                  and ROOT_DOC.search(str(n["path"]))]
        return (readme or picked)[:1]
    return picked[:max_files]


# ----------------------------------------------------------------------------
# docs repo -> passages (was ingest_docs_repo.py)
# ----------------------------------------------------------------------------
_WS = re.compile(r"[ \t]+")
_MULTINL = re.compile(r"\n{3,}")


# ----------------------------------------------------------------------------
# text extraction
# ----------------------------------------------------------------------------
def strip_frontmatter(text: str) -> str:
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            nl = text.find("\n", end + 1)
            return text[nl + 1:] if nl != -1 else ""
    return text


_ROLE = re.compile(r":[a-zA-Z:]+:`~?([^`<]+?)(?:\s*<[^>]+>)?`")   # :func:`~torch.x` -> torch.x
_RST_OPT = re.compile(r"^\s*:[\w][\w .-]*:\s*.*$")                 # directive option line
_RST_UNDERLINE = re.compile(r'^[=\-~^"\'#*+`.]{3,}\s*$')
# formatting directives whose *decoration* we drop (content, if any, is kept by
# de-indenting); autodoc directives keep their symbol argument.
_RST_DROP_DIR = re.compile(
    r"^\s*\.\.\s+(toctree|contents|currentmodule|module|highlight|include|"
    r"image|figure|raw|only|tabularcolumns|autosummary|list-table|csv-table|"
    r"index|meta|sectionauthor|deprecated|versionadded|versionchanged|rubric|"
    r"seealso|graphviz|plot|bibliography|glossary|literalinclude)::.*$", re.I)
_RST_AUTODOC = re.compile(
    r"^\s*\.\.\s+auto(module|class|function|method|data|attribute|exception|"
    r"decorator|property)::\s*(.+?)\s*$", re.I)
_RST_ANYDIR = re.compile(r"^\s*\.\.\s+[\w:+-]+::\s*(.*)$")        # any directive incl. py:module:: / domain roles


_RST_TARGET = re.compile(r"^\s*\.\.\s+_[^:]*:\s*.*$")            # .. _label: hyperlink target
_RST_SUBST = re.compile(r"^\s*\.\.\s+\|[^|]+\|\s.*$")            # .. |sub| replace:: …


def render_rst(text: str) -> str:
    out = []
    for ln in text.splitlines():
        if _RST_UNDERLINE.match(ln):
            continue
        if _RST_TARGET.match(ln) or _RST_SUBST.match(ln):
            continue
        m = _RST_AUTODOC.match(ln)
        if m:
            out.append(m.group(2))
            continue
        if _RST_DROP_DIR.match(ln):
            continue
        if _RST_OPT.match(ln):
            continue
        m = _RST_ANYDIR.match(ln)          # note::/warning::/code-block:: … keep arg text
        if m:
            arg = m.group(1).strip()
            if arg:
                out.append(arg)
            continue
        ln = ln.replace("~", "") if ln.strip().startswith("~") else ln
        ln = _ROLE.sub(r"\1", ln)
        ln = re.sub(r"`([^`<]+?)\s*<[^>]+>`__?", r"\1", ln)   # `text <url>`_ inline link
        ln = re.sub(r"``([^`]+)``", r"\1", ln)
        ln = re.sub(r"\*\*([^*]+)\*\*", r"\1", ln)
        out.append(ln)
    return "\n".join(out)


_MD_COMMENT = re.compile(r"<!--.*?-->", re.S)
_MD_IMPORT = re.compile(r"^\s*(import|export)\s.+$", re.M)          # MDX
_JSX_SELFCLOSE = re.compile(r"<[A-Za-z][\w.]*(?:\s[^<>]*?)?/>")
_JSX_TAG = re.compile(r"</?[A-Za-z][\w.]*(?:\s[^<>]*?)?>")          # <Tab ...> </Tab>
_MD_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\((?:[^)]*)\)")
_MD_HEADING = re.compile(r"^#{1,6}\s+", re.M)
_MYST_ROLE = re.compile(r"\{[\w:+-]+\}`~?([^`<]+?)(?:\s*<[^>]+>)?`")  # {py:func}`x`


def _cap_code(body: str, max_words: int = 40) -> str:
    w = body.split()
    return " ".join(w[:max_words])


def render_markdown(text: str) -> str:
    text = strip_frontmatter(text)
    text = _MD_COMMENT.sub(" ", text)
    text = _MD_IMPORT.sub("", text)
    # walk fenced code blocks: ```{eval-rst}=RST, ```lang=capped code, ```=capped
    lines = text.splitlines()
    out, i = [], 0
    fence_re = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
    while i < len(lines):
        m = fence_re.match(lines[i])
        if m:
            fence, info = m.group(1), m.group(2).strip()
            body = []
            i += 1
            # CommonMark: the closer uses the opener's character and is at least as long
            close_re = re.compile(r"^\s*" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}\s*$")
            while i < len(lines) and not close_re.match(lines[i]):
                body.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            inner = "\n".join(body)
            if info.startswith("{") and "eval-rst" in info:
                out.append(render_rst(inner))
            elif info.startswith("{"):
                # MyST directive: {note}/{admonition}/{code-block}/… keep prose,
                # drop toctree/contents
                if any(d in info for d in ("toctree", "contents")):
                    pass
                else:
                    out.append(render_rst(inner))
            else:
                out.append(_cap_code(inner))
            continue
        out.append(lines[i])
        i += 1
    text = "\n".join(out)
    text = _MYST_ROLE.sub(r"\1", text)
    text = re.sub(r"\([\w:.-]+\)=", " ", text)             # MyST (anchor-label)= target
    text = _MD_IMG.sub(" ", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _JSX_SELFCLOSE.sub(" ", text)
    text = _JSX_TAG.sub(" ", text)
    text = _MD_HEADING.sub("", text)
    text = re.sub(r"[`*>|]", " ", text)
    text = re.sub(r"^\s*[-+]\s+", "", text, flags=re.M)
    return text


def render_notebook(raw: str) -> str:
    try:
        nb = json.loads(raw)
    except Exception:  # noqa: BLE001
        return ""
    parts = []
    for cell in nb.get("cells", []):
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        src = src.strip()
        if not src:
            continue
        if cell.get("cell_type") == "markdown":
            parts.append(render_markdown(src))
        elif cell.get("cell_type") == "code":
            parts.append(_cap_code(src))
    return "\n\n".join(parts)


def to_text(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    ext = os.path.splitext(path)[1].lower()
    if ext == ".ipynb":
        txt = render_notebook(raw)
    elif ext == ".rst":
        txt = render_rst(raw)
    else:  # .md .mdx
        txt = render_markdown(raw)
    txt = _WS.sub(" ", txt)
    txt = _MULTINL.sub("\n\n", txt)
    # collapse lone-word lines into paragraphs a bit for chunk()'s paragraph model
    return txt.strip()


def first_heading(raw_path: str, fallback: str) -> str:
    ext = os.path.splitext(raw_path)[1].lower()
    if ext == ".ipynb":
        try:
            with open(raw_path, encoding="utf-8", errors="replace") as f:
                nb = json.load(f)
            for cell in nb.get("cells", []):
                if cell.get("cell_type") != "markdown":
                    continue
                s = cell.get("source", "")
                s = "".join(s) if isinstance(s, list) else s
                for ln in s.splitlines():
                    m = re.match(r"^#{1,3}\s+(.+)", ln)
                    if m:
                        return re.sub(r"[`*]", "", m.group(1)).strip()
        except Exception:  # noqa: BLE001
            pass
        return fallback
    try:
        with open(raw_path, encoding="utf-8", errors="replace") as f:
            head = f.read(4000)
    except Exception:  # noqa: BLE001
        return fallback
    head = strip_frontmatter(head)
    for ln in head.splitlines():
        m = re.match(r"^#{1,3}\s+(.+)", ln)   # md/mdx
        if m:
            return re.sub(r"[`*]", "", m.group(1)).strip()
    # rst: first non-empty line followed by an underline
    lines = [l for l in head.splitlines()]
    for j in range(len(lines) - 1):
        t = lines[j].strip()
        if t and _RST_UNDERLINE.match(lines[j + 1]) and not t.startswith(".."):
            return _ROLE.sub(r"\1", t)
    return fallback


# ----------------------------------------------------------------------------
# per-repo configuration:  root subdir, exts, skip regex, url(relpath)->url|None
# ----------------------------------------------------------------------------
def _drop_ext(rel: str) -> str:
    return re.sub(r"\.(mdx?|rst|ipynb)$", "", rel)


def _mint_url(host: str, srcroot: str):
    def f(rel: str):
        p = _drop_ext(rel)
        p = re.sub(r"/index$", "", p)
        if p == "index":
            p = ""
        return f"https://{host}/{p}".rstrip("/")
    return f


SKIP_COMMON = re.compile(
    r"(^|/)(_static|_templates|_build|_sources|images|_images|assets|"
    r"CHANGELOG|changelog|CONTRIBUTING|contributing|whats_new|whatsnew|"
    r"release[_-]?notes|releases|redirects|snippets)(/|$|\.)", re.I)

REPOS = {
    "pytorch": dict(
        subdir="pytorch/docs/source", exts=(".md", ".rst"), domain="pytorch.org",
        skip=re.compile(r"(^|/)(community|scripts|conf\.py|index\.rst)", re.I),
        url=lambda rel: f"https://pytorch.org/docs/stable/{_drop_ext(rel)}.html",
    ),
    "scikit-learn": dict(
        subdir="scikit-learn/doc", exts=(".rst",), domain="scikit-learn.org",
        skip=re.compile(r"(^|/)(developers|templates|whats_new|whatsnew|"
                        r"testimonials|binder|conftest|includes|min_dependency|"
                        r"index\.rst$)", re.I),
        url=lambda rel: f"https://scikit-learn.org/stable/{_drop_ext(rel)}.html",
    ),
    "tensorflow": dict(
        subdir="tensorflow/site/en", exts=(".md", ".ipynb"), domain="tensorflow.org",
        skip=re.compile(r"(^|/)(community|contribute|r1|_book|_index|README|"
                        r"index\.(md|ipynb)$)", re.I),
        url=lambda rel: f"https://www.tensorflow.org/{_drop_ext(rel)}",
    ),
    "jax": dict(
        subdir="jax/docs", exts=(".md", ".rst", ".ipynb"), domain="docs.jax.dev",
        skip=re.compile(r"(^|/)(_static|_templates|_autosummary|conf\.py|"
                        r"index\.rst$|glossary)", re.I),
        url=lambda rel: f"https://docs.jax.dev/en/latest/{_drop_ext(rel)}.html",
    ),
    "keras": dict(
        subdir="keras-io/templates", exts=(".md",), domain="keras.io",
        skip=re.compile(r"(^|/)(contributing)", re.I),
        url=_mint_url("keras.io", "keras-io/templates"),
    ),
    "langchain": dict(
        subdir="langchain/src", exts=(".mdx",), domain="docs.langchain.com",
        skip=re.compile(r"(^|/)(snippets|images|fonts|code-samples)(/|$)", re.I),
        url=_mint_url("docs.langchain.com", "langchain/src"),
        sitemap="langchain/sitemap.xml",
    ),
    "ollama": dict(
        subdir="ollama/docs", exts=(".mdx", ".md"), domain="docs.ollama.com",
        skip=re.compile(r"(^|/)(images|development\.md$)", re.I),
        url=_mint_url("docs.ollama.com", "ollama/docs"),
        sitemap="ollama/sitemap.txt",   # live sitemap saved next to repo
    ),
    "transformers": dict(
        subdir="transformers/docs/source/en", exts=(".md",), domain="huggingface.co",
        skip=re.compile(r"(^|/)(_toctree|_redirects|index\.md$)", re.I),
        url=lambda rel: f"https://huggingface.co/docs/transformers/{_drop_ext(rel)}",
    ),
}


def load_sitemap(path: str) -> set[str] | None:
    if not path or not os.path.exists(path):
        return None
    urls = set()
    with open(path, encoding="utf-8", errors="replace") as f:
        data = f.read()
    for m in re.finditer(r"https?://[^\s<\"]+", data):
        urls.add(m.group(0).rstrip("/"))
    return urls or None



def docs_repo_rows(repo: str, root: str, *, trust: float = 0.85) -> tuple[list[dict], dict]:
    """The passages of ONE configured docs repo under `root` (the dir holding the clones) and the legacy
    SUMMARY counters. Row order is the sorted file order, as the driver wrote it."""
    cfg = REPOS[repo]
    base = os.path.join(root, cfg["subdir"])
    sitemap = load_sitemap(os.path.join(root, cfg["sitemap"])) if cfg.get("sitemap") else None
    files = []
    for ext in cfg["exts"]:
        files += glob.glob(os.path.join(base, "**", f"*{ext}"), recursive=True)
    files = sorted(set(files))
    stats = {"files_seen": len(files), "skipped": 0, "off_sitemap": 0, "empty": 0,
             "files_with_passages": 0, "passages": 0}
    rows: list[dict] = []
    for path in files:
        rel = os.path.relpath(path, base)
        if cfg["skip"].search(rel) or SKIP_COMMON.search(rel):
            stats["skipped"] += 1
            continue
        url = cfg["url"](rel)
        if not url:
            stats["skipped"] += 1
            continue
        if sitemap is not None and url.rstrip("/") not in sitemap:
            stats["off_sitemap"] += 1
            continue
        if not os.path.isfile(path):     # broken symlink outside sparse tree
            stats["skipped"] += 1
            continue
        try:
            text = to_text(path)
        except (OSError, ValueError):
            stats["skipped"] += 1
            continue
        if len(text.split()) < 20:
            stats["empty"] += 1
            continue
        title = first_heading(path, os.path.splitext(os.path.basename(rel))[0])
        made = 0
        for unit in chunk_generic_records(f"{title}\n{text}"):
            ch = unit.text
            if low_quality(ch):
                continue
            rows.append(passage_corpus_record(passage_from_chunk(
                unit, url=url, title=title[:300], snippet=" ".join(ch.split()[:25]),
                lang="en", domain=cfg["domain"], tier="B", source_type="docs", trust_tier=trust)))
            made += 1
        if made:
            stats["files_with_passages"] += 1
            stats["passages"] += made
        else:
            stats["empty"] += 1
    return rows, stats



# ----------------------------------------------------------------------------
# react.dev and the Rust Book (was ingest_techdocs.py)
# ----------------------------------------------------------------------------
TECHDOC_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)

TECHDOC_SOURCES = {
    "react": {"domain": "react.dev", "last_seen": None},
    "rust": {"domain": "doc.rust-lang.org", "last_seen": None},
}


# --------------------------------------------------------------------------- #
# cleaning
# --------------------------------------------------------------------------- #
def _techdoc_clean_common(body: str) -> str:
    body = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", body)          # md images
    body = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", body)        # md links -> text
    body = re.sub(r"(?m)^```.*$", "", body)                     # code-fence markers
    body = re.sub(r"<[^>]+>", " ", body)                        # html / JSX tags
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def _techdoc_clean_react(body: str) -> str:
    body = re.sub(r"(?m)^(import|export)\s+.*$", "", body)      # ESM lines
    body = re.sub(r"\{/\*.*?\*/\}", "", body, flags=re.DOTALL)  # MDX heading anchors
    return _techdoc_clean_common(body)


def _techdoc_clean_rust(body: str) -> str:
    body = re.sub(r"\{\{#.*?\}\}", "", body, flags=re.DOTALL)   # mdBook directives
    return _techdoc_clean_common(body)


def _techdoc_first_heading(body: str) -> str | None:
    # first ATX heading of ANY level — the Rust Book titles pages with `## `,
    # react MDX (when it lacks frontmatter) with `# `.
    m = re.search(r"(?m)^#{1,6}\s+(.+?)\s*$", body)
    return m.group(1).strip() if m else None


# --------------------------------------------------------------------------- #
# per-source file -> (url, title, raw_body)
# --------------------------------------------------------------------------- #
def iter_react_pages(repo: str):
    base = os.path.join(repo, "src", "content")
    for dp, _, files in os.walk(base):
        for fn in sorted(files):
            if not fn.endswith(".md"):
                continue
            path = os.path.join(dp, fn)
            rel = os.path.relpath(path, base)[:-3]             # drop .md
            rel = rel.replace(os.sep, "/")
            if rel.endswith("/index"):
                rel = rel[:-len("/index")]
            if rel == "index":
                rel = ""
            url = "https://react.dev/" + rel if rel else "https://react.dev"
            yield path, url


def iter_rust_pages(repo: str):
    base = os.path.join(repo, "src")
    for fn in sorted(os.listdir(base)):
        if not fn.endswith(".md") or fn == "SUMMARY.md":
            continue
        stem = fn[:-3]
        url = f"https://doc.rust-lang.org/book/{stem}.html"
        yield os.path.join(base, fn), url


def techdoc_parse_file(path: str, source: str):
    raw = open(path, encoding="utf-8").read()
    m = TECHDOC_FRONTMATTER.match(raw)
    title = None
    if m:
        for line in m.group(1).splitlines():
            mt = re.match(r"\s*title:\s*(.+?)\s*$", line)
            if mt:
                title = mt.group(1).strip().strip('"').strip("'")
                break
        body = m.group(2)
    else:
        body = raw
    if not title:
        title = _techdoc_first_heading(body)
    clean = _techdoc_clean_react(body) if source == "react" else _techdoc_clean_rust(body)
    if not title:
        title = os.path.basename(path)[:-3]
    return title, clean


# --------------------------------------------------------------------------- #

def techdocs_rows(source: str, repo: str, *, last_seen: str | None = None) -> tuple[list[dict], dict]:
    """The passages of react.dev (`source="react"`) or the Rust Book (`"rust"`) from a clone at `repo`:
    one row per unique point id, tagged `provenance=tech_gapfill_react_rust`. Counters as the driver printed."""
    domain = TECHDOC_SOURCES[source]["domain"]
    pages = iter_react_pages(repo) if source == "react" else iter_rust_pages(repo)
    stats = {"docs": 0, "empty": 0, "passages": 0}
    rows: list[dict] = []
    seen_ids: set[str] = set()
    for path, url in pages:
        title, clean = techdoc_parse_file(path, source)
        chunks = chunk_generic_records(clean)
        if not chunks:
            stats["empty"] += 1
            continue
        stats["docs"] += 1
        for unit in chunks:
            p = passage_from_chunk(
                unit, url=url, title=title, snippet=extractive_snippet(unit.text, 25),
                lang="en", domain=domain, tier="B", source_type="docs", trust_tier=0.85,
                last_seen=last_seen)
            if p.point_id in seen_ids:
                continue
            seen_ids.add(p.point_id)
            row = passage_corpus_record(p)
            row["provenance"] = "tech_gapfill_react_rust"
            rows.append(row)
            stats["passages"] += 1
    return rows, stats


def verify_urls(urls, client, *, sleep=time.sleep, clock=time.time) -> tuple[dict, dict]:
    """The live HTTP-200 gate: GET every distinct URL (>= 0.2 s apart per host); returns `(ok, bad)` as
    `{url: status}`. A transport error is `bad` with `ERR:<type>`. `client` is an `httpx.Client`."""
    last_host_t: dict[str, float] = {}
    ok: dict = {}
    bad: dict = {}
    for u in dict.fromkeys(urls):
        host = u.split("/")[2]
        dt = clock() - last_host_t.get(host, 0)
        if dt < 0.2:
            sleep(0.2 - dt)
        try:
            code = client.get(u).status_code
        except Exception as e:  # noqa: BLE001
            code = f"ERR:{type(e).__name__}"
        last_host_t[host] = clock()
        (ok if code == 200 else bad)[u] = code
    return ok, bad
