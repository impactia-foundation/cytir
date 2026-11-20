"""German de-compounding for the BM25 leg: ONE home for the rule (ADR-004, ADR-121).

The method is `dtuggener/CharSplit` (MIT, commit 9710624e): a character n-gram model
that ranks the cut points of a word. This module wraps it behind one function,
`decompound_parts`, so the write side (`cytir.core.common`) and the read side
(`cytir/serve/schema.py`) cannot drift. Output is bit-identical to the GPL-3.0
package `compound_split` 1.0.2 that ran before (study: specs/compound-split-replacement.md,
698,369 corpus tokens + 200,000 fuzz inputs; pinned by tests/test_compound_split_oracle.py and its sealed fixture).
Any difference would force a rebuild of every `lang=de` BM25 row, so do not "improve" the rule.

Two things come from outside the repo, both pinned:

* the splitter CODE (`charsplit/splitter.py`): the `charsplit` package, installed from
  the pinned upstream commit with `pip install "charsplit @ <url>"` (`PIN_INSTALL`; in no extra). The PyPI name `charsplit` is ANOTHER project: never install it.
  This module reads the file's source, checks its sha256, and runs it. It never imports
  the package's `__init__`.
* the n-gram TABLE (`ngram_probs.json`, 40 MB), accepted from two places, both checked
  against the sha256 pin: (1) the fetched model asset, see
  `cytir.embed.model_assets.CHARSPLIT_NGRAM` (`python3 -m cytir.embed.fetch_models` fetches it);
  (2) the copy that pip installs inside the `charsplit` package, used when (1) is absent.
  An explicit `AW_CHARSPLIT_NGRAM` is absolute: only that path is tried.

If either is missing or fails its sha256, `DecompoundUnavailable` is raised. There is no
fallback to "no splitting": that would index and query German with different tokens and
change ranking without a trace.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import tempfile
import threading
import types
from pathlib import Path

# Pin of the splitter source at upstream commit 9710624ef0b0df0edd12a721c33533ff1d07b557.
SPLITTER_SHA256 = "10854d3fe2a57be38ddeed9650a27d49efe765759237d5d0b8aaac8b3d7968ce"

MIN_TOKEN = 12     # tokens shorter than this are never split
THRESHOLD = 0.7    # score gate on the best split
MIN_PART = 3       # each part must have at least this many characters


class DecompoundUnavailable(RuntimeError):
    """The pinned splitter code or its n-gram table is absent or not the pinned bytes."""


_lock = threading.Lock()
# PyPI refuses direct-URL requirements, so no extra carries the splitter code (task 2989).
PIN_INSTALL = "charsplit @ https://github.com/dtuggener/CharSplit/archive/9710624ef0b0df0edd12a721c33533ff1d07b557.tar.gz"
_splitter = None


def _splitter_source() -> Path:
    try:
        spec = importlib.util.find_spec("charsplit")
    except (ImportError, ValueError):
        spec = None
    locs = list(spec.submodule_search_locations or []) if spec else []
    for loc in locs:
        p = Path(loc) / "splitter.py"
        if p.is_file():
            return p
    raise DecompoundUnavailable(
        "German de-compounding needs the `charsplit` package from the pinned upstream "
        "commit (dtuggener/CharSplit 9710624e). It is not installed. Install it with: "
        f"pip install \"{PIN_INSTALL}\" . Not the PyPI project of the same name.")


def _package_table(splitter_py: Path) -> Path | None:
    """The table pip installs next to the splitter source, if present."""
    p = splitter_py.parent / "ngram_probs.json"
    return p if p.is_file() else None


def _find_table(asset, splitter_py: Path) -> str:
    """The verified table path. Fetched asset first, then the pip-installed copy.

    A file that exists but fails its pin RAISES (a wrong file is never skipped for another).
    An explicit env var is absolute: no other location is tried.
    """
    explicit = os.environ.get(asset.env_var)
    if explicit:
        searched = [Path(explicit)]
    else:
        searched = list(asset.search_paths())
        pkg = _package_table(splitter_py)
        if pkg is not None:
            searched.append(pkg)
    for cand in searched:
        try:
            present = cand.is_file()
        except OSError:
            continue
        if not present:
            continue
        try:
            asset.verify(cand)
        except ValueError as exc:
            raise DecompoundUnavailable(str(exc)) from exc
        return str(cand)
    where = "\n".join(f"    {c}" for c in searched)
    raise DecompoundUnavailable(
        f"{asset.name}: n-gram table missing. Searched:\n{where}\n"
        f"  why: {asset.why}\n{asset.howto()}")


def _build():
    from cytir.embed.model_assets import CHARSPLIT_NGRAM as asset
    src_path = _splitter_source()
    source = src_path.read_bytes()
    got = hashlib.sha256(source).hexdigest()
    if got != SPLITTER_SHA256:
        raise DecompoundUnavailable(
            f"{src_path} is not the pinned CharSplit splitter (sha256 {got}, "
            f"expected {SPLITTER_SHA256}). Wrong package? Install the pinned commit.")
    table = _find_table(asset, src_path)
    # The upstream module loads `Path(__file__).parent / "ngram_probs.json"` when it runs.
    # Run it with `__file__` in a scratch dir that links to the VERIFIED table.
    with tempfile.TemporaryDirectory(prefix="aw-charsplit-") as tmp:
        os.symlink(os.path.abspath(table), os.path.join(tmp, "ngram_probs.json"))
        mod = types.ModuleType("aw_charsplit_splitter")
        mod.__file__ = os.path.join(tmp, "splitter.py")
        exec(compile(source, str(src_path), "exec"), mod.__dict__)  # noqa: S102 - sha256-pinned
    return mod.Splitter().split_compound


def split_candidates(word: str):
    """The raw upstream result: [(score, head, tail), ...], best first."""
    global _splitter
    if _splitter is None:
        with _lock:
            if _splitter is None:
                _splitter = _build()
    return _splitter(word)


def ensure_ready() -> None:
    """Load now (raises `DecompoundUnavailable`). Call at start-up to fail fast."""
    split_candidates("a" * MIN_TOKEN)


def decompound_parts(token: str) -> tuple[str, ...]:
    """Lower-cased (head, tail) of the best split, or () for no split."""
    if len(token) < MIN_TOKEN:
        return ()
    if _splitter is None:
        ensure_ready()
    try:
        res = _splitter(token)
        if res and res[0][0] >= THRESHOLD:
            return tuple(p.lower() for p in (res[0][1], res[0][2]) if len(p) >= MIN_PART)
    except Exception:  # noqa: BLE001 - per-token failure = no split, as before
        pass
    return ()
