"""One home for the non-HuggingFace model FILES the pipeline needs (doctrine 2:
config, not prose; doctrine "One Truth One Home").

Before this module the fastText language-scope model was a bare path default in
``fetch_process.py`` — `<models-dir>/lid.176.bin`, one box's layout — with
**no URL and no checksum anywhere in the repo** (task 429, blocker D). A cold
builder could not obtain the file at all, so the EN/DE/FR/IT scope gate was
unreachable outside the build box and only dump-based sources were buildable.

The URL and digest below were VERIFIED, not recalled: the file at ``url`` was
downloaded on 2026-08-19 and is byte-identical (same size, same sha256) to the
``<models-dir>/lid.176.bin`` that the build box has been running since 2026-07-10.

Fetch with::

    python3 -m cytir.embed.fetch_models            # verifies the digest, refuses on mismatch
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from cytir.core.paths import operator_settings

REPO_ROOT = Path(__file__).resolve().parents[2]


class ShaMismatch(ValueError):
    """Right size, wrong digest: the source served other bytes. Final, never retried on a mirror."""


class ModelAsset:
    """A pinned, checksummed model file that is fetched rather than pip-installed."""

    def __init__(self, name: str, filename: str, url: str, sha256: str,
                 size_bytes: int, env_var: str, why: str,
                 mirrors: tuple[str, ...] = ()):
        self.name = name
        self.filename = filename
        # Ordered source list: the canonical URL first, then byte-identical mirrors.
        # ``url`` stays the first entry for compatibility. The pinned sha256 is the only
        # trust anchor, so a mirror needs no extra trust (task 2776).
        self.urls: tuple[str, ...] = (url, *mirrors)
        self.url = url
        self.sha256 = sha256
        self.size_bytes = size_bytes
        self.env_var = env_var
        self.why = why

    # --- where the file may live -------------------------------------------
    def search_paths(self) -> list[Path]:
        """Candidate locations, most-specific first.

        ``AW_MODELS_DIR`` lets a self-hoster put the models anywhere.  The
        operator's own model directory (setting ``fallback_models_dir``) stays FIRST among the
        implicit candidates so that every existing production invocation resolves to exactly the
        byte it resolved to before this module existed — this is additive discovery, not a
        relocation.
        """
        out: list[Path] = []
        env_dir = os.environ.get("AW_MODELS_DIR")
        if env_dir:
            out.append(Path(env_dir) / self.filename)
        fallback = operator_settings().fallback_models_dir    # the operator's model dir, if any
        if fallback:
            out.append(Path(fallback) / self.filename)
        out.append(REPO_ROOT / "models" / self.filename)      # repo-local (cold start)
        out.append(Path.home() / ".cache" / "cytir" / "models" / self.filename)
        out.append(Path.home() / ".cache" / "aw" / "models" / self.filename)  # rename-codemod: keep (legacy cache dir on the boxes)
        return out

    def resolve(self) -> tuple[str, bool]:
        """Return (path, exists).

        An explicitly set ``env_var`` is ABSOLUTE: it is returned whether or not
        it exists and no other candidate is tried.  Pointing the variable at a
        missing file is how callers deliberately disable the gate, and silently
        substituting a different file would defeat that.
        """
        explicit = os.environ.get(self.env_var)
        if explicit:
            return explicit, os.path.exists(explicit)
        candidates = self.search_paths()
        for c in candidates:
            # A candidate under an unreadable directory (e.g. /root at 0700
            # probed by a non-root cold builder) raises PermissionError from
            # Path.exists(); treat it as absent, never as fatal.
            try:
                if c.exists():
                    return str(c), True
            except OSError:
                continue
        return str(candidates[-2]), False   # repo-local is the advertised default

    def default_dest(self) -> Path:
        explicit = os.environ.get(self.env_var)
        if explicit:
            return Path(explicit)
        env_dir = os.environ.get("AW_MODELS_DIR")
        if env_dir:
            return Path(env_dir) / self.filename
        return REPO_ROOT / "models" / self.filename

    # --- verification -------------------------------------------------------
    def digest(self, path: str | os.PathLike) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()

    def verify(self, path: str | os.PathLike) -> None:
        """Raise unless ``path`` is byte-for-byte the pinned artifact."""
        actual_size = os.path.getsize(path)
        if actual_size != self.size_bytes:
            raise ValueError(
                f"{self.name}: size mismatch at {path} — expected {self.size_bytes} "
                f"bytes, got {actual_size}. Refusing to use an unverified model file.")
        actual = self.digest(path)
        if actual != self.sha256:
            raise ShaMismatch(
                f"{self.name}: sha256 mismatch at {path}\n"
                f"  expected {self.sha256}\n"
                f"  actual   {actual}\n"
                f"Refusing to use an unverified model file. Re-fetch with "
                f"`python3 -m cytir.embed.fetch_models --force`.")

    def howto(self) -> str:
        """The actionable acquisition instructions, for failure messages."""
        return (f"    python3 -m cytir.embed.fetch_models --asset {self.name}\n"
                f"  or by hand:\n"
                f"    mkdir -p {self.default_dest().parent}\n"
                f"    curl -fSL -o {self.default_dest()} <one of the URLs below, in order>\n"
                + "".join(f"      {u}\n" for u in self.urls) +
                f"    echo '{self.sha256}  {self.default_dest()}' | sha256sum -c -\n"
                f"  (or set {self.env_var} to an existing copy)")


LID_176 = ModelAsset(
    name="lid.176",
    filename="lid.176.bin",
    # Canonical upstream: Facebook AI Research's fastText language-identification
    # model, the one the task-166 audit measured (98.0% label accuracy, 0.0%
    # in-scope false-drop) and the one the build box runs.
    url="https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
    # Mirror: the same bytes on Hugging Face, pinned to a commit. Added 2026-11-12 (task 2776)
    # after the canonical host's TLS certificate expired on 2026-11-11 23:59:59Z. Checked by
    # sha256 against the pin below, not trusted by name.
    mirrors=("https://huggingface.co/julien-c/fasttext-language-id/resolve/"
             "9f2e94d2fd6b2615d26f5ec5b34cf7e2f38cb86e/lid.176.bin",),
    sha256="7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e",
    size_bytes=131266198,
    env_var="AW_LID_MODEL",
    why=("fastText EN/DE/FR/IT content language scope-gate (task 166/238). "
         "Without it every non-dump ingest path refuses to run, because the "
         "alternative is stamping out-of-scope pages with a URL-guessed label "
         "and leaking them into the corpus."),
)

CHARSPLIT_NGRAM = ModelAsset(
    name="charsplit-ngram",
    filename="ngram_probs.json",
    # dtuggener/CharSplit (MIT) at the pinned commit; the same commit the `charsplit`
    # requirement installs. The raw URL names the commit, not a branch. Tim's ruling
    # 26 (2026-10-05): fetched with a sha256 pin, not vendored (ADR-121, study
    # specs/compound-split-replacement.md). A table of n-gram probabilities, no text.
    url=("https://raw.githubusercontent.com/dtuggener/CharSplit/"
         "9710624ef0b0df0edd12a721c33533ff1d07b557/charsplit/ngram_probs.json"),
    sha256="b145f5fba901698479d3e889255cf9342fe7e6c3593f80f7dc1932d1759be7bf",
    size_bytes=40447815,
    env_var="AW_CHARSPLIT_NGRAM",
    why=("German de-compounding table for the BM25 leg (ADR-004, ADR-121). Without it "
         "German tokens of 12+ characters cannot be split, so a build or a query would "
         "tokenise German differently from the stored index. There is no fallback."),
)

ASSETS = {a.name: a for a in (LID_176, CHARSPLIT_NGRAM)}
