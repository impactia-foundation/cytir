#!/usr/bin/env python3
"""Fetch the pinned model FILES the pipeline needs but pip cannot install.

Today that is exactly one file: `lid.176.bin`, the fastText model behind the
EN/DE/FR/IT language scope-gate.  Before task 432 the repo named that file in a
path default and nowhere else — no URL, no checksum, no fetch step — so a cold
builder could not obtain it and the gate was unreachable off the build box.

    cyt embed models              # fetch everything missing   (== python3 -m cytir.embed.fetch_models)
    cyt embed models --check      # verify only; never downloads; also checks the embedder pin
    cyt embed models --force      # re-fetch even if present
    cyt embed models --tokenizer  # also fetch the pinned embedder's TOKENIZER files (22 MB, not the 9.4 GB model)

The embedder itself (`cytir.core.model_pin`: model id + 40-hex revision) is fetched by the first
embed, not here; every run prints the pin, and `--check` fails when `AW_MODEL_REVISION`
overrides it (vectors made under another revision do not match the index).

Every download is checksum-verified against the pin in `model_assets.py` and a
mismatched file is deleted, not kept.  Exit code is non-zero on any failure —
this script is safe to put in front of a build in a shell that uses `set -e`.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from cytir.consent.block import ConsentBlocked
from cytir.core.user_agent import UnidentifiedCrawler
from cytir.embed.model_assets import ASSETS, ModelAsset, ShaMismatch
from cytir.sources.family import HttpSpec
from cytir.sources.http import make_client


_DOWNLOAD_ATTEMPTS = 3


def _cause(exc: Exception) -> str:
    """Name why a source failed: TLS, HTTP status, or other transport trouble."""
    import httpx  # noqa: PLC0415
    import ssl  # noqa: PLC0415
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    chain, e = [], exc
    while e is not None and len(chain) < 6:
        chain.append(e)
        e = e.__cause__ or e.__context__
    if any(isinstance(c, ssl.SSLError) for c in chain) or "CERTIFICATE_VERIFY_FAILED" in str(exc):
        return f"TLS error ({exc})"
    return f"{type(exc).__name__} ({exc})"


def _is_source_failure(exc: Exception) -> bool:
    """A TLS or HTTP-status failure is deterministic: go to the next source, do not retry it."""
    import httpx  # noqa: PLC0415
    return isinstance(exc, httpx.HTTPStatusError) or _cause(exc).startswith("TLS error")


def _get_one(asset: ModelAsset, url: str, tmp: Path, spec: HttpSpec) -> None:
    """Fetch ``url`` into ``tmp`` and verify. A truncated body (size mismatch, task 1712) or a
    flaky transport is retried from scratch; a TLS/HTTP failure raises at once (next source);
    a sha256 mismatch at the right size raises at once and is final for the whole asset."""
    # iter_bytes, not iter_raw: raw.githubusercontent.com gzips the charsplit table (13 MB on the
    # wire, 40 MB real), and raw bytes failed the size check on a stranger's clean run (task 2251).
    # The host is external: the consent guard applies, redirects included.
    for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
        try:
            with make_client(spec) as client, client.stream("GET", url) as r, open(tmp, "wb") as fh:
                r.raise_for_status()
                for block in r.iter_bytes(1 << 20):
                    fh.write(block)
            asset.verify(tmp)
            return
        except Exception as exc:
            if os.path.exists(tmp):
                os.unlink(tmp)
            if _is_source_failure(exc) or isinstance(exc, (ShaMismatch, ConsentBlocked, UnidentifiedCrawler)) or attempt == _DOWNLOAD_ATTEMPTS:
                raise
            print(f"[fetch_models] {asset.name}: attempt {attempt}/{_DOWNLOAD_ATTEMPTS} failed, "
                  f"retrying from scratch: {exc}", file=sys.stderr)


def _download(asset: ModelAsset, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[fetch_models] {asset.name}: -> {dest} ({asset.size_bytes/1e6:.0f} MB)")
    # Fetch to a .part file and only rename after the digest checks out, so an
    # interrupted or corrupted download can never masquerade as the real model.
    spec = HttpSpec(timeout=120.0, headers={"user-agent": "authoritative-web-fetch-models"})
    failures: list[str] = []
    urls = getattr(asset, "urls", None) or (asset.url,)
    for i, url in enumerate(urls):
        print(f"[fetch_models] {asset.name}: GET {url}"
              + ("" if i == 0 else f"  (mirror {i}, after {len(failures)} failed source(s))"))
        try:
            _get_one(asset, url, tmp, spec)
        except (ShaMismatch, ConsentBlocked, UnidentifiedCrawler):
            raise                          # wrong bytes, a declined host or no crawler identity are a refusal: never a retry, never a silent next mirror
        except Exception as exc:
            cause = _cause(exc)
            failures.append(f"{url}: {cause}")
            print(f"[fetch_models] {asset.name}: source failed: {cause}", file=sys.stderr)
            continue
        os.replace(tmp, dest)
        print(f"[fetch_models] {asset.name}: OK served by {url} sha256={asset.sha256}")
        return
    raise RuntimeError(f"{asset.name}: every source failed: " + "; ".join(failures))


def fetch_tokenizer() -> bool:
    """Put the pinned embedder's tokenizer files (json/model/txt, ~22 MB) in the HF cache.

    The chunker refuses to write word-only chunks (task 2289), so any box that chunks needs these.
    Idempotent: a cached snapshot is a no-op. Needs `transformers` (it brings huggingface_hub)."""
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
        from cytir.core import model_pin  # noqa: PLC0415
        path = snapshot_download(model_pin.EMBED_MODEL_ID, revision=model_pin.embed_model_revision(),
                                 allow_patterns=["*.json", "*.model", "*.txt"])
    except Exception as exc:  # noqa: BLE001
        print(f"[fetch_models] tokenizer: FETCH FAILED: {type(exc).__name__}: {exc}\n"
              f"  needs `pip install transformers` and network access to huggingface.co", file=sys.stderr)
        return False
    print(f"[fetch_models] tokenizer: OK at {path}")
    return True


def handle(asset: ModelAsset, *, check_only: bool, force: bool) -> bool:
    path, exists = asset.resolve()
    if exists and not force:
        try:
            asset.verify(path)
        except Exception as exc:
            print(f"[fetch_models] {asset.name}: PRESENT BUT INVALID at {path}\n  {exc}",
                  file=sys.stderr)
            return False
        print(f"[fetch_models] {asset.name}: present and verified at {path}")
        return True
    if check_only:
        print(f"[fetch_models] {asset.name}: MISSING (expected at {path})\n"
              f"  why: {asset.why}\n{asset.howto()}", file=sys.stderr)
        return False
    try:
        _download(asset, asset.default_dest())
    except Exception as exc:
        print(f"[fetch_models] {asset.name}: FETCH FAILED: {exc}\n{asset.howto()}",
              file=sys.stderr)
        return False
    return True


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asset", action="append", choices=sorted(ASSETS),
                    help="fetch only these (default: all)")
    ap.add_argument("--check", action="store_true",
                    help="verify presence + checksum, never download")
    ap.add_argument("--tokenizer", action="store_true",
                    help="also fetch the pinned embedder's tokenizer files into the HF cache")
    ap.add_argument("--force", action="store_true", help="re-fetch even if present")
    args = ap.parse_args(argv)

    from cytir.core import model_pin
    try:
        print(f"[fetch_models] {model_pin.pin_note()}")
        pin_ok = model_pin.is_pinned_default() or not args.check
    except ValueError as exc:                    # AW_MODEL_REVISION is not a 40-hex sha
        print(f"[fetch_models] PIN INVALID: {exc}", file=sys.stderr)
        pin_ok = False
    if not pin_ok:
        print("[fetch_models] the embedder pin is overridden (see above); --check wants the "
              "committed pin", file=sys.stderr)
    names = args.asset or ([] if args.tokenizer else sorted(ASSETS))
    ok = pin_ok
    if args.tokenizer and not args.check:
        ok &= fetch_tokenizer()
    for name in names:
        ok &= handle(ASSETS[name], check_only=args.check, force=args.force)
    if not ok:
        print("[fetch_models] one or more assets are missing or invalid", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
