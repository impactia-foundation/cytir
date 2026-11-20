"""Builder corpus text index, published/conf sidecar maps, the lmdb corpus-text
KV store and its fold (moved from the b1 builder script, task 2106).

The code below is the original block, unchanged except for three named
exceptions: `build_corpus_text_index` and `build_corpus_kv` take the corpus
directory as the keyword `corpus_dir` (it was the builder's module constant),
and `build_corpus_kv` takes the builder's `save_state` as a keyword. `lmdb`
stays an optional dependency: it is imported inside `kv_open`, as before."""
from __future__ import annotations

import glob
import gzip
import hashlib
import json
import os

import cytir.core.common as AC
import cytir.core.ids as ID


# --------------------------------------------------------------------------
# corpus text index — single targeted pass over parquet for a needed url-set
# --------------------------------------------------------------------------
def _canon_key(url: str, anchor) -> str:
    """Join key: canonical_url() (aw_common) + '#' + real anchor (identity.md
    §1.2 / task 301 R1: join by (url[,anchor]), NOT point_id)."""
    a = ID._clean_anchor(anchor) or ""
    return AC.canonical_url(url or "") + "#" + a


def build_corpus_text_index(needed_keys: set[str], waves: list[str] | None = None,
                            max_rows: int = 0, log=print, *,
                            corpus_dir: str) -> dict[str, dict]:
    """One pass over the corpus parquet keeping only rows whose (canon_url,anchor)
    key is in `needed_keys`. Returns key -> {texts: [..], license, consent_*, ...}.
    ALL distinct texts under a key are kept (KV-v2 semantics, task 319/GAP A33):
    an anchor-less multi-chunk page shares one join key across its chunks, and
    keeping only the first text is exactly the 318 chunk-collapse.

    `waves`: optional wave-dir allowlist (slice speed). None = all waves.
    `max_rows`: if >0, a scan budget — any needed key not matched within budget
    becomes a no-text TAIL record (kept sparse, tok=1). This is safe by the
    architecture (the tail is honest), and bounds slice wall-clock over the giant
    waves (e.g. 28M pubmed). Smaller waves are scanned first so the budget favors
    high-value re-tokenizable content.
    """
    import pyarrow.parquet as pq
    cols = ["url", "anchor", "text", "lang", "country", "source_type",
            "consent_search_ok", "consent_train_ok", "consent_checked_at",
            "extraction_version", "extra"]
    out: dict[str, dict] = {}
    wave_dirs = sorted(glob.glob(os.path.join(corpus_dir, "wave=*")))
    if waves:
        wave_dirs = [w for w in wave_dirs
                     if os.path.basename(w).replace("wave=", "") in waves]
    files = []
    for w in wave_dirs:
        files += glob.glob(os.path.join(w, "**", "*.parquet"), recursive=True)
    # smallest-first so the row budget favors the small high-value waves
    files.sort(key=lambda f: os.path.getsize(f))
    log(f"[corpus] scanning {len(files)} parquet files across {len(wave_dirs)} "
        f"waves for {len(needed_keys):,} keys (budget={max_rows or 'none'})")
    scanned = 0
    for fi, f in enumerate(files):
        if max_rows and scanned >= max_rows:
            log(f"[corpus] row budget {max_rows:,} reached — remaining keys -> tail")
            break
        if len(out) >= len(needed_keys):
            log("[corpus] all needed keys matched — stop early")
            break
        try:
            pf = pq.ParquetFile(f)
        except Exception as e:  # noqa: BLE001
            log(f"[corpus] skip unreadable {f}: {e}")
            continue
        avail = set(pf.schema_arrow.names)
        use = [c for c in cols if c in avail]
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg, columns=use).to_pylist()
            for r in tbl:
                scanned += 1
                txt = r.get("text")
                if not txt:
                    continue
                k = _canon_key(r.get("url"), r.get("anchor"))
                if k in needed_keys:
                    row = out.get(k)
                    if row is None:
                        row = dict(r)
                        row["texts"] = [txt]
                        out[k] = row
                    elif txt not in row["texts"]:
                        row["texts"].append(txt)
        if (fi + 1) % 10 == 0:
            log(f"[corpus]  {fi+1}/{len(files)} files, scanned {scanned:,}, "
                f"matched {len(out):,}/{len(needed_keys):,}")
    log(f"[corpus] DONE scanned {scanned:,} rows, matched {len(out):,}/"
        f"{len(needed_keys):,} ({100*len(out)/max(len(needed_keys),1):.1f}%)")
    return out


# --------------------------------------------------------------------------
# sidecar loaders (published dates for news, resolve_confidence for cards)
# --------------------------------------------------------------------------
def load_published_map(paths: list[str], log=print) -> dict[str, str]:
    """canonical_url -> ISO published date, from newsfresh work-dir jsonl(.gz)."""
    m: dict[str, str] = {}
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        op = gzip.open if p.endswith(".gz") else open
        try:
            with op(p, "rt", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    u, pub = d.get("url"), d.get("published")
                    if u and pub:
                        m[AC.canonical_url(u)] = pub
        except Exception as e:  # noqa: BLE001
            log(f"[published] skip {p}: {e}")
    log(f"[published] loaded {len(m):,} url->date")
    return m


def load_resolve_conf_map(path: str, log=print) -> dict[str, float]:
    """domain -> resolve_confidence (the `conf` field, ADR-041), from the
    cards_p2_audit.jsonl sidecar (task 296)."""
    m: dict[str, float] = {}
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                dom, conf = d.get("domain"), d.get("conf")
                if dom is not None and conf is not None:
                    m[str(dom).lower().lstrip(".")] = float(conf)
    log(f"[resolve_conf] loaded {len(m):,} domain->conf")
    return m


# --- LMDB corpus-text KV ---------------------------------------------------
_KV_MAP_SIZE = 260 * 1024**3   # 260 GB sparse map (actual grows on nvme4t)

# KV format v3: the v2 multi-text contract plus an optional, atomic pair of
# dense/sparse surfaces for source-specific representations (Fedlex R2).
# A key holds ALL distinct chunk texts for
# its (canonical_url, anchor) join key, \x00-separated (text never contains
# NUL). v1 was first-write-wins — anchor-less multi-chunk pages kept only ONE
# chunk's text, which collapsed every sibling chunk to one digest/one id in
# the 318 build. A v1 KV is UNUSABLE for a correct build; version-stamp and
# refuse to read a mismatched KV (fail loud, never silently rebuild wrong).
_KV_VERSION_KEY = b"__aw_kv_version__"      # 18 bytes: cannot collide with
_KV_VERSION = b"3"                          # the 16-byte blake2b record keys
_KV_SEP = b"\x00"
_KV_CHANNEL_PREFIX = b"C"                    # 17-byte metadata key; record hashes are 16 bytes


def kv_open(path, readonly):
    import lmdb
    env = lmdb.open(path, map_size=_KV_MAP_SIZE, subdir=True,
                    readonly=readonly, lock=(not readonly), readahead=False,
                    max_dbs=0, writemap=(not readonly))
    if readonly:
        with env.begin() as txn:
            v = txn.get(_KV_VERSION_KEY)
        if v != _KV_VERSION:
            env.close()
            raise SystemExit(
                f"[FATAL] corpus KV at {path} is version {v!r}, need "
                f"{_KV_VERSION!r} (v1 = first-write-wins; v2 lacks explicit "
                f"dense/sparse channels). Rebuild the kv phase; do not reuse.")
    else:
        with env.begin(write=True) as txn:
            v = txn.get(_KV_VERSION_KEY)
            if v is None:
                txn.put(_KV_VERSION_KEY, _KV_VERSION)
            elif v != _KV_VERSION:
                env.close()
                raise SystemExit(
                    f"[FATAL] corpus KV at {path} is version {v!r}, need "
                    f"{_KV_VERSION!r} — delete/rebuild it, never mix formats.")
    return env


def _kv_key(canon_key: str) -> bytes:
    """LMDB caps keys at 511 bytes; our canonical join keys (long wiki/fedlex
    URLs) exceed that. Hash to a fixed 16-byte blake2b digest (collision-safe at
    corpus scale). Both put and get sides use this — the join is unchanged."""
    return hashlib.blake2b(canon_key.encode("utf-8"), digest_size=16).digest()


def _kv_put_multi(txn, k: bytes, text: str) -> bool:
    """Append `text` to key k unless already present (KV v2 multi-value).
    Returns True if the KV grew (new key or new distinct text)."""
    new = text.encode("utf-8")
    cur = txn.get(k)
    if cur is None:
        txn.put(k, new)
        return True
    if new in cur.split(_KV_SEP):
        return False
    txn.put(k, cur + _KV_SEP + new)
    return True


def _kv_get_texts(txn, k: bytes) -> list[str] | None:
    v = txn.get(k)
    if v is None:
        return None
    return [p.decode("utf-8") for p in v.split(_KV_SEP)]


def _kv_channel_key(k: bytes) -> bytes:
    return _KV_CHANNEL_PREFIX + k


def _kv_put_channels(txn, k: bytes, dense_text: str, sparse_text: str,
                     bundle_digest: str | None = None) -> None:
    """Persist a paired channel contract beside the normal corpus body.

    There is no safe "first wins" rule for two competing channel pairs on the
    same (url, anchor): it would make a rerun depend on input order. Reject it
    so the source-specific builder has to resolve the conflict explicitly.
    """
    if bundle_digest is not None and (
            not isinstance(bundle_digest, str) or len(bundle_digest) != 64):
        raise ValueError("invalid corpus channel bundle_digest")
    channel = {"dense_text": dense_text, "sparse_text": sparse_text}
    if bundle_digest is not None:
        channel["bundle_digest"] = bundle_digest
    value = json.dumps(channel, ensure_ascii=False,
                       separators=(",", ":")).encode("utf-8")
    ck = _kv_channel_key(k)
    old = txn.get(ck)
    if old is not None and old != value:
        raise ValueError("conflicting dense/sparse channels for one corpus key")
    txn.put(ck, value)


def _kv_get_channels(txn, k: bytes) -> dict | None:
    value = txn.get(_kv_channel_key(k))
    if value is None:
        return None
    try:
        channel = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid corpus dense/sparse channel payload") from exc
    if not isinstance(channel, dict) or not channel.get("dense_text") \
            or not channel.get("sparse_text"):
        raise ValueError("incomplete corpus dense/sparse channel payload")
    digest = channel.get("bundle_digest")
    if digest is not None and (not isinstance(digest, str) or len(digest) != 64):
        raise ValueError("invalid corpus channel bundle_digest")
    return channel


def _fold_jsonl_into_kv(env, path, marker, done, st, log):
    """Fold a jsonl into the KV, retaining explicit paired text channels.

    The ordinary text multi-value contract remains intact for all inputs.  A
    channel-bearing row must carry both fields; this is a build input error,
    not a reason to silently fall back to the legacy surface.
    """
    if not path or not os.path.exists(path) or (marker in done):
        return 0
    put = 0
    op = gzip.open if path.endswith(".gz") else open
    with env.begin(write=True) as txn:
        for line in op(path, "rt", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if not d.get("text"):
                continue
            k = _kv_key(_canon_key(d.get("url"), d.get("anchor")))
            if _kv_put_multi(txn, k, d["text"]):
                put += 1
            dense_text = d.get("dense_text")
            sparse_text = d.get("sparse_text")
            if dense_text is not None or sparse_text is not None:
                if not dense_text or not sparse_text:
                    raise ValueError(
                        f"{path}: incomplete dense/sparse text channels for "
                        f"{d.get('url')!r}")
                _kv_put_channels(txn, k, dense_text, sparse_text,
                                 d.get("bundle_digest"))
    done.add(marker)
    st["kv_done_files"] = sorted(done)
    log(f"[kv] folded {os.path.basename(path)}: put {put:,}")
    return put


def build_corpus_kv(kv_path, fedlex_path, out_dir, st, log, extra_jsonls=None,
                    *, corpus_dir, save_state):
    import pyarrow.parquet as pq
    env = kv_open(kv_path, readonly=False)
    done = set(st.get("kv_done_files", []))
    wave_dirs = sorted(glob.glob(os.path.join(corpus_dir, "wave=*")))
    files = []
    for w in wave_dirs:
        files += glob.glob(os.path.join(w, "**", "*.parquet"), recursive=True)
    files.sort()
    log(f"[kv] {len(files)} parquet files, {len(done)} already done")
    for fi, f in enumerate(files):
        if f in done:
            continue
        try:
            pf = pq.ParquetFile(f)
        except Exception as e:  # noqa: BLE001
            log(f"[kv] skip unreadable {f}: {e}")
            done.add(f)
            continue
        avail = set(pf.schema_arrow.names)
        cols = [c for c in ("url", "anchor", "text") if c in avail]
        put = skip = 0
        for rg in range(pf.num_row_groups):
            rows = pf.read_row_group(rg, columns=cols).to_pylist()
            with env.begin(write=True) as txn:   # one txn per row-group (bounds dirty mem)
                for r in rows:
                    txt = r.get("text")
                    if not txt:
                        continue
                    k = _kv_key(_canon_key(r.get("url"), r.get("anchor")))
                    if _kv_put_multi(txn, k, txt):
                        put += 1
                    else:
                        skip += 1
        done.add(f)
        st["kv_done_files"] = sorted(done)
        if (fi + 1) % 5 == 0 or put > 100000:
            n = env.stat()["entries"]
            log(f"[kv] {fi+1}/{len(files)} {os.path.basename(os.path.dirname(f))} "
                f"put {put:,} dup {skip:,} | total entries {n:,}")
            save_state(out_dir, st)
    # fedlex.jsonl (re-extracted Swiss law text — Tier-A, must fold)
    _fold_jsonl_into_kv(env, fedlex_path, "__fedlex__", done, st, log)
    # extra law re-extraction jsonls (accent-bearing DE/FR/IT law tail, hybrid C)
    for p in (extra_jsonls or []):
        _fold_jsonl_into_kv(env, p, "__extra__" + os.path.basename(p), done, st, log)
    n = env.stat()["entries"]
    env.sync()
    env.close()
    log(f"[kv] DONE — {n:,} distinct corpus-text keys")
    st["kv_entries"] = n
    save_state(out_dir, st)
    return n
