"""Which versioned chunk policy runs: `chunker_selection`, `chunk_markdown_channels`
(moved out of `cytir.core.common`, task 2333)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os
import re

from cytir.core.paths import data_path as _data_path
from cytir.core.common import (  # noqa: E402
    ChunkTextChannels,
)


# --- task 340 / 373: which versioned chunk policy runs --------------------
# `legacy` = fp.chunk 180-word paragraph windows, overlap 30 -> chunk@1 (LIVE)
# `structured` = chunk.py driven by one explicit, hash-pinned policy.
#
# The mode deliberately carries no size. Boundaries belong to the versioned
# policy artifact, not to an enum copied into every dispatcher. Candidate runs
# must name both AW_CHUNK_POLICY and its exact byte SHA. At a future alias flip,
# the tracked profile binds the same path/SHA/epoch/legend atomically. A policy
# change gets a new artifact + epoch + legend; it never edits an old artifact
# in place (task 373, verification doctrine 2/5).
CHUNKER_MODES = ("legacy", "structured")
_LEGACY_CHUNK_EPOCH = 1
_LEGACY_CHUNK_LEGEND = "legacy-180w-o30@1"
_CHUNKER_ANNOUNCED = {"selection": None}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ChunkerSelection:
    mode: str
    epoch: int
    legend: str
    policy_path: Optional[str] = None
    policy_sha256: Optional[str] = None
    tokenizer_revision: Optional[str] = None

    # --- task 407: the selection already KNOWS the whole rule; stamp it ------
    # `Passage.stamps` used to carry `{"chunk": <int>}` and nothing else, so a
    # row could not name the policy that produced it. Five tracked policy
    # artifacts declare epoch 3, and Tim's 2026-08-13 go-forward policy
    # (…-tok2560, epoch 4) differs from the shipped epoch-3 one in a SINGLE
    # field (`token_hard_max.exact` 2000 -> 2560) — identical output on every
    # unit that does not hit the backstop. The epoch int cannot separate them;
    # the policy sha can. This is the single site every producer inherits.
    def provenance(self) -> dict:
        """The `aw.chunk-provenance.v1` record for rows this selection makes."""
        from cytir.chunk import provenance as CP

        if self.mode == "legacy":
            # fetch_process.chunk: paragraph-aware ~180-word windows with a
            # 30-word carry-over. It has no policy artifact — that absence is
            # a fact about the legacy chunker, not a missing measurement, so
            # the policy fields are simply not present rather than null-filled.
            chunker = {"mode": "legacy", "chunk_epoch": self.epoch,
                       "rule_legend": self.legend,
                       "algorithm": "paragraph-window-with-overlap",
                       "target_words": 180, "overlap_words": 30}
            # chunk@1 emits ONE surface: dense and sparse both read the
            # canonical text (see chunk_markdown_channels' legacy branch).
            channels = {"canonical": "body", "dense": "body", "sparse": "body",
                        "channels_split": False}
        else:
            chunker = {"mode": self.mode, "chunk_epoch": self.epoch,
                       "rule_legend": self.legend,
                       "policy_path": _portable_policy_path(self.policy_path),
                       "policy_sha256": self.policy_sha256,
                       "chunk_tokenizer_revision": self.tokenizer_revision}
            # The structured chunker's whole point: the canonical/BM25 surface
            # keeps the full breadcrumb, the dense surface keeps only the
            # deepest retained heading (chunk.dense_text).
            channels = {"canonical": "hierarchy-plus-body",
                        "dense": "leaf-heading-plus-body",
                        "sparse": "hierarchy-plus-body",
                        "channels_split": True}
        family = ("generic-legacy-window" if self.mode == "legacy"
                  else "generic-structured")
        return CP.build(family=family, chunker=chunker, channels=channels,
                        confidence="stamped")

    def stamps(self) -> dict:
        """Per-row stamp: the epoch family value PLUS the interned rule id."""
        return {"chunk": self.epoch, "prov_id": self.provenance()["prov_id"]}


def _portable_policy_path(path: Optional[str]) -> Optional[str]:
    """The policy path as a RECORD may carry it: repo-relative, never a host path (task 771).

    `ChunkerSelection.policy_path` is the absolute path the file was loaded from; the provenance
    record (and so `prov_id`) must not depend on the checkout. The rule is the file's bytes, pinned
    by `policy_sha256` in the same record. A file outside the repo (a candidate copy) is named
    `data/<file name>`, the form the stage builders already write in their manifests."""
    if not path:
        return path
    from pathlib import Path as _Path
    p = _Path(path)
    if not p.is_absolute():
        return p.as_posix()
    try:
        return p.resolve().relative_to(_Path(__file__).resolve().parents[2]).as_posix()
    except ValueError:
        return f"data/{p.name}"


def _chunk_policy_path(raw: str) -> str:
    from pathlib import Path as _Path
    path = _Path(raw).expanduser()
    if not path.is_absolute():
        path = _Path(__file__).resolve().parents[2] / path
    return str(path.resolve())


def _load_pinned_chunk_policy(path_raw: object, sha_raw: object):
    """Load through the canonical policy module and verify the byte pin."""
    if not isinstance(path_raw, str) or not path_raw.strip():
        raise ValueError("structured chunker requires a non-empty chunk_policy_path")
    if not isinstance(sha_raw, str) or not _SHA256_RE.fullmatch(sha_raw):
        raise ValueError(
            "structured chunker requires a lowercase 64-hex chunk_policy_sha256")
    from cytir.chunk.policy import load_chunk_policy
    policy = load_chunk_policy(_chunk_policy_path(path_raw))
    if policy.sha256 != sha_raw:
        raise ValueError(
            f"chunk policy SHA mismatch for {policy.path}: "
            f"profile/env={sha_raw}, actual={policy.sha256}")
    return policy


def _chunk_tokenizer_revision(raw: object, source: str) -> str:
    """Require the immutable tokenizer used to enforce structured bounds."""
    if not isinstance(raw, str) or not re.fullmatch(r"[0-9a-f]{40}", raw):
        raise ValueError(
            f"structured chunker requires a lowercase 40-hex "
            f"chunk_tokenizer_revision from {source}")
    return raw


def chunker_selection() -> ChunkerSelection:
    """Resolve and verify the complete chunker selection.

    An explicit candidate selects `AW_CHUNKER=structured` plus
    `AW_CHUNK_POLICY`, `AW_CHUNK_POLICY_SHA256`, and the immutable
    `AW_CHUNK_TOKENIZER_REVISION`. Without an explicit mode, the tracked build
    profile is authoritative. Unknown, missing, stale-hash, mutable tokenizer,
    or epoch/legend-inconsistent values raise before content is chunked.
    """
    explicit = os.environ.get("AW_CHUNKER")
    if explicit is not None:
        mode = explicit.strip().lower()
        if mode not in CHUNKER_MODES:
            raise ValueError(
                f"AW_CHUNKER={mode!r} unknown; expected one of {CHUNKER_MODES}")
        if mode == "legacy":
            return ChunkerSelection(
                mode, _LEGACY_CHUNK_EPOCH, _LEGACY_CHUNK_LEGEND)
        policy = _load_pinned_chunk_policy(
            os.environ.get("AW_CHUNK_POLICY"),
            os.environ.get("AW_CHUNK_POLICY_SHA256"))
        revision = _chunk_tokenizer_revision(
            os.environ.get("AW_CHUNK_TOKENIZER_REVISION"), "environment")
        return ChunkerSelection(
            mode, policy.epoch, policy.rule_legend, str(policy.path),
            policy.sha256, revision)

    import json as _json
    from pathlib import Path as _Path
    profile = _Path(os.environ.get(
        "AW_BUILD_CONFIG",
        str(_data_path("build_config.json"))))
    try:
        cfg = _json.loads(profile.read_text(encoding="utf-8"))
        mode = str(cfg["chunker_mode"]).strip().lower()
        epoch = cfg["chunk_epoch"]
        legend = cfg["chunk_rule_legend"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            f"cannot resolve chunker from tracked build profile {profile}: "
            f"{exc}") from exc
    if mode not in CHUNKER_MODES:
        raise ValueError(
            f"inconsistent chunk profile {profile}: unknown mode={mode!r}")
    if mode == "legacy":
        if (epoch, legend) != (_LEGACY_CHUNK_EPOCH, _LEGACY_CHUNK_LEGEND):
            raise ValueError(
                f"inconsistent chunk profile {profile}: mode={mode!r}, "
                f"epoch={epoch!r}, legend={legend!r}")
        if cfg.get("chunk_policy_path") or cfg.get("chunk_policy_sha256"):
            raise ValueError(
                f"inconsistent chunk profile {profile}: legacy mode must not "
                "claim a structured chunk policy")
        return ChunkerSelection(mode, epoch, legend)

    policy = _load_pinned_chunk_policy(
        cfg.get("chunk_policy_path"), cfg.get("chunk_policy_sha256"))
    revision = _chunk_tokenizer_revision(
        cfg.get("chunk_tokenizer_revision"), str(profile))
    if (epoch, legend) != (policy.epoch, policy.rule_legend):
        raise ValueError(
            f"inconsistent chunk profile {profile}: mode={mode!r}, "
            f"epoch={epoch!r}, legend={legend!r}; policy requires "
            f"epoch={policy.epoch!r}, legend={policy.rule_legend!r}")
    return ChunkerSelection(
        mode, epoch, legend, str(policy.path), policy.sha256, revision)


def explicit_structured_chunker_selection(
        policy_path: str, policy_sha256: str,
        tokenizer_revision: str) -> ChunkerSelection:
    """Build the canonical structured selection from declared CLI pins.

    Batch reconstruction commands already declare these three immutable
    inputs.  Routing them through this helper lets the canonical dispatcher use
    the exact same validation as env/build-profile callers without consulting
    hidden process state or requiring ``data/build_config.json`` in a minimal
    release.
    """
    policy = _load_pinned_chunk_policy(policy_path, policy_sha256)
    revision = _chunk_tokenizer_revision(
        tokenizer_revision, "explicit structured chunker selection")
    return ChunkerSelection(
        "structured", policy.epoch, policy.rule_legend, str(policy.path),
        policy.sha256, revision)


def _announce_chunker(selection: ChunkerSelection) -> None:
    key = (selection.mode, selection.epoch, selection.policy_sha256,
           selection.tokenizer_revision)
    if _CHUNKER_ANNOUNCED["selection"] == key:
        return
    _CHUNKER_ANNOUNCED["selection"] = key
    if selection.mode != "legacy":
        import sys as _sys
        print(
            f"[aw_common] chunker={selection.mode} rule-epoch "
            f"chunk@{selection.epoch} policy_sha256={selection.policy_sha256} "
            f"tokenizer_revision={selection.tokenizer_revision} "
            f"— NOT the live 180w windower; records carry "
            f"stamps={{'chunk': {selection.epoch}}}",
            file=_sys.stderr, flush=True)


def chunker_mode() -> str:
    """Backward-compatible mode-only view of the verified selection."""
    selection = chunker_selection()
    _announce_chunker(selection)
    return selection.mode


def chunk_markdown_channels(
        md: str, selection: Optional[ChunkerSelection] = None
        ) -> list[ChunkTextChannels]:
    """Return canonical chunks with profile-bound retrieval text channels."""
    selection = selection or chunker_selection()
    if not isinstance(selection, ChunkerSelection):
        raise TypeError("selection must be a verified ChunkerSelection")
    _announce_chunker(selection)
    if selection.mode == "structured":
        from cytir.chunk import chunker as aw_chunk
        aw_chunk.select_tokenizer_revision(selection.tokenizer_revision)
        policy = aw_chunk.select_policy(
            selection.policy_path, allow_env_overrides=False)
        # task 2289: the funnel for every family stage; word-only chunks never ship.
        aw_chunk.require_exact_tokenizer("chunk_markdown_channels")
        if (policy.sha256 != selection.policy_sha256 or
                policy.epoch != selection.epoch or
                policy.rule_legend != selection.legend or
                not aw_chunk.policy_matches_effective(policy)):
            raise ValueError(
                "structured chunk policy changed between dispatcher "
                "verification and activation")
        # task 407: the stamp is minted from the SELECTION (which carries the
        # policy path + byte sha + tokenizer revision), not from the chunk
        # module's globals — one home. The module's own epoch is still
        # cross-checked, because a disagreement here means the activated
        # policy is not the verified one.
        if aw_chunk.stamps().get("chunk") != selection.epoch:
            raise ValueError(
                f"chunk module epoch {aw_chunk.stamps().get('chunk')!r} "
                f"disagrees with the verified selection {selection.epoch!r}")
        rule_stamps = selection.stamps()
        return [ChunkTextChannels(
            text=unit.text,
            dense_text=aw_chunk.dense_text(unit),
            sparse_text=unit.text,
            stamps=rule_stamps,
        ) for unit in aw_chunk.chunk_markdown(md)]
    from cytir.acquire import fetch_process as fp
    # task 407: legacy rows are stamped too. Leaving them unstamped is what
    # made "which chunker produced this row" unanswerable in the first place —
    # an absent stamp reads as "no information", not as "chunk@1". The legacy
    # chunker emits ONE surface, so dense_text/sparse_text stay None and
    # `record_text_channels` keeps its historical fallback to `text`; only the
    # additive `stamps` field is new.
    rule_stamps = selection.stamps()
    return [ChunkTextChannels(text=text, dense_text=None, sparse_text=None,
                              stamps=rule_stamps)
            for text in fp.chunk(md)]


def chunk_markdown_units(md: str) -> tuple[list[str], Optional[dict]]:
    """Backward-compatible text-only projection of the central dispatcher."""
    units = chunk_markdown_channels(md)
    # An empty document still has a producing rule; report the selection's
    # stamp rather than None, so "no chunks" and "no provenance" stay
    # distinguishable (an error is not an empty).
    stamps = units[0].stamps if units else chunker_selection().stamps()
    return [unit.text for unit in units], stamps
