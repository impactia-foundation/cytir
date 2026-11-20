"""Shared website-content contract for source-neutral entity cards."""
from __future__ import annotations

import datetime as dt
import hashlib
import os
import re
import urllib.parse

import cytir.core.common as AC
from cytir.extract import html_extract as _html_extract
from cytir.consent import policy as CP


# --- Website retrieval-projection budget (ONE HOME) -------------------------
# Task 388 ratified a 1,024-token TOTAL website-content budget (2026-08-06);
# task 393 found the budget had been ratified but never applied, because the
# capture stage (card_website_capture.normalized_surface) still
# clamped at 900 characters and hashed that clamped surface into the receipt.
# Result: the full 9,008-row capture retained only 26.9% of available homepage
# words while nominally being "at the 1,024 budget" (task 393 §2).
#
# The failure mode was that the *capture* cap and the *composition* cap were
# two independent literals. They now resolve here, in the single module both
# sides already import, so a future budget change cannot apply to one side
# only. Overridable for pilots; never re-declare these numbers at a call site.
WEBSITE_TOKEN_BUDGET = int(os.environ.get("AW_WEBSITE_TOKEN_BUDGET", "1024"))
# The composed dense surface ceiling is chunk policy's `token_hard_max`
# (exact=2000, count_special_tokens=true) — asserted before embedding, not
# assumed. Kept here so the website budget is always chosen against it.
WEBSITE_DENSE_TOKEN_CEILING = int(
    os.environ.get("AW_WEBSITE_DENSE_TOKEN_CEILING", "2000"))
# Token measurement is a mirror contract with the embedding worker, so the
# tokenizer snapshot is pinned rather than tracking a mutable branch. This is
# the revision carried by the live arctic profile (data/build_config.json
# embed_model) and used by tasks 374/375/381/389.
WEBSITE_TOKENIZER_REVISION = os.environ.get(
    "AW_WEBSITE_TOKENIZER_REVISION",
    "ac6544c8a46e00af67e330e85a9028c66b8cfd9a")

WEBSITE_RECEIPT_SCHEMA = CP.WEBSITE_RECEIPT_SCHEMA
ASSOCIATION_RECEIPT_SCHEMA = "aw.entity-card.site-association.v1"
LICENSE_EVIDENCE_SCHEMA = "aw.entity-card.website-license-evidence.v1"
ACQUISITION_RECEIPT_SCHEMA = "aw.entity-card.website-acquisition.v1"
UNIDENTIFIED_LICENSE_MARKERS = (
    "noassertion", "unverified", "unknown", "no license",
)
RESTRICTIVE_LICENSE_MARKERS = (
    "all rights reserved", "all-rights-reserved", "proprietary-index-only",
    "proprietary", "copyright",
)


def normalized_content(value: str, cap: int) -> str:
    """Normalize and bound one master-side source surface before hashing.

    SUPERSEDED for website surfaces by :func:`normalized_content_tokens` — a
    character cap is not the ratified unit and silently under-delivers a token
    budget (task 393). Retained only for non-website callers that genuinely
    bound by characters.
    """
    text = " ".join((value or "").split())
    return text[:cap].rsplit(" ", 1)[0] if len(text) > cap else text


def _website_tokenizer():
    """Return the revision-pinned Arctic tokenizer, or fail loud.

    Never falls back to a character or whitespace-word approximation: that
    substitution is exactly how the ratified budget silently became a 900-char
    clamp (task 393). A missing tokenizer must stop the build, not quietly
    change the unit of the budget.
    """
    from cytir.chunk import chunker as chunk  # local: keeps the HF/transformers import off the hot path

    chunk.select_tokenizer_revision(WEBSITE_TOKENIZER_REVISION)
    tokenizer = chunk._get_tokenizer()
    if tokenizer is None:
        raise RuntimeError(
            "website token budget requires the pinned Arctic tokenizer "
            f"({chunk.TOKENIZER_MODEL} @ {WEBSITE_TOKENIZER_REVISION}); "
            "it is unavailable — refusing to approximate the budget")
    return tokenizer


def website_token_count(value: str) -> int:
    """Exact content-token count of one website surface (no special tokens)."""
    tokenizer = _website_tokenizer()
    return len(tokenizer(" ".join((value or "").split()),
                         add_special_tokens=False)["input_ids"])


def normalized_content_tokens(value: str, budget: int | None = None) -> str:
    """Normalize and bound a website surface to an exact Arctic-token budget.

    ``budget`` is the *remaining* total website budget, so a caller composing
    homepage + depth-1 pages spends one shared allowance across sources rather
    than granting each source its own cap.
    """
    limit = WEBSITE_TOKEN_BUDGET if budget is None else budget
    text = " ".join((value or "").split())
    if limit <= 0 or not text:
        return ""
    tokenizer = _website_tokenizer()
    # Must be IDEMPOTENT: this function both mints a receipt's content hash at
    # recomposition and re-derives it at composition, so f(f(x)) != f(x) shows
    # up as "content_sha256 does not match website_content" (observed on the
    # first task-393 projection run). decode(encode(x)[:n]) can re-tokenize to
    # MORE than n tokens, so truncate to a fixed point instead of once.
    for _ in range(8):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= limit:
            return text
        cut = tokenizer.decode(ids[:limit], skip_special_tokens=True).strip()
        cut = " ".join(cut.split())
        if not cut or cut == text:
            break
        text = cut
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) > limit:
        raise RuntimeError(
            "website token truncation did not converge; refusing to emit a "
            "surface whose hash would not reproduce")
    return text


def license_is_identified(value: str) -> bool:
    normalized = " ".join((value or "").lower().replace("_", " ").split())
    return bool(normalized) and not any(
        marker in normalized for marker in UNIDENTIFIED_LICENSE_MARKERS)


def license_is_restrictive(value: str) -> bool:
    normalized = " ".join((value or "").lower().replace("_", " ").split())
    return any(marker in normalized for marker in RESTRICTIVE_LICENSE_MARKERS)


def _domain(url: str) -> str:
    return _html_extract.reg_domain((urllib.parse.urlsplit(url).hostname or "").lower())


def _iso_time(value: str, field: str) -> None:
    raw = str(value or "").strip().replace("Z", "+00:00")
    try:
        dt.datetime.fromisoformat(raw)
    except ValueError:
        try:
            dt.date.fromisoformat(raw)
        except ValueError as exc:
            raise ValueError(f"{field} is not an ISO date/time") from exc


def entity_key(seed: dict) -> str:
    value = seed.get("register_id") or seed.get("qid")
    if value:
        return "".join(character for character in str(value).upper()
                       if character.isalnum())
    value = seed.get("domain")
    if not value:
        raise ValueError("website content seed lacks entity identity")
    return _html_extract.reg_domain(str(value).lower())


def validate_association_receipt(seed: dict, receipt: dict, *,
                                 require_admitted: bool = True) -> dict:
    """Validate evidence-graded entity↔site association, never claim ownership."""
    required = (
        "entity", "domain", "status", "source", "observed_at",
        "evidence_sha256", "source_artifact_sha256", "artifact_pointer",
        "resolve_confidence", "resolve_method", "evidence_class",
    )
    if (not isinstance(receipt, dict)
            or receipt.get("schema") != ASSOCIATION_RECEIPT_SCHEMA
            or any(receipt.get(key) in (None, "") for key in required)):
        raise ValueError("website content requires a complete site-association receipt")
    status = receipt["status"]
    if status not in {"admitted", "held"}:
        raise ValueError("unknown site-association evidence grade")
    if require_admitted and status != "admitted":
        raise ValueError("site-association evidence grade is held")
    classes = {"uid_on_page", "slug_live_name_domain_address", "cc_heuristic"}
    if receipt["evidence_class"] not in classes:
        raise ValueError("unknown site-association evidence class")
    if status == "admitted" and receipt["evidence_class"] == "cc_heuristic":
        raise ValueError("CC-only heuristic site association cannot be admitted")
    _iso_time(receipt["observed_at"], "site-association observed_at")
    if _html_extract.reg_domain(str(receipt["domain"]).lower()) != _domain(seed.get("url") or ""):
        raise ValueError("site-association receipt does not match the card domain")
    observed_entity = entity_key({"register_id": receipt["entity"]})
    if observed_entity != entity_key(seed):
        raise ValueError("site-association receipt does not match the card entity")
    if not all(re.fullmatch(r"[0-9a-fA-F]{64}", str(receipt[key]))
               for key in ("evidence_sha256", "source_artifact_sha256")):
        raise ValueError("site-association evidence digests are not SHA-256")
    try:
        confidence = float(receipt["resolve_confidence"])
    except (TypeError, ValueError) as exc:
        raise ValueError("site-association resolve_confidence is invalid") from exc
    if not 0.0 < confidence <= 1.0:
        raise ValueError("site-association resolve_confidence is outside (0,1]")
    return {"schema": ASSOCIATION_RECEIPT_SCHEMA,
            **{key: receipt[key] for key in required}}


def validate_license_evidence(source_url: str, license_value: str,
                              evidence: dict) -> dict:
    required = ("basis", "source_url", "checked_at")
    if (not isinstance(evidence, dict)
            or evidence.get("schema") != LICENSE_EVIDENCE_SCHEMA
            or any(evidence.get(key) in (None, "") for key in required)):
        raise ValueError("website license/status lacks evidence")
    if _domain(evidence["source_url"]) != _domain(source_url):
        raise ValueError("website license evidence crosses source domain")
    if evidence["basis"] == "default-copyright":
        if not license_is_restrictive(license_value):
            raise ValueError("default copyright cannot assert a permissive license")
    elif evidence["basis"] == "reviewed-license":
        for key in ("evidence_sha256", "artifact_pointer"):
            if evidence.get(key) in (None, ""):
                raise ValueError("reviewed website license evidence is incomplete")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", str(evidence["evidence_sha256"])):
            raise ValueError("website license evidence digest is not SHA-256")
    else:
        raise ValueError("unknown website license evidence basis")
    _iso_time(evidence["checked_at"], "website license checked_at")
    return dict(evidence)


def validate_acquisition_receipt(source_url: str, fetched_at: str,
                                 consent_mode: str, receipt: dict) -> dict:
    """Bind content to either a deterministic CC locator or direct fallback."""
    required = ("route", "url", "retrieved_at")
    if (not isinstance(receipt, dict)
            or receipt.get("schema") != ACQUISITION_RECEIPT_SCHEMA
            or any(receipt.get(key) in (None, "") for key in required)):
        raise ValueError("website acquisition receipt is incomplete")
    if _domain(receipt["url"]) != _domain(source_url):
        raise ValueError("website acquisition receipt crosses source domain")
    route = receipt["route"]
    _iso_time(receipt["retrieved_at"], "website acquisition retrieved_at")
    if route in {"common-crawl-s3-warc-range",
                 "common-crawl-https-warc-range"}:
        if consent_mode != "cc":
            raise ValueError("Common Crawl capture requires CC-mode consent")
        cc_required = (
            "crawl", "fetch_time", "warc_filename", "offset", "length",
            "content_digest", "content_truncated", "index_manifest_sha256",
            "selection_version",
        )
        if any(receipt.get(key) in (None, "") for key in cc_required):
            raise ValueError("Common Crawl acquisition locator is incomplete")
        if receipt["fetch_time"] != fetched_at:
            raise ValueError("Common Crawl capture time disagrees with fetched_at")
        if receipt["content_truncated"] is not False:
            raise ValueError("truncated Common Crawl capture is not admissible")
        if not re.fullmatch(r"[0-9a-fA-F]{64}",
                            str(receipt["index_manifest_sha256"])):
            raise ValueError("Common Crawl index manifest digest is not SHA-256")
        try:
            offset, length = int(receipt["offset"]), int(receipt["length"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Common Crawl byte-range locator is invalid") from exc
        if offset < 0 or length <= 0:
            raise ValueError("Common Crawl byte-range locator is invalid")
    elif route == "direct-crawl-fallback":
        if consent_mode != "direct":
            raise ValueError("direct fallback requires direct-mode consent")
        for key in ("fetch_time", "user_agent", "robots_checked_at"):
            if receipt.get(key) in (None, ""):
                raise ValueError("direct-crawl fallback receipt is incomplete")
        if receipt["fetch_time"] != fetched_at:
            raise ValueError("direct-crawl fetch time disagrees with fetched_at")
    else:
        raise ValueError("unknown website acquisition route")
    return dict(receipt)


def validate_receipt(seed: dict, text: str, receipt: dict) -> dict:
    """Validate and return the exact content-free master receipt."""
    required = (
        "source_url", "status", "search_ok", "train_ok", "consent_mode", "license",
        "attribution", "checked_at", "fetched_at", "content_sha256",
        "raw_sha256", "raw_artifact_pointer",
        "association_receipt", "license_evidence", "acquisition_receipt",
    )
    if not isinstance(receipt, dict) or receipt.get("schema") != WEBSITE_RECEIPT_SCHEMA:
        raise ValueError("website_receipt is incomplete or has an unknown schema")
    if any(key not in receipt or receipt[key] in (None, "") for key in required):
        raise ValueError("website_receipt is incomplete or has an unknown schema")
    CP.require_resolved_search_index_receipt(receipt)
    _iso_time(receipt["checked_at"], "website consent checked_at")
    _iso_time(receipt["fetched_at"], "website fetched_at")
    association = validate_association_receipt(seed, receipt["association_receipt"])
    if not license_is_identified(str(receipt["license"])):
        raise ValueError("website_content requires an identified license/copyright status")
    if license_is_restrictive(str(receipt["license"])) and receipt["train_ok"] is not False:
        raise ValueError("restrictive website content must retain train_ok=false")
    if receipt["content_sha256"] != hashlib.sha256(text.encode("utf-8")).hexdigest():
        raise ValueError("website_receipt content_sha256 does not match website_content")
    if not all(isinstance(receipt[key], str)
               and re.fullmatch(r"[0-9a-fA-F]{64}", receipt[key])
               for key in ("content_sha256", "raw_sha256")):
        raise ValueError("website_receipt hashes must be SHA-256 hex digests")
    if (_domain(receipt["source_url"]) != _domain(seed.get("url") or "")
            or not _domain(seed.get("url") or "")):
        raise ValueError("website_receipt source_url must belong to the card domain")
    license_evidence = validate_license_evidence(
        receipt["source_url"], str(receipt["license"]), receipt["license_evidence"])
    acquisition = validate_acquisition_receipt(
        receipt["source_url"], receipt["fetched_at"], receipt["consent_mode"],
        receipt["acquisition_receipt"])
    return {"schema": WEBSITE_RECEIPT_SCHEMA,
            **{key: (association if key == "association_receipt" else
                     license_evidence if key == "license_evidence" else
                     acquisition if key == "acquisition_receipt" else receipt[key])
               for key in required}}
