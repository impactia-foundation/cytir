"""Guard 1 of task 710: a German pack must carry its de-compounded parts.

Why it exists. Three writers wrote 6.93M German BM25 vectors with NO parts (task 710): each ran
a Python without the splitter, and the old rule returns "no parts" and no error. This guard
looks at the rows a writer is ABOUT to write (or just wrote), asks an independent predictor
which rows must carry parts, and compares. The rule is `cytir.core.decompound` (PR #628): the guard calls `projection.saturated_bm25` as the
writer and reads the rule through `rule_module()`. After #628 a blind writer of this codebase
raises; the guard still catches rows written without parts by older code or other writers.

Thresholds, one place: `CARRY_SHARE_MIN` (text basis, the 99 % rule) and `LATTICE_CARRY_SHARE_MIN`.
One switch, one place: the operator setting `german_parts_guard` (env `AW_GERMAN_PARTS_GUARD`),
`warn` (default) or `refuse`. The value is checked once at drain start (`startup_refusal`).

What the 99 % text floor really is. It reads as a share, but in practice it is a binary "has the
writer a splitter" detector: with a splitter every predicted row carries parts (share 1.0), without
one none does (share 0.0). Nothing in between was ever measured on the text basis, so do not read
a 99 % limit as a quality bar for partial writers.

Who is the authority. For SEALED packs the DRAIN-side check (`check_joined`) is the authority: it
runs in the interpreter that writes the points. The compose-side check runs in the compose
interpreter (the project venv, which has a splitter) and so can read PASS for a pack that the
drain, under `/usr/bin/python3`, writes without parts (tranche 12 did exactly that). The compose
manifest block therefore names the interpreter that measured it (`measured_by`). For news
`write_pack` compose IS the writer, so its check is the authority there.

Guard errors never stop a drain. A guard that raises is a guard that stops the drain for every pack
behind it. `guarded` turns any exception into a receipt block `{"result": "error", "error": ...}`
and one loud line; `warn` goes on, `refuse` refuses only that pack.

Two bases, by what a pack holds:

* TEXT (sealed packs, stage/compose side): the row's sparse INPUT text and `lang`. A row is
  PREDICTED when one of its tokens (>= 12 chars) splits with a score >= 0.7 (the rule's own
  gate). It CARRIES parts when the writer's posting differs from the same text written with
  no parts (`saturated_bm25(text, lang)` against `saturated_bm25(text, "en")`).
* LATTICE (content packs, which hold the vector but not the text): a part adds exactly 0.5 to
  the tf, so a writer with parts leaves half-integer tf values in the saturated weights; a
  writer without never does (task 710 §5.1). The lattice assumes tf = 1 is the MODAL weight of a
  row (it reads the base weight off the most common value); a row where another tf dominates is
  mis-read, and `LATTICE_MIN_POSTINGS` leaves the short rows, where that happens, out.
  The row is PREDICTED from its title, site name and snippet (all part of the sparse input). Rows with < `LATTICE_MIN_POSTINGS` postings are
  undecidable on the lattice and are left out of both counts.

A pack is judged on German rows only (`lang == "de"`). A pack with no German rows is untouched.
A pack with rows that have no `lang` is COUNTED and reported, never refused for that.
A content pack whose title-predicted rows are mostly undecided (no `bm25` vector, too few
postings), or fewer than `LATTICE_MIN_PREDICTED`, is UNJUDGED: a WARN line in both modes, never
a refusal.
"""
from __future__ import annotations

import collections
import importlib
import random
from typing import Callable, Iterable

SCHEMA = "aw.german-parts-guard.v1"

#: THE threshold: the share of German rows with predicted parts that must carry them.
CARRY_SHARE_MIN = 0.99

#: The same limit on the LATTICE basis, which is noisier (task 710 section 5.1; measured on real
#: twin rows 2026-10-05: writers WITH parts read 85-100 % of predicted rows there, the three
#: writers WITHOUT read 0-0.1 %). Its floor is the tripwire's own alarm line, so one reading of
#: the lattice has one threshold. The text basis keeps `CARRY_SHARE_MIN`.
LATTICE_CARRY_SHARE_MIN = 0.8

#: A content pack is judged on the lattice only with at least this many predicted decidable rows.
#: Healthy packs read >= 97.6 % (n = 208, task 710); a writer that loses 40 % of the parts reads
#: about 0.59, which passed the old 0.5 floor. Below this count the pack is UNJUDGED (loud line,
#: never a refusal): a share of 3 rows is no evidence.
LATTICE_MIN_PREDICTED = 30

#: German rows judged per pack, and the characters of each row's sparse text that are judged.
#: A deterministic sample (seeded), because the share is a proportion: 1,000 rows give a 99 % limit
#: a resolution far finer than any real failure (a no-parts writer reads 0 to 1.6 %, task 710), and
#: the splitter costs ~10 ms per 400-word row (measured: 5,000 rows of 4,400 words took 13 min,
#: so the pack check must stay small enough to run in the hourly drain).
SAMPLE_MAX = 1000
TEXT_CHARS_MAX = 8000
SAMPLE_SEED = 710

#: The rule's own gate (`cytir.core.common._decompound`): tokens shorter than this never split.
MIN_TOKEN = 12

#: Lattice rows shorter than this are undecidable (the mode weight is not always tf = 1).
LATTICE_MIN_POSTINGS = 20

MODES = ("warn", "refuse")
DEFAULT_MODE = "warn"
SETTING = "german_parts_guard"

#: Used only when there is no external predictor: a German word that a sighted writer splits.
_CANARY = "Handelsregisterauszug"        # splits at 0.94 (Handelsregister + Auszug)


class GermanPartsRefused(RuntimeError):
    """A pack's German rows do not carry the parts the rule predicts (mode `refuse`)."""


def mode() -> str:
    """`warn` (default) or `refuse`, from the operator setting (env `AW_GERMAN_PARTS_GUARD`).
    Any other value raises: a typo must not read as a quiet default."""
    from cytir.core import paths
    value = str(getattr(paths.operator_settings(), SETTING)).strip().lower()
    if value not in MODES:
        raise ValueError(f"{SETTING}={value!r}: must be one of {MODES}")
    return value


def startup_refusal() -> int | None:
    """Check the mode ONCE at drain start (next to the config echo), so a typo refuses the run
    with one clear line instead of raising on the first pack. Returns 2 (and names it on stderr)
    or None."""
    import sys
    try:
        mode()
    except Exception as exc:  # noqa: BLE001 -- a bad value, or any read failure of the setting
        print(f"[REFUSED] promote_queue: setting {SETTING}: {exc}", file=sys.stderr, flush=True)
        return 2
    return None


def guarded(check: Callable, pack_id: str, *args, out=print, **kwargs) -> tuple[dict, str | None]:
    """Run `check(*args, pack_id, ...)` so that it can never stop the caller. On any Exception:
    the report is `{"result": "error", "error": "<type>: <msg>"}`, one loud line is printed, and
    the reason is None in `warn` (the pack goes on) or a refusal of THIS pack in `refuse`. A mode
    that cannot be read counts as `refuse` (nothing says the operator wanted `warn`)."""
    try:
        return check(*args, pack_id, out=out, **kwargs)
    except Exception as exc:  # noqa: BLE001 -- the whole point: a guard error is data, not a stop
        err = f"{type(exc).__name__}: {exc}"
        try:
            refuse = mode() == "refuse"
        except Exception:  # noqa: BLE001
            refuse = True
        out(f"[german_parts] ERROR {pack_id}: the guard raised {err}; "
            + ("mode=refuse refuses this pack" if refuse else "mode=warn writes the pack anyway"))
        return ({"schema": SCHEMA, "result": "error", "error": err},
                f"german_parts: guard error {err}" if refuse else None)


# ------------------------------------------------------------------ the predictor
def _token_re():
    from cytir.core import common
    from cytir.core import bm25_tokenizer as _tokenize
    return _tokenize._TOKEN_RE


def rule_module():
    """The seam to the rule: the module `cytir.core.decompound` (ADR-121). The guard reads the rule
    through this one function, so a test simulates "no splitter" by patching it or the module's
    loader, never a private table of `cytir.core.common`."""
    return importlib.import_module("cytir.core.decompound")


def external_predictor() -> tuple[str, Callable[[str], tuple]] | None:
    """The rule's own splitter as predictor, or None when the rule cannot load.

    After PR #628 there is one splitter, so the predictor is no longer independent of the
    writer's code. It is still useful: it tells which rows MUST carry parts, and the per-row
    reading then catches rows written by OLDER code or other writers. A rule that cannot load
    (`DecompoundUnavailable`, which a writer in this codebase raises at its first German token)
    gives None: the caller then judges by the writer itself (`writer_is_blind`)."""
    try:
        mod = rule_module()
        mod.ensure_ready()
        return "cytir.core.decompound", mod.decompound_parts
    except Exception:  # noqa: BLE001 -- a rule that cannot load is "no predictor", never a raise
        return None


def _default_writer(text: str, lang):
    from cytir.core import projection
    return projection.saturated_bm25(text, lang)


def _posting(writer, text: str, lang) -> dict:
    idx, vals = writer(text, lang)
    return dict(zip(idx, vals))


def predicts_parts(text: str, parts_fn: Callable[[str], tuple]) -> bool:
    for tok in set(_token_re().findall(text.lower())):
        if len(tok) >= MIN_TOKEN and parts_fn(tok):
            return True
    return False


def writer_is_blind(writer=None) -> bool:
    """True when the writer, asked for German, cannot write parts: it raises (the rule module
    cannot load, so a writer of this codebase fails at the first German token) or writes the same
    posting as for a non-German row on a word that always splits (older code or another writer
    without a splitter). Only used with no predictor."""
    writer = writer or _default_writer
    try:
        return _posting(writer, _CANARY, "de") == _posting(writer, _CANARY, "en")
    except Exception:  # noqa: BLE001 -- a writer that raises on German is no sighted writer
        return True


# ------------------------------------------------------------------ the lattice
def lattice(values) -> tuple[int, int] | None:
    """(postings, half-integer-tf postings) of a saturated bm25 vector, from the weights alone
    (task 710 §5.1): `w = 2.2*tf/(tf + c)`, the modal weight is tf = 1. None when undecidable."""
    if not values:
        return None
    cnt = collections.Counter(round(v, 4) for v in values)
    w1 = cnt.most_common(1)[0][0]
    if not (0 < w1 < 2.2):
        return None
    c = 2.2 / w1 - 1
    half = 0
    for v in values:
        if v >= 2.2:
            continue
        t = v * c / (2.2 - v)
        k = round(2 * t)
        if abs(2 * t - k) < 0.06 * max(1.0, t / 4) and k % 2 == 1:
            half += 1
    return len(values), half


def bm25_of(point: dict) -> dict:
    """The `bm25` sparse vector of a point as `{indices, values}`, else `{}`. A point whose
    `vector` is not a mapping (a bare dense list, as a point from a single-vector collection
    carries) has no named bm25 vector: it is read as a vector with none, so undecided."""
    vec = point.get("vector")
    if not isinstance(vec, dict):
        return {}
    bm = vec.get("bm25")
    return bm if isinstance(bm, dict) else {}


# ------------------------------------------------------------------ sampling
def sample_german(rows: Iterable, lang_of: Callable, *, n: int = SAMPLE_MAX,
                  seed: int = SAMPLE_SEED) -> tuple[list, dict]:
    """A seeded reservoir of up to `n` German rows, and the pack's counts:
    `rows`, `german_rows`, `no_lang_rows`. A row is German when `lang_of(row) == "de"`;
    a row with no `lang` is counted apart and never judged."""
    rng = random.Random(seed)
    sample: list = []
    counts = {"rows": 0, "german_rows": 0, "no_lang_rows": 0}
    for row in rows:
        counts["rows"] += 1
        lang = lang_of(row)
        if lang in (None, ""):
            counts["no_lang_rows"] += 1
        elif lang == "de":
            counts["german_rows"] += 1
            if len(sample) < n:
                sample.append(row)
            else:
                j = rng.randrange(counts["german_rows"])
                if j < n:
                    sample[j] = row
    return sample, counts


# ------------------------------------------------------------------ the verdict
def _report(counts: dict, *, basis: str, predictor: str, sampled: int, predicted: int,
            carried: int, undecided: int = 0, note: str = "") -> dict:
    share = (carried / predicted) if predicted else None
    floor = LATTICE_CARRY_SHARE_MIN if basis == "lattice" else CARRY_SHARE_MIN
    if predictor == "none" and basis == "lattice":
        verdict = "NA"
    elif basis == "lattice" and (predicted + undecided) and (
            undecided > predicted or predicted < LATTICE_MIN_PREDICTED):
        # undecided rows dominate, or too few rows to judge: say so, never PASS silently
        verdict = "UNJUDGED"
        note = note or (f"{predicted} decidable and {undecided} undecided title-predicted rows "
                        f"(needs >= {LATTICE_MIN_PREDICTED} decidable, and no more undecided than "
                        "decidable)")
    elif share is None:
        verdict = "PASS"            # no German row with predicted parts: nothing to carry
    else:
        verdict = "PASS" if share >= floor else "FAIL"
    return {"schema": SCHEMA, **counts, "basis": basis, "predictor": predictor,
            "threshold": floor, "sampled": sampled, "predicted": predicted,
            "carried": carried, "undecided": undecided,
            "share": None if share is None else round(share, 4), "verdict": verdict,
            **({"note": note} if note else {})}


def evaluate_texts(texts: list[str], counts: dict, *, predictor=None, writer=None) -> dict:
    """TEXT basis. `texts` = the sparse input of the sampled German rows."""
    # a row with no sparse text was not predicted: nothing to compare it with (never a raise)
    no_text = sum(1 for t in texts if not isinstance(t, str))
    texts = [t for t in texts if isinstance(t, str)]
    if no_text:
        counts = {**counts, "no_text_rows": no_text}
    if not texts:      # no German row: nothing to resolve (and a receipt that does not depend on the box)
        return _report(counts, basis="text", predictor="-", sampled=0, predicted=0, carried=0)
    writer = writer or _default_writer
    texts = [t[:TEXT_CHARS_MAX] for t in texts]
    ext = predictor if predictor is not None else external_predictor()
    if ext is None:
        # no independent splitter: judge by the writer itself. A blind writer cannot carry
        # parts for ANY row with a long token, so those rows count as predicted and uncarried.
        blind = writer_is_blind(writer)
        kind, parts_fn = ("none-writer-blind" if blind else "none-writer-sighted"), None
    else:
        kind, parts_fn = ext
    sampled = predicted = carried = 0
    for text in texts:
        sampled += 1
        if parts_fn is not None:
            if not predicts_parts(text, parts_fn):
                continue
        elif kind == "none-writer-blind":
            if not any(len(t) >= MIN_TOKEN for t in _token_re().findall(text.lower())):
                continue
        else:
            continue            # sighted writer, nothing independent to compare with
        predicted += 1
        try:
            has = _posting(writer, text, "de") != _posting(writer, text, "en")
        except Exception:  # noqa: BLE001 -- a writer that raises is a writer without parts
            has = False
        carried += has
    if kind == "none-writer-sighted":
        return _report(counts, basis="text", predictor=kind, sampled=sampled, predicted=0,
                       carried=0, note="no independent splitter; the writer splits on its own")
    return _report(counts, basis="text", predictor=kind, sampled=sampled, predicted=predicted,
                   carried=carried)


def evaluate_points(points: Iterable[dict], *, predictor=None, n: int = SAMPLE_MAX) -> dict:
    """LATTICE basis, for content packs: `points` = `{id, payload, vector: {bm25: {indices,
    values}}}`. Needs a predictor (title, site name and snippet are the text there is)."""
    sample, counts = sample_german(points, lambda p: (p.get("payload") or {}).get("lang"), n=n)
    if not sample:
        return _report(counts, basis="lattice", predictor="-", sampled=0, predicted=0, carried=0)
    ext = predictor if predictor is not None else external_predictor()
    if ext is None:
        return _report(counts, basis="lattice", predictor="none", sampled=len(sample),
                       predicted=0, carried=0,
                       note="no splitter in this interpreter: cannot predict parts from a vector")
    kind, parts_fn = ext
    predicted = carried = undecided = 0
    for p in sample:
        pl = p.get("payload") or {}
        text = " ".join(str(pl.get(k) or "") for k in ("title", "site_name", "snippet"))
        if not predicts_parts(text, parts_fn):
            continue
        lat = lattice(bm25_of(p).get("values") or [])
        if lat is None or lat[0] < LATTICE_MIN_POSTINGS:
            undecided += 1
            continue
        predicted += 1
        carried += lat[1] > 0
    return _report(counts, basis="lattice", predictor=kind, sampled=len(sample),
                   predicted=predicted, carried=carried, undecided=undecided)


# ------------------------------------------------------------------ the policy
def decide(report: dict, pack_id: str, *, guard_mode: str | None = None) -> tuple[str | None, list[str]]:
    """(refusal reason or None, loud lines). The only place a verdict becomes a refusal."""
    guard_mode = guard_mode or mode()
    lines = []
    if report["no_lang_rows"]:
        lines.append(
            f"[german_parts] {pack_id}: {report['no_lang_rows']} of {report['rows']} rows have no "
            "`lang` field; counted, not judged"
            + (" (the whole pack has none: nothing here can be checked)"
               if report["no_lang_rows"] == report["rows"] else ""))
    if report["verdict"] == "NA" and report["german_rows"]:
        lines.append(f"[german_parts] {pack_id}: NOT CHECKED, {report.get('note')} "
                     f"({report['german_rows']} German rows)")
    if report["verdict"] == "UNJUDGED":
        lines.append(f"[german_parts] WARN {pack_id}: NOT JUDGED, {report.get('note')} "
                     f"({report['german_rows']} German rows, basis {report['basis']})")
    if report["verdict"] in ("NA", "UNJUDGED"):
        return None, lines
    if report["verdict"] == "FAIL":
        what = (f"only {report['carried']} of {report['predicted']} German rows with predicted "
                f"parts carry them ({report['share']:.1%}, limit {report['threshold']:.0%}; "
                f"{report['german_rows']} German rows, {report['sampled']} sampled, "
                f"basis {report['basis']}, predictor {report['predictor']})")
        if guard_mode == "refuse":
            lines.append(f"[german_parts] REFUSE {pack_id}: {what}")
            return f"german_parts: {what}", lines
        lines.append(f"[german_parts] WARN {pack_id}: {what}; mode=warn writes the pack anyway, "
                     f"{SETTING}=refuse would refuse it")
    elif report["verdict"] == "PASS" and report["german_rows"]:
        # One line per judged German pack, so the journal shows the check RAN (task 732: until then a
        # PASS printed nothing, and "no PASS line ever" read as "the check never ran").
        share = "-" if report["share"] is None else f"{report['share']:.1%}"
        lines.append(f"[german_parts] PASS {pack_id}: {report['carried']} of {report['predicted']} German "
                     f"rows with predicted parts carry them ({share}); {report['german_rows']} German rows, "
                     f"{report['sampled']} sampled, basis {report['basis']}, predictor {report['predictor']}")
    return None, lines



# ------------------------------------------------------------------ the call sites
def check_joined(joined: dict, pack_id: str, *, guard_mode: str | None = None, out=print,
                 predictor=None, writer=None) -> tuple[dict, str | None]:
    """The drain's sealed-pack call: `joined` = `{point_id: (dense_text, sparse_text, payload,
    rec)}` from `plan()`. Returns (report for the receipt, refusal reason or None)."""
    sample, counts = sample_german(joined.values(), lambda v: (v[2] or {}).get("lang"))
    report = evaluate_texts([v[1] for v in sample], counts, predictor=predictor, writer=writer)
    reason, lines = decide(report, pack_id, guard_mode=guard_mode)
    for line in lines:
        out(line)
    return report, reason


def check_points(points: list[dict], pack_id: str, *, guard_mode: str | None = None, out=print,
                 predictor=None) -> tuple[dict, str | None]:
    """The drain's content-pack call: the pack's own points (vector, no text)."""
    report = evaluate_points(points, predictor=predictor)
    reason, lines = decide(report, pack_id, guard_mode=guard_mode)
    for line in lines:
        out(line)
    return report, reason
