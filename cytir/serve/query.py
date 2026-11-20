"""Query-side analysis that is a pure function of the query text (S10 slice 1, task 752; moved
verbatim from `retrieval.py`, which imports it back).

Pure means: no environment read at call time, no I/O. Step 1 of task 1061 added the
commercial-intent test, which uses `textmatch._content_terms`. Step 3 (task 1083) adds the
person-name and required-word query side; it reads two frozen constants from `settings`
(`_NAMEGATE_MAXTERMS`, `_NAMEGATE_SHAPE_V2`), exactly as `retrieval.py` did.
"""
from __future__ import annotations

import re

from .schema import bm25_token_groups
from .settings import SETTINGS as _S
from .textmatch import _ALPHA_RE, _STOPWORDS, _WORD, _card_aliases, _content_terms


# Academic intent (task 089): the query side of the academic down-weight in
# `signals._academic_penalty`. Measured on labelled eval queries: recall 0.92,
# precision 0.96 (task 089).
# explicit publication vocabulary — a query using these words wants a paper
_AC_WORDS = frozenset({
    "paper", "papers", "study", "studies", "preprint", "arxiv", "pubmed",
    "pmc", "doi", "thesis", "dissertation", "proceedings", "journal",
    "citation", "seminal", "infoscience",
})
_AC_PHRASES = ("et al", "open access", "research collection", "open problems",
               "original paper")
# distinctive research method/model/bio/physics terms that only appear in
# paper-seeking queries (NOT general encyclopedic topics like "transformer
# wikipedia" — 'wikipedia' is an explicit non-academic override below)
_AC_PAT = re.compile(
    r"\b("
    r"p53|cas9|sars-cov-2|ace2|toll-like|angiotensin|aldosterone|renin|"
    r"higgs boson|gravitational waves|ricci flow|"
    r"resnet|word2vec|instructgpt|roberta|efficientnet|chinchilla|"
    r"ddpm|dqn|vae|gan|ppo|yolo|gru|zfnet|"
    r"vision transformer|gpt-3|few-shot learners|low-rank adaptation|"
    r"chain-of-thought|selective state spaces|compute-optimal|"
    r"co-adaptation|weak supervision|jointly align translate|"
    r"transferable visual models|instance segmentation|promptable segmentation|"
    r"sequence alignment map|internal covariate shift|denoising diffusion|"
    r"adversarial networks|residual learning|policy optimization"
    r")\b")


def _query_academic_intent(q: str) -> bool:
    """Cheap heuristic: does the query want an academic paper/study? True keeps
    arxiv/pubmed at full weight; False lets the academic penalty demote them."""
    ql = (q or "").lower()
    if "wikipedia" in ql:            # explicit encyclopedia intent, not a paper
        return False
    toks = set(re.findall(r"[a-z0-9-]+", ql))
    if toks & _AC_WORDS:
        return True
    if any(p in ql for p in _AC_PHRASES):
        return True
    return bool(_AC_PAT.search(ql))


# Identifier detection (task 170): an identifier-DOMINANT query (UID, LEI, SIRET,
# SIREN, EU VAT; a trailing scheme word allowed) for the exact payload lookup.
# The fast-path switch `AW_ID_FASTPATH` and the lookup stay in `retrieval`.
_ID_SCHEME_SUFFIX = re.compile(
    r"\s*\b(mwst-?nr|mwst|tva|iva|vat|uid|ide|idi|hr|nr)\b\.?\s*$", re.I)
_ID_UID_RE = re.compile(r"^che[-\s]?(\d{3})[.\s]?(\d{3})[.\s]?(\d{3})$", re.I)
_ID_LEI_RE = re.compile(r"^([A-Z0-9]{18}\d{2})$", re.I)
_ID_SIRET_RE = re.compile(r"^(\d{14})$")
_ID_SIREN_RE = re.compile(r"^(\d{9})$")
_ID_VAT_RE = re.compile(
    r"^((?:AT|BE|BG|HR|CY|CZ|DE|DK|EE|EL|ES|FI|FR|HU|IE|IT|LT|LU|LV|MT|NL|PL"
    r"|PT|RO|SE|SI|SK)[0-9A-Z]{8,12})$", re.I)


def _detect_identifier(q: str):
    """(payload_field, canonical_value) for an identifier-DOMINANT query, else
    None. Order (UID, LEI-20, SIRET-14, SIREN-9, VAT) avoids length collisions."""
    s = _ID_SCHEME_SUFFIX.sub("", (q or "").strip()).strip()
    m = _ID_UID_RE.match(s)
    if m:
        return "register_id", f"CHE-{m.group(1)}.{m.group(2)}.{m.group(3)}"
    m = _ID_LEI_RE.match(s)
    if m:
        return "lei", m.group(1).upper()
    for rx in (_ID_SIRET_RE, _ID_SIREN_RE):
        m = rx.match(s)
        if m:
            return "register_id", m.group(1)
    m = _ID_VAT_RE.match(s)
    if m:
        return "register_id", m.group(1).upper()
    return None


# --------------------------------------------------------------------------- #
#  COMMERCIAL / PRODUCT INTENT (task 254) — intent-conditioned down-weight of
#  reference (Wikipedia) + government/regulator trust so a commercial/product
#  lookup ("cash loan", "car leasing", "Barkredit", "Privatkredit", "credit
#  card") surfaces actual PROVIDERS / entity-cards instead of the encyclopedia
#  article or the regulator. Complements the ADR-036 intent trust-head: the head
#  gates the *general* trust-over-relevance flood, but on a commercial query the
#  genuinely-authoritative grounding source is the OFFER/PROVIDER, not the
#  finma/seco/europa.eu regulatory page — a signal the head does not own (it
#  still keeps gov ON TOP for these, measured live 2026-07-17: "personal loan
#  switzerland" -> seco/kmu.admin.ch flood; "cash loan" -> Wikipedia x9;
#  "Barkredit" -> finma + Wikipedia). Cheap multilingual lexicon (EN/DE/FR/IT),
#  NO model, in the hot path a handful of substring/token tests.
#
#  GUARDED against LEGAL / STATUTE / REGULATORY intent (Konsumkredit*gesetz*,
#  "consumer credit *act*", "Art. X CO", anything naming a statute / supervisor /
#  regulation) so authority lookups KEEP gov/reference on top — the whole point
#  is intent-conditioning, exactly like ADR-036. The lexicon is deliberately
#  scoped to FINANCIAL PRODUCTS + leasing + explicit shopping; bare "price" /
#  "buy" / "acheter" are EXCLUDED (they collide with statistic gold "consumer
#  price index" and gov-service gold "acheter une voiture d'occasion" ->
#  carte-grise). Flag-gated AW_COMMERCIAL (default OFF, byte-identical); the
#  down-weights are weights (commercial_ref <= 0 additive on reference;
#  commercial_gov_scale in [0,1] multiplies the gov trust term). rollback =
#  AW_COMMERCIAL=0 or commercial_gov_scale=1.0 + commercial_ref=0.0.
# --------------------------------------------------------------------------- #
_COMM_PHRASES = (
    "cash loan", "personal loan", "payday loan", "car loan", "auto loan",
    "student loan", "home loan", "quick loan", "online loan", "instant loan",
    "small loan", "best loan", "best mortgage", "car leasing", "auto leasing",
    "vehicle leasing", "leasing auto", "leasing voiture", "credit card",
    "carte de crédit", "carte de credit", "carta di credito", "consumer credit",
    "crédit à la consommation", "crédit conso", "credito al consumo",
    "geld leihen", "geld borgen", "kredit aufnehmen", "borrow money",
    "prêt personnel", "pret personnel", "crédit privé", "credit prive",
    "prestito personale", "prestito online", "online shop", "buy online",
    "for sale", "best deal", "cheapest",
)
_COMM_TOKENS = frozenset({
    "loan", "loans", "leasing", "mortgage", "hypothek", "hypotheken",
    "hypothèque", "ipoteca", "darlehen", "prestito", "prestiti", "kredit",
    "barkredit", "privatkredit", "kleinkredit", "sofortkredit", "konsumkredit",
    "ratenkredit", "autokredit", "onlinekredit", "kreditkarte", "kreditkarten",
})
_COMM_SUBSTR = re.compile(
    r"(barkredit|privatkredit|kleinkredit|sofortkredit|konsumkredit|ratenkredit"
    r"|autokredit|onlinekredit|kreditkarte|autoleasing)", re.I)
# LEGAL / regulatory markers -> NOT commercial (authority keeps gov/reference).
_COMM_LEGAL_GUARD = re.compile(
    r"(?<!\w)(art|artikel|§|act|law|gesetz|verordnung|obligationenrecht|zgb|stgb"
    r"|loi|code|directive|richtlinie|règlement|reglement|regulation|statute"
    r"|ordinance|urteil|jugement|rechtsprechung|aufsicht|supervis"
    r"|compliance|finma|regulator|basel)(?!\w)", re.I)


def _query_commercial_intent(q: str, lang: str | None = None) -> bool:
    """True when the query is a commercial/product lookup (financial product,
    leasing, explicit shopping) and NOT a legal/regulatory one. Multilingual,
    lexicon-only, no model. Guarded so statute/regulator queries return False."""
    ql = (q or "").lower()
    if not ql or _COMM_LEGAL_GUARD.search(ql):
        return False
    if any(p in ql for p in _COMM_PHRASES):
        return True
    if _COMM_SUBSTR.search(ql):
        return True
    return bool(_content_terms(ql, lang) & _COMM_TOKENS)


# --- S10 step 3 (task 1083): person-name shape and required words, moved verbatim --- #
_NAMEGATE_MAXTERMS = _S.fusion.namegate_maxterms
_NAMEGATE_SHAPE_V2 = _S.fusion.namegate_shape_v2
# Nobiliary / patronymic particles — the ONLY lowercase infix a bare person name
# may carry ("Ursula von der Leyen", "Ludwig van Beethoven", "Charles de
# Gaulle", "Leonardo da Vinci", "Ibn Sina"). A closed linguistic set, not a
# name list: it says which FUNCTION WORDS can sit inside a name, so the
# contiguity rule below can reject German sentence nouns ("Inflation IN der
# Volkswirtschaft" — "in" is not a particle) without also rejecting real names.
_NOBILIARY = frozenset({
    "von", "van", "de", "del", "della", "der", "den", "di", "da", "das", "dos",
    "du", "la", "le", "les", "ter", "ten", "af", "av", "bin", "ibn", "bint",
    "al", "el", "mac", "mc", "st", "san", "santa", "zu", "zum", "vom",
})


def _person_name_tokens(q: str, q_content: set[str]) -> set[str] | None:
    """The queried person's NAME TOKENS when the raw query is person-name-
    shaped (AW_NAMEGATE trigger), else None.

    Shape, all required: 2..AW_NAMEGATE_MAXTERMS content terms; no digit
    anywhere; every content token in the RAW query starts uppercase (>= 2 of
    them). One lowercase content word ("newest Mistral AI model…") disqualifies
    — a bare name is capitalized in full or it is not a bare name. Stopwords
    ("who", "was", "wer", "qui") are ignored, so "who was Ada Lovelace"
    qualifies and its name tokens are {ada, lovelace} — coverage is measured
    against the NAME, never against interrogative filler. Deterministic on the
    query string: no head, no language model.

    SHAPE v2 (task 406, AW_NAMEGATE_SHAPE_V2) adds TWO deterministic rejections
    that cure the MEASURED German-noun-capitalisation false positive (404 §5.2:
    bb-008 "was ist Inflation in der Volkswirtschaft" 4->8). German capitalises
    every noun, so the v1 "all content tokens capitalised" shape is
    uninformative there; the two rules restore the discrimination without a
    language list or a lexicon. Off => byte-identical to v1.

    NOT included, and why (the census is the evidence, not an opinion): a third
    candidate rule — reject any token the German de-compounder splits — drops
    ZERO additional gate queries (bb-008 is already rejected by contiguity) and
    has a real cost: `_decompound("schwarzenegger")` returns ('schwarzen',
    'egger'), so it would silently disarm the lever on long true surnames.
    Dropped on the measurement.

    The OTHER 404 §5.2 claim is REFUTED here: dom-de-01 "Kuendigungsfrist
    Arbeitsvertrag BGB" was never a detector false positive at all — with the
    live de-compounder its `q_content` is 5 terms, over AW_NAMEGATE_MAXTERMS=4,
    so v1 already returns None. Its 5->6 in the 404 gate has some other cause.
    """
    if not (2 <= len(q_content) <= _NAMEGATE_MAXTERMS):
        return None
    if any(ch.isdigit() for ch in (q or "")):
        return None
    caps: set[str] = set()
    pos: list[int] = []      # raw word positions of the capitalised tokens
    infix: set[str] = set()  # every raw token strictly between the first and
    #                          last capitalised one (stopwords included — that
    #                          is exactly where nobiliary particles live)
    for i, m in enumerate(_WORD.findall(q or "")):
        t = m.replace("'", "").replace("’", "")
        tl = t.lower()
        if pos:
            infix.add((i, tl))
        if len(tl) < 2 or tl in _STOPWORDS or not _ALPHA_RE.search(tl):
            continue
        if t[:1].isupper():
            # (v2-1) an ALL-CAPS token >=2 chars is an acronym (BGB, ABB, ETH,
            # AHV, DGX) — an organisation/statute/spec reference, never a
            # person's name. Measured on the gate set (task 406/
            # shape-census.txt): drops ec-name-16 "ABB Schweiz", ec-name-23
            # "ETH Zürich", ch-01 "AHV …", dom-de-02 "… BGB", g0b7085b8ed84
            # "NVIDIA DGX Spark …" — five non-person queries, zero true names.
            if _NAMEGATE_SHAPE_V2 and t.isupper():
                return None
            caps.add(tl)
            pos.append(i)
        elif _NAMEGATE_SHAPE_V2 and tl in _NOBILIARY:
            # a lowercase NOBILIARY PARTICLE is part of the name, not a
            # disqualifying lowercase content word ("Ludwig van Beethoven" —
            # "van" is not in _STOPWORDS, so v1 rejected the whole query). It
            # is NOT a name token: coverage is measured on {ludwig, beethoven}.
            continue
        else:
            return None
    if len(caps) < 2:
        return None
    # (v2-2) a person's name is a CONTIGUOUS nominal phrase in the raw query
    # ("who was Ada Lovelace"), the only permitted infix being a NOBILIARY
    # PARTICLE ("Ursula von der Leyen", "Ludwig van Beethoven"). Capitalised
    # tokens separated by anything else are sentence nouns, not a name —
    # German capitalises every noun, which is the whole false-positive class
    # ("… Inflation in der Volkswirtschaft"): "in" is not a particle, so that
    # query is rejected, while "von der" keeps zero_signal/gd-14 alive. Measured
    # on the gate set: drops bb-005 "what was the Peace of Westphalia", bb-008,
    # ge34acab08efc "Ellen White über die Flache Erde" — zero bare names.
    if _NAMEGATE_SHAPE_V2 and pos and (pos[-1] - pos[0]) != len(pos) - 1:
        between = {tl for i, tl in infix if pos[0] < i < pos[-1]} - caps
        if not between or not between <= _NOBILIARY:
            return None
    return caps


def _person_name_query(q: str, q_content: set[str]) -> bool:
    """Boolean view of _person_name_tokens (kept for tests/readability)."""
    return _person_name_tokens(q, q_content) is not None


def _name_token_primaries(name_tokens, lang) -> dict:
    """{name_token: primary bm25 hash}. The PRIMARY (joined/whole-word) token
    only — never a de-compounded fragment: `_decompound('schwarzenegger')`
    returns ('schwarzen','egger'), and a page containing only `egger` does not
    name the person (task 406 §3.1 dropped a de-compounding rule for exactly
    this reason). Returns {} when a token emits nothing."""
    out = {}
    for t in (name_tokens or ()):
        g = bm25_token_groups(t, lang)
        if g:
            out[t] = g[0][1]
    return out


def _required_word_groups(q: str, lang, distinctive: set, maxq: int) -> list:
    """[(source_word, (distinctive hashes it emitted, ...))] — the query words a
    candidate MUST match on the sparse leg (task 411 directive c).

    A word qualifies when it emitted at least one DISTINCTIVE token; it is
    matched by ANY of those (so `O'Hear` is satisfied by `ohear` without
    requiring the junk `hear`, and `Arbeitsvertrag` by any of its pieces).
    Non-distinctive words are not requirements at all — which is what keeps a
    verbose question's recall intact."""
    out = []
    for w, _primary, hs in bm25_token_groups(q or "", lang):
        if w in _STOPWORDS:
            continue
        d = tuple(h for h in hs if h in distinctive)
        if d:
            out.append((w, d))
        if maxq and len(out) >= maxq:
            break
    return out


def _card_alias_in_query(q_content: set[str], title: str, names: str,
                         lang: str | None = None) -> bool:
    """True iff SOME alias of this entity card is fully contained in the query
    (exact token equality, like _card_name_match's focus — entity names are
    not morphological variants). This is the REF-CANON entity-query exemption:
    `_card_name_match >= THR` is the wrong test here because a descriptive
    entity query ("roche holding pharma unternehmen sitz basel") dilutes
    coverage far below the card-boost threshold while still plainly NAMING the
    entity — containment of one full alias is the naming test."""
    for a in _card_aliases(title, names):
        at = _content_terms(a, lang)
        if at and all(t in q_content for t in at):
            return True
    return False


# ---- search operators (task 2640) -------------------------------------------------------
# `site:HOST -site:HOST source:TYPE lang:XX country:XX` typed inside the query text. Pure
# parse only; the host table and the Qdrant filter live in `operators.py`.
_OP_TOKEN = re.compile(r"^(-?site|source|lang|country):(\S+)$", re.IGNORECASE)


def _norm_host(value: str) -> str:
    """The host as the index stores it: lower case, no scheme/path, no `www.`."""
    v = re.sub(r"^[a-z][a-z0-9+.-]*://", "", value.strip().lower())
    v = v.split("/", 1)[0].split("?", 1)[0].rstrip(".")
    return v[4:] if v.startswith("www.") else v


def _uniq(xs) -> tuple:
    return tuple(dict.fromkeys(xs))


class Operators:
    """The parsed operators. Immutable value object (plain class: this module is std-lib `re` only)."""
    __slots__ = ("site", "exclude_site", "source", "lang", "country")

    def __init__(self, site: tuple = (), exclude_site: tuple = (), source: tuple = (),
                 lang: str | None = None, country: str | None = None):
        for k, v in (("site", site), ("exclude_site", exclude_site), ("source", source),
                     ("lang", lang), ("country", country)):
            object.__setattr__(self, k, v)

    def __setattr__(self, k, v):
        raise AttributeError("Operators is immutable")

    def _key(self):
        return (self.site, self.exclude_site, self.source, self.lang, self.country)

    def __eq__(self, other):
        # duck-typed: a reloaded `query` module (tests do) yields a distinct class with the same value
        k = getattr(other, "_key", None)
        return callable(k) and self._key() == k()

    def __hash__(self):
        return hash(self._key())

    def __repr__(self):
        return "Operators(site=%r, exclude_site=%r, source=%r, lang=%r, country=%r)" % self._key()

    def any(self) -> bool:
        return bool(self.site or self.exclude_site or self.source or self.lang or self.country)

    def merged(self, *, site=(), exclude_site=(), source=(), lang=None, country=None):
        """Operators plus the named API parameters. Lists merge; for lang/country the operator wins."""
        return Operators(
            site=_uniq([*self.site, *[h for h in map(_norm_host, site) if h]]),
            exclude_site=_uniq([*self.exclude_site, *[h for h in map(_norm_host, exclude_site) if h]]),
            source=_uniq([*self.source, *[s.lower() for s in source if s]]),
            lang=self.lang or (lang.lower() if lang else None),
            country=self.country or (country.upper() if country else None))


def parse_operators(q: str):
    """Split `q` into (clean_q, Operators). A token that is not one of the five operators (with a
    non-empty value) keeps its text. With no operator the query comes back UNCHANGED (no whitespace
    normalisation), so operator-free serving stays byte-identical."""
    site, exclude, source = [], [], []
    lang = country = None
    kept = []
    for tok in q.split():
        m = _OP_TOKEN.match(tok)
        name, h = (m.group(1).lower(), _norm_host(m.group(2))) if m else (None, None)
        if name in ("site", "-site"):
            if not h:
                kept.append(tok)
            else:
                (site if name == "site" else exclude).append(h)
        elif name == "source":
            source.append(m.group(2).lower())
        elif name == "lang":
            lang = m.group(2).lower()
        elif name == "country":
            country = m.group(2).upper()
        else:
            kept.append(tok)
    ops = Operators(site=tuple(dict.fromkeys(site)), exclude_site=tuple(dict.fromkeys(exclude)),
                    source=tuple(dict.fromkeys(source)), lang=lang, country=country)
    return (" ".join(kept), ops) if ops.any() else (q, ops)
