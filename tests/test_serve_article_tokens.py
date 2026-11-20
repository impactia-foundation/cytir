from cytir.serve.article_tokens import glued_article_variants
from cytir.serve.schema import bm25_tokens, _hash
from cytir.serve.settings import Query


def test_glued_gets_split_form_appended():
    assert glued_article_variants("Art. 336c illness") == "Art. 336c illness 336 c"


def test_no_glued_token_is_identity():
    for q in ("Art. 336 c", "Artikel 12", "3 days", "336cd", "a336c"):
        assert glued_article_variants(q) == q


def test_split_form_adds_the_numeric_token():
    off = bm25_tokens("Art. 336c", "en")
    on = bm25_tokens(glued_article_variants("Art. 336c"), "en")
    assert _hash("336") not in off and _hash("336") in on
    assert set(off) <= set(on)


def test_flag_is_off_by_default():
    assert Query.__dataclass_fields__["query_glued_article"].metadata["default"] == "0"


# task 2792: exact-article signal
from cytir.serve.article_tokens import article_exact, query_article_ids, row_article_id

_T = "Obligationenrecht (OR; SR 220) — Art. 336 c — a. Kündigung zur Unzeit"
_U = "https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de#art_336_c"


def test_query_ids_glued_and_split_and_languages():
    assert query_article_ids("Art. 336c OR") == {("336", "c")}
    assert query_article_ids("art 336 c schweiz") == {("336", "c")}
    assert query_article_ids("Artikel 336") == {("336", "")}
    assert query_article_ids("Article 12b and Art. 5") == {("12", "b"), ("5", "")}
    assert query_article_ids("336c without keyword") == frozenset()
    assert query_article_ids("Art. 336 Abs. 2") == {("336", "")}


def test_row_id_prefers_fragment_then_title():
    assert row_article_id(_T, _U) == ("336", "c")
    assert row_article_id(_T, "https://x/y") == ("336", "c")
    assert row_article_id("Art. 336 — a. Grundsatz", "") == ("336", "")
    assert row_article_id("no article here", "") is None


def test_exact_means_same_number_and_letter():
    ids = query_article_ids("Art. 336c")
    assert article_exact(ids, _T, _U)
    assert not article_exact(ids, "OR — Art. 336 b — c. Verfahren", "u#art_336_b")
    assert not article_exact(ids, "OR — Art. 336 — a. Grundsatz", "u#art_336")
    assert not article_exact(query_article_ids("no ref"), _T, _U)


def test_weight_flag_is_off_by_default():
    assert Query.__dataclass_fields__["article_exact_w"].metadata["default"] == "0"


# task 2792 review: the stage-2 builder is a no-op at weight 0, adds exactly W at W > 0, never subtracts
def _build(q, monkeypatch, w):
    from cytir.serve import levers
    monkeypatch.setenv("AW_ARTICLE_EXACT_W", w)
    pls = {1: {"url": "https://x/eli#art_336_c", "title": "OR — Art. 336 c — a", "last_seen": "2026-01-01"},
           2: {"url": "https://x/eli#art_336_b", "title": "OR — Art. 336 b — c", "last_seen": "2026-01-01"}}
    z = dict(_name_primaries={}, _reqterm={}, authority_on=False, base_off_country=0.0, commercial=False,
             cov={}, hubfloor_on=False, k_bm25_scale=1.0, k_dense_scale=1.0, k_exact=1.0, k_machine=0.0,
             k_off_country=0.0, k_recency=0.0, k_title=1.0, lang="de", namegate_tokens=(),
             norm={"dense": {1: 0.5, 2: 0.5}}, payloads=pls, presence={}, q=q, q_academic=False,
             q_cantons=(), q_content=[], q_country=None, q_country_soft=None, q_terms=[], rare_toks=[],
             scope_conf=0.0, scoped=False, target_bm25_weight=0.0, translated_bm25_weight=0.0,
             w_canton=0.0, w_foreign=0.0)
    out = levers.build_rows(**z)
    return {r[0]: r[2] for r in out[-1]}


def test_weight_zero_is_a_noop_in_the_stage2_builder(monkeypatch):
    base = _build("kündigung sperrfrist", monkeypatch, "0")   # names no article
    off = _build("Art. 336c kündigung sperrfrist", monkeypatch, "0")
    assert off[1] == base[1] and off[2] == base[2]
    assert _build("Art. 336c kündigung sperrfrist", monkeypatch, "-2.0") == off   # clamped: never a penalty


def test_weight_adds_exactly_w_to_the_matching_row_only(monkeypatch):
    off = _build("Art. 336c kündigung sperrfrist", monkeypatch, "0")
    on = _build("Art. 336c kündigung sperrfrist", monkeypatch, "1.0")
    assert on[1] - off[1] == 1.0 and on[2] == off[2]


# --- task 2797: statute narrowing ---------------------------------------------------------
from cytir.serve import article_tokens as at

_OR = ("Obligationenrecht (OR; SR 220) — Art. 336 — a. Grundsatz", "https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de#art_336")
_BV = ("Bundesverfassung (BV; SR 101) — Art. 7 — 1. Kapitel: Grundrechte", "https://www.fedlex.admin.ch/eli/cc/1999/404/de#art_7")
_GG = ("Art 7 GG", "https://www.gesetze-im-internet.de/gg/art_7.html")
_WEB = ("Art. 7 explained", "https://example.org/art-7")


def test_query_statutes_shapes():
    f = at.query_statutes
    assert f("Art. 336c OR Kündigung") == {"220"}                      # DE abbreviation
    assert f("article 41 du Code des obligations") == {"220"}          # FR name
    assert f("art. 93 LP saisie") == {"281.1"}                         # FR abbreviation
    assert f("articolo 41 codice delle obbligazioni") == {"220"}       # IT name
    assert f("Art. 7 Federal Constitution") == {"101"}                 # EN name
    assert f("SR 220 Art. 97") == {"220"} and f("RS 101 art. 7") == {"101"}  # SR / RS number
    assert f("article 7 de la Charte canadienne") == {"x:ca-charter"}
    assert f("Art. 7") == frozenset() and f("art 12 or something") == frozenset()  # no statute / lower-case or


def test_narrowed_keeps_exact_statute_and_drops_other():
    ids, keys = at.query_article_ids("Art. 7 BV"), at.query_statutes("Art. 7 BV")
    assert at.article_exact_narrowed(ids, keys, *_BV)
    assert not at.article_exact_narrowed(ids, keys, _OR[0].replace("336", "7"), _OR[1].replace("336", "7"))
    assert not at.article_exact_narrowed(ids, keys, *_WEB)


def test_narrowed_foreign_statute_matches_only_its_own_rows():
    ids = at.query_article_ids("Wortlaut Artikel 7 Grundgesetz")
    keys = at.query_statutes("Wortlaut Artikel 7 Grundgesetz")
    assert at.article_exact_narrowed(ids, keys, *_GG) and not at.article_exact_narrowed(ids, keys, *_BV)
    ids = at.query_article_ids("article 7 de la Charte canadienne")
    keys = at.query_statutes("article 7 de la Charte canadienne")
    assert at.article_exact(ids, *_BV) and not at.article_exact_narrowed(ids, keys, *_BV)  # ca-law-022


def test_narrowed_no_statute_is_the_2792_behaviour():
    ids = at.query_article_ids("Art. 7")
    for row in (_BV, _OR, _GG, _WEB):
        assert at.article_exact_narrowed(ids, frozenset(), *row) == at.article_exact(ids, *row)


def test_narrowing_never_adds():
    for q in ("Art. 7 BV", "Art. 7", "SR 220 Art. 7", "Art. 7 Charte canadienne"):
        ids, keys = at.query_article_ids(q), at.query_statutes(q)
        for row in (_BV, _OR, _GG, _WEB):
            assert (not at.article_exact_narrowed(ids, keys, *row)) or at.article_exact(ids, *row)


def test_flag_registered_off_by_default(monkeypatch):
    from cytir.serve import settings
    monkeypatch.delenv("AW_ARTICLE_EXACT_STATUTE", raising=False)
    assert not settings.current("AW_ARTICLE_EXACT_STATUTE")


def test_article_exact_257_is_not_257e_task2822():
    """Pins probe b001 (task 2822): `Artikel 257` matches `#art_257` only, never the sibling article 257e."""
    from cytir.serve.article_tokens import article_exact, article_exact_narrowed, query_article_ids, query_statutes
    q = "was regelt Artikel 257 des Schweizer Obligationenrechts zur Miete"
    ids, keys = query_article_ids(q), query_statutes(q)
    base = "https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de"
    t257 = "Obligationenrecht (OR; SR 220) — Art. 257 — 1. Mietzins"
    t257e = "Obligationenrecht (OR; SR 220) — Art. 257 e — II. Sicherheiten durch den Mieter"
    assert ids == frozenset({("257", "")}) and keys == frozenset({"220"})
    assert article_exact(ids, t257, base + "#art_257") and article_exact_narrowed(ids, keys, t257, base + "#art_257")
    assert not article_exact(ids, t257e, base + "#art_257_e")
    assert query_article_ids("Art. 257e OR") == frozenset({("257", "e")})
