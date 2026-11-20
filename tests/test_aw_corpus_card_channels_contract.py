"""The register-first card rule, as a stage-time refusal.

Replayed on the sealed Vector8 source-evidence rows (the same fixture
`test_aw_corpus_card_channels.py` uses), so the guard is proven against the
row that caused the incident rather than a hand-written example.
"""
from __future__ import annotations

from cytir.corpus import card_channels as ECC       # noqa: E402
from cytir.index import representation_contract as RC            # noqa: E402
from tests.test_aw_corpus_card_channels import vector8_rows  # noqa: E402

FAM = RC.load_registry()["families"]["entity_card"]


def _stage_row(res: dict) -> dict:
    """The record as the task-467 producer will now seal it: both channels plus
    the register evidence the contract checks the dense surface against."""
    ev = res["register_evidence"]
    return {"point_id": "2a5bc752-ce61-58e7-9453-d63302b86e06",
            "text": res["dense_text"], "dense_text": res["dense_text"],
            "sparse_text": res["sparse_text"],
            "dense_class": res["dense_class"],
            "source_type": "entity_card",
            "title": ev.get("title"), "uid": ev.get("uid"),
            "register_purpose": ev.get("register_purpose"),
            "register_officers": ev.get("register_officers")}


def test_register_first_surface_passes_clean():
    rec = _stage_row(ECC.resolve_channels(vector8_rows()))
    v = RC.check_row(rec, FAM, embedded=rec["text"],
                     sparse_surface_text=rec["sparse_text"])
    assert v == [], v


def test_the_website_blurb_surface_is_refused():
    """The shipped rebuild's surface, rebuilt from the same sealed rows by
    removing only the ruling: the website row wins, everything else is equal."""
    rows = vector8_rows()
    web = dict(next(r for lbl, r in rows if lbl == "task393_apply"))
    res = ECC.resolve_channels(vector8_rows())
    rec = _stage_row(res)
    rec["text"] = rec["dense_text"] = web["dense_text"][:1600]
    rec["sparse_text"] = rec["text"]           # the binder's pre-528a collapse
    rec["dense_class"] = ECC.DENSE_WEBSITE
    checks = {x["check"] for x in RC.check_row(
        rec, FAM, embedded=rec["text"], sparse_surface_text=rec["sparse_text"])}
    # the ruling, refused by class
    assert "dense.class_refused" in checks
    # the register purpose is not in the marketing copy
    assert "identity.dense" in checks
    # officers truncated away: element-wise, not a joined substring
    assert "identity.dense_each" in checks
    # and the UID hook never reaches the posting
    assert "identity.sparse" in checks


def test_officers_truncated_at_the_tail_are_caught_element_wise():
    """A surface that names the FIRST officer and cuts the rest must fail --
    the shipped card ended '… Marc Maret (…); Ni'."""
    res = ECC.resolve_channels(vector8_rows())
    rec = _stage_row(res)
    cut = rec["text"].index("Niclas Delfs")
    rec["text"] = rec["dense_text"] = rec["text"][:cut] + "Ni"
    v = RC.check_row(rec, FAM, embedded=rec["text"],
                     sparse_surface_text=rec["sparse_text"])
    each = [x for x in v if x["check"] == "identity.dense_each"]
    assert each, v
    assert each[0]["field"] == "register_officers"
    assert any("Niclas Delfs" in m for m in each[0]["missing"])


def test_compose_index_text_is_idempotent():
    """Load-bearing for task 528a-2's durable stages.

    The F2-HTML and news stages are sealed carrying the COMPOSED surface
    (`"<title>. <site_name>. <body>"`), which is what their vectors were made
    from. A rebuild hands that text back to the binder, whose `_retokenize`
    composes again -- so composition must be a no-op on an already-composed
    surface, or a rebuild would prepend the title twice and quietly change the
    representation a second time. It is, because `compose_index_text` is
    dedup-guarded: a head token already present in the body is not repeated.
    """
    import cytir.core.common as AC

    for rec in (
        {"text": "Body about cats. More body.", "title": "Cats Page",
         "site_name": "Example"},
        {"text": "Arbeitssicherheit\nBeschreibung ...",
         "title": "Arbeitssicherheit", "site_name": "Suva"},
        {"text": "a body with no title match", "title": "Totally Other",
         "site_name": None},
    ):
        once = AC.compose_index_text(rec)
        twice = AC.compose_index_text({**rec, "text": once})
        assert once == twice, rec["title"]
