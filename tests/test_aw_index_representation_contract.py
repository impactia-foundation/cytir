#!/usr/bin/env python3
"""The representation contract, replayed against the rows that actually shipped.

Every fixture is a verbatim line from a sealed
`shard-NNNN.texts.jsonl.gz` plus the payload aw_e8 carries for that point id
(task 527's fixture-making script). A guard that only fails on a
hand-written example proves nothing about the incident it claims to prevent.

Run: python3 -m pytest tests/test_aw_index_representation_contract.py
"""
import json
import pathlib
import unittest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
from cytir.index import representation_contract as RC  # noqa: E402

FIXTURES = (HERE / "testdata" / "contract-fixtures.json")


def load():
    if not FIXTURES.exists():
        raise unittest.SkipTest(f"fixtures not present: {FIXTURES}")
    return json.loads(FIXTURES.read_text())


def merged(fx):
    """The record as the contract sees it: the staged row plus the payload
    fields the collection carries (title / site_name / source_type), which is
    what a stage-time check would have from the producer's own record."""
    rec = dict(fx["record"])
    pay = fx.get("build_payload") or fx.get("twin_payload") or {}
    for k in ("title", "site_name", "source_type", "lang", "url"):
        rec.setdefault(k, pay.get(k))
    return rec


class TestRegistry(unittest.TestCase):
    def test_registry_loads_and_surfaces_are_closed(self):
        reg = RC.load_registry()
        self.assertEqual(reg["schema"], "aw.representation-contract.v1")
        self.assertTrue(reg["families"])
        for key, fam in reg["families"].items():
            self.assertIn(fam["dense"], RC.SURFACES, key)
            self.assertIn(fam["sparse"], RC.SURFACES, key)

    @unittest.skipUnless((REPO / "data" / "assembly" / "sealed-populations.v1.json").is_file(),
                         "sealed-populations is a repo-only register (build-host paths); absent from the cut")
    def test_every_sealed_family_has_a_contract(self):
        """The registry must cover the populations that contribute vectors, or
        the guard has a blind spot with no name."""
        sealed = json.loads(
            (REPO / "data" / "assembly"
             / "sealed-populations.v1.json").read_text())
        covered = set()
        for fam in RC.load_registry()["families"].values():
            covered.update(fam.get("applies_to", []))
        covered.add("wikipedia_413a")
        missing = [p["set_id"] for p in sealed["populations"]
                   if p["set_id"] not in covered
                   and p["set_id"] not in {
                       "task417_families",          # covered per pack source
                       "task427_postconsent",       # composes; audited, see 527 sec.2
                       "task477_direct_capture_epoch4_chunks",
                       "task471_f2_pdf_docling_epoch4_chunks",
                       "task505_narrow_b2_track1a_chunks"}]
        self.assertEqual(missing, [], f"populations with no contract: {missing}")


class TestProvenanceShare(unittest.TestCase):
    def test_document_text_scores_zero(self):
        share, _, _ = RC.provenance_share(
            "Art. 336 CO\n\nLe congé est abusif lorsqu'il est donné par une "
            "partie pour une raison inhérente à la personnalité de l'autre.")
        self.assertEqual(share, 0.0)

    def test_identity_block_scores_high(self):
        share, prov, total = RC.provenance_share(
            "Article identity\n- Canonical URL: https://x/y\n"
            "- Source record: abc\n- Jurisdiction: Germany (federal)\n"
            "§ 3\nDie Münzen sind gesetzliches Zahlungsmittel.")
        self.assertGreater(share, 0.6)
        self.assertEqual(prov + 2, total)


class TestReplayedFailures(unittest.TestCase):
    """The two proven representation changes, and the control that must pass."""

    @classmethod
    def setUpClass(cls):
        cls.fx = load()
        cls.reg = RC.load_registry()

    def test_law_provenance_head_fails_the_cap(self):
        fx = self.fx["law_provenance_head"]
        rec = merged(fx)
        fam = self.reg["families"]["law_article_pack"]
        v = RC.check_row(rec, fam, embedded=rec["text"],
                         sparse_surface_text=rec["text"])
        checks = {x["check"] for x in v}
        self.assertIn("provenance.cap", checks,
                      "the guard does not see the materializer identity block")
        self.assertIn("conformance.sparse", checks,
                      "the guard does not see sparse_text being bypassed")
        cap = [x for x in v if x["check"] == "provenance.cap"][0]
        self.assertGreater(cap["share"], 0.5)

    def test_f2_html_raw_body_fails_identity(self):
        fx = self.fx["f2_html_body_only"]
        rec = merged(fx)
        fam = self.reg["families"]["f2_html"]
        v = RC.check_row(rec, fam, embedded=rec["text"],
                         sparse_surface_text=rec["text"])
        checks = {x["check"] for x in v}
        self.assertIn("identity.dense", checks,
                      "the guard does not see the title missing from the "
                      "embedded dense surface")
        self.assertIn("conformance.dense", checks)

    def test_card_website_blurb_loses_the_uid_hook(self):
        """ADR-060's sparse channel is name + UID + hooks. The build hands the
        binder the DENSE prose, which carries neither the UID nor (past the
        1,600-char truncation) the last officers."""
        fx = self.fx["card_website_blurb"]
        rec = merged(fx)
        pay = fx.get("build_payload") or {}
        rec["uid"] = (pay.get("register_id") or "").replace("-", "").replace(".", "")
        fam = self.reg["families"]["entity_card"]
        # the binder's own line: surface = rec["text"] or rec["dense_text"]
        sparse_surface = rec.get("text") or rec.get("dense_text")
        v = RC.check_row(rec, fam, embedded=rec.get("text"),
                         sparse_surface_text=sparse_surface)
        checks = {x["check"] for x in v}
        self.assertIn("identity.sparse", checks,
                      "the guard does not see the register UID absent from "
                      "the surface that becomes the BM25 posting")
        self.assertTrue(
            any(x.get("field") == "uid" for x in v
                if x["check"] == "identity.sparse"))

    def test_wikipedia_control_passes(self):
        fx = self.fx["wiki_composed"]
        rec = merged(fx)
        fam = self.reg["families"]["wikipedia_413a"]
        v = RC.check_row(rec, fam, embedded=rec["text"],
                         sparse_surface_text=rec["text"])
        self.assertEqual(v, [], f"the control family must pass cleanly: {v}")


class InsertedCitationIsNotAMissingIdentity(unittest.TestCase):
    """Task 528b: 80% of gesetze failed identity.dense while carrying its title.

    The composed surface is "<law name> (<abbr>) - Art N. <law name> ... Art N"
    and the declared identity field is the payload title "<law name> - Art N",
    so the whole identity is present and none of it is contiguous. The check now
    splits on the separator and requires EVERY part -- strict enough that a
    dropped article number still fails.
    """

    def test_a_citation_inserted_into_the_title_still_carries(self):
        from cytir.index.representation_contract import carries_composed
        title = "Gesetz ueber die Auspraegung einer 1-DM-Goldmuenze - Art 1"
        dense = ("Gesetz ueber die Auspraegung einer 1-DM-Goldmuenze "
                 "(1-DM-GoldmuenzG) - Art 1. Die Deutsche Bundesbank ...")
        self.assertTrue(carries_composed(dense, title))

    def test_a_dropped_article_number_still_fails(self):
        from cytir.index.representation_contract import carries_composed
        title = "Gesetz ueber die Auspraegung einer 1-DM-Goldmuenze - Art 1"
        dense = "Gesetz ueber die Auspraegung einer 1-DM-Goldmuenze. Die Bank"
        self.assertFalse(carries_composed(dense, title))

    def test_a_missing_title_still_fails(self):
        from cytir.index.representation_contract import carries_composed
        self.assertFalse(carries_composed("Article identity - Canonical URL: x",
                                          "Code de commerce - Article L110-2"))


def main() -> int:
    """Propagate a real exit code -- the cov351_verify defect, generalised:
    a bare `unittest.main()` call under `__main__` is invisible to
    `test_suite_roster.py`'s AST check for `raise SystemExit(main())`, and a
    driver checking `$?` on it would be checking nothing. See
    test_suite_roster.py::test_a_gate_propagates_its_own_verdict.
    """
    result = unittest.main(verbosity=2, exit=False).result
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
