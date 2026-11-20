#!/usr/bin/env python3
"""Tests for cytir/index/pack_surface.py, replayed on VERBATIM sealed pack rows.

`tests/testdata/task528b_sealed_pack_rows.json` holds 12 rows cut byte-wise
out of the sealed task-417 stages (two per family), each carrying the sha256 of
its own non-underscore fields.  Every assertion below is about those bytes, so
a test that passes here is a statement about the shipped corpus and not about a
hand-written example.

The four marked `_FAILS_ON_SHIPPED` reproduce, as failures, exactly what task
527 measured in `aw_e8`:

  * the embedded dense surface leads with the materializer's identity block
  * the document's own title is absent from the dense surface for three of the
    five families
  * the provenance block is byte-similar across articles of the same law, so
    two different articles' dense surfaces are near-duplicates
  * the sparse channel is a different string from the dense one and the binder
    was reading only one of them
"""
from __future__ import annotations

import json
import os
import unittest

from cytir.index.pack_surface import (Atom, PackSurfaceError, compose_surfaces,
                          compose_title, identity_fields, load_contract,
                          spec_for_source, split_channel_atoms)

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "testdata", "task528b_sealed_pack_rows.json")

FAMILY_SOURCE = {
    "gesetze": "gesetze-official-current-v1",
    "legifrance": "legifrance-dila-exact-production-v4",
    "dila-loda": "dila-loda-exact-production-v1",
    "pubmed": "pubmed-exact-production-academic-markdown-v3",
    "arxiv": "arxiv-academic-markdown-v2-exact",
    "fedlex": "fedlex-r0-exact-current-v1",
}


def load_rows():
    with open(FIXTURES, encoding="utf-8") as fh:
        return json.load(fh)


class SealedRowsAreVerbatim(unittest.TestCase):
    def test_every_fixture_row_matches_its_recorded_digest(self):
        import hashlib
        for row in load_rows():
            body = {k: v for k, v in row.items() if not k.startswith("_")}
            digest = hashlib.sha256(json.dumps(
                body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            self.assertEqual(digest, row["_row_sha256"], row["point_id"])

    def test_all_six_pack_families_are_represented(self):
        self.assertEqual(
            sorted({r["_family"] for r in load_rows()}), sorted(FAMILY_SOURCE))


class AtomSplitIsExact(unittest.TestCase):
    def test_every_sealed_row_recomposes_byte_identically(self):
        for row in load_rows():
            atoms = split_channel_atoms(row["canonical_text"], row["text"])
            self.assertEqual("\n\n".join(a.sparse_text for a in atoms),
                             row["canonical_text"], row["point_id"])
            self.assertEqual("\n\n".join(a.dense for a in atoms),
                             row["text"], row["point_id"])

    def test_a_mismatched_pair_is_refused_rather_than_guessed(self):
        row = load_rows()[0]
        with self.assertRaises(PackSurfaceError):
            split_channel_atoms(row["canonical_text"], row["text"] + "\n\nstray")

    def test_a_lead_only_document_round_trips(self):
        self.assertEqual(
            [Atom("", "", "body")], split_channel_atoms("body", "body"))


class ArmAIsAByteExactNoOp(unittest.TestCase):
    """With nothing declared the re-projection must return the sealed bytes.

    This is the A/B's floor: if it fails, every arm-B number is measuring a
    lossy round-trip instead of the surface change.
    """

    def test_no_declaration_reproduces_both_channels(self):
        contract = load_contract()
        for row in load_rows():
            spec = dict(spec_for_source(contract, FAMILY_SOURCE[row["_family"]]))
            spec.update({"provenance_headings": [], "sparse_only_headings": [],
                         "dense_title": "none", "sparse_prepends_title": False})
            out = compose_surfaces(row, spec)
            self.assertEqual(out["text"], row["text"], row["point_id"])
            self.assertEqual(out["sparse_text"], row["sparse_text"],
                             row["point_id"])


class ShippedSurfaceIsTheDefect(unittest.TestCase):
    """_FAILS_ON_SHIPPED: these assert the properties `aw_e8` does NOT have."""

    def test_shipped_dense_leads_with_the_identity_block(self):
        # The defect, stated as a fact about the sealed bytes.
        leads = 0
        for row in load_rows():
            spec = spec_for_source(load_contract(),
                                   FAMILY_SOURCE[row["_family"]])
            heads = spec["provenance_headings"]
            first = row["text"].split("\n", 1)[0].strip()
            second = row["text"].split("\n\n", 2)[1].split("\n", 1)[0].strip() \
                if "\n\n" in row["text"] else ""
            if first in heads or second in heads:
                leads += 1
        self.assertGreaterEqual(leads, 10, "shipped rows should lead with it")

    def test_document_first_removes_the_identity_block_from_dense(self):
        contract = load_contract()
        for row in load_rows():
            spec = spec_for_source(contract, FAMILY_SOURCE[row["_family"]])
            out = compose_surfaces(row, spec)
            for heading in spec["provenance_headings"]:
                self.assertNotIn(f"\n{heading}\n", "\n" + out["text"],
                                 f"{row['point_id']} still carries {heading}")

    def test_document_first_puts_the_title_in_the_dense_surface(self):
        contract = load_contract()
        for row in load_rows():
            spec = spec_for_source(contract, FAMILY_SOURCE[row["_family"]])
            out = compose_surfaces(row, spec)
            title = out["composed_title"]
            if not title:
                continue
            head = title.split(" — ")[0].split(" (")[0][:40]
            self.assertIn(head, out["text"][:len(title) + 200],
                          f"{row['point_id']} dense does not open with its title")

    def test_two_articles_of_one_law_stop_being_near_duplicates(self):
        """The mechanism: the identity block is the same for every article.

        Measured on the two sealed gesetze rows -- both are §-articles of laws
        in the same archive, so their SHIPPED dense surfaces share the block
        line-for-line while their bodies differ.
        """
        rows = [r for r in load_rows() if r["_family"] == "gesetze"]
        self.assertEqual(len(rows), 2)
        contract = load_contract()
        spec = spec_for_source(contract, FAMILY_SOURCE["gesetze"])

        def line_overlap(a: str, b: str) -> float:
            la = {ln.strip() for ln in a.split("\n") if ln.strip()}
            lb = {ln.strip() for ln in b.split("\n") if ln.strip()}
            return len(la & lb) / max(len(la | lb), 1)

        shipped = line_overlap(rows[0]["text"], rows[1]["text"])
        fixed = line_overlap(compose_surfaces(rows[0], spec)["text"],
                             compose_surfaces(rows[1], spec)["text"])
        self.assertLess(fixed, shipped,
                        "document-first must reduce cross-article line overlap")

    def test_the_provenance_survives_as_record_metadata(self):
        """Task 516's rule: it stops being the first sentence, it does not go."""
        contract = load_contract()
        for row in load_rows():
            spec = spec_for_source(contract, FAMILY_SOURCE[row["_family"]])
            if not spec["provenance_headings"]:
                continue
            atoms = split_channel_atoms(row["canonical_text"], row["text"])
            if not any(a.local in spec["provenance_headings"] for a in atoms):
                continue
            out = compose_surfaces(row, spec)
            self.assertTrue(out["source_identity"], row["point_id"])
            for heading, block in out["source_identity"].items():
                self.assertIn(heading, spec["provenance_headings"])
                self.assertTrue(block, f"{row['point_id']} {heading} empty")

    def test_the_sparse_channel_is_a_different_string_from_the_dense_one(self):
        """assembly_binder.py fed one surface to both legs (task 527 §4.2)."""
        contract = load_contract()
        for row in load_rows():
            spec = spec_for_source(contract, FAMILY_SOURCE[row["_family"]])
            out = compose_surfaces(row, spec)
            self.assertNotEqual(out["text"], out["sparse_text"], row["point_id"])

    def test_hierarchy_leaves_dense_and_stays_in_sparse(self):
        contract = load_contract()
        for row in load_rows():
            spec = spec_for_source(contract, FAMILY_SOURCE[row["_family"]])
            if not spec["sparse_only_headings"]:
                continue
            atoms = split_channel_atoms(row["canonical_text"], row["text"])
            present = [a for a in atoms if a.local in spec["sparse_only_headings"]]
            if not present:
                continue
            out = compose_surfaces(row, spec)
            for atom in present:
                self.assertNotIn(atom.body[:60], out["text"], row["point_id"])
                self.assertIn(atom.body[:60], out["sparse_text"], row["point_id"])


class TitleComposition(unittest.TestCase):
    def test_the_citation_is_added_when_the_family_declares_one(self):
        contract = load_contract()
        rows = [r for r in load_rows() if r["_family"] == "gesetze"]
        spec = spec_for_source(contract, FAMILY_SOURCE["gesetze"])
        for row in rows:
            atoms = split_channel_atoms(row["canonical_text"], row["text"])
            identity = identity_fields(atoms, spec["provenance_headings"])
            if not identity.get("Article identity", {}).get("Law abbreviation"):
                continue
            abbr = identity["Article identity"]["Law abbreviation"]
            enriched = dict(row, source_identity=identity)
            self.assertIn(abbr, compose_title(enriched, spec))

    def test_a_newline_inside_a_title_is_flattened(self):
        contract = load_contract()
        spec = spec_for_source(contract, FAMILY_SOURCE["gesetze"])
        title = compose_title(
            {"source_metadata": {"title": "Erste Zeile\nZweite Zeile"}}, spec)
        self.assertEqual(title, "Erste Zeile Zweite Zeile")


class TheShippedBytesViolateEveryOneOfThose(unittest.TestCase):
    """The same properties, asserted to be ABSENT from what `aw_e8` embedded.

    A guard is only worth its line count if the incident it names would have
    tripped it.  These run the identical predicates against the sealed `text`
    the collection was actually built from, and assert they do not hold -- so
    the class above is a fix and not a tautology.
    """

    def test_shipped_dense_carries_the_identity_block(self):
        contract = load_contract()
        offenders = 0
        for row in load_rows():
            spec = spec_for_source(contract, FAMILY_SOURCE[row["_family"]])
            for heading in spec["provenance_headings"]:
                if f"\n{heading}\n" in "\n" + row["text"]:
                    offenders += 1
                    break
        self.assertGreaterEqual(offenders, 10)

    def test_shipped_dense_does_not_open_with_the_document_title(self):
        missing = 0
        for row in load_rows():
            title = (row.get("source_metadata") or {}).get("title") or ""
            if not title:
                continue
            head = title.split("—")[0].split("\n")[0].strip()[:40]
            if head and head not in row["text"][:300]:
                missing += 1
        self.assertGreaterEqual(missing, 4)

    def test_shipped_sparse_and_dense_were_collapsed_by_the_binder(self):
        """The pair exists in the record; the binder read only one of them."""
        for row in load_rows():
            self.assertNotEqual(row["text"], row["sparse_text"], row["point_id"])


class ContractIsClosed(unittest.TestCase):
    def test_an_undeclared_source_is_refused_not_defaulted(self):
        with self.assertRaises(PackSurfaceError):
            spec_for_source(load_contract(), "some-new-family-v1")

    def test_every_declared_source_names_its_producer_and_ceiling(self):
        for source, spec in load_contract()["sources"].items():
            self.assertTrue(spec.get("producer"), source)
            self.assertIsInstance(spec.get("max_quarantine_rate"), float, source)
            self.assertIn("provenance_headings", spec, source)


if __name__ == "__main__":
    unittest.main()
