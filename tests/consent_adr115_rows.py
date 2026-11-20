"""Real twin rows (one per ADR-115 rule) and the derive context the consent tests share.

Used by the derive tests and by `test_aw_consent_apply.py`; one copy (task 1241).
"""
from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from cytir.consent.derive import DeriveContext  # noqa: E402
FAMILIES = json.loads((REPO / "data" / "source-family-contracts.v1.json").read_text())["families"]
AS_OF = "2026-10-01T00:00:00Z"


def ctx(cache=None, arxiv=None, pmc=None):
    return DeriveContext(FAMILIES, {"__unparseable__": 0, **(cache or {})}, AS_OF, arxiv, pmc)


def rec(search, train, at="2026-07-19T09:30:38Z", status="ok"):
    return {"cc": {"status": status, "search_ok": search, "train_ok": train, "checked_at": at}}


WIKI = {"url": "https://en.wikipedia.org/wiki/Craigavon", "domain": "en.wikipedia.org",
        "source_type": "reference", "license": "CC-BY-SA-4.0", "consent_checked_at": "2026-08-02T02:08:30Z"}
SE = {"url": "https://math.stackexchange.com/a/4159296", "domain": "math.stackexchange.com",
      "source_type": "community", "license": "CC-BY-SA-4.0"}
ARXIV = {"url": "https://arxiv.org/abs/2503.21080", "domain": "arxiv.org", "source_type": "reference",
         "license": "arXiv-metadata-CC0"}
PUBMED = {"url": "https://pubmed.ncbi.nlm.nih.gov/19815838/", "domain": "pubmed.ncbi.nlm.nih.gov",
          "source_type": "reference", "license": "NLM-PubMed-Terms"}
HN = {"url": "https://news.ycombinator.com/item?id=48604851", "domain": "news.ycombinator.com",
      "license": "api-only", "provenance": "hn-discussion-firebase", "search_ok": True, "train_ok": False,
      "consent_state": "resolved", "consent_basis": "official-hn-firebase-api-only",
      "consent_checked_at": "2026-06-21"}
WEKO_CARRIED = {"url": "https://www.weko.admin.ch/dam/fr/sd-web/X1c86wvpMLVZ/dpc_2015-4.pdf",
                "domain": "weko.admin.ch", "provenance": "weko_docs", "license": "ch-gov-open-cite",
                "consent_state": "carried", "consent_basis": "carried-from-superseded-row",
                "consent_checked_at": "2026-09-09"}
IAIS_CARRIED = {"url": "https://www.iais.org/uploads/2025/06/IAIS-ICPs-and-ComFrame-December-2024.pdf",
                "domain": "iais.org", "provenance": "iais_docs", "license": "iais-noncommercial-attribution",
                "consent_state": "carried", "consent_basis": "carried-from-superseded-row"}
DILA = [
    {"url": "https://www.legifrance.gouv.fr/conv_coll/article/KALIARTI000054454368",
     "domain": "legifrance.gouv.fr", "provenance": "kali_article", "license": "Etalab-2.0"},
    {"url": "https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000006902584",
     "domain": "legifrance.gouv.fr", "provenance": "legi_code", "license": "Etalab-2.0"},
    {"url": "https://www.legifrance.gouv.fr/ceta/id/CETATEXT000053565418",
     "domain": "legifrance.gouv.fr", "provenance": "ce_juri", "license": "Etalab-2.0"},
    {"url": "https://www.legifrance.gouv.fr/jorf/id/JORFTEXT000051146673",
     "domain": "legifrance.gouv.fr", "provenance": "jorf_recent", "license": "Etalab-2.0"},
    {"url": "https://www.legifrance.gouv.fr/juri/id/JURITEXT000044300024",
     "domain": "legifrance.gouv.fr", "provenance": "cass_juri", "license": "Etalab-2.0"},
    {"url": "https://www.legifrance.gouv.fr/conv_coll/id/KALICONT000029178674",
     "domain": "legifrance.gouv.fr", "provenance": "kali_conv", "license": "Etalab-2.0"},
]
LODA_DUMP = {"url": "https://www.legifrance.gouv.fr/loda/article_lc/LEGIARTI000024415582",
             "domain": "legifrance.gouv.fr", "provenance": "loda", "license": "Etalab-2.0",
             "consent_state": "dump", "consent_basis": "official-dump-license"}
RII = {"url": "https://www.rechtsprechung-im-internet.de/jportal/?quelle=jlink&docid=KORE625042017",
       "domain": "rechtsprechung-im-internet.de", "provenance": "rii_courts",
       "license": "public-domain-official-work-UrhG-5"}
