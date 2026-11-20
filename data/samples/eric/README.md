# ERIC inventory fixture (task 1386)

One API page of five REAL ERIC records (`ED659153`, `ED659156`, `ED659161`, `ED659172`, `ED659173`),
laid out as the operator's inventory loop writes it: `pages/000000000.{body.json,headers,curl.json,metadata.json,receipt.json}`.

Provenance (read-only, operator host, 2026-10-28):
- source page: the operator's ERIC inventory directory, page 000000000 (`pages/000000000.*`)
  (page 0 of 2,632, fetched 2026-08-15T13:13:48Z from `api.ies.ed.gov/eric/`, query `e_fulltextauth:1`, 200 rows)
- source body sha256 `ba1b56be05eb0f44c0f22633df10ca0167edad71e6e677f47a1029bdb5215022` (546,788 bytes);
  source headers sha256 `00a21215f236a669120f9123a7a5d1ba73b4dfb928fa22ed66142a951e1d24c1` (copied verbatim).
  headers: the shipped file has the `x-amz*` request-id headers removed (release hygiene); the receipt and metadata bind the shipped bytes; the sha256 above is the original's.

Re-encoding (the fixture is NOT the original page): the body keeps the first 5 `docs` verbatim and sets
`numFound` to 5; the request is the real one with `rows=5`; the curl metrics, transport metadata and page receipt
are rebuilt so every digest and URL binds the new body. `content-length` in the copied headers still names the
original 200-row body. The real fetch time is kept. No live ERIC call was made.

golden/: outputs of the pre-move script `acquire_eric_api_inventory.py` (origin/main 3c29bf94f, before the move) on this fixture, frozen 2026-10-28 (task 1386 review). Re-frozen 2026-10-28 (CI fixes, second round) after the headers rebinding: two runs byte-identical.
