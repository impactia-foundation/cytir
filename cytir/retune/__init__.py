"""cytir.retune -- the fusion-weights retune: it EMITS a candidate and a report, and nothing else.

ADR-099 amendment 3 (Tim, ruling 27 T4, 2026-10-07):
  1. the retune never deploys; it writes `data/retune/candidates/<date>-<sha>.*`;
  2. promotion of fusion weights is a PR that Tim merges;
  3. it is judged on the v2 chooser and the held-out battery, per query, with a null pair;
  4. it scores with the SERVED ranker (`GET /search` on the server's own app), never a second copy.

Candidates land in data/retune/candidates/ when run from a checkout; the operator's runner copies them to its own directory.
Layout: `arm` scores one weights file in a fresh process under prod's serving env; `grid` proposes
vectors; `judge` is the pure comparison (ADR-043 ranking bar, null band); `emit` is the command.
"""
