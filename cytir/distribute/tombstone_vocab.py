"""The public tombstone vocabulary (task 2532): a stdlib-only leaf, no `aw` imports, no file I/O.

It lives apart from `pack.py` because `cytir/index/change_log.py` needs it and the promote-queue drain
imports the change log: the drain's import closure is a guarded surface (no unguarded file write),
and `pack.py` holds real writers. `pack.TOMBSTONE_REASONS` stays a re-export of these names.
"""

# ------------------------------------------------------------------ tombstones
#: `superseded_generation` (ADR-114 am.1): an old generation of rows retired because the new rows
#: are already served at the same urls. An ordinary delete in a pack; it is NOT a takedown and
#: never belongs to the takedown set below.
TOMBSTONE_REASONS = frozenset({"superseded", "aged_out", "takedown", "superseded_generation"})
#: The reasons a publisher/legal takedown list is built from. One member on purpose.
TAKEDOWN_REASONS = frozenset({"takedown"})
