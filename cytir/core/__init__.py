"""cytir.core — the identity/canon layer (task 646 S2, task 643 §6.1).

Doctrine (engineering-doctrine, project interpretation): `cytir.core` is the
only legal source of point-id minting, URL canonicalization, and the
embedder pin. Everything else imports from here; a second definition of
any of these elsewhere is a defect by construction (ratcheted in S3).
"""
from __future__ import annotations
