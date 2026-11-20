"""The vectors a retune measures: ONE key moved per arm (task 533's method).

The grid only PROPOSES. Every proposal is measured by the served ranker (`arm`), so a key the
grid does not move keeps the incumbent's value and a key the old simulator could not model
(the trust head, canton, parasite...) is still in force, because the server applies it.
"""
from __future__ import annotations

#: key -> absolute alternative values. A value equal to the incumbent's is skipped.
ALTERNATIVES: dict[str, list[float]] = {
    "academic": [0.0, -0.6, -1.0, -1.5, -3.0],
    "machine": [-0.5, -1.0],
    "trust": [2.5, 3.5],
    "title": [2.0, 3.0],
    "exact": [0.6, 1.0],
    "bm25": [0.8, 1.2],
}


def proposals(incumbent: dict, keys: list[str] | None = None,
              alternatives: dict[str, list[float]] | None = None) -> list[tuple[str, dict]]:
    """[(arm_name, weights)] -- the incumbent plus exactly one changed key each. Never mutates
    its input. A key absent from the incumbent is not proposed (it would add a knob)."""
    alt = ALTERNATIVES if alternatives is None else alternatives
    out = []
    for key in (keys if keys is not None else list(alt)):
        if key not in incumbent:
            continue
        for v in alt.get(key, []):
            if float(v) == float(incumbent[key]):
                continue
            out.append((f"{key}={v:g}", {**incumbent, key: float(v)}))
    return out
