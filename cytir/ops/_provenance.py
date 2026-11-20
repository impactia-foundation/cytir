"""Provenance sidecar of a generated, committed file (task 1074): `<file>.provenance.json` names the producer
verb, the digest of each input and the digest of the output, so a stranger can regenerate and compare."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def write_provenance(out_path, verb: str, inputs: dict[str, str]) -> Path:
    """inputs: label -> path. Writes `<out_path>.provenance.json`; returns its path."""
    out = Path(out_path)
    side = out.with_name(out.name + ".provenance.json")
    doc = {"producer": f"aw ops {verb}", "module": "aw.ops." + verb.replace("-", "_"),
           "inputs": {k: {"name": Path(v).name, "sha256": sha256_file(v)} for k, v in sorted(inputs.items())},
           "output": {"name": out.name, "sha256": sha256_file(out)}}
    side.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return side


def arg_inputs(args, *outputs) -> dict[str, str]:
    """The existing input files an argparse namespace names (every path-valued option that is a file and is
    not one of `outputs`), as label -> path, for `write_provenance`."""
    skip = {str(Path(o).resolve()) for o in outputs if o}
    found = {}
    for key, val in sorted(vars(args).items()):
        if isinstance(val, (str, Path)) and val and Path(val).is_file() and str(Path(val).resolve()) not in skip:
            found[key] = str(val)
    return found
