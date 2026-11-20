"""ADR-062 Phase-2 (country knob) — GRADUATE the off-country magnitude knob.

Phase 1 shipped every head-driven knob at alpha=0 (byte-identical cold-start). Phase 2
graduates ONE knob — `off_country` (spec §1 knob 12, the off-country penalty MAGNITUDE)
— by raising its alpha so the live scorer reads the value from the head instead of the
static constant. EVERY OTHER KNOB STAYS AT alpha=0 (untouched cold-start).

The Phase-1 off_country head was self-distilled to the static default -2.0 (w=0, b=-2.0
— the exact L2->default minimiser; report.json shows target_var=0, fit_mae=0, |w|=0).
The Phase-2 oracle (the country oracle, a retired one-off) CONFIRMS -2.0 is the per-query MRR@10
optimum on the labelled sets (the -2.0 penalty already saturates the off-country lane),
so the graduated head == the self-distilled -2.0 head; graduating it is a genuine head-
driven path that emits -2.0. This is the honest, gated Phase-2 step: raise alpha, keep
the head at its oracle-confirmed value, let the ADR-043 gate prove zero regression.

  cyt ops graduate-qhead-country --in cytir/serve/qhead.json \
      --out cytir/serve/qhead.json --alpha 1.0 [--oracle /tmp/oracle.json]
"""
from __future__ import annotations
import argparse, json


def main(argv=None, prog=None):
    ap = argparse.ArgumentParser(prog=prog)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--knob", default="off_country")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--oracle", default=None, help="oracle.json to fold in as provenance")
    args = ap.parse_args(argv)

    art = json.load(open(args.inp))
    kn = art["knobs"].get(args.knob)
    if kn is None or kn.get("driver") != "head":
        raise SystemExit(f"knob {args.knob!r} is not a head-driven knob in {args.inp}")

    prev_alpha = kn.get("alpha", 0.0)
    kn["alpha"] = float(args.alpha)
    # keep w,b as self-distilled (== oracle-confirmed -2.0). Record provenance.
    kn["graduated"] = {
        "adr": "ADR-062", "phase": "2-country", "prev_alpha": prev_alpha,
        "note": ("off-country penalty MAGNITUDE (knob 12) graduated; head self-distilled "
                 "to -2.0 and oracle-confirmed optimal; scope stays deterministic (Q2)."),
    }
    if args.oracle:
        orc = json.load(open(args.oracle))
        kn["graduated"]["oracle"] = {
            "n": orc.get("n"), "dist": orc.get("dist"),
            "default_optimal": orc.get("default_optimal"),
            "n_improved_over_default": len(orc.get("improved", [])),
        }

    art["phase"] = "2-country"
    art["_comment"] = (
        "ADR-062 Phase-2 (country). Graduates ONLY the off_country magnitude knob "
        "(spec §1 knob 12): alpha>0 so the live scorer reads it from the head "
        "(self-distilled + oracle-confirmed -2.0). Every OTHER head-driven knob stays "
        "alpha=0 (Phase-1 cold-start, untouched). Scope detection stays deterministic "
        "(Q2). Flag-gated AW_QHEAD=0 default; per-knob AW_QHEAD_KNOBS=off_country_mag.")
    # sanity: no other head knob was accidentally graduated
    others = {k: v.get("alpha") for k, v in art["knobs"].items()
              if v.get("driver") == "head" and k != args.knob}
    assert all((a in (0, 0.0)) for a in others.values()), \
        f"other head knobs must stay alpha=0: {others}"

    json.dump(art, open(args.out, "w"))
    from cytir.ops._provenance import write_provenance
    write_provenance(args.out, "graduate-qhead-country", {"qhead_in": args.inp, **({"oracle": args.oracle} if args.oracle else {})})
    print(f"graduated knob={args.knob} alpha {prev_alpha}->{args.alpha}; other head knobs alpha=0")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
