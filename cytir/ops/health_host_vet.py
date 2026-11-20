"""Vet health-class outlink hosts against POLICY section 4.6 (task 991).

Assigns each host of the roster class `health` ONE category with a `basis` string. The decision uses the host
name, its TLD and the roster fields only. No page is fetched. Rules live in data/health-host-vet-rules.v1.json.

Categories (academic_other = university host with no medical signal; out until Tim rules): official, igo, society, academic_medical, publisher, hospital_provider, commercial_wellness,
gov_nonhealth (government TLD with no health signal: out until Tim rules), unknown.
Pattern-only matches on a commercial TLD (com, net, info, biz, io, shop) never reach a trusted category:
"when in doubt, out".

    python -m cytir.ops.health_host_vet --roster R --out DIR [--rules RULES]
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

from cytir.core.paths import data_path

DEFAULT_RULES = data_path("health-host-vet-rules.v1.json")
COMMERCIAL_TLDS = {"com", "net", "info", "biz", "io", "shop", "store", "app", "co"}
CATEGORIES = ["official", "igo", "society", "academic_medical", "publisher", "hospital_provider",
              "commercial_wellness", "gov_nonhealth", "academic_other", "unknown"]
PATTERN_CATS = ["official", "igo", "society", "academic_medical", "publisher"]
ADMIT_SET = ["official", "igo", "society", "academic_medical"]


def load_rules(path: Path = DEFAULT_RULES) -> dict:
    r = json.loads(Path(path).read_text())
    r["_re"] = {k: [re.compile(p) for p in v] for k, v in r["patterns"].items()}
    return r


def _named(host: str, names) -> str | None:
    for n in names:
        if host == n or host.endswith("." + n):
            return n
    return None


def country(host: str, rules: dict) -> str:
    tld = host.rsplit(".", 1)[-1]
    for c, tlds in rules["tld_country"].items():
        if tld in tlds:
            return c
    if host.endswith(".gov.uk") or host.endswith(".co.uk"):
        return "uk"
    if tld in ("int", "org", "com", "net", "info", "io"):
        return "intl"
    return "other"


def classify(host: str, rules: dict) -> tuple[str, str]:
    host = host.lower().strip(".")
    tld = host.rsplit(".", 1)[-1]
    commercial_tld = tld in COMMERCIAL_TLDS
    # 1. named hosts, in policy order (commercial first so a portal is never lifted by a keyword)
    for cat in rules["order"]:
        hit = _named(host, rules["named"].get(cat, []))
        if hit:
            return cat, f"named:{hit}"
    # 2. commercial wellness patterns
    for rx in rules["_re"]["commercial_wellness"]:
        if rx.search(host):
            return "commercial_wellness", f"pattern:{rx.pattern}"
    # 3. government TLD
    gov = [rx for rx in rules["_re"]["official_tld"] if rx.search(host)]
    if gov:
        kw = [k for k in rules["patterns"]["official_health_kw"] if k in host]
        if kw:
            return "official", f"gov-tld+health-kw:{kw[0]}"
        for rx in rules["_re"]["official_inst"]:
            if rx.search(host):
                return "official", f"gov-tld+inst:{rx.pattern}"
        return "gov_nonhealth", f"gov-tld:{gov[0].pattern}"
    # 4. other pattern categories (never on a commercial TLD)
    med = next((rx for rx in rules["_re"]["medical_token"] if rx.search(host)), None)
    inst = next((rx for rx in rules["_re"]["society_inst"] if rx.search(host)), None)
    cands: list[tuple[str, str]] = []
    for cat in ["igo", "society", "academic_medical", "publisher"]:
        for rx in rules["_re"][cat]:
            if rx.search(host):
                cands.append((cat, f"pattern:{rx.pattern}"))
                break
    if med and inst:
        cands.append(("society", "pattern:society-token+medical-token"))
    acad = next((rx for rx in rules["_re"]["academic_generic"] if rx.search(host)), None)
    if acad:
        cands.append(("academic_medical", "pattern:academic-tld+medical-token") if med
                     else ("academic_other", f"pattern:academic-no-medical-signal:{acad.pattern}"))
    for cat, basis in cands:
        if commercial_tld:
            return "unknown", f"{cat}-pattern on commercial tld"
        return cat, basis
    # 5. providers: out by default, any TLD
    for rx in rules["_re"]["hospital_provider"]:
        if rx.search(host):
            return "hospital_provider", f"pattern:{rx.pattern}"
    return "unknown", "no rule matched"


def vet_rows(roster: list[dict], rules: dict) -> list[dict]:
    out = []
    for x in roster:
        if x.get("class") != "health":
            continue
        cat, basis = classify(x["domain"], rules)
        out.append({"host": x["domain"], "category": cat, "basis": basis, "n_src": x.get("n_src", 0),
                    "endorsers": x.get("n_endorser_regs", 0), "class_basis": x.get("class_basis"),
                    "country": country(x["domain"].lower(), rules)})
    return out


def summarize(rows: list[dict], rows_per_host: int = 145, min_src: int = 5) -> dict:
    s: dict = {"rows_per_host": rows_per_host, "min_src": min_src, "total_hosts": len(rows), "categories": {}}
    for cat in CATEGORIES:
        rs = [r for r in rows if r["category"] == cat]
        big = [r for r in rs if r["n_src"] >= min_src]
        s["categories"][cat] = {
            "hosts": len(rs), "hosts_ge5": len(big),
            "est_rows": len(rs) * rows_per_host, "est_rows_ge5": len(big) * rows_per_host,
            "by_country": dict(collections.Counter(r["country"] for r in rs)),
            "by_country_ge5": dict(collections.Counter(r["country"] for r in big)),
            "top_basis": collections.Counter(r["basis"].split(":")[0] for r in rs).most_common(5),
        }
    adm = [r for r in rows if r["category"] in ADMIT_SET]
    s["recommended_admit"] = {"categories": ADMIT_SET, "hosts": len(adm),
                              "hosts_ge5": sum(r["n_src"] >= min_src for r in adm),
                              "est_rows": len(adm) * rows_per_host,
                              "est_rows_ge5": sum(r["n_src"] >= min_src for r in adm) * rows_per_host}
    s["unknown_share"] = round(s["categories"]["unknown"]["hosts"] / max(1, len(rows)), 4)
    return s


def main(argv=None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog)
    ap.add_argument("--roster", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rules", default=str(DEFAULT_RULES))
    a = ap.parse_args(argv)
    rules = load_rules(Path(a.rules))
    rows = vet_rows(json.loads(Path(a.roster).read_text()), rules)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "health-vet.v1.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    (out / "summary.json").write_text(json.dumps(summarize(rows), indent=1, sort_keys=True))
    rng = random.Random(991)
    samples = {c: rng.sample([r for r in rows if r["category"] == c],
                             min(25, sum(r["category"] == c for r in rows))) for c in CATEGORIES}
    samples["unknown_top40"] = sorted([r for r in rows if r["category"] == "unknown"],
                                      key=lambda r: (-r["n_src"], r["host"]))[:40]
    (out / "samples.json").write_text(json.dumps(samples, indent=1))
    print(json.dumps(summarize(rows)["recommended_admit"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
