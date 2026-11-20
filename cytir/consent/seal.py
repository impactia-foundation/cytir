"""`cyt consent seal` -- the package sealer for `aw.pool-a-consent.v1` (task 738).

Task 629 P7 produced the sealed per-host probe file with exploration scripts on the corpus box
(`run_consent.py` -> `patch_terminal.py` -> `host_census.py`, hand-assembled). This module is
the package's one way to do it again. The scripts were read as the SPEC of the format and
re-implemented here; none is executed.

What a sealed file is (the shape `cytir.consent.derive.load_sealed_probes` reads):

* `host_consent_resolution[host]` -- the resolver record, finalised: `origin`/`origin_scheme`
  stamped from the probed transport, `gate_bucket` (legacy both-flags), `gate_bucket_legacy_both_flags`,
  `gate_bucket_search_indexing` and the direct-terminal verdict at the root, under
  `terminal_schema_version`.
* `family_verdicts[family] = {verdict, reason, hosts}` -- `hosts` is what ties a host to a family.
* provenance: probe vantage (hostname, egress IP), operator identity, the shas of the resolver
  files used, the settings.

Rules kept: probes run on the corpus box, declared with `--probe-box NAME` (the maintainer
machine is not a vantage: the declared name must equal the machine's own hostname, so a run
cannot be labelled with a box it did not run on), `direct_crawl=True` (we fetch ourselves), module politeness defaults
unchanged, a fresh empty cache dir so every host is a live first resolution, and a CCBot or
training-token disallow prices `train_ok=False` only (ADR-024 am.5, enforced by the resolver,
not here). An unreadable robots (403/418/5xx) is recorded as what the resolver says; this
module never turns it into a decline.

The roster and conservation sections (`family_origin_roster`, `pack_to_origin_conservation`)
describe which packs' rows sit behind which host. They come from a pack census over shards
on the corpus box, which this module does not repeat; `--carry-from PREV` carries them
forward for the sealed families, marked `carried_forward_from` with the previous file's sha256.

Seal line: sha256 of the written file + host count, printed and written to `<out>.seal`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import socket
import sys
import tempfile
from datetime import datetime, timezone

SCHEMA = "aw.pool-a-consent.v1"
REPO = pathlib.Path(__file__).resolve().parents[2]
#: the files whose bytes decide a verdict; their shas go into the seal
RESOLVER_FILES = ("cytir/consent/resolver.py", "cytir/core/user_agent.py",
                  "cytir/consent/terminal.py", "cytir/consent/policy.py")
ADMISSIBLE_BUCKETS = ("ok", "ok_search_only")
_REASON_BOTH = "search_ok=true and train_ok=true at every host in the family's origin set"


class SealRefused(ValueError):
    """The sealer refuses to write a file that would not be evidence."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def finalize_host_record(raw: dict) -> dict:
    """The patched record of task 629's `patch_terminal.py`, no network: stamp the origin from
    the transport actually probed, then the three gate buckets and the root terminal verdict."""
    from cytir.consent import policy as CP
    from cytir.consent import resolver as CR
    from cytir.consent import terminal as DTC

    rec = {k: v for k, v in raw.items() if k not in ("from_cache", "_robots_body")}
    host = rec.get("host")
    sig = rec.get("signals") or {}
    robots_t = (sig.get("robots") or {}).get("transport")
    tdm_t = (sig.get("tdmrep") or {}).get("transport")
    scheme = robots_t or tdm_t
    if scheme not in ("http", "https"):
        raise SealRefused(f"{host}: no recorded transport scheme, cannot stamp the origin")
    if robots_t and tdm_t and robots_t != tdm_t:
        raise SealRefused(f"{host}: robots/tdmrep probed different schemes "
                          f"({robots_t} vs {tdm_t}); not collapsed silently")
    rec["origin_scheme"] = scheme
    rec["origin"] = f"{scheme}://{host}"
    rec["gate_bucket"] = CR.bucket(rec)
    rec["gate_bucket_legacy_both_flags"] = rec["gate_bucket"]
    rec["gate_bucket_search_indexing"] = CP.gate_bucket(rec, search_indexing=True)
    try:
        terminal = DTC.consent_terminal(rec, rec["origin"],
                                        schema_version=DTC.CURRENT_SCHEMA_VERSION)
    except DTC.DirectTerminalContractError as exc:
        raise SealRefused(f"{host}: direct-terminal contract error: {exc}") from exc
    rec["direct_terminal"] = ({"status": "allowed_for_fetch_at_root", "reason": None}
                              if terminal is None
                              else {"status": terminal[0], "reason": terminal[1]})
    rec["terminal_schema_version"] = DTC.CURRENT_SCHEMA_VERSION
    return dict(sorted(rec.items()))


def family_verdict(hosts: list, records: dict) -> dict:
    """admissible / held for one family from its hosts' finalised records."""
    bad, train_off = [], []
    for h in hosts:
        r = records[h]
        term = r["direct_terminal"]
        if r["gate_bucket_search_indexing"] not in ADMISSIBLE_BUCKETS \
                or term["status"] != "allowed_for_fetch_at_root":
            bad.append(f"{h} gate_bucket={r['gate_bucket_search_indexing']} direct_terminal={term}")
        elif r["train_ok"] is not True:
            train_off.append(h)
    if bad:
        shown = [b.split(" ", 1)[0] for b in bad]
        return {"verdict": "held",
                "reason": f"host(s) {shown} not search-admissible: " + "; ".join(bad),
                "hosts": list(hosts)}
    if train_off:
        return {"verdict": "admissible",
                "reason": (f"search_ok=true at every host; train_ok=false at {train_off} "
                           "(CCBot/training-only-token disallow, ADR-024 am.5) -- admit for "
                           "search/retrieval only, excluded from any training export"),
                "hosts": list(hosts)}
    return {"verdict": "admissible", "reason": _REASON_BOTH, "hosts": list(hosts)}


def resolver_shas(root=REPO) -> dict:
    return {rel: sha256_file(pathlib.Path(root) / rel) for rel in RESOLVER_FILES}


def require_declared_vantage(hostname: str, declared: str | None) -> None:
    """The probe box is declared by the operator and must be the machine this runs on."""
    if not declared or declared.lower().split(".")[0] != hostname.lower().split(".")[0]:
        raise SealRefused(f"probes run from the corpus box (memory consent-probe-vantage): "
                          f"--probe-box {declared!r} does not match this machine {hostname!r}")


def build(*, family_hosts: dict, raw_records: dict, task: int, precondition: str,
          sealed_at: str, hostname: str, egress_ip: str | None, operator: dict,
          shas: dict, settings_used: str, probe_box: str | None, carry: dict | None = None,
          carry_sha: str | None = None, carry_path: str | None = None) -> dict:
    """Pure assembly. `raw_records` are resolver records; each listed host must have one."""
    require_declared_vantage(hostname, probe_box)
    hosts = sorted({h for hs in family_hosts.values() for h in hs})
    missing = [h for h in hosts if h not in raw_records]
    if missing:
        raise SealRefused(f"no resolver record for {missing}")
    records = {h: finalize_host_record(raw_records[h]) for h in hosts}
    out = {
        "schema": SCHEMA, "task": task, "precondition": precondition,
        "sealed_at_utc": sealed_at,
        "probe_vantage": {
            "hostname": hostname, "egress_ip": egress_ip,
            "note": "all live network probing (robots.txt, .well-known/tdmrep.json) ran on "
                    "the named corpus box, per memory consent-probe-vantage"},
        "operator_identity": operator,
        "resolver": {
            "module": "cytir/consent/resolver.py",
            "entry_point": "cytir.consent.resolver.resolve_hosts(hosts, cache_dir=<fresh>, direct_crawl=True)",
            "sealer": "cytir/consent/seal.py (aw consent seal)",  # rename-codemod: keep (persisted: provenance written into the sealed consent record)
            "sha256": shas, "settings_used": settings_used,
            "search_admission_gate": "cytir.consent.policy.gate_bucket(rec, search_indexing=True)",
            "direct_terminal_gate": "cytir.consent.terminal.consent_terminal at the origin root"},
        "terminal_schema_version": next(iter(records.values()))["terminal_schema_version"],
    }
    if carry is not None:
        fams = set(family_hosts)
        roster = {f: v for f, v in (carry.get("family_origin_roster") or {}).items() if f in fams}
        cons = carry.get("pack_to_origin_conservation") or {}
        packs = [p for p in cons.get("packs", []) if p.get("family") in fams]
        out["carried_forward_from"] = {
            "path": carry_path, "sha256": carry_sha,
            "what": "family_origin_roster and pack_to_origin_conservation, for the sealed "
                    "families only, copied unchanged: which pack rows sit behind which host "
                    "is a fact about the packs, not about the probe, and was not recounted"}
        out["pack_to_origin_conservation"] = {
            "description": cons.get("description"),
            "total_expected_rows": sum(p["expected"] for p in packs),
            "total_actual_rows": sum(p["actual"] for p in packs),
            "all_packs_match": all(p.get("match") for p in packs), "packs": packs}
        out["family_origin_roster"] = roster
    out["host_consent_resolution"] = records
    out["family_verdicts"] = {f: family_verdict(sorted_keep(hs), records)
                              for f, hs in sorted(family_hosts.items())}
    return out


def sorted_keep(hosts) -> list:
    """Host order as given (the previous file's order), de-duplicated."""
    seen, out = set(), []
    for h in hosts:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


def dumps(doc: dict) -> str:
    return json.dumps(doc, indent=1, ensure_ascii=False) + "\n"


def write_sealed(doc: dict, out_path) -> str:
    out_path = pathlib.Path(out_path)
    if out_path.exists():
        raise SealRefused(f"{out_path} exists; a sealed file is never overwritten")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(dumps(doc), encoding="utf-8")
    sha = sha256_file(out_path)
    n = len(doc["host_consent_resolution"])
    line = f"{sha}  {out_path.name}  hosts={n} families={len(doc['family_verdicts'])}"
    pathlib.Path(str(out_path) + ".seal").write_text(line + "\n", encoding="utf-8")
    return line


def operator_identity() -> dict:
    from cytir.consent import resolver as CR
    return {"source": "cytir.core.user_agent.user_agent() (operator.json / env, the identity every "
                      "fetch presents)", "ua_string": CR.UA_STR}


def main(argv=None, prog="cyt consent seal") -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n")[0])
    ap.add_argument("--family-hosts-from", required=True, metavar="SEALED_JSON",
                    help="a previous sealed file: each --family's hosts come from its family_verdicts")
    ap.add_argument("--family", action="append", required=True,
                    help="family id to seal (repeatable)")
    ap.add_argument("--out", required=True, help="new sealed file (must not exist)")
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--precondition", default="r6-refresh")
    ap.add_argument("--egress-ip", default=None, help="this box's public egress IP, recorded")
    ap.add_argument("--carry-from", default=None, metavar="SEALED_JSON",
                    help="carry roster+conservation for the sealed families from this file")
    ap.add_argument("--probe-box", required=True,
                    help="the corpus box you are running on (must match its hostname)")
    args = ap.parse_args(argv)

    hostname = socket.gethostname()
    require_declared_vantage(hostname, args.probe_box)
    prev = json.loads(pathlib.Path(args.family_hosts_from).read_text())
    family_hosts = {}
    for f in args.family:
        hs = (prev.get("family_verdicts") or {}).get(f, {}).get("hosts")
        if not hs:
            raise SealRefused(f"{f}: no hosts in {args.family_hosts_from}")
        family_hosts[f] = hs
    hosts = sorted({h for hs in family_hosts.values() for h in hs})

    from cytir.consent import resolver as CR
    with tempfile.TemporaryDirectory(prefix="aw-seal-cache-") as cache:
        raw = CR.resolve_hosts(hosts, cache_dir=cache, direct_crawl=True)
    carry = carry_sha = None
    if args.carry_from:
        carry = json.loads(pathlib.Path(args.carry_from).read_text())
        carry_sha = sha256_file(args.carry_from)
    doc = build(
        family_hosts=family_hosts, raw_records=raw, task=args.task,
        precondition=args.precondition, sealed_at=utc_now(), hostname=hostname,
        egress_ip=args.egress_ip, probe_box=args.probe_box, operator=operator_identity(), shas=resolver_shas(),
        settings_used=(f"module defaults, unmodified: concurrency={CR.CONCURRENCY}, "
                       f"HOST_DELAY={CR.HOST_DELAY}s, TIMEOUT={CR.TIMEOUT}s, one pass"),
        carry=carry, carry_sha=carry_sha, carry_path=args.carry_from)
    print(write_sealed(doc, args.out))
    for h, r in sorted(doc["host_consent_resolution"].items()):
        print(f"{h}: status={r['status']} search_ok={r['search_ok']} train_ok={r['train_ok']} "
              f"bucket={r['gate_bucket_search_indexing']} terminal={r['direct_terminal']['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
