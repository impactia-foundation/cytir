"""The serving process's typed settings and `GET /config` (S10 step 2, task 1071).

Contracts:
  1. One declaration per `AW_*` name: no serving module but the tokenizer reads the environment.
  2. The 44 fixed knobs are module constants: setting their old env names changes nothing.
  3. Every name prod's compose env (aw-api, every overlay, aw-mcp) and the shipped default env set
     is a declared field; an unknown `AW_*` name is one WARNING line, never a refusal.
  4. The server imports and `/config` answers with no `AW_*` set and with prod's declared env.
  5. A module purge re-reads the settings (spec risk R1); `settings` imports no server module.
  6. `/config` never echoes a secret: no field names one, URL userinfo and queries are redacted.
"""
from __future__ import annotations

import ast
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest

from cytir.ops import prod_env
from cytir.ops import serve_settings_table as T
from cytir.serve import settings as S

ROOT = pathlib.Path(__file__).resolve().parent.parent
SERVER = ROOT / "server"
SRV = ROOT / "cytir" / "serve"


def _settings():
    return T.load_settings_module(SRV)


def _run(code: str, env: dict) -> subprocess.CompletedProcess:
    base = {k: v for k, v in os.environ.items() if not k.startswith(("AW_", "CYTIR_"))}
    base.update(env)
    base["PYTHONPATH"] = os.pathsep.join([str(SERVER), str(ROOT)])
    return subprocess.run([sys.executable, "-c", code], env=base, capture_output=True, text=True,
                          timeout=300, cwd=str(ROOT))


def _json_line(cp: subprocess.CompletedProcess) -> dict:
    assert cp.returncode == 0, cp.stderr[-2000:]
    return json.loads([ln for ln in cp.stdout.splitlines() if ln[:1] in "{["][-1])


def _prod_env(tree: pathlib.Path = ROOT) -> dict:
    env = prod_env.default_env().env
    out = {}
    for k, v in env.items():
        if v.startswith("/app/cytir/serve/"):
            v = str(SRV / v[len("/app/cytir/serve/"):])
        out[k] = v
    out.pop("AW_MODELS_DIR", None)           # a container path; the server never opens it
    return out


def test_only_the_tokenizer_and_settings_touch_the_environment():
    for p in sorted(SRV.glob("*.py")):
        if p.name.startswith("test_") or p.name in ("schema.py", "settings.py"):
            continue
        src = p.read_text(encoding="utf-8")
        for n in ast.walk(ast.parse(src)):
            if isinstance(n, ast.Attribute) and ast.unparse(n) in ("os.environ", "os.getenv"):
                pytest.fail(f"{p.name}:{n.lineno} reads the environment; declare a settings field")


def test_every_declared_field_has_a_kind_a_type_and_a_unique_name():
    st = _settings()
    names = [m["env"] for _, _, m in st.specs() if m["env"]]
    assert len(names) == len(set(names))
    for g, a, m in st.specs():
        assert m["kind"] in st._PY_TYPE, (g, a)
        assert m["env"] or m["weights"], f"{g}.{a}: neither an env name nor a weights key"
        assert m["why"], f"{g}.{a}: no why"
    assert sum(1 for _, _, m in st.specs() if m["weights"]) == 12


def test_settings_is_a_leaf_module():
    tree = ast.parse((SRV / "settings.py").read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            # the allowed edges: cytir.core.paths and cytir.core.envnames (both stdlib-only) names the build-side env (task 1792)
            if n.module == "cytir.core":
                assert [a.name for a in n.names] in (["paths"], ["envnames"]), ast.unparse(n)
                continue
            assert n.level == 0 and not (n.module or "").startswith(("aw_server", "cytir")), ast.unparse(n)


def test_parsing_keeps_todays_semantics():
    st = _settings()
    meta = {g + "." + a: m for g, a, m in st.specs()}
    flag, num, trust = meta["fusion.primacy"], meta["fusion.primacy_keep"], meta["trust.trust_mode"]
    assert st._parse(flag, {"AW_PRIMACY": "true"}, {}, ())[0] is False      # only "1" is on
    assert st._parse(flag, {"AW_PRIMACY": "1"}, {}, ())[0] is True
    with pytest.raises(ValueError):
        st._parse(num, {"AW_PRIMACY_KEEP": ""}, {}, ())                     # set-but-empty: as before
    assert st._parse(trust, {"AW_TRUST_MODE": ""}, {"trust_mode": "gated"}, {"trust_mode"}) == \
        ("gated", "weights:trust_mode")                                     # empty = unset here
    s = st.Settings.load({"AW_PRIMACY": "yes"})
    assert s.fusion.primacy is False and any("AW_PRIMACY" in w for w in s.parse_warnings)


def test_the_tokenizer_mirror_equals_what_schema_parsed():
    cp = _run("import json\nfrom cytir.serve import schema, settings as s\nS = s.SETTINGS\n"
              "print(json.dumps({'a': [schema.COLLECTION, schema._FOLD_ACCENTS, schema._QSTOP,"
              " schema._QSTOP_IDF, schema._QSTOP_MINTOKENS, schema._IDF_TABLE], 'b': [S.runtime.collection,"
              " S.query.fold_accents, S.query.bm25_qstop, S.query.bm25_qstop_idf, S.query.bm25_qstop_mintokens,"
              " S.runtime.bm25_idf_table]}))",
              {"AW_FOLD_ACCENTS": " Yes ", "AW_BM25_QSTOP_IDF": "2.5", "AW_COLLECTION": "aw_x"})
    d = _json_line(cp)
    assert d["a"] == d["b"]


def _fixed() -> dict:
    sources = {p.name: p.read_text(encoding="utf-8") for p in SRV.glob("*.py") if not p.name.startswith("test_")}
    return T.fixed_constants(sources)


def test_the_fixed_knobs_are_constants_that_ignore_their_old_env_names():
    fixed = _fixed()
    assert len(fixed) == 41  # 38 (step 2) + 6 (step 3, task 1079) - 3 (task 1708: the inert denseboost trio)
    code = ("import json, importlib\nfrom cytir.serve import retrieval, authority\n"
            "class _M(dict):\n    def __missing__(self, f):\n        return importlib.import_module('cytir.serve.' + f[:-3])\n"
            "mods = _M({'retrieval.py': retrieval, 'authority.py': authority})\n"
            f"fixed = {json.dumps({k: [v['file'], v['target']] for k, v in fixed.items()})}\n"
            "print(json.dumps({k: repr(getattr(mods[f], t)) for k, (f, t) in fixed.items()}))")
    quiet = _json_line(_run(code, {}))
    perturb = {}
    for k, v in fixed.items():
        val = v["value"]
        perturb[k] = ("0" if val else "1") if isinstance(val, bool) else "zzz" if isinstance(val, (str, frozenset)) else "7"
    loud = _json_line(_run(code, perturb))
    assert quiet == loud
    assert quiet == {k: repr(v["value"]) for k, v in fixed.items()}


def test_every_name_prod_sets_is_declared():
    st = _settings()
    declared = set(st.ENV_NAMES)
    names = set(prod_env.default_env().env)
    compose = ROOT / "deploy" / "docker-compose.yaml"
    if compose.is_file():
        import yaml
        names |= set(prod_env.resolve_env().env)
        doc = yaml.safe_load(compose.read_text(encoding="utf-8"))
        names |= set(doc["services"]["aw-mcp"].get("environment") or {})
    missing = sorted(n for n in names if n.startswith("AW_") and n not in declared)
    assert not missing, f"prod sets names no settings field declares: {missing}"


def test_cold_start_with_no_aw_env_answers_config_from_defaults():
    code = ("import json\nfrom fastapi.testclient import TestClient\nfrom cytir.serve import app\n"
            "r = TestClient(app.api).get('/config')\nd = r.json()\n"
            "print(json.dumps({'status': r.status_code, 'schema': d['schema'], 'unknown': d['unknown_env'],"
            " 'warn': d['parse_warnings'], 'sources': sorted({f['source'] for f in d['fields'].values()}),"
            " 'n': len(d['fields']), 'arts': sorted(d['artifacts']), 'tree': d['tree']}))")
    d = _json_line(_run(code, {}))
    assert d["status"] == 200 and d["schema"] == "aw.serve-config.v1"
    assert d["unknown"] == [] and d["warn"] == []
    assert all(s == "default" or s.startswith("weights:") for s in d["sources"]), d["sources"]
    assert d["n"] == len(_settings().specs()) and "fusion_weights" in d["arts"]
    assert d["tree"] == {"sha": None, "file": "SERVING_RELEASE"}, "a checkout has no rendered release sha"


def test_prods_declared_env_parses_with_no_unknowns_and_no_warnings():
    env = _prod_env()
    code = ("import json\nfrom fastapi.testclient import TestClient\nfrom cytir.serve import app\n"
            "d = TestClient(app.api).get('/config').json()\n"
            "print(json.dumps({'unknown': d['unknown_env'], 'warn': d['parse_warnings'],"
            " 'env': sorted(f['env'] for f in d['fields'].values() if f['source'] == 'env'),"
            " 'loaded': {k: v['loaded'] for k, v in d['artifacts'].items()}}))")
    d = _json_line(_run(code, env))
    assert d["unknown"] == [] and d["warn"] == []
    assert set(d["env"]) == set(env)
    assert d["loaded"]["fusion_weights"] and d["loaded"]["hub_hosts"]


def test_an_unknown_name_is_one_warning_not_a_refusal():
    cp = _run("from cytir.serve import settings\nprint('{\"ok\": true}')", {"AW_NO_SUCH_KNOB": "1"})
    assert _json_line(cp) == {"ok": True}
    lines = [ln for ln in cp.stderr.splitlines() if "AW_NO_SUCH_KNOB" in ln]
    assert len(lines) == 1


def test_a_module_purge_rereads_the_settings():
    code = ("import os, sys, json, importlib\nfrom cytir.serve import retrieval\na = retrieval._PAGECAP\n"
            "os.environ['AW_PAGECAP'] = '3'\n"
            "for m in [m for m in list(sys.modules) if m.startswith('cytir.serve')]: del sys.modules[m]\n"
            "r = importlib.import_module('cytir.serve.retrieval')\n"
            "print(json.dumps([a, r._PAGECAP, r.SETTINGS.legs.pagecap, r.effective_config()['pagecap']['cap']]))")
    assert _json_line(_run(code, {})) == [0, 3, 3, 3]


def test_health_config_block_keeps_its_keys():
    code = ("import json\nfrom cytir.serve import retrieval\nprint(json.dumps(sorted(retrieval.effective_config())))")
    assert _json_line(_run(code, {})) == sorted(
        ["collection", "mrl_dim", "dense_rescore", "dense_oversampling", "dense_ignore_quant", "rankhead",
         "rankhead_v2", "primacy", "pagecap", "refcanon", "bm25_zero_guard", "version_match", "fusion_weights"])


def test_config_echoes_no_secret():
    st = _settings()
    secret = re.compile(r"(?i)(secret|passw|api_?key|credential|bearer|access_token|auth_token)")
    assert not [m["env"] for _, _, m in st.specs() if m["env"] and secret.search(m["env"])]
    assert st.redact("https://user:pw@q.example:6333/x?api_key=1") == "https://q.example:6333/x"
    assert st.redact("http://u:p@h?token=x") == "http://h"
    doc = st.config_document(st.Settings.load({"AW_QDRANT_URL": "http://u:p@h:1/?t=2"}),
                             env={"AW_QDRANT_URL": "http://u:p@h:1/?t=2"})
    assert doc["fields"]["runtime.qdrant_url"]["value"] == "http://h:1/"
    assert "p@" not in json.dumps(doc)


def test_the_proof_table_classifies_a_read(tmp_path):
    src = ('import os\nX = float(os.environ.get("AW_A", "0.5"))\nY = os.environ.get("AW_B", "0") == "1"\n'
           'Z = os.environ.get("AW_C") or WEIGHTS.get("c", "d")\n')
    rows = {r["env"]: r for r in T.read_sites({"m.py": src})}
    assert (rows["AW_A"]["kind"], rows["AW_B"]["kind"], rows["AW_C"]["kind"]) == ("float", "flag", "str_or_weights")
    assert rows["AW_C"]["weights"] == "c"
    fixed = T.fixed_constants({"m.py": "Q = 3  # fixed: was env AW_Q (S10 D5: no product setter)\n"})
    assert fixed == {"AW_Q": {"file": "m.py", "target": "Q", "value": 3}}


def test_step3_frozen_names_are_no_longer_read_and_are_echoed():
    st = _settings()
    gone = {"AW_COUNTRY_BOOST", "AW_HUBFLOOR_DF_MIN", "AW_HUBFLOOR_MAXTRUST",
            "AW_REQTERM_MODE", "AW_REQTERM_DFMAX", "AW_REQTERM_MAXQ"}
    assert set(st.FROZEN) == gone
    assert not gone & set(st.ENV_NAMES)
    assert gone <= set(_fixed())
    doc = st.config_document(env={"AW_COUNTRY_BOOST": "9"}, artifacts={})
    fz = {f["env"]: f for f in doc["frozen"]}
    assert set(fz) == gone and fz["AW_COUNTRY_BOOST"]["set_in_env"] is True
    assert fz["AW_COUNTRY_BOOST"]["value"] == 0.6 and not fz["AW_HUBFLOOR_DF_MIN"]["set_in_env"]


def test_config_names_the_tree_sha_frozen_at_startup(tmp_path):
    import dataclasses
    st = _settings()
    f = tmp_path / "SERVING_RELEASE"
    assert st._read_serving_release(str(f)) is None                    # absent: None, never a guess
    f.write_text("\n")
    assert st._read_serving_release(str(f)) is None
    f.write_text("0123abcd\n")
    assert st._read_serving_release(str(f)) == "0123abcd"
    s = dataclasses.replace(st.Settings.load({}), serving_release="0123abcd")
    f.write_text("changed-after-startup\n")
    assert st.config_document(s, env={})["tree"] == {"sha": "0123abcd", "file": "SERVING_RELEASE"}


_SCHEMA_KNOBS = ("AW_COLLECTION", "AW_FOLD_ACCENTS", "AW_BM25_QSTOP", "AW_BM25_QSTOP_IDF",
                 "AW_BM25_QSTOP_MINTOKENS", "AW_BM25_IDF_TABLE", "AW_TRUSTFLOOR_ABS")
_SCHEMA_PROBE = ("import json\nfrom cytir.serve import schema, settings as s\nS = s.SETTINGS\n"
                 "print(json.dumps({'a': [schema.COLLECTION, schema._FOLD_ACCENTS, schema._QSTOP,"
                 " schema._QSTOP_IDF, schema._QSTOP_MINTOKENS, schema._IDF_TABLE,"
                 " s.current('AW_TRUSTFLOOR_ABS')], 'b': [S.runtime.collection, S.query.fold_accents,"
                 " S.query.bm25_qstop, S.query.bm25_qstop_idf, S.query.bm25_qstop_mintokens,"
                 " S.runtime.bm25_idf_table, S.trust.trustfloor_abs or False]}))")


def test_the_seven_schema_knobs_are_declared_and_schema_reads_no_environ():
    assert set(_SCHEMA_KNOBS) <= S.ENV_NAMES
    src = (pathlib.Path(S.__file__).parent / "schema.py").read_text(encoding="utf-8")
    assert "os.environ" not in src


def test_schema_values_equal_settings_values_by_default_and_when_set():
    d = _json_line(_run(_SCHEMA_PROBE, {}))
    assert d["a"][:6] == d["b"][:6] and d["a"][2] is False
    d = _json_line(_run(_SCHEMA_PROBE, {"AW_FOLD_ACCENTS": "1", "AW_BM25_QSTOP": "yes",
                                        "AW_BM25_QSTOP_IDF": "2.5", "AW_BM25_QSTOP_MINTOKENS": "7",
                                        "AW_COLLECTION": "aw_x", "AW_BM25_IDF_TABLE": "/x/t.json",
                                        "AW_TRUSTFLOOR_ABS": "true"}))
    assert d["a"][:6] == d["b"][:6]
    assert d["a"][:6] == ["aw_x", True, True, 2.5, 7, "/x/t.json"]
    assert d["a"][6] is False and d["b"][6] is False   # one reading: a FLAG, "true" is OFF on both sides
    d = _json_line(_run(_SCHEMA_PROBE, {"AW_TRUSTFLOOR_ABS": "1"}))
    assert d["a"][6] is True and d["b"][6] is True


def test_build_side_package_names_are_not_unread_but_unknown_names_are():
    """task 1792: AW_WORK_ROOT is a package setting (cytir.core.paths), so the server hygiene check does not flag it."""
    S = _settings()
    env = {"AW_WORK_ROOT": "/tmp/x", "AW_TOTALLY_UNKNOWN": "1"}
    assert S.Settings.load(env).unknown_env == ("AW_TOTALLY_UNKNOWN",)
