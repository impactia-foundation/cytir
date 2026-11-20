"""The news discovery path must import without duckdb installed (it crashed a 12-month RTS run on a venv that has
no duckdb: sitemap_ingest -> cc_columnar imported it at module level)."""
import subprocess
import sys
import textwrap

MODULES = ["cytir.acquire.cc_columnar", "cytir.acquire.sitemap_ingest", "cytir.news.rts_section_discovery",
           "cytir.news.coverage_backfill"]


def test_news_discovery_modules_import_without_duckdb():
    code = textwrap.dedent(f"""
        import importlib, os, sys
        os.environ.setdefault("AW_ALLOW_UNIDENTIFIED_CRAWL", "1")
        class Block:
            def find_spec(self, name, path=None, target=None):
                if name == "duckdb" or name.startswith("duckdb."):
                    raise ModuleNotFoundError("No module named 'duckdb'", name="duckdb")
        sys.meta_path.insert(0, Block())
        sys.modules.pop("duckdb", None)
        for m in {MODULES!r}:
            importlib.import_module(m)
        assert "duckdb" not in sys.modules
        import cytir.acquire.cc_columnar as ccx
        try:
            ccx._con()
        except ModuleNotFoundError as e:
            assert e.name == "duckdb"      # the CC-index path still needs it, and says so when it runs
        else:
            raise SystemExit("_con() must need duckdb")
        print("OK")
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout + r.stderr
