"""P0 eval gate (`cyt gate p0-eval`): URL normalisation, hit-rank rule, main() with a fake client."""
from cytir.gates import p0_eval
from cytir.gates.p0_eval import hit_rank, norm


def test_hit_rank_prefix_match_anchor():
    assert hit_rank("https://docs.python.org/3/library/os.html",
                    [{"url": "https://docs.python.org/3/library/os.html#os.listdir"}]) == 1


def test_hit_rank_miss_is_none():
    assert hit_rank("https://x.org/a", [{"url": "https://y.org/b"}]) is None


def test_norm_strips_trailing_slash_and_lowercases_host():
    assert norm("https://A.org/p/") == "a.org/p"


def test_main_counts_errors_with_fake_client(tmp_path, monkeypatch, capsys):
    ev = tmp_path / "e.yaml"
    ev.write_text("eval:\n- {id: q1, question: a, target_url: 'https://x.org/a', lang: en}\n"
                  "- {id: q2, question: b, target_url: 'https://x.org/b', lang: en}\n")

    class R:
        def __init__(self, ok):
            self.ok = ok

        def raise_for_status(self):
            if not self.ok:
                raise RuntimeError("503")

        def json(self):
            return {"results": [{"url": "https://x.org/a"}]}

    calls = []

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self, url, params):
            calls.append(url)
            return R(params["q"] == "a")

    monkeypatch.setattr(p0_eval, "make_client", lambda spec: FakeClient())
    p0_eval.main(["--eval", str(ev), "--api", "http://fake", "--k", "5"])
    out = capsys.readouterr().out
    assert "hit@1 : 1/2" in out and "errors: 1/2" in out
    assert all(c.endswith("/search") for c in calls)
