"""`Env.lane_value` (task 2121): order is env override, then the lane config file
(last assignment wins), then the LANE_DEFAULTS table; an unknown name is refused.
The table's equality with the lane script lives in the unshipped characterization test."""
from __future__ import annotations

import pytest

from cytir.sources.families import news_tier2_apply as T


def _env(config, **extra):
    environ = {"AW_RELEASE_DIR": "/release", "AW_NEWS_RELEASE_MANIFEST_SHA256": "deadbeef",
               "AW_NEWS_WORK": "/w", "AW_NEWS_LANGGATE_PY": "/py",
               "AW_QPY": "/py", "AW_PROD_LOCK_HOST": "lockhost",
               "AW_NEWS_CONFIG": str(config), **extra}
    return T.Env(environ)


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "current.env"
    path.write_text("AW_NEWS_RETAIN_MONTHS=24\nAW_NEWS_RETAIN_MONTHS=30\n", encoding="utf-8")
    return path


def test_environment_override_wins(config):
    assert _env(config, AW_NEWS_RETAIN_MONTHS="99").lane_value("AW_NEWS_RETAIN_MONTHS") == "99"


def test_config_file_beats_table_and_last_assignment_wins(config):
    assert _env(config).lane_value("AW_NEWS_RETAIN_MONTHS") == "30"


def test_table_is_the_last_fallback(config):
    env = _env(config)
    for name, value in T.LANE_DEFAULTS.items():
        if name != "AW_NEWS_RETAIN_MONTHS":
            assert env.lane_value(name) == value


def test_table_values():
    assert T.LANE_DEFAULTS == {
        "AW_NEWS_FRESH_MONTHS": "12",
        "AW_NEWS_RETAIN_MONTHS": "36",
        "AW_RECENT_CAP": "1500",
        "AW_NEWS_RESUME": "0",
    }


def test_unknown_name_is_refused(config):
    with pytest.raises(SystemExit, match="AW_NO_SUCH_KNOB"):
        _env(config).lane_value("AW_NO_SUCH_KNOB")
