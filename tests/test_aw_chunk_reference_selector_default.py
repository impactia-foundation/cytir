"""Task 2070: the default selector of `cyt chunk reference-select` is v4, the one the public cut ships."""
from __future__ import annotations

import json
import pathlib

from cytir.chunk import reference_selector as RS

APP = pathlib.Path(__file__).resolve().parent.parent
V4 = "data/chunk-reference-selector.v4.json"


def test_default_selector_is_v4_and_exists():
    assert pathlib.Path(RS.DEFAULT_CONFIG).name == pathlib.Path(V4).name
    assert pathlib.Path(RS.DEFAULT_CONFIG).is_file()


def test_default_selector_is_in_the_package_data_and_the_cut():
    name = pathlib.Path(V4).name
    assert f'"{name}"' in (APP / "pyproject.toml").read_text("utf-8")
    cut = (APP / "data" / "public-cut.v1.json").read_text("utf-8")
    assert V4 in json.dumps(json.loads(cut))


def test_default_selector_loads():
    assert RS.Selector().config_path.name == pathlib.Path(V4).name
