"""Task 551 -- the two fusion-weights candidates the guard's refusal message names exist.

The shell-driver and stamper pins of task 551 pinned frozen task scripts; task 2544 moved
them to the task 551 script folder (never collected by CI). What
constrains the package stays here: both candidate files ship in cytir/serve/ and parse.
"""
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
CANDIDATES = ("fusion_weights.candidate519.json", "fusion_weights.candidate533.json")


def test_both_named_candidates_actually_exist_in_the_package():
    for name in CANDIDATES:
        path = REPO / "cytir" / "serve" / name
        assert path.is_file(), name
        assert isinstance(json.loads(path.read_text()), dict), name
