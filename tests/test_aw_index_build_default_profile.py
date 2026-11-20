"""`cyt index build --pack` defaults to PROD_E9; `served` stays selectable by name (task 1271, ruling R1)."""
import pathlib

import pytest
from cytir.core import serving_profile as SP
from cytir.index import build as B


def _resolved(*extra):
    args = B._parser("build", "cyt index build").parse_args(
        ["--pack", "stage", "--qdrant", "qdrant-url", "--collection", "c", "--out", "r", *extra])
    return SP.get(args.profile or B.DEFAULT_PACK_PROFILE)


def test_unqualified_pack_build_is_the_prod_e9_shape():
    prof = _resolved()
    assert prof.name == "prod-e9" and prof.expected_shape() == SP.PROD_E9.expected_shape()
    assert prof.hnsw_m == 8 and prof.sparse_index_datatype == "uint8"      # not the arm's m=16 / float16


def test_served_still_resolves_to_the_old_shape_by_name():
    prof = _resolved("--profile", "served")
    assert prof.expected_shape() == SP.SERVED.expected_shape()
    assert prof.expected_shape() != SP.PROD_E9.expected_shape()


def test_both_names_are_public_and_the_help_names_the_default():
    assert set(B.PUBLIC_PROFILES) == {"prod-e9", "served"}
    helptext = B._parser("build", "cyt index build").format_help()
    assert "default prod-e9" in helptext and "pre-e9 shape" in helptext


def test_a_private_profile_name_is_refused_with_rc_3_at_the_cli_and_in_run(capsys):
    private = next(n for n in SP.PROFILES if n not in B.PUBLIC_PROFILES)
    rc = B.main(["--pack", "stage", "--qdrant", "http://q", "--collection", "c", "--out", "r",
                 "--profile", private])
    assert rc == B.EXIT_REFUSED == 3
    assert "REFUSED: unknown profile" in capsys.readouterr().err
    with pytest.raises(B.BuildRefusal, match="unknown profile"):
        B.run(mode="build", packs=[], url="http://q", collection="c", profile_name=private,
              out_dir=pathlib.Path("r"), resume=False, recreate=False, allow_remote=False, batch=1,
              as_of=None, consent_cache_dir=None, settle_timeout_s=1)
