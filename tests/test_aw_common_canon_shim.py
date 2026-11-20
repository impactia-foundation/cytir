# Task 646 S2: `canon` moved from aw_common.py to cytir/core/canon.py.
# aw_common.py re-exports these names rather than re-implementing them --
# this is the "mirror byte-identity test becomes an import-identity test"
# acceptance line from 643-open-source-readiness-review.md §6.2 S2,
# applied to aw_common's canon slice (the aw_b1_identity <-> cytir.core.ids
# version of the same proof is
# test_aw_b1_identity_shim.py::test_shim_reexports_are_the_same_objects).
import cytir.core.canon as canon
import cytir.core.common as AC

_REEXPORTED_NAMES = (
    "canonicalize_url",
    "canonical_url",
    "credential_url_hit_count",
    "fold_www",
    "normalize_register_id",
    "reset_credential_url_hit_count",
    "strip_tracking_url",
)


def test_shim_reexports_are_the_same_objects():
    for name in _REEXPORTED_NAMES:
        assert getattr(AC, name) is getattr(canon, name), (
            f"aw_common.{name} is not cytir.core.canon.{name} -- the shim has "
            f"diverged into a second copy")


def test_no_leftover_private_canon_helpers_in_aw_common():
    # These lived in aw_common.py before the move; if any reappear here it is
    # a fork (a second, locally-edited copy), not a re-export.
    leftovers = ("_TRACKING_KEYS", "_TRACKING_PREFIXES", "_PATH_SESSION",
                 "_is_tracking_key", "_CREDENTIAL_PARAM_NAMES", "_JWT_LIKE_RE",
                 "_is_credential_param", "_scrub_credential_params",
                 "_PY_VER", "_PY_LOCALE", "_WWW_RE")
    present = [n for n in leftovers if hasattr(AC, n)]
    assert present == [], f"aw_common still defines {present} locally"
