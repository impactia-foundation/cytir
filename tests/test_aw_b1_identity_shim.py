# Task 646 S2: aw_b1_identity moved to cytir/core/ids.py; aw_b1_identity.py
# is now a shim (`from cytir.core.ids import *`). Proves the shim re-exports the
# SAME objects (not a copy) and that the worked-example self-test still
# passes byte-exactly through the old import path.
import cytir.core.ids as ids
import cytir.core.ids as ID

_REEXPORTED_NAMES = (
    "AW_NAMESPACE",
    "EPOCHS",
    "LAW_SUB_SEPARATOR",
    "NATIVE_ANCHOR_FIELDS",
    "RESOLVED_COMPONENT_FIELDS",
    "RULE_LEGEND",
    "classify_tier",
    "content_digest",
    "has_source_native_anchor",
    "is_positional_anchor",
    "law_sub_anchor_component",
    "mint_point_id",
    "normalize_digest_text",
    "salted_anchor_component",
    "self_test",
    "source_native_anchor",
    "tier_b_fallback_digest_input",
)


def test_shim_reexports_are_the_same_objects():
    for name in _REEXPORTED_NAMES:
        a, b = getattr(ID, name), getattr(ids, name)
        if callable(a):
            assert a is b, (
                f"aw_b1_identity.{name} is not cytir.core.ids.{name} -- the "
                f"shim has diverged into a second copy")
        else:
            assert a == b, f"aw_b1_identity.{name} != cytir.core.ids.{name}"


def test_worked_examples_still_pass_through_the_shim():
    ID.self_test()   # raises on any mismatch; identity.md §6 worked examples


def test_point_id_is_identical_through_either_import_path():
    a = ID.mint_point_id("https://fedlex.admin.ch/eli/cc/1999/404/de", "art_1")
    b = ids.mint_point_id("https://fedlex.admin.ch/eli/cc/1999/404/de", "art_1")
    assert a == b
