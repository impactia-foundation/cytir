"""Where the serving package lives inside any source tree (S10 step 9, task 1130).

A leaf with no imports, so the offline code (the gates, the index writers, the ops verbs) can name the
serving package's place without importing the package itself. `cytir.serve` re-exports `SERVER_REL` and adds
`SERVER_DIR`, this checkout's copy.
"""
import pathlib

SERVER_REL = "cytir/serve"

# --- which package a source tree carries (RULING 82-85: the `aw` -> `cytir` rename) ------------------
# `SERVER_REL` above names THIS checkout's serving package. A gate or a release tool that starts or
# inspects ANOTHER tree (the PR base, the release tree, the tree a box serves) must ask that tree:
# the first release after the rename compares an old tree with a new one. After the rename a tree
# carries `cytir/` and a thin import shim `cytir/` (one file, marked below); before it, `cytir/` alone.
#: The package directory names a tree can carry, the current one first.
PACKAGE_NAMES = ("cytir", "aw")  # rename-codemod: keep
#: The first bytes of the shim's `__init__.py` carry this text (the rename codemod writes it). A tree
#: whose only `cytir/__init__.py` is the shim has no package of its own under that name.
SHIM_MARKER = "rename-codemod: shim"
SHIM_HEAD_BYTES = 400


def _is_shim(init: pathlib.Path) -> bool:
    try:
        return SHIM_MARKER in init.read_bytes()[:SHIM_HEAD_BYTES].decode("utf-8", "replace")
    except OSError:
        return False


def package_name(tree) -> str:
    """`cytir` when `<tree>/cytir/__init__.py` exists (it wins over the shim; beside a real
    old-package dir that is not the shim it is refused), else `aw` when
    `<tree>/aw/__init__.py` exists and is not the shim; anything else is refused. The one resolver
    behind every `<tree>/<package>/...` path of a tree other than this one. `serve_parity` holds an
    inline copy (it is loaded by path inside an arm); tests/test_aw_package_name_resolver.py pins them."""
    tree = pathlib.Path(tree)
    new, old = PACKAGE_NAMES
    init = tree / old / "__init__.py"
    if (tree / new / "__init__.py").is_file():
        if init.is_file() and not _is_shim(init):
            # a half-renamed tree (a copy without delete): never pick one of two real packages silently
            raise SystemExit(f"{tree} carries two real packages, {tree / new} and {tree / old} (its "
                             f"__init__.py is not the import shim): a stale directory, refusing to guess")
        return new
    if init.is_file() and not _is_shim(init):
        return old
    raise SystemExit(f"{tree} carries no serving package ({new}/ or {old}/ with its own __init__.py; "
                     f"a {old}/ that is only the import shim does not count): refusing to guess")


def serve_rel(tree) -> str:
    """`<package>/serve` of `tree`."""
    return f"{package_name(tree)}/serve"
