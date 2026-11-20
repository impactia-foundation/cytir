"""Task 2389: the family registry is restored after each test module (conftest.py fixture).
The fixture body is driven by hand: register inside it, finish it, the key is gone."""
import importlib.util
import pathlib

from cytir.sources import registry

_spec = importlib.util.spec_from_file_location(
    "aw_shared_conftest_2389", pathlib.Path(__file__).with_name("conftest.py"))
_conftest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_conftest)
_body = _conftest._source_family_registry_isolated_per_module.__wrapped__


class _Leaky:
    family_id = "leaky_family_2389"


def test_a_family_registered_in_one_module_is_gone_after_it():
    gen = _body()
    next(gen)                                   # module starts: snapshot taken
    registry.register_family(_Leaky)            # the module registers a family
    assert "leaky_family_2389" in registry._FACTORIES
    for _ in gen:                               # module ends: registry restored
        pass
    assert "leaky_family_2389" not in registry._FACTORIES


def test_snapshot_is_complete_so_real_families_survive_the_restore():
    gen = _body()
    next(gen)
    assert "fedlex_exact" in registry._FACTORIES
    for _ in gen:
        pass
    assert "fedlex_exact" in registry._FACTORIES
