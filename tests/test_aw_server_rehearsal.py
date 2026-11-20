"""Task 2339: the server-side rehearsal refuses a scratch server that is not scratch."""
from types import SimpleNamespace as NS

import pytest

from cytir.distribute import server_rehearsal as sr

TWIN = "http://127.0.0.1:6486"


class FakeScratch:
    def __init__(self, collections=(), aliases=()):
        self._c, self._a = set(collections), list(aliases)

    def collection_exists(self, name):
        return name in self._c

    def get_aliases(self):
        return NS(aliases=[NS(alias_name=a, collection_name=c) for a, c in self._a])


def test_clean_scratch_passes():
    sr.check_scratch(TWIN, "http://127.0.0.1:6499", FakeScratch(["other"]))


@pytest.mark.parametrize("url", ["http://127.0.0.1:6486", "http://localhost:6486"])
def test_refuses_the_twin(url):
    with pytest.raises(sr.ScratchRefusal, match="twin"):
        sr.check_scratch(TWIN, url, FakeScratch())


@pytest.mark.parametrize("port", [6333, 6485])
def test_refuses_a_serving_port(port):
    with pytest.raises(sr.ScratchRefusal, match="serving port"):
        sr.check_scratch(TWIN, f"http://127.0.0.1:{port}", FakeScratch())


@pytest.mark.parametrize("name", ["slice", "applied"])
def test_refuses_an_existing_collection(name):
    with pytest.raises(sr.ScratchRefusal, match="already has"):
        sr.check_scratch(TWIN, "http://127.0.0.1:6499", FakeScratch([name]))


@pytest.mark.parametrize("alias", [("slice", "x"), ("prod", "applied")])
def test_refuses_an_alias_to_or_from_a_scratch_name(alias):
    with pytest.raises(sr.ScratchRefusal, match="alias"):
        sr.check_scratch(TWIN, "http://127.0.0.1:6499", FakeScratch(aliases=[alias]))
