"""Task 1509: the so-driver's producer revision is the digest of the package materializer."""
import hashlib
import inspect

from cytir.core.paths import REPO_ROOT
from cytir.ops import stackoverflow_tranches

MODULE = "cytir.sources.stackoverflow.materialize_production_tranche"
FILE = REPO_ROOT / "cytir/sources/stackoverflow/materialize_production_tranche.py"


def test_the_driver_names_the_package_module_and_file():
    src = inspect.getsource(stackoverflow_tranches)
    assert 'REPO_ROOT / "cytir/sources/stackoverflow/materialize_production_tranche.py"' in src
    assert f'"-m", "{MODULE}"' in src
    assert "hashlib.sha256(materializer.read_bytes())" in src
    assert FILE.is_file()


def test_the_producer_digest_equals_sha256_of_that_file():
    src = inspect.getsource(stackoverflow_tranches)
    assert "materializer = REPO_ROOT /" in src
    digest = hashlib.sha256(FILE.read_bytes()).hexdigest()
    assert len(digest) == 64
    import importlib
    mod = importlib.import_module(MODULE)
    assert hashlib.sha256(open(mod.__file__, "rb").read()).hexdigest() == digest
