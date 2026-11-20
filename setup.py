"""Build shim (task 733). Everything else lives in pyproject.toml.

Why this file exists: the runtime reads `data/` (contracts, registers, the rehearsal fixture).
A non-editable `pip install .` copied only `cytir/`, so a stranger's install could not find them.
pyproject.toml can map a top-level directory onto a package (`[tool.setuptools.package-dir]
"cytir._data" = "data"`) but cannot ADD that package to an auto-discovered list, so the package
list is spelled here: every `cytir` package (namespace dirs included) plus `cytir._data`.

Which data files ship is pyproject's `[tool.setuptools.package-data] "cytir._data"`; a test
(tests/test_package_data.py) holds that list equal to the public cut's `data-contracts` and
`release-rehearsal` rules, so a file the cut withholds can never reach a wheel.
The reader side is `cytir.core.paths.data_dir()` / `data_path()`.
"""
from setuptools import find_namespace_packages, setup

setup(packages=find_namespace_packages(include=["cytir", "cytir.*"]) + ["cytir._data"])
