# Contributing to Cytir

Thank you for your interest in contributing! This guide explains how to get started.

## Installation

```bash
git clone https://github.com/impactia-foundation/cytir.git && cd cytir
pip install -e .[dev]
```

With uv: `uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"`.

This installs the `cytir` package in editable mode with all dev dependencies for testing and local development.

## Running tests

```bash
export CYTIR_WORK_ROOT=$(mktemp -d)    # Cytir has no default work directory; see QUICKSTART 0a
pytest
```

This runs the complete test suite. Tests are organized by module and use pytest conventions.

## The `cyt` CLI

The main entry point is the `cyt` CLI. Explore available commands:

```bash
cyt --help
```

This shows the primary command groups for the search index: `dump`, `apply`, `rehearse`, `torrent`, `source`, and `embed` (`embed shards` makes the vectors; `cyt embed index` and `cyt source run --embed` are retired, the build verb is `cyt index build`).

## Adding a source

The project uses a plugin protocol for data sources. To add a new source family:

1. Read the SourceFamily Protocol in `cytir/sources/family.py`
2. Implement your source following the protocol (inherit from `SourceFamily`, implement `run()`)
3. Register it in the source registry
4. Add tests in the same PR
5. See `ADDING-A-SOURCE.md` for a detailed walkthrough

## Code style and review

- Keep PRs focused on a single concern
- Include tests for new functionality
- Run `pytest` locally before pushing
- PRs should be reviewed before merge; feedback is collaborative

## DCO and sign-off

This project uses the Developer Certificate of Origin (DCO). By contributing, you certify that you wrote the contribution or have the right to submit it under the project's license (Apache-2.0).

Sign your commits with:

```bash
git commit -s
```

This adds your `Signed-off-by: Your Name <email>` line, confirming your authorship and rights.

## Reporting issues

For bugs, performance issues, or feature requests, open a GitHub issue with:

- A clear title describing the problem
- Steps to reproduce (for bugs)
- Expected vs. actual behavior
- Your environment (Python version, OS)

## Questions?

If you have questions about the architecture, design, or how something works, ask in an issue or discussion. The codebase is documented in `CLAUDE.md`, `spec.md`, and inline comments.

Thank you for helping build an open, grounded search index!
