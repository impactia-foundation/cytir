# Cytir

An **open web search engine and open web search index whose primary user is an AI agent**. LLMs hold a
lossy copy of the web and confabulate on precise authoritative facts (what Art. 336 CO
says, a library function's current signature, a federal statistic). Cytir (short form CS) **surfaces the
URL of the authoritative source** — with the minimum context needed to rank — so the
agent can fetch and read it. A finder of authoritative URLs, **not a content host**.

## Status

**Live at https://websear.ch** — a Swiss-focused index of about 69 million passages
(arctic-embed-l-v2, 256-d float16 with product quantization, dense + BM25, on one box).
The MCP endpoint for AI agents is **https://mcp.websear.ch**. P0 = Swiss law (DE/FR/IT) on
`fedlex`; the runway is a billion-ish-passage, self-hostable index.

## Try it

On a fresh Ubuntu 24.04 (or Debian 12) the system `pip` is missing and refuses installs outside a virtualenv (PEP 668),
so use [uv](https://docs.astral.sh/uv/). It needs `git`, `curl` and `minisign` from apt:

```bash
sudo apt-get update && sudo apt-get install -y git curl minisign
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"
git clone https://github.com/impactia-foundation/cytir.git && cd cytir
uv venv && source .venv/bin/activate     # the sample data lives in this repository
uv pip install -e ".[dev]"
cyt rehearse run          # no network, no server; needs `minisign` (apt line above, `brew install minisign` on macOS)
```

With plain pip instead, inside a virtualenv you made (`sudo apt-get install -y python3-venv python3-pip`, then
`python3 -m venv .venv && source .venv/bin/activate`): `pip install -e ".[dev]"`.

The package alone (the `cyt` command and its data files, not the sample files of the clone) is on PyPI. The base install has no
dependencies, so only `cyt --version` and `--help` run; a verb that needs more tells you which extra to install.
Use the `dev` extra for the first commands: `uv tool install "cytir[dev]"` (or `pip install "cytir[dev]"` in a virtualenv),
or one-off `uvx --from "cytir[dev]" cyt rehearse run`.

Anything that fetches from a host (the table below, `cyt embed models`, every source run) refuses until you say who you are,
because Cytir will not send an anonymous User-Agent. Set these two first (your own page and your own contact; a URL or a mailbox):

```bash
export CYTIR_OPERATOR_URL=https://example.org/my-cytir-instance
export CYTIR_OPERATOR_CONTACT=you@example.org
```

German de-compounding (ingest, the server, tuning) also needs the splitter code, which PyPI cannot carry as an extra:
`pip install "charsplit @ https://github.com/dtuggener/CharSplit/archive/9710624ef0b0df0edd12a721c33533ff1d07b557.tar.gz"` (not the PyPI project named `charsplit`; the same quoted requirement works after `uv pip install`), plus the table from `cyt` / `python -m cytir.embed.fetch_models --asset charsplit-ngram`.

`cyt rehearse run` needs only that install (the `dev` extra: no torch, no model files, no tokenizer). It takes the sealed sample index through the release path:
dump, sign, verify, tamper test, apply into a second empty collection, query. It prints one JSON
summary and exits 0 when every step holds.

The first real run, one source family from raw source to a sealed, consent-checked pack, needs more:
the pinned embedder tokenizer (`transformers` plus the tokenizer files; the chunker
refuses without them). Section 1b of [`QUICKSTART.md`](QUICKSTART.md) lists the prerequisites (the `pipeline` extra or
`pip install transformers`, and the tokenizer fetch) and the command (`cyt source run loda_exact`). The guide then continues with `cyt index build`,
`cyt serve` and `cyt gate` on a small fixture.

## What you can rebuild, and what you cannot

**You can rebuild the fixture families.** The four verbs `cyt source run`, `cyt index build`,
`cyt serve` and `cyt gate` take a family from source to a served, gated index on your own
machine. Public CI runs this path on every change, with stand-in vectors. The real embedder
also runs on CPU: [`QUICKSTART.md`](QUICKSTART.md) has the steps and the measured cost.

**You cannot yet rebuild the production index.** The operator rebuilds it from a locked
manifest with private tooling. This repository does not hold that tooling. The public v1
target is the Swiss slice: a stranger rebuilds it from public Common Crawl and public dumps,
and the counts and eval match production's same slice. Until then, treat the production index
as published output, not as something you can reproduce from this repository.

## Where to read

| File | Purpose |
|---|---|
| [`QUICKSTART.md`](QUICKSTART.md) | Install, build, serve and gate a small index |
| [`ADDING-A-SOURCE.md`](ADDING-A-SOURCE.md) | Add a source pack and a family adapter |
| [`specs/dump-format.md`](specs/dump-format.md) | The redistributable dump format |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | How to contribute |
| [`LICENSE`](LICENSE) | Apache-2.0 |

The operator keeps its specification, decision ledger, plan and task records in a private
operator tree. That tree is not in this repository.
