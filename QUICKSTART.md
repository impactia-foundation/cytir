# QUICKSTART — build a tiny Cytir index from a clean clone

**Last verified 2026-11-12** on a clean Ubuntu 24.04 container by task 2774: stand-in path and real-embedder path both PASS (gate `proves: ranking`). The worked example in section 5 queries the fixture collection `aw_gate_fixture` (Swiss-Austrian border-traffic agreement, SR 0.631, DE/FR).

This is deliberately **not** the production rebuild (a manifest-driven runner, currently `blocked` — see [README.md](README.md)). What follows is the *small honest path* that works today.

## The end-to-end path with `cyt` verbs only (run 2026-10-04; re-run 2026-11-03 from a clean clone in a bare `ubuntu:24.04` container, task 2251)

Make one stage from a source sample with `cyt source run`, build an index from it and two sealed
fixture stages (96 points: a Swiss federal act in DE/FR/IT, three French in-force texts and one French
collective-agreement article), serve it, and score a gold slice against it. The same verbs run, as
subprocesses with exactly one Cytir setting (`CYTIR_WORK_ROOT`), in the CI job `index-serve-gate` on every
change that can alter them, against a throwaway Qdrant v1.18.3 started as a plain binary (the
`docker run` of section 1 starts the same release). The build, the server start and the gate take
under a minute together.

**Read this first: the vectors here are stand-ins.** The pinned embedder needs `torch` and 2.3 GB of
weights. This walkthrough uses `tests/standin_embedder.py` instead: a hashed bag of words, for
documents and for queries. The build, the collection shape, the server and the gate are the real
code; the dense leg's geometry is not the model's. So a pass here proves the plumbing, not ranking
quality (all 24 gold queries rank 1 on a 96-point index, which is also why the baseline carries no
ranking information). The gate does not leave this to you: every verdict has a `proves` field, set
from the embedder identity the server reports (the stand-in announces itself), and `cyt gate`
prints it. For this walkthrough it reads `plumbing-only (stand-in vectors; not a ranking claim)`. The real path is the next section (run 2026-10-13, task 911); its cost is in "CPU vs GPU" below.

```bash
python3 -m venv ~/venv-cytir
~/venv-cytir/bin/pip install -e ".[dev]" "uvicorn==0.54.0" "charsplit @ https://github.com/dtuggener/CharSplit/archive/9710624ef0b0df0edd12a721c33533ff1d07b557.tar.gz"
export PATH=~/venv-cytir/bin:$PATH

export CYTIR_WORK_ROOT=$HOME/cytir-work          # must exist: the write locks live under it
mkdir -p "$CYTIR_WORK_ROOT/fixture" "$CYTIR_WORK_ROOT/consent-cache"
export QDRANT=http://127.0.0.1:6333        # a throwaway Qdrant, on any free port
```

The health and search checks use `curl`; on a slim image install it first (`apt-get install -y curl`).

No command below starts Qdrant. Start it first, on the port `QDRANT` names:

```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:v1.18.3
```

`cyt index build` refuses a Qdrant that is not on this machine (`REFUSED: ... a build never targets a
non-local Qdrant ... without --allow-remote`). If you run these steps inside a container and Qdrant
is another container, forward its port to the container's own loopback instead of passing
`--allow-remote` (for example `socat TCP-LISTEN:6333,fork,bind=127.0.0.1 TCP:<qdrant-container-ip>:6333`).

1. **Stages.** A stage is `PACK.json` plus the chunk rows. `data/index-fixture/` holds two sealed ones
   (`loda_exact`, `fedlex_exact`); copy them. The other five you make yourself, through the family door:
   `cyt source run` reads a local sample of a dump (`--from`, offline: no network, no consent call,
   because a dump family contacts no host), runs the family's declared adapter and writes the stage
   `cyt index build` reads (`--out`). The samples are the DILA KALI dump, six Service-Public fiches, seven Stack Exchange
   posts, six OpenAlex works (one small parquet file cut from the public snapshot) and ten Stack Overflow posts (ten questions with their accepted answers). `--today` pins the run date, so the output is byte-identical on every machine; it equals the
   committed `data/index-fixture/dila_kali/`, `data/index-fixture/service_public_fiche/`, `data/index-fixture/stackexchange_nonso_qa_p0/`, `data/index-fixture/openalex_academic_dump/` and `data/index-fixture/stackoverflow_qa/`. Then give
   all seven stand-in vectors (`vectors/`, in the format `cyt embed shards` writes).
   `cyt source run` needs the exact embed tokenizer for every family that chunks text (all but the two copied ones). Without `transformers` and the tokenizer files it **refuses** (`exact embed tokenizer unavailable ... refusing to write word-only chunks`; `cytir/chunk/chunker.py`). There is no word-only fallback. Do the tokenizer step first (it is 22 MB):

   ```bash
   pip install transformers                           # skip if you installed the `pipeline` extra
   HF_HUB_OFFLINE=0 python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Snowflake/snowflake-arctic-embed-l-v2.0', revision='ac6544c8a46e00af67e330e85a9028c66b8cfd9a', allow_patterns=['*.json','*.model','*.txt'])"
   export HF_HUB_OFFLINE=1
   ```

   ```bash
   cp -r data/index-fixture/loda_exact data/index-fixture/fedlex_exact "$CYTIR_WORK_ROOT/fixture/"
   cyt source run dila_kali --from data/samples/dila-kali --out "$CYTIR_WORK_ROOT/fixture/dila_kali" \
       --today 2026-10-03
   cyt source run service_public_fiche --from data/samples/service-public-fiche \
       --out "$CYTIR_WORK_ROOT/fixture/service_public_fiche" --today 2026-10-03
   cyt source run stackexchange_nonso_qa_p0 --from data/samples/stackexchange-nonso \
       --out "$CYTIR_WORK_ROOT/fixture/stackexchange_nonso_qa_p0" --today 2026-10-03
   cyt source run openalex_academic_dump --from data/samples/openalex-academic \
       --out "$CYTIR_WORK_ROOT/fixture/openalex_academic_dump" --today 2026-10-03
   cyt source run stackoverflow_qa --from data/samples/stackoverflow-qa \
       --out "$CYTIR_WORK_ROOT/fixture/stackoverflow_qa" --today 2026-10-03
   python tests/standin_embedder.py vectors "$CYTIR_WORK_ROOT"/fixture/loda_exact \
       "$CYTIR_WORK_ROOT"/fixture/fedlex_exact "$CYTIR_WORK_ROOT"/fixture/dila_kali \
       "$CYTIR_WORK_ROOT"/fixture/service_public_fiche "$CYTIR_WORK_ROOT"/fixture/stackexchange_nonso_qa_p0 "$CYTIR_WORK_ROOT"/fixture/openalex_academic_dump \
       "$CYTIR_WORK_ROOT"/fixture/stackoverflow_qa
   ```

2. **Build** (a few seconds): creates the collection from the `prod-e9` profile (what prod runs; `--profile served` is the pre-e9 shape), applies the packs with
   the same code the production drain runs (single target instead of twin-first), finalizes it,
   and measures **this build's own BM25 IDF table**. `--as-of` pins the consent stamp and the
   table's date, so a rebuild is byte-identical.

   ```bash
   cyt index build \
       --pack "$CYTIR_WORK_ROOT/fixture/loda_exact" --pack "$CYTIR_WORK_ROOT/fixture/fedlex_exact" \
       --pack "$CYTIR_WORK_ROOT/fixture/dila_kali" --pack "$CYTIR_WORK_ROOT/fixture/service_public_fiche" \
       --pack "$CYTIR_WORK_ROOT/fixture/stackexchange_nonso_qa_p0" \
       --pack "$CYTIR_WORK_ROOT/fixture/openalex_academic_dump" --pack "$CYTIR_WORK_ROOT/fixture/stackoverflow_qa" \
       --qdrant "$QDRANT" --collection aw_gate_fixture --out "$CYTIR_WORK_ROOT/run" \
       --as-of 2026-10-03T00:00:00Z --consent-cache-dir "$CYTIR_WORK_ROOT/consent-cache" --recreate
   ```

   It leaves `build-manifest.json` (counts per pack: records, admitted, written), the row
   fingerprints, the IDF snapshot and `bm25_idf.json` in `$CYTIR_WORK_ROOT/run`.

3. **Serve.** `cyt serve` refuses a collection that is not the declared profile, starts the server
   with the shipped default environment (`cytir/ops/serving_env.default.json`; a checkout that has
   `deploy/` uses its compose files instead) and writes `serve-receipt.json`. The query embedder is
   the stand-in again (a stub that answers like a TEI server):

   ```bash
   python tests/standin_embedder.py tei 18081 &
   cyt serve --collection aw_gate_fixture --qdrant "$QDRANT" --tei-url http://127.0.0.1:18081 \
       --idf-table "$CYTIR_WORK_ROOT/run/bm25_idf.json" --port 18080 \
       --receipt "$CYTIR_WORK_ROOT/serve-receipt.json" &
   curl -s http://127.0.0.1:18080/health       # wait until "status":"ok" (10 to 40 s)
   curl -s "http://127.0.0.1:18080/search?q=Art.+336+CO+Kuendigung&k=3"   # one query: JSON with `results`
   ```

4. **Gate.** Scores the 24 pinned golds over HTTP with the ADR-043 per-gold scorer. It passes when
   no gold ranks worse than in the baseline verdict of the same slice and packs (exit 0), fails
   when one does (exit 1, and only then), and refuses (exit 3) when it cannot score: a query that
   errors, an empty index, a server that answers nothing to every query, a baseline made on
   another profile or IDF table (unless you pass `--allow-identity-mismatch`), or any unexpected
   error. None of these is counted as a miss.

   ```bash
   cyt gate --api http://127.0.0.1:18080 --slice data/gate-slices/ci-fixture.v1.json \
       --baseline data/gate-slices/ci-fixture.baseline.v1.json \
       --serve-receipt "$CYTIR_WORK_ROOT/serve-receipt.json" --verdict-out "$CYTIR_WORK_ROOT/verdict.json"
   ```

   The verdict records the slice's sha256, the server receipt's sha256, the profile digest, the
   IDF table's identity and `proves`. The last lines of the output show it:

   ```text
   proves: plumbing-only (stand-in vectors; not a ranking claim)
   ```

   With the collection name and `--as-of` above, the profile digest and the IDF table equal the
   baseline's; if they do not, the gate refuses and names both values.

Stop the two background processes when you are done (`kill` their pids).

**What is not on this path yet.** (a) The step from a source to a *stage* exists for the declared families (no `unknown` in their contract rows), among them `dila_kali`, `service_public_fiche`, `stackexchange_nonso_qa_p0`, `openalex_academic_dump`, `stackoverflow_qa`, `loda_exact` and `fedlex_exact`. `cyt source run` refuses to stage a
family whose declaration holds an `unknown`. All eight fixture stages in `data/index-fixture/` are made by the door (task 2561). `--from` works only for a dump
family that contacts no host. The Service-Public family runs the same door: `cyt source run service_public_fiche
--from data/samples/service-public-fiche --out STAGE --today 2026-10-03` (six real Service-Public fiches,
Etalab-2.0; the chunk policy it declares needs the pinned tokenizer, and runs word-bounded with a
warning when `transformers` is not installed). Its stage is complete, and `cyt index build` accepts its rows under the dump-licence consent rule r1
(Etalab-2.0), so it is in the build, serve and gate steps above: six fiches, three golds. So is the Stack Exchange family: seven passages
(six questions of one site, `data/samples/stackexchange-nonso/`, CC BY-SA 3.0 and 4.0), three golds. So is the OpenAlex family: ten works of one snapshot part (`data/samples/openalex-academic/`, CC0), three golds. So is the Stack Overflow family: ten questions with their accepted answers (`data/samples/stackoverflow-qa/`, CC BY-SA 2.5, 3.0 and 4.0), three golds. The `dila_kali` sample holds ONE in-force article, so its three golds, like the three of the Stack Exchange
stage, test the plumbing of that family and nothing about ranking. (b) Real vectors: `cyt embed shards` needs the model. (c) The
gold slice is ours and tiny; your own golds go in a file of the same shape (`aw.gate-slice.v1`).


## The real-embedder path on a CPU-only box (run 2026-10-13, task 911)

The same verbs, with the pinned model instead of the stand-in. Run from the public cut in a fresh
`python:3.12` container (no GPU, no mounts, no Hugging Face cache, no token). Do the stages (step 1
above) first, then:

```bash
pip install -e ".[pipeline,serve]" "uvicorn==0.54.0" "charsplit @ https://github.com/dtuggener/CharSplit/archive/9710624ef0b0df0edd12a721c33533ff1d07b557.tar.gz" \
    --extra-index-url https://download.pytorch.org/whl/cpu    # 4 min, venv 2.1 GB
cyt embed models                                                # 171 MB, 22 s
for s in loda_exact fedlex_exact dila_kali service_public_fiche stackexchange_nonso_qa_p0 \
         openalex_academic_dump stackoverflow_qa; do
  cyt embed shards --in "$CYTIR_WORK_ROOT/fixture/$s/chunks" --out "$CYTIR_WORK_ROOT/fixture/$s/vectors" \
      --device cpu --compute-dtype float32
done
```

Then `cyt index build` (step 2 above). A second build into the same `--out` is safe with `--recreate`: it deletes the
collection and clears this run's own DONE markers, so every pack is built again. Without `--recreate` or `--resume`
it refuses (the collection exists). A build that ends with 0 points exits non-zero (rc 3) and tells you to use a
fresh `--out`; it never reports success on an empty collection.

Then:

```bash
cyt serve --collection aw_gate_fixture --qdrant "$QDRANT" \
    --idf-table "$CYTIR_WORK_ROOT/run/bm25_idf.json" --port 18080 \
    --receipt "$CYTIR_WORK_ROOT/serve-receipt.json" &
```

Three things differ from the stand-in path, and each one failed on a clean box before this section:

- **Device and dtype resolve automatically.** On a CPU-only box `--device cpu --compute-dtype float32` is the
  default, so the flags are optional; they stay as the explicit form. (Before task 912 the default was
  `cuda:0` and `float16`, and a CPU box stopped with `AssertionError: Torch not compiled with CUDA enabled`.)
- **`.[pipeline]` does not install the server.** `cyt serve` needs `fastapi`, which the `serve` extra
  carries. Without it the server stops with `ModuleNotFoundError: No module named 'fastapi'`.
- **This venv is on numpy>=2, so the language gate is off here.** fastText needs `numpy<2`; the HTML/crawl
  sources need the separate `langgate` venv (section "The second venv" below), not this one.
- **No `--tei-url`: the server embeds queries in-process** with the same pinned model. The first query
  loads it (8 s); later queries take 0.1 s. `cyt gate` then prints an
  `embedder:` line (model, pinned revision, device/dtype, library versions) and the verdict proves `ranking`.

**The model download.** The arctic embedder is fetched on the first `cyt embed shards` from Hugging
Face, at the pinned revision `ac6544c8a46e00af67e330e85a9028c66b8cfd9a` (see "The embedder is
revision-pinned"): **2.2 GB** in `~/.cache/huggingface`, and the first stage (download included)
took 57 s on a fast link. After that first fetch the cache is complete, and `HF_HUB_OFFLINE=1`
is safe for a rebuild that has the cache. Do not set it before the first fetch. The vector manifests
record `revision_pinned` and `revision_resolved`; both read `ac6544c8...`.

**Sealed vectors.** The CPU vectors for the fixture rows are committed in `tests/data/index-fixture-dense-cpu.v1.json`; `tests/test_aw_quickstart_fixture_vectors.py` re-embeds the rows on CPU and checks cosine >= 0.999 against them (it skips when the model is not cached), so you can tell your embeds match ours.

Batch size and text order do not change CPU fp32 document embeddings beyond float rounding (cosine >= 0.9999999, max element delta 3.4e-06 over 186 passages, task 2536); a rebuilt vector below cosine 0.9999 against the reference means a different input text or token cap, not a different batch.

**What hosted CI runs.** The public CI runs the stand-in path only (CPU, 96 points): `cyt source run` x5,
stand-in vectors, `cyt index build`, `cyt serve`, one query, `cyt gate`, about 70 s. It cannot run the
real-embedder path (2.3 GB of weights, torch wheels). The gate there says `plumbing-only`; it proves
that the verbs connect and the golds answer, and it makes no ranking claim.

Measured (8 cores, `nice -n 19`): embed 67 rows in 7 stages, 2.1 to 2.9 GB peak RSS per
stage, 10 s to 57 s each; `cyt index build` 6 s, 0.46 GB; the server 2.5 GB RSS.

---


## What you need

| | |
|---|---|
| OS | Linux, Python 3.12 (3.11 fine) |
| RAM | 8 GB minimum; the CPU embed peaked at **3.4 GB RSS** |
| Disk | ~12 GB with the default torch (the pipeline venv alone is **5.5 GB**: it pulls CUDA wheels even on a CPU-only box). With `--extra-index-url https://download.pytorch.org/whl/cpu` the venv is **2.1 GB** and the embedder cache **2.2 GB** |
| GPU | **Not required for this walkthrough.** See [CPU vs GPU](#cpu-vs-gpu-what-cpu-only-really-costs) before you scale up. |
| Credentials | **None.** No AWS key, no HF token, no Cytir infrastructure. One setting is still required for every verb that fetches from a host: the crawler identity (`CYTIR_OPERATOR_URL` and `CYTIR_OPERATOR_CONTACT`, section 0a). It is your own URL and mailbox, not ours. |
| Network | Public internet: `fedlex.admin.ch`, HuggingFace, Docker Hub, PyPI |

## 0. Clone and install

The package alone installs with `pip install cytir` (the `cyt` command and the data files the package needs). The samples, the fixture stages and the source packs live in this repository, so every step below starts from a clone.

```bash
git clone https://github.com/impactia-foundation/cytir.git && cd cytir
sudo apt-get update && sudo apt-get install -y git curl ca-certificates python3.12-venv python3-pip socat docker.io
# (a bare ubuntu:24.04 image has none of git, curl or ca-certificates; `socat` is the loopback
#  forward for a Qdrant that lives in another container, see the E2E path; `docker.io` is only
#  needed on a host that will run the throwaway Qdrant of section 1)

python3 -m venv ~/venv-pipe
~/venv-pipe/bin/pip install -e ".[pipeline]" "charsplit @ https://github.com/dtuggener/CharSplit/archive/9710624ef0b0df0edd12a721c33533ff1d07b557.tar.gz"      # ~5.5 GB, several minutes (no GPU? add
#   --extra-index-url https://download.pytorch.org/whl/cpu  -> CPU torch, ~0.5 GB download)
```

## 0a. Say where Cytir keeps its work, and who you are

Every setting is a `CYTIR_*` environment variable. The older `AW_*` spelling of the same names still works, and
the two always agree (if you set both to different values, the program refuses to start). One exception: a name the package itself mirrored at import is not yours, so if you then change
the other spelling, the name you changed wins silently.

```bash
export CYTIR_WORK_ROOT=$HOME/cytir-work
export CYTIR_OPERATOR_URL=https://example.org/my-cytir-instance   # YOUR page; every host you fetch logs it
export CYTIR_OPERATOR_CONTACT=you@example.org                  # YOUR mailbox
```

The second pair is the crawler identity. Every verb that crawls a host (`cyt source run` on a crawl family, the fedlex extractor,
the consent resolver, `cyt embed models`) refuses without it,
rather than send an anonymous User-Agent (task 2838: `cyt embed models` without it stops with `UnidentifiedCrawler`
for both files). The offline stand-in path at the top needs only `CYTIR_WORK_ROOT`.
(Alternative: `cp operator.example.json operator.json` and edit it.)

Every queue, lock, scratch and store directory Cytir uses is derived from this one setting. There
is **no default**: an unset work root is a refusal that names the setting, not a quiet
`./cytir-work` that would put your queue and your write locks wherever the process happened to
start (two different directories on two different days, and a lock taken in one is no lock in
the other). `python3 -m cytir.core.paths` prints every setting, its value and where it came from.
A box that runs Cytir for an organisation keeps these values in one file instead,
`/etc/aw/operator.json`, which Cytir reads when `CYTIR_OPERATOR_OVERLAY` is unset.

## 0b. Fetch the pinned model files

Three model artifacts are *not* pip-installable. One is fetched automatically the first
time you embed (the arctic embedder, from HuggingFace). The other two you must fetch
explicitly:

```bash
cyt embed models          # 126 MB + 40 MB, sha256-verified, idempotent
cyt embed models --check  # verify only; exits 1 if anything is missing or the embedder pin is overridden
```

This gets `lid.176.bin`, the fastText model behind the **EN/DE/FR/IT language scope
gate**. The URL and the expected sha256 live in `cytir/embed/model_assets.py` — one home,
checked on every load. The file lands in `models/` (git-ignored); set `CYTIR_MODELS_DIR` to
put it elsewhere, or `CYTIR_LID_MODEL` to point at a copy you already have.
If `dl.fbaipublicfiles.com` fails (its certificate expired on 2026-11-11), the tool falls back to a sha256-checked Hugging Face mirror; by hand: `curl -fSL -o models/lid.176.bin https://huggingface.co/julien-c/fasttext-language-id/resolve/9f2e94d2fd6b2615d26f5ec5b34cf7e2f38cb86e/lid.176.bin`, then check the sha256 in `cytir/embed/model_assets.py`.

The second file is `ngram_probs.json`, the 40 MB n-gram table of the German
de-compounding rule (`cytir.core.decompound`, CharSplit, MIT, ADR-121). It is **required** for
anything that tokenises German for the BM25 leg: ingest, the server and every tuning
script. Without it they **refuse** (a clear error naming this command). They never fall
back to "no splitting", because a build and a query that split differently would change
ranking without a trace. Fetch only this one with `--asset charsplit-ngram`; set
`CYTIR_CHARSPLIT_NGRAM` to point at a copy you already have. The code half is the `charsplit`
package. No extra carries it (PyPI refuses direct-URL requirements), so install it once per venv:
`pip install "charsplit @ https://github.com/dtuggener/CharSplit/archive/9710624ef0b0df0edd12a721c33533ff1d07b557.tar.gz"` (do not install the PyPI package
named `charsplit`: it is a different project).

**You do not need the language model for the fedlex walkthrough below** (dump-based sources carry
authoritative per-document language labels). You *do* need it for every HTML/crawl source
— and if it is absent those scripts **refuse to run**, loudly, naming this command. They
do not quietly fall back to guessing the language from the URL, because that is how
out-of-scope pages leaked into the corpus in task 233.

### The second venv (the language gate needs `numpy<2`)

fastText's `predict()` calls `np.array(..., copy=False)`, which numpy 2.x removed. The
main pipeline venv is on numpy>=2 (docling and opencv require it), so **the gate cannot
run there** and the HTML-extraction path needs a venv of its own:

```bash
python3 -m venv ~/venv-langgate
~/venv-langgate/bin/pip install -e ".[langgate]"   # numpy<2 + fasttext-wheel

~/venv-langgate/bin/python -c \
  "from cytir.acquire import fetch_process as fp; print(fp.langgate_status())"     # -> (True, None)
```

The pin is exact: `numpy<2` plus `fasttext-wheel` (extra `langgate` in `pyproject.toml`). It stays while
fastText's `predict()` calls `np.array(..., copy=False)`; re-check with the probe above after any fastText bump.

`(True, None)` means the gate is live. Anything else prints the reason and the fix.
Run `cyt source run` in **this** venv; keep the docling PDF/table path on the main
numpy>=2 venv as a separate step.

> **Known gap.** The `pipeline` extra is incomplete. It omits packages the wider
> operator's wider build imports — `pyarrow`, `docling`, `boto3`, `sentence-transformers`,
> `sklearn`, `jsonschema`, `lmdb`, `mwxml`, `py7zr`, `pypdf`,
> `playwright`, `htmldate`, `aiohttp`, and `pytest`. The path in this document does not
> need them (`sentence-transformers` arrives transitively via `FlagEmbedding`, both in the `embed-local` extra), but any
> other source will trip on one. Tracked in task 429.

## 1. Start a local Qdrant

```bash
sudo systemctl start docker
docker run -d --name qdrant -p 6333:6333 -v ~/qdrant-storage:/qdrant/storage \
    qdrant/qdrant:v1.18.3
curl -s localhost:6333/     # -> {"title":"qdrant - vector search engine", ...}
```

Pin v1.18.3: it is the release the CI job `index-serve-gate` runs, and a 1.18 binary **panics** on a
1.12 storage directory.

## 1b. The one-command path: run a source family to a sealed pack

`cyt source run <family_id>` is the single door for acquire, extract, chunk and pack. It needs
no Qdrant server, no GPU, no credentials and no Cytir machine.

This family reads the pinned embedder's **tokenizer** (it sizes the chunks), so the run needs
`transformers` (it is in the `pipeline` extra of section 0, not in `dev`) and the tokenizer files in
the HuggingFace cache. Without them it **refuses** and prints the fetch command. The tokenizer files
alone are 22 MB (the full snapshot that command fetches by default is 9.4 GB); this is enough:

```bash
pip install transformers                           # skip if you installed the `pipeline` extra
HF_HUB_OFFLINE=0 python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Snowflake/snowflake-arctic-embed-l-v2.0', revision='ac6544c8a46e00af67e330e85a9028c66b8cfd9a', allow_patterns=['*.json','*.model','*.txt'])"
export HF_HUB_OFFLINE=1
pip install -e .                                   # installs the `cyt` command (no dependencies of its own)
cyt source families                                 # what is registered
cyt source run loda_exact \
    --out-dir build/loda \
    --opt root=data/samples/legi-loda --opt release=aw-sample-loda-v1
```

`data/samples/legi-loda` holds eight real DILA LEGI articles (Etalab-2.0). Point `root=` at
your own extracted DILA LEGI dump (the directory that contains `TNC_en_vigueur/`) to run the
whole family. On the sample the run prints `in=8 admitted=3 excluded=5 quarantined=0
records=3`. The five exclusions are counted by reason in the manifest (articles below the body
floor, one repealed article). Nothing is dropped silently.

What the run leaves in `build/loda/`:

| path | what it is |
|---|---|
| `l0/` | the acquired bytes, content-addressed, sealed before extraction |
| `pack/records.jsonl`, `pack/manifest.json` | the chunked records, and the declaration, consent verdicts and per-unit terminals they were made under |

**`--embed` is retired.** It refuses before reading or writing anything and names `cyt index
build`, the one build verb (it exists, see the E2E path). Vectors for a pack come from
`cyt embed shards`. Add `--dry-run` to see what would be read, writing nothing.
`cyt source run FAMILY --help` lists every flag.

## 2. Fetch + extract + chunk a source

`fedlex` is the P0 source and the most portable entry point in the repo: it reads Akoma
Ntoso XML straight from the federal filestore, needs no crawler, no language model and no
credentials (only the crawler identity of section 0a), and emits one anchored passage per article — so fetch, extract and chunk are
a single step.

```bash
mkdir -p ~/build
python3 -m cytir.sources.families.fedlex \
    --eli cc/1999/404 --langs de fr it \
    --out ~/build/fedlex.jsonl
```

Observed (2.6 s, 3 documents):

```
  · cc/1999/404/de (20240303) -> 241 article-passages
  · cc/1999/404/fr (20240303) -> 247 article-passages
  · cc/1999/404/it (20240303) -> 244 article-passages
wrote 732 fedlex article-passages ... -> ~/build/fedlex.jsonl
```

`cc/1999/404` is the Federal Constitution (SR 101). Any consolidated ELI works.

## 3. Resolve consent — live, per source

Consent is **never** inherited from Common Crawl and never assumed. Resolve it against the
live host before you index anything:

```bash
python3 -m cytir.consent.resolver \
    --hosts www.fedlex.admin.ch,fedlex.data.admin.ch \
    --out ~/build/consent.jsonl
```

```
  [1/2] fedlex.data.admin.ch: ok (no opt-out signal found)
  [2/2] www.fedlex.admin.ch: ok (no opt-out signal found)
CONSENT SUMMARY: {'ok': 2, 'declined': 0, 'unknown': 0} of 2 hosts
```

> **Known gap.** Nothing enforces this. the consent resolver is not wired into
> the fedlex extractor or most of the other extractor scripts — it is a **manual step you must
> remember**, and the embed step will happily index a declined host. If a host returns
> `declined`, do not proceed with it. Tracked in task 429.

## 4. Embed and index (retired; replaced by `cyt index build`)

`cyt embed index` is **retired**. It refuses and names `cyt index build` (the path above). The old command projected rows with a generic materializer that
differs from the one production uses, so the index it built was not the index Cytir serves:

| axis | retired door | production |
|---|---|---|
| point id | uuid5 of the URL | the pack's declared id, verified |
| dense text | raw `text` | `"<title>. <site_name>. <body>"` |
| sparse leg | raw term counts | saturated BM25 weights |
| payload | the whole record | a whitelist |

What you can do today: make vectors for a sealed pack with `cyt embed shards` (1024-d float16
files, pinned model revision), build a servable collection from a stage with `cyt index build`, and
get a ready-made index by applying a published dump. What you cannot do yet: cut a pack into a
stage with one verb for every family (see the end of the path above).

> **The model download and the CPU cost** are in the sections below; they describe
> `cyt embed shards` as well as the retired door.

> **The old single-script index build (the old index-corpus script) no longer exists** (task 432). Use
> `cyt index build`.

## 5. Serve and query

```bash
python3 -m venv ~/venv-srv
~/venv-srv/bin/pip install -r server/requirements.txt                              # serve-only, no torch
# the server embeds queries in-process unless CYTIR_TEI_URL names a TEI sidecar; in-process needs torch:
~/venv-srv/bin/pip install -r server/requirements-embed-local.txt -c server/constraints.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu   # skip this line behind a TEI sidecar

# CYTIR_COLLECTION below is the fixture collection the E2E path built.
# the `cytir` package is not in server/requirements.txt: install it from the app directory (no
# dependencies of its own; the Docker image does the same with a non-editable `pip install .`)
~/venv-srv/bin/pip install --no-deps -e .
cd server
CYTIR_QDRANT_URL=http://localhost:6333 \
CYTIR_COLLECTION=aw_gate_fixture \
CYTIR_EMBEDDER=arctic \
CYTIR_MRL_DIM=256 CYTIR_DENSE_RESCORE=1 CYTIR_DENSE_OVERSAMPLING=1.0 CYTIR_DENSE_IGNORE_QUANT=0 \
CYTIR_WEBAPP=$(pwd)/../webapp/index.html \
~/venv-srv/bin/uvicorn cytir.serve.app:api --host 127.0.0.1 --port 8080
```

A non-loopback Qdrant, TEI or API host must also be set in `CYTIR_QDRANT_URL` / `CYTIR_TEI_URL` / `CYTIR_API_URL`, or the internal client refuses it.

The four `CYTIR_DENSE_*` / `CYTIR_MRL_DIM` variables are the query side of the served layout
(`serving_config.flip_deploy_env()` is their one home). The server's default width is 128, so
a 256-wide collection served without `CYTIR_MRL_DIM=256` fails on a dimension mismatch.

Routes: `/search`, `/find_authority`, `/health`, `/` (human page), `/privacy`.
There is no `/healthz`.

Search operators (type them in `q`, or pass the named parameter; both merge):

| In the query | Parameter | Effect |
|---|---|---|
| `site:admin.ch` | `site=admin.ch` (repeatable) | only that host and its sub-hosts; several = OR; `www.` is ignored |
| `-site:wikipedia.org` | `exclude_site=wikipedia.org` (repeatable) | drop that host and its sub-hosts |
| `source:official` | `source=official` | only that `source_type`; an unknown value stays in the query as text and `filters.note` says so |
| `lang:de` | `lang=de` | same as the `lang` parameter; the operator wins on a clash |
| `country:ch` | `country=CH` | soft BOOST (results are not filtered); the operator and the `country` parameter both echo as `boosts.country`; the operator wins on a clash |

The response then has `query` (operators removed) and `filters` (what was applied, plus a `note`). `country:` is a soft boost, not a filter, so it is echoed under `boosts`, never `filters`. Valid `source:` values include `official`, `reference`, `docs`, `news`, `entity-self`; the vocabulary is a live facet and grows.
Without an operator the response is unchanged.

The collection a stranger holds is the fixture collection `aw_gate_fixture` (the E2E path at the top
builds it). It holds the seven stage packs of `data/index-fixture/`; its law is the Swiss-Austrian
border-traffic agreement (SR 0.631, ELI `cc/1948/195_183_155`) in DE and FR, not the Federal
Constitution. Query that:

```bash
curl -s "localhost:8080/search?q=Grenzverkehr+Abkommen+Schweiz+Oesterreich&k=3"
```

Each result is `{url, title, snippet, source_type, lang, trust_tier, score}`; the top URLs look like
`https://www.fedlex.admin.ch/eli/cc/1948/195_183_155/de#art_<n>`. The anchored `#art_<n>` URL is
the product: Cytir surfaces *where the authoritative text is*, it does not host it. The
Federal Constitution (`cc/1999/404`, section 2) can be fetched and chunked, but no documented verb
turns that into a served collection yet; the production index holds it.

> `/find_authority` returns `{"status":"degraded"}` on a fresh build. It needs a separate
> entity-card collection, and **no documented build path for one exists**. `/search` is
> unaffected. Tracked in task 429.

---

## CPU vs GPU: what CPU-only really costs

Measured on the rehearsal box — 8 shared vCPU, no GPU, `/usr/bin/time -v`:

| | |
|---|---|
| Passages | 732 |
| Wall clock | **10 min 37 s** (incl. one-time model download) |
| Throughput | **~1.2 passages/sec** |
| CPU used | 703 % of 8 cores |
| Peak RSS | **3.4 GB** |
| Precision | **fp32** (`torch.cuda.is_available()` false -> `use_fp16=False`) |

Extrapolating honestly, on a comparable 8-vCPU box:

| index size | CPU-only embed time |
|---|---|
| 10 k passages | ~2.3 hours |
| 100 k passages | ~1 day |
| 1 M passages | **~10 days** |
| 44 M (current production) | **~1.2 years — not viable** |

**CPU-only is genuinely fine for a small personal or single-domain index** (a statute
collection, one institution's docs, a few thousand pages). It is not a path to a
production-scale index; that needs GPUs, and constraint C1 sizes the full build at
4×A6000 for a few days. Retrieval/serving is a different story — serving is NVMe-bound,
not GPU-bound, and a CPU box serves a built index fine.

### Are CPU-built vectors compatible with GPU-built ones?

**Not verified.** Production embeds **fp16 on GPU**; this path embeds **fp32 on CPU**
(`cytir.embed.collection_stage.build_encoder`). The model, the MRL truncation and the L2
renormalization are identical, so they should be interchangeable to within fp16 rounding —
but task 429 **did not compare vectors from the two paths**, and this document will not
assert an equivalence nobody measured. If you intend to mix CPU-built and GPU-built
segments in one collection, measure it first.

### The embedder is revision-pinned

Both sides of the build read one file — the pin module under `cytir/core`, mirrored byte-identically
into `cytir/serve` (its `model_pin` module) because the aw-api image contains only `server/`:

```
Snowflake/snowflake-arctic-embed-l-v2.0 @ ac6544c8a46e00af67e330e85a9028c66b8cfd9a
```

`load_embedder()` applies it by default, so your document vectors come from that exact
commit and not from whatever HuggingFace serves on the day you run. The query side applies
it too (in-process embedder and the TEI sidecar's `--revision`), and aw-api checks TEI's
`/info` at first use and logs loudly if the served revision is not the pinned one — a
mismatch between write-time and read-time embeddings is invisible in every metric except
result quality.

Move the pin only for a deliberate re-embed: `CYTIR_MODEL_REVISION=<40-hex sha>`. It is
honoured, it rejects branch names (`main` is not a pin — it moves), and every log line
says the value is an override. Fixed in task 432; task 429 found it unpinned.

---

## What this walkthrough does not cover

- **The general web path.** Families that are not registered (`cyt source families` lists the
  ones that are) are an operator step: the packs under `sources/` name their source, but a
  stranger can drive only the registered families.
- **Choosing a source from the allowlist.** The YAMLs under `sources/` describe sources; the
  `extractor:` field some of them carry points at a private operator script that this
  repository does not hold.
- **The production rebuild.** The full production index is rebuilt by the operator from a
  locked manifest with private tooling. The public v1 target is the Swiss slice (scoreboard
  slice 14); this walkthrough is the part of it a stranger can run today.

## Politeness

The rehearsal fetched **3 documents from 1 source** (plus a handful of SPARQL metadata and
robots requests), all to `fedlex.admin.ch` hosts, single-threaded. If you follow this
guide, keep it small: this is a wiring test, not corpus acquisition. Respect
`robots.txt`, identify yourself honestly, and resolve consent live — step 3 is part of the
path, not an optional extra.

## MCP tools

The public server answers MCP at `https://mcp.websear.ch` (the `/mcp` path works too). `python -m cytir.serve.mcp_server` serves `/mcp` (Streamable HTTP) with two tools, both thin wrappers over the API routes above (no second implementation):

| Tool | Arguments | Calls | Use |
|---|---|---|---|
| `search` | `query`, `k`, `lang`, `country`, `site`, `exclude_site`, `source` | `GET /search` | the page that states the fact |
| `find_authority` | `query`, `k` (3, max 10), `country` | `GET /find_authority` | which SOURCE is authoritative (an authority card with `navigate.search_url`); `weak_match` or `note` means no good authority is known, fall back to `search` |

The response is passed through unchanged. An API error becomes a tool error with the HTTP status and the API's own `error` text; a non-JSON body (proxy HTML) is never echoed, only the status.
