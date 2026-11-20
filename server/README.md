# server — Cytir retrieval API, human search page, and MCP endpoint

> Until 2026-08-19 this file read *"Backend lives here once the stack is chosen. Empty
> scaffold."* — while `aw_server/` held ~5,600 lines of shipped, serving code. It was the
> single most misleading file in the repo: a stranger reading it concluded there was no
> server. Corrected by task 429.

`cytir/serve/` (formerly `server/aw_server/`; the shim is deleted, task 1147) is the live serving stack behind **https://websear.ch** (FastAPI).

| module | role |
|---|---|
| `app.py` | HTTP surface: `/search`, `/find_authority`, `/health`, `/` (human page), `/privacy` |
| `retrieval.py` | the ranking core — dense + BM25 legs, two-stage weighted score fusion, payload signals and penalties (ADR-025, ADR-035..037) |
| `models.py` | query encoder (`AW_EMBEDDER=arctic` by default) |
| `schema.py` | payload/record shape + env-tunable knobs |
| `mcp_server.py` | thin MCP (Streamable HTTP) proxy in front of the API — the agent-facing surface |

There is a `/health` route. There is **no** `/healthz`.

## Run it against a local index

Everything is env-configured with localhost defaults, so it starts against a local Qdrant
with no edits:

```bash
python3 -m venv ~/venv-srv && ~/venv-srv/bin/pip install -r requirements.txt
# with uv: see "Using uv instead of pip" in ../QUICKSTART.md (use `--python 3.12` and `-c constraints.txt`)

AW_QDRANT_URL=http://localhost:6333 \
AW_COLLECTION=<your-collection> \
AW_EMBEDDER=arctic \
AW_WEBAPP=$(pwd)/../webapp/index.html \
~/venv-srv/bin/uvicorn cytir.serve.app:api --host 127.0.0.1 --port 8080
```

For a full walkthrough that builds an index first, see [`../QUICKSTART.md`](../QUICKSTART.md)
— every command in it was executed on a clean box.

**Note:** `/find_authority` needs a separate entity-card collection and returns
`{"status":"degraded"}` without one. No build path for that collection exists in the repo
yet (task 429). `/search` is unaffected.

Deployment (compose, Caddy, the serving box) is `../deploy/` — `deploy/docker-compose.yaml`
is the single tracked source of the serving config.
