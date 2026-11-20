# Build context is the product root (authoritative-web-app/), not server/ --
# see deploy/docker-compose.yaml's `aw-api` service (task 646 S2) -- so this
# Dockerfile can COPY both `cytir/` and `server/`. Every path below is relative
# to that root.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    AW_QDRANT_URL=http://qdrant:6333

WORKDIR /app

# CPU-only torch first (smaller, no CUDA), then the rest.
# Versions come from server/constraints.txt (exact freeze of what we serve).
COPY server/requirements.txt server/requirements-embed-local.txt server/constraints.txt ./
RUN pip install --no-cache-dir torch -c constraints.txt --index-url https://download.pytorch.org/whl/cpu
# The image serves with in-process query embedding available (requirements-embed-local.txt, task 2281): same package set as before the split.
RUN pip install --no-cache-dir -r requirements.txt -r requirements-embed-local.txt -c constraints.txt

# task 646 S2: install the `aw` package (cytir/core/model_pin.py, cytir/core/
# canon.py, cytir/core/ids.py) so the read side (cytir/serve/models.py) imports
# the SAME identity/canon code the build side does, instead of a hand-
# maintained byte-copy. Non-editable install: this is a built image, not a
# dev checkout. A separate layer from the shim below so a shim-only
# change does not invalidate this (slower, torch-adjacent) layer's cache.
# S10 step 9 (task 1130): the serving package is `cytir/serve`, part of this `aw` tree; the box
# bind-mounts its own render of it over /app/cytir/serve (deploy/docker-compose.yaml).
# Task 1140: the build also needs setup.py (package list), README.md (`readme =`) and data/
# (`[tool.setuptools.package-dir] "cytir._data" = "data"`, task 733); without them
# `pip install .` fails with "package directory 'data' does not exist". The withheld
# data files stay out of the layer via .dockerignore. tests/test_dockerfile_build_context.py
# builds the wheel from exactly these COPY sources.
COPY pyproject.toml setup.py README.md ./
COPY data ./data
COPY cytir ./cytir
RUN pip install --no-cache-dir -c /app/constraints.txt .

# German de-compounding table (ADR-121): fetched, never baked in. It lives in the
# `aw-models` volume like the other models; fetch it BEFORE this image serves:
#   docker compose run --rm aw-api python -m cytir.embed.fetch_models --asset charsplit-ngram
# The server refuses German tokens of 12+ chars without it (no silent no-split).
ENV AW_MODELS_DIR=/models/aw

# Models are NOT baked in: the `aw-models` volume masks /models at runtime, so
# they load there on first request and persist across recreates. Keeping them
# out of the image makes rebuilds fast.

EXPOSE 8080 8081
CMD ["uvicorn", "cytir.serve.app:api", "--host", "0.0.0.0", "--port", "8080"]
