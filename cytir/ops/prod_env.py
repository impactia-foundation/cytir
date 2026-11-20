"""cytir.ops.prod_env — read prod's OWN serving environment from
deploy/docker-compose.yaml's aw-api block, and rewrite its CONTAINER paths
to this host's own copy.

MOVED HERE from task 514's `eval_arm.py` (task 646, weekly goal
2026-09-25): that file now imports these two functions from here instead of
carrying its own copy. There is exactly one place that says what prod's
serving configuration is -- deploy/docker-compose.yaml's aw-api environment
block -- so every caller reads it from here rather than restating it (task
514's own original finding: a second, hand-copied list of forty flags is
how an arm comes to measure a configuration nobody runs).
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

import yaml

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
from cytir.core.serve_layout import SERVER_REL  # noqa: E402

#: where the image bind-mounts the serving package (compose: `../cytir/serve:/app/cytir/serve:ro`)
IMAGE_SERVE_PREFIX = "/app/" + SERVER_REL + "/"
COMPOSE = REPO / "deploy" / "docker-compose.yaml"
OVERLAY_MANIFEST = REPO / "deploy" / "compose-overlays.manifest"
SERVICE = "aw-api"
#: The serving env a tree WITHOUT `deploy/` (the public cut) starts the server with: the compose
#: aw-api block plus its overlays, written out once, minus what is ours (see `NOT_IN_DEFAULT`).
#: Package data. It is GENERATED from the compose files (`--write-default`) and a test fails when
#: it drifts, so there is still exactly one place that says what the serving config is.
DEFAULT_ENV = pathlib.Path(__file__).resolve().parent / "serving_env.default.json"
DEFAULT_ENV_SCHEMA = "aw.serving-env-default.v1"
#: Keys the default never carries: they name a host or a collection of ours (the compose values are
#: container host names and prod's collection name) or a container path (AW_WEBAPP, AW_MODELS_DIR). `cyt serve` sets every one of
#: them from its own flags (`--qdrant`, `--collection`, `--tei-url`), or leaves it unset.
NOT_IN_DEFAULT = ("AW_QDRANT_URL", "AW_COLLECTION", "AW_TEI_URL", "AW_WEBAPP", "AW_MODELS_DIR")


def _service_env(path: pathlib.Path) -> dict:
    """The `aw-api` environment block of one compose file ({} when it has none)."""
    doc = yaml.safe_load(path.read_text()) or {}
    env = ((doc.get("services") or {}).get(SERVICE) or {}).get("environment") or {}
    if isinstance(env, list):
        out = {}
        for item in env:
            k, _, v = str(item).partition("=")
            out[k] = v
        return out
    return {str(k): "" if v is None else str(v) for k, v in env.items()}


def prod_env() -> dict:
    """The BASE compose file's aw-api env, overlays NOT applied. Kept as it was: the two
    live units that run `prod_env_run.py` (their flip values restated as
    `--extra`) must see the same env they always saw. Everything new calls `resolve_env`."""
    return _service_env(COMPOSE)


def overlay_files(manifest: pathlib.Path = OVERLAY_MANIFEST) -> list:
    """The overlays compose applies on top of the base, in order: one path per line of
    `deploy/compose-overlays.manifest` (relative to `deploy/`; blank lines and `#` comments
    ignored). Refuses a listed file that is missing, as `deploy/scripts/compose.sh` does."""
    out = []
    for line in manifest.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        f = manifest.parent / line
        if not f.is_file():
            raise SystemExit(f"{manifest.name} lists {line}, which does not exist at {f}: the "
                             "serving env would silently lack that overlay")
        out.append(f)
    return out


@dataclass
class ResolvedEnv:
    """`env`: the merged aw-api environment; `sources`: key -> the file that set it last."""
    env: dict
    sources: dict = field(default_factory=dict)


def declared_default(*, compose: pathlib.Path | None = None,
                     manifest: pathlib.Path = OVERLAY_MANIFEST) -> dict:
    """What `DEFAULT_ENV` holds: the compose aw-api block + every overlay (container paths left as
    they are, `to_repo_paths` maps them at resolve time), minus `NOT_IN_DEFAULT`."""
    compose = COMPOSE if compose is None else compose
    merged: dict = {}
    for f in [compose] + overlay_files(manifest):
        merged.update(_service_env(f))
    return {k: merged[k] for k in sorted(merged) if k not in NOT_IN_DEFAULT}


def default_document(**kw) -> dict:
    return {
        "schema": DEFAULT_ENV_SCHEMA,
        "why": "The serving environment `cyt serve` starts the API with when the tree has no "
               "deploy/docker-compose.yaml (the public cut). Generated from the compose aw-api "
               "block and its overlays by `python -m cytir.ops.prod_env --write-default`; a test "
               "fails when it differs. Neutral: no host, no collection name, no credential. "
               "`cyt serve` sets AW_COLLECTION, AW_QDRANT_URL, AW_TEI_URL and AW_MRL_DIM itself.",
        "env": declared_default(**kw)}


def default_env() -> "ResolvedEnv":
    """The shipped default, container paths mapped to this checkout (refusing a missing file)."""
    doc = json.loads(DEFAULT_ENV.read_text(encoding="utf-8"))
    if doc.get("schema") != DEFAULT_ENV_SCHEMA:
        raise SystemExit(f"{DEFAULT_ENV}: not an {DEFAULT_ENV_SCHEMA} file")
    env = {str(k): str(v) for k, v in doc["env"].items()}
    return ResolvedEnv(env=to_repo_paths(env), sources={k: DEFAULT_ENV.name for k in env})


def resolve_env(*, overlays: bool = True, compose: pathlib.Path | None = None,
                manifest: pathlib.Path = OVERLAY_MANIFEST) -> ResolvedEnv:
    """The ONE function that says what the aw-api container's environment is: the base
    compose file's aw-api block, then every overlay in the manifest, later files winning
    (docker compose's own merge). Container paths (`/app/cytir/serve/...`) are mapped to this
    checkout (`to_repo_paths`), refusing a file that is not there. `overlays=False` is the
    base alone, the old `prod_env_run.py` behaviour.

    A tree that has no compose file at all (the public cut: `deploy/**` never ships) resolves to
    the shipped default (`default_env`). Where the compose file exists it is the ONLY source: the
    default is not read, not merged, and cannot change what our boxes resolve."""
    default_tree = compose is None
    compose = COMPOSE if compose is None else compose
    if default_tree and not compose.is_file():
        return default_env()
    files = [compose] + (overlay_files(manifest) if overlays else [])
    merged, sources = {}, {}
    for f in files:
        for k, v in _service_env(f).items():
            merged[k] = v
            sources[k] = f.name
    return ResolvedEnv(env=to_repo_paths(merged), sources=sources)


def to_repo_paths(env: dict) -> dict:
    """Compose values are CONTAINER paths; a caller running on the bare host
    (not inside the aw-api container) needs the repo's own copy instead.

    `AW_HUBFLOOR_TABLE=/app/cytir/serve/hub_hosts.aw_b1.json` does not exist on
    the host, and the server's loader answers a missing table by DISABLING
    hubfloor with a one-line warning -- so a caller that copies the compose
    env verbatim silently measures/serves a different ranker than prod's own
    container runs. Rewrite the container mount prefix to the repo's own
    copy, and refuse if the file is still not there rather than letting the
    fallback decide.
    """
    out = dict(env)
    for key, value in env.items():
        if not value.startswith(IMAGE_SERVE_PREFIX):
            continue
        local = REPO / SERVER_REL / value[len(IMAGE_SERVE_PREFIX):]
        if not local.is_file():
            raise SystemExit(
                f"{key}={value} maps to {local}, which does not exist. The "
                "server would disable the feature it configures and the "
                "caller would run/measure a ranker prod does not.")
        out[key] = str(local)
    return out


def main(argv=None) -> int:
    """`python -m cytir.ops.prod_env --write-default | --check-default` (maintainers, from a tree that
    has `deploy/`)."""
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    if args not in (["--write-default"], ["--check-default"]):
        print("usage: python -m cytir.ops.prod_env {--write-default | --check-default}", file=sys.stderr)
        return 2
    want = json.dumps(default_document(), indent=1, ensure_ascii=False) + "\n"
    if args == ["--write-default"]:
        DEFAULT_ENV.write_text(want, encoding="utf-8")
        print(f"wrote {DEFAULT_ENV}")
        return 0
    if DEFAULT_ENV.read_text(encoding="utf-8") != want:
        print(f"{DEFAULT_ENV} differs from the compose files: regenerate with --write-default",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
