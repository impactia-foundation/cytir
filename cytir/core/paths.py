#!/usr/bin/env python3
# Moved from the retired `aw_paths` module (task 677, task 643 S5); the old path is an alias shim.
"""The one home for "where does this box keep things" (task 529).

Eight modules under `pipeline/` compiled an absolute path into a string
literal -- `<data-root>/...` for bench inputs, and in two cases the
repo's own location -- and one gate (`gate_country_restamp.py`) named a
worktree that had been deleted, so it could not run from any checkout that
exists. Engineering doctrine 2: a value that differs between machines is
named, defaulted in ONE place, and overridable; if reconstructing the current
value needs the history, it is archaeology, not configuration.

Two names, both env-overridable:

  AW_REPO_ROOT   the checkout (default: two directories above this file
                 (cytir/core/), which is correct in a worktree, a clone, and a container alike)
  AW_DATA_ROOT   the box's scratch/bench data (default: ~/aw-data)

and one derived answer, `data_dir()` / `data_path()`: where the repo's `data/` contracts live
(task 733). In a checkout (a clone, a worktree, `pip install -e .`, or `AW_REPO_ROOT` set) it is
`<repo>/data`. In a NON-editable install it is `cytir/_data` inside the installed package: setup.py
and pyproject.toml ship the public cut's runtime data files there. `cyt --version` and
`python -m cytir.core.paths --get data_dir` say which one is in force. A reader of a `data/` file calls `data_path(name)`; it never
computes `parents[N] / "data"` (tests/test_package_data.py refuses a new such reader).

Nothing here creates directories or fetches anything. It answers a question.
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import sys

from cytir.core import envnames as _envnames

REPO_ROOT = pathlib.Path(
    os.environ.get("AW_REPO_ROOT")
    or pathlib.Path(__file__).resolve().parents[2])

DATA_ROOT = pathlib.Path(
    os.environ.get("AW_DATA_ROOT")
    or (pathlib.Path.home() / "aw-data"))

# --------------------------------------------------------------------------- operator settings
# Task 696 slice 4. Every value that differs between OUR boxes and a stranger's is a named
# setting, resolved ONCE per process, in this order:
#   1. an explicit argument (`load_settings(explicit=...)`)
#   2. the environment (the canonical `env` of each spec, then its deprecated aliases)
#   3. the operator overlay file: `AW_OPERATOR_OVERLAY`, else the box file `/etc/aw/operator.json`
#      when it exists (and only then: no other implicit location)
#   4. a default that is neutral by construction: it carries no host and no path, it only
#      DERIVES from the work root (`queue_dir` = `<work_root>/promotion-queue`).
# There is NO default for the work root and none for anything that names a host, a release
# or an identity. A setting nobody configured is a loud `SettingRefused` on first ACCESS (never
# at import), naming the setting, its env var and the overlay. A path that silently fell back to
# `./aw-work` would move the promotion queue and the writer locks to wherever the process
# happened to start, and a lock taken in a different place is no lock (review of #563/#567).
#
# `AW_WORK_ROOT` set by argument or env makes a SCRATCH run: every setting derived from the
# work root (queue, locks, GPU lock, corpus, news dirs) is derived from it and the overlay's
# values for them are ignored, so a scratch run cannot half-share a lock or a queue with the box.
# The overlay (`deploy/operator/*.json`) is operator state: outside the public cut.
OVERLAY_ENV = "AW_OPERATOR_OVERLAY"
OVERLAY_SCHEMA = "aw.operator-overlay.v1"
#: The conventional system path of the box-level overlay (a path, not an identity of ours).
BOX_OVERLAY = pathlib.Path("/etc/aw/operator.json")
#: TESTS ONLY: replaces the default overlay lookup (the box file) with this path ("" = no
#: overlay), so a test never reads the machine's real `/etc/aw/operator.json`.
TEST_LOOKUP_ENV = "AW_TEST_OVERLAY_LOOKUP"

_log = logging.getLogger("cytir.core.paths")


class SettingsError(RuntimeError):
    """The overlay is unreadable, malformed, or names a setting that does not exist."""


class SettingRefused(RuntimeError):
    """A setting with no honest default was read and nobody configured it."""


class _Spec:
    __slots__ = ("name", "env", "aliases", "neutral", "what", "derived")

    def __init__(self, name, env, neutral, what, *, aliases=(), derived=False):
        self.name, self.env, self.neutral, self.what = name, env, neutral, what
        self.aliases, self.derived = tuple(aliases), derived


class _Refusal:
    def __init__(self, message: str, root: str | None = None):
        self.message = message
        self.root = root           # the configurable setting this refusal ultimately comes from


class _DependsOnRefused(Exception):
    def __init__(self, base):
        self.base = base


class _Values(dict):
    """The values resolved so far; reading one that was refused aborts the derivation."""

    def __getitem__(self, k):
        v = dict.__getitem__(self, k)
        if isinstance(v, _Refusal):
            raise _DependsOnRefused(k)
        return v


def _local_beside(work_root: str) -> str:
    """`local`: this machine, which is where a configured work root lives. Reading `work_root`
    (the argument) is the point: with none configured the read refuses, so the host does too."""
    return "local"


def _under(base: str, *parts: str):
    return lambda v: os.path.join(v[base], *parts)


#: name, canonical env var, default (callable on the values resolved so far; None = refuse),
#: what. `derived` = computed from the work root, so a scratch run (AW_WORK_ROOT) re-derives it.
#: Order matters: a default may read an earlier setting. Aliases are the older names that already
#: exist in the repo for the same thing; accepted with one deprecation line, a conflict refuses.
_SPECS: tuple[_Spec, ...] = (
    _Spec("work_root", "AW_WORK_ROOT", None, "root of all scratch/staging work on this box"),
    _Spec("hdd_root", "AW_HDD_ROOT", lambda v: v["work_root"],
          "bulk-data disk holding the corpus store", derived=True),
    _Spec("corpus_root", "AW_CORPUS_ROOT", _under("hdd_root", "aw-corpus"), "ADR-026 corpus store",
          derived=True),
    _Spec("l0_store_root", "AW_L0_STORE_ROOT", _under("hdd_root", "aw-corpus-l0"),
          "default root of the compressed L0 store", aliases=("AW_L0_ROOT",), derived=True),
    _Spec("fallback_models_dir", "AW_FALLBACK_MODELS_DIR", lambda v: "",
          "an extra implicit model directory searched after AW_MODELS_DIR (empty = none)"),
    _Spec("releases_root", "AW_RELEASES_ROOT", _under("work_root", "releases"),
          "immutable release cuts", derived=True),
    _Spec("queue_dir", "AW_QUEUE_DIR", _under("work_root", "promotion-queue"), "promotion queue",
          aliases=("AW_PROMOTION_QUEUE_DIR",), derived=True),
    # Delta change log (task 2314, ruling 68d): where the drain writes one segment per landed pack.
    # Empty = OFF (the writer is a no-op and the drain is byte-identical); master-side only, never a pack dir.
    _Spec("change_log_dir", "AW_CHANGE_LOG_DIR", lambda v: "",
          "directory of the drain's change-log segments (empty = no change log written)"),
    # Delta fingerprint ledger (task 2590, specs/delta-fingerprint-base.md): the master-side per-row
    # fingerprints of the last PUBLISHED pack, which `cyt dump` diffs the next delta against. Empty = OFF
    # (a delta is diffed against the base manifest's own sidecar, as before). Never shipped.
    _Spec("fingerprint_ledger_root", "AW_FINGERPRINT_LEDGER_ROOT", lambda v: "",
          "directory of the delta fingerprint ledger (empty = deltas diff against the base manifest)"),
    _Spec("gpu_lock_dir", "AW_GPU_LOCK_DIR", _under("work_root", "GPU-LOCK"),
          "legacy GPU lock directory", derived=True),
    _Spec("news_work", "AW_NEWS_WORK", _under("work_root", "newsfresh"), "news refresh work tree",
          derived=True),
    _Spec("refresh_github_docs_root", "AW_REFRESH_GH_DOCS_ROOT",
          _under("work_root", "refresh_lane_direct", "github-repo-docs"),
          "per-run preserve roots of the refresh lane's github_repo_docs adapter", derived=True),
    _Spec("refresh_direct_work", "AW_REFRESH_DIRECT_WORK", _under("work_root", "refresh_lane_direct"),
          "work tree, state.json and NEXT.md of the refresh lane's direct-compose driver", derived=True),
    # The b1 builder (`cyt index b1-rebuild`, task 2123). The corpus store it reads, the directory it
    # writes its state/manifest/consent cache under, and the command line a STOP message tells the
    # operator to run to resume a full build.
    _Spec("b1_corpus_dir", "AW_B1_CORPUS_DIR", lambda v: v["corpus_root"],
          "corpus store the b1 builder reads (default: the ADR-026 corpus store)", derived=True),
    _Spec("b1_build_dir", "AW_B1_BUILD_DIR", _under("work_root", "aw_b1_build"),
          "directory of the b1 builder's state, manifest, tail list and consent cache", derived=True),
    _Spec("b1_resume_prefix", "AW_B1_RESUME_PREFIX",
          lambda v: f"AW_FOLD_ACCENTS=1 {sys.executable} -m cytir.index.b1_cli",
          "command prefix a b1 full-build STOP message gives for the resume line"),
    _Spec("news_config_name", "AW_NEWS_CONFIG_NAME", lambda v: "current.env",
          "file name of the news config under <news_work>/state"),
    _Spec("hn_work", "AW_HN_WORK", _under("work_root", "hn_discussion"),
          "Hacker News discussion work tree", derived=True),
    _Spec("news_backfill_dir", "AW_NEWS_BACKFILL_DIR", _under("work_root", "news_backfill"),
          "news backfill state", derived=True),
    _Spec("qpy", "AW_QPY", lambda v: sys.executable, "interpreter with qdrant_client for the news chain"),
    # `local` only on a box that HAS a work root: "this machine, next to the work root" is the
    # scratch / single-box case. With no work root there is no honest default for the host (a lock
    # taken on the wrong host is no lock), so it refuses by name like the rest (review of #567).
    _Spec("lock_host", "AW_LOCK_HOST", lambda v: _local_beside(v["work_root"]),
          "host holding the cross-task write locks", derived=True),
    _Spec("lock_dir", "AW_LOCK_DIR", lambda v: v["work_root"],
          "directory of the write locks on lock_host", derived=True),
    _Spec("prod_lock_host", "AW_PROD_LOCK_HOST", None, "host of the production writer mutex"),
    # The between-batches box-health guard of a prod payload write (task 1241/1247). Empty = not
    # configured: only `cyt ops payload-profile published --apply` needs them, and it refuses by name.
    _Spec("prod_qdrant_container", "AW_PROD_QDRANT_CONTAINER", lambda v: "",
          "the prod qdrant container whose RestartCount the write guard watches (empty = none)"),
    _Spec("prod_health_search_url", "AW_PROD_HEALTH_SEARCH_URL", lambda v: "",
          "the one search URL the prod write guard probes between batches (empty = none)"),
    _Spec("twin_ssh_host", "AW_TWIN_SSH_HOST", None, "ssh host that reaches the twin qdrant"),
    # The TWIN: the build box's qdrant that every drain writes first and every gate reads (task 832).
    # ONE place names it, so the aw_e10 flip moves it with one edit (the box overlay), not fifteen.
    # The defaults are today's twin on its own box (the build box, loopback); they carry no secret and no
    # path. `cytir.core.twin` is the accessor; a script takes it with `aw_setting_in "$APP" twin_url`.
    # Not in impactia.json yet: a key enters the box file only when every checkout knows it.
    _Spec("twin_url", "AW_TWIN_URL", lambda v: "http://127.0.0.1:6457",
          "URL of the twin qdrant (the build box's copy that every write goes to first)"),
    _Spec("twin_collection", "AW_TWIN_COLLECTION", lambda v: "aw_e8",
          "physical collection name on the twin qdrant"),
    _Spec("langgate_py", "AW_NEWS_LANGGATE_PY", None,
          "interpreter with the lang-gate stack (numpy<2 + fasttext)",
          aliases=("AW_LANGGATE_PYTHON_BIN",)),
    # NOT an alias of AW_RELEASE_DIR: that name means "the immutable release THIS process runs
    # from" (the ERIC chain sets it to its own release), a different thing from the news lane's
    # pinned release. Treating it as one would move `news.daily.R` under the ERIC environment.
    _Spec("news_release_dir", "AW_NEWS_RELEASE_DIR", None, "the pinned immutable news release"),
    # The crawler identity (`cytir.core.user_agent` builds the User-Agent from it). Env first, then the
    # checkout's own `operator.json`, then these (the box overlay), so every entry point on a box
    # has ONE identity without a per-unit environment line. A stranger has none and still fails
    # closed: there is no default, and `user_agent()` refuses. NOT in impactia.json yet (step B2): a
    # key may enter the box file only when every checkout that reads it knows the key.
    _Spec("operator_url", "AW_OPERATOR_URL", None, "crawler identity: a URL a human can read"),
    _Spec("operator_contact", "AW_OPERATOR_CONTACT", None, "crawler identity: an address a human can write to"),
    # The lock file the hourly promotion unit takes (`flock -n <file> ...`). Empty = none: a
    # stranger's box has no such unit, and a hand run of `cyt index promote` there needs no lock.
    # Not in impactia.json yet: a key enters the box file only when every checkout knows it.
    _Spec("promotion_lock", "AW_PROMOTION_LOCK", lambda v: "",
          "lock file of the hourly promotion unit; a hand run of `cyt index promote` takes it (empty = none)"),
    # Where the drain keeps the restore copy of every row a URL-group supersede deletes (ADR-067
    # am.1 point 4, task 712). Empty = none: no default path on purpose (a default would be a
    # scratch path, and a restore copy must outlive the work tree), and a URL-group supersede
    # REFUSES while it is empty (`cytir.index.url_group_supersede.export_dir`). The directory must
    # already exist. Not in impactia.json yet: a key enters the box file only when every checkout
    # knows it (same as `promotion_lock`).
    _Spec("removed_rows_export_dir", "AW_REMOVED_ROWS_EXPORT_DIR", lambda v: "",
          "durable directory (must exist) for the restore copy of rows a URL-group supersede deletes"
          " (empty = none: such a supersede refuses)"),
    # Guard 1 of task 710 (`cytir.index.german_parts_guard`): `warn` writes a German pack whose rows
    # lack the de-compounded parts and says so loudly; `refuse` refuses it. Defaults to `warn`
    # until the writers all have the splitter (PR #628); one value flips it.
    _Spec("german_parts_guard", "AW_GERMAN_PARTS_GUARD", lambda v: "warn",
          "German parts guard: `warn` (loud line + count) or `refuse` (the pack is refused)"),
    # The sealed term set of the aw_e9 BM25 prune rule (task 717, `cytir.index.sparse_rule`): a sorted uint32
    # file whose sha256 `data/collection-payload-profiles.v1.json` pins. Empty = none, and no default
    # path (a default would be a scratch path): while it is empty a prod write REFUSES by name
    # (`cytir.index.sparse_rule`), never writes an unpruned row. Not in impactia.json yet: a key enters the box file only when every checkout knows it
    # (same as `removed_rows_export_dir`); the drain gets it from a systemd drop-in.
    _Spec("sparse_rule_drop_terms", "AW_SPARSE_RULE_DROP_TERMS", lambda v: "",
          "sealed term set (uint32 file) of the BM25 prune rule the prod collection declares (empty = none: a prod write refuses)"),
    _Spec("storagebox", "AW_BOX", None, "user@host of the off-box evidence store"),
    _Spec("box_rsh", "AW_BOX_RSH", None, "ssh command line that reaches the off-box evidence store"),
)
_BY_NAME = {s.name: s for s in _SPECS}
#: Settings where an empty string is a mistake, not "none": it refuses by name (task 832).
_NEVER_EMPTY = frozenset({"twin_url", "twin_collection"})


class Settings:
    """Resolved operator settings. Attribute access; a refused setting raises on access."""

    def __init__(self, values: dict, sources: dict, overlay=None):
        object.__setattr__(self, "_values", values)
        object.__setattr__(self, "_sources", sources)
        object.__setattr__(self, "overlay_file", overlay)

    def __getattr__(self, name: str):
        try:
            v = self._values[name]
        except KeyError:
            raise AttributeError(name) from None
        if isinstance(v, _Refusal):
            raise SettingRefused(v.message)
        return v

    def __setattr__(self, name, value):
        raise AttributeError("Settings is read-only")

    def source(self, name: str) -> str:
        """Where `name` came from: `arg` | `env VAR` | `overlay PATH (how found)` |
        `default` | `REFUSED`."""
        return self._sources[name]

    def as_dict(self) -> dict:
        """Every setting; a refused one is None (for tests and printing, never for logic)."""
        return {k: (None if isinstance(v, _Refusal) else v) for k, v in self._values.items()}


def _read_overlay(path: pathlib.Path) -> dict:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SettingsError(f"operator overlay {path} could not be read ({exc})") from exc
    if not isinstance(doc, dict) or doc.get("schema") != OVERLAY_SCHEMA \
            or not isinstance(doc.get("settings"), dict):
        raise SettingsError(f"operator overlay {path}: need {{'schema': '{OVERLAY_SCHEMA}', "
                            f"'settings': {{...}}}}")
    unknown = sorted(set(doc["settings"]) - set(_BY_NAME))
    if unknown:
        raise SettingsError(f"operator overlay {path} names unknown settings {unknown}")
    return {k: str(v) for k, v in doc["settings"].items()}


def overlay_lookup(env=None) -> tuple[pathlib.Path | None, str]:
    """(path, how it was found). `AW_OPERATOR_OVERLAY` must name a file (else an error); then
    the box file `/etc/aw/operator.json` if it exists; else none (nothing else is implicit)."""
    env = _envnames.mirrored(os.environ if env is None else env)
    named = env.get(OVERLAY_ENV)
    if named:
        p = pathlib.Path(named)
        if not p.is_file():
            raise SettingsError(f"{OVERLAY_ENV}={named} is not a file")
        return p, f"env {OVERLAY_ENV}"
    if TEST_LOOKUP_ENV in env:
        t = env[TEST_LOOKUP_ENV]
        if t and pathlib.Path(t).is_file():
            return pathlib.Path(t), f"test lookup {TEST_LOOKUP_ENV}"
        return None, "none"
    if BOX_OVERLAY.is_file():
        return BOX_OVERLAY, "box default"
    return None, "none"


def box_file_setting(name: str, env=None) -> str | None:
    """What the BOX file (`/etc/aw/operator.json`) ALONE resolves for setting `name`: the file is
    loaded with an EMPTY environment (no `AW_*` variable, no `AW_OPERATOR_OVERLAY`), through the
    same `load_settings` every caller uses, so a setting the file does not name but derives (impactIA's
    file names `work_root`, and `queue_dir` follows from it) resolves exactly as it does at run time.
    None when there is no box file or the setting REFUSES under it. A test session never reads the
    machine's real file (`TEST_LOOKUP_ENV` replaces it, "" = none). A box file that exists and cannot
    be read raises `SettingsError`."""
    env = _envnames.mirrored(os.environ if env is None else env)
    if TEST_LOOKUP_ENV in env:
        t = env[TEST_LOOKUP_ENV]
        path = pathlib.Path(t) if t and pathlib.Path(t).is_file() else None
    else:
        path = BOX_OVERLAY if BOX_OVERLAY.is_file() else None
    if path is None:
        return None
    try:
        return getattr(load_settings({}, overlay=path), name)
    except SettingRefused:
        return None


def overlay_path(env=None) -> pathlib.Path | None:
    return overlay_lookup(env)[0]


def overlay_value(name: str, env=None) -> str | None:
    """What the overlay in force says for setting `name`, read from the file on every call and
    ignoring the environment and the process cache; None when no overlay or it does not name it.
    The crawler identity uses this so that it keeps reading the environment fresh itself."""
    path = overlay_lookup(env)[0]
    return _read_overlay(path).get(name) if path else None


_deprecated_seen: set = set()


def _from_env(spec: _Spec, env):
    """(value, var) from the canonical env name or an alias; (None, None) when unset.
    An alias warns once per process; two names set to different values is a refusal.
    The `CYTIR_*` spelling of a name is mirrored first, on a copy (task 2964)."""
    env = _envnames.mirrored(env)
    found = [(n, env[n]) for n in (spec.env, *spec.aliases) if env.get(n)]
    if not found:
        return None, None
    if len({v for _, v in found}) > 1:
        return _Refusal(f"setting `{spec.name}`: " + " and ".join(f"{n}={v!r}" for n, v in found)
                        + f" disagree; set only {spec.env}"), "conflict"
    name, value = found[0]
    if name != spec.env and (name, spec.env) not in _deprecated_seen:
        _deprecated_seen.add((name, spec.env))
        _log.warning("%s is deprecated for the `%s` setting; use %s", name, spec.name, spec.env)
    return value, name


def _refusal_text(s: _Spec, needed_by: str | None = None) -> str:
    who = f"setting `{needed_by}` derives from `{s.name}`, which" if needed_by else f"setting `{s.name}`"
    return (f"{who} ({s.what}) is not configured: set {_envnames.shown(s.env)}, or put `{s.name}` in the operator "
            f"overlay ({_envnames.shown(OVERLAY_ENV)}, or the box file {BOX_OVERLAY}). A stranger's first run: "
            f"export CYTIR_WORK_ROOT=$HOME/aw-work")


def load_settings(env=None, *, explicit: dict | None = None, overlay=False) -> Settings:
    """Resolve every setting. `overlay=False` = look it up (`overlay_lookup`); `overlay=None` =
    no overlay at all; a path = that overlay."""
    env = _envnames.mirrored(os.environ if env is None else env)
    explicit = explicit or {}
    bad = sorted(set(explicit) - set(_BY_NAME))
    if bad:
        raise SettingsError(f"unknown settings {bad}")
    how = "given"
    if overlay is False:
        overlay, how = overlay_lookup(env)
    ov = _read_overlay(pathlib.Path(overlay)) if overlay else {}
    ov_tag = f"overlay {overlay}" + (f" ({how})" if how != "given" else "")
    scratch = "work_root" in explicit or _from_env(_BY_NAME["work_root"], env)[0] is not None
    values: _Values = _Values()
    sources: dict = {}
    for s in _SPECS:
        ev, evar = (None, None) if s.name in explicit else _from_env(s, env)
        if s.name in explicit:
            values[s.name], sources[s.name] = str(explicit[s.name]), "arg"
        elif isinstance(ev, _Refusal):
            values[s.name], sources[s.name] = ev, "REFUSED (conflicting env names)"
        elif ev is not None:
            values[s.name], sources[s.name] = ev, f"env {evar}"
        elif s.name in ov and not (scratch and s.derived):
            values[s.name], sources[s.name] = ov[s.name], ov_tag
        elif s.neutral is not None:
            try:
                values[s.name] = s.neutral(values)
                sources[s.name] = "default"
            except _DependsOnRefused as d:
                # name the ROOT of the chain (`corpus_root` -> `hdd_root` -> `work_root`)
                root = dict.__getitem__(values, d.base).root or d.base
                values[s.name] = _Refusal(_refusal_text(_BY_NAME[root], needed_by=s.name), root=root)
                sources[s.name] = "REFUSED"
        else:
            values[s.name] = _Refusal(_refusal_text(s), root=s.name)
            sources[s.name] = "REFUSED"
        v = dict.__getitem__(values, s.name)
        if s.name in _NEVER_EMPTY and isinstance(v, str) and not v.strip():
            # an empty twin silently turns every twin check off downstream (compose skips them)
            values[s.name] = _Refusal(f"setting `{s.name}` ({s.what}) is EMPTY in {sources[s.name]}: "
                                      f"set a value or remove the key", root=s.name)
            sources[s.name] = "REFUSED (empty)"
    return Settings(dict(values), sources, overlay)


_cached: Settings | None = None


def env_names() -> frozenset:
    """Every env name (and alias) the package's declared settings read (task 1792)."""
    return frozenset(n for s in _SPECS for n in (s.env, *s.aliases) if n)


def operator_settings() -> Settings:
    """The process-wide settings, resolved once from the environment at first use."""
    global _cached
    if _cached is None:
        _cached = load_settings()
    return _cached


def reset_settings_cache() -> None:
    """Tests only."""
    global _cached
    _cached = None


def describe(s: Settings, name: str) -> str:
    """`name = value  [source]`, or `name = REFUSED  [source]`; never raises."""
    v = s.as_dict()[name]
    return f"{name} = {'REFUSED' if v is None else v}  [{s.source(name)}]"


def echo_lines(s: Settings | None = None) -> list[str]:
    """Every setting with its effective value and source (the config echo)."""
    s = s or operator_settings()
    return [describe(s, spec.name) for spec in _SPECS]


def location_line(label: str, effective: str | None, setting: str) -> str:
    """One log line for a drain or compose start: where the queue or the locks are, and why.
    `effective` is what the process will really use (it may be a command-line argument); None =
    nothing was passed, so the setting decides. Never raises: a refused setting prints REFUSED."""
    s = operator_settings()
    src = s.source(setting)
    configured = s.as_dict()[setting]
    if effective is None or configured == effective:
        return f"[config] {label} = {configured or 'REFUSED'}  [{src}]"
    return (f"[config] {label} = {effective}  [command-line argument; setting {setting} = "
            f"{configured or 'REFUSED'}  [{src}]]")


def lock_location_line() -> str:
    """The write-lock location (host:dir) with the source of each half, for a startup log line.
    Never raises: an unconfigured location is printed as REFUSED, the use site refuses by name."""
    s = operator_settings()
    env_host, env_dir = os.environ.get("AW_LOCK_HOST"), os.environ.get("AW_LOCK_DIR")
    host = env_host or s.as_dict()["lock_host"] or "REFUSED"
    directory = env_dir or s.as_dict()["lock_dir"] or "REFUSED"
    hsrc = "env AW_LOCK_HOST" if env_host else s.source("lock_host")
    dsrc = "env AW_LOCK_DIR" if env_dir else s.source("lock_dir")
    return f"[config] write-lock = {host}:{directory}  [host: {hsrc}; dir: {dsrc}]"


def refused_settings(*names: str) -> list[str]:
    """The refusal text of every named setting that nobody configured (empty = all resolve)."""
    s = operator_settings()
    out = []
    for n in names:
        try:
            getattr(s, n)
        except SettingRefused as exc:
            out.append(f"{n}: {exc}")
    return out


def startup_refusal(prog: str, *names: str) -> int | None:
    """Fail at START, not mid-run: when any named setting is unconfigured, print one `[REFUSED]`
    line per setting (its name first) to stderr and return the exit code to use (2); else None.
    A driver calls this before it opens a queue or writes anything, so a missing overlay is a red
    unit, not a green one that silently refused every pack (review of #567, item 2)."""
    bad = refused_settings(*names)
    for line in bad:
        print(f"[REFUSED] {prog}: setting {line}", file=sys.stderr, flush=True)
    return 2 if bad else None


def box_overlay_drift(box: pathlib.Path | None = None, repo: pathlib.Path | None = None) -> str | None:
    """A WARNING line when the box file and the checkout's `deploy/operator/impactia.json` both
    exist and name different settings (the box file shadows the repo copy until re-installed);
    None when they agree or either is absent. Never raises. `box` defaults to the box file (or the
    test lookup, `AW_TEST_OVERLAY_LOOKUP`, when a test sets it)."""
    if box is None:
        look = os.environ.get(TEST_LOOKUP_ENV)
        box = BOX_OVERLAY if look is None else (pathlib.Path(look) if look else None)
    repo = repo or REPO_ROOT / "deploy" / "operator" / "impactia.json"
    try:
        if box is None or not (box.is_file() and repo.is_file()):
            return None
        have, want = _read_overlay(box), _read_overlay(repo)
    except (OSError, SettingsError):
        return None
    diff = sorted(k for k in set(have) | set(want) if have.get(k) != want.get(k))
    if not diff:
        return None
    return (f"WARNING: {box} differs from {repo} in {', '.join(diff)}; the box file wins. "
            f"Re-install it from the checkout (`install -D -m 0644 {repo} {BOX_OVERLAY}`) and run "
            f"`python3 deploy/operator/check_box_overlay.py`.")


#: The two roots that are not operator settings (they are derived from the checkout and the user),
#: readable by name through `--get` so a shell script need not re-derive them.
_ROOT_NAMES = {"repo_root": lambda: str(REPO_ROOT), "data_root": lambda: str(DATA_ROOT),
               "data_dir": lambda: str(data_dir())}


def get_value(name: str) -> str:
    """The value of one setting (or `repo_root` / `data_root`) for a shell caller.
    Raises `SettingRefused` (unconfigured), `SettingsError` (bad overlay) or `KeyError` (no such name)."""
    if name in _ROOT_NAMES:
        return _ROOT_NAMES[name]()
    if name not in _BY_NAME:
        raise KeyError(name)
    return str(getattr(operator_settings(), name))


def main(argv=None) -> int:
    """`python -m cytir.core.paths`: print every setting, its value (or REFUSED) and its source.

    `python -m cytir.core.paths --get NAME` prints just that one value (nothing else on stdout) so a
    shell script can take a box path from the settings. Any settings error (an unconfigured
    setting, a malformed overlay, an unknown overlay key) prints ONE `[REFUSED] <setting>: <reason>`
    line to stderr and exits 2; an unknown name exits 64."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--get"]:
        if len(argv) != 2:
            print("usage: python -m cytir.core.paths --get NAME", file=sys.stderr)
            return 64
        try:
            print(get_value(argv[1]))
        except (SettingRefused, SettingsError) as exc:
            # ONE line, no traceback, for every settings error: an unconfigured setting, an
            # unreadable or malformed overlay, an overlay naming a key this checkout does not know.
            print(f"[REFUSED] {argv[1]}: {' '.join(str(exc).split())}", file=sys.stderr)
            return 2
        except KeyError:
            names = ", ".join(sorted([*_BY_NAME, *_ROOT_NAMES]))
            print(f"unknown setting {argv[1]!r}; known: {names}", file=sys.stderr)
            return 64
        return 0
    s = operator_settings()
    print(f"# operator overlay: {s.overlay_file if s.overlay_file else 'none'}")
    print("\n".join(echo_lines(s)))
    d = s.as_dict()
    if d["operator_url"] is None or d["operator_contact"] is None:
        print("# note: operator_url / operator_contact are not in the box overlay yet (step B2). The "
              "crawler identity comes from the environment or the checkout's operator.json; without "
              "either, `user_agent()` refuses. Their REFUSED line above is not a path-setting fault.")
    warn = box_overlay_drift()
    if warn:
        print(warn)
    return 0


class LazyPath(os.PathLike):
    """A path that resolves on USE, not on import.

    Module constants such as `news.daily.W` or `QUEUE_DIR` used to be computed when the module
    loaded. With no default for the work root they would refuse at import, so an unconfigured
    box could not even run `--help`. A `LazyPath` defers: `str(p)`, `os.fspath(p)`, `p / "x"`
    (itself lazy), `p.exists()` and every `pathlib.Path` attribute resolve through
    `operator_settings()` at the moment of use, and a refused setting raises `SettingRefused`
    there, by name. `repr()` never raises."""

    __slots__ = ("_fn", "_label")

    def __init__(self, fn, label: str):
        object.__setattr__(self, "_fn", fn)
        object.__setattr__(self, "_label", label)

    def _p(self) -> pathlib.Path:
        return pathlib.Path(self._fn())

    def __fspath__(self) -> str:
        return os.fspath(self._p())

    def __str__(self) -> str:
        return str(self._p())

    def __repr__(self) -> str:
        try:
            return f"LazyPath({str(self._p())!r})"
        except SettingRefused:
            return f"LazyPath(<{self._label}: not configured>)"

    def __truediv__(self, other):
        return LazyPath(lambda: self._p() / other, f"{self._label}/{other}")

    def __eq__(self, other):
        # An unconfigured setting RAISES here (never `False`): "not equal" would read as an answer.
        try:
            other_path = pathlib.Path(os.fspath(other))
        except TypeError:
            return NotImplemented
        return self._p() == other_path

    def __hash__(self):
        return hash(self._p())

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._p(), name)


def lazy_setting_path(name: str) -> LazyPath:
    """A `LazyPath` for the operator setting `name` (resolved through `operator_settings()` on use)."""
    return LazyPath(lambda: getattr(operator_settings(), name), name)


def work_path(*parts: str) -> str:
    """A path under the work root."""
    return os.path.join(operator_settings().work_root, *parts)


# The ADR-026 corpus store (L0 originals / L1 documents / L2 passages), the layer every collection
# must be rebuildable from. `CORPUS_ROOT` resolves on first ACCESS (module `__getattr__`), never at
# import: an unconfigured box refuses by name where the path is used, not when a module loads.
def __getattr__(name):
    if name == "CORPUS_ROOT":
        return pathlib.Path(operator_settings().corpus_root)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def corpus_path(*parts: str) -> str:
    """A path inside the ADR-026 corpus store."""
    return str(pathlib.Path(operator_settings().corpus_root).joinpath(*parts))


def data_dir_line() -> str:
    """One line saying where `data/` is read from and why (for `cyt --version` and the echo)."""
    why = ("checkout: AW_REPO_ROOT" if os.environ.get("AW_REPO_ROOT")
           else "checkout" if _is_checkout() else "installed package data")
    return f"# data dir: {data_dir()} [{why}]"


def repo_path(*parts: str) -> str:
    """A path inside the checkout."""
    return str(REPO_ROOT.joinpath(*parts))


def _is_checkout() -> bool:
    """True when `REPO_ROOT` is a real checkout, not the site-packages dir of an install."""
    if os.environ.get("AW_REPO_ROOT"):
        return True
    return (REPO_ROOT / "pyproject.toml").is_file() and (REPO_ROOT / "data").is_dir()


def data_dir() -> pathlib.Path:
    """The directory holding the repo's `data/` files (task 733); see the module docstring."""
    explicit = os.environ.get("AW_REPO_ROOT")
    if explicit:
        return pathlib.Path(explicit) / "data"
    if _is_checkout():
        return REPO_ROOT / "data"
    # Non-editable install: `cytir/_data` beside the code (package data, pyproject.toml).
    from importlib import resources
    return pathlib.Path(str(resources.files("cytir") / "_data"))


def data_path(*parts: str) -> pathlib.Path:
    """A file under `data_dir()`. Names the file; does not check it exists."""
    return data_dir().joinpath(*parts)


class SourcesNotFound(RuntimeError):
    """`sources/` (the pack declarations) was asked for and cannot be found."""


def sources_dir() -> pathlib.Path:
    """The directory holding the pack declarations, `sources/**/*.yaml` (task 735).

    `sources/` is POLICY, not package data: it is declared by people, merged by the founder, and
    the public cut ships it as files next to the package. So an installed `aw` has no copy and
    looks here, in order: `AW_SOURCES_DIR`; `<AW_REPO_ROOT>/sources`; the checkout's `sources/`
    (a clone, a worktree, `pip install -e .`). None found = `SourcesNotFound`, which names all
    three. Only code that reads a pack calls this; a family that reads none never needs it.
    """
    explicit = os.environ.get("AW_SOURCES_DIR")
    if explicit:
        d = pathlib.Path(explicit)
        if not d.is_dir():
            raise SourcesNotFound(f"CYTIR_SOURCES_DIR={explicit!r} is not a directory")
        return d
    root = os.environ.get("AW_REPO_ROOT")
    if root and (pathlib.Path(root) / "sources").is_dir():
        return pathlib.Path(root) / "sources"
    if _is_checkout() and (REPO_ROOT / "sources").is_dir():
        return REPO_ROOT / "sources"
    raise SourcesNotFound(
        "no pack declarations found. `sources/` is not part of the installed package. Clone the "
        "public repository (it ships `sources/`) and set CYTIR_SOURCES_DIR=<clone>/sources, or run "
        "from the clone. Looked at: $CYTIR_SOURCES_DIR (unset), $CYTIR_REPO_ROOT/sources, and the "
        "checkout's own sources/ (this install is not a checkout).")


if __name__ == "__main__":
    raise SystemExit(main())
