"""cytir.core.envnames -- the `CYTIR_*` <-> `AW_*` environment-name mirror (task 2964, RULING 85).

Two prefixes name the same knob: `CYTIR_X` and `AW_X`. `normalise(env)` makes them agree:

* only one of the two is set: the other gets the same value (both directions);
* both are set and equal: nothing to do;
* both are set with DIFFERENT values and both are user-set: `EnvNameConflict`, naming both
  variables and the length of both values (never the values: they may be secrets).

Which name wins? For two user-set names: neither. They are equal, or the process refuses. For a
twin the mirror made itself and a user-set name: the user-set name wins (see STALE TWINS).
Calling it twice changes nothing (idempotent). An empty string counts as "set" here; the settings
loaders that treat empty as unset do so after the mirror.

STALE TWINS. The mirror records the twins it creates IN THE ENVIRONMENT, so every child process
inherits the record: one marker variable, `_CYTIR_ENV_MIRRORED` (no knob prefix, so the mirror
never mirrors it), holding `NAME:HASH,NAME:HASH` where HASH = first 12 hex of sha256 of the twin's
value. No value is ever stored or printed (only a hash). A differing pair resolves like this:
exactly one side is a recorded twin whose CURRENT value still matches its hash -> it is stale,
the other side wins and the twin and its record are rewritten; neither side matches a record
(both user-set) -> `EnvNameConflict`. A recorded twin that still matches while its source name is
gone (a `delenv("AW_X")`) is dropped with its record. A twin the user changed is user-set: its
record is dropped. The same rule runs on every mapping, using the marker found IN THAT mapping.
Because a parent's `AW_X=1` leaves `CYTIR_X=1` plus the record in its environment, a child started
with `AW_X=2` (or `CYTIR_X=2`) finds the record and follows the new value instead of refusing.
Prefix filters strip the marker together with the knobs (`is_knob` is true for it).

KNOWN LIMIT. The mirror runs (a) once on `os.environ` when the package is imported and (b) on a
copy of every INJECTED env mapping that the settings loaders take. A `CYTIR_X` that a program puts
into `os.environ` AFTER the import is seen by the settings loaders (they mirror on every call) but
NOT by code that reads `os.environ["AW_X"]` directly; call `normalise(os.environ)` after setting
it. A `monkeypatch.setenv("AW_X", ...)` after import keeps working: the code reads `AW_X`.
Code that REMOVES one knob from an env mapping must remove both spellings: `pop_knob`.

Stdlib only. Imports nothing from the package (the package `__init__` imports this module first).
"""
from __future__ import annotations

import hashlib
from typing import MutableMapping

NEW_PREFIX = "CYTIR_"
OLD_PREFIX = "AW_"
PREFIXES = (NEW_PREFIX, OLD_PREFIX)
MARKER = "_CYTIR_ENV_MIRRORED"


class EnvNameConflict(ValueError):
    """`CYTIR_X` and `AW_X` are both set, to different values."""


def is_knob(name: str) -> bool:
    """True for a configuration name of either prefix and for the mirror's marker: the one filter
    every prefix site uses, so an arm or child never inherits a record that no longer matches."""
    return isinstance(name, str) and (name.startswith(PREFIXES) or name == MARKER)


def twin(name: str) -> str | None:
    """The other spelling of a knob name; None when `name` has neither prefix."""
    if name.startswith(NEW_PREFIX):
        return OLD_PREFIX + name[len(NEW_PREFIX):]
    if name.startswith(OLD_PREFIX):
        return NEW_PREFIX + name[len(OLD_PREFIX):]
    return None


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "surrogateescape")).hexdigest()[:12]


def _records(env) -> dict[str, str]:
    """name -> hash from the marker of `env`, kept only while the named variable still holds
    the hashed value (otherwise somebody else set it: it is user-set now)."""
    out = {}
    for part in (env.get(MARKER) or "").split(","):
        name, _, h = part.partition(":")
        if name and h and name in env and env[name] is not None and _digest(env[name]) == h:
            out[name] = h
    return out


def normalise(env: MutableMapping[str, str]) -> list[str]:
    """Mirror every knob of `env` onto its twin, IN PLACE. Returns the names it added.
    Raises `EnvNameConflict` (before changing anything) when two user-set names disagree."""
    rec = _records(env)
    drop: list[str] = []          # stale twins whose source is gone
    setv: dict[str, str] = {}     # stale twins to re-mirror
    for name in rec:
        if twin(name) not in env:
            drop.append(name)
    for k in sorted(env):
        if k in drop or not k.startswith(NEW_PREFIX):
            continue
        t = twin(k)
        if t not in env or t in drop or env[t] == env[k]:
            continue
        kr, tr = k in rec, t in rec
        if kr != tr:
            stale, other = (k, t) if kr else (t, k)
            setv[stale] = env[other]
        else:
            raise EnvNameConflict(
                f"{k} and {t} are both set to different values "
                f"(lengths {len(env[k])} and {len(env[t])}); set only {k}")
    adds = {}
    for k in env:
        t = twin(k) if (k.startswith(PREFIXES) and k not in drop) else None
        if t is not None and t not in env:
            adds[t] = env[k]
    for n in drop:
        del env[n]
        del rec[n]
    for n, v in {**setv, **adds}.items():
        env[n] = v
        rec[n] = _digest(v)
    if rec:
        env[MARKER] = ",".join(f"{n}:{h}" for n, h in sorted(rec.items()))
    else:
        env.pop(MARKER, None)
    return list(adds)


def user_set_names(env) -> list[str]:
    """The knob names of `env` that a person set, once each: no mirror-made twin, and for a pair
    that is user-set on both sides only the `CYTIR_` spelling."""
    rec = _records(env)
    names = [k for k in env if k.startswith(PREFIXES) and k not in rec]
    return sorted(k for k in names if not (k.startswith(OLD_PREFIX) and twin(k) in names))


def shown(name: str) -> str:
    """The spelling to print to a person: `CYTIR_X` for `AW_X` (and for `CYTIR_X` itself)."""
    return NEW_PREFIX + name[len(OLD_PREFIX):] if name.startswith(OLD_PREFIX) else name


def mirrored(env) -> dict:
    """A mirrored COPY of any mapping; the caller's mapping (and `os.environ`) is never changed."""
    out = dict(env)
    normalise(out)
    return out


def pop_knob(env: MutableMapping[str, str], name: str, default=None):
    """Remove a knob from `env` under BOTH spellings (and return the value of `name`, else of its
    twin, else `default`). Removing only `AW_X` would let the next `import cytir` mirror `CYTIR_X`
    back into place: the guard the removal serves would be bypassed."""
    t = twin(name)
    got = env.pop(name, None)
    other = env.pop(t, None) if t else None
    rec = _records(env)
    for n in (name, t):
        rec.pop(n, None)
    if rec:
        env[MARKER] = ",".join(f"{n}:{h}" for n, h in sorted(rec.items()))
    else:
        env.pop(MARKER, None)
    return got if got is not None else (other if other is not None else default)
