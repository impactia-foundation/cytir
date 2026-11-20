#!/usr/bin/env python3
"""The news stage release, derived from the checkout's own commit (task 683 S7, task 671 follow-up).

WHY. The daily news run fetches and stages from an immutable release directory (a `git archive`
of one commit, closure-checked by `cytir.news.release_preflight`). The pin of that release lived as
five literals in `python -m cytir.news.daily` and had to be re-typed after every
merge that mattered (r18 to r22 in nine days). A forgotten re-pin runs the run on code that is
silently days old, and a half-edited pin refuses at 06:10Z. This module removes the typing:

    release dir      <releases-root>/news-<sha8>            the archive of the checkout's HEAD
    manifest         <work>/release-manifest-<sha8>.json    the file closure (written at cut)
    manifest digest  <work>/release-manifest-<sha8>.sha256  the digest agreed at cut

`sha` is `git rev-parse HEAD` of the checkout the unit runs from (the unit's wrapper already
refuses a dirty checkout or one not on origin/main). There is nothing to edit after a merge: the
operator runs `cut`, and until it has been run for HEAD the run REFUSES, loudly, before any
network (it never falls back to an older release).

WHAT IS KEPT from the hand recipe (task 671's reason for not deriving R from the live tree):
the release is still a SEPARATE immutable copy, and its closure manifest is still hashed at cut
time and checked at run time against that recorded digest. For the two-machine copy check, run
`cut` on the second machine with `--expect-manifest-sha256 <digest printed by the first>`; the
cut refuses on a mismatch and writes the digest file only after it agrees.

    python3 -m cytir.news.release resolve   --checkout . --releases-root R --work W
    python3 -m cytir.news.release cut       --checkout . --releases-root R --work W [--expect-manifest-sha256 H]
    python3 -m cytir.news.release ensure    --checkout . [--releases-root R --work W] [--keep 4]
    python3 -m cytir.news.release prune     --releases-root R --work W [--keep 4] [--protect SHA8 ...]

`ensure` (task 769) is `resolve`, except that a HEAD with NO release is cut first (under a lock
beside the release root, taken only to cut, so the daily and the refresh wrapper never cut at once), then pruned. It is what the
derive-mode entry points call, because the 11:30Z refresh fast-forwards the checkout every day and
a release cut by hand would be stale by the next 06:10Z. An existing release is reused, never
re-cut; a half-present release (dir without digest, or the reverse) still refuses.

Stdlib only.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import time
import posixpath
import re
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

from cytir.news import release_preflight as rp

SHA40 = re.compile(r"[0-9a-f]{40}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
#: The checkout's own application subtree inside the monorepo (the release root is this subtree).
APP_SUBDIR = "authoritative-web-app"


def _link_filter(skipped: list[dict[str, str]]):
    """Extraction filter: the `data` filter, except that a link member that is not safe is
    SKIPPED and recorded instead of failing the cut. Unsafe = an absolute target, or a target
    that leaves the release tree (judged here from the link's own directory: the stdlib `data`
    filter resolves a relative target from the wrong place on some Python versions, which made
    the cut's result depend on the interpreter).

    Why this is safe: the closure (`release_preflight.release_files`) counts regular files only
    and ignores every symlink, so a link member can neither be missing from the closure nor
    change it. A relative link that stays inside the tree is kept as a link, as archived."""
    def keep(member: tarfile.TarInfo, dest: str):
        if member.issym():
            target = member.linkname
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(member.name), target))
            if posixpath.isabs(target) or resolved == ".." or resolved.startswith("../"):
                skipped.append({"path": member.name, "target": target})
                return None
            return tarfile.tar_filter(member, dest)
        if member.islnk():
            try:
                return tarfile.data_filter(member, dest)
            except tarfile.FilterError:
                skipped.append({"path": member.name, "target": member.linkname})
                return None
        return tarfile.data_filter(member, dest)
    return keep


class ReleaseError(RuntimeError):
    """No usable release for this commit. The message names the command that fixes it."""


@dataclass(frozen=True)
class ReleasePin:
    sha: str
    release_dir: Path
    manifest: Path
    manifest_sha256: str

    def env(self) -> dict[str, str]:
        """The `AW_*` names the stage scripts and the preflight read."""
        return {
            "AW_RELEASE_DIR": str(self.release_dir),
            "AW_REPO_DIR": str(self.release_dir),
            "AW_NEWS_RELEASE_MANIFEST": str(self.manifest),
            "AW_NEWS_RELEASE_MANIFEST_SHA256": self.manifest_sha256,
            "AW_NEWS_RELEASE_SHA": self.sha,
        }


def checkout_sha(checkout: Path) -> str:
    done = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                          capture_output=True, text=True, timeout=60)
    sha = done.stdout.strip()
    if done.returncode != 0 or not SHA40.fullmatch(sha):
        raise ReleaseError(f"cannot read HEAD of {checkout}: {done.stderr.strip()[:200]}")
    return sha


def paths_for(sha: str, releases_root: Path, work: Path) -> tuple[Path, Path, Path]:
    if not SHA40.fullmatch(sha):
        raise ReleaseError(f"not a 40-hex git sha: {sha!r}")
    short = sha[:8]
    return (Path(releases_root) / f"news-{short}",
            Path(work) / f"release-manifest-{short}.json",
            Path(work) / f"release-manifest-{short}.sha256")


def derive_pin(checkout: Path, releases_root: Path, work: Path, *, sha: str | None = None) -> ReleasePin:
    """The pin for the checkout's HEAD, validated against its recorded closure digest.
    Raises `ReleaseError` (never a fallback to another release) when HEAD has no cut release."""
    sha = sha or checkout_sha(Path(checkout))
    release_dir, manifest, digest_file = paths_for(sha, releases_root, work)
    cut_cmd = (f"python3 -m cytir.news.release cut --checkout {checkout} "
               f"--releases-root {releases_root} --work {work}")
    missing = [str(p) for p in (release_dir, manifest, digest_file)
               if not (p.is_dir() if p == release_dir else p.is_file())]
    if missing:
        raise ReleaseError(f"no news release has been cut for HEAD {sha[:8]} (absent: {missing}). "
                           f"Cut it, then re-run: {cut_cmd}")
    recorded = digest_file.read_text().strip()
    if not SHA256.fullmatch(recorded):
        raise ReleaseError(f"{digest_file} does not hold a sha256")
    try:
        rp.load_and_validate_release(release_dir, manifest, sha, recorded)
    except rp.PreflightError as exc:
        raise ReleaseError(f"release for {sha[:8]} is not intact: {exc}") from exc
    return ReleasePin(sha, release_dir, manifest, recorded)


def cut(checkout: Path, releases_root: Path, work: Path, *, sha: str | None = None,
        expect_manifest_sha256: str | None = None) -> ReleasePin:
    """Archive the app subtree of `sha`, seal its closure, record the digest. Idempotent for an
    identical result; refuses to overwrite a release that exists but differs."""
    checkout = Path(checkout)
    sha = sha or checkout_sha(checkout)
    release_dir, manifest, digest_file = paths_for(sha, releases_root, work)
    if release_dir.exists():
        raise ReleaseError(f"{release_dir} already exists: a release is immutable. "
                           "Remove it by hand if it is known bad, then cut again.")
    release_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = release_dir.with_name(release_dir.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    try:
        arch = subprocess.run(["git", "-C", str(checkout), "archive", "--format=tar",
                               f"{sha}:{APP_SUBDIR}"], capture_output=True, timeout=300)
        if arch.returncode != 0:
            raise ReleaseError(f"git archive {sha[:8]}:{APP_SUBDIR} failed: "
                               f"{arch.stderr.decode(errors='replace')[:200]}")
        import io
        skipped: list[dict[str, str]] = []
        with tarfile.open(fileobj=io.BytesIO(arch.stdout)) as tar:
            tar.extractall(tmp, filter=_link_filter(skipped))
        (tmp / "GIT_SHA").write_text(sha + "\n")
        value = rp.write_manifest(tmp, sha, manifest,
                                  skipped_links=sorted(skipped, key=lambda d: d["path"]))
        digest = rp.digest(manifest)
        if expect_manifest_sha256 and digest != expect_manifest_sha256:
            manifest.unlink(missing_ok=True)
            raise ReleaseError(f"manifest digest {digest} differs from the expected "
                               f"{expect_manifest_sha256}: the two copies are not byte-identical")
        tmp.rename(release_dir)
        digest_file.write_text(digest + "\n")
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    print(json.dumps({"sha": sha, "files": len(value["files"]), "manifest_sha256": digest,
                      "skipped_links": len(value.get("skipped_links", [])),
                      "release_dir": str(release_dir)}, sort_keys=True))
    return derive_pin(checkout, releases_root, work, sha=sha)


RELEASE_DIR_RE = re.compile(r"news-([0-9a-f]{8})\Z")
#: A release younger than this is never pruned, whatever `--keep` says.
PRUNE_MIN_AGE_DAYS = 7
DEFAULT_KEEP = 4


def _toplevel(checkout: Path) -> Path:
    """The git ROOT. `git archive <sha>:<subdir>` run from inside the app subdir returns almost
    nothing (task 748), so every cut goes through the root, whatever the caller passed."""
    done = subprocess.run(["git", "-C", str(checkout), "rev-parse", "--show-toplevel"],
                          capture_output=True, text=True, timeout=60)
    if done.returncode != 0 or not done.stdout.strip():
        raise ReleaseError(f"{checkout} is not inside a git checkout: {done.stderr.strip()[:200]}")
    return Path(done.stdout.strip())


class _Lock:
    """Exclusive cut lock, a file BESIDE the release root the caller passed (the dir the cut
    writes into). Taken only to cut, never to resolve. Any OS failure is a `ReleaseError`."""
    def __init__(self, releases_root: Path):
        self.path = Path(releases_root) / ".news-release-cut.lock"

    def __enter__(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.fh = open(self.path, "a")
            fcntl.flock(self.fh, fcntl.LOCK_EX)
        except OSError as exc:
            raise ReleaseError(f"cannot take the cut lock {self.path}: {exc}") from exc
        return self

    def __exit__(self, *exc):
        self.fh.close()


def prune(releases_root: Path, work: Path, *, keep: int = DEFAULT_KEEP, protect: tuple[str, ...] = (),
          min_age_days: float = PRUNE_MIN_AGE_DAYS, now: float | None = None) -> list[str]:
    """Delete old `news-<sha8>` releases (1.1 GB each) plus their manifest and digest.
    Kept: the `keep` newest, every sha8 in `protect` (the pinned/current one), and anything
    younger than `min_age_days`. Only dirs that this tool cut (a `GIT_SHA` file and a digest file
    in `work`) are candidates; every other directory under the root is left alone. Age is the
    mtime of `GIT_SHA`, written at cut. Returns the deleted dir names."""
    now = time.time() if now is None else now
    root = Path(releases_root)
    found: list[tuple[float, str, Path]] = []
    if root.is_dir():
        for d in root.iterdir():
            m = RELEASE_DIR_RE.fullmatch(d.name)
            marker = d / "GIT_SHA"
            if m and d.is_dir() and marker.is_file() and \
                    (Path(work) / f"release-manifest-{m.group(1)}.sha256").is_file():
                found.append((marker.stat().st_mtime, m.group(1), d))
    found.sort(reverse=True)
    protected = {p[:8] for p in protect}
    deleted = []
    for i, (mtime, short, d) in enumerate(found):
        if i < keep or short in protected or now - mtime < min_age_days * 86400:
            continue
        shutil.rmtree(d)
        (Path(work) / f"release-manifest-{short}.json").unlink(missing_ok=True)
        (Path(work) / f"release-manifest-{short}.sha256").unlink(missing_ok=True)
        deleted.append(d.name)
    return deleted


def ensure_pin(checkout: Path, releases_root: Path, work: Path, *, keep: int = DEFAULT_KEEP,
               log=print) -> ReleasePin:
    """`derive_pin` for the checkout's HEAD, cutting the release first when HEAD has none."""
    root = _toplevel(Path(checkout))
    sha = checkout_sha(root)
    release_dir, _, _ = paths_for(sha, releases_root, work)
    if release_dir.exists():                         # reuse, lock-free: derive_pin validates it
        return derive_pin(root, releases_root, work, sha=sha)
    try:
        with _Lock(releases_root):
            if release_dir.exists():                 # another process cut it while we waited
                return derive_pin(root, releases_root, work, sha=sha)
            with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet):
                pin = cut(root, releases_root, work, sha=sha)
            log(f"news release: cut {sha[:8]} for HEAD (auto)")
            gone = prune(releases_root, work, keep=keep, protect=(sha, *_env_protect()))
            if gone:
                log(f"news release: pruned {', '.join(gone)}")
            return pin
    except (ReleaseError, OSError) as exc:
        try:                                         # keep the plain refusal text in front
            derive_pin(root, releases_root, work, sha=sha)
        except ReleaseError as plain:
            raise ReleaseError(f"{plain}; auto-cut failed: {exc}") from exc
        raise


def _env_protect() -> tuple[str, ...]:
    """Extra sha8s that must survive a prune (e.g. a release another chain still runs from)."""
    return tuple(x for x in os.environ.get("AW_NEWS_RELEASE_PROTECT", "").replace(",", " ").split() if x)


def _ensure_cli(args, lazy_setting_path) -> int:
    """`ensure`: exactly one outcome line on STDERR (created / reused / FAILED), rc 0 or 2 (task 1475).
    The unit discards stdout (the env block), so the journal only ever sees this line."""
    try:
        root = Path(args.releases_root or lazy_setting_path("releases_root"))
        work = Path(args.work or lazy_setting_path("news_work"))
        sha = checkout_sha(_toplevel(Path(args.checkout)))
        existed = paths_for(sha, root, work)[0].exists()
        pin = ensure_pin(args.checkout, root, work, keep=args.keep,
                         log=lambda m: print(m, file=sys.stderr))
    except (ReleaseError, OSError, subprocess.SubprocessError) as exc:
        print(f"news release ensure: FAILED: {exc}", file=sys.stderr)
        return 2
    print(f"news release ensure: {'reused' if existed else 'created'} {pin.release_dir.name}",
          file=sys.stderr)
    print(json.dumps(pin.env(), indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("resolve", "cut"):
        p = sub.add_parser(name)
        p.add_argument("--checkout", type=Path, required=True)
        p.add_argument("--releases-root", type=Path, required=True)
        p.add_argument("--work", type=Path, required=True)
        p.add_argument("--sha", help="default: HEAD of the checkout")
    sub.choices["cut"].add_argument("--expect-manifest-sha256")
    pe = sub.add_parser("ensure")
    pe.add_argument("--checkout", type=Path, required=True)
    pe.add_argument("--releases-root", type=Path)   # default: operator settings (as cytir.news.daily)
    pe.add_argument("--work", type=Path)
    pe.add_argument("--keep", type=int, default=DEFAULT_KEEP)
    pp = sub.add_parser("prune")
    pp.add_argument("--releases-root", type=Path, required=True)
    pp.add_argument("--work", type=Path, required=True)
    pp.add_argument("--keep", type=int, default=DEFAULT_KEEP)
    pp.add_argument("--protect", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        if args.command == "prune":
            gone = prune(args.releases_root, args.work, keep=args.keep,
                         protect=tuple(args.protect) + _env_protect())
            print(json.dumps({"pruned": gone}, sort_keys=True))
        elif args.command == "ensure":
            from cytir.core.paths import lazy_setting_path
            return _ensure_cli(args, lazy_setting_path)
        elif args.command == "cut":
            pin = cut(args.checkout, args.releases_root, args.work, sha=args.sha,
                      expect_manifest_sha256=args.expect_manifest_sha256)
        else:
            pin = derive_pin(args.checkout, args.releases_root, args.work, sha=args.sha)
            print(json.dumps(pin.env(), indent=2, sort_keys=True))
    except ReleaseError as exc:
        print(f"FATAL news release: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
