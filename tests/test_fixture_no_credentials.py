"""No shipped fixture or data line carries someone else's credential-like value.

Public-release guard (task 2990): URLs taken from a public crawl once carried a
basic-auth password, a JWT and opaque session tokens. Fixtures keep the SHAPE
(same length, same percent-encoding, same position) with an obviously fake value
made of `0`, `x` and `X` only. A line that must keep a fake credential-shaped
value (a scrubber's own input) carries the marker `gitleaks:allow`, as in
`test_aw_common_credential_url_scrub.py`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCAN_DIRS = ("tests/fixtures", "tests/data", "data", "cytir")
SUFFIXES = {".jsonl", ".json", ".txt", ".csv", ".tsv", ".yaml", ".yml", ".ndjson"}
ALLOW = "gitleaks:allow"

USERINFO = re.compile(r'://(?!mailto:)([^/@:\s"\\]+):([^/@\s"\\]+)@([^/?#\s"\\]*)')
JWT = re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}")
QTOKEN = re.compile(
    r"[?&;](?:token|access_token|api_key|apikey|key|sig|signature|auth|password|secret|"
    r"session|sid|jwt)=([A-Za-z0-9_%.\-~+/]{16,})",
    re.I,
)
EMAIL = re.compile(r"(?<![A-Za-z0-9._+%:-])([A-Za-z0-9._+-]+?)(?:@|%40)([A-Za-z0-9.-]+\.[a-z]{2,})", re.I)
ROLE_LOCAL = {"info", "media", "datenschutz", "research", "studentinfo", "wiso", "mppp", "sanita",
              "deutschland", "happiness", "contact", "mail", "office", "webmaster", "admin",
              "support", "kontakt", "service", "mailto", "a", "u", "user", "pw", "p", "ss", "name", "bland"}
PLACEHOLDER_HOST = re.compile(r"(^|\.)(example\.(com|org|net)|a\.com|localhost|test|invalid)(:\d+)?$", re.I)


def synthetic(value: str) -> bool:
    """True when the value carries no information: only 0, x, X and separators."""
    core = re.sub(r"%[0-9A-Fa-f]{2}", "", value)
    return set(re.sub(r"[^A-Za-z0-9]", "", core)) <= set("0xX")


def offences(line: str, emails: bool = False) -> list[str]:
    if ALLOW in line:
        return []
    found = []
    if emails:
        for m in EMAIL.finditer(line):
            loc = re.sub(r"^[0-9A-F]{2}(?=[xX0])", "", m.group(1))  # a leftover %XX lead-in
            if not (PLACEHOLDER_HOST.search(m.group(2)) or synthetic(loc) or loc.lower() in ROLE_LOCAL):
                found.append("person-looking-email")
    for m in USERINFO.finditer(line):
        if not (PLACEHOLDER_HOST.search(m.group(3)) or synthetic(m.group(2)) or len(m.group(2)) <= 2):
            found.append("userinfo-with-password")
    if any(not synthetic(m.group()) for m in JWT.finditer(line)):
        found.append("jwt")
    for m in QTOKEN.finditer(line):
        v = m.group(1)
        if not (synthetic(v) or v.startswith("0123456789")):
            found.append("opaque-query-token")
    return found


def _files():
    """The data files the public cut ships (its allowlist), under the scanned dirs."""
    from cytir.ops import public_cut as pc

    shipped, _ = pc.select(pc.load_allowlist(ROOT), pc.tracked_files(ROOT))
    for rel in sorted(shipped):
        p = ROOT / rel
        if rel.startswith(tuple(d + "/" for d in SCAN_DIRS)) and p.suffix in SUFFIXES \
                and p.is_file() and p.stat().st_size < 64_000_000:
            yield p


def test_no_shipped_data_line_carries_a_credential():
    bad = {}
    for p in _files():
        with p.open(encoding="utf-8", errors="replace") as fh:
            for n, line in enumerate(fh, 1):
                for kind in offences(line, emails=p.parent.name == "fixtures"):
                    bad.setdefault(str(p.relative_to(ROOT)), []).append((n, kind))
    assert not bad, f"credential-like values (line, kind) in: { {k: v[:5] for k, v in bad.items()} }"


@pytest.mark.parametrize(
    "line, expect",
    [
        ('{"u":"https://h.ch/a?sid=' + "a1b2c3d4" * 4 + '"}', ["opaque-query-token"]),
        ('{"u":"https://h.ch/a?sid=' + "0" * 32 + '"}', []),
        ('{"u":"https://alice:hunter22x@h.ch/"}', ["userinfo-with-password"]),
        ('{"u":"https://alice:XXXXXXXX@h.ch/"}', []),
        ('{"u":"https://u:pw12345@example.com/"}', []),
        ('{"u":"https://h.ch/?t=eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ4In0.sig123"}', ["jwt"]),
        ('{"u":"https://h.ch/?t=eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ4In0.sig123"} gitleaks:allow', []),
    ],
)
def test_detector_behaves(line, expect):
    assert offences(line, emails=True) == expect


def test_email_detector():
    assert offences('{"u":"https://h.ch/a?m=jane.doe%40firma.ch"}', emails=True) == ["person-looking-email"]
    assert offences('{"u":"https://h.ch/a?m=xxxx.xxx%40firma.ch"}', emails=True) == []
    assert offences('{"u":"https://h.ch/a?m=info%40firma.ch"}', emails=True) == []
