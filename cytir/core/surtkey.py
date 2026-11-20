"""cytir.core.surtkey -- the Common Crawl `url_surtkey` for a URL or a host.

The CC columnar index keys every capture by `url_surtkey`: the host labels
reversed and comma-joined, then `)`, then the path and the sorted query, all
lower-cased -- `https://www.Example.com/A/b/?z=1&a=2` -> `com,example)/a/b?a=2&z=1`.
`surt_key(url)` computes that key so a lookup finds the capture. A key that
differs by one character makes the lookup silently miss ("0 pages"), so the
contract here is byte parity with the key CC stores. `tests/test_surtkey.py`
and the sealed fixture `tests/fixtures/surtkey_parity.v1.jsonl` hold the proof.

Clean-room provenance (this replaces the AGPL-3.0 `surt` package, ADR-121).
The code was written from public descriptions of the transform and from
black-box observation of input/output pairs. No source of the `surt` package
(or of any other SURT implementation) was opened, copied or translated. Read on
2026-10-03:

* Internet Archive, Heritrix user manual, glossary entry "SURT"
  (http://crawler.archive.org/articles/user_manual/glossary.html): "SURT stands
  for Sort-friendly URI Reordering Transform ... A URI <scheme://domain.tld/path?query>
  has SURT form <scheme://(tld,domain,)/path?query>. Conversion to SURT form
  also involves making all characters lowercase, and changing the 'https'
  scheme to 'http'."
* Internet Archive, Wayback CDX server documentation, the `urlkey` column
  (https://raw.githubusercontent.com/internetarchive/wayback/master/wayback-cdx-server/README.md):
  the capture of `http://www.archive.org:80/` has urlkey `org,archive)/` -- no
  scheme, no leading `www`, no default port, a `)` after the host.
* Common Crawl, "Index to WARC Files and URLs in Columnar Format"
  (https://commoncrawl.org/blog/index-to-warc-files-and-urls-in-columnar-format):
  the column list, `url_surtkey STRING`.
* The Python standard library `idna` codec (RFC 3490, IDNA2003) for non-ASCII hosts.

The rule table below was MEASURED, not read: against Common Crawl's own
`url_surtkey` values for real captures (the ground truth) and by black-box runs
of the `surt` 0.3.1 package, used only as an oracle by tests that skip when it
is not installed. Where the two ever disagree, Common Crawl's key wins.

Rule table (the order is the order of application):

 1. Strip leading/trailing ASCII whitespace; delete TAB, CR and LF anywhere.
 2. A fragment (`#...`) is dropped. The query starts at the first `?`.
 3. The scheme is removed. After `scheme://` (any scheme) the authority starts;
    after `http(s)://` a doubled lower-case `http(s)://` goes too (scraped links
    such as `https://https://host/x`). `http`/`https` also accept 0, 1 or 3+
    slashes (`http:x`, `http:/x`); then the text up to the next `/` or `?` is
    the host as it is, with no user-info and no port read. Another `scheme:`
    without `//` (`mailto:`, `about:`, `host:port`) is not a URL: it comes back
    with the text after the scheme lower-cased.
 4. The authority ends at the first `/` or `?`. User-info (up to the last `@`)
    is dropped. The port is the text after the first `:` (after the `]` for an
    `[ipv6]` host); it must be ASCII digits or empty, else ValueError.
 5. The default port is dropped (80 for http, 443 for https), as are an empty
    port and port 0. Any other port stays: `com,example:8080)/`. A port above
    65535 raises ValueError.
 6. An empty host promotes the first path segment to the host (`http:///x`). No
    host at all (`http://`, a host that decodes to nothing) is not a key: the
    scheme, a colon and the decoded path come back, as the package did.
 7. Host: ASCII-lower-case, percent-decode to a fixpoint, IDNA2003 when it has
    non-ASCII text (a host IDNA rejects stays as it is, escaped); `..` becomes
    `.` once; leading/trailing dots go; an all-numeric host is an IPv4 address
    (a single number is decimal, several parts may be octal when they start with
    `0`) written as dotted decimal; one leading `www` or `wwwNNN` label goes when
    a dot follows it.
 8. The labels are reversed and joined with `,`; an `[ipv6]` host stays one
    label unless it has dots.
 9. Path: ASCII-lower-case, percent-decode to a fixpoint, remove `.` segments and
    `x/..` pairs (a `..` with nothing to cancel stays), collapse runs of `/`,
    remove the LAST ASP.NET cookieless-session segment `/(s(<24 alnum>))` when a
    `.aspx` with a character before it follows, drop one trailing `/` unless the
    path is `/`; an empty path is `/`.
10. Query: ASCII-lower-case, percent-decode to a fixpoint, remove the LAST match
    of each session-id parameter in this order (`jsessionid`, `phpsessid`, `sid`
    with 32 alphanumerics; `aspsessionidXXXXXXXX` with 24 letters;
    `cfid`+`cftoken`) together with its trailing `&`, escape, split on `&`, sort
    by (name, value) with a bare `name` before `name=`; empty pieces stay. An
    empty query leaves no `?`.
11. Escaping: bytes <= 0x20, 0x7f, `%`, `#` and bytes >= 0x80 become lower-case
    `%xx`; everything else stays literal. Upper-case non-ASCII text is NOT
    lower-cased (`%c3%84`), only ASCII is.
12. Key = labels + (`:port`) + `)` + path + (`?` + query).

Known divergences from the package, all on input that is not a URL a crawler
stores: an empty `[]` host with a port, an unclosed `[`, a `scheme:` without
`//` that is not http(s), hosts that percent-decode to invalid UTF-8. They are
listed in the PR that introduced this module.

Pure, no network, no state.
"""
from __future__ import annotations

import re
from typing import Optional

__all__ = ["surt_key", "surt_key_or_none", "assert_key_parity", "PARITY_PROBES"]

_SCHEME = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*):")
_WWW = re.compile(r"^www\d*\.")
_ALL_NUMERIC = re.compile(r"^[0-9]+(?:\.[0-9]+){0,3}$")
_SLASHES = re.compile(r"/{2,}")
_DOT_SEG = re.compile(r"/\.(?=/|$)")
_DOTDOT = re.compile(r"/[^/]*/\.\.(?=/|$)")
# `/(S(<24 alphanumerics>))/` -- the ASP.NET cookieless session segment (also
# `(A(..)F(..))` and other one-letter groups) -- is dropped, but only when a
# `.aspx` with at least one character before it follows.
_ASPNET_SESSION = re.compile(r"/\([a-z]\([0-9a-z]{24}\)(?:[a-z]\([0-9a-z]{24}\))*\)(?=/[^?]+\.aspx)")
_PCT = re.compile(rb"%([0-9a-fA-F]{2})")

# Session-id parameters (rule 10), in this order. A trailing `&` goes with the
# parameter, so removing one never leaves a doubled separator.
_SESSION_RES = tuple(re.compile(p) for p in (
    rb"jsessionid=[0-9a-z]{32}(?:&|$)",
    rb"phpsessid=[0-9a-z]{32}(?:&|$)",
    rb"sid=[0-9a-z]{32}(?:&|$)",
    rb"aspsessionid[a-z]{8}=[a-z]{24}(?:&|$)",
    rb"cfid=[^&]*&cftoken=[^&]*(?:&|$)",
))

_ESCAPE = {i: "%%%02x" % i for i in range(256)
           if i <= 0x20 or i in (0x7F, 0x25, 0x23) or i >= 0x80}


def _unescape_once(b: bytes) -> bytes:
    return _PCT.sub(lambda m: bytes([int(m.group(1), 16)]), b)


def _fixpoint(b: bytes) -> bytes:
    while True:
        nb = _unescape_once(b)
        if nb == b:
            return b
        b = nb


def _escape(b: bytes) -> str:
    return "".join(_ESCAPE.get(c) or chr(c) for c in b)


def _decode_escape(s: str) -> str:
    return _escape(_fixpoint(s.encode("utf-8").lower()).lower())


def _ipv4(host: str) -> str:
    """Rule 7: an all-numeric host as dotted decimal; otherwise unchanged."""
    if not _ALL_NUMERIC.match(host):
        return host
    parts = host.split(".")
    n = len(parts)
    # One part is always decimal. With more parts a leading `0` makes that part
    # octal; when the FIRST part has one, a 8 or 9 anywhere voids the host.
    if n > 1 and parts[0][0] == "0" and re.search("[89]", host):
        return host
    nums = []
    for part in parts:
        try:
            if n > 1 and len(part) > 1 and part[0] == "0":
                nums.append(int(part, 8))
            else:
                nums.append(int(part))
        except ValueError:
            return host
    if n == 1:
        v = nums[0] & 0xFFFFFFFF
    else:
        if any(x > 255 for x in nums[:-1]) or nums[-1] >= 256 ** (5 - n):
            return host
        v = 0
        for x in nums[:-1]:
            v = (v << 8) | x
        v = (v << (8 * (5 - n))) | nums[-1]
    return ".".join(str((v >> s) & 255) for s in (24, 16, 8, 0))


def _host_labels(host: str) -> list[str]:
    hb = _fixpoint(host.encode("utf-8").lower()).lower()
    h = hb.decode("ascii") if hb.isascii() else hb.decode("utf-8", "replace")
    if not h.isascii():
        try:
            h = h.encode("idna").decode("ascii")
        except UnicodeError:
            pass  # the whole host stays as it is and is escaped below
    h = _ipv4(h.replace("..", ".").strip("."))
    h = _WWW.sub("", h, count=1)
    if not h:
        return []
    return [_escape(p.encode("utf-8")) for p in h.split(".")]


def _clean_path(path: str) -> str:
    while True:
        new = _DOT_SEG.sub("", path, count=1)
        if new == path:
            new = _DOTDOT.sub("", path, count=1)
            if new == path:
                break
        path = new
    path = _SLASHES.sub("/", path)
    found = list(_ASPNET_SESSION.finditer(path))
    if found:  # ASP.NET cookieless session: one removal, the LAST match
        path = path[:found[-1].start()] + path[found[-1].end():]
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    return path or "/"


def _clean_query(query: str) -> str:
    b = _fixpoint(query.encode("utf-8").lower())
    for rx in _SESSION_RES:
        found = list(rx.finditer(b))
        if found:  # one removal per pattern, and it is the LAST match
            b = b[:found[-1].start()] + b[found[-1].end():]
    b = b.lower()
    if not b:
        return ""

    def sort_key(piece: str):
        name, eq, value = piece.partition("=")
        return (name, 1 if eq else 0, value)

    return "&".join(sorted(_escape(b).split("&"), key=sort_key))


def _hostless(scheme: str, tail: str) -> str:
    """No host at all (`http://`, `http://?x`): the package returned the scheme
    and what was left, not a key; kept the same so a bad input fails the same way."""
    path, _, query = tail.partition("?")
    p = _decode_escape(path) if path else ""
    if len(p) > 1 and p.endswith("/"):
        p = p[:-1]
    q = _clean_query(query) if query else ""
    return f"{scheme or 'http'}:" + (p or ("/" if q else "")) + (("?" + q) if q else "")


def _key(scheme: str, rest: str, bare: bool = False, carried_port: int = 0) -> str:
    end = len(rest)
    for i, c in enumerate(rest):
        if c in "/?":
            end = i
            break
    authority, tail = rest[:end], rest[end:]
    if bare:
        host, port, host_is_v6 = authority, "", False
    else:
        hostport = authority.rpartition("@")[2]
        if hostport.startswith("["):
            close = hostport.find("]")
            if close < 0:  # unclosed `[`: a trailing `:` is an empty port
                host, port = hostport[1:].removesuffix(":"), ""
            else:
                host = hostport[1:close]
                after_br = hostport[close + 1:]
                port = after_br[1:] if after_br.startswith(":") else ""
            host_is_v6 = True
        else:
            host, _, port = hostport.partition(":")
            host_is_v6 = False
    if port and not (port.isascii() and port.isdigit()):
        raise ValueError("Port could not be cast to integer value")
    port_n = int(port) if port else carried_port
    if port_n > 65535:
        raise ValueError("Port out of range 0-65535")
    if not host:
        stripped = tail.lstrip("/")
        if stripped and stripped[0] != "?":
            return _key(scheme, stripped, True, port_n)
        return _hostless(scheme, tail)
    path, _, query = tail.partition("?")
    if host_is_v6:
        labels = [_escape(p) for p in _fixpoint(host.encode("utf-8").lower()).lower().split(b".") if p]
    else:
        labels = _host_labels(host)
        if not labels:
            return _hostless(scheme, tail)
    default = 80 if scheme.lower() in ("http", "") else 443 if scheme.lower() == "https" else -1
    port_part = f":{port_n}" if port_n and port_n != default else ""
    key = ",".join(reversed(labels)) + port_part + ")" + _clean_path(_decode_escape(path))
    q = _clean_query(query) if query else ""
    return key + ("?" + q if q else "")


def surt_key(url_or_host: str) -> str:
    """The CC `url_surtkey` of a URL (or of a bare host / `host/path`).

    Raises ValueError for a port that is not a number 0..65535 (the package did
    the same); `surt_key_or_none` is the never-raises form."""
    url = url_or_host.strip(" \t\n\r\x0b\x0c")
    url = url.replace("\t", "").replace("\r", "").replace("\n", "")
    if not url:
        return "-"
    url = url.split("#", 1)[0]
    m = _SCHEME.match(url)
    if not m:
        return _key("", url)
    scheme = m.group(1)
    after = url[m.end():]
    if scheme.lower() == "file" and after.startswith("///"):
        return _hostless(scheme, after[2:])
    if after.startswith("//"):
        rest = after[2:]
        if scheme.lower() in ("http", "https"):
            # `https://https://host/x` (a doubled scheme, common in scraped
            # links): the inner lower-case `http(s)://` goes too.
            while rest.startswith(("http://", "https://")):
                rest = rest.split("://", 1)[1]
        return _key(scheme, rest)
    if scheme.lower() in ("http", "https"):
        return _key(scheme, after.lstrip("/"), True)
    return f"{m.group(1)}:{after.lower()}"


#: Real (url, url_surtkey) pairs from Common Crawl's own columnar index, one per
#: rule family. `assert_key_parity` checks them at the start of a lookup run.
PARITY_PROBES: tuple[tuple[str, str], ...] = (
    ("https://www.sciencepg.com/journal/164/archive/1640603",
     "com,sciencepg)/journal/164/archive/1640603"),  # uniform
    ("https://www.old.mandarine.academy/tag/ma-pme-numerique/",
     "academy,mandarine,old)/tag/ma-pme-numerique"),  # uniform
    ("https://www.elfarmaceutico.es/tag/cof-de-madrid?_page=8",
     "es,elfarmaceutico)/tag/cof-de-madrid?_page=8"),  # uniform
    ("https://xn--m1abbbg.me/podborki",
     "me,xn--m1abbbg)/podborki"),  # idn
    ("http://www.omkie.com:3000/",
     "com,omkie:3000)/"),  # port
    ("https://worshipleader.me/blog/tag/Q%26A",
     "me,worshipleader)/blog/tag/q&a"),  # pct
    ("https://ginga101.com/?p=1127",
     "com,ginga101)/?p=1127"),  # query
    ("http://kk2000inc.com//",
     "com,kk2000inc)/"),  # dblslash
    ("http://www.midi247.com/(S(snmadkjm5v3gas3rfbcn3e0p))/Links.aspx",
     "com,midi247)/links.aspx"),  # aspnet_session
    ("http://tulab.mx/(S(o01ai0xjqqen45tmibethj4x))/Error?tipo=1",
     "mx,tulab)/(s(o01ai0xjqqen45tmibethj4x))/error?tipo=1"),  # aspnet_session
    ("https://nymland.com/category-super?sid=14",
     "com,nymland)/category-super?sid=14"),  # session_param
    ("http://phoneche2016.com/?tzzgx/../xxpl/dqgg.htm",
     "com,phoneche2016)/?tzzgx/../xxpl/dqgg.htm"),  # dotseg
    ("http://www.nydldx.com/dianlanNews118.html",
     "com,nydldx)/dianlannews118.html"),  # upper_path
    ("https://orianik.com/?s=sonido+industrial",
     "com,orianik)/?s=sonido+industrial"),  # plus_query
    ("http://vmk1.ru/",
     "ru,vmk1)/"),  # root_noquery
    ("https://www.ginfosoft.com/",
     "com,ginfosoft)/"),  # www
    ("http://guidexp.com/",
     "com,guidexp)/"),  # http
    ("https://my.eu.panasonic.com/#/login?lang=de_DE",
     "com,panasonic,eu,my)/"),  # fragment
)

_PARITY_OK = False


def assert_key_parity(force: bool = False) -> None:
    """Refuse to go on when `surt_key` no longer gives Common Crawl's key for the
    `PARITY_PROBES`. Checked once per process (`force=True` re-checks).

    A wrong key finds nothing, and "nothing" looks like a genuine "no capture";
    so the run stops here instead of recording a wave of false negatives."""
    global _PARITY_OK
    if _PARITY_OK and not force:
        return
    wrong = [(u, want, surt_key_or_none(u)) for u, want in PARITY_PROBES
             if surt_key_or_none(u) != want]
    if wrong:
        raise RuntimeError(f"surt key parity broken for {len(wrong)}/{len(PARITY_PROBES)} "
                           f"probe URLs, first: {wrong[0]!r}; refusing to look up Common Crawl")
    _PARITY_OK = True


def surt_key_or_none(url_or_host: str) -> Optional[str]:
    """`surt_key`, or None when the input cannot be keyed (never raises)."""
    try:
        return surt_key(url_or_host)
    except Exception:  # noqa: BLE001 -- the contract is "None on failure"
        return None
