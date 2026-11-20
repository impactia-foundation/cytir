"""Canonicalize curl transport evidence for one preserved HTTP response."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from cytir.acquire.artifact_io import atomic_publish, sha256_file

REDACTED = "<redacted>"
#: header NAMES whose values are credentials; the names stay, the values never reach disk
SECRET_HEADERS = frozenset({"authorization", "proxy-authorization", "cookie", "set-cookie"})


def redact_url(url: str) -> str:
    """Replace URL userinfo (`user:pass@`) with `<redacted>@`; any other URL is returned unchanged."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return REDACTED if "@" in url else url
    if "@" not in parts.netloc:
        return url
    host = parts.netloc.rsplit("@", 1)[1]
    return urlunsplit(parts._replace(netloc=f"{REDACTED}@{host}"))


#: header NAMES whose values are URLs that may carry userinfo
URL_HEADERS = frozenset({"location", "content-location"})


def redact_header_dump(raw: bytes) -> bytes:
    """Blank credential header values and strip URL userinfo in Location/Content-Location.

    Names and line ends are kept. obs-fold continuation lines (leading space/tab) belong to
    the previous header: blanked with a secret header, joined into the URL for a URL header.
    """
    lines = raw.splitlines(keepends=True)
    out: list[bytes] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        name, sep, value = line.partition(b":")
        key = name.strip().lower().decode("latin-1") if sep and not line[:1] in (b" ", b"\t") else ""
        j = i + 1
        while j < len(lines) and lines[j][:1] in (b" ", b"\t"):
            j += 1
        if key in SECRET_HEADERS:
            end = lines[j - 1][len(lines[j - 1].rstrip(b"\r\n")):]
            out.append(name + b": " + REDACTED.encode() + end)
        elif key in URL_HEADERS:
            end = lines[j - 1][len(lines[j - 1].rstrip(b"\r\n")):]
            text = b"".join([value.rstrip(b"\r\n")] + [x.rstrip(b"\r\n") for x in lines[i + 1:j]])
            clean = redact_url(text.strip().decode("latin-1")).encode("latin-1")
            if j == i + 1 and clean == text.strip():
                out.append(line)
            else:
                out.append(name + b": " + clean + end)
        else:
            out.extend(lines[i:j])
        i = j
    return b"".join(out)


def _redact_json(value):
    if isinstance(value, str):
        return redact_url(value) if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", value) else value
    if isinstance(value, dict):
        # curl's %{json} splits the userinfo out: `url.user`, `url.password`, `urle.user`, ...
        return {k: (REDACTED if v and k.rsplit(".", 1)[-1] in ("user", "password", "userinfo")
                    else _redact_json(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_json(v) for v in value]
    return value


def _redact_json_bytes_changed(raw: bytes, redacted) -> bool:
    return json.loads(raw.decode("utf-8")) != redacted


def _rewrite(path: Path, data: bytes) -> None:
    """Replace a not-yet-sealed transport file with its redacted bytes (temp + rename)."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.redact")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def record(*, body: Path, headers: Path, curl_metrics: Path,
           requested_url: str, fetched_at: str, output: Path) -> dict:
    # Sanitize the raw evidence files in place BEFORE they are hashed or sealed: a credentialed
    # fetch must leave no secret in the headers dump, the curl metrics, the metadata or any log.
    raw_headers = headers.read_bytes()
    clean_headers = redact_header_dump(raw_headers)
    if clean_headers != raw_headers:
        _rewrite(headers, clean_headers)
    metrics = _redact_json(json.loads(curl_metrics.read_text(encoding="utf-8")))
    raw_metrics = curl_metrics.read_bytes()
    clean_metrics = (json.dumps(metrics, indent=2, sort_keys=True) + "\n").encode()
    if _redact_json_bytes_changed(raw_metrics, metrics):
        _rewrite(curl_metrics, clean_metrics)
    requested_url = redact_url(requested_url)
    status = int(metrics.get("http_code") or metrics.get("response_code") or 0)
    final_url = str(metrics.get("url_effective") or "")
    if status != 200 or not final_url:
        raise ValueError(f"HTTP fetch is not a complete success: {status} {redact_url(final_url)!r}")
    result = {
        "schema": "aw.http-fetch-metadata.v1",
        "requested_url": requested_url,
        "final_url": final_url,
        "http_status": status,
        "fetched_at": fetched_at,
        "body": {"bytes": body.stat().st_size, "sha256": sha256_file(body)},
        "response_headers": {
            "encoding": "curl-dump-header-raw",
            "bytes": headers.stat().st_size,
            "sha256": sha256_file(headers),
        },
    }
    data = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    if output.exists():
        if sha256_file(output) != hashlib.sha256(data).hexdigest():
            raise ValueError(f"immutable fetch metadata collision: {output}")
    else:
        atomic_publish(output, data)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body", type=Path, required=True)
    parser.add_argument("--headers", type=Path, required=True)
    parser.add_argument("--curl-metrics", type=Path, required=True)
    parser.add_argument("--requested-url", required=True)
    parser.add_argument("--fetched-at", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = record(body=args.body, headers=args.headers,
                    curl_metrics=args.curl_metrics,
                    requested_url=args.requested_url,
                    fetched_at=args.fetched_at, output=args.out)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
