import hashlib

import pytest

from cytir.extract.routec.w3c_technical_reports import (
    _validate_binding, is_technical_report_url, report_path_identity)


class _Node:
    def __init__(self, href): self.href = href
    def get(self, key): return self.href if key == "href" else None


class _Soup:
    def find(self, name, rel):
        assert name == "link"
        assert rel("canonical")
        return _Node("https://www.w3.org/TR/html-aria/")


def test_w3c_tr_identity_is_exact_path_not_broad_host() -> None:
    assert is_technical_report_url(
        "https://www.w3.org/TR/2017/WD-html-aria-20170909")
    assert is_technical_report_url("https://www.w3.org/TR/WD-script-960124.html")
    assert not is_technical_report_url("https://www.w3.org/blog/2017/spec")
    assert not is_technical_report_url("https://example.org/TR/spec")


def test_report_identity_preserves_version_path_and_declared_canonical() -> None:
    value = report_path_identity(
        "https://www.w3.org/TR/2017/WD-html-aria-20170909",
        "https://www.w3.org/TR/2017/WD-html-aria-20170909/", _Soup())
    assert value["requested_path"] == "/TR/2017/WD-html-aria-20170909"
    assert value["status_hint_from_path"] == "WD"
    assert value["version_date_hint_from_path"] == "20170909"
    assert value["declared_canonical_url"] == "https://www.w3.org/TR/html-aria/"


def test_plain_terminal_receipt_binding_is_not_misread_as_gzip(tmp_path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}\n")
    binding = {"uri": "artifact://receipt.json", "bytes": receipt.stat().st_size,
               "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest()}
    assert _validate_binding(tmp_path, binding) == receipt
    binding["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="plain artifact binding failed"):
        _validate_binding(tmp_path, binding)
