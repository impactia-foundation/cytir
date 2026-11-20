"""The last amendment row's pre_amendment_file_sha256 is verified by `contract --check`."""
import hashlib

import pytest

from cytir.core.paths import data_path
from cytir.sources.contract import FamilyContractError, pre_amendment_sha256


def _build(last_body: str, first_extra: str = "") -> str:
    """Hand-formatted file; the recorded hash is the sha of the file minus the last object."""
    head = ('{\n  "note": "x",\n  "amendments": [\n'
            '    {"old_file_sha256": "aa", "pre_amendment_file_sha256": "bb"%s},\n'
            '    {"old_file_sha256": "cc", "pre_amendment_file_sha256": "dd"},\n' % first_extra)
    tail = '\n  ],\n  "families": {}\n}\n'
    pre = hashlib.sha256((head.rstrip().rstrip(",") + tail).encode()).hexdigest()
    return head + "    {" + last_body.replace("@@", pre) + "}" + tail


_PLAIN = '"old_file_sha256": "ee", "pre_amendment_file_sha256": "@@"'


def test_real_file_passes():
    rec, comp = pre_amendment_sha256(data_path("source-family-contracts.v1.json").read_text("utf-8"))
    assert rec == comp


def test_synthetic_three_rows_pass():
    rec, comp = pre_amendment_sha256(_build(_PLAIN))
    assert rec == comp


def test_flipped_byte_fails():
    text = _build(_PLAIN)
    rec, _ = pre_amendment_sha256(text)
    bad = ("0" if rec[0] != "0" else "1") + rec[1:]
    rec2, comp2 = pre_amendment_sha256(text.replace(rec, bad))
    assert rec2 != comp2


def test_nested_object_before_old_sha_passes():
    body = '"meta": {"a": {"b": 1}}, "old_file_sha256": "ee", "pre_amendment_file_sha256": "@@"'
    rec, comp = pre_amendment_sha256(_build(body))
    assert rec == comp


def test_braces_in_strings_pass():
    text = _build('"why": "{not an object} \\" }", ' + _PLAIN,
                  first_extra=', "t": "{not an object}"')
    rec, comp = pre_amendment_sha256(text)
    assert rec == comp


@pytest.mark.parametrize("text", [
    "not json",
    '{"amendments": []}',
    '{"amendments": [{"old_file_sha256": "a"}]}',
    '[1, 2]',
])
def test_malformed_fails_closed(text):
    with pytest.raises(FamilyContractError):
        pre_amendment_sha256(text)
