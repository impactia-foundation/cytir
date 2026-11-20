"""cytir.core.canon.https_frontier_url replaced the pack `canon` copies (task 2166): same output."""
from __future__ import annotations

import urllib.parse

import pytest

from cytir.acquire import crawl as acquire_crawl
from cytir.core.canon import https_frontier_url
from cytir.sources.rights_canon import crawl as rights_crawl

ALIAS = {"cnil.fr": "www.cnil.fr"}
URLS = ["http://CNIL.fr", "https://cnil.fr/a/?x=1#f", "", "//x", "http://[bad", "relative/p",
        "https://a.b/c/", "http://www.x.org/p?q=1&r=2"]


def _old(url, keep_query=False, alias=None):
    """The body of the three retired private copies, verbatim."""
    alias = alias or {}
    try:
        p = urllib.parse.urlsplit(url)
    except ValueError:
        return ""
    host = (p.netloc or "").lower()
    host = alias.get(host, host)
    return urllib.parse.urlunsplit(("https", host, p.path or "/", p.query if keep_query else "", ""))


@pytest.mark.parametrize("url", URLS)
@pytest.mark.parametrize("keep", [False, True])
def test_matches_retired_copies(url, keep):
    assert https_frontier_url(url, keep, ALIAS) == _old(url, keep, ALIAS)
    assert https_frontier_url(url, keep) == _old(url, keep)
    assert acquire_crawl.make_canon(ALIAS)(url, keep_query=keep) == _old(url, keep, ALIAS)
    assert acquire_crawl.make_canon()(url, keep) == _old(url, keep)
    assert rights_crawl.canon(url, keep) == _old(url, keep, rights_crawl.ALIAS)
