from cytir.acquire import cc_digest as subject


def test_sha1_normalization_accepts_cc_and_warc_forms() -> None:
    bare = subject.EMPTY_SHA1_BASE32
    assert subject.normalize_sha1(bare.lower()) == bare
    assert subject.normalize_sha1(f"sha1:{bare}") == bare
    assert subject.normalize_sha1(f"SHA1:{bare}====") == bare
    assert subject.is_empty_sha1(bare) is True
    assert subject.matches_sha1(b"", bare) is True


def test_sha1_normalization_rejects_wrong_algorithm_or_shape() -> None:
    for value in ("", "sha256:" + subject.EMPTY_SHA1_BASE32,
                  "sha1:NOPE", "not-base32"):
        assert subject.normalize_sha1(value) is None
        assert subject.is_empty_sha1(value) is False
