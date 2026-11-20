import pytest

from cytir.core.common import require_qdrant_point_id


def test_qdrant_point_id_accepts_canonical_uuid_and_uint64():
    value = "a4ae9c62-0e5d-58fe-97c8-5b943a91e18a"
    assert require_qdrant_point_id(value) == value
    assert require_qdrant_point_id(0) == "0"
    assert require_qdrant_point_id(2 ** 64 - 1) == str(2 ** 64 - 1)


@pytest.mark.parametrize("value", [
    "a" * 64,
    "A4AE9C62-0E5D-58FE-97C8-5B943A91E18A",
    -1,
    2 ** 64,
    True,
])
def test_qdrant_point_id_rejects_non_qdrant_or_noncanonical_values(value):
    with pytest.raises(ValueError):
        require_qdrant_point_id(value)
