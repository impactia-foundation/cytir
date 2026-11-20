import cytir.ops.audits.technical_product_profile as subject


def test_volatile_commerce_keys_are_detected_recursively():
    value = {"name": "Robot", "nested": [{"offers": {"price": "99"}}]}
    assert {"offers", "price"}.issubset(subject._keys(value))
    assert subject._keys(value) & subject.VOLATILE


def test_requested_target_is_queryless_and_www_normalized():
    assert subject.TARGET == (
        "https://sunfounder.com/products/"
        "sunfounder-pidog-robot-dog-kit-for-raspberry-pi")
