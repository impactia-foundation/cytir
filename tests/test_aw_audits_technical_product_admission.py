from cytir.ops.audits import technical_product_admission as ADMISSION


def test_requested_pidog_target_consequence_normalizes_www_and_tracking():
    result = ADMISSION._requested_target_consequence([{
        "source_url": "https://www.sunfounder.com/products/"
                      "sunfounder-pidog-robot-dog-kit-for-raspberry-pi"
                      "?srsltid=tracking",
        "source_status": 200,
        "admitted": True,
    }])
    assert result["requested_canonical_url"] == (
        "https://sunfounder.com/products/"
        "sunfounder-pidog-robot-dog-kit-for-raspberry-pi")
    assert result["frontier_selected"] is True
    assert result["source_preserved"] is True
    assert result["admitted"] is True
