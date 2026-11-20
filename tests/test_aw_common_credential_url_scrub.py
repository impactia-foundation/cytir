# Task 646 S0: credential-URL scrub in `canonical_url`, gated OFF by default
# (task 643 §5.1 / §6.2 S0 — canonical_url is an identity input; scrubbing it
# by default would re-key every point whose URL carries a credential param).
#
# Task 646 S2 moved `canonical_url` + the scrub to `cytir.core.canon`;
# `aw_common.canonical_url` is now a re-exported reference to the SAME
# function object, so the flag it reads (`_SCRUB_CREDENTIAL_URL_PARAMS`) must
# be toggled on `cytir.core.canon` -- the module that actually owns it -- not on
# `aw_common` (setting an attribute there would silently create an unrelated
# name on the aw_common module object and change nothing; see
# test_aw_common_canon_shim.py for the identity proof).

import cytir.core.canon as canon
import cytir.core.common as AC

JWT_URL = ("https://tomcat.example.org/changelog"
           "?accessToken=eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ4In0.sig123")  # gitleaks:allow (fake test fixture, not a real credential)


def setup_function(_fn):
    canon.reset_credential_url_hit_count()
    canon._SCRUB_CREDENTIAL_URL_PARAMS = False


def teardown_function(_fn):
    canon.reset_credential_url_hit_count()
    canon._SCRUB_CREDENTIAL_URL_PARAMS = False


def test_flag_off_by_default():
    assert canon._SCRUB_CREDENTIAL_URL_PARAMS is False


def test_scrub_off_leaves_canonical_url_unchanged_but_still_counts():
    # Exercised via aw_common's re-exported name -- proves the shim path
    # still behaves like the pre-move code, not just the new one.
    before = AC.canonical_url(JWT_URL)
    assert "accessToken=" in before          # unchanged: no rekey with flag off
    assert AC.credential_url_hit_count() == 1


def test_scrub_on_strips_credential_param_and_counts():
    canon._SCRUB_CREDENTIAL_URL_PARAMS = True
    after = AC.canonical_url(JWT_URL)
    assert "accessToken" not in after
    assert "sig123" not in after
    assert after == "https://tomcat.example.org/changelog"
    assert AC.credential_url_hit_count() == 1


def test_jwt_shaped_value_caught_even_under_an_unlisted_key_name():
    url = ("https://example.org/doc"
           "?weird_key=eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ4In0.sig")  # gitleaks:allow (fake test fixture, not a real credential)
    canon._SCRUB_CREDENTIAL_URL_PARAMS = True
    after = AC.canonical_url(url)
    assert after == "https://example.org/doc"


def test_functional_params_survive_the_scrub():
    url = "https://fedlex.admin.ch/eli/cc/24/233_245_233?lang=de&accessToken=eyJa.eyJb.c"
    canon._SCRUB_CREDENTIAL_URL_PARAMS = True
    after = AC.canonical_url(url)
    assert "lang=de" in after
    assert "accessToken" not in after


def test_no_credential_param_no_hit():
    url = "https://fedlex.admin.ch/eli/cc/24/233_245_233?lang=de"
    AC.canonical_url(url)
    assert AC.credential_url_hit_count() == 0
