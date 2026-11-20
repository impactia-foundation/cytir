"""ONE home for the crawler's HTTP identity (task 432, item 4).

THE DEFECT THIS REMOVES
-----------------------
Every fetching script in this repo hardcoded a User-Agent naming *our* site and
*our* mailbox — `https://websear.ch`, `bender@impactia.org`. A stranger who
clones this repo and runs the pipeline therefore **identifies as us to every
host they crawl**. That is not cosmetic:

  * site operators' allow/deny decisions and rate limits are made against a
    token, and they would be applying them to traffic we did not send;
  * abuse reports for a third party's crawl arrive in our mailbox, about hosts
    we never touched;
  * our own consent evidence — which is per-host and lives or dies on being an
    honest record of who asked what — gets polluted by other people's requests.

Consent is a founding constraint of this project. Impersonation at the identity
layer makes the consent record unreliable for everyone.

THE CONTRACT
------------
* Build the UA here; never restate it at a call site.
* The identity is **configuration**, not source. It comes from the environment
  or `operator.json`, and the built-in default is an obvious placeholder that
  says out loud, in the UA itself, that nobody configured it.
* We never impersonate a browser. A UA that says `Mozilla/5.0 (X11 …)` is a lie
  told to a machine that is trying to decide whether to trust us, and it makes
  robots.txt meaningless. The UA carries product, version, a URL a human can
  read, and an address a human can write to.

CONFIGURATION, in order of precedence
-------------------------------------
1. ``AW_OPERATOR_URL`` / ``AW_OPERATOR_CONTACT`` environment variables.
2. ``operator.json`` at the repo root (git-ignored; copy `operator.example.json`).
3. The operator overlay (`cytir.core.paths`: settings ``operator_url`` / ``operator_contact`` in
   the file named by ``AW_OPERATOR_OVERLAY``, else the box file), so every entry point on a
   box has one identity. A stranger has no overlay.
4. Nothing — in which case `user_agent()` **raises**. It does not quietly
   substitute a placeholder, because a placeholder on the wire is still a wrong
   answer to "who is asking?", and a lane that starts unconfigured would write
   consent evidence describing a crawler that does not exist. Failing closed
   turns an invisible evidence-corruption into a one-line fix at startup.

   To crawl deliberately without an identity (local experiment, offline test),
   set ``AW_ALLOW_UNIDENTIFIED_CRAWL=1``. You then get the placeholder below —
   honest, non-impersonating, self-describing — plus a loud banner.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# cytir/core/user_agent.py -> repo root (moved from pipeline/, task 677: one level deeper).
REPO_ROOT = Path(__file__).resolve().parents[2]

PRODUCT = "authoritative-web"
#: Bumping this changes the UA every host sees. It is not a release version;
#: it is the crawler-identity version. Leave it alone unless the crawler's
#: behaviour changes in a way an operator would want to distinguish.
VERSION = "0.1"

CONFIG_FILE = REPO_ROOT / "operator.json"
EXAMPLE_FILE = REPO_ROOT / "operator.example.json"

#: Deliberately self-describing: a site operator reading their logs can tell
#: instantly that this is an unconfigured self-hosted fork and not us.
PLACEHOLDER_URL = "https://example.invalid/set-AW_OPERATOR_URL"
PLACEHOLDER_CONTACT = "UNCONFIGURED-set-AW_OPERATOR_CONTACT"

_warned = False


class UnidentifiedCrawler(RuntimeError):
    """No operator identity is configured and unidentified crawling was not
    explicitly allowed. Raised instead of putting a placeholder on the wire."""


def _load_file() -> dict:
    try:
        with open(CONFIG_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {}
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"{CONFIG_FILE} exists but could not be read as JSON ({exc}). Fix it or "
            f"delete it — a malformed operator identity must not fall back to a "
            f"placeholder, because then you would not know who you are crawling as."
        ) from exc


def operator() -> tuple[str, str, bool]:
    """Return (url, contact, configured)."""
    cfg = _load_file()
    url = (os.environ.get("AW_OPERATOR_URL") or cfg.get("url") or "").strip()
    contact = (os.environ.get("AW_OPERATOR_CONTACT") or cfg.get("contact") or "").strip()
    if not (url and contact):
        # the box overlay (`cytir.core.paths`): one identity for every entry point on a box. Last in
        # the order, and only for what the environment and `operator.json` left empty.
        from cytir.core import paths  # noqa: PLC0415 (paths must stay importable without us)
        url = url or (paths.overlay_value("operator_url") or "").strip()
        contact = contact or (paths.overlay_value("operator_contact") or "").strip()
    if url and contact:
        return url, contact, True
    return (url or PLACEHOLDER_URL), (contact or PLACEHOLDER_CONTACT), False


def _warn_once(url: str, contact: str) -> None:
    global _warned
    if _warned:
        return
    _warned = True
    bar = "!" * 72
    sys.stderr.write(
        f"\n{bar}\n"
        f"!! AW CRAWLER IDENTITY IS NOT CONFIGURED\n"
        f"!! Outgoing requests will be sent as:\n"
        f"!!   {PRODUCT}/{VERSION} (+{url}; {contact})\n"
        f"!! Every host you fetch logs that string, and every allow/deny decision,\n"
        f"!! rate limit and abuse report is made against it. Set your OWN identity\n"
        f"!! before crawling anything:\n"
        f"!!   cp operator.example.json operator.json   # then edit it\n"
        f"!! or export CYTIR_OPERATOR_URL=... CYTIR_OPERATOR_CONTACT=...\n"
        f"{bar}\n")
    sys.stderr.flush()


ALLOW_UNIDENTIFIED_ENV = "AW_ALLOW_UNIDENTIFIED_CRAWL"


def _identity() -> tuple[str, str]:
    """(url, contact) of the configured operator; fails closed (see the module docstring)."""
    url, contact, configured = operator()
    if not configured:
        if os.environ.get(ALLOW_UNIDENTIFIED_ENV) != "1":
            raise UnidentifiedCrawler(
                f"AW crawler identity is not configured, so there is no honest "
                f"User-Agent to send.\n"
                f"  Set it:  export CYTIR_OPERATOR_URL=... CYTIR_OPERATOR_CONTACT=...\n"
                f"  or:      cp {EXAMPLE_FILE.name} {CONFIG_FILE.name}   # then edit it\n"
                f"  Looked for {CONFIG_FILE}\n"
                f"  Refusing rather than sending "
                f"'{PRODUCT}/{VERSION} (+{url}; {contact})' — every host you fetch logs "
                f"that string, every allow/deny decision and rate limit is applied to it, "
                f"and our own consent evidence is only worth something if it names the "
                f"crawler that actually made the request.\n"
                f"  To crawl unidentified ON PURPOSE, set {ALLOW_UNIDENTIFIED_ENV}=1.")
        _warn_once(url, contact)
    return url, contact


def user_agent(component: str | None = None) -> str:
    """The User-Agent for every outbound request.

    Raises `UnidentifiedCrawler` when no operator identity is configured, unless
    ``AW_ALLOW_UNIDENTIFIED_CRAWL=1``. See the module docstring for why this
    fails closed rather than falling back.

    `component` is an optional short token for a distinct crawl behaviour that a
    site operator might reasonably want to tell apart (e.g. "consent",
    "sitemap"). It is appended, never substituted — the product, the URL and the
    contact are always present.
    """
    url, contact = _identity()
    ua = f"{PRODUCT}/{VERSION} (+{url}; {contact})"
    if component:
        ua = f"{ua} {component}"
    return ua


#: The pack-fetch crawler (`cytir.acquire.crawl`) has always sent this product token, and sites'
#: robots.txt rules and our own L0 manifests name it; it stays as it is. Only the URL and the
#: contact come from the operator identity, built here and nowhere else (lesson L001).
PACK_FETCH_PRODUCT = "AuthoritativeWebBot"


def pack_fetch_user_agent() -> str:
    """The pack-fetch crawler's User-Agent: `AuthoritativeWebBot (+<url>; <contact>)`, from the
    same operator identity and the same fail-closed rule as `user_agent()`."""
    url, contact = _identity()
    return f"{PACK_FETCH_PRODUCT} (+{url}; {contact})"


def is_configured() -> bool:
    return operator()[2]


def assert_configured(context: str = "") -> None:
    """For entry points that must not crawl under a placeholder identity."""
    url, contact, configured = operator()
    if configured:
        return
    ctx = f" [{context}]" if context else ""
    raise RuntimeError(
        f"AW crawler identity is not configured{ctx}. Requests would go out as "
        f"'{user_agent()}'. Set CYTIR_OPERATOR_URL and CYTIR_OPERATOR_CONTACT, or copy "
        f"{EXAMPLE_FILE.name} to {CONFIG_FILE.name} and edit it.")


def main(argv: list[str] | None = None) -> int:
    # `python3 -m cytir.core.user_agent` prints the UA — this
    # is also how shell scripts get it, so there is exactly one implementation.
    argv = sys.argv[1:] if argv is None else argv
    sys.stdout.write(user_agent(argv[0] if argv else None) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
