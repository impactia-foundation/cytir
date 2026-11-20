"""All reader knobs in one place (task 2710)."""
from __future__ import annotations

import os

from cytir.core.user_agent import VERSION

FETCH_TIMEOUT_S = 20.0
MAX_BYTES = 10 * 1024 * 1024
RENDER_TIMEOUT_S = 40.0
VIRTUAL_TIME_BUDGET_MS = 10000
DEFAULT_MAX_CHARS = 20000
# Non-Chrome renderers run under a CPU limit and an RSS watchdog (RLIMIT_AS aborts them at start). Obscura reached 6.5 GB.
ALT_RENDERER_MAX_RSS = 2 * 1024 ** 3
ALT_RENDERER_RLIMIT_CPU_S = 120
# Shell detection (shell.py): little text from a big page, or an explicit marker.
SHELL_MIN_TEXT_CHARS = 400
SHELL_MIN_HTML_BYTES = 20 * 1024
SHELL_MAX_TEXT_RATIO = 0.01   # text/HTML under 1 % and no <main>/<article> with text = shell
# An honest reader identity. Not a browser impersonation (see cytir/core/user_agent.py); no operator mailbox,
# because a client-side reader acts for its user, not for an operator.
USER_AGENT = f"CytirReader/{VERSION} (+https://websear.ch; local reader acting for one user)"
BROWSER_BINARIES = ("google-chrome", "chromium", "chromium-browser", "msedge", "brave")
ALT_RENDERERS = ("obscura", "moli", "lightpanda")
INSTALL_HINT = ("This page needs JavaScript and no renderer was found. Install Chrome or Chromium "
                "(or set CYTIR_READER_RENDERER to obscura, moli or lightpanda with the binary on PATH).")


def renderer_choice() -> str:
    """`AW_READER_RENDERER`: auto (default = the installed browser) | installed_browser | obscura | moli | lightpanda | none."""
    return os.environ.get("AW_READER_RENDERER", "auto").strip().lower() or "auto"
