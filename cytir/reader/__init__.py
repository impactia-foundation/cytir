"""The local reader (task 2710, Layer 2 of the agent read step; `specs/agent-read-step.md`).

Runs on the CLIENT: `cyt read <url>` and the stdio MCP server `python -m cytir.reader.mcp`. It is not part of the
serving box. Fetch, JavaScript-shell detection, optional render with an installed browser, extraction, anchor trim.
"""
