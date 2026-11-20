"""cytir.probe -- the standing random-probe quality loop: samplers, the polite read-only runner, the
Slice B scorer and the two mechanical checks (empty-snippet rate, novel-query latency).

Moved from the old `probe_*` scripts (task 1521, ops port S1); run as `python3 -m cytir.probe.<name>` by
`deploy/scripts/probe-run.sh`. No forwarding shims: the shell is the only caller.
"""
