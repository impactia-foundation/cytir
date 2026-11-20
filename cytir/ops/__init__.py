"""cytir.ops — operational tooling that production systemd units and shell
drivers call directly (task 646, weekly goal 2026-09-25: "nothing in
production runs from tasks/"). Moved out of tasks/*-scripts/ one file at a
time; each move leaves a shim at the old path (task 643 §6.2).
"""
from __future__ import annotations
