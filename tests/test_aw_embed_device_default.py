"""Task 912 — `cyt embed shards` device/dtype defaults follow the hardware.

Pure-function tests of `resolve_device_dtype`; no torch or GPU needed.
"""
from __future__ import annotations

from cytir.embed.shards import build_parser, resolve_device_dtype


def test_auto_on_cpu_box_resolves_to_cpu_float32():
    assert resolve_device_dtype("auto", "auto", cuda_available=False) == (
        "cpu", "float32")


def test_auto_on_cuda_box_resolves_to_cuda_float16():
    assert resolve_device_dtype("auto", "auto", cuda_available=True) == (
        "cuda:0", "float16")


def test_explicit_values_pass_through_unchanged():
    assert resolve_device_dtype("cpu", "float16", cuda_available=True) == (
        "cpu", "float16")
    assert resolve_device_dtype("cuda:1", "float32", cuda_available=False) == (
        "cuda:1", "float32")


def test_parser_defaults_are_auto():
    args = build_parser().parse_args(["--in", "a", "--out", "b"])
    assert args.device == "auto"
    assert args.compute_dtype == "auto"
