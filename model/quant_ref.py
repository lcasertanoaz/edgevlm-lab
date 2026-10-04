#!/usr/bin/env python3
"""Fixed-point reference for the accelerator: symmetric integer quantization
and integer matrix multiply, plus test-vector export for the Verilog testbench.

This is the "golden" model the RTL must match bit-for-bit: the RTL computes
int(A) @ int(B) exactly, and dequantization (scale multiply) happens outside
the array.

Run:  python model/quant_ref.py          (error vs. precision on random data)
"""
import numpy as np


def quantize(x, bits, axis=None):
    """Symmetric quantization to signed `bits`-bit integers.
    axis=None -> one scale per tensor; axis=k -> one scale per slice along k
    (e.g. axis=1 on a (K, N) weight matrix = per-output-channel scales)."""
    qmax = (1 << (bits - 1)) - 1
    amax = np.max(np.abs(x)) if axis is None else np.max(np.abs(x), axis=1 - axis, keepdims=True)
    scale = np.where(amax == 0, 1.0, amax / qmax)
    q = np.clip(np.round(x / scale), -qmax - 1, qmax).astype(np.int64)
    return q, scale


def int_matmul(qa, qb):
    """Exact integer matmul (what the systolic array computes)."""
    return qa.astype(np.int64) @ qb.astype(np.int64)


def quantized_linear(x, w, bits, per_channel=True):
    """y = x @ w computed with quantized activations and weights."""
    qx, sx = quantize(x, bits)
    qw, sw = quantize(w, bits, axis=1 if per_channel else None)
    return int_matmul(qx, qw) * sx * sw


def write_hex(path, mat, bits):
    """Write a matrix as one two's-complement hex word per line ($readmemh)."""
    digits = (bits + 3) // 4
    mask = (1 << bits) - 1
    with open(path, "w") as f:
        for v in np.asarray(mat).flatten():
            f.write(f"{int(v) & mask:0{digits}x}\n")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # stand-in for one transformer projection: 64 tokens x 576 features -> 576
    x = rng.standard_normal((64, 576)).astype(np.float32)
    w = (rng.standard_normal((576, 576)) / np.sqrt(576)).astype(np.float32)
    ref = x @ w
    print("relative error of a quantized linear layer (random data, not the real model):")
    for bits in (16, 8, 4):
        for pc in (False, True):
            y = quantized_linear(x, w, bits, per_channel=pc)
            err = np.linalg.norm(y - ref) / np.linalg.norm(ref)
            print(f"  {bits:2d}-bit  {'per-channel' if pc else 'per-tensor ':11s}  {err:.2e}")
