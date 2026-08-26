"""
A working `torch._int_mm` on AMD GPUs whose ROCm libraries do not provide one.

INT8 checkpoints reach `torch._int_mm` for their matmuls (comfy-kitchen's
`fast_int8_mm`, and every quantized layout built on it). On ROCm that op is
served by hipBLASLt, which ships kernels only for a subset of architectures --
RDNA 2 (`gfx103X`) is not among them. There is no graceful failure: the call
lands in torch_hip.dll and the process dies with

    Exception Code: 0xC0000005
      ... ?_int_mm@cuda@at@@YA?AVTensor@2@AEBV32@0@Z() + 0x10D byte(s)

on the first sampling step, which reads as a driver problem rather than a
missing kernel. The silicon itself is fine -- RDNA 2 has the INT8 dot-product
instructions -- so this is a gap in the math library, not in the hardware.

Whether the op works is decided by trying it in a *subprocess*: a crash there
costs a few seconds instead of the session, and it catches a wrong answer as
well as a hard fault. The verdict is cached per (torch build, GPU).
"""

import os
import subprocess
import sys

from modules.timer import startup_timer

#: Exercised on the GPU and checked against an exact CPU reference: hipBLASLt
#: gaps show up as a segfault, but a mis-dispatched kernel would quietly return
#: nonsense, and both make the checkpoint unusable.
_PROBE = """
import ctypes, os, sys
if os.name == "nt":
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)  # no crash dialogs

import torch

device = torch.device("cuda", 0)
a = torch.randint(-127, 127, (64, 128), dtype=torch.int8)
b = torch.randint(-127, 127, (128, 64), dtype=torch.int8)

expected = a.to(torch.int32) @ b.to(torch.int32)
actual = torch._int_mm(a.to(device), b.to(device)).cpu()

assert torch.equal(actual, expected), "torch._int_mm returned incorrect results"
"""


def _receipt_path(torch) -> str:
    from modules.paths_internal import script_path

    build = torch.__version__.replace("+", "-")
    return os.path.join(script_path, "tmp", f"int-mm-{build}.ok")


def int_mm_is_usable(torch) -> bool:
    """Whether `torch._int_mm` computes correctly on this GPU."""

    receipt = _receipt_path(torch)
    try:
        with open(receipt, "r", encoding="utf-8") as file:
            return file.read().strip() == "usable"
    except OSError:
        pass

    probe = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True)
    usable = probe.returncode == 0

    try:
        os.makedirs(os.path.dirname(receipt), exist_ok=True)
        with open(receipt, "w", encoding="utf-8") as file:
            file.write("usable" if usable else "broken")
    except OSError:
        pass

    return usable


def _int_mm_via_float32(a, b):
    """
    INT8 x INT8 -> INT32, computed in fp32.

    Every product fits in 15 bits and fp32 carries 24, so the result is exact
    until the accumulation itself passes 2^24. Even then it barely drifts:
    a deliberately saturated K=16384 (every input at +/-127, far beyond
    anything real activations do) lands within 6 of an exact 264,000,000 --
    and the caller multiplies by a dequantisation scale immediately after.

    fp32 here means fp32: the architectures that need this fallback are the
    ones with no reduced-precision matmul path to fall into.
    """

    import torch

    return torch.matmul(a.to(torch.float32), b.to(torch.float32)).round_().to(torch.int32)


def install_int_mm_fallback(torch) -> bool:
    """Swap in the fp32 fallback when the native INT8 matmul cannot be used."""

    if int_mm_is_usable(torch):
        return False

    torch._int_mm = _int_mm_via_float32
    if hasattr(torch, "int8_mm"):
        torch.int8_mm = _int_mm_via_float32

    startup_timer.record("int8 matmul fallback")
    return True
