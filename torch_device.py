#!/usr/bin/env python3
"""
Pick the best torch device available on this machine and describe it.

Priority: CUDA (NVIDIA)  >  MPS (Apple Silicon GPU)  >  CPU.

The original training was done on a Windows/CUDA box; on this Mac the GPU is
Apple's Metal backend ("mps"), which `torch.cuda.is_available()` never reports.
Every inference / training entry point should call `select_device()` instead of
hand-rolling a cuda-or-cpu check.

    from torch_device import select_device, describe, ultralytics_arg
    device = select_device()
    short, pretty = describe(device)          # ("mps", "MPS (Apple GPU)")
    yolo.predict(..., device=ultralytics_arg(device))
"""

import os

# Let unsupported MPS ops fall back to CPU instead of crashing. Must be set
# before the first tensor op; importing this module early makes that reliable.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _mps_available() -> bool:
    try:
        import torch
        return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    except Exception:
        return False


def select_device(prefer: str | None = None):
    """Return a torch.device. `prefer` ("cuda"/"mps"/"cpu") overrides autodetect."""
    import torch

    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe(device) -> tuple[str, str]:
    """(short_name, human_label) for logging / the GUI indicator."""
    kind = getattr(device, "type", str(device))
    if kind == "cuda":
        try:
            import torch
            return "cuda", f"CUDA — {torch.cuda.get_device_name(0)}"
        except Exception:
            return "cuda", "CUDA GPU"
    if kind == "mps":
        return "mps", "MPS — Apple GPU"
    return "cpu", "CPU"


def ultralytics_arg(device):
    """Ultralytics wants 0 / 'mps' / 'cpu' rather than a torch.device."""
    kind = getattr(device, "type", str(device))
    if kind == "cuda":
        return 0
    return kind  # "mps" or "cpu"


def autocast_kwargs(device) -> dict:
    """
    kwargs for `torch.autocast(**kwargs)`. AMP only really helps on CUDA;
    elsewhere we disable it so the same code path runs everywhere.
    """
    kind = getattr(device, "type", str(device))
    return {"device_type": "cuda", "enabled": kind == "cuda"}
