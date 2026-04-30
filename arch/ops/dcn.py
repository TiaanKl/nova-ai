# arch/ops/dcn.py
#
# Thin Python wrapper around the native DCNv2 CUDA op.
# - At runtime, you bind the real CUDA kernels via `bind_native_ops(...)`.
# - For ONNX export, you can monkey‑patch `modulated_deform_conv`
#   to call a fake op (e.g. NovaDcnv2Op.apply) instead of CUDA.

from __future__ import annotations

from typing import Any, Optional
import torch

_native_ops: Optional[Any] = None


def bind_native_ops(native_ops: Any) -> None:
    """Bind the native CUDA ops object (from `load_nova_kernels()`)."""
    global _native_ops
    _native_ops = native_ops


def modulated_deform_conv(feat: torch.Tensor, offset: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    High-level DCNv2 entrypoint used by the model and by ONNX export.

    At runtime this delegates to the bound native ops implementation. The
    compiled extension exposes `dcn_forward(...)` (the C++ bridge) which
    computes an im2col buffer; here we call that and return the final
    reconstructed feature map so callers don't need to manage the im2col
    buffer themselves.

    During ONNX export this symbol can be monkey-patched to emit a custom
    ONNX node instead of calling native code.
    """
    if _native_ops is None:
        raise RuntimeError("modulated_deform_conv was called before native ops were bound or ONNX patching.")

    # Some native implementations (if provided) may already expose a
    # convenience `modulated_deform_conv` that returns the final tensor.
    if hasattr(_native_ops, "modulated_deform_conv"):
        return _native_ops.modulated_deform_conv(feat, offset, mask)

    # Fallback to the bridge API `dcn_forward` used by the compiled torch
    # extension. It writes into an output 'cols' tensor which we reshape
    # and reduce to produce the final aligned feature.
    if not hasattr(_native_ops, "dcn_forward"):
        raise AttributeError("Bound native_ops object has no DCN entrypoint (expected 'dcn_forward' or 'modulated_deform_conv').")

    b, c, h, w = feat.shape

    x_c = feat.contiguous()
    offset_c = offset.contiguous()
    mask_c = mask.contiguous()

    cols = torch.zeros(b, c * 9, h, w, device=x_c.device, dtype=x_c.dtype)

    # Call the native bridge. Signature mirrors the C++ bridge in
    # kernels/nova_bridge.cpp (see utils.cuda_loader).
    _native_ops.dcn_forward(
        x_c, offset_c, mask_c,
        b, c, h, w,
        3, 3, 1, 1, 1, 1, 1, 1, 1, cols
    )

    assert cols.numel() == b * c * 9 * h * w, "DCN im2col size mismatch"

    output = cols.view(b, c, 9, h, w).mean(dim=2)
    return output
