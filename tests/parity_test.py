import torch
import numpy as np
import os
import time
from pathlib import Path
from arch.vsr_backbone import NovaVSRBackbone
from engine.trt import TensorRTNovaBackbone
from utils.checkpoint import load_model_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = PROJECT_ROOT / "weights" / "realbasicvsr.pth"
ENGINE_DEPENDENCIES = [
    PROJECT_ROOT / "arch" / "vsr_backbone.py",
    PROJECT_ROOT / "models" / "export_onnx.py",
    PROJECT_ROOT / "utils" / "checkpoint.py",
    WEIGHTS_PATH,
]


def ensure_engine_is_fresh(engine_path: Path) -> None:
    if not engine_path.exists():
        raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

    engine_mtime = engine_path.stat().st_mtime
    stale_inputs = [path for path in ENGINE_DEPENDENCIES if path.exists() and path.stat().st_mtime > engine_mtime]
    if not stale_inputs:
        return

    stale_names = ", ".join(path.name for path in stale_inputs)
    rebuild_hint = [
        f"Engine {engine_path.name} is older than: {stale_names}",
        "Re-export ONNX and rebuild the engine before trusting parity numbers.",
        "Example:",
        f"  python -m models.export_onnx --weights-path {WEIGHTS_PATH} --onnx-path ./weights/nova_480.onnx --T 7 --H 480 --W 854",
    ]
    if "fp16" in engine_path.name.lower():
        rebuild_hint.append(f"  python -m models.build_trt --onnx ./weights/nova_480.onnx --engine {engine_path} --fp16 --force-rebuild --no-cache")
    else:
        rebuild_hint.append(f"  python -m models.build_trt --onnx ./weights/nova_480.onnx --engine {engine_path} --force-rebuild --no-cache")
    raise RuntimeError("\n".join(rebuild_hint))


def print_diff_summary(label: str, reference: np.ndarray, candidate: np.ndarray) -> None:
    diff = np.abs(reference - candidate).reshape(-1)
    percentiles = np.percentile(diff, [50, 90, 99, 99.9])
    rmse = float(np.sqrt(np.mean(np.square(reference - candidate))))
    print(
        f"{label} mean={diff.mean():.7f} median={np.median(diff):.7f} rmse={rmse:.7f} "
        f"p90={percentiles[1]:.7f} p99={percentiles[2]:.7f} p99.9={percentiles[3]:.7f} max={diff.max():.7f}"
    )

torch.manual_seed(0)
torch_model = NovaVSRBackbone().cuda().eval()
load_model_checkpoint(torch_model, str(WEIGHTS_PATH), "cuda")

engine_path = Path(os.environ.get("NOVA_TRT_ENGINE", "./weights/nova_trt_fp32_480.engine"))
ensure_engine_is_fresh(engine_path)
print("Using engine:", engine_path)
trt_model = TensorRTNovaBackbone(str(engine_path))

if trt_model.opt_input_shape is None:
    raise RuntimeError("TensorRT engine did not provide a profile for input")
input_shape = tuple(trt_model.opt_input_shape)
print("Using input shape:", input_shape)

x = torch.rand(*input_shape).cuda()

with torch.no_grad():
    print("Running PyTorch reference...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    y_torch_tensor = torch_model(x)
    torch.cuda.synchronize()
    torch_forward_time = time.perf_counter() - start
    print(f"PyTorch forward took {torch_forward_time:.2f}s")

    print("Copying PyTorch output to CPU...")
    start = time.perf_counter()
    y_torch = y_torch_tensor.cpu().numpy()
    torch_copy_time = time.perf_counter() - start
    print(f"PyTorch CPU copy took {torch_copy_time:.2f}s")

    print("Running TensorRT...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    y_trt_tensor = trt_model(x)
    torch.cuda.synchronize()
    trt_forward_time = time.perf_counter() - start
    print(f"TensorRT forward took {trt_forward_time:.2f}s")

    print("Copying TensorRT output to CPU...")
    start = time.perf_counter()
    y_trt = y_trt_tensor.cpu().numpy()
    trt_copy_time = time.perf_counter() - start
    print(f"TensorRT CPU copy took {trt_copy_time:.2f}s")

torch_finite = np.isfinite(y_torch)
trt_finite = np.isfinite(y_trt)
diff = y_torch - y_trt

print("Torch finite:", torch_finite.all(), "nan count:", np.isnan(y_torch).sum(), "inf count:", np.isinf(y_torch).sum())
print("TRT finite:", trt_finite.all(), "nan count:", np.isnan(y_trt).sum(), "inf count:", np.isinf(y_trt).sum())
print("Torch range:", float(y_torch.min()), float(y_torch.max()))
print("TRT range:", float(y_trt.min()), float(y_trt.max()))
print_diff_summary("Raw diff", y_torch, y_trt)
print_diff_summary("Clamped [0,1] diff", np.clip(y_torch, 0.0, 1.0), np.clip(y_trt, 0.0, 1.0))
