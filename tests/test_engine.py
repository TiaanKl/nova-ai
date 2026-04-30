from pathlib import Path
import os
import numpy as np
import onnxruntime as ort
from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnx
from polygraphy.backend.trt import EngineFromPath, TrtRunner
from polygraphy.comparator import Comparator

from engine.trt import _load_nova_plugin_library

MODEL_PATH = Path(os.environ.get("NOVA_ONNX_MODEL", "./weights/nova_480.onnx"))
NUM_SAMPLES = int(os.environ.get("NOVA_PARITY_SAMPLES", "1"))
MEAN_ABS_MAX = float(os.environ.get("NOVA_PARITY_MEAN_ABS_MAX", "0.0025"))
P99_ABS_MAX = float(os.environ.get("NOVA_PARITY_P99_ABS_MAX", "0.015"))
MAX_ABS_MAX = float(os.environ.get("NOVA_PARITY_MAX_ABS_MAX", "0.30"))

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"ONNX model not found: {MODEL_PATH}")


def resolve_input_shape(model_path: Path) -> tuple[int, ...]:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_shape = session.get_inputs()[0].shape
    if any(not isinstance(dim, int) for dim in input_shape):
        raise ValueError(
            "Parity tests require a fixed-shape ONNX model. "
            f"Resolved input shape was {input_shape!r}; point NOVA_ONNX_MODEL at a fixed export such as ./weights/nova_480.onnx."
        )
    return tuple(input_shape)


def resolve_ort_providers() -> list[str]:
    available = ort.get_available_providers()
    requested = os.environ.get("NOVA_ORT_PROVIDERS")
    preferred = (
        [provider.strip() for provider in requested.split(",") if provider.strip()]
        if requested
        else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    providers = [provider for provider in preferred if provider in available]
    if not providers:
        raise RuntimeError(
            f"None of the requested ONNXRuntime providers are available. Requested={preferred}, available={available}"
        )
    return providers


INPUT_SHAPE = resolve_input_shape(MODEL_PATH)
ORT_PROVIDERS = resolve_ort_providers()
build_onnxrt_session = SessionFromOnnx(str(MODEL_PATH), providers=ORT_PROVIDERS)

engine_path = Path(os.environ.get("NOVA_TRT_ENGINE", "./weights/nova_trt_fp32_480.engine"))
if not engine_path.exists():
    raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

_load_nova_plugin_library()
trt_engine = EngineFromPath(str(engine_path))

runners = [
    OnnxrtRunner(build_onnxrt_session),
    TrtRunner(trt_engine),
]


def iter_feed_dicts(num_samples=NUM_SAMPLES, input_shape=INPUT_SHAPE, seed=0):
    generator = np.random.default_rng(seed)
    for _ in range(num_samples):
        yield {"input": generator.random(input_shape, dtype=np.float32)}


def assert_parity_metrics(run_results) -> None:
    onnx_name, onnx_iterations = run_results[0]
    trt_name, trt_iterations = run_results[1]

    if len(onnx_iterations) != len(trt_iterations):
        raise AssertionError(
            f"Runner iteration count mismatch: {onnx_name}={len(onnx_iterations)} vs {trt_name}={len(trt_iterations)}"
        )

    for iteration_index, (onnx_outputs, trt_outputs) in enumerate(zip(onnx_iterations, trt_iterations), start=1):
        output_names = list(onnx_outputs.keys())
        if output_names != list(trt_outputs.keys()):
            raise AssertionError(
                f"Runner output mismatch for iteration {iteration_index}: {onnx_name}={output_names} vs {trt_name}={list(trt_outputs.keys())}"
            )

        for output_name in output_names:
            onnx_output = np.asarray(onnx_outputs[output_name], dtype=np.float32)
            trt_output = np.asarray(trt_outputs[output_name], dtype=np.float32)

            if onnx_output.shape != trt_output.shape:
                raise AssertionError(
                    f"Output shape mismatch for {output_name}: {onnx_output.shape} vs {trt_output.shape}"
                )

            if not np.isfinite(onnx_output).all() or not np.isfinite(trt_output).all():
                raise AssertionError(f"Non-finite values detected in output '{output_name}'")

            abs_diff = np.abs(onnx_output - trt_output).reshape(-1)
            mean_abs = float(abs_diff.mean())
            p99_abs = float(np.percentile(abs_diff, 99))
            max_abs = float(abs_diff.max())

            print(
                f"Iteration {iteration_index} {output_name}: "
                f"mean_abs={mean_abs:.6f} p99_abs={p99_abs:.6f} max_abs={max_abs:.6f}"
            )

            if mean_abs > MEAN_ABS_MAX or p99_abs > P99_ABS_MAX or max_abs > MAX_ABS_MAX:
                raise AssertionError(
                    f"Parity metrics exceeded limits for '{output_name}': "
                    f"mean_abs={mean_abs:.6f} (max {MEAN_ABS_MAX}), "
                    f"p99_abs={p99_abs:.6f} (max {P99_ABS_MAX}), "
                    f"max_abs={max_abs:.6f} (max {MAX_ABS_MAX})"
                )

run_results = Comparator.run(runners, data_loader=iter_feed_dicts())
assert_parity_metrics(run_results)