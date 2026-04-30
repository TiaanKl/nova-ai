import os
import ctypes
from pathlib import Path
from typing import TYPE_CHECKING
import argparse
from typing import Sequence

import tensorrt as trt

if TYPE_CHECKING:
    from tensorrt import (
        Logger, # type: ignore[attr-defined]
        Builder, # type: ignore[attr-defined]
        NetworkDefinitionCreationFlag, # type: ignore[attr-defined]
        OnnxParser, # type: ignore[attr-defined]
        BuilderFlag, # type: ignore[attr-defined]
    )

ONNX_PATH = "./weights/nova_480.onnx"
ENGINE_PATH = "./weights/nova_trt_fp16_480.engine"
FAST_ONNX_PATH = "./weights/nova_240.onnx"
FAST_ENGINE_PATH = "./weights/nova_trt_fp16_fast.engine"
_base = Path(__file__).resolve().parents[1] / "trt_plugins" / "build"
_candidates = [
    _base / "Release" / "nova_dcnv2_plugin.dll",
    _base / "Debug" / "nova_dcnv2_plugin.dll",
    _base / "nova_dcnv2_plugin.dll",
]
PLUGIN_PATH = None
for p in _candidates:
    if p.exists():
        PLUGIN_PATH = p
        break
if PLUGIN_PATH is None:
    PLUGIN_PATH = _candidates[0]


def load_plugin_library(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Plugin library not found: {path}")
    lib = ctypes.CDLL(path)
    try:
        if hasattr(lib, "pluginInitPlugin"):
            lib.pluginInitPlugin()
    except Exception:
        pass
    print("[*] Loaded plugin library:", path)
    return lib


def build_engine(
    onnx_path: str = ONNX_PATH,
    engine_path: str = ENGINE_PATH,
    fast_dev: bool = False,
    workspace_mb: int = 4096,
    use_fp16: bool = True,  # FP16 by default — required for the 10–15 FPS @ 1080p target.
    use_bf16: bool = False,
    use_cache: bool = True,
    force_rebuild: bool = False,
    fixed_shape: Sequence[int] | None = None,
):
    logger = trt.Logger(trt.Logger.INFO)  # type: ignore[attr-defined]

    if fast_dev and onnx_path == ONNX_PATH and os.path.exists(FAST_ONNX_PATH):
        onnx_path = FAST_ONNX_PATH
    if fast_dev and engine_path == ENGINE_PATH:
        engine_path = FAST_ENGINE_PATH

    runtime = trt.Runtime(logger)  # type: ignore[attr-defined]

    if use_cache and not force_rebuild and os.path.exists(engine_path):
        try:
            with open(engine_path, "rb") as f:
                blob = f.read()
            engine = runtime.deserialize_cuda_engine(blob)  # type: ignore[attr-defined]
            if engine:
                print("[*] Loaded cached engine:", engine_path)
                return engine
        except Exception as e:
            print("[i] Failed to load cached engine (will rebuild):", e)

    try:
        load_plugin_library(str(PLUGIN_PATH))
    except FileNotFoundError:
        print("[!] Plugin library not found; continuing without plugin (ONNX nodes will not resolve).")

    trt.init_libnvinfer_plugins(logger, "")  # type: ignore[attr-defined]

    builder = trt.Builder(logger)  # type: ignore[attr-defined]
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)  # type: ignore[attr-defined]
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)  # type: ignore[attr-defined]

    parse_ok = False
    if hasattr(parser, "parse_from_file"):
        parse_ok = parser.parse_from_file(str(Path(onnx_path).resolve()))
    else:
        with open(onnx_path, "rb") as f:
            parse_ok = parser.parse(f.read())

    if not parse_ok:
        print("[!] ONNX parsing failed:")
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        return

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024)  # type: ignore[attr-defined]
    print(f"[i] Workspace size set to {workspace_mb} MB")

    if hasattr(config, "builder_optimization_level"):
        if fast_dev:
            config.builder_optimization_level = 0
            print("[i] Fast-dev: builder optimization level set to 0")
        else:
            # Highest level — exhaustive tactic search. Only affects build time, not runtime.
            config.builder_optimization_level = 5
            print("[i] Builder optimization level set to 5 (production)")

    # Persistent timing cache — reuses kernel-tactic timings across rebuilds.
    timing_cache = None
    cache_path = Path(engine_path).with_suffix(".timing_cache")
    try:
        cache_blob = cache_path.read_bytes() if cache_path.exists() else b""
        timing_cache = config.create_timing_cache(cache_blob)  # type: ignore[attr-defined]
        config.set_timing_cache(timing_cache, ignore_mismatch=False)  # type: ignore[attr-defined]
        if cache_blob:
            print(f"[i] Loaded timing cache: {cache_path} ({len(cache_blob)} bytes)")
    except Exception as e:
        print(f"[i] Timing cache unavailable: {e}")
        timing_cache = None

    if use_bf16:
        if use_bf16 and hasattr(trt.BuilderFlag, "BF16"):   # type: ignore[attr-defined]
            try:
                config.set_flag(trt.BuilderFlag.BF16)   # type: ignore[attr-defined]
                print("[*] Requested BF16")
            except Exception as e:
                print("[!] Failed to enable BF16, falling back:", e)
                if use_fp16 and getattr(builder, "platform_has_fast_fp16", False):
                    config.set_flag(trt.BuilderFlag.FP16)  # type: ignore[attr-defined]
                    print("[*] Falling back to FP16")
                else:
                    print("[*] Falling back to FP32")
        else:
            print("[!] BF16 not exposed by this TensorRT build; using fallback precision")
            if use_fp16 and getattr(builder, "platform_has_fast_fp16", False):
                config.set_flag(trt.BuilderFlag.FP16)  # type: ignore[attr-defined]
                print("[*] Falling back to FP16")
            else:
                print("[*] Falling back to FP32")
    elif use_fp16 and getattr(builder, "platform_has_fast_fp16", False):
        config.set_flag(trt.BuilderFlag.FP16)  # type: ignore[attr-defined]
        # Mark network I/O bindings as FP16 so TRT does not insert FP32↔FP16 cast layers
        # at the entry/exit of the engine. The 4× upscaled output is the largest tensor
        # in the graph; eliminating its epilogue cast saves a full bandwidth-bound pass.
        try:
            network.get_input(0).dtype = trt.DataType.HALF  # type: ignore[attr-defined]
            for i in range(network.num_outputs):
                network.get_output(i).dtype = trt.DataType.HALF  # type: ignore[attr-defined]
            print("[*] Using FP16 with FP16 I/O bindings")
        except Exception as e:
            print(f"[!] Could not set FP16 I/O dtypes ({e}); using FP32 I/O with FP16 internals")
    else:
        print("[*] Using FP32")

    # Dynamic shapes for [N, T, C, H, W]
    input_tensor = network.get_input(0)
    profile = builder.create_optimization_profile()
    static_network_shape = None
    try:
        candidate_shape = tuple(int(dim) for dim in input_tensor.shape)
        if all(dim > 0 for dim in candidate_shape):
            static_network_shape = candidate_shape
    except Exception:
        static_network_shape = None

    if fixed_shape is not None:
        fixed = tuple(int(dim) for dim in fixed_shape)
        min_shape = opt_shape = max_shape = fixed
        print("[i] Using fixed profile", fixed)
    elif static_network_shape is not None:
        min_shape = opt_shape = max_shape = static_network_shape
        print("[i] ONNX input is static; using fixed profile", static_network_shape)
    elif fast_dev:
        # Fixed single-shape profile for fast iteration
        fixed = (1, 7, 3, 240, 426)
        min_shape = opt_shape = max_shape = fixed
        print("[i] Fast-dev: using fixed profile", fixed)
    else:
        min_shape = (1, 1, 3, 240, 426)
        opt_shape = (1, 7, 3, 480, 854)
        max_shape = (1, 15, 3, 720, 1280)
    profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    print("[*] Building TensorRT engine...")

    engine = None
    # Try multiple builder entrypoints for compatibility
    try:
        if hasattr(builder, "build_engine"):
            engine = builder.build_engine(network, config)
        elif hasattr(builder, "build_serialized_network"):
            serialized = builder.build_serialized_network(network, config)
            if serialized:
                engine = runtime.deserialize_cuda_engine(serialized)  # type: ignore[attr-defined]
        elif hasattr(builder, "build_cuda_engine"):
            engine = builder.build_cuda_engine(network)
        else:
            raise AttributeError("No compatible build method found on Builder")
    except Exception as e:
        print("[!] Engine build raised an exception:", e)
        try:
            print("[i] Builder methods:", dir(builder))
        except Exception:
            pass
        return

    if engine is None:
        print("[!] Engine build failed")
        return

    outdir = os.path.dirname(engine_path)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(engine.serialize())

    # Persist timing cache for the next build.
    if timing_cache is not None:
        try:
            cache_bytes = bytes(timing_cache.serialize())  # type: ignore[attr-defined]
            cache_path.write_bytes(cache_bytes)
            print(f"[+] Timing cache saved to: {cache_path} ({len(cache_bytes)} bytes)")
        except Exception as e:
            print(f"[i] Could not persist timing cache: {e}")

    print(f"[+] Engine saved to: {engine_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build or load a TensorRT engine from ONNX")
    ap.add_argument("--fast-dev", action="store_true", help="Use a single fixed profile and smaller workspace for fast iteration")
    ap.add_argument("--workspace-mb", type=int, default=4096, help="Workspace memory limit in MB")
    ap.add_argument("--fp16", action="store_true", help="(Default ON) Build with FP16 internals + FP16 I/O bindings.")
    ap.add_argument("--bf16", action="store_true", help="Opt in to BF16 build. Use BF16 where supported (may reduce NaNs on supported hardware).")
    ap.add_argument("--no-fp16", action="store_true", help="Disable FP16; build a pure FP32 engine.")
    ap.add_argument("--no-cache", action="store_true", help="Do not attempt to load/save cached engine")
    ap.add_argument("--force-rebuild", action="store_true", help="Force rebuild even if cached engine exists")
    ap.add_argument("--onnx", type=str, default=ONNX_PATH, help="Path to input ONNX file")
    ap.add_argument("--engine", type=str, default=ENGINE_PATH, help="Output engine path")
    ap.add_argument("--fixed-shape", type=int, nargs=5, metavar=("N", "T", "C", "H", "W"), help="Build a single fixed profile for the given input shape")
    args = ap.parse_args()
    # FP16 is now the default; --no-fp16 explicitly disables it.
    use_fp16 = (not args.no_fp16) and not args.bf16
    build_engine(
        onnx_path=args.onnx,
        engine_path=args.engine,
        fast_dev=args.fast_dev,
        workspace_mb=args.workspace_mb,
        use_fp16=use_fp16,
        use_bf16=args.bf16,
        use_cache=not args.no_cache,
        force_rebuild=args.force_rebuild,
        fixed_shape=args.fixed_shape,
    )
