from engine.pipeline import VideoProcessor
from arch.vsr_backbone import NovaVSRBackbone
from engine.trt import TensorRTNovaBackbone
from pathlib import Path
import torch
from utils.checkpoint import load_model_checkpoint
from utils.gpu_telemetry import describe_gpu_telemetry_backend
from utils.logger import configure_logging, format_bytes, get_logger


LOGGER = get_logger(__name__)

VIDEO_INPUT = "./samples/input.mp4"
VIDEO_OUTPUT = "./output/output.mp4"

USE_TRT = True
CAPTURE_BACKEND = "opencv"

# Prefer FP16 engine — it's the path that hits the 10–15 FPS @ 1080p target.
# FP32 remains as a fallback for debugging / numerical-stability checks only.
TRT_ENGINE_CANDIDATES = (
    "./weights/nova_trt_fp32_480.engine",
    "./weights/nova_trt_fp16_480.engine",
)

WEIGHTS_CANDIDATES = (
    "./weights/realbasicvsr_wogan.pth",
    "./weights/realbasicvsr.pth",
)

WINDOW_SIZE = 7
FRAMES_PER_INFERENCE = 3

ENABLE_TEXTURE_RECOVERY = False
ENABLE_TEXTURE_STABILIZER = False
ENABLE_GPU_TELEMETRY = True

TEXTURE_ALPHA = 0.00

WHITE_BALANCE_STRENGTH = 0.0
EXPOSURE_TARGET = 0.00
INPUT_GAMMA = 1.00
CONTRAST_STRENGTH = 0.00

TEMPORAL_DENOISE_STRENGTH = 0.00
TEMPORAL_ANTI_FLICKER_STRENGTH = 0.00

FINAL_TOUCHUP_STRENGTH = 0.0

TEXTURE_STABILIZER_BLEND = 0.00
TEXTURE_STABILIZER_SIGMA = 1.8

OUTPUT_CODEC = "auto"
OUTPUT_CRF = 22
OUTPUT_PRESET = "slow"
OUTPUT_NVENC_PRESET = "p4"

DEBUG_DUMP_STAGES = False
DEBUG_DUMP_WINDOW_INDEX = 1
PROFILE_PIPELINE = True
PERFORMANCE_LOG_INTERVAL = 5

LOG_LEVEL = "INFO"
LOG_FILE_LEVEL = "DEBUG"
LOG_FILE = "./output/realbasic.log"


def resolve_trt_engine_path():
    for engine_path in TRT_ENGINE_CANDIDATES:
        if Path(engine_path).exists():
            return engine_path
    return TRT_ENGINE_CANDIDATES[-1]


def resolve_weights_path():
    for weights_path in WEIGHTS_CANDIDATES:
        if Path(weights_path).exists():
            return weights_path
    return WEIGHTS_CANDIDATES[-1]


def ensure_engine_is_fresh(engine_path, weights_path):
    engine_file = Path(engine_path)
    if not engine_file.exists():
        raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

    dependency_paths = [
        Path("./arch/vsr_backbone.py"),
        Path("./models/export_onnx.py"),
        Path("./utils/checkpoint.py"),
        Path(weights_path),
    ]
    engine_mtime = engine_file.stat().st_mtime
    stale_inputs = [path for path in dependency_paths if path.exists() and path.stat().st_mtime > engine_mtime]
    if not stale_inputs:
        return

    stale_names = ", ".join(path.name for path in stale_inputs)
    rebuild_hint = [
        f"Engine {engine_file.name} is older than: {stale_names}",
        "Re-export ONNX and rebuild the engine before trusting output quality.",
        "Example:",
        f"  python -m models.export_onnx --weights-path {weights_path} --onnx-path ./weights/nova_480.onnx --T 7 --H 480 --W 854",
    ]
    if "fp16" in engine_file.name.lower():
        rebuild_hint.append(f"  python -m models.build_trt --onnx ./weights/nova_480.onnx --engine {engine_path} --fp16 --force-rebuild --no-cache")
    else:
        rebuild_hint.append(f"  python -m models.build_trt --onnx ./weights/nova_480.onnx --engine {engine_path} --force-rebuild --no-cache")
    raise RuntimeError("\n".join(rebuild_hint))

def build_pytorch_model(device):
    model = NovaVSRBackbone().to(device)

    weights_path = resolve_weights_path()
    LOGGER.info("Loading PyTorch weights: %s", weights_path)
    load_model_checkpoint(model, weights_path, device)
    model.eval()
    return model, None

def build_trt_model(device):
    weights_path = resolve_weights_path()
    engine_path = resolve_trt_engine_path()
    ensure_engine_is_fresh(engine_path, weights_path)
    LOGGER.info("Using weights reference for TensorRT parity: %s", weights_path)
    LOGGER.info("Loading TensorRT engine: %s", engine_path)
    model = TensorRTNovaBackbone(engine_path)
    return model, None


def log_runtime_environment(device):
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        LOGGER.info(
            "CUDA device: name=%s capability=%s.%s vram=%s",
            properties.name,
            properties.major,
            properties.minor,
            format_bytes(properties.total_memory),
        )
    else:
        LOGGER.info("Using device: %s", device)

    LOGGER.info(
        "Run config: input=%s output=%s use_trt=%s capture_backend=%s hw_decode=%s frames_per_inference=%s profile_pipeline=%s perf_interval=%s",
        VIDEO_INPUT,
        VIDEO_OUTPUT,
        USE_TRT,
        CAPTURE_BACKEND,
        FRAMES_PER_INFERENCE,
        PROFILE_PIPELINE,
        PERFORMANCE_LOG_INTERVAL,
    )
    LOGGER.info(
        "Enhancement config: texture_recovery=%s texture_stabilizer=%s blend=%.2f sigma=%.2f anti_flicker=%.2f temporal_denoise=%.2f white_balance=%.2f exposure_target=%.2f input_gamma=%.2f detail_prep=%.2f final_touchup=%.2f",
        ENABLE_TEXTURE_RECOVERY,
        ENABLE_TEXTURE_STABILIZER,
        TEXTURE_STABILIZER_BLEND,
        TEXTURE_STABILIZER_SIGMA,
        TEMPORAL_ANTI_FLICKER_STRENGTH,
        TEMPORAL_DENOISE_STRENGTH,
        WHITE_BALANCE_STRENGTH,
        EXPOSURE_TARGET,
        INPUT_GAMMA,
        CONTRAST_STRENGTH,
        FINAL_TOUCHUP_STRENGTH,
    )
    LOGGER.info("Debug dump config: enabled=%s window_index=%s", DEBUG_DUMP_STAGES, DEBUG_DUMP_WINDOW_INDEX)
    LOGGER.info("GPU telemetry backend: enabled=%s backend=%s", ENABLE_GPU_TELEMETRY, describe_gpu_telemetry_backend())

def run():
    configure_logging(level=LOG_LEVEL, log_file=LOG_FILE, file_level=LOG_FILE_LEVEL)
    LOGGER.info("Logging configured: console_level=%s file=%s file_level=%s", LOG_LEVEL, Path(LOG_FILE).resolve(), LOG_FILE_LEVEL)
    device = torch.device('cuda')
    log_runtime_environment(device)

    if USE_TRT:
        LOGGER.info("Using TensorRT backbone")
        model, native_ops = build_trt_model(device)
    else:
        LOGGER.info("Using PyTorch backbone")
        model, native_ops = build_pytorch_model(device)

    processor = VideoProcessor(
        model=model,
        native_ops=native_ops,
        device=device,
        window_size=7,
        texture_alpha=TEXTURE_ALPHA,
        output_crf=OUTPUT_CRF,
        output_preset=OUTPUT_PRESET,
        output_codec=OUTPUT_CODEC,
        output_nvenc_preset=OUTPUT_NVENC_PRESET,
        capture_backend=CAPTURE_BACKEND,
        profile_performance=PROFILE_PIPELINE,
        performance_log_interval=PERFORMANCE_LOG_INTERVAL,
        frames_per_inference=FRAMES_PER_INFERENCE,
        temporal_denoise_strength=TEMPORAL_DENOISE_STRENGTH,
        white_balance_strength=WHITE_BALANCE_STRENGTH,
        exposure_target=EXPOSURE_TARGET,
        input_gamma=INPUT_GAMMA,
        contrast_strength=CONTRAST_STRENGTH,
        final_touchup_strength=FINAL_TOUCHUP_STRENGTH,
        enable_texture_recovery=ENABLE_TEXTURE_RECOVERY,
        enable_gpu_telemetry=ENABLE_GPU_TELEMETRY,
        enable_texture_stabilizer=ENABLE_TEXTURE_STABILIZER,
        texture_stabilizer_blend=TEXTURE_STABILIZER_BLEND,
        texture_stabilizer_sigma=TEXTURE_STABILIZER_SIGMA,
        temporal_anti_flicker_strength=TEMPORAL_ANTI_FLICKER_STRENGTH,
        dump_debug_frames=DEBUG_DUMP_STAGES,
        debug_dump_window_index=DEBUG_DUMP_WINDOW_INDEX,
    )

    processor.process(
        input_path=VIDEO_INPUT,
        output_path=VIDEO_OUTPUT,
        width=854,
        height=480,
        fps=24
    )

if __name__ == "__main__":
    run()
