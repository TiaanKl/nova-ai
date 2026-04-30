import ffmpeg
import logging
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import subprocess
import time
import torch
import torchvision.transforms.v2.functional as F_vision
from engine.texture_recovery import recover_biological_texture
from engine.tiling import TileHandler
from utils.gpu_telemetry import query_gpu_telemetry
from utils.logger import format_bytes, get_logger
from utils.videocapture import OptimizedVideoCapture


LOGGER = get_logger(__name__)


class RollingFrameBuffer:
    def __init__(self, window_size):
        self.window_size = window_size
        self.storage = None
        self.count = 0

    def push(self, frame):
        if self.storage is None:
            channels, height, width = frame.shape
            self.storage = torch.empty(
                (self.window_size * 2, channels, height, width),
                device=frame.device,
                dtype=frame.dtype,
            )

        slot = self.count % self.window_size
        self.storage[slot].copy_(frame)
        self.storage[slot + self.window_size].copy_(frame)
        self.count += 1

    @property
    def is_full(self):
        return self.count >= self.window_size

    def current_window(self):
        if self.storage is None or self.count == 0:
            raise RuntimeError("Rolling frame buffer is empty")

        if self.count <= self.window_size:
            return self.storage[: self.count]

        start = self.count % self.window_size
        return self.storage[start:start + self.window_size]


_FFMPEG_ENCODERS = None
_FFMPEG_DECODERS = None


_CUVID_DECODER_BY_CODEC = {
    "av1": "av1_cuvid",
    "h264": "h264_cuvid",
    "hevc": "hevc_cuvid",
    "mjpeg": "mjpeg_cuvid",
    "mpeg1video": "mpeg1_cuvid",
    "mpeg2video": "mpeg2_cuvid",
    "mpeg4": "mpeg4_cuvid",
    "vc1": "vc1_cuvid",
    "vp8": "vp8_cuvid",
    "vp9": "vp9_cuvid",
}


def _get_ffmpeg_encoders():
    global _FFMPEG_ENCODERS
    if _FFMPEG_ENCODERS is not None:
        return _FFMPEG_ENCODERS

    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            check=True,
            capture_output=True,
            text=True,
        )
        _FFMPEG_ENCODERS = result.stdout
    except Exception:
        _FFMPEG_ENCODERS = ""

    return _FFMPEG_ENCODERS


def _get_ffmpeg_decoders():
    global _FFMPEG_DECODERS
    if _FFMPEG_DECODERS is not None:
        return _FFMPEG_DECODERS

    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-decoders"],
            check=True,
            capture_output=True,
            text=True,
        )
        _FFMPEG_DECODERS = result.stdout
    except Exception:
        _FFMPEG_DECODERS = ""

    return _FFMPEG_DECODERS


def _probe_video_codec(path):
    try:
        probe = ffmpeg.probe(path)
    except Exception:
        return None

    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            return stream.get("codec_name")

    return None


def _resolve_hw_video_decoder(path):
    codec_name = _probe_video_codec(path)
    if not codec_name:
        return None

    decoder_name = _CUVID_DECODER_BY_CODEC.get(codec_name)
    if not decoder_name:
        return None

    if decoder_name not in _get_ffmpeg_decoders():
        return None

    return decoder_name

class VideoProcessor:
    def __init__(self, model, native_ops, device, window_size=15, texture_alpha=0.24, output_crf=22, output_preset="slow", frames_per_inference=1, output_codec="auto", output_nvenc_preset="p4", use_hw_decode=True, temporal_denoise_strength=0.18, white_balance_strength=0.15, exposure_target=0.46, input_gamma=0.98, contrast_strength=0.12, final_touchup_strength=0.18, texture_stabilizer_blend=0.55, texture_stabilizer_sigma=1.5, temporal_anti_flicker_strength=0.14, capture_backend="opencv", capture_cache_size=50, profile_performance=False, performance_log_interval=5, enable_texture_recovery=True, enable_gpu_telemetry=True, enable_texture_stabilizer=True, dump_debug_frames=False, debug_dump_dir="./output/debug_frames", debug_dump_window_index=1):
        self.model = model
        self.ops = native_ops
        self.window_size = window_size
        self.device = device
        self.texture_alpha = texture_alpha
        self.enable_texture_recovery = bool(enable_texture_recovery and texture_alpha > 0)
        self.enable_gpu_telemetry = bool(enable_gpu_telemetry)
        self.output_crf = output_crf
        self.output_preset = output_preset
        self.frames_per_inference = max(1, min(frames_per_inference, window_size))
        self.output_codec = output_codec
        self.output_nvenc_preset = output_nvenc_preset
        self.use_hw_decode = use_hw_decode
        self.capture_backend = capture_backend
        self.capture_cache_size = capture_cache_size
        self.profile_performance = profile_performance
        self.performance_log_interval = max(1, int(performance_log_interval))
        self.temporal_denoise_strength = temporal_denoise_strength
        self.white_balance_strength = white_balance_strength
        self.exposure_target = exposure_target
        self.input_gamma = input_gamma
        self.contrast_strength = contrast_strength
        self.final_touchup_strength = max(0.0, float(final_touchup_strength))
        self.enable_texture_stabilizer = bool(enable_texture_stabilizer)
        self.texture_stabilizer_blend = max(0.0, min(float(texture_stabilizer_blend), 1.0))
        self.texture_stabilizer_sigma = max(0.1, float(texture_stabilizer_sigma))
        self.temporal_anti_flicker_strength = max(0.0, min(float(temporal_anti_flicker_strength), 1.0))
        self.dump_debug_frames = bool(dump_debug_frames)
        self.debug_dump_dir = Path(debug_dump_dir)
        self.debug_dump_window_index = max(1, int(debug_dump_window_index))
        model_tile_shape = getattr(model, "input_spatial_shape", None)
        tile_size = 256
        if model_tile_shape is not None:
            tile_size = max(int(model_tile_shape[0]), int(model_tile_shape[1]))
        self.tiler = TileHandler(tile_size=tile_size, overlap=32)
        self._logged_inference_mode = False
        self._logged_emit_mode = False
        self._logged_encoder_mode = False
        self._logged_preprocess_mode = False
        self._logged_postprocess_mode = False
        self._logged_stabilizer_mode = False
        self._logged_input_mode = False
        self._logged_runtime_configuration = False
        self._logged_gpu_telemetry_unavailable = False
        self._logged_debug_dump_mode = False
        self._debug_frame_dumped = False
        self._last_perf_interval_total = None
        self._last_bottleneck_warning_signature = None
        self._pinned_hwc_uint8: torch.Tensor | None = None

    def _sync_device(self):
        if isinstance(self.device, torch.device) and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _reset_cuda_peak_memory_stats(self):
        if isinstance(self.device, torch.device) and self.device.type == "cuda":
            try:
                torch.cuda.reset_peak_memory_stats(self.device)
            except Exception:
                pass

    def _get_cuda_memory_stats(self):
        if not isinstance(self.device, torch.device) or self.device.type != "cuda":
            return None

        return {
            "allocated": int(torch.cuda.memory_allocated(self.device)),
            "reserved": int(torch.cuda.memory_reserved(self.device)),
            "max_allocated": int(torch.cuda.max_memory_allocated(self.device)),
            "max_reserved": int(torch.cuda.max_memory_reserved(self.device)),
        }

    def _get_device_index(self):
        if not isinstance(self.device, torch.device) or self.device.type != "cuda":
            return None
        if self.device.index is not None:
            return self.device.index
        return torch.cuda.current_device()

    def _get_gpu_telemetry(self):
        if not self.enable_gpu_telemetry:
            return None

        device_index = self._get_device_index()
        if device_index is None:
            return None

        snapshot = query_gpu_telemetry(device_index)
        if snapshot is None and not self._logged_gpu_telemetry_unavailable:
            LOGGER.warning("GPU telemetry is enabled but nvidia-smi telemetry could not be read")
            self._logged_gpu_telemetry_unavailable = True
        return snapshot

    def _stage_timer_start(self):
        if not self.profile_performance:
            return None
        self._sync_device()
        return time.perf_counter()

    def _stage_timer_end(self, start_time):
        if start_time is None:
            return 0.0
        self._sync_device()
        return time.perf_counter() - start_time

    def _get_emit_range(self):
        emit_start = max(0, (self.window_size - self.frames_per_inference) // 2)
        emit_end = emit_start + self.frames_per_inference
        return emit_start, emit_end

    def _can_run_full_frame_shape(self, shape):
        can_accept_input_shape = getattr(self.model, "can_accept_input_shape", None)
        if not callable(can_accept_input_shape):
            return False
        return can_accept_input_shape(tuple(int(dim) for dim in shape))

    def _log_runtime_configuration(self, input_path, requested_width, requested_height, requested_fps, actual_width, actual_height, actual_fps, output_path):
        if self._logged_runtime_configuration:
            return

        input_shape = (1, self.window_size, 3, int(actual_height), int(actual_width))
        LOGGER.info(
            "Stream config: input=%s requested=%sx%s@%.3f actual=%sx%s@%.3f output=%s window=%s emit=%s texture_recovery=%s texture_stabilizer=%s blend=%.2f anti_flicker=%.2f final_touchup=%.3f debug_dump=%s gpu_telemetry=%s alpha=%.3f",
            input_path,
            requested_width,
            requested_height,
            requested_fps,
            actual_width,
            actual_height,
            actual_fps,
            output_path,
            self.window_size,
            self.frames_per_inference,
            self.enable_texture_recovery,
            self.enable_texture_stabilizer,
            self.texture_stabilizer_blend,
            self.temporal_anti_flicker_strength,
            self.final_touchup_strength,
            self.dump_debug_frames,
            self.enable_gpu_telemetry,
            self.texture_alpha,
        )
        LOGGER.info(
            "Model profile: query_shape=%s full_frame=%s min=%s opt=%s max=%s",
            input_shape,
            self._can_run_full_frame_shape(input_shape),
            getattr(self.model, "min_input_shape", None),
            getattr(self.model, "opt_input_shape", None),
            getattr(self.model, "max_input_shape", None),
        )
        self._logged_runtime_configuration = True

    def _log_window_debug(self, window_index, input_tensor, upscaled, decode_time, prepare_time, model_time, post_time, write_time, emitted_frames):
        if not LOGGER.isEnabledFor(logging.DEBUG):
            return

        stage_times = {
            "decode": decode_time,
            "prepare": prepare_time,
            "trt": model_time,
            "post": post_time,
            "write": write_time,
        }
        bottleneck_stage = max(stage_times.items(), key=lambda item: item[1])[0]
        memory_stats = self._get_cuda_memory_stats()
        memory_suffix = ""
        if memory_stats is not None:
            memory_suffix = (
                f" gpu_alloc={format_bytes(memory_stats['allocated'])}"
                f" gpu_reserved={format_bytes(memory_stats['reserved'])}"
                f" gpu_peak={format_bytes(memory_stats['max_allocated'])}"
            )

        LOGGER.debug(
            "Window %s timings: input=%s output=%s decode=%.3fs prepare=%.3fs trt=%.3fs post=%.3fs write=%.3fs emitted=%s bottleneck=%s%s",
            window_index,
            tuple(int(dim) for dim in input_tensor.shape),
            tuple(int(dim) for dim in upscaled.shape),
            decode_time,
            prepare_time,
            model_time,
            post_time,
            write_time,
            emitted_frames,
            bottleneck_stage,
            memory_suffix,
        )

    def _log_perf_interval(self, perf_stats):
        total_time = (
            perf_stats["decode"]
            + perf_stats["prepare"]
            + perf_stats["model"]
            + perf_stats["post"]
            + perf_stats["write"]
        )
        avg_total = total_time / perf_stats["windows"]
        avg_stage_times = {
            "decode": perf_stats["decode"] / perf_stats["windows"],
            "prepare": perf_stats["prepare"] / perf_stats["windows"],
            "trt": perf_stats["model"] / perf_stats["windows"],
            "post": perf_stats["post"] / perf_stats["windows"],
            "write": perf_stats["write"] / perf_stats["windows"],
        }
        effective_fps = perf_stats["emitted_frames"] / total_time if total_time > 0 else 0.0
        bottleneck_stage = max(avg_stage_times.items(), key=lambda item: item[1])[0]
        bottleneck_share = (avg_stage_times[bottleneck_stage] / avg_total) * 100.0 if avg_total > 0 else 0.0

        memory_stats = self._get_cuda_memory_stats()
        telemetry_snapshot = self._get_gpu_telemetry()
        memory_suffix = ""
        if memory_stats is not None:
            memory_suffix = (
                f" gpu_alloc={format_bytes(memory_stats['allocated'])}"
                f" gpu_reserved={format_bytes(memory_stats['reserved'])}"
                f" gpu_peak={format_bytes(memory_stats['max_allocated'])}"
            )
        telemetry_suffix = "" if telemetry_snapshot is None else telemetry_snapshot.format_inline()

        LOGGER.info(
            "Perf avg windows=%s decode=%.3fs prepare=%.3fs trt=%.3fs post=%.3fs write=%.3fs total=%.3fs out_fps=%.2f bottleneck=%s(%.0f%%)%s%s",
            perf_stats["windows"],
            avg_stage_times["decode"],
            avg_stage_times["prepare"],
            avg_stage_times["trt"],
            avg_stage_times["post"],
            avg_stage_times["write"],
            avg_total,
            effective_fps,
            bottleneck_stage,
            bottleneck_share,
            memory_suffix,
            telemetry_suffix,
        )

        if perf_stats["artifact_samples"] > 0:
            LOGGER.info(
                "Artifact avg windows=%s prep_detail=%.4f prep_dark_edge=%.4f dark_halo=%.4f bright_halo=%.4f edge_dark=%.2f%% touchup_lift=%.4f stabilize_mean=%.4f stabilize_max=%.4f anti_flicker_mean=%.4f anti_flicker_max=%.4f",
                perf_stats["windows"],
                perf_stats["prep_detail_lift"] / perf_stats["windows"],
                perf_stats["prep_dark_edge"] / perf_stats["windows"],
                perf_stats["dark_halo"] / perf_stats["artifact_samples"],
                perf_stats["bright_halo"] / perf_stats["artifact_samples"],
                (perf_stats["edge_dark_ratio"] / perf_stats["artifact_samples"]) * 100.0,
                perf_stats["touchup_lift"] / perf_stats["artifact_samples"],
                perf_stats["stabilize_blend_mean"] / perf_stats["artifact_samples"],
                perf_stats["stabilize_blend_max"] / perf_stats["artifact_samples"],
                perf_stats["anti_flicker_blend_mean"] / perf_stats["artifact_samples"],
                perf_stats["anti_flicker_blend_max"] / perf_stats["artifact_samples"],
            )

        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(
                "Perf shares: decode=%.0f%% prepare=%.0f%% trt=%.0f%% post=%.0f%% write=%.0f%%",
                (avg_stage_times["decode"] / avg_total) * 100.0 if avg_total > 0 else 0.0,
                (avg_stage_times["prepare"] / avg_total) * 100.0 if avg_total > 0 else 0.0,
                (avg_stage_times["trt"] / avg_total) * 100.0 if avg_total > 0 else 0.0,
                (avg_stage_times["post"] / avg_total) * 100.0 if avg_total > 0 else 0.0,
                (avg_stage_times["write"] / avg_total) * 100.0 if avg_total > 0 else 0.0,
            )

        if self._last_perf_interval_total is not None and avg_total > self._last_perf_interval_total * 1.25:
            LOGGER.warning(
                "Perf drift detected: total_window_time %.3fs -> %.3fs (+%.0f%%) bottleneck=%s",
                self._last_perf_interval_total,
                avg_total,
                ((avg_total - self._last_perf_interval_total) / self._last_perf_interval_total) * 100.0,
                bottleneck_stage,
            )

        bottleneck_signature = (bottleneck_stage, int(bottleneck_share // 5))
        if bottleneck_share >= 60.0 and bottleneck_signature != self._last_bottleneck_warning_signature:
            LOGGER.warning(
                "Dominant bottleneck: stage=%s avg=%.3fs share=%.0f%%",
                bottleneck_stage,
                avg_stage_times[bottleneck_stage],
                bottleneck_share,
            )
            self._last_bottleneck_warning_signature = bottleneck_signature

        self._last_perf_interval_total = avg_total

    def _log_process_summary(self, input_path, output_path, decoded_frames, processed_windows, emitted_frames, elapsed_time):
        average_output_fps = emitted_frames / elapsed_time if elapsed_time > 0 else 0.0
        LOGGER.info(
            "Run summary: input=%s output=%s decoded_frames=%s windows=%s emitted_frames=%s elapsed=%.2fs avg_out_fps=%.2f",
            input_path,
            output_path,
            decoded_frames,
            processed_windows,
            emitted_frames,
            elapsed_time,
            average_output_fps,
        )

        memory_stats = self._get_cuda_memory_stats()
        telemetry_snapshot = self._get_gpu_telemetry()
        if memory_stats is not None:
            LOGGER.info(
                "CUDA memory summary: alloc=%s reserved=%s peak_alloc=%s peak_reserved=%s",
                format_bytes(memory_stats["allocated"]),
                format_bytes(memory_stats["reserved"]),
                format_bytes(memory_stats["max_allocated"]),
                format_bytes(memory_stats["max_reserved"]),
            )
        if telemetry_snapshot is not None:
            LOGGER.info("GPU telemetry summary: name=%s%s", telemetry_snapshot.name, telemetry_snapshot.format_inline())

    def _to_luma(self, image):
        weights = image.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
        return (image * weights).sum(dim=1, keepdim=True)

    def _rescale_to_luma(self, image, target_luma, min_gain=0.0, max_gain=2.0):
        current_luma = self._to_luma(image)
        gain = (target_luma / current_luma.clamp_min(1e-4)).clamp(min_gain, max_gain)
        return (image * gain).clamp(0.0, 1.0)

    def _upsample_reference(self, lr_frame, size):
        return torch.nn.functional.interpolate(lr_frame, size=size, mode="bicubic", align_corners=False).clamp(0.0, 1.0)

    def _build_texture_test_base(self, lr_frame, size, reference=None):
        if reference is not None:
            base = reference
        else:
            base = torch.nn.functional.interpolate(lr_frame, size=size, mode="bicubic", align_corners=False).clamp(0.0, 1.0)
        return F_vision.gaussian_blur(base, kernel_size=[5, 5], sigma=[self.texture_stabilizer_sigma, self.texture_stabilizer_sigma]).clamp(0.0, 1.0)

    def _tensor_to_rgb8(self, frame):
        if frame.dim() == 4:
            frame = frame.squeeze(0)
        return (
            frame.detach()
            .permute(1, 2, 0)
            .clamp(0, 1)
            .mul(255)
            .byte()
            .cpu()
            .numpy()
        )

    def _save_debug_frame_strip(self, processed_window_index, frame_index, source_lr, prepared_lr, raw_sr, stabilized_sr, refined_sr, final_frame):
        if self._debug_frame_dumped or not self.dump_debug_frames:
            return

        if processed_window_index != self.debug_dump_window_index:
            return

        if not self._logged_debug_dump_mode:
            LOGGER.info("Saving debug stage dumps for window=%s into %s", self.debug_dump_window_index, self.debug_dump_dir)
            self._logged_debug_dump_mode = True

        self.debug_dump_dir.mkdir(parents=True, exist_ok=True)

        size = raw_sr.shape[-2:]
        stages = [
            ("lr_source", self._upsample_reference(source_lr, size)),
            ("lr_prepared", self._upsample_reference(prepared_lr, size)),
            ("sr_raw", raw_sr),
            ("sr_stabilized", stabilized_sr),
            ("sr_refined", refined_sr),
            ("sr_final", final_frame),
        ]

        arrays = [(label, self._tensor_to_rgb8(tensor)) for label, tensor in stages]
        separator = 8
        label_band = 28
        panel_height, panel_width = arrays[0][1].shape[:2]
        strip_width = len(arrays) * panel_width + (len(arrays) - 1) * separator
        strip_height = panel_height + label_band
        strip = Image.new("RGB", (strip_width, strip_height), color=(12, 12, 12))
        draw = ImageDraw.Draw(strip)

        x = 0
        for label, array in arrays:
            image = Image.fromarray(array, mode="RGB")
            strip.paste(image, (x, label_band))
            draw.text((x + 8, 6), label, fill=(235, 235, 235))
            image.save(self.debug_dump_dir / f"w{processed_window_index:03d}_f{frame_index:02d}_{label}.png")
            x += panel_width + separator

        strip_path = self.debug_dump_dir / f"w{processed_window_index:03d}_f{frame_index:02d}_comparison_strip.png"
        strip.save(strip_path)
        LOGGER.info("Saved debug comparison strip: %s", strip_path)
        self._debug_frame_dumped = True

    def _box_blur(self, image, kernel_size):
        padding = kernel_size // 2
        image = torch.nn.functional.pad(image, (padding, padding, padding, padding), mode="reflect")
        return torch.nn.functional.avg_pool2d(image, kernel_size=kernel_size, stride=1)

    def _temporal_denoise_window(self, window):
        if self.temporal_denoise_strength <= 0 or window.shape[0] < 3:
            return window

        temporal_mean = window.mean(dim=0, keepdim=True)
        deviation = (window - temporal_mean).abs().mean(dim=1, keepdim=True)
        luma = self._to_luma(window)
        edge_mask = ((luma - self._box_blur(luma, kernel_size=3)).abs() * 24.0).clamp(0.0, 1.0)
        flat_mask = (1.0 - edge_mask * 1.2).clamp(0.0, 1.0)
        midtone_mask = (1.0 - (luma - 0.58).abs() / 0.24).clamp(0.0, 1.0)
        stable_mask = (1.0 - deviation / 0.08).clamp(0.0, 1.0)
        denoise_weight = self.temporal_denoise_strength * stable_mask * (0.35 + 0.65 * flat_mask * midtone_mask)
        return torch.lerp(window, temporal_mean.expand_as(window), denoise_weight)

    def _color_correct_window(self, window):
        corrected = window

        if self.white_balance_strength > 0:
            channel_means = corrected.mean(dim=(0, 2, 3), keepdim=True).clamp_min(1e-4)
            neutral_mean = channel_means.mean(dim=1, keepdim=True)
            wb_gain = (neutral_mean / channel_means).clamp(0.85, 1.18)
            corrected = (corrected * torch.pow(wb_gain, self.white_balance_strength)).clamp(0.0, 1.0)

        if self.exposure_target > 0:
            mean_luma = self._to_luma(corrected).mean().clamp_min(1e-4)
            exposure_scale = (corrected.new_tensor(self.exposure_target) / mean_luma).clamp(0.90, 1.12)
            corrected = (corrected * exposure_scale).clamp(0.0, 1.0)

        if self.input_gamma != 1.0:
            corrected = corrected.clamp(0.0, 1.0).pow(self.input_gamma)

        return corrected

    def _enhance_contrast(self, window):
        if self.contrast_strength <= 0:
            return window

        luma = self._to_luma(window)
        local_base = self._box_blur(luma, kernel_size=7)
        local_detail = luma - local_base
        positive_detail = local_detail.clamp(min=0.0)
        detail_energy = self._box_blur(local_detail.abs(), kernel_size=5)
        flat_mask = (1.0 - detail_energy * 18.0).clamp(0.0, 1.0)
        midtone_mask = (1.0 - (luma - 0.55).abs() / 0.35).clamp(0.0, 1.0)
        detail_boost = self.contrast_strength * positive_detail * (0.35 + 0.65 * flat_mask) * midtone_mask
        target_luma = (luma + detail_boost).clamp(0.0, 1.0)
        return self._rescale_to_luma(window, target_luma, min_gain=1.0, max_gain=1.08)

    def _final_touchup_frame(self, frame, lr_frame):
        if self.final_touchup_strength <= 0:
            return frame

        reference = self._upsample_reference(lr_frame, frame.shape[-2:])
        frame_luma = self._to_luma(frame)
        reference_luma = self._to_luma(reference)
        edge_response = (reference_luma - self._box_blur(reference_luma, kernel_size=3)).abs()
        edge_mask = (edge_response * 28.0).clamp(0.0, 1.0)
        dark_halo = (reference_luma - frame_luma - 0.008).clamp(min=0.0)
        target_luma = (frame_luma + self.final_touchup_strength * dark_halo * edge_mask).clamp(0.0, 1.0)
        return self._rescale_to_luma(frame, target_luma, min_gain=1.0, max_gain=1.06)

    def _compute_preprocess_metrics(self, source_frame, prepared_frame):
        source_luma = self._to_luma(source_frame)
        prepared_luma = self._to_luma(prepared_frame)
        edge_mask = ((source_luma - self._box_blur(source_luma, kernel_size=3)).abs() * 24.0).clamp(0.0, 1.0)
        flat_mask = (1.0 - edge_mask).clamp(0.0, 1.0)
        return {
            "prep_detail_lift": float(((prepared_luma - source_luma).clamp(min=0.0) * flat_mask).mean().item()),
            "prep_dark_edge": float(((source_luma - prepared_luma).clamp(min=0.0) * edge_mask).mean().item()),
        }

    def _compute_output_artifact_metrics(self, refined_frame, final_frame, lr_frame):
        reference = self._upsample_reference(lr_frame, final_frame.shape[-2:])
        reference_luma = self._to_luma(reference)
        refined_luma = self._to_luma(refined_frame)
        final_luma = self._to_luma(final_frame)
        edge_mask = ((reference_luma - self._box_blur(reference_luma, kernel_size=3)).abs() * 24.0).clamp(0.0, 1.0)
        edge_dark_ratio = (((reference_luma - final_luma) > 0.035) & (edge_mask > 0.20)).float().mean().item()
        return {
            "dark_halo": float(((reference_luma - final_luma).clamp(min=0.0) * edge_mask).mean().item()),
            "bright_halo": float(((final_luma - reference_luma).clamp(min=0.0) * edge_mask).mean().item()),
            "edge_dark_ratio": float(edge_dark_ratio),
            "touchup_lift": float(((final_luma - refined_luma).clamp(min=0.0) * edge_mask).mean().item()),
        }

    def _stabilize_sr_frame(self, sr_frame, lr_frame, lr_reference=None):
        if not self.enable_texture_stabilizer or self.texture_stabilizer_blend <= 0:
            return sr_frame, {
                "stabilize_blend_mean": sr_frame.new_zeros((sr_frame.shape[0],)),
                "stabilize_blend_max": sr_frame.new_zeros((sr_frame.shape[0],)),
            }

        if not self._logged_stabilizer_mode:
            LOGGER.info("Applying test_texture-style SR stabilizer before recovery (blend=%.2f sigma=%.2f)", self.texture_stabilizer_blend, self.texture_stabilizer_sigma)
            self._logged_stabilizer_mode = True

        sr_luma = self._to_luma(sr_frame)
        sr_luma_self_blur = self._box_blur(sr_luma, kernel_size=5)
        sr_local_deviation = (sr_luma - sr_luma_self_blur).abs()

        sr_edge = ((sr_luma - self._box_blur(sr_luma, kernel_size=3)).abs() * 28.0).clamp(0.0, 1.0)
        flat_mask = (1.0 - sr_edge * 1.3).clamp(0.0, 1.0)

        instability_mask = ((sr_local_deviation - 0.04) * 14.0).clamp(0.0, 1.0)
        blend_weight = (self.texture_stabilizer_blend * flat_mask * instability_mask).clamp(0.0, self.texture_stabilizer_blend)

        smooth_base = self._build_texture_test_base(lr_frame, sr_frame.shape[-2:], reference=lr_reference)
        stabilized = torch.lerp(sr_frame, smooth_base, blend_weight)
        return stabilized, {
            "stabilize_blend_mean": blend_weight.mean(dim=(1, 2, 3)),
            "stabilize_blend_max": blend_weight.amax(dim=(1, 2, 3)),
        }

    def _reduce_temporal_flicker(self, frame_batch, lr_batch):
        if self.temporal_anti_flicker_strength <= 0 or frame_batch.shape[0] < 2:
            return frame_batch, {
                "anti_flicker_blend_mean": frame_batch.new_zeros((frame_batch.shape[0],)),
                "anti_flicker_blend_max": frame_batch.new_zeros((frame_batch.shape[0],)),
            }

        reference_luma = self._to_luma(self._upsample_reference(lr_batch, frame_batch.shape[-2:]))
        edge_mask = ((reference_luma - self._box_blur(reference_luma, kernel_size=3)).abs() * 24.0).clamp(0.0, 1.0)
        flat_mask = (1.0 - edge_mask * 1.4).clamp(0.0, 1.0)
        midtone_mask = (1.0 - (reference_luma - 0.58).abs() / 0.22).clamp(0.0, 1.0)

        prev_reference_luma = torch.cat((reference_luma[:1], reference_luma[:-1]), dim=0)
        next_reference_luma = torch.cat((reference_luma[1:], reference_luma[-1:]), dim=0)
        current_luma = self._to_luma(frame_batch)
        prev_luma = torch.cat((current_luma[:1], current_luma[:-1]), dim=0)
        next_luma = torch.cat((current_luma[1:], current_luma[-1:]), dim=0)

        prev_stability = (1.0 - (reference_luma - prev_reference_luma).abs() / 0.025).clamp(0.0, 1.0)
        next_stability = (1.0 - (reference_luma - next_reference_luma).abs() / 0.025).clamp(0.0, 1.0)
        stability_mask = torch.maximum(prev_stability, next_stability)

        target_luma = (current_luma + prev_luma * prev_stability + next_luma * next_stability) / (1.0 + prev_stability + next_stability).clamp_min(1e-6)
        shimmer_mask = ((current_luma - target_luma).abs() - 0.003).mul(72.0).clamp(0.0, 1.0)
        temporal_weight = (
            self.temporal_anti_flicker_strength
            * flat_mask
            * midtone_mask
            * stability_mask
            * shimmer_mask
        ).clamp(0.0, self.temporal_anti_flicker_strength)

        smoothed_target_luma = torch.lerp(current_luma, target_luma, temporal_weight)
        smoothed = self._rescale_to_luma(frame_batch, smoothed_target_luma, min_gain=0.97, max_gain=1.03)
        return smoothed, {
            "anti_flicker_blend_mean": temporal_weight.mean(dim=(1, 2, 3)),
            "anti_flicker_blend_max": temporal_weight.amax(dim=(1, 2, 3)),
        }

    def _prepare_window(self, window):
        prepared = window.clamp(0.0, 1.0)

        if not self._logged_preprocess_mode:
            LOGGER.info("Applying temporal denoise, color correction, and detail preparation before SR")
            self._logged_preprocess_mode = True

        prepared = self._temporal_denoise_window(prepared)
        prepared = self._color_correct_window(prepared)
        prepared = self._enhance_contrast(prepared)
        return prepared

    def _can_run_full_frame(self, input_tensor):
        return self._can_run_full_frame_shape(tuple(int(dim) for dim in input_tensor.shape))

    def _run_model(self, input_tensor):
        if self._can_run_full_frame(input_tensor):
            if not self._logged_inference_mode:
                LOGGER.info("Using direct full-frame inference")
                self._logged_inference_mode = True
            return self.model(input_tensor)

        if not self._logged_inference_mode:
            LOGGER.info("Using tiled inference")
            self._logged_inference_mode = True

        LOGGER.debug(">>> input_tensor shape: %s", input_tensor.shape)
        return self.tiler.process_tiles(input_tensor, self.model)

    def _refine_frame(self, sr_frame, lr_frame, lr_reference=None):
        if not self.enable_texture_recovery:
            return sr_frame

        return recover_biological_texture(sr_frame, lr_frame, alpha=self.texture_alpha, lr_resized=lr_reference)

    def _postprocess_frames(self, sr_frame, lr_frame):
        if not self._logged_postprocess_mode:
            active_steps = []
            if self.enable_texture_stabilizer and self.texture_stabilizer_blend > 0:
                active_steps.append("stabilizer")
            if self.enable_texture_recovery:
                active_steps.append("texture_recovery")
            if self.temporal_anti_flicker_strength > 0:
                active_steps.append("anti_flicker")
            if self.final_touchup_strength > 0:
                active_steps.append("final_touchup")

            if active_steps:
                LOGGER.info("Applying postprocess pipeline after SR: %s", ", ".join(active_steps))
            else:
                LOGGER.info("Postprocess pipeline disabled; using raw SR output")
            self._logged_postprocess_mode = True

        lr_reference = None
        needs_reference = (
            (self.enable_texture_stabilizer and self.texture_stabilizer_blend > 0)
            or self.enable_texture_recovery
        )
        if needs_reference:
            lr_reference = self._upsample_reference(lr_frame, sr_frame.shape[-2:])

        stabilized, stabilizer_metrics = self._stabilize_sr_frame(sr_frame, lr_frame, lr_reference=lr_reference)
        refined = self._refine_frame(stabilized, lr_frame, lr_reference=lr_reference)
        flicker_reduced, anti_flicker_metrics = self._reduce_temporal_flicker(refined, lr_frame)
        final_frame = self._final_touchup_frame(flicker_reduced, lr_frame)
        return stabilized, refined, final_frame, {**stabilizer_metrics, **anti_flicker_metrics}

    def _get_ffmpeg_input(self, path, width, height):
        """Replaces FrameReader.cs - Pipes raw RGB bytes from FFmpeg."""
        input_kwargs = {}
        input_mode = "software"

        if self.use_hw_decode:
            hw_decoder = _resolve_hw_video_decoder(path)
            if hw_decoder is not None:
                input_kwargs = {
                    "hwaccel": "cuda",
                    "hwaccel_output_format": "cuda",
                    "vcodec": hw_decoder,
                }
                input_mode = hw_decoder
            else:
                LOGGER.warning("Hardware decode requested but unavailable for %s; falling back to software decode", path)

        if not self._logged_input_mode:
            LOGGER.info("Using input decoder: %s", input_mode)
            self._logged_input_mode = True

        input_stream = ffmpeg.input(path, **input_kwargs)
        if input_mode != "software":
            # Decode on GPU, then download into a concrete software pixel format
            # before piping raw RGB frames back to Python.
            input_stream = input_stream.filter("hwdownload").filter("format", "nv12")

        return (
            input_stream
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run_async(pipe_stdout=True)
        )

    def _get_opencv_input(self, path):
        if not self._logged_input_mode:
            LOGGER.info("Using input decoder: opencv")
            if self.use_hw_decode:
                LOGGER.info("OpenCV capture selected; ffmpeg hardware decode is only used with the ffmpeg input backend")
            self._logged_input_mode = True
        return OptimizedVideoCapture(path, cache_size=self.capture_cache_size)

    def _get_ffmpeg_output(self, path, width, height, fps):
        """Replaces FrameWriter.cs - Pipes tensors back to FFmpeg for encoding."""
        encoders = _get_ffmpeg_encoders()
        use_nvenc = self.output_codec == "h264_nvenc" or (
            self.output_codec == "auto" and "h264_nvenc" in encoders
        )

        if use_nvenc:
            output_kwargs = {
                "vcodec": "h264_nvenc",
                "pix_fmt": "yuv420p",
                "preset": self.output_nvenc_preset,
                "rc": "vbr",
                "cq": self.output_crf,
                "b:v": "0",
                "movflags": "+faststart",
            }
            encoder_name = "h264_nvenc"
        else:
            output_kwargs = {
                "vcodec": "libx264",
                "pix_fmt": "yuv420p",
                "preset": self.output_preset,
                "crf": self.output_crf,
                "movflags": "+faststart",
            }
            encoder_name = "libx264"

        if not self._logged_encoder_mode:
            LOGGER.info("Using output encoder: %s target=%s size=%sx%s fps=%.3f", encoder_name, path, width, height, fps)
            self._logged_encoder_mode = True

        return (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}', framerate=fps)
            .output(path, **output_kwargs)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

    def process(self, input_path, output_path, width, height, fps):
        in_pipe = None
        capture = None
        actual_width = width
        actual_height = height
        actual_fps = fps
        decoded_frames = 0
        processed_windows = 0
        emitted_frames_total = 0
        process_start_time = time.perf_counter()
        self._last_perf_interval_total = None
        self._last_bottleneck_warning_signature = None
        self._reset_cuda_peak_memory_stats()

        if self.capture_backend == "opencv":
            capture = self._get_opencv_input(input_path)
            actual_width = capture.width
            actual_height = capture.height
            if capture.fps > 0:
                actual_fps = capture.fps
        else:
            in_pipe = self._get_ffmpeg_input(input_path, width, height)

        self._log_runtime_configuration(input_path, width, height, fps, actual_width, actual_height, actual_fps, output_path)

        out_pipe = self._get_ffmpeg_output(output_path, actual_width * 4, actual_height * 4, actual_fps)

        frame_buffer = RollingFrameBuffer(self.window_size)
        last_inference_frame_count = None
        pending_decode_time = 0.0
        perf_stats = {
            "decode": 0.0,
            "prepare": 0.0,
            "model": 0.0,
            "post": 0.0,
            "write": 0.0,
            "prep_detail_lift": 0.0,
            "prep_dark_edge": 0.0,
            "dark_halo": 0.0,
            "bright_halo": 0.0,
            "edge_dark_ratio": 0.0,
            "touchup_lift": 0.0,
            "stabilize_blend_mean": 0.0,
            "stabilize_blend_max": 0.0,
            "anti_flicker_blend_mean": 0.0,
            "anti_flicker_blend_max": 0.0,
            "artifact_samples": 0,
            "windows": 0,
            "emitted_frames": 0,
        }

        LOGGER.info("Processing %s...", input_path)

        try:
            while True:
                decode_start = self._stage_timer_start()
                # Fast decode path:
                # - Persistent pinned host buffer enables async H2D (vs. unpinned, which serializes).
                # - All channel-flip/permute/cast/normalize happens GPU-side, in parallel with the
                #   next host-side decode, instead of through 3-4 numpy→torch→permute→/255 ops.
                if capture is not None:
                    frame_bgr = capture.read()
                    if frame_bgr is None:
                        break
                    h, w = frame_bgr.shape[0], frame_bgr.shape[1]
                    if self._pinned_hwc_uint8 is None or tuple(self._pinned_hwc_uint8.shape) != (h, w, 3):
                        self._pinned_hwc_uint8 = torch.empty((h, w, 3), dtype=torch.uint8, pin_memory=True)
                    np.copyto(self._pinned_hwc_uint8.numpy(), frame_bgr)
                    device_hwc = self._pinned_hwc_uint8.to(self.device, non_blocking=True)
                    # BGR HWC uint8 → RGB CHW fp32 [0,1]; channel reverse on GPU via flip(-1).
                    frame_t = device_hwc.flip(-1).permute(2, 0, 1).to(torch.float32).mul_(1.0 / 255.0)
                else:
                    assert in_pipe is not None
                    in_bytes = in_pipe.stdout.read(actual_width * actual_height * 3)
                    if not in_bytes:
                        break
                    if self._pinned_hwc_uint8 is None or tuple(self._pinned_hwc_uint8.shape) != (actual_height, actual_width, 3):
                        self._pinned_hwc_uint8 = torch.empty((actual_height, actual_width, 3), dtype=torch.uint8, pin_memory=True)
                    # ffmpeg pix_fmt=rgb24 → already RGB HWC uint8, no channel flip.
                    np.copyto(
                        self._pinned_hwc_uint8.numpy(),
                        np.frombuffer(in_bytes, np.uint8).reshape(actual_height, actual_width, 3),
                    )
                    device_hwc = self._pinned_hwc_uint8.to(self.device, non_blocking=True)
                    frame_t = device_hwc.permute(2, 0, 1).to(torch.float32).mul_(1.0 / 255.0)
                decoded_frames += 1
                pending_decode_time += self._stage_timer_end(decode_start)
                frame_buffer.push(frame_t)

                if frame_buffer.is_full:
                    if last_inference_frame_count is not None:
                        frames_since_last_inference = frame_buffer.count - last_inference_frame_count
                        if frames_since_last_inference < self.frames_per_inference:
                            continue

                    prepare_start = self._stage_timer_start()
                    window = frame_buffer.current_window()
                    prepared_window = self._prepare_window(window)
                    emit_start, emit_end = self._get_emit_range()
                    input_tensor = prepared_window.unsqueeze(0)
                    prepare_time = self._stage_timer_end(prepare_start)
                    center_index = min(prepared_window.shape[0] - 1, self.window_size // 2)
                    # Skip diagnostic preprocess metrics on the hot path. Each call forces
                    # an .item() sync (GPU→CPU) and runs an extra box_blur on the full window.
                    if self.profile_performance:
                        preprocess_metrics = self._compute_preprocess_metrics(
                            window[center_index].unsqueeze(0),
                            prepared_window[center_index].unsqueeze(0),
                        )
                    else:
                        preprocess_metrics = {"prep_detail_lift": 0.0, "prep_dark_edge": 0.0}

                    model_start = self._stage_timer_start()
                    with torch.no_grad():
                        upscaled = self._run_model(input_tensor)
                    model_time = self._stage_timer_end(model_start)

                    post_time = 0.0
                    write_time = 0.0
                    emitted_frames = emit_end - emit_start
                    processed_windows += 1
                    emitted_frames_total += emitted_frames
                    artifact_metrics = None
                    artifact_frame_idx = emit_start + emitted_frames // 2
                    artifact_batch_index = emitted_frames // 2
                    window_postprocess_metrics = {
                        "stabilize_blend_mean": 0.0,
                        "stabilize_blend_max": 0.0,
                        "anti_flicker_blend_mean": 0.0,
                        "anti_flicker_blend_max": 0.0,
                    }

                    if not self._logged_emit_mode:
                        LOGGER.info("Emitting %s frame(s) per inference", emitted_frames)
                        self._logged_emit_mode = True

                    post_start = self._stage_timer_start()
                    sr_batch = upscaled[:, emit_start:emit_end].squeeze(0)
                    source_lr_batch = window[emit_start:emit_end]
                    lr_batch = prepared_window[emit_start:emit_end]
                    stabilized_batch, refined_batch, final_batch, stabilizer_metrics = self._postprocess_frames(sr_batch, lr_batch)
                    post_time += self._stage_timer_end(post_start)

                    window_postprocess_metrics = {
                        "stabilize_blend_mean": float(stabilizer_metrics["stabilize_blend_mean"].mean().item()),
                        "stabilize_blend_max": float(stabilizer_metrics["stabilize_blend_max"].max().item()),
                        "anti_flicker_blend_mean": float(stabilizer_metrics["anti_flicker_blend_mean"].mean().item()),
                        "anti_flicker_blend_max": float(stabilizer_metrics["anti_flicker_blend_max"].max().item()),
                    }

                    # Diagnostic only — running this every window forces 4 .item() syncs and an
                    # extra full-resolution bicubic upsample. Skip when profiling is off.
                    if self.profile_performance:
                        artifact_metrics = self._compute_output_artifact_metrics(
                            refined_batch[artifact_batch_index].unsqueeze(0),
                            final_batch[artifact_batch_index].unsqueeze(0),
                            lr_batch[artifact_batch_index].unsqueeze(0),
                        )
                    self._save_debug_frame_strip(
                        processed_windows,
                        artifact_frame_idx,
                        source_lr_batch[artifact_batch_index].unsqueeze(0),
                        lr_batch[artifact_batch_index].unsqueeze(0),
                        sr_batch[artifact_batch_index].unsqueeze(0),
                        stabilized_batch[artifact_batch_index].unsqueeze(0),
                        refined_batch[artifact_batch_index].unsqueeze(0),
                        final_batch[artifact_batch_index].unsqueeze(0),
                    )

                    for batch_idx in range(emitted_frames):
                        final_frame = final_batch[batch_idx]
                        out_frame = (
                            final_frame
                                .permute(1, 2, 0)
                                .clamp(0, 1)
                                .mul(255)
                                .byte()
                                .cpu()
                                .numpy()
                        )

                        write_start = self._stage_timer_start()
                        out_pipe.stdin.write(out_frame.tobytes())
                        write_time += self._stage_timer_end(write_start)

                    self._log_window_debug(
                        processed_windows,
                        input_tensor,
                        upscaled,
                        pending_decode_time,
                        prepare_time,
                        model_time,
                        post_time,
                        write_time,
                        emitted_frames,
                    )

                    if self.profile_performance:
                        perf_stats["decode"] += pending_decode_time
                        perf_stats["prepare"] += prepare_time
                        perf_stats["model"] += model_time
                        perf_stats["post"] += post_time
                        perf_stats["write"] += write_time
                        perf_stats["prep_detail_lift"] += preprocess_metrics["prep_detail_lift"]
                        perf_stats["prep_dark_edge"] += preprocess_metrics["prep_dark_edge"]
                        if artifact_metrics is not None:
                            perf_stats["dark_halo"] += artifact_metrics["dark_halo"]
                            perf_stats["bright_halo"] += artifact_metrics["bright_halo"]
                            perf_stats["edge_dark_ratio"] += artifact_metrics["edge_dark_ratio"]
                            perf_stats["touchup_lift"] += artifact_metrics["touchup_lift"]
                            perf_stats["stabilize_blend_mean"] += window_postprocess_metrics["stabilize_blend_mean"]
                            perf_stats["stabilize_blend_max"] += window_postprocess_metrics["stabilize_blend_max"]
                            perf_stats["anti_flicker_blend_mean"] += window_postprocess_metrics["anti_flicker_blend_mean"]
                            perf_stats["anti_flicker_blend_max"] += window_postprocess_metrics["anti_flicker_blend_max"]
                            perf_stats["artifact_samples"] += 1
                        perf_stats["windows"] += 1
                        perf_stats["emitted_frames"] += emitted_frames

                        if perf_stats["windows"] % self.performance_log_interval == 0:
                            self._log_perf_interval(perf_stats)
                            perf_stats = {
                                "decode": 0.0,
                                "prepare": 0.0,
                                "model": 0.0,
                                "post": 0.0,
                                "write": 0.0,
                                "prep_detail_lift": 0.0,
                                "prep_dark_edge": 0.0,
                                "dark_halo": 0.0,
                                "bright_halo": 0.0,
                                "edge_dark_ratio": 0.0,
                                "touchup_lift": 0.0,
                                "stabilize_blend_mean": 0.0,
                                "stabilize_blend_max": 0.0,
                                "anti_flicker_blend_mean": 0.0,
                                "anti_flicker_blend_max": 0.0,
                                "artifact_samples": 0,
                                "windows": 0,
                                "emitted_frames": 0,
                            }

                    pending_decode_time = 0.0

                    last_inference_frame_count = frame_buffer.count
        finally:
            elapsed_time = time.perf_counter() - process_start_time
            if capture is not None:
                capture.close()
            if in_pipe is not None:
                in_pipe.terminate()
            out_pipe.stdin.close()
            out_pipe.wait()
            LOGGER.info("Processing complete.")
            self._log_process_summary(input_path, output_path, decoded_frames, processed_windows, emitted_frames_total, elapsed_time)


def get_video_stream(input_path, width, height):
    process = (
        ffmpeg
        .input(input_path)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run_async(pipe_stdout=True)
    )

    while True:
        in_bytes = process.stdout.read(width * height * 3)
        if not in_bytes:
            break
        frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
        yield torch.from_numpy(frame).to('cuda').permute(2, 0, 1).float() / 255.0