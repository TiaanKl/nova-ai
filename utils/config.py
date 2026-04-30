import os

class VideoEnhancerConfig:
    """Configuration for AI Video Enhancer on RTX 4070 Super"""

    INPUT_RESOLUTION = "1080p"

    GPU_DEVICE = 0
    GPU_VRAM_RESERVED = 0.3
    WORKSPACE_SIZE = 3 * 1024**3

    BUFFER_SIZE = 30
    CACHE_SIZE = 40

    BATCH_SIZE = 6
    BATCH_TIMEOUT = 0.05
    DYNAMIC_BATCH = True

    TENSORRT_PRECISION = "FP16"
    MAX_AUX_STREAMS = 5
    TIMING_ITERATIONS = 10

    OPTIMIZATION_PROFILES = [
        (1, 6, 8),
    ]

    NUM_IO_THREADS = 1
    NUM_INFERENCE_THREADS = 1
    NUM_WORKER_THREADS = 4

    PIN_MEMORY = True
    ENABLE_CUDA_GRAPHS = True

    ENABLE_LAYER_FUSION = True
    ENABLE_TENSOR_CORES = True

    INFERENCE_MODE = "async"
    STREAM_SYNCHRONIZATION = "event"

    OUTPUT_CODEC = "h264"
    OUTPUT_BITRATE = "50M"

    @classmethod
    def get_batch_size(cls):
        """Get batch size based on resolution"""
        batch_sizes = {
            "720p": 8,
            "1080p": 6,
            "1440p": 4,
            "4K": 2,
        }
        return batch_sizes.get(cls.INPUT_RESOLUTION, 6)

    @classmethod
    def get_cache_settings(cls):
        """Get cache settings based on resolution"""
        settings = {
            "720p": {"buffer": 40, "cache": 60},
            "1080p": {"buffer": 30, "cache": 40},
            "1440p": {"buffer": 20, "cache": 25},
            "4K": {"buffer": 15, "cache": 20},
        }
        return settings.get(cls.INPUT_RESOLUTION, settings["1080p"])