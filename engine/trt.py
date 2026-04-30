from __future__ import annotations
import ctypes
from pathlib import Path
import torch
import tensorrt as trt  # type: ignore
from typing import List, Tuple
from utils.logger import get_logger


LOGGER = get_logger(__name__)


# --- Local protocol stubs to satisfy Pylance ---
class TRTLogger:
    WARNING: int
    def __init__(self, severity: int): ...


class TRTRuntime:
    def deserialize_cuda_engine(self, data: bytes): ...


class TRTICudaEngine:
    def create_execution_context(self): ...
    def get_binding_index(self, name: str) -> int: ...
    def get_tensor_profile_shape(self, name: str, profile_index: int): ...
    @property
    def num_bindings(self) -> int: ...
    @property
    def num_io_tensors(self) -> int: ...


class TRTExecutionContext:
    def set_binding_shape(self, index: int, shape: Tuple[int, ...]): ...
    def get_binding_shape(self, index: int) -> Tuple[int, ...]: ...
    def set_input_shape(self, name: str, shape: Tuple[int, ...]): ...
    def get_tensor_shape(self, name: str) -> Tuple[int, ...]: ...
    def set_tensor_address(self, name: str, address: int): ...
    def execute_async_v2(self, bindings: List[int], stream_handle: int): ...
    def execute_async_v3(self, stream_handle: int): ...

def _load_nova_plugin_library() -> None:
    project_root = Path(__file__).resolve().parents[1]
    plugin_base = project_root / "trt_plugins" / "build"
    candidates = [
        plugin_base / "Release" / "nova_dcnv2_plugin.dll",
        plugin_base / "Debug" / "nova_dcnv2_plugin.dll",
        plugin_base / "nova_dcnv2_plugin.dll",
    ]

    plugin_path = next((path for path in candidates if path.exists()), None)
    if plugin_path is None:
        raise FileNotFoundError("Nova TensorRT plugin DLL was not found")

    lib = ctypes.CDLL(str(plugin_path))
    try:
        if hasattr(lib, "pluginInitPlugin"):
            lib.pluginInitPlugin()
    except Exception:
        pass

    LOGGER.info("Loaded TensorRT plugin: %s", plugin_path)


class TensorRTNovaBackbone(torch.nn.Module):
    """
    Clean, type-safe TensorRT engine wrapper for RealBasic++.
    Pylance-friendly while still using the classic TensorRT API.
    """

    def __init__(self, engine_path: str):
        super().__init__()

        self.logger: TRTLogger = trt.Logger(trt.Logger.WARNING)  # type: ignore[attr-defined]

        _load_nova_plugin_library()

        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:  # type: ignore[attr-defined]
            engine = runtime.deserialize_cuda_engine(f.read())  # type: ignore[attr-defined]

        self.engine: TRTICudaEngine = engine  # type: ignore[assignment]
        self.context: TRTExecutionContext = engine.create_execution_context()  # type: ignore[assignment]

        self.uses_tensor_api = hasattr(self.engine, "num_io_tensors") and hasattr(self.context, "set_tensor_address")
        self.min_input_shape: Tuple[int, ...] | None = None
        self.opt_input_shape: Tuple[int, ...] | None = None
        self.max_input_shape: Tuple[int, ...] | None = None
        self.input_spatial_shape: Tuple[int, int] | None = None
        self._static_input_shape: Tuple[int, ...] | None = None
        self._cached_output: torch.Tensor | None = None
        self._cached_input_shape: Tuple[int, ...] | None = None
        self._engine_input_dtype: torch.dtype = torch.float32
        self._engine_output_dtype: torch.dtype = torch.float32

        if self.uses_tensor_api:
            self.input_name = "input"
            self.output_name = "output"
        else:
            self.input_idx = self.engine.get_binding_index("input")
            self.output_idx = self.engine.get_binding_index("output")

        if hasattr(self.engine, "get_tensor_profile_shape"):
            try:
                profile_shapes = self.engine.get_tensor_profile_shape("input", 0)
                if profile_shapes is not None:
                    profile_min, profile_opt, profile_max = profile_shapes
                    self.min_input_shape = tuple(int(dim) for dim in profile_min)
                    self.opt_input_shape = tuple(int(dim) for dim in profile_opt)
                    self.max_input_shape = tuple(int(dim) for dim in profile_max)
                    self.input_spatial_shape = (self.max_input_shape[-2], self.max_input_shape[-1])
                    if self.min_input_shape == self.max_input_shape:
                        # Single-shape profile — engine is fully static, skip per-call shape ops.
                        self._static_input_shape = self.min_input_shape
            except Exception:
                self.min_input_shape = None
                self.opt_input_shape = None
                self.max_input_shape = None
                self.input_spatial_shape = None

        # Detect FP16 I/O bindings (TRT >= 8.5 exposes get_tensor_dtype).
        try:
            if hasattr(self.engine, "get_tensor_dtype"):
                in_dt = self.engine.get_tensor_dtype("input") # type: ignore[attr-defined]
                out_dt = self.engine.get_tensor_dtype("output") # type: ignore[attr-defined]
                self._engine_input_dtype = (
                    torch.float16 if in_dt == trt.DataType.HALF else torch.float32  # type: ignore[attr-defined]
                )
                self._engine_output_dtype = (
                    torch.float16 if out_dt == trt.DataType.HALF else torch.float32  # type: ignore[attr-defined]
                )
        except Exception:
            self._engine_input_dtype = torch.float32
            self._engine_output_dtype = torch.float32

    def can_accept_input_shape(self, shape: Tuple[int, ...]) -> bool:
        if self.min_input_shape is None or self.max_input_shape is None:
            return True

        if len(shape) != len(self.min_input_shape):
            return False

        return all(
            min_dim <= dim <= max_dim
            for dim, min_dim, max_dim in zip(shape, self.min_input_shape, self.max_input_shape)
        )

    def forward(self, lrs: torch.Tensor) -> torch.Tensor:
        assert lrs.is_cuda, "Input tensor must be on CUDA"

        if lrs.dtype != self._engine_input_dtype:
            lrs = lrs.to(self._engine_input_dtype)

        cur_shape = tuple(int(d) for d in lrs.shape)

        shape_changed = self._cached_input_shape != cur_shape
        if shape_changed:
            if self.uses_tensor_api:
                self.context.set_input_shape(self.input_name, cur_shape)
                out_shape = tuple(self.context.get_tensor_shape(self.output_name))
            else:
                self.context.set_binding_shape(self.input_idx, cur_shape)
                out_shape = tuple(self.context.get_binding_shape(self.output_idx))
            self._cached_output = torch.empty(
                out_shape, device=lrs.device, dtype=self._engine_output_dtype
            )
            self._cached_input_shape = cur_shape

        assert self._cached_output is not None
        output = self._cached_output

        stream_handle = torch.cuda.current_stream(device=lrs.device).cuda_stream

        if self.uses_tensor_api:
            self.context.set_tensor_address(self.input_name, lrs.data_ptr())
            self.context.set_tensor_address(self.output_name, output.data_ptr())
            self.context.execute_async_v3(stream_handle)  # type: ignore[attr-defined]
        else:
            bindings: List[int] = [0] * self.engine.num_bindings
            bindings[self.input_idx] = lrs.data_ptr()
            bindings[self.output_idx] = output.data_ptr()
            self.context.execute_async_v2(bindings, stream_handle)  # type: ignore[attr-defined]

        if self._engine_output_dtype != torch.float32:
            return output.to(torch.float32)
        return output
