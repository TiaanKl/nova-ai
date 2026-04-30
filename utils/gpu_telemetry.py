from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import shutil
import subprocess


_QUERY_FIELDS = (
    "name",
    "utilization.gpu",
    "utilization.memory",
    "clocks.sm",
    "clocks.mem",
    "temperature.gpu",
    "power.draw",
    "power.limit",
)


def _parse_int(value: str) -> int | None:
    value = value.strip()
    if not value or value in {"N/A", "[N/A]"}:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _parse_float(value: str) -> float | None:
    value = value.strip()
    if not value or value in {"N/A", "[N/A]"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


@dataclass(frozen=True)
class GpuTelemetrySnapshot:
    name: str
    gpu_utilization: int | None
    memory_utilization: int | None
    sm_clock_mhz: int | None
    memory_clock_mhz: int | None
    temperature_c: int | None
    power_draw_w: float | None
    power_limit_w: float | None

    def format_inline(self) -> str:
        parts: list[str] = []
        if self.gpu_utilization is not None:
            parts.append(f" gpu_util={self.gpu_utilization}%")
        if self.memory_utilization is not None:
            parts.append(f" mem_util={self.memory_utilization}%")
        if self.sm_clock_mhz is not None:
            parts.append(f" sm_clock={self.sm_clock_mhz}MHz")
        if self.memory_clock_mhz is not None:
            parts.append(f" mem_clock={self.memory_clock_mhz}MHz")
        if self.temperature_c is not None:
            parts.append(f" temp={self.temperature_c}C")
        if self.power_draw_w is not None:
            power_text = f" power={self.power_draw_w:.0f}W"
            if self.power_limit_w is not None:
                power_text += f"/{self.power_limit_w:.0f}W"
            parts.append(power_text)
        return "".join(parts)


@lru_cache(maxsize=1)
def get_nvidia_smi_path() -> str | None:
    return shutil.which("nvidia-smi")


def describe_gpu_telemetry_backend() -> str:
    nvidia_smi_path = get_nvidia_smi_path()
    if nvidia_smi_path is None:
        return "unavailable"
    return f"nvidia-smi ({nvidia_smi_path})"


def query_gpu_telemetry(device_index: int | None = None) -> GpuTelemetrySnapshot | None:
    nvidia_smi_path = get_nvidia_smi_path()
    if nvidia_smi_path is None:
        return None

    command = [
        nvidia_smi_path,
        f"--query-gpu={','.join(_QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    if device_index is not None:
        command.append(f"--id={device_index}")

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return None

    parts = [part.strip() for part in lines[0].split(",")]
    if len(parts) != len(_QUERY_FIELDS):
        return None

    return GpuTelemetrySnapshot(
        name=parts[0],
        gpu_utilization=_parse_int(parts[1]),
        memory_utilization=_parse_int(parts[2]),
        sm_clock_mhz=_parse_int(parts[3]),
        memory_clock_mhz=_parse_int(parts[4]),
        temperature_c=_parse_int(parts[5]),
        power_draw_w=_parse_float(parts[6]),
        power_limit_w=_parse_float(parts[7]),
    )