from __future__ import annotations

import logging
import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def ffmpeg_convert_video(
    input_path: str,
    output_path: str,
    *,
    vf: str | None = None,
    pix_fmt: str | None = None,
    video_codec: str | None = None,
    overwrite: bool = True,
    extra_args: list[str] | None = None,
) -> None:
    """Run an ffmpeg conversion step while keeping OpenCV for frame access."""
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    command.append("-y" if overwrite else "-n")
    command.extend(["-i", input_path])

    if vf:
        command.extend(["-vf", vf])
    if video_codec:
        command.extend(["-c:v", video_codec])
    if pix_fmt:
        command.extend(["-pix_fmt", pix_fmt])
    if extra_args:
        command.extend(extra_args)

    command.append(output_path)
    subprocess.run(command, check=True)


class OptimizedVideoCapture:
    """MMCV-style OpenCV video reader with caching and validated seeking."""

    def __init__(self, video_path: str, buffer_size: int = 30, cache_size: int = 50):
        self.video_path = video_path
        self.buffer_size = buffer_size
        self.cache_size = max(1, int(cache_size))

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))

        self._position = 0
        self._frame_cache: OrderedDict[int, np.ndarray] = OrderedDict()

        logger.info(
            "VideoCapture initialized: %sx%s @ %.3f FPS, %s frames",
            self.width,
            self.height,
            self.fps,
            self.total_frames,
        )

    @property
    def opened(self) -> bool:
        return self.cap.isOpened()

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def frame_cnt(self) -> int:
        return self.total_frames

    @property
    def position(self) -> int:
        return self._position

    @property
    def vcap(self) -> cv2.VideoCapture:
        return self.cap

    def _cache_get(self, frame_idx: int) -> Optional[np.ndarray]:
        frame = self._frame_cache.get(frame_idx)
        if frame is None:
            return None
        self._frame_cache.move_to_end(frame_idx)
        return frame.copy()

    def _cache_put(self, frame_idx: int, frame: np.ndarray) -> None:
        self._frame_cache[frame_idx] = frame.copy()
        self._frame_cache.move_to_end(frame_idx)
        while len(self._frame_cache) > self.cache_size:
            self._frame_cache.popitem(last=False)

    def _get_real_position(self) -> int:
        return int(round(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))

    def _set_real_position(self, frame_idx: int) -> None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        real_position = self._get_real_position()
        for _ in range(max(frame_idx - real_position, 0)):
            self.cap.read()
        self._position = frame_idx

    def seek(self, frame_idx: int) -> bool:
        if frame_idx < 0 or frame_idx >= self.total_frames:
            return False
        self._set_real_position(frame_idx)
        return True

    def read(self) -> Optional[np.ndarray]:
        cached = self._cache_get(self._position)
        if cached is not None:
            self._position += 1
            return cached

        if self._position != self._get_real_position():
            self._set_real_position(self._position)

        ret, frame = self.cap.read()
        if not ret:
            return None

        frame_idx = self._position
        self._cache_put(frame_idx, frame)
        self._position += 1
        return frame.copy()

    def read_next(self) -> tuple[bool, Optional[np.ndarray]]:
        frame = self.read()
        return frame is not None, frame

    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        if frame_idx < 0 or frame_idx >= self.total_frames:
            return None

        if frame_idx == self._position:
            return self.read()

        cached = self._cache_get(frame_idx)
        if cached is not None:
            self._position = frame_idx + 1
            return cached

        self._set_real_position(frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return None

        self._cache_put(self._position, frame)
        self._position += 1
        return frame.copy()

    def current_frame(self) -> Optional[np.ndarray]:
        if self._position == 0:
            return None
        return self._cache_get(self._position - 1)

    def cvt2frames(
        self,
        frame_dir: str,
        file_start: int = 0,
        filename_tmpl: str = "{:06d}.jpg",
        start: int = 0,
        max_num: int = 0,
    ) -> None:
        output_dir = Path(frame_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if max_num == 0:
            task_num = self.frame_cnt - start
        else:
            task_num = min(self.frame_cnt - start, max_num)

        if task_num <= 0:
            raise ValueError("start must be less than total frame number")

        self._set_real_position(start)
        for file_idx in range(file_start, file_start + task_num):
            frame = self.read()
            if frame is None:
                break
            output_path = output_dir / filename_tmpl.format(file_idx)
            cv2.imwrite(str(output_path), frame)

    def close(self) -> None:
        if self.cap.isOpened():
            self.cap.release()
            logger.info("VideoCapture closed")

    def __len__(self) -> int:
        return self.frame_cnt

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self.get_frame(frame_idx) for frame_idx in range(*idx.indices(self.frame_cnt))]
        if idx < 0:
            idx += self.frame_cnt
        frame = self.get_frame(int(idx))
        if frame is None:
            raise IndexError(f"Frame {idx} out of range")
        return frame

    def __iter__(self):
        self._set_real_position(0)
        return self

    def __next__(self) -> np.ndarray:
        frame = self.read()
        if frame is None:
            raise StopIteration
        return frame

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception as exc:
            logger.error("Error while closing VideoCapture: %s", exc)