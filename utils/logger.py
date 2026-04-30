from __future__ import annotations

import logging
from pathlib import Path
import sys


def _resolve_level(level: str | int) -> int:
	if isinstance(level, int):
		return level
	return getattr(logging, str(level).upper(), logging.INFO)


def _root_level(*levels: int | None) -> int:
	resolved_levels = [level for level in levels if level is not None]
	if not resolved_levels:
		return logging.INFO
	return min(resolved_levels)


def format_bytes(num_bytes: int | float) -> str:
	size = float(num_bytes)
	units = ("B", "KiB", "MiB", "GiB", "TiB")
	for unit in units:
		if abs(size) < 1024.0 or unit == units[-1]:
			if unit == "B":
				return f"{int(size)}{unit}"
			return f"{size:.2f}{unit}"
		size /= 1024.0
	return f"{size:.2f}{units[-1]}"


def configure_logging(level: str | int = "INFO", log_file: str | None = None, console_level: str | int | None = None, file_level: str | int | None = None) -> None:
	root_logger = logging.getLogger("realbasic")
	resolved_console_level = _resolve_level(console_level if console_level is not None else level)
	resolved_file_level = _resolve_level(file_level if file_level is not None else level) if log_file else None
	root_logger.handlers.clear()
	root_logger.setLevel(_root_level(resolved_console_level, resolved_file_level))
	root_logger.propagate = False

	formatter = logging.Formatter(
		fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
		datefmt="%H:%M:%S",
	)

	console_handler = logging.StreamHandler(sys.stdout)
	console_handler.setLevel(resolved_console_level)
	console_handler.setFormatter(formatter)
	root_logger.addHandler(console_handler)

	if log_file:
		assert resolved_file_level is not None
		log_path = Path(log_file)
		log_path.parent.mkdir(parents=True, exist_ok=True)
		file_handler = logging.FileHandler(log_path, encoding="utf-8")
		file_handler.setLevel(resolved_file_level)
		file_handler.setFormatter(formatter)
		root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
	if not logging.getLogger("realbasic").handlers:
		configure_logging()

	if not name or name == "__main__":
		return logging.getLogger("realbasic")

	return logging.getLogger(f"realbasic.{name}")
