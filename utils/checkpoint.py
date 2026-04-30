from __future__ import annotations

from multiprocessing.util import get_logger
from pathlib import Path
from typing import Any

import torch


def _extract_state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        if isinstance(state_dict, dict):
            return state_dict
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError("Unsupported checkpoint format")


def _normalize_state_dict(state_dict: dict[str, Any], model: torch.nn.Module) -> dict[str, Any]:
    model_keys = set(model.state_dict().keys())

    filtered = {key: value for key, value in state_dict.items() if key != "step_counter"}
    if set(filtered.keys()) == model_keys:
        return filtered

    if any(key.startswith("generator.") for key in filtered):
        stripped = {
            key[len("generator.") :]: value
            for key, value in filtered.items()
            if key.startswith("generator.")
        }
        if stripped:
            return stripped

    return filtered


def load_model_checkpoint(model: torch.nn.Module, checkpoint_path: str, map_location: str | torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    state_dict = _normalize_state_dict(_extract_state_dict(checkpoint), model)

    result = model.load_state_dict(state_dict, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)

    if not missing and not unexpected:
        return

    logger = get_logger()

    if unexpected:
        logger.warning(
            "Checkpoint contains %d unexpected key(s) (architecture smaller than training). "
            "These weights will be ignored. First few: %s",
            len(unexpected),
            ", ".join(unexpected[:5]),
        )

    if missing:
        details = [
            f"Checkpoint mismatch for {Path(checkpoint_path).name} and {model.__class__.__name__}.",
            f"Missing keys: {len(missing)}",
            "Missing sample: " + ", ".join(missing[:5]),
        ]
        raise RuntimeError("\n".join(details))