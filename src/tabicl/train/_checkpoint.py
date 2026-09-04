"""Checkpoint metadata helpers for reproducible pre-training experiments."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tabicl.prior.graph_lib._config import PriorConfig

CHECKPOINT_METADATA_VERSION = 1
_GRAPH_U_KEYS = (
    "graph_u_enabled",
    "graph_u_query_location",
    "graph_u_query_scale",
    "graph_u_force_gaussian",
    "graph_u_max_attempts",
)


def normalize_config(value: Any) -> Any:
    """Convert configuration values into JSON- and weights-only-safe objects."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return normalize_config(value.item())
    if isinstance(value, (Path, os.PathLike, torch.device, torch.dtype)):
        return str(value)
    if isinstance(value, Enum):
        return normalize_config(value.value)
    if isinstance(value, argparse.Namespace):
        return normalize_config(vars(value))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return normalize_config(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): normalize_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [normalize_config(item) for item in value]
    raise TypeError(
        f"Configuration value of type {type(value).__name__} is not serializable: {value!r}"
    )


def checkpoint_training_metadata(
    training_config: Mapping[str, Any],
    prior_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the extra checkpoint fields used for experiment provenance."""

    normalized_training = normalize_config(training_config)
    normalized_prior = normalize_config(prior_config)
    assert isinstance(normalized_training, dict)
    assert normalized_prior is None or isinstance(normalized_prior, dict)
    return {
        "checkpoint_metadata_version": CHECKPOINT_METADATA_VERSION,
        "training_config": normalized_training,
        "prior_config": normalized_prior,
    }


def build_run_config(
    *,
    model_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    prior_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the human-readable run manifest saved beside checkpoints."""

    metadata = checkpoint_training_metadata(training_config, prior_config)
    normalized_model = normalize_config(model_config)
    assert isinstance(normalized_model, dict)
    return {
        "checkpoint_metadata_version": metadata["checkpoint_metadata_version"],
        "model_config": normalized_model,
        "training_config": metadata["training_config"],
        "prior_config": metadata["prior_config"],
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
    os.replace(temporary_path, path)


def save_run_config(
    checkpoint_dir: str | os.PathLike[str],
    payload: Mapping[str, Any],
    *,
    current_step: int,
) -> Path:
    """Persist a run manifest without silently replacing a different launch."""

    directory = Path(checkpoint_dir)
    primary_path = directory / "run_config.json"
    normalized_payload = normalize_config(payload)
    assert isinstance(normalized_payload, dict)

    if primary_path.exists():
        with primary_path.open("r", encoding="utf-8") as file:
            existing_payload = json.load(file)
        if existing_payload == normalized_payload:
            return primary_path
        target_path = directory / f"run_config.step-{current_step}.json"
        collision_index = 1
        while target_path.exists():
            with target_path.open("r", encoding="utf-8") as file:
                existing_payload = json.load(file)
            if existing_payload == normalized_payload:
                return target_path
            target_path = directory / (
                f"run_config.step-{current_step}.{collision_index}.json"
            )
            collision_index += 1
    else:
        target_path = primary_path

    _atomic_write_json(target_path, normalized_payload)
    return target_path


def get_matched_graph_u_config(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the Graph-U shift used to train a metadata-aware checkpoint.

    Older upstream checkpoints do not contain training provenance. In that case
    callers must supply evaluation shift parameters explicitly; silently using
    defaults could mislabel an evaluation as matched.
    """

    training_config = checkpoint.get("training_config")
    prior_config = checkpoint.get("prior_config")
    if not isinstance(training_config, Mapping) or not isinstance(
        prior_config, Mapping
    ):
        raise TypeError(
            "Checkpoint does not contain training_config and prior_config metadata; "
            "the matched Graph-U shift is unknown. Supply evaluation shift parameters "
            "explicitly for legacy checkpoints."
        )

    # For a pre-generated prior, its metadata is the effective configuration;
    # the launch-time --prior_type value may merely be an unused parser default.
    prior_type = prior_config.get("prior_type", training_config.get("prior_type"))
    if prior_type != "graph_scm":
        raise ValueError(
            "Matched Graph-U evaluation requires a checkpoint trained with "
            f"prior_type='graph_scm', got {prior_type!r}."
        )

    missing = [key for key in _GRAPH_U_KEYS if key not in prior_config]
    if missing:
        raise ValueError(
            "Checkpoint prior_config is missing required Graph-U fields: "
            + ", ".join(missing)
        )
    if prior_config["graph_u_enabled"] is not True:
        raise ValueError(
            "Checkpoint was not trained with Graph-U enabled; it has no matched "
            "Graph-U shift. Supply an evaluation condition explicitly."
        )

    validated = PriorConfig(
        graph_u_enabled=True,
        graph_u_query_location=prior_config["graph_u_query_location"],
        graph_u_query_scale=prior_config["graph_u_query_scale"],
        graph_u_force_gaussian=prior_config["graph_u_force_gaussian"],
        graph_u_max_attempts=prior_config["graph_u_max_attempts"],
    )
    return {
        "prior_type": prior_type,
        "graph_u_enabled": validated.graph_u_enabled,
        "graph_u_query_location": validated.graph_u_query_location,
        "graph_u_query_scale": validated.graph_u_query_scale,
        "graph_u_force_gaussian": validated.graph_u_force_gaussian,
        "graph_u_max_attempts": validated.graph_u_max_attempts,
    }


def load_matched_graph_u_config(
    checkpoint_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Load a checkpoint and return its matched Graph-U evaluation settings."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint payload must be a mapping")
    return get_matched_graph_u_config(checkpoint)
