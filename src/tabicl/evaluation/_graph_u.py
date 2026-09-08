"""Deterministic, on-the-fly evaluation on Graph-U synthetic tasks."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

from tabicl.__about__ import __version__
from tabicl._model.inference_config import InferenceConfig
from tabicl._model.tabicl import TabICL
from tabicl.prior._dataset import GraphPrior
from tabicl.prior.graph_lib._config import PriorConfig
from tabicl.train._checkpoint import get_matched_graph_u_config, normalize_config

TaskType = Literal["classification", "regression"]
_CONDITION_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_PRIOR_CONFIG_FIELDS = {field.name for field in dataclasses.fields(PriorConfig)}


@dataclass(frozen=True)
class ConditionSpec:
    """A parsed condition before checkpoint-dependent settings are resolved."""

    name: str
    kind: Literal["ordinary", "identity", "matched", "custom"]
    query_location: float | None = None
    query_scale: float | None = None
    force_gaussian: bool | None = None


@dataclass(frozen=True)
class EvaluationCondition:
    """A complete Graph-U environment used to generate evaluation tasks."""

    name: str
    graph_u_enabled: bool
    query_location: float
    query_scale: float
    force_gaussian: bool
    source: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "graph_u_enabled": self.graph_u_enabled,
            "graph_u_query_location": self.query_location,
            "graph_u_query_scale": self.query_scale,
            "graph_u_force_gaussian": self.force_gaussian,
            "source": self.source,
        }


@dataclass(frozen=True)
class CheckpointDescriptor:
    """Small, state-dict-free description of an evaluation checkpoint."""

    path: Path
    label: str
    task_type: TaskType
    model_config: dict[str, Any]
    training_config: dict[str, Any] | None
    prior_config: dict[str, Any] | None
    curr_step: int | None
    size_bytes: int
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "label": self.label,
            "task_type": self.task_type,
            "curr_step": self.curr_step,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "model_config": self.model_config,
            "training_config": self.training_config,
            "prior_config": self.prior_config,
        }


@dataclass(frozen=True)
class EvaluationTask:
    """One generated task, retained only long enough to score it."""

    X: torch.Tensor
    y: torch.Tensor
    train_size: int
    active_features: int
    seed: int
    fingerprint: str


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Expected a boolean, got {value!r}")


def parse_condition(value: str) -> ConditionSpec:
    """Parse ordinary, identity, matched, or NAME:LOCATION:SCALE[:BOOL]."""

    value = value.strip()
    lowered = value.lower()
    if lowered in {"ordinary", "identity", "matched"}:
        return ConditionSpec(name=lowered, kind=lowered)

    parts = value.split(":")
    if len(parts) not in {3, 4}:
        raise ValueError(
            "Condition must be ordinary, identity, matched, or "
            "NAME:LOCATION:SCALE[:FORCE_GAUSSIAN]"
        )
    name = parts[0].strip()
    if not _CONDITION_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "Custom condition names may contain only letters, digits, '.', '_', and '-'"
        )
    try:
        location = float(parts[1])
        scale = float(parts[2])
    except ValueError as error:
        raise ValueError(f"Invalid location or scale in condition {value!r}") from error
    force_gaussian = _parse_bool(parts[3]) if len(parts) == 4 else None

    # Reuse the prior's validation for finite location and positive finite scale.
    PriorConfig(
        graph_u_query_location=location,
        graph_u_query_scale=scale,
    )
    return ConditionSpec(
        name=name,
        kind="custom",
        query_location=location,
        query_scale=scale,
        force_gaussian=force_gaussian,
    )


def _mapping_or_none(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    normalized = normalize_config(value)
    assert isinstance(normalized, dict)
    return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_checkpoint(path: str | os.PathLike[str]) -> CheckpointDescriptor:
    """Validate a checkpoint and read only the metadata needed by evaluation."""

    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {resolved_path}")
    checkpoint = torch.load(resolved_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint payload must be a mapping: {resolved_path}")
    if not isinstance(checkpoint.get("config"), Mapping):
        raise TypeError(f"Checkpoint has no model config: {resolved_path}")
    if not isinstance(checkpoint.get("state_dict"), Mapping):
        raise TypeError(f"Checkpoint has no state_dict: {resolved_path}")

    model_config = _mapping_or_none(checkpoint["config"])
    assert model_config is not None
    max_classes = model_config.get("max_classes")
    if (
        not isinstance(max_classes, int)
        or isinstance(max_classes, bool)
        or max_classes < 0
    ):
        raise ValueError(
            f"Checkpoint config has invalid max_classes={max_classes!r}: {resolved_path}"
        )
    task_type: TaskType = "regression" if max_classes == 0 else "classification"
    curr_step = checkpoint.get("curr_step")
    if not isinstance(curr_step, int) or isinstance(curr_step, bool):
        curr_step = None

    return CheckpointDescriptor(
        path=resolved_path,
        label=resolved_path.stem,
        task_type=task_type,
        model_config=model_config,
        training_config=_mapping_or_none(checkpoint.get("training_config")),
        prior_config=_mapping_or_none(checkpoint.get("prior_config")),
        curr_step=curr_step,
        size_bytes=resolved_path.stat().st_size,
        sha256=_sha256_file(resolved_path),
    )


def disambiguate_checkpoint_labels(
    descriptors: Sequence[CheckpointDescriptor],
) -> list[CheckpointDescriptor]:
    """Reject duplicate paths and make equal checkpoint basenames readable."""

    paths = [descriptor.path for descriptor in descriptors]
    duplicates = [str(path) for path, count in Counter(paths).items() if count > 1]
    if duplicates:
        raise ValueError(
            "A checkpoint path was supplied more than once: " + ", ".join(duplicates)
        )

    stem_counts = Counter(descriptor.label for descriptor in descriptors)
    candidates = [
        (
            f"{descriptor.path.parent.name}/{descriptor.label}"
            if stem_counts[descriptor.label] > 1
            else descriptor.label
        )
        for descriptor in descriptors
    ]
    candidate_counts = Counter(candidates)
    return [
        dataclasses.replace(
            descriptor,
            label=(
                str(descriptor.path) if candidate_counts[candidate] > 1 else candidate
            ),
        )
        for descriptor, candidate in zip(descriptors, candidates, strict=True)
    ]


def _checkpoint_metadata(descriptor: CheckpointDescriptor) -> dict[str, Any]:
    return {
        "training_config": descriptor.training_config,
        "prior_config": descriptor.prior_config,
    }


def _matched_config_or_none(
    descriptor: CheckpointDescriptor,
) -> dict[str, Any] | None:
    try:
        return get_matched_graph_u_config(_checkpoint_metadata(descriptor))
    except (TypeError, ValueError):
        return None


def choose_reference_checkpoint(
    descriptors: Sequence[CheckpointDescriptor],
    explicit_reference: str | os.PathLike[str] | None,
) -> CheckpointDescriptor:
    """Choose the prior/shift reference applied consistently to every model."""

    if explicit_reference is not None:
        explicit_path = Path(explicit_reference).expanduser().resolve()
        for descriptor in descriptors:
            if descriptor.path == explicit_path:
                return descriptor
        return inspect_checkpoint(explicit_path)

    for descriptor in descriptors:
        if _matched_config_or_none(descriptor) is not None:
            return descriptor
    return descriptors[0]


def default_condition_specs(reference: CheckpointDescriptor) -> list[ConditionSpec]:
    """Return useful defaults, adding matched only when it is knowable."""

    specs = [parse_condition("ordinary"), parse_condition("identity")]
    if _matched_config_or_none(reference) is not None:
        specs.append(parse_condition("matched"))
    return specs


def resolve_conditions(
    specs: Sequence[ConditionSpec],
    reference: CheckpointDescriptor,
) -> list[EvaluationCondition]:
    """Resolve checkpoint-dependent settings once for all evaluated models."""

    prior_config = reference.prior_config or {}
    matched_config = _matched_config_or_none(reference)
    force_gaussian_recorded = "graph_u_force_gaussian" in prior_config
    base_force_gaussian = prior_config.get("graph_u_force_gaussian", False)
    if not isinstance(base_force_gaussian, bool):
        raise TypeError("Reference graph_u_force_gaussian must be a boolean")

    resolved: list[EvaluationCondition] = []
    names: set[str] = set()
    for spec in specs:
        if spec.name in names:
            raise ValueError(f"Duplicate evaluation condition name: {spec.name!r}")
        names.add(spec.name)

        if spec.kind == "ordinary":
            condition = EvaluationCondition(
                name=spec.name,
                graph_u_enabled=False,
                query_location=0.0,
                query_scale=1.0,
                force_gaussian=base_force_gaussian,
                source="ordinary graph_scm without Graph-U conditioning",
            )
        elif spec.kind == "identity":
            policy_source = (
                f"source-family policy from {reference.path}"
                if force_gaussian_recorded
                else "evaluator default source-family policy (checkpoint metadata unavailable)"
            )
            condition = EvaluationCondition(
                name=spec.name,
                graph_u_enabled=True,
                query_location=0.0,
                query_scale=1.0,
                force_gaussian=base_force_gaussian,
                source=policy_source,
            )
        elif spec.kind == "matched":
            if matched_config is None:
                raise ValueError(
                    "The matched condition was requested, but the reference checkpoint "
                    "does not record a Graph-U training shift. Use --reference-checkpoint "
                    "with a metadata-aware Graph-U checkpoint or specify a custom condition."
                )
            condition = EvaluationCondition(
                name=spec.name,
                graph_u_enabled=True,
                query_location=float(matched_config["graph_u_query_location"]),
                query_scale=float(matched_config["graph_u_query_scale"]),
                force_gaussian=bool(matched_config["graph_u_force_gaussian"]),
                source=f"matched to {reference.path}",
            )
        else:
            assert spec.query_location is not None and spec.query_scale is not None
            condition = EvaluationCondition(
                name=spec.name,
                graph_u_enabled=True,
                query_location=spec.query_location,
                query_scale=spec.query_scale,
                force_gaussian=(
                    base_force_gaussian
                    if spec.force_gaussian is None
                    else spec.force_gaussian
                ),
                source="explicit CLI condition",
            )
        resolved.append(condition)
    return resolved


def prior_reconstruction_details(
    reference: CheckpointDescriptor,
) -> dict[str, Any]:
    """Describe whether the reference stored every base-prior field."""

    stored = dict(reference.prior_config or {})
    if "graph_noise" in stored and "add_gaussian_noise" not in stored:
        stored["add_gaussian_noise"] = stored["graph_noise"]
    missing = sorted(_PRIOR_CONFIG_FIELDS - stored.keys())
    return {
        "source": (
            "reference checkpoint prior_config"
            if reference.prior_config is not None
            else "evaluator PriorConfig defaults"
        ),
        "complete": not missing,
        "missing_fields_using_evaluator_defaults": missing,
    }


def code_provenance() -> dict[str, Any]:
    """Record runtime versions and local Git state when available."""

    repository_root = Path(__file__).resolve().parents[3]
    git_commit = None
    git_dirty = None
    try:
        commit_result = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        status_result = subprocess.run(
            ["git", "-C", str(repository_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        git_commit = commit_result.stdout.strip()
        git_dirty = bool(status_result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass

    return {
        "tabicl_version": __version__,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }


def build_prior_config(
    reference: CheckpointDescriptor,
    condition: EvaluationCondition,
    *,
    max_attempts: int | None = None,
) -> PriorConfig:
    """Reconstruct the reference prior and apply one evaluation intervention."""

    stored = dict(reference.prior_config or {})
    # Pre-generated prior metadata uses the CLI spelling for this field.
    if "graph_noise" in stored and "add_gaussian_noise" not in stored:
        stored["add_gaussian_noise"] = stored["graph_noise"]
    kwargs = {
        key: value for key, value in stored.items() if key in _PRIOR_CONFIG_FIELDS
    }
    kwargs.update(
        graph_u_enabled=condition.graph_u_enabled,
        graph_u_query_location=condition.query_location,
        graph_u_query_scale=condition.query_scale,
        graph_u_force_gaussian=condition.force_gaussian,
    )
    if max_attempts is not None:
        kwargs["graph_u_max_attempts"] = max_attempts
    return PriorConfig(**kwargs)


@contextmanager
def isolated_task_seed(seed: int):
    """Make one task reproducible without perturbing caller RNG state."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        # Graph generation is CPU-only. Seed only the CPU default generator so
        # evaluation task generation cannot perturb CUDA/XPU/MPS RNG streams.
        torch.random.default_generator.manual_seed(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


def derive_task_seed(base_seed: int, task_index: int) -> int:
    """Derive independent, stable seeds; conditions intentionally share them."""

    digest = hashlib.blake2b(
        f"tabicl-graph-u-eval:{base_seed}:{task_index}".encode(),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little") % (2**31)


def _tensor_fingerprint(*tensors: torch.Tensor, metadata: str = "") -> str:
    digest = hashlib.sha256(metadata.encode())
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def generate_task(
    *,
    task_type: TaskType,
    prior_config: PriorConfig,
    support_size: int,
    query_size: int,
    num_features: int,
    num_classes: int,
    seed: int,
) -> EvaluationTask:
    """Generate one production-faithful task on CPU and discard no labels yet."""

    if support_size <= 0 or query_size <= 0 or num_features <= 0:
        raise ValueError("support_size, query_size, and num_features must be positive")
    if task_type == "classification" and num_classes < 2:
        raise ValueError("Classification evaluation requires num_classes >= 2")
    if task_type == "classification" and (support_size < 2 or query_size < 2):
        raise ValueError(
            "Classification requires support_size and query_size to each be at least 2"
        )

    regression = task_type == "regression"
    with isolated_task_seed(seed):
        prior = GraphPrior(
            regression=regression,
            batch_size=1,
            min_features=num_features,
            max_features=num_features,
            max_classes=num_classes,
            config=prior_config,
            n_jobs=1,
            device="cpu",
            generation_max_attempts=prior_config.graph_u_max_attempts,
        )
        params = {
            "regression": regression,
            "seq_len": support_size + query_size,
            "train_size": support_size,
            "num_features": num_features,
            "max_features": num_features,
            "num_classes": None if regression else num_classes,
            "device": "cpu",
            "config": prior_config,
        }
        X, y, d = prior.generate_dataset(params)

    active_features = int(d.item())
    X = X[:, :active_features].detach().cpu().contiguous()
    y = y.detach().cpu().contiguous()
    fingerprint = _tensor_fingerprint(
        X,
        y,
        metadata=f"{support_size}:{query_size}:{active_features}:{seed}",
    )
    return EvaluationTask(
        X=X,
        y=y,
        train_size=support_size,
        active_features=active_features,
        seed=seed,
        fingerprint=fingerprint,
    )


def _encode_classification_targets(y: torch.Tensor, train_size: int) -> torch.Tensor:
    support_classes = torch.unique(y[:train_size], sorted=True)
    query_classes = torch.unique(y[train_size:], sorted=True)
    if not torch.equal(support_classes, query_classes):
        raise ValueError(
            "Support and query must contain the same classification labels"
        )
    encoded = torch.searchsorted(support_classes, y)
    if not torch.equal(support_classes[encoded], y):
        raise ValueError("Could not encode classification labels")
    return encoded


def score_classification_task(
    model: Any,
    task: EvaluationTask,
    *,
    device: torch.device,
    inference_config: InferenceConfig | None,
    softmax_temperature: float,
) -> dict[str, float | int]:
    """Score a task while passing only support labels into the model."""

    encoded_y = _encode_classification_targets(task.y, task.train_size)
    X = task.X.unsqueeze(0).to(device)
    y_support = encoded_y[: task.train_size].unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(
            X,
            y_support,
            return_logits=True,
            softmax_temperature=softmax_temperature,
            inference_config=inference_config,
        ).float()

    # Query labels are accessed only after inference has returned.
    y_query = encoded_y[task.train_size :].long().to(logits.device)
    logits = logits.squeeze(0)
    if logits.shape[0] != len(y_query):
        raise RuntimeError(
            f"Model returned {logits.shape[0]} query rows for {len(y_query)} labels"
        )
    if not torch.isfinite(logits).all():
        raise RuntimeError("Model returned non-finite classification logits")
    probabilities = torch.softmax(logits / softmax_temperature, dim=-1)
    predictions = probabilities.argmax(dim=-1)
    log_loss = F.cross_entropy(logits / softmax_temperature, y_query)
    one_hot = F.one_hot(y_query, num_classes=probabilities.shape[-1]).float()
    brier = (probabilities - one_hot).square().sum(dim=-1).mean()
    return {
        "accuracy": float((predictions == y_query).float().mean().item()),
        "log_loss": float(log_loss.item()),
        "brier_score": float(brier.item()),
        "num_classes": int(probabilities.shape[-1]),
    }


def score_regression_task(
    model: Any,
    task: EvaluationTask,
    *,
    device: torch.device,
    inference_config: InferenceConfig | None,
) -> dict[str, float | None]:
    """Score point predictions and the checkpoint's quantile objective."""

    X = task.X.unsqueeze(0).to(device)
    y_support = task.y[: task.train_size].unsqueeze(0).to(device)
    with torch.inference_mode():
        raw_quantiles = (
            model(
                X,
                y_support,
                inference_config=inference_config,
            )
            .float()
            .squeeze(0)
        )

    # Query labels are accessed only after inference has returned.
    y_query = task.y[task.train_size :].float().to(raw_quantiles.device)
    if raw_quantiles.shape[0] != len(y_query):
        raise RuntimeError(
            f"Model returned {raw_quantiles.shape[0]} query rows for {len(y_query)} labels"
        )
    num_quantiles = raw_quantiles.shape[-1]
    alphas = torch.linspace(
        0.0,
        1.0,
        num_quantiles + 2,
        device=raw_quantiles.device,
        dtype=raw_quantiles.dtype,
    )[1:-1]
    if not torch.isfinite(raw_quantiles).all():
        raise RuntimeError("Model returned non-finite regression quantiles")
    raw_errors = y_query.unsqueeze(-1) - raw_quantiles
    raw_pinball = torch.maximum(alphas * raw_errors, (alphas - 1) * raw_errors).mean()

    # TabICL's public regression path uses the same sorting policy to repair
    # crossing quantiles. Their mean is unchanged by sorting.
    monotonic_quantiles = raw_quantiles.sort(dim=-1).values
    errors = y_query.unsqueeze(-1) - monotonic_quantiles
    predictive_pinball = torch.maximum(alphas * errors, (alphas - 1) * errors).mean()
    mean_prediction = monotonic_quantiles.mean(dim=-1)
    median_prediction = torch.quantile(monotonic_quantiles, 0.5, dim=-1)
    residual = mean_prediction - y_query
    total_variation = (y_query - y_query.mean()).square().sum()
    r2 = None
    if total_variation.item() > 0:
        r2 = float((1.0 - residual.square().sum() / total_variation).item())
    return {
        "rmse": float(residual.square().mean().sqrt().item()),
        "mae": float(residual.abs().mean().item()),
        "median_mae": float((median_prediction - y_query).abs().mean().item()),
        "r2": r2,
        "pinball_loss": float(predictive_pinball.item()),
        "crps_approximation": float((2.0 * predictive_pinball).item()),
        "raw_training_pinball_loss": float(raw_pinball.item()),
    }


def load_model(
    descriptor: CheckpointDescriptor,
    device: torch.device,
) -> TabICL:
    """Load one raw prior-native model without sklearn preprocessing ensembles."""

    checkpoint = torch.load(descriptor.path, map_location="cpu", weights_only=True)
    model = TabICL(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def build_inference_config(device: torch.device, use_amp: bool) -> InferenceConfig:
    """Use an explicit device/precision policy rather than environment defaults."""

    manager_config = {
        "device": str(device),
        "use_amp": use_amp,
        "use_fa3": False,
        "offload": False,
        "use_async": False,
    }
    return InferenceConfig(
        COL_CONFIG=dict(manager_config),
        ROW_CONFIG=dict(manager_config),
        ICL_CONFIG=dict(manager_config),
    )


def resolve_device(value: str) -> torch.device:
    """Resolve auto to the first available accelerator, otherwise CPU."""

    if value != "auto":
        device = torch.device(value)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if device.type == "xpu" and (
        not hasattr(torch, "xpu") or not torch.xpu.is_available()
    ):
        raise ValueError("XPU was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    return device


def _summarize_task_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metric_names = sorted(
        {
            key
            for record in records
            for key, value in record["metrics"].items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        - {"num_classes"}
    )
    summary: dict[str, Any] = {"tasks": len(records)}
    for metric_name in metric_names:
        values = np.asarray(
            [
                record["metrics"][metric_name]
                for record in records
                if record["metrics"].get(metric_name) is not None
            ],
            dtype=float,
        )
        if not len(values):
            continue
        standard_error = (
            float(values.std(ddof=1) / math.sqrt(len(values)))
            if len(values) > 1
            else 0.0
        )
        summary[metric_name] = {
            "count": len(values),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "standard_error": standard_error,
            "ci95_low": float(values.mean() - 1.96 * standard_error),
            "ci95_high": float(values.mean() + 1.96 * standard_error),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def evaluate(
    *,
    descriptors: Sequence[CheckpointDescriptor],
    reference: CheckpointDescriptor,
    conditions: Sequence[EvaluationCondition],
    datasets: int,
    support_size: int,
    query_size: int,
    num_features: int,
    num_classes: int,
    base_seed: int,
    device: torch.device,
    use_amp: bool,
    softmax_temperature: float,
    max_attempts: int | None,
    progress: bool = True,
) -> dict[str, Any]:
    """Evaluate all checkpoints on deterministically regenerated task streams."""

    if not descriptors:
        raise ValueError("At least one checkpoint is required")
    if datasets <= 0:
        raise ValueError("datasets must be positive")
    task_types = {descriptor.task_type for descriptor in descriptors}
    if len(task_types) != 1:
        raise ValueError(
            "Classification and regression checkpoints cannot be evaluated together"
        )
    task_type = next(iter(task_types))
    if task_type == "classification" and num_classes < 2:
        raise ValueError("Classification evaluation requires num_classes >= 2")
    if task_type == "classification" and (support_size < 2 or query_size < 2):
        raise ValueError(
            "Classification requires support_size and query_size to each be at least 2"
        )
    if softmax_temperature <= 0 or not math.isfinite(softmax_temperature):
        raise ValueError("softmax_temperature must be finite and positive")

    inference_config = build_inference_config(device, use_amp)
    expected_fingerprints: dict[tuple[str, int], str] = {}
    model_results: list[dict[str, Any]] = []

    for descriptor in descriptors:
        model = load_model(descriptor, device)
        condition_results: list[dict[str, Any]] = []
        for condition in conditions:
            prior_config = build_prior_config(
                reference,
                condition,
                max_attempts=max_attempts,
            )
            records: list[dict[str, Any]] = []
            for task_index in range(datasets):
                seed = derive_task_seed(base_seed, task_index)
                task = generate_task(
                    task_type=task_type,
                    prior_config=prior_config,
                    support_size=support_size,
                    query_size=query_size,
                    num_features=num_features,
                    num_classes=num_classes,
                    seed=seed,
                )
                fingerprint_key = (condition.name, task_index)
                expected = expected_fingerprints.setdefault(
                    fingerprint_key, task.fingerprint
                )
                if expected != task.fingerprint:
                    raise RuntimeError(
                        "Deterministic task regeneration failed; checkpoints would not "
                        f"receive identical data for {condition.name} task {task_index}"
                    )

                try:
                    if task_type == "classification":
                        metrics = score_classification_task(
                            model,
                            task,
                            device=device,
                            inference_config=inference_config,
                            softmax_temperature=softmax_temperature,
                        )
                    else:
                        metrics = score_regression_task(
                            model,
                            task,
                            device=device,
                            inference_config=inference_config,
                        )
                    non_finite_metrics = [
                        name
                        for name, value in metrics.items()
                        if isinstance(value, float) and not math.isfinite(value)
                    ]
                    if non_finite_metrics:
                        raise RuntimeError(
                            "Non-finite metrics: " + ", ".join(non_finite_metrics)
                        )
                except Exception as error:
                    raise RuntimeError(
                        f"Failed to score checkpoint {descriptor.path}, condition "
                        f"{condition.name!r}, task {task_index} (seed {seed}): {error}"
                    ) from error
                records.append(
                    {
                        "task_index": task_index,
                        "seed": seed,
                        "fingerprint": task.fingerprint,
                        "active_features": task.active_features,
                        "metrics": metrics,
                    }
                )
                if progress:
                    print(
                        f"\r{descriptor.label} | {condition.name} | "
                        f"{task_index + 1}/{datasets}",
                        end="",
                        flush=True,
                    )
            if progress:
                print()
            condition_results.append(
                {
                    "condition": condition.as_dict(),
                    "prior_config": normalize_config(prior_config),
                    "summary": _summarize_task_metrics(records),
                    "tasks": records,
                }
            )
        model_results.append(
            {
                "checkpoint": descriptor.as_dict(),
                "conditions": condition_results,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pairing_caveat = (
        "Graph-U mechanisms, converters, final preprocessing, feature retention, "
        "and optional dataset filtering are fitted on support rows only. Thus a "
        "same-seed identity/shift realization preserves support while changing "
        "query U. Classification conditions are still distribution-level rather "
        "than guaranteed paired-SCM counterfactuals: shift-dependent class-split "
        "validation can reject one realization and accept a later SCM under the "
        "same seed. Checkpoints are exactly paired within each condition."
    )
    reconstruction = prior_reconstruction_details(reference)
    warnings = [pairing_caveat]
    if not reconstruction["complete"]:
        warnings.append(
            "The reference checkpoint did not record every PriorConfig field; "
            "missing base-prior values were filled from evaluator defaults."
        )

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_provenance": code_provenance(),
        "warnings": warnings,
        "protocol": {
            "name": "graph_u_prior_native",
            "task_type": task_type,
            "datasets_per_condition": datasets,
            "support_size": support_size,
            "query_size": query_size,
            "num_features": num_features,
            "num_classes": None if task_type == "regression" else num_classes,
            "base_seed": base_seed,
            "device": str(device),
            "use_amp": use_amp,
            "softmax_temperature": (
                softmax_temperature if task_type == "classification" else None
            ),
            "stores_generated_tensors": False,
            "query_labels_passed_to_model": False,
            "checkpoint_pairing": "exact within each condition, verified by hashes",
            "condition_pairing": "distribution-level; not paired-SCM counterfactuals",
            "cross_condition_pairing_caveat": pairing_caveat,
            "confidence_intervals": "normal approximation: mean +/- 1.96 * SE",
        },
        "reference_checkpoint": reference.as_dict(),
        "prior_reconstruction": reconstruction,
        "conditions": [condition.as_dict() for condition in conditions],
        "models": model_results,
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
    os.replace(temporary_path, path)


def write_csv_results(
    output_path: Path, result: Mapping[str, Any]
) -> tuple[Path, Path]:
    """Write compact summary and per-task tables beside the JSON result."""

    summary_path = output_path.with_suffix(".summary.csv")
    tasks_path = output_path.with_suffix(".tasks.csv")
    summary_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []

    for model_result in result["models"]:
        checkpoint = model_result["checkpoint"]
        for condition_result in model_result["conditions"]:
            condition_name = condition_result["condition"]["name"]
            for metric, statistics in condition_result["summary"].items():
                if metric == "tasks":
                    continue
                summary_rows.append(
                    {
                        "checkpoint": checkpoint["path"],
                        "label": checkpoint["label"],
                        "step": checkpoint["curr_step"],
                        "condition": condition_name,
                        "metric": metric,
                        **statistics,
                    }
                )
            for task in condition_result["tasks"]:
                base = {
                    "checkpoint": checkpoint["path"],
                    "label": checkpoint["label"],
                    "step": checkpoint["curr_step"],
                    "condition": condition_name,
                    "task_index": task["task_index"],
                    "seed": task["seed"],
                    "fingerprint": task["fingerprint"],
                    "active_features": task["active_features"],
                }
                for metric, value in task["metrics"].items():
                    task_rows.append({**base, "metric": metric, "value": value})

    summary_fields = [
        "checkpoint",
        "label",
        "step",
        "condition",
        "metric",
        "count",
        "mean",
        "std",
        "standard_error",
        "ci95_low",
        "ci95_high",
        "min",
        "max",
    ]
    task_fields = [
        "checkpoint",
        "label",
        "step",
        "condition",
        "task_index",
        "seed",
        "fingerprint",
        "active_features",
        "metric",
        "value",
    ]
    for path, rows, fields in (
        (summary_path, summary_rows, summary_fields),
        (tasks_path, task_rows, task_fields),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, path)
    return summary_path, tasks_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tabicl.evaluation",
        description=(
            "Evaluate TabICL checkpoints on deterministic Graph-U tasks generated "
            "on demand. Only metrics, seeds, configs, and tensor hashes are saved."
        ),
    )
    parser.add_argument("checkpoints", nargs="+", help="Checkpoint files to compare")
    parser.add_argument(
        "--reference-checkpoint",
        default=None,
        help=(
            "Checkpoint supplying the base prior and matched shift. Defaults to the "
            "first metadata-aware Graph-U checkpoint, otherwise the first checkpoint."
        ),
    )
    parser.add_argument(
        "--condition",
        action="append",
        default=None,
        help=(
            "Repeat for each condition: ordinary, identity, matched, or "
            "NAME:LOCATION:SCALE[:FORCE_GAUSSIAN]. Defaults to ordinary+identity "
            "and, when available, matched."
        ),
    )
    parser.add_argument("--datasets", type=int, default=100)
    parser.add_argument("--support-size", type=int, default=128)
    parser.add_argument("--query-size", type=int, default=128)
    parser.add_argument("--features", type=int, default=10)
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use accelerator mixed precision. Defaults to false on CPU, true otherwise.",
    )
    parser.add_argument("--softmax-temperature", type=float, default=1.0)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help=(
            "Maximum generation attempts per task in every condition. Defaults to "
            "the reference Graph-U limit, normally 1000."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("graph_u_evaluation.json"),
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Write only JSON, without summary and per-task CSV files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing result files. By default, evaluation refuses to overwrite.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    descriptors = disambiguate_checkpoint_labels(
        [inspect_checkpoint(path) for path in args.checkpoints]
    )
    task_types = {descriptor.task_type for descriptor in descriptors}
    if len(task_types) != 1:
        raise ValueError(
            "The checkpoint list mixes classification and regression architectures; "
            "evaluate them in separate commands."
        )
    reference = choose_reference_checkpoint(descriptors, args.reference_checkpoint)
    specs = (
        [parse_condition(value) for value in args.condition]
        if args.condition
        else default_condition_specs(reference)
    )
    conditions = resolve_conditions(specs, reference)
    device = resolve_device(args.device)
    use_amp = args.amp if args.amp is not None else device.type != "cpu"
    output_path = args.output.expanduser().resolve()
    output_paths = [output_path]
    if not args.no_csv:
        output_paths.extend(
            [
                output_path.with_suffix(".summary.csv"),
                output_path.with_suffix(".tasks.csv"),
            ]
        )
    existing_paths = [path for path in output_paths if path.exists()]
    if existing_paths and not getattr(args, "overwrite", False):
        raise FileExistsError(
            "Refusing to overwrite existing evaluation output: "
            + ", ".join(str(path) for path in existing_paths)
            + ". Pass --overwrite to replace it."
        )

    reconstruction = prior_reconstruction_details(reference)
    if not args.quiet and not reconstruction["complete"]:
        print(
            "Warning: reference prior metadata is incomplete; evaluator defaults "
            "will fill the missing PriorConfig fields."
        )

    result = evaluate(
        descriptors=descriptors,
        reference=reference,
        conditions=conditions,
        datasets=args.datasets,
        support_size=args.support_size,
        query_size=args.query_size,
        num_features=args.features,
        num_classes=args.classes,
        base_seed=args.seed,
        device=device,
        use_amp=use_amp,
        softmax_temperature=args.softmax_temperature,
        max_attempts=args.max_attempts,
        progress=not args.quiet,
    )
    _atomic_write_json(output_path, result)
    if not args.no_csv:
        summary_path, tasks_path = write_csv_results(output_path, result)
        if not args.quiet:
            print(f"Saved summary CSV to {summary_path}")
            print(f"Saved per-task CSV to {tasks_path}")
    if not args.quiet:
        print(f"Saved evaluation JSON to {output_path}")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_from_args(args)
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        parser.error(str(error))
