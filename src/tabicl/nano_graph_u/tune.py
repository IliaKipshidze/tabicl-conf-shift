"""Run fixed-budget learning-rate trials on the saved Nano Graph-U banks."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from tabicl.prior.graph_lib._config import PriorConfig

from .data import _SEED_SCHEME, NanoPriorDumpLoader
from .evaluate import evaluate_checkpoint
from .train import PAPER_MODEL_CONFIG, _sha256_file, train_model

LEARNING_RATES = (
    0.0001,
    0.0002,
    0.0003,
    0.0005,
    0.0007,
    0.001,
    0.0015,
    0.002,
    0.003,
    0.004,
)
TRIAL_FORMAT = "nano_graph_u_lr_trial_v1"
TRAIN_STEPS = 2500
VALIDATION_STEPS = 100
BATCH_SIZE = 32


def trial_spec(
    task_id: int, data_root: str | Path, output_root: str | Path
) -> dict[str, Any]:
    """Map the 100 array tasks to five repetitions, two conditions and ten rates."""
    if (
        isinstance(task_id, bool)
        or not isinstance(task_id, int)
        or not 1 <= task_id <= 100
    ):
        raise ValueError("task_id must be an integer from 1 to 100")
    repetition, slot = divmod(task_id - 1, 20)
    repetition += 1
    condition = "identity" if slot < 10 else "shift"
    lr_index = slot % 10
    learning_rate = LEARNING_RATES[lr_index]
    tag = format(learning_rate, "g").replace(".", "p")
    data_root = Path(data_root).expanduser().resolve()
    candidate = (
        Path(output_root).expanduser().resolve()
        / f"rep_{repetition}"
        / condition
        / f"lr_{tag}"
    )
    return {
        "task_id": task_id,
        "repetition": repetition,
        "condition": condition,
        "model_seed": 41 + repetition,
        "lr_index": lr_index,
        "learning_rate": learning_rate,
        "train_dump": str(
            data_root
            / f"rep_{repetition}"
            / f"{condition}_train_seed{1000 + repetition}_2500steps_b32_r150_f5.h5"
        ),
        "validation_dump": str(
            data_root
            / "validation"
            / f"{condition}_validation_seed3001_100steps_b32_r150_f5.h5"
        ),
        "checkpoint_dir": str(candidate / "checkpoints"),
        "result_path": str(candidate / "validation.json"),
    }


def _validate_metadata(
    metadata: dict[str, Any], condition: str, split: str, repetition: int | None = None
) -> int:
    if condition not in ("identity", "shift") or split not in ("train", "validation"):
        raise ValueError(
            "Only identity/shift training and validation banks are accepted"
        )
    if split == "train" and repetition not in range(1, 6):
        raise ValueError("Training banks require a repetition from 1 to 5")
    steps = TRAIN_STEPS if split == "train" else VALIDATION_STEPS
    seed = 1000 + repetition if split == "train" else 3001
    expected = {
        "format_version": 1,
        "steps": steps,
        "batch_size": BATCH_SIZE,
        "seed": seed,
        "seed_scheme": _SEED_SCHEME,
        "rows": 150,
        "features": 5,
        "batch_size_per_gp": 4,
        "batch_size_per_subgp": 4,
        "seq_len_per_gp": False,
        "prior_type": "graph_scm",
        "min_train_size": 0.3,
        "max_train_size": 0.9,
        "n_jobs": 1,
        "max_step_retries": 8,
    }
    prior_config = PriorConfig(
        graph_u_enabled=True,
        graph_u_structure_mode="add_root",
        graph_u_query_location=0.0 if condition == "identity" else 2.0,
        graph_u_query_scale=1.0 if condition == "identity" else 1.5,
        graph_u_force_gaussian=False,
        graph_u_max_attempts=1000,
        filter_unpredictable_graphs=True,
        filter_unpredictable_datasets=True,
        allow_act_warping=False,
    )
    expected["prior_config"] = asdict(prior_config)
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Unexpected {condition} {split} bank metadata: {key}")
    fingerprint = metadata.get("generator_source_sha256")
    if not _is_digest(fingerprint):
        raise ValueError("Bank has no valid generator source fingerprint")
    return steps


def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def validate_bank(
    path: str | Path, condition: str, split: str, repetition: int | None = None
) -> dict[str, Any]:
    """Require complete, correctly configured banks before any optimization."""
    loader = NanoPriorDumpLoader(path, batch_size=BATCH_SIZE, device="cpu")
    steps = _validate_metadata(loader.metadata, condition, split, repetition)
    if loader.committed_steps != steps:
        raise ValueError(
            f"Incomplete {split} bank: {loader.committed_steps}/{steps} batches committed"
        )
    return loader.metadata


def validate_trial_result(
    result: dict[str, Any],
    spec: dict[str, Any],
    *,
    verify_files: bool = True,
    hash_cache: dict[tuple[str, int, int, int], str] | None = None,
) -> float:
    """Reject stale/incomplete trial reports; return the selection AUROC."""

    def digest(path: Path) -> str:
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if hash_cache is None:
            return _sha256_file(path)
        if key not in hash_cache:
            hash_cache[key] = _sha256_file(path)
        return hash_cache[key]

    if result.get("format") != TRIAL_FORMAT or result.get("trial") != spec:
        raise ValueError("Trial result belongs to a different candidate/configuration")
    if result.get("selection_metric") != "roc_auc":
        raise ValueError("Trial selection metric must be roc_auc")
    report = result.get("validation_report", {})
    expected_checkpoint = str(Path(spec["checkpoint_dir"]) / "latest.pt")
    if (
        report.get("format") != "nano_graph_u_evaluation_v1"
        or report.get("checkpoint") != expected_checkpoint
        or report.get("checkpoint_step") != TRAIN_STEPS
        or report.get("dump") != spec["validation_dump"]
        or report.get("task_count") != VALIDATION_STEPS * BATCH_SIZE
        or report.get("metric_task_counts", {}).get("roc_auc")
        != VALIDATION_STEPS * BATCH_SIZE
        or not _is_digest(report.get("checkpoint_sha256"))
        or not _is_digest(report.get("dump_sha256"))
    ):
        raise ValueError(
            "Trial must score the final checkpoint on all 3,200 matching validation tasks"
        )
    _validate_metadata(report.get("dump_metadata", {}), spec["condition"], "validation")
    score = report.get("metrics", {}).get("roc_auc")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
        or not 0 <= score <= 1
    ):
        raise ValueError("Trial has no finite validation AUROC in [0, 1]")
    if verify_files:
        train_metadata = validate_bank(
            spec["train_dump"], spec["condition"], "train", spec["repetition"]
        )
        validation_metadata = validate_bank(
            spec["validation_dump"], spec["condition"], "validation"
        )
        if (
            train_metadata["generator_source_sha256"]
            != validation_metadata["generator_source_sha256"]
        ):
            raise ValueError(
                "Training and validation banks were generated by different code"
            )
        if report["dump_metadata"] != validation_metadata or report[
            "dump_sha256"
        ] != digest(Path(spec["validation_dump"])):
            raise ValueError("Validation bank changed after this trial was scored")
        checkpoint_path = Path(expected_checkpoint)
        if report["checkpoint_sha256"] != digest(checkpoint_path):
            raise ValueError("Checkpoint changed after this trial was scored")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        expected_training = {
            "seed": spec["model_seed"],
            "batch_size": BATCH_SIZE,
            "learning_rate": spec["learning_rate"],
            "optimizer": "schedulefree.AdamWScheduleFree",
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "loss": "query_cross_entropy",
            "precision": "float32",
            "dump_path": spec["train_dump"],
            "dump_sha256": digest(Path(spec["train_dump"])),
        }
        if (
            checkpoint.get("format") != "nano_graph_u_v1"
            or checkpoint.get("step") != TRAIN_STEPS
            or checkpoint.get("model_config") != PAPER_MODEL_CONFIG
            or checkpoint.get("dump_metadata") != train_metadata
            or checkpoint.get("prior_config") != train_metadata["prior_config"]
            or checkpoint.get("train_config") != expected_training
        ):
            raise ValueError("Checkpoint does not match this fixed-budget LR trial")
    return float(score)


def run_trial(
    task_id: int,
    data_root: str | Path,
    output_root: str | Path,
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    """Train one candidate and score only its matching validation bank."""
    spec = trial_spec(task_id, data_root, output_root)
    training = validate_bank(
        spec["train_dump"], spec["condition"], "train", spec["repetition"]
    )
    validation = validate_bank(spec["validation_dump"], spec["condition"], "validation")
    if training["generator_source_sha256"] != validation["generator_source_sha256"]:
        raise ValueError(
            "Training and validation banks were generated by different code"
        )
    result_path = Path(spec["result_path"])
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        validate_trial_result(result, spec)
        print(f"Already scored task={task_id} result={result_path}", flush=True)
        return result
    checkpoint = train_model(
        spec["train_dump"],
        spec["checkpoint_dir"],
        steps=TRAIN_STEPS,
        batch_size=BATCH_SIZE,
        seed=spec["model_seed"],
        device=device,
        learning_rate=spec["learning_rate"],
        save_every=250,
        log_every=25,
        resume=(Path(spec["checkpoint_dir"]) / "latest.pt").is_file(),
    )
    report = evaluate_checkpoint(
        checkpoint, spec["validation_dump"], batch_size=BATCH_SIZE, device=device
    )
    result = {
        "format": TRIAL_FORMAT,
        "trial": spec,
        "selection_metric": "roc_auc",
        "validation_report": report,
    }
    validate_trial_result(result, spec)
    temporary = result_path.with_name(f".{result_path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, result_path)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        f"TRIAL_COMPLETE task={task_id} validation_auroc={report['metrics']['roc_auc']:.6f} result={result_path}",
        flush=True,
    )
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    trial = subcommands.add_parser(
        "trial", help="Train and validate one of the 100 candidates"
    )
    trial.add_argument("--task-id", type=int, required=True)
    trial.add_argument("--data-root", type=Path, required=True)
    trial.add_argument("--output-root", type=Path, required=True)
    trial.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    run_trial(args.task_id, args.data_root, args.output_root, device=args.device)


if __name__ == "__main__":
    main()
