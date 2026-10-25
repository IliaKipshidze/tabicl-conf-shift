"""Evaluate the ten retained winners on the twenty frozen test combinations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from .data import NanoPriorDumpLoader
from .evaluate import evaluate_checkpoint
from .select import WINNERS_FORMAT
from .train import PAPER_MODEL_CONFIG, _sha256_file
from .tune import (
    BATCH_SIZE,
    LEARNING_RATES,
    TRAIN_STEPS,
    _is_digest,
    _validate_metadata,
    trial_spec,
)

__test__ = False
TEST_FORMAT = "nano_graph_u_winner_test_v1"
TEST_STEPS = 100
METRIC_NAMES = ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "nll", "brier")


def test_spec(
    task_id: int,
    data_root: str | Path,
    search_root: str | Path,
    result_root: str | Path,
) -> dict[str, Any]:
    """Map each repetition to identity/shift winners crossed with both test banks."""
    if (
        isinstance(task_id, bool)
        or not isinstance(task_id, int)
        or not 1 <= task_id <= 20
    ):
        raise ValueError("task_id must be an integer from 1 to 20")
    repetition, slot = divmod(task_id - 1, 4)
    repetition += 1
    training_condition = "identity" if slot < 2 else "shift"
    test_condition = "identity" if slot % 2 == 0 else "shift"
    data_root = Path(data_root).expanduser().resolve()
    search_root = Path(search_root).expanduser().resolve()
    result_root = Path(result_root).expanduser().resolve()
    return {
        "task_id": task_id,
        "repetition": repetition,
        "training_condition": training_condition,
        "test_condition": test_condition,
        "model_seed": 41 + repetition,
        "checkpoint": str(
            search_root
            / "winners"
            / f"rep_{repetition}"
            / training_condition
            / "model.pt"
        ),
        "manifest_path": str(search_root / "winners" / "winners.json"),
        "test_dump": str(
            data_root
            / f"rep_{repetition}"
            / f"{test_condition}_test_seed{2000 + repetition}_100steps_b32_r150_f5.h5"
        ),
        "result_path": str(
            result_root
            / f"rep_{repetition}"
            / f"{training_condition}_on_{test_condition}.json"
        ),
    }


def _validate_test_metadata(
    metadata: dict[str, Any], condition: str, repetition: int
) -> None:
    if metadata.get("seed") != 2000 + repetition:
        raise ValueError("Test bank must use this repetition's independent test seed")
    # Test and validation banks have the same scientific configuration and size;
    # only their root seeds differ. Validate a copy without changing provenance.
    normalized = dict(metadata, seed=3001)
    _validate_metadata(normalized, condition, "validation")


def validate_test_bank(
    path: str | Path, condition: str, repetition: int
) -> dict[str, Any]:
    loader = NanoPriorDumpLoader(path, batch_size=BATCH_SIZE, device="cpu")
    _validate_test_metadata(loader.metadata, condition, repetition)
    if loader.committed_steps != TEST_STEPS:
        raise ValueError(
            f"Incomplete test bank: {loader.committed_steps}/{TEST_STEPS} batches committed"
        )
    return loader.metadata


def _validate_winner_entry(
    winner: dict[str, Any], spec: dict[str, Any]
) -> dict[str, Any]:
    data_root = Path(spec["test_dump"]).parents[1]
    search_root = Path(spec["manifest_path"]).parents[1]
    candidate = trial_spec(winner.get("task_id"), data_root, search_root)
    expected = {
        "repetition": spec["repetition"],
        "condition": spec["training_condition"],
        "learning_rate": candidate["learning_rate"],
        "model_seed": spec["model_seed"],
        "checkpoint_step": TRAIN_STEPS,
        "checkpoint": spec["checkpoint"],
        "source_checkpoint": str(Path(candidate["checkpoint_dir"]) / "latest.pt"),
        "validation_dump": candidate["validation_dump"],
        "validation_result": candidate["result_path"],
    }
    if (
        candidate["repetition"] != spec["repetition"]
        or candidate["condition"] != spec["training_condition"]
    ):
        raise ValueError(
            "Winner belongs to a different repetition or training condition"
        )
    if any(winner.get(key) != value for key, value in expected.items()):
        raise ValueError("Winner record does not match the selected trial")
    score = winner.get("validation_roc_auc")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
        or not 0 <= score <= 1
    ):
        raise ValueError("Winner has no finite validation AUROC")
    if not _is_digest(winner.get("checkpoint_sha256")) or not _is_digest(
        winner.get("validation_dump_sha256")
    ):
        raise ValueError("Winner has invalid checkpoint/validation provenance")
    return candidate


def _load_winner(spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
    contents = Path(spec["manifest_path"]).read_bytes()
    manifest = json.loads(contents)
    expected = {
        "format": WINNERS_FORMAT,
        "data_root": str(Path(spec["test_dump"]).parents[1]),
        "output_root": str(Path(spec["manifest_path"]).parents[1]),
        "selection_metric": "roc_auc",
        "validation_policy": "matching_training_condition",
        "training_steps": TRAIN_STEPS,
        "trial_count": 100,
        "learning_rates": list(LEARNING_RATES),
    }
    if any(
        manifest.get(key) != value for key, value in expected.items()
    ) or not _is_digest(manifest.get("generator_source_sha256")):
        raise ValueError(
            "Winner manifest does not match this completed validation selection"
        )
    winners = manifest.get("winners", [])
    if (
        not isinstance(winners, list)
        or len(winners) != 10
        or not all(isinstance(winner, dict) for winner in winners)
    ):
        raise ValueError("Winner manifest must contain exactly ten selected models")
    groups = [(winner.get("repetition"), winner.get("condition")) for winner in winners]
    if sorted(groups) != [
        (rep, cond) for rep in range(1, 6) for cond in ("identity", "shift")
    ]:
        raise ValueError(
            "Winner manifest must contain one model per repetition and condition"
        )
    winner = winners[groups.index((spec["repetition"], spec["training_condition"]))]
    _validate_winner_entry(winner, spec)
    return manifest, winner, hashlib.sha256(contents).hexdigest()


def _digest(path: Path, cache: dict | None) -> str:
    if cache is None:
        return _sha256_file(path)
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if key not in cache:
        cache[key] = _sha256_file(path)
    return cache[key]


def _validate_checkpoint(
    winner: dict[str, Any], spec: dict[str, Any], fingerprint: str, cache: dict | None
) -> None:
    if _digest(Path(spec["checkpoint"]), cache) != winner["checkpoint_sha256"]:
        raise ValueError("Retained winner checkpoint changed after selection")
    checkpoint = torch.load(spec["checkpoint"], map_location="cpu", weights_only=True)
    metadata = checkpoint.get("dump_metadata", {})
    _validate_metadata(
        metadata, spec["training_condition"], "train", spec["repetition"]
    )
    candidate = _validate_winner_entry(winner, spec)
    config = checkpoint.get("train_config", {})
    expected = {
        "seed": spec["model_seed"],
        "batch_size": BATCH_SIZE,
        "learning_rate": winner["learning_rate"],
        "optimizer": "schedulefree.AdamWScheduleFree",
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "loss": "query_cross_entropy",
        "precision": "float32",
        "dump_path": candidate["train_dump"],
    }
    if (
        checkpoint.get("format") != "nano_graph_u_v1"
        or checkpoint.get("step") != TRAIN_STEPS
        or checkpoint.get("model_config") != PAPER_MODEL_CONFIG
        or checkpoint.get("prior_config") != metadata["prior_config"]
        or metadata["generator_source_sha256"] != fingerprint
        or any(config.get(key) != value for key, value in expected.items())
        or not _is_digest(config.get("dump_sha256"))
    ):
        raise ValueError(
            "Retained checkpoint does not match the selected model and training recipe"
        )


def validate_test_result(
    result: dict[str, Any],
    spec: dict[str, Any],
    *,
    verify_files: bool = True,
    hash_cache: dict | None = None,
) -> None:
    """Check one complete test report without selecting or changing any model."""
    if (
        result.get("format") != TEST_FORMAT
        or result.get("test") != spec
        or not _is_digest(result.get("manifest_sha256"))
    ):
        raise ValueError("Test result belongs to a different evaluation/selection")
    winner = result.get("winner", {})
    _validate_winner_entry(winner, spec)
    report = result.get("evaluation_report", {})
    if (
        report.get("format") != "nano_graph_u_evaluation_v1"
        or report.get("checkpoint") != spec["checkpoint"]
        or report.get("checkpoint_sha256") != winner["checkpoint_sha256"]
        or report.get("checkpoint_step") != TRAIN_STEPS
        or report.get("dump") != spec["test_dump"]
        or report.get("task_count") != TEST_STEPS * BATCH_SIZE
        or not _is_digest(report.get("dump_sha256"))
        or not isinstance(report.get("query_row_count"), int)
        or not 0 < report["query_row_count"] < TEST_STEPS * BATCH_SIZE * 150
    ):
        raise ValueError(
            "Test result must score the selected final model on all 3,200 test tasks"
        )
    _validate_test_metadata(
        report.get("dump_metadata", {}), spec["test_condition"], spec["repetition"]
    )
    for name in METRIC_NAMES:
        value = report.get("metrics", {}).get(name)
        if (
            report.get("metric_task_counts", {}).get(name) != TEST_STEPS * BATCH_SIZE
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or (name != "nll" and value > 1)
        ):
            raise ValueError(f"Test result has incomplete or invalid {name} metrics")
    if verify_files:
        manifest, selected, manifest_hash = _load_winner(spec)
        if result["manifest_sha256"] != manifest_hash or winner != selected:
            raise ValueError("Winner selection changed after testing")
        metadata = validate_test_bank(
            spec["test_dump"], spec["test_condition"], spec["repetition"]
        )
        if (
            report["dump_metadata"] != metadata
            or metadata["generator_source_sha256"]
            != manifest["generator_source_sha256"]
            or report["dump_sha256"] != _digest(Path(spec["test_dump"]), hash_cache)
        ):
            raise ValueError("Test bank changed or uses a different generator")
        _validate_checkpoint(
            selected, spec, manifest["generator_source_sha256"], hash_cache
        )


def run_test(
    task_id: int,
    data_root: str | Path,
    search_root: str | Path,
    result_root: str | Path,
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    spec = test_spec(task_id, data_root, search_root, result_root)
    manifest, winner, manifest_hash = _load_winner(spec)
    metadata = validate_test_bank(
        spec["test_dump"], spec["test_condition"], spec["repetition"]
    )
    if metadata["generator_source_sha256"] != manifest["generator_source_sha256"]:
        raise ValueError(
            "Test bank uses a different generator from the selected models"
        )
    _validate_checkpoint(winner, spec, manifest["generator_source_sha256"], None)
    path = Path(spec["result_path"])
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        validate_test_result(result, spec)
        print(f"Already tested task={task_id} result={path}", flush=True)
        return result
    report = evaluate_checkpoint(
        spec["checkpoint"], spec["test_dump"], batch_size=BATCH_SIZE, device=device
    )
    result = {
        "format": TEST_FORMAT,
        "test": spec,
        "manifest_sha256": manifest_hash,
        "winner": winner,
        "evaluation_report": report,
    }
    validate_test_result(result, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        f"TEST_COMPLETE task={task_id} auroc={report['metrics']['roc_auc']:.6f} result={path}",
        flush=True,
    )
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True, type=int)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--search-root", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    run_test(
        args.task_id,
        args.data_root,
        args.search_root,
        args.result_root,
        device=args.device,
    )


if __name__ == "__main__":
    main()
