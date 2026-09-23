"""Read-only prediction diagnostics on a frozen Nano-Graph-U prior dump.

This is not a replacement for the independent evaluation.  It exposes
per-table predictions and compares them with an ExtraTrees model fitted only
on that table's support rows.  The output JSON is create-only, and neither
the dump nor any checkpoint is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score

from .data import NanoPriorDumpLoader
from .model import NanoTabPFNModel
from .train import _schedulefree_adamw


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scores(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    margin: np.ndarray | None = None,
    exact_nll: float | None = None,
) -> dict[str, float | int | bool | None]:
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if targets.ndim != 1 or probabilities.shape != targets.shape:
        raise ValueError("expected matching one-dimensional query targets and scores")
    if not np.all(np.isfinite(probabilities)) or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ValueError("non-finite or out-of-range query probabilities")
    predicted = (probabilities > 0.5).astype(np.int64)
    correct_prob = np.where(targets == 1, probabilities, 1 - probabilities)
    result: dict[str, float | int | bool | None] = {
        "query_rows": len(targets),
        "query_positive_fraction": float(np.mean(targets)),
        "roc_auc": float(roc_auc_score(targets, probabilities))
        if len(np.unique(targets)) == 2
        else None,
        "nll": exact_nll
        if exact_nll is not None
        else float(-np.log(np.clip(correct_prob, 1e-12, 1)).mean()),
        "accuracy": float(np.mean(predicted == targets)),
        "predicted_positive_fraction": float(np.mean(predicted)),
        "constant_predicted_class": bool(np.all(predicted == predicted[0])),
        "positive_probability_mean": float(np.mean(probabilities)),
        "positive_probability_std": float(np.std(probabilities)),
        "positive_probability_min": float(np.min(probabilities)),
        "positive_probability_max": float(np.max(probabilities)),
        "positive_probability_range": float(np.ptp(probabilities)),
    }
    if margin is not None:
        margin = np.asarray(margin, dtype=np.float64)
        if margin.shape != targets.shape or not np.all(np.isfinite(margin)):
            raise ValueError("invalid query logit margins")
        result["logit_margin_std"] = float(np.std(margin))
        result["logit_margin_range"] = float(np.ptp(margin))
    return result


def _summary(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    if not rows:
        raise ValueError("cannot summarize zero tables")
    numeric = (
        "roc_auc",
        "nll",
        "accuracy",
        "query_positive_fraction",
        "predicted_positive_fraction",
        "positive_probability_mean",
        "positive_probability_std",
        "positive_probability_range",
        "logit_margin_std",
        "logit_margin_range",
    )
    result: dict[str, float | int | None] = {"task_count": len(rows)}
    for name in numeric:
        values = [row[name] for row in rows if row.get(name) is not None]
        if values:
            result[f"mean_{name}"] = float(np.mean(values))
    result["constant_predicted_class_fraction"] = float(
        np.mean([row["constant_predicted_class"] for row in rows])
    )
    result["auc_task_count"] = sum(row["roc_auc"] is not None for row in rows)
    return result


def _score_model(
    model: NanoTabPFNModel,
    batches: list[tuple[torch.Tensor, torch.Tensor, int, int]],
    compute_device: torch.device,
) -> list[dict[str, Any]]:
    """Score one model-weight view on the already loaded frozen batches."""
    model.to(compute_device).eval()
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for x_batch, y_batch, split, offset in batches:
            x = x_batch.to(compute_device)
            support_y = y_batch[:, :split].to(compute_device).float()
            logits = model((x, support_y), train_test_split_index=split)
            if logits.shape != (len(x_batch), len(y_batch[0]) - split, 2):
                raise ValueError("checkpoint produced an unexpected query shape")
            probabilities = logits.softmax(dim=-1)[..., 1].cpu().numpy()
            margins = (logits[..., 1] - logits[..., 0]).cpu().numpy()
            query_targets = y_batch[:, split:].to(compute_device)
            log_probs = logits.log_softmax(dim=-1)
            exact_nll = (
                -log_probs.gather(-1, query_targets.unsqueeze(-1))
                .squeeze(-1)
                .mean(dim=1)
                .cpu()
                .numpy()
            )
            for index in range(len(x_batch)):
                y = y_batch[index].numpy()
                row = _scores(
                    y[split:],
                    probabilities[index],
                    margin=margins[index],
                    exact_nll=float(exact_nll[index]),
                )
                row.update(
                    task_index=offset + index,
                    support_rows=split,
                    support_positive_fraction=float(np.mean(y[:split])),
                )
                rows.append(row)
    return rows


def _checkpoint_weight_views(
    checkpoint: dict[str, Any],
    compute_device: torch.device,
    *,
    include_training_weights: bool,
    checkpoint_label: str,
) -> list[tuple[str, NanoTabPFNModel]]:
    """Load saved averaged weights and optionally reconstruct live weights."""
    model_config = checkpoint["model_config"]
    averaged_model = NanoTabPFNModel(**model_config)
    averaged_model.load_state_dict(checkpoint["model"])
    views = [("schedulefree_averaged_x", averaged_model)]
    if not include_training_weights:
        return views

    train_config = checkpoint.get("train_config", {})
    if "optimizer" not in checkpoint:
        raise ValueError(
            "checkpoint has no optimizer state for training-weight reconstruction: "
            f"{checkpoint_label}"
        )
    if "learning_rate" not in train_config:
        raise ValueError(
            "checkpoint has no learning rate for training-weight reconstruction: "
            f"{checkpoint_label}"
        )
    training_model = NanoTabPFNModel(**model_config).to(compute_device).float()
    training_model.load_state_dict(checkpoint["model"])
    optimizer = _schedulefree_adamw(
        training_model.parameters(),
        learning_rate=float(train_config["learning_rate"]),
    )
    optimizer.load_state_dict(checkpoint["optimizer"])
    if not all(not group.get("train_mode", True) for group in optimizer.param_groups):
        raise ValueError("checkpoint optimizer was not saved in evaluation mode")
    optimizer.train()
    views.append(("schedulefree_training_y", training_model))
    return views


def diagnose_dump(
    checkpoint_paths: list[str | Path],
    dump_path: str | Path,
    *,
    max_tasks: int = 128,
    device: str = "cpu",
    baseline: bool = True,
    baseline_trees: int = 100,
    baseline_seed: int = 0,
    include_training_weights: bool = False,
) -> dict[str, Any]:
    """Score fixed query rows without fitting on them or updating checkpoints."""
    if not checkpoint_paths:
        raise ValueError("at least one checkpoint is required")
    if max_tasks < 1 or baseline_trees < 1:
        raise ValueError("max_tasks and baseline_trees must be positive")
    dump_path = Path(dump_path).expanduser().resolve()
    with h5py.File(dump_path, "r") as file:
        if "metadata_json" not in file.attrs:
            raise ValueError("not a Nano-Graph-U prior dump")
        batch_size = int(json.loads(file.attrs["metadata_json"])["batch_size"])
    loader = NanoPriorDumpLoader(dump_path, batch_size=batch_size, device="cpu")
    if loader.committed_steps != int(loader.metadata["steps"]):
        raise ValueError("diagnostic requires a complete frozen evaluation dump")
    batches: list[tuple[torch.Tensor, torch.Tensor, int, int]] = []
    task_count = 0
    for batch in loader:
        count = min(max_tasks - task_count, batch["x"].shape[0])
        batches.append(
            (
                batch["x"][:count],
                batch["y"][:count],
                int(batch["train_test_split_index"]),
                task_count,
            )
        )
        task_count += count
        if task_count == max_tasks:
            break
    if not batches:
        raise ValueError("dump has no committed tasks")

    dump_hash = _sha256(dump_path)
    output: dict[str, Any] = {
        "format": "nano_graph_u_diagnostic_v2",
        "dump": str(dump_path),
        "dump_sha256": dump_hash,
        "dump_metadata": loader.metadata,
        "task_count": task_count,
        "models": [],
    }
    if baseline:
        baseline_rows: list[dict[str, Any]] = []
        for x_batch, y_batch, split, offset in batches:
            for index in range(len(x_batch)):
                x = x_batch[index].numpy()
                y = y_batch[index].numpy()
                forest = ExtraTreesClassifier(
                    n_estimators=baseline_trees,
                    random_state=baseline_seed,
                    n_jobs=1,
                )
                forest.fit(x[:split], y[:split])
                class_one = int(np.flatnonzero(forest.classes_ == 1)[0])
                probabilities = forest.predict_proba(x[split:])[:, class_one]
                row = _scores(y[split:], probabilities)
                row.update(
                    task_index=offset + index,
                    support_rows=split,
                    support_positive_fraction=float(np.mean(y[:split])),
                )
                baseline_rows.append(row)
        output["baseline"] = {
            "name": "ExtraTreesClassifier",
            "fit": "each task's support rows only",
            "n_estimators": baseline_trees,
            "random_state": baseline_seed,
            "summary": _summary(baseline_rows),
            "per_task": baseline_rows,
        }

    compute_device = torch.device(device)
    if compute_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu")
    for checkpoint_arg in checkpoint_paths:
        checkpoint_path = Path(checkpoint_arg).expanduser().resolve()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("format") != "nano_graph_u_v1":
            raise ValueError(f"not a Nano-Graph-U v1 checkpoint: {checkpoint_path}")
        model_config = checkpoint["model_config"]
        if int(model_config["num_outputs"]) != 2:
            raise ValueError("diagnostic currently supports binary Nano tasks only")
        train_config = checkpoint.get("train_config", {})
        matches_training_dump = train_config.get("dump_sha256") == dump_hash or (
            train_config.get("dump_path") is not None
            and Path(train_config["dump_path"]).resolve() == dump_path
        )
        views = _checkpoint_weight_views(
            checkpoint,
            compute_device,
            include_training_weights=include_training_weights,
            checkpoint_label=str(checkpoint_path),
        )

        checkpoint_hash = _sha256(checkpoint_path)
        for weight_view, model in views:
            model_rows = _score_model(model, batches, compute_device)
            output["models"].append(
                {
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": checkpoint_hash,
                    "checkpoint_step": int(checkpoint["step"]),
                    "weight_view": weight_view,
                    "matches_training_dump": matches_training_dump,
                    "summary": _summary(model_rows),
                    "per_task": model_rows,
                }
            )
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--dump", required=True)
    parser.add_argument("--max-tasks", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument(
        "--include-training-weights",
        action="store_true",
        help="also reconstruct and score ScheduleFree's live training weights",
    )
    parser.add_argument("--baseline-trees", type=int, default=100)
    parser.add_argument(
        "--output", help="New JSON path; existing files are never overwritten"
    )
    args = parser.parse_args(argv)
    result = diagnose_dump(
        args.checkpoint,
        args.dump,
        max_tasks=args.max_tasks,
        device=args.device,
        baseline=not args.no_baseline,
        baseline_trees=args.baseline_trees,
        include_training_weights=args.include_training_weights,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        output_path = Path(args.output)
        with output_path.open("x", encoding="utf-8") as stream:
            stream.write(rendered + "\n")
        print(
            json.dumps(
                {
                    "output": str(output_path),
                    "task_count": result["task_count"],
                    "models": [
                        {
                            "checkpoint": model["checkpoint"],
                            "checkpoint_step": model["checkpoint_step"],
                            "weight_view": model["weight_view"],
                            "matches_training_dump": model["matches_training_dump"],
                            "summary": model["summary"],
                        }
                        for model in result["models"]
                    ],
                    "baseline_summary": result.get("baseline", {}).get("summary"),
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
    else:
        print(rendered)


if __name__ == "__main__":
    main()
