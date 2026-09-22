"""Evaluate a Nano-Graph-U checkpoint on a frozen synthetic prior dump.

The dump is generated separately from the training dump, so the same tasks can
be scored by identity- and shift-trained checkpoints without regenerating them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from .data import NanoPriorDumpLoader
from .model import NanoTabPFNModel


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    dump_path: str | Path,
    *,
    batch_size: int = 32,
    max_batches: int | None = None,
    device: str = "cpu",
    allow_training_dump: bool = False,
) -> dict[str, Any]:
    """Score each query set, then average metrics equally across tables."""

    checkpoint_path = Path(checkpoint_path).resolve()
    dump_path = Path(dump_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "nano_graph_u_v1":
        raise ValueError("Expected a Nano-Graph-U v1 checkpoint")
    dump_sha256 = _sha256(dump_path)
    recorded_training_dump = checkpoint.get("train_config", {}).get("dump_path")
    if not allow_training_dump and (
        (
            recorded_training_dump is not None
            and Path(recorded_training_dump).resolve() == dump_path
        )
        or checkpoint.get("train_config", {}).get("dump_sha256") == dump_sha256
    ):
        raise ValueError(
            "Refusing to evaluate on the training dump; generate an independently "
            "seeded evaluation dump instead"
        )
    model_config = checkpoint["model_config"]
    if int(model_config["num_outputs"]) != 2:
        raise ValueError(
            "The Nano-Graph-U evaluator currently supports binary tasks only"
        )
    model = NanoTabPFNModel(**model_config)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    totals = {
        "roc_auc": 0.0,
        "nll": 0.0,
        "brier": 0.0,
        "accuracy": 0.0,
        "balanced_accuracy": 0.0,
        "macro_f1": 0.0,
    }
    counts = {name: 0 for name in totals}
    tasks = 0
    query_rows = 0
    loader = NanoPriorDumpLoader(
        dump_path,
        num_steps=max_batches,
        batch_size=batch_size,
        device="cpu",
    )

    with torch.inference_mode():
        for batch in loader:
            split = int(batch["train_test_split_index"])
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            logits = model((x, y[:, :split].float()), train_test_split_index=split)
            targets = y[:, split:].long()
            if logits.shape[:2] != targets.shape or logits.shape[-1] != 2:
                raise ValueError(
                    "Model output and query labels have incompatible shapes"
                )
            log_probs = F.log_softmax(logits, dim=-1)
            positive_probs = log_probs[..., 1].exp()
            predictions = logits.argmax(dim=-1)
            task_nll = (
                -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).mean(dim=1)
            )
            task_brier = ((positive_probs - targets.float()) ** 2).mean(dim=1)
            task_accuracy = (predictions == targets).float().mean(dim=1)

            for index in range(x.shape[0]):
                true = targets[index].cpu().numpy()
                pred = predictions[index].cpu().numpy()
                probability = positive_probs[index].cpu().numpy()
                per_task = {
                    "nll": float(task_nll[index].cpu()),
                    "brier": float(task_brier[index].cpu()),
                    "accuracy": float(task_accuracy[index].cpu()),
                    "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
                    "macro_f1": float(
                        f1_score(true, pred, average="macro", zero_division=0)
                    ),
                }
                # AUC is undefined if the query set contains only one class.
                if len(np.unique(true)) == 2:
                    per_task["roc_auc"] = float(roc_auc_score(true, probability))
                for name, value in per_task.items():
                    totals[name] += value
                    counts[name] += 1
                tasks += 1
                query_rows += len(true)

    if tasks == 0:
        raise ValueError("Evaluation dump contains no tasks")
    return {
        "format": "nano_graph_u_evaluation_v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "dump": str(dump_path),
        "dump_sha256": dump_sha256,
        "dump_metadata": loader.metadata,
        "task_count": tasks,
        "query_row_count": query_rows,
        "metric_task_counts": counts,
        "metrics": {
            name: totals[name] / counts[name] if counts[name] else None
            for name in totals
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dump", required=True)
    parser.add_argument(
        "--output", help="New JSON path; existing files are never overwritten"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    result = evaluate_checkpoint(
        args.checkpoint,
        args.dump,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        device=args.device,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        with output_path.open("x", encoding="utf-8") as stream:
            stream.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
