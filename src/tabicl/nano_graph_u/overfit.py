"""Diagnostic: repeatedly fit one *fixed* Graph-U table with a fresh Nano model.

This intentionally trains on the selected table's query labels. It tests
whether the model and optimization path can memorize a single task; its score
is *not* an unbiased evaluation or a generalization estimate. The input dump
is opened read-only, and no model checkpoint is written.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .data import NanoPriorDumpLoader, _require_h5py
from .model import NanoTabPFNModel
from .train import PAPER_MODEL_CONFIG, _schedulefree_adamw, _sha256_file


def _query_metrics(
    model: NanoTabPFNModel,
    x: torch.Tensor,
    y: torch.Tensor,
    split: int,
) -> dict[str, float]:
    """Measure the fixed query rows while supplying only support labels."""
    model.eval()
    with torch.no_grad():
        logits = model((x, y[:, :split].float()), train_test_split_index=split)
        targets = y[:, split:]
        if tuple(logits.shape) != (*targets.shape, 2):
            raise ValueError("Nano output shape does not match fixed-table targets")
        nll = float(F.cross_entropy(logits.reshape(-1, 2), targets.reshape(-1)))
        positive_probs = logits.softmax(dim=-1)[..., 1].reshape(-1).cpu().numpy()
        labels = targets.reshape(-1).cpu().numpy()
        predictions = logits.argmax(dim=-1).reshape(-1).cpu().numpy()
    if not math.isfinite(nll) or not np.isfinite(positive_probs).all():
        raise FloatingPointError("Non-finite fixed-table predictions")
    if len(np.unique(labels)) != 2:
        raise ValueError("Selected query partition must contain both classes")
    return {
        "query_nll": nll,
        "query_roc_auc": float(roc_auc_score(labels, positive_probs)),
        "query_accuracy": float(np.mean(predictions == labels)),
        "positive_probability_mean": float(np.mean(positive_probs)),
        "positive_probability_variance": float(np.var(positive_probs)),
        "positive_probability_min": float(np.min(positive_probs)),
        "positive_probability_max": float(np.max(positive_probs)),
    }


def overfit_fixed_table(
    dump_path: str | Path,
    *,
    table_index: int = 0,
    steps: int = 1000,
    seed: int = 42,
    learning_rate: float = 0.004,
    log_every: int = 50,
    device: str = "cuda",
) -> dict[str, Any]:
    """Repeat a single committed training table for ``steps`` optimizer updates.

    ``table_index`` is zero-based across the entire dump. This diagnostic
    deliberately exposes that table's query labels as loss targets on every
    update; they are never supplied as model input. It never updates the
    production checkpoint or the dump.
    """
    if (
        isinstance(table_index, bool)
        or not isinstance(table_index, int)
        or table_index < 0
    ):
        raise ValueError("table_index must be a non-negative integer")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if isinstance(log_every, bool) or not isinstance(log_every, int) or log_every < 1:
        raise ValueError("log_every must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; use --device cpu for a local smoke test"
        )

    dump_path = Path(dump_path).expanduser().resolve()
    if not dump_path.is_file():
        raise FileNotFoundError(f"Nano prior dump not found: {dump_path}")
    # Read only the batch containing the requested table. The standard loader
    # verifies all per-table boundaries and labels before we select one table.
    with _require_h5py().File(dump_path, "r") as stream:
        if "metadata_json" not in stream.attrs:
            raise ValueError("file is not a Nano-Graph-U prior dump")
        batch_size = int(json.loads(stream.attrs["metadata_json"])["batch_size"])
    info = NanoPriorDumpLoader(dump_path, device="cpu", batch_size=batch_size)
    if info.committed_steps != int(info.metadata["steps"]):
        raise ValueError("fixed-table diagnostic requires a completed prior dump")
    committed_tables = info.committed_steps * batch_size
    if table_index >= committed_tables:
        raise ValueError(
            f"table_index {table_index} is outside {committed_tables} committed tables"
        )
    batch_step, within_batch = divmod(table_index, batch_size)
    loader = NanoPriorDumpLoader(
        dump_path,
        batch_size=batch_size,
        start_step=batch_step,
        num_steps=batch_step + 1,
        device="cpu",
    )
    batch = next(iter(loader))
    split = int(batch["train_test_split_index"])
    compute_device = torch.device(device)
    x = batch["x"][within_batch : within_batch + 1].to(
        device=compute_device, dtype=torch.float32
    )
    y = batch["y"][within_batch : within_batch + 1].to(
        device=compute_device, dtype=torch.long
    )
    if not 0 < split < y.shape[1]:
        raise ValueError("Selected table has an invalid support/query boundary")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model_config = dict(PAPER_MODEL_CONFIG)
    model = NanoTabPFNModel(**model_config).to(compute_device).float()
    initial_parameters = [
        parameter.detach().clone() for parameter in model.parameters()
    ]
    optimizer = _schedulefree_adamw(model.parameters(), learning_rate=learning_rate)

    history = [{"step": 0, **_query_metrics(model, x, y, split)}]
    print(
        f"fixed_table={table_index} support={split} query={y.shape[1] - split} "
        f"step=0/{steps} query_nll={history[0]['query_nll']:.6f}",
        flush=True,
    )
    optimizer.train()
    model.train()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model((x, y[:, :split].float()), train_test_split_index=split)
        targets = y[:, split:]
        loss = F.cross_entropy(logits.reshape(-1, 2), targets.reshape(-1))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite fixed-table loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % log_every == 0 or step == steps:
            # Report the schedule-free averaged weights used for evaluation,
            # then restore training weights before the next update.
            optimizer.eval()
            try:
                metrics = _query_metrics(model, x, y, split)
                history.append({"step": step, **metrics})
                print(
                    f"step={step}/{steps} query_nll={metrics['query_nll']:.6f} "
                    f"query_auc={metrics['query_roc_auc']:.6f}",
                    flush=True,
                )
            finally:
                if step != steps:
                    optimizer.train()
                    model.train()

    squared_delta = sum(
        float(torch.sum((parameter.detach() - original) ** 2).cpu())
        for parameter, original in zip(
            model.parameters(), initial_parameters, strict=True
        )
    )
    result = {
        "format": "nano_graph_u_fixed_table_overfit_v1",
        "purpose": "memorization diagnostic; query labels reused as loss targets, not a generalization score",
        "dump": str(dump_path),
        "dump_sha256": _sha256_file(dump_path),
        "dump_metadata": info.metadata,
        "table_index": table_index,
        "batch_step": batch_step,
        "within_batch_index": within_batch,
        "support_rows": split,
        "query_rows": int(y.shape[1] - split),
        "support_positive_count": int(y[:, :split].sum().item()),
        "query_positive_count": int(y[:, split:].sum().item()),
        "steps": steps,
        "seed": seed,
        "device": str(compute_device),
        "model_config": model_config,
        "learning_rate": learning_rate,
        "optimizer": "schedulefree.AdamWScheduleFree",
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "parameter_change_l2": math.sqrt(squared_delta),
        "history": history,
        "initial": history[0],
        "final": history[-1],
    }
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dump", required=True, help="Existing, completed training HDF5 dump"
    )
    parser.add_argument(
        "--output", required=True, help="New JSON report path; never overwritten"
    )
    parser.add_argument("--table-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.004)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {output}")
    result = overfit_fixed_table(
        args.dump,
        table_index=args.table_index,
        steps=args.steps,
        seed=args.seed,
        learning_rate=args.learning_rate,
        log_every=args.log_every,
        device=args.device,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(f"report={output}", flush=True)


if __name__ == "__main__":
    main()
