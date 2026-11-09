"""Tune epoch budgets on existing repetition-one fixed-shift banks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .data import NanoPriorDumpLoader
from .evaluate import evaluate_checkpoint
from .fixed_experiments import FIXED_SHIFTS, _row, _source_roots, experiment_spec
from .sampled import MODES, _lock
from .select import _atomic_write, _retain_checkpoint
from .summarize_tests import _csv_bytes
from .test_winners import METRIC_NAMES, _digest
from .train import BATCH_ORDER_SCHEME, PAPER_MODEL_CONFIG, train_model
from .tune import TRAIN_STEPS, _validate_metadata

EPOCHS = (3, 6, 10)
LEARNING_RATES = (0.0007, 0.001, 0.002)


def task_spec(cluster_root, fixed_root, root, task_id):
    """Map thirty jobs to ten model/data settings at each epoch budget."""
    if isinstance(task_id, bool) or not isinstance(task_id, int) or not 1 <= task_id <= 30:
        raise ValueError("task_id must be an integer from 1 to 30")
    cluster, fixed, output = [Path(p).expanduser().resolve() for p in (cluster_root, fixed_root, root)]
    sources = [fixed, cluster / "tabicl-conf-shift-priors",
               cluster / "tabicl-conf-shift-checkpoints", cluster / "tabicl-conf-shift-evaluations"]
    if output == cluster or output in cluster.parents or any(
        output == source or output in source.parents or source in output.parents for source in sources
    ):
        raise ValueError("Epoch output must be separate from the existing experiment/data roots")
    budget, slot = divmod(task_id - 1, 10)
    epochs = EPOCHS[budget]
    mode = MODES[slot % 2]
    condition = "identity" if slot < 2 else "shift"
    pair_id = None if slot < 2 else (slot - 2) // 2 + 1
    pair = experiment_spec(fixed, pair_id) if pair_id else None
    data = pair["directory"] / "data" / mode if pair else _source_roots(cluster, mode)[0]
    tag = pair["directory"].name if pair else "iid"
    return {
        "task_id": task_id, "epochs": epochs, "total_steps": TRAIN_STEPS * epochs,
        "mode": mode, "condition": condition, "pair_id": pair_id,
        "fixed_shift_config": pair["fixed_shift_config"] if pair else None,
        "repetition": 1, "model_seed": 42,
        "shuffle_batches": True, "batch_order_scheme": BATCH_ORDER_SCHEME,
        "cluster_root": str(cluster), "fixed_root": str(fixed), "root": str(output),
        "directory": str(output / f"epochs_{epochs}" / tag / mode),
        "train_dump": str(data / "rep_1" / f"{condition}_train_seed1001_2500steps_b32_r150_f5.h5"),
        "validation_dump": str(data / "validation" / f"{condition}_validation_seed3001_100steps_b32_r150_f5.h5"),
    }


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


def _bank(path, condition, split, mode, profile=None):
    path = Path(path).resolve()
    loader = NanoPriorDumpLoader(path, batch_size=32)
    metadata = loader.metadata
    if split == "test":
        if metadata.get("seed") != 2001:
            raise ValueError("Test bank must use repetition-one seed 2001")
        checked_metadata, checked_split = dict(metadata, seed=3001), "validation"
    else:
        checked_metadata, checked_split = metadata, split
    expected_steps = _validate_metadata(
        checked_metadata, condition, checked_split, 1, graph_u_mode=mode,
        fixed_shift_config=profile,
    )
    if loader.committed_steps != expected_steps:
        raise ValueError(f"Incomplete {split} bank: {loader.committed_steps}/{expected_steps}: {path}")
    return {"dump": str(path), "metadata": metadata}


def _test_banks(spec):
    banks = []
    for mode in MODES:
        data = _source_roots(Path(spec["cluster_root"]), mode)[0]
        banks.append({"mode": mode, "condition": "identity", "pair_id": None,
                      **_bank(data / "rep_1/identity_test_seed2001_100steps_b32_r150_f5.h5",
                              "identity", "test", mode)})
        for pair_id in ([spec["pair_id"]] if spec["pair_id"] else range(1, 5)):
            pair = experiment_spec(spec["fixed_root"], pair_id)
            banks.append({"mode": mode, "condition": "shift", "pair_id": pair_id,
                          **_bank(pair["directory"] / "data" / mode / "rep_1"
                                  / "shift_test_seed2001_100steps_b32_r150_f5.h5",
                                  "shift", "test", mode, pair["fixed_shift_config"])})
    return banks


def _check_report(report, checkpoint, bank, total_steps, cache, *, task_count=3200):
    if (
        report.get("format") != "nano_graph_u_evaluation_v1"
        or report.get("checkpoint") != str(checkpoint)
        or report.get("checkpoint_sha256") != _digest(Path(checkpoint), cache)
        or report.get("checkpoint_step") != total_steps
        or report.get("dump") != bank["dump"]
        or report.get("dump_sha256") != _digest(Path(bank["dump"]), cache)
        or report.get("dump_metadata") != bank["metadata"]
        or report.get("task_count") != task_count
        or isinstance(report.get("query_row_count"), bool)
        or not isinstance(report.get("query_row_count"), int)
        or not 0 < report["query_row_count"] < task_count * 150
    ):
        raise ValueError(f"Report must match the epoch-budget checkpoint and all {task_count:,} evaluation tasks")
    for metric in METRIC_NAMES:
        value = report.get("metrics", {}).get(metric)
        if (report.get("metric_task_counts", {}).get(metric) != task_count
            or isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (metric != "nll" and value > 1)):
            raise ValueError(f"Incomplete or invalid metric: {metric}")


def _check_checkpoint(checkpoint, spec, rate, training, cache):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected_training = {
        "seed": 42, "batch_size": 32, "learning_rate": rate,
        "optimizer": "schedulefree.AdamWScheduleFree", "weight_decay": 0.0,
        "gradient_clip_norm": 1.0, "loss": "query_cross_entropy", "precision": "float32",
        "dump_path": training["dump"], "dump_sha256": _digest(Path(training["dump"]), cache),
        "epochs": spec["epochs"], "steps_per_epoch": TRAIN_STEPS,
        "shuffle_batches": True, "batch_order_scheme": BATCH_ORDER_SCHEME,
    }
    if (payload.get("format") != "nano_graph_u_v1"
        or payload.get("step") != spec["total_steps"]
        or payload.get("model_config") != PAPER_MODEL_CONFIG
        or payload.get("dump_metadata") != training["metadata"]
        or payload.get("prior_config") != training["metadata"]["prior_config"]
        or payload.get("train_config") != expected_training):
        raise ValueError("Checkpoint does not match this LR, training bank and full epoch budget")


def _candidate(spec, rate, training, validation, device, cache):
    directory = Path(spec["directory"]) / "candidates" / f"lr_{format(rate, 'g').replace('.', 'p')}"
    checkpoint, path = directory / "checkpoints/latest.pt", directory / "validation.json"
    binding = {"task": spec, "learning_rate": rate, "selection_metric": "roc_auc"}
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        if any(result.get(key) != value for key, value in binding.items()):
            raise ValueError(f"Existing validation belongs to a different candidate: {path}")
        report = result["validation_report"]
    else:
        train_model(training["dump"], checkpoint.parent, steps=TRAIN_STEPS, epochs=spec["epochs"],
                    shuffle_batches=True,
                    batch_size=32, seed=42, learning_rate=rate, device=device,
                    save_every=TRAIN_STEPS, log_every=250, resume=checkpoint.is_file())
        _check_checkpoint(checkpoint, spec, rate, training, cache)
        report = evaluate_checkpoint(checkpoint, validation["dump"], batch_size=32, device=device)
    _check_checkpoint(checkpoint, spec, rate, training, cache)
    _check_report(report, checkpoint, validation, spec["total_steps"], cache)
    _json(path, {**binding, "validation_report": report})
    print(f"CANDIDATE_READY task={spec['task_id']} epochs={spec['epochs']} lr={rate:g} "
          f"validation_auroc={report['metrics']['roc_auc']:.6f}", flush=True)
    return {"learning_rate": rate, "validation_roc_auc": report["metrics"]["roc_auc"],
            "source_checkpoint": str(checkpoint), "checkpoint_sha256": report["checkpoint_sha256"],
            "validation_result": str(path)}


def _evaluate(spec, winner, banks, device, cache):
    rows = []
    for bank in banks:
        tag = f"{bank['mode']}_{bank['condition']}_pair_{bank['pair_id'] or 'iid'}"
        path = Path(spec["directory"]) / "tests" / f"{tag}.json"
        binding = {"winner": winner, "test_graph_u_mode": bank["mode"],
                   "test_condition": bank["condition"], "pair_id": bank["pair_id"]}
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if any(saved.get(key) != value for key, value in binding.items()):
                raise ValueError(f"Existing test belongs to a different winner/bank: {path}")
            report = saved["evaluation_report"]
        else:
            if device is None:
                raise FileNotFoundError(f"Completed task is missing its cached test report: {path}")
            report = evaluate_checkpoint(winner["checkpoint"], bank["dump"], batch_size=32, device=device)
        _check_report(report, winner["checkpoint"], bank, spec["total_steps"], cache)
        _json(path, {**binding, "evaluation_report": report})
        pairs = [spec["pair_id"] or bank["pair_id"]] if (spec["pair_id"] or bank["pair_id"]) else range(1, 5)
        for pair_id in pairs:
            location, scale = FIXED_SHIFTS[pair_id - 1]
            rows.append({"epochs": spec["epochs"], "training_steps": spec["total_steps"],
                         "shuffle_batches": True, "batch_order_scheme": BATCH_ORDER_SCHEME,
                         "fixed_location": location, "fixed_scale": scale,
                         **_row(report, winner, bank["mode"], bank["condition"], path,
                                "shared_iid" if bank["pair_id"] is None and spec["pair_id"] is None
                                else "epoch_evaluation")})
        if device is not None:
            print(f"TEST_READY task={spec['task_id']} bank={tag}", flush=True)
    return rows


def _collect_if_complete(cluster_root, fixed_root, root):
    root = Path(root).resolve()
    specs = [task_spec(cluster_root, fixed_root, root, task) for task in range(1, 31)]
    if any(not (Path(spec["directory"]) / "complete.json").is_file() for spec in specs):
        return False
    with _lock(root / ".summary.job.lock", blocking=True):
        rows, winners, cache = [], [], {}
        for spec in specs:
            result = json.loads((Path(spec["directory"]) / "complete.json").read_text(encoding="utf-8"))
            if result.get("format") != "nano_graph_u_epoch_task_v1" or result.get("task") != spec:
                raise ValueError("Completed task belongs to a different epoch experiment")
            winner = result["winner"]
            training = _bank(spec["train_dump"], spec["condition"], "train", spec["mode"], spec["fixed_shift_config"])
            _check_checkpoint(winner["checkpoint"], spec, winner["learning_rate"], training, cache)
            if _digest(Path(winner["checkpoint"]), cache) != winner["checkpoint_sha256"]:
                raise ValueError("Selected checkpoint changed after testing")
            # Re-read cached reports, validating their checkpoint and data hashes.
            expected_rows = _evaluate(spec, winner, _test_banks(spec), None, cache)
            if result["rows"] != expected_rows:
                raise ValueError("Completed test rows differ from their evaluation reports")
            rows.extend(expected_rows)
            winners.append({"epochs": spec["epochs"], "training_pair_id": spec["pair_id"],
                            "shuffle_batches": True, "batch_order_scheme": BATCH_ORDER_SCHEME,
                            "training_graph_u_mode": spec["mode"], "training_condition": spec["condition"],
                            "repetition": 1, "learning_rate": winner["learning_rate"],
                            "validation_roc_auc": winner["validation_roc_auc"],
                            "checkpoint": winner["checkpoint"], "checkpoint_sha256": winner["checkpoint_sha256"]})
        columns = ["epochs", "fixed_location", "fixed_scale", "training_graph_u_mode", "training_condition",
                   "test_graph_u_mode", "test_condition"]
        expected = {(epoch, *pair, mode, condition, test_mode, test_condition)
                    for epoch in EPOCHS for pair in FIXED_SHIFTS for mode in MODES
                    for condition in ("identity", "shift") for test_mode in MODES
                    for test_condition in ("identity", "shift")}
        if len(rows) != 192 or {tuple(row[key] for key in columns) for row in rows} != expected:
            raise ValueError("Epoch results must contain all 192 repetition-one comparisons exactly once")
        rows.sort(key=lambda row: tuple(row[key] for key in columns))
        cells = [{**{key: row[key] for key in columns}, "repetition_count": 1,
                  "shuffle_batches": True, "batch_order_scheme": BATCH_ORDER_SCHEME,
                  **{f"{metric}_mean": row[metric] for metric in METRIC_NAMES}} for row in rows]
        result_dir = root / "results"
        result_dir.mkdir(parents=True, exist_ok=True)
        for name, values in (("per_repetition.csv", rows), ("summary.csv", cells), ("winners.csv", winners)):
            _atomic_write(result_dir / name, _csv_bytes(values, list(values[0])))
        _json(result_dir / "summary.json", {"format": "nano_graph_u_epoch_summary_v1",
              "epochs": list(EPOCHS), "learning_rates": list(LEARNING_RATES), "repetitions": [1],
              "shuffle_batches": True, "batch_order_scheme": BATCH_ORDER_SCHEME,
              "selection_metric": "matching_validation_roc_auc", "winners": winners,
              "cells": cells, "per_repetition": rows})
    print(f"EPOCH_RESULTS_COMPLETE winners=30 comparisons=192 results={result_dir}", flush=True)
    return True


def run_experiment(cluster_root, fixed_root, root, task_id, *, device="cuda"):
    spec = task_spec(cluster_root, fixed_root, root, task_id)
    directory = Path(spec["directory"])
    with _lock(directory / ".job.lock"):
        training = _bank(spec["train_dump"], spec["condition"], "train", spec["mode"], spec["fixed_shift_config"])
        validation = _bank(spec["validation_dump"], spec["condition"], "validation", spec["mode"], spec["fixed_shift_config"])
        if training["metadata"]["generator_source_sha256"] != validation["metadata"]["generator_source_sha256"]:
            raise ValueError("Training and shared validation banks were generated by different prior code")
        banks, cache = _test_banks(spec), {}
        _json(directory / "task.json", spec)
        candidates = [_candidate(spec, rate, training, validation, device, cache) for rate in LEARNING_RATES]
        selected = max(candidates, key=lambda candidate: (candidate["validation_roc_auc"], -candidate["learning_rate"]))
        winner = {**selected, "training_graph_u_mode": spec["mode"], "training_condition": spec["condition"],
                  "repetition": 1, "model_seed": 42, "epochs": spec["epochs"],
                  "shuffle_batches": True, "batch_order_scheme": BATCH_ORDER_SCHEME,
                  "checkpoint": str(directory / "winner/model.pt"),
                  "query_location": 0.0 if spec["condition"] == "identity" else spec["fixed_shift_config"]["graph_u_query_location"],
                  "query_scale": 1.0 if spec["condition"] == "identity" else spec["fixed_shift_config"]["graph_u_query_scale"],
                  "generator_source_sha256": training["metadata"]["generator_source_sha256"]}
        _retain_checkpoint(Path(winner["source_checkpoint"]), Path(winner["checkpoint"]), winner["checkpoint_sha256"])
        _json(directory / "winner.json", winner)
        print(f"WINNER_READY task={task_id} epochs={spec['epochs']} lr={winner['learning_rate']:g}", flush=True)
        rows = _evaluate(spec, winner, banks, device, cache)
        result = {"format": "nano_graph_u_epoch_task_v1", "task": spec, "winner": winner, "rows": rows}
        _json(directory / "complete.json", result)
    _collect_if_complete(cluster_root, fixed_root, root)
    print(f"EPOCH_EXPERIMENT_COMPLETE task={task_id} epochs={spec['epochs']}", flush=True)
    return result


def _selected_training(spec, cache):
    """Bind training-set scoring to the already selected and tested winner."""
    directory = Path(spec["directory"])
    source = directory / "complete.json"
    completed = json.loads(source.read_text(encoding="utf-8"))
    if completed.get("format") != "nano_graph_u_epoch_task_v1" or completed.get("task") != spec:
        raise ValueError("Training evaluation requires this task's completed test results")
    winner = completed["winner"]
    if json.loads((directory / "winner.json").read_text(encoding="utf-8")) != winner:
        raise ValueError("Selected winner differs from the model used for testing")
    expected = {"epochs": spec["epochs"], "training_steps": spec["total_steps"],
                "training_graph_u_mode": spec["mode"], "training_condition": spec["condition"],
                "repetition": 1, "model_seed": 42, "shuffle_batches": True,
                "batch_order_scheme": BATCH_ORDER_SCHEME, "learning_rate": winner["learning_rate"],
                "checkpoint": str(directory / "winner/model.pt"),
                "checkpoint_sha256": winner["checkpoint_sha256"]}
    pairs = [FIXED_SHIFTS[spec["pair_id"] - 1]] if spec["pair_id"] else FIXED_SHIFTS
    expected_cases = {(*pair, mode, condition) for pair in pairs for mode in MODES
                      for condition in ("identity", "shift")}
    rows = completed.get("rows", [])
    if (winner["learning_rate"] not in LEARNING_RATES or winner["checkpoint"] != expected["checkpoint"]
        or len(rows) != len(expected_cases)
        or {(row["fixed_location"], row["fixed_scale"], row["test_graph_u_mode"], row["test_condition"])
            for row in rows} != expected_cases
        or any(any(row.get(key) != value for key, value in expected.items()) for row in rows)):
        raise ValueError("Completed test rows do not identify this frozen selected model")
    training = _bank(spec["train_dump"], spec["condition"], "train", spec["mode"], spec["fixed_shift_config"])
    _check_checkpoint(winner["checkpoint"], spec, winner["learning_rate"], training, cache)
    if _digest(Path(winner["checkpoint"]), cache) != winner["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed after testing")
    binding = {"task": spec, "winner": winner, "test_results_sha256": _digest(source, cache),
               "training_dump_sha256": _digest(Path(training["dump"]), cache)}
    return training, binding


def _load_training_evaluation(spec, cache):
    training, binding = _selected_training(spec, cache)
    saved = json.loads((Path(spec["directory"]) / "training_evaluation.json").read_text(encoding="utf-8"))
    if saved.get("format") != "nano_graph_u_epoch_training_evaluation_v1" or any(
        saved.get(key) != value for key, value in binding.items()
    ):
        raise ValueError("Saved training evaluation belongs to a different selected model")
    report = saved["evaluation_report"]
    _check_report(report, binding["winner"]["checkpoint"], training, spec["total_steps"], cache, task_count=80000)
    return report, binding["winner"]


def _collect_training_if_complete(cluster_root, fixed_root, root):
    root = Path(root).resolve()
    specs = [task_spec(cluster_root, fixed_root, root, task) for task in range(1, 31)]
    if any(not (Path(spec["directory"]) / "training_evaluation.json").is_file() for spec in specs):
        return False
    with _lock(root / "results/.training_summary.job.lock", blocking=True):
        rows, cache = [], {}
        for spec in specs:
            report, winner = _load_training_evaluation(spec, cache)
            pairs = [FIXED_SHIFTS[spec["pair_id"] - 1]] if spec["pair_id"] else FIXED_SHIFTS
            for location, scale in pairs:
                rows.append({"epochs": spec["epochs"], "training_steps": spec["total_steps"],
                    "fixed_location": location, "fixed_scale": scale,
                    "training_graph_u_mode": spec["mode"], "training_condition": spec["condition"],
                    "repetition": 1, "model_seed": 42, "learning_rate": winner["learning_rate"],
                    "shuffle_batches": True, "batch_order_scheme": BATCH_ORDER_SCHEME,
                    "evaluation_split": "training", "scored_rows": "query", "task_count": 80000,
                    "query_row_count": report["query_row_count"], **report["metrics"],
                    "checkpoint": report["checkpoint"], "checkpoint_sha256": report["checkpoint_sha256"],
                    "training_dump": report["dump"], "training_dump_sha256": report["dump_sha256"],
                    "result_path": str(Path(spec["directory"]) / "training_evaluation.json"),
                    "result_origin": "shared_iid" if spec["pair_id"] is None else "own_shift_training"})
        keys = ["epochs", "fixed_location", "fixed_scale", "training_graph_u_mode", "training_condition"]
        expected = {(epoch, *pair, mode, condition) for epoch in EPOCHS for pair in FIXED_SHIFTS
                    for mode in MODES for condition in ("identity", "shift")}
        if len(rows) != 48 or {tuple(row[key] for key in keys) for row in rows} != expected:
            raise ValueError("Training scores require all 48 epoch/model/shift comparisons")
        rows.sort(key=lambda row: tuple(row[key] for key in keys))
        cells = [{**{key: row[key] for key in keys}, "repetition_count": 1,
                  "task_count_per_repetition": 80000, "evaluation_split": "training", "scored_rows": "query",
                  **{f"{metric}_mean": row[metric] for metric in METRIC_NAMES}} for row in rows]
        for name, values in (("training_per_repetition.csv", rows), ("training_summary.csv", cells)):
            path = root / "results" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(path, _csv_bytes(values, list(values[0])))
    print(f"EPOCH_TRAINING_RESULTS_COMPLETE models=30 rows=48 results={root / 'results'}", flush=True)
    return True


def run_training_evaluation(cluster_root, fixed_root, root, task_id, *, device="cuda"):
    """Evaluate frozen winners on their own training tasks; never optimize/select."""
    spec = task_spec(cluster_root, fixed_root, root, task_id)
    directory = Path(spec["directory"])
    with _lock(directory / ".training_evaluation.job.lock"):
        if (directory / "training_evaluation.json").is_file():
            report, _ = _load_training_evaluation(spec, {})
        else:
            training, binding = _selected_training(spec, {})
            checkpoint = binding["winner"]["checkpoint"]
            report = evaluate_checkpoint(checkpoint, training["dump"], batch_size=32, max_batches=None,
                                         device=device, allow_training_dump=True)
            _check_report(report, checkpoint, training, spec["total_steps"], {}, task_count=80000)
            if (report["dump_sha256"] != binding["training_dump_sha256"]
                or _digest(Path(checkpoint), None) != binding["winner"]["checkpoint_sha256"]
                or _digest(directory / "complete.json", None) != binding["test_results_sha256"]):
                raise ValueError("Selected model or its inputs changed during training evaluation")
            _json(directory / "training_evaluation.json", {
                "format": "nano_graph_u_epoch_training_evaluation_v1", **binding, "evaluation_report": report})
    _collect_training_if_complete(cluster_root, fixed_root, root)
    print(f"EPOCH_TRAINING_EVALUATION_COMPLETE task={task_id} tasks=80000", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-root", required=True, type=Path)
    parser.add_argument("--fixed-root", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--task-id", required=True, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--training-eval-only", action="store_true",
                        help="Score already selected winners on their own training banks; no optimization")
    args = parser.parse_args(argv)
    run = run_training_evaluation if args.training_eval_only else run_experiment
    run(args.cluster_root, args.fixed_root, args.root, args.task_id, device=args.device)


if __name__ == "__main__":
    main()
