"""Score fixed-experiment winners on their own saved pretraining tasks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

import torch

from .evaluate import evaluate_checkpoint
from .fixed_experiments import FIXED_SHIFTS, _summarize, _write_json
from .sampled import MODES, _lock
from .select import _atomic_write, _check_existing
from .summarize_tests import _csv_bytes
from .test_winners import METRIC_NAMES, _digest
from .train import PAPER_MODEL_CONFIG
from .tune import validate_bank


def _source(root):
    path = root / "results" / "summary.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    rows = saved.get("per_repetition", [])
    keys = ("fixed_location", "fixed_scale", "training_graph_u_mode", "training_condition",
            "test_graph_u_mode", "test_condition", "repetition")
    expected = {(location, scale, mode, condition, test_mode, test_condition, repetition)
                for location, scale in FIXED_SHIFTS for mode in MODES
                for condition in ("identity", "shift") for test_mode in MODES
                for test_condition in ("identity", "shift") for repetition in range(1, 6)}
    if (saved.get("format") != "nano_graph_u_fixed_experiments_summary_v1"
            or len(rows) != 320 or len(saved.get("cells", [])) != 64
            or {tuple(row[key] for key in keys) for row in rows} != expected):
        raise ValueError("Expected the complete, selected four-experiment test summary")
    for location, scale in FIXED_SHIFTS:
        group = [row for row in rows if (row["fixed_location"], row["fixed_scale"]) == (location, scale)]
        cells = [cell for cell in saved["cells"] if (cell["fixed_location"], cell["fixed_scale"]) == (location, scale)]
        if _summarize(group) != cells:
            raise ValueError("Original test summary does not match its repetition scores")
    return rows, _digest(path, None)


def _task_spec(root, task_id, rows, source_hash):
    if isinstance(task_id, bool) or not isinstance(task_id, int) or not 1 <= task_id <= 50:
        raise ValueError("task_id must be an integer from 1 to 50")
    if task_id <= 10:
        condition, slot, pair = "identity", task_id - 1, None
    else:
        pair_index, slot = divmod(task_id - 11, 10)
        condition, pair = "shift", FIXED_SHIFTS[pair_index]
    mode, repetition = MODES[slot // 5], slot % 5 + 1
    matches = [row for row in rows
               if (row["training_graph_u_mode"], row["training_condition"], row["repetition"])
               == (mode, condition, repetition)
               and (pair is None or (row["fixed_location"], row["fixed_scale"]) == pair)]
    fields = ("checkpoint", "checkpoint_sha256", "model_seed", "learning_rate", "training_generator_source_sha256")
    if len(matches) != (16 if pair is None else 4) or any(
        any(row[field] != matches[0][field] for field in fields) for row in matches
    ):
        raise ValueError("Test results disagree about the selected model")
    selected = matches[0]
    return {
        "task_id": task_id, "training_graph_u_mode": mode, "training_condition": condition,
        "repetition": repetition, "fixed_location": None if pair is None else pair[0],
        "fixed_scale": None if pair is None else pair[1],
        **{field: selected[field] for field in fields},
        "checkpoint": str(Path(selected["checkpoint"]).resolve()),
        "test_results_sha256": source_hash,
        "result_path": str(root / "results" / "training_evaluations" / f"task_{task_id:02d}.json"),
    }


def task_spec(root, task_id):
    if isinstance(task_id, bool) or not isinstance(task_id, int) or not 1 <= task_id <= 50:
        raise ValueError("task_id must be an integer from 1 to 50")
    root = Path(root).expanduser().resolve()
    return _task_spec(root, task_id, *_source(root))


def _training_bank(spec, cache):
    checkpoint_path = Path(spec["checkpoint"])
    if _digest(checkpoint_path, cache) != spec["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed after test evaluation")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint.get("train_config", {})
    if not config.get("dump_path"):
        raise ValueError("Checkpoint does not record its own training dump")
    dump = Path(config["dump_path"]).resolve()
    profile = None if spec["training_condition"] == "identity" else {
        "graph_u_query_location": spec["fixed_location"], "graph_u_query_scale": spec["fixed_scale"],
    }
    metadata = validate_bank(dump, spec["training_condition"], "train", spec["repetition"],
        graph_u_mode=spec["training_graph_u_mode"], fixed_shift_config=profile)
    dump_hash = _digest(dump, cache)
    expected_training = {
        "seed": spec["model_seed"], "batch_size": 32, "learning_rate": spec["learning_rate"],
        "optimizer": "schedulefree.AdamWScheduleFree", "weight_decay": 0.0,
        "gradient_clip_norm": 1.0, "loss": "query_cross_entropy", "precision": "float32",
        "dump_sha256": dump_hash,
    }
    if (checkpoint.get("format") != "nano_graph_u_v1" or checkpoint.get("step") != 2500
            or checkpoint.get("model_config") != PAPER_MODEL_CONFIG
            or checkpoint.get("dump_metadata") != metadata
            or checkpoint.get("prior_config") != metadata["prior_config"]
            or metadata["generator_source_sha256"] != spec["training_generator_source_sha256"]
            or spec["model_seed"] != 41 + spec["repetition"]
            or any(config.get(key) != value for key, value in expected_training.items())):
        raise ValueError("Checkpoint and bank do not match this selected model's original training")
    return dump, metadata, dump_hash


def _validate_report(report, spec, dump, metadata, dump_hash):
    count = report.get("query_row_count")
    if (report.get("format") != "nano_graph_u_evaluation_v1"
            or report.get("checkpoint") != spec["checkpoint"]
            or report.get("checkpoint_sha256") != spec["checkpoint_sha256"]
            or report.get("checkpoint_step") != 2500 or report.get("dump") != str(dump)
            or report.get("dump_sha256") != dump_hash or report.get("dump_metadata") != metadata
            or report.get("task_count") != 80000 or isinstance(count, bool)
            or not isinstance(count, int) or not 0 < count < 80000 * 150):
        raise ValueError("Training evaluation must score the selected checkpoint on all 80,000 original tasks")
    for metric in METRIC_NAMES:
        value = report.get("metrics", {}).get(metric)
        if (report.get("metric_task_counts", {}).get(metric) != 80000
                or isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0 or (metric != "nll" and value > 1)):
            raise ValueError(f"Incomplete or invalid training metric: {metric}")


def _load_result(spec, cache):
    saved = json.loads(Path(spec["result_path"]).read_text(encoding="utf-8"))
    if saved.get("format") != "nano_graph_u_own_training_evaluation_v1" or saved.get("model") != spec:
        raise ValueError("Saved training evaluation belongs to a different selected model")
    _validate_report(saved["evaluation_report"], spec, *_training_bank(spec, cache))
    return saved["evaluation_report"]


def _collect_if_complete(root):
    rows, source_hash = _source(root)
    specs = [_task_spec(root, task, rows, source_hash) for task in range(1, 51)]
    if not all(Path(spec["result_path"]).is_file() for spec in specs):
        return False
    with _lock(root / "results" / ".training_summary.job.lock", blocking=True):
        cache = {}
        reports = {spec["task_id"]: _load_result(spec, cache) for spec in specs}
        per_repetition = []
        for pair_index, (location, scale) in enumerate(FIXED_SHIFTS):
            for slot in range(10):
                for condition in ("identity", "shift"):
                    task = slot + 1 if condition == "identity" else 11 + pair_index * 10 + slot
                    spec, report = specs[task - 1], reports[task]
                    prior = report["dump_metadata"]["prior_config"]
                    per_repetition.append({
                        "fixed_location": location, "fixed_scale": scale,
                        "training_graph_u_mode": spec["training_graph_u_mode"], "training_condition": condition,
                        "repetition": spec["repetition"], "model_seed": spec["model_seed"],
                        "learning_rate": spec["learning_rate"], "evaluation_split": "training", "scored_rows": "query",
                        "training_query_location": prior["graph_u_query_location"],
                        "training_query_scale": prior["graph_u_query_scale"],
                        "task_count": report["task_count"], "query_row_count": report["query_row_count"],
                        **report["metrics"], "checkpoint": report["checkpoint"],
                        "checkpoint_sha256": report["checkpoint_sha256"], "training_dump": report["dump"],
                        "training_dump_sha256": report["dump_sha256"], "result_path": spec["result_path"],
                        "result_origin": "shared_iid" if condition == "identity" else "own_shift_training",
                    })
        keys = ("fixed_location", "fixed_scale", "training_graph_u_mode", "training_condition")
        groups = {}
        for row in per_repetition:
            groups.setdefault(tuple(row[key] for key in keys), []).append(row)
        cells = []
        for key, group in sorted(groups.items()):
            if len(group) != 5 or {row["repetition"] for row in group} != set(range(1, 6)):
                raise ValueError("Training summaries require five repetitions per model setting")
            cell = {**dict(zip(keys, key)), "repetition_count": 5, "task_count_per_repetition": 80000,
                    "evaluation_split": "training", "scored_rows": "query"}
            for metric in METRIC_NAMES:
                cell[f"{metric}_mean"] = mean(row[metric] for row in group)
                cell[f"{metric}_std"] = stdev(row[metric] for row in group)
            cells.append(cell)
        outputs = {"training_summary.csv": _csv_bytes(cells, list(cells[0])),
                   "training_per_repetition.csv": _csv_bytes(per_repetition, list(per_repetition[0]))}
        for name, contents in outputs.items():
            _check_existing(root / "results" / name, contents)
        for name, contents in outputs.items():
            _atomic_write(root / "results" / name, contents)
    print(f"TRAINING_RESULTS_COMPLETE unique_evaluations=50 comparisons=16 rows=80 results={root / 'results'}", flush=True)
    return True


def run_evaluation(root, task_id, *, device="cuda"):
    root = Path(root).expanduser().resolve()
    spec = task_spec(root, task_id)
    path = Path(spec["result_path"])
    with _lock(path.with_suffix(".job.lock")):
        if path.exists():
            report = _load_result(spec, {})
            print(f"TRAINING_EVALUATION_REUSED task={task_id}", flush=True)
        else:
            dump, metadata, dump_hash = _training_bank(spec, {})
            report = evaluate_checkpoint(spec["checkpoint"], dump, batch_size=32, max_batches=None,
                                         device=device, allow_training_dump=True)
            _validate_report(report, spec, dump, metadata, dump_hash)
            if _digest(Path(spec["checkpoint"]), None) != spec["checkpoint_sha256"]:
                raise ValueError("Checkpoint changed during training-set evaluation")
            if _digest(dump, None) != dump_hash:
                raise ValueError("Training bank changed during training-set evaluation")
            _write_json(path, {"format": "nano_graph_u_own_training_evaluation_v1", "model": spec,
                               "evaluation_report": report})
    _collect_if_complete(root)
    print(f"TRAINING_EVALUATION_COMPLETE task={task_id} tasks=80000 result={path}", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--task-id", required=True, type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    run_evaluation(args.root, args.task_id, device=args.device)


if __name__ == "__main__":
    main()
