"""Score frozen epoch winners with fixed test shifts 1.5 times their training shifts."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path

from . import epoch_experiments as epochs
from .data import _generator_source_sha256
from .evaluate import evaluate_checkpoint
from .fixed_experiments import (
    FIXED_SHIFTS,
    _bank_status,
    _generate_banks,
    _row,
    experiment_spec,
)
from .sampled import MODES, _lock
from .select import _atomic_write
from .summarize_tests import _csv_bytes
from .test_winners import METRIC_NAMES, _digest

MULTIPLIER = 1.5
TEST_SEED = 6001
_KEYS = ("epochs", "fixed_location", "fixed_scale", "training_graph_u_mode", "training_condition",
         "test_graph_u_mode", "test_condition")


def _context(cluster_root, fixed_root, source_root, root):
    cluster, fixed, source, output = [Path(p).expanduser().resolve()
                                       for p in (cluster_root, fixed_root, source_root, root)]
    protected = [fixed, source, cluster / "tabicl-conf-shift-priors",
                 cluster / "tabicl-conf-shift-checkpoints", cluster / "tabicl-conf-shift-evaluations"]
    if output == cluster or output in cluster.parents or any(
        output == path or output in path.parents or path in output.parents for path in protected
    ):
        raise ValueError("Test-shift output must be separate from all existing source roots")
    return {"cluster": cluster, "fixed": fixed, "source": source, "root": output}


def bank_spec(cluster_root, fixed_root, source_root, root, task_id):
    if isinstance(task_id, bool) or not isinstance(task_id, int) or not 1 <= task_id <= 8:
        raise ValueError("task_id must be an integer from 1 to 8")
    context = _context(cluster_root, fixed_root, source_root, root)
    pair_id, mode_index = divmod(task_id - 1, 2)
    pair_id += 1
    pair = experiment_spec(context["root"], pair_id)
    mode = MODES[mode_index]
    location, scale = pair["location"] * MULTIPLIER, pair["scale"] * MULTIPLIER
    return {**context, "task_id": task_id, "pair_id": pair_id, "directory": pair["directory"],
            "graph_u_mode": mode, "split": "test", "repetition": 1, "seed": TEST_SEED, "steps": 100,
            "fixed_location": pair["location"], "fixed_scale": pair["scale"],
            "shifted_test_location": location, "shifted_test_scale": scale,
            "fixed_shift_config": {"graph_u_query_location": location, "graph_u_query_scale": scale},
            "dump": pair["directory"] / "data" / mode / "rep_1"
                    / f"shift_test_seed{TEST_SEED}_100steps_b32_r150_f5.h5"}


def _recipe(context):
    value = {"format": "nano_graph_u_epoch_test_shift_v1", "test_shift_multiplier": MULTIPLIER,
             "cluster_root": str(context["cluster"]), "fixed_root": str(context["fixed"]),
             "source_root": str(context["source"]), "test_seed": TEST_SEED, "repetitions": [1],
             "epochs": list(epochs.EPOCHS), "steps": 100, "batch_size": 32, "rows": 150, "features": 5,
             "profiles": [{"fixed_location": location, "fixed_scale": scale,
                           "shifted_test_location": location * MULTIPLIER,
                           "shifted_test_scale": scale * MULTIPLIER} for location, scale in FIXED_SHIFTS]}
    with _lock(context["root"] / ".recipe.job.lock", blocking=True):
        epochs._json(context["root"] / "experiment.json", value)
    return value


def generate_bank(cluster_root, fixed_root, source_root, root, task_id, *, workers=8):
    spec = bank_spec(cluster_root, fixed_root, source_root, root, task_id)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    _recipe(spec)
    with _lock(spec["directory"] / f".generate_{spec['graph_u_mode']}.job.lock"):
        _generate_banks([spec], workers)
    print(f"TEST_SHIFT_BANK_COMPLETE task={task_id} train={spec['fixed_location']}/{spec['fixed_scale']} "
          f"test={spec['shifted_test_location']}/{spec['shifted_test_scale']} dump={spec['dump']}", flush=True)
    return spec


def _model_key(epoch, mode, condition, pair):
    return epoch, mode, condition, None if condition == "identity" else pair


def _reference_tables(source, models, training_reports):
    """Validate the existing CSVs before copying them byte for byte."""
    names = ("winners.csv", "training_summary.csv", "training_per_repetition.csv")
    contents = {name: (source / "results" / name).read_bytes() for name in names}
    expected = {(epoch, location, scale, mode, condition) for epoch in epochs.EPOCHS
                for location, scale in FIXED_SHIFTS for mode in MODES for condition in ("identity", "shift")}
    for name in names:
        rows = list(csv.DictReader(io.StringIO(contents[name].decode("utf-8"))))
        seen = set()
        for row in rows:
            epoch, mode, condition = int(row["epochs"]), row["training_graph_u_mode"], row["training_condition"]
            if int(row.get("repetition", row.get("repetition_count", 0))) != 1:
                raise ValueError("Source references must use repetition one")
            if name == "winners.csv":
                pair = None if condition == "identity" else int(float(row["training_pair_id"]))
                key = _model_key(epoch, mode, condition, pair)
                model = models[key]
                if (row["checkpoint_sha256"] != model["checkpoint_sha256"]
                    or row["checkpoint"] != model["checkpoint"]
                    or float(row["learning_rate"]) != model["learning_rate"]
                    or float(row["validation_roc_auc"]) != model["validation_roc_auc"]):
                    raise ValueError("Winner CSV does not match the frozen selected models")
            else:
                location, scale = float(row["fixed_location"]), float(row["fixed_scale"])
                pair = FIXED_SHIFTS.index((location, scale)) + 1
                model_key = _model_key(epoch, mode, condition, pair)
                report, model = training_reports[model_key], models[model_key]
                key = epoch, location, scale, mode, condition
                summary = name == "training_summary.csv"
                if (row["evaluation_split"] != "training" or row["scored_rows"] != "query"
                    or int(row["task_count_per_repetition" if summary else "task_count"]) != 80000
                    or int(row["repetition_count" if summary else "repetition"]) != 1
                    or any(not math.isclose(float(row[f"{metric}_mean" if summary else metric]), report["metrics"][metric],
                                            rel_tol=0, abs_tol=1e-12) for metric in METRIC_NAMES)
                    or (not summary and (row["checkpoint_sha256"] != model["checkpoint_sha256"]
                                         or row["training_dump_sha256"] != report["dump_sha256"]
                                         or int(row["training_steps"]) != 2500 * epoch
                                         or int(row["model_seed"]) != 42
                                         or row["shuffle_batches"] != "True"
                                         or row["batch_order_scheme"] != epochs.BATCH_ORDER_SCHEME
                                         or float(row["learning_rate"]) != model["learning_rate"]))):
                    raise ValueError("Training CSV does not match the existing frozen-model training scores")
            if key in seen:
                raise ValueError(f"Duplicate source CSV row: {name}")
            seen.add(key)
        if seen != (set(models) if name == "winners.csv" else expected):
            raise ValueError(f"Incomplete source CSV: {name}")
    return contents


def _source(context, cache):
    path = context["source"] / "results/summary.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    rows = saved.get("per_repetition", [])
    expected = {(epoch, *pair, mode, condition, test_mode, test_condition)
                for epoch in epochs.EPOCHS for pair in FIXED_SHIFTS for mode in MODES
                for condition in ("identity", "shift") for test_mode in MODES
                for test_condition in ("identity", "shift")}
    if (saved.get("format") != "nano_graph_u_epoch_summary_v1" or len(rows) != 192
        or len(saved.get("cells", [])) != 192 or len(saved.get("winners", [])) != 30
        or {tuple(row[key] for key in _KEYS) for row in rows} != expected):
        raise ValueError("Expected the complete selected repetition-one epoch experiment")
    models, training = {}, {}
    for task in range(1, 31):
        spec = epochs.task_spec(context["cluster"], context["fixed"], context["source"], task)
        report, winner = epochs._load_training_evaluation(spec, cache)
        key = _model_key(spec["epochs"], spec["mode"], spec["condition"], spec["pair_id"])
        models[key], training[key] = winner, report
    tables = _reference_tables(context["source"], models, training)
    provenance = {"source_summary_sha256": _digest(path, cache),
                  "checkpoints": {str(key): value["checkpoint_sha256"] for key, value in models.items()},
                  "reference_csv_sha256": {name: _digest(context["source"] / "results" / name, cache) for name in tables}}
    return models, rows, tables, provenance


def _iid_report(source_rows, pair, epoch, model, test_mode, cache):
    matches = [row for row in source_rows if
               (row["fixed_location"], row["fixed_scale"], row["epochs"], row["training_graph_u_mode"],
                row["training_condition"], row["test_graph_u_mode"], row["test_condition"])
               == (*pair, epoch, model["training_graph_u_mode"], model["training_condition"], test_mode, "identity")]
    if len(matches) != 1:
        raise ValueError("Missing source IID test reference")
    row = matches[0]
    path = Path(row["result_path"])
    saved = json.loads(path.read_text(encoding="utf-8"))
    if (saved.get("winner") != model or saved.get("test_graph_u_mode") != test_mode
        or saved.get("test_condition") != "identity" or saved.get("pair_id") is not None):
        raise ValueError("Source IID report belongs to a different selected model")
    report = saved["evaluation_report"]
    bank = epochs._bank(report["dump"], "identity", "test", test_mode)
    epochs._check_report(report, model["checkpoint"], bank, 2500 * epoch, cache)
    if (row["checkpoint_sha256"] != model["checkpoint_sha256"]
        or row["learning_rate"] != model["learning_rate"]
        or row["test_dump_sha256"] != report["dump_sha256"]
        or any(row[metric] != report["metrics"][metric] for metric in METRIC_NAMES)):
        raise ValueError("Source IID summary differs from its frozen evaluation report")
    return report, path


def _evaluate_pair(context, pair_id, models, source_rows, device, cache):
    pair = experiment_spec(context["root"], pair_id)
    rows = []
    for epoch in epochs.EPOCHS:
        for mode in MODES:
            for condition in ("identity", "shift"):
                model = models[_model_key(epoch, mode, condition, pair_id)]
                for test_mode in MODES:
                    report, path = _iid_report(source_rows, (pair["location"], pair["scale"]), epoch, model, test_mode, cache)
                    for test_condition in ("identity", "shift"):
                        if test_condition == "shift":
                            spec = bank_spec(context["cluster"], context["fixed"], context["source"], context["root"],
                                             (pair_id - 1) * 2 + MODES.index(test_mode) + 1)
                            bank = {"dump": str(spec["dump"]), "metadata": _bank_status(spec).metadata}
                            path = pair["directory"] / "results" / f"epochs_{epoch}" / f"{mode}_{condition}_on_{test_mode}.json"
                            binding = {"winner": model, "test_graph_u_mode": test_mode,
                                       "test_shift_multiplier": MULTIPLIER, "fixed_shift_config": spec["fixed_shift_config"]}
                            if path.exists():
                                saved = json.loads(path.read_text(encoding="utf-8"))
                                if any(saved.get(key) != value for key, value in binding.items()):
                                    raise ValueError("Cached test score belongs to a different frozen model/test profile")
                                report = saved["evaluation_report"]
                            else:
                                if device is None:
                                    raise FileNotFoundError(f"Missing completed test score: {path}")
                                report = evaluate_checkpoint(model["checkpoint"], bank["dump"], batch_size=32, device=device)
                            epochs._check_report(report, model["checkpoint"], bank, 2500 * epoch, cache)
                            epochs._json(path, {**binding, "evaluation_report": report})
                        rows.append({"epochs": epoch, "training_steps": 2500 * epoch,
                            "shuffle_batches": True, "batch_order_scheme": epochs.BATCH_ORDER_SCHEME,
                            "fixed_location": pair["location"], "fixed_scale": pair["scale"],
                            "shifted_test_location": pair["location"] * MULTIPLIER,
                            "shifted_test_scale": pair["scale"] * MULTIPLIER, "test_shift_multiplier": MULTIPLIER,
                            **_row(report, model, test_mode, test_condition, path,
                                   "reused_iid" if test_condition == "identity" else "new_test_shift")})
    return rows


def _collect_if_complete(context):
    paths = [experiment_spec(context["root"], pair)["directory"] / "complete.json" for pair in range(1, 5)]
    if not all(path.is_file() for path in paths):
        return False
    with _lock(context["root"] / ".summary.job.lock", blocking=True):
        cache = {}
        models, source_rows, tables, provenance = _source(context, cache)
        rows = []
        for pair_id, path in enumerate(paths, 1):
            saved = json.loads(path.read_text(encoding="utf-8"))
            expected_rows = _evaluate_pair(context, pair_id, models, source_rows, None, cache)
            if saved.get("source") != provenance or saved.get("rows") != expected_rows:
                raise ValueError("Completed test-shift results differ from their frozen inputs/reports")
            rows.extend(expected_rows)
        rows.sort(key=lambda row: tuple(row[key] for key in _KEYS))
        if len(rows) != 192 or len({tuple(row[key] for key in _KEYS) for row in rows}) != 192:
            raise ValueError("Expected all 192 test-shift comparisons")
        cells = [{**{key: row[key] for key in _KEYS}, "repetition_count": 1,
                  "shifted_test_location": row["shifted_test_location"], "shifted_test_scale": row["shifted_test_scale"],
                  "test_shift_multiplier": MULTIPLIER, "shuffle_batches": True,
                  "batch_order_scheme": epochs.BATCH_ORDER_SCHEME,
                  **{f"{metric}_mean": row[metric] for metric in METRIC_NAMES}} for row in rows]
        output = context["root"] / "results"
        output.mkdir(parents=True, exist_ok=True)
        for name, values in (("summary.csv", cells), ("per_repetition.csv", rows)):
            _atomic_write(output / name, _csv_bytes(values, list(values[0])))
        for name, content in tables.items():
            _atomic_write(output / name, content)
        epochs._json(output / "summary.json", {"format": "nano_graph_u_epoch_test_shift_summary_v1",
            "source": provenance, "test_shift_multiplier": MULTIPLIER, "new_evaluation_count": 96,
            "reused_iid_count": 96, "cells": cells, "per_repetition": rows})
    print(f"TEST_SHIFT_RESULTS_COMPLETE banks=8 new_evaluations=96 reused_iid=96 results={output}", flush=True)
    return True


def evaluate_pair(cluster_root, fixed_root, source_root, root, task_id, *, device="cuda"):
    if isinstance(task_id, bool) or not isinstance(task_id, int) or not 1 <= task_id <= 4:
        raise ValueError("task_id must be an integer from 1 to 4")
    context = _context(cluster_root, fixed_root, source_root, root)
    for task in range(1, 9):
        spec = bank_spec(cluster_root, fixed_root, source_root, root, task)
        if _bank_status(spec).metadata["generator_source_sha256"] != _generator_source_sha256():
            raise ValueError("New test banks were generated by different prior code")
    _recipe(context)
    pair = experiment_spec(context["root"], task_id)
    with _lock(pair["directory"] / ".evaluation.job.lock"):
        cache = {}
        models, source_rows, _, provenance = _source(context, cache)
        rows = _evaluate_pair(context, task_id, models, source_rows, device, cache)
        epochs._json(pair["directory"] / "complete.json", {"source": provenance, "rows": rows})
    _collect_if_complete(context)
    print(f"TEST_SHIFT_EVALUATION_COMPLETE pair={task_id} new_evaluations=24 reused_iid=24", flush=True)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    for stage in ("generate", "evaluate"):
        command = commands.add_parser(stage)
        for name in ("cluster-root", "fixed-root", "source-root", "root"):
            command.add_argument(f"--{name}", required=True, type=Path)
        command.add_argument("--task-id", type=int, required=True)
        if stage == "generate":
            command.add_argument("--workers", type=int, default=8)
        else:
            command.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    function = generate_bank if args.stage == "generate" else evaluate_pair
    kwargs = {"workers": args.workers} if args.stage == "generate" else {"device": args.device}
    function(args.cluster_root, args.fixed_root, args.source_root, args.root, args.task_id, **kwargs)


if __name__ == "__main__":
    main()
