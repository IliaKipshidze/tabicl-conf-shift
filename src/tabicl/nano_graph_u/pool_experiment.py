"""Five-repeat Nano experiments with uniform, shifted and mixed support pools."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean, stdev

import torch

from tabicl.prior.graph_lib._pool import GRAPH_U_POOL_MIXED_STD_FRACTIONS

from ._settings import experiment_prior_config
from .data import NanoPriorDumpLoader, _metadata, generate_dump
from .epoch_experiments import _json
from .evaluate import evaluate_checkpoint
from .sampled import _lock
from .select import _atomic_write, _retain_checkpoint
from .summarize_tests import _csv_bytes
from .test_winners import METRIC_NAMES, _digest
from .train import BATCH_ORDER_SCHEME, PAPER_MODEL_CONFIG, train_model
from .tune import LEARNING_RATES

MODES = ("confounding", "covariate")
CONDITIONS = ("uniform", "shift", "mixed")
REPETITIONS = tuple(range(1, 6))
ROWS, FEATURES, BATCH_SIZE = 150, 5, 32
TRAIN_STEPS, EVAL_STEPS = 2500, 100
BANK_COUNT, TRIAL_COUNT, WINNER_COUNT = 66, 300, 30
_MODEL_KEYS = ("training_graph_u_mode", "training_condition")
_TEST_KEYS = (*_MODEL_KEYS, "test_graph_u_mode", "test_condition")


def prior_config(condition, mode):
    """Keep the original graph prior and replace only the U intervention."""
    if condition not in CONDITIONS or mode not in MODES:
        raise ValueError("Unknown common-support condition or graph mode")
    return replace(
        experiment_prior_config("identity", mode),
        graph_u_intervention_mode="support_pool",
        graph_u_pool_sampling="gaussian" if condition == "shift" else condition,
        graph_u_pool_peak_percentile=75.0,
        graph_u_pool_std_fraction=0.5,
    )


def _recipe():
    return {
        "format": "nano_graph_u_pool_experiment_v1",
        "modes": list(MODES),
        "conditions": list(CONDITIONS),
        "repetitions": list(REPETITIONS),
        "rows": ROWS,
        "features": FEATURES,
        "batch_size": BATCH_SIZE,
        "train_steps": TRAIN_STEPS,
        "evaluation_steps": EVAL_STEPS,
        "epochs": 1,
        "train_tasks": TRAIN_STEPS * BATCH_SIZE,
        "evaluation_tasks": EVAL_STEPS * BATCH_SIZE,
        "learning_rates": list(LEARNING_RATES),
        "model_seed_rule": "41 + repetition",
        "shuffle_batches": True,
        "batch_order_scheme": BATCH_ORDER_SCHEME,
        "model_config": PAPER_MODEL_CONFIG,
        "selection_metric": "matching_validation_roc_auc",
        "tie_break": "smaller_learning_rate",
        "mixed_component_probabilities": [0.25] * 4,
        "mixed_gaussian_std_fractions": list(GRAPH_U_POOL_MIXED_STD_FRACTIONS),
        "priors": {
            mode: {
                condition: asdict(prior_config(condition, mode))
                for condition in CONDITIONS
            }
            for mode in MODES
        },
    }


def _initialize(root):
    root = Path(root).expanduser().resolve()
    repository = Path(__file__).resolve().parents[3]
    if (
        root == Path(root.anchor)
        or root == repository
        or root in repository.parents
        or repository in root.parents
    ):
        raise ValueError(
            "Experiment root must be a dedicated directory outside the repository"
        )
    with _lock(root / ".experiment.lock", blocking=True):
        if not (root / "experiment.json").exists() and any(
            path.name != ".experiment.lock" for path in root.iterdir()
        ):
            raise ValueError(
                "Refusing to use a non-empty directory belonging to another experiment"
            )
        _json(root / "experiment.json", _recipe())
    return root


def _index(task_id, count):
    if (
        isinstance(task_id, bool)
        or not isinstance(task_id, int)
        or not 1 <= task_id <= count
    ):
        raise ValueError(f"task_id must be an integer from 1 to {count}")
    return task_id - 1


def bank_spec(root, task_id):
    mode_index, local = divmod(_index(task_id, BANK_COUNT), 33)
    mode, condition = MODES[mode_index], CONDITIONS[local % 3]
    repetition = local // 6 + 1 if local < 30 else None
    split = ("train" if local % 6 < 3 else "test") if local < 30 else "validation"
    seed = (
        3001
        if split == "validation"
        else (1000 if split == "train" else 2000) + repetition
    )
    steps = TRAIN_STEPS if split == "train" else EVAL_STEPS
    root = Path(root).expanduser().resolve()
    directory = (
        root
        / "data"
        / mode
        / ("validation" if repetition is None else f"rep_{repetition}")
    )
    return {
        "root": str(root),
        "task_id": task_id,
        "mode": mode,
        "condition": condition,
        "repetition": repetition,
        "split": split,
        "seed": seed,
        "steps": steps,
        "dump": str(
            directory / f"{condition}_{split}_seed{seed}_{steps}steps_b32_r150_f5.h5"
        ),
    }


def _bank(root, mode, condition, split, repetition):
    local = (
        30 + CONDITIONS.index(condition)
        if split == "validation"
        else (
            (repetition - 1) * 6
            + (0 if split == "train" else 3)
            + CONDITIONS.index(condition)
        )
    )
    return bank_spec(root, MODES.index(mode) * 33 + local + 1)


def _setting(root, task_id):
    mode_index, local = divmod(_index(task_id, WINNER_COUNT), 15)
    repetition, condition_index = divmod(local, 3)
    repetition += 1
    mode, condition = MODES[mode_index], CONDITIONS[condition_index]
    root = str(Path(root).expanduser().resolve())
    return {
        "root": root,
        "mode": mode,
        "condition": condition,
        "repetition": repetition,
        "model_seed": 41 + repetition,
        "epochs": 1,
        "total_steps": TRAIN_STEPS,
        "train_dump": _bank(root, mode, condition, "train", repetition)["dump"],
        "validation_dump": _bank(root, mode, condition, "validation", repetition)[
            "dump"
        ],
    }


def trial_spec(root, task_id):
    mode_index, local = divmod(_index(task_id, TRIAL_COUNT), 150)
    model_index, lr_index = divmod(local, 10)
    spec = _setting(root, mode_index * 15 + model_index + 1)
    rate = LEARNING_RATES[lr_index]
    directory = (
        Path(spec["root"])
        / "checkpoints"
        / spec["mode"]
        / f"rep_{spec['repetition']}"
        / spec["condition"]
        / f"lr_{format(rate, 'g').replace('.', 'p')}"
    )
    return {
        **spec,
        "task_id": task_id,
        "lr_index": lr_index,
        "learning_rate": rate,
        "directory": str(directory),
        "checkpoint_dir": str(directory / "checkpoints"),
        "result_path": str(directory / "validation.json"),
    }


def winner_spec(root, task_id):
    spec = _setting(root, task_id)
    suffix = Path(spec["mode"]) / f"rep_{spec['repetition']}" / spec["condition"]
    directory = Path(spec["root"]) / "winners" / suffix
    results = Path(spec["root"]) / "results" / suffix
    return {
        **spec,
        "task_id": task_id,
        "directory": str(directory),
        "checkpoint": str(directory / "model.pt"),
        "results_directory": str(results),
        "result_path": str(results / "complete.json"),
    }


def _validate_bank(spec, *, complete=True):
    loader = NanoPriorDumpLoader(spec["dump"], batch_size=BATCH_SIZE)
    expected = _metadata(
        prior_config(spec["condition"], spec["mode"]),
        steps=spec["steps"],
        batch_size=BATCH_SIZE,
        seed=spec["seed"],
        rows=ROWS,
        features=FEATURES,
        min_train_size=0.3,
        max_train_size=0.9,
        n_jobs=1,
    )
    if loader.metadata != expected:
        raise ValueError(
            f"Bank does not match the common-support experiment: {spec['dump']}"
        )
    if not 0 <= loader.committed_steps <= spec["steps"] or (
        complete and loader.committed_steps != spec["steps"]
    ):
        raise ValueError(
            f"Incomplete {spec['split']} bank: {loader.committed_steps}/{spec['steps']}"
        )
    return loader.metadata


def generate_bank(root, task_id, *, workers=8):
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    spec = bank_spec(root, task_id)
    _initialize(root)
    with _lock(Path(spec["root"]) / "locks" / f"bank_{task_id}.lock"):
        exists = Path(spec["dump"]).exists()
        if exists:
            _validate_bank(spec, complete=False)
        print(
            f"GENERATION_START task={task_id} mode={spec['mode']} condition={spec['condition']} "
            f"split={spec['split']} tasks={spec['steps'] * BATCH_SIZE} dump={spec['dump']}",
            flush=True,
        )
        if (
            not exists
            or NanoPriorDumpLoader(spec["dump"], batch_size=BATCH_SIZE).committed_steps
            < spec["steps"]
        ):
            generate_dump(
                spec["dump"],
                steps=spec["steps"],
                batch_size=BATCH_SIZE,
                seed=spec["seed"],
                prior_config=prior_config(spec["condition"], spec["mode"]),
                rows=ROWS,
                features=FEATURES,
                min_train_size=0.3,
                max_train_size=0.9,
                n_jobs=1,
                workers=workers,
                resume=exists,
            )
        _validate_bank(spec)
    print(f"POOL_BANK_COMPLETE task={task_id} dump={spec['dump']}", flush=True)
    return spec


def _checkpoint(path, spec, rate, cache):
    training = _validate_bank(
        _bank(
            spec["root"], spec["mode"], spec["condition"], "train", spec["repetition"]
        )
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = {
        "seed": spec["model_seed"],
        "batch_size": BATCH_SIZE,
        "learning_rate": rate,
        "optimizer": "schedulefree.AdamWScheduleFree",
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "loss": "query_cross_entropy",
        "precision": "float32",
        "dump_path": spec["train_dump"],
        "dump_sha256": _digest(Path(spec["train_dump"]), cache),
        "shuffle_batches": True,
        "batch_order_scheme": BATCH_ORDER_SCHEME,
    }
    if (
        payload.get("format") != "nano_graph_u_v1"
        or payload.get("step") != TRAIN_STEPS
        or payload.get("model_config") != PAPER_MODEL_CONFIG
        or payload.get("dump_metadata") != training
        or payload.get("prior_config") != training["prior_config"]
        or payload.get("train_config") != config
    ):
        raise ValueError(
            "Checkpoint does not match the model, prior, LR, seed or full one-epoch budget"
        )


def _report(report, checkpoint, bank, cache):
    count, metadata = bank["steps"] * BATCH_SIZE, _validate_bank(bank)
    if (
        report.get("format") != "nano_graph_u_evaluation_v1"
        or report.get("checkpoint") != str(checkpoint)
        or report.get("checkpoint_sha256") != _digest(Path(checkpoint), cache)
        or report.get("checkpoint_step") != TRAIN_STEPS
        or report.get("dump") != bank["dump"]
        or report.get("dump_sha256") != _digest(Path(bank["dump"]), cache)
        or report.get("dump_metadata") != metadata
        or report.get("task_count") != count
        or isinstance(report.get("query_row_count"), bool)
        or not isinstance(report.get("query_row_count"), int)
        or not 0 < report["query_row_count"] < count * ROWS
    ):
        raise ValueError(
            "Evaluation must score the pinned final checkpoint on the full expected bank"
        )
    for name in METRIC_NAMES:
        value = report.get("metrics", {}).get(name)
        if (
            report.get("metric_task_counts", {}).get(name) != count
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or (name != "nll" and value > 1)
        ):
            raise ValueError(f"Incomplete or invalid common-support metric: {name}")


def _trial_result(spec, cache):
    result = json.loads(Path(spec["result_path"]).read_text(encoding="utf-8"))
    if (
        result.get("format") != "nano_graph_u_pool_trial_v1"
        or result.get("trial") != spec
    ):
        raise ValueError(
            "Validation result belongs to a different common-support candidate"
        )
    checkpoint = Path(spec["checkpoint_dir"]) / "latest.pt"
    _checkpoint(checkpoint, spec, spec["learning_rate"], cache)
    _report(
        result["validation_report"],
        checkpoint,
        _bank(
            spec["root"],
            spec["mode"],
            spec["condition"],
            "validation",
            spec["repetition"],
        ),
        cache,
    )
    return result


def run_trial(root, task_id, *, device="cuda"):
    spec = trial_spec(root, task_id)
    _initialize(root)
    with _lock(Path(spec["directory"]) / ".trial.lock"):
        for split in ("train", "validation"):
            _validate_bank(
                _bank(root, spec["mode"], spec["condition"], split, spec["repetition"])
            )
        cache = {}
        if not Path(spec["result_path"]).exists():
            checkpoint_dir = Path(spec["checkpoint_dir"])
            checkpoint = train_model(
                spec["train_dump"],
                checkpoint_dir,
                steps=TRAIN_STEPS,
                epochs=1,
                shuffle_batches=True,
                batch_size=BATCH_SIZE,
                seed=spec["model_seed"],
                device=device,
                learning_rate=spec["learning_rate"],
                save_every=250,
                log_every=25,
                resume=(checkpoint_dir / "latest.pt").is_file(),
            )
            _checkpoint(checkpoint, spec, spec["learning_rate"], cache)
            report = evaluate_checkpoint(
                checkpoint,
                spec["validation_dump"],
                batch_size=BATCH_SIZE,
                device=device,
            )
            _report(
                report,
                checkpoint,
                _bank(
                    root,
                    spec["mode"],
                    spec["condition"],
                    "validation",
                    spec["repetition"],
                ),
                cache,
            )
            _json(
                spec["result_path"],
                {
                    "format": "nano_graph_u_pool_trial_v1",
                    "trial": spec,
                    "validation_report": report,
                },
            )
        result = _trial_result(spec, cache)
    print(
        f"POOL_TRIAL_COMPLETE task={task_id} lr={spec['learning_rate']} "
        f"validation_auroc={result['validation_report']['metrics']['roc_auc']:.6f}",
        flush=True,
    )
    return result


def select_winners(root):
    """Require every full-budget candidate; select using matching validation only."""
    root = _initialize(root)
    with _lock(root / ".selection.lock"):
        specs = [trial_spec(root, task) for task in range(1, TRIAL_COUNT + 1)]
        missing = [
            spec["result_path"]
            for spec in specs
            if not Path(spec["result_path"]).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Cannot select: {len(missing)} of {TRIAL_COUNT} validation results missing; first: {missing[0]}"
            )
        cache = {}
        trials = [_trial_result(spec, cache) for spec in specs]
        winners = []
        for task_id in range(1, WINNER_COUNT + 1):
            spec = winner_spec(root, task_id)
            candidates = [
                result
                for result in trials
                if all(
                    result["trial"][key] == spec[key]
                    for key in ("mode", "condition", "repetition")
                )
            ]
            selected = max(
                candidates,
                key=lambda result: (
                    result["validation_report"]["metrics"]["roc_auc"],
                    -result["trial"]["learning_rate"],
                ),
            )
            trial, report = selected["trial"], selected["validation_report"]
            winner = {
                "task_id": task_id,
                "mode": spec["mode"],
                "condition": spec["condition"],
                "repetition": spec["repetition"],
                "model_seed": spec["model_seed"],
                "learning_rate": trial["learning_rate"],
                "validation_roc_auc": report["metrics"]["roc_auc"],
                "source_checkpoint": report["checkpoint"],
                "checkpoint": spec["checkpoint"],
                "checkpoint_sha256": report["checkpoint_sha256"],
                "validation_dump": report["dump"],
                "validation_dump_sha256": report["dump_sha256"],
                "train_dump": spec["train_dump"],
                "train_dump_sha256": _digest(Path(spec["train_dump"]), cache),
            }
            winners.append(winner)
        for winner in winners:
            _retain_checkpoint(
                Path(winner["source_checkpoint"]),
                Path(winner["checkpoint"]),
                winner["checkpoint_sha256"],
            )
            _json(Path(winner["checkpoint"]).parent / "winner.json", winner)
            print(
                f"rep={winner['repetition']} mode={winner['mode']} condition={winner['condition']} "
                f"lr={winner['learning_rate']} validation_auc={winner['validation_roc_auc']:.6f}",
                flush=True,
            )
        manifest = {
            "format": "nano_graph_u_pool_winners_v1",
            "recipe": _recipe(),
            "winners": winners,
        }
        _json(root / "winners" / "winners.json", manifest)
    print(
        f"POOL_SELECTION_COMPLETE winners={WINNER_COUNT} manifest={root / 'winners' / 'winners.json'}",
        flush=True,
    )
    return manifest


def _winner(spec, cache):
    manifest = json.loads(
        (Path(spec["root"]) / "winners" / "winners.json").read_text(encoding="utf-8")
    )
    records = manifest.get("winners", [])
    if (
        manifest.get("format") != "nano_graph_u_pool_winners_v1"
        or manifest.get("recipe") != _recipe()
        or len(records) != WINNER_COUNT
        or [item.get("task_id") for item in records] != list(range(1, WINNER_COUNT + 1))
    ):
        raise ValueError("Invalid or incomplete frozen winner manifest")
    winner = records[spec["task_id"] - 1]
    if (
        any(
            winner.get(key) != spec[key]
            for key in (
                "mode",
                "condition",
                "repetition",
                "model_seed",
                "checkpoint",
                "train_dump",
                "validation_dump",
            )
        )
        or winner.get("learning_rate") not in LEARNING_RATES
        or json.loads(
            (Path(spec["directory"]) / "winner.json").read_text(encoding="utf-8")
        )
        != winner
    ):
        raise ValueError(
            "Selected winner does not match its frozen repetition and condition"
        )
    _checkpoint(winner["checkpoint"], spec, winner["learning_rate"], cache)
    if (
        _digest(Path(winner["checkpoint"]), cache) != winner["checkpoint_sha256"]
        or _digest(Path(spec["train_dump"]), cache) != winner["train_dump_sha256"]
    ):
        raise ValueError("Selected checkpoint or training bank changed after selection")
    return winner


def _score(path, winner, bank, cache, device=None):
    path = Path(path)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("winner") != winner:
            raise ValueError("Existing score belongs to a different frozen winner")
        report = saved["evaluation_report"]
    else:
        if device is None:
            raise FileNotFoundError(f"Missing completed evaluation: {path}")
        report = evaluate_checkpoint(
            winner["checkpoint"],
            bank["dump"],
            batch_size=BATCH_SIZE,
            device=device,
            allow_training_dump=bank["split"] == "train",
        )
    _report(report, winner["checkpoint"], bank, cache)
    _json(path, {"winner": winner, "evaluation_report": report})
    return report


def _rows(spec, winner, cache, device=None):
    common = {
        "training_graph_u_mode": spec["mode"],
        "training_condition": spec["condition"],
        "repetition": spec["repetition"],
        "model_seed": spec["model_seed"],
        "epochs": 1,
        "training_steps": TRAIN_STEPS,
        "learning_rate": winner["learning_rate"],
        "validation_roc_auc": winner["validation_roc_auc"],
        "rows": ROWS,
        "features": FEATURES,
        "shuffle_batches": True,
        "batch_order_scheme": BATCH_ORDER_SCHEME,
        "checkpoint": winner["checkpoint"],
        "checkpoint_sha256": winner["checkpoint_sha256"],
    }
    results, tests = Path(spec["results_directory"]), []
    for mode in MODES:
        for condition in CONDITIONS:
            bank = _bank(spec["root"], mode, condition, "test", spec["repetition"])
            report = _score(
                results / f"test_{mode}_{condition}.json", winner, bank, cache, device
            )
            tests.append(
                {
                    **common,
                    "test_graph_u_mode": mode,
                    "test_condition": condition,
                    "task_count": report["task_count"],
                    "query_row_count": report["query_row_count"],
                    **report["metrics"],
                    "test_dump": report["dump"],
                    "test_dump_sha256": report["dump_sha256"],
                }
            )
    bank = _bank(
        spec["root"], spec["mode"], spec["condition"], "train", spec["repetition"]
    )
    report = _score(results / "training.json", winner, bank, cache, device)
    training = {
        **common,
        "evaluation_split": "training",
        "scored_rows": "query",
        "task_count": report["task_count"],
        "query_row_count": report["query_row_count"],
        **report["metrics"],
        "training_dump": report["dump"],
        "training_dump_sha256": report["dump_sha256"],
    }
    if _digest(Path(winner["checkpoint"]), cache) != winner["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed during evaluation")
    return tests, training


def evaluate_winner(root, task_id, *, device="cuda"):
    spec = winner_spec(root, task_id)
    _initialize(root)
    with _lock(Path(spec["results_directory"]) / ".test.lock"):
        cache = {}
        winner = _winner(spec, cache)
        tests, own = _rows(spec, winner, cache, device)
        result = {
            "format": "nano_graph_u_pool_test_v1",
            "spec": spec,
            "winner": winner,
            "test_rows": tests,
            "training_row": own,
        }
        _json(spec["result_path"], result)
    print(
        f"POOL_TEST_COMPLETE task={task_id} test_banks=6 training_banks=1 result={spec['result_path']}",
        flush=True,
    )
    return result


def _aggregate(rows, keys):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    cells = []
    for group, values in sorted(groups.items()):
        if sorted(row["repetition"] for row in values) != list(REPETITIONS):
            raise ValueError(
                "Every summary cell must contain exactly the five repetitions"
            )
        cells.append(
            {
                **dict(zip(keys, group)),
                "repetition_count": len(REPETITIONS),
                **{
                    f"{name}_{stat}": function(row[name] for row in values)
                    for name in METRIC_NAMES
                    for stat, function in (("mean", mean), ("std", stdev))
                },
            }
        )
    return cells


def summarize_results(root):
    root = _initialize(root)
    with _lock(root / ".summary.lock"):
        specs = [winner_spec(root, task) for task in range(1, WINNER_COUNT + 1)]
        missing = [
            spec["result_path"]
            for spec in specs
            if not Path(spec["result_path"]).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Cannot summarize: {len(missing)} of {WINNER_COUNT} winner evaluations missing; first: {missing[0]}"
            )
        tests, training, winners, cache = [], [], [], {}
        for spec in specs:
            winner = _winner(spec, cache)
            rows, own = _rows(spec, winner, cache)
            expected = {
                "format": "nano_graph_u_pool_test_v1",
                "spec": spec,
                "winner": winner,
                "test_rows": rows,
                "training_row": own,
            }
            if (
                json.loads(Path(spec["result_path"]).read_text(encoding="utf-8"))
                != expected
            ):
                raise ValueError(
                    "Completed winner result differs from its pinned evaluations"
                )
            tests.extend(rows)
            training.append(own)
            winners.append(winner)
        if len(tests) != 180 or len(training) != 30:
            raise ValueError(
                "The complete design requires 180 test and 30 training rows"
            )
        tests.sort(
            key=lambda row: tuple(row[key] for key in (*_TEST_KEYS, "repetition"))
        )
        training.sort(
            key=lambda row: tuple(row[key] for key in (*_MODEL_KEYS, "repetition"))
        )
        cells, own_cells = (
            _aggregate(tests, _TEST_KEYS),
            _aggregate(training, _MODEL_KEYS),
        )
        if len(cells) != 36 or len(own_cells) != 6:
            raise ValueError(
                "The complete design requires 36 test and six training summary cells"
            )
        output = root / "results"
        output.mkdir(parents=True, exist_ok=True)
        for name, values in (
            ("summary.csv", cells),
            ("per_repetition.csv", tests),
            ("training_summary.csv", own_cells),
            ("training_per_repetition.csv", training),
            ("winners.csv", winners),
        ):
            _atomic_write(output / name, _csv_bytes(values, list(values[0])))
        summary = {
            "format": "nano_graph_u_pool_summary_v1",
            "recipe": _recipe(),
            "cells": cells,
            "per_repetition": tests,
            "training_cells": own_cells,
            "training_rows": training,
            "winners": winners,
        }
        _json(output / "summary.json", summary)
    print(
        f"POOL_SUMMARY_COMPLETE test_rows=180 training_rows=30 test_cells=36 results={output}",
        flush=True,
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    for name in ("generate", "trial", "select", "test", "summarize", "repair"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        if name in ("generate", "trial", "test"):
            command.add_argument("--task-id", type=int, required=True)
        if name == "generate":
            command.add_argument("--workers", type=int, default=8)
        if name in ("trial", "test"):
            command.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.stage == "generate":
        generate_bank(args.root, args.task_id, workers=args.workers)
    elif args.stage == "trial":
        run_trial(args.root, args.task_id, device=args.device)
    elif args.stage == "select":
        select_winners(args.root)
    elif args.stage == "test":
        evaluate_winner(args.root, args.task_id, device=args.device)
    elif args.stage == "repair":
        from .pool_repair import repair_banks

        repair_banks(args.root)
    else:
        summarize_results(args.root)


if __name__ == "__main__":
    main()
