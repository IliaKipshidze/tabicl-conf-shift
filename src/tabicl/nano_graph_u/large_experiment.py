"""Fixed-shift Nano experiment with 1,000-row, eight-feature tasks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from ._settings import experiment_prior_config
from .data import NanoPriorDumpLoader, _metadata, generate_dump
from .epoch_experiments import _json
from .evaluate import evaluate_checkpoint
from .sampled import MODES, _lock
from .select import _atomic_write, _retain_checkpoint
from .summarize_tests import _csv_bytes
from .test_winners import METRIC_NAMES, _digest
from .train import (
    BATCH_ORDER_SCHEME,
    PAPER_MODEL_CONFIG,
    _schedulefree_adamw,
    train_model,
)

ROWS, FEATURES, BATCH_SIZE = 1000, 8, 32
TRAIN_STEPS, EVAL_STEPS = 15625, 1000
EPOCHS, LEARNING_RATES = (1, 3, 9), (0.0007, 0.001, 0.002)
CONDITIONS, SPLITS = ("identity", "shift"), ("train", "validation", "test")
_TEST_KEYS = ("epochs", "training_graph_u_mode", "training_condition", "test_graph_u_mode", "test_condition")


def _recipe():
    return {"format": "nano_graph_u_large_experiment_v1", "rows": ROWS, "features": FEATURES,
            "batch_size": BATCH_SIZE, "train_tasks": TRAIN_STEPS * BATCH_SIZE,
            "validation_tasks": EVAL_STEPS * BATCH_SIZE, "test_tasks": EVAL_STEPS * BATCH_SIZE,
            "epochs": list(EPOCHS), "learning_rates": list(LEARNING_RATES), "repetitions": [1],
            "model_seed": 42, "shift_location": 2.0, "shift_scale": 1.5,
            "shuffle_batches": True, "batch_order_scheme": BATCH_ORDER_SCHEME,
            "model_config": PAPER_MODEL_CONFIG, "selection_metric": "matching_validation_roc_auc"}


def _initialize(root):
    root = Path(root).expanduser().resolve()
    if root == Path(root.anchor):
        raise ValueError("Experiment root must be a dedicated directory")
    with _lock(root / ".experiment.lock", blocking=True):
        path = root / "experiment.json"
        if not path.exists() and any(p.name != ".experiment.lock" for p in root.iterdir()):
            raise ValueError("Refusing to use a non-empty directory belonging to another experiment")
        _json(path, _recipe())
    return root


def _index(task_id, count):
    if isinstance(task_id, bool) or not isinstance(task_id, int) or not 1 <= task_id <= count:
        raise ValueError(f"task_id must be an integer from 1 to {count}")
    return task_id - 1


def bank_spec(root, task_id):
    kind, split_index = divmod(_index(task_id, 12), 3)
    mode, condition, split = MODES[kind // 2], CONDITIONS[kind % 2], SPLITS[split_index]
    steps = TRAIN_STEPS if split == "train" else EVAL_STEPS
    seed = {"train": 1001, "validation": 3001, "test": 2001}[split]
    root = Path(root).expanduser().resolve()
    return {"root": root, "task_id": task_id, "mode": mode, "condition": condition,
            "split": split, "steps": steps, "seed": seed,
            "dump": root / "data" / mode / condition
                    / f"{split}_seed{seed}_{steps}steps_b32_r1000_f8.h5"}


def _bank(root, mode, condition, split):
    kind = MODES.index(mode) * 2 + CONDITIONS.index(condition)
    return bank_spec(root, kind * 3 + SPLITS.index(split) + 1)


def _validate_bank(spec, *, complete=True):
    loader = NanoPriorDumpLoader(spec["dump"], batch_size=BATCH_SIZE)
    expected = _metadata(experiment_prior_config(spec["condition"], spec["mode"]),
        steps=spec["steps"], batch_size=BATCH_SIZE, seed=spec["seed"], rows=ROWS, features=FEATURES,
        min_train_size=0.3, max_train_size=0.9, n_jobs=1)
    if loader.metadata != expected:
        raise ValueError(f"Bank does not match the large experiment's shape/prior/seed: {spec['dump']}")
    if not 0 <= loader.committed_steps <= spec["steps"] or (complete and loader.committed_steps != spec["steps"]):
        raise ValueError(f"Incomplete {spec['split']} bank: {loader.committed_steps}/{spec['steps']}")
    return loader.metadata


def generate_bank(root, task_id, *, workers=8):
    spec = bank_spec(root, task_id)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    _initialize(root)
    with _lock(spec["root"] / "locks" / f"bank_{task_id}.lock"):
        exists = spec["dump"].exists()
        if exists:
            _validate_bank(spec, complete=False)
        print(f"GENERATION_START task={task_id} split={spec['split']} tasks={spec['steps'] * BATCH_SIZE} "
              f"mode={spec['mode']} condition={spec['condition']} dump={spec['dump']}", flush=True)
        if not exists or NanoPriorDumpLoader(spec["dump"], batch_size=BATCH_SIZE).committed_steps < spec["steps"]:
            generate_dump(spec["dump"], steps=spec["steps"], batch_size=BATCH_SIZE, seed=spec["seed"],
                prior_config=experiment_prior_config(spec["condition"], spec["mode"]), rows=ROWS, features=FEATURES,
                min_train_size=0.3, max_train_size=0.9, n_jobs=1, workers=workers, resume=exists)
        _validate_bank(spec)
    print(f"LARGE_BANK_COMPLETE task={task_id} tasks={spec['steps'] * BATCH_SIZE}", flush=True)
    return spec


def _require_smoke(root):
    path = Path(root).resolve() / "smoke.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report.get("format") != "nano_graph_u_large_memory_smoke_v1" or report.get("status") != "passed"
        or report.get("recipe") != _recipe() or report.get("support_rows") != 900
        or not str(report.get("device", "")).startswith("cuda")):
        raise ValueError("Run the full batch-32 CUDA memory smoke for this exact experiment first")
    return report


def _check_capacity(smoke, device):
    if not device.startswith("cuda"):
        return
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot see the allocated GPU")
    if torch.cuda.get_device_properties(torch.device(device)).total_memory < smoke["peak_allocated_bytes"]:
        raise RuntimeError("This GPU has less memory than the full-size smoke required; retain batch size 32 and use a larger GPU")


def run_smoke(root, *, device="cuda"):
    """Probe the unchanged model's memory with a random batch; no production data."""
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("The large experiment memory smoke requires the allocated CUDA GPU")
    root = _initialize(root)
    with _lock(root / ".smoke.lock"):
        if (root / "smoke.json").exists():
            report = _require_smoke(root)
        else:
            from .model import NanoTabPFNModel

            torch.manual_seed(42)
            target_device = torch.device(device)
            torch.cuda.reset_peak_memory_stats(target_device)
            model = NanoTabPFNModel(**PAPER_MODEL_CONFIG).to(target_device).float().train()
            optimizer = _schedulefree_adamw(model.parameters(), learning_rate=0.001)
            optimizer.train()
            x = torch.randn(BATCH_SIZE, ROWS, FEATURES, device=target_device)
            y = (torch.arange(ROWS, device=target_device) % 2).expand(BATCH_SIZE, -1)
            logits = model((x, y[:, :900].float()), train_test_split_index=900)
            loss = F.cross_entropy(logits.reshape(-1, 2), y[:, 900:].reshape(-1))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss in the full-size CUDA smoke")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            torch.cuda.synchronize(target_device)
            report = {"format": "nano_graph_u_large_memory_smoke_v1", "status": "passed", "recipe": _recipe(),
                "support_rows": 900, "device": str(target_device), "gpu": torch.cuda.get_device_name(target_device),
                "torch_version": str(torch.__version__), "parameter_count": sum(p.numel() for p in model.parameters()),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(target_device), "query_nll": float(loss.detach())}
            _json(root / "smoke.json", report)
    print(f"LARGE_MEMORY_SMOKE_COMPLETE parameters={report['parameter_count']} "
          f"peak_allocated_GiB={report['peak_allocated_bytes'] / 2**30:.3f}", flush=True)
    return report


def _setting(root, budget, kind):
    mode, condition = MODES[kind // 2], CONDITIONS[kind % 2]
    root = Path(root).expanduser().resolve()
    return {"root": str(root), "epochs": EPOCHS[budget], "total_steps": TRAIN_STEPS * EPOCHS[budget],
            "mode": mode, "condition": condition, "model_seed": 42, "repetition": 1,
            "train_dump": str(_bank(root, mode, condition, "train")["dump"]),
            "validation_dump": str(_bank(root, mode, condition, "validation")["dump"])}


def trial_spec(root, task_id):
    budget, slot = divmod(_index(task_id, 36), 12)
    kind, lr_index = divmod(slot, 3)
    spec = _setting(root, budget, kind)
    rate = LEARNING_RATES[lr_index]
    directory = Path(spec["root"]) / "checkpoints" / f"epochs_{spec['epochs']}" / spec["mode"] / spec["condition"] \
                / f"lr_{format(rate, 'g').replace('.', 'p')}"
    return {**spec, "task_id": task_id, "learning_rate": rate, "directory": str(directory),
            "checkpoint_dir": str(directory / "checkpoints"), "result_path": str(directory / "validation.json")}


def winner_spec(root, task_id):
    budget, kind = divmod(_index(task_id, 12), 4)
    spec = _setting(root, budget, kind)
    directory = Path(spec["root"]) / "winners" / f"epochs_{spec['epochs']}" / spec["mode"] / spec["condition"]
    return {**spec, "task_id": task_id, "directory": str(directory), "checkpoint": str(directory / "model.pt"),
            "result_path": str(directory / "complete.json")}


def _checkpoint(path, spec, rate, cache):
    training = _validate_bank(_bank(spec["root"], spec["mode"], spec["condition"], "train"))
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = {"seed": 42, "batch_size": BATCH_SIZE, "learning_rate": rate,
        "optimizer": "schedulefree.AdamWScheduleFree", "weight_decay": 0.0, "gradient_clip_norm": 1.0,
        "loss": "query_cross_entropy", "precision": "float32", "dump_path": spec["train_dump"],
        "dump_sha256": _digest(Path(spec["train_dump"]), cache), "shuffle_batches": True,
        "batch_order_scheme": BATCH_ORDER_SCHEME}
    if spec["epochs"] != 1:
        config.update(epochs=spec["epochs"], steps_per_epoch=TRAIN_STEPS)
    if (payload.get("format") != "nano_graph_u_v1" or payload.get("step") != spec["total_steps"]
        or payload.get("model_config") != PAPER_MODEL_CONFIG or payload.get("dump_metadata") != training
        or payload.get("prior_config") != training["prior_config"] or payload.get("train_config") != config):
        raise ValueError("Checkpoint does not match this large-experiment model, LR or full epoch budget")


def _report(report, checkpoint, bank, total_steps, cache):
    metadata = _validate_bank(bank)
    count = bank["steps"] * BATCH_SIZE
    if (report.get("format") != "nano_graph_u_evaluation_v1" or report.get("checkpoint") != str(checkpoint)
        or report.get("checkpoint_sha256") != _digest(Path(checkpoint), cache)
        or report.get("checkpoint_step") != total_steps or report.get("dump") != str(bank["dump"])
        or report.get("dump_sha256") != _digest(bank["dump"], cache) or report.get("dump_metadata") != metadata
        or report.get("task_count") != count or isinstance(report.get("query_row_count"), bool)
        or not isinstance(report.get("query_row_count"), int) or not 0 < report["query_row_count"] < count * ROWS):
        raise ValueError("Evaluation must score the matching final checkpoint on the entire expected bank")
    for name in METRIC_NAMES:
        value = report.get("metrics", {}).get(name)
        if (report.get("metric_task_counts", {}).get(name) != count or isinstance(value, bool)
            or not isinstance(value, (int, float)) or not math.isfinite(value)
            or value < 0 or (name != "nll" and value > 1)):
            raise ValueError(f"Incomplete or invalid large-experiment metric: {name}")


def _trial_result(spec, cache):
    result = json.loads(Path(spec["result_path"]).read_text(encoding="utf-8"))
    if result.get("format") != "nano_graph_u_large_trial_v1" or result.get("trial") != spec:
        raise ValueError("Validation result belongs to a different large-experiment candidate")
    checkpoint = Path(spec["checkpoint_dir"]) / "latest.pt"
    _checkpoint(checkpoint, spec, spec["learning_rate"], cache)
    _report(result["validation_report"], checkpoint,
            _bank(spec["root"], spec["mode"], spec["condition"], "validation"), spec["total_steps"], cache)
    return result


def run_trial(root, task_id, *, device="cuda"):
    spec = trial_spec(root, task_id)
    smoke = _require_smoke(root)
    _check_capacity(smoke, device)
    _initialize(root)
    with _lock(Path(spec["directory"]) / ".trial.lock"):
        for split in ("train", "validation"):
            _validate_bank(_bank(root, spec["mode"], spec["condition"], split))
        if not Path(spec["result_path"]).exists():
            checkpoint_dir = Path(spec["checkpoint_dir"])
            checkpoint = train_model(spec["train_dump"], checkpoint_dir, steps=TRAIN_STEPS, epochs=spec["epochs"],
                shuffle_batches=True, batch_size=BATCH_SIZE, seed=42, device=device,
                learning_rate=spec["learning_rate"], save_every=2500, log_every=250,
                resume=(checkpoint_dir / "latest.pt").is_file())
            _checkpoint(checkpoint, spec, spec["learning_rate"], {})
            report = evaluate_checkpoint(checkpoint, spec["validation_dump"], batch_size=BATCH_SIZE, device=device)
            _report(report, checkpoint, _bank(root, spec["mode"], spec["condition"], "validation"), spec["total_steps"], {})
            _json(spec["result_path"], {"format": "nano_graph_u_large_trial_v1", "trial": spec, "validation_report": report})
        result = _trial_result(spec, {})
    print(f"LARGE_TRIAL_COMPLETE task={task_id} epochs={spec['epochs']} lr={spec['learning_rate']} "
          f"validation_auroc={result['validation_report']['metrics']['roc_auc']:.6f}", flush=True)
    return result


def _selected(spec, cache):
    budget = EPOCHS.index(spec["epochs"])
    kind = MODES.index(spec["mode"]) * 2 + CONDITIONS.index(spec["condition"])
    candidates = []
    for lr_index in range(3):
        trial = trial_spec(spec["root"], budget * 12 + kind * 3 + lr_index + 1)
        report = _trial_result(trial, cache)["validation_report"]
        candidates.append((report["metrics"]["roc_auc"], -trial["learning_rate"], trial, report))
    _, _, trial, report = max(candidates, key=lambda value: value[:2])
    return {"epochs": spec["epochs"], "training_graph_u_mode": spec["mode"],
            "training_condition": spec["condition"], "repetition": 1, "model_seed": 42,
            "learning_rate": trial["learning_rate"], "validation_roc_auc": report["metrics"]["roc_auc"],
            "source_checkpoint": report["checkpoint"], "checkpoint": spec["checkpoint"],
            "checkpoint_sha256": report["checkpoint_sha256"]}


def _score(path, spec, winner, bank, cache, device=None):
    path = Path(path)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("winner") != winner:
            raise ValueError("Existing score belongs to a different selected large-experiment model")
        report = saved["evaluation_report"]
    else:
        if device is None:
            raise FileNotFoundError(f"Missing completed evaluation: {path}")
        report = evaluate_checkpoint(winner["checkpoint"], bank["dump"], batch_size=BATCH_SIZE, device=device,
                                     allow_training_dump=bank["split"] == "train")
    _report(report, winner["checkpoint"], bank, spec["total_steps"], cache)
    _json(path, {"winner": winner, "evaluation_report": report})
    return report


def _rows(spec, winner, cache, device=None):
    common = {"epochs": spec["epochs"], "training_steps": spec["total_steps"], "rows": ROWS,
              "features": FEATURES, "fixed_location": 2.0, "fixed_scale": 1.5,
              "training_graph_u_mode": spec["mode"], "training_condition": spec["condition"],
              "training_query_location": 0.0 if spec["condition"] == "identity" else 2.0,
              "training_query_scale": 1.0 if spec["condition"] == "identity" else 1.5,
              "repetition": 1, "model_seed": 42, "learning_rate": winner["learning_rate"],
              "shuffle_batches": True, "batch_order_scheme": BATCH_ORDER_SCHEME,
              "checkpoint": winner["checkpoint"], "checkpoint_sha256": winner["checkpoint_sha256"]}
    tests = []
    for mode in MODES:
        for condition in CONDITIONS:
            bank = _bank(spec["root"], mode, condition, "test")
            report = _score(Path(spec["directory"]) / f"test_{mode}_{condition}.json", spec, winner, bank, cache, device)
            tests.append({**common, "test_graph_u_mode": mode, "test_condition": condition,
                          "test_query_location": 0.0 if condition == "identity" else 2.0,
                          "test_query_scale": 1.0 if condition == "identity" else 1.5,
                          "task_count": report["task_count"], "query_row_count": report["query_row_count"],
                          **report["metrics"], "test_dump": report["dump"], "test_dump_sha256": report["dump_sha256"]})
    bank = _bank(spec["root"], spec["mode"], spec["condition"], "train")
    report = _score(Path(spec["directory"]) / "training.json", spec, winner, bank, cache, device)
    training = {**common, "evaluation_split": "training", "scored_rows": "query",
                "task_count": report["task_count"], "query_row_count": report["query_row_count"],
                **report["metrics"], "training_dump": report["dump"], "training_dump_sha256": report["dump_sha256"]}
    _checkpoint(winner["checkpoint"], spec, winner["learning_rate"], cache)
    if _digest(Path(winner["checkpoint"]), cache) != winner["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed during evaluation")
    return tests, training


def _collect_if_complete(root):
    specs = [winner_spec(root, task) for task in range(1, 13)]
    if any(not Path(spec["result_path"]).is_file() for spec in specs):
        return False
    root = Path(root).resolve()
    with _lock(root / ".summary.lock", blocking=True):
        tests, training, winners, cache = [], [], [], {}
        for spec in specs:
            result = json.loads(Path(spec["result_path"]).read_text(encoding="utf-8"))
            winner = _selected(spec, cache)
            _checkpoint(winner["checkpoint"], spec, winner["learning_rate"], cache)
            if _digest(Path(winner["checkpoint"]), cache) != winner["checkpoint_sha256"]:
                raise ValueError("Selected checkpoint changed after evaluation")
            rows, own = _rows(spec, winner, cache)
            if result != {"format": "nano_graph_u_large_winner_v1", "spec": spec,
                          "winner": winner, "test_rows": rows, "training_row": own}:
                raise ValueError("Completed winner scores differ from their pinned inputs")
            tests.extend(rows)
            training.append(own)
            winners.append(winner)
        if len(tests) != 48 or len({tuple(row[key] for key in _TEST_KEYS) for row in tests}) != 48 or len(training) != 12:
            raise ValueError("Large experiment requires 48 test and 12 training comparisons")
        tests.sort(key=lambda row: tuple(row[key] for key in _TEST_KEYS))
        model_keys = _TEST_KEYS[:3]
        cells = [{**{key: row[key] for key in _TEST_KEYS}, "repetition_count": 1,
                  "rows": ROWS, "features": FEATURES, "train_tasks": TRAIN_STEPS * BATCH_SIZE,
                  "test_tasks": EVAL_STEPS * BATCH_SIZE, "fixed_location": 2.0, "fixed_scale": 1.5,
                  **{f"{name}_mean": row[name] for name in METRIC_NAMES}} for row in tests]
        own_cells = [{**{key: row[key] for key in model_keys}, "repetition_count": 1,
                      "evaluation_split": "training", "scored_rows": "query", "task_count_per_repetition": 500000,
                      **{f"{name}_mean": row[name] for name in METRIC_NAMES}} for row in training]
        output = root / "results"
        output.mkdir(parents=True, exist_ok=True)
        for name, values in (("summary.csv", cells), ("per_repetition.csv", tests),
                             ("training_summary.csv", own_cells), ("training_per_repetition.csv", training),
                             ("winners.csv", winners)):
            _atomic_write(output / name, _csv_bytes(values, list(values[0])))
        _json(output / "summary.json", {"format": "nano_graph_u_large_summary_v1",
                                       "recipe": _recipe(), "cells": cells, "per_repetition": tests,
                                       "training_rows": training, "winners": winners})
    print(f"LARGE_RESULTS_COMPLETE winners=12 test_rows=48 training_rows=12 results={output}", flush=True)
    return True


def evaluate_winner(root, task_id, *, device="cuda"):
    spec = winner_spec(root, task_id)
    smoke = _require_smoke(root)
    _check_capacity(smoke, device)
    _initialize(root)
    with _lock(Path(spec["directory"]) / ".winner.lock"):
        cache = {}
        for mode in MODES:
            for condition in CONDITIONS:
                _validate_bank(_bank(root, mode, condition, "test"))
        winner = _selected(spec, cache)
        _retain_checkpoint(Path(winner["source_checkpoint"]), Path(winner["checkpoint"]), winner["checkpoint_sha256"])
        _json(Path(spec["directory"]) / "winner.json", winner)
        tests, own = _rows(spec, winner, cache, device)
        result = {"format": "nano_graph_u_large_winner_v1", "spec": spec,
                  "winner": winner, "test_rows": tests, "training_row": own}
        _json(spec["result_path"], result)
    _collect_if_complete(root)
    print(f"LARGE_WINNER_COMPLETE task={task_id} epochs={spec['epochs']} lr={winner['learning_rate']}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    for name in ("smoke", "generate", "trial", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        if name != "smoke":
            command.add_argument("--task-id", type=int, required=True)
        if name == "generate":
            command.add_argument("--workers", type=int, default=8)
        else:
            command.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.stage == "smoke":
        run_smoke(args.root, device=args.device)
    elif args.stage == "generate":
        generate_bank(args.root, args.task_id, workers=args.workers)
    elif args.stage == "trial":
        run_trial(args.root, args.task_id, device=args.device)
    else:
        evaluate_winner(args.root, args.task_id, device=args.device)


if __name__ == "__main__":
    main()
