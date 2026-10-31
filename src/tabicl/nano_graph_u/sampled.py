"""Separate, manually submitted stages of the sampled-shift Nano experiment."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from tabicl.prior.graph_lib._config import GRAPH_U_SAMPLING_DEFAULTS

from ._settings import experiment_prior_config, normalize_sampling_config
from .data import NanoPriorDumpLoader, generate_dump
from .select import _atomic_write, _check_existing, select_winners
from .summarize_tests import _csv_bytes, summarize_tests
from .test_winners import (
    _load_winner,
    _validate_checkpoint,
    _validate_test_metadata,
    run_test,
    test_spec,
    validate_test_result,
)
from .train import PAPER_MODEL_CONFIG
from .tune import LEARNING_RATES, _validate_metadata, run_trial, trial_spec, validate_trial_result

MODES = ("confounding", "covariate")
_FORMAT = "nano_graph_u_sampled_experiment_v1"
_PROTOCOL = {
    "repetitions": 5, "training_steps": 2500, "heldout_steps": 100,
    "batch_size": 32, "rows": 150, "features": 5,
    "learning_rates": list(LEARNING_RATES), "model_config": PAPER_MODEL_CONFIG,
}


def _slot(task_id: int, count: int) -> int:
    if isinstance(task_id, bool) or not isinstance(task_id, int) or not 1 <= task_id <= count:
        raise ValueError(f"task-id must be an integer from 1 to {count}")
    return task_id - 1


@contextmanager
def _lock(path: Path, *, blocking: bool = False):
    """Use OS locks, automatically released even if a job is killed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            operation = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
            try:
                msvcrt.locking(stream.fileno(), operation, 1)
            except OSError as exc:
                raise RuntimeError(f"Another job is using {path}") from exc
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            except BlockingIOError as exc:
                raise RuntimeError(f"Another job is using {path}") from exc
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)


def load_experiment(root: Path) -> dict[str, Any]:
    payload = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    if payload.get("format") != _FORMAT or payload.get("protocol") != _PROTOCOL:
        raise ValueError("Experiment manifest uses a different scientific protocol")
    return normalize_sampling_config(payload["sampling_config"])


def initialize_experiment(root: Path, overrides: dict[str, Any]) -> dict[str, Any]:
    """Publish one immutable recipe, safely shared by generation array jobs."""
    with _lock(root / ".experiment.lock", blocking=True):
        path = root / "experiment.json"
        settings = load_experiment(root) if path.exists() else {
            **GRAPH_U_SAMPLING_DEFAULTS, "graph_u_shift_mode": "sampled"
        }
        settings = normalize_sampling_config({**settings, **overrides})
        payload = {"format": _FORMAT, "protocol": _PROTOCOL, "sampling_config": settings}
        contents = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        _check_existing(path, contents)
        _atomic_write(path, contents)
    return settings


def bank_spec(root: Path, task_id: int) -> dict[str, Any]:
    mode_index, slot = divmod(_slot(task_id, 44), 22)
    mode = MODES[mode_index]
    condition = "identity" if slot % 2 == 0 else "shift"
    if slot < 20:
        repetition = slot // 4 + 1
        split = "train" if slot % 4 < 2 else "test"
        seed = (1000 if split == "train" else 2000) + repetition
        subdir = f"rep_{repetition}"
    else:
        repetition, split, seed, subdir = None, "validation", 3001, "validation"
    steps = 2500 if split == "train" else 100
    return {
        "task_id": task_id, "graph_u_mode": mode, "condition": condition,
        "repetition": repetition, "split": split, "seed": seed, "steps": steps,
        "dump": root / "data" / mode / subdir
        / f"{condition}_{split}_seed{seed}_{steps}steps_b32_r150_f5.h5",
    }


def trial_for_task(root: Path, task_id: int, settings: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    mode_index, slot = divmod(_slot(task_id, 200), 100)
    mode = MODES[mode_index]
    return mode, trial_spec(
        slot + 1, root / "data" / mode, root / "checkpoints" / mode,
        graph_u_mode=mode, sampling_config=settings,
    )


def test_for_task(root: Path, task_id: int, settings: dict[str, Any]) -> dict[str, Any]:
    pair, slot = divmod(_slot(task_id, 80), 20)
    training_mode, test_mode = MODES[pair // 2], MODES[pair % 2]
    return test_spec(
        slot + 1, root / "data" / test_mode, root / "checkpoints" / training_mode,
        root / "results" / f"{training_mode}_on_{test_mode}",
        graph_u_mode=training_mode, test_graph_u_mode=test_mode,
        training_data_root=root / "data" / training_mode, sampling_config=settings,
    )


def _bank_status(spec: dict[str, Any], settings: dict[str, Any], *, complete: bool = True):
    loader = NanoPriorDumpLoader(spec["dump"], batch_size=32)
    kwargs = {"graph_u_mode": spec["graph_u_mode"], "sampling_config": settings}
    if spec["split"] == "test":
        _validate_test_metadata(loader.metadata, spec["condition"], spec["repetition"], **kwargs)
    else:
        _validate_metadata(
            loader.metadata, spec["condition"], spec["split"], spec["repetition"], **kwargs
        )
    if not 0 <= loader.committed_steps <= spec["steps"]:
        raise ValueError("Bank has an invalid number of committed batches")
    if complete and loader.committed_steps != spec["steps"]:
        raise ValueError(f"Incomplete bank: {loader.committed_steps}/{spec['steps']} batches committed")
    return loader


def generate_task(root: Path, task_id: int, settings: dict[str, Any], *, workers: int):
    spec = bank_spec(root, task_id)
    with _lock(spec["dump"].with_suffix(".h5.job.lock")):
        exists = spec["dump"].exists()
        if exists and _bank_status(spec, settings, complete=False).committed_steps == spec["steps"]:
            print(f"Bank already complete: task={task_id} dump={spec['dump']}", flush=True)
            return
        generate_dump(
            spec["dump"], steps=spec["steps"], batch_size=32, seed=spec["seed"],
            prior_config=experiment_prior_config(spec["condition"], spec["graph_u_mode"], settings),
            rows=150, features=5, min_train_size=0.3, max_train_size=0.9,
            n_jobs=1, workers=workers, resume=exists,
        )
        _bank_status(spec, settings)
    print(f"BANK_READY task={task_id} dump={spec['dump']}", flush=True)


def check_stage(root: Path, stage: str, settings: dict[str, Any]) -> bool:
    """Read-only checks, reporting global array indices that need attention."""
    counts = {"data": 44, "tune": 200, "select": 2, "test": 80}
    failures, fingerprints, hash_cache = [], set(), {}
    for task in range(1, counts[stage] + 1):
        try:
            if stage == "data":
                metadata = _bank_status(bank_spec(root, task), settings).metadata
                fingerprints.add(metadata["generator_source_sha256"])
            elif stage == "tune":
                _, spec = trial_for_task(root, task, settings)
                result = json.loads(Path(spec["result_path"]).read_text(encoding="utf-8"))
                validate_trial_result(result, spec, hash_cache=hash_cache)
            elif stage == "select":
                for local in (rep * 4 + offset for rep in range(5) for offset in (1, 3)):
                    spec = test_for_task(root, (task - 1) * 60 + local, settings)
                    manifest, winner, _ = _load_winner(spec)
                    _validate_checkpoint(winner, spec, manifest["generator_source_sha256"], hash_cache)
            else:
                spec = test_for_task(root, task, settings)
                result = json.loads(Path(spec["result_path"]).read_text(encoding="utf-8"))
                validate_test_result(result, spec, hash_cache=hash_cache)
        except Exception as exc:
            failures.append(task)
            print(f"INVALID stage={stage} task={task}: {exc}", flush=True)
    if len(fingerprints) > 1:
        print("DATA_NOT_READY: banks were generated by different prior code versions", flush=True)
        return False
    if failures:
        print(f"CHECK_FAILED stage={stage} array_tasks={','.join(map(str, failures))}", flush=True)
        return False
    print(f"STAGE_READY stage={stage} completed={counts[stage]}/{counts[stage]}", flush=True)
    return True


def summarize_experiment(root: Path, settings: dict[str, Any]):
    rows, cells = [], []
    for training_mode in MODES:
        for test_mode in MODES:
            summary = summarize_tests(
                root / "data" / test_mode, root / "checkpoints" / training_mode,
                root / "results" / f"{training_mode}_on_{test_mode}",
                graph_u_mode=training_mode, test_graph_u_mode=test_mode,
                training_data_root=root / "data" / training_mode, sampling_config=settings,
            )
            modes = {"training_graph_u_mode": training_mode, "test_graph_u_mode": test_mode}
            rows.extend({**row, **modes} for row in summary["per_repetition"])
            cells.extend({**cell, **modes} for cell in summary["cells"])
    # All four model settings must reuse the same bank for a given test setting.
    models, banks = {}, {}
    for row in rows:
        model_key = (row["training_graph_u_mode"], row["repetition"], row["training_condition"])
        bank_key = (row["test_graph_u_mode"], row["repetition"], row["test_condition"])
        if models.setdefault(model_key, row["checkpoint_sha256"]) != row["checkpoint_sha256"]:
            raise ValueError("A selected model changed between test settings")
        if banks.setdefault(bank_key, row["test_dump_sha256"]) != row["test_dump_sha256"]:
            raise ValueError("Different models were evaluated on different test banks")
    combined = {
        "format": "nano_graph_u_sampled_summary_v1", "sampling_config": settings,
        "evaluation_count": 80, "repetition_count": 5, "standard_deviation_ddof": 1,
        "cells": cells, "per_repetition": rows,
    }
    csv_rows = [{
        **{key: value for key, value in cell.items() if key != "metrics"},
        **{f"{metric}_{stat}": value for metric, stats in cell["metrics"].items() for stat, value in stats.items()},
    } for cell in cells]
    outputs = {
        "summary.json": (json.dumps(combined, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
        "summary.csv": _csv_bytes(csv_rows, list(csv_rows[0])),
        "per_repetition.csv": _csv_bytes(rows, list(rows[0])),
    }
    for name, contents in outputs.items():
        _check_existing(root / "results" / name, contents)
    for name, contents in outputs.items():
        _atomic_write(root / "results" / name, contents)
    print(f"SUMMARY_READY evaluations=80 cells=16 path={root / 'results' / 'summary.csv'}", flush=True)
    return combined


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for stage in ("generate", "tune", "select", "test", "summarize", "check"):
        command = subcommands.add_parser(stage)
        command.add_argument("--root", required=True, type=Path)
        if stage in {"generate", "tune", "select", "test"}:
            command.add_argument("--task-id", required=True, type=int)
        if stage in {"tune", "test"}:
            command.add_argument("--device", default="cuda")
        if stage == "generate":
            command.add_argument("--workers", type=int, default=8)
            for parameter in ("scale", "location"):
                for statistic in ("mean", "std", "min", "max"):
                    command.add_argument(f"--query-{parameter}-{statistic}", type=float)
        if stage == "check":
            command.add_argument("--stage", choices=("data", "tune", "select", "test"), required=True)
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    if args.command in {"generate", "tune", "select", "test"}:
        _slot(args.task_id, {"generate": 44, "tune": 200, "select": 2, "test": 80}[args.command])
    if args.command == "generate":
        if args.workers < 1:
            parser.error("workers must be positive")
        overrides = {
            f"graph_u_query_{parameter}_{statistic}": getattr(args, f"query_{parameter}_{statistic}")
            for parameter in ("scale", "location") for statistic in ("mean", "std", "min", "max")
            if getattr(args, f"query_{parameter}_{statistic}") is not None
        }
        settings = initialize_experiment(root, overrides)
        generate_task(root, args.task_id, settings, workers=args.workers)
        return
    settings = load_experiment(root)
    if args.command == "check":
        if not check_stage(root, args.stage, settings):
            raise SystemExit(1)
    elif args.command == "tune":
        mode, spec = trial_for_task(root, args.task_id, settings)
        with _lock(Path(spec["checkpoint_dir"]).parent / ".job.lock"):
            run_trial(
                spec["task_id"], root / "data" / mode, root / "checkpoints" / mode,
                device=args.device, graph_u_mode=mode, sampling_config=settings,
            )
    elif args.command == "select":
        mode = MODES[_slot(args.task_id, 2)]
        with _lock(root / "checkpoints" / mode / ".selection.job.lock"):
            manifest = select_winners(
                root / "data" / mode, root / "checkpoints" / mode,
                graph_u_mode=mode, sampling_config=settings,
            )
        print(f"WINNERS_READY mode={mode} count={len(manifest['winners'])}", flush=True)
    elif args.command == "test":
        pair, local = divmod(_slot(args.task_id, 80), 20)
        training_mode, test_mode = MODES[pair // 2], MODES[pair % 2]
        spec = test_for_task(root, args.task_id, settings)
        with _lock(Path(spec["result_path"]).with_suffix(".json.job.lock")):
            run_test(
                local + 1, root / "data" / test_mode, root / "checkpoints" / training_mode,
                root / "results" / f"{training_mode}_on_{test_mode}",
                device=args.device, graph_u_mode=training_mode, test_graph_u_mode=test_mode,
                training_data_root=root / "data" / training_mode, sampling_config=settings,
            )
    else:
        with _lock(root / ".summary.job.lock"):
            summarize_experiment(root, settings)


if __name__ == "__main__":
    main()
