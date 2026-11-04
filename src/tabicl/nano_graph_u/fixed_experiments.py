"""Repeat fixed-shift training while reusing the original fixed IID controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

from ._settings import experiment_prior_config, normalize_fixed_shift_config
from .data import NanoPriorDumpLoader, _generator_source_sha256, generate_dump
from .evaluate import evaluate_checkpoint
from .sampled import MODES, _lock
from .scale_sweep import _validate_report
from .select import _atomic_write, _check_existing, _retain_checkpoint
from .summarize_tests import _csv_bytes
from .test_winners import METRIC_NAMES, _digest, test_spec, validate_test_result
from .tune import (
    LEARNING_RATES,
    _validate_metadata,
    run_trial,
    trial_spec,
    validate_trial_result,
)

FIXED_SHIFTS = ((4.0, 1.5), (20.0, 2.0), (60.0, 3.0), (100.0, 4.0))


def experiment_spec(root, task_id):
    if isinstance(task_id, bool) or not isinstance(task_id, int) or not 1 <= task_id <= 4:
        raise ValueError("task_id must be an integer from 1 to 4")
    location, scale = FIXED_SHIFTS[task_id - 1]
    tag = lambda value: format(value, "g").replace(".", "p")
    return {
        "task_id": task_id, "location": location, "scale": scale,
        "directory": Path(root).expanduser().resolve() / f"loc_{tag(location)}_scale_{tag(scale)}",
        "fixed_shift_config": normalize_fixed_shift_config({
            "graph_u_query_location": location, "graph_u_query_scale": scale,
        }),
    }


def _source_roots(cluster, mode):
    suffix = "" if mode == "confounding" else "_covariate"
    return (
        cluster / "tabicl-conf-shift-priors/nano_graph_u" / f"repetitions{suffix}",
        cluster / "tabicl-conf-shift-checkpoints/nano_graph_u" / f"lr_search{suffix}",
    )


def _check_roots(cluster, root):
    protected = [path for mode in MODES for path in _source_roots(cluster, mode)]
    protected.append(cluster / "tabicl-conf-shift-evaluations")
    if root == cluster or any(root == path or root in path.parents or path in root.parents for path in protected):
        raise ValueError("Fixed experiment output must be separate from the original data, models and results")


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def _recipe(cluster, spec):
    return {
        "format": "nano_graph_u_fixed_experiment_v1", "cluster_root": str(cluster),
        "task_id": spec["task_id"], "fixed_shift_config": spec["fixed_shift_config"],
        "shift_mode": "fixed", "repetitions": 5, "train_steps": 2500,
        "evaluation_steps": 100, "batch_size": 32, "rows": 150, "features": 5,
        "train_seeds": list(range(1001, 1006)), "test_seeds": list(range(2001, 2006)),
        "validation_seed": 3001, "model_seeds": list(range(42, 47)),
        "learning_rates": list(LEARNING_RATES), "selection_metric": "roc_auc",
    }


def bank_specs(directory, profile):
    profile = normalize_fixed_shift_config(profile)
    specs = []
    for mode in MODES:
        for repetition in range(1, 6):
            for split, steps, seed in (("train", 2500, 1000 + repetition), ("test", 100, 2000 + repetition)):
                specs.append({
                    "graph_u_mode": mode, "split": split, "repetition": repetition,
                    "steps": steps, "seed": seed, "fixed_shift_config": profile,
                    "dump": Path(directory) / "data" / mode / f"rep_{repetition}"
                    / f"shift_{split}_seed{seed}_{steps}steps_b32_r150_f5.h5",
                })
        specs.append({
            "graph_u_mode": mode, "split": "validation", "repetition": None,
            "steps": 100, "seed": 3001, "fixed_shift_config": profile,
            "dump": Path(directory) / "data" / mode / "validation"
            / "shift_validation_seed3001_100steps_b32_r150_f5.h5",
        })
    return specs


def _bank_status(spec, *, complete=True):
    loader = NanoPriorDumpLoader(spec["dump"], batch_size=32)
    metadata = loader.metadata
    if spec["split"] == "test":
        if metadata.get("seed") != spec["seed"]:
            raise ValueError("Fixed test bank has the wrong independent seed")
        metadata = dict(metadata, seed=3001)
    _validate_metadata(
        metadata, "shift", "validation" if spec["split"] == "test" else spec["split"],
        spec["repetition"], graph_u_mode=spec["graph_u_mode"],
        fixed_shift_config=spec["fixed_shift_config"],
    )
    if not 0 <= loader.committed_steps <= spec["steps"] or (complete and loader.committed_steps != spec["steps"]):
        raise ValueError(f"Incomplete bank: {loader.committed_steps}/{spec['steps']} batches: {spec['dump']}")
    return loader


def _generate_banks(specs, workers):
    fingerprint = _generator_source_sha256()
    for spec in specs:
        if spec["dump"].exists() and _bank_status(spec, complete=False).metadata["generator_source_sha256"] != fingerprint:
            raise ValueError(f"Existing fixed bank was generated by different prior code: {spec['dump']}")
    for index, spec in enumerate(specs, 1):
        exists = spec["dump"].exists()
        if not exists or _bank_status(spec, complete=False).committed_steps < spec["steps"]:
            generate_dump(
                spec["dump"], steps=spec["steps"], batch_size=32, seed=spec["seed"],
                prior_config=experiment_prior_config("shift", spec["graph_u_mode"],
                    fixed_shift_config=spec["fixed_shift_config"]),
                rows=150, features=5, min_train_size=0.3, max_train_size=0.9,
                n_jobs=1, workers=workers, resume=exists,
            )
        _bank_status(spec)
        print(f"BANK_READY {index}/{len(specs)} mode={spec['graph_u_mode']} "
              f"split={spec['split']} rep={spec['repetition']}", flush=True)


def generate_experiment(cluster_root, root, task_id, *, workers=8):
    cluster, root = Path(cluster_root).resolve(), Path(root).resolve()
    _check_roots(cluster, root)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    spec = experiment_spec(root, task_id)
    with _lock(spec["directory"] / ".experiment.job.lock"):
        _write_json(spec["directory"] / "experiment.json", _recipe(cluster, spec))
        _generate_banks(bank_specs(spec["directory"], spec["fixed_shift_config"]), workers)
    print(f"GENERATION_COMPLETE task={task_id} banks=22/22 directory={spec['directory']}", flush=True)


def _row(report, model, test_mode, condition, path, origin):
    prior = report["dump_metadata"]["prior_config"]
    return {
        "training_graph_u_mode": model["training_graph_u_mode"],
        "training_condition": model["training_condition"], "test_graph_u_mode": test_mode,
        "test_condition": condition, "repetition": model["repetition"],
        "model_seed": model["model_seed"], "learning_rate": model["learning_rate"],
        "training_query_location": model["query_location"], "training_query_scale": model["query_scale"],
        "test_query_location": prior["graph_u_query_location"], "test_query_scale": prior["graph_u_query_scale"],
        "test_data_seed": report["dump_metadata"]["seed"], "task_count": report["task_count"],
        "query_row_count": report["query_row_count"],
        **{metric: report["metrics"][metric] for metric in METRIC_NAMES},
        "checkpoint": report["checkpoint"], "checkpoint_sha256": report["checkpoint_sha256"],
        "test_dump": report["dump"], "test_dump_sha256": report["dump_sha256"],
        "training_generator_source_sha256": model["generator_source_sha256"],
        "test_generator_source_sha256": report["dump_metadata"]["generator_source_sha256"],
        "result_path": str(path), "result_origin": origin,
    }


def load_original_iid(cluster_root):
    """Read only the historical fixed IID winners, IID banks and IID-on-IID reports."""
    cluster = Path(cluster_root).resolve()
    models, banks, rows, provenance, cache = {}, {}, [], {}, {}
    for mode in MODES:
        training_root, search_root = _source_roots(cluster, mode)
        for test_mode in MODES:
            test_root, _ = _source_roots(cluster, test_mode)
            suffix = ("" if mode == "confounding" else "_covariate") if mode == test_mode else f"_{mode}_on_{test_mode}"
            result_root = cluster / "tabicl-conf-shift-evaluations/nano_graph_u" / f"lr_search_tests{suffix}"
            for repetition in range(1, 6):
                spec = test_spec(
                    (repetition - 1) * 4 + 1, test_root, search_root, result_root,
                    graph_u_mode=mode, test_graph_u_mode=test_mode,
                    training_data_root=training_root if mode != test_mode else None,
                )
                path = Path(spec["result_path"])
                result = json.loads(path.read_text(encoding="utf-8"))
                validate_test_result(result, spec, verify_files=True, hash_cache=cache)
                report, winner = result["evaluation_report"], result["winner"]
                model = {
                    "training_graph_u_mode": mode, "training_condition": "identity", "repetition": repetition,
                    "model_seed": winner["model_seed"], "learning_rate": winner["learning_rate"],
                    "checkpoint": report["checkpoint"], "checkpoint_sha256": report["checkpoint_sha256"],
                    "query_location": 0.0, "query_scale": 1.0,
                    "generator_source_sha256": result.get("training_generator_source_sha256",
                        report["dump_metadata"]["generator_source_sha256"]),
                }
                if models.setdefault((mode, repetition, "identity"), model) != model:
                    raise ValueError("Historical IID reports use different selected models")
                bank = {"dump": Path(report["dump"]), "metadata": report["dump_metadata"], "sha256": report["dump_sha256"]}
                if banks.setdefault((test_mode, repetition), bank) != bank:
                    raise ValueError("Historical IID reports use different test banks")
                rows.append(_row(report, model, test_mode, "identity", path, "reused_iid"))
                provenance[str(path)] = _digest(path, cache)
    if len(models) != 10 or len(banks) != 10 or len(rows) != 20:
        raise ValueError("Expected ten original IID models, ten IID banks and twenty IID evaluations")
    return models, banks, rows, provenance


def _select_shifted(directory, mode, profile):
    data_root, output_root = directory / "data" / mode, directory / "checkpoints" / mode
    candidates, cache = [], {}
    for repetition in range(1, 6):
        for index in range(len(LEARNING_RATES)):
            spec = trial_spec((repetition - 1) * 20 + 11 + index, data_root, output_root,
                graph_u_mode=mode, fixed_shift_config=profile)
            result = json.loads(Path(spec["result_path"]).read_text(encoding="utf-8"))
            score = validate_trial_result(result, spec, verify_files=True, hash_cache=cache)
            candidates.append((spec, result["validation_report"], score))
    winners, models = [], {}
    for repetition in range(1, 6):
        spec, report, score = max((item for item in candidates if item[0]["repetition"] == repetition),
                                 key=lambda item: (item[2], -item[0]["learning_rate"]))
        winner = {
            "training_graph_u_mode": mode, "training_condition": "shift", "repetition": repetition,
            "model_seed": spec["model_seed"], "learning_rate": spec["learning_rate"],
            "checkpoint": str(output_root / "winners" / f"rep_{repetition}" / "shift" / "model.pt"),
            "checkpoint_sha256": report["checkpoint_sha256"], "source_checkpoint": report["checkpoint"],
            "validation_roc_auc": score, "validation_result": spec["result_path"],
            "query_location": profile["graph_u_query_location"], "query_scale": profile["graph_u_query_scale"],
            "generator_source_sha256": report["dump_metadata"]["generator_source_sha256"],
        }
        winners.append(winner)
        models[(mode, repetition, "shift")] = winner
    manifest = {"format": "nano_graph_u_fixed_shift_winners_v1", "fixed_shift_config": profile,
                "selection_metric": "roc_auc", "tie_break": "lowest_learning_rate", "trial_count": 50,
                "training_steps": 2500, "learning_rates": list(LEARNING_RATES), "winners": winners}
    path = output_root / "winners" / "winners.json"
    contents = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    _check_existing(path, contents)
    for winner in winners:
        _retain_checkpoint(Path(winner["source_checkpoint"]), Path(winner["checkpoint"]), winner["checkpoint_sha256"])
    _write_json(path, manifest)
    print(f"WINNERS_READY mode={mode} winners=5 manifest={path}", flush=True)
    return models


def _evaluate_models(directory, profile, models, iid_banks, specs, device):
    rows, cache = [], {}
    shifted_banks = {(spec["graph_u_mode"], spec["repetition"]): spec for spec in specs if spec["split"] == "test"}
    for (mode, repetition, training_condition), model in sorted(models.items()):
        if _digest(Path(model["checkpoint"]), cache) != model["checkpoint_sha256"]:
            raise ValueError("Selected model changed before evaluation")
        for test_mode in MODES:
            for condition in ("identity", "shift"):
                if training_condition == condition == "identity":
                    continue
                if condition == "identity":
                    bank = iid_banks[(test_mode, repetition)]
                    metadata, dump = bank["metadata"], bank["dump"]
                    if _digest(dump, cache) != bank["sha256"]:
                        raise ValueError("Original IID test bank changed")
                else:
                    spec = shifted_banks[(test_mode, repetition)]
                    metadata, dump = _bank_status(spec).metadata, spec["dump"]
                path = directory / "results" / f"{mode}_on_{test_mode}" / f"rep_{repetition}" / f"{training_condition}_on_{condition}.json"
                binding = {"format": "nano_graph_u_fixed_test_v1", "fixed_shift_config": profile,
                           "model": model, "test_graph_u_mode": test_mode, "test_condition": condition}
                if path.exists():
                    saved = json.loads(path.read_text(encoding="utf-8"))
                    if any(saved.get(key) != value for key, value in binding.items()):
                        raise ValueError(f"Existing test belongs to another experiment: {path}")
                    report = saved["evaluation_report"]
                else:
                    report = evaluate_checkpoint(model["checkpoint"], dump, batch_size=32, device=device)
                _validate_report(report, model, {"dump": dump}, metadata, cache)
                _write_json(path, {**binding, "evaluation_report": report})
                rows.append(_row(report, model, test_mode, condition, path, "new_evaluation"))
                print(f"TEST_READY {len(rows)}/60 train={mode}/{training_condition} test={test_mode}/{condition} rep={repetition}", flush=True)
    return rows


def _summarize(rows):
    keys = ("training_graph_u_mode", "training_condition", "test_graph_u_mode", "test_condition")
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    expected = {(mode, condition, test_mode, test_condition)
                for mode in MODES for condition in ("identity", "shift")
                for test_mode in MODES for test_condition in ("identity", "shift")}
    if set(groups) != expected or len(rows) != 80:
        raise ValueError("Each fixed experiment requires all eighty evaluations and sixteen comparisons")
    cells = []
    for key, group in sorted(groups.items()):
        if len(group) != 5 or {row["repetition"] for row in group} != set(range(1, 6)):
            raise ValueError("Each comparison requires all five repetitions exactly once")
        cell = {"fixed_location": group[0]["fixed_location"], "fixed_scale": group[0]["fixed_scale"],
                **dict(zip(keys, key)), "repetition_count": 5}
        for metric in METRIC_NAMES:
            cell[f"{metric}_mean"] = mean(row[metric] for row in group)
            cell[f"{metric}_std"] = stdev(row[metric] for row in group)
        cells.append(cell)
    return cells


def _save_results(directory, result):
    path = directory / "results"
    path.mkdir(parents=True, exist_ok=True)
    outputs = {"summary.csv": _csv_bytes(result["cells"], list(result["cells"][0])),
               "per_repetition.csv": _csv_bytes(result["per_repetition"], list(result["per_repetition"][0]))}
    for name, content in outputs.items():
        _check_existing(path / name, content)
    for name, content in outputs.items():
        _atomic_write(path / name, content)
    _write_json(path / "summary.json", result)


def _collect_if_complete(root):
    with _lock(root / ".summary.job.lock", blocking=True):
        paths = [experiment_spec(root, task)["directory"] / "results" / "summary.json" for task in range(1, 5)]
        if not all(path.is_file() for path in paths):
            return
        results = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        rows, cells, controls = [], [], {}
        for task, result in enumerate(results, 1):
            profile = experiment_spec(root, task)["fixed_shift_config"]
            if result["recipe"]["fixed_shift_config"] != profile:
                raise ValueError("Combined results contain a different fixed experiment")
            for row in result["per_repetition"]:
                if (row["fixed_location"], row["fixed_scale"]) != (
                    profile["graph_u_query_location"], profile["graph_u_query_scale"]
                ):
                    raise ValueError("Repetition labels do not match the fixed experiment recipe")
                bindings = []
                if row["training_condition"] == "identity":
                    bindings.append((("model", row["training_graph_u_mode"], row["repetition"]), row["checkpoint_sha256"]))
                if row["test_condition"] == "identity":
                    bindings.append((("bank", row["test_graph_u_mode"], row["repetition"]), row["test_dump_sha256"]))
                if row["training_condition"] == row["test_condition"] == "identity":
                    bindings.append((("scores", row["training_graph_u_mode"], row["test_graph_u_mode"], row["repetition"]),
                                     tuple(row[metric] for metric in METRIC_NAMES)))
                for key, value in bindings:
                    if controls.setdefault(key, value) != value:
                        raise ValueError("Original IID controls differ across fixed experiments")
            if _summarize(result["per_repetition"]) != result["cells"]:
                raise ValueError("Saved fixed summary does not match its repetition scores")
            rows.extend(result["per_repetition"])
            cells.extend(result["cells"])
        _save_results(root, {"format": "nano_graph_u_fixed_experiments_summary_v1",
                            "repetition_count": 5, "standard_deviation_ddof": 1,
                            "cells": cells, "per_repetition": rows})
        print(f"ALL_FIXED_EXPERIMENTS_COMPLETE comparisons=64 evaluations=320 summary={root / 'results/summary.csv'}", flush=True)


def run_experiment(cluster_root, root, task_id, *, device="cuda"):
    cluster, root = Path(cluster_root).resolve(), Path(root).resolve()
    _check_roots(cluster, root)
    spec = experiment_spec(root, task_id)
    directory, profile = spec["directory"], spec["fixed_shift_config"]
    with _lock(directory / ".experiment.job.lock"):
        _write_json(directory / "experiment.json", _recipe(cluster, spec))
        specs = bank_specs(directory, profile)
        fingerprints = {_bank_status(bank).metadata["generator_source_sha256"] for bank in specs}
        if len(fingerprints) != 1:
            raise ValueError("New fixed banks were generated by different prior code versions")
        models, iid_banks, iid_rows, provenance = load_original_iid(cluster)
        _write_json(directory / "source_iid.json", provenance)
        completed = 0
        for mode in MODES:
            for repetition in range(1, 6):
                for index in range(len(LEARNING_RATES)):
                    run_trial((repetition - 1) * 20 + 11 + index, directory / "data" / mode,
                              directory / "checkpoints" / mode, device=device,
                              graph_u_mode=mode, fixed_shift_config=profile)
                    completed += 1
                    print(f"TUNING_READY {completed}/100 mode={mode} rep={repetition}", flush=True)
        for mode in MODES:
            models.update(_select_shifted(directory, mode, profile))
        rows = iid_rows + _evaluate_models(directory, profile, models, iid_banks, specs, device)
        rows = [{"fixed_location": spec["location"], "fixed_scale": spec["scale"], **row} for row in rows]
        result = {"format": "nano_graph_u_fixed_experiment_summary_v1", "recipe": _recipe(cluster, spec),
                  "new_evaluation_count": 60, "reused_iid_count": 20, "repetition_count": 5,
                  "standard_deviation_ddof": 1, "cells": _summarize(rows), "per_repetition": rows}
        _save_results(directory, result)
    _collect_if_complete(root)
    print(f"FIXED_EXPERIMENT_COMPLETE task={task_id} evaluations=80 summary={directory / 'results/summary.csv'}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="stage", required=True)
    for stage in ("generate", "run"):
        command = subcommands.add_parser(stage)
        command.add_argument("--cluster-root", required=True, type=Path)
        command.add_argument("--root", required=True, type=Path)
        command.add_argument("--task-id", required=True, type=int)
        if stage == "generate":
            command.add_argument("--workers", type=int, default=8)
        else:
            command.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.stage == "generate":
        generate_experiment(args.cluster_root, args.root, args.task_id, workers=args.workers)
    else:
        run_experiment(args.cluster_root, args.root, args.task_id, device=args.device)


if __name__ == "__main__":
    main()
