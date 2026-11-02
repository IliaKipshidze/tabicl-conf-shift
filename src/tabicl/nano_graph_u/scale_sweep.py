"""Generate new shift-scale test banks and score frozen, selected Nano models."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from ._settings import experiment_prior_config, normalize_sampling_config
from .data import NanoPriorDumpLoader, generate_dump
from .evaluate import evaluate_checkpoint
from .sampled import MODES, _lock, load_experiment, test_for_task
from .select import _atomic_write, _check_existing
from .summarize_tests import _csv_bytes
from .test_winners import (
    METRIC_NAMES,
    _digest,
    _validate_test_metadata,
    validate_test_result,
)

SCALE_MEANS = (1.3, 1.7, 1.9, 2.2)
_SEED_BASE = 4000


def _tag(value: float) -> str:
    return format(value, ".12g").replace(".", "p")


def bank_specs(root: Path, settings: dict[str, Any], scale_means=SCALE_MEANS, half_width=0.2):
    """One bank per mean, graph mode and repetition, reused by four models."""
    values = [float(value) for value in scale_means]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Scale means must be a nonempty list of finite values")
    if len({_tag(value) for value in values}) != len(values):
        raise ValueError("Scale means must be distinct")
    if not math.isfinite(half_width) or half_width <= 0:
        raise ValueError("Scale half-width must be finite and positive")
    specs = []
    for value in values:
        profile = normalize_sampling_config({
            **settings, "graph_u_query_scale_mean": value,
            "graph_u_query_scale_min": round(value - half_width, 12),
            "graph_u_query_scale_max": round(value + half_width, 12),
        })
        for mode in MODES:
            for repetition in range(1, 6):
                seed = _SEED_BASE + repetition
                specs.append({
                    "scale_mean": value, "sampling_config": profile,
                    "graph_u_mode": mode, "repetition": repetition, "seed": seed, "steps": 100,
                    "dump": root / "data" / f"scale_{_tag(value)}" / mode / f"rep_{repetition}"
                    / f"shift_test_seed{seed}_100steps_b32_r150_f5.h5",
                })
    return specs


def _validate_bank(spec, *, complete=True):
    loader = NanoPriorDumpLoader(spec["dump"], batch_size=32)
    if loader.metadata["seed"] != spec["seed"]:
        raise ValueError("Sweep bank uses the wrong independent test seed")
    # Check the original test protocol, allowing only the explicitly new seed
    # and test sampling profile. The saved metadata itself is never modified.
    _validate_test_metadata(
        dict(loader.metadata, seed=2000 + spec["repetition"]), "shift", spec["repetition"],
        graph_u_mode=spec["graph_u_mode"], sampling_config=spec["sampling_config"],
    )
    if not 0 <= loader.committed_steps <= 100 or (complete and loader.committed_steps != 100):
        raise ValueError(f"Incomplete sweep bank: {loader.committed_steps}/100 batches")
    return loader


def _generate_banks(specs, workers):
    for index, spec in enumerate(specs, 1):
        exists = spec["dump"].exists()
        if not exists or _validate_bank(spec, complete=False).committed_steps < 100:
            generate_dump(
                spec["dump"], steps=100, batch_size=32, seed=spec["seed"],
                prior_config=experiment_prior_config("shift", spec["graph_u_mode"], spec["sampling_config"]),
                rows=150, features=5, min_train_size=0.3, max_train_size=0.9,
                n_jobs=1, workers=workers, resume=exists,
            )
        _validate_bank(spec)
        parameter = spec.get("sweep_parameter", "scale")
        print(f"BANK_READY {index}/{len(specs)} mean={spec[f'{parameter}_mean']} "
              f"mode={spec['graph_u_mode']} rep={spec['repetition']}", flush=True)


def _row(report, model, test_mode, profile, result_path, origin, *, parameter="scale"):
    row = {
        "training_graph_u_mode": model["training_graph_u_mode"],
        "training_condition": model["training_condition"], "test_graph_u_mode": test_mode,
        "test_condition": "identity" if profile is None else "shift",
        **{f"scale_{field}": None if profile is None else profile[f"graph_u_query_scale_{field}"]
           for field in ("mean", "std", "min", "max")},
        "repetition": model["repetition"], "model_seed": model["model_seed"],
        "learning_rate": model["learning_rate"], "test_data_seed": report["dump_metadata"]["seed"],
        "task_count": report["task_count"], "query_row_count": report["query_row_count"],
        **{metric: report["metrics"][metric] for metric in METRIC_NAMES},
        "checkpoint": report["checkpoint"], "checkpoint_sha256": report["checkpoint_sha256"],
        "test_dump": report["dump"], "test_dump_sha256": report["dump_sha256"],
        "result_path": str(result_path), "result_origin": origin,
    }
    if parameter == "location":
        row.update({
            f"location_{field}": None if profile is None else profile[f"graph_u_query_location_{field}"]
            for field in ("mean", "std", "min", "max")
        })
    return row


def load_source(source):
    """Read the selected winners and existing IID reports, without selection or inference."""
    settings, models, iid_rows, manifests, cache = load_experiment(source), {}, [], {}, {}
    for task in range(1, 81):
        spec = test_for_task(source, task, settings)
        if spec["test_condition"] != "identity":
            continue
        path = Path(spec["result_path"])
        result = json.loads(path.read_text(encoding="utf-8"))
        validate_test_result(result, spec, hash_cache=cache)
        report, winner = result["evaluation_report"], result["winner"]
        training_mode = spec.get("graph_u_mode", "confounding")
        test_mode = spec.get("test_graph_u_mode", training_mode)
        model = {
            "training_graph_u_mode": training_mode, "training_condition": spec["training_condition"],
            "repetition": spec["repetition"], "model_seed": spec["model_seed"],
            "learning_rate": winner["learning_rate"], "checkpoint": report["checkpoint"],
            "checkpoint_sha256": winner["checkpoint_sha256"],
        }
        key = (training_mode, spec["repetition"], spec["training_condition"])
        if models.setdefault(key, model) != model:
            raise ValueError("Source IID reports refer to different selected checkpoints")
        if manifests.setdefault(training_mode, result["manifest_sha256"]) != result["manifest_sha256"]:
            raise ValueError("Source IID reports refer to different winner selections")
        iid_rows.append(_row(report, model, test_mode, None, path, "reused_iid"))
    if len(models) != 20 or len(iid_rows) != 40:
        raise ValueError("Expected 20 selected models and 40 existing IID evaluations")
    return settings, models, iid_rows, manifests


def _validate_report(report, model, spec, metadata, cache):
    if (
        report.get("format") != "nano_graph_u_evaluation_v1"
        or report.get("checkpoint") != model["checkpoint"]
        or report.get("checkpoint_sha256") != model["checkpoint_sha256"]
        or report.get("checkpoint_step") != 2500
        or report.get("dump") != str(spec["dump"])
        or report.get("dump_sha256") != _digest(spec["dump"], cache)
        or report.get("dump_metadata") != metadata
        or report.get("task_count") != 3200
        or not isinstance(report.get("query_row_count"), int)
        or not 0 < report["query_row_count"] < 3200 * 150
    ):
        raise ValueError("Sweep report does not match this frozen model and complete test bank")
    for metric in METRIC_NAMES:
        value = report.get("metrics", {}).get(metric)
        if (
            report.get("metric_task_counts", {}).get(metric) != 3200
            or isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (metric != "nll" and value > 1)
        ):
            raise ValueError(f"Incomplete or invalid sweep metric: {metric}")


def _evaluate_banks(root, specs, models, device):
    rows, cache = [], {}
    for spec in specs:
        metadata = _validate_bank(spec).metadata
        parameter = spec.get("sweep_parameter", "scale")
        value = spec[f"{parameter}_mean"]
        for mode in MODES:
            for condition in ("identity", "shift"):
                model = models[(mode, spec["repetition"], condition)]
                path = root / "results" / f"{parameter}_{_tag(value)}" \
                    / f"{mode}_on_{spec['graph_u_mode']}" / f"rep_{spec['repetition']}" / f"{condition}.json"
                binding = {"format": f"nano_graph_u_{parameter}_sweep_evaluation_v1", "model": model,
                           f"{parameter}_mean": value, "test_sampling_config": spec["sampling_config"]}
                if _digest(Path(model["checkpoint"]), cache) != model["checkpoint_sha256"]:
                    raise ValueError("Selected checkpoint changed during the sweep")
                if path.exists():
                    saved = json.loads(path.read_text(encoding="utf-8"))
                    if any(saved.get(key) != value for key, value in binding.items()):
                        raise ValueError(f"Existing report belongs to a different sweep: {path}")
                    report = saved["evaluation_report"]
                else:
                    report = evaluate_checkpoint(model["checkpoint"], spec["dump"], batch_size=32, device=device)
                _validate_report(report, model, spec, metadata, cache)
                path.parent.mkdir(parents=True, exist_ok=True)
                _atomic_write(path, (json.dumps({**binding, "evaluation_report": report},
                              indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
                rows.append(_row(report, model, spec["graph_u_mode"], spec["sampling_config"], path,
                                 "new_sweep", parameter=parameter))
                print(f"EVALUATION_READY {len(rows)}/{4 * len(specs)} "
                      f"mean={value} train={mode}/{condition} test={spec['graph_u_mode']} "
                      f"rep={spec['repetition']}", flush=True)
    return rows


def _summarize(rows, *, parameter="scale"):
    if parameter not in {"scale", "location"}:
        raise ValueError("Sweep parameter must be scale or location")
    keys = ("training_graph_u_mode", "training_condition", "test_graph_u_mode", "test_condition",
            f"{parameter}_mean")
    parameters = ("scale",) if parameter == "scale" else ("scale", "location")
    profile_fields = [f"{name}_{field}" for name in parameters for field in ("mean", "std", "min", "max")
                      if f"{name}_{field}" not in keys]
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    cells = []
    for key, group in groups.items():
        if len(group) != 5 or {row["repetition"] for row in group} != set(range(1, 6)):
            raise ValueError("Each sweep summary cell requires all five repetitions exactly once")
        if any(row[field] != group[0][field] for row in group for field in profile_fields):
            raise ValueError("Sweep summary cell contains different sampling profiles")
        cell = {**dict(zip(keys, key)), "repetition_count": 5,
                **{field: group[0][field] for field in profile_fields}}
        for metric in METRIC_NAMES:
            cell[f"{metric}_mean"] = mean(row[metric] for row in group)
            cell[f"{metric}_std"] = stdev(row[metric] for row in group)
        cells.append(cell)
    return cells


def run_sweep(source_root, root, *, scale_means=SCALE_MEANS, scale_half_width=0.2, workers=8, device="cuda"):
    source, root = Path(source_root).resolve(), Path(root).resolve()
    if source == root or source in root.parents or root in source.parents:
        raise ValueError("Sweep output must be in a separate directory from the source experiment")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    settings, models, iid_rows, manifests = load_source(source)
    specs = bank_specs(root, settings, scale_means, scale_half_width)
    recipe = {
        "format": "nano_graph_u_scale_sweep_v1", "source_root": str(source),
        "source_sampling_config": settings, "source_winner_manifests_sha256": manifests,
        "source_checkpoints_sha256": {"/".join(map(str, key)): value["checkpoint_sha256"]
                                      for key, value in models.items()},
        "scale_means": [float(value) for value in scale_means], "scale_half_width": scale_half_width,
        "test_seed_base": _SEED_BASE, "steps": 100, "batch_size": 32, "rows": 150, "features": 5,
    }
    with _lock(root / ".sweep.job.lock"):
        recipe_path = root / "experiment.json"
        _atomic_write(recipe_path, (json.dumps(recipe, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
        print(f"SWEEP_START banks={len(specs)} new_evaluations={4 * len(specs)} reused_iid=40", flush=True)
        _generate_banks(specs, workers)
        print(f"GENERATION_COMPLETE banks={len(specs)}/{len(specs)}", flush=True)
        new_rows = _evaluate_banks(root, specs, models, device)
        rows = iid_rows + new_rows
        if len(new_rows) != 4 * len(specs) or len(rows) != 40 + 4 * len(specs):
            raise ValueError("Sweep evaluation count is incomplete")
        cells = _summarize(rows)
        result = {"format": "nano_graph_u_scale_sweep_summary_v1", "recipe": recipe,
                  "new_evaluation_count": len(new_rows), "reused_iid_count": len(iid_rows),
                  "repetition_count": 5, "standard_deviation_ddof": 1,
                  "cells": cells, "per_repetition": rows}
        outputs = {
            "summary.csv": _csv_bytes(cells, list(cells[0])),
            "per_repetition.csv": _csv_bytes(rows, list(rows[0])),
            "summary.json": (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
        }
        for name, contents in outputs.items():
            _check_existing(root / "results" / name, contents)
        for name, contents in outputs.items():
            _atomic_write(root / "results" / name, contents)
        print(f"SWEEP_COMPLETE banks={len(specs)} new_evaluations={len(new_rows)} "
              f"reused_iid={len(iid_rows)} cells={len(cells)} summary={root / 'results' / 'summary.csv'}", flush=True)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--scale-means", type=float, nargs="+", default=SCALE_MEANS)
    parser.add_argument("--scale-half-width", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    run_sweep(args.source_root, args.root, scale_means=args.scale_means,
              scale_half_width=args.scale_half_width, workers=args.workers, device=args.device)


if __name__ == "__main__":
    main()
