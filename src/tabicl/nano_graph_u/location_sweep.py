"""Generate new shift-location test banks and score frozen, selected Nano models."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from ._settings import normalize_sampling_config
from .sampled import MODES, _lock
from .scale_sweep import _evaluate_banks, _generate_banks, _summarize, _tag, load_source
from .select import _atomic_write, _check_existing
from .summarize_tests import _csv_bytes

LOCATION_PROFILES = (
    (0.3, 0.1, 0.5),
    (1.3, 1.1, 1.5),
    (1.7, 1.5, 1.9),
    (1.9, 1.7, 2.1),
    (2.2, 2.0, 2.4),
    (4.5, 4.3, 4.7),
    (10.0, 9.8, 10.2),
    (100.0, 85.0, 115.0),
)
_SEED_BASE = 5000


def _normalize_profiles(location_profiles):
    """Validate explicit means and bounds before planning any output."""
    profiles = []
    try:
        for profile in location_profiles:
            if isinstance(profile, (str, bytes)):
                raise TypeError("Each location profile must contain exactly three values")
            values = tuple(float(value) for value in profile)
            if len(values) != 3:
                raise ValueError("Each location profile must contain exactly three values")
            mean, lower, upper = values
            if not all(math.isfinite(value) for value in values):
                raise ValueError("Location profile values must be finite")
            if not lower < upper or not lower <= mean <= upper:
                raise ValueError("Location profiles require lower < upper and lower <= mean <= upper")
            profiles.append(values)
    except (TypeError, OverflowError) as exc:
        raise ValueError("Location profiles must contain triples of finite numbers") from exc
    if not profiles:
        raise ValueError("Location profiles must be nonempty")
    if len({_tag(profile[0]) for profile in profiles}) != len(profiles):
        raise ValueError("Location means must be distinct")
    return tuple(profiles)


def bank_specs(root: Path, settings: dict[str, Any], location_profiles=LOCATION_PROFILES):
    """One bank per location profile, graph mode and repetition, reused by four models."""
    profiles = _normalize_profiles(location_profiles)
    settings = normalize_sampling_config(settings)
    root = Path(root)
    specs = []
    for mean, lower, upper in profiles:
        sampling_config = normalize_sampling_config({
            **settings,
            "graph_u_query_location_mean": mean,
            "graph_u_query_location_min": lower,
            "graph_u_query_location_max": upper,
        })
        for mode in MODES:
            for repetition in range(1, 6):
                seed = _SEED_BASE + repetition
                specs.append({
                    "sweep_parameter": "location", "location_mean": mean,
                    "sampling_config": sampling_config, "graph_u_mode": mode,
                    "repetition": repetition, "seed": seed, "steps": 100,
                    "dump": root / "data" / f"location_{_tag(mean)}" / mode / f"rep_{repetition}"
                    / f"shift_test_seed{seed}_100steps_b32_r150_f5.h5",
                })
    return specs


def run_sweep(source_root, root, *, location_profiles=LOCATION_PROFILES, workers=8, device="cuda"):
    source, root = Path(source_root).resolve(), Path(root).resolve()
    if source == root or source in root.parents or root in source.parents:
        raise ValueError("Sweep output must be in a separate directory from the source experiment")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    profiles = _normalize_profiles(location_profiles)
    settings, models, iid_rows, manifests = load_source(source)
    settings = normalize_sampling_config(settings)
    specs = bank_specs(root, settings, profiles)
    recipe = {
        "format": "nano_graph_u_location_sweep_v1", "source_root": str(source),
        "source_sampling_config": settings, "source_winner_manifests_sha256": manifests,
        "source_checkpoints_sha256": {"/".join(map(str, key)): value["checkpoint_sha256"]
                                      for key, value in models.items()},
        "location_profiles": [list(profile) for profile in profiles],
        "test_seed_base": _SEED_BASE, "steps": 100, "batch_size": 32, "rows": 150, "features": 5,
    }
    recipe_bytes = (json.dumps(recipe, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with _lock(root / ".sweep.job.lock"):
        recipe_path = root / "experiment.json"
        _check_existing(recipe_path, recipe_bytes)
        _atomic_write(recipe_path, recipe_bytes)
        print("SOURCE_SAMPLING " + " ".join(
            f"{field}={settings[f'graph_u_query_{field}']}"
            for field in ("scale_mean", "scale_std", "scale_min", "scale_max", "location_std")
        ), flush=True)
        print(f"SWEEP_START banks={len(specs)} new_evaluations={4 * len(specs)} reused_iid=40", flush=True)
        _generate_banks(specs, workers)
        print(f"GENERATION_COMPLETE banks={len(specs)}/{len(specs)}", flush=True)
        new_rows = _evaluate_banks(root, specs, models, device)
        iid_rows = [{**row, **{f"location_{field}": None for field in ("mean", "std", "min", "max")}}
                    for row in iid_rows]
        rows = iid_rows + new_rows
        if len(iid_rows) != 40 or len(new_rows) != 4 * len(specs) or len(rows) != 40 + 4 * len(specs):
            raise ValueError("Sweep evaluation count is incomplete")
        cells = _summarize(rows, parameter="location")
        if len(cells) != 8 * (len(profiles) + 1):
            raise ValueError("Sweep summary cell count is incomplete")
        result = {"format": "nano_graph_u_location_sweep_summary_v1", "recipe": recipe,
                  "new_evaluation_count": len(new_rows), "reused_iid_count": len(iid_rows),
                  "repetition_count": 5, "standard_deviation_ddof": 1,
                  "cells": cells, "per_repetition": rows}
        outputs = {
            "summary.csv": _csv_bytes(cells, list(cells[0])),
            "per_repetition.csv": _csv_bytes(rows, list(rows[0])),
            "summary.json": (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
        }
        result_dir = root / "results"
        for name, contents in outputs.items():
            _check_existing(result_dir / name, contents)
        result_dir.mkdir(parents=True, exist_ok=True)
        for name, contents in outputs.items():
            _atomic_write(result_dir / name, contents)
        print(f"SWEEP_COMPLETE banks={len(specs)} new_evaluations={len(new_rows)} "
              f"reused_iid={len(iid_rows)} cells={len(cells)} summary={result_dir / 'summary.csv'}", flush=True)
        return result


def _parse_profile(value):
    try:
        return _normalize_profiles([value.split(":")])[0]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected a mean:lower:upper location profile: {exc}") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--location-profiles", type=_parse_profile, nargs="+", default=LOCATION_PROFILES,
                        metavar="MEAN:LOWER:UPPER")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    run_sweep(args.source_root, args.root, location_profiles=args.location_profiles,
              workers=args.workers, device=args.device)


if __name__ == "__main__":
    main()
