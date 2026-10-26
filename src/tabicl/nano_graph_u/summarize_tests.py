"""Summarize the twenty held-out winner evaluations across five repetitions."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .select import _atomic_write, _check_existing
from .test_winners import METRIC_NAMES as METRICS
from .test_winners import test_spec, validate_test_result

SUMMARY_FORMAT = "nano_graph_u_winner_test_summary_v1"


def _csv_bytes(rows: list[dict[str, Any]], columns: list[str]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def summarize_tests(
    data_root: str | Path,
    search_root: str | Path,
    result_root: str | Path,
    *,
    graph_u_mode: str = "confounding",
) -> dict[str, Any]:
    """Require the complete matched design and report mean/sample std per cell.

    Every input metric is already a mean over one bank's 3,200 tasks. The
    standard deviation measures variation across the five repetitions, not
    variation across individual tasks or uncertainty of the estimated mean.
    """
    data_root = Path(data_root).expanduser().resolve()
    search_root = Path(search_root).expanduser().resolve()
    result_root = Path(result_root).expanduser().resolve()
    specs = [
        test_spec(task, data_root, search_root, result_root, graph_u_mode=graph_u_mode)
        for task in range(1, 21)
    ]
    missing = [
        spec["result_path"] for spec in specs if not Path(spec["result_path"]).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Cannot summarize: {len(missing)} of 20 test results are missing; "
            f"first missing result: {missing[0]}"
        )

    rows = []
    hash_cache: dict = {}
    checkpoint_hashes: dict[tuple[int, str], str] = {}
    test_hashes: dict[tuple[int, str], str] = {}
    manifest_hash = None
    for spec in specs:
        result = json.loads(Path(spec["result_path"]).read_text(encoding="utf-8"))
        validate_test_result(result, spec, verify_files=True, hash_cache=hash_cache)
        report = result["evaluation_report"]
        current_manifest_hash = result["manifest_sha256"]
        if manifest_hash is None:
            manifest_hash = current_manifest_hash
        elif current_manifest_hash != manifest_hash:
            raise ValueError("Test results refer to different winner selections")
        repetition = spec["repetition"]
        training_condition = spec["training_condition"]
        test_condition = spec["test_condition"]
        model_key = (repetition, training_condition)
        bank_key = (repetition, test_condition)
        if (
            checkpoint_hashes.setdefault(model_key, report["checkpoint_sha256"])
            != report["checkpoint_sha256"]
        ):
            raise ValueError("A winner changed between its two test evaluations")
        if (
            test_hashes.setdefault(bank_key, report["dump_sha256"])
            != report["dump_sha256"]
        ):
            raise ValueError("The two models were evaluated on different test banks")
        winner = result["winner"]
        rows.append(
            {
                "repetition": repetition,
                "training_condition": training_condition,
                "test_condition": test_condition,
                "learning_rate": winner["learning_rate"],
                "model_seed": winner["model_seed"],
                "test_data_seed": report["dump_metadata"]["seed"],
                "task_count": report["task_count"],
                "query_row_count": report["query_row_count"],
                **{metric: report["metrics"][metric] for metric in METRICS},
                "checkpoint": report["checkpoint"],
                "checkpoint_sha256": report["checkpoint_sha256"],
                "test_dump": report["dump"],
                "test_dump_sha256": report["dump_sha256"],
                "result_path": spec["result_path"],
            }
        )

    cells, csv_rows = [], []
    for training_condition in ("identity", "shift"):
        for test_condition in ("identity", "shift"):
            group = [
                row
                for row in rows
                if row["training_condition"] == training_condition
                and row["test_condition"] == test_condition
            ]
            if len(group) != 5 or {row["repetition"] for row in group} != set(
                range(1, 6)
            ):
                raise ValueError("Each summary cell must contain all five repetitions")
            metrics = {
                metric: {
                    "mean": mean(row[metric] for row in group),
                    "std": stdev(row[metric] for row in group),
                }
                for metric in METRICS
            }
            cell = {
                "training_condition": training_condition,
                "test_condition": test_condition,
                "repetition_count": 5,
                "metrics": metrics,
            }
            cells.append(cell)
            csv_rows.append(
                {
                    key: cell[key]
                    for key in (
                        "training_condition",
                        "test_condition",
                        "repetition_count",
                    )
                }
                | {
                    f"{metric}_{stat}": metrics[metric][stat]
                    for metric in METRICS
                    for stat in ("mean", "std")
                }
            )

    summary = {
        "format": SUMMARY_FORMAT,
        "data_root": str(data_root),
        "search_root": str(search_root),
        "result_root": str(result_root),
        "winners_manifest": str(search_root / "winners" / "winners.json"),
        "winners_manifest_sha256": manifest_hash,
        "evaluation_count": 20,
        "repetition_count": 5,
        "standard_deviation_ddof": 1,
        "aggregation": "equal-weight mean and sample std of five repetition-level task means",
        "cells": cells,
        "per_repetition": rows,
    }
    if graph_u_mode != "confounding":
        summary["graph_u_mode"] = graph_u_mode
    outputs = {
        result_root / "summary.json": (
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8"),
        result_root / "summary.csv": _csv_bytes(
            csv_rows,
            ["training_condition", "test_condition", "repetition_count"]
            + [f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std")],
        ),
        result_root / "per_repetition.csv": _csv_bytes(rows, list(rows[0])),
    }
    # Refuse a changed summary before replacing any existing output.
    for path, contents in outputs.items():
        _check_existing(path, contents)
    for path, contents in outputs.items():
        _atomic_write(path, contents)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--search-root", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument(
        "--graph-u-mode", choices=("confounding", "covariate"), default="confounding"
    )
    args = parser.parse_args(argv)
    summary = summarize_tests(
        args.data_root,
        args.search_root,
        args.result_root,
        graph_u_mode=args.graph_u_mode,
    )
    for cell in summary["cells"]:
        scores = " ".join(
            f"{metric}={cell['metrics'][metric]['mean']:.6f}+/-{cell['metrics'][metric]['std']:.6f}"
            for metric in METRICS
        )
        print(
            f"training={cell['training_condition']} test={cell['test_condition']} "
            f"n=5 {scores}",
            flush=True,
        )
    print(
        f"SUMMARY_READY path={Path(args.result_root).expanduser().resolve() / 'summary.csv'}"
    )


if __name__ == "__main__":
    main()
