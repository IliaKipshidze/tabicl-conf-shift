"""The sampled experiment combines all four frozen-test directions consistently."""

from __future__ import annotations

import csv
import hashlib
import json
from statistics import mean, stdev

import pytest

from tabicl.nano_graph_u import sampled
from tabicl.prior.graph_lib._config import GRAPH_U_SAMPLING_DEFAULTS


@pytest.fixture
def stage_summaries(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    (root / "results").mkdir(parents=True)
    settings = {**GRAPH_U_SAMPLING_DEFAULTS, "graph_u_shift_mode": "sampled"}
    calls, outputs = [], {}
    for training_mode in sampled.MODES:
        for test_mode in sampled.MODES:
            rows, cells = [], []
            for repetition in range(1, 6):
                for training in ("identity", "shift"):
                    for testing in ("identity", "shift"):
                        rows.append({
                            "repetition": repetition,
                            "training_condition": training,
                            "test_condition": testing,
                            "accuracy": 0.7 + repetition * 0.01,
                            "roc_auc": 0.8 + repetition * 0.01,
                            "learning_rate": 0.001,
                            "model_seed": 41 + repetition,
                            "test_data_seed": 2000 + repetition,
                            "checkpoint_sha256": hashlib.sha256(
                                f"{training_mode}/{repetition}/{training}".encode()
                            ).hexdigest(),
                            "test_dump_sha256": hashlib.sha256(
                                f"{test_mode}/{repetition}/{testing}".encode()
                            ).hexdigest(),
                        })
            for training in ("identity", "shift"):
                for testing in ("identity", "shift"):
                    group = [row for row in rows if row["training_condition"] == training and row["test_condition"] == testing]
                    cells.append({
                        "training_condition": training,
                        "test_condition": testing,
                        "repetition_count": 5,
                        "metrics": {
                            metric: {"mean": mean(row[metric] for row in group), "std": stdev(row[metric] for row in group)}
                            for metric in ("accuracy", "roc_auc")
                        },
                    })
            if training_mode != test_mode:
                modes = {"training_graph_u_mode": training_mode, "test_graph_u_mode": test_mode}
                rows = [{**modes, **row} for row in rows]
                cells = [{**modes, **cell} for cell in cells]
            outputs[(training_mode, test_mode)] = {"per_repetition": rows, "cells": cells}

    def collect(data_root, search_root, result_root, **kwargs):
        training, testing = kwargs["graph_u_mode"], kwargs["test_graph_u_mode"]
        assert data_root == root / "data" / testing
        assert search_root == root / "checkpoints" / training
        assert result_root == root / "results" / f"{training}_on_{testing}"
        assert kwargs["training_data_root"] == root / "data" / training
        assert kwargs["sampling_config"] == settings
        calls.append((training, testing))
        return outputs[(training, testing)]

    monkeypatch.setattr(sampled, "summarize_tests", collect)
    return root, settings, calls, outputs


def test_combined_summary_contains_all_80_reports_and_16_cells(stage_summaries):
    root, settings, calls, _ = stage_summaries
    summary = sampled.summarize_experiment(root, settings)
    assert calls == [(a, b) for a in sampled.MODES for b in sampled.MODES]
    assert summary["evaluation_count"] == len(summary["per_repetition"]) == 80
    assert len(summary["cells"]) == 16
    assert summary["sampling_config"] == settings
    assert summary["standard_deviation_ddof"] == 1
    assert len({(row["training_graph_u_mode"], row["training_condition"], row["test_graph_u_mode"], row["test_condition"], row["repetition"]) for row in summary["per_repetition"]}) == 80
    assert all(cell["repetition_count"] == 5 for cell in summary["cells"])
    for name, count in (("per_repetition.csv", 80), ("summary.csv", 16)):
        with (root / "results" / name).open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == count
        assert {(row["training_graph_u_mode"], row["test_graph_u_mode"]) for row in rows} == set(calls)
    assert json.loads((root / "results" / "summary.json").read_text()) == summary


def test_combined_summary_is_idempotent_and_refuses_changed_outputs(stage_summaries):
    root, settings, _, outputs = stage_summaries
    first = sampled.summarize_experiment(root, settings)
    files = [root / "results" / name for name in ("summary.json", "summary.csv", "per_repetition.csv")]
    before = {path: path.read_bytes() for path in files}
    assert sampled.summarize_experiment(root, settings) == first
    assert {path: path.read_bytes() for path in files} == before
    outputs[("confounding", "confounding")]["cells"][0]["metrics"]["roc_auc"]["mean"] += 0.01
    with pytest.raises(FileExistsError, match="different selection"):
        sampled.summarize_experiment(root, settings)
    assert {path: path.read_bytes() for path in files} == before


@pytest.mark.parametrize("field,message", [
    ("checkpoint_sha256", "selected model changed"),
    ("test_dump_sha256", "different test banks"),
])
def test_combined_summary_rejects_cross_direction_model_or_bank_changes(stage_summaries, field, message):
    root, settings, _, outputs = stage_summaries
    outputs[("confounding", "covariate")]["per_repetition"][0][field] = "f" * 64
    with pytest.raises(ValueError, match=message):
        sampled.summarize_experiment(root, settings)
    assert not (root / "results" / "summary.json").exists()
    assert not (root / "results" / "summary.csv").exists()
