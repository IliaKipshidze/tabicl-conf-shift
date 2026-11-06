"""Training scores reuse selected fixed-experiment models and their own banks."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from statistics import mean, stdev

import pytest
import torch

from tabicl.nano_graph_u import fixed_experiments as fixed
from tabicl.nano_graph_u import training_eval
from tabicl.nano_graph_u._settings import experiment_prior_config
from tabicl.nano_graph_u.data import _metadata
from tabicl.nano_graph_u.train import PAPER_MODEL_CONFIG, _sha256_file
from tabicl.nano_graph_u.tune import LEARNING_RATES

MODES = ("confounding", "covariate")
METRICS = (
    "roc_auc",
    "nll",
    "brier",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _sparse_bank(path, metadata):
    h5py = pytest.importorskip("h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    count = metadata["steps"] * metadata["batch_size"]
    with h5py.File(path, "w") as stream:
        stream.attrs["metadata_json"] = json.dumps(metadata)
        stream.attrs["committed_steps"] = metadata["steps"]
        # HDF5 reserves these shapes without allocating the 80,000 tables.
        stream.create_dataset("x", (count, 150, 5), dtype="f4")
        stream.create_dataset("y", (count, 150), dtype="i8")
        stream.create_dataset("train_sizes", (count,), dtype="i4")


@pytest.fixture
def completed_experiments(tmp_path):
    """Create selected checkpoints and a complete historical test summary."""
    root = tmp_path / "fixed"
    cluster = tmp_path / "cluster"
    models = {}
    for pair in (None, *range(1, 5)):
        condition = "identity" if pair is None else "shift"
        experiment = fixed.experiment_spec(root, pair) if pair is not None else None
        profile = experiment["fixed_shift_config"] if experiment else None
        for mode in MODES:
            if experiment:
                data = experiment["directory"] / "data" / mode
                search = experiment["directory"] / "checkpoints" / mode
            else:
                suffix = "" if mode == "confounding" else "_covariate"
                data = (
                    cluster
                    / "tabicl-conf-shift-priors/nano_graph_u"
                    / f"repetitions{suffix}"
                )
                search = (
                    cluster
                    / "tabicl-conf-shift-checkpoints/nano_graph_u"
                    / f"lr_search{suffix}"
                )
            for repetition in range(1, 6):
                metadata = _metadata(
                    experiment_prior_config(
                        condition, mode, fixed_shift_config=profile
                    ),
                    steps=2500,
                    batch_size=32,
                    seed=1000 + repetition,
                    rows=150,
                    features=5,
                    min_train_size=0.3,
                    max_train_size=0.9,
                    n_jobs=1,
                )
                metadata["generator_source_sha256"] = "a" * 64
                dump = (
                    data
                    / f"rep_{repetition}"
                    / (
                        f"{condition}_train_seed{1000 + repetition}_2500steps_b32_r150_f5.h5"
                    )
                )
                _sparse_bank(dump, metadata)
                checkpoint = (
                    search / "winners" / f"rep_{repetition}" / condition / "model.pt"
                )
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "format": "nano_graph_u_v1",
                    "step": 2500,
                    "model_config": copy.deepcopy(PAPER_MODEL_CONFIG),
                    "model": {"fixture": torch.tensor(float(repetition))},
                    "dump_metadata": metadata,
                    "prior_config": metadata["prior_config"],
                    "train_config": {
                        "seed": 41 + repetition,
                        "batch_size": 32,
                        "learning_rate": LEARNING_RATES[repetition - 1],
                        "optimizer": "schedulefree.AdamWScheduleFree",
                        "weight_decay": 0.0,
                        "gradient_clip_norm": 1.0,
                        "loss": "query_cross_entropy",
                        "precision": "float32",
                        "dump_path": str(dump),
                        "dump_sha256": _sha256_file(dump),
                    },
                }
                torch.save(payload, checkpoint)
                models[(pair, mode, repetition)] = {
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": _sha256_file(checkpoint),
                    "dump": str(dump),
                    "metadata": metadata,
                    "payload": payload,
                }

    rows, cells = [], []
    for pair in range(1, 5):
        experiment = fixed.experiment_spec(root, pair)
        pair_rows = []
        for mode in MODES:
            for condition in ("identity", "shift"):
                for test_mode in MODES:
                    for test_condition in ("identity", "shift"):
                        for repetition in range(1, 6):
                            model = models[
                                (
                                    None if condition == "identity" else pair,
                                    mode,
                                    repetition,
                                )
                            ]
                            location, scale = (
                                (0.0, 1.0)
                                if condition == "identity"
                                else fixed.FIXED_SHIFTS[pair - 1]
                            )
                            test_location, test_scale = (
                                (0.0, 1.0)
                                if test_condition == "identity"
                                else fixed.FIXED_SHIFTS[pair - 1]
                            )
                            pair_rows.append(
                                {
                                    "fixed_location": experiment["location"],
                                    "fixed_scale": experiment["scale"],
                                    "training_graph_u_mode": mode,
                                    "training_condition": condition,
                                    "test_graph_u_mode": test_mode,
                                    "test_condition": test_condition,
                                    "repetition": repetition,
                                    "model_seed": 41 + repetition,
                                    "learning_rate": LEARNING_RATES[repetition - 1],
                                    "training_query_location": location,
                                    "training_query_scale": scale,
                                    "test_query_location": test_location,
                                    "test_query_scale": test_scale,
                                    "test_data_seed": 2000 + repetition,
                                    "task_count": 3200,
                                    "query_row_count": 240000,
                                    **{
                                        metric: 0.5 + repetition / 100
                                        for metric in METRICS
                                    },
                                    "checkpoint": model["checkpoint"],
                                    "checkpoint_sha256": model["checkpoint_sha256"],
                                    "test_dump": str(
                                        cluster
                                        / "test"
                                        / test_mode
                                        / test_condition
                                        / f"rep_{repetition}.h5"
                                    ),
                                    "test_dump_sha256": "b" * 64,
                                    "training_generator_source_sha256": "a" * 64,
                                    "test_generator_source_sha256": "a" * 64,
                                    "result_path": str(
                                        experiment["directory"]
                                        / "results"
                                        / f"{mode}_on_{test_mode}"
                                        / f"rep_{repetition}"
                                        / f"{condition}_on_{test_condition}.json"
                                    ),
                                    "result_origin": "reused_iid"
                                    if condition == test_condition == "identity"
                                    else "new_evaluation",
                                }
                            )
        rows.extend(pair_rows)
        cells.extend(fixed._summarize(pair_rows))
    summary = {
        "format": "nano_graph_u_fixed_experiments_summary_v1",
        "repetition_count": 5,
        "standard_deviation_ddof": 1,
        "cells": cells,
        "per_repetition": rows,
    }
    _write_json(root / "results/summary.json", summary)
    # These are frozen, already-published test outputs.
    (root / "results/summary.csv").write_text(
        "existing,test,summary\n", encoding="utf-8"
    )
    (root / "results/per_repetition.csv").write_text(
        "existing,test,repetitions\n", encoding="utf-8"
    )
    return {"root": root, "cluster": cluster, "models": models, "summary": summary}


def _report(checkpoint, dump, *, score=None):
    h5py = pytest.importorskip("h5py")
    checkpoint, dump = Path(checkpoint), Path(dump)
    with h5py.File(dump, "r") as stream:
        metadata = json.loads(stream.attrs["metadata_json"])
    if score is None:
        score = 0.6 + (metadata["seed"] - 1000) / 100
    return {
        "format": "nano_graph_u_evaluation_v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "checkpoint_step": 2500,
        "dump": str(dump),
        "dump_sha256": _sha256_file(dump),
        "dump_metadata": metadata,
        "task_count": 80000,
        "query_row_count": 6000000,
        "metrics": {metric: score for metric in METRICS},
        "metric_task_counts": {metric: 80000 for metric in METRICS},
    }


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _input_hashes(fixture):
    paths = [
        path
        for base in (fixture["root"], fixture["cluster"])
        for path in base.rglob("*")
        if path.is_file()
    ]
    return {path: _sha256_file(path) for path in paths}


def _save_checkpoint(fixture, model, payload):
    """Rebind selection hashes so a test reaches semantic preflight checks."""
    checkpoint = Path(model["checkpoint"])
    torch.save(payload, checkpoint)
    model["checkpoint_sha256"] = _sha256_file(checkpoint)
    for row in fixture["summary"]["per_repetition"]:
        if row["checkpoint"] == str(checkpoint):
            row["checkpoint_sha256"] = model["checkpoint_sha256"]
    _write_json(fixture["root"] / "results/summary.json", fixture["summary"])


def test_fifty_task_mapping_covers_each_selected_model_once(completed_experiments):
    fixture = completed_experiments
    root = fixture["root"]
    expected = [
        fixture["models"][(pair, mode, repetition)]["checkpoint"]
        for pair in (None, *range(1, 5))
        for mode in MODES
        for repetition in range(1, 6)
    ]
    specs = [training_eval.task_spec(root, task) for task in range(1, 51)]
    assert [spec["checkpoint"] for spec in specs] == expected
    assert len(set(expected)) == 50
    assert [spec["result_path"] for spec in specs] == [
        str(root / "results/training_evaluations" / f"task_{task:02d}.json")
        for task in range(1, 51)
    ]


@pytest.mark.parametrize("task", [0, 51, True, "1", 1.0])
def test_task_mapping_rejects_invalid_ids_before_loading_sources(tmp_path, task):
    with pytest.raises(ValueError, match="1 to 50"):
        training_eval.task_spec(tmp_path, task)


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_row",
        "missing_cell",
        "extra_cell",
        "model_hash",
        "model_seed",
        "learning_rate",
    ],
)
def test_source_summary_must_be_complete_and_bind_one_model_per_selection(
    completed_experiments,
    corruption,
):
    fixture = completed_experiments
    summary = fixture["summary"]
    if corruption == "missing_row":
        summary["per_repetition"].pop()
    elif corruption == "missing_cell":
        summary["cells"].pop()
    elif corruption == "extra_cell":
        extra = dict(summary["cells"][0], fixed_location=999.0)
        summary["cells"].append(extra)
    else:
        row = summary["per_repetition"][0]
        field = {
            "model_hash": "checkpoint_sha256",
            "model_seed": "model_seed",
            "learning_rate": "learning_rate",
        }[corruption]
        row[field] = "f" * 64 if field == "checkpoint_sha256" else 0.123
    _write_json(fixture["root"] / "results/summary.json", summary)
    with pytest.raises(ValueError):
        training_eval.task_spec(fixture["root"], 1)


@pytest.mark.parametrize("task", [1, 6, 11, 16, 41, 50])
def test_evaluation_uses_own_recorded_bank_and_keeps_inputs_frozen(
    completed_experiments,
    monkeypatch,
    task,
):
    fixture = completed_experiments
    root = fixture["root"]
    spec = training_eval.task_spec(root, task)
    payload = torch.load(spec["checkpoint"], map_location="cpu", weights_only=True)
    dump = payload["train_config"]["dump_path"]
    before = _input_hashes(fixture)
    calls = []

    def evaluate(checkpoint, own_dump, **kwargs):
        assert str(checkpoint) == spec["checkpoint"]
        assert str(own_dump) == dump
        assert kwargs == {
            "batch_size": 32,
            "max_batches": None,
            "device": "cpu",
            "allow_training_dump": True,
        }
        calls.append((checkpoint, own_dump))
        return _report(checkpoint, own_dump)

    monkeypatch.setattr(training_eval, "evaluate_checkpoint", evaluate)
    initial = training_eval.run_evaluation(root, task, device="cpu")
    assert initial["task_count"] == 80000
    assert training_eval.run_evaluation(root, task, device="cpu") == initial
    assert len(calls) == 1
    assert all(_sha256_file(path) == digest for path, digest in before.items())
    assert Path(spec["result_path"]).is_file()
    assert not (root / "results/training_summary.csv").exists()
    assert not (root / "results/training_per_repetition.csv").exists()


@pytest.mark.parametrize(
    "corruption",
    [
        "wrong_seed",
        "wrong_condition",
        "wrong_track",
        "wrong_profile",
        "partial",
        "recorded_hash",
        "recorded_path",
        "recipe",
        "model_config",
        "checkpoint_step",
    ],
)
def test_own_bank_and_training_recipe_are_validated_before_inference(
    completed_experiments,
    monkeypatch,
    corruption,
):
    h5py = pytest.importorskip("h5py")
    fixture = completed_experiments
    root = fixture["root"]
    model = fixture["models"][(1, "confounding", 1)]
    payload = copy.deepcopy(model["payload"])
    metadata = copy.deepcopy(model["metadata"])
    if corruption in {
        "wrong_seed",
        "wrong_condition",
        "wrong_track",
        "wrong_profile",
        "partial",
    }:
        if corruption == "wrong_seed":
            metadata["seed"] = 1002
        elif corruption == "wrong_condition":
            metadata["prior_config"]["graph_u_query_location"] = 0.0
            metadata["prior_config"]["graph_u_query_scale"] = 1.0
        elif corruption == "wrong_track":
            metadata["prior_config"]["graph_u_mode"] = "covariate"
        elif corruption == "wrong_profile":
            metadata["prior_config"]["graph_u_query_location"] = 20.0
        with h5py.File(model["dump"], "r+") as stream:
            stream.attrs["metadata_json"] = json.dumps(metadata)
            if corruption == "partial":
                stream.attrs["committed_steps"] = 2499
        payload["dump_metadata"] = metadata
        payload["prior_config"] = metadata["prior_config"]
        payload["train_config"]["dump_sha256"] = _sha256_file(Path(model["dump"]))
    elif corruption == "recorded_hash":
        payload["train_config"]["dump_sha256"] = "f" * 64
    elif corruption == "recorded_path":
        payload["train_config"]["dump_path"] = fixture["models"][(1, "confounding", 2)][
            "dump"
        ]
    elif corruption == "recipe":
        payload["train_config"]["optimizer"] = "torch.optim.AdamW"
    elif corruption == "model_config":
        payload["model_config"]["num_outputs"] = 3
    elif corruption == "checkpoint_step":
        payload["step"] = 2499
    _save_checkpoint(fixture, model, payload)
    monkeypatch.setattr(
        training_eval,
        "evaluate_checkpoint",
        lambda *a, **kw: pytest.fail("Must not evaluate invalid inputs"),
    )
    with pytest.raises(ValueError):
        training_eval.run_evaluation(root, 11, device="cpu")
    assert not (root / "results/training_evaluations/task_11.json").exists()


@pytest.mark.parametrize(
    "corruption", ["task_count", *METRICS, "nan_metric", "missing_metric"]
)
def test_every_report_requires_all_eighty_thousand_tasks_and_six_metrics(
    completed_experiments,
    monkeypatch,
    corruption,
):
    fixture = completed_experiments
    root = fixture["root"]

    def evaluate(checkpoint, dump, **kwargs):
        report = _report(checkpoint, dump)
        if corruption == "task_count":
            report["task_count"] = 79999
        elif corruption == "nan_metric":
            report["metrics"]["nll"] = float("nan")
        elif corruption == "missing_metric":
            del report["metrics"]["macro_f1"]
        else:
            report["metric_task_counts"][corruption] = 79999
        return report

    monkeypatch.setattr(training_eval, "evaluate_checkpoint", evaluate)
    with pytest.raises(ValueError):
        training_eval.run_evaluation(root, 1, device="cpu")
    assert not (root / "results/training_evaluations/task_01.json").exists()


@pytest.mark.parametrize("corruption", ["report", "checkpoint", "bank"])
def test_cached_scores_revalidate_reports_and_frozen_inputs(
    completed_experiments,
    monkeypatch,
    corruption,
):
    h5py = pytest.importorskip("h5py")
    fixture = completed_experiments
    root = fixture["root"]
    calls = []

    def evaluate(checkpoint, dump, **kwargs):
        calls.append("evaluate")
        return _report(checkpoint, dump)

    monkeypatch.setattr(training_eval, "evaluate_checkpoint", evaluate)
    training_eval.run_evaluation(root, 1, device="cpu")
    result_path = root / "results/training_evaluations/task_01.json"
    model = fixture["models"][(None, "confounding", 1)]
    if corruption == "report":
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        saved["evaluation_report"]["metric_task_counts"]["accuracy"] = 79999
        _write_json(result_path, saved)
    elif corruption == "checkpoint":
        with Path(model["checkpoint"]).open("ab") as stream:
            stream.write(b"changed after scoring")
    else:
        with h5py.File(model["dump"], "r+") as stream:
            stream.attrs["committed_steps"] = 2499
    with pytest.raises(ValueError):
        training_eval.run_evaluation(root, 1, device="cpu")
    assert calls == ["evaluate"]


def test_bank_changed_during_inference_is_not_cached(
    completed_experiments, monkeypatch
):
    h5py = pytest.importorskip("h5py")
    root = completed_experiments["root"]

    def evaluate(checkpoint, dump, **kwargs):
        report = _report(checkpoint, dump)
        with h5py.File(dump, "r+") as stream:
            stream.attrs["changed_during_inference"] = True
        return report

    monkeypatch.setattr(training_eval, "evaluate_checkpoint", evaluate)
    with pytest.raises(ValueError, match="changed"):
        training_eval.run_evaluation(root, 1, device="cpu")
    assert not (root / "results/training_evaluations/task_01.json").exists()


def test_all_fifty_scores_produce_eighty_rows_and_sixteen_sample_sd_cells(
    completed_experiments,
    monkeypatch,
):
    fixture = completed_experiments
    root = fixture["root"]
    before = _input_hashes(fixture)
    calls = []

    def evaluate(checkpoint, dump, **kwargs):
        assert kwargs == {
            "batch_size": 32,
            "max_batches": None,
            "device": "cpu",
            "allow_training_dump": True,
        }
        calls.append((str(checkpoint), str(dump)))
        report = _report(checkpoint, dump)
        prior = report["dump_metadata"]["prior_config"]
        offset = (0.05 if prior["graph_u_mode"] == "covariate" else 0.0) + (
            prior["graph_u_query_location"] / 1000
        )
        report["metrics"] = {
            metric: value + offset for metric, value in report["metrics"].items()
        }
        return report

    monkeypatch.setattr(training_eval, "evaluate_checkpoint", evaluate)
    for task in range(1, 50):
        training_eval.run_evaluation(root, task, device="cpu")
    assert not (root / "results/training_summary.csv").exists()
    assert not (root / "results/training_per_repetition.csv").exists()
    training_eval.run_evaluation(root, 50, device="cpu")
    assert len(calls) == len(set(calls)) == 50
    rows = _read_csv(root / "results/training_per_repetition.csv")
    cells = _read_csv(root / "results/training_summary.csv")
    assert len(rows) == 80
    assert len(cells) == 16
    assert all(int(row["task_count"]) == 80000 for row in rows)
    assert {float(row["fixed_location"]) for row in rows} == {4.0, 20.0, 60.0, 100.0}
    for mode in MODES:
        for repetition in range(1, 6):
            group = [
                row
                for row in rows
                if row["training_condition"] == "identity"
                and row["training_graph_u_mode"] == mode
                and int(row["repetition"]) == repetition
            ]
            assert len(group) == 4
            assert len({row["checkpoint"] for row in group}) == 1
            assert len({row["result_path"] for row in group}) == 1
            assert len({row["roc_auc"] for row in group}) == 1
    for cell in cells:
        assert int(cell["repetition_count"]) == 5
        offset = (0.05 if cell["training_graph_u_mode"] == "covariate" else 0.0) + (
            float(cell["fixed_location"]) / 1000
            if cell["training_condition"] == "shift"
            else 0.0
        )
        scores = [0.6 + repetition / 100 + offset for repetition in range(1, 6)]
        for metric in METRICS:
            assert float(cell[f"{metric}_mean"]) == pytest.approx(mean(scores))
            assert float(cell[f"{metric}_std"]) == pytest.approx(stdev(scores))
    # Reenter the completion path from a cached task without repeating inference.
    csv_bytes = {
        name: (root / "results" / name).read_bytes()
        for name in (
            "training_summary.csv",
            "training_per_repetition.csv",
        )
    }
    training_eval.run_evaluation(root, 50, device="cpu")
    assert len(calls) == 50
    assert all(
        (root / "results" / name).read_bytes() == content
        for name, content in csv_bytes.items()
    )
    assert all(_sha256_file(path) == digest for path, digest in before.items())
