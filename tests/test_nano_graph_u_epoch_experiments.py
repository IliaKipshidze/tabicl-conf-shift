"""Epoch budgets use the saved repetition-one banks and matching validation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from tabicl.nano_graph_u import epoch_experiments as epochs
from tabicl.nano_graph_u._settings import experiment_prior_config
from tabicl.nano_graph_u.data import _metadata
from tabicl.nano_graph_u.fixed_experiments import experiment_spec
from tabicl.nano_graph_u.train import PAPER_MODEL_CONFIG, _sha256_file

MODES = ("confounding", "covariate")


def _bank_file(path, condition, split, mode, profile=None):
    h5py = pytest.importorskip("h5py")
    path = Path(path)
    steps = 2500 if split == "train" else 100
    metadata = _metadata(
        experiment_prior_config(condition, mode, fixed_shift_config=profile),
        steps=steps, batch_size=32, seed={"train": 1001, "test": 2001, "validation": 3001}[split],
        rows=150, features=5, min_train_size=0.3, max_train_size=0.9, n_jobs=1,
    )
    metadata["generator_source_sha256"] = "a" * 64
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as stream:
        stream.attrs["metadata_json"] = json.dumps(metadata)
        stream.attrs["committed_steps"] = steps
        for key, shape, dtype in (
            ("x", (steps * 32, 150, 5), "f4"),
            ("y", (steps * 32, 150), "i8"),
            ("train_sizes", (steps * 32,), "i4"),
        ):
            stream.create_dataset(key, shape, dtype=dtype)


@pytest.fixture
def banks(tmp_path):
    cluster, fixed, root = [tmp_path / name for name in ("cluster", "fixed", "epoch outputs")]
    for mode in MODES:
        suffix = "" if mode == "confounding" else "_covariate"
        sources = [(cluster / "tabicl-conf-shift-priors/nano_graph_u" / f"repetitions{suffix}",
                    "identity", None)]
        for pair_id in range(1, 5):
            spec = experiment_spec(fixed, pair_id)
            sources.append((spec["directory"] / "data" / mode, "shift", spec["fixed_shift_config"]))
        for source, condition, profile in sources:
            for split, steps, seed in (("train", 2500, 1001), ("test", 100, 2001),
                                       ("validation", 100, 3001)):
                directory = source / ("validation" if split == "validation" else "rep_1")
                path = directory / f"{condition}_{split}_seed{seed}_{steps}steps_b32_r150_f5.h5"
                _bank_file(path, condition, split, mode, profile)
    return cluster, fixed, root


def test_task_mapping_covers_all_budgets_and_reuses_only_repetition_one(banks):
    cluster, fixed, root = banks
    specs = [epochs.task_spec(cluster, fixed, root, task) for task in range(1, 31)]
    assert epochs.EPOCHS == (3, 6, 10)
    assert epochs.LEARNING_RATES == (0.0007, 0.001, 0.002)
    assert len({str(spec["directory"]) for spec in specs}) == 30
    assert {spec["repetition"] for spec in specs} == {1}
    assert {spec["model_seed"] for spec in specs} == {42}
    for budget, expected_epochs in enumerate(epochs.EPOCHS):
        group = specs[budget * 10 : (budget + 1) * 10]
        assert {spec["epochs"] for spec in group} == {expected_epochs}
        assert {spec["total_steps"] for spec in group} == {2500 * expected_epochs}
        assert [(spec["mode"], spec["condition"], spec["pair_id"]) for spec in group] == [
            (mode, "identity", None) for mode in MODES
        ] + [(mode, "shift", pair) for pair in range(1, 5) for mode in MODES]
        for spec in group:
            assert Path(spec["train_dump"]).name.endswith("train_seed1001_2500steps_b32_r150_f5.h5")
            assert Path(spec["validation_dump"]).name.endswith("validation_seed3001_100steps_b32_r150_f5.h5")
            assert Path(spec["train_dump"]).is_file() and Path(spec["validation_dump"]).is_file()
            assert Path(spec["directory"]).is_relative_to(root)
    assert len({spec["train_dump"] for spec in specs}) == 10
    assert len({spec["validation_dump"] for spec in specs}) == 10


@pytest.mark.parametrize("task", [0, 31, True, "1"])
def test_invalid_task_is_rejected_without_creating_outputs(tmp_path, task):
    with pytest.raises(ValueError, match="1 to 30"):
        epochs.task_spec(tmp_path / "cluster", tmp_path / "fixed", tmp_path / "output", task)
    assert not (tmp_path / "output").exists()


@pytest.fixture
def compute(monkeypatch):
    h5py = pytest.importorskip("h5py")
    calls = {"train": [], "validation": [], "test": []}

    def metadata(path):
        with h5py.File(path, "r") as stream:
            return json.loads(stream.attrs["metadata_json"])

    def train(dump, directory, **options):
        calls["train"].append((Path(dump), Path(directory), options.copy()))
        assert options["steps"] == 2500 and options["epochs"] in epochs.EPOCHS
        assert options["batch_size"] == 32 and options["seed"] == 42
        assert options["learning_rate"] in epochs.LEARNING_RATES
        data = metadata(dump)
        config = {
            "seed": 42, "batch_size": 32, "learning_rate": options["learning_rate"],
            "optimizer": "schedulefree.AdamWScheduleFree", "weight_decay": 0.0,
            "gradient_clip_norm": 1.0, "loss": "query_cross_entropy", "precision": "float32",
            "dump_path": str(Path(dump).resolve()), "dump_sha256": _sha256_file(Path(dump)),
            "epochs": options["epochs"], "steps_per_epoch": 2500,
        }
        path = Path(directory) / "latest.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format": "nano_graph_u_v1", "step": 2500 * options["epochs"],
                    "model_config": PAPER_MODEL_CONFIG, "model": {"fixture": torch.tensor(0)},
                    "dump_metadata": data, "prior_config": data["prior_config"],
                    "train_config": config}, path)
        return path

    def evaluate(checkpoint, dump, **options):
        assert options == {"batch_size": 32, "device": "cpu"}
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        config, data = payload["train_config"], metadata(dump)
        split = "validation" if "validation" in Path(dump).name else "test"
        calls[split].append((Path(checkpoint), Path(dump), config.copy()))
        if split == "validation":
            scores = {3: (0.9, 0.9, 0.7), 6: (0.7, 0.9, 0.8), 10: (0.7, 0.8, 0.9)}
            score = scores[config["epochs"]][epochs.LEARNING_RATES.index(config["learning_rate"])]
        else:
            source = payload["prior_config"]
            target = data["prior_config"]
            score = (0.60 + 0.001 * config["epochs"]
                     + 0.01 * MODES.index(source["graph_u_mode"])
                     + 0.02 * bool(source["graph_u_query_location"])
                     + 0.03 * MODES.index(target["graph_u_mode"])
                     + 0.04 * bool(target["graph_u_query_location"])
                     + 0.0001 * target["graph_u_query_location"])
        return {
            "format": "nano_graph_u_evaluation_v1", "checkpoint": str(Path(checkpoint).resolve()),
            "checkpoint_sha256": _sha256_file(Path(checkpoint)), "checkpoint_step": payload["step"],
            "dump": str(Path(dump).resolve()), "dump_sha256": _sha256_file(Path(dump)),
            "dump_metadata": data, "task_count": 3200, "query_row_count": 160000,
            "metric_task_counts": {name: 3200 for name in epochs.METRIC_NAMES},
            "metrics": {name: score for name in epochs.METRIC_NAMES},
        }

    monkeypatch.setattr(epochs, "train_model", train)
    monkeypatch.setattr(epochs, "evaluate_checkpoint", evaluate)
    return calls


def test_all_budgets_select_matching_validation_winners_and_collect_without_sd(banks, compute, monkeypatch):
    cluster, fixed, root = banks
    lock, summary_locks = epochs._lock, []

    def checked_lock(path, *args, **kwargs):
        if Path(path).name == ".summary.job.lock":
            summary_locks.append(kwargs.get("blocking"))
        return lock(path, *args, **kwargs)

    monkeypatch.setattr(epochs, "_lock", checked_lock)
    before = {path: _sha256_file(path) for base in (cluster, fixed) for path in base.rglob("*.h5")}
    for task in range(1, 31):
        epochs.run_experiment(cluster, fixed, root, task, device="cpu")
        if task < 30:
            assert not (root / "results/summary.csv").exists()
    assert len(compute["train"]) == len(compute["validation"]) == 90
    assert len(compute["test"]) == 156
    assert summary_locks and all(summary_locks)
    winners = {3: 0.0007, 6: 0.001, 10: 0.002}
    for _, dump, config in compute["validation"]:
        assert dump.name.startswith(Path(config["dump_path"]).name.split("_train_")[0] + "_validation_")
    assert all(config["learning_rate"] == winners[config["epochs"]]
               for _, _, config in compute["test"])
    rows = pd.read_csv(root / "results/per_repetition.csv")
    summary = pd.read_csv(root / "results/summary.csv")
    assert len(rows) == len(summary) == 192
    assert set(rows["repetition"]) == {1}
    assert set(rows["epochs"]) == set(epochs.EPOCHS)
    assert not any(column.endswith("_std") for column in summary)
    keys = ["epochs", "fixed_location", "fixed_scale", "training_graph_u_mode",
            "training_condition", "test_graph_u_mode", "test_condition"]
    assert not rows.duplicated(keys).any()
    assert rows.groupby(["epochs", "fixed_location", "fixed_scale"]).size().eq(16).all()
    assert len(rows["checkpoint_sha256"].unique()) == 30
    assert all(_sha256_file(path) == digest for path, digest in before.items())
    # Completed tasks reuse their own selected candidate and cached tests.
    epochs.run_experiment(cluster, fixed, root, 1, device="cpu")
    assert len(compute["train"]) == len(compute["validation"]) == 90
    assert len(compute["test"]) == 156
    # A complete marker must not cause collection to run inference for a lost report.
    first = Path(epochs.task_spec(cluster, fixed, root, 1)["directory"])
    next((first / "tests").glob("*.json")).unlink()
    with pytest.raises(FileNotFoundError, match="missing its cached test report"):
        epochs._collect_if_complete(cluster, fixed, root)
    assert len(compute["train"]) == len(compute["validation"]) == 90
    assert len(compute["test"]) == 156


@pytest.mark.parametrize("split", ["train", "validation"])
def test_partial_source_bank_fails_before_optimization(banks, compute, split):
    h5py = pytest.importorskip("h5py")
    cluster, fixed, root = banks
    spec = epochs.task_spec(cluster, fixed, root, 3)
    with h5py.File(spec[f"{split}_dump"], "r+") as stream:
        stream.attrs["committed_steps"] = 2499 if split == "train" else 99
    with pytest.raises(ValueError, match="[Ii]ncomplete"):
        epochs.run_experiment(cluster, fixed, root, 3, device="cpu")
    assert not compute["train"] and not compute["validation"] and not compute["test"]


@pytest.mark.parametrize("corruption", ["checkpoint_budget", "report_budget", "metric_count"])
def test_completed_candidate_rejects_wrong_epoch_or_incomplete_scores(banks, compute, corruption):
    cluster, fixed, root = banks
    epochs.run_experiment(cluster, fixed, root, 3, device="cpu")
    directory = Path(epochs.task_spec(cluster, fixed, root, 3)["directory"])
    candidate = directory / "candidates/lr_0p0007"
    if corruption == "checkpoint_budget":
        path = candidate / "checkpoints/latest.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload["step"] = 2500
        torch.save(payload, path)
    else:
        path = candidate / "validation.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        report = payload["validation_report"]
        if corruption == "report_budget":
            report["checkpoint_step"] = 2500
        else:
            report["metric_task_counts"]["roc_auc"] = 3199
        path.write_text(json.dumps(payload), encoding="utf-8")
    counts = {key: len(values) for key, values in compute.items()}
    with pytest.raises(ValueError):
        epochs.run_experiment(cluster, fixed, root, 3, device="cpu")
    assert counts == {key: len(values) for key, values in compute.items()}


def test_interruption_after_training_resumes_same_candidate_before_validation(banks, compute, monkeypatch):
    cluster, fixed, root = banks
    evaluate = epochs.evaluate_checkpoint

    def interrupted(*args, **kwargs):
        raise RuntimeError("simulated interruption before validation")

    monkeypatch.setattr(epochs, "evaluate_checkpoint", interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        epochs.run_experiment(cluster, fixed, root, 3, device="cpu")
    assert len(compute["train"]) == 1
    assert not compute["train"][0][2]["resume"]
    assert not compute["validation"] and not compute["test"]
    monkeypatch.setattr(epochs, "evaluate_checkpoint", evaluate)
    result = epochs.run_experiment(cluster, fixed, root, 3, device="cpu")
    assert len(compute["train"]) == 4
    assert compute["train"][1][2]["resume"]
    assert len(compute["validation"]) == 3 and len(compute["test"]) == 4
    assert result["winner"]["learning_rate"] == 0.0007
