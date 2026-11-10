"""The larger Nano experiment retains its data, epoch and selection contracts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from tabicl.nano_graph_u import large_experiment as large
from tabicl.nano_graph_u._settings import experiment_prior_config
from tabicl.nano_graph_u.data import _metadata
from tabicl.nano_graph_u.train import (
    BATCH_ORDER_SCHEME,
    PAPER_MODEL_CONFIG,
    _sha256_file,
)


def _sparse_bank(spec):
    h5py = pytest.importorskip("h5py")
    path = Path(spec["dump"])
    data = _metadata(
        experiment_prior_config(spec["condition"], spec["mode"]),
        steps=spec["steps"], batch_size=32, seed=spec["seed"],
        rows=1000, features=8, min_train_size=0.3, max_train_size=0.9, n_jobs=1,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    count = spec["steps"] * 32
    with h5py.File(path, "w") as stream:
        stream.attrs["metadata_json"] = json.dumps(data, sort_keys=True)
        stream.attrs["committed_steps"] = spec["steps"]
        stream.create_dataset("x", (count, 1000, 8), chunks=(32, 1000, 8), dtype="f4")
        stream.create_dataset("y", (count, 1000), chunks=(32, 1000), dtype="i8")
        stream.create_dataset("train_sizes", (count,), chunks=(32,), dtype="i4")
    assert path.stat().st_size < 100_000  # Full-sized shapes, no allocated task chunks.
    return data


@pytest.fixture
def banks(tmp_path):
    root = tmp_path / "large experiment"
    large._initialize(root)
    for task in range(1, 13):
        _sparse_bank(large.bank_spec(root, task))
    return root


@pytest.fixture
def compute(monkeypatch):
    h5py = pytest.importorskip("h5py")
    calls = {"fit": [], "validation": [], "test": [], "training": []}
    fitting = {"done": False}

    def metadata(path):
        with h5py.File(path, "r") as stream:
            return json.loads(stream.attrs["metadata_json"])

    def train(dump, directory, **options):
        assert not fitting["done"]
        assert options["steps"] == 15625 and options["epochs"] in (1, 3, 9)
        assert options["batch_size"] == 32 and options["seed"] == 42
        assert options["shuffle_batches"] is True
        assert options["learning_rate"] in (0.0007, 0.001, 0.002)
        calls["fit"].append((Path(dump), options.copy()))
        data = metadata(dump)
        assert data["rows"] == 1000 and data["features"] == 8 and data["steps"] * 32 == 500000
        config = {
            "seed": 42, "batch_size": 32, "learning_rate": options["learning_rate"],
            "optimizer": "schedulefree.AdamWScheduleFree", "weight_decay": 0.0,
            "gradient_clip_norm": 1.0, "loss": "query_cross_entropy", "precision": "float32",
            "dump_path": str(Path(dump).resolve()), "dump_sha256": _sha256_file(Path(dump)),
            "shuffle_batches": True, "batch_order_scheme": BATCH_ORDER_SCHEME,
        }
        if options["epochs"] != 1:
            config.update(epochs=options["epochs"], steps_per_epoch=15625)
        checkpoint = Path(directory) / "latest.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format": "nano_graph_u_v1", "step": 15625 * options["epochs"],
                    "model_config": PAPER_MODEL_CONFIG, "model": {"fixture": torch.tensor(0)},
                    "dump_metadata": data, "prior_config": data["prior_config"], "train_config": config},
                   checkpoint)
        return checkpoint

    def evaluate(checkpoint, dump, **options):
        assert options["batch_size"] == 32 and options["device"] == "cpu"
        assert options.get("max_batches") is None
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        config, data = payload["train_config"], metadata(dump)
        budget = payload["step"] // 15625
        if Path(dump).resolve() == Path(config["dump_path"]).resolve():
            split = "training"
            assert options["allow_training_dump"] is True
        else:
            split = "validation" if "validation" in Path(dump).name else "test"
            assert not options.get("allow_training_dump", False)
        assert (split == "validation") != fitting["done"]
        calls[split].append((Path(checkpoint), Path(dump), budget, config.copy()))
        if split == "validation":
            scores = {1: (0.9, 0.9, 0.7), 3: (0.7, 0.9, 0.8), 9: (0.7, 0.8, 0.9)}
            score = scores[budget][large.LEARNING_RATES.index(config["learning_rate"])]
        else:
            prior = data["prior_config"]
            source = payload["prior_config"]
            score = (0.60 + 0.001 * budget + 0.01 * (source["graph_u_mode"] == "covariate")
                     + 0.02 * bool(source["graph_u_query_location"])
                     + 0.03 * (prior["graph_u_mode"] == "covariate")
                     + 0.04 * bool(prior["graph_u_query_location"]))
        count = data["steps"] * data["batch_size"]
        return {"format": "nano_graph_u_evaluation_v1", "checkpoint": str(Path(checkpoint).resolve()),
                "checkpoint_sha256": _sha256_file(Path(checkpoint)), "checkpoint_step": payload["step"],
                "dump": str(Path(dump).resolve()), "dump_sha256": _sha256_file(Path(dump)),
                "dump_metadata": data, "task_count": count, "query_row_count": count * 500,
                "metrics": {metric: score for metric in large.METRIC_NAMES},
                "metric_task_counts": {metric: count for metric in large.METRIC_NAMES}}

    monkeypatch.setattr(large, "train_model", train)
    monkeypatch.setattr(large, "evaluate_checkpoint", evaluate)
    return calls, fitting


def test_task_mapping_retains_large_dimensions_counts_and_epoch_budgets(tmp_path):
    root = tmp_path / "new experiment"
    assert (large.ROWS, large.FEATURES, large.BATCH_SIZE) == (1000, 8, 32)
    assert (large.TRAIN_STEPS, large.EVAL_STEPS) == (15625, 1000)
    assert large.EPOCHS == (1, 3, 9)
    assert large.LEARNING_RATES == (0.0007, 0.001, 0.002)
    specs = [large.bank_spec(root, task) for task in range(1, 13)]
    assert len({str(spec["dump"]) for spec in specs}) == 12
    assert [(spec["mode"], spec["condition"], spec["split"]) for spec in specs] == [
        (mode, condition, split) for mode in ("confounding", "covariate")
        for condition in ("identity", "shift") for split in ("train", "validation", "test")]
    for spec in specs:
        assert spec["steps"] == (15625 if spec["split"] == "train" else 1000)
        assert spec["seed"] == {"train": 1001, "validation": 3001, "test": 2001}[spec["split"]]
    trials = [large.trial_spec(root, task) for task in range(1, 37)]
    for index, budget in enumerate((1, 3, 9)):
        group = trials[index * 12 : (index + 1) * 12]
        assert {trial["epochs"] for trial in group} == {budget}
        assert {trial["total_steps"] for trial in group} == {15625 * budget}
        assert [trial["learning_rate"] for trial in group] == list(large.LEARNING_RATES) * 4
    assert len({str(trial["result_path"]) for trial in trials}) == 36
    winners = [large.winner_spec(root, task) for task in range(1, 13)]
    assert [(winner["epochs"], winner["mode"], winner["condition"]) for winner in winners] == [
        (budget, mode, condition) for budget in (1, 3, 9)
        for mode in ("confounding", "covariate") for condition in ("identity", "shift")]
    assert len({str(winner["checkpoint"]) for winner in winners}) == 12
    assert not root.exists()


def test_generation_uses_requested_prior_sizes_and_resumes_partial_banks(tmp_path, monkeypatch):
    h5py = pytest.importorskip("h5py")
    root = tmp_path / "generated"
    specs = [large.bank_spec(root, task) for task in range(1, 13)]
    calls = []

    def generate(path, **options):
        spec = next(spec for spec in specs if Path(spec["dump"]) == Path(path))
        assert options["steps"] == spec["steps"] and options["seed"] == spec["seed"]
        assert (options["batch_size"], options["rows"], options["features"]) == (32, 1000, 8)
        assert options["workers"] == 2 and options["n_jobs"] == 1
        prior = options["prior_config"]
        assert prior.graph_u_mode == spec["mode"] and not prior.graph_u_force_gaussian
        assert (prior.graph_u_query_location, prior.graph_u_query_scale) == (
            (0, 1) if spec["condition"] == "identity" else (2, 1.5))
        if options["resume"] and Path(path).exists():
            with h5py.File(path, "r") as stream:
                if stream.attrs["committed_steps"] == options["steps"]:
                    return path
        calls.append(options.copy())
        _sparse_bank(spec)
        return path

    monkeypatch.setattr(large, "generate_dump", generate)
    for task in range(1, 13):
        large.generate_bank(root, task, workers=2)
    assert len(calls) == 12
    large.generate_bank(root, 1, workers=2)
    assert len(calls) == 12
    with h5py.File(specs[0]["dump"], "r+") as stream:
        stream.attrs["committed_steps"] = 15624
    large.generate_bank(root, 1, workers=2)
    assert len(calls) == 13 and calls[-1]["resume"] is True


def test_full_workflow_selects_final_budget_winners_and_scores_all_banks(banks, compute, monkeypatch):
    calls, fitting = compute
    monkeypatch.setattr(large, "_require_smoke", lambda root: None)
    before = {path: _sha256_file(path) for path in banks.rglob("*.h5")}
    for task in range(1, 37):
        large.run_trial(banks, task, device="cpu")
    assert len(calls["fit"]) == len(calls["validation"]) == 36
    assert not calls["test"] and not calls["training"]
    fitting["done"] = True
    for task in range(1, 13):
        large.evaluate_winner(banks, task, device="cpu")
        if task < 12:
            assert not (banks / "results/summary.csv").exists()
    assert len(calls["test"]) == 48 and len(calls["training"]) == 12
    winners = {1: 0.0007, 3: 0.001, 9: 0.002}
    assert all(config["learning_rate"] == winners[budget]
               for _, _, budget, config in calls["test"] + calls["training"])
    raw = pd.read_csv(banks / "results/per_repetition.csv")
    summary = pd.read_csv(banks / "results/summary.csv")
    training = pd.read_csv(banks / "results/training_per_repetition.csv")
    training_summary = pd.read_csv(banks / "results/training_summary.csv")
    selected = pd.read_csv(banks / "results/winners.csv")
    assert len(raw) == len(summary) == 48
    assert len(training) == len(training_summary) == len(selected) == 12
    assert raw["task_count"].eq(32000).all() and training["task_count"].eq(500000).all()
    assert raw["repetition"].eq(1).all() and training["repetition"].eq(1).all()
    assert raw["checkpoint_sha256"].nunique() == training["checkpoint_sha256"].nunique() == 12
    assert raw["training_steps"].eq(15625 * raw["epochs"]).all()
    assert not any(column.endswith("_std") for frame in (summary, training_summary) for column in frame)
    keys = ["epochs", "training_graph_u_mode", "training_condition", "test_graph_u_mode", "test_condition"]
    assert not raw.duplicated(keys).any() and raw.groupby("epochs").size().eq(16).all()
    np.testing.assert_allclose(summary.set_index(keys).sort_index()[[f"{m}_mean" for m in large.METRIC_NAMES]],
                               raw.set_index(keys).sort_index()[list(large.METRIC_NAMES)], rtol=0, atol=1e-12)
    assert all(_sha256_file(path) == digest for path, digest in before.items())
    counts = {key: len(value) for key, value in calls.items()}
    # Cache reads may validate files but must neither optimize nor numerically score again.
    large.run_trial(banks, 1, device="cpu")
    large.evaluate_winner(banks, 1, device="cpu")
    assert counts == {key: len(value) for key, value in calls.items()}


@pytest.mark.parametrize("split", ["train", "validation"])
def test_incomplete_bank_fails_before_fitting(banks, compute, monkeypatch, split):
    h5py = pytest.importorskip("h5py")
    monkeypatch.setattr(large, "_require_smoke", lambda root: None)
    spec = large.bank_spec(banks, 1 if split == "train" else 2)
    with h5py.File(spec["dump"], "r+") as stream:
        stream.attrs["committed_steps"] = spec["steps"] - 1
    with pytest.raises(ValueError, match="[Ii]ncomplete"):
        large.run_trial(banks, 1, device="cpu")
    assert not compute[0]["fit"]


def test_missing_memory_probe_stops_before_fitting(banks, compute):
    with pytest.raises((ValueError, FileNotFoundError), match="[Ss]moke|[Pp]robe"):
        large.run_trial(banks, 1, device="cpu")
    assert not compute[0]["fit"]


@pytest.mark.parametrize("corruption", ["shape", "device", "status"])
def test_memory_probe_must_match_the_full_cuda_training_configuration(tmp_path, corruption):
    root = large._initialize(tmp_path / "probe")
    report = {"format": "nano_graph_u_large_memory_smoke_v1", "status": "passed",
              "recipe": large._recipe(), "support_rows": 900, "device": "cuda:0"}
    if corruption == "shape":
        report["recipe"]["rows"] = 150
    elif corruption == "device":
        report["device"] = "cpu"
    else:
        report["status"] = "failed"
    (root / "smoke.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="[Ss]moke"):
        large._require_smoke(root)


def test_trial_resumes_saved_weights_after_interruption_before_validation(banks, compute, monkeypatch):
    monkeypatch.setattr(large, "_require_smoke", lambda root: None)
    evaluate = large.evaluate_checkpoint

    def interrupted(*args, **kwargs):
        raise RuntimeError("interrupted before validation")

    monkeypatch.setattr(large, "evaluate_checkpoint", interrupted)
    with pytest.raises(RuntimeError, match="interrupted before validation"):
        large.run_trial(banks, 1, device="cpu")
    assert len(compute[0]["fit"]) == 1 and not compute[0]["fit"][0][1]["resume"]
    monkeypatch.setattr(large, "evaluate_checkpoint", evaluate)
    large.run_trial(banks, 1, device="cpu")
    assert len(compute[0]["fit"]) == 2 and compute[0]["fit"][1][1]["resume"]
    assert len(compute[0]["validation"]) == 1


@pytest.mark.parametrize("task", [0, 37, True, "1"])
def test_invalid_candidate_has_no_filesystem_side_effect(tmp_path, task):
    root = tmp_path / "invalid"
    with pytest.raises(ValueError):
        large.trial_spec(root, task)
    assert not root.exists()


@pytest.mark.parametrize("corruption", ["step", "metric_count", "dump"])
def test_stale_validation_checkpoint_and_bank_binding_are_rejected(banks, compute, monkeypatch, corruption):
    monkeypatch.setattr(large, "_require_smoke", lambda root: None)
    large.run_trial(banks, 1, device="cpu")
    spec = large.trial_spec(banks, 1)
    path = Path(spec["result_path"])
    saved = json.loads(path.read_text(encoding="utf-8"))
    report = saved["validation_report"]
    if corruption == "step":
        report["checkpoint_step"] = 2500
    elif corruption == "metric_count":
        report["metric_task_counts"]["roc_auc"] = 31999
    else:
        report["dump"] = str(large.bank_spec(banks, 3)["dump"])
    path.write_text(json.dumps(saved), encoding="utf-8")
    counts = {key: len(value) for key, value in compute[0].items()}
    with pytest.raises(ValueError):
        large.run_trial(banks, 1, device="cpu")
    assert counts == {key: len(value) for key, value in compute[0].items()}


def test_capacity_guard_skips_cuda_queries_for_cpu_smoke_tests(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("CPU tests must not query CUDA availability or device properties")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected)
    monkeypatch.setattr(torch.cuda, "get_device_properties", unexpected)
    large._check_capacity(None, "cpu")


def test_capacity_guard_requires_available_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="cannot see the allocated GPU"):
        large._check_capacity({"peak_allocated_bytes": 10}, "cuda")


@pytest.mark.parametrize("capacity, sufficient", [(9, False), (10, True)])
def test_capacity_guard_compares_allocated_gpu_to_full_size_probe(monkeypatch, capacity, sufficient):
    devices = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def properties(device):
        devices.append(device)
        return SimpleNamespace(total_memory=capacity)

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    if sufficient:
        large._check_capacity({"peak_allocated_bytes": 10}, "cuda:2")
    else:
        with pytest.raises(RuntimeError, match="less memory than the full-size smoke"):
            large._check_capacity({"peak_allocated_bytes": 10}, "cuda:2")
    assert devices == [torch.device("cuda:2")]
