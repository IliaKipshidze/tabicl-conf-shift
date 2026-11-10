"""Stronger test shifts score frozen epoch winners and reuse IID references."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from test_nano_graph_u_epoch_experiments import banks as banks  # noqa: PLC0414
from test_nano_graph_u_epoch_experiments import compute as compute  # noqa: PLC0414
from test_nano_graph_u_epoch_training_eval import (
    training_compute as training_compute,  # noqa: PLC0414
)

from tabicl.nano_graph_u import epoch_experiments as epochs
from tabicl.nano_graph_u import epoch_test_shift as sweep
from tabicl.nano_graph_u import fixed_experiments as fixed
from tabicl.nano_graph_u.data import _metadata
from tabicl.nano_graph_u.train import _sha256_file


@pytest.fixture
def source(banks, compute, training_compute):
    for task in range(1, 31):
        epochs.run_experiment(*banks, task, device="cpu")
    training_compute()
    for task in range(1, 31):
        epochs.run_training_evaluation(*banks, task, device="cpu")
    cluster, fixed, original = banks
    return cluster, fixed, original, original.parent / "stronger test shifts"


def _fingerprints(roots):
    return {path: _sha256_file(path) for root in roots for path in root.rglob("*")
            if path.is_file() and path.suffix in (".h5", ".pt", ".json", ".csv")}


@pytest.fixture
def scoring(source, monkeypatch):
    h5py = pytest.importorskip("h5py")
    calls = {"generate": [], "evaluate": []}

    def forbidden(*args, **kwargs):
        pytest.fail("test-only experiment entered optimization or model selection")

    def generate(path, **options):
        calls["generate"].append((Path(path), options.copy()))
        assert options["steps"] == 100 and options["batch_size"] == 32
        assert options["seed"] == 6001 and options["rows"] == 150 and options["features"] == 5
        assert options["n_jobs"] == 1
        metadata = _metadata(options["prior_config"], steps=100, batch_size=32, seed=6001,
                             rows=150, features=5, min_train_size=0.3, max_train_size=0.9, n_jobs=1)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as stream:
            stream.attrs["metadata_json"] = json.dumps(metadata)
            stream.attrs["committed_steps"] = 100
            for key, shape, dtype in (("x", (3200, 150, 5), "f4"), ("y", (3200, 150), "i8"),
                                      ("train_sizes", (3200,), "i4")):
                stream.create_dataset(key, shape, dtype=dtype)
        return path

    def evaluate(checkpoint, dump, **options):
        assert options == {"batch_size": 32, "device": "cpu"}
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        with h5py.File(dump, "r") as stream:
            metadata = json.loads(stream.attrs["metadata_json"])
        assert metadata["seed"] == 6001
        assert payload["step"] == 2500 * payload["train_config"]["epochs"]
        calls["evaluate"].append((Path(checkpoint), Path(dump)))
        prior = metadata["prior_config"]
        score = (0.6 + 0.001 * payload["train_config"]["epochs"]
                 + 0.0001 * prior["graph_u_query_location"]
                 + 0.01 * (prior["graph_u_mode"] == "covariate"))
        return {
            "format": "nano_graph_u_evaluation_v1", "checkpoint": str(Path(checkpoint).resolve()),
            "checkpoint_sha256": _sha256_file(Path(checkpoint)), "checkpoint_step": payload["step"],
            "dump": str(Path(dump).resolve()), "dump_sha256": _sha256_file(Path(dump)),
            "dump_metadata": metadata, "task_count": 3200, "query_row_count": 160000,
            "metric_task_counts": {metric: 3200 for metric in epochs.METRIC_NAMES},
            "metrics": {metric: score for metric in epochs.METRIC_NAMES},
        }

    for name in ("train_model", "_candidate", "run_experiment"):
        monkeypatch.setattr(epochs, name, forbidden)
    monkeypatch.setattr(fixed, "generate_dump", generate)
    monkeypatch.setattr(sweep, "evaluate_checkpoint", evaluate)
    return calls


def test_eight_test_banks_have_only_the_requested_fixed_parameters(tmp_path):
    roots = [tmp_path / name for name in ("cluster", "fixed", "source", "output")]
    specs = [sweep.bank_spec(*roots, task) for task in range(1, 9)]
    assert sweep.MULTIPLIER == 1.5 and sweep.TEST_SEED == 6001
    assert len({spec["dump"] for spec in specs}) == 8
    assert [(spec["pair_id"], spec["graph_u_mode"]) for spec in specs] == [
        (pair, mode) for pair in range(1, 5) for mode in ("confounding", "covariate")]
    for spec in specs:
        assert spec["split"] == "test" and spec["repetition"] == 1
        assert spec["seed"] == 6001 and spec["steps"] == 100
        assert Path(spec["dump"]).is_relative_to(roots[-1])
        assert spec["fixed_shift_config"] == {
            "graph_u_query_location": 1.5 * spec["fixed_location"],
            "graph_u_query_scale": 1.5 * spec["fixed_scale"],
        }
    assert not roots[-1].exists()


@pytest.mark.parametrize("task", [0, 9, True, "1"])
def test_invalid_generation_tasks_do_not_create_outputs(tmp_path, task):
    roots = [tmp_path / name for name in ("cluster", "fixed", "source", "output")]
    with pytest.raises(ValueError, match="1 to 8"):
        sweep.bank_spec(*roots, task)
    assert not roots[-1].exists()


def test_all_stronger_test_banks_score_frozen_winners_and_reuse_references(source, scoring):
    cluster, fixed, original, output = source
    before = _fingerprints((cluster, fixed, original))
    for task in range(1, 9):
        sweep.generate_bank(*source, task, workers=2)
    assert len(scoring["generate"]) == 8
    profiles = {(options["prior_config"].graph_u_mode,
                 options["prior_config"].graph_u_query_location,
                 options["prior_config"].graph_u_query_scale)
                for _, options in scoring["generate"]}
    assert profiles == {(mode, location, scale) for mode in ("confounding", "covariate")
                        for location, scale in ((6, 2.25), (30, 3), (90, 4.5), (150, 6))}
    # Evaluation must reject a partial bank before scoring any model; generation
    # resumes just that bank rather than trusting its filename as completion.
    h5py = pytest.importorskip("h5py")
    first_bank = sweep.bank_spec(*source, 1)["dump"]
    with h5py.File(first_bank, "r+") as stream:
        stream.attrs["committed_steps"] = 99
    with pytest.raises(ValueError, match="[Ii]ncomplete"):
        sweep.evaluate_pair(*source, 1, device="cpu")
    assert not scoring["evaluate"]
    sweep.generate_bank(*source, 1, workers=2)
    assert len(scoring["generate"]) == 9 and scoring["generate"][-1][1]["resume"] is True
    for task in range(1, 5):
        sweep.evaluate_pair(*source, task, device="cpu")
        if task < 4:
            assert not (output / "results/summary.csv").exists()
    assert len(scoring["evaluate"]) == 96
    raw = pd.read_csv(output / "results/per_repetition.csv")
    summary = pd.read_csv(output / "results/summary.csv")
    assert len(raw) == len(summary) == 192
    assert raw["checkpoint_sha256"].nunique() == 30
    assert raw["repetition"].eq(1).all() and summary["repetition_count"].eq(1).all()
    assert raw["task_count"].eq(3200).all()
    assert not any(column.endswith("_std") for column in summary)
    keys = ["epochs", "fixed_location", "fixed_scale", "training_graph_u_mode",
            "training_condition", "test_graph_u_mode", "test_condition"]
    assert not raw.duplicated(keys).any()
    assert raw.groupby(["epochs", "fixed_location", "fixed_scale"]).size().eq(16).all()
    for parameter, iid_value in (("location", 0), ("scale", 1)):
        np.testing.assert_allclose(raw[f"test_query_{parameter}"],
            np.where(raw["test_condition"].eq("identity"), iid_value, 1.5 * raw[f"fixed_{parameter}"]))
        np.testing.assert_allclose(raw[f"training_query_{parameter}"],
            np.where(raw["training_condition"].eq("identity"), iid_value, raw[f"fixed_{parameter}"]))
    old = pd.read_csv(original / "results/per_repetition.csv")
    iid = raw.loc[raw["test_condition"].eq("identity")].set_index(keys).sort_index()
    expected = old.loc[old["test_condition"].eq("identity")].set_index(keys).sort_index()
    np.testing.assert_allclose(iid[list(epochs.METRIC_NAMES)], expected[list(epochs.METRIC_NAMES)])
    assert iid["test_dump_sha256"].equals(expected["test_dump_sha256"])
    for name in ("training_summary.csv", "training_per_repetition.csv", "winners.csv"):
        assert (output / "results" / name).read_bytes() == (original / "results" / name).read_bytes()
    assert all(_sha256_file(path) == digest for path, digest in before.items())
    # Reruns neither generate a new bank nor rescore any frozen checkpoint.
    sweep.generate_bank(*source, 1, workers=2)
    sweep.evaluate_pair(*source, 1, device="cpu")
    assert len(scoring["generate"]) == 9 and len(scoring["evaluate"]) == 96
    # Cached scores remain bound to the exact epoch checkpoint.
    path = Path(raw.loc[raw["test_condition"].eq("shift") & raw["fixed_location"].eq(4), "result_path"].iloc[0])
    original_bytes = path.read_bytes()
    saved = json.loads(original_bytes)
    report = next(value for value in saved.values()
                  if isinstance(value, dict) and value.get("format") == "nano_graph_u_evaluation_v1")
    report["checkpoint_step"] = 2500
    try:
        path.write_text(json.dumps(saved), encoding="utf-8")
        with pytest.raises(ValueError):
            sweep.evaluate_pair(*source, 1, device="cpu")
        assert len(scoring["evaluate"]) == 96
    finally:
        path.write_bytes(original_bytes)


def test_evaluation_requires_existing_models_and_data_without_generating_them(tmp_path, monkeypatch):
    roots = [tmp_path / name for name in ("cluster", "fixed", "source", "output")]

    def forbidden(*args, **kwargs):
        pytest.fail("evaluation tried to create models or data")

    monkeypatch.setattr(fixed, "generate_dump", forbidden)
    monkeypatch.setattr(sweep, "evaluate_checkpoint", forbidden)
    monkeypatch.setattr(epochs, "train_model", forbidden)
    with pytest.raises((ValueError, FileNotFoundError)):
        sweep.evaluate_pair(*roots, 1, device="cpu")
