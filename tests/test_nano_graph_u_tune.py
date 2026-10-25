"""Checks for the fixed-budget LR trial runner without costly model fits."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from tabicl.nano_graph_u import tune
from tabicl.nano_graph_u.data import _metadata
from tabicl.nano_graph_u.train import PAPER_MODEL_CONFIG, _sha256_file
from tabicl.prior.graph_lib._config import PriorConfig


def _prior(condition):
    return PriorConfig(
        graph_u_enabled=True,
        graph_u_structure_mode="add_root",
        graph_u_query_location=0.0 if condition == "identity" else 2.0,
        graph_u_query_scale=1.0 if condition == "identity" else 1.5,
        graph_u_force_gaussian=False,
        graph_u_max_attempts=1000,
        filter_unpredictable_graphs=True,
        filter_unpredictable_datasets=True,
        allow_act_warping=False,
    )


def _bank(path, condition, steps, seed):
    h5py = pytest.importorskip("h5py")
    metadata = _metadata(
        _prior(condition),
        steps=steps,
        batch_size=32,
        seed=seed,
        rows=150,
        features=5,
        min_train_size=0.3,
        max_train_size=0.9,
        n_jobs=1,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as file:
        file.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
        file.attrs["committed_steps"] = steps
        # Unallocated HDF5 datasets suffice for metadata/preflight tests.
        file.create_dataset("x", (steps * 32, 150, 5), dtype="f4")
        file.create_dataset("y", (steps * 32, 150), dtype="i8")
        file.create_dataset("train_sizes", (steps * 32,), dtype="i4")
    return metadata


def _setup(tmp_path, task=1):
    spec = tune.trial_spec(task, tmp_path / "data", tmp_path / "output")
    training = _bank(
        spec["train_dump"], spec["condition"], 2500, 1000 + spec["repetition"]
    )
    validation = _bank(spec["validation_dump"], spec["condition"], 100, 3001)
    return spec, training, validation


def _checkpoint(spec, metadata):
    path = Path(spec["checkpoint_dir"]) / "latest.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "nano_graph_u_v1",
            "step": 2500,
            "model_config": PAPER_MODEL_CONFIG,
            "dump_metadata": metadata,
            "prior_config": metadata["prior_config"],
            "train_config": {
                "seed": spec["model_seed"],
                "batch_size": 32,
                "learning_rate": spec["learning_rate"],
                "optimizer": "schedulefree.AdamWScheduleFree",
                "weight_decay": 0.0,
                "gradient_clip_norm": 1.0,
                "loss": "query_cross_entropy",
                "precision": "float32",
                "dump_path": spec["train_dump"],
                "dump_sha256": _sha256_file(Path(spec["train_dump"])),
            },
        },
        path,
    )
    return path


def _report(spec, metadata):
    checkpoint = Path(spec["checkpoint_dir"]) / "latest.pt"
    return {
        "format": "nano_graph_u_evaluation_v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "checkpoint_step": 2500,
        "dump": spec["validation_dump"],
        "dump_sha256": _sha256_file(Path(spec["validation_dump"])),
        "dump_metadata": metadata,
        "task_count": 3200,
        "query_row_count": 160000,
        "metric_task_counts": {"roc_auc": 3200},
        "metrics": {"roc_auc": 0.83},
    }


def test_lr_trial_mapping_covers_all_candidates_with_matched_initializations(tmp_path):
    specs = [
        tune.trial_spec(task, tmp_path / "data", tmp_path / "output")
        for task in range(1, 101)
    ]
    assert len({spec["result_path"] for spec in specs}) == 100
    assert {spec["model_seed"] for spec in specs} == {42, 43, 44, 45, 46}
    for repetition in range(1, 6):
        for condition in ("identity", "shift"):
            group = [
                spec
                for spec in specs
                if spec["repetition"] == repetition and spec["condition"] == condition
            ]
            assert tuple(spec["learning_rate"] for spec in group) == tune.LEARNING_RATES
            assert {spec["model_seed"] for spec in group} == {41 + repetition}
            assert len({spec["train_dump"] for spec in group}) == 1
            assert len({spec["validation_dump"] for spec in group}) == 1
            assert f"{condition}_validation_seed3001" in group[0]["validation_dump"]


@pytest.mark.parametrize("task", [0, 101, True, "1"])
def test_lr_trial_mapping_rejects_invalid_array_ids(tmp_path, task):
    with pytest.raises(ValueError, match="1 to 100"):
        tune.trial_spec(task, tmp_path, tmp_path)


@pytest.mark.parametrize(
    "corruption", ["partial", "wrong_condition", "wrong_seed", "wrong_scheme"]
)
def test_trial_preflight_rejects_bad_validation_before_training(
    tmp_path, monkeypatch, corruption
):
    h5py = pytest.importorskip("h5py")
    spec, _, metadata = _setup(tmp_path)
    with h5py.File(spec["validation_dump"], "r+") as file:
        if corruption == "partial":
            file.attrs["committed_steps"] = 99
        else:
            if corruption == "wrong_condition":
                metadata["prior_config"] = asdict(_prior("shift"))
            elif corruption == "wrong_seed":
                metadata["seed"] = 2001
            else:
                metadata["seed_scheme"] = "additive"
            file.attrs["metadata_json"] = json.dumps(metadata)
    monkeypatch.setattr(
        tune, "train_model", lambda *args, **kwargs: pytest.fail("Must not train")
    )
    with pytest.raises(ValueError):
        tune.run_trial(1, tmp_path / "data", tmp_path / "output", device="cpu")
    assert not Path(spec["result_path"]).exists()


@pytest.mark.parametrize("task", [1, 31])
def test_trial_reuses_existing_trainer_and_matching_validator_then_is_idempotent(
    tmp_path, monkeypatch, task
):
    spec, training, validation = _setup(tmp_path, task)
    calls = []

    def train(dump, checkpoint_dir, **kwargs):
        assert dump == spec["train_dump"]
        assert checkpoint_dir == spec["checkpoint_dir"]
        assert kwargs == {
            "steps": 2500,
            "batch_size": 32,
            "seed": spec["model_seed"],
            "device": "cpu",
            "learning_rate": spec["learning_rate"],
            "save_every": 250,
            "log_every": 25,
            "resume": False,
        }
        calls.append("train")
        return _checkpoint(spec, training)

    def evaluate(checkpoint, dump, **kwargs):
        assert Path(checkpoint) == Path(spec["checkpoint_dir"]) / "latest.pt"
        assert dump == spec["validation_dump"]
        assert kwargs == {"batch_size": 32, "device": "cpu"}
        calls.append("validation")
        return _report(spec, validation)

    monkeypatch.setattr(tune, "train_model", train)
    monkeypatch.setattr(tune, "evaluate_checkpoint", evaluate)
    first = tune.run_trial(task, tmp_path / "data", tmp_path / "output", device="cpu")
    assert calls == ["train", "validation"]
    assert first == json.loads(Path(spec["result_path"]).read_text())
    assert tune.validate_trial_result(first, spec) == 0.83
    second = tune.run_trial(task, tmp_path / "data", tmp_path / "output", device="cpu")
    assert first == second
    assert calls == ["train", "validation"]
    assert not list((tmp_path / "data").rglob("*test*"))


@pytest.mark.parametrize(
    "field,value", [("checkpoint_step", 2250), ("task_count", 3199)]
)
def test_result_rejects_incomplete_trial_report(tmp_path, field, value):
    spec, training, validation = _setup(tmp_path)
    _checkpoint(spec, training)
    result = {
        "format": tune.TRIAL_FORMAT,
        "trial": spec,
        "selection_metric": "roc_auc",
        "validation_report": _report(spec, validation),
    }
    result["validation_report"][field] = value
    with pytest.raises(ValueError, match="final checkpoint"):
        tune.validate_trial_result(result, spec)


def test_result_rejects_changed_checkpoints_and_wrong_model_seed(tmp_path):
    spec, training, validation = _setup(tmp_path)
    path = _checkpoint(spec, training)
    result = {
        "format": tune.TRIAL_FORMAT,
        "trial": spec,
        "selection_metric": "roc_auc",
        "validation_report": _report(spec, validation),
    }
    payload = torch.load(path, weights_only=True)
    payload["train_config"]["seed"] += 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="changed"):
        tune.validate_trial_result(result, spec)
    updated = copy.deepcopy(result)
    updated["validation_report"]["checkpoint_sha256"] = _sha256_file(path)
    with pytest.raises(ValueError, match="fixed-budget"):
        tune.validate_trial_result(updated, spec)


def test_trial_resumes_checkpoint_when_validation_result_is_missing(
    tmp_path, monkeypatch
):
    spec, training, validation = _setup(tmp_path)
    checkpoint = _checkpoint(spec, training)
    called = []

    def resume(dump, checkpoint_dir, **kwargs):
        assert kwargs["resume"] is True
        assert kwargs["steps"] == 2500
        called.append("resume")
        return checkpoint

    monkeypatch.setattr(tune, "train_model", resume)
    monkeypatch.setattr(
        tune, "evaluate_checkpoint", lambda *args, **kwargs: _report(spec, validation)
    )
    tune.run_trial(1, tmp_path / "data", tmp_path / "output", device="cpu")
    assert called == ["resume"]
    assert Path(spec["result_path"]).is_file()


def test_result_hash_cache_reuses_reads_but_detects_changed_files(
    tmp_path, monkeypatch
):
    spec, training, validation = _setup(tmp_path)
    checkpoint = _checkpoint(spec, training)
    result = {
        "format": tune.TRIAL_FORMAT,
        "trial": spec,
        "selection_metric": "roc_auc",
        "validation_report": _report(spec, validation),
    }
    hashed = []
    original_hash = tune._sha256_file

    def digest(path):
        hashed.append(path)
        return original_hash(path)

    monkeypatch.setattr(tune, "_sha256_file", digest)
    cache = {}
    tune.validate_trial_result(result, spec, hash_cache=cache)
    tune.validate_trial_result(result, spec, hash_cache=cache)
    assert len(hashed) == 3
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Checkpoint changed"):
        tune.validate_trial_result(result, spec, hash_cache=cache)
    assert len(hashed) == 4
