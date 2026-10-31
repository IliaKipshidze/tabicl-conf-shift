"""The sampled experiment keeps matching-validation tuning and frozen winners."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from tabicl.nano_graph_u import select as selection
from tabicl.nano_graph_u import tune
from tabicl.nano_graph_u.data import _metadata
from tabicl.nano_graph_u.train import _sha256_file
from tabicl.prior.graph_lib._config import GRAPH_U_SAMPLING_DEFAULTS
from test_nano_graph_u_tune import _checkpoint, _prior, _report


def _sampling():
    return {
        **GRAPH_U_SAMPLING_DEFAULTS,
        "graph_u_shift_mode": "sampled",
        "graph_u_query_scale_std": 0.08,
        "graph_u_query_location_std": 0.12,
    }


def _bank_metadata(condition, mode, sampling, *, steps, seed):
    values = dict(sampling)
    if condition == "identity":
        values["graph_u_shift_mode"] = "fixed"
    return _metadata(
        replace(_prior(condition, mode), **values),
        steps=steps, batch_size=32, seed=seed, rows=150, features=5,
        min_train_size=0.3, max_train_size=0.9, n_jobs=1,
    )


def _write_bank(path, metadata):
    h5py = pytest.importorskip("h5py")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = metadata["steps"] * 32
    with h5py.File(path, "w") as stream:
        stream.attrs["metadata_json"] = json.dumps(metadata)
        stream.attrs["committed_steps"] = metadata["steps"]
        stream.create_dataset("x", (count, 150, 5), dtype="f4")
        stream.create_dataset("y", (count, 150), dtype="i8")
        stream.create_dataset("train_sizes", (count,), dtype="i4")


@pytest.mark.parametrize("mode", ["confounding", "covariate"])
@pytest.mark.parametrize("task", [1, 11])
def test_sampled_trial_keeps_same_recipe_and_matches_its_validation_bank(
    tmp_path, monkeypatch, mode, task
):
    sampling = _sampling()
    before = copy.deepcopy(sampling)
    spec = tune.trial_spec(
        task, tmp_path / "data", tmp_path / "search",
        graph_u_mode=mode, sampling_config=sampling,
    )
    assert spec["sampling_config"] == sampling
    assert spec["model_seed"] == 42
    assert spec["learning_rate"] == 0.0001
    banks = {}
    for split, steps, seed, path in (
        ("train", 2500, 1001, spec["train_dump"]),
        ("validation", 100, 3001, spec["validation_dump"]),
    ):
        metadata = _bank_metadata(spec["condition"], mode, sampling, steps=steps, seed=seed)
        _write_bank(path, metadata)
        banks[split] = metadata
        expected_mode = "fixed" if spec["condition"] == "identity" else "sampled"
        assert metadata["prior_config"]["graph_u_shift_mode"] == expected_mode
        tune.validate_bank(
            path, spec["condition"], split, 1 if split == "train" else None,
            graph_u_mode=mode, sampling_config=sampling,
        )
    calls = []

    def train(dump, directory, **kwargs):
        assert dump == spec["train_dump"]
        assert directory == spec["checkpoint_dir"]
        assert kwargs == dict(
            steps=2500, batch_size=32, seed=42, device="cpu",
            learning_rate=0.0001, save_every=250, log_every=25, resume=False,
        )
        calls.append("train")
        return _checkpoint(spec, banks["train"])

    def evaluate(checkpoint, dump, **kwargs):
        assert dump == spec["validation_dump"]
        assert "_test_" not in Path(dump).name
        assert kwargs == dict(batch_size=32, device="cpu")
        calls.append("validation")
        return _report(spec, banks["validation"])

    monkeypatch.setattr(tune, "train_model", train)
    monkeypatch.setattr(tune, "evaluate_checkpoint", evaluate)
    options = dict(device="cpu", graph_u_mode=mode, sampling_config=sampling)
    result = tune.run_trial(task, tmp_path / "data", tmp_path / "search", **options)
    assert tune.validate_trial_result(result, spec) == 0.83
    assert tune.run_trial(task, tmp_path / "data", tmp_path / "search", **options) == result
    assert calls == ["train", "validation"]
    assert sampling == before

    altered = copy.deepcopy(result)
    altered["validation_report"]["dump_metadata"]["prior_config"]["graph_u_query_scale_std"] = 0.05
    with pytest.raises(ValueError, match="prior_config"):
        tune.validate_trial_result(altered, spec, verify_files=False)
    if spec["condition"] == "shift":
        with pytest.raises(ValueError, match="prior_config"):
            tune._validate_metadata(banks["validation"], "shift", "validation", graph_u_mode=mode)


@pytest.mark.parametrize("mode", ["confounding", "covariate"])
def test_sampled_selection_checks_all_trials_and_records_its_profile(tmp_path, monkeypatch, mode):
    sampling = _sampling()
    data_root, search_root = tmp_path / "data", tmp_path / "search"
    specs = [
        tune.trial_spec(task, data_root, search_root, graph_u_mode=mode, sampling_config=sampling)
        for task in range(1, 101)
    ]
    for spec in specs:
        checkpoint = Path(spec["checkpoint_dir"]) / "latest.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(f"model-{spec['task_id']}".encode())
        metadata = _bank_metadata(spec["condition"], mode, sampling, steps=100, seed=3001)
        score = 0.9 if spec["lr_index"] == (2 if spec["condition"] == "identity" else 5) else 0.7
        result = dict(
            format=tune.TRIAL_FORMAT, trial=spec, selection_metric="roc_auc",
            validation_report=dict(
                format="nano_graph_u_evaluation_v1", checkpoint=str(checkpoint),
                checkpoint_sha256=_sha256_file(checkpoint), checkpoint_step=2500,
                dump=spec["validation_dump"], dump_sha256="a" * 64,
                dump_metadata=metadata, task_count=3200,
                metric_task_counts=dict(roc_auc=3200), metrics=dict(roc_auc=score),
            ),
        )
        Path(spec["result_path"]).write_text(json.dumps(result), encoding="utf-8")
    validated = []

    def validate(result, spec, **kwargs):
        assert kwargs["verify_files"] is True
        assert spec["sampling_config"] == sampling
        validated.append(spec["task_id"])
        return tune.validate_trial_result(result, spec, verify_files=False)

    monkeypatch.setattr(selection, "validate_trial_result", validate)
    manifest = selection.select_winners(
        data_root, search_root, graph_u_mode=mode, sampling_config=sampling
    )
    assert validated == list(range(1, 101))
    assert manifest["sampling_config"] == sampling
    assert len(manifest["winners"]) == 10
    for winner in manifest["winners"]:
        assert winner["learning_rate"] == (0.0003 if winner["condition"] == "identity" else 0.001)
        assert Path(winner["checkpoint"]).is_file()
    assert selection.select_winners(
        data_root, search_root, graph_u_mode=mode, sampling_config=sampling
    ) == manifest
