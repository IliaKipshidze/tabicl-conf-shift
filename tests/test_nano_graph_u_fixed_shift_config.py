"""Fixed shift pairs retain the original training and validation protocol."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from test_nano_graph_u_tune import _checkpoint, _prior, _report

from tabicl.nano_graph_u import tune
from tabicl.nano_graph_u._settings import (
    experiment_prior_config,
    normalize_fixed_shift_config,
)
from tabicl.nano_graph_u.data import _metadata
from tabicl.prior.graph_lib._config import GRAPH_U_SAMPLING_DEFAULTS

PAIRS = ((4, 1.5), (20, 2), (60, 3), (100, 4))


def _config(pair):
    return {"graph_u_query_location": pair[0], "graph_u_query_scale": pair[1]}


def _bank_metadata(pair, mode, split):
    steps, seed = (2500, 1001) if split == "train" else (100, 3001)
    return _metadata(
        replace(_prior("shift", mode), **_config(pair)),
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


@pytest.mark.parametrize("pair", PAIRS)
@pytest.mark.parametrize("mode", ["confounding", "covariate"])
@pytest.mark.parametrize("split", ["train", "validation"])
def test_each_pair_requires_matching_fixed_train_and_validation_metadata(pair, mode, split):
    config = _config(pair)
    metadata = _bank_metadata(pair, mode, split)
    before = copy.deepcopy(metadata)
    repetition = 1 if split == "train" else None
    assert tune._validate_metadata(
        metadata, "shift", split, repetition, graph_u_mode=mode,
        fixed_shift_config=config,
    ) == metadata["steps"]
    assert metadata == before
    with pytest.raises(ValueError, match="prior_config"):
        tune._validate_metadata(
            metadata, "shift", split, repetition, graph_u_mode=mode,
            fixed_shift_config=_config((pair[0] + 1, pair[1])),
        )
    with pytest.raises(ValueError, match="prior_config"):
        tune._validate_metadata(metadata, "shift", split, repetition, graph_u_mode=mode)

    metadata["prior_config"]["graph_u_shift_mode"] = "sampled"
    with pytest.raises(ValueError, match="prior_config"):
        tune._validate_metadata(
            metadata, "shift", split, repetition, graph_u_mode=mode,
            fixed_shift_config=config,
        )


@pytest.mark.parametrize("pair", PAIRS)
@pytest.mark.parametrize("mode", ["confounding", "covariate"])
def test_fixed_pair_trial_validates_files_binds_recipe_and_reuses_result(
    tmp_path, monkeypatch, pair, mode,
):
    config = _config(pair)
    before = copy.deepcopy(config)
    spec = tune.trial_spec(
        11, tmp_path / "data", tmp_path / "search",
        graph_u_mode=mode, fixed_shift_config=config,
    )
    banks = {split: _bank_metadata(pair, mode, split) for split in ("train", "validation")}
    for split, metadata in banks.items():
        _write_bank(spec[f"{split}_dump"], metadata)
        assert tune.validate_bank(
            spec[f"{split}_dump"], "shift", split, 1 if split == "train" else None,
            graph_u_mode=mode, fixed_shift_config=config,
        ) == metadata
    calls = []

    def train(dump, checkpoint_dir, **kwargs):
        assert dump == spec["train_dump"]
        assert checkpoint_dir == spec["checkpoint_dir"]
        assert kwargs["steps"] == 2500
        assert kwargs["seed"] == 42
        assert kwargs["learning_rate"] == tune.LEARNING_RATES[0]
        calls.append("train")
        return _checkpoint(spec, banks["train"])

    def evaluate(checkpoint, dump, **kwargs):
        assert dump == spec["validation_dump"]
        assert "_test_" not in Path(dump).name
        calls.append("validation")
        return _report(spec, banks["validation"])

    monkeypatch.setattr(tune, "train_model", train)
    monkeypatch.setattr(tune, "evaluate_checkpoint", evaluate)
    options = {"device": "cpu", "graph_u_mode": mode, "fixed_shift_config": config}
    result = tune.run_trial(11, tmp_path / "data", tmp_path / "search", **options)
    assert tune.validate_trial_result(result, spec) == 0.83
    assert tune.run_trial(11, tmp_path / "data", tmp_path / "search", **options) == result
    assert calls == ["train", "validation"]
    assert config == before
    assert spec["fixed_shift_config"] == _config(tuple(map(float, pair)))
    assert "sampling_config" not in spec

    altered = copy.deepcopy(result)
    altered["trial"]["fixed_shift_config"]["graph_u_query_location"] += 1
    with pytest.raises(ValueError, match="prior_config"):
        tune.validate_trial_result(altered, altered["trial"], verify_files=False)


@pytest.mark.parametrize("pair", PAIRS)
def test_fixed_validation_mismatch_is_rejected_before_training(tmp_path, monkeypatch, pair):
    config = _config(pair)
    spec = tune.trial_spec(11, tmp_path / "data", tmp_path / "search", fixed_shift_config=config)
    _write_bank(spec["train_dump"], _bank_metadata(pair, "confounding", "train"))
    _write_bank(
        spec["validation_dump"],
        _bank_metadata((pair[0], pair[1] + 1), "confounding", "validation"),
    )
    monkeypatch.setattr(tune, "train_model", lambda *args, **kwargs: pytest.fail("Must not train"))
    with pytest.raises(ValueError, match="prior_config"):
        tune.run_trial(
            11, tmp_path / "data", tmp_path / "search", device="cpu",
            fixed_shift_config=config,
        )


@pytest.mark.parametrize("pair", PAIRS)
@pytest.mark.parametrize("mode", ["confounding", "covariate"])
def test_explicit_pair_changes_only_shift_parameters_and_keeps_iid_original(pair, mode):
    config = _config(pair)
    assert asdict(experiment_prior_config("shift", mode, fixed_shift_config=config)) == asdict(
        replace(_prior("shift", mode), **config)
    )
    assert asdict(experiment_prior_config("identity", mode, fixed_shift_config=config)) == asdict(
        _prior("identity", mode)
    )


@pytest.mark.parametrize("condition,task", [("identity", 1), ("shift", 11)])
@pytest.mark.parametrize("mode", ["confounding", "covariate"])
def test_legacy_defaults_and_trial_payloads_remain_unchanged(tmp_path, condition, task, mode):
    assert asdict(experiment_prior_config(condition, mode)) == asdict(_prior(condition, mode))
    original = tune.trial_spec(task, tmp_path, tmp_path, graph_u_mode=mode)
    explicit_none = tune.trial_spec(
        task, tmp_path, tmp_path, graph_u_mode=mode, fixed_shift_config=None,
    )
    assert original == explicit_none
    assert "fixed_shift_config" not in original
    assert "sampling_config" not in original


@pytest.mark.parametrize(
    "config",
    [
        {}, {"graph_u_query_location": 4}, {**_config((4, 1.5)), "extra": 0},
        {"location": 4, "scale": 1.5},
        *(_config((value, 1.5)) for value in (True, "4", 1 + 2j, float("nan"), float("inf"))),
        *(_config((4, value)) for value in (False, "1.5", 0, -1, float("nan"), float("-inf"))),
    ],
)
def test_invalid_fixed_shift_recipes_are_rejected(config):
    with pytest.raises(ValueError):
        normalize_fixed_shift_config(config)


def test_normalization_copies_values_to_plain_floats():
    config = _config((20, 2))
    normalized = normalize_fixed_shift_config(config)
    assert normalized == config
    assert all(type(value) is float for value in normalized.values())
    normalized["graph_u_query_location"] = 60
    assert config == _config((20, 2))


def test_fixed_and_sampled_recipes_are_mutually_exclusive_everywhere(tmp_path):
    options = {
        "sampling_config": {**GRAPH_U_SAMPLING_DEFAULTS, "graph_u_shift_mode": "sampled"},
        "fixed_shift_config": _config((4, 1.5)),
    }
    with pytest.raises(ValueError, match="mutually exclusive"):
        experiment_prior_config("shift", "confounding", **options)
    with pytest.raises(ValueError, match="mutually exclusive"):
        tune.trial_spec(11, tmp_path, tmp_path, **options)
    with pytest.raises(ValueError, match="mutually exclusive"):
        tune._validate_metadata(_bank_metadata((4, 1.5), "confounding", "validation"), "shift", "validation", **options)
    with pytest.raises(ValueError, match="mutually exclusive"):
        tune.run_trial(11, tmp_path, tmp_path, **options)
