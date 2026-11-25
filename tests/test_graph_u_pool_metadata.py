"""Pool interventions survive checkpoints and evaluation reconstruction."""

import dataclasses
from copy import deepcopy

import pytest
import torch

from tabicl.evaluation._graph_u import (
    CheckpointDescriptor,
    build_prior_config,
    parse_condition,
    prior_reconstruction_details,
    resolve_conditions,
)
from tabicl.nano_graph_u._settings import experiment_prior_config
from tabicl.nano_graph_u.data import _metadata
from tabicl.nano_graph_u.tune import _validate_metadata
from tabicl.prior.graph_lib._config import GRAPH_U_POOL_DEFAULTS, PriorConfig
from tabicl.train._checkpoint import (
    checkpoint_training_metadata,
    get_matched_graph_u_config,
    load_matched_graph_u_config,
)


def _pool_config(mode="confounding", sampling="gaussian"):
    return PriorConfig(
        graph_u_enabled=True,
        graph_u_structure_mode="add_root",
        graph_u_mode=mode,
        graph_u_intervention_mode="support_pool",
        graph_u_pool_sampling=sampling,
        graph_u_pool_peak_percentile=82.0,
        graph_u_pool_std_fraction=0.17,
    )


def _checkpoint(config):
    return checkpoint_training_metadata(
        {"prior_type": "graph_scm"}, dataclasses.asdict(config)
    )


def _descriptor(tmp_path, payload):
    return CheckpointDescriptor(
        path=tmp_path / "model.ckpt", label="model", task_type="classification",
        model_config={"max_classes": 2}, training_config=payload["training_config"],
        prior_config=payload["prior_config"], curr_step=2500,
        size_bytes=100, sha256="0" * 64,
    )


@pytest.mark.parametrize("mode", ["confounding", "covariate"])
@pytest.mark.parametrize("sampling", ["uniform", "gaussian"])
def test_pool_checkpoint_and_matched_evaluation_round_trip(tmp_path, mode, sampling):
    config = _pool_config(mode, sampling)
    payload = _checkpoint(config)
    path = tmp_path / "model.ckpt"
    torch.save(payload, path)
    matched = load_matched_graph_u_config(path)
    for key in GRAPH_U_POOL_DEFAULTS:
        assert matched[key] == getattr(config, key)
    reference = _descriptor(tmp_path, payload)
    condition = resolve_conditions([parse_condition("matched")], reference)[0]
    restored = build_prior_config(reference, condition)
    assert dataclasses.asdict(restored) == dataclasses.asdict(config)
    for key in GRAPH_U_POOL_DEFAULTS:
        assert condition.as_dict()[key] == getattr(config, key)
    assert prior_reconstruction_details(reference)["complete"] is True


@pytest.mark.parametrize("missing", GRAPH_U_POOL_DEFAULTS)
def test_pool_checkpoint_never_guesses_missing_pool_parameters(missing):
    payload = _checkpoint(_pool_config())
    del payload["prior_config"][missing]
    with pytest.raises(ValueError, match=missing):
        get_matched_graph_u_config(payload)


def test_pool_checkpoint_does_not_require_unused_affine_parameters():
    payload = _checkpoint(_pool_config())
    del payload["prior_config"]["graph_u_query_location"]
    del payload["prior_config"]["graph_u_query_scale"]
    assert get_matched_graph_u_config(payload)["graph_u_intervention_mode"] == "support_pool"


def test_partial_pool_metadata_is_not_reported_as_complete(tmp_path):
    payload = _checkpoint(_pool_config())
    del payload["prior_config"]["graph_u_intervention_mode"]
    details = prior_reconstruction_details(_descriptor(tmp_path, payload))
    assert details["complete"] is False
    assert "graph_u_intervention_mode" in details["missing_fields_using_evaluator_defaults"]


def test_pool_reference_identity_evaluates_uniform_pool(tmp_path):
    reference = _descriptor(tmp_path, _checkpoint(_pool_config()))
    condition = resolve_conditions([parse_condition("identity")], reference)[0]
    config = build_prior_config(reference, condition)
    assert config.graph_u_enabled is True
    assert config.graph_u_intervention_mode == "support_pool"
    assert config.graph_u_pool_sampling == "uniform"
    assert config.graph_u_pool_peak_percentile == 82.0
    assert config.graph_u_pool_std_fraction == 0.17


@pytest.mark.parametrize("name", ["ordinary", "explicit:4:2"])
def test_nonpool_evaluation_does_not_inherit_pool_intervention(tmp_path, name):
    reference = _descriptor(tmp_path, _checkpoint(_pool_config()))
    condition = resolve_conditions([parse_condition(name)], reference)[0]
    config = build_prior_config(reference, condition)
    assert config.graph_u_intervention_mode == "affine"
    assert config.graph_u_enabled is (name != "ordinary")
    if name.startswith("explicit"):
        assert (config.graph_u_query_location, config.graph_u_query_scale) == (4.0, 2.0)


def test_historical_affine_checkpoint_keeps_old_matched_representation(tmp_path):
    config = PriorConfig(graph_u_enabled=True, graph_u_structure_mode="add_root")
    payload = _checkpoint(config)
    for key in GRAPH_U_POOL_DEFAULTS:
        del payload["prior_config"][key]
    reference = _descriptor(tmp_path, payload)
    matched = get_matched_graph_u_config(payload)
    assert not set(GRAPH_U_POOL_DEFAULTS).intersection(matched)
    condition = resolve_conditions([parse_condition("matched")], reference)[0]
    assert build_prior_config(reference, condition).graph_u_intervention_mode == "affine"
    assert prior_reconstruction_details(reference)["complete"] is True


def _training_metadata():
    return _metadata(
        experiment_prior_config("shift", "confounding"), steps=2500,
        batch_size=32, seed=1001, rows=150, features=5,
        min_train_size=0.3, max_train_size=0.9, n_jobs=1,
    )


def test_historical_nano_bank_normalization_does_not_rewrite_metadata():
    metadata = _training_metadata()
    for key in GRAPH_U_POOL_DEFAULTS:
        del metadata["prior_config"][key]
    original = deepcopy(metadata)
    assert _validate_metadata(metadata, "shift", "train", 1) == 2500
    assert metadata == original


def test_fixed_nano_recipe_rejects_pool_bank_instead_of_mislabeling_it():
    metadata = _training_metadata()
    metadata["prior_config"]["graph_u_intervention_mode"] = "support_pool"
    with pytest.raises(ValueError, match="prior_config"):
        _validate_metadata(metadata, "shift", "train", 1)


def test_partial_new_bank_is_not_mistaken_for_a_legacy_affine_bank():
    metadata = _training_metadata()
    del metadata["prior_config"]["graph_u_intervention_mode"]
    with pytest.raises(ValueError, match="prior_config"):
        _validate_metadata(metadata, "shift", "train", 1)
