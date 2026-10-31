"""Sampled source shifts survive metadata storage and evaluation reconstruction."""

import dataclasses
import json

import pytest
import torch

from tabicl.evaluation._graph_u import (
    CheckpointDescriptor,
    build_prior_config,
    parse_condition,
    prior_reconstruction_details,
    resolve_conditions,
)
from tabicl.prior._genload import SavePriorDataset
from tabicl.prior.graph_lib._config import GRAPH_U_SAMPLING_DEFAULTS, PriorConfig
from tabicl.train._checkpoint import (
    checkpoint_training_metadata,
    get_matched_graph_u_config,
    load_matched_graph_u_config,
)
from tabicl.train._train_config import build_parser


def _sampled_config(mode="confounding"):
    return PriorConfig(
        graph_u_enabled=True,
        graph_u_structure_mode="add_root",
        graph_u_mode=mode,
        graph_u_shift_mode="sampled",
        graph_u_query_scale_mean=1.55,
        graph_u_query_scale_std=0.07,
        graph_u_query_location_mean=2.05,
        graph_u_query_location_std=0.12,
    )


def _checkpoint(config):
    return checkpoint_training_metadata(
        {"prior_type": "graph_scm"}, dataclasses.asdict(config)
    )


def _descriptor(tmp_path, config):
    payload = _checkpoint(config)
    return CheckpointDescriptor(
        path=tmp_path / "model.ckpt",
        label="model",
        task_type="classification",
        model_config={"max_classes": 2},
        training_config=payload["training_config"],
        prior_config=payload["prior_config"],
        curr_step=2500,
        size_bytes=100,
        sha256="0" * 64,
    )


@pytest.mark.parametrize("mode", ["confounding", "covariate"])
def test_sampled_checkpoint_round_trip_and_matched_evaluation(tmp_path, mode):
    config = _sampled_config(mode)
    path = tmp_path / "model.ckpt"
    torch.save(_checkpoint(config), path)
    matched = load_matched_graph_u_config(path)
    for key in GRAPH_U_SAMPLING_DEFAULTS:
        assert matched[key] == getattr(config, key)

    reference = _descriptor(tmp_path, config)
    condition = resolve_conditions([parse_condition("matched")], reference)[0]
    restored = build_prior_config(reference, condition)
    assert dataclasses.asdict(restored) == dataclasses.asdict(config)
    for key in GRAPH_U_SAMPLING_DEFAULTS:
        assert condition.as_dict()[key] == getattr(config, key)
    assert prior_reconstruction_details(reference)["complete"] is True


@pytest.mark.parametrize(
    "key", [key for key in GRAPH_U_SAMPLING_DEFAULTS if key != "graph_u_shift_mode"]
)
def test_sampled_checkpoint_never_guesses_missing_distribution_parameters(key):
    checkpoint = _checkpoint(_sampled_config())
    del checkpoint["prior_config"][key]
    with pytest.raises(ValueError, match=key):
        get_matched_graph_u_config(checkpoint)


@pytest.mark.parametrize("name", ["identity", "ordinary", "explicit:2:1.5"])
def test_explicit_evaluation_conditions_do_not_inherit_sampling(tmp_path, name):
    reference = _descriptor(tmp_path, _sampled_config())
    condition = resolve_conditions([parse_condition(name)], reference)[0]
    config = build_prior_config(reference, condition)
    assert config.graph_u_shift_mode == "fixed"
    expected = (2.0, 1.5) if name.startswith("explicit") else (0.0, 1.0)
    assert (config.graph_u_query_location, config.graph_u_query_scale) == expected


def test_historical_checkpoints_without_sampling_fields_remain_fixed(tmp_path):
    config = PriorConfig(
        graph_u_enabled=True,
        graph_u_structure_mode="add_root",
        graph_u_query_location=2.0,
        graph_u_query_scale=1.5,
    )
    reference = _descriptor(tmp_path, config)
    for key in GRAPH_U_SAMPLING_DEFAULTS:
        del reference.prior_config[key]
    matched = get_matched_graph_u_config(
        {"training_config": reference.training_config, "prior_config": reference.prior_config}
    )
    assert matched["graph_u_query_scale"] == 1.5
    condition = resolve_conditions([parse_condition("matched")], reference)[0]
    assert build_prior_config(reference, condition).graph_u_shift_mode == "fixed"
    assert prior_reconstruction_details(reference)["complete"] is True


def test_sparse_prior_writer_records_sampling_distribution(tmp_path):
    writer = SavePriorDataset.__new__(SavePriorDataset)
    writer.args = build_parser().parse_args([])
    writer.args.regression = False
    for key in GRAPH_U_SAMPLING_DEFAULTS:
        setattr(writer.args, key, getattr(_sampled_config(), key))
    writer.save_dir = tmp_path
    writer.save_metadata()
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    for key in GRAPH_U_SAMPLING_DEFAULTS:
        assert metadata[key] == getattr(_sampled_config(), key)
