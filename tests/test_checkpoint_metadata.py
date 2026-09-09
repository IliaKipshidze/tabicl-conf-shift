"""Tests for reproducible pre-training checkpoint metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tabicl.prior.graph_lib._config import PriorConfig
from tabicl.train._checkpoint import (
    build_run_config,
    checkpoint_training_metadata,
    get_matched_graph_u_config,
    load_matched_graph_u_config,
    normalize_config,
    save_run_config,
)
from tabicl.train._train_config import build_parser


def _graph_u_checkpoint() -> dict:
    return {
        "config": {"embed_dim": 128},
        "state_dict": {},
        **checkpoint_training_metadata(
            {
                "prior_type": "graph_scm",
                "graph_u_enabled": True,
                "graph_u_structure_mode": "add_root",
                "graph_u_query_location": 2.25,
                "graph_u_query_scale": 1.5,
                "graph_u_force_gaussian": False,
                "graph_u_max_attempts": 321,
            },
            vars(
                PriorConfig(
                    graph_u_enabled=True,
                    graph_u_structure_mode="add_root",
                    graph_u_query_location=2.25,
                    graph_u_query_scale=1.5,
                    graph_u_force_gaussian=False,
                    graph_u_max_attempts=321,
                )
            ),
        ),
    }


def test_normalize_config_copies_and_serializes_supported_values():
    args = argparse.Namespace(
        path=Path("checkpoints/run"),
        integer=np.int64(7),
        device=torch.device("cpu"),
        values=(1, 2),
    )

    normalized = normalize_config(args)
    args.integer = 99

    assert normalized == {
        "path": str(Path("checkpoints/run")),
        "integer": 7,
        "device": "cpu",
        "values": [1, 2],
    }
    assert json.loads(json.dumps(normalized)) == normalized


def test_training_cli_snapshot_preserves_graph_u_shift_parameters():
    args = build_parser().parse_args(
        [
            "--prior_type",
            "graph_scm",
            "--graph_u_enabled",
            "true",
            "--graph_u_structure_mode",
            "add_root",
            "--graph_u_query_location",
            "-2.25",
            "--graph_u_query_scale",
            "1.5",
            "--graph_u_force_gaussian",
            "false",
        ]
    )

    snapshot = normalize_config(args)

    assert snapshot["prior_type"] == "graph_scm"
    assert snapshot["graph_u_enabled"] is True
    assert snapshot["graph_u_structure_mode"] == "add_root"
    assert snapshot["graph_u_query_location"] == -2.25
    assert snapshot["graph_u_query_scale"] == 1.5
    assert snapshot["graph_u_force_gaussian"] is False


def test_checkpoint_training_metadata_round_trips_with_weights_only(tmp_path):
    checkpoint = _graph_u_checkpoint()
    path = tmp_path / "model.ckpt"
    torch.save(checkpoint, path)

    loaded = torch.load(path, map_location="cpu", weights_only=True)

    assert loaded["config"] == {"embed_dim": 128}
    assert loaded["training_config"]["graph_u_query_location"] == 2.25
    assert loaded["prior_config"]["graph_u_force_gaussian"] is False


def test_run_config_is_atomic_and_does_not_overwrite_a_different_launch(tmp_path):
    first = build_run_config(
        model_config={"embed_dim": 128},
        training_config={"prior_type": "graph_scm", "run": 1},
        prior_config=vars(PriorConfig()),
    )
    first_path = save_run_config(tmp_path, first, current_step=0)
    same_path = save_run_config(tmp_path, first, current_step=10)
    second = build_run_config(
        model_config={"embed_dim": 128},
        training_config={"prior_type": "graph_scm", "run": 2},
        prior_config=vars(PriorConfig()),
    )
    second_path = save_run_config(tmp_path, second, current_step=10)

    assert first_path == tmp_path / "run_config.json"
    assert same_path == first_path
    assert second_path == tmp_path / "run_config.step-10.json"
    assert json.loads(first_path.read_text(encoding="utf-8")) == first
    assert json.loads(second_path.read_text(encoding="utf-8")) == second
    assert not list(tmp_path.glob("*.tmp"))

    third = build_run_config(
        model_config={"embed_dim": 128},
        training_config={"prior_type": "graph_scm", "run": 3},
        prior_config=vars(PriorConfig()),
    )
    third_path = save_run_config(tmp_path, third, current_step=10)

    assert third_path == tmp_path / "run_config.step-10.1.json"
    assert json.loads(second_path.read_text(encoding="utf-8")) == second
    assert json.loads(third_path.read_text(encoding="utf-8")) == third


def test_get_matched_graph_u_config_returns_exact_training_shift():
    matched = get_matched_graph_u_config(_graph_u_checkpoint())

    assert matched == {
        "prior_type": "graph_scm",
        "graph_u_enabled": True,
        "graph_u_structure_mode": "add_root",
        "graph_u_query_location": 2.25,
        "graph_u_query_scale": 1.5,
        "graph_u_force_gaussian": False,
        "graph_u_max_attempts": 321,
    }


def test_matched_config_prefers_effective_pre_generated_prior_metadata():
    checkpoint = _graph_u_checkpoint()
    checkpoint["training_config"]["prior_type"] = "mix_scm"
    checkpoint["prior_config"]["prior_type"] = "graph_scm"

    matched = get_matched_graph_u_config(checkpoint)

    assert matched["prior_type"] == "graph_scm"


def test_matched_config_maps_pre_structure_mode_checkpoint_to_reject():
    checkpoint = _graph_u_checkpoint()
    del checkpoint["prior_config"]["graph_u_structure_mode"]

    matched = get_matched_graph_u_config(checkpoint)

    assert matched["graph_u_structure_mode"] == "reject"


def test_load_matched_graph_u_config_reads_checkpoint_path(tmp_path):
    path = tmp_path / "model.ckpt"
    torch.save(_graph_u_checkpoint(), path)

    assert load_matched_graph_u_config(path) == get_matched_graph_u_config(
        _graph_u_checkpoint()
    )


def test_matched_graph_u_config_rejects_legacy_checkpoint_without_metadata():
    with pytest.raises(TypeError, match="legacy checkpoints"):
        get_matched_graph_u_config({"config": {}, "state_dict": {}})


@pytest.mark.parametrize(
    "training_updates,prior_updates,error",
    [
        ({"prior_type": "mlp_scm"}, {}, "prior_type='graph_scm'"),
        ({}, {"graph_u_enabled": False}, "not trained with Graph-U"),
        ({}, {"graph_u_structure_mode": "unknown"}, "must be 'reject' or 'add_root'"),
        ({}, {"graph_u_query_scale": 0.0}, "finite and positive"),
    ],
)
def test_matched_graph_u_config_rejects_incompatible_metadata(
    training_updates, prior_updates, error
):
    checkpoint = _graph_u_checkpoint()
    checkpoint["training_config"].update(training_updates)
    checkpoint["prior_config"].update(prior_updates)

    with pytest.raises(ValueError, match=error):
        get_matched_graph_u_config(checkpoint)


def test_matched_graph_u_config_rejects_missing_field():
    checkpoint = _graph_u_checkpoint()
    del checkpoint["prior_config"]["graph_u_query_location"]

    with pytest.raises(ValueError, match="graph_u_query_location"):
        get_matched_graph_u_config(checkpoint)


def test_graph_u_structure_mode_validation_uses_base_dag_size():
    add_root = PriorConfig(
        graph_u_enabled=True,
        graph_u_structure_mode="add_root",
        max_n_nodes=2,
    )

    assert add_root.graph_u_structure_mode == "add_root"
    with pytest.raises(ValueError, match="'reject'.*max_n_nodes >= 3"):
        PriorConfig(
            graph_u_enabled=True,
            graph_u_structure_mode="reject",
            max_n_nodes=2,
        )
