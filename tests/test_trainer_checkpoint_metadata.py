"""Integration coverage for metadata written by the pre-training Trainer."""

from __future__ import annotations

import argparse

import pytest
import torch

pytest.importorskip("wandb")
pytest.importorskip("transformers")

from tabicl.train._run import Trainer


def test_trainer_checkpoint_keeps_model_config_and_adds_training_provenance(tmp_path):
    trainer = Trainer.__new__(Trainer)
    trainer.config = argparse.Namespace(checkpoint_dir=str(tmp_path))
    trainer.model_config = {"embed_dim": 16, "max_classes": 3}
    trainer.raw_model = torch.nn.Linear(2, 3)
    trainer.optimizer = torch.optim.AdamW(trainer.raw_model.parameters())
    trainer.scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=1)
    trainer.curr_step = 12
    trainer.training_config = {
        "prior_type": "graph_scm",
        "graph_u_enabled": True,
        "graph_u_query_location": 2.0,
        "graph_u_query_scale": 1.5,
        "graph_u_force_gaussian": False,
    }
    trainer.prior_config = {
        "prior_type": "graph_scm",
        "graph_u_enabled": True,
        "graph_u_query_location": 2.0,
        "graph_u_query_scale": 1.5,
        "graph_u_force_gaussian": False,
        "graph_u_max_attempts": 1000,
    }

    trainer.save_checkpoint("step-12.ckpt")
    checkpoint = torch.load(
        tmp_path / "step-12.ckpt",
        map_location="cpu",
        weights_only=True,
    )

    assert checkpoint["config"] == trainer.model_config
    assert checkpoint["curr_step"] == 12
    assert checkpoint["training_config"] == trainer.training_config
    assert checkpoint["prior_config"] == trainer.prior_config
    assert checkpoint["checkpoint_metadata_version"] == 1
