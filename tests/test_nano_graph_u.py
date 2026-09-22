"""Focused checks for the separate paper-scale Nano-Graph-U path."""

from __future__ import annotations

import pytest
import torch

from tabicl.nano_graph_u.data import NanoPriorDumpLoader, adapt_batch, generate_dump
from tabicl.nano_graph_u.model import NanoTabPFNModel
from tabicl.prior.graph_lib._config import PriorConfig


def test_paper_nano_model_parameter_count_and_query_shape():
    model = NanoTabPFNModel(
        embedding_size=96,
        num_attention_heads=4,
        mlp_hidden_size=192,
        num_layers=3,
        num_outputs=2,
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 356_066
    with torch.inference_mode():
        logits = model((torch.randn(2, 12, 5), torch.zeros(2, 7)), 7)
    assert logits.shape == (2, 5, 2)


def test_nano_adapter_refuses_different_boundaries_within_batch():
    raw = (
        torch.zeros(2, 12, 3),
        torch.zeros(2, 12, dtype=torch.long),
        torch.tensor([3, 3]),
        torch.tensor([12, 12]),
        torch.tensor([7, 8]),
    )
    with pytest.raises(ValueError, match="same support/query split"):
        adapt_batch(raw)


def test_nano_rejects_splits_too_small_for_normalization(tmp_path):
    pytest.importorskip("h5py")
    with pytest.raises(ValueError, match="at least two support"):
        generate_dump(
            tmp_path / "too_small.h5",
            steps=1,
            batch_size=2,
            seed=7,
            prior_config=PriorConfig(
                graph_u_enabled=True, graph_u_structure_mode="add_root"
            ),
            rows=4,
            features=3,
        )


def test_nano_parallel_generation_is_identical_and_resume_checks_code(
    tmp_path, monkeypatch
):
    h5py = pytest.importorskip("h5py")
    prior_config = PriorConfig(graph_u_enabled=True, graph_u_structure_mode="add_root")
    serial = tmp_path / "serial.h5"
    parallel = tmp_path / "parallel.h5"
    kwargs = {
        "steps": 2,
        "batch_size": 2,
        "seed": 19,
        "prior_config": prior_config,
        "rows": 64,
        "features": 3,
    }
    generate_dump(serial, workers=1, **kwargs)
    generate_dump(parallel, workers=2, **kwargs)
    with h5py.File(serial, "r") as first, h5py.File(parallel, "r") as second:
        for key in ("x", "y", "train_sizes"):
            assert (first[key][:] == second[key][:]).all()

    monkeypatch.setattr(
        "tabicl.nano_graph_u.data._generator_source_sha256", lambda: "changed-code"
    )
    with pytest.raises(ValueError, match="different generation metadata"):
        generate_dump(serial, workers=1, resume=True, **kwargs)


def test_nano_graph_u_generation_training_and_loader_boundary(tmp_path):
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("schedulefree")
    from tabicl.nano_graph_u.evaluate import evaluate_checkpoint
    from tabicl.nano_graph_u.train import train_model

    dump = tmp_path / "shifted.h5"
    prior_config = PriorConfig(
        graph_u_enabled=True,
        graph_u_structure_mode="add_root",
        graph_u_query_location=2.0,
        graph_u_query_scale=1.5,
        graph_u_force_gaussian=False,
    )
    generate_dump(
        dump,
        steps=2,
        batch_size=2,
        seed=7,
        prior_config=prior_config,
        rows=64,
        features=3,
        n_jobs=1,
    )
    with h5py.File(dump, "r") as stream:
        assert stream.attrs["committed_steps"] == 2
        assert stream["x"].shape == (4, 64, 3)
        assert stream["train_sizes"][0] == stream["train_sizes"][1]

    batch = next(iter(NanoPriorDumpLoader(dump, batch_size=2)))
    split = batch["train_test_split_index"]
    assert 0 < split < 64
    assert set(torch.unique(batch["y"]).tolist()) == {0, 1}

    checkpoint_dir = tmp_path / "checkpoints"
    train_model(
        dump,
        checkpoint_dir,
        steps=1,
        batch_size=2,
        seed=11,
        device="cpu",
        save_every=1,
        log_every=1,
    )
    checkpoint = train_model(
        dump,
        checkpoint_dir,
        steps=2,
        batch_size=2,
        seed=11,
        device="cpu",
        save_every=1,
        log_every=1,
        resume=True,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["format"] == "nano_graph_u_v1"
    assert payload["step"] == 2
    assert payload["prior_config"]["graph_u_query_location"] == 2.0
    assert payload["prior_config"]["graph_u_query_scale"] == 1.5

    uninterrupted = train_model(
        dump,
        tmp_path / "uninterrupted",
        steps=2,
        batch_size=2,
        seed=11,
        device="cpu",
        save_every=2,
        log_every=1,
    )
    uninterrupted_payload = torch.load(
        uninterrupted, map_location="cpu", weights_only=True
    )
    for name, weight in payload["model"].items():
        assert torch.equal(weight, uninterrupted_payload["model"][name])

    # Wiring smoke only: a real evaluation must use an independent dump.
    with pytest.raises(ValueError, match="training dump"):
        evaluate_checkpoint(checkpoint, dump, batch_size=2, device="cpu")
    report = evaluate_checkpoint(
        checkpoint, dump, batch_size=2, device="cpu", allow_training_dump=True
    )
    assert report["task_count"] == 4
    assert report["metric_task_counts"]["roc_auc"] == 4

    with h5py.File(dump, "r+") as stream:
        stream["train_sizes"][1] = split + 1
    with pytest.raises(ValueError, match="same support/query split"):
        next(iter(NanoPriorDumpLoader(dump, batch_size=2)))
