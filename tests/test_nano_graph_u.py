"""Focused checks for the separate paper-scale Nano-Graph-U path."""

from __future__ import annotations

import json
import random

import numpy as np
import pytest
import torch

from tabicl.nano_graph_u import data as nano_data
from tabicl.nano_graph_u.data import (
    NanoPriorDumpLoader,
    _generate_step,
    _seed_batch,
    adapt_batch,
    generate_dump,
)
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


@pytest.mark.parametrize(
    "first,second",
    [
        ((1001, 1000, 0), (2001, 0, 0)),
        ((1001, 1, 0), (1002, 0, 0)),
        ((1001, 0, 1), (1001 + 1_000_003, 0, 0)),
        ((1001, 0, 0), (1001, 0, 1)),
    ],
)
def test_nano_batch_seeds_separate_files_batches_and_retries(first, second):
    def draw(identifiers):
        _seed_batch(*identifiers)
        return (
            [random.getrandbits(64) for _ in range(8)],
            np.random.randint(0, 2**32, size=8, dtype=np.uint32),
            torch.rand(8),
        )

    initial = draw(first)
    other = draw(second)
    repeated = draw(first)
    assert initial[0] == repeated[0]
    np.testing.assert_array_equal(initial[1], repeated[1])
    assert torch.equal(initial[2], repeated[2])
    assert initial[0] != other[0]
    assert not np.array_equal(initial[1], other[1])
    assert not torch.equal(initial[2], other[2])


def test_nano_previously_overlapping_train_test_seeds_generate_distinct_batches():
    kwargs = {
        "prior_config": PriorConfig(
            graph_u_enabled=True,
            graph_u_structure_mode="add_root",
            graph_u_query_location=2.0,
            graph_u_query_scale=1.5,
        ),
        "batch_size": 2,
        "rows": 64,
        "features": 3,
        "min_train_size": 0.3,
        "max_train_size": 0.9,
    }
    training = _generate_step(1000, seed=1001, **kwargs)
    testing = _generate_step(0, seed=2001, **kwargs)
    repeated = _generate_step(1000, seed=1001, **kwargs)
    assert not np.array_equal(training[0], testing[0])
    for original, replay in zip(training, repeated):
        np.testing.assert_array_equal(original, replay)


def test_nano_parallel_generation_is_identical_and_resume_checks_code(
    tmp_path, monkeypatch
):
    h5py = pytest.importorskip("h5py")
    prior_config = PriorConfig(graph_u_enabled=True, graph_u_structure_mode="add_root")
    serial = tmp_path / "serial.h5"
    parallel = tmp_path / "parallel.h5"
    resumed = tmp_path / "resumed.h5"
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

    write_step = nano_data._write_step

    def interrupt_second_batch(file, step, batch_size, arrays):
        if step == 1:
            # Simulate a partially written batch that was never committed.
            file["x"][step * batch_size] = np.nan
            file.flush()
            raise RuntimeError("simulated interruption")
        write_step(file, step, batch_size, arrays)

    with monkeypatch.context() as patch:
        patch.setattr(nano_data, "_write_step", interrupt_second_batch)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            generate_dump(resumed, workers=1, **kwargs)
    with h5py.File(resumed, "r") as stream:
        assert stream.attrs["committed_steps"] == 1
        assert np.isnan(stream["x"][kwargs["batch_size"]]).all()

    generate_dump(resumed, workers=2, resume=True, **kwargs)
    with (
        h5py.File(serial, "r") as first,
        h5py.File(parallel, "r") as second,
        h5py.File(resumed, "r") as third,
    ):
        for key in ("x", "y", "train_sizes"):
            np.testing.assert_array_equal(first[key][:], second[key][:])
            np.testing.assert_array_equal(first[key][:], third[key][:])
        metadata = json.loads(first.attrs["metadata_json"])
        assert metadata["seed_scheme"] == "seedsequence_step_retry_file_v1"
        assert first.attrs["metadata_json"] == third.attrs["metadata_json"]

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
