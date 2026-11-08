"""Whole-batch shuffling is reproducible, split-preserving and resumable."""

import hashlib
import json
import shutil

import numpy as np
import pytest
import torch
from test_nano_graph_u_epochs import _assert_equal

from tabicl.nano_graph_u import train
from tabicl.nano_graph_u.model import NanoTabPFNModel

h5py = pytest.importorskip("h5py")
pytest.importorskip("schedulefree")


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _dump(path):
    # Each pair of tasks has a different batch ID and shared support boundary.
    x = np.random.default_rng(12).normal(size=(10, 8, 2)).astype(np.float32)
    for batch in range(5):
        x[2 * batch : 2 * batch + 2, 0, 0] = batch
    metadata = {"format_version": 1, "steps": 5, "batch_size": 2,
                "rows": 8, "features": 2, "prior_config": {"graph_u_enabled": True}}
    with h5py.File(path, "w") as stream:
        stream.attrs["metadata_json"] = json.dumps(metadata)
        stream.attrs["committed_steps"] = 5
        stream.create_dataset("x", data=x)
        stream.create_dataset("y", data=np.tile(np.arange(8) % 2, (10, 1)))
        stream.create_dataset("train_sizes", data=np.repeat(np.arange(2, 7), 2))
    return path


def _train(dump, directory, **kwargs):
    options = {"steps": 5, "epochs": 3, "batch_size": 2, "device": "cpu", "seed": 42,
               "learning_rate": 0.001, "save_every": 1, "log_every": 5, "shuffle_batches": True}
    options.update(kwargs)
    path = train.train_model(dump, directory, **options)
    return torch.load(path, map_location="cpu", weights_only=True)


def _record(monkeypatch):
    observed = []
    forward = NanoTabPFNModel.forward

    def record(self, src, train_test_split_index):
        x, y = src
        batch = int(x[0, 0, 0])
        assert x[1, 0, 0] == batch
        assert train_test_split_index == batch + 2
        assert y.shape == (2, batch + 2)
        observed.append(batch)
        return forward(self, src, train_test_split_index)

    monkeypatch.setattr(NanoTabPFNModel, "forward", record)
    return observed


def test_epoch_orders_cover_every_batch_and_do_not_consume_global_rng():
    np.random.seed(71)
    before = np.random.get_state()
    orders = [np.asarray(train._epoch_batch_order(2500, 42, epoch)) for epoch in range(10)]
    after = np.random.get_state()
    _assert_equal(before[0], after[0])
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    for epoch, order in enumerate(orders):
        np.testing.assert_array_equal(np.sort(order), np.arange(2500))
        np.testing.assert_array_equal(order, train._epoch_batch_order(2500, 42, epoch))
        assert not np.array_equal(order, train._epoch_batch_order(2500, 43, epoch))
    assert len({tuple(order) for order in orders}) == 10


def test_shuffled_training_preserves_batches_and_matches_order_across_lrs(tmp_path, monkeypatch):
    dump = _dump(tmp_path / "train.h5")
    digest = hashlib.sha256(dump.read_bytes()).hexdigest()
    observed = _record(monkeypatch)
    sequences = []
    for lr in (0.0007, 0.001, 0.002):
        observed.clear()
        payload = _train(dump, tmp_path / str(lr), learning_rate=lr)
        sequences.append(observed.copy())
        assert payload["step"] == 15
        assert payload["train_config"]["shuffle_batches"] is True
        assert payload["train_config"]["batch_order_scheme"] == "seedsequence_model_epoch_v1"
        for epoch in range(3):
            assert sorted(observed[5 * epoch : 5 * (epoch + 1)]) == list(range(5))
        assert hashlib.sha256(dump.read_bytes()).hexdigest() == digest
    assert sequences[0] == sequences[1] == sequences[2]
    assert sequences[0][:5] != sequences[0][5:10]
    # A longer requested budget must preserve the actual first three epochs.
    observed.clear()
    _train(dump, tmp_path / "six_epochs", epochs=6)
    assert observed[:15] == sequences[0]
    prefix = torch.load(tmp_path / "six_epochs/step-15.pt", map_location="cpu", weights_only=True)
    reference = torch.load(tmp_path / "0.001/latest.pt", map_location="cpu", weights_only=True)
    _assert_equal(prefix["model"], reference["model"])
    _assert_equal(prefix["optimizer"], reference["optimizer"])


@pytest.mark.parametrize("completed_step", [5, 7])
def test_shuffled_resume_is_exact_at_boundary_and_mid_epoch(tmp_path, monkeypatch, completed_step):
    dump = _dump(tmp_path / "train.h5")
    seen = _record(monkeypatch)
    uninterrupted = _train(dump, tmp_path / "complete")
    full_order = seen.copy()
    resumed_dir = tmp_path / "resumed"
    resumed_dir.mkdir()
    shutil.copyfile(tmp_path / "complete" / f"step-{completed_step}.pt", resumed_dir / "latest.pt")
    seen.clear()
    resumed = _train(dump, resumed_dir, resume=True)
    assert seen == full_order[completed_step:]
    _assert_equal(resumed, uninterrupted)


@pytest.mark.parametrize("initial_shuffle", [False, True])
def test_shuffle_mode_cannot_be_changed_while_resuming(tmp_path, initial_shuffle):
    dump = _dump(tmp_path / "train.h5")
    directory = tmp_path / "model"
    payload = _train(dump, directory, shuffle_batches=initial_shuffle)
    if not initial_shuffle:
        assert "shuffle_batches" not in payload["train_config"]
        assert "batch_order_scheme" not in payload["train_config"]
    before = (directory / "latest.pt").read_bytes()
    with pytest.raises(ValueError, match="training configuration differs"):
        _train(dump, directory, shuffle_batches=not initial_shuffle, resume=True)
    assert (directory / "latest.pt").read_bytes() == before


@pytest.mark.parametrize("value", [1, "true", None])
def test_shuffle_option_requires_a_boolean_before_writing(tmp_path, value):
    with pytest.raises(TypeError, match="shuffle_batches must be a boolean"):
        train.train_model(tmp_path / "missing.h5", tmp_path / "output", device="cpu", shuffle_batches=value)
    assert not (tmp_path / "output").exists()
