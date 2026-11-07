"""Repeated passes preserve task order, optimizer state and resumability."""

import hashlib
import json
import shutil

import numpy as np
import pytest
import torch

from tabicl.nano_graph_u.model import NanoTabPFNModel
from tabicl.nano_graph_u.train import train_model

h5py = pytest.importorskip("h5py")
pytest.importorskip("schedulefree")


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _dump(path, *, copies=1):
    """Two distinguishable batches, optionally duplicated into a flat bank."""
    rng = np.random.default_rng(12)
    x = rng.normal(size=(4, 8, 2)).astype(np.float32)
    x[0, 0, 0], x[2, 0, 0] = 0, 1
    y = np.tile(np.arange(8, dtype=np.int64) % 2, (4, 1))
    metadata = {
        "format_version": 1,
        "steps": 2 * copies,
        "batch_size": 2,
        "rows": 8,
        "features": 2,
        "prior_config": {"graph_u_enabled": True},
    }
    with h5py.File(path, "w") as stream:
        stream.attrs["metadata_json"] = json.dumps(metadata)
        stream.attrs["committed_steps"] = metadata["steps"]
        stream.create_dataset("x", data=np.tile(x, (copies, 1, 1)))
        stream.create_dataset("y", data=np.tile(y, (copies, 1)))
        stream.create_dataset("train_sizes", data=np.full(4 * copies, 4))
    return path


def _train(dump, directory, **kwargs):
    options = {
        "steps": 2, "batch_size": 2, "device": "cpu", "learning_rate": 0.001,
        "seed": 42, "save_every": 1, "log_every": 1,
    }
    options.update(kwargs)
    result = train_model(dump, directory, **options)
    return torch.load(result, map_location="cpu", weights_only=True)


def _assert_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_equal(left, right)
    else:
        assert actual == expected


def _record_batches(monkeypatch):
    seen = []
    forward = NanoTabPFNModel.forward

    def record(self, src, train_test_split_index):
        x, support_y = src
        assert support_y.shape == (2, train_test_split_index)
        seen.append(int(x[0, 0, 0]))
        return forward(self, src, train_test_split_index)

    monkeypatch.setattr(NanoTabPFNModel, "forward", record)
    return seen


def test_epochs_repeat_same_batches_with_continuous_optimizer(tmp_path, monkeypatch):
    dump = _dump(tmp_path / "training.h5")
    before = hashlib.sha256(dump.read_bytes()).hexdigest()
    seen = _record_batches(monkeypatch)
    repeated = _train(dump, tmp_path / "repeated", epochs=3)
    assert seen == [0, 1, 0, 1, 0, 1]
    assert repeated["step"] == 6
    assert repeated["train_config"]["epochs"] == 3
    assert repeated["train_config"]["steps_per_epoch"] == 2
    assert repeated["dump_metadata"]["steps"] == 2
    assert hashlib.sha256(dump.read_bytes()).hexdigest() == before

    # A flat bank with exactly the same six batches must produce identical
    # model AND optimizer state, ruling out a hidden reset between epochs.
    flat = _train(_dump(tmp_path / "flat.h5", copies=3), tmp_path / "flat", steps=6)
    _assert_equal(repeated["model"], flat["model"])
    _assert_equal(repeated["optimizer"], flat["optimizer"])


@pytest.mark.parametrize("completed_step", [2, 3])
def test_resume_is_exact_at_epoch_boundary_and_mid_epoch(tmp_path, monkeypatch, completed_step):
    dump = _dump(tmp_path / "training.h5")
    complete_dir = tmp_path / "complete"
    uninterrupted = _train(dump, complete_dir, epochs=3)
    resume_dir = tmp_path / "resume"
    resume_dir.mkdir()
    shutil.copyfile(complete_dir / f"step-{completed_step}.pt", resume_dir / "latest.pt")
    seen = _record_batches(monkeypatch)
    resumed = _train(dump, resume_dir, epochs=3, resume=True)
    assert seen == [step % 2 for step in range(completed_step, 6)]
    assert resumed["step"] == 6
    _assert_equal(resumed, uninterrupted)


def test_default_one_epoch_checkpoint_schema_and_results_unchanged(tmp_path):
    dump = _dump(tmp_path / "training.h5")
    default = _train(dump, tmp_path / "default")
    explicit = _train(dump, tmp_path / "explicit", epochs=1)
    _assert_equal(default, explicit)
    assert default["step"] == 2
    assert set(default["train_config"]) == {
        "seed", "batch_size", "learning_rate", "optimizer", "weight_decay",
        "gradient_clip_norm", "loss", "precision", "dump_path", "dump_sha256",
    }


def test_completed_multi_epoch_resume_does_not_update_or_write(tmp_path, monkeypatch):
    dump = _dump(tmp_path / "training.h5")
    directory = tmp_path / "model"
    _train(dump, directory, epochs=3)
    before = (directory / "latest.pt").read_bytes()
    seen = _record_batches(monkeypatch)
    _train(dump, directory, epochs=3, resume=True)
    assert seen == []
    assert (directory / "latest.pt").read_bytes() == before


@pytest.mark.parametrize("epochs", [0, -1, True, 1.5, "3", None])
def test_invalid_epochs_rejected_before_any_files_are_created(tmp_path, epochs):
    with pytest.raises(ValueError, match="epochs must be a positive integer"):
        train_model(tmp_path / "missing.h5", tmp_path / "model", epochs=epochs, device="cpu")
    assert not (tmp_path / "model").exists()


def test_changed_multi_epoch_budget_cannot_silently_resume(tmp_path):
    dump = _dump(tmp_path / "training.h5")
    directory = tmp_path / "model"
    _train(dump, directory, epochs=3)
    before = (directory / "latest.pt").read_bytes()
    with pytest.raises(ValueError, match="training configuration differs"):
        _train(dump, directory, epochs=6, resume=True)
    assert (directory / "latest.pt").read_bytes() == before
