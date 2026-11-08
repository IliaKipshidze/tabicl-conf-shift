"""Whole-batch reordering preserves each table and its stored split."""

import hashlib
import json

import numpy as np
import pytest
import torch

from tabicl.nano_graph_u.data import NanoPriorDumpLoader

h5py = pytest.importorskip("h5py")


@pytest.fixture
def dump(tmp_path):
    path = tmp_path / "tasks.h5"
    metadata = {"format_version": 1, "steps": 4, "batch_size": 2, "rows": 8, "features": 2}
    x = np.arange(8 * 8 * 2, dtype=np.float32).reshape(8, 8, 2)
    y = np.tile(np.arange(8, dtype=np.int64) % 2, (8, 1))
    with h5py.File(path, "w") as file:
        file.attrs["metadata_json"] = json.dumps(metadata)
        file.attrs["committed_steps"] = 4
        file.create_dataset("x", data=x)
        file.create_dataset("y", data=y)
        file.create_dataset("train_sizes", data=np.repeat([2, 3, 4, 5], 2))
    return path


def test_reordered_batches_preserve_all_tasks_and_each_boundary(dump):
    before = hashlib.sha256(dump.read_bytes()).hexdigest()
    loader = NanoPriorDumpLoader(dump, batch_size=2)
    sequential = list(loader)
    assert [batch["train_test_split_index"] for batch in sequential] == [2, 3, 4, 5]
    reordered = list(loader.iter_batches(np.array([3, 0, 2, 1], dtype=np.int64)))
    for batch, original_index in zip(reordered, [3, 0, 2, 1]):
        original = sequential[original_index]
        assert torch.equal(batch["x"], original["x"])
        assert torch.equal(batch["y"], original["y"])
        assert batch["train_test_split_index"] == original["train_test_split_index"]
    assert hashlib.sha256(dump.read_bytes()).hexdigest() == before


def test_resumed_subset_respects_configured_range(dump):
    loader = NanoPriorDumpLoader(dump, batch_size=2, start_step=1, num_steps=4)
    assert len(loader) == 3
    assert [batch["train_test_split_index"] for batch in loader] == [3, 4, 5]
    assert [batch["train_test_split_index"] for batch in loader.iter_batches([3, 1])] == [5, 3]
    assert list(loader.iter_batches([])) == []
    with pytest.raises(ValueError, match="outside"):
        list(loader.iter_batches([0]))


@pytest.mark.parametrize("indices,error", [
    ([0, 0], ValueError), ([4], ValueError), ([-1], ValueError),
    ([True], TypeError), ([0.5], TypeError), (["1"], TypeError), ([np.bool_(False)], TypeError),
])
def test_invalid_order_rejected_before_opening_the_file(dump, monkeypatch, indices, error):
    loader = NanoPriorDumpLoader(dump, batch_size=2)

    def unexpected_open(*args, **kwargs):
        pytest.fail("invalid batch order opened the HDF5 file")

    monkeypatch.setattr(h5py, "File", unexpected_open)
    with pytest.raises(error, match="batch ind"):
        list(loader.iter_batches(indices))


def test_reordered_batches_still_validate_shared_boundaries(dump):
    loader = NanoPriorDumpLoader(dump, batch_size=2)
    with h5py.File(dump, "r+") as file:
        file["train_sizes"][6] = 4
    with pytest.raises(ValueError, match="same support/query split"):
        list(loader.iter_batches([3]))


def test_each_iteration_opens_the_dump_once(dump, monkeypatch):
    loader = NanoPriorDumpLoader(dump, batch_size=2)
    opened = []
    original = h5py.File

    def record(*args, **kwargs):
        opened.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(h5py, "File", record)
    assert len(list(loader.iter_batches([3, 1, 0, 2]))) == 4
    assert len(opened) == 1
    assert opened[0][0][1] == "r"
