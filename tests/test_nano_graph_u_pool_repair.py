"""Repair only provenance for the exact finite-pool validation/retry change."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tabicl.nano_graph_u import pool_experiment as experiment
from tabicl.nano_graph_u import pool_repair as repair
from tabicl.nano_graph_u.data import NanoPriorDumpLoader
from tabicl.nano_graph_u.sampled import _lock


def _bank(root, task_id=1, *, fingerprint="legacy", committed=1):
    h5py = pytest.importorskip("h5py")
    spec = experiment.bank_spec(root, task_id)
    path = Path(spec["dump"])
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(repair._expected(spec), generator_source_sha256=fingerprint)
    count = spec["steps"] * 32
    with h5py.File(path, "w") as stream:
        stream.attrs["metadata_json"] = json.dumps(
            metadata, sort_keys=True, allow_nan=False
        )
        stream.attrs["committed_steps"] = committed
        x = stream.create_dataset("x", (count, 150, 5), chunks=(32, 150, 5), dtype="f4")
        y = stream.create_dataset("y", (count, 150), chunks=(32, 150), dtype="i8")
        splits = stream.create_dataset(
            "train_sizes", (count,), chunks=(32,), dtype="i4"
        )
        x[:32] = np.arange(32 * 150 * 5, dtype="f4").reshape(32, 150, 5)
        y[:32] = np.arange(32 * 150, dtype="i8").reshape(32, 150) % 2
        splits[:32] = 100
    return spec, metadata


def _contents(path):
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "r") as stream:
        return {
            key: stream[key][:32].tobytes() for key in ("x", "y", "train_sizes")
        }, int(stream.attrs["committed_steps"])


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(repair, "_approved_legacy_hashes", lambda: {"legacy"})
    return experiment._initialize(tmp_path / "pool repair")


def test_partial_and_complete_banks_retain_exact_data_and_original_audit(root):
    h5py = pytest.importorskip("h5py")
    specs = [_bank(root, 1), _bank(root, 4, committed=100)]
    before = {spec["dump"]: _contents(spec["dump"]) for spec, _ in specs}
    counts = repair.repair_banks(root)
    assert counts == {"repaired": 2, "already_current": 0, "missing": 64}
    for spec, previous in specs:
        assert _contents(spec["dump"]) == before[spec["dump"]]
        metadata = NanoPriorDumpLoader(spec["dump"]).metadata
        assert metadata == repair._expected(spec)
        with h5py.File(spec["dump"], "r") as stream:
            audit = json.loads(stream.attrs[repair._AUDIT_ATTRIBUTE])
            assert audit["original_metadata"] == previous
            assert json.loads(audit["original_metadata_json"]) == previous
            assert audit["committed_steps"] == before[spec["dump"]][1]
            assert audit["previous_source_sha256"] == "legacy"
            assert audit["new_source_sha256"] == metadata["generator_source_sha256"]
    before_files = {spec["dump"]: Path(spec["dump"]).read_bytes() for spec, _ in specs}
    assert repair.repair_banks(root) == {
        "repaired": 0,
        "already_current": 2,
        "missing": 64,
    }
    assert all(Path(path).read_bytes() == data for path, data in before_files.items())


@pytest.mark.parametrize(
    "corruption", ["fingerprint", "seed", "prior", "committed", "shape"]
)
def test_unrecognized_or_invalid_banks_fail_without_modifying_them(root, corruption):
    h5py = pytest.importorskip("h5py")
    spec, metadata = _bank(root)
    path = Path(spec["dump"])
    with h5py.File(path, "r+") as stream:
        if corruption == "fingerprint":
            metadata["generator_source_sha256"] = "unrecognized"
        elif corruption == "seed":
            metadata["seed"] += 1
        elif corruption == "prior":
            metadata["prior_config"]["graph_u_pool_peak_percentile"] = 25
        elif corruption == "committed":
            stream.attrs["committed_steps"] = spec["steps"] + 1
        else:
            del stream["x"]
            stream.create_dataset("x", (1, 150, 5), dtype="f4")
        stream.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        repair.repair_banks(root)
    assert path.read_bytes() == before


@pytest.mark.parametrize("folder", ["checkpoints", "winners"])
def test_migration_refuses_banks_already_bound_to_models(root, folder):
    spec, _ = _bank(root)
    model = root / folder / "rep_1/model.pt"
    model.parent.mkdir(parents=True)
    model.touch()
    before = Path(spec["dump"]).read_bytes()
    with pytest.raises(ValueError, match="before training"):
        repair.repair_banks(root)
    assert Path(spec["dump"]).read_bytes() == before


def test_active_generation_lock_prevents_repair(root):
    spec, _ = _bank(root)
    before = Path(spec["dump"]).read_bytes()
    with (
        _lock(root / "locks/bank_1.lock"),
        pytest.raises(RuntimeError, match="Another job"),
    ):
        repair.repair_banks(root)
    assert Path(spec["dump"]).read_bytes() == before


def test_audit_flushed_before_interruption_allows_safe_retry(root, monkeypatch):
    h5py = pytest.importorskip("h5py")
    spec, metadata = _bank(root)
    path = Path(spec["dump"])
    expected = repair._expected(spec)
    with h5py.File(path, "r+") as stream:
        audit = {
            "format": "nano_graph_u_pool_nonfinite_repair_v1",
            "original_metadata": metadata,
            "original_metadata_json": stream.attrs["metadata_json"],
            "committed_steps": 1,
            "previous_source_sha256": "legacy",
            "new_source_sha256": expected["generator_source_sha256"],
        }
        stream.attrs[repair._AUDIT_ATTRIBUTE] = json.dumps(
            audit, sort_keys=True, allow_nan=False
        )
    before = _contents(path)
    assert repair.repair_banks(root)["repaired"] == 1
    assert _contents(path) == before
    with h5py.File(path, "r") as stream:
        assert json.loads(stream.attrs[repair._AUDIT_ATTRIBUTE]) == audit


def test_conflicting_audit_is_not_overwritten(root):
    h5py = pytest.importorskip("h5py")
    spec, _ = _bank(root)
    path = Path(spec["dump"])
    with h5py.File(path, "r+") as stream:
        stream.attrs[repair._AUDIT_ATTRIBUTE] = '{"different":"repair"}'
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Conflicting"):
        repair.repair_banks(root)
    assert path.read_bytes() == before


def test_legacy_fingerprints_reconstruct_only_previous_two_source_files():
    package = Path(repair.__file__).resolve().parents[1]
    repository = package.parents[1]
    expected = hashlib.sha256()
    for path in sorted(
        [package / "nano_graph_u/data.py", *package.joinpath("prior").rglob("*.py")],
        key=lambda path: path.as_posix(),
    ):
        relative = path.relative_to(package).as_posix()
        contents = path.read_bytes()
        if relative in repair._PATCHED_SOURCES:
            contents = subprocess.run(
                ["git", "show", f"{repair._BASE_COMMIT}:src/tabicl/{relative}"],
                cwd=repository,
                check=True,
                capture_output=True,
            ).stdout.replace(b"\r\n", b"\n")
        expected.update(relative.encode() + b"\0" + contents + b"\0")
    hashes = repair._approved_legacy_hashes()
    assert expected.hexdigest() in hashes
    assert len(hashes) == 4


def test_actual_previous_source_fingerprint_is_accepted_for_metadata_only_repair(
    tmp_path,
):
    root = experiment._initialize(tmp_path / "actual legacy provenance")
    fingerprint = min(repair._approved_legacy_hashes())
    spec, _ = _bank(root, fingerprint=fingerprint)
    before = _contents(spec["dump"])
    assert repair.repair_banks(root) == {
        "repaired": 1,
        "already_current": 0,
        "missing": 65,
    }
    assert _contents(spec["dump"]) == before
    assert NanoPriorDumpLoader(spec["dump"]).metadata == repair._expected(spec)


def test_unapproved_future_patch_cannot_enable_old_fingerprints(monkeypatch):
    monkeypatch.setattr(
        repair, "_PATCHED_SOURCES", {"prior/graph_lib/_pool.py": "different"}
    )
    with pytest.raises(ValueError, match="exact nonfinite-U retry fix"):
        repair._approved_legacy_hashes()


def test_any_other_generator_source_change_invalidates_the_previous_fingerprints(
    monkeypatch,
):
    before = repair._approved_legacy_hashes()
    source = Path(repair.__file__).resolve().parents[1] / "prior/graph_lib/_config.py"
    actual_read = Path.read_bytes

    def changed_read(path):
        contents = actual_read(path)
        return contents + b"\n# unrelated change\n" if path == source else contents

    monkeypatch.setattr(Path, "read_bytes", changed_read)
    assert before.isdisjoint(repair._approved_legacy_hashes())
