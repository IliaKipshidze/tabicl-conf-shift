"""Manual sampled-experiment stages map tasks and resume complete artifacts."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest

from tabicl.nano_graph_u import sampled
from tabicl.nano_graph_u._settings import experiment_prior_config
from tabicl.nano_graph_u.data import _metadata
from tabicl.nano_graph_u.tune import LEARNING_RATES
from tabicl.prior.graph_lib._config import GRAPH_U_SAMPLING_DEFAULTS


def _settings():
    return {**GRAPH_U_SAMPLING_DEFAULTS, "graph_u_shift_mode": "sampled"}


def _write_bank(spec, settings, *, committed=None):
    h5py = pytest.importorskip("h5py")
    path = spec["dump"]
    metadata = _metadata(
        experiment_prior_config(spec["condition"], spec["graph_u_mode"], settings),
        steps=spec["steps"], batch_size=32, seed=spec["seed"], rows=150, features=5,
        min_train_size=0.3, max_train_size=0.9, n_jobs=1,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    count = spec["steps"] * 32
    with h5py.File(path, "w") as stream:
        stream.attrs["metadata_json"] = json.dumps(metadata)
        stream.attrs["committed_steps"] = spec["steps"] if committed is None else committed
        stream.create_dataset("x", (count, 150, 5), dtype="f4")
        stream.create_dataset("y", (count, 150), dtype="i8")
        stream.create_dataset("train_sizes", (count,), dtype="i4")
    return metadata


def test_all_stage_mappings_preserve_design_and_reuse_each_test_bank(tmp_path):
    root, settings = tmp_path / "experiment", _settings()
    banks = [sampled.bank_spec(root, task) for task in range(1, 45)]
    assert len({bank["dump"] for bank in banks}) == 44
    assert Counter(bank["split"] for bank in banks) == dict(train=20, test=20, validation=4)
    for mode in sampled.MODES:
        group = [bank for bank in banks if bank["graph_u_mode"] == mode]
        assert len(group) == 22
        assert Counter(bank["condition"] for bank in group) == dict(identity=11, shift=11)
        for bank in group:
            assert root / "data" / mode in bank["dump"].parents
            if bank["split"] == "validation":
                assert bank["seed"] == 3001 and bank["repetition"] is None
            else:
                base = 1000 if bank["split"] == "train" else 2000
                assert bank["seed"] == base + bank["repetition"]
            assert bank["steps"] == (2500 if bank["split"] == "train" else 100)

    trials = [sampled.trial_for_task(root, task, settings) for task in range(1, 201)]
    assert Counter(mode for mode, _ in trials) == dict(confounding=100, covariate=100)
    assert len({trial["checkpoint_dir"] for _, trial in trials}) == 200
    assert len({trial["train_dump"] for _, trial in trials}) == 20
    assert len({trial["validation_dump"] for _, trial in trials}) == 4
    for mode in sampled.MODES:
        for repetition in range(1, 6):
            for condition in ("identity", "shift"):
                group = [trial for current_mode, trial in trials if current_mode == mode
                         and trial["repetition"] == repetition and trial["condition"] == condition]
                assert tuple(trial["learning_rate"] for trial in group) == LEARNING_RATES
                assert {trial["model_seed"] for trial in group} == {41 + repetition}
                assert all(trial["sampling_config"] == settings for trial in group)

    tests = [sampled.test_for_task(root, task, settings) for task in range(1, 81)]
    assert len({test["result_path"] for test in tests}) == 80
    assert len({test["checkpoint"] for test in tests}) == 20
    assert len({test["test_dump"] for test in tests}) == 20
    assert set(Counter(test["test_dump"] for test in tests).values()) == {4}
    assert set(Counter(test["checkpoint"] for test in tests).values()) == {4}
    assert {Path(test["test_dump"]) for test in tests} == {
        bank["dump"].resolve() for bank in banks if bank["split"] == "test"
    }
    assert not root.exists()


def test_manifest_is_immutable_and_reads_never_initialize_a_directory(tmp_path):
    root = tmp_path / "experiment"
    with pytest.raises(FileNotFoundError):
        sampled.load_experiment(root)
    assert not root.exists()
    settings = sampled.initialize_experiment(root, {"graph_u_query_scale_std": 0.08})
    assert settings["graph_u_shift_mode"] == "sampled"
    assert settings["graph_u_query_scale_std"] == 0.08
    manifest = root / "experiment.json"
    before = manifest.read_bytes()
    assert sampled.load_experiment(root) == settings
    assert sampled.initialize_experiment(root, {}) == settings
    assert sampled.initialize_experiment(root, {"graph_u_query_scale_std": 0.08}) == settings
    with pytest.raises(FileExistsError):
        sampled.initialize_experiment(root, {"graph_u_query_scale_std": 0.06})
    assert manifest.read_bytes() == before


@pytest.mark.parametrize("stage,count", [("generate", 44), ("tune", 200), ("select", 2), ("test", 80)])
def test_invalid_array_ids_cannot_create_outputs(tmp_path, stage, count):
    root = tmp_path / "experiment"
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    for task in (0, count + 1):
        with pytest.raises(ValueError, match="task-id"):
            sampled.main([stage, "--root", str(root), "--task-id", str(task)])
    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()} == before
    assert not root.exists()


@pytest.mark.parametrize("task", [1, 2, 21, 22, 23, 24, 43, 44])
def test_complete_bank_is_validated_and_skipped_without_generator_call(tmp_path, monkeypatch, task):
    root, settings = tmp_path / "experiment", _settings()
    spec = sampled.bank_spec(root, task)
    _write_bank(spec, settings)

    def unexpected_generation(*_args, **_kwargs):
        raise AssertionError("An already complete bank must not be regenerated")

    monkeypatch.setattr(sampled, "generate_dump", unexpected_generation)
    before = spec["dump"].read_bytes()
    sampled.generate_task(root, task, settings, workers=8)
    assert spec["dump"].read_bytes() == before


@pytest.mark.parametrize("already_exists", [False, True])
def test_generation_starts_or_resumes_the_expected_bank(tmp_path, monkeypatch, already_exists):
    root, settings = tmp_path / "experiment", _settings()
    spec = sampled.bank_spec(root, 24)
    if already_exists:
        _write_bank(spec, settings, committed=19)
    calls = []

    def generate(path, **kwargs):
        assert path == spec["dump"]
        assert kwargs["resume"] is already_exists
        assert kwargs["steps"] == 2500 and kwargs["batch_size"] == 32
        assert kwargs["rows"] == 150 and kwargs["features"] == 5
        assert kwargs["seed"] == 1001 and kwargs["workers"] == 8
        assert kwargs["n_jobs"] == 1
        assert kwargs["prior_config"].graph_u_mode == "covariate"
        assert kwargs["prior_config"].graph_u_shift_mode == "sampled"
        calls.append(path)
        _write_bank(spec, settings)

    monkeypatch.setattr(sampled, "generate_dump", generate)
    sampled.generate_task(root, 24, settings, workers=8)
    assert calls == [spec["dump"]]
    assert sampled._bank_status(spec, settings).committed_steps == 2500


def test_read_only_data_check_reports_global_failed_indices(tmp_path, capsys):
    root, settings = tmp_path / "experiment", _settings()
    for task in range(1, 45):
        if task != 24:
            _write_bank(sampled.bank_spec(root, task), settings, committed=0 if task == 3 else None)
    before = {path.relative_to(root) for path in root.rglob("*")}
    assert sampled.check_stage(root, "data", settings) is False
    output = capsys.readouterr().out
    assert "CHECK_FAILED stage=data array_tasks=3,24" in output
    assert {path.relative_to(root) for path in root.rglob("*")} == before
    _write_bank(sampled.bank_spec(root, 3), settings)
    _write_bank(sampled.bank_spec(root, 24), settings)
    assert sampled.check_stage(root, "data", settings) is True
    assert "STAGE_READY stage=data completed=44/44" in capsys.readouterr().out


def test_check_cli_uses_required_stage_without_writing_outputs(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    settings = sampled.initialize_experiment(root, {})
    received = []

    def check(path, stage, config):
        received.append((path, stage, config))
        return stage == "data"

    monkeypatch.setattr(sampled, "check_stage", check)
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    sampled.main(["check", "--root", str(root), "--stage", "data"])
    with pytest.raises(SystemExit) as error:
        sampled.main(["check", "--root", str(root), "--stage", "tune"])
    assert error.value.code == 1
    assert received == [(root.resolve(), "data", settings), (root.resolve(), "tune", settings)]
    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()} == before
