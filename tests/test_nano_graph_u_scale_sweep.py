"""Frozen-model scale sweeps preserve task settings and reuse IID results."""

from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from tabicl.nano_graph_u import scale_sweep as sweep
from tabicl.nano_graph_u._settings import experiment_prior_config
from tabicl.nano_graph_u.data import _metadata
from tabicl.prior.graph_lib._config import GRAPH_U_SAMPLING_DEFAULTS


def _settings():
    return {**GRAPH_U_SAMPLING_DEFAULTS, "graph_u_shift_mode": "sampled"}


def _write_bank(spec, *, committed=100, changed_mean=False):
    h5py = pytest.importorskip("h5py")
    settings = dict(spec["sampling_config"])
    if changed_mean:
        settings["graph_u_query_scale_mean"] += 0.01
    metadata = _metadata(
        experiment_prior_config("shift", spec["graph_u_mode"], settings),
        steps=100, batch_size=32, seed=spec["seed"], rows=150, features=5,
        min_train_size=0.3, max_train_size=0.9, n_jobs=1,
    )
    spec["dump"].parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(spec["dump"], "w") as stream:
        stream.attrs["metadata_json"] = json.dumps(metadata)
        stream.attrs["committed_steps"] = committed
        stream.create_dataset("x", (3200, 150, 5), dtype="f4")
        stream.create_dataset("y", (3200, 150), dtype="i8")
        stream.create_dataset("train_sizes", (3200,), dtype="i4")


def test_bank_plan_has_40_distinct_banks_with_only_scale_parameters_changed(tmp_path):
    settings = _settings()
    before = deepcopy(settings)
    specs = sweep.bank_specs(tmp_path, settings)
    assert len(specs) == len({spec["dump"] for spec in specs}) == 40
    assert Counter(spec["graph_u_mode"] for spec in specs) == dict(confounding=20, covariate=20)
    assert set(Counter(spec["scale_mean"] for spec in specs).values()) == {10}
    assert set(Counter(spec["repetition"] for spec in specs).values()) == {8}
    for spec in specs:
        mean, profile = spec["scale_mean"], spec["sampling_config"]
        assert profile["graph_u_query_scale_mean"] == mean
        assert profile["graph_u_query_scale_min"] == pytest.approx(mean - 0.2)
        assert profile["graph_u_query_scale_max"] == pytest.approx(mean + 0.2)
        for key, value in settings.items():
            if key not in {"graph_u_query_scale_mean", "graph_u_query_scale_min", "graph_u_query_scale_max"}:
                assert profile[key] == value
        assert spec["seed"] == 4000 + spec["repetition"]
        assert spec["steps"] == 100
    assert settings == before


@pytest.mark.parametrize("means,width", [([], 0.2), ([1.3, 1.3], 0.2), ([float("nan")], 0.2),
                                          ([1.3], 0), ([1.3], -0.2), ([0.1], 0.2)])
def test_invalid_sweep_ranges_fail_before_output_creation(tmp_path, means, width):
    output = tmp_path / "sweep"
    with pytest.raises(ValueError):
        sweep.bank_specs(output, _settings(), means, width)
    assert not output.exists()


def test_complete_and_partial_bank_validation_checks_actual_batch_counter(tmp_path):
    spec = sweep.bank_specs(tmp_path, _settings())[0]
    _write_bank(spec, committed=19)
    assert sweep._validate_bank(spec, complete=False).committed_steps == 19
    with pytest.raises(ValueError, match="[Ii]ncomplete"):
        sweep._validate_bank(spec)
    _write_bank(spec)
    assert sweep._validate_bank(spec).committed_steps == 100
    _write_bank(spec, changed_mean=True)
    with pytest.raises(ValueError):
        sweep._validate_bank(spec)


@pytest.mark.parametrize("committed", [None, 19, 100])
def test_bank_generation_starts_resumes_or_skips_without_replacing_complete_data(
    tmp_path, monkeypatch, committed
):
    spec = sweep.bank_specs(tmp_path, _settings())[0]
    if committed is not None:
        _write_bank(spec, committed=committed)
    calls = []

    def generate(path, **kwargs):
        assert path == spec["dump"]
        assert kwargs["steps"] == 100 and kwargs["batch_size"] == 32
        assert kwargs["rows"] == 150 and kwargs["features"] == 5
        assert kwargs["seed"] == 4001 and kwargs["workers"] == 4
        assert kwargs["resume"] is (committed is not None)
        assert kwargs["prior_config"].graph_u_query_scale_mean == spec["scale_mean"]
        calls.append(path)
        _write_bank(spec)

    monkeypatch.setattr(sweep, "generate_dump", generate)
    sweep._generate_banks([spec], workers=4)
    assert len(calls) == (0 if committed == 100 else 1)
    assert sweep._validate_bank(spec).committed_steps == 100


def _rows():
    rows = []
    for train_mode in ("confounding", "covariate"):
        for train_condition in ("identity", "shift"):
            for test_mode in ("confounding", "covariate"):
                for mean in (None, 1.3, 1.7, 1.9, 2.2):
                    for repetition in range(1, 6):
                        value = 0.6 + repetition * 0.01
                        rows.append({
                            "training_graph_u_mode": train_mode, "training_condition": train_condition,
                            "test_graph_u_mode": test_mode,
                            "test_condition": "identity" if mean is None else "shift",
                            "scale_mean": mean, "scale_std": None if mean is None else 0.1,
                            "scale_min": None if mean is None else mean - 0.2,
                            "scale_max": None if mean is None else mean + 0.2,
                            "repetition": repetition,
                            **{metric: value for metric in ("accuracy", "balanced_accuracy", "macro_f1",
                                                           "roc_auc", "nll", "brier")},
                        })
    return rows


def test_summary_includes_iid_once_per_cell_and_sample_std_across_five_repetitions():
    rows = _rows()
    assert len(rows) == 200
    cells = sweep._summarize(rows)
    assert len(cells) == 40
    assert sum(cell["test_condition"] == "identity" for cell in cells) == 8
    for cell in cells:
        assert cell["repetition_count"] == 5
        for metric in sweep.METRIC_NAMES:
            assert cell[f"{metric}_mean"] == pytest.approx(0.63)
            assert cell[f"{metric}_std"] == pytest.approx(0.01 * math.sqrt(2.5))


@pytest.mark.parametrize("problem", ["missing", "duplicate"])
def test_summary_rejects_missing_or_duplicate_repetitions(problem):
    rows = _rows()
    rows = rows[1:] if problem == "missing" else rows + [dict(rows[0])]
    with pytest.raises(ValueError):
        sweep._summarize(rows)


def test_whole_sweep_runs_only_new_evaluations_then_reuses_completed_outputs(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "sweep"
    settings, models, iid_rows = _settings(), {}, []
    source.mkdir()
    for mode in sweep.MODES:
        for repetition in range(1, 6):
            for condition in ("identity", "shift"):
                checkpoint = source / f"{mode}_{repetition}_{condition}.pt"
                checkpoint.write_bytes(f"frozen/{mode}/{repetition}/{condition}".encode())
                model = {
                    "training_graph_u_mode": mode, "training_condition": condition,
                    "repetition": repetition, "model_seed": 41 + repetition,
                    "learning_rate": 0.001, "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sweep._digest(checkpoint, None),
                }
                models[(mode, repetition, condition)] = model
                for test_mode in sweep.MODES:
                    report = {
                        "checkpoint": str(checkpoint), "checkpoint_sha256": model["checkpoint_sha256"],
                        "dump": str(source / f"iid_{test_mode}_{repetition}.h5"),
                        "dump_sha256": "a" * 64, "dump_metadata": {"seed": 2000 + repetition},
                        "task_count": 3200, "query_row_count": 240000,
                        "metrics": {metric: 0.6 + repetition * 0.01 for metric in sweep.METRIC_NAMES},
                    }
                    iid_rows.append(sweep._row(report, model, test_mode, None,
                                               source / "existing_iid_report.json", "reused_iid"))
    source_bytes = {path: path.read_bytes() for path in source.iterdir()}
    monkeypatch.setattr(sweep, "load_source", lambda path: (settings, models, iid_rows, {"frozen": "b" * 64}))
    generated, evaluated = [], []
    specs = sweep.bank_specs(output, settings)
    by_path = {spec["dump"]: spec for spec in specs}

    def generate(path, **kwargs):
        assert not kwargs["resume"]
        assert kwargs["workers"] == 2
        generated.append(path)
        _write_bank(by_path[path])

    def evaluate(checkpoint, dump, **kwargs):
        assert kwargs == {"batch_size": 32, "device": "cpu"}
        assert output / "data" in dump.parents
        evaluated.append((checkpoint, dump))
        metadata = sweep._validate_bank(by_path[dump]).metadata
        return {
            "format": "nano_graph_u_evaluation_v1", "checkpoint": checkpoint,
            "checkpoint_sha256": sweep._digest(Path(checkpoint), None), "checkpoint_step": 2500,
            "dump": str(dump), "dump_sha256": sweep._digest(dump, None), "dump_metadata": metadata,
            "task_count": 3200, "query_row_count": 240000,
            "metrics": {metric: 0.7 for metric in sweep.METRIC_NAMES},
            "metric_task_counts": {metric: 3200 for metric in sweep.METRIC_NAMES},
        }

    monkeypatch.setattr(sweep, "generate_dump", generate)
    monkeypatch.setattr(sweep, "evaluate_checkpoint", evaluate)
    result = sweep.run_sweep(source, output, workers=2, device="cpu")
    assert len(generated) == 40 and len(evaluated) == 160
    assert set(Counter(dump for _, dump in evaluated).values()) == {4}
    assert result["new_evaluation_count"] == 160 and result["reused_iid_count"] == 40
    assert len(result["per_repetition"]) == 200 and len(result["cells"]) == 40
    assert sum(row["result_origin"] == "reused_iid" for row in result["per_repetition"]) == 40
    before = {path: path.read_bytes() for path in (output / "results").rglob("*.json")}
    assert sweep.run_sweep(source, output, workers=2, device="cpu") == result
    assert len(generated) == 40 and len(evaluated) == 160
    assert all(path.read_bytes() == data for path, data in before.items())
    assert all(path.read_bytes() == data for path, data in source_bytes.items())
    with pytest.raises(FileExistsError):
        sweep.run_sweep(source, output, scale_half_width=0.15, workers=2, device="cpu")
    assert len(generated) == 40 and len(evaluated) == 160
