"""Location sweeps reuse frozen models and bind cached results to their profiles."""

import csv
import json
import math
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from tabicl.nano_graph_u import location_sweep as sweep
from tabicl.nano_graph_u import scale_sweep as shared
from tabicl.nano_graph_u._settings import experiment_prior_config
from tabicl.nano_graph_u.data import _metadata
from tabicl.prior.graph_lib._config import GRAPH_U_SAMPLING_DEFAULTS


def _write_bank(spec, settings=None):
    h5py = pytest.importorskip("h5py")
    metadata = _metadata(
        experiment_prior_config("shift", spec["graph_u_mode"], settings or spec["sampling_config"]),
        steps=100, batch_size=32, seed=spec["seed"], rows=150, features=5,
        min_train_size=0.3, max_train_size=0.9, n_jobs=1,
    )
    spec["dump"].parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(spec["dump"], "w") as stream:
        stream.attrs["metadata_json"] = json.dumps(metadata)
        stream.attrs["committed_steps"] = 100
        stream.create_dataset("x", (3200, 150, 5), dtype="f4")
        stream.create_dataset("y", (3200, 150), dtype="i8")
        stream.create_dataset("train_sizes", (3200,), dtype="i4")


def _report(checkpoint, spec):
    return {
        "format": "nano_graph_u_evaluation_v1", "checkpoint": checkpoint,
        "checkpoint_sha256": shared._digest(Path(checkpoint), None), "checkpoint_step": 2500,
        "dump": str(spec["dump"]), "dump_sha256": shared._digest(spec["dump"], None),
        "dump_metadata": shared._validate_bank(spec).metadata,
        "task_count": 3200, "query_row_count": 240000,
        "metrics": {metric: 0.7 for metric in shared.METRIC_NAMES},
        "metric_task_counts": {metric: 3200 for metric in shared.METRIC_NAMES},
    }


@pytest.fixture
def frozen_source(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    settings = {
        **GRAPH_U_SAMPLING_DEFAULTS, "graph_u_shift_mode": "sampled",
        "graph_u_query_scale_mean": 1.6, "graph_u_query_scale_std": 0.12,
        "graph_u_query_scale_min": 1.2, "graph_u_query_scale_max": 1.9,
    }
    models, iid_rows = {}, []
    for mode in sweep.MODES:
        for repetition in range(1, 6):
            for condition in ("identity", "shift"):
                checkpoint = source / f"{mode}_{repetition}_{condition}.pt"
                checkpoint.write_bytes(f"frozen/{mode}/{repetition}/{condition}".encode())
                model = {
                    "training_graph_u_mode": mode, "training_condition": condition,
                    "repetition": repetition, "model_seed": 41 + repetition,
                    "learning_rate": 0.001, "checkpoint": str(checkpoint),
                    "checkpoint_sha256": shared._digest(checkpoint, None),
                }
                models[(mode, repetition, condition)] = model
                for test_mode in sweep.MODES:
                    report = {
                        "checkpoint": str(checkpoint), "checkpoint_sha256": model["checkpoint_sha256"],
                        "dump": str(source / f"iid_{test_mode}_{repetition}.h5"),
                        "dump_sha256": "a" * 64, "dump_metadata": {"seed": 2000 + repetition},
                        "task_count": 3200, "query_row_count": 240000,
                        "metrics": {metric: 0.6 + repetition * 0.01 for metric in shared.METRIC_NAMES},
                    }
                    path = source / f"{mode}_{repetition}_{condition}_on_{test_mode}.json"
                    path.write_text(json.dumps(report))
                    iid_rows.append(shared._row(report, model, test_mode, None, path, "reused_iid"))
    monkeypatch.setattr(sweep, "load_source", lambda _: (settings, models, iid_rows, {"frozen": "b" * 64}))
    return source, settings, models, iid_rows


def test_whole_location_sweep_reuses_iid_and_completed_outputs(frozen_source, tmp_path, monkeypatch):
    source, settings, models, iid_rows = frozen_source
    output = tmp_path / "sweep"
    specs = sweep.bank_specs(output, settings)
    by_path = {spec["dump"]: spec for spec in specs}
    source_bytes = {path: path.read_bytes() for path in source.iterdir()}
    original_iid_rows = deepcopy(iid_rows)
    original_settings = deepcopy(settings)
    generated, evaluated = [], []

    def generate(path, **kwargs):
        spec = by_path[path]
        assert not kwargs["resume"] and kwargs["workers"] == 2
        assert kwargs["steps"] == 100 and kwargs["batch_size"] == 32
        assert kwargs["rows"] == 150 and kwargs["features"] == 5
        assert kwargs["seed"] == 5000 + spec["repetition"]
        prior = kwargs["prior_config"]
        for key, value in spec["sampling_config"].items():
            assert getattr(prior, key) == value
        generated.append(path)
        _write_bank(spec)

    def evaluate(checkpoint, dump, **kwargs):
        assert kwargs == {"batch_size": 32, "device": "cpu"}
        assert output / "data" in dump.parents
        evaluated.append((checkpoint, dump))
        return _report(checkpoint, by_path[dump])

    monkeypatch.setattr(shared, "generate_dump", generate)
    monkeypatch.setattr(shared, "evaluate_checkpoint", evaluate)
    result = sweep.run_sweep(source, output, workers=2, device="cpu")
    assert len(generated) == 80 and len(evaluated) == 320
    assert set(Counter(dump for _, dump in evaluated).values()) == {4}
    assert set(Counter(checkpoint for checkpoint, _ in evaluated).values()) == {16}
    assert len(models) == len(result["recipe"]["source_checkpoints_sha256"]) == 20
    assert result["new_evaluation_count"] == 320 and result["reused_iid_count"] == 40
    rows, cells = result["per_repetition"], result["cells"]
    assert len(rows) == 360 and len(cells) == 72
    assert Counter(row["result_origin"] for row in rows) == {"reused_iid": 40, "new_sweep": 320}
    assert Counter(cell["test_condition"] for cell in cells) == {"identity": 8, "shift": 64}
    assert result["standard_deviation_ddof"] == 1
    for row in rows:
        if row["result_origin"] == "reused_iid":
            assert all(row[f"location_{field}"] is None for field in ("mean", "std", "min", "max"))
        else:
            assert row["location_std"] == 0.15
            assert (row["location_mean"], row["location_min"], row["location_max"]) in sweep.LOCATION_PROFILES
            for field in ("mean", "std", "min", "max"):
                assert row[f"scale_{field}"] == settings[f"graph_u_query_scale_{field}"]
    for cell in cells:
        assert cell["repetition_count"] == 5
        if cell["test_condition"] == "identity":
            assert cell["roc_auc_mean"] == pytest.approx(0.63)
            assert cell["roc_auc_std"] == pytest.approx(0.01 * math.sqrt(2.5))
        else:
            assert cell["roc_auc_mean"] == pytest.approx(0.7) and cell["roc_auc_std"] == 0
            assert cell["location_std"] == 0.15
    assert json.loads((output / "results" / "summary.json").read_text()) == result
    with (output / "results" / "per_repetition.csv").open(newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 360 and csv_rows[0]["location_mean"] == ""
    assert {f"{parameter}_{field}" for parameter in ("scale", "location")
            for field in ("mean", "std", "min", "max")} <= set(csv_rows[0])
    before = {path: path.read_bytes() for path in (output / "results").rglob("*") if path.is_file()}
    assert sweep.run_sweep(source, output, workers=2, device="cpu") == result
    assert len(generated) == 80 and len(evaluated) == 320
    assert all(path.read_bytes() == contents for path, contents in before.items())
    assert all(path.read_bytes() == contents for path, contents in source_bytes.items())
    assert iid_rows == original_iid_rows and settings == original_settings


@pytest.mark.parametrize("change", ["mean", "location_profile", "scale_profile", "metadata", "dump_hash", "checkpoint"])
def test_cached_location_report_requires_unchanged_profile_model_and_bank(
    frozen_source, tmp_path, monkeypatch, change
):
    _, settings, models, _ = frozen_source
    root = tmp_path / "sweep"
    spec = sweep.bank_specs(root, settings)[0]
    _write_bank(spec)
    evaluated = []

    def evaluate(checkpoint, dump, **_kwargs):
        assert dump == spec["dump"]
        evaluated.append(checkpoint)
        return _report(checkpoint, spec)

    monkeypatch.setattr(shared, "evaluate_checkpoint", evaluate)
    shared._evaluate_banks(root, [spec], models, "cpu")
    path = next((root / "results").rglob("*.json"))
    saved = json.loads(path.read_text())
    if change == "mean":
        saved["location_mean"] += 0.01
    elif change in ("location_profile", "scale_profile"):
        parameter = change.removesuffix("_profile")
        saved["test_sampling_config"][f"graph_u_query_{parameter}_std"] += 0.01
    elif change == "metadata":
        saved["evaluation_report"]["dump_metadata"]["prior_config"]["graph_u_query_location_mean"] += 0.01
    elif change == "dump_hash":
        saved["evaluation_report"]["dump_sha256"] = "0" * 64
    else:
        Path(saved["model"]["checkpoint"]).write_bytes(b"changed frozen model")
    path.write_text(json.dumps(saved))
    before = {path: path.read_bytes() for path in (root / "results").rglob("*.json")}
    with pytest.raises(ValueError):
        shared._evaluate_banks(root, [spec], models, "cpu")
    assert len(evaluated) == 4
    assert all(path.read_bytes() == contents for path, contents in before.items())


@pytest.mark.parametrize("parameter", ["location", "scale"])
def test_location_bank_metadata_must_match_both_sampling_distributions(
    frozen_source, tmp_path, monkeypatch, parameter
):
    _, settings, models, _ = frozen_source
    root = tmp_path / "sweep"
    spec = sweep.bank_specs(root, settings)[0]
    changed = {**spec["sampling_config"], f"graph_u_query_{parameter}_std": 0.2}
    _write_bank(spec, changed)

    def unexpected_inference(*_args, **_kwargs):
        raise AssertionError("A mismatched bank must fail before inference")

    monkeypatch.setattr(shared, "evaluate_checkpoint", unexpected_inference)
    with pytest.raises(ValueError):
        shared._evaluate_banks(root, [spec], models, "cpu")
    assert not (root / "results").exists()
