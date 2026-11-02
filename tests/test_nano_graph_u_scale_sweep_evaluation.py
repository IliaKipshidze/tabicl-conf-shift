"""Sweep evaluation reuses frozen models and validates cached reports."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tabicl.nano_graph_u import scale_sweep as sweep
from tabicl.prior.graph_lib._config import GRAPH_U_SAMPLING_DEFAULTS


def _settings():
    return {**GRAPH_U_SAMPLING_DEFAULTS, "graph_u_shift_mode": "sampled"}


def _hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _report(model, spec, metadata):
    return {
        "format": "nano_graph_u_evaluation_v1", "checkpoint": model["checkpoint"],
        "checkpoint_sha256": model["checkpoint_sha256"], "checkpoint_step": 2500,
        "dump": str(spec["dump"]), "dump_sha256": sweep._digest(spec["dump"], None),
        "dump_metadata": metadata, "task_count": 3200, "query_row_count": 32000,
        "metrics": {metric: 0.5 for metric in sweep.METRIC_NAMES},
        "metric_task_counts": {metric: 3200 for metric in sweep.METRIC_NAMES},
    }


@pytest.fixture
def one_bank(tmp_path, monkeypatch):
    root = tmp_path / "sweep"
    spec = sweep.bank_specs(root, _settings())[0]
    spec["dump"].parent.mkdir(parents=True)
    spec["dump"].write_bytes(b"bank fixture")
    metadata = {"seed": spec["seed"], "prior_config": spec["sampling_config"]}
    monkeypatch.setattr(sweep, "_validate_bank", lambda _: SimpleNamespace(metadata=metadata))
    models = {}
    for mode in sweep.MODES:
        for condition in ("identity", "shift"):
            checkpoint = tmp_path / f"{mode}_{condition}.pt"
            checkpoint.write_bytes(f"{mode}_{condition}".encode())
            models[(mode, 1, condition)] = {
                "training_graph_u_mode": mode, "training_condition": condition,
                "repetition": 1, "model_seed": 42, "learning_rate": 0.001,
                "checkpoint": str(checkpoint), "checkpoint_sha256": sweep._digest(checkpoint, None),
            }
    calls = []

    def infer(checkpoint, dump, *, batch_size, device):
        assert batch_size == 32 and device == "cpu" and dump == spec["dump"]
        calls.append(checkpoint)
        model = next(model for model in models.values() if model["checkpoint"] == checkpoint)
        return _report(model, spec, metadata)

    monkeypatch.setattr(sweep, "evaluate_checkpoint", infer)
    return root, spec, metadata, models, calls, infer


def test_interrupted_inference_reuses_complete_reports_and_only_finishes_missing_models(one_bank, monkeypatch):
    root, spec, _, models, calls, infer = one_bank
    attempts = []

    def interrupt(*args, **kwargs):
        attempts.append(args[0])
        if len(attempts) == 3:
            raise RuntimeError("interrupted evaluation")
        return infer(*args, **kwargs)

    monkeypatch.setattr(sweep, "evaluate_checkpoint", interrupt)
    with pytest.raises(RuntimeError, match="interrupted evaluation"):
        sweep._evaluate_banks(root, [spec], models, "cpu")
    assert len(list((root / "results").rglob("*.json"))) == 2
    monkeypatch.setattr(sweep, "evaluate_checkpoint", infer)
    rows = sweep._evaluate_banks(root, [spec], models, "cpu")
    assert len(rows) == 4 and len(calls) == 4
    before = {p: p.read_bytes() for p in (root / "results").rglob("*.json")}
    assert sweep._evaluate_banks(root, [spec], models, "cpu") == rows
    assert len(calls) == 4
    assert all(p.read_bytes() == content for p, content in before.items())
    assert {row["test_condition"] for row in rows} == {"shift"}
    assert {row["scale_mean"] for row in rows} == {1.3}


@pytest.mark.parametrize("change", ["profile", "model", "metadata", "task_count", "metric_count", "metric_nan"])
def test_cached_report_must_match_the_frozen_model_bank_and_complete_metrics(one_bank, change):
    root, spec, _, models, calls, _ = one_bank
    sweep._evaluate_banks(root, [spec], models, "cpu")
    path = next((root / "results").rglob("*.json"))
    saved = json.loads(path.read_text())
    report = saved["evaluation_report"]
    if change == "profile":
        saved["test_sampling_config"]["graph_u_query_scale_mean"] = 1.4
    elif change == "model":
        saved["model"]["checkpoint_sha256"] = "0" * 64
    elif change == "metadata":
        report["dump_metadata"]["seed"] = 999
    elif change == "task_count":
        report["task_count"] = 3199
    elif change == "metric_count":
        report["metric_task_counts"]["roc_auc"] = 3199
    else:
        report["metrics"]["nll"] = float("nan")
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        sweep._evaluate_banks(root, [spec], models, "cpu")
    assert len(calls) == 4


def test_source_loading_validates_40_iid_reports_without_inference_or_source_writes(tmp_path, monkeypatch):
    source = tmp_path / "source"
    settings, validated = _settings(), []
    monkeypatch.setattr(sweep, "load_experiment", lambda _: settings)
    for task in range(1, 81):
        spec = sweep.test_for_task(source, task, settings)
        if spec["test_condition"] != "identity":
            continue
        mode = spec.get("graph_u_mode", "confounding")
        checkpoint_hash = _hash(spec["checkpoint"])
        result = {
            "test": spec, "manifest_sha256": _hash(mode),
            "winner": {"learning_rate": 0.001, "checkpoint_sha256": checkpoint_hash},
            "evaluation_report": {
                "checkpoint": spec["checkpoint"], "checkpoint_sha256": checkpoint_hash,
                "dump": spec["test_dump"], "dump_sha256": _hash(spec["test_dump"]),
                "dump_metadata": {"seed": 2000 + spec["repetition"]},
                "task_count": 3200, "query_row_count": 32000,
                "metrics": {metric: 0.5 for metric in sweep.METRIC_NAMES},
            },
        }
        path = Path(spec["result_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result))

    def validate(result, spec, *, hash_cache):
        assert result["test"] == spec
        assert spec["test_condition"] == "identity" and spec["sampling_config"] == settings
        assert isinstance(hash_cache, dict)
        validated.append(spec["task_id"])

    def unexpected_inference(*_args, **_kwargs):
        raise AssertionError("IID evaluations must be reused")

    monkeypatch.setattr(sweep, "validate_test_result", validate)
    monkeypatch.setattr(sweep, "evaluate_checkpoint", unexpected_inference)
    before = {path: path.read_bytes() for path in source.rglob("*.json")}
    profile, models, rows, manifests = sweep.load_source(source)
    assert profile == settings and len(models) == 20 and len(rows) == 40 and len(manifests) == 2
    assert len(validated) == 40
    assert {row["result_origin"] for row in rows} == {"reused_iid"}
    assert all(row["scale_mean"] is None and row["test_condition"] == "identity" for row in rows)
    assert all(path.read_bytes() == content for path, content in before.items())
