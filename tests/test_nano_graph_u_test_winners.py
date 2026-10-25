"""Final testing preserves validation selection and reuses frozen test banks."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tabicl.nano_graph_u import test_winners as final_tests
from tabicl.nano_graph_u.data import _metadata
from tabicl.nano_graph_u.model import NanoTabPFNModel
from tabicl.nano_graph_u.select import WINNERS_FORMAT
from tabicl.nano_graph_u.train import _sha256_file
from tabicl.nano_graph_u.tune import LEARNING_RATES, trial_spec
from tabicl.prior.graph_lib._config import PriorConfig


def _metadata_for(condition, *, seed, steps):
    prior = PriorConfig(
        graph_u_enabled=True,
        graph_u_structure_mode="add_root",
        graph_u_query_location=0.0 if condition == "identity" else 2.0,
        graph_u_query_scale=1.0 if condition == "identity" else 1.5,
        graph_u_max_attempts=1000,
        filter_unpredictable_graphs=True,
        filter_unpredictable_datasets=True,
        allow_act_warping=False,
    )
    metadata = _metadata(
        prior,
        steps=steps,
        batch_size=32,
        seed=seed,
        rows=150,
        features=5,
        min_train_size=0.3,
        max_train_size=0.9,
        n_jobs=1,
    )
    metadata["generator_source_sha256"] = "a" * 64
    return metadata


@pytest.fixture
def make_inputs(tmp_path):
    h5py = pytest.importorskip("h5py")

    def create(task_id=1, *, actual_model=False):
        data, search, results = (
            tmp_path / "banks",
            tmp_path / "search",
            tmp_path / "tests",
        )
        spec = final_tests.test_spec(task_id, data, search, results)
        winners = []
        for repetition in range(1, 6):
            for offset, condition in ((0, "identity"), (10, "shift")):
                candidate = trial_spec((repetition - 1) * 20 + offset + 1, data, search)
                checkpoint = (
                    search / "winners" / f"rep_{repetition}" / condition / "model.pt"
                )
                winner = {
                    "repetition": repetition,
                    "condition": condition,
                    "task_id": candidate["task_id"],
                    "learning_rate": candidate["learning_rate"],
                    "model_seed": 41 + repetition,
                    "validation_roc_auc": 0.9,
                    "checkpoint_step": 2500,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": "b" * 64,
                    "source_checkpoint": str(
                        Path(candidate["checkpoint_dir"]) / "latest.pt"
                    ),
                    "validation_dump": candidate["validation_dump"],
                    "validation_dump_sha256": "c" * 64,
                    "validation_result": candidate["result_path"],
                }
                if str(checkpoint) == spec["checkpoint"]:
                    metadata = _metadata_for(
                        condition, seed=1000 + repetition, steps=2500
                    )
                    checkpoint.parent.mkdir(parents=True)
                    payload = {
                        "format": "nano_graph_u_v1",
                        "step": 2500,
                        "model_config": final_tests.PAPER_MODEL_CONFIG,
                        "model": NanoTabPFNModel(
                            **final_tests.PAPER_MODEL_CONFIG
                        ).state_dict()
                        if actual_model
                        else {"fixture": torch.tensor(0)},
                        "dump_metadata": metadata,
                        "prior_config": metadata["prior_config"],
                        "train_config": {
                            "seed": 41 + repetition,
                            "batch_size": 32,
                            "learning_rate": candidate["learning_rate"],
                            "optimizer": "schedulefree.AdamWScheduleFree",
                            "weight_decay": 0.0,
                            "gradient_clip_norm": 1.0,
                            "loss": "query_cross_entropy",
                            "precision": "float32",
                            "dump_path": candidate["train_dump"],
                            "dump_sha256": "d" * 64,
                        },
                    }
                    torch.save(payload, checkpoint)
                    winner["checkpoint_sha256"] = _sha256_file(checkpoint)
                winners.append(winner)
        manifest = {
            "format": WINNERS_FORMAT,
            "data_root": str(data),
            "output_root": str(search),
            "selection_metric": "roc_auc",
            "validation_policy": "matching_training_condition",
            "training_steps": 2500,
            "trial_count": 100,
            "generator_source_sha256": "a" * 64,
            "learning_rates": list(LEARNING_RATES),
            "winners": winners,
        }
        Path(spec["manifest_path"]).write_text(json.dumps(manifest), encoding="utf-8")
        test_metadata = _metadata_for(
            spec["test_condition"], seed=2000 + spec["repetition"], steps=100
        )
        dump = Path(spec["test_dump"])
        dump.parent.mkdir(parents=True)
        with h5py.File(dump, "w") as stream:
            stream.attrs["metadata_json"] = json.dumps(test_metadata)
            stream.attrs["committed_steps"] = 100
            stream.create_dataset("x", (3200, 150, 5), dtype="f4")
            stream.create_dataset("y", (3200, 150), dtype="i8")
            stream.create_dataset("train_sizes", (3200,), dtype="i4")
            if actual_model:
                labels = np.tile(np.arange(150) % 2, (3200, 1))
                stream["x"][:, :, 0] = labels
                stream["y"][:] = labels
                stream["train_sizes"][:] = 75
        return spec, manifest, test_metadata

    return create


def _report(spec, metadata):
    return {
        "format": "nano_graph_u_evaluation_v1",
        "checkpoint": spec["checkpoint"],
        "checkpoint_sha256": _sha256_file(Path(spec["checkpoint"])),
        "checkpoint_step": 2500,
        "dump": spec["test_dump"],
        "dump_sha256": _sha256_file(Path(spec["test_dump"])),
        "dump_metadata": metadata,
        "task_count": 3200,
        "query_row_count": 240000,
        "metrics": {name: 0.8 for name in final_tests.METRIC_NAMES},
        "metric_task_counts": {name: 3200 for name in final_tests.METRIC_NAMES},
    }


def _result(spec, manifest, metadata):
    return {
        "format": final_tests.TEST_FORMAT,
        "test": spec,
        "manifest_sha256": hashlib.sha256(
            Path(spec["manifest_path"]).read_bytes()
        ).hexdigest(),
        "winner": next(
            winner
            for winner in manifest["winners"]
            if winner["checkpoint"] == spec["checkpoint"]
        ),
        "evaluation_report": _report(spec, metadata),
    }


def test_final_test_mapping_crosses_both_winners_with_both_banks(tmp_path):
    specs = [
        final_tests.test_spec(task, tmp_path, tmp_path, tmp_path)
        for task in range(1, 21)
    ]
    assert len({spec["result_path"] for spec in specs}) == 20
    assert len({spec["checkpoint"] for spec in specs}) == 10
    assert len({spec["test_dump"] for spec in specs}) == 10
    for repetition in range(1, 6):
        group = [spec for spec in specs if spec["repetition"] == repetition]
        assert {
            (spec["training_condition"], spec["test_condition"]) for spec in group
        } == {
            (training, testing)
            for training in ("identity", "shift")
            for testing in ("identity", "shift")
        }
        assert {spec["model_seed"] for spec in group} == {41 + repetition}
        assert all(
            f"test_seed{2000 + repetition}" in spec["test_dump"] for spec in group
        )


@pytest.mark.parametrize("task", [0, 21, True, "1"])
def test_final_test_mapping_rejects_invalid_tasks(tmp_path, task):
    with pytest.raises(ValueError, match="1 to 20"):
        final_tests.test_spec(task, tmp_path, tmp_path, tmp_path)


@pytest.mark.parametrize("task", [2, 3])
def test_final_testing_reuses_evaluator_on_cross_condition_bank_and_is_idempotent(
    make_inputs, monkeypatch, task
):
    spec, _manifest, metadata = make_inputs(task)
    calls = []

    def evaluate(checkpoint, dump, **kwargs):
        assert checkpoint == spec["checkpoint"]
        assert dump == spec["test_dump"]
        assert kwargs == {"batch_size": 32, "device": "cpu"}
        calls.append("test")
        return _report(spec, metadata)

    monkeypatch.setattr(final_tests, "evaluate_checkpoint", evaluate)
    data = Path(spec["test_dump"]).parents[1]
    search = Path(spec["manifest_path"]).parents[1]
    results = Path(spec["result_path"]).parents[1]
    initial = final_tests.run_test(task, data, search, results, device="cpu")
    assert initial["winner"]["learning_rate"] == LEARNING_RATES[0]
    assert final_tests.run_test(task, data, search, results, device="cpu") == initial
    assert calls == ["test"]
    assert not list(data.rglob("*train*"))
    assert not list(data.rglob("*validation*"))
    assert not (search / "rep_1").exists()
    assert initial["winner"]["condition"] != spec["test_condition"]


@pytest.mark.parametrize(
    "corruption", ["partial", "wrong_seed", "wrong_condition", "checkpoint"]
)
def test_final_test_preflight_rejects_changed_inputs_before_inference(
    make_inputs, monkeypatch, corruption
):
    h5py = pytest.importorskip("h5py")
    spec, _, metadata = make_inputs()
    if corruption == "checkpoint":
        Path(spec["checkpoint"]).write_bytes(b"changed")
    else:
        with h5py.File(spec["test_dump"], "r+") as stream:
            if corruption == "partial":
                stream.attrs["committed_steps"] = 99
            else:
                if corruption == "wrong_seed":
                    metadata["seed"] = 3001
                else:
                    metadata["prior_config"]["graph_u_query_location"] = 2.0
                stream.attrs["metadata_json"] = json.dumps(metadata)
    monkeypatch.setattr(
        final_tests,
        "evaluate_checkpoint",
        lambda *args, **kwargs: pytest.fail("Must not evaluate"),
    )
    with pytest.raises(ValueError):
        final_tests.run_test(
            1,
            Path(spec["test_dump"]).parents[1],
            Path(spec["manifest_path"]).parents[1],
            Path(spec["result_path"]).parents[1],
            device="cpu",
        )
    assert not Path(spec["result_path"]).exists()


@pytest.mark.parametrize("corruption", ["count", "metric", "selection"])
def test_final_result_rejects_partial_metrics_or_changed_selection(
    make_inputs, corruption
):
    spec, manifest, metadata = make_inputs()
    result = _result(spec, manifest, metadata)
    if corruption == "count":
        result["evaluation_report"]["metric_task_counts"]["roc_auc"] = 3199
    elif corruption == "metric":
        result["evaluation_report"]["metrics"]["nll"] = float("nan")
    else:
        result["manifest_sha256"] = "e" * 64
    with pytest.raises(ValueError):
        final_tests.validate_test_result(result, spec)


def test_final_result_detects_changed_test_file_with_shared_hash_cache(make_inputs):
    spec, manifest, metadata = make_inputs()
    result = _result(spec, manifest, metadata)
    cache = {}
    final_tests.validate_test_result(result, spec, hash_cache=cache)
    Path(spec["test_dump"]).touch()
    final_tests.validate_test_result(result, spec, hash_cache=cache)
    changed = copy.deepcopy(result)
    changed["evaluation_report"]["dump_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="Test bank changed"):
        final_tests.validate_test_result(changed, spec, hash_cache=cache)


def test_final_testing_actual_cpu_inference_on_all_test_tasks(make_inputs, monkeypatch):
    # A small test-only architecture keeps the inference smoke fast; the
    # production checkpoint checks continue to require PAPER_MODEL_CONFIG.
    monkeypatch.setattr(
        final_tests,
        "PAPER_MODEL_CONFIG",
        {
            "embedding_size": 8,
            "num_attention_heads": 2,
            "mlp_hidden_size": 16,
            "num_layers": 1,
            "num_outputs": 2,
        },
    )
    spec, _, _ = make_inputs(actual_model=True)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        report = final_tests.run_test(
            1,
            Path(spec["test_dump"]).parents[1],
            Path(spec["manifest_path"]).parents[1],
            Path(spec["result_path"]).parents[1],
            device="cpu",
        )["evaluation_report"]
    finally:
        torch.set_num_threads(previous_threads)
    assert report["task_count"] == 3200
    assert report["query_row_count"] == 3200 * 75
    assert all(count == 3200 for count in report["metric_task_counts"].values())
    assert 0 <= report["metrics"]["roc_auc"] <= 1
