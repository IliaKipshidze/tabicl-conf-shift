"""Tests for deterministic, on-the-fly Graph-U checkpoint evaluation."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from tabicl._model.tabicl import TabICL
from tabicl.evaluation._graph_u import (
    CheckpointDescriptor,
    EvaluationTask,
    build_prior_config,
    default_condition_specs,
    derive_task_seed,
    disambiguate_checkpoint_labels,
    evaluate,
    generate_task,
    inspect_checkpoint,
    parse_condition,
    prior_reconstruction_details,
    resolve_conditions,
    run_from_args,
    score_classification_task,
    score_regression_task,
    write_csv_results,
)
from tabicl.prior.graph_lib._config import PriorConfig
from tabicl.train._checkpoint import checkpoint_training_metadata


def _descriptor(
    tmp_path: Path,
    *,
    task_type: str = "classification",
    graph_u: bool = True,
) -> CheckpointDescriptor:
    model_config = {
        "max_classes": 0 if task_type == "regression" else 3,
        "num_quantiles": 5,
    }
    prior_config = vars(
        PriorConfig(
            graph_u_enabled=graph_u,
            graph_u_structure_mode="add_root",
            graph_u_query_location=2.5,
            graph_u_query_scale=1.75,
            graph_u_force_gaussian=False,
            graph_u_max_attempts=123,
        )
    )
    return CheckpointDescriptor(
        path=(tmp_path / f"{task_type}.ckpt").resolve(),
        label=task_type,
        task_type=task_type,
        model_config=model_config,
        training_config={"prior_type": "graph_scm"},
        prior_config=prior_config,
        curr_step=10,
        size_bytes=100,
        sha256="0" * 64,
    )


def _tiny_model_config(*, regression: bool = False) -> dict:
    return {
        "max_classes": 0 if regression else 3,
        "num_quantiles": 5,
        "embed_dim": 8,
        "col_num_blocks": 1,
        "col_nhead": 2,
        "col_num_inds": 4,
        "col_affine": False,
        "col_feature_group": False,
        "col_feature_group_size": 3,
        "col_target_aware": True,
        "col_ssmax": False,
        "row_num_blocks": 1,
        "row_nhead": 2,
        "row_num_cls": 1,
        "row_rope_base": 10000,
        "row_rope_interleaved": True,
        "icl_num_blocks": 1,
        "icl_nhead": 2,
        "icl_ssmax": False,
        "ff_factor": 2,
        "dropout": 0.0,
        "activation": "gelu",
        "norm_first": True,
        "bias_free_ln": False,
        "zero_init": False,
        "recompute": False,
    }


def _save_tiny_checkpoint(path: Path, *, regression: bool = False) -> None:
    model_config = _tiny_model_config(regression=regression)
    model = TabICL(**model_config)
    training_config = {"prior_type": "graph_scm"}
    if regression:
        training_config["regression_method"] = "quantile"
    torch.save(
        {
            "config": model_config,
            "state_dict": model.state_dict(),
            **checkpoint_training_metadata(
                training_config,
                {
                    **vars(PriorConfig()),
                    "prior_type": "graph_scm",
                },
            ),
        },
        path,
    )


@pytest.mark.parametrize(
    "value,expected",
    [
        ("ordinary", ("ordinary", "ordinary", None, None, None)),
        ("identity", ("identity", "identity", None, None, None)),
        ("matched", ("matched", "matched", None, None, None)),
        ("strong:4:2", ("strong", "custom", 4.0, 2.0, None)),
        ("reverse:-2:0.5:true", ("reverse", "custom", -2.0, 0.5, True)),
    ],
)
def test_parse_condition(value, expected):
    spec = parse_condition(value)
    assert (
        spec.name,
        spec.kind,
        spec.query_location,
        spec.query_scale,
        spec.force_gaussian,
    ) == expected


@pytest.mark.parametrize(
    "value,error",
    [
        ("broken", "Condition must be"),
        ("bad name:1:2", "condition names"),
        ("shift:nan:1", "finite"),
        ("shift:0:0", "positive"),
        ("shift:0:1:maybe", "boolean"),
    ],
)
def test_parse_condition_rejects_invalid_specs(value, error):
    with pytest.raises(ValueError, match=error):
        parse_condition(value)


def test_condition_resolution_uses_one_reference_for_all_models(tmp_path):
    reference = _descriptor(tmp_path)
    conditions = resolve_conditions(
        [
            parse_condition("ordinary"),
            parse_condition("identity"),
            parse_condition("matched"),
            parse_condition("strong:5:2"),
        ],
        reference,
    )

    assert [condition.name for condition in conditions] == [
        "ordinary",
        "identity",
        "matched",
        "strong",
    ]
    assert conditions[0].graph_u_enabled is False
    assert all(condition.structure_mode == "add_root" for condition in conditions)
    assert (conditions[1].query_location, conditions[1].query_scale) == (0.0, 1.0)
    assert (conditions[2].query_location, conditions[2].query_scale) == (2.5, 1.75)
    assert (conditions[3].query_location, conditions[3].query_scale) == (5.0, 2.0)
    assert all(condition.force_gaussian is False for condition in conditions)

    config = build_prior_config(reference, conditions[2])
    assert config.graph_u_enabled is True
    assert config.graph_u_structure_mode == "add_root"
    assert config.graph_u_query_location == 2.5
    assert config.graph_u_query_scale == 1.75
    assert config.graph_u_max_attempts == 123


def test_default_conditions_only_include_matched_when_recorded(tmp_path):
    graph_u_reference = _descriptor(tmp_path, graph_u=True)
    legacy_reference = CheckpointDescriptor(
        **{
            **graph_u_reference.__dict__,
            "training_config": None,
            "prior_config": None,
        }
    )

    assert [spec.name for spec in default_condition_specs(graph_u_reference)] == [
        "ordinary",
        "identity",
        "matched",
    ]
    assert [spec.name for spec in default_condition_specs(legacy_reference)] == [
        "ordinary",
        "identity",
    ]

    identity = resolve_conditions([parse_condition("identity")], legacy_reference)[0]
    assert "evaluator default" in identity.source
    assert prior_reconstruction_details(graph_u_reference)["complete"] is True
    legacy_details = prior_reconstruction_details(legacy_reference)
    assert legacy_details["complete"] is False
    assert legacy_details["missing_fields_using_evaluator_defaults"]


def test_legacy_graph_u_prior_reconstructs_historical_reject_mode(tmp_path):
    reference = _descriptor(tmp_path)
    assert reference.prior_config is not None
    old_prior_config = dict(reference.prior_config)
    del old_prior_config["graph_u_structure_mode"]
    reference = dataclasses.replace(reference, prior_config=old_prior_config)

    conditions = resolve_conditions(
        [parse_condition("identity"), parse_condition("matched")], reference
    )

    assert [condition.structure_mode for condition in conditions] == [
        "reject",
        "reject",
    ]
    assert (
        build_prior_config(reference, conditions[1]).graph_u_structure_mode == "reject"
    )
    assert prior_reconstruction_details(reference)["complete"] is True


def test_duplicate_checkpoint_stems_get_unambiguous_labels(tmp_path):
    descriptor = _descriptor(tmp_path)
    first = dataclasses.replace(
        descriptor,
        path=(tmp_path / "baseline" / "step-100.ckpt").resolve(),
        label="step-100",
    )
    second = dataclasses.replace(
        descriptor,
        path=(tmp_path / "graph-u" / "step-100.ckpt").resolve(),
        label="step-100",
    )

    labeled = disambiguate_checkpoint_labels([first, second])

    assert [item.label for item in labeled] == [
        "baseline/step-100",
        "graph-u/step-100",
    ]
    with pytest.raises(ValueError, match="supplied more than once"):
        disambiguate_checkpoint_labels([first, first])


def test_task_generation_is_reproducible_and_restores_rng_state():
    seed = derive_task_seed(42, 0)
    config = PriorConfig(
        graph_u_enabled=True,
        graph_u_query_location=2.0,
        graph_u_query_scale=1.5,
        graph_u_max_attempts=1000,
    )
    random.seed(8)
    np.random.seed(8)
    torch.manual_seed(8)
    expected_after = (random.random(), np.random.random(), torch.rand(1))
    random.seed(8)
    np.random.seed(8)
    torch.manual_seed(8)

    first = generate_task(
        task_type="regression",
        prior_config=config,
        support_size=16,
        query_size=8,
        num_features=3,
        num_classes=2,
        seed=seed,
    )
    actual_after = (random.random(), np.random.random(), torch.rand(1))
    random.random()
    np.random.random()
    torch.rand(10)
    second = generate_task(
        task_type="regression",
        prior_config=config,
        support_size=16,
        query_size=8,
        num_features=3,
        num_classes=2,
        seed=seed,
    )

    assert actual_after[0] == expected_after[0]
    assert actual_after[1] == expected_after[1]
    torch.testing.assert_close(actual_after[2], expected_after[2])
    assert first.fingerprint == second.fingerprint
    torch.testing.assert_close(first.X, second.X)
    torch.testing.assert_close(first.y, second.y)


def test_task_generation_rejects_impossible_classification_split():
    with pytest.raises(ValueError, match="each be at least 2"):
        generate_task(
            task_type="classification",
            prior_config=PriorConfig(),
            support_size=1,
            query_size=10,
            num_features=2,
            num_classes=4,
            seed=1,
        )


def test_requested_class_capacity_may_exceed_split_size():
    task = generate_task(
        task_type="classification",
        prior_config=PriorConfig(),
        support_size=2,
        query_size=2,
        num_features=2,
        num_classes=4,
        seed=1,
    )

    assert task.train_size == 2
    assert len(torch.unique(task.y[:2])) == 2
    assert set(task.y[:2].tolist()) == set(task.y[2:].tolist())


class _RecordingClassifier:
    def __init__(self):
        self.seen_X = None
        self.seen_y = None

    def __call__(self, X, y_support, **_kwargs):
        self.seen_X = X.detach().cpu()
        self.seen_y = y_support.detach().cpu()
        return torch.tensor([[[0.0, 3.0], [3.0, 0.0]]], device=X.device)


def test_classification_scorer_never_passes_query_labels():
    model = _RecordingClassifier()
    task = EvaluationTask(
        X=torch.arange(12, dtype=torch.float32).reshape(4, 3),
        y=torch.tensor([10, 20, 20, 10]),
        train_size=2,
        active_features=3,
        seed=1,
        fingerprint="test",
    )

    metrics = score_classification_task(
        model,
        task,
        device=torch.device("cpu"),
        inference_config=None,
        softmax_temperature=1.0,
    )

    assert model.seen_X.shape == (1, 4, 3)
    assert model.seen_y.tolist() == [[0, 1]]
    assert metrics["accuracy"] == 1.0
    assert metrics["log_loss"] > 0
    assert metrics["brier_score"] > 0


class _ExtremeLogitClassifier:
    def __call__(self, X, y_support, **_kwargs):
        return torch.tensor(
            [[[-200.0, 0.0], [0.0, -200.0]]],
            device=X.device,
        )


def test_classification_log_loss_is_stable_for_extreme_logits():
    task = EvaluationTask(
        X=torch.zeros(4, 2),
        y=torch.tensor([0, 1, 0, 1]),
        train_size=2,
        active_features=2,
        seed=1,
        fingerprint="test",
    )

    metrics = score_classification_task(
        _ExtremeLogitClassifier(),
        task,
        device=torch.device("cpu"),
        inference_config=None,
        softmax_temperature=1.0,
    )

    assert metrics["log_loss"] == pytest.approx(200.0)


class _HierarchicalClassifierStub:
    def __init__(self):
        self.temperatures = []

    def __call__(self, X, y_support, *, softmax_temperature, **_kwargs):
        self.temperatures.append(softmax_temperature)
        probabilities = torch.tensor(
            [
                [
                    [0.05, 0.05, 0.1, 0.8],
                    [0.7, 0.1, 0.1, 0.1],
                    [0.1, 0.7, 0.1, 0.1],
                    [0.1, 0.1, 0.7, 0.1],
                ]
            ],
            device=X.device,
        )
        # This is how TabICL represents hierarchical probabilities when the
        # caller asks for logits.
        return softmax_temperature * torch.log(probabilities + 1e-6)


def test_classification_temperature_is_forwarded_for_hierarchical_logits():
    model = _HierarchicalClassifierStub()
    task = EvaluationTask(
        X=torch.zeros(8, 2),
        y=torch.tensor([0, 1, 2, 3, 3, 0, 1, 2]),
        train_size=4,
        active_features=2,
        seed=1,
        fingerprint="test",
    )

    low_temperature = score_classification_task(
        model,
        task,
        device=torch.device("cpu"),
        inference_config=None,
        softmax_temperature=0.5,
    )
    high_temperature = score_classification_task(
        model,
        task,
        device=torch.device("cpu"),
        inference_config=None,
        softmax_temperature=1.7,
    )

    assert model.temperatures == [0.5, 1.7]
    assert low_temperature["accuracy"] == 1.0
    assert low_temperature["log_loss"] == pytest.approx(high_temperature["log_loss"])


class _RecordingRegressor:
    def __init__(self):
        self.seen_y = None

    def __call__(self, X, y_support, **_kwargs):
        self.seen_y = y_support.detach().cpu()
        return torch.tensor(
            [[[0.5, 1.0, 1.5], [2.5, 3.0, 3.5]]],
            device=X.device,
        )


def test_regression_scorer_never_passes_query_targets():
    model = _RecordingRegressor()
    task = EvaluationTask(
        X=torch.zeros(4, 2),
        y=torch.tensor([-1.0, 0.0, 1.0, 3.0]),
        train_size=2,
        active_features=2,
        seed=1,
        fingerprint="test",
    )

    metrics = score_regression_task(
        model,
        task,
        device=torch.device("cpu"),
        inference_config=None,
    )

    assert model.seen_y.tolist() == [[-1.0, 0.0]]
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)
    assert metrics["pinball_loss"] >= 0


def test_inspect_checkpoint_detects_architecture_task_type(tmp_path):
    classification_path = tmp_path / "classification.ckpt"
    regression_path = tmp_path / "regression.ckpt"
    metadata = checkpoint_training_metadata(
        {"prior_type": "graph_scm"},
        vars(PriorConfig()),
    )
    torch.save(
        {
            "config": {"max_classes": 10, "num_quantiles": 9},
            "state_dict": {},
            "curr_step": 12,
            **metadata,
        },
        classification_path,
    )
    torch.save(
        {
            "config": {"max_classes": 0, "num_quantiles": 9},
            "state_dict": {},
            **metadata,
        },
        regression_path,
    )

    classification = inspect_checkpoint(classification_path)
    regression = inspect_checkpoint(regression_path)

    assert classification.task_type == "classification"
    assert classification.curr_step == 12
    assert len(classification.sha256) == 64
    assert regression.task_type == "regression"


def test_csv_results_contain_summary_and_per_task_rows(tmp_path):
    output = tmp_path / "results.json"
    result = {
        "models": [
            {
                "checkpoint": {
                    "path": "model.ckpt",
                    "label": "model",
                    "curr_step": 5,
                },
                "conditions": [
                    {
                        "condition": {"name": "identity"},
                        "summary": {
                            "tasks": 1,
                            "accuracy": {
                                "mean": 1.0,
                                "std": 0.0,
                                "standard_error": 0.0,
                                "min": 1.0,
                                "max": 1.0,
                            },
                        },
                        "tasks": [
                            {
                                "task_index": 0,
                                "seed": 7,
                                "fingerprint": "abc",
                                "active_features": 3,
                                "metrics": {"accuracy": 1.0},
                            }
                        ],
                    }
                ],
            }
        ]
    }

    summary_path, tasks_path = write_csv_results(output, result)

    with summary_path.open(newline="", encoding="utf-8") as file:
        summary_rows = list(csv.DictReader(file))
    with tasks_path.open(newline="", encoding="utf-8") as file:
        task_rows = list(csv.DictReader(file))
    assert summary_rows[0]["metric"] == "accuracy"
    assert summary_rows[0]["mean"] == "1.0"
    assert task_rows[0]["fingerprint"] == "abc"
    assert task_rows[0]["value"] == "1.0"


def test_run_from_args_rejects_mixed_checkpoint_types(tmp_path):
    paths = []
    for name, max_classes in (("classification", 2), ("regression", 0)):
        path = tmp_path / f"{name}.ckpt"
        torch.save(
            {
                "config": {"max_classes": max_classes, "num_quantiles": 5},
                "state_dict": {},
            },
            path,
        )
        paths.append(str(path))
    args = argparse.Namespace(
        checkpoints=paths,
        reference_checkpoint=None,
        condition=["ordinary"],
        datasets=1,
        support_size=4,
        query_size=2,
        features=2,
        classes=2,
        seed=1,
        device="cpu",
        amp=False,
        softmax_temperature=1.0,
        max_attempts=10,
        output=tmp_path / "result.json",
        no_csv=False,
        overwrite=False,
        quiet=True,
    )

    with pytest.raises(ValueError, match="mixes classification and regression"):
        run_from_args(args)


def test_run_from_args_writes_real_results_without_tensors_and_refuses_overwrite(
    tmp_path,
):
    checkpoint_path = tmp_path / "tiny-classifier.ckpt"
    output_path = tmp_path / "result.json"
    _save_tiny_checkpoint(checkpoint_path)
    args = argparse.Namespace(
        checkpoints=[str(checkpoint_path)],
        reference_checkpoint=None,
        condition=["identity"],
        datasets=1,
        support_size=16,
        query_size=8,
        features=3,
        classes=2,
        seed=5,
        device="cpu",
        amp=False,
        softmax_temperature=1.0,
        max_attempts=1000,
        output=output_path,
        no_csv=False,
        overwrite=False,
        quiet=True,
    )

    run_from_args(args)

    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    task_record = loaded["models"][0]["conditions"][0]["tasks"][0]
    assert loaded["protocol"]["stores_generated_tensors"] is False
    assert loaded["protocol"]["query_labels_passed_to_model"] is False
    assert set(task_record) == {
        "active_features",
        "fingerprint",
        "metrics",
        "seed",
        "task_index",
    }
    assert output_path.with_suffix(".summary.csv").is_file()
    assert output_path.with_suffix(".tasks.csv").is_file()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_from_args(args)
    args.overwrite = True
    run_from_args(args)


def test_evaluate_runs_end_to_end_with_real_tiny_checkpoint(tmp_path):
    path = tmp_path / "tiny-classifier.ckpt"
    _save_tiny_checkpoint(path)
    descriptor = inspect_checkpoint(path)
    condition = resolve_conditions([parse_condition("identity")], descriptor)

    result = evaluate(
        descriptors=[descriptor],
        reference=descriptor,
        conditions=condition,
        datasets=1,
        support_size=16,
        query_size=8,
        num_features=3,
        num_classes=2,
        base_seed=7,
        device=torch.device("cpu"),
        use_amp=False,
        softmax_temperature=1.0,
        max_attempts=1000,
        progress=False,
    )

    condition_result = result["models"][0]["conditions"][0]
    assert result["protocol"]["task_type"] == "classification"
    assert condition_result["summary"]["tasks"] == 1
    assert 0.0 <= condition_result["summary"]["accuracy"]["mean"] <= 1.0
    assert len(condition_result["tasks"][0]["fingerprint"]) == 64


def test_evaluate_runs_end_to_end_with_real_tiny_regression_checkpoint(tmp_path):
    path = tmp_path / "tiny-regressor.ckpt"
    _save_tiny_checkpoint(path, regression=True)
    descriptor = inspect_checkpoint(path)
    condition = resolve_conditions([parse_condition("identity")], descriptor)

    result = evaluate(
        descriptors=[descriptor],
        reference=descriptor,
        conditions=condition,
        datasets=1,
        support_size=16,
        query_size=8,
        num_features=3,
        num_classes=2,
        base_seed=11,
        device=torch.device("cpu"),
        use_amp=False,
        softmax_temperature=1.0,
        max_attempts=1000,
        progress=False,
    )

    condition_result = result["models"][0]["conditions"][0]
    assert result["protocol"]["task_type"] == "regression"
    assert condition_result["summary"]["tasks"] == 1
    assert condition_result["summary"]["rmse"]["mean"] >= 0.0
    assert condition_result["summary"]["pinball_loss"]["mean"] >= 0.0


def test_multiple_checkpoints_receive_identical_regenerated_tasks(tmp_path):
    first_path = tmp_path / "first.ckpt"
    second_path = tmp_path / "second.ckpt"
    _save_tiny_checkpoint(first_path)
    _save_tiny_checkpoint(second_path)
    descriptors = [inspect_checkpoint(first_path), inspect_checkpoint(second_path)]
    condition = resolve_conditions([parse_condition("identity")], descriptors[0])

    result = evaluate(
        descriptors=descriptors,
        reference=descriptors[0],
        conditions=condition,
        datasets=1,
        support_size=16,
        query_size=8,
        num_features=3,
        num_classes=2,
        base_seed=19,
        device=torch.device("cpu"),
        use_amp=False,
        softmax_temperature=1.0,
        max_attempts=1000,
        progress=False,
    )

    fingerprints = [
        model_result["conditions"][0]["tasks"][0]["fingerprint"]
        for model_result in result["models"]
    ]
    assert fingerprints[0] == fingerprints[1]
