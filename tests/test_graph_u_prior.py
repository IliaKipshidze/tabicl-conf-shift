"""Focused tests for the Graph-U source-shift prior."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pytest
import torch

from tabicl.prior import _dataset as prior_dataset_module
from tabicl.prior import _graph_scm as graph_scm_module
from tabicl.prior._dataset import GraphPrior, Prior, PriorDataset
from tabicl.prior._graph_scm import GraphSCM
from tabicl.prior._reg2cls import outlier_removing, standard_scaling
from tabicl.prior.graph_lib import _dataset as dataset_module
from tabicl.prior.graph_lib import _function as function_module
from tabicl.prior.graph_lib import _graph_function as graph_function_module
from tabicl.prior.graph_lib._activation import Standardize
from tabicl.prior.graph_lib._base import (
    Context,
    Dataset,
    DatasetProperties,
    FeatureSpec,
)
from tabicl.prior.graph_lib._config import PriorConfig
from tabicl.prior.graph_lib._dataset import (
    RandomDataset,
    construct_graph_u_root,
    find_graph_u_candidates,
)
from tabicl.prior.graph_lib._graph_function import RandomGraphFunction
from tabicl.prior.graph_lib._points import (
    RandomGaussianPoints,
    RandomPoints,
    RandomUniformPoints,
)


def _feature_specs(*groups: str) -> dict[str, FeatureSpec]:
    return {
        f"{group}_{idx}": FeatureSpec(group=group) for idx, group in enumerate(groups)
    }


@pytest.mark.parametrize(
    "graph,node_feature_specs,expected",
    [
        (
            [[], [0], [0]],
            [{}, _feature_specs("x"), _feature_specs("y")],
            [0],
        ),
        pytest.param(
            [[], [0], [0]],
            [_feature_specs("x"), _feature_specs("x"), _feature_specs("y")],
            [],
            id="observed-root-is-not-hidden",
        ),
        pytest.param(
            [[], [0], [1], [1]],
            [{}, {}, _feature_specs("x"), _feature_specs("y")],
            [],
            id="non-root-common-parent-is-not-a-source",
        ),
        pytest.param(
            [[], [0], [0], [1], [2]],
            [{}, {}, {}, _feature_specs("x"), _feature_specs("y")],
            [],
            id="ancestor-without-direct-observed-children-is-not-eligible",
        ),
        pytest.param(
            [[], [0]],
            [{}, _feature_specs("x", "y")],
            [],
            id="x-and-y-must-be-on-distinct-child-nodes",
        ),
        pytest.param(
            [[], [], [0], [0], [1], [1]],
            [
                {},
                {},
                _feature_specs("x"),
                _feature_specs("y"),
                _feature_specs("x"),
                _feature_specs("y"),
            ],
            [0, 1],
            id="all-eligible-roots-are-returned-in-node-order",
        ),
    ],
)
def test_find_graph_u_candidates_requires_hidden_root_with_distinct_direct_x_y_children(
    graph, node_feature_specs, expected
):
    assert find_graph_u_candidates(graph, node_feature_specs) == expected


def test_construct_graph_u_root_preserves_base_dag_and_feature_placement():
    graph = [[], [0], [0, 1]]
    node_feature_specs = [
        _feature_specs("x"),
        _feature_specs("x"),
        _feature_specs("y"),
    ]

    augmented_graph, augmented_specs, x_child, y_child = construct_graph_u_root(
        graph, node_feature_specs
    )

    assert augmented_graph[0] == []
    assert augmented_specs[0] == {}
    assert x_child != y_child
    assert 0 in augmented_graph[x_child]
    assert 0 in augmented_graph[y_child]
    for base_idx, base_parents in enumerate(graph):
        shifted_idx = base_idx + 1
        forced_parent = [0] if shifted_idx in {x_child, y_child} else []
        assert augmented_graph[shifted_idx] == [
            *forced_parent,
            *(parent_idx + 1 for parent_idx in base_parents),
        ]
        assert augmented_specs[shifted_idx] == node_feature_specs[base_idx]
    assert find_graph_u_candidates(augmented_graph, augmented_specs)[0] == 0


def test_construct_graph_u_root_repairs_single_shared_x_y_node_without_rejection():
    graph = [[], [0]]
    node_feature_specs = [
        _feature_specs("x", "y"),
        {},
    ]

    augmented_graph, augmented_specs, x_child, y_child = construct_graph_u_root(
        graph, node_feature_specs
    )

    assert x_child == 2
    assert y_child == 1
    assert any(spec.group == "x" for spec in augmented_specs[x_child].values())
    assert any(spec.group == "y" for spec in augmented_specs[y_child].values())
    assert 0 in augmented_graph[x_child]
    assert 0 in augmented_graph[y_child]
    assert sum(
        spec.group == "x" for node_specs in augmented_specs for spec in node_specs.values()
    ) == 1
    assert sum(
        spec.group == "y" for node_specs in augmented_specs for spec in node_specs.values()
    ) == 1


class _IdentityRandomFunction:
    def __init__(self, _context, _in_features: int, _out_features: int):
        pass

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        return points


def test_graph_u_forced_gaussian_applies_query_location_and_scale_deterministically(
    monkeypatch,
):
    monkeypatch.setattr(function_module, "RandomFunction", _IdentityRandomFunction)
    config = PriorConfig(
        graph_u_enabled=True,
        graph_u_query_location=2.25,
        graph_u_query_scale=1.75,
        graph_u_force_gaussian=True,
    )
    points = RandomPoints(Context(config=config))
    n_samples, n_train, n_features = 12, 5, 3

    torch.manual_seed(1729)
    expected = torch.randn(n_samples, n_features)
    expected[n_train:] = (
        config.graph_u_query_scale * expected[n_train:] + config.graph_u_query_location
    )

    torch.manual_seed(1729)
    actual = points.sample(n_samples, n_features, graph_u_n_train=n_train)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_random_points_without_graph_u_boundary_preserves_legacy_sampling(monkeypatch):
    """An enabled experiment must not shift unrelated roots."""
    monkeypatch.setattr(function_module, "RandomFunction", _IdentityRandomFunction)
    config = PriorConfig(
        graph_u_enabled=True,
        graph_u_query_location=100.0,
        graph_u_query_scale=7.0,
        graph_u_force_gaussian=True,
    )
    context = Context(config=config)
    choices: list[str] = []

    def choose_gaussian(name, _categories, _mode="meta"):
        choices.append(name)
        return RandomGaussianPoints

    monkeypatch.setattr(context.sampler, "choice", choose_gaussian)

    torch.manual_seed(81)
    expected = torch.randn(10, 2)
    torch.manual_seed(81)
    actual = RandomPoints(context).sample(10, 2)

    assert choices == ["random_base_points"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_graph_u_affine_shift_also_supports_non_gaussian_sources(monkeypatch):
    monkeypatch.setattr(function_module, "RandomFunction", _IdentityRandomFunction)
    config = PriorConfig(
        graph_u_enabled=True,
        graph_u_query_location=-0.5,
        graph_u_query_scale=2.0,
        graph_u_force_gaussian=False,
    )
    context = Context(config=config)
    monkeypatch.setattr(
        context.sampler,
        "choice",
        lambda name, _categories, _mode="meta": RandomUniformPoints,
    )

    torch.manual_seed(99)
    expected = -1 + 2 * torch.rand(8, 2)
    expected[3:] = 2.0 * expected[3:] - 0.5
    torch.manual_seed(99)
    actual = RandomPoints(context).sample(8, 2, graph_u_n_train=3)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_direct_graph_u_points_fit_root_function_on_support_only():
    def generate(query_location: float, query_scale: float) -> torch.Tensor:
        np.random.seed(91)
        torch.manual_seed(91)
        config = PriorConfig(
            graph_u_enabled=True,
            graph_u_query_location=query_location,
            graph_u_query_scale=query_scale,
            graph_u_force_gaussian=True,
            fct_types="tree",
        )
        return RandomPoints(Context(config=config)).sample(
            20, 3, graph_u_n_train=12
        )

    identity = generate(0.0, 1.0)
    shifted = generate(4.0, 2.0)

    torch.testing.assert_close(identity[:12], shifted[:12], rtol=0, atol=0)
    assert not torch.equal(identity[12:], shifted[12:])


@pytest.mark.parametrize("graph_u_n_train", [-1, 0, 8, 9])
def test_graph_u_rejects_invalid_support_query_boundary(graph_u_n_train):
    points = RandomPoints(Context(config=PriorConfig(graph_u_enabled=True)))

    with pytest.raises(ValueError, match="0 < graph_u_n_train < n_batch"):
        points.sample(8, 2, graph_u_n_train=graph_u_n_train)


def test_graph_u_boundary_cannot_activate_when_disabled():
    points = RandomPoints(Context(config=PriorConfig(graph_u_enabled=False)))

    with pytest.raises(ValueError, match="graph_u_enabled=False"):
        points.sample(8, 2, graph_u_n_train=4)


@pytest.mark.parametrize(
    "scale", [0.0, -1.0, float("nan"), float("inf"), -float("inf")]
)
def test_graph_u_rejects_nonpositive_or_nonfinite_query_scale(scale):
    with pytest.raises(
        ValueError, match="graph_u_query_scale must be finite and positive"
    ):
        PriorConfig(graph_u_query_scale=scale)


@pytest.mark.parametrize("location", [float("nan"), float("inf"), -float("inf")])
def test_graph_u_rejects_nonfinite_query_location(location):
    with pytest.raises(ValueError, match="graph_u_query_location must be finite"):
        PriorConfig(graph_u_query_location=location)


@pytest.mark.parametrize("attempts", [True, 1.5, "10"])
def test_graph_u_rejects_noninteger_max_attempts(attempts):
    with pytest.raises(TypeError, match="graph_u_max_attempts must be an integer"):
        PriorConfig(graph_u_max_attempts=attempts)


def test_graph_u_rejects_ensure_iid_and_impossible_graph_size():
    with pytest.raises(ValueError, match="does not currently support ensure_iid=True"):
        PriorConfig(graph_u_enabled=True, ensure_iid=True)

    with pytest.raises(ValueError, match="max_n_nodes >= 3"):
        PriorConfig(graph_u_enabled=True, max_n_nodes=2)


def test_graph_u_cli_config_round_trip():
    parser = argparse.ArgumentParser()
    PriorConfig.add_args_to_parser(parser)
    args = parser.parse_args(
        [
            "--graph_u_enabled",
            "true",
            "--graph_u_query_location",
            "-1.25",
            "--graph_u_query_scale",
            "2.5",
            "--graph_u_max_attempts",
            "17",
        ]
    )

    config = PriorConfig.from_args(args)

    assert config.graph_u_enabled is True
    assert config.graph_u_query_location == -1.25
    assert config.graph_u_query_scale == 2.5
    assert config.graph_u_force_gaussian is False
    assert config.graph_u_max_attempts == 17


def test_graph_u_does_not_force_gaussian_by_default():
    assert PriorConfig().graph_u_force_gaussian is False


def test_graph_u_lazy_transformers_fit_on_support_only():
    context = Context(config=PriorConfig(graph_u_enabled=True)).with_fit_boundary(4, 6)
    support = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    first = torch.cat([support, torch.tensor([[4.0], [5.0]])])
    second = torch.cat([support, torch.tensor([[4000.0], [5000.0]])])

    first_result = Standardize(context)(first)
    second_result = Standardize(
        Context(config=PriorConfig(graph_u_enabled=True)).with_fit_boundary(4, 6)
    )(second)

    torch.testing.assert_close(first_result[:4], second_result[:4], rtol=0, atol=0)
    assert not torch.equal(first_result[4:], second_result[4:])


def test_graph_u_final_preprocessing_fits_both_passes_on_support_only():
    support = torch.tensor([[0.0], [0.0], [0.0], [100.0]])
    first = torch.cat([support, torch.tensor([[0.0], [0.0]])])
    second = torch.cat([support, torch.tensor([[1000.0], [1000.0]])])

    first_clipped = outlier_removing(first, threshold=1, fit_size=4)
    second_clipped = outlier_removing(second, threshold=1, fit_size=4)
    first_scaled = standard_scaling(first_clipped, fit_size=4)
    second_scaled = standard_scaling(second_clipped, fit_size=4)

    torch.testing.assert_close(first_clipped[:4], second_clipped[:4], rtol=0, atol=0)
    torch.testing.assert_close(first_scaled[:4], second_scaled[:4], rtol=0, atol=0)
    assert not torch.equal(first_scaled[4:], second_scaled[4:])


def test_graph_u_feature_retention_is_decided_from_support_only():
    support = torch.tensor([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    first = torch.cat([support, torch.tensor([[1.0, 4.0], [1.0, 5.0]])])
    second = torch.cat([support, torch.tensor([[10.0, 4.0], [20.0, 5.0]])])
    d = torch.tensor([2])

    first_result, first_d = Prior.delete_unique_features(
        first.unsqueeze(0), d, fit_size=4
    )
    second_result, second_d = Prior.delete_unique_features(
        second.unsqueeze(0), d, fit_size=4
    )

    assert first_d.item() == second_d.item() == 1
    torch.testing.assert_close(
        first_result[0, :4], second_result[0, :4], rtol=0, atol=0
    )
    torch.testing.assert_close(first_result[0, :, 0], first[:, 1], rtol=0, atol=0)


@pytest.mark.parametrize(
    "regression,fct_types",
    [
        (False, "default"),
        (True, "default"),
        (True, "tree"),
        (True, "disc"),
        (True, "em"),
        (True, "mlp"),
        (True, "prod"),
    ],
)
@pytest.mark.parametrize("structure_mode", ["reject", "add_root"])
def test_same_seed_graph_u_shift_preserves_final_support(
    regression, fct_types, structure_mode
):
    def generate(query_location: float, query_scale: float):
        np.random.seed(7)
        torch.manual_seed(7)
        config = PriorConfig(
            graph_u_enabled=True,
            graph_u_structure_mode=structure_mode,
            graph_u_query_location=query_location,
            graph_u_query_scale=query_scale,
            graph_u_force_gaussian=True,
            graph_u_max_attempts=1000,
            min_n_nodes=3,
            max_n_nodes=8,
            fct_types=fct_types,
        )
        scm = GraphSCM(
            regression=regression,
            seq_len=48,
            train_size=32,
            num_features=4,
            max_features=4,
            num_classes=3,
            permute_features=False,
            permute_labels=False,
            config=config,
        )
        X, y = scm()
        return X, y, scm.metadata_

    identity_X, identity_y, identity_metadata = generate(0.0, 1.0)
    shifted_X, shifted_y, shifted_metadata = generate(3.0, 1.7)

    assert identity_metadata["graph"] == shifted_metadata["graph"]
    assert identity_metadata["graph_u_node_idx"] == shifted_metadata["graph_u_node_idx"]
    assert identity_metadata["graph_u_fit_policy"] == "support_only"
    torch.testing.assert_close(identity_X[:32], shifted_X[:32], rtol=0, atol=0)
    torch.testing.assert_close(identity_y[:32], shifted_y[:32], rtol=0, atol=0)
    assert not (
        torch.equal(identity_X[32:], shifted_X[32:])
        and torch.equal(identity_y[32:], shifted_y[32:])
    )


def test_graph_u_requires_graph_scm_prior_type():
    with pytest.raises(ValueError, match="prior_type='graph_scm'"):
        PriorDataset(
            prior_type="mlp_scm",
            config=PriorConfig(graph_u_enabled=True),
        )


def test_cls_sanity_does_not_mix_graph_u_environments():
    X = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1)
    y = torch.tensor([[0, 0, 1, 1]])
    X_before = X.clone()
    y_before = y.clone()

    valid = Prior.cls_sanity_check(
        X, y, train_size=2, allow_cross_split_permutation=False
    )

    assert not valid
    torch.testing.assert_close(X, X_before)
    torch.testing.assert_close(y, y_before)


def test_cls_sanity_legacy_path_can_repair_split_by_permuting_rows(monkeypatch):
    X = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1)
    y = torch.tensor([[0, 0, 1, 1]])
    permutation = torch.tensor([0, 2, 1, 3])
    monkeypatch.setattr(torch, "randperm", lambda _n: permutation)

    valid = Prior.cls_sanity_check(
        X, y, train_size=2, n_attempts=1, allow_cross_split_permutation=True
    )

    assert valid
    torch.testing.assert_close(X.squeeze(-1), torch.tensor([[0.0, 2.0, 1.0, 3.0]]))
    torch.testing.assert_close(y, torch.tensor([[0, 1, 0, 1]]))


class _FakeGraphDataset:
    def __init__(self, properties: DatasetProperties):
        self.kwargs = {
            "sentinel": "metadata",
            "n_train": properties.n_train,
            "n_test": properties.n_test,
        }
        self._dataset = Dataset(
            tensors={
                "x_0": torch.arange(
                    properties.n_train + properties.n_test, dtype=torch.float32
                ).unsqueeze(-1),
                "y_0": torch.arange(
                    properties.n_train + properties.n_test, dtype=torch.float32
                ).unsqueeze(-1),
            },
            feature_specs=properties.feature_specs,
            **self.kwargs,
        )

    def get_concat_tensors(self):
        return self._dataset.get_concat_tensors()


class _CapturingRandomDataset:
    properties: DatasetProperties | None = None

    def __init__(self, _context):
        pass

    def sample(self, properties: DatasetProperties):
        self.__class__.properties = properties
        return _FakeGraphDataset(properties)


def test_graph_scm_propagates_train_size_and_retains_metadata(monkeypatch):
    monkeypatch.setattr(graph_scm_module, "RandomDataset", _CapturingRandomDataset)
    scm = GraphSCM(
        regression=True,
        seq_len=6,
        train_size=np.int64(4),
        num_features=1,
        max_features=1,
        permute_features=False,
    )

    X, y = scm()

    assert X.shape == (6, 1)
    assert y.shape == (6,)
    assert _CapturingRandomDataset.properties is not None
    assert _CapturingRandomDataset.properties.n_train == 4
    assert _CapturingRandomDataset.properties.n_test == 2
    assert scm.metadata_["sentinel"] == "metadata"


class _QueryNaNRandomDataset:
    def __init__(self, _context):
        pass

    def sample(self, properties: DatasetProperties):
        n_samples = properties.n_train + properties.n_test
        X = torch.arange(n_samples, dtype=torch.float32).unsqueeze(-1)
        X[properties.n_train] = torch.nan
        y = torch.arange(n_samples, dtype=torch.float32).unsqueeze(-1)
        return Dataset(
            tensors={"x_0": X, "y_0": y},
            feature_specs=properties.feature_specs,
        )


def test_graph_scm_marks_query_nan_invalid_without_overwriting_support(monkeypatch):
    monkeypatch.setattr(graph_scm_module, "RandomDataset", _QueryNaNRandomDataset)
    scm = GraphSCM(
        regression=True,
        seq_len=6,
        train_size=4,
        num_features=1,
        max_features=1,
        permute_features=False,
        config=PriorConfig(graph_u_enabled=True, max_n_nodes=3),
    )

    X, y = scm()

    assert scm.invalid_generation_ is True
    assert not torch.all(X[:4] == 0)
    assert not torch.any(y[:4] == -100)
    torch.testing.assert_close(X[4:], torch.zeros(2, 1))
    torch.testing.assert_close(y[4:], torch.full((2,), -100.0))


def test_graph_prior_retries_graph_scm_marked_invalid(monkeypatch):
    class InvalidOnceGraphSCM:
        calls = 0

        def __init__(self, **_kwargs):
            self.invalid_generation_ = False

        def __call__(self):
            self.__class__.calls += 1
            self.invalid_generation_ = self.__class__.calls == 1
            X = torch.arange(12, dtype=torch.float32).reshape(6, 2)
            y = torch.arange(6, dtype=torch.float32)
            if self.invalid_generation_:
                X[4:] = 0
                y[4:] = -100
            return X, y

    monkeypatch.setattr(prior_dataset_module, "GraphSCM", InvalidOnceGraphSCM)
    InvalidOnceGraphSCM.calls = 0
    config = PriorConfig(graph_u_enabled=True, max_n_nodes=3)
    prior = GraphPrior(
        regression=True,
        config=config,
        n_jobs=1,
        generation_max_attempts=2,
    )
    params = {
        "regression": True,
        "seq_len": 6,
        "train_size": 4,
        "num_features": 2,
        "max_features": 2,
        "num_classes": None,
        "device": "cpu",
        "config": config,
    }

    X, y, d = prior.generate_dataset(params)

    assert InvalidOnceGraphSCM.calls == 2
    assert d.item() == 2
    torch.testing.assert_close(X, torch.arange(12, dtype=torch.float32).reshape(6, 2))
    torch.testing.assert_close(y, torch.arange(6, dtype=torch.float32))


def test_graph_prior_bounds_outer_retries_and_preserves_environment_boundary(
    monkeypatch,
):
    class AlwaysInvalidGraphSCM:
        calls = 0

        def __init__(self, **_kwargs):
            pass

        def __call__(self):
            self.__class__.calls += 1
            X = torch.arange(8, dtype=torch.float32).reshape(4, 2)
            y = torch.tensor([0, 0, 1, 1])
            return X, y

    config = PriorConfig(graph_u_enabled=True, graph_u_max_attempts=2, max_n_nodes=3)
    prior = GraphPrior(config=config, n_jobs=1)
    allow_permutation: list[bool] = []

    def reject_split(
        _X, _y, _train_size, *, allow_cross_split_permutation=True, **_kwargs
    ):
        allow_permutation.append(allow_cross_split_permutation)
        return False

    monkeypatch.setattr(prior_dataset_module, "GraphSCM", AlwaysInvalidGraphSCM)
    monkeypatch.setattr(prior, "cls_sanity_check", reject_split)
    params = {
        "regression": False,
        "seq_len": 4,
        "train_size": 2,
        "num_features": 2,
        "max_features": 2,
        "num_classes": 2,
        "device": "cpu",
        "config": config,
    }

    with pytest.raises(
        RuntimeError,
        match="Unable to generate a valid Graph-U dataset after 2 attempts",
    ):
        prior.generate_dataset(params)

    assert AlwaysInvalidGraphSCM.calls == 2
    assert allow_permutation == [False, False]


def test_graph_u_optional_dataset_filter_receives_support_only(monkeypatch):
    class FixedGraphSCM:
        def __init__(self, **_kwargs):
            pass

        def __call__(self):
            X = torch.tensor(
                [
                    [0.0, 0.0],
                    [1.0, 1.0],
                    [2.0, 2.0],
                    [3.0, 3.0],
                    [1000.0, 1000.0],
                    [2000.0, 2000.0],
                ]
            )
            y = torch.arange(6, dtype=torch.float32)
            return X, y

    seen: list[tuple[torch.Tensor, torch.Tensor]] = []

    def record_filter(X, y, *_args, **_kwargs):
        seen.append((X.clone(), y.clone()))
        return False

    monkeypatch.setattr(prior_dataset_module, "GraphSCM", FixedGraphSCM)
    monkeypatch.setattr(prior_dataset_module, "should_filter", record_filter)
    config = PriorConfig(graph_u_enabled=True, max_n_nodes=3)
    prior = GraphPrior(regression=True, config=config, n_jobs=1)
    params = {
        "regression": True,
        "seq_len": 6,
        "train_size": 4,
        "num_features": 2,
        "max_features": 2,
        "num_classes": None,
        "device": "cpu",
        "config": config,
    }

    prior.generate_dataset(params)

    assert len(seen) == 1
    assert seen[0][0].shape == (4, 2)
    assert seen[0][1].shape == (4,)
    torch.testing.assert_close(seen[0][0], FixedGraphSCM()()[0][:4])
    torch.testing.assert_close(seen[0][1], FixedGraphSCM()()[1][:4])


def test_evaluator_can_bound_ordinary_graph_prior_retries(monkeypatch):
    class AlwaysInvalidGraphSCM:
        calls = 0

        def __init__(self, **_kwargs):
            pass

        def __call__(self):
            self.__class__.calls += 1
            X = torch.arange(8, dtype=torch.float32).reshape(4, 2)
            y = torch.tensor([0, 0, 1, 1])
            return X, y

    config = PriorConfig(graph_u_enabled=False)
    prior = GraphPrior(
        config=config,
        n_jobs=1,
        generation_max_attempts=2,
    )
    allow_permutation: list[bool] = []

    def reject_split(
        _X, _y, _train_size, *, allow_cross_split_permutation=True, **_kwargs
    ):
        allow_permutation.append(allow_cross_split_permutation)
        return False

    monkeypatch.setattr(prior_dataset_module, "GraphSCM", AlwaysInvalidGraphSCM)
    monkeypatch.setattr(prior, "cls_sanity_check", reject_split)
    params = {
        "regression": False,
        "seq_len": 4,
        "train_size": 2,
        "num_features": 2,
        "max_features": 2,
        "num_classes": 2,
        "device": "cpu",
        "config": config,
    }

    with pytest.raises(
        RuntimeError,
        match="Unable to generate a valid dataset after 2 attempts",
    ):
        prior.generate_dataset(params)

    assert AlwaysInvalidGraphSCM.calls == 2
    assert allow_permutation == [True, True]


def test_random_dataset_bounds_graph_u_dag_rejection(monkeypatch):
    class NeverConfoundedDAG:
        calls = 0

        def __init__(self, _context):
            pass

        def sample(self, n_nodes: int) -> list[list[int]]:
            self.__class__.calls += 1
            return [[] for _ in range(n_nodes)]

    monkeypatch.setattr(dataset_module, "RandomDAG", NeverConfoundedDAG)
    NeverConfoundedDAG.calls = 0
    config = PriorConfig(
        graph_u_enabled=True,
        graph_u_max_attempts=3,
        min_n_nodes=3,
        max_n_nodes=3,
    )
    properties = DatasetProperties(n_train=4, n_test=2, cat_sizes={"x": [0], "y": [0]})

    with pytest.raises(RuntimeError, match="after 3 attempts"):
        RandomDataset(Context(config=config)).sample(properties)

    assert NeverConfoundedDAG.calls == 3


class _FixedDAG:
    def __init__(self, _context):
        pass

    def sample(self, n_nodes: int) -> list[list[int]]:
        assert n_nodes == 3
        return [[], [0], [0]]


class _RecordingGraphFunction:
    calls: list[dict[str, Any]] = []

    def __init__(
        self,
        _context,
        dag,
        node_feature_specs,
        *,
        graph_u_node_idx=None,
        n_train=None,
    ):
        self.node_feature_specs = node_feature_specs
        self.__class__.calls.append(
            {
                "dag": dag,
                "node_feature_specs": node_feature_specs,
                "graph_u_node_idx": graph_u_node_idx,
                "n_train": n_train,
            }
        )

    def __call__(self, n_samples: int) -> dict[str, torch.Tensor]:
        return {
            name: torch.zeros(n_samples, 1)
            for node_specs in self.node_feature_specs
            for name in node_specs
        }


def test_random_dataset_records_selected_graph_u_and_shift_config(monkeypatch):
    monkeypatch.setattr(dataset_module, "RandomDAG", _FixedDAG)
    monkeypatch.setattr(dataset_module, "RandomGraphFunction", _RecordingGraphFunction)
    _RecordingGraphFunction.calls.clear()
    feature_placements = iter([np.array([1]), np.array([2])])

    def controlled_choice(values, *args, **kwargs):
        if kwargs.get("size") is not None:
            return next(feature_placements)
        return np.asarray(values)[0]

    monkeypatch.setattr(dataset_module.np.random, "choice", controlled_choice)
    config = PriorConfig(
        graph_u_enabled=True,
        graph_u_query_location=1.5,
        graph_u_query_scale=0.75,
        graph_u_force_gaussian=True,
        graph_u_max_attempts=5,
        min_n_nodes=2,
        max_n_nodes=3,
        subsample_feature_nodes=False,
    )
    properties = DatasetProperties(n_train=4, n_test=2, cat_sizes={"x": [0], "y": [0]})

    dataset = RandomDataset(Context(config=config)).sample(properties)

    call = _RecordingGraphFunction.calls[-1]
    x_child = next(
        idx
        for idx, specs in enumerate(call["node_feature_specs"])
        if any(spec.group == "x" for spec in specs.values())
    )
    y_child = next(
        idx
        for idx, specs in enumerate(call["node_feature_specs"])
        if any(spec.group == "y" for spec in specs.values())
    )
    assert call["graph_u_node_idx"] == 0
    assert call["n_train"] == properties.n_train
    assert dataset.kwargs["graph_u_enabled"] is True
    assert dataset.kwargs["graph_u_node_idx"] == 0
    assert dataset.kwargs["graph_u_candidate_node_idxs"] == (0,)
    assert dataset.kwargs["graph_u_x_child_node_idxs"] == (x_child,)
    assert dataset.kwargs["graph_u_y_child_node_idxs"] == (y_child,)
    assert dataset.kwargs["graph_u_attempts"] == 1
    assert dataset.kwargs["graph_u_config"]["graph_u_query_location"] == 1.5
    assert dataset.kwargs["graph_u_config"]["graph_u_query_scale"] == 0.75
    assert dataset.kwargs["n_train"] == properties.n_train
    assert dataset.kwargs["n_test"] == properties.n_test


class _FixedTwoNodeDAG:
    calls = 0

    def __init__(self, _context):
        pass

    def sample(self, n_nodes: int) -> list[list[int]]:
        self.__class__.calls += 1
        assert n_nodes == 2
        return [[], [0]]


def test_constructed_graph_u_handles_94_features_with_one_dag_proposal(monkeypatch):
    monkeypatch.setattr(dataset_module, "RandomDAG", _FixedTwoNodeDAG)
    monkeypatch.setattr(dataset_module, "RandomGraphFunction", _RecordingGraphFunction)
    _FixedTwoNodeDAG.calls = 0
    _RecordingGraphFunction.calls.clear()

    original_choice = dataset_module.np.random.choice

    def place_all_features_on_first_node(values, *args, **kwargs):
        size = kwargs.get("size")
        if size is not None:
            return np.zeros(size, dtype=int)
        return original_choice(values, *args, **kwargs)

    monkeypatch.setattr(dataset_module.np.random, "choice", place_all_features_on_first_node)
    config = PriorConfig(
        graph_u_enabled=True,
        graph_u_structure_mode="add_root",
        graph_u_max_attempts=1,
        min_n_nodes=2,
        max_n_nodes=2,
        subsample_feature_nodes=False,
        filter_unpredictable_graphs=True,
    )
    properties = DatasetProperties(
        n_train=4,
        n_test=2,
        cat_sizes={"x": [0] * 94, "y": [2]},
    )

    dataset = RandomDataset(Context(config=config)).sample(properties)

    assert _FixedTwoNodeDAG.calls == 1
    assert dataset.kwargs["graph_u_attempts"] == 1
    assert dataset.kwargs["graph_u_structure_mode"] == "add_root"
    assert dataset.kwargs["graph_u_node_idx"] == 0
    assert dataset.kwargs["graph_u_candidate_node_idxs"] == (0,)
    assert dataset.kwargs["graph_u_base_n_nodes"] == 2
    assert len(dataset.kwargs["graph"]) == 3
    assert len(dataset.tensors) == 95

    call = _RecordingGraphFunction.calls[-1]
    assert call["graph_u_node_idx"] == 0
    assert call["node_feature_specs"][0] == {}
    assert call["graph_u_node_idx"] in call["dag"][dataset.kwargs["graph_u_constructed_x_child_node_idx"]]
    assert call["graph_u_node_idx"] in call["dag"][dataset.kwargs["graph_u_constructed_y_child_node_idx"]]


class _DeterministicNodeFunction:
    boundaries: list[int | None] = []
    fit_boundaries: list[tuple[int | None, int | None]] = []

    def __init__(self, _context, feature_specs, *, graph_u_n_train=None):
        self.feature_specs = feature_specs
        self.graph_u_n_train = graph_u_n_train
        self.__class__.boundaries.append(graph_u_n_train)
        self.__class__.fit_boundaries.append(
            (_context.fit_n_samples, _context.total_n_samples)
        )

    def __call__(self, parents: list[torch.Tensor], n_samples: int):
        if not parents:
            values = torch.arange(n_samples, dtype=torch.float32).unsqueeze(-1)
            if self.graph_u_n_train is not None:
                values[self.graph_u_n_train :] += 100
        else:
            values = torch.stack(parents).sum(dim=0) + 1
        return values, {name: values.clone() for name in self.feature_specs}


def test_graph_u_source_shift_propagates_through_all_descendants(monkeypatch):
    monkeypatch.setattr(
        graph_function_module, "RandomNodeFunction", _DeterministicNodeFunction
    )
    _DeterministicNodeFunction.boundaries.clear()
    _DeterministicNodeFunction.fit_boundaries.clear()
    graph = [[], [0], [0], [2]]
    node_feature_specs = [
        {},
        _feature_specs("x"),
        _feature_specs("y"),
        {"x_descendant": FeatureSpec(group="x")},
    ]
    graph_function = RandomGraphFunction(
        Context(config=PriorConfig(graph_u_enabled=True)),
        dag=graph,
        node_feature_specs=node_feature_specs,
        graph_u_node_idx=0,
        n_train=2,
    )

    features = graph_function(4)

    assert _DeterministicNodeFunction.boundaries == [2, None, None, None]
    assert _DeterministicNodeFunction.fit_boundaries == [(2, 4)] * 4
    torch.testing.assert_close(
        features["x_0"].squeeze(-1), torch.tensor([1.0, 2.0, 103.0, 104.0])
    )
    torch.testing.assert_close(
        features["y_0"].squeeze(-1), torch.tensor([1.0, 2.0, 103.0, 104.0])
    )
    torch.testing.assert_close(
        features["x_descendant"].squeeze(-1),
        torch.tensor([2.0, 3.0, 104.0, 105.0]),
    )

    with pytest.raises(RuntimeError, match="cannot be evaluated twice"):
        graph_function(4)


@pytest.mark.parametrize(
    "graph,node_feature_specs,node_idx,error",
    [
        (
            [[], [0], [0]],
            [{}, _feature_specs("x"), _feature_specs("y")],
            True,
            "integer",
        ),
        (
            [[], [0], [0]],
            [_feature_specs("x"), _feature_specs("x"), _feature_specs("y")],
            0,
            "hidden",
        ),
        ([[1], [], [1]], [{}, {}, _feature_specs("y")], 0, "root"),
        ([[], [0]], [{}, _feature_specs("x", "y")], 0, "distinct observed X and Y"),
    ],
)
def test_graph_function_rejects_invalid_selected_u(
    graph, node_feature_specs, node_idx, error
):
    with pytest.raises((TypeError, ValueError), match=error):
        RandomGraphFunction(
            Context(config=PriorConfig(graph_u_enabled=True)),
            dag=graph,
            node_feature_specs=node_feature_specs,
            graph_u_node_idx=node_idx,
            n_train=2,
        )
