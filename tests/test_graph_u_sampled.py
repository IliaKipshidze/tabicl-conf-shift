"""Per-task Graph-U shift sampling without changing the fixed-shift path."""

from __future__ import annotations

import argparse

import numpy as np
import pytest
import torch

from tabicl.prior._graph_scm import GraphSCM
from tabicl.prior.graph_lib import _function as function_module
from tabicl.prior.graph_lib import _points as points_module
from tabicl.prior.graph_lib._base import Context
from tabicl.prior.graph_lib._config import PriorConfig
from tabicl.prior.graph_lib._points import RandomGaussianPoints, RandomPoints
from tabicl.prior.graph_lib._shift import sample_graph_u_query_shift


class _IdentityFunction:
    def __init__(self, *_args):
        pass

    def __call__(self, values):
        return values


def test_fixed_shift_default_consumes_no_numpy_or_torch_randomness():
    config = PriorConfig(graph_u_query_scale=1.75, graph_u_query_location=-2.0)
    assert config.graph_u_shift_mode == "fixed"
    np.random.seed(41)
    torch.manual_seed(41)
    expected_numpy, expected_torch = np.random.rand(8), torch.rand(8)
    np.random.seed(41)
    torch.manual_seed(41)
    assert sample_graph_u_query_shift(config) == (1.75, -2.0)
    np.testing.assert_array_equal(np.random.rand(8), expected_numpy)
    torch.testing.assert_close(torch.rand(8), expected_torch, rtol=0, atol=0)


def test_shared_prior_cli_exposes_sampling_settings_and_keeps_fixed_default():
    parser = argparse.ArgumentParser()
    PriorConfig.add_args_to_parser(parser)
    assert PriorConfig.from_args(parser.parse_args([])).graph_u_shift_mode == "fixed"
    overrides = {
        "graph_u_shift_mode": "sampled",
        "graph_u_query_scale_mean": 1.6,
        "graph_u_query_scale_std": 0.04,
        "graph_u_query_scale_min": 1.5,
        "graph_u_query_scale_max": 1.7,
        "graph_u_query_location_mean": 2.1,
        "graph_u_query_location_std": 0.08,
        "graph_u_query_location_min": 1.9,
        "graph_u_query_location_max": 2.4,
    }
    arguments = [part for key, value in overrides.items() for part in (f"--{key}", str(value))]
    config = PriorConfig.from_args(parser.parse_args(arguments))
    assert {key: getattr(config, key) for key in overrides} == overrides


def test_sampled_parameters_are_bounded_concentrated_independent_and_reproducible():
    config = PriorConfig(graph_u_shift_mode="sampled")
    np.random.seed(715)
    draws = np.array([sample_graph_u_query_shift(config) for _ in range(2048)])
    np.random.seed(715)
    repeated = np.array([sample_graph_u_query_shift(config) for _ in range(2048)])
    np.testing.assert_array_equal(draws, repeated)
    scales, locations = draws.T
    assert np.all((1.3 < scales) & (scales < 1.7))
    assert np.all((1.7 < locations) & (locations < 2.3))
    assert abs(scales.mean() - 1.5) < 0.01
    assert abs(locations.mean() - 2.0) < 0.02
    # A clipped Gaussian would have mass at the bounds. A uniform would have
    # equal density in these equally wide central and outer regions.
    for values, mean, parent_std in ((scales, 1.5, 0.1), (locations, 2.0, 0.15)):
        distance = abs(values - mean) / parent_std
        assert np.count_nonzero(distance < 0.5) > 2 * np.count_nonzero(distance > 1.5)
    assert abs(np.corrcoef(scales, locations)[0, 1]) < 0.1


def test_sampling_honors_custom_distribution_parameters_and_ignores_fixed_values():
    config = PriorConfig(
        graph_u_shift_mode="sampled",
        graph_u_query_scale=100,
        graph_u_query_location=100,
        graph_u_query_scale_mean=2.5,
        graph_u_query_scale_std=0.01,
        graph_u_query_scale_min=2.4,
        graph_u_query_scale_max=2.6,
        graph_u_query_location_mean=-2,
        graph_u_query_location_std=0.02,
        graph_u_query_location_min=-2.1,
        graph_u_query_location_max=-1.9,
    )
    np.random.seed(30)
    scales, locations = np.array([sample_graph_u_query_shift(config) for _ in range(128)]).T
    assert np.all((2.4 <= scales) & (scales <= 2.6))
    assert np.all((-2.1 <= locations) & (locations <= -1.9))
    assert abs(scales.mean() - 2.5) < 0.005
    assert abs(locations.mean() + 2) < 0.01


@pytest.mark.parametrize(
    "kwargs",
    [
        {"graph_u_shift_mode": "unknown"},
        {"graph_u_query_scale_std": 0},
        {"graph_u_query_scale_std": -0.1},
        {"graph_u_query_location_std": float("nan")},
        {"graph_u_query_location_std": float("inf")},
        {"graph_u_query_scale_min": 0},
        {"graph_u_query_scale_min": -1},
        {"graph_u_query_scale_min": 1.7},
        {"graph_u_query_scale_max": 1.3},
        {"graph_u_query_location_min": 2.3},
        {"graph_u_query_location_max": 1.7},
        {"graph_u_query_scale_mean": float("nan")},
        {"graph_u_query_scale_mean": 1.8},
        {"graph_u_query_location_mean": float("inf")},
        {"graph_u_query_location_mean": 1.0},
        {"graph_u_query_scale_min": float("nan")},
        {"graph_u_query_location_max": float("inf")},
    ],
)
def test_invalid_sampling_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        PriorConfig(**{"graph_u_shift_mode": "sampled", **kwargs})


@pytest.mark.parametrize("force_gaussian", [False, True])
def test_one_pair_per_task_is_shared_across_query_rows_and_source_coordinates(
    monkeypatch, force_gaussian
):
    monkeypatch.setattr(function_module, "RandomFunction", _IdentityFunction)
    config = PriorConfig(
        graph_u_enabled=True,
        graph_u_shift_mode="sampled",
        graph_u_force_gaussian=force_gaussian,
        meta_sampling_mode="global",
    )
    context = Context(config=config)
    if not force_gaussian:
        monkeypatch.setattr(
            context.sampler, "choice", lambda *_args: points_module.RandomUniformPoints
        )
    sampled_pairs = []

    def capture_pair(prior_config):
        pair = sample_graph_u_query_shift(prior_config)
        sampled_pairs.append(pair)
        return pair

    monkeypatch.setattr(points_module, "sample_graph_u_query_shift", capture_pair)
    np.random.seed(21)
    # Sharing the meta-context must not turn the shift into one global draw.
    for _ in range(3):
        torch.manual_seed(91)
        base = torch.randn(12, 4) if force_gaussian else -1 + 2 * torch.rand(12, 4)
        torch.manual_seed(91)
        points = RandomPoints(context)
        actual = points.sample(12, 4, graph_u_n_train=5)
        scale, location = sampled_pairs[-1]
        assert points.graph_u_query_scale_ == scale
        assert points.graph_u_query_location_ == location
        torch.testing.assert_close(actual[:5], base[:5], rtol=0, atol=0)
        torch.testing.assert_close(actual[5:], scale * base[5:] + location, rtol=0, atol=0)
    assert len(sampled_pairs) == 3
    assert len(set(sampled_pairs)) == 3


def test_sampling_is_not_applied_to_other_root_sources(monkeypatch):
    monkeypatch.setattr(function_module, "RandomFunction", _IdentityFunction)
    context = Context(config=PriorConfig(graph_u_enabled=True, graph_u_shift_mode="sampled"))
    monkeypatch.setattr(context.sampler, "choice", lambda *_args: RandomGaussianPoints)

    def unexpected_sampling(_config):
        raise AssertionError("An ordinary root must not sample Graph-U shift parameters")

    monkeypatch.setattr(points_module, "sample_graph_u_query_shift", unexpected_sampling)
    torch.manual_seed(45)
    expected = torch.randn(12, 3)
    torch.manual_seed(45)
    actual = RandomPoints(context).sample(12, 3)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["confounding", "covariate"])
def test_actual_tasks_record_realized_shift_and_keep_u_hidden(monkeypatch, mode):
    config = PriorConfig(
        graph_u_enabled=True,
        graph_u_structure_mode="add_root",
        graph_u_mode=mode,
        graph_u_shift_mode="sampled",
        min_n_nodes=3,
        max_n_nodes=5,
        fct_types="lin",
    )
    actual_pairs = []

    def capture_pair(prior_config):
        pair = sample_graph_u_query_shift(prior_config)
        actual_pairs.append(pair)
        return pair

    monkeypatch.setattr(points_module, "sample_graph_u_query_shift", capture_pair)
    np.random.seed(7)
    torch.manual_seed(7)
    recorded = []
    for _ in range(3):
        scm = GraphSCM(
            regression=True,
            seq_len=24,
            train_size=16,
            num_features=2,
            max_features=2,
            config=config,
        )
        x, y = scm()
        assert x.shape == (24, 2)
        assert y.shape == (24,)
        assert torch.isfinite(x).all() and torch.isfinite(y).all()
        metadata = scm.metadata_
        pair = (metadata["graph_u_query_scale"], metadata["graph_u_query_location"])
        recorded.append(pair)
        assert pair == actual_pairs[-1]
        assert metadata["graph_u_shift_mode"] == "sampled"
        assert metadata["graph_u_config"]["graph_u_shift_mode"] == "sampled"
        assert metadata["graph_u_node_idx"] == 0
        assert metadata["graph_u_fit_policy"] == "support_only"
        assert bool(metadata["graph_u_y_child_node_idxs"]) == (mode == "confounding")
        assert not any(isinstance(value, torch.Tensor) for value in metadata.values())
    assert len(actual_pairs) == 3
    assert len(set(recorded)) == 3
