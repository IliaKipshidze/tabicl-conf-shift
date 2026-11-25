"""Common-support resampling of final hidden-U vectors before propagation."""

from __future__ import annotations

import json
import random

import numpy as np
import pytest
import torch

from tabicl.prior._graph_scm import GraphSCM
from tabicl.prior.graph_lib import _function as function_module
from tabicl.prior.graph_lib import _graph_function as graph_function_module
from tabicl.prior.graph_lib import _points as points_module
from tabicl.prior.graph_lib._base import Context, FeatureSpec
from tabicl.prior.graph_lib._config import PriorConfig
from tabicl.prior.graph_lib._graph_function import RandomGraphFunction
from tabicl.prior.graph_lib._pool import (
    graph_u_pool_probabilities,
    resample_graph_u_pool,
)


def _config(**overrides):
    return PriorConfig(
        **{
            "graph_u_enabled": True,
            "graph_u_structure_mode": "add_root",
            "graph_u_intervention_mode": "support_pool",
            **overrides,
        }
    )


def _assert_query_in_support(values, n_support):
    matches = (values[n_support:, None, :] == values[None, :n_support, :]).all(-1)
    assert matches.any(dim=1).all()


@pytest.mark.parametrize("n_support", [1, 3, 32])
def test_uniform_pool_probabilities_are_exact_and_do_not_draw_randomness(n_support):
    config = _config(graph_u_pool_sampling="uniform")
    torch.manual_seed(19)
    before = torch.get_rng_state().clone()
    probabilities = graph_u_pool_probabilities(n_support, config)
    assert torch.equal(torch.get_rng_state(), before)
    assert probabilities.dtype == torch.float64
    torch.testing.assert_close(
        probabilities, torch.full((n_support,), 1 / n_support, dtype=torch.float64),
        rtol=0, atol=0,
    )


def test_gaussian_weights_use_original_index_percentile_and_relative_width():
    config = _config(graph_u_pool_peak_percentile=80, graph_u_pool_std_fraction=0.2)
    actual = graph_u_pool_probabilities(11, config)
    indices = np.arange(11)
    weights = np.exp(-0.5 * ((indices - 8) / 2.2) ** 2)
    np.testing.assert_allclose(actual.numpy(), weights / weights.sum(), rtol=1e-12)
    assert actual.argmax().item() == 8
    # The width is a fraction of pool size, not a fixed count of rows.
    larger = graph_u_pool_probabilities(101, config)
    assert larger.argmax().item() == 80
    assert larger[60] / larger[80] > actual[0] / actual[8]


@pytest.mark.parametrize("percentile, expected", [(0, 0), (100, 8)])
def test_pool_peak_percentile_reaches_both_endpoints(percentile, expected):
    config = _config(graph_u_pool_peak_percentile=percentile)
    probabilities = graph_u_pool_probabilities(9, config)
    assert probabilities.argmax().item() == expected
    assert torch.isfinite(probabilities).all()
    assert probabilities.sum().item() == pytest.approx(1)


@pytest.mark.parametrize("width", [1e-250, 1e250])
def test_extreme_pool_widths_keep_probabilities_finite(width):
    probabilities = graph_u_pool_probabilities(
        20, _config(graph_u_pool_peak_percentile=75, graph_u_pool_std_fraction=width)
    )
    assert torch.isfinite(probabilities).all()
    assert (probabilities >= 0).all()
    assert probabilities.sum().item() == pytest.approx(1)
    if width < 1:
        assert probabilities.argmax().item() == 14
    else:
        torch.testing.assert_close(
            probabilities, torch.full((20,), 1 / 20, dtype=torch.float64), rtol=0, atol=0
        )


def test_pool_preserves_original_order_and_samples_complete_vectors():
    support = torch.tensor([[8.0, -8.0], [-9.0, 9.0], [3.0, -3.0]])
    original = support.clone()
    config = _config(graph_u_pool_peak_percentile=0, graph_u_pool_std_fraction=0.001)
    combined, metadata = resample_graph_u_pool(support, 40, config)
    torch.testing.assert_close(support, original, rtol=0, atol=0)
    torch.testing.assert_close(combined[:3], original, rtol=0, atol=0)
    # Index zero is selected, regardless of where its numerical U values rank.
    torch.testing.assert_close(combined[3:], original[:1].expand(40, -1), rtol=0, atol=0)
    _assert_query_in_support(combined, 3)
    assert metadata["graph_u_pool_size"] == 3
    assert metadata["graph_u_pool_constant"] is False
    assert metadata["graph_u_pool_reweighted"] is True
    assert metadata["graph_u_pool_total_variation"] > 0.6
    json.dumps(metadata)


def test_query_sampling_is_reproducible_with_replacement_and_changes_frequencies():
    support = torch.tensor([[17.0, 31.0], [-2.0, 7.0], [4.0, -9.0]])
    config = _config(graph_u_pool_peak_percentile=100, graph_u_pool_std_fraction=0.2)
    torch.manual_seed(72)
    first, metadata = resample_graph_u_pool(support, 6000, config)
    torch.manual_seed(72)
    second, repeated_metadata = resample_graph_u_pool(support, 6000, config)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert metadata == repeated_metadata
    _assert_query_in_support(first, 3)
    last_row_frequency = (first[3:] == support[-1]).all(-1).float().mean().item()
    assert last_row_frequency > 0.75
    assert len(torch.unique(first[3:], dim=0)) <= len(support)


def test_constant_pool_is_reported_as_no_effective_u_reweighting():
    support = torch.tensor([[4.0, -2.0]]).expand(8, -1).clone()
    combined, metadata = resample_graph_u_pool(support, 12, _config())
    torch.testing.assert_close(combined, support[:1].expand(20, -1), rtol=0, atol=0)
    assert metadata["graph_u_pool_constant"] is True
    assert metadata["graph_u_pool_unique_support_values"] == 1
    assert metadata["graph_u_pool_total_variation"] == 0
    assert metadata["graph_u_pool_reweighted"] is False


@pytest.mark.parametrize("family", [
    points_module.RandomUniformPoints, points_module.RandomCirclePoints,
    points_module.RandomGaussianPoints, points_module.RandomCovariancePoints,
])
def test_pool_root_keeps_source_families_without_sampling_affine_parameters(monkeypatch, family):
    class Identity:
        def __init__(self, *_args):
            pass

        def __call__(self, values):
            return values

    def unexpected_shift(_config):
        raise AssertionError("Pool mode must not sample affine parameters")

    monkeypatch.setattr(function_module, "RandomFunction", Identity)
    monkeypatch.setattr(points_module, "sample_graph_u_query_shift", unexpected_shift)
    config = _config(graph_u_shift_mode="sampled", graph_u_query_location=100,
                     graph_u_query_scale=40)
    context = Context(config=config, seed=17)
    original_choice = context.sampler.choice

    def choose_source(name, categories, *args, **kwargs):
        if name == "random_base_points":
            return family
        return original_choice(name, categories, *args, **kwargs)

    monkeypatch.setattr(context.sampler, "choice", choose_source)
    source_calls = []
    original_sample = family.sample

    def capture_source(self, rows, features):
        source_calls.append(rows)
        return original_sample(self, rows, features)

    monkeypatch.setattr(family, "sample", capture_source)
    values = points_module.RandomPoints(context).sample(12, 3, graph_u_n_train=12)
    assert values.shape == (12, 3)
    assert torch.isfinite(values).all()
    assert source_calls == [12]


@pytest.mark.parametrize("mode", ["confounding", "covariate"])
def test_pool_vectors_reach_descendants_without_resampling_unrelated_roots(monkeypatch, mode):
    support = torch.tensor([[9.0, -4.0], [-2.0, 5.0], [7.0, 11.0], [0.0, 1.0]])
    calls = []

    class Node:
        def __init__(self, context, feature_specs, *, graph_u_n_train=None):
            self.context, self.feature_specs = context, feature_specs
            self.boundary = graph_u_n_train

        def __call__(self, parents, n_samples):
            calls.append((self.boundary, n_samples, self.context.fit_n_samples,
                          self.context.total_n_samples))
            if self.boundary is not None:
                assert n_samples == len(support)
                values = support.clone()
            elif not parents:
                values = -2 * torch.arange(n_samples, dtype=torch.float32)[:, None]
            else:
                values = sum(parents) + 1
            return values, {name: values.clone() for name in self.feature_specs}

    chosen = torch.tensor([3, 0, 1, 3, 2, 0])
    monkeypatch.setattr(graph_function_module, "RandomNodeFunction", Node)
    monkeypatch.setattr(torch, "multinomial", lambda _p, count, replacement: chosen.clone())
    graph = RandomGraphFunction(
        Context(config=_config(graph_u_mode=mode)),
        dag=[[], [], [0, 1], [0] if mode == "confounding" else [], [2]],
        node_feature_specs=[{}, {"other": FeatureSpec(group="x")},
                            {"x": FeatureSpec(group="x")}, {"y": FeatureSpec(group="y")},
                            {"descendant": FeatureSpec(group="x")}],
        graph_u_node_idx=0, n_train=4,
    )
    features = graph(10)
    final_u = torch.cat([support, support[chosen]])
    other = -2 * torch.arange(10, dtype=torch.float32)[:, None]
    torch.testing.assert_close(features["other"], other)
    torch.testing.assert_close(features["x"], final_u + other + 1)
    torch.testing.assert_close(features["descendant"], final_u + other + 2)
    torch.testing.assert_close(features["y"], final_u + 1 if mode == "confounding" else other)
    assert calls[0] == (4, 4, 4, 4)
    assert all(call == (None, 10, 4, 10) for call in calls[1:])


@pytest.mark.parametrize("mode", ["confounding", "covariate"])
@pytest.mark.parametrize("add_noise", [False, True])
def test_actual_graph_scm_uses_final_u_pool_and_keeps_u_hidden(monkeypatch, mode, add_noise):
    captured = []
    actual_resample = graph_function_module.resample_graph_u_pool

    def capture(support, n_query, config):
        result, metadata = actual_resample(support, n_query, config)
        captured.append((support.clone(), result.clone()))
        return result, metadata

    monkeypatch.setattr(graph_function_module, "resample_graph_u_pool", capture)
    np.random.seed(7)
    torch.manual_seed(7)
    scm = GraphSCM(
        regression=True, seq_len=24, train_size=16, num_features=2, max_features=2,
        config=_config(graph_u_mode=mode, min_n_nodes=3, max_n_nodes=5,
                       fct_types="lin", add_gaussian_noise=add_noise),
    )
    x, y = scm()
    assert x.shape == (24, 2) and y.shape == (24,)
    assert torch.isfinite(x).all() and torch.isfinite(y).all()
    assert len(captured) == 1
    support, final_u = captured[0]
    assert len(support) == 16 and len(final_u) == 24
    torch.testing.assert_close(final_u[:16], support, rtol=0, atol=0)
    _assert_query_in_support(final_u, 16)
    assert scm.metadata_["graph_u_pool_size"] == 16
    assert scm.metadata_["graph_u_intervention_mode"] == "support_pool"
    assert bool(scm.metadata_["graph_u_y_child_node_idxs"]) == (mode == "confounding")
    assert not any(isinstance(value, torch.Tensor) for value in scm.metadata_.values())
    json.dumps({name: value for name, value in scm.metadata_.items()
                if name.startswith("graph_u_pool_")})


def _seeded_scm(config):
    np.random.seed(7)
    torch.manual_seed(7)
    random.seed(7)
    scm = GraphSCM(
        regression=True, seq_len=24, train_size=16, num_features=2, max_features=2,
        config=config,
    )
    x, y = scm()
    return x, y, scm.metadata_, torch.get_rng_state(), np.random.get_state(), random.getstate()


@pytest.mark.parametrize("mode", ["confounding", "covariate"])
def test_uniform_and_gaussian_pool_modes_keep_final_support_identical(mode):
    common = {"graph_u_mode": mode, "min_n_nodes": 3, "max_n_nodes": 5, "fct_types": "lin"}
    uniform = _seeded_scm(_config(**common, graph_u_pool_sampling="uniform"))
    gaussian = _seeded_scm(_config(**common, graph_u_pool_sampling="gaussian"))
    assert uniform[2]["graph"] == gaussian[2]["graph"]
    torch.testing.assert_close(uniform[0][:16], gaussian[0][:16], rtol=0, atol=0)
    torch.testing.assert_close(uniform[1][:16], gaussian[1][:16], rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["confounding", "covariate"])
def test_pool_generation_and_rng_ignore_all_affine_shift_parameters(mode):
    common = {"graph_u_mode": mode, "min_n_nodes": 3, "max_n_nodes": 5, "fct_types": "lin"}
    default = _seeded_scm(_config(**common))
    ignored = _seeded_scm(_config(
        **common,
        graph_u_shift_mode="sampled",
        graph_u_query_location=1e8,
        graph_u_query_scale=77,
        graph_u_query_scale_mean=4,
        graph_u_query_scale_std=0.04,
        graph_u_query_scale_min=3.8,
        graph_u_query_scale_max=4.2,
        graph_u_query_location_mean=-12,
        graph_u_query_location_std=0.3,
        graph_u_query_location_min=-13,
        graph_u_query_location_max=-11,
    ))
    torch.testing.assert_close(default[0], ignored[0], rtol=0, atol=0)
    torch.testing.assert_close(default[1], ignored[1], rtol=0, atol=0)
    assert torch.equal(default[3], ignored[3])
    assert default[4][0] == ignored[4][0]
    np.testing.assert_array_equal(default[4][1], ignored[4][1])
    assert default[4][2:] == ignored[4][2:]
    assert default[5] == ignored[5]
