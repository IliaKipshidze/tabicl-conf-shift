"""Task-level common-support mixtures and their public entry points."""

import argparse
import json
from collections import Counter
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from tabicl.nano_graph_u import __main__ as nano_cli
from tabicl.nano_graph_u.data import _seed_batch, create_prior
from tabicl.prior._dataset import GraphPrior
from tabicl.prior.graph_lib import _graph_function as graph_function_module
from tabicl.prior.graph_lib import _points as points_module
from tabicl.prior.graph_lib._config import GRAPH_U_POOL_DEFAULTS, PriorConfig
from tabicl.prior.graph_lib._pool import (
    GRAPH_U_POOL_MIXED_STD_FRACTIONS,
    graph_u_pool_probabilities,
    resample_graph_u_pool,
)
from tabicl.train._checkpoint import (
    checkpoint_training_metadata,
    load_matched_graph_u_config,
)


def _config(**overrides):
    return PriorConfig(**{
        "graph_u_enabled": True,
        "graph_u_structure_mode": "add_root",
        "graph_u_intervention_mode": "support_pool",
        "graph_u_pool_sampling": "mixed",
        "graph_u_pool_peak_percentile": 75,
        **overrides,
    })


def _assert_pool(values, support):
    assert torch.equal(values[:len(support)], support)
    matches = (values[len(support):, None, :] == support[None, :, :]).all(-1)
    assert matches.any(dim=1).all()


def test_mixed_probabilities_require_one_resolved_task_profile():
    state = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="resolve one task profile"):
        graph_u_pool_probabilities(16, _config())
    assert torch.equal(torch.get_rng_state(), state)


@pytest.mark.parametrize("component", range(4))
def test_one_profile_governs_all_query_rows_and_preserves_whole_vectors(monkeypatch, component):
    support = torch.tensor([[8., -2.], [-4., 9.], [5., 1.], [0., -7.], [3., 6.]])
    config = _config(graph_u_pool_std_fraction=0.123)
    draws, probabilities = [], []
    actual_multinomial = torch.multinomial

    def choose_profile(high, size, **kwargs):
        draws.append((high, size))
        return torch.tensor(component, **kwargs)

    def capture_probabilities(weights, count, replacement):
        probabilities.append(weights.clone())
        return actual_multinomial(weights, count, replacement)

    monkeypatch.setattr(torch, "randint", choose_profile)
    monkeypatch.setattr(torch, "multinomial", capture_probabilities)
    values, metadata = resample_graph_u_pool(support, 73, config)
    assert draws == [(4, ())]
    assert len(probabilities) == 1
    width = None if component == 0 else GRAPH_U_POOL_MIXED_STD_FRACTIONS[component - 1]
    if width is None:
        expected = np.full(len(support), 1 / len(support))
        assert metadata["graph_u_pool_total_variation"] == 0
    else:
        indices = np.arange(len(support))
        weights = np.exp(-0.5 * ((indices - 0.75 * (len(support) - 1)) / (width * len(support))) ** 2)
        expected = weights / weights.sum()
        assert metadata["graph_u_pool_total_variation"] > 0
    np.testing.assert_allclose(probabilities[0].numpy(), expected, rtol=1e-12)
    _assert_pool(values, support)
    assert config.graph_u_pool_sampling == "mixed" and config.graph_u_pool_std_fraction == 0.123
    assert metadata["graph_u_pool_sampling"] == "mixed"
    assert metadata["graph_u_pool_mixture_component"] == component
    assert metadata["graph_u_pool_realized_sampling"] == ("uniform" if width is None else "gaussian")
    assert metadata["graph_u_pool_realized_std_fraction"] == width
    json.dumps(metadata)


def test_seed_reproduces_profiles_values_and_following_random_stream():
    support = torch.arange(24, dtype=torch.float32).reshape(8, 3)

    def run():
        _seed_batch(1001, 2, 0)
        outputs = [resample_graph_u_pool(support, 17, _config()) for _ in range(24)]
        return outputs, torch.rand(8)

    first, following = run()
    second, repeated_following = run()
    for (values, metadata), (repeated_values, repeated_metadata) in zip(first, second):
        assert torch.equal(values, repeated_values)
        assert metadata == repeated_metadata
    assert torch.equal(following, repeated_following)
    assert {metadata["graph_u_pool_mixture_component"] for _, metadata in first} == set(range(4))


def test_mixture_components_are_equally_likely_across_tasks():
    torch.manual_seed(2481)
    support = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    counts = Counter(
        resample_graph_u_pool(support, 3, _config())[1]["graph_u_pool_mixture_component"]
        for _ in range(2000)
    )
    assert set(counts) == set(range(4))
    # Broad deterministic tolerance guards accidental unequal component weights.
    assert all(0.20 < count / 2000 < 0.30 for count in counts.values())


@pytest.mark.parametrize("mode", ["confounding", "covariate"])
def test_actual_batch_contains_multiple_task_profiles_without_affine_shifts(monkeypatch, mode):
    accepted, generated = [], []
    actual_resample = graph_function_module.resample_graph_u_pool
    actual_generate = GraphPrior.generate_dataset

    def capture_pool(support, n_query, config):
        values, metadata = actual_resample(support, n_query, config)
        _assert_pool(values, support)
        generated.append(metadata)
        return values, metadata

    def capture_accepted(self, params):
        result = actual_generate(self, params)
        accepted.append(generated[-1])
        return result

    def reject_affine(_config):
        raise AssertionError("A common-support task must never sample affine shift parameters")

    monkeypatch.setattr(graph_function_module, "resample_graph_u_pool", capture_pool)
    monkeypatch.setattr(GraphPrior, "generate_dataset", capture_accepted)
    monkeypatch.setattr(points_module, "sample_graph_u_query_shift", reject_affine)
    _seed_batch(1001, 0, 0)
    prior = create_prior(
        _config(graph_u_mode=mode, min_n_nodes=3, max_n_nodes=5, fct_types="lin",
                graph_u_shift_mode="sampled", graph_u_query_location=100, graph_u_query_scale=40),
        batch_size=8, rows=48, features=3,
    )
    x, y, _, lengths, boundaries = prior.get_batch()
    assert x.shape == (8, 48, 3) and y.shape == (8, 48)
    assert torch.isfinite(x).all() and torch.isfinite(y).all()
    assert torch.equal(lengths, torch.full_like(lengths, 48))
    assert boundaries.unique().numel() == 1
    assert len(accepted) == 8
    assert len({item["graph_u_pool_mixture_component"] for item in accepted}) > 1


@pytest.mark.parametrize("sampling", ["uniform", "gaussian"])
def test_existing_modes_keep_rng_and_metadata_without_a_mixture_draw(sampling):
    support = torch.arange(30, dtype=torch.float32).reshape(10, 3)
    config = _config(graph_u_pool_sampling=sampling)
    torch.manual_seed(28)
    probabilities = graph_u_pool_probabilities(len(support), config)
    expected_indices = torch.multinomial(probabilities, 21, replacement=True)
    expected_next = torch.rand(5)
    torch.manual_seed(28)
    values, metadata = resample_graph_u_pool(support, 21, config)
    assert torch.equal(values[10:], support[expected_indices])
    assert torch.equal(torch.rand(5), expected_next)
    assert not any("mixture" in key or "realized" in key for key in metadata)


def test_mixed_constant_pool_reports_no_effective_u_shift():
    values, metadata = resample_graph_u_pool(torch.ones(12, 3), 10, _config())
    assert torch.equal(values, torch.ones(22, 3))
    assert metadata["graph_u_pool_constant"] is True
    assert metadata["graph_u_pool_total_variation"] == 0
    assert metadata["graph_u_pool_reweighted"] is False


def test_mixed_shared_and_nano_clis_and_checkpoint_metadata_round_trip(monkeypatch, tmp_path):
    parser = argparse.ArgumentParser()
    PriorConfig.add_args_to_parser(parser)
    parsed = PriorConfig.from_args(parser.parse_args([
        "--graph_u_intervention_mode", "support_pool", "--graph_u_pool_sampling", "mixed",
    ]))
    assert parsed.graph_u_pool_sampling == "mixed"
    generated = []
    monkeypatch.setattr(nano_cli, "generate_dump", lambda *args, **kwargs: generated.append(kwargs))
    nano_cli.main([
        "generate", "--dump", str(tmp_path / "train.h5"), "--steps", "1",
        "--intervention-mode", "support_pool", "--pool-sampling", "mixed",
        "--pool-peak-percentile", "75", "--pool-std-fraction", "0.5",
    ])
    config = generated[0]["prior_config"]
    assert config.graph_u_pool_sampling == "mixed"
    checkpoint = tmp_path / "model.pt"
    torch.save(checkpoint_training_metadata({"prior_type": "graph_scm"}, asdict(config)), checkpoint)
    restored = load_matched_graph_u_config(checkpoint)
    assert {key: restored[key] for key in GRAPH_U_POOL_DEFAULTS} == {
        key: getattr(config, key) for key in GRAPH_U_POOL_DEFAULTS
    }
    assert replace(config).graph_u_pool_sampling == "mixed"
