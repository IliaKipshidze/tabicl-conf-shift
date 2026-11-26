"""Reject numerical failures in a U pool without hiding malformed inputs."""

import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from tabicl.prior import _dataset as dataset_module
from tabicl.prior._dataset import GraphPrior
from tabicl.prior.graph_lib._config import PriorConfig
from tabicl.prior.graph_lib._node_function import RandomNodeFunction
from tabicl.prior.graph_lib._pool import (
    GRAPH_U_POOL_MIXED_STD_FRACTIONS,
    NonFiniteGraphUPoolError,
    graph_u_pool_probabilities,
    resample_graph_u_pool,
)


def _config(**options):
    return PriorConfig(**{
        "graph_u_enabled": True,
        "graph_u_structure_mode": "add_root",
        "graph_u_intervention_mode": "support_pool",
        "min_n_nodes": 3,
        "max_n_nodes": 5,
        "fct_types": "lin",
        **options,
    })


def _params(config):
    return {"regression": True, "seq_len": 24, "train_size": 16, "num_features": 2,
            "max_features": 2, "num_classes": None, "device": "cpu", "config": config}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_pool_has_specific_rejection_type_without_sampling(value):
    support = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    support[2, 1] = value
    before = torch.get_rng_state().clone()
    with pytest.raises(NonFiniteGraphUPoolError, match="finite, nonempty"):
        resample_graph_u_pool(support, 8, _config(graph_u_pool_sampling="mixed"))
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize("mode", ["confounding", "covariate"])
@pytest.mark.parametrize("sampling", ["uniform", "gaussian", "mixed"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_real_graph_prior_recovers_from_nonfinite_hidden_root(monkeypatch, mode, sampling, value):
    original = RandomNodeFunction._transform
    root_calls = []

    def fail_first_root(self, parents, rows):
        values, features = original(self, parents, rows)
        if self.graph_u_n_train is not None:
            root_calls.append(values.clone())
            if len(root_calls) == 1:
                values = values.clone()
                values[0, 0] = value
        return values, features

    monkeypatch.setattr(RandomNodeFunction, "_transform", fail_first_root)
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    config = _config(graph_u_mode=mode, graph_u_pool_sampling=sampling)
    prior = GraphPrior(regression=True, config=config, n_jobs=1, generation_max_attempts=50)
    x, y, active = prior.generate_dataset(_params(config))
    assert len(root_calls) >= 2
    assert x.shape == (24, 2) and y.shape == (24,)
    assert torch.isfinite(x).all() and torch.isfinite(y).all()
    assert active.item() > 0


@pytest.mark.parametrize("mode", ["confounding", "covariate"])
def test_nonfinite_pool_rejections_preserve_attempt_limit_and_counter(monkeypatch, mode):
    class InvalidSCM:
        calls = 0

        def __init__(self, **options):
            self.config = options["config"]

        def __call__(self):
            type(self).calls += 1
            return resample_graph_u_pool(torch.full((16, 2), torch.nan), 8, self.config)

    monkeypatch.setattr(dataset_module, "GraphSCM", InvalidSCM)
    config = _config(graph_u_mode=mode)
    prior = GraphPrior(regression=True, config=config, n_jobs=1, generation_max_attempts=2)
    with pytest.raises(RuntimeError, match="after 2 attempts; rejections=.*'non_finite_values': 2"):
        prior.generate_dataset(_params(config))
    assert InvalidSCM.calls == 2


@pytest.mark.parametrize("support", [torch.ones(3), torch.empty(0, 2), torch.empty(4, 0)])
def test_malformed_pools_are_not_retryable_numeric_failures(monkeypatch, support):
    class MalformedSCM:
        calls = 0

        def __init__(self, **options):
            self.config = options["config"]

        def __call__(self):
            type(self).calls += 1
            return resample_graph_u_pool(support, 8, self.config)

    monkeypatch.setattr(dataset_module, "GraphSCM", MalformedSCM)
    config = _config()
    prior = GraphPrior(regression=True, config=config, n_jobs=1, generation_max_attempts=2)
    with pytest.raises(ValueError) as error:
        prior.generate_dataset(_params(config))
    assert not isinstance(error.value, NonFiniteGraphUPoolError)
    assert MalformedSCM.calls == 1


def test_unrelated_generation_errors_are_not_swallowed(monkeypatch):
    class BrokenSCM:
        calls = 0

        def __init__(self, **_options):
            pass

        def __call__(self):
            type(self).calls += 1
            raise ValueError("invalid graph construction")

    monkeypatch.setattr(dataset_module, "GraphSCM", BrokenSCM)
    config = _config()
    prior = GraphPrior(regression=True, config=config, n_jobs=1, generation_max_attempts=2)
    with pytest.raises(ValueError, match="invalid graph construction"):
        prior.generate_dataset(_params(config))
    assert BrokenSCM.calls == 1


@pytest.mark.parametrize("sampling", ["uniform", "gaussian", "mixed"])
def test_finite_pool_keeps_existing_sampling_and_rng(sampling):
    support = torch.arange(28, dtype=torch.float32).reshape(14, 2)
    config = _config(graph_u_pool_sampling=sampling)
    torch.manual_seed(42)
    resolved = config
    if sampling == "mixed":
        component = torch.randint(4, ()).item()
        if component == 0:
            resolved = replace(config, graph_u_pool_sampling="uniform")
        else:
            resolved = replace(config, graph_u_pool_sampling="gaussian",
                               graph_u_pool_std_fraction=GRAPH_U_POOL_MIXED_STD_FRACTIONS[component - 1])
    indices = torch.multinomial(graph_u_pool_probabilities(14, resolved), 17, replacement=True)
    after = torch.get_rng_state().clone()
    torch.manual_seed(42)
    combined, _ = resample_graph_u_pool(support, 17, config)
    assert torch.equal(combined, torch.cat((support, support[indices])))
    assert torch.equal(torch.get_rng_state(), after)
