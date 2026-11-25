"""Pool-intervention configuration and public generation entry points."""

import argparse

import pytest

from tabicl.nano_graph_u import __main__ as nano_cli
from tabicl.prior.graph_lib._config import GRAPH_U_POOL_DEFAULTS, PriorConfig


@pytest.mark.parametrize("overrides", [
    {"graph_u_intervention_mode": "unknown"},
    {"graph_u_pool_sampling": "unknown"},
    {"graph_u_pool_peak_percentile": -1},
    {"graph_u_pool_peak_percentile": 101},
    {"graph_u_pool_peak_percentile": float("nan")},
    {"graph_u_pool_peak_percentile": float("inf")},
    {"graph_u_pool_peak_percentile": True},
    {"graph_u_pool_peak_percentile": "75"},
    {"graph_u_pool_std_fraction": 0},
    {"graph_u_pool_std_fraction": -0.1},
    {"graph_u_pool_std_fraction": float("nan")},
    {"graph_u_pool_std_fraction": float("inf")},
    {"graph_u_pool_std_fraction": True},
    {"graph_u_pool_std_fraction": "0.1"},
])
def test_invalid_pool_configuration_is_rejected(overrides):
    with pytest.raises(ValueError):
        PriorConfig(**overrides)


def test_shared_prior_cli_round_trips_pool_configuration():
    parser = argparse.ArgumentParser()
    PriorConfig.add_args_to_parser(parser)
    defaults = PriorConfig.from_args(parser.parse_args([]))
    assert defaults.graph_u_intervention_mode == "affine"
    expected = {
        "graph_u_intervention_mode": "support_pool",
        "graph_u_pool_sampling": "gaussian",
        "graph_u_pool_peak_percentile": 85,
        "graph_u_pool_std_fraction": 0.2,
    }
    arguments = [part for key, value in expected.items() for part in (f"--{key}", str(value))]
    actual = PriorConfig.from_args(parser.parse_args(arguments))
    assert {key: getattr(actual, key) for key in expected} == expected
    assert set(GRAPH_U_POOL_DEFAULTS) == set(expected)


@pytest.mark.parametrize("sampling", ["uniform", "gaussian"])
def test_nano_generation_cli_passes_pool_parameters_to_the_real_prior(monkeypatch, tmp_path, sampling):
    captured = []
    monkeypatch.setattr(nano_cli, "generate_dump", lambda *args, **kwargs: captured.append(kwargs))
    nano_cli.main([
        "generate", "--dump", str(tmp_path / "tasks.h5"), "--steps", "1",
        "--batch-size", "2", "--rows", "64", "--features", "3",
        "--intervention-mode", "support_pool", "--pool-sampling", sampling,
        "--pool-peak-percentile", "65", "--pool-std-fraction", "0.15",
        "--graph-u-mode", "covariate",
    ])
    assert len(captured) == 1
    config = captured[0]["prior_config"]
    assert config.graph_u_intervention_mode == "support_pool"
    assert config.graph_u_pool_sampling == sampling
    assert config.graph_u_pool_peak_percentile == 65
    assert config.graph_u_pool_std_fraction == 0.15
    assert config.graph_u_mode == "covariate"
    assert config.graph_u_force_gaussian is False


def test_nano_generation_keeps_affine_default(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr(nano_cli, "generate_dump", lambda *args, **kwargs: captured.append(kwargs))
    nano_cli.main(["generate", "--dump", str(tmp_path / "tasks.h5")])
    config = captured[0]["prior_config"]
    assert config.graph_u_intervention_mode == "affine"
    assert config.graph_u_shift_mode == "fixed"
    assert config.graph_u_query_location == 2
    assert config.graph_u_query_scale == 1.5
