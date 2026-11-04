"""Shared scientific settings for the fixed and sampled shift experiments."""

import math
from numbers import Real
from typing import Any

from tabicl.prior.graph_lib._config import GRAPH_U_SAMPLING_DEFAULTS, PriorConfig


def normalize_sampling_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate a complete, explicit sampled-shift recipe."""
    if set(config) != set(GRAPH_U_SAMPLING_DEFAULTS):
        raise ValueError("sampling_config must contain exactly the nine Graph-U sampling fields")
    prior = PriorConfig(**config)
    if prior.graph_u_shift_mode != "sampled":
        raise ValueError("This experiment requires graph_u_shift_mode='sampled'")
    return {name: getattr(prior, name) for name in GRAPH_U_SAMPLING_DEFAULTS}


def normalize_fixed_shift_config(config: dict[str, Any]) -> dict[str, float]:
    """Validate one explicit location/scale pair, without sampling a shift."""
    fields = {"graph_u_query_location", "graph_u_query_scale"}
    if not isinstance(config, dict) or set(config) != fields:
        raise ValueError(
            "fixed_shift_config must contain exactly graph_u_query_location and graph_u_query_scale"
        )
    normalized = {}
    for name in sorted(fields):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, Real):
            # Malformed experiment recipes consistently raise ValueError.
            raise ValueError(f"{name} must be a finite real number")  # noqa: TRY004
        try:
            value = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite real number") from exc
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite real number")
        if name == "graph_u_query_scale" and value <= 0:
            raise ValueError("graph_u_query_scale must be positive")
        normalized[name] = value
    return normalized


def experiment_prior_config(
    condition: str,
    graph_u_mode: str,
    sampling_config: dict[str, Any] | None = None,
    *,
    fixed_shift_config: dict[str, Any] | None = None,
) -> PriorConfig:
    """Preserve the old prior, replacing only shifted tasks' source parameters."""
    if condition not in {"identity", "shift"}:
        raise ValueError("condition must be identity or shift")
    if sampling_config is not None and fixed_shift_config is not None:
        raise ValueError("sampling_config and fixed_shift_config are mutually exclusive")
    settings = {} if sampling_config is None else normalize_sampling_config(sampling_config)
    fixed = (
        {"graph_u_query_location": 2.0, "graph_u_query_scale": 1.5}
        if fixed_shift_config is None else normalize_fixed_shift_config(fixed_shift_config)
    )
    if fixed_shift_config is not None:
        settings["graph_u_shift_mode"] = "fixed"
    if condition == "identity":
        settings["graph_u_shift_mode"] = "fixed"
    return PriorConfig(
        graph_u_enabled=True,
        graph_u_mode=graph_u_mode,
        graph_u_structure_mode="add_root",
        graph_u_query_location=0.0 if condition == "identity" else fixed["graph_u_query_location"],
        graph_u_query_scale=1.0 if condition == "identity" else fixed["graph_u_query_scale"],
        graph_u_force_gaussian=False,
        graph_u_max_attempts=1000,
        filter_unpredictable_graphs=True,
        filter_unpredictable_datasets=True,
        allow_act_warping=False,
        **settings,
    )
