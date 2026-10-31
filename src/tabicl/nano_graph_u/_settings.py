"""Shared scientific settings for the fixed and sampled shift experiments."""

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


def experiment_prior_config(
    condition: str, graph_u_mode: str, sampling_config: dict[str, Any] | None = None
) -> PriorConfig:
    """Preserve the old prior, replacing only shifted tasks' source parameters."""
    if condition not in {"identity", "shift"}:
        raise ValueError("condition must be identity or shift")
    settings = {} if sampling_config is None else normalize_sampling_config(sampling_config)
    if condition == "identity":
        settings["graph_u_shift_mode"] = "fixed"
    return PriorConfig(
        graph_u_enabled=True,
        graph_u_mode=graph_u_mode,
        graph_u_structure_mode="add_root",
        graph_u_query_location=0.0 if condition == "identity" else 2.0,
        graph_u_query_scale=1.0 if condition == "identity" else 1.5,
        graph_u_force_gaussian=False,
        graph_u_max_attempts=1000,
        filter_unpredictable_graphs=True,
        filter_unpredictable_datasets=True,
        allow_act_warping=False,
        **settings,
    )
