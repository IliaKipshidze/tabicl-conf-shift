"""Resample final hidden-U vectors from the support rows in their original order."""

from dataclasses import replace
from numbers import Integral

import torch

from ._config import PriorConfig

GRAPH_U_POOL_MIXED_STD_FRACTIONS = (0.2, 0.5, 1.0)


def graph_u_pool_probabilities(
    n_support: int,
    config: PriorConfig,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return categorical weights over row indices, without inspecting U values.

    Percentiles span the first through last support row. The Gaussian width is
    ``std_fraction * n_support``; its center/width are not the resulting moments.
    """
    if isinstance(n_support, bool) or not isinstance(n_support, Integral) or n_support < 1:
        raise ValueError("n_support must be a positive integer")
    n_support = int(n_support)
    if config.graph_u_pool_sampling == "mixed":
        raise ValueError("Mixed pool sampling must resolve one task profile before computing probabilities")
    if config.graph_u_pool_sampling == "uniform":
        return torch.full((n_support,), 1.0 / n_support, dtype=torch.float64, device=device)

    indices = torch.arange(n_support, dtype=torch.float64, device=device)
    center = (n_support - 1) * (config.graph_u_pool_peak_percentile / 100.0)
    squared_distance = ((indices - center) / n_support).square()
    # Subtract the nearest row's distance before dividing. This remains finite
    # at the peak even for widths whose square would underflow to zero.
    distance_delta = (squared_distance - squared_distance.min()).clamp_min(0)
    width = config.graph_u_pool_std_fraction
    logits = -0.5 * (distance_delta / width / width)
    return torch.softmax(logits, dim=0)


def resample_graph_u_pool(
    support_u: torch.Tensor,
    n_query: int,
    config: PriorConfig,
) -> tuple[torch.Tensor, dict]:
    """Keep support U and append whole vectors sampled with replacement.

    Diagnostics contain scalar summaries only. Constant pools and reweightings
    erased by duplicate U states are recorded rather than rejected.
    """
    if not isinstance(support_u, torch.Tensor) or support_u.ndim != 2:
        raise ValueError("support_u must be a matrix of complete U vectors")
    if support_u.shape[0] < 1 or support_u.shape[1] < 1 or not bool(torch.isfinite(support_u).all()):
        raise ValueError("support_u must contain finite, nonempty U vectors")
    if isinstance(n_query, bool) or not isinstance(n_query, Integral) or n_query < 1:
        raise ValueError("n_query must be a positive integer")
    n_support = support_u.shape[0]
    task_config = config
    mixture_metadata = {}
    if config.graph_u_pool_sampling == "mixed":
        component = int(torch.randint(
            len(GRAPH_U_POOL_MIXED_STD_FRACTIONS) + 1, (), device=support_u.device
        ).item())
        sampling = "uniform" if component == 0 else "gaussian"
        width = None if component == 0 else GRAPH_U_POOL_MIXED_STD_FRACTIONS[component - 1]
        task_config = replace(config, graph_u_pool_sampling=sampling,
                              graph_u_pool_std_fraction=config.graph_u_pool_std_fraction if width is None else width)
        mixture_metadata = {
            "graph_u_pool_mixture_component": component,
            "graph_u_pool_realized_sampling": sampling,
            "graph_u_pool_realized_std_fraction": width,
        }
    probabilities = graph_u_pool_probabilities(n_support, task_config, device=support_u.device)
    indices = torch.multinomial(probabilities, int(n_query), replacement=True)
    query_u = support_u.index_select(0, indices)

    # Aggregate duplicate vectors to measure a shift in U, not just in row indices.
    unique_u, states = torch.unique(support_u, dim=0, return_inverse=True)
    n_states = unique_u.shape[0]
    support_mass = torch.bincount(states, minlength=n_states).to(torch.float64) / n_support
    query_mass = probabilities.new_zeros(n_states).scatter_add_(0, states, probabilities)
    total_variation = float((support_mass - query_mass).abs().sum() / 2)
    if n_states == 1 or task_config.graph_u_pool_sampling == "uniform":
        total_variation = 0.0
    metadata = {
        "graph_u_pool_sampling": config.graph_u_pool_sampling,
        "graph_u_pool_peak_percentile": config.graph_u_pool_peak_percentile,
        "graph_u_pool_std_fraction": config.graph_u_pool_std_fraction,
        "graph_u_pool_size": n_support,
        "graph_u_pool_unique_support_values": n_states,
        "graph_u_pool_unique_query_values": int(states.index_select(0, indices).unique().numel()),
        "graph_u_pool_constant": n_states == 1,
        "graph_u_pool_total_variation": total_variation,
        "graph_u_pool_reweighted": total_variation > 1e-12,
        "graph_u_pool_effective_sample_size": float(1 / probabilities.square().sum()),
        "graph_u_pool_max_probability": float(probabilities.max()),
        **mixture_metadata,
    }
    return torch.cat((support_u, query_u), dim=0), metadata
