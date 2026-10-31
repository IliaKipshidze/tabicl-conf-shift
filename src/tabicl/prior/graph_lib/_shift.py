"""Task-level source shifts; independent of the sampled SCM/source family."""

from scipy.stats import truncnorm

from tabicl.prior.graph_lib._config import PriorConfig


def sample_graph_u_query_shift(config: PriorConfig) -> tuple[float, float]:
    """Return (scale, location), drawing once per parameter in sampled mode.

    SciPy uses the seeded NumPy random stream. Inverse-CDF sampling gives a
    truncated normal without clipping or retrying graphs. Fixed mode consumes
    no randomness, preserving the original generator's seeded behavior.
    """
    if config.graph_u_shift_mode == "fixed":
        return config.graph_u_query_scale, config.graph_u_query_location

    def draw(parameter: str) -> float:
        prefix = f"graph_u_query_{parameter}"
        mean = getattr(config, f"{prefix}_mean")
        std = getattr(config, f"{prefix}_std")
        lower = getattr(config, f"{prefix}_min")
        upper = getattr(config, f"{prefix}_max")
        return float(truncnorm.rvs((lower - mean) / std, (upper - mean) / std, loc=mean, scale=std))

    return draw("scale"), draw("location")
