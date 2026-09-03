"""Generate real Graph-U datasets and inspect the hidden source shift.

This is a diagnostic script, not a training entry point. It temporarily wraps
the real generator in memory so that the selected hidden root can be inspected;
the production dataset/API remains unchanged and still exposes only X and y.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch

from tabicl.prior._graph_scm import GraphSCM
from tabicl.prior.graph_lib._config import PriorConfig
from tabicl.prior.graph_lib._function import RandomFunction
from tabicl.prior.graph_lib._node_function import RandomNodeFunction
from tabicl.prior.graph_lib._points import (
    RandomCirclePoints,
    RandomCovariancePoints,
    RandomGaussianPoints,
    RandomPoints,
    RandomUniformPoints,
)


@dataclass
class GraphUTrace:
    """Values from the selected hidden root at successive generation stages."""

    n_support: int
    source_family: str | None = None
    raw_source: torch.Tensor | None = None
    shifted_source: torch.Tensor | None = None
    root_function_output: torch.Tensor | None = None
    hidden_u: torch.Tensor | None = None


class _TraceState:
    def __init__(self) -> None:
        self.active = False
        self.base_depth = 0
        self.current: GraphUTrace | None = None
        self.by_points_id: dict[int, GraphUTrace] = {}
        self.traces: list[GraphUTrace] = []


@contextmanager
def capture_graph_u_values() -> Iterator[list[GraphUTrace]]:
    """Capture Graph-U values while leaving all numerical behavior unchanged."""

    state = _TraceState()
    original_points_sample = RandomPoints.sample
    original_function_transform = RandomFunction._transform
    original_node_transform = RandomNodeFunction._transform
    source_classes = (
        RandomUniformPoints,
        RandomCirclePoints,
        RandomGaussianPoints,
        RandomCovariancePoints,
    )
    original_source_samples = {source_cls: source_cls.sample for source_cls in source_classes}

    def traced_points_sample(
        self: RandomPoints,
        n_batch: int,
        n: int,
        *,
        graph_u_n_train: int | None = None,
    ) -> torch.Tensor:
        if graph_u_n_train is None:
            return original_points_sample(self, n_batch, n)
        if state.active:
            raise RuntimeError("Nested selected-U source sampling is not supported")

        trace = GraphUTrace(n_support=graph_u_n_train)
        state.active = True
        state.current = trace
        try:
            result = original_points_sample(
                self,
                n_batch,
                n,
                graph_u_n_train=graph_u_n_train,
            )
            trace.root_function_output = result.detach().cpu().clone()
        finally:
            state.active = False
            state.current = None

        state.by_points_id[id(self)] = trace
        state.traces.append(trace)
        return result

    def traced_function_transform(
        self: RandomFunction,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if (
            state.active
            and self.__class__ is RandomFunction
            and state.current is not None
            and state.current.shifted_source is None
        ):
            state.current.shifted_source = x.detach().cpu().clone()
        return original_function_transform(self, x)

    def traced_node_transform(
        self: RandomNodeFunction,
        x: list[torch.Tensor],
        n_samples: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        result = original_node_transform(self, x, n_samples)
        if self.graph_u_n_train is not None:
            trace = state.by_points_id.pop(id(self.random_points_))
            trace.hidden_u = result[0].detach().cpu().clone()
        return result

    def make_traced_source_sample(source_cls: type[RandomPoints]):
        original = original_source_samples[source_cls]

        def traced_source_sample(
            self: RandomPoints,
            n_batch: int,
            n: int,
        ) -> torch.Tensor:
            is_outer_source = (
                state.active
                and state.base_depth == 0
                and state.current is not None
                and state.current.raw_source is None
            )
            state.base_depth += 1
            try:
                result = original(self, n_batch, n)
            finally:
                state.base_depth -= 1
            if is_outer_source:
                assert state.current is not None
                state.current.source_family = source_cls.__name__
                # The generator subsequently modifies the query slice in place.
                state.current.raw_source = result.detach().cpu().clone()
            return result

        return traced_source_sample

    RandomPoints.sample = traced_points_sample
    RandomFunction._transform = traced_function_transform
    RandomNodeFunction._transform = traced_node_transform
    for cls in source_classes:
        cls.sample = make_traced_source_sample(cls)

    try:
        yield state.traces
    finally:
        RandomPoints.sample = original_points_sample
        RandomFunction._transform = original_function_transform
        RandomNodeFunction._transform = original_node_transform
        for cls, original in original_source_samples.items():
            cls.sample = original


def _require_trace_tensors(trace: GraphUTrace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if trace.raw_source is None or trace.shifted_source is None or trace.hidden_u is None:
        raise RuntimeError("The diagnostic hooks did not capture a complete Graph-U trace")
    return trace.raw_source, trace.shifted_source, trace.hidden_u


def _standardized_mean_difference(values: torch.Tensor, n_support: int) -> float:
    support = values[:n_support].float()
    query = values[n_support:].float()
    support_std = support.std(dim=0, correction=0)
    query_std = query.std(dim=0, correction=0)
    pooled_std = torch.sqrt((support_std.square() + query_std.square()) / 2)
    informative = pooled_std > 1e-8
    if not informative.any():
        return 0.0
    differences = (query.mean(dim=0) - support.mean(dim=0)).abs()
    return (differences[informative] / pooled_std[informative]).mean().item()


def _format_histogram(values: list[int]) -> str:
    counts = Counter(values)
    return ", ".join(f"{value}: {counts[value]}" for value in sorted(counts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=int, default=20)
    parser.add_argument("--support", type=int, default=128)
    parser.add_argument("--query", type=int, default=128)
    parser.add_argument("--features", type=int, default=5)
    parser.add_argument("--query-location", type=float, default=2.0)
    parser.add_argument("--query-scale", type=float, default=1.5)
    parser.add_argument("--force-gaussian", action="store_true")
    parser.add_argument("--seed", type=int, default=20260903)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.datasets <= 0 or args.support <= 0 or args.query <= 0 or args.features <= 0:
        raise ValueError("datasets, support, query, and features must all be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = PriorConfig(
        graph_u_enabled=True,
        graph_u_query_location=args.query_location,
        graph_u_query_scale=args.query_scale,
        graph_u_force_gaussian=args.force_gaussian,
    )
    attempts: list[int] = []
    candidate_counts: list[int] = []
    source_errors: list[float] = []
    query_errors: list[float] = []
    source_shift_scores: list[float] = []
    hidden_u_shift_scores: list[float] = []
    collapsed_hidden_u = 0
    source_families: Counter[str] = Counter()
    captured_traces: list[GraphUTrace] = []

    with torch.no_grad(), capture_graph_u_values() as traces:
        for _ in range(args.datasets):
            before = len(traces)
            scm = GraphSCM(
                regression=True,
                seq_len=args.support + args.query,
                train_size=args.support,
                num_features=args.features,
                max_features=args.features,
                permute_features=False,
                config=config,
            )
            X, y = scm()
            if len(traces) != before + 1:
                raise RuntimeError("Expected exactly one selected-U trace per dataset")

            trace = traces[-1]
            raw_source, shifted_source, hidden_u = _require_trace_tensors(trace)
            expected_query = args.query_scale * raw_source[args.support :] + args.query_location
            support_error = (
                shifted_source[: args.support] - raw_source[: args.support]
            ).abs().max().item()
            query_error = (
                shifted_source[args.support :] - expected_query
            ).abs().max().item()

            # U must stay hidden: it has no feature specification and is absent
            # from the returned X/y tensors.
            metadata = scm.metadata_
            selected_u = metadata["graph_u_node_idx"]
            if selected_u is None or metadata["graph_u_enabled"] is not True:
                raise RuntimeError("Generated dataset is missing Graph-U metadata")
            expected_x_shape = (args.support + args.query, args.features)
            if X.shape != expected_x_shape or y.shape != (args.support + args.query,):
                raise RuntimeError(
                    f"Unexpected observed shapes: X={tuple(X.shape)}, y={tuple(y.shape)}"
                )
            if not torch.isfinite(X).all() or not torch.isfinite(y).all():
                raise RuntimeError("Generated observed data contain non-finite values")

            attempts.append(metadata["graph_u_attempts"])
            candidate_counts.append(len(metadata["graph_u_candidate_node_idxs"]))
            source_errors.append(support_error)
            query_errors.append(query_error)
            source_shift_scores.append(
                _standardized_mean_difference(shifted_source, args.support)
            )
            hidden_u_shift_scores.append(
                _standardized_mean_difference(hidden_u, args.support)
            )
            collapsed_hidden_u += int(hidden_u.float().std(correction=0).item() <= 1e-8)
            source_families[trace.source_family or "unknown"] += 1
            captured_traces.append(trace)

    total_proposals = sum(attempts)
    total_rejections = total_proposals - args.datasets
    attempts_array = np.asarray(attempts)
    print("Graph-U generation-only smoke test")
    print(f"datasets accepted: {args.datasets}")
    print(f"rows per dataset: {args.support} support + {args.query} query")
    print(f"observed columns: {args.features} X + 1 y; hidden U columns exposed: 0")
    print(f"force Gaussian: {args.force_gaussian}")
    print(f"source families: {dict(source_families)}")
    print(f"graph proposals: {total_proposals}")
    print(f"graphs rejected before generation: {total_rejections}")
    print(f"accepted on first proposal: {attempts.count(1)}/{args.datasets}")
    print(
        "attempts per accepted dataset: "
        f"mean={attempts_array.mean():.2f}, median={np.median(attempts_array):.1f}, "
        f"max={attempts_array.max()}"
    )
    print(f"attempt histogram (attempts: datasets): {_format_histogram(attempts)}")
    print(f"candidate-count histogram: {_format_histogram(candidate_counts)}")
    print(f"max support source identity error: {max(source_errors):.3g}")
    print(f"max query affine-formula error: {max(query_errors):.3g}")
    print(
        "mean absolute standardized support-query difference: "
        f"Z_U={np.mean(source_shift_scores):.3f}, final U={np.mean(hidden_u_shift_scores):.3f}"
    )
    hidden_scores = np.asarray(hidden_u_shift_scores)
    print(
        "final-U shift score distribution: "
        f"min={hidden_scores.min():.3f}, median={np.median(hidden_scores):.3f}, "
        f"max={hidden_scores.max():.3f}; "
        f"score > 0.1 in {(hidden_scores > 0.1).sum()}/{args.datasets} datasets"
    )
    print(f"constant final-U tensors: {collapsed_hidden_u}/{args.datasets}")

    example_trace = max(
        captured_traces,
        key=lambda trace: _standardized_mean_difference(
            _require_trace_tensors(trace)[2], args.support
        ),
    )
    raw_source, shifted_source, hidden_u = _require_trace_tensors(example_trace)
    support_u = hidden_u[: args.support].float()
    query_u = hidden_u[args.support :].float()
    coordinate = int((query_u.mean(dim=0) - support_u.mean(dim=0)).abs().argmax())
    print("example dataset with visible final-U mean separation:")
    print(
        "  support raw E_U -> shifted Z_U: "
        f"{raw_source[:3, 0].tolist()} -> {shifted_source[:3, 0].tolist()}"
    )
    print(
        "  query raw E_U -> shifted Z_U: "
        f"{raw_source[args.support : args.support + 3, 0].tolist()} -> "
        f"{shifted_source[args.support : args.support + 3, 0].tolist()}"
    )
    print(
        f"  final hidden U coordinate {coordinate} support/query means: "
        f"{support_u[:, coordinate].mean().item():.3f} / "
        f"{query_u[:, coordinate].mean().item():.3f}"
    )

    if max(source_errors) > 1e-6 or max(query_errors) > 1e-6:
        raise RuntimeError("The captured source values do not match the configured shift")


if __name__ == "__main__":
    main()
