from typing import Dict, List

import numpy as np

from tabicl.prior.graph_lib._base import PriorComponent, Dataset, DatasetProperties, FeatureSpec
from tabicl.prior.graph_lib._graph import RandomDAG
from tabicl.prior.graph_lib._graph_function import RandomGraphFunction


def check_x_y_ancestors_overlap(graph: List[List[int]], node_feature_specs: List[Dict[str, FeatureSpec]]) -> bool:
    """
    There can only be a functional relation between x and y if one of their ancestors overlap.

    :param graph: Graph (list of parent node idxs for each node)
    :param node_feature_specs: feature specs with feature names for each node
    :return: False if x and y are completely independent according to the graph, otherwise True.
    """
    ancestors_arrs = {key: np.zeros(len(graph), dtype=np.bool_) for key in ['x', 'y']}
    for node_idx in reversed(range(len(graph))):
        for feature_spec in node_feature_specs[node_idx].values():
            if feature_spec.group in ['x', 'y']:
                # set this node as one of the ancestors from this group
                ancestors_arrs[feature_spec.group][node_idx] = True
        for group in ['x', 'y']:
            if ancestors_arrs[group][node_idx]:
                for parent_idx in graph[node_idx]:
                    ancestors_arrs[group][parent_idx] = True

    return np.any(np.logical_and(ancestors_arrs['x'], ancestors_arrs['y']))


def find_graph_u_candidates(
    graph: List[List[int]], node_feature_specs: List[Dict[str, FeatureSpec]]
) -> List[int]:
    """Return hidden root nodes that directly confound distinct observed X and Y nodes.

    ``graph[i]`` contains the parents of node ``i``. A Graph-U candidate must
    therefore have no parents, expose no feature of its own, and occur in the
    parent lists of at least one observed X node and one *different* observed Y
    node. Requiring different child nodes rules out treating a single node that
    happens to expose both X and Y as a two-variable confounding structure.
    """
    if len(graph) != len(node_feature_specs):
        raise ValueError(
            "graph and node_feature_specs must describe the same number of nodes "
            f"(got {len(graph)} and {len(node_feature_specs)})"
        )

    observed_x_nodes = {
        node_idx
        for node_idx, feature_specs in enumerate(node_feature_specs)
        if any(feature_spec.group == "x" for feature_spec in feature_specs.values())
    }
    observed_y_nodes = {
        node_idx
        for node_idx, feature_specs in enumerate(node_feature_specs)
        if any(feature_spec.group == "y" for feature_spec in feature_specs.values())
    }

    candidates = []
    for node_idx, parents in enumerate(graph):
        if parents or node_feature_specs[node_idx]:
            continue

        x_children = {child_idx for child_idx in observed_x_nodes if node_idx in graph[child_idx]}
        y_children = {child_idx for child_idx in observed_y_nodes if node_idx in graph[child_idx]}
        if any(x_child != y_child for x_child in x_children for y_child in y_children):
            candidates.append(node_idx)

    return candidates


class RandomDataset(PriorComponent):
    def sample(self, data_prop: DatasetProperties) -> Dataset:
        graph_u_enabled = bool(getattr(self.config, "graph_u_enabled", False))
        graph_u_max_attempts = getattr(self.config, "graph_u_max_attempts", 100)
        if graph_u_enabled:
            if not isinstance(graph_u_max_attempts, (int, np.integer)) or graph_u_max_attempts < 1:
                raise ValueError(
                    "graph_u_max_attempts must be a positive integer when Graph-U is enabled, "
                    f"got {graph_u_max_attempts!r}"
                )
            graph_u_max_attempts = int(graph_u_max_attempts)
            if data_prop.n_train <= 0 or data_prop.n_test <= 0:
                raise ValueError(
                    "Graph-U requires non-empty support and query splits; "
                    f"got n_train={data_prop.n_train}, n_test={data_prop.n_test}. "
                    "Pass train_size strictly between zero and seq_len to GraphSCM."
                )

        n_attempts = 0
        graph_u_candidates: List[int] = []
        while True:
            n_attempts += 1
            # ----- Create computation graph -----
            min_n_nodes = max(self.config.min_n_nodes, 3) if graph_u_enabled else self.config.min_n_nodes
            n_nodes = self.sampler.randint(
                "n_nodes", min_n_nodes, self.config.max_n_nodes + 1, use_log=True
            )
            graph = RandomDAG(self.context).sample(n_nodes)

            node_feature_specs = [dict() for _ in range(n_nodes)]

            feature_groups = list(set(spec.group for spec in data_prop.feature_specs.values()))
            for feature_group in feature_groups:
                feature_specs = {
                    key: value for key, value in data_prop.feature_specs.items() if value.group == feature_group
                }
                if self.config.subsample_feature_nodes:
                    n_feature_nodes = self.sampler.randint("n_feature_nodes", 1, n_nodes + 1)
                    feature_nodes = np.random.permutation(n_nodes)[:n_feature_nodes]
                else:
                    feature_nodes = np.arange(n_nodes)
                feature_node_idxs = np.random.choice(feature_nodes, replace=True, size=len(feature_specs))

                for idx, (feature_name, feature_spec) in enumerate(feature_specs.items()):
                    node_feature_specs[feature_node_idxs[idx]][feature_name] = feature_spec

            graph_is_predictable = (
                (not self.config.filter_unpredictable_graphs)
                or check_x_y_ancestors_overlap(graph, node_feature_specs)
            )
            if graph_u_enabled:
                graph_u_candidates = find_graph_u_candidates(graph, node_feature_specs)

            if graph_is_predictable and ((not graph_u_enabled) or graph_u_candidates):
                break

            if graph_u_enabled and n_attempts >= graph_u_max_attempts:
                n_x_features = sum(
                    feature_spec.group == "x" for feature_spec in data_prop.feature_specs.values()
                )
                n_y_features = sum(
                    feature_spec.group == "y" for feature_spec in data_prop.feature_specs.values()
                )
                raise RuntimeError(
                    "Unable to sample a valid Graph-U dataset after "
                    f"{graph_u_max_attempts} attempts. A valid graph needs an unobserved root "
                    "that is a direct parent of distinct observed X and Y nodes "
                    f"(n_x_features={n_x_features}, n_y_features={n_y_features}, "
                    f"n_nodes_range=[{min_n_nodes}, {self.config.max_n_nodes}]). "
                    "Increase graph_u_max_attempts or use a larger/denser graph prior."
                )

        graph_u_node_idx = int(np.random.choice(graph_u_candidates)) if graph_u_enabled else None

        graph_func_kwargs = {}
        if graph_u_enabled:
            graph_func_kwargs = {
                "graph_u_node_idx": graph_u_node_idx,
                "n_train": data_prop.n_train,
            }
        graph_func = RandomGraphFunction(
            self.context,
            dag=graph,
            node_feature_specs=node_feature_specs,
            **graph_func_kwargs,
        )

        graph_u_x_child_node_idxs = []
        graph_u_y_child_node_idxs = []
        if graph_u_node_idx is not None:
            for child_idx, parents in enumerate(graph):
                if graph_u_node_idx not in parents:
                    continue
                groups = {feature_spec.group for feature_spec in node_feature_specs[child_idx].values()}
                if "x" in groups:
                    graph_u_x_child_node_idxs.append(child_idx)
                if "y" in groups:
                    graph_u_y_child_node_idxs.append(child_idx)

        # ----- Evaluate computation graph -----
        n_samples = data_prop.n_train + data_prop.n_test
        if self.config.ensure_iid:
            graph_func(n_samples)  # fit the graph function on separate data
        tensors = graph_func(n_samples)
        graph_u_config = {
            name: value for name, value in vars(self.config).items() if name.startswith("graph_u_")
        }
        return Dataset(
            tensors=tensors,
            feature_specs=data_prop.feature_specs,
            graph=graph,
            n_train=data_prop.n_train,
            n_test=data_prop.n_test,
            graph_u_enabled=graph_u_enabled,
            graph_u_node_idx=graph_u_node_idx,
            graph_u_candidate_node_idxs=tuple(graph_u_candidates),
            graph_u_x_child_node_idxs=tuple(graph_u_x_child_node_idxs),
            graph_u_y_child_node_idxs=tuple(graph_u_y_child_node_idxs),
            graph_u_attempts=n_attempts,
            graph_u_config=graph_u_config,
            graph_u_fit_policy="support_only" if graph_u_enabled else None,
        )
