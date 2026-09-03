from numbers import Integral
from typing import Dict, List, Optional

import torch

from tabicl.prior.graph_lib._base import RandomTransformer, Context, FeatureSpec
from tabicl.prior.graph_lib._node_function import RandomNodeFunction


class RandomGraphFunction(RandomTransformer):
    """
    Samples a dataset by propagating data through a graph.
    """
    def __init__(
        self,
        context: Context,
        dag: List[List[int]],
        node_feature_specs: List[Dict[str, FeatureSpec]],
        *,
        graph_u_node_idx: Optional[int] = None,
        n_train: Optional[int] = None,
    ):
        super().__init__(context=context)
        self.dag = dag
        self.node_feature_specs = node_feature_specs
        self.graph_u_node_idx = graph_u_node_idx
        self.n_train = n_train

        if self.config.graph_u_enabled:
            if self.graph_u_node_idx is None:
                raise ValueError("graph_u_node_idx is required when graph_u_enabled=True")
            if self.n_train is None:
                raise ValueError("n_train is required when graph_u_enabled=True")
            if not isinstance(self.graph_u_node_idx, Integral) or isinstance(
                    self.graph_u_node_idx, bool
            ):
                raise TypeError("graph_u_node_idx must be an integer")
            if not isinstance(self.n_train, Integral) or isinstance(self.n_train, bool):
                raise TypeError("n_train must be an integer")
            self.graph_u_node_idx = int(self.graph_u_node_idx)
            self.n_train = int(self.n_train)
            if len(self.dag) != len(self.node_feature_specs):
                raise ValueError("dag and node_feature_specs must describe the same number of nodes")
            if not 0 <= self.graph_u_node_idx < len(self.dag):
                raise ValueError(f"graph_u_node_idx={self.graph_u_node_idx} is outside the DAG")
            if self.dag[self.graph_u_node_idx]:
                raise ValueError("The Graph-U confounder must be a root node")
            if self.node_feature_specs[self.graph_u_node_idx]:
                raise ValueError("The Graph-U confounder must be hidden")

            x_children = {
                node_idx
                for node_idx, feature_specs in enumerate(self.node_feature_specs)
                if self.graph_u_node_idx in self.dag[node_idx]
                and any(feature_spec.group == "x" for feature_spec in feature_specs.values())
            }
            y_children = {
                node_idx
                for node_idx, feature_specs in enumerate(self.node_feature_specs)
                if self.graph_u_node_idx in self.dag[node_idx]
                and any(feature_spec.group == "y" for feature_spec in feature_specs.values())
            }
            if not any(x_child != y_child for x_child in x_children for y_child in y_children):
                raise ValueError(
                    "The Graph-U confounder must directly parent distinct observed X and Y nodes"
                )
        elif self.graph_u_node_idx is not None or self.n_train is not None:
            raise ValueError("Graph-U node and boundary were supplied while graph_u_enabled=False")

    def _validate_graph_u_boundary(self, n_samples: int) -> None:
        if not self.config.graph_u_enabled:
            return
        if self.n_train is None or not 0 < self.n_train < n_samples:
            raise ValueError(
                f"Graph-U requires 0 < n_train < n_samples; got n_train={self.n_train}, n_samples={n_samples}"
            )

    def _fit(self, n_samples: int):
        self._validate_graph_u_boundary(n_samples)
        self.nodes_ = []
        for node_idx, feature_specs in enumerate(self.node_feature_specs):
            if node_idx == self.graph_u_node_idx:
                node = RandomNodeFunction(
                    self.context,
                    feature_specs=feature_specs,
                    graph_u_n_train=self.n_train,
                )
            else:
                node = RandomNodeFunction(self.context, feature_specs=feature_specs)
            self.nodes_.append(node)
        # for efficiency, prune nodes whose values don't need to be computed
        self.should_compute_ = [False for _ in range(len(self.nodes_))]
        for node_idx in reversed(range(len(self.nodes_))):
            if len(self.node_feature_specs[node_idx]) >= 1:
                self.should_compute_[node_idx] = True
            if self.should_compute_[node_idx]:  # could have been set by successors or by itself
                for parent in self.dag[node_idx]:
                    self.should_compute_[parent] = True

        if self.config.graph_u_enabled:
            assert self.graph_u_node_idx is not None
            if not self.should_compute_[self.graph_u_node_idx]:
                raise ValueError("The selected Graph-U confounder is not an ancestor of an observed feature")

    def _transform(self, n_samples: int) -> Dict[str, torch.Tensor]:
        self._validate_graph_u_boundary(n_samples)
        n_nodes = len(self.node_feature_specs)
        node_values = [None for _ in range(n_nodes)]
        features = dict()
        for node_idx in range(len(self.node_feature_specs)):
            if self.should_compute_[node_idx]:
                node_values[node_idx], out_features = self.nodes_[node_idx](
                    [node_values[parent] for parent in self.dag[node_idx]], n_samples
                )
                features = features | out_features
        return features
