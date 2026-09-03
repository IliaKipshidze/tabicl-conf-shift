import argparse
import math
from dataclasses import dataclass
from numbers import Integral
from typing import Literal


def str2bool(value):
    return value.lower() == "true"


@dataclass
class PriorConfig:
    """Configuration options for the prior."""

    add_gaussian_noise: bool = False
    use_node_feature_importance: bool = True
    use_node_importance: bool = True
    use_node_l2_norm: bool = True
    allow_act_warping: bool = False
    allow_kumaraswamy_warping: bool = True
    disallow_y_warping: bool = False
    # maximum number of discretization levels in random discretization function
    max_discretization_cardinality: int = 256
    filter_unpredictable_datasets: bool = False
    filter_unpredictable_graphs: bool = False
    min_n_nodes: int = 2
    max_n_nodes: int = 32
    cauchy_dag_offset: float = 0.0
    meta_sampling_mode: Literal['meta', 'local', 'global'] = 'meta'
    random_matrix_types: Literal['default', 'gaussian'] = 'default'
    fct_types: str = 'default'
    multi_fct_types: Literal['default', 'concat', 'agg'] = 'default'
    cat_modes: str = 'default'
    random_weights_types: Literal['simple', 'uniform'] = 'simple'
    subsample_feature_nodes: bool = True
    seq_lens_multiple_of: int = 1
    use_corrected_num_converters: bool = False
    use_corrected_cat_sizes: bool = False
    ensure_iid: bool = False
    remove_trivial_datasets: bool = False
    trivial_dataset_threshold: float = 0.05
    use_corrected_cat_meta_sampling: bool = False
    graph_u_enabled: bool = False
    graph_u_query_location: float = 0.0
    graph_u_query_scale: float = 1.0
    graph_u_force_gaussian: bool = False
    graph_u_max_attempts: int = 1000

    def __post_init__(self) -> None:
        if not math.isfinite(self.graph_u_query_location):
            raise ValueError("graph_u_query_location must be finite")
        if not math.isfinite(self.graph_u_query_scale) or self.graph_u_query_scale <= 0:
            raise ValueError("graph_u_query_scale must be finite and positive")
        if not isinstance(self.graph_u_max_attempts, Integral) or isinstance(
                self.graph_u_max_attempts, bool
        ):
            raise TypeError("graph_u_max_attempts must be an integer")
        if self.graph_u_max_attempts <= 0:
            raise ValueError("graph_u_max_attempts must be positive")
        self.graph_u_max_attempts = int(self.graph_u_max_attempts)
        if self.graph_u_enabled and self.max_n_nodes < 3:
            raise ValueError("Graph-U requires max_n_nodes >= 3")
        if self.graph_u_enabled and self.ensure_iid:
            raise ValueError(
                "Graph-U does not currently support ensure_iid=True because the "
                "extra graph pass resamples root-level mechanisms"
            )

    @staticmethod
    def from_args(args) -> "PriorConfig":
        return PriorConfig(add_gaussian_noise=args.graph_noise,
                           use_node_feature_importance=args.use_node_feature_importance,
                           use_node_importance=args.use_node_importance,
                           use_node_l2_norm=args.use_node_l2_norm,
                           allow_act_warping=args.allow_act_warping,
                           filter_unpredictable_datasets=args.filter_unpredictable_datasets,
                           filter_unpredictable_graphs=args.filter_unpredictable_graphs,
                           min_n_nodes=args.min_n_nodes,
                           max_n_nodes=args.max_n_nodes,
                           cauchy_dag_offset=args.cauchy_dag_offset,
                           meta_sampling_mode=args.meta_sampling_mode,
                           random_matrix_types=args.random_matrix_types,
                           fct_types=args.fct_types,
                           multi_fct_types=args.multi_fct_types,
                           cat_modes=args.cat_modes,
                           random_weights_types=args.random_weights_types,
                           subsample_feature_nodes=args.subsample_feature_nodes,
                           allow_kumaraswamy_warping=args.allow_kumaraswamy_warping,
                           disallow_y_warping=args.disallow_y_warping,
                           seq_lens_multiple_of=args.seq_lens_multiple_of,
                           use_corrected_num_converters=args.use_corrected_num_converters,
                           use_corrected_cat_sizes=args.use_corrected_cat_sizes,
                           ensure_iid=args.ensure_iid,
                           remove_trivial_datasets=args.remove_trivial_datasets,
                           trivial_dataset_threshold=args.trivial_dataset_threshold,
                           use_corrected_cat_meta_sampling=args.use_corrected_cat_meta_sampling,
                           graph_u_enabled=getattr(args, "graph_u_enabled", False),
                           graph_u_query_location=getattr(args, "graph_u_query_location", 0.0),
                           graph_u_query_scale=getattr(args, "graph_u_query_scale", 1.0),
                           graph_u_force_gaussian=getattr(args, "graph_u_force_gaussian", False),
                           graph_u_max_attempts=getattr(args, "graph_u_max_attempts", 1000))

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--graph_noise",
            default=False,
            type=str2bool,
            help="Whether to add Gaussian noise to the nodes of DAG when using graph_scm prior",
        )
        parser.add_argument(
            "--use_node_feature_importance",
            default=True,
            type=str2bool,
            help="Whether to sample random feature importances on nodes when using the graph_scm prior",
        )
        parser.add_argument(
            "--use_node_importance",
            default=True,
            type=str2bool,
            help="Whether to sample random node importances when using the graph_scm prior",
        )
        parser.add_argument(
            "--use_node_l2_norm",
            default=True,
            type=str2bool,
            help="Whether to L2-normalize the features on the node when using the graph_scm prior",
        )
        parser.add_argument(
            "--allow_act_warping",
            default=False,
            type=str2bool,
            help="Whether to include random activations as a numerical feature warping method in the graph_scm prior",
        )
        parser.add_argument(
            "--allow_kumaraswamy_warping",
            default=True,
            type=str2bool,
            help="Whether to include Kumaraswamy warping in the graph_scm prior",
        )
        parser.add_argument(
            "--disallow_y_warping",
            default=False,
            type=str2bool,
            help="Whether to exclude warping on y",
        )
        parser.add_argument(
            "--filter_unpredictable_datasets",
            default=False,
            type=str2bool,
            help="Whether to remove datasets from the prior where a simple ExtraTrees is not better than a dummy model",
        )
        parser.add_argument(
            "--filter_unpredictable_graphs",
            default=False,
            type=str2bool,
            help="Whether to filter graphs from the prior where y is independent of x",
        )
        parser.add_argument(
            "--min_n_nodes",
            default=2,
            type=int,
            help="Minimum number of nodes in the causal graph",
        )
        parser.add_argument(
            "--max_n_nodes",
            default=32,
            type=int,
            help="Maximum number of nodes in the causal graph",
        )
        parser.add_argument(
            "--cauchy_dag_offset",
            default=0.0,
            type=float,
            help="Offset for node density calculation in Cauchy DAG, default=0.0. "
                 "Larger values will generate denser graphs on average.",
        )
        parser.add_argument(
            "--meta_sampling_mode",
            default="meta",
            type=str,
            help="One of 'meta' (default), 'local' (will always sample iid stuff), "
                 "'global' (will always sample the same thing)"
        )
        parser.add_argument(
            "--random_matrix_types",
            default="default",
            type=str,
            help="Which types of random matrix to sample ('default' or 'gaussian' for only gaussian)"
        )
        parser.add_argument(
            "--fct_types",
            default="default",
            type=str,
            help="Which types of functions to sample ('default', 'tabpfnv2' for only NN/Tree/Discretization)"
        )
        parser.add_argument(
            "--multi_fct_types",
            default="default",
            type=str,
            help="Which types of multi-functions to sample ('default' for both, 'concat', 'agg')"
        )
        parser.add_argument(
            "--cat_modes",
            default="default",
            type=str,
            help="Which categorical discretization modes to use ('default', 'neighbor', 'softmax', 'neighbor_id', "
                 "'neighbor_disc', 'neighbor_func', 'neighbor_int', 'softmax_id', 'softmax_disc', 'softmax_int')"
        )
        parser.add_argument(
            "--random_weights_types",
            default="simple",
            type=str,
            help="Which types of random weights to sample ('simple' or 'uniform')"
        )
        parser.add_argument(
            "--subsample_feature_nodes",
            default=True,
            type=str2bool,
            help="Whether to sample a subset of nodes such that features can only be placed on those nodes",
        )
        parser.add_argument(
            "--seq_lens_multiple_of",
            default=1,
            type=int,
            help="Sequence lengths are rounded to multiples of this number",
        )
        parser.add_argument(
            "--use_corrected_num_converters",
            default=False,
            type=str2bool,
            help="Whether to apply warping to the column output instead of the propagated node values",
        )
        parser.add_argument(
            "--use_corrected_cat_sizes",
            default=False,
            type=str2bool,
            help="Whether to allow larger categorical cardinalities than 9",
        )
        parser.add_argument(
            "--ensure_iid",
            default=False,
            type=str2bool,
            help="Whether to ensure that the data is IID by first fitting all the transforms on a separate forward pass.",
        )
        parser.add_argument(
            "--remove_trivial_datasets",
            default=False,
            type=str2bool,
            help="Whether to remove datasets from the prior where a simple ExtraTrees is almost perfect",
        )
        parser.add_argument(
            "--trivial_dataset_threshold",
            default=0.05,
            type=float,
            help="If remove_trivial_dataset=True, datasets are filtered if the extratrees RMSE is below the threshold times the dummy RMSE",
        )
        parser.add_argument(
            "--use_corrected_cat_meta_sampling",
            default=False,
            type=str2bool,
            help="Whether to use the corrected meta-sampling for categoricals.",
        )
        parser.add_argument(
            "--graph_u_enabled",
            default=False,
            type=str2bool,
            help="Whether to require and shift one hidden root confounder in graph_scm tasks.",
        )
        parser.add_argument(
            "--graph_u_query_location",
            default=0.0,
            type=float,
            help="Location added to the selected confounder's query source samples.",
        )
        parser.add_argument(
            "--graph_u_query_scale",
            default=1.0,
            type=float,
            help="Positive scale applied to the selected confounder's query source samples.",
        )
        parser.add_argument(
            "--graph_u_force_gaussian",
            default=False,
            type=str2bool,
            help=(
                "Whether the selected confounder must use a Gaussian base source "
                "instead of the usual randomly selected source family."
            ),
        )
        parser.add_argument(
            "--graph_u_max_attempts",
            default=1000,
            type=int,
            help="Maximum graph or dataset attempts used to produce a valid Graph-U task.",
        )

