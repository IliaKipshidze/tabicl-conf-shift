# The TabICLv2 Prior

The TabICLv2 prior is divided into samplers for different types of objects: 
datasets, functions, graphs, matrices, and so on.
Many sampler types are implemented as base classes with subclasses,
where the base class (e.g., RandomFunction) samples a random subclass 
(e.g., RandomGPFunction),
and then the subclass implements a specific mechanism for sampling.

One can sample a dataset as follows:
```python
from tabicl.prior.graph_lib.dataset import RandomDataset
from tabicl.prior.graph_lib.base import Context, DatasetProperties

# sample an ordinary dataset with 1000 total samples
# (Graph-U additionally uses n_train as the support/query boundary)
# get 2 numerical features (cat size 0), 
# one categorical with (up to) 4 different values,
# and one target with (up to) 3 different classes.
ds_prop = DatasetProperties(n_train=1000, 
                            n_test=0, 
                            cat_sizes={"x": [0, 0, 4], "y": [3]})
tensors = RandomDataset(Context()).sample(ds_prop).get_concat_tensors()
x_num = tensors['x_num']  # (n_samples, n_features)
x_cat = tensors['x_cat']
y_cat = tensors['y_cat']
```

However, this is **not the full dataset sampling logic**: 
Preprocessing, filtering, and some hyperparameter sampling are done in
`prior/dataset.py` and `prior/graph_scm.py`.

## Essential classes
- `PriorConfig` (`config.py`) contains configuration options for the prior. 
(This includes options to fix some unintended behavior 
that was present during TabICLv2 pre-training; 
we set the defaults to retain the TabICLv2 behavior.)
- `GlobalSampler` (`base.py`) allows sampling 
scalar variables from different distributions, 
with different correlation modes:
  - `'global'` samples a single value per name, 
  - `'meta'` samples correlated values for the same name,
  - `'local'` samples independent values for the same name.
- `Context` (`base.py`): Stores a `PriorConfig` and a `GlobalSampler`.
Gets passed to every sampling class.
- `PriorComponent` (`base.py`): Base class for other sampling classes.
Takes a `Context` object and uses it to track potential 
for infinite recursions.
- `RandomTransformer` (`base.py`) and subclasses: 
Implement a simple fit-predict interface for things like random functions
such that they can be applied to multiple tensors 
while only being fitted on the first one. 
Currently, we only apply them once, 
so the fit-transform paradigm is not necessary.

`base.py` also contains classes to 
store and specify datasets and their properties.

## Dataset sampling hierarchy:
A random dataset (`dataset.py`) is sampled 
by sampling a random graph (`graph.py`), assigning features to nodes, and
evaluating the graph using a random graph function (`graph_function.py`).
The graph function evaluates the nodes 
using random node functions (`node_function.py`).
The node function applies different processing steps, 
including converters for extracting dataset features (`converter.py`),
sampling random points (`points.py`) on root nodes, 
and applying random multi-functions (`multi_function.py`) on other nodes. 
The multi-functions use aggregation mechanisms 
with random functions (`function.py`), 
which can use random matrices (`matrix.py`), 
random activations (`activation.py`),
and random weights (`weights.py`).

In addition, `properties.py` provides a way to sample categorical sizes, 
which can be passed to `RandomDataset`.

## Graph-U confounder source shift (fork extension)

`graph_u_enabled=True` conditions the graph prior on the existence of an
unobserved root that is a direct parent of both an observed-X node and a
distinct target node. Only this selected root receives an environment-specific
source distribution; its effects then propagate through the ordinary graph
evaluation.

By default, the selected confounder retains the graph prior's usual randomly
selected base-source family. The implementation samples the support and query
parts together, applies

```text
Z_query = graph_u_query_scale * Z_query + graph_u_query_location
```

only to the query slice, and then applies one shared random root function to
the combined tensor. This affine transformation shifts the exogenous source
feeding the hidden node for any source family. Set
`graph_u_force_gaussian=True` for a Gaussian-only controlled ablation. The
final hidden node can still have a non-Gaussian distribution after TabICL's
random root function and node transformations.

```python
from tabicl.prior import PriorDataset
from tabicl.prior.graph_lib._config import PriorConfig

config = PriorConfig(
    graph_u_enabled=True,
    graph_u_query_location=1.0,
    graph_u_query_scale=1.5,
)
prior = PriorDataset(
    prior_type="graph_scm",
    config=config,
    batch_size=4,
    max_seq_len=256,
    min_train_size=0.5,
    max_train_size=0.5,
    n_jobs=1,
)
X, y, active_features, seq_lens, train_sizes = next(prior)
```

Use the following three conditions to separate graph-conditioning effects from
the source shift:

- original prior: `graph_u_enabled=False`;
- Graph-U identity control: enabled with location `0` and scale `1`;
- Graph-U shift: enabled with a nonzero location and/or nonunit scale.

Run the generation-only diagnostic (no model training) with:

```bash
python scripts/smoke_graph_u.py --datasets 20
```

The diagnostic defaults to a visible shift (location `2`, scale `1.5`). Add
`--query-location 0 --query-scale 1` for the identity control. It reports
rejected graph proposals and temporarily captures the selected hidden root in
memory to verify the source transformation. These latent values are not added
to the generated dataset or saved by the normal prior API.

`ensure_iid=True` is intentionally rejected with Graph-U in this version,
because TabICL's extra evaluation pass currently resamples root-level
mechanisms. Also note that some original TabICL transformations are fitted on
the combined support/query tensor; the intervention is shared within an
episode, but this first implementation does not make those transformations
independent of the sampled environments. Consequently, same-seed identity and
shift tasks select the same graph and U, but their support tensors are not
guaranteed to be identical. Use the Graph-U identity condition as the matched
control; a stricter support-only mechanism-fitting variant is future work.

Per-task structural metadata, including the selected U and its observed
children, is available as `GraphSCM.metadata_` after direct generation. The
standard `PriorDataset` batch interface still returns tensors only. Saved-prior
metadata records the global Graph-U settings, but not a per-task graph ledger.


