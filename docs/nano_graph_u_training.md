# Paper-scale Nano-Graph-U at CISPA

Nano-Graph-U is a **separate preliminary experiment**. It uses the paper-era
[`nanoTabPFN` model](https://github.com/automl/nanoTabPFN/tree/530670098e4befabe80825a3eefed408a926e34a)
and training recipe with this repository's existing constructed Graph-U
`graph_scm` prior. It does not replace `python -m tabicl.train`, modify full
TabICL checkpoints, or claim that Graph-U was validated by the Nano paper.

The default full Nano run follows the [paper's](https://arxiv.org/html/2511.03634v2)
model/training scale: 3 layers, 4 heads, embedding 96, hidden MLP 192, binary
output, 150 rows, 5 requested features, batch 32, 2,500 optimizer updates
(80,000 tables), `schedulefree.AdamWScheduleFree` with learning rate 0.004 and
zero weight decay, query cross-entropy, and gradient clipping at 1. The model
has 356,066 parameters. Data are pre-generated as in the paper. The Graph-U
prior is intentionally different from the paper's ordinary synthetic prior.

## What is preserved from the full Graph-U path

- The same `PriorDataset`/`GraphPrior` implementation and `graph_scm` mechanisms.
- The `add_root` hidden confounder and its propagation through structural
  equations. Hidden U is not an observed input column.
- Random root-source families (`graph_u_force_gaussian=False`).
- Graph/dataset predictability filters and support-only transformation fitting.
- The existing 0.3-0.9 support fraction, rather than the paper dump's broader
  0.1-0.9 range. This is an explicit prior-distribution difference.

Each Nano batch contains 32 distinct tables, not one shared SCM. The prior
still uses groups of four. `seq_len_per_gp=False` makes all 32 tables in a
single batch share one newly sampled support/query row index; the next batch
samples another. The dump stores an index for **every table**, and the loader
rejects any batch in which those indices differ. Consequently the same index
controls the generated U shift, the labels shown to Nano, and the query loss.
This deliberately avoids the paper loader's `single_eval_pos[0]` behavior when
stored per-table indices differ.

Generation uses one CPU worker **within** each batch and, by default, eight
independent batches in parallel for full jobs. Every batch has its own
deterministic seed; the main process writes them to HDF5 in order, so an
interrupted dump can resume without changing its task stream. Pre-generation
can still take much longer than model training; the paper's reported training
time excludes prior generation and evaluation. The supplied Slurm scripts use
the known-working GPU partition, so the allocated GPU sits idle during CPU
pre-generation. A failed prior batch is retried with up to eight deterministic
seeds; if every retry fails, the job stops with a clear error. Resumption also
checks a fingerprint of the generator source files, preventing a partially
generated dump from silently mixing code versions.

## Install/update the existing CISPA environment

If the working `tabicl-conf-shift` environment already exists, install just
the two Nano-specific packages and refresh the editable checkout without
letting pip replace PyTorch. Run these commands from the repository directory:

```bash
source /home/bin/CISPA-scratch/c01ilki/miniconda3/etc/profile.d/conda.sh
conda activate /home/bin/CISPA-scratch/c01ilki/miniconda3/envs/tabicl-conf-shift
python -m pip install --no-deps 'h5py>=3,<4' 'schedulefree>=1.4,<2'
python -m pip install --no-deps -e .
```

For a fresh environment or a broader environment refresh, the existing setup
script remains available:

```bash
bash scripts/setup_cispa_env.sh
```

The setup script updates the whole environment and reinstalls the
CUDA-12.1-compatible PyTorch 2.5.1 wheel; the narrow commands above are safer
for an already-working environment. Do **not** install
the whole TFM-Playground package in this environment: its current Python 3.12
and PyTorch 2.9 requirements do not match the working cluster setup.

## First, an end-to-end smoke job

```bash
sbatch scripts/slurm_nano_graph_u.sh
```

The default smoke job generates four 64-row, 3-feature Graph-U tables and runs
two genuine Nano updates. It tests wiring only; it is **not** a paper-scale
result. Check the `slurm-nano-graph-u-<job-id>.out` and `.err` files under
`/home/bin/CISPA-scratch/c01ilki` and require a final `checkpoint=...` line.

## Full identity and shifted training runs

Run the no-shift control and treatment in separate checkpoint/dump directories:

```bash
sbatch --export=ALL,RUN_MODE=full,CONDITION=identity scripts/slurm_nano_graph_u.sh
sbatch --export=ALL,RUN_MODE=full,CONDITION=shift,QUERY_LOCATION=2.0,QUERY_SCALE=1.5 scripts/slurm_nano_graph_u.sh
```

The control has query location 0 and scale 1; the treatment uses location 2 and
scale 1.5. Both use the same model and prior settings. The default data/model
seeds are 42, but the generated tasks are not guaranteed to remain one-to-one
paired across conditions because classification validity checks may reject
different sampled SCMs. Never compare either run's score on its own training
dump as scientific evaluation.

The script resumes an interrupted HDF dump or `latest.pt` checkpoint on an
identical resubmission. Generation metadata and checkpoint settings are checked
before resuming. Do not change seeds, model settings, or dump contents while
resuming. The script locks the dump and checkpoint directory, so an accidental
duplicate submission fails rather than writing concurrently. Every 250
updates, training saves `latest.pt` and a
numbered `step-*.pt`. These files are separate from TabICL's `.ckpt` files.

## Frozen synthetic evaluation

Use independent evaluation data (default seed 424242). Evaluate **both**
checkpoints on the same identity dump and the same shifted dump:

```bash
sbatch --export=ALL,CHECKPOINT=/absolute/path/to/nano_identity/latest.pt,EVAL_CONDITION=identity scripts/slurm_nano_graph_u_eval.sh
sbatch --export=ALL,CHECKPOINT=/absolute/path/to/nano_shift/latest.pt,EVAL_CONDITION=identity scripts/slurm_nano_graph_u_eval.sh
sbatch --export=ALL,CHECKPOINT=/absolute/path/to/nano_identity/latest.pt,EVAL_CONDITION=shift scripts/slurm_nano_graph_u_eval.sh
sbatch --export=ALL,CHECKPOINT=/absolute/path/to/nano_shift/latest.pt,EVAL_CONDITION=shift scripts/slurm_nano_graph_u_eval.sh
```

Substitute the real checkpoint paths printed by the training jobs. Evaluation
dumps are keyed by condition/seed/size and protected by a file lock, so both
models see the exact same task bank **within each evaluation condition**.
JSON reports are written under
`/home/bin/CISPA-scratch/c01ilki/tabicl-conf-shift-evaluations/nano_graph_u/`.
They contain mean per-task ROC-AUC, query NLL, Brier score, accuracy, balanced
accuracy, macro-F1, task counts, and checkpoint/dump hashes. AUC is omitted
for a task whose query set has only one class, with its count reported.

This synthetic evaluation tests the confounder shift. Reproducing the Nano
paper's TabArena real-data benchmark would be a separate validation exercise;
the provided commands do not claim to do so. The primary comparison is
shift-trained versus identity-trained Nano on the same shifted evaluation
bank, together with their performance on the identity bank. It does not by
itself establish the effect for the full-size TabICL model.
