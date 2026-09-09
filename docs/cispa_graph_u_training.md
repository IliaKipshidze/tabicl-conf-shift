# Training the constructed Graph-U classifier at CISPA

These commands target the account layout recorded for `c01ilki`:

- repository: `/home/bin/CISPA-scratch/c01ilki/tabicl-conf-shift`
- Miniconda: `/home/bin/CISPA-scratch/c01ilki/miniconda3`
- environment: `/home/bin/CISPA-scratch/c01ilki/miniconda3/envs/tabicl-conf-shift`
- checkpoints: `/home/bin/CISPA-scratch/c01ilki/tabicl-conf-shift-checkpoints`
- Slurm partition: `gpu`, one GPU and eight CPUs

Runtime repository, environment, and checkpoint paths can be overridden through
the variables documented in the scripts; the Slurm log directives intentionally
remain fixed to this `c01ilki` layout.

## 1. Clone the Linux copy

From `lllogin`:

```bash
cd /home/bin/CISPA-scratch/c01ilki
git clone --branch graph-u-source-shift \
  https://github.com/IliaKipshidze/tabicl-conf-shift.git
cd tabicl-conf-shift
```

Cloning is preferable to copying the Windows folder because the local Windows
`.venv` cannot be used on Linux. If the repository already exists, update it:

```bash
cd /home/bin/CISPA-scratch/c01ilki/tabicl-conf-shift
git switch graph-u-source-shift
git pull --ff-only
```

## 2. Create the environment once

```bash
bash scripts/setup_cispa_env.sh
```

`environment.cispa.yml` selects Python 3.11 and the setup script installs this
repository with its `pretrain` and `test` extras. It first pins the official
PyTorch 2.5.1 CUDA 12.1 wheel, which is compatible with the CUDA 12.x driver on
the current CISPA A100 nodes. This also prevents the unbounded `torch>=2.2`
dependency from selecting an incompatible CUDA 13 wheel. CUDA is validated
inside the Slurm GPU job because a login node may legitimately report no
visible GPU.

## 3. Submit the generator/GPU smoke test

The defaults reproduce the diagnostic shift used during development:
location `2.0`, scale `1.5`, 94 observed features, and a random source family.
The 94-feature default directly covers the case that defeated rejection
sampling. The default structure mode is `add_root`: TabICL first samples its
ordinary base graph and then inserts a hidden root U with direct edges to
distinct observed-X and target nodes. It does not wait for a suitable
confounder to occur by chance.

```bash
sbatch scripts/slurm_graph_u_smoke.sh
```

To inspect another condition:

```bash
sbatch --export=ALL,GRAPH_U_STRUCTURE_MODE=add_root,GRAPH_U_FEATURES=94,GRAPH_U_QUERY_LOCATION=1.0,GRAPH_U_QUERY_SCALE=1.5 \
  scripts/slurm_graph_u_smoke.sh
```

The diagnostic prints `base graphs rejected for a missing U: 0` in
`add_root` mode and verifies that every dataset records the inserted U as a
hidden root. The older rejection-conditioned implementation remains available
for comparison with `GRAPH_U_STRUCTURE_MODE=reject`; only that mode can report
multiple structural attempts.

The output and error filenames include the Slurm job ID and are written under
`/home/bin/CISPA-scratch/c01ilki`.

## 4. Submit a short training pilot

The training script deliberately defaults to a 100-step pilot:

```bash
sbatch scripts/slurm_train_graph_u_clf_stage1.sh
```

This is a real `add_root` training run, but its checkpoint directory is separate
from the full run. Check its runtime, GPU memory, loss, generator throughput,
and saved checkpoints before starting the long run. Both checkpoint paths and
run names contain the structure mode, preventing `add_root` and `reject` jobs
from resuming each other's checkpoints.

## 5. Submit the full Stage-1 run

After choosing the experimental shift parameters, specify them explicitly:

```bash
sbatch --export=ALL,RUN_MODE=full,GRAPH_U_STRUCTURE_MODE=add_root,GRAPH_U_QUERY_LOCATION=2.0,GRAPH_U_QUERY_SCALE=1.5 \
  scripts/slurm_train_graph_u_clf_stage1.sh
```

The values above are an example matching the development smoke test, not a
recommendation for the final experiment. Full mode refuses to start unless
both shift variables are explicitly supplied.

Stage 1 follows the upstream 500,000-step TabICLv2 classifier recipe. A single
48-hour allocation may not finish it. Resubmitting the identical command uses
the same parameter-specific checkpoint directory; TabICL finds the latest
checkpoint there and restores the model, optimizer, scheduler, and step.

Do not change `MAX_STEPS`, structure mode, location, scale, or the checkpoint
directory between resubmissions of the same experiment. A pilot checkpoint
should not seed the full run because its learning-rate schedule was created for
a different number of steps. More generally, do not change other training
settings while reusing a checkpoint directory, and do not run two jobs against
the same directory concurrently; not every override is encoded in its name.

Useful monitoring commands:

```bash
squeue -u "$USER"
tail -f /home/bin/CISPA-scratch/c01ilki/slurm-tabicl-gu-clf-s1-<job-id>.out
tail -f /home/bin/CISPA-scratch/c01ilki/slurm-tabicl-gu-clf-s1-<job-id>.err
```
