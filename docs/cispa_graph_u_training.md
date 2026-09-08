# Training the Graph-U classifier at CISPA

These commands target the account layout recorded for `c01ilki`:

- repository: `/home/bin/CISPA-scratch/c01ilki/tabicl-conf-shift`
- Miniconda: `/home/bin/CISPA-scratch/c01ilki/miniconda3`
- environment: `/home/bin/CISPA-scratch/c01ilki/miniconda3/envs/tabicl-conf-shift`
- checkpoints: `/home/bin/CISPA-scratch/c01ilki/tabicl-conf-shift-checkpoints`
- Slurm partition: `gpu`, one GPU and eight CPUs

All paths can be overridden through the environment variables documented in
the scripts, but no overrides are needed for this layout.

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
repository with its `pretrain` and `test` extras. CUDA is validated inside the
Slurm GPU job because a login node may legitimately report no visible GPU.

## 3. Submit the generator/GPU smoke test

The defaults reproduce the diagnostic shift used during development:
location `2.0`, scale `1.5`, with a random source family.

```bash
sbatch scripts/slurm_graph_u_smoke.sh
```

To inspect another condition:

```bash
sbatch --export=ALL,GRAPH_U_QUERY_LOCATION=1.0,GRAPH_U_QUERY_SCALE=1.5 \
  scripts/slurm_graph_u_smoke.sh
```

The output and error filenames include the Slurm job ID and are written under
`/home/bin/CISPA-scratch/c01ilki`.

## 4. Submit a short training pilot

The training script deliberately defaults to a 100-step pilot:

```bash
sbatch scripts/slurm_train_graph_u_clf_stage1.sh
```

This is a real training run, but its checkpoint directory is separate from the
full run. Check its runtime, GPU memory, loss, generator throughput, and saved
checkpoints before starting the long run.

## 5. Submit the full Stage-1 run

After choosing the experimental shift parameters, specify them explicitly:

```bash
sbatch --export=ALL,RUN_MODE=full,GRAPH_U_QUERY_LOCATION=2.0,GRAPH_U_QUERY_SCALE=1.5 \
  scripts/slurm_train_graph_u_clf_stage1.sh
```

The values above are an example matching the development smoke test, not a
recommendation for the final experiment. Full mode refuses to start unless
both shift variables are explicitly supplied.

Stage 1 follows the upstream 500,000-step TabICLv2 classifier recipe. A single
48-hour allocation may not finish it. Resubmitting the identical command uses
the same parameter-specific checkpoint directory; TabICL finds the latest
checkpoint there and restores the model, optimizer, scheduler, and step.

Do not change `MAX_STEPS`, location, scale, or the checkpoint directory between
resubmissions of the same experiment. A pilot checkpoint should not seed the
full run because its learning-rate schedule was created for a different number
of steps.

Useful monitoring commands:

```bash
squeue -u "$USER"
tail -f /home/bin/CISPA-scratch/c01ilki/slurm-tabicl-gu-clf-s1-<job-id>.out
tail -f /home/bin/CISPA-scratch/c01ilki/slurm-tabicl-gu-clf-s1-<job-id>.err
```
