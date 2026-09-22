# Diagnosing chance-level Nano-Graph-U results

The shifted Nano run reached 2,500 steps, but its independent evaluation was
near chance in both shifted and identity conditions. This is not evidence for
or against the confounder-shift hypothesis: first determine whether the model
learned useful query predictions at all.

The diagnostic job reads the existing training and evaluation dumps and the
saved step-250, step-1000, and step-2500 checkpoints. It never changes them.
It performs three separate checks:

1. Score all three checkpoints on the same first 128 tables of each frozen
   evaluation bank, recording predictive metrics and probability variation.
2. Fit a support-only ExtraTrees classifier separately on each of those tables
   and score its query rows. This tests whether the *saved* tasks carry an
   accessible support-to-query signal; it is not a pretrained baseline.
3. Initialize a fresh Nano model and repeatedly train it on one fixed table
   from the training dump. This tests whether the architecture, loss, and
   optimizer can memorize that table; it is not a generalization evaluation.

From `/home/bin/CISPA-scratch/c01ilki/tabicl-conf-shift`, submit:

```bash
sbatch scripts/slurm_nano_graph_u_diagnose.sh
```

Check the returned job ID with `sacct` and the matching
`slurm-nano-gu-diag-<job-id>.out/.err` files under the scratch directory.
The job prints its new result directory, normally
`tabicl-conf-shift-evaluations/nano_graph_u/diagnostics/job-<job-id>/`.
There are three JSON reports there: `shifted_eval.json`,
`identity_eval.json`, and `fixed_table_overfit.json`. Result paths are
create-only, so a repeated diagnostic job gets a distinct directory.

These are diagnostic samples, not replacements for the 3,200-table full
evaluation. A nearly constant Nano probability together with a useful
ExtraTrees query AUC would point toward model training or optimization rather
than an inherently impossible prior. If the fixed-table NLL falls far below
0.693, Nano can learn at least one table; if it remains near 0.693, investigate
the model/optimizer/data interface before any additional full training run.
Comparing the three saved checkpoints can show whether performance collapsed
late or never rose above chance. A low ExtraTrees AUC on this 128-table subset
does not prove that all Graph-U tasks lack signal, and successful fixed-table
overfitting establishes memorization only, not generalization.

The script defaults to the existing shifted 2,500-step run and evaluation
banks. It requests one GPU and eight CPUs. Do not run it on the login node.
