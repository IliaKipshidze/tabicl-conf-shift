"""Check common-support stage resources, mappings and shell execution locally."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import pytest
from test_nano_graph_u_slurm import shell_path

pytest_plugins = ("test_nano_graph_u_slurm",)
REPO = Path(__file__).resolve().parents[1]
STAGES = {"generate": 66, "tune": 300, "select": 1, "test": 30, "summarize": 1}
ARRAYS = {"generate": 66, "tune": 300, "test": 30}
COMMAND_STAGE = {"tune": "trial"}


def script(stage):
    return REPO / "scripts" / f"slurm_nano_graph_u_pool_{stage}.sh"


@pytest.fixture
def pool_env(tmp_path):
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith("NANO_") or key in ("SLURM_ARRAY_TASK_ID", "WORKERS"):
            env.pop(key, None)
    env.update(NANO_CLUSTER_ROOT=shell_path(tmp_path), NANO_REPO_DIR=shell_path(REPO),
               NANO_CONDA_ROOT=shell_path(tmp_path / "conda"),
               NANO_ENV_PATH=shell_path(tmp_path / "env"),
               NANO_POOL_ROOT=shell_path(tmp_path / "pool experiment"),
               NANO_EXPERIMENT_ROOT=shell_path(tmp_path / "old experiment"),
               SLURM_CPUS_PER_TASK="8", POOL_TEST_LOG=shell_path(tmp_path / "arguments.txt"),
               POOL_TEST_EXIT="0", POOL_TEST_GPU_EXIT="0")
    return env


def run_stage(bash, env, stage, *args):
    return subprocess.run([bash, str(script(stage)), *args], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=30, check=False)


def commands(output):
    return [shlex.split(line.removeprefix("Command:")) for line in output.splitlines()
            if line.startswith("Command:")]


def plans(output, prefix):
    return [dict(item.split("=", 1) for item in line.split())
            for line in output.splitlines() if line.startswith(prefix + "=")]


@pytest.mark.parametrize("stage,count", STAGES.items())
def test_syntax_and_read_only_dry_run_without_python_or_conda(bash, pool_env, tmp_path, stage, count):
    syntax = subprocess.run([bash, "-n", str(script(stage))], capture_output=True, text=True, check=False)
    assert syntax.returncode == 0, syntax.stderr
    pool_env["PATH"] = "/no-commands-needed-for-preview"
    before = set(tmp_path.rglob("*"))
    result = run_stage(bash, pool_env, stage, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert set(tmp_path.rglob("*")) == before
    calls = commands(result.stdout)
    assert len(calls) == count
    for task_id, call in enumerate(calls, 1):
        assert call[:5] == ["python", "-u", "-m", "tabicl.nano_graph_u.pool_experiment",
                            COMMAND_STAGE.get(stage, stage)]
        options = dict(zip(call[5::2], call[6::2]))
        assert options["--root"] == pool_env["NANO_POOL_ROOT"]
        if stage in ARRAYS:
            assert options["--task-id"] == str(task_id)
        else:
            assert "--task-id" not in options
        assert "--device" in options if stage in ("tune", "test") else "--device" not in options
    assert "mixed equal per-task choices=uniform,0.2n,0.5n,1n" in result.stdout


def test_generation_preview_matches_all_python_bank_specs(bash, pool_env, tmp_path):
    from tabicl.nano_graph_u.pool_experiment import bank_spec

    result = run_stage(bash, pool_env, "generate", "--dry-run")
    rows = plans(result.stdout, "bank")
    dumps = [line.removeprefix("Dump=") for line in result.stdout.splitlines() if line.startswith("Dump=")]
    assert len(rows) == len(set(dumps)) == 66
    for task_id, (row, dump) in enumerate(zip(rows, dumps), 1):
        spec = bank_spec(tmp_path / "pool experiment", task_id)
        assert row["graph_u_mode"] == spec["mode"]
        assert row["condition"] == spec["condition"]
        assert row["repetition"] == str(spec["repetition"] or "shared")
        assert row["split"] == spec["split"]
        assert int(row["seed"]) == spec["seed"]
        assert int(row["batches"]) == spec["steps"]
        assert int(row["tables"]) == spec["steps"] * 32
        assert row["workers"] == "8"
        assert dump == shell_path(Path(spec["dump"]))


def test_tuning_preview_matches_all_python_candidate_specs(bash, pool_env, tmp_path):
    from tabicl.nano_graph_u.pool_experiment import trial_spec

    result = run_stage(bash, pool_env, "tune", "--dry-run")
    rows = plans(result.stdout, "trial")
    candidates = [line.removeprefix("Candidate=") for line in result.stdout.splitlines()
                  if line.startswith("Candidate=")]
    assert len(rows) == len(set(candidates)) == 300
    for task_id, (row, candidate) in enumerate(zip(rows, candidates), 1):
        spec = trial_spec(tmp_path / "pool experiment", task_id)
        assert (row["graph_u_mode"], row["condition"], int(row["repetition"])) == (
            spec["mode"], spec["condition"], spec["repetition"])
        assert float(row["learning_rate"]) == spec["learning_rate"]
        assert int(row["model_seed"]) == spec["model_seed"]
        assert row["epochs"] == "1" and row["updates"] == "2500"
        assert candidate == shell_path(Path(spec["directory"]))


def test_testing_preview_matches_all_frozen_winners(bash, pool_env, tmp_path):
    from tabicl.nano_graph_u.pool_experiment import winner_spec

    result = run_stage(bash, pool_env, "test", "--dry-run")
    rows = plans(result.stdout, "test")
    winners = [line.removeprefix("Winner=") for line in result.stdout.splitlines() if line.startswith("Winner=")]
    assert len(rows) == len(set(winners)) == 30
    for task_id, (row, winner) in enumerate(zip(rows, winners), 1):
        spec = winner_spec(tmp_path / "pool experiment", task_id)
        assert (row["training_mode"], row["training"], int(row["repetition"])) == (
            spec["mode"], spec["condition"], spec["repetition"])
        assert row["test_banks"] == "6" and row["test_tables_per_bank"] == "3200"
        assert row["own_training_tables"] == "80000"
        assert winner == shell_path(Path(spec["checkpoint"]))


@pytest.mark.parametrize("stage,count", ARRAYS.items())
@pytest.mark.parametrize("invalid", ["", "0", "999", "../1", "01", "1+1"])
def test_invalid_array_task_fails_before_activation(bash, pool_env, tmp_path, stage, count, invalid):
    pool_env["SLURM_ARRAY_TASK_ID"] = invalid
    result = run_stage(bash, pool_env, stage, "--dry-run")
    assert result.returncode != 0
    assert f"from 1 to {count}" in result.stderr
    assert not (tmp_path / "pool experiment").exists()


@pytest.mark.parametrize("stage,count", ARRAYS.items())
def test_array_task_override_previews_only_requested_task(bash, pool_env, stage, count):
    pool_env["SLURM_ARRAY_TASK_ID"] = str(count)
    result = run_stage(bash, pool_env, stage, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert len(commands(result.stdout)) == 1
    assert f"--task-id {count}" in result.stdout


@pytest.mark.parametrize("stage", STAGES)
def test_default_root_ignores_old_experiment_variable(bash, pool_env, stage):
    pool_env.pop("NANO_POOL_ROOT")
    pool_env["SLURM_ARRAY_TASK_ID"] = "1"
    result = run_stage(bash, pool_env, stage, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert commands(result.stdout)[0][6] == pool_env["NANO_CLUSTER_ROOT"] + "/tabicl-conf-shift-common-support-experiment"


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("target", ["relative", "repo", "old", "parent", "normalized"])
def test_unsafe_output_roots_fail_without_writes(bash, pool_env, tmp_path, stage, target):
    roots = {"relative": "relative/directory", "repo": pool_env["NANO_REPO_DIR"] + "/outputs",
             "old": pool_env["NANO_EXPERIMENT_ROOT"], "parent": pool_env["NANO_CLUSTER_ROOT"],
             "normalized": pool_env["NANO_CLUSTER_ROOT"] + "/new/../tabicl-conf-shift-priors/outputs"}
    pool_env["NANO_POOL_ROOT"] = roots[target]
    result = run_stage(bash, pool_env, stage, "--dry-run")
    assert result.returncode != 0
    assert "absolute path" in result.stderr or "must not overlap" in result.stderr
    assert not (tmp_path / "pool experiment").exists()


def test_workers_follow_allocation_and_reject_oversubscription(bash, pool_env):
    pool_env.update(SLURM_ARRAY_TASK_ID="1", SLURM_CPUS_PER_TASK="4")
    result = run_stage(bash, pool_env, "generate", "--dry-run")
    assert result.returncode == 0 and "--workers 4" in result.stdout
    pool_env["WORKERS"] = "5"
    result = run_stage(bash, pool_env, "generate", "--dry-run")
    assert result.returncode != 0 and "must not exceed" in result.stderr


@pytest.mark.parametrize("stage", STAGES)
def test_unknown_arguments_rejected(bash, pool_env, stage):
    result = run_stage(bash, pool_env, stage, "--unknown")
    assert result.returncode != 0 and "Usage:" in result.stderr


@pytest.fixture
def fake_environment(tmp_path):
    initialization = tmp_path / "conda" / "etc" / "profile.d" / "conda.sh"
    initialization.parent.mkdir(parents=True)
    initialization.write_text(
        'conda() { return 0; }\n'
        'python() { if [[ "${1:-}" == "-" ]]; then return "${POOL_TEST_GPU_EXIT}"; fi; '
        'printf "%s\\n" "$@" > "${POOL_TEST_LOG}"; return "${POOL_TEST_EXIT}"; }\n', encoding="utf-8")


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("exit_code", ["0", "7"])
def test_stage_executes_one_command_and_propagates_failure(
    bash, pool_env, tmp_path, fake_environment, stage, exit_code
):
    pool_env.update(SLURM_ARRAY_TASK_ID="1", POOL_TEST_EXIT=exit_code)
    result = run_stage(bash, pool_env, stage)
    assert result.returncode == int(exit_code), result.stderr
    arguments = (tmp_path / "arguments.txt").read_text().splitlines()
    command_stage = COMMAND_STAGE.get(stage, stage)
    assert arguments[:4] == ["-u", "-m", "tabicl.nano_graph_u.pool_experiment", command_stage]
    assert (f"POOL_{command_stage.upper()}_COMPLETE" in result.stdout) == (exit_code == "0")


@pytest.mark.parametrize("stage", ["tune", "test"])
def test_gpu_preflight_failure_prevents_model_command(bash, pool_env, tmp_path, fake_environment, stage):
    pool_env.update(SLURM_ARRAY_TASK_ID="1", POOL_TEST_GPU_EXIT="9")
    result = run_stage(bash, pool_env, stage)
    assert result.returncode == 9
    assert not (tmp_path / "arguments.txt").exists()


def test_stage_resource_requests_and_thread_limits():
    resources = {"generate": ("wdr", 8, "48:00:00"), "tune": ("gpu", 2, "24:00:00"),
                 "select": ("wdr", 1, "01:00:00"), "test": ("gpu", 2, "12:00:00"),
                 "summarize": ("wdr", 1, "01:00:00")}
    for stage, (partition, cpus, time) in resources.items():
        source = script(stage).read_text()
        assert source.startswith("#!/usr/bin/env bash\n") and "set -euo pipefail" in source
        assert f"#SBATCH --partition={partition}" in source
        assert f"#SBATCH --cpus-per-task={cpus}" in source
        assert f"#SBATCH --time={time}" in source
        assert ("#SBATCH --gres=gpu:1" in source) == (partition == "gpu")
        if stage in ARRAYS:
            assert f"#SBATCH --array=1-{ARRAYS[stage]}%2" in source and "%A_%a.out" in source
        else:
            assert "#SBATCH --array=" not in source and "%j.out" in source
        for setting in ("OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1", "NUMEXPR_NUM_THREADS=1"):
            assert setting in source
