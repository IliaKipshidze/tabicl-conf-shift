"""Validate the sampled-experiment stage scripts without submitting Slurm jobs."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import pytest

from test_nano_graph_u_slurm import bash, shell_path

REPO = Path(__file__).resolve().parents[1]
STAGES = {"generate": 44, "tune": 200, "select": 2, "test": 80, "summarize": 1}


@pytest.fixture
def sampled_env(tmp_path):
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith("SAMPLE_") or key in ("SLURM_ARRAY_TASK_ID", "WORKERS"):
            env.pop(key, None)
    env.update(
        NANO_CLUSTER_ROOT=shell_path(tmp_path),
        NANO_REPO_DIR=shell_path(REPO),
        NANO_CONDA_ROOT=shell_path(tmp_path / "conda"),
        NANO_ENV_PATH=shell_path(tmp_path / "env"),
        NANO_EXPERIMENT_ROOT=shell_path(tmp_path / "sampled experiment"),
        SLURM_CPUS_PER_TASK="8",
        SAMPLE_TEST_LOG=shell_path(tmp_path / "arguments.txt"),
        SAMPLE_TEST_EXIT_CODE="0",
    )
    return env


def script(stage):
    return REPO / "scripts" / f"slurm_nano_graph_u_sampled_{stage}.sh"


def run_stage(bash, env, stage, *args):
    return subprocess.run(
        [bash, str(script(stage)), *args], cwd=REPO, env=env,
        capture_output=True, text=True, timeout=30, check=False,
    )


def commands(output):
    return [shlex.split(line.removeprefix("Command:"))
            for line in output.splitlines() if line.startswith("Command:")]


def plans(output, prefix):
    return [dict(item.split("=", 1) for item in line.split())
            for line in output.splitlines() if line.startswith(prefix + "=")]


@pytest.mark.parametrize("stage,count", STAGES.items())
def test_dry_run_is_read_only_and_uses_separate_root(bash, sampled_env, tmp_path, stage, count):
    result = run_stage(bash, sampled_env, stage, "--dry-run")
    assert result.returncode == 0, result.stderr
    calls = commands(result.stdout)
    assert len(calls) == count
    assert not (tmp_path / "sampled experiment").exists()
    assert not (tmp_path / "arguments.txt").exists()
    for index, call in enumerate(calls, 1):
        assert call[:5] == ["python", "-u", "-m", "tabicl.nano_graph_u.sampled", stage]
        options = dict(zip(call[5::2], call[6::2]))
        assert options["--root"] == sampled_env["NANO_EXPERIMENT_ROOT"]
        if stage != "summarize":
            assert options["--task-id"] == str(index)
        assert "lr_search_tests" not in result.stdout
    assert "slurm_nano_graph_u_generate.sh" not in result.stdout


def test_generation_maps_exactly_all_44_banks(bash, sampled_env):
    result = run_stage(bash, sampled_env, "generate", "--dry-run")
    observed = plans(result.stdout, "bank")
    expected = set()
    for mode in ("confounding", "covariate"):
        for rep in range(1, 6):
            for split, seed, batches in (("train", 1000 + rep, 2500), ("test", 2000 + rep, 100)):
                for condition in ("identity", "shift"):
                    expected.add((mode, str(rep), split, condition, str(seed), str(batches)))
        for condition in ("identity", "shift"):
            expected.add((mode, "shared", "validation", condition, "3001", "100"))
    assert {(row["graph_u_mode"], row["repetition"], row["split"], row["condition"],
             row["seed"], row["batches"]) for row in observed} == expected
    dumps = [line for line in result.stdout.splitlines() if line.startswith("Dump=")]
    assert len(set(dumps)) == 44
    for row in observed:
        assert row["shift_mode"] == ("fixed" if row["condition"] == "identity" else "sampled")
        assert int(row["tables"]) == int(row["batches"]) * 32
        assert row["workers"] == "8"


def test_tuning_maps_all_candidates_with_same_seed_per_repetition(bash, sampled_env):
    result = run_stage(bash, sampled_env, "tune", "--dry-run")
    rates = (0.0001, 0.0002, 0.0003, 0.0005, 0.0007, 0.001, 0.0015, 0.002, 0.003, 0.004)
    rows = plans(result.stdout, "trial")
    keys = set()
    for global_id, row in enumerate(rows, 1):
        local = (global_id - 1) % 100 + 1
        assert row["graph_u_mode"] == ("confounding" if global_id <= 100 else "covariate")
        assert int(row["local_trial"]) == local
        assert int(row["model_seed"]) == 41 + int(row["repetition"])
        assert float(row["learning_rate"]) == rates[(local - 1) % 10]
        keys.add((row["graph_u_mode"], row["repetition"], row["condition"], row["learning_rate"]))
    assert len(keys) == 200


def test_final_testing_covers_full_four_by_four_matrix(bash, sampled_env):
    result = run_stage(bash, sampled_env, "test", "--dry-run")
    rows = plans(result.stdout, "test")
    keys = {(row["training_mode"], row["training"], row["test_mode"], row["evaluation"], row["repetition"])
            for row in rows}
    expected = {(train_mode, train_condition, test_mode, test_condition, str(rep))
                for train_mode in ("confounding", "covariate")
                for train_condition in ("identity", "shift")
                for test_mode in ("confounding", "covariate")
                for test_condition in ("identity", "shift") for rep in range(1, 6)}
    assert keys == expected
    assert all(row["tables"] == "3200" for row in rows)
    results = [line for line in result.stdout.splitlines() if line.startswith("Result=")]
    assert len(set(results)) == 80


def test_custom_distribution_parameters_are_exposed_by_generation(bash, sampled_env):
    sampled_env.update(SLURM_ARRAY_TASK_ID="2", SAMPLE_SCALE_MEAN="1.6", SAMPLE_SCALE_STD="0.05",
                       SAMPLE_SCALE_MIN="1.4", SAMPLE_SCALE_MAX="1.8", SAMPLE_LOCATION_MEAN="2.2",
                       SAMPLE_LOCATION_STD="0.1", SAMPLE_LOCATION_MIN="2.0", SAMPLE_LOCATION_MAX="2.4")
    result = run_stage(bash, sampled_env, "generate", "--dry-run")
    options = dict(zip(commands(result.stdout)[0][5::2], commands(result.stdout)[0][6::2]))
    for parameter in ("scale", "location"):
        for statistic in ("mean", "std", "min", "max"):
            assert options[f"--query-{parameter}-{statistic}"] == sampled_env[f"SAMPLE_{parameter}_{statistic}".upper()]


@pytest.mark.parametrize("stage,count", [(s, c) for s, c in STAGES.items() if s != "summarize"])
@pytest.mark.parametrize("invalid", ["", "0", "999", "../1"])
def test_invalid_array_id_fails_before_environment_activation(bash, sampled_env, tmp_path, stage, count, invalid):
    sampled_env["SLURM_ARRAY_TASK_ID"] = invalid
    result = run_stage(bash, sampled_env, stage)
    assert result.returncode != 0
    assert f"from 1 to {count}" in result.stderr
    assert not (tmp_path / "sampled experiment").exists()


def test_worker_count_follows_allocation_and_rejects_oversubscription(bash, sampled_env):
    sampled_env.update(SLURM_ARRAY_TASK_ID="1", SLURM_CPUS_PER_TASK="4")
    result = run_stage(bash, sampled_env, "generate", "--dry-run")
    assert result.returncode == 0
    assert "--workers 4" in result.stdout
    sampled_env["WORKERS"] = "5"
    result = run_stage(bash, sampled_env, "generate", "--dry-run")
    assert result.returncode != 0
    assert "must not exceed" in result.stderr


@pytest.fixture
def fake_environment(tmp_path):
    initialization = tmp_path / "conda" / "etc" / "profile.d" / "conda.sh"
    initialization.parent.mkdir(parents=True)
    initialization.write_text(
        'conda() { return 0; }\n'
        'python() { if [[ "${1:-}" == "-" ]]; then return 0; fi; '
        'printf "%s\\n" "$@" > "${SAMPLE_TEST_LOG}"; return "${SAMPLE_TEST_EXIT_CODE}"; }\n',
        encoding="utf-8",
    )


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("exit_code", ["0", "7"])
def test_each_stage_runs_only_its_command_and_propagates_failures(
    bash, sampled_env, tmp_path, fake_environment, stage, exit_code
):
    sampled_env.update(SLURM_ARRAY_TASK_ID="1", SAMPLE_TEST_EXIT_CODE=exit_code)
    result = run_stage(bash, sampled_env, stage)
    assert result.returncode == int(exit_code), result.stderr
    arguments = (tmp_path / "arguments.txt").read_text().splitlines()
    assert arguments[:4] == ["-u", "-m", "tabicl.nano_graph_u.sampled", stage]
    assert (f"SAMPLED_{stage.upper()}_COMPLETE" in result.stdout) == (exit_code == "0")


def test_resource_requests_match_stage_work():
    for stage, count in STAGES.items():
        source = script(stage).read_text()
        assert source.startswith("#!/usr/bin/env bash\n")
        assert "set -euo pipefail" in source
        if stage in ("tune", "test"):
            assert "#SBATCH --partition=gpu" in source
            assert "#SBATCH --gres=gpu:1" in source
        else:
            assert "#SBATCH --partition=wdr" in source
            assert "--gres" not in source
        if stage != "summarize":
            assert f"#SBATCH --array=1-{count}%2" in source
