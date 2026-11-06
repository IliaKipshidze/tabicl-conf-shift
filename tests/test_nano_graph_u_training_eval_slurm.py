"""The GPU array evaluates each selected model on its own saved training bank."""

import os
import shlex
import subprocess
from pathlib import Path

import pytest
from test_nano_graph_u_slurm import (
    bash as bash,  # noqa: PLC0414 - re-export the pytest fixture
)
from test_nano_graph_u_slurm import shell_path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "slurm_nano_graph_u_training_eval.sh"
PAIRS = [(4, 1.5), (20, 2), (60, 3), (100, 4)]


@pytest.fixture
def training_env(tmp_path):
    env = dict(os.environ)
    env.pop("SLURM_ARRAY_TASK_ID", None)
    env.update(
        NANO_CLUSTER_ROOT=shell_path(tmp_path),
        NANO_REPO_DIR=shell_path(REPO),
        NANO_CONDA_ROOT=shell_path(tmp_path / "conda"),
        NANO_ENV_PATH=shell_path(tmp_path / "env"),
        NANO_FIXED_ROOT=shell_path(tmp_path / "fixed experiments"),
        # Old experiment settings cannot redirect the selected fixed sweep.
        NANO_EXPERIMENT_ROOT=shell_path(tmp_path / "old experiment"),
        NANO_DATA_ROOT=shell_path(tmp_path / "old banks"),
        NANO_SEARCH_ROOT=shell_path(tmp_path / "old models"),
        NANO_TEST_ROOT=shell_path(tmp_path / "old tests"),
        GRAPH_U_MODE="covariate",
        SLURM_CPUS_PER_TASK="2",
        TRAINING_TEST_LOG=shell_path(tmp_path / "arguments.txt"),
        TRAINING_TEST_ENV_LOG=shell_path(tmp_path / "environment.txt"),
        TRAINING_TEST_EXIT_CODE="0",
        TRAINING_TEST_GPU_EXIT_CODE="0",
    )
    return env


def run_script(bash, env, *args):
    return subprocess.run(
        [bash, str(SCRIPT), *args], cwd=REPO, env=env,
        capture_output=True, text=True, timeout=30, check=False,
    )


def commands(output):
    return [
        shlex.split(line.removeprefix("Command:"))
        for line in output.splitlines() if line.startswith("Command:")
    ]


def test_dry_run_maps_all_50_unique_evaluations_without_touching_files(bash, training_env, tmp_path):
    result = run_script(bash, training_env, "--dry-run")
    assert result.returncode == 0, result.stderr
    planned = commands(result.stdout)
    assert len(planned) == 50
    for task_id, args in enumerate(planned, start=1):
        assert args[:4] == ["python", "-u", "-m", "tabicl.nano_graph_u.training_eval"]
        assert dict(zip(args[4::2], args[5::2])) == {
            "--root": training_env["NANO_FIXED_ROOT"],
            "--task-id": str(task_id),
            "--device": "cuda",
        }
        model_index = (task_id - 1) if task_id <= 10 else ((task_id - 11) % 10)
        mode = "confounding" if model_index < 5 else "covariate"
        repetition = model_index % 5 + 1
        condition = "identity" if task_id <= 10 else "shift"
        label = f"task={task_id} condition={condition} mode={mode} repetition={repetition}"
        if task_id <= 10:
            label += " IID result reused for all four pairs"
        else:
            pair_id = (task_id - 11) // 10 + 1
            location, scale = PAIRS[pair_id - 1]
            label += f" pair={pair_id} location={location} scale={scale:g}"
        assert label + "; own saved training bank; tasks=80000" in result.stdout
    assert result.stdout.count("condition=identity") == 10
    assert result.stdout.count("condition=shift") == 40
    assert result.stdout.count("tasks=80000") == 50
    assert "existing checkpoints and training banks are read-only" in result.stdout
    assert "existing test CSVs are unchanged" in result.stdout
    assert f"{training_env['NANO_FIXED_ROOT']}/results/training_summary.csv (16 cells)" in result.stdout
    assert f"{training_env['NANO_FIXED_ROOT']}/results/training_per_repetition.csv (80 rows; IID results reused per pair)" in result.stdout
    for key in ("NANO_EXPERIMENT_ROOT", "NANO_DATA_ROOT", "NANO_SEARCH_ROOT", "NANO_TEST_ROOT"):
        assert training_env[key] not in result.stdout
    assert not list(tmp_path.iterdir())


def test_default_root_uses_existing_fixed_experiment(bash, training_env, tmp_path):
    training_env.pop("NANO_FIXED_ROOT")
    result = run_script(bash, training_env, "--dry-run")
    assert result.returncode == 0, result.stderr
    for args in commands(result.stdout):
        assert args[args.index("--root") + 1] == training_env["NANO_CLUSTER_ROOT"] + "/tabicl-conf-shift-fixed-experiments"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("task_id", ["1", "10", "11", "20", "21", "40", "41", "50"])
def test_array_dry_run_previews_only_its_evaluation(bash, training_env, tmp_path, task_id):
    training_env["SLURM_ARRAY_TASK_ID"] = task_id
    result = run_script(bash, training_env, "--dry-run")
    assert result.returncode == 0, result.stderr
    planned = commands(result.stdout)
    assert len(planned) == 1
    assert planned[0][planned[0].index("--task-id") + 1] == task_id
    assert result.stdout.count("tasks=80000") == 1
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("task_id", ["", "0", "51", "01", "050", "../1", "1.0", "-1"])
@pytest.mark.parametrize("dry_run", [False, True])
def test_invalid_task_fails_before_activation(bash, training_env, tmp_path, task_id, dry_run):
    training_env["SLURM_ARRAY_TASK_ID"] = task_id
    result = run_script(bash, training_env, *(["--dry-run"] if dry_run else []))
    assert result.returncode != 0
    assert "from 1 to 50" in result.stderr
    assert not list(tmp_path.iterdir())


def test_execution_requires_array_task(bash, training_env, tmp_path):
    result = run_script(bash, training_env)
    assert result.returncode != 0
    assert "from 1 to 50" in result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("key,value", [
    ("SLURM_CPUS_PER_TASK", "0"), ("SLURM_CPUS_PER_TASK", "2.5"),
    ("NANO_CLUSTER_ROOT", "relative/cluster"), ("NANO_FIXED_ROOT", "relative/output"),
    ("NANO_REPO_DIR", "relative/repo"), ("NANO_CONDA_ROOT", "relative/conda"),
    ("NANO_ENV_PATH", "relative/env"),
])
def test_invalid_configuration_fails_before_files(bash, training_env, tmp_path, key, value):
    training_env[key] = value
    result = run_script(bash, training_env, "--dry-run")
    assert result.returncode != 0
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("args", [("--unknown",), ("--dry-run", "extra")])
def test_unknown_arguments_fail_before_activation(bash, training_env, tmp_path, args):
    result = run_script(bash, training_env, *args)
    assert result.returncode != 0
    assert "Usage:" in result.stderr
    assert not list(tmp_path.iterdir())


@pytest.fixture
def fake_environment(tmp_path):
    initialization = tmp_path / "conda" / "etc" / "profile.d" / "conda.sh"
    initialization.parent.mkdir(parents=True)
    initialization.write_text(
        'conda() { return 0; }\n'
        'python() { if [[ "${1:-}" == "-" ]]; then return "${TRAINING_TEST_GPU_EXIT_CODE}"; fi; '
        'printf "%s\\n" "$@" >> "${TRAINING_TEST_LOG}"; '
        'printf "%s\\n" "$PYTHONUNBUFFERED" "$OMP_NUM_THREADS" "$OPENBLAS_NUM_THREADS" '
        '"$MKL_NUM_THREADS" "$NUMEXPR_NUM_THREADS" >> "${TRAINING_TEST_ENV_LOG}"; '
        'return "${TRAINING_TEST_EXIT_CODE}"; }\n',
        encoding="utf-8",
    )


@pytest.mark.parametrize("task_id", ["1", "10", "11", "21", "41", "50"])
@pytest.mark.parametrize("exit_code", ["0", "7"])
def test_task_invokes_one_controller_and_propagates_failure(
    bash, training_env, tmp_path, fake_environment, task_id, exit_code,
):
    training_env.update(SLURM_ARRAY_TASK_ID=task_id, TRAINING_TEST_EXIT_CODE=exit_code)
    result = run_script(bash, training_env)
    assert result.returncode == int(exit_code), result.stderr
    args = (tmp_path / "arguments.txt").read_text().splitlines()
    assert args[:3] == ["-u", "-m", "tabicl.nano_graph_u.training_eval"]
    assert dict(zip(args[3::2], args[4::2])) == {
        "--root": training_env["NANO_FIXED_ROOT"], "--task-id": task_id, "--device": "cuda",
    }
    assert args.count("-m") == 1
    assert (tmp_path / "environment.txt").read_text().splitlines() == ["1"] * 5
    assert ("TRAINING_EVAL_JOB_COMPLETE" in result.stdout) == (exit_code == "0")


def test_gpu_check_failure_stops_before_controller(bash, training_env, tmp_path, fake_environment):
    training_env.update(SLURM_ARRAY_TASK_ID="1", TRAINING_TEST_GPU_EXIT_CODE="9")
    result = run_script(bash, training_env)
    assert result.returncode == 9
    assert not (tmp_path / "arguments.txt").exists()
    assert "TRAINING_EVAL_JOB_COMPLETE" not in result.stdout


def test_launcher_is_one_gpu_array_with_no_training_or_additional_submissions():
    source = SCRIPT.read_text(encoding="utf-8")
    for directive in (
        "--job-name=nano-gu-train-eval", "--partition=gpu", "--gres=gpu:1",
        "--cpus-per-task=2", "--time=08:00:00", "--array=1-50%2",
        "--output=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-train-eval-%A_%a.out",
        "--error=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-train-eval-%A_%a.err",
    ):
        assert f"#SBATCH {directive}" in source
    assert "set -euo pipefail" in source
    assert "import torch" in source
    assert "torch.cuda.is_available()" in source
    assert "--cluster-root" not in source
    assert not any(line.startswith(("sbatch ", "srun ")) for line in source.splitlines())
    assert "tabicl.nano_graph_u.fixed_experiments" not in source
    assert b"\r" not in SCRIPT.read_bytes()
