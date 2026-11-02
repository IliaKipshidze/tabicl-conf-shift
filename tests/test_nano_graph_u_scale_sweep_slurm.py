"""The single sweep job invokes only the frozen-model evaluation controller."""

import os
import shlex
import subprocess
from pathlib import Path

import pytest

from test_nano_graph_u_slurm import bash, shell_path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "slurm_nano_graph_u_scale_sweep.sh"


@pytest.fixture
def sweep_env(tmp_path):
    env = dict(os.environ)
    for key in ("WORKERS", "SWEEP_SCALE_MEANS", "SWEEP_SCALE_HALF_WIDTH"):
        env.pop(key, None)
    env.update(
        NANO_CLUSTER_ROOT=shell_path(tmp_path),
        NANO_REPO_DIR=shell_path(REPO),
        NANO_CONDA_ROOT=shell_path(tmp_path / "conda"),
        NANO_ENV_PATH=shell_path(tmp_path / "env"),
        NANO_SOURCE_ROOT=shell_path(tmp_path / "source experiment"),
        NANO_SWEEP_ROOT=shell_path(tmp_path / "scale sweep"),
        # A leftover setting from earlier runs must not control this job.
        NANO_EXPERIMENT_ROOT=shell_path(tmp_path / "old experiment"),
        SLURM_CPUS_PER_TASK="8",
        SWEEP_TEST_LOG=shell_path(tmp_path / "arguments.txt"),
        SWEEP_TEST_EXIT_CODE="0",
    )
    return env


def run_script(bash, env, *args):
    return subprocess.run(
        [bash, str(SCRIPT), *args], cwd=REPO, env=env,
        capture_output=True, text=True, timeout=30, check=False,
    )


def command(output):
    lines = [line.removeprefix("Command:") for line in output.splitlines() if line.startswith("Command:")]
    assert len(lines) == 1
    return shlex.split(lines[0])


def test_sweep_dry_run_is_read_only_and_uses_separate_paths(bash, sweep_env, tmp_path):
    result = run_script(bash, sweep_env, "--dry-run")
    assert result.returncode == 0, result.stderr
    args = command(result.stdout)
    assert args[:4] == ["python", "-u", "-m", "tabicl.nano_graph_u.scale_sweep"]
    assert args[args.index("--source-root") + 1] == sweep_env["NANO_SOURCE_ROOT"]
    assert args[args.index("--root") + 1] == sweep_env["NANO_SWEEP_ROOT"]
    assert args[args.index("--scale-means") + 1:args.index("--scale-half-width")] == ["1.3", "1.7", "1.9", "2.2"]
    assert args[args.index("--scale-half-width") + 1] == "0.2"
    assert args[args.index("--workers") + 1] == "8"
    assert "New test banks=40; new evaluations=160; reused IID evaluations=40" in result.stdout
    assert sweep_env["NANO_EXPERIMENT_ROOT"] not in result.stdout
    assert not (tmp_path / "scale sweep").exists()
    assert not (tmp_path / "source experiment").exists()
    assert not (tmp_path / "arguments.txt").exists()


def test_sweep_options_and_cpu_allocation_are_configurable(bash, sweep_env):
    sweep_env.update(SWEEP_SCALE_MEANS="1.4 2.0", SWEEP_SCALE_HALF_WIDTH="0.1", SLURM_CPUS_PER_TASK="4")
    result = run_script(bash, sweep_env, "--dry-run")
    assert result.returncode == 0, result.stderr
    args = command(result.stdout)
    assert args[args.index("--scale-means") + 1:args.index("--scale-half-width")] == ["1.4", "2.0"]
    assert args[args.index("--scale-half-width") + 1] == "0.1"
    assert args[args.index("--workers") + 1] == "4"
    assert "New test banks=20; new evaluations=80; reused IID evaluations=40" in result.stdout
    sweep_env["WORKERS"] = "5"
    failed = run_script(bash, sweep_env, "--dry-run")
    assert failed.returncode != 0
    assert "must not exceed" in failed.stderr


@pytest.mark.parametrize("key,value", [
    ("WORKERS", "0"), ("WORKERS", "abc"), ("SLURM_CPUS_PER_TASK", "0"),
    ("NANO_SOURCE_ROOT", "relative/source"), ("NANO_SWEEP_ROOT", "relative/output"),
])
def test_invalid_configuration_fails_before_activation(bash, sweep_env, tmp_path, key, value):
    sweep_env[key] = value
    result = run_script(bash, sweep_env)
    assert result.returncode != 0
    assert not (tmp_path / "arguments.txt").exists()


def test_source_and_output_cannot_be_the_same(bash, sweep_env):
    sweep_env["NANO_SWEEP_ROOT"] = sweep_env["NANO_SOURCE_ROOT"] + "/"
    result = run_script(bash, sweep_env, "--dry-run")
    assert result.returncode != 0
    assert "must differ" in result.stderr


@pytest.fixture
def fake_environment(tmp_path):
    initialization = tmp_path / "conda" / "etc" / "profile.d" / "conda.sh"
    initialization.parent.mkdir(parents=True)
    initialization.write_text(
        'conda() { return 0; }\n'
        'python() { if [[ "${1:-}" == "-" ]]; then return 0; fi; '
        'printf "%s\\n" "$@" > "${SWEEP_TEST_LOG}"; return "${SWEEP_TEST_EXIT_CODE}"; }\n',
        encoding="utf-8",
    )


@pytest.mark.parametrize("exit_code", ["0", "7"])
def test_sweep_invokes_one_controller_and_propagates_failure(
    bash, sweep_env, tmp_path, fake_environment, exit_code
):
    sweep_env["SWEEP_TEST_EXIT_CODE"] = exit_code
    result = run_script(bash, sweep_env)
    assert result.returncode == int(exit_code), result.stderr
    args = (tmp_path / "arguments.txt").read_text().splitlines()
    assert args[:3] == ["-u", "-m", "tabicl.nano_graph_u.scale_sweep"]
    assert args[args.index("--device") + 1] == "cuda"
    assert "train" not in args and "trial" not in args and "select" not in args
    assert ("SCALE_SWEEP_COMPLETE" in result.stdout) == (exit_code == "0")


def test_sweep_is_one_gpu_job_without_array_or_additional_submissions():
    source = SCRIPT.read_text()
    assert "#SBATCH --partition=gpu" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "#SBATCH --cpus-per-task=8" in source
    assert "#SBATCH --time=24:00:00" in source
    assert "#SBATCH --array" not in source
    assert "set -euo pipefail" in source
    assert not any(line.startswith("sbatch ") for line in source.splitlines())
