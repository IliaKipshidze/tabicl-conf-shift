"""The single location sweep job evaluates frozen models in a separate root."""

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
SCRIPT = REPO / "scripts" / "slurm_nano_graph_u_location_sweep.sh"
DEFAULT_PROFILES = [
    "0.3:0.1:0.5", "1.3:1.1:1.5", "1.7:1.5:1.9", "1.9:1.7:2.1",
    "2.2:2.0:2.4", "4.5:4.3:4.7", "10:9.8:10.2", "100:85:115",
]


@pytest.fixture
def sweep_env(tmp_path):
    env = dict(os.environ)
    for key in ("WORKERS", "SWEEP_LOCATION_PROFILES"):
        env.pop(key, None)
    env.update(
        NANO_CLUSTER_ROOT=shell_path(tmp_path),
        NANO_REPO_DIR=shell_path(REPO),
        NANO_CONDA_ROOT=shell_path(tmp_path / "conda"),
        NANO_ENV_PATH=shell_path(tmp_path / "env"),
        NANO_SOURCE_ROOT=shell_path(tmp_path / "source experiment"),
        NANO_LOCATION_ROOT=shell_path(tmp_path / "location sweep"),
        # Settings from prior experiments must not redirect the location sweep.
        NANO_SWEEP_ROOT=shell_path(tmp_path / "old scale sweep"),
        NANO_EXPERIMENT_ROOT=shell_path(tmp_path / "old experiment"),
        SLURM_CPUS_PER_TASK="8",
        SWEEP_TEST_LOG=shell_path(tmp_path / "arguments.txt"),
        SWEEP_TEST_EXIT_CODE="0",
        SWEEP_TEST_GPU_EXIT_CODE="0",
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


def test_location_sweep_dry_run_is_read_only_and_uses_separate_paths(bash, sweep_env, tmp_path):
    result = run_script(bash, sweep_env, "--dry-run")
    assert result.returncode == 0, result.stderr
    args = command(result.stdout)
    assert args[:4] == ["python", "-u", "-m", "tabicl.nano_graph_u.location_sweep"]
    assert args[args.index("--source-root") + 1] == sweep_env["NANO_SOURCE_ROOT"]
    assert args[args.index("--root") + 1] == sweep_env["NANO_LOCATION_ROOT"]
    assert args[args.index("--location-profiles") + 1:args.index("--workers")] == DEFAULT_PROFILES
    assert args[args.index("--workers") + 1] == "8"
    assert args[args.index("--device") + 1] == "cuda"
    assert "Location profiles=" + " ".join(DEFAULT_PROFILES) in result.stdout
    assert "New test banks=80; new evaluations=320; reused IID evaluations=40; total records=360" in result.stdout
    assert "inherited from source" in result.stdout
    assert "scale mean=1.5 SD=0.1 bounds=[1.3,1.7]; location SD=0.15" in result.stdout
    assert sweep_env["NANO_SWEEP_ROOT"] not in result.stdout
    assert sweep_env["NANO_EXPERIMENT_ROOT"] not in result.stdout
    assert not list(tmp_path.iterdir())


def test_default_roots_ignore_stale_scale_and_experiment_settings(bash, sweep_env, tmp_path):
    sweep_env.pop("NANO_SOURCE_ROOT")
    sweep_env.pop("NANO_LOCATION_ROOT")
    result = run_script(bash, sweep_env, "--dry-run")
    assert result.returncode == 0, result.stderr
    args = command(result.stdout)
    assert args[args.index("--source-root") + 1] == sweep_env["NANO_CLUSTER_ROOT"] + "/tabicl-conf-shift-sampled-experiment"
    assert args[args.index("--root") + 1] == sweep_env["NANO_CLUSTER_ROOT"] + "/tabicl-conf-shift-location-sweep"
    assert sweep_env["NANO_SWEEP_ROOT"] not in result.stdout
    assert sweep_env["NANO_EXPERIMENT_ROOT"] not in result.stdout
    assert not list(tmp_path.iterdir())


def test_location_profiles_and_workers_are_configurable(bash, sweep_env):
    sweep_env.update(SWEEP_LOCATION_PROFILES="1.4:1.2:1.6\n2.0:1.8:2.2", SLURM_CPUS_PER_TASK="4")
    result = run_script(bash, sweep_env, "--dry-run")
    assert result.returncode == 0, result.stderr
    args = command(result.stdout)
    assert args[args.index("--location-profiles") + 1:args.index("--workers")] == ["1.4:1.2:1.6", "2.0:1.8:2.2"]
    assert args[args.index("--workers") + 1] == "4"
    assert "New test banks=20; new evaluations=80; reused IID evaluations=40; total records=120" in result.stdout
    sweep_env["WORKERS"] = "2"
    configured = run_script(bash, sweep_env, "--dry-run")
    assert configured.returncode == 0, configured.stderr
    assert command(configured.stdout)[-3:] == ["2", "--device", "cuda"]
    sweep_env["WORKERS"] = "5"
    failed = run_script(bash, sweep_env, "--dry-run")
    assert failed.returncode != 0
    assert "must not exceed" in failed.stderr


@pytest.mark.parametrize("key,value", [
    ("WORKERS", "0"), ("WORKERS", "abc"), ("SLURM_CPUS_PER_TASK", "0"),
    ("NANO_SOURCE_ROOT", "relative/source"), ("NANO_LOCATION_ROOT", "relative/output"),
    ("SWEEP_LOCATION_PROFILES", " \t "),
])
def test_invalid_configuration_fails_before_activation(bash, sweep_env, tmp_path, key, value):
    sweep_env[key] = value
    result = run_script(bash, sweep_env)
    assert result.returncode != 0
    assert not list(tmp_path.iterdir())


def test_source_and_output_cannot_be_the_same(bash, sweep_env):
    sweep_env["NANO_LOCATION_ROOT"] = sweep_env["NANO_SOURCE_ROOT"] + "/"
    result = run_script(bash, sweep_env, "--dry-run")
    assert result.returncode != 0
    assert "must differ" in result.stderr


@pytest.mark.parametrize("args", [("--unknown",), ("--dry-run", "extra")])
def test_unknown_arguments_fail_before_activation(bash, sweep_env, tmp_path, args):
    result = run_script(bash, sweep_env, *args)
    assert result.returncode != 0
    assert "Usage:" in result.stderr
    assert not list(tmp_path.iterdir())


@pytest.fixture
def fake_environment(tmp_path):
    initialization = tmp_path / "conda" / "etc" / "profile.d" / "conda.sh"
    initialization.parent.mkdir(parents=True)
    initialization.write_text(
        'conda() { return 0; }\n'
        'python() { if [[ "${1:-}" == "-" ]]; then return "${SWEEP_TEST_GPU_EXIT_CODE}"; fi; '
        'printf "%s\\n" "$@" >> "${SWEEP_TEST_LOG}"; return "${SWEEP_TEST_EXIT_CODE}"; }\n',
        encoding="utf-8",
    )


@pytest.mark.parametrize("exit_code", ["0", "7"])
def test_location_sweep_invokes_one_controller_and_propagates_failure(
    bash, sweep_env, tmp_path, fake_environment, exit_code
):
    sweep_env["SWEEP_TEST_EXIT_CODE"] = exit_code
    result = run_script(bash, sweep_env)
    assert result.returncode == int(exit_code), result.stderr
    args = (tmp_path / "arguments.txt").read_text().splitlines()
    assert args[:3] == ["-u", "-m", "tabicl.nano_graph_u.location_sweep"]
    assert args.count("-m") == 1
    assert args[args.index("--location-profiles") + 1:args.index("--workers")] == DEFAULT_PROFILES
    assert args[args.index("--device") + 1] == "cuda"
    assert "train" not in args and "trial" not in args and "select" not in args
    assert ("LOCATION_SWEEP_COMPLETE" in result.stdout) == (exit_code == "0")


def test_gpu_check_failure_stops_before_controller(bash, sweep_env, tmp_path, fake_environment):
    sweep_env["SWEEP_TEST_GPU_EXIT_CODE"] = "9"
    result = run_script(bash, sweep_env)
    assert result.returncode == 9
    assert not (tmp_path / "arguments.txt").exists()
    assert "LOCATION_SWEEP_COMPLETE" not in result.stdout


def test_location_sweep_is_one_gpu_job_without_array_or_additional_submissions():
    source = SCRIPT.read_text()
    assert "#SBATCH --partition=gpu" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "#SBATCH --cpus-per-task=8" in source
    assert "#SBATCH --time=24:00:00" in source
    assert "slurm-nano-gu-location-sweep-%j.out" in source
    assert "slurm-nano-gu-location-sweep-%j.err" in source
    assert "#SBATCH --array" not in source
    assert "set -euo pipefail" in source
    assert not any(line.startswith("sbatch ") for line in source.splitlines())
