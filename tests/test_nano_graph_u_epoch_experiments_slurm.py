"""Check the multi-epoch Slurm plan and failure handling without a cluster."""

import os
import shlex
import subprocess
from pathlib import Path

import pytest
from test_nano_graph_u_slurm import bash as bash  # noqa: PLC0414
from test_nano_graph_u_slurm import shell_path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/slurm_nano_graph_u_epoch_experiments.sh"
PAIRS = [(4, 1.5), (20, 2), (60, 3), (100, 4)]


@pytest.fixture
def job_env(tmp_path):
    env = dict(os.environ)
    env.pop("SLURM_ARRAY_TASK_ID", None)
    env.update(
        NANO_CLUSTER_ROOT=shell_path(tmp_path),
        NANO_FIXED_ROOT=shell_path(tmp_path / "fixed source"),
        NANO_EPOCH_ROOT=shell_path(tmp_path / "new epochs"),
        NANO_REPO_DIR=shell_path(REPO),
        NANO_CONDA_ROOT=shell_path(tmp_path / "conda"),
        NANO_ENV_PATH=shell_path(tmp_path / "env"),
        SLURM_CPUS_PER_TASK="2",
        EPOCH_TEST_LOG=shell_path(tmp_path / "args.txt"),
        EPOCH_TEST_EXIT="0",
        EPOCH_TEST_GPU_EXIT="0",
    )
    return env


def run(bash, env, *args):
    return subprocess.run([bash, str(SCRIPT), *args], env=env, cwd=REPO,
                          capture_output=True, text=True, timeout=30, check=False)


def test_syntax_and_resource_request(bash):
    assert subprocess.run([bash, "-n", str(SCRIPT)], check=False).returncode == 0
    source = SCRIPT.read_text(encoding="utf-8")
    for directive in ("--job-name=nano-gu-epochs", "--partition=gpu", "--gres=gpu:1",
                      "--cpus-per-task=2", "--array=1-30%2", "--time=48:00:00"):
        assert f"#SBATCH {directive}" in source
    assert b"\r" not in SCRIPT.read_bytes()
    assert not any(line.startswith(("sbatch ", "srun ")) for line in source.splitlines())


def test_all_30_task_plans_use_rep1_shared_validation_and_separate_outputs(bash, job_env, tmp_path):
    result = run(bash, job_env, "--dry-run")
    assert result.returncode == 0, result.stderr
    plans = [line for line in result.stdout.splitlines() if line.startswith("task=")]
    commands = [shlex.split(line.removeprefix("Command:"))
                for line in result.stdout.splitlines() if line.startswith("Command:")]
    assert len(plans) == len(commands) == 30
    for task, (plan, command) in enumerate(zip(plans, commands), 1):
        budget, slot = divmod(task - 1, 10)
        epochs = (3, 6, 10)[budget]
        mode = "confounding" if slot % 2 == 0 else "covariate"
        condition = "identity" if slot < 2 else "shift"
        assert f"task={task} epochs={epochs} updates={2500 * epochs} mode={mode} condition={condition} repetition=1" in plan
        if slot >= 2:
            location, scale = PAIRS[(slot - 2) // 2]
            assert f"location={location} scale={scale:g}" in plan
        assert command[:4] == ["python", "-u", "-m", "tabicl.nano_graph_u.epoch_experiments"]
        assert dict(zip(command[4::2], command[5::2])) == {
            "--cluster-root": job_env["NANO_CLUSTER_ROOT"], "--fixed-root": job_env["NANO_FIXED_ROOT"],
            "--root": job_env["NANO_EPOCH_ROOT"], "--task-id": str(task), "--device": "cuda"}
    assert result.stdout.count("Learning rates=0.0007,0.001,0.002") == 30
    assert result.stdout.count("Batch order=seeded permutation of whole batches per epoch") == 30
    assert result.stdout.count("/rep_1/") == 30
    assert result.stdout.count("/validation/") == 30
    assert "validation_seed3001_100steps_b32_r150_f5.h5" in result.stdout
    assert "192 rows" in result.stdout and "156 unique test evaluations" in result.stdout
    assert not list(tmp_path.iterdir())


def test_default_output_keeps_shuffled_runs_separate(bash, job_env, tmp_path):
    job_env.pop("NANO_EPOCH_ROOT")
    result = run(bash, job_env, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert f"New outputs={job_env['NANO_CLUSTER_ROOT']}/tabicl-conf-shift-epoch-experiments-shuffled;" in result.stdout
    assert not list(tmp_path.iterdir())


def test_array_preview_selects_only_one_task(bash, job_env, tmp_path):
    job_env["SLURM_ARRAY_TASK_ID"] = "30"
    result = run(bash, job_env, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("Command:") == 1
    assert "task=30 epochs=10 updates=25000 mode=covariate condition=shift" in result.stdout
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("task", ["", "0", "31", "01", "1.0", "../1"])
def test_invalid_array_tasks_fail_without_files(bash, job_env, tmp_path, task):
    job_env["SLURM_ARRAY_TASK_ID"] = task
    result = run(bash, job_env, "--dry-run")
    assert result.returncode != 0 and "1 to 30" in result.stderr
    assert not list(tmp_path.iterdir())


def test_normal_execution_requires_array_id(bash, job_env, tmp_path):
    result = run(bash, job_env)
    assert result.returncode != 0 and "1 to 30" in result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("source", ["fixed source", "fixed source/child", "new/../fixed source",
                                     "tabicl-conf-shift-priors/child", "."])
def test_output_cannot_overlap_lexically_resolved_read_only_sources(bash, job_env, tmp_path, source):
    job_env["NANO_EPOCH_ROOT"] = job_env["NANO_CLUSTER_ROOT"] + "/" + source
    result = run(bash, job_env, "--dry-run")
    assert result.returncode != 0 and "must not overlap" in result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("setting,value", [("NANO_EPOCH_ROOT", "relative"), ("SLURM_CPUS_PER_TASK", "0")])
def test_invalid_settings_fail_before_activation(bash, job_env, tmp_path, setting, value):
    job_env[setting] = value
    assert run(bash, job_env, "--dry-run").returncode != 0
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("exit_code,gpu_exit", [("0", "0"), ("7", "0"), ("0", "9")])
def test_execution_runs_one_controller_and_propagates_failures(bash, job_env, tmp_path, exit_code, gpu_exit):
    initialization = tmp_path / "conda/etc/profile.d/conda.sh"
    initialization.parent.mkdir(parents=True)
    initialization.write_text(
        'conda() { return 0; }\n'
        'python() { if [[ "${1:-}" == "-" ]]; then return "$EPOCH_TEST_GPU_EXIT"; fi; '
        'printf "%s\\n" "$@" > "$EPOCH_TEST_LOG"; return "$EPOCH_TEST_EXIT"; }\n', encoding="utf-8")
    job_env.update(SLURM_ARRAY_TASK_ID="13", EPOCH_TEST_EXIT=exit_code, EPOCH_TEST_GPU_EXIT=gpu_exit)
    result = run(bash, job_env)
    assert result.returncode == int(gpu_exit if gpu_exit != "0" else exit_code)
    if gpu_exit != "0":
        assert not (tmp_path / "args.txt").exists()
    else:
        args = (tmp_path / "args.txt").read_text().splitlines()
        assert args[:3] == ["-u", "-m", "tabicl.nano_graph_u.epoch_experiments"]
        assert dict(zip(args[3::2], args[4::2])) == {
            "--cluster-root": job_env["NANO_CLUSTER_ROOT"], "--fixed-root": job_env["NANO_FIXED_ROOT"],
            "--root": job_env["NANO_EPOCH_ROOT"], "--task-id": "13", "--device": "cuda"}
    assert ("EPOCH_EXPERIMENT_JOB_COMPLETE" in result.stdout) == (exit_code == gpu_exit == "0")
