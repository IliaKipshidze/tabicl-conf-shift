"""Preview and execute the CPU/GPU test-only plans without a cluster."""

import os
import shlex
import subprocess
from pathlib import Path

import pytest
from test_nano_graph_u_slurm import bash as bash  # noqa: PLC0414
from test_nano_graph_u_slurm import shell_path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = {
    "generate": REPO / "scripts/slurm_nano_graph_u_test_shift_generate.sh",
    "evaluate": REPO / "scripts/slurm_nano_graph_u_test_shift_evaluate.sh",
}
PAIRS = [(4, 1.5, 6, 2.25), (20, 2, 30, 3), (60, 3, 90, 4.5), (100, 4, 150, 6)]


@pytest.fixture
def job_env(tmp_path):
    env = dict(os.environ)
    env.pop("SLURM_ARRAY_TASK_ID", None)
    env.update(
        NANO_CLUSTER_ROOT=shell_path(tmp_path), NANO_FIXED_ROOT=shell_path(tmp_path / "fixed source"),
        NANO_EPOCH_SOURCE_ROOT=shell_path(tmp_path / "epoch source"),
        NANO_TEST_SHIFT_ROOT=shell_path(tmp_path / "new test sweep"), NANO_REPO_DIR=shell_path(REPO),
        NANO_CONDA_ROOT=shell_path(tmp_path / "conda"), NANO_ENV_PATH=shell_path(tmp_path / "env"),
        SLURM_CPUS_PER_TASK="8", WORKERS="8", SHIFT_TEST_LOG=shell_path(tmp_path / "args.txt"),
        SHIFT_TEST_EXIT="0", SHIFT_TEST_GPU_EXIT="0",
    )
    return env


def run(bash, env, stage, *options):
    return subprocess.run([bash, str(SCRIPTS[stage]), *options], env=env, cwd=REPO,
                          capture_output=True, text=True, timeout=30, check=False)


@pytest.mark.parametrize("stage", SCRIPTS)
def test_syntax_resources_and_no_nested_submissions(bash, stage):
    script = SCRIPTS[stage]
    assert subprocess.run([bash, "-n", str(script)], check=False).returncode == 0
    source = script.read_text(encoding="utf-8")
    assert b"\r" not in script.read_bytes()
    assert not any(line.startswith(("sbatch ", "srun ")) for line in source.splitlines())
    expected = ("--partition=wdr", "--cpus-per-task=8", "--array=1-8%2", "--time=08:00:00") if stage == "generate" else (
        "--partition=gpu", "--gres=gpu:1", "--cpus-per-task=2", "--array=1-4%2", "--time=24:00:00")
    assert all(f"#SBATCH {value}" in source for value in expected)
    if stage == "generate":
        assert "#SBATCH --gres" not in source


@pytest.mark.parametrize("stage,task_count", [("generate", 8), ("evaluate", 4)])
def test_dry_run_covers_only_agreed_test_settings_without_files(bash, job_env, tmp_path, stage, task_count):
    result = run(bash, job_env, stage, "--dry-run")
    assert result.returncode == 0, result.stderr
    plans = [line for line in result.stdout.splitlines() if line.startswith("task=")]
    commands = [shlex.split(line.removeprefix("Command:")) for line in result.stdout.splitlines()
                if line.startswith("Command:")]
    assert len(plans) == len(commands) == task_count
    for task, (plan, command) in enumerate(zip(plans, commands), 1):
        location, scale, test_location, test_scale = PAIRS[(task - 1) // 2 if stage == "generate" else task - 1]
        assert f"train_location={location} train_scale={scale:g} test_location={test_location} test_scale={test_scale:g} repetition=1" in plan
        assert command[:5] == ["python", "-u", "-m", "tabicl.nano_graph_u.epoch_test_shift", stage]
        options = dict(zip(command[5::2], command[6::2]))
        assert options == {
            "--cluster-root": job_env["NANO_CLUSTER_ROOT"], "--fixed-root": job_env["NANO_FIXED_ROOT"],
            "--source-root": job_env["NANO_EPOCH_SOURCE_ROOT"], "--root": job_env["NANO_TEST_SHIFT_ROOT"],
            "--task-id": str(task), **({"--workers": "8"} if stage == "generate" else {"--device": "cuda"}),
        }
    if stage == "generate":
        assert result.stdout.count("mode=confounding") == result.stdout.count("mode=covariate") == 4
        assert result.stdout.count("seed=6001 tasks=3200 batches=100") == 8
    else:
        assert result.stdout.count("24 new shifted evaluations + 24 reused IID test rows") == 4
        assert "192 rows plus reused training CSVs" in result.stdout
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("stage,last", [("generate", "8"), ("evaluate", "4")])
def test_selected_task_and_default_new_output_root(bash, job_env, tmp_path, stage, last):
    job_env["SLURM_ARRAY_TASK_ID"] = last
    job_env.pop("NANO_TEST_SHIFT_ROOT")
    result = run(bash, job_env, stage, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("Command:") == 1
    assert "train_location=100 train_scale=4 test_location=150 test_scale=6" in result.stdout
    assert f"{job_env['NANO_CLUSTER_ROOT']}/tabicl-conf-shift-epoch-test-shift-1p5" in result.stdout
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("stage,task", [("generate", "0"), ("generate", "9"), ("evaluate", "5"), ("evaluate", "01")])
def test_invalid_array_task_rejected_before_activation(bash, job_env, tmp_path, stage, task):
    job_env["SLURM_ARRAY_TASK_ID"] = task
    result = run(bash, job_env, stage, "--dry-run")
    assert result.returncode != 0 and "SLURM_ARRAY_TASK_ID" in result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("stage", SCRIPTS)
@pytest.mark.parametrize("target", ["fixed source", "epoch source/child", "new/../epoch source", "."])
def test_output_cannot_overlap_read_only_sources(bash, job_env, tmp_path, stage, target):
    job_env["NANO_TEST_SHIFT_ROOT"] = job_env["NANO_CLUSTER_ROOT"] + "/" + target
    result = run(bash, job_env, stage, "--dry-run")
    assert result.returncode != 0 and "must not overlap" in result.stderr
    assert not list(tmp_path.iterdir())


def test_generation_rejects_workers_beyond_allocation(bash, job_env, tmp_path):
    job_env["WORKERS"] = "9"
    result = run(bash, job_env, "generate", "--dry-run")
    assert result.returncode != 0 and "must not exceed" in result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("stage", SCRIPTS)
@pytest.mark.parametrize("exit_code,gpu_exit", [("0", "0"), ("7", "0"), ("0", "9")])
def test_one_controller_call_and_failure_propagation(bash, job_env, tmp_path, stage, exit_code, gpu_exit):
    initialization = tmp_path / "conda/etc/profile.d/conda.sh"
    initialization.parent.mkdir(parents=True)
    initialization.write_text(
        'conda() { return 0; }\n'
        'python() { if [[ "${1:-}" == "-" ]]; then return "$SHIFT_TEST_GPU_EXIT"; fi; '
        'printf "%s\\n" "$@" > "$SHIFT_TEST_LOG"; return "$SHIFT_TEST_EXIT"; }\n', encoding="utf-8")
    job_env.update(SLURM_ARRAY_TASK_ID="1", SHIFT_TEST_EXIT=exit_code, SHIFT_TEST_GPU_EXIT=gpu_exit)
    result = run(bash, job_env, stage)
    expected_code = int(gpu_exit) if stage == "evaluate" and gpu_exit != "0" else int(exit_code)
    assert result.returncode == expected_code, result.stderr
    log = tmp_path / "args.txt"
    if stage == "evaluate" and gpu_exit != "0":
        assert not log.exists()
    else:
        args = log.read_text(encoding="utf-8").splitlines()
        assert args[:4] == ["-u", "-m", "tabicl.nano_graph_u.epoch_test_shift", stage]
        assert args[args.index("--task-id") + 1] == "1"
    assert ("JOB_COMPLETE" in result.stdout) == (expected_code == 0)
