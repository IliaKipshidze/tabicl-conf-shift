"""Large-task Slurm plans retain the requested protocol and propagate failures."""

import os
import shlex
import subprocess
from pathlib import Path

import pytest
from test_nano_graph_u_slurm import bash as bash  # noqa: PLC0414
from test_nano_graph_u_slurm import shell_path

REPO = Path(__file__).resolve().parents[1]
GPU = REPO / "scripts/slurm_nano_graph_u_large.sh"
CPU = REPO / "scripts/slurm_nano_graph_u_large_generate.sh"


@pytest.fixture
def job_env(tmp_path):
    env = dict(os.environ)
    for key in ("SLURM_ARRAY_TASK_ID", "WORKERS"):
        env.pop(key, None)
    env.update(NANO_CLUSTER_ROOT=shell_path(tmp_path), NANO_LARGE_ROOT=shell_path(tmp_path / "large output"),
               NANO_REPO_DIR=shell_path(REPO), NANO_CONDA_ROOT=shell_path(tmp_path / "conda"),
               NANO_ENV_PATH=shell_path(tmp_path / "env"), SLURM_CPUS_PER_TASK="8",
               LARGE_TEST_LOG=shell_path(tmp_path / "args.txt"), LARGE_TEST_EXIT="0", LARGE_TEST_GPU_EXIT="0")
    return env


def run(bash, env, script, *options):
    return subprocess.run([bash, str(script), *options], env=env, cwd=REPO,
                          capture_output=True, text=True, timeout=30, check=False)


def commands(output):
    return [shlex.split(line.removeprefix("Command:")) for line in output.splitlines() if line.startswith("Command:")]


@pytest.mark.parametrize("script", [CPU, GPU])
def test_resources_syntax_and_no_nested_submissions(bash, script):
    assert subprocess.run([bash, "-n", str(script)], check=False).returncode == 0
    assert b"\r" not in script.read_bytes()
    text = script.read_text(encoding="utf-8")
    directives = ("--partition=wdr", "--cpus-per-task=8", "--array=1-12%2") if script == CPU else (
        "--partition=gpu", "--gres=gpu:1", "--cpus-per-task=2", "--array=1")
    assert all(f"#SBATCH {value}" in text for value in (*directives, "--time=48:00:00"))
    assert not any(line.startswith(("sbatch ", "srun ")) for line in text.splitlines())


def test_generation_all_twelve_bank_counts_and_paths_are_preview_only(bash, job_env, tmp_path):
    result = run(bash, job_env, CPU, "--dry-run")
    assert result.returncode == 0, result.stderr
    plans = [line for line in result.stdout.splitlines() if line.startswith("task=")]
    assert len(plans) == len(commands(result.stdout)) == 12
    for index, (plan, command) in enumerate(zip(plans, commands(result.stdout))):
        setting, split_index = divmod(index, 3)
        mode = "confounding" if setting < 2 else "covariate"
        condition = "identity" if setting % 2 == 0 else "shift"
        split, tasks, steps, seed = [("train", 500000, 15625, 1001), ("validation", 32000, 1000, 3001),
                                     ("test", 32000, 1000, 2001)][split_index]
        assert f"task={index + 1} mode={mode} condition={condition} split={split} tasks={tasks} batches={steps} seed={seed}" in plan
        assert command[:5] == ["python", "-u", "-m", "tabicl.nano_graph_u.large_experiment", "generate"]
        assert command[command.index("--task-id") + 1] == str(index + 1)
    assert "90.24 GB" in result.stdout and "rows=1000 features=8" in result.stdout
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("stage,count", [("smoke", 1), ("trial", 36), ("evaluate", 12)])
def test_gpu_stage_maps_full_budget_and_selected_models(bash, job_env, tmp_path, stage, count):
    result = run(bash, job_env, GPU, stage, "--dry-run")
    assert result.returncode == 0, result.stderr
    plans = [line for line in result.stdout.splitlines() if line.startswith("stage=")]
    assert len(plans) == len(commands(result.stdout)) == count
    for index, (plan, command) in enumerate(zip(plans, commands(result.stdout))):
        assert command[:5] == ["python", "-u", "-m", "tabicl.nano_graph_u.large_experiment", stage]
        if stage == "smoke":
            assert "batch=32 rows=1000 features=8 support_rows=900" in plan
            assert "--task-id" not in command
            continue
        epoch_index, slot = divmod(index, 12 if stage == "trial" else 4)
        epoch = [1, 3, 9][epoch_index]
        setting = slot // 3 if stage == "trial" else slot
        mode, condition = ("confounding" if setting < 2 else "covariate"), ("identity" if setting % 2 == 0 else "shift")
        assert f"epochs={epoch} updates={15625 * epoch} mode={mode} condition={condition}" in plan
        assert command[command.index("--task-id") + 1] == str(index + 1)
    if stage == "trial":
        assert all(result.stdout.count(f"learning_rate={rate};") == 12 for rate in ("0.0007", "0.001", "0.002"))
    assert not list(tmp_path.iterdir())


def test_default_gpu_stage_and_output_root_are_smoke_only(bash, job_env, tmp_path):
    job_env.pop("NANO_LARGE_ROOT")
    result = run(bash, job_env, GPU, "--dry-run")
    assert result.returncode == 0 and len(commands(result.stdout)) == 1
    assert commands(result.stdout)[0][4] == "smoke"
    assert "tabicl-conf-shift-nano-large-r1000-f8" in result.stdout
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("script,options,task", [(CPU, [], "13"), (GPU, ["smoke"], "2"),
    (GPU, ["trial"], "37"), (GPU, ["evaluate"], "13"), (GPU, ["trial"], "01")])
def test_invalid_task_is_rejected_before_environment_activation(bash, job_env, tmp_path, script, options, task):
    job_env["SLURM_ARRAY_TASK_ID"] = task
    result = run(bash, job_env, script, *options, "--dry-run")
    assert result.returncode != 0 and "SLURM_ARRAY_TASK_ID" in result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("script", [CPU, GPU])
@pytest.mark.parametrize("target", [".", "tabicl-conf-shift-priors/new", "new/../tabicl-conf-shift-fixed-experiments"])
def test_protected_output_roots(bash, job_env, tmp_path, script, target):
    job_env["NANO_LARGE_ROOT"] = job_env["NANO_CLUSTER_ROOT"] + "/" + target
    result = run(bash, job_env, script, "--dry-run")
    assert result.returncode != 0 and "must not overlap" in result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("script,options", [(CPU, []), (GPU, ["smoke"]), (GPU, ["trial"]), (GPU, ["evaluate"])])
@pytest.mark.parametrize("exit_code", ["0", "7"])
def test_one_controller_call_and_exit_propagation(bash, job_env, tmp_path, script, options, exit_code):
    init = tmp_path / "conda/etc/profile.d/conda.sh"
    init.parent.mkdir(parents=True)
    init.write_text('conda() { return 0; }\n'
                    'python() { if [[ "${1:-}" == "-" ]]; then return "$LARGE_TEST_GPU_EXIT"; fi; '
                    'printf "%s\\n" "$@" > "$LARGE_TEST_LOG"; return "$LARGE_TEST_EXIT"; }\n', encoding="utf-8")
    job_env.update(SLURM_ARRAY_TASK_ID="1", LARGE_TEST_EXIT=exit_code)
    result = run(bash, job_env, script, *options)
    assert result.returncode == int(exit_code), result.stderr
    args = (tmp_path / "args.txt").read_text(encoding="utf-8").splitlines()
    stage = options[0] if options else "generate"
    assert args[:4] == ["-u", "-m", "tabicl.nano_graph_u.large_experiment", stage]
    assert ("JOB_COMPLETE" in result.stdout) == (exit_code == "0")


def test_gpu_preflight_failure_never_runs_controller(bash, job_env, tmp_path):
    init = tmp_path / "conda/etc/profile.d/conda.sh"
    init.parent.mkdir(parents=True)
    init.write_text('conda() { return 0; }\npython() { return 9; }\n', encoding="utf-8")
    job_env["SLURM_ARRAY_TASK_ID"] = "1"
    result = run(bash, job_env, GPU, "smoke")
    assert result.returncode == 9 and "JOB_COMPLETE" not in result.stdout
    assert not (tmp_path / "args.txt").exists()


def test_generation_rejects_cpu_oversubscription(bash, job_env, tmp_path):
    job_env["WORKERS"] = "9"
    result = run(bash, job_env, CPU, "--dry-run")
    assert result.returncode != 0 and "must not exceed" in result.stderr
    assert not list(tmp_path.iterdir())
