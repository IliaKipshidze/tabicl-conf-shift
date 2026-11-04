"""Separate CPU generation and GPU arrays preserve the four fixed-pair plans."""

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
SCRIPT = REPO / "scripts" / "slurm_nano_graph_u_fixed_experiments.sh"
GENERATE_SCRIPT = REPO / "scripts" / "slurm_nano_graph_u_fixed_generate.sh"
PAIRS = [(4, 1.5), (20, 2), (60, 3), (100, 4)]
COUNTS = {
    "generate": "New banks=22; generation only; original IID banks reused read-only",
    "run": "Existing generated banks=22; training trials=100; new winners=10; new evaluations=60; reused IID results=20",
}


@pytest.fixture(params=["generate", "run"])
def fixed_env(tmp_path, request):
    env = dict(os.environ)
    for key in ("SLURM_ARRAY_TASK_ID", "WORKERS"):
        env.pop(key, None)
    env.update(
        NANO_CLUSTER_ROOT=shell_path(tmp_path),
        NANO_REPO_DIR=shell_path(REPO),
        NANO_CONDA_ROOT=shell_path(tmp_path / "conda"),
        NANO_ENV_PATH=shell_path(tmp_path / "env"),
        NANO_FIXED_ROOT=shell_path(tmp_path / "fixed experiments"),
        # Prior sampled-experiment and single-mode settings cannot redirect it.
        NANO_EXPERIMENT_ROOT=shell_path(tmp_path / "old sampled experiment"),
        NANO_DATA_ROOT=shell_path(tmp_path / "old banks"),
        NANO_SEARCH_ROOT=shell_path(tmp_path / "old search"),
        NANO_TEST_ROOT=shell_path(tmp_path / "old tests"),
        GRAPH_U_MODE="covariate",
        SLURM_CPUS_PER_TASK="8",
        FIXED_TEST_LOG=shell_path(tmp_path / "arguments.txt"),
        FIXED_TEST_ENV_LOG=shell_path(tmp_path / "environment.txt"),
        FIXED_TEST_EXIT_CODE="0",
        FIXED_TEST_GPU_EXIT_CODE="0",
        FIXED_TEST_STAGE=request.param,
    )
    return env


def run_script(bash, env, *args):
    script = GENERATE_SCRIPT if env["FIXED_TEST_STAGE"] == "generate" else SCRIPT
    return subprocess.run(
        [bash, str(script), *args], cwd=REPO, env=env,
        capture_output=True, text=True, timeout=30, check=False,
    )


def commands(output):
    return [
        shlex.split(line.removeprefix("Command:"))
        for line in output.splitlines() if line.startswith("Command:")
    ]


def test_dry_run_previews_all_pairs_and_original_sources_without_files(bash, fixed_env, tmp_path):
    result = run_script(bash, fixed_env, "--dry-run")
    assert result.returncode == 0, result.stderr
    planned = commands(result.stdout)
    assert len(planned) == 4
    stage = fixed_env["FIXED_TEST_STAGE"]
    for task_id, (args, (location, scale)) in enumerate(zip(planned, PAIRS), start=1):
        assert args[:5] == ["python", "-u", "-m", "tabicl.nano_graph_u.fixed_experiments", stage]
        expected = {
            "--cluster-root": fixed_env["NANO_CLUSTER_ROOT"],
            "--root": fixed_env["NANO_FIXED_ROOT"],
            "--task-id": str(task_id),
        }
        expected.update({"--workers": "8"} if stage == "generate" else {"--device": "cuda"})
        assert dict(zip(args[5::2], args[6::2])) == expected
        assert f"pair={task_id} location={location} scale={scale:g} modes=confounding,covariate repetitions=5 learning_rates=10" in result.stdout
        tag = f"loc_{location}_scale_{scale:g}".replace(".", "p")
        assert f"Pair output={fixed_env['NANO_FIXED_ROOT']}/{tag}" in result.stdout
    assert result.stdout.count(COUNTS[stage]) == 4
    assert "read-only; reused without retraining" in result.stdout
    for suffix in ("", "_covariate"):
        base = fixed_env["NANO_CLUSTER_ROOT"]
        assert f"{base}/tabicl-conf-shift-priors/nano_graph_u/repetitions{suffix}" in result.stdout
        assert f"{base}/tabicl-conf-shift-checkpoints/nano_graph_u/lr_search{suffix}/winners" in result.stdout
        for repetition in range(1, 6):
            assert f"/lr_search{suffix}/winners/rep_{repetition}/identity/model.pt" in result.stdout
    for suffix in ("", "_covariate", "_confounding_on_covariate", "_covariate_on_confounding"):
        assert f"{base}/tabicl-conf-shift-evaluations/nano_graph_u/lr_search_tests{suffix}" in result.stdout
    for key in ("NANO_EXPERIMENT_ROOT", "NANO_DATA_ROOT", "NANO_SEARCH_ROOT", "NANO_TEST_ROOT"):
        assert fixed_env[key] not in result.stdout
    assert not list(tmp_path.iterdir())


def test_default_root_is_separate_from_previous_experiments(bash, fixed_env, tmp_path):
    fixed_env.pop("NANO_FIXED_ROOT")
    result = run_script(bash, fixed_env, "--dry-run")
    assert result.returncode == 0, result.stderr
    for args in commands(result.stdout):
        assert args[args.index("--root") + 1] == fixed_env["NANO_CLUSTER_ROOT"] + "/tabicl-conf-shift-fixed-experiments"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("task_id", ["1", "2", "3", "4"])
def test_array_task_previews_only_its_pair(bash, fixed_env, task_id):
    fixed_env["SLURM_ARRAY_TASK_ID"] = task_id
    result = run_script(bash, fixed_env, "--dry-run")
    assert result.returncode == 0, result.stderr
    planned = commands(result.stdout)
    assert len(planned) == 1
    assert planned[0][planned[0].index("--task-id") + 1] == task_id
    location, scale = PAIRS[int(task_id) - 1]
    assert f"pair={task_id} location={location} scale={scale:g}" in result.stdout
    assert result.stdout.count(COUNTS[fixed_env["FIXED_TEST_STAGE"]]) == 1


@pytest.mark.parametrize("task_id", ["", "0", "5", "01", "../1", "1.0"])
@pytest.mark.parametrize("dry_run", [False, True])
def test_invalid_task_fails_before_activation(bash, fixed_env, tmp_path, task_id, dry_run):
    fixed_env["SLURM_ARRAY_TASK_ID"] = task_id
    result = run_script(bash, fixed_env, *(["--dry-run"] if dry_run else []))
    assert result.returncode != 0
    assert "from 1 to 4" in result.stderr
    assert not list(tmp_path.iterdir())


def test_execution_requires_an_array_task(bash, fixed_env, tmp_path):
    result = run_script(bash, fixed_env)
    assert result.returncode != 0
    assert "from 1 to 4" in result.stderr
    assert not list(tmp_path.iterdir())


def test_workers_follow_cpu_allocation_and_support_override(bash, fixed_env):
    if fixed_env["FIXED_TEST_STAGE"] != "generate":
        pytest.skip("Only CPU generation uses workers")
    fixed_env.update(SLURM_ARRAY_TASK_ID="1", SLURM_CPUS_PER_TASK="4")
    result = run_script(bash, fixed_env, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert commands(result.stdout)[0][-2:] == ["--workers", "4"]
    fixed_env["WORKERS"] = "2"
    result = run_script(bash, fixed_env, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert commands(result.stdout)[0][-2:] == ["--workers", "2"]
    fixed_env["WORKERS"] = "5"
    result = run_script(bash, fixed_env, "--dry-run")
    assert result.returncode != 0
    assert "must not exceed" in result.stderr


@pytest.mark.parametrize("key,value", [
    ("SLURM_CPUS_PER_TASK", "0"), ("SLURM_CPUS_PER_TASK", "2.5"),
    ("NANO_CLUSTER_ROOT", "relative/source"), ("NANO_FIXED_ROOT", "relative/output"),
    ("NANO_REPO_DIR", "relative/repo"), ("NANO_CONDA_ROOT", "relative/conda"),
    ("NANO_ENV_PATH", "relative/env"),
])
def test_invalid_configuration_fails_before_files(bash, fixed_env, tmp_path, key, value):
    fixed_env[key] = value
    result = run_script(bash, fixed_env, "--dry-run")
    assert result.returncode != 0
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("value", ["0", "abc", "-1"])
def test_invalid_generation_workers_fail_before_files(bash, fixed_env, tmp_path, value):
    if fixed_env["FIXED_TEST_STAGE"] != "generate":
        pytest.skip("Only CPU generation uses workers")
    fixed_env["WORKERS"] = value
    result = run_script(bash, fixed_env, "--dry-run")
    assert result.returncode != 0
    assert "WORKERS must be a positive integer" in result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("suffix", [
    "tabicl-conf-shift-priors/nano_graph_u/repetitions/",
    "tabicl-conf-shift-priors/nano_graph_u/repetitions_covariate/new",
    "tabicl-conf-shift-checkpoints/nano_graph_u/lr_search/../lr_search_covariate",
    "tabicl-conf-shift-checkpoints",
    "tabicl-conf-shift-evaluations/nano_graph_u/lr_search_tests",
    "tabicl-conf-shift-evaluations/nano_graph_u/lr_search_tests_covariate",
    "tabicl-conf-shift-evaluations/nano_graph_u/lr_search_tests_confounding_on_covariate",
    "tabicl-conf-shift-evaluations/nano_graph_u/lr_search_tests_covariate_on_confounding",
    ".",
])
def test_output_cannot_overlap_original_read_only_sources(bash, fixed_env, tmp_path, suffix):
    fixed_env["NANO_FIXED_ROOT"] = fixed_env["NANO_CLUSTER_ROOT"] + "/" + suffix
    result = run_script(bash, fixed_env, "--dry-run")
    assert result.returncode != 0
    assert "must differ from and not overlap" in result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("args", [("--unknown",), ("--dry-run", "extra")])
def test_unknown_arguments_fail_before_activation(bash, fixed_env, tmp_path, args):
    result = run_script(bash, fixed_env, *args)
    assert result.returncode != 0
    assert "Usage:" in result.stderr
    assert not list(tmp_path.iterdir())


@pytest.fixture
def fake_environment(tmp_path):
    initialization = tmp_path / "conda" / "etc" / "profile.d" / "conda.sh"
    initialization.parent.mkdir(parents=True)
    initialization.write_text(
        'conda() { return 0; }\n'
        'python() { if [[ "${1:-}" == "-" ]]; then return "${FIXED_TEST_GPU_EXIT_CODE}"; fi; '
        'printf "%s\\n" "$@" >> "${FIXED_TEST_LOG}"; '
        'printf "%s\\n" "$PYTHONUNBUFFERED" "$OMP_NUM_THREADS" "$OPENBLAS_NUM_THREADS" '
        '"$MKL_NUM_THREADS" "$NUMEXPR_NUM_THREADS" >> "${FIXED_TEST_ENV_LOG}"; '
        'return "${FIXED_TEST_EXIT_CODE}"; }\n',
        encoding="utf-8",
    )


@pytest.mark.parametrize("task_id", ["1", "2", "3", "4"])
@pytest.mark.parametrize("exit_code", ["0", "7"])
def test_each_task_invokes_one_controller_and_propagates_failure(
    bash, fixed_env, tmp_path, fake_environment, task_id, exit_code
):
    fixed_env.update(SLURM_ARRAY_TASK_ID=task_id, FIXED_TEST_EXIT_CODE=exit_code, WORKERS="3")
    result = run_script(bash, fixed_env)
    assert result.returncode == int(exit_code), result.stderr
    args = (tmp_path / "arguments.txt").read_text().splitlines()
    stage = fixed_env["FIXED_TEST_STAGE"]
    assert args[:4] == ["-u", "-m", "tabicl.nano_graph_u.fixed_experiments", stage]
    expected = {
        "--cluster-root": fixed_env["NANO_CLUSTER_ROOT"],
        "--root": fixed_env["NANO_FIXED_ROOT"],
        "--task-id": task_id,
    }
    expected.update({"--workers": "3"} if stage == "generate" else {"--device": "cuda"})
    assert dict(zip(args[4::2], args[5::2])) == expected
    assert args.count("-m") == 1
    assert (tmp_path / "environment.txt").read_text().splitlines() == ["1"] * 5
    marker = "FIXED_GENERATION_JOB_COMPLETE" if stage == "generate" else "FIXED_EXPERIMENT_JOB_COMPLETE"
    assert (marker in result.stdout) == (exit_code == "0")


def test_gpu_check_failure_stops_before_controller(bash, fixed_env, tmp_path, fake_environment):
    if fixed_env["FIXED_TEST_STAGE"] != "run":
        pytest.skip("CPU generation does not check a GPU")
    fixed_env.update(SLURM_ARRAY_TASK_ID="1", FIXED_TEST_GPU_EXIT_CODE="9")
    result = run_script(bash, fixed_env)
    assert result.returncode == 9
    assert not (tmp_path / "arguments.txt").exists()
    assert "FIXED_EXPERIMENT_JOB_COMPLETE" not in result.stdout


@pytest.mark.parametrize("script,partition,cpus,log", [
    (GENERATE_SCRIPT, "wdr", 8, "fixed-data"),
    (SCRIPT, "gpu", 2, "fixed"),
])
def test_each_stage_is_one_array_submission_with_no_extra_launchers(script, partition, cpus, log):
    source = script.read_text(encoding="utf-8")
    for directive in (
        f"--partition={partition}", f"--cpus-per-task={cpus}",
        "--time=48:00:00", "--array=1-4%2",
        f"--output=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-{log}-%A_%a.out",
        f"--error=/home/bin/CISPA-scratch/c01ilki/slurm-nano-gu-{log}-%A_%a.err",
    ):
        assert f"#SBATCH {directive}" in source
    assert "set -euo pipefail" in source
    assert not any(line.startswith(("sbatch ", "srun ")) for line in source.splitlines())
    assert b"\r" not in script.read_bytes()
    if script == GENERATE_SCRIPT:
        assert "#SBATCH --gres" not in source
        assert "import torch" not in source
        assert "--device cuda" not in source
        assert "run/validate" not in source
    else:
        assert "#SBATCH --gres=gpu:1" in source
        assert "import torch" in source
        assert "Stages=generate" not in source
        assert "--workers" not in source
