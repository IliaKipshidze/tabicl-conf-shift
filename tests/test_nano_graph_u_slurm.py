"""Check generation-job mapping and execution without a Slurm cluster."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "slurm_nano_graph_u_generate.sh"


@pytest.fixture(scope="module")
def bash():
    candidates = [
        shutil.which("bash"),
        "C:/Program Files/Git/bin/bash.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            probe = subprocess.run(
                [candidate, "-c", "exit 0"], capture_output=True, timeout=10, check=False
            )
            if probe.returncode == 0:
                return candidate
    pytest.skip("A working Bash executable is required for Slurm launcher checks")


def shell_path(path: Path) -> str:
    value = path.resolve().as_posix()
    if os.name == "nt":
        return f"/{value[0].lower()}{value[2:]}"
    return value


@pytest.fixture
def job_env(tmp_path):
    env = dict(os.environ)
    for key in ("SLURM_ARRAY_TASK_ID", "WORKERS"):
        env.pop(key, None)
    env.update(
        NANO_CLUSTER_ROOT=shell_path(tmp_path),
        NANO_REPO_DIR=shell_path(REPO),
        NANO_CONDA_ROOT=shell_path(tmp_path / "conda"),
        NANO_ENV_PATH=shell_path(tmp_path / "env"),
        NANO_DATA_ROOT=shell_path(tmp_path / "data banks"),
        SLURM_CPUS_PER_TASK="8",
        NANO_TEST_ARGS_LOG=shell_path(tmp_path / "arguments.txt"),
        NANO_TEST_EXIT_CODE="0",
        NANO_TEST_FLOCK_EXIT_CODE="0",
    )
    return env


def run_script(bash, env, *arguments):
    return subprocess.run(
        [bash, str(SCRIPT), *arguments],
        env=env,
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_generation_array_plans_exactly_the_22_agreed_banks(bash, job_env, tmp_path):
    result = run_script(bash, job_env, "--dry-run")
    assert result.returncode == 0, result.stderr
    commands = [
        shlex.split(line.removeprefix("Command:"))
        for line in result.stdout.splitlines()
        if line.startswith("Command:")
    ]
    expected = set()
    for repetition in range(1, 6):
        for split, steps, seed in (
            ("train", "2500", 1000 + repetition),
            ("test", "100", 2000 + repetition),
        ):
            for condition in ("identity", "shift"):
                expected.add((f"rep_{repetition}", condition, split, str(seed), steps))
    for condition in ("identity", "shift"):
        expected.add(("validation", condition, "validation", "3001", "100"))

    observed = set()
    dumps = set()
    for command in commands:
        assert command[:5] == ["python", "-u", "-m", "tabicl.nano_graph_u", "generate"]
        options = dict(zip(command[5::2], command[6::2]))
        assert options["--batch-size"] == "32"
        assert options["--rows"] == "150"
        assert options["--features"] == "5"
        assert options["--n-jobs"] == "1"
        assert options["--workers"] == "8"
        assert options["--min-train-size"] == "0.3"
        assert options["--max-train-size"] == "0.9"
        assert options["--structure-mode"] == "add_root"
        assert options["--max-attempts"] == "1000"
        assert "--force-gaussian" not in command
        dump = options["--dump"]
        directory, name = dump.rsplit("/", 1)
        condition, split = name.split("_")[:2]
        assert (options["--query-location"], options["--query-scale"]) == (
            ("0.0", "1.0") if condition == "identity" else ("2.0", "1.5")
        )
        observed.add(
            (directory.rsplit("/", 1)[1], condition, split, options["--seed"], options["--steps"])
        )
        dumps.add(dump)
    assert len(commands) == len(dumps) == 22
    assert observed == expected
    assert not (tmp_path / "data banks").exists()


@pytest.mark.parametrize("task", ["", "0", "23", "../1"])
def test_generation_job_rejects_invalid_task_before_writes(bash, job_env, tmp_path, task):
    job_env["SLURM_ARRAY_TASK_ID"] = task
    result = run_script(bash, job_env)
    assert result.returncode != 0
    assert "from 1 to 22" in result.stderr
    assert not (tmp_path / "data banks").exists()


def test_generation_workers_follow_cpu_allocation_and_cannot_exceed_it(bash, job_env):
    job_env.update(SLURM_CPUS_PER_TASK="4", SLURM_ARRAY_TASK_ID="1")
    result = run_script(bash, job_env, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "--workers 4" in result.stdout
    job_env["WORKERS"] = "5"
    result = run_script(bash, job_env, "--dry-run")
    assert result.returncode != 0
    assert "must not exceed" in result.stderr


@pytest.fixture
def mock_conda(tmp_path):
    initialization = tmp_path / "conda" / "etc" / "profile.d" / "conda.sh"
    initialization.parent.mkdir(parents=True)
    initialization.write_text(
        "conda() { return 0; }\n"
        'python() { printf "%s\\n" "$@" > "${NANO_TEST_ARGS_LOG}"; '
        'return "${NANO_TEST_EXIT_CODE}"; }\n'
        'flock() { return "${NANO_TEST_FLOCK_EXIT_CODE}"; }\n',
        encoding="utf-8",
    )


@pytest.mark.parametrize("task", ["1", "4", "21", "22"])
def test_generation_job_only_invokes_generate_and_resumes_existing_bank(
    bash, job_env, tmp_path, mock_conda, task
):
    job_env["SLURM_ARRAY_TASK_ID"] = task
    result = run_script(bash, job_env)
    assert result.returncode == 0, result.stderr
    arguments = (tmp_path / "arguments.txt").read_text().splitlines()
    assert arguments[:4] == ["-u", "-m", "tabicl.nano_graph_u", "generate"]
    assert "--resume" not in arguments
    assert "GENERATION_COMPLETE" in result.stdout

    dump = next((tmp_path / "data banks").rglob("*.job.lock"))
    dump = dump.with_suffix("").with_suffix("")
    dump.write_bytes(b"existing bank")
    result = run_script(bash, job_env)
    assert result.returncode == 0, result.stderr
    arguments = (tmp_path / "arguments.txt").read_text().splitlines()
    assert arguments[-1] == "--resume"
    assert dump.read_bytes() == b"existing bank"


@pytest.mark.parametrize("failure", ["generator", "lock"])
def test_generation_failure_cannot_report_completion(
    bash, job_env, tmp_path, mock_conda, failure
):
    job_env["SLURM_ARRAY_TASK_ID"] = "1"
    if failure == "generator":
        job_env["NANO_TEST_EXIT_CODE"] = "7"
    else:
        job_env["NANO_TEST_FLOCK_EXIT_CODE"] = "1"
    result = run_script(bash, job_env)
    assert result.returncode != 0
    assert "GENERATION_COMPLETE" not in result.stdout
    if failure == "lock":
        assert not (tmp_path / "arguments.txt").exists()
        assert "Another generation job" in result.stderr
