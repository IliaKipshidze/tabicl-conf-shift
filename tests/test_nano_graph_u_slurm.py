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
TUNE_SCRIPT = REPO / "scripts" / "slurm_nano_graph_u_tune.sh"
SELECT_SCRIPT = REPO / "scripts" / "slurm_nano_graph_u_select.sh"
TEST_SCRIPT = REPO / "scripts" / "slurm_nano_graph_u_test.sh"
SUMMARY_SCRIPT = REPO / "scripts" / "slurm_nano_graph_u_summarize.sh"


@pytest.fixture(scope="module")
def bash():
    candidates = [
        shutil.which("bash"),
        "C:/Program Files/Git/bin/bash.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            probe = subprocess.run(
                [candidate, "-c", "exit 0"],
                capture_output=True,
                timeout=10,
                check=False,
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
    for key in (
        "SLURM_ARRAY_TASK_ID",
        "WORKERS",
        "GRAPH_U_MODE",
        "TEST_GRAPH_U_MODE",
        "NANO_TRAIN_DATA_ROOT",
    ):
        env.pop(key, None)
    env.update(
        NANO_CLUSTER_ROOT=shell_path(tmp_path),
        NANO_REPO_DIR=shell_path(REPO),
        NANO_CONDA_ROOT=shell_path(tmp_path / "conda"),
        NANO_ENV_PATH=shell_path(tmp_path / "env"),
        NANO_DATA_ROOT=shell_path(tmp_path / "data banks"),
        NANO_SEARCH_ROOT=shell_path(tmp_path / "search outputs"),
        NANO_TEST_ROOT=shell_path(tmp_path / "test outputs"),
        SLURM_CPUS_PER_TASK="8",
        NANO_TEST_ARGS_LOG=shell_path(tmp_path / "arguments.txt"),
        NANO_TEST_EXIT_CODE="0",
        NANO_TEST_FLOCK_EXIT_CODE="0",
    )
    return env


def run_script(bash, env, *arguments, script=SCRIPT):
    return subprocess.run(
        [bash, str(script), *arguments],
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
            (
                directory.rsplit("/", 1)[1],
                condition,
                split,
                options["--seed"],
                options["--steps"],
            )
        )
        dumps.add(dump)
    assert len(commands) == len(dumps) == 22
    assert observed == expected
    assert not (tmp_path / "data banks").exists()


@pytest.mark.parametrize("task", ["", "0", "23", "../1"])
def test_generation_job_rejects_invalid_task_before_writes(
    bash, job_env, tmp_path, task
):
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
        'python() { if [[ "${1:-}" == "-" ]]; then return 0; fi; '
        'printf "%s\\n" "$@" > "${NANO_TEST_ARGS_LOG}"; '
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


def test_tuning_slurm_plan_matches_all_100_python_trial_specs(bash, job_env, tmp_path):
    from tabicl.nano_graph_u.tune import trial_spec

    result = run_script(bash, job_env, "--dry-run", script=TUNE_SCRIPT)
    assert result.returncode == 0, result.stderr
    plans = [
        dict(field.split("=", 1) for field in line.split())
        for line in result.stdout.splitlines()
        if line.startswith("trial=")
    ]
    directories = [
        line.removeprefix("Candidate directory=")
        for line in result.stdout.splitlines()
        if line.startswith("Candidate directory=")
    ]
    assert len(plans) == len(set(directories)) == 100
    for task_id, plan in enumerate(plans, start=1):
        spec = trial_spec(task_id, tmp_path / "data", tmp_path / "search")
        assert int(plan["trial"]) == spec["task_id"]
        assert int(plan["repetition"]) == spec["repetition"]
        assert plan["condition"] == spec["condition"]
        assert int(plan["model_seed"]) == spec["model_seed"]
        assert float(plan["learning_rate"]) == spec["learning_rate"]
        assert int(plan["steps"]) == 2500
        assert directories[task_id - 1].endswith(
            f"/rep_{spec['repetition']}/{spec['condition']}/"
            + Path(spec["checkpoint_dir"]).parent.name
        )
    assert "tabicl.nano_graph_u.tune trial" in result.stdout
    assert "tabicl.nano_graph_u generate" not in result.stdout
    assert not (tmp_path / "search outputs").exists()


@pytest.mark.parametrize("task", ["1", "10", "11", "20", "81", "100"])
def test_tuning_job_runs_only_its_candidate_and_validation(
    bash, job_env, tmp_path, mock_conda, task
):
    job_env["SLURM_ARRAY_TASK_ID"] = task
    result = run_script(bash, job_env, script=TUNE_SCRIPT)
    assert result.returncode == 0, result.stderr
    arguments = (tmp_path / "arguments.txt").read_text().splitlines()
    assert arguments[:4] == ["-u", "-m", "tabicl.nano_graph_u.tune", "trial"]
    assert arguments[arguments.index("--task-id") + 1] == task
    assert arguments[arguments.index("--device") + 1] == "cuda"
    assert "TUNING_JOB_COMPLETE" in result.stdout


@pytest.mark.parametrize("task", ["", "0", "101"])
def test_tuning_job_rejects_missing_or_invalid_task(bash, job_env, tmp_path, task):
    job_env["SLURM_ARRAY_TASK_ID"] = task
    result = run_script(bash, job_env, script=TUNE_SCRIPT)
    assert result.returncode != 0
    assert "from 1 to 100" in result.stderr
    assert not (tmp_path / "arguments.txt").exists()


def test_selection_job_calls_only_winner_selection(bash, job_env, tmp_path, mock_conda):
    result = run_script(bash, job_env, script=SELECT_SCRIPT)
    assert result.returncode == 0, result.stderr
    arguments = (tmp_path / "arguments.txt").read_text().splitlines()
    assert arguments[:3] == ["-u", "-m", "tabicl.nano_graph_u.select"]
    assert "SELECTION_COMPLETE" in result.stdout


@pytest.mark.parametrize("script", [TUNE_SCRIPT, SELECT_SCRIPT])
def test_tuning_or_selection_failure_cannot_report_completion(
    bash, job_env, tmp_path, mock_conda, script
):
    job_env.update(SLURM_ARRAY_TASK_ID="1", NANO_TEST_EXIT_CODE="7")
    result = run_script(bash, job_env, script=script)
    assert result.returncode == 7
    assert "JOB_COMPLETE" not in result.stdout
    assert "SELECTION_COMPLETE" not in result.stdout


def test_winner_test_plan_matches_all_20_python_specs(bash, job_env, tmp_path):
    from tabicl.nano_graph_u.test_winners import test_spec as make_spec

    result = run_script(bash, job_env, "--dry-run", script=TEST_SCRIPT)
    assert result.returncode == 0, result.stderr
    plans = [
        dict(field.split("=", 1) for field in line.split())
        for line in result.stdout.splitlines()
        if line.startswith("test=")
    ]
    checkpoints = [
        line.removeprefix("Checkpoint=")
        for line in result.stdout.splitlines()
        if line.startswith("Checkpoint=")
    ]
    dumps = [
        line.removeprefix("Test bank=")
        for line in result.stdout.splitlines()
        if line.startswith("Test bank=")
    ]
    reports = [
        line.removeprefix("Result=")
        for line in result.stdout.splitlines()
        if line.startswith("Result=")
    ]
    assert len(plans) == len(set(reports)) == 20
    assert len(set(checkpoints)) == len(set(dumps)) == 10
    for task_id, plan in enumerate(plans, start=1):
        spec = make_spec(
            task_id, tmp_path / "banks", tmp_path / "search", tmp_path / "results"
        )
        assert int(plan["test"]) == spec["task_id"]
        assert int(plan["repetition"]) == spec["repetition"]
        assert plan["training"] == spec["training_condition"]
        assert plan["evaluation"] == spec["test_condition"]
        assert int(plan["test_seed"]) == 2000 + spec["repetition"]
        assert int(plan["tables"]) == 3200
        for paths, key, levels in (
            (checkpoints, "checkpoint", 4),
            (dumps, "test_dump", 2),
            (reports, "result_path", 2),
        ):
            suffix = "/".join(Path(spec[key]).parts[-levels:])
            assert paths[task_id - 1].endswith("/" + suffix)
    assert "tabicl.nano_graph_u.test_winners" in result.stdout
    assert "tabicl.nano_graph_u generate" not in result.stdout
    assert not (tmp_path / "test outputs").exists()


@pytest.mark.parametrize("task", ["1", "2", "3", "4", "20"])
def test_winner_test_job_calls_only_frozen_model_evaluation(
    bash, job_env, tmp_path, mock_conda, task
):
    job_env["SLURM_ARRAY_TASK_ID"] = task
    result = run_script(bash, job_env, script=TEST_SCRIPT)
    assert result.returncode == 0, result.stderr
    arguments = (tmp_path / "arguments.txt").read_text().splitlines()
    assert arguments[:3] == ["-u", "-m", "tabicl.nano_graph_u.test_winners"]
    assert arguments[arguments.index("--task-id") + 1] == task
    assert arguments[arguments.index("--device") + 1] == "cuda"
    assert "TEST_JOB_COMPLETE" in result.stdout


@pytest.mark.parametrize("task", ["", "0", "21"])
def test_winner_test_job_rejects_invalid_task(bash, job_env, tmp_path, task):
    job_env["SLURM_ARRAY_TASK_ID"] = task
    result = run_script(bash, job_env, script=TEST_SCRIPT)
    assert result.returncode != 0
    assert "from 1 to 20" in result.stderr
    assert not (tmp_path / "arguments.txt").exists()


def test_summary_job_only_collects_saved_test_reports(
    bash, job_env, tmp_path, mock_conda
):
    result = run_script(bash, job_env, script=SUMMARY_SCRIPT)
    assert result.returncode == 0, result.stderr
    arguments = (tmp_path / "arguments.txt").read_text().splitlines()
    assert arguments[:3] == ["-u", "-m", "tabicl.nano_graph_u.summarize_tests"]
    assert "SUMMARY_COMPLETE" in result.stdout


@pytest.mark.parametrize("script", [TEST_SCRIPT, SUMMARY_SCRIPT])
def test_final_testing_failure_cannot_report_completion(
    bash, job_env, tmp_path, mock_conda, script
):
    job_env.update(SLURM_ARRAY_TASK_ID="1", NANO_TEST_EXIT_CODE="7")
    result = run_script(bash, job_env, script=script)
    assert result.returncode == 7
    assert "JOB_COMPLETE" not in result.stdout
    assert "SUMMARY_COMPLETE" not in result.stdout


@pytest.mark.parametrize("script", [SCRIPT, TUNE_SCRIPT, TEST_SCRIPT])
def test_covariate_plans_use_separate_default_roots_and_explicit_mode(
    bash, job_env, tmp_path, script
):
    job_env["GRAPH_U_MODE"] = "covariate"
    for name in ("NANO_DATA_ROOT", "NANO_SEARCH_ROOT", "NANO_TEST_ROOT"):
        job_env.pop(name)
    result = run_script(bash, job_env, "--dry-run", script=script)
    assert result.returncode == 0, result.stderr
    assert "--graph-u-mode covariate" in result.stdout
    if script == SCRIPT:
        assert "/repetitions_covariate/" in result.stdout
        assert "--query-location 0.0 --query-scale 1.0" in result.stdout
        assert "--query-location 2.0 --query-scale 1.5" in result.stdout
        assert result.stdout.count("Command:") == 22
    else:
        assert "/lr_search_covariate/" in result.stdout
    if script == TEST_SCRIPT:
        assert "/lr_search_tests_covariate/" in result.stdout
        assert "/repetitions_covariate/" in result.stdout
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "script", [SCRIPT, TUNE_SCRIPT, SELECT_SCRIPT, TEST_SCRIPT, SUMMARY_SCRIPT]
)
def test_covariate_mode_reaches_each_python_entrypoint(
    bash, job_env, tmp_path, mock_conda, script
):
    job_env.update(GRAPH_U_MODE="covariate", SLURM_ARRAY_TASK_ID="1")
    result = run_script(bash, job_env, script=script)
    assert result.returncode == 0, result.stderr
    arguments = (tmp_path / "arguments.txt").read_text().splitlines()
    assert arguments[arguments.index("--graph-u-mode") + 1] == "covariate"


@pytest.mark.parametrize(
    "script", [SCRIPT, TUNE_SCRIPT, SELECT_SCRIPT, TEST_SCRIPT, SUMMARY_SCRIPT]
)
def test_unknown_graph_u_mode_fails_before_writes(bash, job_env, tmp_path, script):
    job_env["GRAPH_U_MODE"] = "invalid"
    result = run_script(bash, job_env, "--dry-run", script=script)
    assert result.returncode != 0
    assert "GRAPH_U_MODE must be" in result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "training_mode,test_mode",
    [("confounding", "covariate"), ("covariate", "confounding")],
)
def test_cross_evaluation_plans_match_model_and_bank_roots(
    bash, job_env, tmp_path, training_mode, test_mode
):
    from tabicl.nano_graph_u.test_winners import test_spec as make_spec

    job_env.update(GRAPH_U_MODE=training_mode, TEST_GRAPH_U_MODE=test_mode)
    for name in ("NANO_DATA_ROOT", "NANO_SEARCH_ROOT", "NANO_TEST_ROOT"):
        job_env.pop(name)
    result = run_script(bash, job_env, "--dry-run", script=TEST_SCRIPT)
    assert result.returncode == 0, result.stderr
    commands = [
        shlex.split(line.removeprefix("Command:"))
        for line in result.stdout.splitlines()
        if line.startswith("Command:")
    ]
    model_suffix = "_covariate" if training_mode == "covariate" else ""
    bank_suffix = "_covariate" if test_mode == "covariate" else ""
    for task, command in enumerate(commands, start=1):
        options = dict(zip(command[4::2], command[5::2]))
        assert options["--graph-u-mode"] == training_mode
        assert options["--test-graph-u-mode"] == test_mode
        assert options["--data-root"].endswith(f"/repetitions{bank_suffix}")
        assert options["--training-data-root"].endswith(f"/repetitions{model_suffix}")
        assert options["--search-root"].endswith(f"/lr_search{model_suffix}")
        assert options["--result-root"].endswith(
            f"/lr_search_tests_{training_mode}_on_{test_mode}"
        )
        spec = make_spec(
            task,
            options["--data-root"],
            options["--search-root"],
            options["--result-root"],
            graph_u_mode=training_mode,
            test_graph_u_mode=test_mode,
            training_data_root=options["--training-data-root"],
        )
        assert spec["repetition"] == (task - 1) // 4 + 1
        assert (
            f"/rep_{spec['repetition']}/{spec['training_condition']}/model.pt"
            in spec["checkpoint"].replace("\\", "/")
        )
    assert len(commands) == 20
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("script", [TEST_SCRIPT, SUMMARY_SCRIPT])
@pytest.mark.parametrize(
    "training_mode,test_mode",
    [("confounding", "covariate"), ("covariate", "confounding")],
)
def test_cross_scripts_propagate_both_modes_and_training_root(
    bash, job_env, tmp_path, mock_conda, script, training_mode, test_mode
):
    job_env.update(
        GRAPH_U_MODE=training_mode,
        TEST_GRAPH_U_MODE=test_mode,
        SLURM_ARRAY_TASK_ID="1",
        NANO_TRAIN_DATA_ROOT=shell_path(tmp_path / "original training banks"),
    )
    result = run_script(bash, job_env, script=script)
    assert result.returncode == 0, result.stderr
    arguments = (tmp_path / "arguments.txt").read_text().splitlines()
    assert arguments[arguments.index("--graph-u-mode") + 1] == training_mode
    assert arguments[arguments.index("--test-graph-u-mode") + 1] == test_mode
    assert (
        arguments[arguments.index("--training-data-root") + 1]
        == job_env["NANO_TRAIN_DATA_ROOT"]
    )
    assert "generate" not in arguments and "train" not in arguments


@pytest.mark.parametrize("script", [TEST_SCRIPT, SUMMARY_SCRIPT])
def test_unknown_test_graph_u_mode_fails_before_writes(bash, job_env, tmp_path, script):
    job_env["TEST_GRAPH_U_MODE"] = "invalid"
    result = run_script(bash, job_env, script=script)
    assert result.returncode != 0
    assert "TEST_GRAPH_U_MODE must be" in result.stderr
    assert not list(tmp_path.iterdir())
