"""Held-out summaries retain the four cells and five repetition-level scores."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

from tabicl.nano_graph_u import summarize_tests as summary_module


@pytest.fixture
def completed_tests(tmp_path, monkeypatch):
    data_root, search_root, result_root = (
        tmp_path / "data",
        tmp_path / "search",
        tmp_path / "results",
    )
    specs, records, validated = [], [], []
    for task in range(1, 21):
        spec = summary_module.test_spec(task, data_root, search_root, result_root)
        repetition = spec["repetition"]
        training = spec["training_condition"]
        testing = spec["test_condition"]
        offset = 0.02 * (training == "identity") + 0.03 * (testing == "identity")
        accuracy = 0.6 + repetition * 0.01 + offset
        metrics = {
            "accuracy": accuracy,
            "balanced_accuracy": accuracy - 0.02,
            "macro_f1": accuracy - 0.01,
            "roc_auc": accuracy + 0.1,
            "nll": 0.5 - repetition * 0.01,
            "brier": 0.2 - repetition * 0.005,
        }
        record = {
            "format": "nano_graph_u_winner_test_v1",
            "test": spec,
            "manifest_sha256": "a" * 64,
            "winner": {
                "learning_rate": 0.0015 if training == "identity" else 0.001,
                "model_seed": 41 + repetition,
            },
            "evaluation_report": {
                "checkpoint": spec["checkpoint"],
                "checkpoint_sha256": hashlib.sha256(
                    f"model/{repetition}/{training}".encode()
                ).hexdigest(),
                "checkpoint_step": 2500,
                "dump": spec["test_dump"],
                "dump_sha256": hashlib.sha256(
                    f"bank/{repetition}/{testing}".encode()
                ).hexdigest(),
                "dump_metadata": {"seed": 2000 + repetition},
                "task_count": 3200,
                "query_row_count": 3200 * (15 + repetition * 10),
                "metric_task_counts": {key: 3200 for key in metrics},
                "metrics": metrics,
            },
        }
        target = Path(spec["result_path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record), encoding="utf-8")
        specs.append(spec)
        records.append(record)

    def validate(record, spec, *, verify_files, hash_cache):
        assert verify_files is True
        assert isinstance(hash_cache, dict)
        validated.append(spec["task_id"])
        if record["test"] != spec or record["format"] != "nano_graph_u_winner_test_v1":
            raise ValueError("Invalid test result")
        if record["evaluation_report"]["task_count"] != 3200:
            raise ValueError("Incomplete test result")

    # File-integrity checks are exercised in the test-runner suite; isolate
    # aggregation here while still requiring every result to pass its validator.
    monkeypatch.setattr(summary_module, "validate_test_result", validate)
    return data_root, search_root, result_root, specs, records, validated


def rewrite(spec, record):
    Path(spec["result_path"]).write_text(json.dumps(record), encoding="utf-8")


def test_test_summary_uses_sample_std_of_five_task_means_and_is_idempotent(
    completed_tests,
):
    data, search, output, _, _, validated = completed_tests
    summary = summary_module.summarize_tests(data, search, output)
    assert validated == list(range(1, 21))
    assert summary["evaluation_count"] == 20
    assert summary["repetition_count"] == 5
    assert summary["standard_deviation_ddof"] == 1
    assert summary["winners_manifest_sha256"] == "a" * 64
    assert len(summary["cells"]) == 4
    for cell in summary["cells"]:
        offset = 0.02 * (cell["training_condition"] == "identity") + 0.03 * (
            cell["test_condition"] == "identity"
        )
        assert cell["repetition_count"] == 5
        assert cell["metrics"]["accuracy"]["mean"] == pytest.approx(0.63 + offset)
        assert cell["metrics"]["accuracy"]["std"] == pytest.approx(
            0.01 * math.sqrt(2.5)
        )
        assert cell["metrics"]["brier"]["std"] == pytest.approx(0.005 * math.sqrt(2.5))
    assert len(summary["per_repetition"]) == 20
    assert {row["learning_rate"] for row in summary["per_repetition"]} == {
        0.001,
        0.0015,
    }
    for name, count in (("per_repetition.csv", 20), ("summary.csv", 4)):
        with (output / name).open(encoding="utf-8", newline="") as stream:
            assert len(list(csv.DictReader(stream))) == count
    assert json.loads((output / "summary.json").read_text()) == summary
    before = {path.name: path.read_bytes() for path in output.glob("*.csv")}
    assert summary_module.summarize_tests(data, search, output) == summary
    assert before == {path.name: path.read_bytes() for path in output.glob("*.csv")}


def test_summary_requires_all_twenty_results_before_writing(completed_tests):
    data, search, output, specs, _, validated = completed_tests
    Path(specs[9]["result_path"]).unlink()
    with pytest.raises(FileNotFoundError, match="1 of 20"):
        summary_module.summarize_tests(data, search, output)
    assert not validated
    assert not (output / "summary.json").exists()


@pytest.mark.parametrize(
    "corruption", ["manifest", "winner", "test_bank", "count", "cell"]
)
def test_summary_refuses_mismatched_or_incomplete_results(completed_tests, corruption):
    data, search, output, specs, records, _ = completed_tests
    index = 1 if corruption == "winner" else 2
    record = records[index]
    if corruption == "manifest":
        record["manifest_sha256"] = "b" * 64
    elif corruption == "winner":
        record["evaluation_report"]["checkpoint_sha256"] = "b" * 64
    elif corruption == "test_bank":
        record["evaluation_report"]["dump_sha256"] = "b" * 64
    elif corruption == "count":
        record["evaluation_report"]["task_count"] = 3199
    else:
        record["test"] = dict(record["test"], repetition=5)
    rewrite(specs[index], record)
    with pytest.raises(ValueError):
        summary_module.summarize_tests(data, search, output)
    assert not (output / "summary.json").exists()


def test_summary_preserves_existing_outputs_when_scores_change(completed_tests):
    data, search, output, specs, records, _ = completed_tests
    summary_module.summarize_tests(data, search, output)
    paths = [
        output / name for name in ("summary.json", "summary.csv", "per_repetition.csv")
    ]
    before = {path: path.read_bytes() for path in paths}
    records[0]["evaluation_report"]["metrics"]["accuracy"] += 0.01
    rewrite(specs[0], records[0])
    with pytest.raises(FileExistsError, match="different selection"):
        summary_module.summarize_tests(data, search, output)
    assert before == {path: path.read_bytes() for path in paths}


def test_summary_cli_prints_four_cells_and_summary_location(completed_tests, capsys):
    data, search, output, _, _, _ = completed_tests
    summary_module.main(
        [
            "--data-root",
            str(data),
            "--search-root",
            str(search),
            "--result-root",
            str(output),
        ]
    )
    lines = capsys.readouterr().out.splitlines()
    assert len([line for line in lines if line.startswith("training=")]) == 4
    assert all("n=5" in line for line in lines if line.startswith("training="))
    assert "SUMMARY_READY" in lines[-1]
