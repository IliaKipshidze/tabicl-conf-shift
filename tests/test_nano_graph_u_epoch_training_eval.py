"""Own-training-bank scoring uses frozen epoch winners without fitting again."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from test_nano_graph_u_epoch_experiments import banks as banks  # noqa: PLC0414
from test_nano_graph_u_epoch_experiments import compute as compute  # noqa: PLC0414

from tabicl.nano_graph_u import epoch_experiments as epochs
from tabicl.nano_graph_u.train import _sha256_file


@pytest.fixture
def selected(banks, compute):
    def prepare(task=3):
        epochs.run_experiment(*banks, task, device="cpu")
        return epochs.task_spec(*banks, task)

    return prepare


@pytest.fixture
def training_compute(monkeypatch):
    """Install only after the original training/test workflow has completed."""
    calls = []

    def install():
        h5py = pytest.importorskip("h5py")

        def forbidden(*args, **kwargs):
            pytest.fail("training-only evaluation must not train, select or test again")

        def evaluate(checkpoint, dump, **options):
            assert options == {"batch_size": 32, "max_batches": None,
                               "device": "cpu", "allow_training_dump": True}
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            config = payload["train_config"]
            assert Path(dump).resolve() == Path(config["dump_path"]).resolve()
            assert payload["step"] == config["epochs"] * 2500
            with h5py.File(dump, "r") as stream:
                metadata = json.loads(stream.attrs["metadata_json"])
            calls.append((Path(checkpoint), Path(dump)))
            prior = metadata["prior_config"]
            score = (0.6 + 0.001 * config["epochs"]
                     + 0.01 * (prior["graph_u_mode"] == "covariate")
                     + 0.0001 * prior["graph_u_query_location"])
            return {
                "format": "nano_graph_u_evaluation_v1",
                "checkpoint": str(Path(checkpoint).resolve()),
                "checkpoint_sha256": _sha256_file(Path(checkpoint)),
                "checkpoint_step": payload["step"], "dump": str(Path(dump).resolve()),
                "dump_sha256": _sha256_file(Path(dump)), "dump_metadata": metadata,
                "task_count": 80000, "query_row_count": 5000000,
                "metric_task_counts": {metric: 80000 for metric in epochs.METRIC_NAMES},
                "metrics": {metric: score for metric in epochs.METRIC_NAMES},
            }

        monkeypatch.setattr(epochs, "train_model", forbidden)
        monkeypatch.setattr(epochs, "_candidate", forbidden)
        monkeypatch.setattr(epochs, "_evaluate", forbidden)
        monkeypatch.setattr(epochs, "evaluate_checkpoint", evaluate)
        return calls

    return install


def _existing_hashes(banks):
    return {path: _sha256_file(path) for base in banks for path in base.rglob("*")
            if path.is_file() and path.suffix in (".h5", ".pt", ".json", ".csv")}


def test_scores_all_thirty_frozen_winners_and_collects_48_rows(banks, selected, training_compute):
    for task in range(1, 31):
        selected(task)
    before = _existing_hashes(banks)
    calls = training_compute()
    cluster, fixed, root = banks
    for task in range(1, 31):
        epochs.run_training_evaluation(cluster, fixed, root, task, device="cpu")
        if task < 30:
            assert not (root / "results/training_summary.csv").exists()
    assert len(calls) == 30
    assert all(_sha256_file(path) == digest for path, digest in before.items())
    raw = pd.read_csv(root / "results/training_per_repetition.csv")
    summary = pd.read_csv(root / "results/training_summary.csv")
    assert len(raw) == len(summary) == 48
    assert raw["checkpoint_sha256"].nunique() == 30
    assert raw["task_count"].eq(80000).all() and summary["task_count_per_repetition"].eq(80000).all()
    assert raw["evaluation_split"].eq("training").all() and raw["scored_rows"].eq("query").all()
    assert raw["repetition"].eq(1).all() and summary["repetition_count"].eq(1).all()
    assert raw["shuffle_batches"].all() and raw["training_steps"].eq(2500 * raw["epochs"]).all()
    assert not any(column.endswith("_std") for column in summary)
    keys = ["epochs", "fixed_location", "fixed_scale", "training_graph_u_mode", "training_condition"]
    assert not raw.duplicated(keys).any()
    assert raw.groupby(["epochs", "fixed_location", "fixed_scale"]).size().eq(4).all()
    np.testing.assert_allclose(
        summary.set_index(keys).sort_index()[[f"{metric}_mean" for metric in epochs.METRIC_NAMES]],
        raw.set_index(keys).sort_index()[list(epochs.METRIC_NAMES)], rtol=0, atol=1e-12)
    iid = raw.loc[raw["training_condition"].eq("identity")]
    assert iid["checkpoint_sha256"].nunique() == 6
    assert iid.groupby("checkpoint_sha256").size().eq(4).all()
    winners = pd.read_csv(root / "results/winners.csv")
    matched = raw.merge(winners[["checkpoint_sha256", "learning_rate", "epochs"]],
                        on=["checkpoint_sha256", "learning_rate", "epochs"], validate="many_to_one")
    assert len(matched) == 48
    csv_before = {path: path.read_bytes() for path in (root / "results").glob("*.csv")}
    epochs.run_training_evaluation(cluster, fixed, root, 1, device="cpu")
    assert len(calls) == 30
    assert all(path.read_bytes() == content for path, content in csv_before.items())


@pytest.mark.parametrize("corruption", ["missing_winner", "checkpoint_budget", "partial_bank"])
def test_invalid_selected_inputs_are_rejected_before_inference(banks, selected, training_compute, corruption):
    spec = selected()
    directory = Path(spec["directory"])
    winner = json.loads((directory / "winner.json").read_text(encoding="utf-8"))
    if corruption == "missing_winner":
        (directory / "winner.json").unlink()
    elif corruption == "checkpoint_budget":
        path = Path(winner["checkpoint"])
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload["step"] = 2500
        torch.save(payload, path)
    else:
        h5py = pytest.importorskip("h5py")
        with h5py.File(spec["train_dump"], "r+") as stream:
            stream.attrs["committed_steps"] = 2499
    calls = training_compute()
    with pytest.raises((ValueError, FileNotFoundError)):
        epochs.run_training_evaluation(*banks, 3, device="cpu")
    assert calls == []
    assert not (directory / "training_evaluation.json").exists()


@pytest.mark.parametrize("corruption", ["wrong_dump", "metric_count", "short_prefix"])
def test_cached_training_scores_must_cover_the_own_complete_bank(banks, selected, training_compute, corruption):
    spec = selected()
    calls = training_compute()
    epochs.run_training_evaluation(*banks, 3, device="cpu")
    path = Path(spec["directory"]) / "training_evaluation.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    report = next(value for value in saved.values()
                  if isinstance(value, dict) and value.get("format") == "nano_graph_u_evaluation_v1")
    if corruption == "wrong_dump":
        report["dump"] = spec["validation_dump"]
    elif corruption == "metric_count":
        report["metric_task_counts"]["roc_auc"] = 79999
    else:
        report["task_count"] = 79968
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError):
        epochs.run_training_evaluation(*banks, 3, device="cpu")
    assert len(calls) == 1


def test_training_only_mode_requires_an_already_selected_model(banks, training_compute):
    calls = training_compute()
    with pytest.raises(FileNotFoundError):
        epochs.run_training_evaluation(*banks, 3, device="cpu")
    assert calls == []


def test_cli_training_only_dispatch_cannot_enter_optimization(tmp_path, monkeypatch):
    calls = []

    def forbidden(*args, **kwargs):
        pytest.fail("training-only CLI entered the training/tuning workflow")

    monkeypatch.setattr(epochs, "run_experiment", forbidden)
    monkeypatch.setattr(epochs, "run_training_evaluation", lambda *args, **kwargs: calls.append((args, kwargs)))
    epochs.main(["--cluster-root", str(tmp_path / "cluster"), "--fixed-root", str(tmp_path / "fixed"),
                 "--root", str(tmp_path / "outputs"), "--task-id", "23", "--device", "cpu",
                 "--training-eval-only"])
    assert calls == [((tmp_path / "cluster", tmp_path / "fixed", tmp_path / "outputs", 23), {"device": "cpu"})]
