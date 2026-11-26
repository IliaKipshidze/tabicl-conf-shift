"""Selection and cross-mode scoring contracts for common-support experiments."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tabicl.nano_graph_u import pool_experiment as pool
from tabicl.nano_graph_u.data import _metadata
from tabicl.nano_graph_u.train import (
    BATCH_ORDER_SCHEME,
    PAPER_MODEL_CONFIG,
    _sha256_file,
)


def _sparse_bank(spec):
    """Exercise full bank contracts without generating or allocating training data."""
    h5py = pytest.importorskip("h5py")
    metadata = _metadata(
        pool.prior_config(spec["condition"], spec["mode"]),
        steps=spec["steps"],
        batch_size=32,
        seed=spec["seed"],
        rows=150,
        features=5,
        min_train_size=0.3,
        max_train_size=0.9,
        n_jobs=1,
    )
    path = Path(spec["dump"])
    path.parent.mkdir(parents=True, exist_ok=True)
    count = spec["steps"] * 32
    with h5py.File(path, "w") as stream:
        stream.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
        stream.attrs["committed_steps"] = spec["steps"]
        stream.create_dataset("x", (count, 150, 5), chunks=(32, 150, 5), dtype="f4")
        stream.create_dataset("y", (count, 150), chunks=(32, 150), dtype="i8")
        stream.create_dataset("train_sizes", (count,), chunks=(32,), dtype="i4")
    assert path.stat().st_size < 100_000
    return metadata


@pytest.fixture
def banks(tmp_path):
    root = pool._initialize(tmp_path / "pool experiment")
    for task_id in range(1, 67):
        _sparse_bank(pool.bank_spec(root, task_id))
    return root


@pytest.fixture
def compute(monkeypatch):
    h5py = pytest.importorskip("h5py")
    calls = {"train": [], "validation": [], "test": [], "training": []}

    def metadata(path):
        with h5py.File(path, "r") as stream:
            return json.loads(stream.attrs["metadata_json"])

    def train(dump, directory, **options):
        assert options["steps"] == 2500 and options["epochs"] == 1
        assert options["shuffle_batches"] is True and options["batch_size"] == 32
        assert options["seed"] in range(42, 47)
        calls["train"].append((str(dump), options))
        data = metadata(dump)
        config = {
            "seed": options["seed"],
            "batch_size": 32,
            "learning_rate": options["learning_rate"],
            "optimizer": "schedulefree.AdamWScheduleFree",
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "loss": "query_cross_entropy",
            "precision": "float32",
            "dump_path": str(Path(dump).resolve()),
            "dump_sha256": _sha256_file(Path(dump)),
            "shuffle_batches": True,
            "batch_order_scheme": BATCH_ORDER_SCHEME,
        }
        path = Path(directory) / "latest.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "nano_graph_u_v1",
                "step": 2500,
                "model_config": PAPER_MODEL_CONFIG,
                "model": {"fixture": torch.tensor(0)},
                "prior_config": data["prior_config"],
                "dump_metadata": data,
                "train_config": config,
            },
            path,
        )
        return path

    def evaluate(checkpoint, dump, **options):
        assert options["batch_size"] == 32 and options["device"] == "cpu"
        assert options.get("max_batches") is None
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        config, data = payload["train_config"], metadata(dump)
        if Path(dump).resolve() == Path(config["dump_path"]).resolve():
            split = "training"
            assert options["allow_training_dump"] is True
        else:
            split = "validation" if "validation" in Path(dump).name else "test"
            assert not options.get("allow_training_dump", False)
        calls[split].append((str(checkpoint), str(dump)))
        prior = payload["prior_config"]
        condition = {"uniform": 0, "gaussian": 1, "mixed": 2}[
            prior["graph_u_pool_sampling"]
        ]
        # Two tied winners in every setting; the lower LR must be selected.
        best_index = (config["seed"] - 42 + condition) % 9
        score = (
            0.9
            if pool.LEARNING_RATES.index(config["learning_rate"])
            in (best_index, best_index + 1)
            else 0.6
        )
        if split != "validation":
            score = (
                0.7
                + 0.001 * config["seed"]
                + 0.01 * condition
                + 0.02 * (data["prior_config"]["graph_u_mode"] == "covariate")
            )
        count = data["steps"] * 32
        return {
            "format": "nano_graph_u_evaluation_v1",
            "checkpoint": str(Path(checkpoint).resolve()),
            "checkpoint_sha256": _sha256_file(Path(checkpoint)),
            "checkpoint_step": 2500,
            "dump": str(Path(dump).resolve()),
            "dump_sha256": _sha256_file(Path(dump)),
            "dump_metadata": data,
            "task_count": count,
            "query_row_count": count * 75,
            "metrics": {name: score for name in pool.METRIC_NAMES},
            "metric_task_counts": {name: count for name in pool.METRIC_NAMES},
        }

    monkeypatch.setattr(pool, "train_model", train)
    monkeypatch.setattr(pool, "evaluate_checkpoint", evaluate)
    return calls


def test_bank_and_candidate_design_has_no_path_or_seed_aliases(tmp_path):
    root = tmp_path / "experiment"
    specs = [pool.bank_spec(root, task) for task in range(1, 67)]
    assert len({spec["dump"] for spec in specs}) == 66
    for spec in specs:
        assert spec["steps"] == (2500 if spec["split"] == "train" else 100)
        assert spec["seed"] == (
            3001
            if spec["split"] == "validation"
            else (1000 if spec["split"] == "train" else 2000) + spec["repetition"]
        )
        prior = pool.prior_config(spec["condition"], spec["mode"])
        assert prior.graph_u_intervention_mode == "support_pool"
        assert prior.graph_u_pool_sampling == {"shift": "gaussian"}.get(
            spec["condition"], spec["condition"]
        )
        assert (
            prior.graph_u_pool_peak_percentile,
            prior.graph_u_pool_std_fraction,
        ) == (75, 0.5)
        assert (prior.graph_u_query_location, prior.graph_u_query_scale) == (0, 1)
        assert not prior.graph_u_force_gaussian
    trials = [pool.trial_spec(root, task) for task in range(1, 301)]
    assert len({spec["result_path"] for spec in trials}) == 300
    assert len({spec["train_dump"] for spec in trials}) == 30
    assert len({spec["validation_dump"] for spec in trials}) == 6
    assert [spec["learning_rate"] for spec in trials] == list(pool.LEARNING_RATES) * 30
    assert all(spec["model_seed"] == 41 + spec["repetition"] for spec in trials)
    assert not {spec["dump"] for spec in specs if spec["split"] == "test"}.intersection(
        {spec["train_dump"] for spec in trials}
        | {spec["validation_dump"] for spec in trials}
    )
    assert not root.exists()


def test_generation_skips_complete_banks_and_resumes_committed_prefix(
    tmp_path, monkeypatch
):
    h5py = pytest.importorskip("h5py")
    root, calls = tmp_path / "generated", []

    def generate(path, **options):
        spec = pool.bank_spec(root, 1)
        assert str(path) == spec["dump"]
        assert options["workers"] == 2 and options["n_jobs"] == 1
        calls.append(options)
        _sparse_bank(spec)

    monkeypatch.setattr(pool, "generate_dump", generate)
    pool.generate_bank(root, 1, workers=2)
    pool.generate_bank(root, 1, workers=2)
    assert len(calls) == 1 and calls[0]["resume"] is False
    with h5py.File(pool.bank_spec(root, 1)["dump"], "r+") as stream:
        stream.attrs["committed_steps"] = 2499
    pool.generate_bank(root, 1, workers=2)
    assert len(calls) == 2 and calls[1]["resume"] is True


def test_selection_scoring_and_summary_keep_repetitions_and_banks_separate(
    banks, compute, monkeypatch
):
    for task_id in range(1, 301):
        pool.run_trial(banks, task_id, device="cpu")
    assert len(compute["train"]) == len(compute["validation"]) == 300
    assert not compute["test"] and not compute["training"]
    # Selection must remain possible when every test bank is unavailable.
    for spec in [pool.bank_spec(banks, task) for task in range(1, 67)]:
        if spec["split"] == "test":
            Path(spec["dump"]).rename(Path(spec["dump"]).with_suffix(".hidden"))
    winners = pool.select_winners(banks)
    for item in winners["winners"]:
        condition = pool.CONDITIONS.index(item["condition"])
        assert (
            item["learning_rate"]
            == pool.LEARNING_RATES[(item["repetition"] - 1 + condition) % 9]
        )
    assert len(winners["winners"]) == 30 and not compute["test"]
    for path in banks.rglob("*.hidden"):
        path.rename(path.with_suffix(".h5"))
    for task_id in range(1, 31):
        result = pool.evaluate_winner(banks, task_id, device="cpu")
        assert len(result["test_rows"]) == 6
        assert {row["repetition"] for row in result["test_rows"]} == {
            result["spec"]["repetition"]
        }
        assert result["training_row"]["task_count"] == 80000
    assert len(compute["test"]) == 180 and len(compute["training"]) == 30
    summary = pool.summarize_results(banks)
    assert len(summary["cells"]) == 36 and len(summary["training_cells"]) == 6
    raw = pd.read_csv(banks / "results/per_repetition.csv")
    cells = pd.read_csv(banks / "results/summary.csv")
    assert len(raw) == 180 and raw["task_count"].eq(3200).all()
    keys = list(pool._TEST_KEYS)
    expected = raw.groupby(keys)[list(pool.METRIC_NAMES)].agg(["mean", "std"])
    expected.columns = [f"{name}_{stat}" for name, stat in expected.columns]
    pd.testing.assert_frame_equal(
        cells.set_index(keys).sort_index()[expected.columns],
        expected,
        check_exact=False,
        atol=1e-12,
        rtol=0,
    )
    assert (
        pd.read_csv(banks / "results/training_per_repetition.csv")["task_count"]
        .eq(80000)
        .all()
    )
    assert len(pd.read_csv(banks / "results/winners.csv")) == 30
    counts = {key: len(values) for key, values in compute.items()}
    pool.run_trial(banks, 1, device="cpu")
    pool.select_winners(banks)
    pool.evaluate_winner(banks, 1, device="cpu")
    pool.summarize_results(banks)
    assert counts == {key: len(values) for key, values in compute.items()}


@pytest.mark.parametrize("split,bank_id", [("train", 1), ("validation", 31)])
def test_partial_banks_cannot_start_training(banks, compute, split, bank_id):
    h5py = pytest.importorskip("h5py")
    spec = pool.bank_spec(banks, bank_id)
    with h5py.File(spec["dump"], "r+") as stream:
        stream.attrs["committed_steps"] = spec["steps"] - 1
    with pytest.raises(ValueError, match="[Ii]ncomplete"):
        pool.run_trial(banks, 1, device="cpu")
    assert not compute["train"]


@pytest.mark.parametrize("corruption", ["step", "metric_count", "dump"])
def test_stale_validation_is_rejected_without_retraining(banks, compute, corruption):
    pool.run_trial(banks, 1, device="cpu")
    path = Path(pool.trial_spec(banks, 1)["result_path"])
    saved = json.loads(path.read_text())
    report = saved["validation_report"]
    if corruption == "step":
        report["checkpoint_step"] = 2499
    elif corruption == "metric_count":
        report["metric_task_counts"]["roc_auc"] = 3199
    else:
        report["dump"] = pool.bank_spec(banks, 4)["dump"]
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        pool.run_trial(banks, 1, device="cpu")
    assert len(compute["train"]) == len(compute["validation"]) == 1


def test_missing_candidates_and_winners_fail_before_output(tmp_path):
    root = tmp_path / "incomplete"
    with pytest.raises(FileNotFoundError, match="300 validation results missing"):
        pool.select_winners(root)
    assert not (root / "winners/winners.json").exists()
    with pytest.raises(FileNotFoundError, match="30 winner evaluations missing"):
        pool.summarize_results(root)
    assert not (root / "results/summary.csv").exists()


def test_recipe_is_pinned_and_other_experiments_are_not_adopted(tmp_path):
    root = pool._initialize(tmp_path / "pinned")
    path = root / "experiment.json"
    recipe = json.loads(path.read_text())
    recipe["rows"] = 1000
    path.write_text(json.dumps(recipe))
    with pytest.raises((FileExistsError, ValueError)):
        pool._initialize(root)
    assert json.loads(path.read_text())["rows"] == 1000
    other = tmp_path / "other"
    other.mkdir()
    (other / "train.h5").touch()
    with pytest.raises(ValueError, match="non-empty"):
        pool._initialize(other)


@pytest.mark.parametrize(
    "function,count",
    [(pool.bank_spec, 66), (pool.trial_spec, 300), (pool.winner_spec, 30)],
)
@pytest.mark.parametrize("invalid", [0, -1, True, "1", 301])
def test_invalid_array_ids_do_not_create_directories(
    tmp_path, function, count, invalid
):
    root = tmp_path / "bad"
    with pytest.raises(ValueError):
        function(root, invalid)
    assert not root.exists()


def test_aggregation_uses_sample_sd_and_requires_each_repeat():
    rows = [
        {
            "condition": "shift",
            "repetition": repetition,
            **{name: 0.1 * repetition for name in pool.METRIC_NAMES},
        }
        for repetition in range(1, 6)
    ]
    cell = pool._aggregate(rows, ("condition",))[0]
    assert cell["roc_auc_std"] == pytest.approx(
        np.std([0.1, 0.2, 0.3, 0.4, 0.5], ddof=1)
    )
    with pytest.raises(ValueError, match="five repetitions"):
        pool._aggregate(rows[:-1], ("condition",))


def test_repair_cli_only_dispatches_metadata_repair(tmp_path, monkeypatch):
    from tabicl.nano_graph_u import pool_repair

    calls = []
    monkeypatch.setattr(pool_repair, "repair_banks", lambda root: calls.append(root))
    root = tmp_path / "existing pool experiment"
    pool.main(["repair", "--root", str(root)])
    assert calls == [root]
    assert not root.exists()
