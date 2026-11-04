"""Fixed-pair experiments reuse the original IID assets and train only shifts."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from tabicl.nano_graph_u import fixed_experiments as fixed
from tabicl.nano_graph_u import test_winners as historical
from tabicl.nano_graph_u import tune
from tabicl.nano_graph_u._settings import experiment_prior_config
from tabicl.nano_graph_u.data import _metadata
from tabicl.nano_graph_u.select import WINNERS_FORMAT
from tabicl.nano_graph_u.train import PAPER_MODEL_CONFIG, _sha256_file
from tabicl.nano_graph_u.tune import LEARNING_RATES, trial_spec

MODES = ("confounding", "covariate")


def test_four_fixed_pairs_have_separate_directories_and_explicit_profiles(tmp_path):
    assert fixed.FIXED_SHIFTS == ((4.0, 1.5), (20.0, 2.0), (60.0, 3.0), (100.0, 4.0))
    directories = []
    for task, pair in enumerate(fixed.FIXED_SHIFTS, start=1):
        spec = fixed.experiment_spec(tmp_path, task)
        assert (spec["location"], spec["scale"]) == pair
        assert spec["fixed_shift_config"] == {
            "graph_u_query_location": pair[0], "graph_u_query_scale": pair[1],
        }
        directories.append(Path(spec["directory"]))
    assert [directory.name for directory in directories] == [
        "loc_4_scale_1p5", "loc_20_scale_2", "loc_60_scale_3", "loc_100_scale_4",
    ]
    assert len(set(directories)) == 4
    assert all(directory.parent == tmp_path for directory in directories)


@pytest.mark.parametrize("task", [0, 5, True, "1"])
def test_fixed_pair_mapping_rejects_invalid_tasks(tmp_path, task):
    with pytest.raises(ValueError, match="1 to 4"):
        fixed.experiment_spec(tmp_path, task)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _bank_metadata(condition, split, repetition, mode, *, fixed_shift_config=None):
    seed = {"train": 1000 + repetition, "test": 2000 + repetition,
            "validation": 3001}[split]
    metadata = _metadata(
        experiment_prior_config(condition, mode, fixed_shift_config=fixed_shift_config),
        steps=2500 if split == "train" else 100,
        batch_size=32,
        seed=seed,
        rows=150,
        features=5,
        min_train_size=0.3,
        max_train_size=0.9,
        n_jobs=1,
    )
    # Historical assets deliberately use a fingerprint distinct from the code
    # currently producing the additional shifted banks.
    metadata["generator_source_sha256"] = "a" * 64
    return metadata


def _sparse_bank(path, metadata):
    h5py = pytest.importorskip("h5py")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = metadata["steps"] * metadata["batch_size"]
    with h5py.File(path, "w") as stream:
        stream.attrs["metadata_json"] = json.dumps(metadata)
        stream.attrs["committed_steps"] = metadata["steps"]
        stream.create_dataset("x", (count, 150, 5), dtype="f4")
        stream.create_dataset("y", (count, 150), dtype="i8")
        stream.create_dataset("train_sizes", (count,), dtype="i4")


def _report(checkpoint, dump, *, score=0.8):
    h5py = pytest.importorskip("h5py")
    with h5py.File(dump, "r") as stream:
        metadata = json.loads(stream.attrs["metadata_json"])
    return {
        "format": "nano_graph_u_evaluation_v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256_file(Path(checkpoint)),
        "checkpoint_step": 2500,
        "dump": str(dump),
        "dump_sha256": _sha256_file(Path(dump)),
        "dump_metadata": metadata,
        "task_count": 3200,
        "query_row_count": 240000,
        "metrics": {name: score for name in historical.METRIC_NAMES},
        "metric_task_counts": {name: 3200 for name in historical.METRIC_NAMES},
    }


@pytest.fixture
def original_iid(tmp_path):
    """Make all original IID tests, without creating any old shifted bank."""
    cluster = tmp_path / "cluster"
    roots = {}
    for mode in MODES:
        suffix = "" if mode == "confounding" else "_covariate"
        data = cluster / "tabicl-conf-shift-priors/nano_graph_u" / f"repetitions{suffix}"
        search = cluster / "tabicl-conf-shift-checkpoints/nano_graph_u" / f"lr_search{suffix}"
        roots[mode] = (data, search)
        winners = []
        for repetition in range(1, 6):
            for condition, offset in (("identity", 0), ("shift", 10)):
                candidate = trial_spec(
                    (repetition - 1) * 20 + offset + 1,
                    data,
                    search,
                    graph_u_mode=mode,
                )
                checkpoint = search / "winners" / f"rep_{repetition}" / condition / "model.pt"
                winner = {
                    "repetition": repetition,
                    "condition": condition,
                    "task_id": candidate["task_id"],
                    "learning_rate": candidate["learning_rate"],
                    "model_seed": 41 + repetition,
                    "validation_roc_auc": 0.9,
                    "checkpoint_step": 2500,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": "b" * 64,
                    "source_checkpoint": str(Path(candidate["checkpoint_dir"]) / "latest.pt"),
                    "validation_dump": candidate["validation_dump"],
                    "validation_dump_sha256": "c" * 64,
                    "validation_result": candidate["result_path"],
                }
                if condition == "identity":
                    metadata = _bank_metadata(condition, "train", repetition, mode)
                    # Exercise compatibility with the original confounding dump.
                    if mode == "confounding":
                        metadata["prior_config"].pop("graph_u_mode")
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "format": "nano_graph_u_v1",
                        "step": 2500,
                        "model_config": PAPER_MODEL_CONFIG,
                        "model": {"fixture": torch.tensor(0)},
                        "dump_metadata": metadata,
                        "prior_config": metadata["prior_config"],
                        "train_config": {
                            "seed": 41 + repetition,
                            "batch_size": 32,
                            "learning_rate": candidate["learning_rate"],
                            "optimizer": "schedulefree.AdamWScheduleFree",
                            "weight_decay": 0.0,
                            "gradient_clip_norm": 1.0,
                            "loss": "query_cross_entropy",
                            "precision": "float32",
                            "dump_path": candidate["train_dump"],
                            "dump_sha256": "d" * 64,
                        },
                    }, checkpoint)
                    winner["checkpoint_sha256"] = _sha256_file(checkpoint)
                winners.append(winner)
            dump = data / f"rep_{repetition}" / f"identity_test_seed{2000 + repetition}_100steps_b32_r150_f5.h5"
            metadata = _bank_metadata("identity", "test", repetition, mode)
            if mode == "confounding":
                metadata["prior_config"].pop("graph_u_mode")
            _sparse_bank(dump, metadata)
        manifest = {
            "format": WINNERS_FORMAT,
            "data_root": str(data),
            "output_root": str(search),
            "selection_metric": "roc_auc",
            "validation_policy": "matching_training_condition",
            "training_steps": 2500,
            "trial_count": 100,
            "generator_source_sha256": "a" * 64,
            "learning_rates": list(LEARNING_RATES),
            "winners": winners,
        }
        if mode != "confounding":
            manifest["graph_u_mode"] = mode
        _write_json(search / "winners/winners.json", manifest)
    results = []
    for training_mode in MODES:
        training_data, search = roots[training_mode]
        manifest_path = search / "winners/winners.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for test_mode in MODES:
            suffix = "" if training_mode == "confounding" else "_covariate"
            if training_mode != test_mode:
                suffix = f"_{training_mode}_on_{test_mode}"
            output = cluster / "tabicl-conf-shift-evaluations/nano_graph_u" / f"lr_search_tests{suffix}"
            for repetition in range(1, 6):
                spec = historical.test_spec(
                    (repetition - 1) * 4 + 1,
                    roots[test_mode][0],
                    search,
                    output,
                    graph_u_mode=training_mode,
                    test_graph_u_mode=test_mode,
                    training_data_root=training_data,
                )
                winner = next(w for w in manifest["winners"] if w["checkpoint"] == spec["checkpoint"])
                result = {
                    "format": historical.TEST_FORMAT,
                    "test": spec,
                    "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                    "winner": winner,
                    "evaluation_report": _report(spec["checkpoint"], spec["test_dump"], score=0.5 + repetition / 100),
                }
                if training_mode != test_mode:
                    result["training_generator_source_sha256"] = "a" * 64
                    result["test_generator_source_sha256"] = "a" * 64
                _write_json(spec["result_path"], result)
                results.append(result)
    return cluster, roots, results


def test_original_iid_loader_reads_only_frozen_iid_models_banks_and_results(original_iid):
    cluster, roots, reports = original_iid
    before = {str(path): _sha256_file(path) for path in cluster.rglob("*") if path.is_file()}
    models, banks, rows, provenance = fixed.load_original_iid(cluster)
    assert set(models) == {(mode, rep, "identity") for mode in MODES for rep in range(1, 6)}
    assert set(banks) == {(mode, rep) for mode in MODES for rep in range(1, 6)}
    assert len(rows) == len(reports) == 20
    assert provenance
    assert before == {str(path): _sha256_file(path) for path in cluster.rglob("*") if path.is_file()}
    assert all(not list(data.rglob("shift*.h5")) for data, _ in roots.values())
    assert all(not list(search.glob("winners/rep_*/shift/model.pt")) for _, search in roots.values())


@pytest.mark.parametrize("corruption", ["checkpoint", "bank_profile", "report"])
def test_original_iid_loader_rejects_corrupt_or_incomplete_sources(original_iid, corruption):
    h5py = pytest.importorskip("h5py")
    cluster, _, reports = original_iid
    report = reports[0]
    if corruption == "checkpoint":
        Path(report["test"]["checkpoint"]).write_bytes(b"changed after selection")
    elif corruption == "bank_profile":
        with h5py.File(report["test"]["test_dump"], "r+") as stream:
            metadata = json.loads(stream.attrs["metadata_json"])
            metadata["prior_config"]["graph_u_query_location"] = 4.0
            stream.attrs["metadata_json"] = json.dumps(metadata)
    else:
        report = copy.deepcopy(report)
        report["evaluation_report"]["metric_task_counts"]["roc_auc"] = 3199
        _write_json(report["test"]["result_path"], report)
    with pytest.raises(ValueError):
        fixed.load_original_iid(cluster)


@pytest.fixture
def compute(monkeypatch):
    """Replace expensive numerical work, retaining all real artifact checks."""
    h5py = pytest.importorskip("h5py")
    calls = {"generate": [], "train": [], "validation": [], "test": []}

    def generate(dump, **options):
        calls["generate"].append((Path(dump), options))
        assert options["batch_size"] == 32
        assert options["rows"] == 150
        assert options["features"] == 5
        assert options["min_train_size"] == 0.3
        assert options["max_train_size"] == 0.9
        assert options["n_jobs"] == 1
        prior = options["prior_config"]
        assert prior.graph_u_shift_mode == "fixed"
        assert prior.graph_u_query_location in {4.0, 20.0, 60.0, 100.0}
        assert prior.graph_u_max_attempts == 1000
        assert prior.filter_unpredictable_graphs
        assert prior.filter_unpredictable_datasets
        assert not prior.allow_act_warping
        metadata = _metadata(
            prior,
            **{key: options[key] for key in (
                "steps", "batch_size", "seed", "rows", "features",
                "min_train_size", "max_train_size", "n_jobs",
            )},
        )
        _sparse_bank(dump, metadata)
        return Path(dump)

    def train(dump, checkpoint_dir, **options):
        calls["train"].append((Path(dump), Path(checkpoint_dir), options))
        assert options["steps"] == 2500
        assert options["batch_size"] == 32
        assert options["device"] == "cpu"
        assert options["save_every"] == 250
        assert options["log_every"] == 25
        assert Path(dump).name.startswith("shift_train_")
        with h5py.File(dump, "r") as stream:
            metadata = json.loads(stream.attrs["metadata_json"])
        assert options["seed"] == metadata["seed"] - 959
        config = {
            "seed": options["seed"],
            "batch_size": 32,
            "learning_rate": options["learning_rate"],
            "optimizer": "schedulefree.AdamWScheduleFree",
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "loss": "query_cross_entropy",
            "precision": "float32",
            "dump_path": str(Path(dump)),
            "dump_sha256": _sha256_file(Path(dump)),
        }
        destination = Path(checkpoint_dir) / "latest.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format": "nano_graph_u_v1",
            "step": 2500,
            "model_config": PAPER_MODEL_CONFIG,
            "model": {"fixture": torch.tensor(0)},
            "dump_metadata": metadata,
            "prior_config": metadata["prior_config"],
            "train_config": config,
        }, destination)
        return destination

    def evaluate(checkpoint, dump, **options):
        assert options == {"batch_size": 32, "device": "cpu"}
        split = "validation" if "validation" in Path(dump).name else "test"
        calls[split].append((Path(checkpoint), Path(dump)))
        if split == "validation":
            config = torch.load(checkpoint, map_location="cpu", weights_only=True)["train_config"]
            # The best AUROC is tied at two rates. NLL and iteration order must
            # not choose the larger learning rate.
            score = 0.9 if config["learning_rate"] in LEARNING_RATES[2:4] else 0.7
        else:
            with h5py.File(dump, "r") as stream:
                seed = json.loads(stream.attrs["metadata_json"])["seed"]
            score = 0.6 + (seed - 2000) / 100
        return _report(checkpoint, dump, score=score)

    monkeypatch.setattr(fixed, "generate_dump", generate)
    monkeypatch.setattr(tune, "train_model", train)
    monkeypatch.setattr(tune, "evaluate_checkpoint", evaluate)
    monkeypatch.setattr(fixed, "evaluate_checkpoint", evaluate)
    return calls


def test_generation_creates_only_22_shift_banks_and_resumes_one_partial_bank(
    original_iid, tmp_path, compute,
):
    h5py = pytest.importorskip("h5py")
    cluster, _, _ = original_iid
    root = tmp_path / "fixed"
    fixed.generate_experiment(cluster, root, 1, workers=3)
    assert len(compute["generate"]) == 22
    directory = Path(fixed.experiment_spec(root, 1)["directory"])
    for mode in MODES:
        group = [(path, options) for path, options in compute["generate"]
                 if options["prior_config"].graph_u_mode == mode]
        assert len(group) == 11
        assert sorted(options["seed"] for _, options in group if options["steps"] == 2500) == list(range(1001, 1006))
        assert sorted(options["seed"] for _, options in group if options["steps"] == 100) == [2001, 2002, 2003, 2004, 2005, 3001]
        assert all(options["workers"] == 3 for _, options in group)
        assert all(not options["resume"] for _, options in group)
        assert all(options["prior_config"].graph_u_query_scale == 1.5 for _, options in group)
        assert all(path.is_relative_to(directory) and path.name.startswith("shift_") for path, _ in group)
    assert len(list(directory.rglob("*.h5"))) == 22
    fixed.generate_experiment(cluster, root, 1, workers=3)
    assert len(compute["generate"]) == 22
    partial = next(path for path, opts in compute["generate"] if opts["steps"] == 2500)
    with h5py.File(partial, "r+") as stream:
        stream.attrs["committed_steps"] = 2499
    fixed.generate_experiment(cluster, root, 1, workers=2)
    assert len(compute["generate"]) == 23
    assert compute["generate"][-1][0] == partial
    assert compute["generate"][-1][1]["resume"] is True
    assert compute["generate"][-1][1]["workers"] == 2
    assert not compute["train"]
    assert not compute["validation"]
    assert not compute["test"]


@pytest.mark.parametrize("field", ["profile", "generator"])
def test_generation_refuses_an_existing_bank_from_another_recipe(
    original_iid, tmp_path, compute, field,
):
    h5py = pytest.importorskip("h5py")
    cluster, _, _ = original_iid
    root = tmp_path / "fixed"
    fixed.generate_experiment(cluster, root, 1)
    path = compute["generate"][0][0]
    with h5py.File(path, "r+") as stream:
        metadata = json.loads(stream.attrs["metadata_json"])
        if field == "profile":
            metadata["prior_config"]["graph_u_query_location"] = 20.0
        else:
            metadata["generator_source_sha256"] = "f" * 64
        stream.attrs["metadata_json"] = json.dumps(metadata)
    with pytest.raises((ValueError, FileExistsError)):
        fixed.generate_experiment(cluster, root, 1)
    assert len(compute["generate"]) == 22


@pytest.mark.parametrize("corruption", ["missing", "partial"])
def test_run_requires_every_generated_bank_before_any_training(
    original_iid, tmp_path, compute, monkeypatch, corruption,
):
    h5py = pytest.importorskip("h5py")
    cluster, _, _ = original_iid
    root = tmp_path / "fixed"
    fixed.generate_experiment(cluster, root, 1)
    path = compute["generate"][-1][0]
    if corruption == "missing":
        path.unlink()
    else:
        with h5py.File(path, "r+") as stream:
            stream.attrs["committed_steps"] = 99
    monkeypatch.setattr(fixed, "_generate_banks", lambda *args, **kwargs: pytest.fail("run must not generate banks"))
    with pytest.raises((ValueError, FileNotFoundError)):
        fixed.run_experiment(cluster, root, 1, device="cpu")
    assert len(compute["generate"]) == 22
    assert not compute["train"]
    assert not compute["validation"]
    assert not compute["test"]


def test_full_design_resumes_selects_by_auc_and_collects_all_four_pairs(
    original_iid, tmp_path, compute, monkeypatch,
):
    cluster, _, source_results = original_iid
    source_hashes = {path: _sha256_file(path) for path in cluster.rglob("*") if path.is_file()}
    root = tmp_path / "fixed"
    fixed.generate_experiment(cluster, root, 1)
    directory = Path(fixed.experiment_spec(root, 1)["directory"])
    evaluate = fixed.evaluate_checkpoint

    def interrupted_evaluation(*args, **kwargs):
        if len(compute["test"]) == 3:
            raise RuntimeError("simulated interruption")
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(fixed, "evaluate_checkpoint", interrupted_evaluation)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        fixed.run_experiment(cluster, root, 1, device="cpu")
    assert len(compute["train"]) == len(compute["validation"]) == 100
    assert len(compute["test"]) == 3
    assert len(list((directory / "results").rglob("*_on_*.json"))) == 3
    monkeypatch.setattr(fixed, "evaluate_checkpoint", evaluate)
    result = fixed.run_experiment(cluster, root, 1, device="cpu")
    assert len(compute["train"]) == len(compute["validation"]) == 100
    assert len(compute["test"]) == 60
    assert len(compute["generate"]) == 22

    for mode in MODES:
        training = [(path, options) for path, _, options in compute["train"]
                    if path.parent.parent.name == mode]
        assert len(training) == 50
        for repetition in range(1, 6):
            group = [(path, options) for path, options in training
                     if path.parent.name == f"rep_{repetition}"]
            assert len(group) == 10
            assert {options["learning_rate"] for _, options in group} == set(LEARNING_RATES)
            assert {options["seed"] for _, options in group} == {41 + repetition}
            assert len({path for path, _ in group}) == 1
        trials = [json.loads(path.read_text(encoding="utf-8"))["trial"]
                  for path in (directory / "checkpoints" / mode).rglob("validation.json")]
        assert {trial["task_id"] for trial in trials} == {
            (rep - 1) * 20 + 11 + index for rep in range(1, 6) for index in range(10)
        }
        assert {trial["condition"] for trial in trials} == {"shift"}
        assert all(trial["fixed_shift_config"] == result["recipe"]["fixed_shift_config"] for trial in trials)
        manifest = json.loads((directory / "checkpoints" / mode / "winners/winners.json").read_text(encoding="utf-8"))
        assert manifest["trial_count"] == 50
        assert manifest["selection_metric"] == "roc_auc"
        assert manifest["tie_break"] == "lowest_learning_rate"
        assert len(manifest["winners"]) == 5
        assert {winner["learning_rate"] for winner in manifest["winners"]} == {LEARNING_RATES[2]}
        assert {winner["model_seed"] for winner in manifest["winners"]} == set(range(42, 47))
        assert all(Path(winner["checkpoint"]).is_file() for winner in manifest["winners"])
    assert len({dump for _, dump in compute["validation"]}) == 2
    assert len({path for path, _, _ in compute["train"]}) == 10

    assert result["new_evaluation_count"] == 60
    assert result["reused_iid_count"] == 20
    assert result["repetition_count"] == 5
    assert result["standard_deviation_ddof"] == 1
    assert len(result["cells"]) == 16
    assert len(result["per_repetition"]) == 80
    reused = [row for row in result["per_repetition"] if row["result_origin"] == "reused_iid"]
    assert len(reused) == 20
    assert {row["result_path"] for row in reused} == {report["test"]["result_path"] for report in source_results}
    assert all(row["training_condition"] == row["test_condition"] == "identity" for row in reused)
    for row in result["per_repetition"]:
        assert row["training_query_location"] == (0.0 if row["training_condition"] == "identity" else 4.0)
        assert row["training_query_scale"] == (1.0 if row["training_condition"] == "identity" else 1.5)
        assert row["test_query_location"] == (0.0 if row["test_condition"] == "identity" else 4.0)
        assert row["test_query_scale"] == (1.0 if row["test_condition"] == "identity" else 1.5)
        assert row["test_data_seed"] == 2000 + row["repetition"]
        assert row["task_count"] == 3200
    for cell in result["cells"]:
        expected = 0.53 if cell["training_condition"] == cell["test_condition"] == "identity" else 0.63
        assert cell["repetition_count"] == 5
        assert cell["roc_auc_mean"] == pytest.approx(expected)
        assert cell["roc_auc_std"] == pytest.approx(0.0158113883008419)
    assert not (root / "results/summary.json").exists()
    assert fixed.run_experiment(cluster, root, 1, device="cpu") == result
    assert (len(compute["train"]), len(compute["validation"]), len(compute["test"])) == (100, 100, 60)

    for task, pair in enumerate(fixed.FIXED_SHIFTS[1:], start=2):
        fixed.generate_experiment(cluster, root, task)
        completed = fixed.run_experiment(cluster, root, task, device="cpu")
        assert len(completed["cells"]) == 16
        assert len(completed["per_repetition"]) == 80
        assert {(row["fixed_location"], row["fixed_scale"]) for row in completed["per_repetition"]} == {pair}
        assert {row["training_query_location"] for row in completed["per_repetition"]} == {0.0, pair[0]}
        assert {row["test_query_scale"] for row in completed["per_repetition"]} == {1.0, pair[1]}
    combined = json.loads((root / "results/summary.json").read_text(encoding="utf-8"))
    assert len(combined["cells"]) == 64
    assert len(combined["per_repetition"]) == 320
    assert len(compute["generate"]) == 88
    assert (len(compute["train"]), len(compute["validation"]), len(compute["test"])) == (400, 400, 240)
    assert source_hashes == {path: _sha256_file(path) for path in cluster.rglob("*") if path.is_file()}
    assert len((root / "results/summary.csv").read_text(encoding="utf-8").splitlines()) == 65
    assert len((root / "results/per_repetition.csv").read_text(encoding="utf-8").splitlines()) == 321

    # Even a valid source report gaining an unrelated field changes provenance;
    # a resumed experiment must reject it before reusing any completed work.
    changed = copy.deepcopy(source_results[0])
    changed["annotation"] = "source changed after this experiment"
    _write_json(changed["test"]["result_path"], changed)
    with pytest.raises(FileExistsError):
        fixed.run_experiment(cluster, root, 1, device="cpu")
    assert (len(compute["train"]), len(compute["validation"]), len(compute["test"])) == (400, 400, 240)


@pytest.fixture
def saved_fixed_summaries(tmp_path):
    """Save a complete design with identical IID controls in all four pairs."""
    root = tmp_path / "fixed"
    paths = []
    for task in range(1, 5):
        spec = fixed.experiment_spec(root, task)
        rows = []
        for training_mode in MODES:
            for training_condition in ("identity", "shift"):
                for test_mode in MODES:
                    for test_condition in ("identity", "shift"):
                        for repetition in range(1, 6):
                            model_source = "original" if training_condition == "identity" else f"pair_{task}"
                            bank_source = "original" if test_condition == "identity" else f"pair_{task}"
                            iid = training_condition == test_condition == "identity"
                            rows.append({
                                "fixed_location": spec["location"],
                                "fixed_scale": spec["scale"],
                                "training_graph_u_mode": training_mode,
                                "training_condition": training_condition,
                                "test_graph_u_mode": test_mode,
                                "test_condition": test_condition,
                                "repetition": repetition,
                                "checkpoint_sha256": hashlib.sha256(
                                    f"model:{model_source}:{training_mode}:{repetition}".encode()
                                ).hexdigest(),
                                "test_dump_sha256": hashlib.sha256(
                                    f"bank:{bank_source}:{test_mode}:{repetition}".encode()
                                ).hexdigest(),
                                **{metric: 0.5 + repetition / 100 + (0 if iid else task / 100)
                                   for metric in historical.METRIC_NAMES},
                            })
        result = {
            "format": "nano_graph_u_fixed_experiment_summary_v1",
            "recipe": {"fixed_shift_config": spec["fixed_shift_config"]},
            "new_evaluation_count": 60,
            "reused_iid_count": 20,
            "repetition_count": 5,
            "standard_deviation_ddof": 1,
            "cells": fixed._summarize(rows),
            "per_repetition": rows,
        }
        fixed._save_results(spec["directory"], result)
        paths.append(spec["directory"] / "results/summary.json")
    return root, paths


def test_aggregation_accepts_complete_pairs_with_matching_original_iid_controls(saved_fixed_summaries):
    root, _ = saved_fixed_summaries
    fixed._collect_if_complete(root)
    result = json.loads((root / "results/summary.json").read_text(encoding="utf-8"))
    assert len(result["cells"]) == 64
    assert len(result["per_repetition"]) == 320


@pytest.mark.parametrize("field", [
    "fixed_location", "fixed_scale", "checkpoint_sha256", "test_dump_sha256",
    *historical.METRIC_NAMES,
])
def test_aggregation_rejects_wrong_pair_labels_or_changed_original_iid_controls(
    saved_fixed_summaries, field,
):
    root, paths = saved_fixed_summaries
    result = json.loads(paths[-1].read_text(encoding="utf-8"))
    for row in result["per_repetition"]:
        if field in {"fixed_location", "fixed_scale"}:
            row[field] += 1.0
        elif field == "checkpoint_sha256":
            # Change every use of one IID model in this pair, so the pair is
            # internally consistent while disagreeing with the other pairs.
            if (row["training_graph_u_mode"], row["training_condition"], row["repetition"]) == ("confounding", "identity", 1):
                row[field] = "f" * 64
        elif field == "test_dump_sha256":
            if (row["test_graph_u_mode"], row["test_condition"], row["repetition"]) == ("covariate", "identity", 3):
                row[field] = "f" * 64
        elif (
            row["training_graph_u_mode"], row["training_condition"],
            row["test_graph_u_mode"], row["test_condition"], row["repetition"],
        ) == ("confounding", "identity", "covariate", "identity", 3):
            row[field] += 0.1
    # Recompute the statistics so rejection must come from provenance and
    # labels, independently of the per-pair arithmetic integrity check.
    result["cells"] = fixed._summarize(result["per_repetition"])
    _write_json(paths[-1], result)
    message = "labels" if field in {"fixed_location", "fixed_scale"} else "IID controls"
    with pytest.raises(ValueError, match=message):
        fixed._collect_if_complete(root)
    assert not (root / "results/summary.json").exists()
