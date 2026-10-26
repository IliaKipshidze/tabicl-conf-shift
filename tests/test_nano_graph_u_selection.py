"""Winner retention and selection use validation reports, never test banks."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from tabicl.nano_graph_u import select as selection
from tabicl.nano_graph_u.data import _metadata
from tabicl.nano_graph_u.train import PAPER_MODEL_CONFIG, _sha256_file
from tabicl.nano_graph_u.tune import TRIAL_FORMAT, trial_spec
from tabicl.prior.graph_lib._config import PriorConfig


@pytest.fixture
def make_search(tmp_path):
    h5py = pytest.importorskip("h5py")

    def create(*, separate_fingerprints=False, graph_u_mode="confounding"):
        data_root, output_root = tmp_path / "banks", tmp_path / "search"
        metadata_by_path = {}
        specs = [
            trial_spec(task, data_root, output_root, graph_u_mode=graph_u_mode)
            for task in range(1, 101)
        ]
        for spec in specs:
            condition = spec["condition"]
            prior = PriorConfig(
                graph_u_enabled=True,
                graph_u_mode=graph_u_mode,
                graph_u_structure_mode="add_root",
                graph_u_query_location=0.0 if condition == "identity" else 2.0,
                graph_u_query_scale=1.0 if condition == "identity" else 1.5,
                graph_u_max_attempts=1000,
                filter_unpredictable_graphs=True,
                filter_unpredictable_datasets=True,
                allow_act_warping=False,
            )
            for split, steps, seed, path in (
                ("train", 2500, 1000 + spec["repetition"], spec["train_dump"]),
                ("validation", 100, 3001, spec["validation_dump"]),
            ):
                if path in metadata_by_path:
                    continue
                metadata = _metadata(
                    prior,
                    steps=steps,
                    batch_size=32,
                    seed=seed,
                    rows=150,
                    features=5,
                    min_train_size=0.3,
                    max_train_size=0.9,
                    n_jobs=1,
                )
                metadata["generator_source_sha256"] = (
                    "b" * 64
                    if separate_fingerprints and condition == "shift"
                    else "a" * 64
                )
                metadata_by_path[path] = metadata
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                # Sparse fixtures validate provenance and shapes without training.
                with h5py.File(path, "w") as stream:
                    stream.attrs["metadata_json"] = json.dumps(metadata)
                    stream.attrs["committed_steps"] = steps
                    stream.create_dataset("x", shape=(steps * 32, 150, 5), dtype="f4")
                    stream.create_dataset("y", shape=(steps * 32, 150), dtype="i8")
                    stream.create_dataset(
                        "train_sizes", shape=(steps * 32,), dtype="i4"
                    )

        hashes = {path: _sha256_file(Path(path)) for path in metadata_by_path}
        for spec in specs:
            checkpoint = Path(spec["checkpoint_dir"]) / "latest.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            training = {
                "seed": spec["model_seed"],
                "batch_size": 32,
                "learning_rate": spec["learning_rate"],
                "optimizer": "schedulefree.AdamWScheduleFree",
                "weight_decay": 0.0,
                "gradient_clip_norm": 1.0,
                "loss": "query_cross_entropy",
                "precision": "float32",
                "dump_path": spec["train_dump"],
                "dump_sha256": hashes[spec["train_dump"]],
            }
            torch.save(
                {
                    "format": "nano_graph_u_v1",
                    "step": 2500,
                    "model_config": PAPER_MODEL_CONFIG,
                    "dump_metadata": metadata_by_path[spec["train_dump"]],
                    "prior_config": metadata_by_path[spec["train_dump"]][
                        "prior_config"
                    ],
                    "train_config": training,
                    "model": {"fixture": torch.tensor(spec["task_id"])},
                },
                checkpoint,
            )
            target = (
                spec["repetition"] - 1
                if spec["condition"] == "identity"
                else 10 - spec["repetition"]
            )
            score = (
                0.9 if spec["lr_index"] == target else 0.6 + spec["lr_index"] * 0.001
            )
            result = {
                "format": TRIAL_FORMAT,
                "trial": spec,
                "selection_metric": "roc_auc",
                "validation_report": {
                    "format": "nano_graph_u_evaluation_v1",
                    "checkpoint": str(checkpoint),
                    "checkpoint_step": 2500,
                    "checkpoint_sha256": _sha256_file(checkpoint),
                    "dump": spec["validation_dump"],
                    "dump_sha256": hashes[spec["validation_dump"]],
                    "dump_metadata": metadata_by_path[spec["validation_dump"]],
                    "task_count": 3200,
                    "metric_task_counts": {"roc_auc": 3200},
                    "metrics": {
                        "roc_auc": score,
                        "nll": 0.01 if spec["lr_index"] == 0 else 10.0,
                    },
                },
            }
            Path(spec["result_path"]).write_text(json.dumps(result), encoding="utf-8")
        return data_root, output_root, specs

    return create


def rewrite_report(spec, edit):
    path = Path(spec["result_path"])
    result = json.loads(path.read_text(encoding="utf-8"))
    edit(result)
    path.write_text(json.dumps(result), encoding="utf-8")


def test_selection_retains_ten_individual_winners_and_is_idempotent(make_search):
    data_root, output_root, specs = make_search()
    manifest = selection.select_winners(data_root, output_root)
    assert len(manifest["winners"]) == 10
    assert manifest["selection_metric"] == "roc_auc"
    assert manifest["generator_source_sha256"] == "a" * 64
    for winner in manifest["winners"]:
        index = (
            winner["repetition"] - 1
            if winner["condition"] == "identity"
            else 10 - winner["repetition"]
        )
        assert winner["learning_rate"] == selection.LEARNING_RATES[index]
        assert winner["model_seed"] == 41 + winner["repetition"]
        assert winner["validation_roc_auc"] == 0.9
        assert _sha256_file(Path(winner["checkpoint"])) == winner["checkpoint_sha256"]
        assert (
            Path(winner["checkpoint"]).read_bytes()
            == Path(winner["source_checkpoint"]).read_bytes()
        )
    with (output_root / "winners" / "summary.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        assert len(list(csv.DictReader(stream))) == 10
    assert all(Path(spec["result_path"]).is_file() for spec in specs)
    assert selection.select_winners(data_root, output_root) == manifest


def test_selection_requires_every_candidate_before_writing_winners(make_search):
    data_root, output_root, specs = make_search()
    Path(specs[50]["result_path"]).unlink()
    with pytest.raises(FileNotFoundError, match="1 of 100"):
        selection.select_winners(data_root, output_root)
    assert not (output_root / "winners").exists()


def test_selection_exact_auc_ties_choose_smaller_learning_rate(make_search):
    data_root, output_root, specs = make_search()
    for spec in specs[:2]:
        rewrite_report(
            spec,
            lambda result: result["validation_report"]["metrics"].update(roc_auc=0.99),
        )
    manifest = selection.select_winners(data_root, output_root)
    assert manifest["winners"][0]["task_id"] == 1


@pytest.mark.parametrize(
    "field,value", [("checkpoint_step", 2499), ("task_count", 3199)]
)
def test_selection_refuses_incomplete_validation_or_training(make_search, field, value):
    data_root, output_root, specs = make_search()
    rewrite_report(
        specs[0], lambda result: result["validation_report"].update({field: value})
    )
    with pytest.raises(ValueError, match="final checkpoint"):
        selection.select_winners(data_root, output_root)
    assert not (output_root / "winners").exists()


def test_selection_rejects_stale_checkpoint(make_search):
    data_root, output_root, specs = make_search()
    (Path(specs[0]["checkpoint_dir"]) / "latest.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Checkpoint changed"):
        selection.select_winners(data_root, output_root)
    assert not (output_root / "winners").exists()


def test_selection_requires_same_generator_in_both_conditions(make_search):
    data_root, output_root, _ = make_search(separate_fingerprints=True)
    with pytest.raises(ValueError, match="different prior code"):
        selection.select_winners(data_root, output_root)
    assert not (output_root / "winners").exists()


def test_selection_preserves_existing_different_winner_manifest(make_search):
    data_root, output_root, specs = make_search()
    original = selection.select_winners(data_root, output_root)
    path = output_root / "winners" / "winners.json"
    before = path.read_bytes()
    rewrite_report(
        specs[1],
        lambda result: result["validation_report"]["metrics"].update(roc_auc=1.0),
    )
    with pytest.raises(FileExistsError, match="different selection"):
        selection.select_winners(data_root, output_root)
    assert path.read_bytes() == before
    assert (
        _sha256_file(Path(original["winners"][0]["checkpoint"]))
        == original["winners"][0]["checkpoint_sha256"]
    )


def test_selection_recovers_after_interrupted_winner_retention(
    make_search, monkeypatch
):
    data_root, output_root, _ = make_search()
    retain = selection._retain_checkpoint
    count = 0

    def interrupt(source, destination, digest):
        nonlocal count
        count += 1
        if count == 3:
            raise RuntimeError("interrupted copy")
        retain(source, destination, digest)

    with monkeypatch.context() as patch:
        patch.setattr(selection, "_retain_checkpoint", interrupt)
        with pytest.raises(RuntimeError, match="interrupted copy"):
            selection.select_winners(data_root, output_root)
    assert not (output_root / "winners" / "winners.json").exists()
    manifest = selection.select_winners(data_root, output_root)
    assert len(manifest["winners"]) == 10


def test_covariate_selection_retains_mode_and_rejects_confounding_request(make_search):
    data, output, _ = make_search(graph_u_mode="covariate")
    manifest = selection.select_winners(data, output, graph_u_mode="covariate")
    assert manifest["graph_u_mode"] == "covariate"
    assert len(manifest["winners"]) == 10
    assert selection.select_winners(data, output, graph_u_mode="covariate") == manifest
    with pytest.raises(ValueError, match="different candidate"):
        selection.select_winners(data, output)
