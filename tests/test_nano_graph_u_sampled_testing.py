"""Frozen testing keeps sampled experiment settings and original fixed results separate."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from tabicl.nano_graph_u import test_winners as final_tests
from tabicl.nano_graph_u import summarize_tests as summaries
from tabicl.nano_graph_u._settings import experiment_prior_config
from tabicl.nano_graph_u.train import _sha256_file
from tabicl.prior.graph_lib._config import GRAPH_U_SAMPLING_DEFAULTS
from test_nano_graph_u_test_winners import make_inputs, _result
from test_nano_graph_u_test_summary import completed_tests, rewrite


def _profile():
    return {
        **GRAPH_U_SAMPLING_DEFAULTS,
        "graph_u_shift_mode": "sampled",
        "graph_u_query_scale_std": 0.08,
    }


def _sampled_inputs(make_inputs, task_id, mode, target_mode):
    spec, manifest, metadata = make_inputs(
        task_id=task_id,
        graph_u_mode=mode,
        test_graph_u_mode=target_mode,
    )
    profile = _profile()
    spec["sampling_config"] = profile
    manifest["sampling_config"] = profile
    payload = torch.load(spec["checkpoint"], map_location="cpu", weights_only=True)
    config = asdict(experiment_prior_config(spec["training_condition"], mode, profile))
    payload["prior_config"] = config
    payload["dump_metadata"]["prior_config"] = config
    torch.save(payload, spec["checkpoint"])
    winner = next(row for row in manifest["winners"] if row["checkpoint"] == spec["checkpoint"])
    winner["checkpoint_sha256"] = _sha256_file(Path(spec["checkpoint"]))
    Path(spec["manifest_path"]).write_text(json.dumps(manifest), encoding="utf-8")
    metadata["prior_config"] = asdict(experiment_prior_config(spec["test_condition"], target_mode, profile))
    h5py = pytest.importorskip("h5py")
    with h5py.File(spec["test_dump"], "r+") as stream:
        stream.attrs["metadata_json"] = json.dumps(metadata)
    return spec, manifest, metadata


@pytest.mark.parametrize("mode,target_mode", [
    ("confounding", "confounding"),
    ("covariate", "covariate"),
    ("confounding", "covariate"),
    ("covariate", "confounding"),
])
@pytest.mark.parametrize("task_id", [1, 2, 3, 4])
def test_sampled_winner_testing_validates_all_four_training_test_conditions(
    make_inputs, task_id, mode, target_mode
):
    spec, manifest, metadata = _sampled_inputs(make_inputs, task_id, mode, target_mode)
    result = _result(spec, manifest, metadata)
    final_tests.validate_test_result(result, spec)
    assert metadata["prior_config"]["graph_u_shift_mode"] == (
        "sampled" if spec["test_condition"] == "shift" else "fixed"
    )
    candidate = final_tests._validate_winner_entry(result["winner"], spec)
    assert candidate["sampling_config"] == _profile()


def test_test_spec_binds_profile_without_changing_fixed_spec(tmp_path):
    original = final_tests.test_spec(4, tmp_path, tmp_path, tmp_path)
    sampled = final_tests.test_spec(4, tmp_path, tmp_path, tmp_path, sampling_config=_profile())
    assert "sampling_config" not in original
    assert sampled.pop("sampling_config") == _profile()
    assert sampled == original


@pytest.mark.parametrize("wrong_profile", [None, {"graph_u_query_scale_std": 0.07}])
def test_winner_manifest_rejects_different_or_missing_sampling_profile(make_inputs, wrong_profile):
    spec, _, _ = _sampled_inputs(make_inputs, 4, "confounding", "confounding")
    if wrong_profile is None:
        del spec["sampling_config"]
    else:
        spec["sampling_config"] = {**_profile(), **wrong_profile}
    with pytest.raises(ValueError, match="sampling configuration"):
        final_tests._load_winner(spec)


def test_sampled_shift_bank_is_not_accepted_as_fixed_experiment(make_inputs):
    spec, _, metadata = _sampled_inputs(make_inputs, 4, "confounding", "confounding")
    with pytest.raises(ValueError, match="prior_config"):
        final_tests._validate_test_metadata(metadata, "shift", 1)
    final_tests._validate_test_metadata(metadata, "shift", 1, sampling_config=_profile())


def test_summary_carries_profile_through_every_report(completed_tests):
    data, search, output, specs, records, validated = completed_tests
    for spec, record in zip(specs, records):
        record["test"]["sampling_config"] = _profile()
        rewrite(spec, record)
    summary = summaries.summarize_tests(data, search, output, sampling_config=_profile())
    assert summary["sampling_config"] == _profile()
    assert summary["evaluation_count"] == 20
    assert validated == list(range(1, 21))
    assert len(summary["cells"]) == 4
