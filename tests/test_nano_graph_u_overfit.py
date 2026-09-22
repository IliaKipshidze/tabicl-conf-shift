"""Checks for the isolated, fixed-table Nano memorization diagnostic."""

from __future__ import annotations

import json
import math

import pytest

from tabicl.nano_graph_u.data import generate_dump
from tabicl.nano_graph_u.overfit import main, overfit_fixed_table
from tabicl.nano_graph_u.train import _sha256_file
from tabicl.prior.graph_lib._config import PriorConfig


def test_fixed_table_diagnostic_isolated_and_reports_predictions(tmp_path):
    pytest.importorskip("h5py")
    pytest.importorskip("schedulefree")
    dump = tmp_path / "train.h5"
    generate_dump(
        dump,
        steps=1,
        batch_size=2,
        seed=23,
        prior_config=PriorConfig(
            graph_u_enabled=True,
            graph_u_structure_mode="add_root",
            graph_u_query_location=2.0,
            graph_u_query_scale=1.5,
        ),
        rows=64,
        features=3,
    )
    original_hash = _sha256_file(dump)
    report_path = tmp_path / "overfit.json"
    main(
        [
            "--dump",
            str(dump),
            "--output",
            str(report_path),
            "--table-index",
            "1",
            "--steps",
            "3",
            "--log-every",
            "2",
            "--device",
            "cpu",
        ]
    )
    result = json.loads(report_path.read_text(encoding="utf-8"))
    assert result["format"] == "nano_graph_u_fixed_table_overfit_v1"
    assert result["table_index"] == 1
    assert [item["step"] for item in result["history"]] == [0, 2, 3]
    assert 0 < result["support_rows"] < 64
    assert result["query_rows"] == 64 - result["support_rows"]
    assert result["parameter_change_l2"] > 0
    assert math.isfinite(result["initial"]["query_nll"])
    assert math.isfinite(result["final"]["query_nll"])
    assert 0 <= result["final"]["query_roc_auc"] <= 1
    assert _sha256_file(dump) == original_hash
    with pytest.raises(FileExistsError, match="overwrite"):
        main(["--dump", str(dump), "--output", str(report_path), "--device", "cpu"])


def test_fixed_table_rejects_out_of_range_index(tmp_path):
    pytest.importorskip("h5py")
    dump = tmp_path / "train.h5"
    generate_dump(
        dump,
        steps=1,
        batch_size=2,
        seed=25,
        prior_config=PriorConfig(
            graph_u_enabled=True, graph_u_structure_mode="add_root"
        ),
        rows=64,
        features=3,
    )
    with pytest.raises(ValueError, match="committed tables"):
        overfit_fixed_table(dump, table_index=2, steps=1, device="cpu")
