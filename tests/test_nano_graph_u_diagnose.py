"""Read-only Nano prediction diagnostics on a small committed dump."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tabicl.nano_graph_u.diagnose import (
    _checkpoint_weight_views,
    diagnose_dump,
    main,
)
from tabicl.nano_graph_u.model import NanoTabPFNModel
from tabicl.nano_graph_u.train import _schedulefree_adamw


def test_diagnostic_scores_query_and_support_only_baseline(tmp_path, capsys):
    h5py = pytest.importorskip("h5py")
    dump = tmp_path / "eval.h5"
    x = np.arange(4 * 8 * 3, dtype=np.float32).reshape(4, 8, 3) / 10
    y = np.array(
        [
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 1, 0, 1, 0],
            [1, 0, 1, 0, 0, 1, 0, 1],
        ],
        dtype=np.int64,
    )
    with h5py.File(dump, "x") as file:
        file.create_dataset("x", data=x)
        file.create_dataset("y", data=y)
        file.create_dataset("train_sizes", data=np.array([4, 4, 4, 4]))
        file.attrs["metadata_json"] = json.dumps(
            {"format_version": 1, "steps": 2, "batch_size": 2, "rows": 8, "features": 3}
        )
        file.attrs["committed_steps"] = 2

    config = {
        "embedding_size": 96,
        "num_attention_heads": 4,
        "mlp_hidden_size": 192,
        "num_layers": 3,
        "num_outputs": 2,
    }
    model = NanoTabPFNModel(**config)
    checkpoint = tmp_path / "step-1.pt"
    torch.save(
        {
            "format": "nano_graph_u_v1",
            "model_config": config,
            "model": model.state_dict(),
            "step": 1,
            "train_config": {"dump_path": str(tmp_path / "train.h5")},
        },
        checkpoint,
    )
    report = diagnose_dump([checkpoint], dump, max_tasks=3, baseline_trees=4)
    assert report["task_count"] == 3
    assert len(report["models"]) == 1
    assert len(report["models"][0]["per_task"]) == 3
    assert report["models"][0]["matches_training_dump"] is False
    assert report["models"][0]["weight_view"] == "schedulefree_averaged_x"
    assert report["baseline"]["fit"] == "each task's support rows only"
    assert report["baseline"]["summary"]["task_count"] == 3
    assert report["models"][0]["per_task"][-1]["task_index"] == 2
    assert 0 <= report["models"][0]["summary"]["mean_roc_auc"] <= 1
    assert report["models"][0]["per_task"][0]["support_rows"] == 4
    assert report["models"][0]["per_task"][0]["query_rows"] == 4

    same_dump_payload = torch.load(checkpoint, weights_only=True)
    same_dump_payload["train_config"]["dump_path"] = str(dump)
    same_dump_checkpoint = tmp_path / "same-dump.pt"
    torch.save(same_dump_payload, same_dump_checkpoint)
    same_dump_report = diagnose_dump(
        [same_dump_checkpoint], dump, max_tasks=1, baseline=False
    )
    assert same_dump_report["models"][0]["matches_training_dump"] is True

    output = tmp_path / "diagnostic.json"
    main(
        [
            "--checkpoint",
            str(checkpoint),
            "--dump",
            str(dump),
            "--max-tasks",
            "1",
            "--no-baseline",
            "--output",
            str(output),
        ]
    )
    console = capsys.readouterr().out
    assert '"task_count": 1' in console
    assert '"per_task"' not in console
    assert json.loads(output.read_text())["task_count"] == 1
    assert "per_task" in json.loads(output.read_text())["models"][0]
    with pytest.raises(ValueError, match="no optimizer state"):
        diagnose_dump(
            [checkpoint],
            dump,
            max_tasks=1,
            baseline=False,
            include_training_weights=True,
        )
    with pytest.raises(FileExistsError):
        main(
            [
                "--checkpoint",
                str(checkpoint),
                "--dump",
                str(dump),
                "--max-tasks",
                "1",
                "--no-baseline",
                "--output",
                str(output),
            ]
        )

    with h5py.File(dump, "r+") as file:
        file.attrs["committed_steps"] = 1
    with pytest.raises(ValueError, match="complete frozen evaluation dump"):
        diagnose_dump([checkpoint], dump, max_tasks=1)


def test_diagnostic_reconstructs_schedulefree_training_weights(tmp_path):
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("schedulefree")
    dump = tmp_path / "eval.h5"
    x = np.arange(2 * 8 * 3, dtype=np.float32).reshape(2, 8, 3) / 10
    y = np.array(
        [
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0],
        ],
        dtype=np.int64,
    )
    with h5py.File(dump, "x") as file:
        file.create_dataset("x", data=x)
        file.create_dataset("y", data=y)
        file.create_dataset("train_sizes", data=np.array([4, 4]))
        file.attrs["metadata_json"] = json.dumps(
            {"format_version": 1, "steps": 1, "batch_size": 2, "rows": 8, "features": 3}
        )
        file.attrs["committed_steps"] = 1

    config = {
        "embedding_size": 8,
        "num_attention_heads": 2,
        "mlp_hidden_size": 16,
        "num_layers": 1,
        "num_outputs": 2,
    }
    model = NanoTabPFNModel(**config)
    optimizer = _schedulefree_adamw(model.parameters(), learning_rate=0.004)
    optimizer.train()
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)
    logits = model((x_tensor, y_tensor[:, :4].float()), train_test_split_index=4)
    loss = F.cross_entropy(logits.reshape(-1, 2), y_tensor[:, 4:].reshape(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    expected_training_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    optimizer.eval()

    checkpoint = tmp_path / "step-1.pt"
    torch.save(
        {
            "format": "nano_graph_u_v1",
            "model_config": config,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": 1,
            "train_config": {"learning_rate": 0.004, "dump_path": "different.h5"},
        },
        checkpoint,
    )
    report = diagnose_dump(
        [checkpoint],
        dump,
        max_tasks=2,
        baseline=False,
        include_training_weights=True,
    )
    assert [model_report["weight_view"] for model_report in report["models"]] == [
        "schedulefree_averaged_x",
        "schedulefree_training_y",
    ]
    assert all(
        model_report["summary"]["task_count"] == 2 for model_report in report["models"]
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    views = _checkpoint_weight_views(
        payload,
        torch.device("cpu"),
        include_training_weights=True,
        checkpoint_label=str(checkpoint),
    )
    reconstructed_training_state = views[1][1].state_dict()
    for name, expected in expected_training_state.items():
        assert torch.equal(reconstructed_training_state[name], expected)
