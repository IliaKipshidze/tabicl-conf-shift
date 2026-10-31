"""Sampled source-shift configuration through Nano generation and dump resumption."""

from dataclasses import asdict, replace
import json

import numpy as np
import pytest
import torch

from tabicl.nano_graph_u import __main__ as nano_cli
from tabicl.nano_graph_u import data as nano_data
from tabicl.prior.graph_lib._config import PriorConfig


@pytest.mark.parametrize("mode", ["fixed", "sampled"])
def test_nano_cli_passes_shift_configuration(monkeypatch, tmp_path, mode):
    received = {}
    monkeypatch.setattr(
        nano_cli, "generate_dump", lambda path, **kwargs: received.update(kwargs)
    )
    argv = ["generate", "--dump", str(tmp_path / "data.h5")]
    if mode == "sampled":
        argv += ["--shift-mode", mode]
    overrides = {
        "scale_mean": 1.6, "scale_std": 0.07, "scale_min": 1.4, "scale_max": 1.8,
        "location_mean": 2.1, "location_std": 0.12, "location_min": 1.8, "location_max": 2.4,
    }
    for name, value in overrides.items():
        argv += [f"--query-{name.replace('_', '-')}", str(value)]
    nano_cli.main(argv)
    config = received["prior_config"]
    assert config.graph_u_shift_mode == mode
    assert config.graph_u_query_scale == 1.5
    assert config.graph_u_query_location == 2.0
    for name, value in overrides.items():
        assert getattr(config, f"graph_u_query_{name}") == value


def test_nano_cli_sampled_defaults_match_prior():
    args = nano_cli.build_parser().parse_args(["generate", "--dump", "unused.h5"])
    config = PriorConfig()
    assert args.shift_mode == "fixed"
    for parameter in ("scale", "location"):
        for statistic in ("mean", "std", "min", "max"):
            name = f"query_{parameter}_{statistic}"
            assert getattr(args, name) == getattr(config, f"graph_u_{name}")


@pytest.mark.parametrize("graph_u_mode", ["confounding", "covariate"])
def test_sampled_dump_replays_and_resumes_with_distribution_metadata(
    tmp_path, monkeypatch, graph_u_mode
):
    h5py = pytest.importorskip("h5py")
    config = PriorConfig(
        graph_u_enabled=True,
        graph_u_structure_mode="add_root",
        graph_u_mode=graph_u_mode,
        graph_u_shift_mode="sampled",
    )
    kwargs = dict(
        steps=2, batch_size=2, seed=37, prior_config=config, rows=64, features=3
    )
    complete, resumed = tmp_path / "complete.h5", tmp_path / "resumed.h5"
    nano_data.generate_dump(complete, **kwargs)
    write_step = nano_data._write_step

    def interrupt(file, step, batch_size, arrays):
        if step == 1:
            file["x"][step * batch_size] = np.nan
            file.flush()
            raise RuntimeError("interrupted sampled generation")
        write_step(file, step, batch_size, arrays)

    with monkeypatch.context() as patch:
        patch.setattr(nano_data, "_write_step", interrupt)
        with pytest.raises(RuntimeError, match="interrupted sampled generation"):
            nano_data.generate_dump(resumed, **kwargs)
    nano_data.generate_dump(resumed, resume=True, **kwargs)
    with h5py.File(complete, "r") as first, h5py.File(resumed, "r") as second:
        assert first.attrs["metadata_json"] == second.attrs["metadata_json"]
        assert json.loads(first.attrs["metadata_json"])["prior_config"] == asdict(config)
        for name in ("x", "y", "train_sizes"):
            np.testing.assert_array_equal(first[name][:], second[name][:])

    for changed in (
        replace(config, graph_u_shift_mode="fixed"),
        replace(config, graph_u_query_scale_std=0.08),
        replace(config, graph_u_query_location_max=2.2),
    ):
        with pytest.raises(ValueError, match="different generation metadata"):
            nano_data.generate_dump(
                complete, resume=True, **{**kwargs, "prior_config": changed}
            )


def test_sampled_bank_trains_and_evaluates_with_complete_metadata(tmp_path, monkeypatch):
    pytest.importorskip("h5py")
    pytest.importorskip("schedulefree")
    from tabicl.nano_graph_u import train as nano_train
    from tabicl.nano_graph_u.evaluate import evaluate_checkpoint

    monkeypatch.setattr(nano_train, "PAPER_MODEL_CONFIG", {
        "embedding_size": 16, "num_attention_heads": 2,
        "mlp_hidden_size": 32, "num_layers": 1, "num_outputs": 2,
    })
    config = PriorConfig(
        graph_u_enabled=True,
        graph_u_structure_mode="add_root",
        graph_u_shift_mode="sampled",
    )
    training_dump, evaluation_dump = tmp_path / "train.h5", tmp_path / "test.h5"
    for path, seed in ((training_dump, 83), (evaluation_dump, 84)):
        nano_data.generate_dump(
            path, steps=1, batch_size=2, seed=seed, prior_config=config,
            rows=64, features=3,
        )
    checkpoint = nano_train.train_model(
        training_dump, tmp_path / "checkpoints", steps=1, batch_size=2,
        seed=7, device="cpu", save_every=1, log_every=1,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["step"] == 1
    assert payload["prior_config"] == asdict(config)
    assert payload["dump_metadata"]["prior_config"] == asdict(config)

    report = evaluate_checkpoint(checkpoint, evaluation_dump, batch_size=2, device="cpu")
    assert report["task_count"] == 2
    assert report["dump_metadata"]["prior_config"] == asdict(config)
    assert report["dump_metadata"]["seed"] == 84
    assert np.isfinite(report["metrics"]["nll"])
