"""Location plans inherit source settings and pin an immutable frozen-model recipe."""

import json
import math
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from tabicl.nano_graph_u import location_sweep as sweep
from tabicl.nano_graph_u.test_winners import METRIC_NAMES
from tabicl.prior.graph_lib._config import GRAPH_U_SAMPLING_DEFAULTS


def _settings():
    return {**GRAPH_U_SAMPLING_DEFAULTS, "graph_u_shift_mode": "sampled"}


def test_default_plan_has_80_distinct_banks_with_only_location_profile_changed(tmp_path):
    settings = _settings()
    before = deepcopy(settings)
    specs = sweep.bank_specs(tmp_path, settings)
    assert sweep.LOCATION_PROFILES == (
        (0.3, 0.1, 0.5), (1.3, 1.1, 1.5), (1.7, 1.5, 1.9), (1.9, 1.7, 2.1),
        (2.2, 2.0, 2.4), (4.5, 4.3, 4.7), (10.0, 9.8, 10.2), (100.0, 85.0, 115.0),
    )
    assert len(specs) == len({spec["dump"] for spec in specs}) == 80
    assert Counter(spec["graph_u_mode"] for spec in specs) == {"confounding": 40, "covariate": 40}
    assert set(Counter(spec["location_mean"] for spec in specs).values()) == {10}
    assert set(Counter(spec["repetition"] for spec in specs).values()) == {16}
    profiles = {mean: (lower, upper) for mean, lower, upper in sweep.LOCATION_PROFILES}
    for spec in specs:
        mean, profile = spec["location_mean"], spec["sampling_config"]
        assert spec["sweep_parameter"] == "location"
        assert profile["graph_u_query_location_mean"] == mean
        assert (profile["graph_u_query_location_min"], profile["graph_u_query_location_max"]) == profiles[mean]
        for key, value in settings.items():
            if key not in {"graph_u_query_location_mean", "graph_u_query_location_min",
                           "graph_u_query_location_max"}:
                assert profile[key] == value
        seed = 5000 + spec["repetition"]
        assert spec["seed"] == seed and spec["steps"] == 100
        assert spec["dump"] == (tmp_path / "data" / f"location_{sweep._tag(mean)}" / spec["graph_u_mode"]
                                / f"rep_{spec['repetition']}" / f"shift_test_seed{seed}_100steps_b32_r150_f5.h5")
    assert settings == before
    assert not (tmp_path / "data").exists()


def test_custom_source_scale_parameters_and_location_std_are_inherited(tmp_path):
    settings = {**_settings(), "graph_u_query_scale_mean": 2.5, "graph_u_query_scale_std": 0.4,
                "graph_u_query_scale_min": 2.1, "graph_u_query_scale_max": 2.9,
                "graph_u_query_location_std": 0.35}
    specs = sweep.bank_specs(str(tmp_path), settings, ((-2, -3, -1), (0, 0, 0.4)))
    assert len(specs) == 20
    for spec in specs:
        for field in ("mean", "std", "min", "max"):
            key = f"graph_u_query_scale_{field}"
            assert spec["sampling_config"][key] == settings[key]
        assert spec["sampling_config"]["graph_u_query_location_std"] == 0.35


@pytest.mark.parametrize("profiles", [
    (), ((1.3, 1.1, 1.5), (1.3, 1.0, 1.6)),
    ((1.3, 1.1),), ((1.3, 1.1, 1.5, 1.7),), (1.3,), ("123",),
    ((float("nan"), 1, 2),), ((1.3, float("-inf"), 2),), ((1.3, 1, float("inf")),),
    ((1.3, 1.3, 1.3),), ((1.3, 1.5, 1.1),), ((1.3, 1.4, 1.5),), ((1.3, 1.1, 1.2),),
])
def test_invalid_profiles_fail_before_output_creation(tmp_path, profiles):
    root = tmp_path / "sweep"
    with pytest.raises(ValueError):
        sweep.bank_specs(root, _settings(), profiles)
    assert not root.exists()


@pytest.mark.parametrize("change", [{"graph_u_query_scale_std": 0}, {"graph_u_query_location_std": 0},
                                   {"graph_u_shift_mode": "fixed"}, {"unknown": 1}])
def test_bank_plan_normalizes_and_validates_source_settings(tmp_path, change):
    with pytest.raises(ValueError):
        sweep.bank_specs(tmp_path / "sweep", {**_settings(), **change})


def _rows():
    rows = []
    for train_mode in sweep.MODES:
        for condition in ("identity", "shift"):
            for test_mode in sweep.MODES:
                for profile in (None, *sweep.LOCATION_PROFILES):
                    for repetition in range(1, 6):
                        values = (None, None, None) if profile is None else profile
                        rows.append({
                            "training_graph_u_mode": train_mode, "training_condition": condition,
                            "test_graph_u_mode": test_mode,
                            "test_condition": "identity" if profile is None else "shift",
                            "location_mean": values[0], "location_std": None if profile is None else 0.15,
                            "location_min": values[1], "location_max": values[2],
                            **{f"scale_{field}": None if profile is None else _settings()[f"graph_u_query_scale_{field}"]
                               for field in ("mean", "std", "min", "max")},
                            "repetition": repetition,
                            **{metric: 0.6 + 0.01 * repetition for metric in METRIC_NAMES},
                        })
    return rows


def test_location_summary_groups_360_records_into_72_cells_with_sample_std():
    rows = _rows()
    assert len(rows) == 360
    cells = sweep._summarize(rows, parameter="location")
    assert len(cells) == 72
    assert sum(cell["test_condition"] == "identity" for cell in cells) == 8
    for cell in cells:
        assert cell["repetition_count"] == 5
        if cell["test_condition"] == "shift":
            assert cell["scale_mean"] == 1.5 and cell["scale_std"] == 0.1
            assert cell["location_std"] == 0.15
        for metric in METRIC_NAMES:
            assert cell[f"{metric}_mean"] == pytest.approx(0.63)
            assert cell[f"{metric}_std"] == pytest.approx(0.01 * math.sqrt(2.5))


@pytest.mark.parametrize("problem", ["missing", "duplicate"])
def test_location_summary_requires_five_distinct_repetitions(problem):
    rows = _rows()
    rows = rows[1:] if problem == "missing" else rows + [dict(rows[0])]
    with pytest.raises(ValueError):
        sweep._summarize(rows, parameter="location")


@pytest.mark.parametrize("change", ["profile", "settings", "manifest", "checkpoint", "seed"])
def test_recipe_is_pinned_before_generation_and_rejects_changed_recipe(tmp_path, monkeypatch, change):
    source, output = tmp_path / "source", tmp_path / "sweep"
    source.mkdir()
    settings = _settings()
    models = {("confounding", 1, "identity"): {"checkpoint_sha256": "a" * 64}}
    manifests = {"confounding": "b" * 64}
    monkeypatch.setattr(sweep, "load_source", lambda path: (settings, models, [], manifests))
    generations = []
    writes = []
    atomic_write = sweep._atomic_write

    def write(path, contents):
        writes.append(path)
        atomic_write(path, contents)

    def interrupted_generation(specs, workers):
        assert (output / "experiment.json").is_file()
        recipe = json.loads((output / "experiment.json").read_text())
        assert recipe["format"] == "nano_graph_u_location_sweep_v1"
        assert recipe["test_seed_base"] == 5000
        assert recipe["location_profiles"] == [list(profile) for profile in sweep.LOCATION_PROFILES]
        generations.append(specs)
        raise RuntimeError("generation interrupted")

    monkeypatch.setattr(sweep, "_atomic_write", write)
    monkeypatch.setattr(sweep, "_generate_banks", interrupted_generation)
    with pytest.raises(RuntimeError, match="generation interrupted"):
        sweep.run_sweep(source, output)
    pinned = (output / "experiment.json").read_bytes()
    assert len(generations) == len(writes) == 1
    kwargs = {}
    if change == "profile":
        kwargs["location_profiles"] = ((0.3, 0.2, 0.4), *sweep.LOCATION_PROFILES[1:])
    elif change == "settings":
        settings["graph_u_query_location_std"] = 0.2
    elif change == "manifest":
        manifests["confounding"] = "c" * 64
    elif change == "checkpoint":
        models[("confounding", 1, "identity")]["checkpoint_sha256"] = "d" * 64
    else:
        monkeypatch.setattr(sweep, "_SEED_BASE", 6000)
    with pytest.raises(FileExistsError):
        sweep.run_sweep(source, output, **kwargs)
    assert len(generations) == len(writes) == 1
    assert (output / "experiment.json").read_bytes() == pinned
    assert not (output / "data").exists()


@pytest.mark.parametrize("relationship", ["same", "ancestor", "descendant"])
def test_source_and_output_must_be_disjoint_before_loading(tmp_path, monkeypatch, relationship):
    source = tmp_path / "source"
    root = {"same": source, "ancestor": tmp_path, "descendant": source / "sweep"}[relationship]
    monkeypatch.setattr(sweep, "load_source", lambda path: pytest.fail("Must reject before reading source"))
    with pytest.raises(ValueError, match="separate directory"):
        sweep.run_sweep(source, root)


@pytest.mark.parametrize("workers", [True, 0, -1, 1.5])
def test_workers_validation_precedes_source_loading(tmp_path, monkeypatch, workers):
    monkeypatch.setattr(sweep, "load_source", lambda path: pytest.fail("Must reject before reading source"))
    with pytest.raises(ValueError, match="positive integer"):
        sweep.run_sweep(tmp_path / "source", tmp_path / "sweep", workers=workers)


def test_cli_parses_profiles_and_worker_device_overrides(monkeypatch):
    calls = []
    monkeypatch.setattr(sweep, "run_sweep", lambda *args, **kwargs: calls.append((args, kwargs)))
    sweep.main(["--source-root", "source", "--root", "sweep", "--location-profiles",
                "0.3:0.1:0.5", "100:85:115", "--workers", "3", "--device", "cpu"])
    assert calls == [((Path("source"), Path("sweep")), {
        "location_profiles": [(0.3, 0.1, 0.5), (100.0, 85.0, 115.0)], "workers": 3, "device": "cpu",
    })]


def test_cli_uses_default_profiles_workers_and_device(monkeypatch):
    calls = []
    monkeypatch.setattr(sweep, "run_sweep", lambda *args, **kwargs: calls.append(kwargs))
    sweep.main(["--source-root", "source", "--root", "sweep"])
    assert calls == [{"location_profiles": sweep.LOCATION_PROFILES, "workers": 8, "device": "cuda"}]


@pytest.mark.parametrize("profile", ["1:2", "1:2:3:4", "nan:0:1", "1:2:3", "x:0:1"])
def test_cli_rejects_invalid_profile_without_starting_run(monkeypatch, profile):
    monkeypatch.setattr(sweep, "run_sweep", lambda *args, **kwargs: pytest.fail("Must reject before running"))
    with pytest.raises(SystemExit) as exc:
        sweep.main(["--source-root", "source", "--root", "sweep", "--location-profiles", profile])
    assert exc.value.code == 2
