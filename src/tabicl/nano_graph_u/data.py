"""Paper-sized nanoTabPFN batches from the existing Graph-U prior.

The nanoTabPFN model takes a single support/query split index for a whole
batch.  GraphPrior can retain its usual groups of four independent tasks while
sampling one split for the entire batch (``seq_len_per_gp=False``).  We store
the split for *every* task and check equality both before writing and before
feeding a batch to the model; never silently take only the first split.

HDF5 generation commits one complete batch at a time.  ``committed_steps`` is
advanced only after the batch data have been flushed, so a preempted job can
resume with the same configuration and deterministic per-step random seeds.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import random
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tabicl.prior import PriorDataset
from tabicl.prior.graph_lib._config import PriorConfig

_FORMAT_VERSION = 1
_MAX_STEP_RETRIES = 8


def _generator_source_sha256() -> str:
    """Fingerprint code that constructs Graph-U tables for safe dump resumption."""
    package_dir = Path(__file__).resolve().parents[1]
    prior_dir = package_dir / "prior"
    sources = [Path(__file__).resolve(), *prior_dir.rglob("*.py")]
    digest = hashlib.sha256()
    for source in sorted(sources, key=lambda path: path.as_posix()):
        digest.update(source.relative_to(package_dir).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "Nano-Graph-U prior dumps require h5py; install h5py in the "
            "Nano training environment without replacing PyTorch."
        ) from exc
    return h5py


def _validate_shape_and_config(
    prior_config: PriorConfig,
    *,
    batch_size: int,
    rows: int,
    features: int,
    min_train_size: float,
    max_train_size: float,
    n_jobs: int,
) -> None:
    if not isinstance(prior_config, PriorConfig):
        raise TypeError("prior_config must be a PriorConfig")
    if (
        not prior_config.graph_u_enabled
        or prior_config.graph_u_structure_mode != "add_root"
    ):
        raise ValueError(
            "Nano-Graph-U requires graph_u_enabled=True and "
            "graph_u_structure_mode='add_root'"
        )
    if prior_config.ensure_iid:
        raise ValueError("Graph-U cannot use ensure_iid=True")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 4:
        raise ValueError("rows must be an integer of at least 4")
    if isinstance(features, bool) or not isinstance(features, int) or features < 1:
        raise ValueError("features must be a positive integer")
    if not (0 < min_train_size < max_train_size < 1):
        raise ValueError(
            "split ratios must satisfy 0 < min_train_size < max_train_size < 1"
        )
    if int(rows * min_train_size) < 2 or int(rows * max_train_size) > rows - 2:
        raise ValueError(
            "split ratios must leave at least two support and two query rows"
        )
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs != 1:
        raise ValueError(
            "n_jobs must be 1 for deterministic, resumable Graph-U dumps; "
            "the existing within-batch fork pool does not independently seed workers"
        )


def create_prior(
    prior_config: PriorConfig,
    *,
    batch_size: int = 32,
    rows: int = 150,
    features: int = 5,
    min_train_size: float = 0.3,
    max_train_size: float = 0.9,
    n_jobs: int = 1,
) -> PriorDataset:
    """Create the unchanged Graph-U generator restricted to Nano task sizes."""
    _validate_shape_and_config(
        prior_config,
        batch_size=batch_size,
        rows=rows,
        features=features,
        min_train_size=min_train_size,
        max_train_size=max_train_size,
        n_jobs=n_jobs,
    )
    return PriorDataset(
        regression=False,
        batch_size=batch_size,
        batch_size_per_gp=4,
        batch_size_per_subgp=4,
        min_features=features,
        max_features=features,
        max_classes=2,
        min_seq_len=None,
        max_seq_len=rows,
        log_seq_len=False,
        log_n_features=False,
        seq_len_per_gp=False,
        min_train_size=min_train_size,
        max_train_size=max_train_size,
        replay_small=False,
        prior_type="graph_scm",
        config=prior_config,
        n_jobs=n_jobs,
        num_threads_per_generate=1,
        device="cpu",
    )


def _validate_batch_tensors(
    x: torch.Tensor, y: torch.Tensor, splits: torch.Tensor
) -> int:
    if x.ndim != 3 or y.ndim != 2 or splits.ndim != 1:
        raise ValueError("expected x[B,T,F], y[B,T], and one split per table")
    batch_size, rows, _ = x.shape
    if y.shape != (batch_size, rows) or splits.shape != (batch_size,):
        raise ValueError("x, y and per-table splits disagree on batch/row dimensions")
    if not bool(torch.all(splits == splits[0])):
        raise ValueError(
            "Nano batches require the same support/query split for all tables; "
            f"got {splits.tolist()}"
        )
    split = int(splits[0].item())
    if not 2 <= split <= rows - 2:
        raise ValueError(f"invalid support/query split {split} for {rows} rows")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("features contain non-finite values")
    if not bool(torch.isfinite(y).all()):
        raise ValueError("labels contain non-finite values")
    if not bool(torch.all((y == 0) | (y == 1))):
        raise ValueError("Nano classification requires binary 0/1 labels")
    for partition in (y[:, :split], y[:, split:]):
        has_zero = torch.any(partition == 0, dim=1)
        has_one = torch.any(partition == 1, dim=1)
        if not bool(torch.all(has_zero & has_one)):
            raise ValueError(
                "every table must contain both classes in support and query"
            )
    return split


def adapt_batch(raw: tuple[torch.Tensor, ...]) -> dict[str, Any]:
    """Adapt a Graph-U prior batch to Nano's shared-split model interface."""
    if len(raw) != 5:
        raise ValueError("expected PriorDataset.get_batch()'s five tensors")
    x, y, active_features, seq_lens, train_sizes = raw
    if not all(isinstance(t, torch.Tensor) for t in raw):
        raise TypeError("prior batch must contain five dense torch tensors")
    if x.is_nested or y.is_nested:
        raise ValueError("Nano requires dense, fixed-length tables")
    if seq_lens.shape != (x.shape[0],) or not bool(torch.all(seq_lens == x.shape[1])):
        raise ValueError("Nano requires one fixed row count throughout the batch")
    if active_features.shape != (x.shape[0],) or not bool(
        torch.all((active_features > 0) & (active_features <= x.shape[2]))
    ):
        raise ValueError("invalid active feature counts")
    split = _validate_batch_tensors(x, y, train_sizes)
    return {
        "x": x.to(dtype=torch.float32).contiguous(),
        "y": y.to(dtype=torch.long).contiguous(),
        "train_test_split_index": split,
    }


def _metadata(
    prior_config: PriorConfig,
    *,
    steps: int,
    batch_size: int,
    seed: int,
    rows: int,
    features: int,
    min_train_size: float,
    max_train_size: float,
    n_jobs: int,
) -> dict[str, Any]:
    return {
        "format_version": _FORMAT_VERSION,
        "steps": steps,
        "batch_size": batch_size,
        "seed": seed,
        "rows": rows,
        "features": features,
        "min_train_size": min_train_size,
        "max_train_size": max_train_size,
        "batch_size_per_gp": 4,
        "batch_size_per_subgp": 4,
        "seq_len_per_gp": False,
        "prior_type": "graph_scm",
        "prior_config": asdict(prior_config),
        "generator_source_sha256": _generator_source_sha256(),
        "max_step_retries": _MAX_STEP_RETRIES,
        "n_jobs": n_jobs,
        "split_policy": "one split per generated batch, stored for every table",
    }


def _generate_step(
    step: int,
    *,
    seed: int,
    prior_config: PriorConfig,
    batch_size: int,
    rows: int,
    features: int,
    min_train_size: float,
    max_train_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate one independent batch, identically in parent or spawned worker."""
    for retry in range(_MAX_STEP_RETRIES):
        step_seed = seed + step + retry * 1_000_003
        random.seed(step_seed)
        np.random.seed(step_seed % (2**32))
        torch.manual_seed(step_seed)
        prior = create_prior(
            prior_config,
            batch_size=batch_size,
            rows=rows,
            features=features,
            min_train_size=min_train_size,
            max_train_size=max_train_size,
            n_jobs=1,
        )
        try:
            raw = prior.get_batch(batch_size)
        except RuntimeError as exc:
            if not str(exc).startswith("Unable to generate a valid Graph-U dataset"):
                raise
            if retry == _MAX_STEP_RETRIES - 1:
                raise RuntimeError(
                    f"Nano batch {step} failed after {_MAX_STEP_RETRIES} "
                    "deterministic prior-generation retries"
                ) from exc
            continue
        adapted = adapt_batch(raw)
        return (
            adapted["x"].cpu().numpy(),
            adapted["y"].cpu().numpy(),
            raw[4].cpu().numpy(),
        )
    raise AssertionError("unreachable: retry loop always returns or raises")


def _write_step(
    file: Any,
    step: int,
    batch_size: int,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Write a complete batch and only then advance the durable commit marker."""
    start, end = step * batch_size, (step + 1) * batch_size
    x, y, train_sizes = arrays
    file["x"][start:end] = x
    file["y"][start:end] = y
    # Do not collapse this to the scalar Nano split: all per-table split values
    # must remain available for the loader's equality assertion.
    file["train_sizes"][start:end] = train_sizes
    file.flush()
    file.attrs["committed_steps"] = step + 1
    file.flush()


def generate_dump(
    path: str | Path,
    *,
    steps: int,
    batch_size: int,
    seed: int,
    prior_config: PriorConfig,
    rows: int = 150,
    features: int = 5,
    min_train_size: float = 0.3,
    max_train_size: float = 0.9,
    n_jobs: int = 1,
    workers: int = 1,
    resume: bool = False,
) -> Path:
    """Generate a resumable HDF5 dump of ``steps * batch_size`` Graph-U tasks.

    ``resume=False`` refuses to overwrite an existing file.  On resumption,
    every metadata field must match and generation restarts at the first batch
    not marked committed. ``workers`` parallelizes independent batches, not
    generation within a batch; it may change when resuming and is deliberately
    absent from scientific metadata. Per-step seeds make output independent of
    the worker count and preemption point.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    _validate_shape_and_config(
        prior_config,
        batch_size=batch_size,
        rows=rows,
        features=features,
        min_train_size=min_train_size,
        max_train_size=max_train_size,
        n_jobs=n_jobs,
    )
    metadata = _metadata(
        prior_config,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        rows=rows,
        features=features,
        min_train_size=min_train_size,
        max_train_size=max_train_size,
        n_jobs=n_jobs,
    )
    metadata_json = json.dumps(metadata, sort_keys=True, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite existing Nano prior dump: {path}")
    h5py = _require_h5py()
    with h5py.File(path, "r+" if path.exists() else "x") as file:
        if "metadata_json" in file.attrs:
            if file.attrs["metadata_json"] != metadata_json:
                raise ValueError(
                    "cannot resume dump with different generation metadata"
                )
            committed = int(file.attrs["committed_steps"])
            if not 0 <= committed <= steps:
                raise ValueError("dump has invalid committed_steps metadata")
        else:
            if resume and len(file):
                raise ValueError("existing dump has no Nano metadata")
            total = steps * batch_size
            file.create_dataset(
                "x",
                (total, rows, features),
                dtype="f4",
                chunks=(batch_size, rows, features),
            )
            file.create_dataset(
                "y", (total, rows), dtype="i8", chunks=(batch_size, rows)
            )
            file.create_dataset(
                "train_sizes", (total,), dtype="i4", chunks=(batch_size,)
            )
            file.attrs["metadata_json"] = metadata_json
            file.attrs["committed_steps"] = 0
            file.flush()
            committed = 0

        task_kwargs = {
            "seed": seed,
            "prior_config": prior_config,
            "batch_size": batch_size,
            "rows": rows,
            "features": features,
            "min_train_size": min_train_size,
            "max_train_size": max_train_size,
        }
        if committed < steps:
            # Always spawn, even for one worker: the underlying prior iterates
            # a Python set whose order depends on the interpreter hash seed.
            # A fixed child hash seed makes serial and parallel dumps identical
            # without changing the existing prior's behavior for TabICL runs.
            # Submit only a bounded number of jobs and write in step order.
            previous_hash_seed = os.environ.get("PYTHONHASHSEED")
            os.environ["PYTHONHASHSEED"] = "0"
            try:
                with ProcessPoolExecutor(
                    max_workers=workers, mp_context=mp.get_context("spawn")
                ) as executor:
                    pending = {}
                    next_step = committed
                    prefetch = min(steps - committed, 2 * workers)
                    for _ in range(prefetch):
                        pending[next_step] = executor.submit(
                            _generate_step, next_step, **task_kwargs
                        )
                        next_step += 1
                    for step in range(committed, steps):
                        arrays = pending.pop(step).result()
                        if next_step < steps:
                            pending[next_step] = executor.submit(
                                _generate_step, next_step, **task_kwargs
                            )
                            next_step += 1
                        _write_step(file, step, batch_size, arrays)
            finally:
                if previous_hash_seed is None:
                    os.environ.pop("PYTHONHASHSEED", None)
                else:
                    os.environ["PYTHONHASHSEED"] = previous_hash_seed
    return path


class NanoPriorDumpLoader:
    """Read committed HDF5 batches, validating every stored split on use.

    ``start_step`` and ``num_steps`` are absolute batch indices in the dump;
    iteration yields ``[start_step, num_steps)``.  If ``num_steps`` is omitted,
    all currently committed batches are available.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        num_steps: int | None = None,
        batch_size: int = 32,
        start_step: int = 0,
        device: str | torch.device = "cpu",
    ) -> None:
        self.path = Path(path)
        self.device = torch.device(device)
        self.batch_size = batch_size
        h5py = _require_h5py()
        with h5py.File(self.path, "r") as file:
            if "metadata_json" not in file.attrs or "committed_steps" not in file.attrs:
                raise ValueError("file is not a committed Nano-Graph-U prior dump")
            self.metadata = json.loads(file.attrs["metadata_json"])
            self.committed_steps = int(file.attrs["committed_steps"])
            if self.metadata.get("format_version") != _FORMAT_VERSION:
                raise ValueError("unsupported Nano prior dump format")
            if batch_size != self.metadata["batch_size"]:
                raise ValueError(
                    "loader batch_size must match the generation batch_size"
                )
            total = self.metadata["steps"] * batch_size
            expected_shapes = {
                "x": (total, self.metadata["rows"], self.metadata["features"]),
                "y": (total, self.metadata["rows"]),
                "train_sizes": (total,),
            }
            for key, shape in expected_shapes.items():
                if key not in file or file[key].shape != shape:
                    raise ValueError(
                        f"invalid or missing {key} dataset in Nano prior dump"
                    )
        if (
            isinstance(start_step, bool)
            or not isinstance(start_step, int)
            or start_step < 0
        ):
            raise ValueError("start_step must be a non-negative integer")
        if num_steps is None:
            num_steps = self.committed_steps
        if isinstance(num_steps, bool) or not isinstance(num_steps, int):
            raise TypeError("num_steps must be an integer or None")
        if not start_step <= num_steps <= self.committed_steps:
            raise ValueError(
                "requested steps exceed the committed dump or precede start_step"
            )
        self.start_step = start_step
        self.num_steps = num_steps

    def __len__(self) -> int:
        return self.num_steps - self.start_step

    def __iter__(self) -> Iterator[dict[str, Any]]:
        h5py = _require_h5py()
        with h5py.File(self.path, "r") as file:
            for step in range(self.start_step, self.num_steps):
                start, end = step * self.batch_size, (step + 1) * self.batch_size
                x = torch.from_numpy(file["x"][start:end]).to(self.device)
                y = torch.from_numpy(file["y"][start:end]).to(self.device)
                splits = torch.from_numpy(file["train_sizes"][start:end])
                split = _validate_batch_tensors(x, y, splits)
                yield {"x": x, "y": y, "train_test_split_index": split}
