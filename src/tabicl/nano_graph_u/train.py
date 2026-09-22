"""Paper-sized nanoTabPFN training on pre-generated Graph-U tasks.

This is intentionally a separate training path from ``tabicl.train``.  In
particular, its checkpoints are not TabICL checkpoints.  Each HDF5 batch has
one support/query split shared by all its tables, and the same split is used
for label masking and the query loss.
"""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .data import NanoPriorDumpLoader
from .model import NanoTabPFNModel

CHECKPOINT_FORMAT = "nano_graph_u_v1"
PAPER_MODEL_CONFIG = {
    "embedding_size": 96,
    "num_attention_heads": 4,
    "mlp_hidden_size": 192,
    "num_layers": 3,
    "num_outputs": 2,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _schedulefree_adamw(parameters, *, learning_rate: float):
    try:
        from schedulefree import AdamWScheduleFree
    except ImportError as exc:
        raise RuntimeError(
            "Nano-Graph-U training needs schedulefree. Install the Nano optional "
            "dependencies (h5py and schedulefree) in the existing environment; "
            "do not install TFM-Playground's incompatible PyTorch build."
        ) from exc
    return AdamWScheduleFree(parameters, lr=learning_rate, weight_decay=0.0)


def _atomic_save(payload: dict[str, Any], target: Path) -> None:
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint_payload(
    *,
    model: NanoTabPFNModel,
    optimizer: Any,
    step: int,
    model_config: dict[str, Any],
    dump_metadata: dict[str, Any],
    train_config: dict[str, Any],
) -> dict[str, Any]:
    numpy_rng_state = np.random.get_state()
    return {
        "format": CHECKPOINT_FORMAT,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "model_config": model_config,
        "prior_config": dump_metadata["prior_config"],
        "train_config": train_config,
        "dump_metadata": dump_metadata,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
        # Use only tensors and built-in containers so torch.load(...,
        # weights_only=True) can read the checkpoint during evaluation.
        "numpy_rng_state": (
            numpy_rng_state[0],
            numpy_rng_state[1].tolist(),
            numpy_rng_state[2],
            numpy_rng_state[3],
            numpy_rng_state[4],
        ),
        "python_rng_state": random.getstate(),
    }


def train_model(
    dump: str | Path,
    checkpoint_dir: str | Path,
    *,
    steps: int = 2500,
    batch_size: int = 32,
    seed: int = 42,
    device: str = "cuda",
    learning_rate: float = 0.004,
    save_every: int = 250,
    log_every: int = 25,
    resume: bool = False,
) -> Path:
    """Train the paper's small binary Nano model, one HDF5 batch per update.

    The dump must have at least ``steps`` complete batches.  The loader checks
    that every table in a batch has the same support/query split; no split is
    silently taken from another table.  ``steps`` counts completed optimizer
    updates, so a resumed run reads only the remaining dump batches.
    """
    if steps <= 0 or batch_size <= 0 or save_every <= 0 or log_every <= 0:
        raise ValueError(
            "steps, batch_size, save_every, and log_every must be positive"
        )
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; use --device cpu only for a smoke test"
        )

    dump_path = Path(dump).expanduser().resolve()
    output_dir = Path(checkpoint_dir).expanduser().resolve()
    latest = output_dir / "latest.pt"
    if not dump_path.is_file():
        raise FileNotFoundError(f"Nano prior dump not found: {dump_path}")
    if not resume and latest.exists():
        raise FileExistsError(
            f"{latest} exists; pass --resume to continue rather than overwrite it"
        )
    if resume and not latest.is_file():
        raise FileNotFoundError(f"No checkpoint to resume: {latest}")

    # Open once for metadata before model initialization; iterating later
    # starts at the checkpoint's completed-step index.
    metadata_loader = NanoPriorDumpLoader(
        dump_path, num_steps=steps, batch_size=batch_size, start_step=0, device="cpu"
    )
    dump_metadata = metadata_loader.metadata
    if dump_metadata["batch_size"] != batch_size:
        raise ValueError("Requested batch size differs from the generated dump")
    if dump_metadata["rows"] != 150 or dump_metadata["features"] != 5:
        print(
            "Warning: this dump does not use the paper's 150-row, 5-feature task size; "
            "treat this as a smoke or ablation run, not the paper-sized experiment.",
            flush=True,
        )
    if steps != 2500 or batch_size != 32:
        print(
            "Warning: steps or batch size differs from nanoTabPFN's paper experiment.",
            flush=True,
        )

    model_config = dict(PAPER_MODEL_CONFIG)
    train_config = {
        "seed": seed,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": "schedulefree.AdamWScheduleFree",
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "loss": "query_cross_entropy",
        "precision": "float32",
        "dump_path": str(dump_path),
        "dump_sha256": _sha256_file(dump_path),
    }

    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    compute_device = torch.device(device)
    model = NanoTabPFNModel(**model_config).to(compute_device).float()
    optimizer = _schedulefree_adamw(model.parameters(), learning_rate=learning_rate)
    start_step = 0
    if resume:
        checkpoint = torch.load(latest, map_location="cpu", weights_only=True)
        if checkpoint.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"Not a compatible Nano-Graph-U checkpoint: {latest}")
        if checkpoint.get("model_config") != model_config:
            raise ValueError("Checkpoint model configuration differs from this trainer")
        if checkpoint.get("dump_metadata") != dump_metadata:
            raise ValueError(
                "Checkpoint was trained on a different prior dump/configuration"
            )
        if checkpoint.get("train_config") != train_config:
            raise ValueError("Checkpoint training configuration differs from this run")
        start_step = int(checkpoint["step"])
        if not 0 <= start_step <= steps:
            raise ValueError(f"Checkpoint step {start_step} is outside 0..{steps}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if (
            torch.cuda.is_available()
            and checkpoint.get("cuda_rng_state_all") is not None
        ):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        numpy_rng_state = checkpoint["numpy_rng_state"]
        np.random.set_state(
            (
                numpy_rng_state[0],
                np.asarray(numpy_rng_state[1], dtype=np.uint32),
                numpy_rng_state[2],
                numpy_rng_state[3],
                numpy_rng_state[4],
            )
        )
        random.setstate(checkpoint["python_rng_state"])

    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer.train()
    model.train()
    if start_step == steps:
        print(f"Already completed {steps} updates: {latest}", flush=True)
        return latest

    print(
        f"Nano-Graph-U: {sum(p.numel() for p in model.parameters()):,} parameters; "
        f"updates {start_step + 1}..{steps}; batch={batch_size}; device={compute_device}",
        flush=True,
    )
    loader = NanoPriorDumpLoader(
        dump_path,
        num_steps=steps,
        batch_size=batch_size,
        start_step=start_step,
        device="cpu",
    )
    for step, batch in enumerate(loader, start=start_step + 1):
        x = batch["x"].to(device=compute_device, dtype=torch.float32)
        y = batch["y"].to(device=compute_device, dtype=torch.long)
        split = int(batch["train_test_split_index"])
        if not 0 < split < y.shape[1]:
            raise ValueError(
                f"Invalid shared support/query boundary {split} at step {step}"
            )
        if x.shape[0] != batch_size or y.shape != x.shape[:2]:
            raise ValueError(
                f"Unexpected Nano batch shape at step {step}: {x.shape}, {y.shape}"
            )
        if not torch.all((y >= 0) & (y < model_config["num_outputs"])):
            raise ValueError(f"Labels outside binary class range at step {step}")

        # The only labels supplied to Nano are from rows before `split`.
        # Query labels remain exclusively in the loss target.
        optimizer.zero_grad(set_to_none=True)
        logits = model((x, y[:, :split].float()), train_test_split_index=split)
        query_targets = y[:, split:]
        expected_shape = (*query_targets.shape, model_config["num_outputs"])
        if tuple(logits.shape) != expected_shape:
            raise ValueError(
                f"Nano output shape {tuple(logits.shape)} != expected {expected_shape}"
            )
        loss = F.cross_entropy(logits.reshape(-1, 2), query_targets.reshape(-1))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite query loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step == 1 or step % log_every == 0 or step == steps:
            print(f"step={step}/{steps} query_nll={loss.item():.6f}", flush=True)
        if step % save_every == 0 or step == steps:
            # Schedule-free AdamW uses averaged (evaluation) weights after
            # optimizer.eval(); save those for standalone evaluation.  Its
            # optimizer state is saved too so optimizer.train() can resume.
            optimizer.eval()
            try:
                payload = _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    model_config=model_config,
                    dump_metadata=dump_metadata,
                    train_config=train_config,
                )
                numbered = output_dir / f"step-{step}.pt"
                _atomic_save(payload, numbered)
                _atomic_save(payload, latest)
            finally:
                optimizer.train()
            print(f"checkpoint={numbered}", flush=True)
    return latest
