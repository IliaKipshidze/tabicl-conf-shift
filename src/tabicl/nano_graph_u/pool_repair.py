"""Adopt valid pool batches after the narrowly scoped nonfinite-U retry fix."""

from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
from pathlib import Path

from . import pool_experiment as experiment
from .data import NanoPriorDumpLoader, _metadata, _require_h5py
from .sampled import _lock

_BASE_COMMIT = "23fd4230c89f2a0f311af5ce738baa75fc9b185e"
_PATCHED_SOURCES = {
    "prior/_dataset.py": "8e453a169f22b5d24a0d1399367fc5b7cbe50529d6362a6cfdf8a0c6081f3393",
    "prior/graph_lib/_pool.py": "a2961c2221d81ac37de4e0a858688e1294809668f6352b81e97aa27ef5d9caf4",
}
_AUDIT_ATTRIBUTE = "pool_nonfinite_repair_json"


def _approved_legacy_hashes() -> set[str]:
    """Reconstruct only the source version preceding this exact retry fix.

    Other generator files retain their current bytes. A change to any of them
    therefore prevents a match to a bank generated before that change.
    """
    package = Path(__file__).resolve().parents[1]
    repository = package.parents[1]
    sources = sorted(
        [package / "nano_graph_u/data.py", *package.joinpath("prior").rglob("*.py")],
        key=lambda path: path.as_posix(),
    )
    previous = {}
    for relative, approved in _PATCHED_SOURCES.items():
        current = (package / relative).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(current).hexdigest() != approved:
            raise ValueError(
                "Pool repair is approved only for the exact nonfinite-U retry fix"
            )
        result = subprocess.run(
            ["git", "show", f"{_BASE_COMMIT}:src/tabicl/{relative}"],
            cwd=repository,
            check=True,
            capture_output=True,
        )
        previous[relative] = result.stdout.replace(b"\r\n", b"\n")
    fingerprints = set()
    # Git stores LF; an earlier Windows checkout can contain CRLF per file.
    for crlf in itertools.product((False, True), repeat=len(previous)):
        replacements = {
            name: contents.replace(b"\n", b"\r\n") if windows else contents
            for (name, contents), windows in zip(previous.items(), crlf)
        }
        digest = hashlib.sha256()
        for source in sources:
            relative = source.relative_to(package).as_posix()
            digest.update(relative.encode("utf-8") + b"\0")
            digest.update(replacements.get(relative, source.read_bytes()))
            digest.update(b"\0")
        fingerprints.add(digest.hexdigest())
    return fingerprints


def _expected(spec):
    return _metadata(
        experiment.prior_config(spec["condition"], spec["mode"]),
        steps=spec["steps"],
        batch_size=experiment.BATCH_SIZE,
        seed=spec["seed"],
        rows=experiment.ROWS,
        features=experiment.FEATURES,
        min_train_size=0.3,
        max_train_size=0.9,
        n_jobs=1,
    )


def repair_banks(root):
    """Update provenance only; leave task tensors and commit markers untouched.

    The former implementation aborted before accepting a nonfinite U pool, so
    every committed batch already passed the same finite-pool requirement.
    """
    root = experiment._initialize(root)
    h5py = _require_h5py()
    counts = {"repaired": 0, "already_current": 0, "missing": 0}
    legacy_hashes = None
    for task_id in range(1, experiment.BANK_COUNT + 1):
        spec = experiment.bank_spec(root, task_id)
        path = Path(spec["dump"])
        with _lock(root / "locks" / f"bank_{task_id}.lock"):
            if not path.exists():
                counts["missing"] += 1
                continue
            loader = NanoPriorDumpLoader(path, batch_size=experiment.BATCH_SIZE)
            expected = _expected(spec)
            if not 0 <= loader.committed_steps <= spec["steps"]:
                raise ValueError(f"Invalid committed batch count: {path}")
            if loader.metadata == expected:
                counts["already_current"] += 1
                continue
            old_hash = loader.metadata.get("generator_source_sha256")
            if (
                dict(
                    loader.metadata,
                    generator_source_sha256=expected["generator_source_sha256"],
                )
                != expected
            ):
                raise ValueError(f"Cannot repair differing generation settings: {path}")
            if legacy_hashes is None:
                legacy_hashes = _approved_legacy_hashes()
            if old_hash not in legacy_hashes:
                raise ValueError(
                    f"Cannot repair an unrecognized generator version: {path}"
                )
            if any(
                any((root / folder).rglob("*.pt"))
                for folder in ("checkpoints", "winners")
            ):
                raise ValueError(
                    "Repair must run before training: existing checkpoints pin bank hashes"
                )
            with h5py.File(path, "r+") as stream:
                audit = {
                    "format": "nano_graph_u_pool_nonfinite_repair_v1",
                    "original_metadata": loader.metadata,
                    "original_metadata_json": stream.attrs["metadata_json"],
                    "committed_steps": loader.committed_steps,
                    "previous_source_sha256": old_hash,
                    "new_source_sha256": expected["generator_source_sha256"],
                }
                audit_json = json.dumps(audit, sort_keys=True, allow_nan=False)
                if (
                    _AUDIT_ATTRIBUTE in stream.attrs
                    and stream.attrs[_AUDIT_ATTRIBUTE] != audit_json
                ):
                    raise ValueError(f"Conflicting pool repair audit: {path}")
                stream.attrs[_AUDIT_ATTRIBUTE] = audit_json
                stream.flush()
                stream.attrs["metadata_json"] = json.dumps(
                    expected, sort_keys=True, allow_nan=False
                )
                stream.flush()
            experiment._validate_bank(spec, complete=False)
            counts["repaired"] += 1
            print(
                f"POOL_BANK_REPAIRED task={task_id} committed={loader.committed_steps}/{spec['steps']} dump={path}",
                flush=True,
            )
    print(
        "POOL_REPAIR_COMPLETE "
        + " ".join(f"{key}={value}" for key, value in counts.items()),
        flush=True,
    )
    return counts
