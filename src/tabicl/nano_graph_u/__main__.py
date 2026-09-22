"""CLI for the separate, paper-sized Nano-Graph-U experiment."""

from __future__ import annotations

import argparse

from tabicl.prior.graph_lib._config import PriorConfig

from .data import generate_dump
from .train import train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tabicl.nano_graph_u",
        description="Generate and train a paper-sized nanoTabPFN with Graph-U tasks.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    generate = subcommands.add_parser(
        "generate", help="Pre-generate Graph-U HDF5 batches"
    )
    generate.add_argument("--dump", required=True, help="New HDF5 dump path")
    generate.add_argument("--steps", type=int, default=2500)
    generate.add_argument("--batch-size", type=int, default=32)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--n-jobs", type=int, default=1)
    generate.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent batches generated in parallel; --n-jobs stays 1 within each batch",
    )
    generate.add_argument("--rows", type=int, default=150)
    generate.add_argument("--features", type=int, default=5)
    generate.add_argument("--min-train-size", type=float, default=0.3)
    generate.add_argument("--max-train-size", type=float, default=0.9)
    generate.add_argument("--query-location", type=float, default=2.0)
    generate.add_argument("--query-scale", type=float, default=1.5)
    generate.add_argument("--structure-mode", choices=("add_root",), default="add_root")
    generate.add_argument("--max-attempts", type=int, default=1000)
    generate.add_argument("--force-gaussian", action="store_true")
    generate.add_argument(
        "--resume", action="store_true", help="Continue an interrupted dump"
    )

    train = subcommands.add_parser("train", help="Train the paper-sized Nano model")
    train.add_argument("--dump", required=True)
    train.add_argument("--checkpoint-dir", required=True)
    train.add_argument("--steps", type=int, default=2500)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="cuda")
    train.add_argument("--learning-rate", type=float, default=0.004)
    train.add_argument("--save-every", type=int, default=250)
    train.add_argument("--log-every", type=int, default=25)
    train.add_argument("--resume", action="store_true", help="Continue from latest.pt")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        prior_config = PriorConfig(
            graph_u_enabled=True,
            graph_u_structure_mode=args.structure_mode,
            graph_u_query_location=args.query_location,
            graph_u_query_scale=args.query_scale,
            graph_u_force_gaussian=args.force_gaussian,
            graph_u_max_attempts=args.max_attempts,
            filter_unpredictable_graphs=True,
            filter_unpredictable_datasets=True,
            allow_act_warping=False,
        )
        generate_dump(
            args.dump,
            steps=args.steps,
            batch_size=args.batch_size,
            seed=args.seed,
            prior_config=prior_config,
            rows=args.rows,
            features=args.features,
            min_train_size=args.min_train_size,
            max_train_size=args.max_train_size,
            n_jobs=args.n_jobs,
            workers=args.workers,
            resume=args.resume,
        )
    elif args.command == "train":
        train_model(
            args.dump,
            args.checkpoint_dir,
            steps=args.steps,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
            learning_rate=args.learning_rate,
            save_every=args.save_every,
            log_every=args.log_every,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
