#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from common import evaluate_checkpoint, evaluate_fixed, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--fixed", action="store_true")
    parser.add_argument("--dataset")
    parser.add_argument("--train-pairs", type=int, choices=(32, 128))
    parser.add_argument("--split", choices=("internal", "external"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    if args.fixed:
        if args.dataset is None or args.train_pairs is None:
            parser.error("--fixed requires --dataset and --train-pairs")
        path = evaluate_fixed(
            cfg, args.dataset, args.train_pairs, args.split, args.device, force=args.force
        )
    else:
        if args.checkpoint is None:
            parser.error("--checkpoint is required unless --fixed is used")
        path = evaluate_checkpoint(cfg, args.checkpoint, args.split, args.device, force=args.force)
    print(path)


if __name__ == "__main__":
    main()

