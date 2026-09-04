#!/usr/bin/env python3
from __future__ import annotations

import argparse

from common import load_config, train_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train-pairs", type=int, choices=(32, 128), required=True)
    parser.add_argument("--group-size", type=int, choices=(4, 8, 16), required=True)
    parser.add_argument("--kind", choices=("global", "cnn"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    print(train_policy(
        cfg, args.dataset, args.train_pairs, args.group_size, args.kind, args.device, force=args.force
    ))


if __name__ == "__main__":
    main()

