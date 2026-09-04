#!/usr/bin/env python3
from __future__ import annotations

import argparse

from common import build_dataset_cache, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    if args.dataset not in cfg["datasets"]:
        raise ValueError(args.dataset)
    print(build_dataset_cache(cfg, args.dataset, args.device, force=args.force))


if __name__ == "__main__":
    main()

