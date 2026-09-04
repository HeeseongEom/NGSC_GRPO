#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from common import build_dataset_split, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    datasets = [args.dataset] if args.dataset else cfg["datasets"]
    for dataset in datasets:
        for count in cfg["split"]["train_pair_counts"]:
            root = build_dataset_split(cfg, dataset, int(count), force=args.force)
            metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
            print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

