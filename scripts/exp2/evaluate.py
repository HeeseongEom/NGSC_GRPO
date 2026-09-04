#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from common import ROOT, evaluate_checkpoint, load_exp2_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "exp2.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("internal", "external"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_exp2_config(args.config)
    print(evaluate_checkpoint(cfg, args.checkpoint, args.split, args.device, args.force))


if __name__ == "__main__":
    main()
