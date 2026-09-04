#!/usr/bin/env python3
from __future__ import annotations

import argparse

from common import ROOT, load_exp2_config, train_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "exp2.yaml"))
    parser.add_argument("--method", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--kind", choices=("global", "conditional"), required=True)
    parser.add_argument("--action-set", required=True)
    parser.add_argument("--reward", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--optimization", choices=("trips_replication", "main_optimization"), required=True)
    parser.add_argument("--heldout-domain")
    parser.add_argument("--state-mode", choices=("base11", "prompt12"), default="base11")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_exp2_config(args.config)
    print(train_policy(
        cfg, args.method, args.run_name, args.kind, args.action_set, args.reward, args.seed,
        args.device, args.optimization, args.heldout_domain, args.state_mode, args.force
    ))


if __name__ == "__main__":
    main()
