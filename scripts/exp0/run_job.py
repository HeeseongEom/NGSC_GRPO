#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from build_matrix import build_rows
from common import (
    evaluate_checkpoint,
    evaluate_fixed,
    load_config,
    train_policy,
)
from upper_bound import run as run_upper_bound


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    jobs = {row["job_id"]: row for row in build_rows(cfg)}
    if args.job_id not in jobs:
        raise KeyError(f"Unknown job {args.job_id}; run build_matrix.py to inspect valid IDs")
    job = jobs[args.job_id]
    dataset = job["dataset"]
    train_pairs = int(job["train_pairs"])
    outputs = []
    if job["job_type"] == "upper_bound":
        outputs.append(str(run_upper_bound(cfg, dataset, args.device, force=args.force)))
    else:
        group_size = int(job["group_size"])
        # The fixed baseline is group-size independent.  Evaluate it once for the
        # smallest-group job instead of repeating the same full external pass 3x.
        if group_size == min(int(value) for value in cfg["optimization"]["group_sizes"]):
            evaluate_fixed(cfg, dataset, train_pairs, "internal", args.device, force=args.force)
            evaluate_fixed(cfg, dataset, train_pairs, "external", args.device, force=args.force)
        for kind in ("global", "cnn"):
            checkpoint = train_policy(
                cfg, dataset, train_pairs, group_size, kind, args.device, force=args.force
            )
            outputs.append(str(checkpoint))
            outputs.append(str(evaluate_checkpoint(
                cfg, checkpoint, "internal", args.device, force=args.force
            )))
            outputs.append(str(evaluate_checkpoint(
                cfg, checkpoint, "external", args.device, force=args.force
            )))
    print(json.dumps({"job": job, "outputs": outputs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
