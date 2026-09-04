#!/usr/bin/env python3
"""Run one exact dense upper-bound queue per GPU."""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

from common import load_config, output_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--gpus", nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    gpus = args.gpus or [str(value) for value in cfg["runtime"]["gpu_ids"]]
    jobs = queue.Queue()
    # HAM10000 dominates the runtime, so both GPUs evaluate one disjoint half.
    # The remaining seven datasets are queued after the two shards.
    jobs.put(("HAM10000", 0, 2))
    jobs.put(("HAM10000", 1, 2))
    for dataset in cfg["datasets"]:
        if dataset != "HAM10000":
            jobs.put((dataset, None, 1))
    log_root = output_root(cfg) / "runner_logs" / "dense_upper_0p1"
    log_root.mkdir(parents=True, exist_ok=True)
    failures = []
    lock = threading.Lock()

    def worker(gpu: str) -> None:
        while True:
            try:
                dataset, shard_index, num_shards = jobs.get_nowait()
            except queue.Empty:
                return
            command = [
                sys.executable, str(Path(__file__).with_name("dense_upper.py")),
                "--dataset", dataset, "--device", f"cuda:{gpu}",
            ]
            if shard_index is not None:
                command.extend((
                    "--shard-index", str(shard_index), "--num-shards", str(num_shards),
                ))
            if args.config:
                command.extend(("--config", args.config))
            if args.force:
                command.append("--force")
            log_name = dataset if shard_index is None else f"{dataset}_shard{shard_index}"
            with (log_root / f"{log_name}.log").open("w", encoding="utf-8") as handle:
                result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=os.environ.copy())
            if result.returncode:
                with lock:
                    failures.append((log_name, result.returncode))
            jobs.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError(f"Dense upper-bound failures: {failures}; inspect {log_root}")
    merge = [
        sys.executable, str(Path(__file__).with_name("dense_upper.py")),
        "--dataset", "HAM10000", "--num-shards", "2", "--merge-shards",
    ]
    if args.config:
        merge.extend(("--config", args.config))
    subprocess.run(merge, check=True, env=os.environ.copy())
    print(f"Completed dense 0.1 upper bounds for {len(cfg['datasets'])} datasets")


if __name__ == "__main__":
    main()
