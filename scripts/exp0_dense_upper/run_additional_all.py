#!/usr/bin/env python3
"""Run n32-internal and full-target dense upper bounds on two GPUs."""

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
    gpus = args.gpus or [str(v) for v in cfg["runtime"]["gpu_ids"]]
    jobs = queue.Queue()
    for scope in ("internal_n32", "external_full"):
        for shard in range(2):
            jobs.put((scope, "HAM10000", shard, 2))
    for scope in ("internal_n32", "external_full"):
        for dataset in cfg["datasets"]:
            if dataset != "HAM10000":
                jobs.put((scope, dataset, None, 1))
    log_root = output_root(cfg) / "runner_logs" / "dense_upper_additional"
    log_root.mkdir(parents=True, exist_ok=True)
    failures = []
    lock = threading.Lock()

    def worker(gpu: str) -> None:
        while True:
            try:
                scope, dataset, shard, num_shards = jobs.get_nowait()
            except queue.Empty:
                return
            command = [
                sys.executable, str(Path(__file__).with_name("additional_upper.py")),
                "--scope", scope, "--dataset", dataset, "--device", f"cuda:{gpu}",
            ]
            if shard is not None:
                command.extend(("--shard-index", str(shard), "--num-shards", str(num_shards)))
            if args.config:
                command.extend(("--config", args.config))
            if args.force:
                command.append("--force")
            name = f"{scope}_{dataset}" + ("" if shard is None else f"_shard{shard}")
            with (log_root / f"{name}.log").open("w", encoding="utf-8") as handle:
                result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=os.environ.copy())
            if result.returncode:
                with lock:
                    failures.append((name, result.returncode))
            jobs.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError(f"Additional upper-bound failures: {failures}; inspect {log_root}")
    for scope in ("internal_n32", "external_full"):
        command = [
            sys.executable, str(Path(__file__).with_name("additional_upper.py")),
            "--scope", scope, "--dataset", "HAM10000", "--num-shards", "2", "--merge-shards",
        ]
        if args.config:
            command.extend(("--config", args.config))
        subprocess.run(command, check=True, env=os.environ.copy())
    print("Completed internal n32 and full external-target dense upper bounds")


if __name__ == "__main__":
    main()
