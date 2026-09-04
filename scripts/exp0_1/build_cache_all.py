#!/usr/bin/env python3
"""Build the eight dataset caches with one sequential worker per GPU."""

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
    for dataset in cfg["datasets"]:
        jobs.put(dataset)
    log_root = output_root(cfg) / "runner_logs" / "cache"
    log_root.mkdir(parents=True, exist_ok=True)
    failures = []
    lock = threading.Lock()

    def worker(gpu: str) -> None:
        while True:
            try:
                dataset = jobs.get_nowait()
            except queue.Empty:
                return
            command = [
                sys.executable, str(Path(__file__).with_name("build_cache.py")),
                "--dataset", dataset, "--device", f"cuda:{gpu}",
            ]
            if args.config:
                command.extend(("--config", args.config))
            if args.force:
                command.append("--force")
            with (log_root / f"{dataset}.log").open("w", encoding="utf-8") as handle:
                result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=os.environ.copy())
            if result.returncode:
                with lock:
                    failures.append((dataset, result.returncode))
            jobs.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError(f"Cache failures: {failures}; inspect {log_root}")
    print(f"All {len(cfg['datasets'])} dataset caches are ready in {output_root(cfg) / 'cache'}")


if __name__ == "__main__":
    main()

