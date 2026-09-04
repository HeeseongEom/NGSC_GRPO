#!/usr/bin/env python3
"""Run the 56 EXP0 jobs with configurable independent workers per GPU."""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

from build_matrix import build_rows
from common import load_config, output_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--gpus", nargs="+", default=None)
    parser.add_argument("--job-type", choices=("upper_bound", "ablation"))
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--workers-per-gpu", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    gpus = args.gpus or [str(value) for value in cfg["runtime"]["gpu_ids"]]
    workers_per_gpu = int(
        args.workers_per_gpu or cfg["runtime"].get("workers_per_gpu", 1)
    )
    if workers_per_gpu < 1:
        raise ValueError("workers_per_gpu must be positive")
    selected = [row for row in build_rows(cfg) if args.job_type is None or row["job_type"] == args.job_type]
    if args.max_jobs is not None:
        selected = selected[: args.max_jobs]
    jobs = queue.Queue()
    for row in selected:
        jobs.put(row)
    log_root = output_root(cfg) / "runner_logs" / "jobs"
    log_root.mkdir(parents=True, exist_ok=True)
    failures = []
    lock = threading.Lock()

    def worker(gpu: str) -> None:
        while True:
            try:
                row = jobs.get_nowait()
            except queue.Empty:
                return
            command = [
                sys.executable, str(Path(__file__).with_name("run_job.py")),
                "--job-id", row["job_id"], "--device", f"cuda:{gpu}",
            ]
            if args.config:
                command.extend(("--config", args.config))
            if args.force:
                command.append("--force")
            with (log_root / f"{row['job_id']}.log").open("w", encoding="utf-8") as handle:
                result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=os.environ.copy())
            if result.returncode:
                with lock:
                    failures.append((row["job_id"], result.returncode))
            jobs.task_done()

    threads = [
        threading.Thread(target=worker, args=(gpu,), daemon=False)
        for gpu in gpus for _ in range(workers_per_gpu)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError(f"Job failures: {failures}; inspect {log_root}")
    print(
        f"Completed {len(selected)} jobs on GPUs {gpus} "
        f"with {workers_per_gpu} workers per GPU"
    )


if __name__ == "__main__":
    main()
