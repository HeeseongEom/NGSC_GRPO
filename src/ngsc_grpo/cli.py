from __future__ import annotations

import argparse
import json

from .config import BASELINES, METHODS, load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Leakage-safe GRPO-NGSC feasibility runner")
    parser.add_argument("--config", default="configs/feasibility.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    split = sub.add_parser("make-splits")
    split.add_argument("--force", action="store_true")
    validate = sub.add_parser("validate")
    validate.add_argument("--verify-large-hashes", action="store_true")
    cache = sub.add_parser("cache")
    cache.add_argument("--method", required=True, choices=METHODS)
    cache.add_argument("--role", required=True, choices=("source", "target"))
    cache.add_argument("--device")
    cache.add_argument("--force", action="store_true")
    static = sub.add_parser("static-search")
    static.add_argument("--method", required=True, choices=METHODS)
    static.add_argument("--device")
    static.add_argument("--force", action="store_true")
    train = sub.add_parser("train")
    train.add_argument("--method", required=True, choices=METHODS)
    train.add_argument("--seed", required=True, type=int)
    train.add_argument("--device")
    train.add_argument("--force", action="store_true")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--method", required=True, choices=METHODS)
    evaluate.add_argument("--baseline", required=True, choices=BASELINES)
    evaluate.add_argument("--seed", type=int)
    evaluate.add_argument("--device")
    evaluate.add_argument("--force", action="store_true")
    summarize = sub.add_parser("summarize")
    summarize.add_argument("--method", choices=METHODS)
    summarize.add_argument("--all-methods", action="store_true")
    return parser


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "make-splits":
        from .splits import build_splits
        result = build_splits(cfg, force=args.force)
    elif args.command == "validate":
        from .provenance import validate_project
        result = validate_project(cfg, verify_large_hashes=args.verify_large_hashes)
    elif args.command == "cache":
        from .cache import build_feature_cache
        result = build_feature_cache(cfg, args.method, args.role, args.device, args.force)
    elif args.command == "static-search":
        from .training import run_static_search
        result = run_static_search(cfg, args.method, args.device, args.force)
    elif args.command == "train":
        if args.seed not in [int(value) for value in cfg["grpo"]["seeds"]]:
            raise ValueError(f"seed {args.seed} is not configured in grpo.seeds")
        from .training import train_grpo
        result = train_grpo(cfg, args.method, args.seed, args.device, args.force)
    elif args.command == "evaluate":
        if args.baseline == "conditional_grpo" and args.seed is None:
            raise ValueError("conditional_grpo requires --seed")
        from .evaluation import evaluate_target
        result = evaluate_target(cfg, args.method, args.baseline, args.seed, args.device, args.force)
    elif args.command == "summarize":
        from .reporting import summarize_all, summarize_method
        if args.all_methods:
            result = summarize_all(cfg)
        elif args.method:
            result = summarize_method(cfg, args.method)
        else:
            raise ValueError("summarize requires --method or --all-methods")
    else:
        raise AssertionError(args.command)
    serializable = result if isinstance(result, (dict, list, int, float, bool)) else str(result)
    print(json.dumps({"result": serializable}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
