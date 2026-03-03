from __future__ import annotations

import argparse
import json
import os
from typing import Any, Callable

from jax import config

from .schema import BenchmarkRecord, BenchmarkResult
from .stats import summarize, throughput
from .timing import timed_trials


def add_harness_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mode", choices=["kernel", "transfer", "e2e"], default="kernel"
    )
    parser.add_argument(
        "--transfer-policy",
        choices=["none", "payload_only", "full_tree"],
        default="none",
    )
    parser.add_argument("--inner-steps", type=int, default=200)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--measure-iters", type=int, default=10)
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Alias for --measure-iters (legacy compatibility).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-float32", action="store_true")
    parser.add_argument("--use-float64", action="store_true")


def configure_precision(args: argparse.Namespace) -> None:
    if args.use_float64 and args.use_float32:
        raise ValueError("Cannot use both --use-float32 and --use-float64")
    if args.use_float64:
        config.update("jax_enable_x64", True)
    if args.use_float32:
        config.update("jax_enable_x64", False)


def _resolved_measure_iters(args: argparse.Namespace) -> int:
    if args.trials is not None:
        return int(args.trials)
    return int(args.measure_iters)


def run_case(
    result: BenchmarkResult,
    name: str,
    params: dict[str, Any],
    payload_items: int,
    fn: Callable[[], Any],
    args: argparse.Namespace,
) -> None:
    measure_iters = _resolved_measure_iters(args)
    durations, bytes_per_iter, bytes_per_sec = timed_trials(
        fn,
        mode=args.mode,
        transfer_policy=args.transfer_policy,
        warmup_iters=args.warmup_iters,
        measure_iters=measure_iters,
    )
    call_ms = [d * 1e3 for d in durations]
    step_ms = [d * 1e3 / max(1, args.inner_steps) for d in durations]
    metrics = {
        **summarize(call_ms, "call_time_ms"),
        **summarize(step_ms, "step_time_ms"),
        **throughput(payload_items, durations),
        "bytes_transferred_per_iter": float(bytes_per_iter),
        "bytes_transferred_per_sec": float(bytes_per_sec),
    }
    result.records.append(
        BenchmarkRecord(
            name=name,
            mode=args.mode,
            transfer_policy=args.transfer_policy,
            params=params,
            metrics=metrics,
        )
    )


def finalize_result(
    result: BenchmarkResult,
    args: argparse.Namespace,
    output_path: str,
    extra_run: dict[str, Any] | None = None,
) -> None:
    result.run = {
        "mode": args.mode,
        "transfer_policy": args.transfer_policy,
        "inner_steps": args.inner_steps,
        "warmup_iters": args.warmup_iters,
        "measure_iters": _resolved_measure_iters(args),
        "seed": args.seed,
    }
    if extra_run:
        result.run.update(extra_run)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"Saved benchmark result: {output_path}")
