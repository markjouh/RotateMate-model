#!/usr/bin/env python3
"""Simple parameter sweep orchestrator for RotateMate."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from run import (
    ensure_writable_dir,
    load_config,
    setup_logging,
    step_download,
    step_export,
    step_train,
)


DEFAULT_SWEEP = {
    "learning_rate": [5e-5, 1e-4, 2e-4],
    "weight_decay": [5e-5, 1e-4],
    "seeds": [0, 1],
}


def parse_float_list(values: Sequence[str] | None, fallback: Sequence[float]) -> List[float]:
    if not values:
        return list(fallback)
    return [float(value) for value in values]


def parse_int_list(values: Sequence[str] | None, fallback: Sequence[int]) -> List[int]:
    if not values:
        return list(fallback)
    return [int(value) for value in values]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a parameter sweep for RotateMate.")
    parser.add_argument("--config", default="config.yaml", help="Path to base configuration file")
    parser.add_argument("--learning-rate", nargs="*", default=None, help="Learning rates to sweep")
    parser.add_argument("--weight-decay", nargs="*", default=None, help="Weight decay values to sweep")
    parser.add_argument("--seeds", nargs="*", default=None, help="Random seeds to evaluate")
    parser.add_argument("--export", action="store_true", help="Export each run after training")
    args = parser.parse_args()

    base_config = load_config(args.config)

    lr_values = parse_float_list(args.learning_rate, DEFAULT_SWEEP["learning_rate"])
    wd_values = parse_float_list(args.weight_decay, DEFAULT_SWEEP["weight_decay"])
    seeds = parse_int_list(args.seeds, DEFAULT_SWEEP["seeds"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_root = ensure_writable_dir(Path(base_config["training"]["output_dir"]).parent, "sweep.base_output")
    sweep_root = ensure_writable_dir(base_output_root / f"sweep_{timestamp}", "sweep.root")
    master_log_dir = ensure_writable_dir(sweep_root / "master_logs", "sweep.logs")

    setup_logging(master_log_dir)
    sweep_logger = logging.getLogger("sweep")
    sweep_logger.setLevel(logging.INFO)

    sweep_logger.info("Starting sweep")
    sweep_logger.info("Learning rates: %s", lr_values)
    sweep_logger.info("Weight decays: %s", wd_values)
    sweep_logger.info("Seeds: %s", seeds)

    splits = step_download(base_config)

    combinations = list(itertools.product(lr_values, wd_values, seeds))
    results: List[Dict[str, object]] = []

    for lr, wd, seed in combinations:
        run_name = f"lr{lr:.0e}_wd{wd:.0e}_seed{seed}"
        sweep_logger.info("Running configuration %s", run_name)

        run_dir = ensure_writable_dir(sweep_root / run_name, f"sweep.{run_name}")
        checkpoints_dir = ensure_writable_dir(run_dir / "checkpoints", f"{run_name}.checkpoints")
        logs_dir = ensure_writable_dir(run_dir / "logs", f"{run_name}.logs")

        cfg = deepcopy(base_config)
        cfg["training"]["learning_rate"] = lr
        cfg["training"]["weight_decay"] = wd
        cfg["training"]["output_dir"] = str(checkpoints_dir)
        cfg["training"]["logs_dir"] = str(logs_dir)
        cfg.setdefault("logging", {})["logs_dir"] = str(logs_dir)

        set_seed(seed)

        setup_logging(logs_dir)
        run_logger = logging.getLogger("sweep.run")
        run_logger.setLevel(logging.INFO)
        run_logger.info("Starting run %s", run_name)

        train_results = step_train(cfg, splits)
        best_model = Path(train_results["best_model"]).resolve()
        best_metric = float(train_results["best_metric"])

        run_logger.info("Run %s finished - Val accuracy: %.4f", run_name, best_metric)

        export_paths = {}
        if args.export:
            exports = step_export(cfg, best_model)
            export_paths = {key: str(Path(path).resolve()) for key, path in exports.items()}

        setup_logging(master_log_dir)
        sweep_logger.info(
            "Completed %s | val_acc=%.4f | best_model=%s",
            run_name,
            best_metric,
            best_model,
        )

        result_entry = {
            "run": run_name,
            "learning_rate": lr,
            "weight_decay": wd,
            "seed": seed,
            "val_accuracy": best_metric,
            "best_model": str(best_model),
        }
        if export_paths:
            result_entry["exports"] = export_paths
        results.append(result_entry)

    results.sort(key=lambda item: item["val_accuracy"], reverse=True)

    summary_path = sweep_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump({"results": results}, handle, indent=2)

    # Write a Markdown summary table for convenience.
    summary_md_path = sweep_root / "summary.md"
    with summary_md_path.open("w", encoding="utf-8") as handle:
        handle.write("| Rank | Run | Learning Rate | Weight Decay | Seed | Val Accuracy |\n")
        handle.write("| --- | --- | --- | --- | --- | --- |\n")
        for rank, entry in enumerate(results, start=1):
            handle.write(
                f"| {rank} | {entry['run']} | {entry['learning_rate']:.2e} | "
                f"{entry['weight_decay']:.2e} | {entry['seed']} | {entry['val_accuracy']:.4f} |\n"
            )

    sweep_logger.info("Sweep complete. Top runs:")
    for entry in results[: min(5, len(results))]:
        sweep_logger.info(
            "%s | val_acc=%.4f | lr=%g | wd=%g",
            entry["run"],
            entry["val_accuracy"],
            entry["learning_rate"],
            entry["weight_decay"],
        )

    sweep_logger.info("Summary saved to %s", summary_path)
    sweep_logger.info("Markdown table saved to %s", summary_md_path)


if __name__ == "__main__":
    main()
