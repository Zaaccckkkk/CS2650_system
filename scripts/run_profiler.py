#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from phase1_graph_profiler import GraphProfiler
from phase1_graph_profiler.models import build_model_bundle


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Phase 1 graph profiler for one iteration.")
    p.add_argument("--model", required=True, choices=["resnet152", "bert"])
    p.add_argument("--batch-size", type=int, help="Run profiling for one mini-batch size.")
    p.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        help="Run profiling for multiple mini-batch sizes and generate the sweep graph.",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default="outputs")
    return p.parse_args()


def run_single_profile(
    model_name: str,
    batch_size: int,
    device: str,
    output_root: Path,
) -> dict:
    """
    Run the existing profiler once for one batch size.
    Save outputs under: output_root/model_name/bs_{batch_size}
    Return a small summary dict containing peak memory info.
    """
    out_dir = output_root / model_name / f"bs_{batch_size}"

    bundle = build_model_bundle(model_name, batch_size, device)
    optimizer = bundle.optimizer_ctor(bundle.model.parameters())

    profiler = GraphProfiler(
        model=bundle.model,
        optimizer=optimizer,
        criterion=bundle.criterion,
        output_dir=out_dir,
        model_name=model_name,
        device=device,
    )

    # Warm up once before the profiled iteration.
    _ = profiler.warmup_one_iteration(batch=bundle.batch, target=bundle.target)
    loss = profiler.profile_one_iteration(batch=bundle.batch, target=bundle.target)
    artifacts = profiler.write_artifacts()

    # Read summary.json to extract peak memory.
    summary_path = Path(artifacts.summary_path)
    with summary_path.open("r") as f:
        summary = json.load(f)

    peak_bytes = summary["peak_breakdown"]["total_live_bytes"]
    peak_mb = peak_bytes / (1024 ** 2)

    result = {
        "model_name": model_name,
        "device": device,
        "batch_size": batch_size,
        "loss": float(loss.item()),
        "peak_memory_bytes": peak_bytes,
        "peak_memory_mb": peak_mb,
        "summary_json": str(summary_path),
        "peak_breakdown_json": str(artifacts.peak_breakdown_path),
    }

    print(f"[done] model={model_name}, bs={batch_size}, loss={float(loss.item()):.6f}, peak={peak_mb:.2f} MB")
    return result


def write_sweep_csv(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_name",
        "device",
        "batch_size",
        "loss",
        "peak_memory_bytes",
        "peak_memory_mb",
        "summary_json",
        "peak_breakdown_json",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def write_sweep_summary(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": results[0]["model_name"] if results else None,
        "device": results[0]["device"] if results else None,
        "num_runs": len(results),
        "batch_sizes": [r["batch_size"] for r in results],
        "peak_memory_bytes": [r["peak_memory_bytes"] for r in results],
        "peak_memory_mb": [r["peak_memory_mb"] for r in results],
        "results": results,
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def plot_peak_memory_vs_batch_size(results: list[dict], path: Path) -> None:
    """
    Generate:
    Peak Memory Consumption vs Mini-Batch Size (w/o AC)
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    results = sorted(results, key=lambda x: x["batch_size"])
    batch_sizes = [r["batch_size"] for r in results]
    peak_mb = [r["peak_memory_mb"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.bar([str(bs) for bs in batch_sizes], peak_mb)
    plt.title("Peak Memory Consumption vs Mini-Batch Size (w/o AC)")
    plt.xlabel("Mini-Batch Size")
    plt.ylabel("Peak Memory Consumption (MB)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def run_batch_size_sweep(
    model_name: str,
    batch_sizes: list[int],
    device: str,
    output_root: Path,
) -> list[dict]:
    """
    Run the profiler for multiple batch sizes and save:
    - one normal profiler output directory per batch size
    - one aggregate CSV
    - one aggregate summary JSON
    - one final bar graph
    """
    results = []

    for bs in batch_sizes:
        result = run_single_profile(
            model_name=model_name,
            batch_size=bs,
            device=device,
            output_root=output_root,
        )
        results.append(result)

    model_dir = output_root / model_name
    write_sweep_csv(results, model_dir / "batch_size_sweep.csv")
    write_sweep_summary(results, model_dir / "batch_size_sweep_summary.json")
    plot_peak_memory_vs_batch_size(results, model_dir / "peak_memory_vs_batch_size.png")

    print(f"[saved] {model_dir / 'batch_size_sweep.csv'}")
    print(f"[saved] {model_dir / 'batch_size_sweep_summary.json'}")
    print(f"[saved] {model_dir / 'peak_memory_vs_batch_size.png'}")
    return results


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)

    # Validate arguments:
    # exactly one of --batch-size or --batch-sizes should be provided.
    if (args.batch_size is None) == (args.batch_sizes is None):
        raise ValueError("Please provide exactly one of --batch-size or --batch-sizes.")

    # Single-run mode: preserve existing behavior.
    if args.batch_size is not None:
        result = run_single_profile(
            model_name=args.model,
            batch_size=args.batch_size,
            device=args.device,
            output_root=output_root,
        )
        print(json.dumps(result, indent=2))
        return

    # Sweep mode: new Phase 1(b) flow.
    run_batch_size_sweep(
        model_name=args.model,
        batch_sizes=args.batch_sizes,
        device=args.device,
        output_root=output_root,
    )


if __name__ == "__main__":
    main()