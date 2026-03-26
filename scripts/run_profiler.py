#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

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


def bytes_to_mb(x: int | float) -> float:
    return x / (1024 ** 2)


def run_single_profile(
    model_name: str,
    batch_size: int,
    device: str,
    output_root: Path,
) -> dict:
    """
    Run the existing profiler once for one batch size.
    Save outputs under: output_root/model_name/bs_{batch_size}
    Return a summary dict containing peak memory info and its breakdown.
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

    # Read summary.json to extract peak memory and breakdown.
    summary_path = Path(artifacts.summary_path)
    with summary_path.open("r") as f:
        summary = json.load(f)

    peak = summary["peak_breakdown"]

    parameter_bytes = int(peak.get("parameter_bytes", 0))
    gradient_bytes = int(peak.get("gradient_bytes", 0))
    activation_bytes = int(peak.get("activation_bytes", 0))
    optimizer_state_bytes = int(peak.get("optimizer_state_bytes", 0))
    other_bytes = int(peak.get("other_bytes", 0))
    peak_bytes = int(peak.get("total_live_bytes", 0))

    result = {
        "model_name": model_name,
        "device": device,
        "batch_size": batch_size,
        "loss": float(loss.item()),

        "parameter_bytes": parameter_bytes,
        "gradient_bytes": gradient_bytes,
        "activation_bytes": activation_bytes,
        "optimizer_state_bytes": optimizer_state_bytes,
        "other_bytes": other_bytes,
        "peak_memory_bytes": peak_bytes,

        "parameter_mb": bytes_to_mb(parameter_bytes),
        "gradient_mb": bytes_to_mb(gradient_bytes),
        "activation_mb": bytes_to_mb(activation_bytes),
        "optimizer_state_mb": bytes_to_mb(optimizer_state_bytes),
        "other_mb": bytes_to_mb(other_bytes),
        "peak_memory_mb": bytes_to_mb(peak_bytes),

        "summary_json": str(summary_path),
        "peak_breakdown_json": str(artifacts.peak_breakdown_path),
    }

    print(
        f"[done] model={model_name}, bs={batch_size}, "
        f"loss={float(loss.item()):.6f}, peak={result['peak_memory_mb']:.2f} MB"
    )
    return result


def write_sweep_csv(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_name",
        "device",
        "batch_size",
        "loss",
        "parameter_bytes",
        "gradient_bytes",
        "activation_bytes",
        "optimizer_state_bytes",
        "other_bytes",
        "peak_memory_bytes",
        "parameter_mb",
        "gradient_mb",
        "activation_mb",
        "optimizer_state_mb",
        "other_mb",
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

    results = sorted(results, key=lambda x: x["batch_size"])

    payload = {
        "model_name": results[0]["model_name"] if results else None,
        "device": results[0]["device"] if results else None,
        "num_runs": len(results),
        "batch_sizes": [r["batch_size"] for r in results],
        "parameter_mb": [r["parameter_mb"] for r in results],
        "gradient_mb": [r["gradient_mb"] for r in results],
        "activation_mb": [r["activation_mb"] for r in results],
        "optimizer_state_mb": [r["optimizer_state_mb"] for r in results],
        "other_mb": [r["other_mb"] for r in results],
        "peak_memory_mb": [r["peak_memory_mb"] for r in results],
        "results": results,
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def plot_peak_memory_vs_batch_size(results: list[dict], path: Path) -> None:
    """
    Generate a stacked bar chart:
    Peak Memory Consumption vs Mini-Batch Size (w/o AC)

    Each bar includes the peak-memory breakdown:
    - parameter
    - gradient
    - activation
    - optimizer_state
    - other
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    results = sorted(results, key=lambda x: x["batch_size"])
    labels = [str(r["batch_size"]) for r in results]

    parameter_mb = [r["parameter_mb"] for r in results]
    gradient_mb = [r["gradient_mb"] for r in results]
    activation_mb = [r["activation_mb"] for r in results]
    optimizer_state_mb = [r["optimizer_state_mb"] for r in results]
    other_mb = [r["other_mb"] for r in results]

    x = list(range(len(labels)))

    plt.figure(figsize=(10, 6))

    b1 = plt.bar(x, parameter_mb, label="parameter")
    b2 = plt.bar(x, gradient_mb, bottom=parameter_mb, label="gradient")

    bottom_3 = [parameter_mb[i] + gradient_mb[i] for i in range(len(x))]
    b3 = plt.bar(x, activation_mb, bottom=bottom_3, label="activation")

    bottom_4 = [bottom_3[i] + activation_mb[i] for i in range(len(x))]
    b4 = plt.bar(x, optimizer_state_mb, bottom=bottom_4, label="optimizer_state")

    bottom_5 = [bottom_4[i] + optimizer_state_mb[i] for i in range(len(x))]
    b5 = plt.bar(x, other_mb, bottom=bottom_5, label="other")

    totals = [
        parameter_mb[i] + gradient_mb[i] + activation_mb[i] + optimizer_state_mb[i] + other_mb[i]
        for i in range(len(x))
    ]

    # Put total peak memory labels above each stacked bar.
    for i, total in enumerate(totals):
        plt.text(i, total, f"{total:.1f}", ha="center", va="bottom", fontsize=9)

    plt.xticks(x, labels)
    plt.title("Peak Memory Consumption vs Mini-Batch Size (w/o AC)")
    plt.xlabel("Mini-Batch Size")
    plt.ylabel("Peak Memory Consumption (MB)")
    plt.legend()
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
    - one final stacked bar graph
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

    # Exactly one of --batch-size or --batch-sizes should be provided.
    if (args.batch_size is None) == (args.batch_sizes is None):
        raise ValueError("Please provide exactly one of --batch-size or --batch-sizes.")

    # Single-run mode.
    if args.batch_size is not None:
        result = run_single_profile(
            model_name=args.model,
            batch_size=args.batch_size,
            device=args.device,
            output_root=output_root,
        )
        print(json.dumps(result, indent=2))
        return

    # Sweep mode.
    run_batch_size_sweep(
        model_name=args.model,
        batch_sizes=args.batch_sizes,
        device=args.device,
        output_root=output_root,
    )


if __name__ == "__main__":
    main()