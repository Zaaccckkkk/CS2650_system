#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List

import torch

from phase1_graph_profiler import GraphProfiler
from phase1_graph_profiler.models import build_model_bundle


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep batch sizes for Phase 1 baseline (no AC).")
    p.add_argument("--model", required=True, choices=["resnet152", "bert"])
    p.add_argument("--batch-sizes", required=True, nargs="+", type=int)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default="outputs")
    return p.parse_args()


def maybe_plot(csv_path: Path, out_dir: Path, model: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    rows: List[dict] = []
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    x = [int(r["batch_size"]) for r in rows]
    peak = [int(r["peak_live_bytes"]) for r in rows]
    latency_ms = [float(r["iteration_time_ms"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([str(v) for v in x], peak)
    ax.set_title(f"Peak Memory vs Batch Size (w/o AC) - {model}")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Peak live bytes")
    fig.tight_layout()
    fig.savefig(out_dir / f"{model}_peak_memory_vs_batch_wo_ac.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, latency_ms, marker="o")
    ax.set_title(f"Iteration Latency vs Batch Size (w/o AC) - {model}")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Iteration time (ms)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{model}_latency_vs_batch_wo_ac.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root_out = Path(args.output_dir)
    sweep_dir = root_out / args.model / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for bs in args.batch_sizes:
        bundle = build_model_bundle(args.model, bs, args.device)
        optimizer = bundle.optimizer_ctor(bundle.model.parameters())

        run_out = root_out / args.model / f"bs_{bs}"
        profiler = GraphProfiler(
            model=bundle.model,
            optimizer=optimizer,
            criterion=bundle.criterion,
            output_dir=run_out,
            model_name=args.model,
            device=args.device,
        )
        _ = profiler.profile_one_iteration(batch=bundle.batch, target=bundle.target)
        artifacts = profiler.write_artifacts()

        summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "model": args.model,
                "batch_size": bs,
                "peak_live_bytes": summary["peak_breakdown"]["total_live_bytes"],
                "iteration_time_ms": summary["total_time_us"] / 1000.0,
                "summary_json": str(artifacts.summary_path),
            }
        )

    csv_path = sweep_dir / "phase1_baseline_sweep.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    maybe_plot(csv_path=csv_path, out_dir=sweep_dir, model=args.model)
    print(f"wrote sweep csv: {csv_path}")


if __name__ == "__main__":
    main()
