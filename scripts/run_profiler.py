#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from phase1_graph_profiler import GraphProfiler
from phase1_graph_profiler.models import build_model_bundle


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Phase 1 graph profiler for one iteration.")
    p.add_argument("--model", required=True, choices=["resnet152", "bert"])
    p.add_argument("--batch-size", required=True, type=int)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default="outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir) / args.model / f"bs_{args.batch_size}"

    bundle = build_model_bundle(args.model, args.batch_size, args.device)
    optimizer = bundle.optimizer_ctor(bundle.model.parameters())

    profiler = GraphProfiler(
        model=bundle.model,
        optimizer=optimizer,
        criterion=bundle.criterion,
        output_dir=out_dir,
        model_name=args.model,
        device=args.device,
    )
    loss = profiler.profile_one_iteration(batch=bundle.batch, target=bundle.target)
    artifacts = profiler.write_artifacts()

    print(f"loss={float(loss.item()):.6f}")
    print(json.dumps({k: str(v) for k, v in artifacts.__dict__.items()}, indent=2))


if __name__ == "__main__":
    main()
