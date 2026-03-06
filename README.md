# CS2650 Systems Project - Phase 1 (Graph Profiler)

This repository contains a Phase 1 implementation for the CS265 systems project:

- Builds a dynamic computation graph over one training iteration (`forward + backward + optimizer`)
- Profiles per-operator compute time and output memory bytes
- Categorizes tensors: `parameter`, `gradient`, `activation`, `optimizer_state`, `other`
- Runs static activation analysis (`first_use_node_id`, `last_use_node_id`)
- Produces a peak memory breakdown artifact and optional plot

## Structure

- `phase1_graph_profiler/core.py`: profiler + graph/tensor analysis
- `phase1_graph_profiler/models.py`: model/input factories (`resnet152`, `bert`)
- `scripts/run_profiler.py`: one-run profiler entrypoint
- `scripts/sweep_phase1.py`: batch-size sweep to build Phase 1 baseline plots (w/o AC)
- `docs/PHASE1_REQUIREMENT_MAPPING.md`: requirement-to-artifact mapping

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Phase 1 Profiler (single run)

```bash
python3 scripts/run_profiler.py --model resnet152 --batch-size 4 --device cuda --output-dir outputs
python3 scripts/run_profiler.py --model bert --batch-size 8 --device cuda --output-dir outputs
```

## Sweep Batch Sizes (w/o AC)

```bash
python3 scripts/sweep_phase1.py --model resnet152 --batch-sizes 1 2 4 8 --device cuda --output-dir outputs
python3 scripts/sweep_phase1.py --model bert --batch-sizes 2 4 8 16 --device cuda --output-dir outputs
```

## Output artifacts (per run)

Generated in `outputs/<model>/bs_<N>/`:

- `nodes.csv`: op-level nodes with phase, timing, and memory bytes
- `edges.csv`: producer-consumer dataflow edges
- `tensors.csv`: tensor metadata and category labels
- `activation_liveness.csv`: activation first/last-use static analysis
- `memory_timeline.csv`: simulated live-memory timeline by category
- `peak_breakdown.json`: peak-memory composition by category
- `summary.json`: run summary with phase times and graph stats
- `peak_memory_breakdown.png`: optional bar chart (if matplotlib available)

Sweep results go to `outputs/<model>/sweep/`.
