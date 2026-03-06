# Phase 1 Requirement Mapping (from project PDF)

## Requirement 1
Collect computation time and memory usage per operator in topological execution order.

- Implemented by: `phase1_graph_profiler/core.py`
- Artifacts: `nodes.csv`
- Fields used: `duration_us`, `output_bytes`, `node_id`, `phase`

## Requirement 2
Categorize each op input/output as parameter, gradient, activation, optimizer state, or other.

- Implemented by: `GraphProfiler._categorize_tensor`
- Artifacts: `tensors.csv`
- Fields used: `category`, `created_phase`, `producer_node_id`

## Requirement 3
Static activation analysis for first/last use across forward/backward.

- Implemented by: `GraphProfiler._finalize_tensor_use_ranges`
- Artifacts: `activation_liveness.csv`
- Fields used: `first_use_node_id`, `last_use_node_id`

## Requirement 4
Generate peak-memory breakdown graph from collected statistics.

- Implemented by: `GraphProfiler._compute_memory_timeline`, `_maybe_plot_peak_breakdown`
- Artifacts: `peak_breakdown.json`, `memory_timeline.csv`, `peak_memory_breakdown.png`

## Midway check-in support

For deliverables 4(a), 4(b) without AC:

- Run batch sweep: `scripts/sweep_phase1.py`
- Outputs:
  - `phase1_baseline_sweep.csv`
  - `<model>_peak_memory_vs_batch_wo_ac.png`
  - `<model>_latency_vs_batch_wo_ac.png`
