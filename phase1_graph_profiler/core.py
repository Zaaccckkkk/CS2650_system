from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode


TensorId = int


@dataclass
class NodeRecord:
    node_id: int
    op_name: str
    phase: str
    start_ns: int
    end_ns: int
    duration_us: float
    input_tensor_ids: List[TensorId]
    output_tensor_ids: List[TensorId]
    output_bytes: int


@dataclass
class TensorRecord:
    tensor_id: TensorId
    shape: List[int]
    dtype: str
    bytes: int
    category: str
    producer_node_id: int
    first_use_node_id: int
    last_use_node_id: int
    created_phase: str


@dataclass
class ProfileArtifacts:
    nodes_path: Path
    edges_path: Path
    tensors_path: Path
    activation_liveness_path: Path
    memory_timeline_path: Path
    peak_breakdown_path: Path
    summary_path: Path
    peak_breakdown_plot_path: Optional[Path]


class OpTraceDispatchMode(TorchDispatchMode):
    def __init__(self, profiler: "GraphProfiler") -> None:
        super().__init__()
        self.profiler = profiler

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # type: ignore[override]
        kwargs = kwargs or {}
        in_tensors = [x for x in pytree.tree_leaves((args, kwargs)) if isinstance(x, torch.Tensor)]
        input_ids = [id(t) for t in in_tensors]

        start_ns = time.perf_counter_ns()
        out = func(*args, **kwargs)
        end_ns = time.perf_counter_ns()

        out_tensors = [x for x in pytree.tree_leaves(out) if isinstance(x, torch.Tensor)]
        output_ids = [id(t) for t in out_tensors]
        output_bytes = sum(t.numel() * t.element_size() for t in out_tensors)

        self.profiler._record_op(
            op_name=str(func),
            phase=self.profiler.current_phase,
            start_ns=start_ns,
            end_ns=end_ns,
            input_tensors=in_tensors,
            output_tensors=out_tensors,
            input_ids=input_ids,
            output_ids=output_ids,
            output_bytes=output_bytes,
        )
        return out


class GraphProfiler:
    """Profiles one training iteration and builds a dynamic op graph.

    Graph nodes are executed operators. Directed edges are inferred from tensor
    producer/consumer relationships.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        output_dir: Path | str,
        model_name: str,
        device: str,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.device = device

        self.current_phase = "unknown"
        self.nodes: List[NodeRecord] = []
        self.edges: Set[Tuple[int, int, str]] = set()
        self.tensor_records: Dict[TensorId, TensorRecord] = {}
        self._producer_by_tensor: Dict[TensorId, int] = {}
        self._consumer_nodes_by_tensor: Dict[TensorId, List[int]] = defaultdict(list)

        self._param_ids: Set[TensorId] = set()
        self._grad_ids: Set[TensorId] = set()
        self._optim_state_ids: Set[TensorId] = set()

        self._capture_model_state_ids()

    def _capture_model_state_ids(self) -> None:
        self._param_ids = {id(p) for p in self.model.parameters()}
        self._grad_ids = {id(p.grad) for p in self.model.parameters() if p.grad is not None}
        self._optim_state_ids = set()
        for state in self.optimizer.state.values():
            for v in state.values():
                if isinstance(v, torch.Tensor):
                    self._optim_state_ids.add(id(v))

    @contextmanager
    def phase(self, name: str):
        old = self.current_phase
        self.current_phase = name
        try:
            yield
        finally:
            self.current_phase = old

    def _categorize_tensor(self, t: torch.Tensor) -> str:
        tid = id(t)
        if tid in self._param_ids:
            return "parameter"
        if tid in self._grad_ids:
            return "gradient"
        if tid in self._optim_state_ids:
            return "optimizer_state"
        if t.requires_grad:
            return "activation"
        return "other"

    def _record_op(
        self,
        op_name: str,
        phase: str,
        start_ns: int,
        end_ns: int,
        input_tensors: Sequence[torch.Tensor],
        output_tensors: Sequence[torch.Tensor],
        input_ids: Sequence[TensorId],
        output_ids: Sequence[TensorId],
        output_bytes: int,
    ) -> None:
        node_id = len(self.nodes)
        node = NodeRecord(
            node_id=node_id,
            op_name=op_name,
            phase=phase,
            start_ns=start_ns,
            end_ns=end_ns,
            duration_us=(end_ns - start_ns) / 1000.0,
            input_tensor_ids=list(input_ids),
            output_tensor_ids=list(output_ids),
            output_bytes=output_bytes,
        )
        self.nodes.append(node)

        for tid in input_ids:
            self._consumer_nodes_by_tensor[tid].append(node_id)
            producer = self._producer_by_tensor.get(tid)
            if producer is not None:
                self.edges.add((producer, node_id, "tensor_flow"))

        for t in output_tensors:
            tid = id(t)
            bytes_ = t.numel() * t.element_size()
            category = self._categorize_tensor(t)
            record = self.tensor_records.get(tid)
            if record is None:
                self.tensor_records[tid] = TensorRecord(
                    tensor_id=tid,
                    shape=list(t.shape),
                    dtype=str(t.dtype).replace("torch.", ""),
                    bytes=bytes_,
                    category=category,
                    producer_node_id=node_id,
                    first_use_node_id=node_id,
                    last_use_node_id=node_id,
                    created_phase=phase,
                )
            else:
                record.bytes = bytes_
                record.category = category
                record.last_use_node_id = node_id

            self._producer_by_tensor[tid] = node_id

    def _finalize_tensor_use_ranges(self) -> None:
        self._capture_model_state_ids()
        for tid, rec in self.tensor_records.items():
            if tid in self._param_ids:
                rec.category = "parameter"
            elif tid in self._grad_ids:
                rec.category = "gradient"
            elif tid in self._optim_state_ids:
                rec.category = "optimizer_state"

            consumers = self._consumer_nodes_by_tensor.get(tid, [])
            if consumers:
                rec.first_use_node_id = min(rec.first_use_node_id, min(consumers))
                rec.last_use_node_id = max(rec.last_use_node_id, max(consumers))

    def _compute_memory_timeline(self) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        by_start: Dict[int, List[TensorRecord]] = defaultdict(list)
        by_end: Dict[int, List[TensorRecord]] = defaultdict(list)
        for rec in self.tensor_records.values():
            by_start[rec.producer_node_id].append(rec)
            by_end[rec.last_use_node_id].append(rec)

        live_by_category: Dict[str, int] = defaultdict(int)
        timeline: List[Dict[str, Any]] = []
        peak_total = -1
        peak_breakdown: Dict[str, int] = {}

        for node in self.nodes:
            for rec in by_start.get(node.node_id, []):
                live_by_category[rec.category] += rec.bytes

            total = sum(live_by_category.values())
            point = {
                "node_id": node.node_id,
                "phase": node.phase,
                "op_name": node.op_name,
                "total_live_bytes": total,
                "parameter_bytes": live_by_category.get("parameter", 0),
                "gradient_bytes": live_by_category.get("gradient", 0),
                "activation_bytes": live_by_category.get("activation", 0),
                "optimizer_state_bytes": live_by_category.get("optimizer_state", 0),
                "other_bytes": live_by_category.get("other", 0),
            }
            timeline.append(point)

            if total > peak_total:
                peak_total = total
                peak_breakdown = {
                    "parameter_bytes": point["parameter_bytes"],
                    "gradient_bytes": point["gradient_bytes"],
                    "activation_bytes": point["activation_bytes"],
                    "optimizer_state_bytes": point["optimizer_state_bytes"],
                    "other_bytes": point["other_bytes"],
                    "total_live_bytes": total,
                    "node_id": node.node_id,
                    "phase": node.phase,
                    "op_name": node.op_name,
                }

            for rec in by_end.get(node.node_id, []):
                live_by_category[rec.category] -= rec.bytes

        return timeline, peak_breakdown

    def profile_one_iteration(self, batch: Any, target: torch.Tensor) -> torch.Tensor:
        self.nodes.clear()
        self.edges.clear()
        self.tensor_records.clear()
        self._producer_by_tensor.clear()
        self._consumer_nodes_by_tensor.clear()

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        with OpTraceDispatchMode(self):
            with self.phase("forward"):
                if isinstance(batch, dict):
                    output = self.model(**batch)
                else:
                    output = self.model(batch)
                loss = self.criterion(output, target)

            with self.phase("backward"):
                loss.backward()

            with self.phase("optimizer"):
                self.optimizer.step()

        self._finalize_tensor_use_ranges()
        return loss.detach()

    def write_artifacts(self) -> ProfileArtifacts:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        nodes_path = self.output_dir / "nodes.csv"
        edges_path = self.output_dir / "edges.csv"
        tensors_path = self.output_dir / "tensors.csv"
        activation_liveness_path = self.output_dir / "activation_liveness.csv"
        memory_timeline_path = self.output_dir / "memory_timeline.csv"
        peak_breakdown_path = self.output_dir / "peak_breakdown.json"
        summary_path = self.output_dir / "summary.json"

        self._write_csv(nodes_path, [asdict(n) for n in self.nodes])
        self._write_csv(
            edges_path,
            [
                {"src_node_id": s, "dst_node_id": d, "edge_type": t}
                for (s, d, t) in sorted(self.edges)
            ],
        )
        self._write_csv(tensors_path, [asdict(t) for t in self.tensor_records.values()])

        activation_rows = [
            asdict(t)
            for t in self.tensor_records.values()
            if t.category == "activation"
        ]
        self._write_csv(activation_liveness_path, activation_rows)

        timeline, peak_breakdown = self._compute_memory_timeline()
        self._write_csv(memory_timeline_path, timeline)
        with peak_breakdown_path.open("w", encoding="utf-8") as f:
            json.dump(peak_breakdown, f, indent=2)

        summary = {
            "model_name": self.model_name,
            "device": self.device,
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "num_tensors": len(self.tensor_records),
            "num_activations": len(activation_rows),
            "peak_breakdown": peak_breakdown,
            "total_time_us": sum(n.duration_us for n in self.nodes),
            "phase_time_us": {
                phase: sum(n.duration_us for n in self.nodes if n.phase == phase)
                for phase in ["forward", "backward", "optimizer"]
            },
        }
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        plot_path = self._maybe_plot_peak_breakdown(peak_breakdown)

        return ProfileArtifacts(
            nodes_path=nodes_path,
            edges_path=edges_path,
            tensors_path=tensors_path,
            activation_liveness_path=activation_liveness_path,
            memory_timeline_path=memory_timeline_path,
            peak_breakdown_path=peak_breakdown_path,
            summary_path=summary_path,
            peak_breakdown_plot_path=plot_path,
        )

    def _maybe_plot_peak_breakdown(self, peak_breakdown: Dict[str, Any]) -> Optional[Path]:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            return None

        labels = ["parameter", "gradient", "activation", "optimizer_state", "other"]
        values = [
            peak_breakdown.get("parameter_bytes", 0),
            peak_breakdown.get("gradient_bytes", 0),
            peak_breakdown.get("activation_bytes", 0),
            peak_breakdown.get("optimizer_state_bytes", 0),
            peak_breakdown.get("other_bytes", 0),
        ]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(labels, values)
        ax.set_title(f"Peak Memory Breakdown ({self.model_name})")
        ax.set_ylabel("Bytes")
        ax.grid(axis="y", alpha=0.3)

        out = self.output_dir / "peak_memory_breakdown.png"
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        return out

    @staticmethod
    def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
        rows = list(rows)
        if not rows:
            with path.open("w", encoding="utf-8", newline="") as f:
                f.write("")
            return

        keys = list(rows[0].keys())
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
