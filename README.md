# oxide-capacity

GPU cluster capacity planning with ternary utilization signals.

## Why This Exists

GPU clusters are expensive, and utilization is the only lever that matters. But "utilization" as a single number is a lie — a node at 95% memory but 20% compute isn't "overutilized," it's *misconfigured*. You need multi-dimensional bin packing, and you need a signal that captures the three states that actually drive decisions: **waste money** (underutilized, scale down), **sweet spot** (balanced, hold), **risk** (overloaded, scale up).

The ternary classification (< 0.4 → underutilized, 0.4–0.8 → balanced, > 0.8 → overloaded) maps directly to operational decisions. No dashboards to interpret, no thresholds to tune per workload. One signal, one action.

## Architecture

```
┌──────────────────────────────────────────────────┐
│               CapacityPlanner                    │
│                                                  │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐         │
│  │  Node 0 │  │  Node 1 │  │  Node 2 │  ...    │
│  │ A100 80G│  │ A100 80G│  │ A10 24G │         │
│  │ [w1, w2]│  │ [w3]   │  │ [w4, w5]│         │
│  │ mem: 62%│  │ mem: 30%│  │ mem: 90%│         │
│  │ cmp: 50%│  │ cmp: 25%│  │ cmp: 70%│         │
│  │ Balanced│  │ Underut.│  │ Overld. │         │
│  └─────────┘  └─────────┘  └─────────┘         │
│                                                  │
│  bin_pack_best_fit_decreasing(workloads)         │
│  scale_recommendation() → ScaleRecommendation    │
│  predict_utilization(history, horizon) → f64     │
│  cost_efficiency() → (util, score)               │
└──────────────────────────────────────────────────┘

Workload ──→ bin_pack ──→ Node assignment
                         └─→ unplaced (overflow)
```

**Key types:**

- `TernarySignal` — `Underutilized(+1)`, `Balanced(0)`, `Overloaded(-1)`
- `ResourceProfile` — hardware spec: GPU memory, compute units, bandwidth
- `Workload` — resource demands: memory, compute, bandwidth
- `Node` — a GPU node with profile and assigned workloads
- `CapacityPlanner` — the planning engine
- `ScaleRecommendation` — signal + human-readable recommendation + utilization

## Usage

```rust
use oxide_capacity::*;

let a100 = ResourceProfile::new(80.0, 108, 2039.0); // 80 GiB, 108 SMs, 2039 GiB/s

let mut planner = CapacityPlanner::new();
planner.add_node(Node::new("gpu-0", a100));
planner.add_node(Node::new("gpu-1", a100));

// Pack workloads onto nodes (best-fit decreasing by memory)
let workloads = vec![
    Workload::new("training-run-42", 40.0, 54, 1000.0),
    Workload::new("inference-api", 20.0, 30, 500.0),
    Workload::new("data-pipeline", 30.0, 40, 800.0),
    Workload::new("analytics", 15.0, 20, 300.0),
];
let unplaced = planner.bin_pack_best_fit_decreasing(workloads);
assert!(unplaced.is_empty());

// Check cluster health
let rec = planner.scale_recommendation();
println!("Signal: {} ({:.0}%)", rec.signal, rec.utilization * 100.0);

// Predict future utilization
let history = vec![0.2, 0.3, 0.4, 0.5, 0.6];
let predicted = CapacityPlanner::predict_utilization(&history, 3); // 3 steps ahead
// predicted ≈ 0.9 (linear extrapolation from last two points)

// Cost efficiency
let (util, score) = planner.cost_efficiency();
```

## API Reference

### `TernarySignal`

```rust
pub enum TernarySignal {
    Underutilized = 1,  // < 40% utilization
    Balanced = 0,       // 40–80% utilization
    Overloaded = -1,    // > 80% utilization
}
```

- `from_utilization(ratio: f64) -> Self` — classify a 0.0–1.0 ratio
- `impl Display` — formats as `+1 (underutilized)`, ` 0 (balanced)`, `-1 (overloaded)`

### `ResourceProfile`

- `new(gpu_memory_gib, compute_units, bandwidth_gib_s) -> Self`

### `Workload`

- `new(id, gpu_memory_gib, compute_units, bandwidth_gib_s) -> Self`

### `Node`

- `new(id, profile) -> Self`
- `assign(workload)` / `can_fit(&workload) -> bool`
- `allocated_gpu_memory() -> f64` / `allocated_compute() -> u32` / `allocated_bandwidth() -> f64`
- `memory_utilization() -> f64` / `compute_utilization() -> f64` / `composite_utilization() -> f64`
- `ternary_signal() -> TernarySignal` / `remaining_gpu_memory() -> f64`

### `CapacityPlanner`

- `new() -> Self` / `add_node(node)`
- `cluster_utilization() -> f64` / `aggregate_signal() -> TernarySignal`
- `bin_pack_best_fit_decreasing(workloads) -> Vec<Workload>` — returns unplaced workloads
- `scale_recommendation() -> ScaleRecommendation`
- `predict_utilization(history: &[f64], horizon: usize) -> f64` (associated function)
- `cost_efficiency() -> (f64, f64)` — (utilization_ratio, efficiency_score)
- `node_signals() -> Vec<(String, TernarySignal, f64)>` — per-node breakdown

### `ScaleRecommendation`

- `signal: TernarySignal`, `recommendation: String`, `utilization: f64`

## The Deeper Idea

This is the **capacity layer** in the oxide stack's resource planning architecture. The ternary utilization signal drives both immediate decisions (bin packing) and strategic ones (scale up/down). The bin-packing algorithm uses best-fit decreasing — sort by memory demand descending, then place each workload on the node with the least remaining memory that still fits. This minimizes fragmentation.

The prediction engine is intentionally simple: linear extrapolation from the last two data points. For GPU capacity planning, the signal-to-noise ratio of complex models rarely justifies their overhead. A slope of +0.1/step over three steps tells you everything you need to know about whether you'll need more nodes next week.

## Related Crates

- **oxide-tenancy** — per-tenant isolation that feeds into node utilization calculations
- **oxide-federation** — cross-cluster federation that uses capacity signals for routing
- **oxide-health-monitor** — GPU health that affects effective capacity
- **oxide-compile-cache** — reduces effective workload by skipping recompilation
