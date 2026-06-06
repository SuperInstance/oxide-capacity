# oxide-capacity

GPU cluster capacity planning with ternary utilization signals. Bin packing, scale recommendations, trend prediction.

## Overview

# oxide-capacity

GPU cluster capacity planning with ternary utilization signals.

## Architecture

This crate sits within the **five-layer Oxide Stack**:

| Layer | Crate | Role |
|-------|-------|------|
| 1 | open-parallel | Async runtime (tokio fork) |
| 2 | pincher | "Vector DB as runtime, LLM as compiler" |
| 3 | flux-core | Bytecode VM + A2A agent protocol |
| 4 | cuda-oxide | Flux→MIR→Pliron→NVVM→PTX compiler |
| 5 | cudaclaw | Persistent GPU kernels, warp consensus, SmartCRDT |

The key insight: **ternary values {-1, 0, +1} map directly to GPU compute**. They pack 16× denser than FP32, enable XNOR+popcount matmul, and conservation laws become compile-time checks.

## Stats

| Metric | Value |
|--------|-------|
| Tests | 11 |
| Lines of Code | 461 |
| Public API Surface | 29 items |
| License | MIT |

## Installation

```toml
[dependencies]
oxide-capacity = "0.1.0"
```

## Usage

```rust
use oxide_capacity::*;
// See src/lib.rs tests for complete working examples
```

### Key Types

```
- pub enum TernarySignal {
    pub fn from_utilization(ratio: f64) -> Self {
- pub struct ResourceProfile {
    pub fn new(gpu_memory_gib: f64, compute_units: u32, bandwidth_gib_s: f64) -> Self {
- pub struct Workload {
    pub fn new(id: impl Into<String>, gpu_memory_gib: f64, compute_units: u32, bandwidth_gib_s: f64) -> Self {
- pub struct Node {
    pub fn new(id: impl Into<String>, profile: ResourceProfile) -> Self {
    pub fn allocated_gpu_memory(&self) -> f64 {
    pub fn allocated_compute(&self) -> u32 {
```

## Design Philosophy

This crate uses **ternary algebra** (Z₃) where every value is {-1, 0, +1}:

- **+1** → positive signal (healthy, allocated, converged, ready)
- **0** → neutral (pending, balanced, monitoring, degraded)
- **-1** → negative signal (failed, free, diverged, overloaded)

This isn't arbitrary — ternary is the natural encoding for:
1. **BitNet b1.58** (Microsoft) — ternary neural networks at 60% less power
2. **GPU warp voting** — hardware ballot instructions return ternary consensus
3. **Conservation laws** — {-1, 0, +1} preserves quantity (what goes in must come out)

## Testing

```bash
git clone https://github.com/SuperInstance/oxide-capacity.git
cd oxide-capacity
cargo test
```

## License

MIT
