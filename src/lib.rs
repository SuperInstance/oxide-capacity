//! # oxide-capacity
//!
//! GPU cluster capacity planning with ternary utilization signals.
//! Provides bin packing, scale recommendations, trend prediction, and cost efficiency analysis.

use std::cmp::Ordering;

// ── Ternary Utilization ──────────────────────────────────────────────────────

/// Ternary utilization signal: underutilized, balanced, or overloaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TernarySignal {
    Underutilized = 1,
    Balanced = 0,
    Overloaded = -1,
}

impl TernarySignal {
    /// Classify a utilization ratio (0.0–1.0) into a ternary signal.
    ///
    /// - `< 0.4` → Underutilized
    /// - `0.4–0.8` → Balanced
    /// - `> 0.8` → Overloaded
    pub fn from_utilization(ratio: f64) -> Self {
        if ratio < 0.4 {
            TernarySignal::Underutilized
        } else if ratio > 0.8 {
            TernarySignal::Overloaded
        } else {
            TernarySignal::Balanced
        }
    }
}

impl std::fmt::Display for TernarySignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TernarySignal::Underutilized => write!(f, "+1 (underutilized)"),
            TernarySignal::Balanced => write!(f, " 0 (balanced)"),
            TernarySignal::Overloaded => write!(f, "-1 (overloaded)"),
        }
    }
}

// ── Resource Profile ─────────────────────────────────────────────────────────

/// Hardware resource profile for a GPU node.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResourceProfile {
    /// GPU memory in GiB.
    pub gpu_memory_gib: f64,
    /// Number of compute units (e.g. SMs, CUs).
    pub compute_units: u32,
    /// Memory bandwidth in GiB/s.
    pub bandwidth_gib_s: f64,
}

impl ResourceProfile {
    pub fn new(gpu_memory_gib: f64, compute_units: u32, bandwidth_gib_s: f64) -> Self {
        Self {
            gpu_memory_gib,
            compute_units,
            bandwidth_gib_s,
        }
    }
}

// ── Workload ─────────────────────────────────────────────────────────────────

/// A workload requesting GPU resources.
#[derive(Debug, Clone, PartialEq)]
pub struct Workload {
    pub id: String,
    pub gpu_memory_gib: f64,
    pub compute_units: u32,
    pub bandwidth_gib_s: f64,
}

impl Workload {
    pub fn new(id: impl Into<String>, gpu_memory_gib: f64, compute_units: u32, bandwidth_gib_s: f64) -> Self {
        Self {
            id: id.into(),
            gpu_memory_gib,
            compute_units,
            bandwidth_gib_s,
        }
    }
}

// ── Node ─────────────────────────────────────────────────────────────────────

/// A single GPU node in the cluster.
#[derive(Debug, Clone)]
pub struct Node {
    pub id: String,
    pub profile: ResourceProfile,
    pub workloads: Vec<Workload>,
}

impl Node {
    pub fn new(id: impl Into<String>, profile: ResourceProfile) -> Self {
        Self {
            id: id.into(),
            profile,
            workloads: Vec::new(),
        }
    }

    /// How much GPU memory is currently allocated (sum of workload demands).
    pub fn allocated_gpu_memory(&self) -> f64 {
        self.workloads.iter().map(|w| w.gpu_memory_gib).sum()
    }

    /// How much compute is currently allocated.
    pub fn allocated_compute(&self) -> u32 {
        self.workloads.iter().map(|w| w.compute_units).sum()
    }

    /// How much bandwidth is currently allocated.
    pub fn allocated_bandwidth(&self) -> f64 {
        self.workloads.iter().map(|w| w.bandwidth_gib_s).sum()
    }

    /// Utilization ratio: allocated / capacity (0.0–1.0+ if overcommitted).
    pub fn memory_utilization(&self) -> f64 {
        if self.profile.gpu_memory_gib == 0.0 {
            return 0.0;
        }
        self.allocated_gpu_memory() / self.profile.gpu_memory_gib
    }

    pub fn compute_utilization(&self) -> f64 {
        if self.profile.compute_units == 0 {
            return 0.0;
        }
        self.allocated_compute() as f64 / self.profile.compute_units as f64
    }

    /// Composite utilization: average of memory and compute ratios.
    pub fn composite_utilization(&self) -> f64 {
        (self.memory_utilization() + self.compute_utilization()) / 2.0
    }

    /// Ternary classification of this node.
    pub fn ternary_signal(&self) -> TernarySignal {
        TernarySignal::from_utilization(self.composite_utilization())
    }

    /// Can this node accept the workload without exceeding capacity?
    pub fn can_fit(&self, workload: &Workload) -> bool {
        self.allocated_gpu_memory() + workload.gpu_memory_gib <= self.profile.gpu_memory_gib
            && self.allocated_compute() + workload.compute_units <= self.profile.compute_units
            && self.allocated_bandwidth() + workload.bandwidth_gib_s <= self.profile.bandwidth_gib_s
    }

    /// Assign a workload to this node.
    pub fn assign(&mut self, workload: Workload) {
        self.workloads.push(workload);
    }

    /// Remaining GPU memory.
    pub fn remaining_gpu_memory(&self) -> f64 {
        (self.profile.gpu_memory_gib - self.allocated_gpu_memory()).max(0.0)
    }
}

// ── Capacity Planner ─────────────────────────────────────────────────────────

/// Tracks per-node utilization and classifies ternary signals.
#[derive(Debug, Clone)]
pub struct CapacityPlanner {
    pub nodes: Vec<Node>,
}

impl CapacityPlanner {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn add_node(&mut self, node: Node) {
        self.nodes.push(node);
    }

    /// Overall cluster utilization ratio.
    pub fn cluster_utilization(&self) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        let total_cap: f64 = self.nodes.iter().map(|n| n.profile.gpu_memory_gib).sum();
        let total_alloc: f64 = self.nodes.iter().map(|n| n.allocated_gpu_memory()).sum();
        if total_cap == 0.0 {
            return 0.0;
        }
        total_alloc / total_cap
    }

    /// Aggregate ternary signal across all nodes.
    pub fn aggregate_signal(&self) -> TernarySignal {
        TernarySignal::from_utilization(self.cluster_utilization())
    }

    /// Bin-pack workloads onto nodes using best-fit decreasing.
    ///
    /// Sorts workloads by GPU memory descending, then places each workload on
    /// the node with the *least* remaining memory that still fits.
    pub fn bin_pack_best_fit_decreasing(&mut self, mut workloads: Vec<Workload>) -> Vec<Workload> {
        // Sort descending by memory demand
        workloads.sort_by(|a, b| {
            b.gpu_memory_gib
                .partial_cmp(&a.gpu_memory_gib)
                .unwrap_or(Ordering::Equal)
        });

        let mut unplaced = Vec::new();
        for wl in workloads {
            let mut best_idx: Option<usize> = None;
            let mut best_remaining = f64::MAX;

            for (i, node) in self.nodes.iter().enumerate() {
                if node.can_fit(&wl) {
                    let rem = node.remaining_gpu_memory() - wl.gpu_memory_gib;
                    if rem >= 0.0 && rem < best_remaining {
                        best_remaining = rem;
                        best_idx = Some(i);
                    }
                }
            }

            match best_idx {
                Some(idx) => self.nodes[idx].assign(wl),
                None => unplaced.push(wl),
            }
        }

        unplaced
    }

    /// Scale recommendation: ternary signal for the whole cluster.
    pub fn scale_recommendation(&self) -> ScaleRecommendation {
        let signal = self.aggregate_signal();
        let recommendation = match signal {
            TernarySignal::Underutilized => "scale-down: cluster is underutilized, consider removing nodes",
            TernarySignal::Balanced => "hold: cluster utilization is balanced",
            TernarySignal::Overloaded => "scale-up: cluster is overloaded, consider adding nodes",
        };
        ScaleRecommendation {
            signal,
            recommendation: recommendation.to_string(),
            utilization: self.cluster_utilization(),
        }
    }

    /// Predict future utilization using simple linear extrapolation.
    ///
    /// Given a series of past utilization measurements and a horizon (number of
    /// steps to extrapolate), returns the predicted utilization.
    pub fn predict_utilization(history: &[f64], horizon: usize) -> f64 {
        if history.is_empty() {
            return 0.0;
        }
        if history.len() == 1 {
            return history[0];
        }
        // Simple linear regression: slope from last two points, extrapolate.
        let n = history.len();
        let last = history[n - 1];
        let prev = history[n - 2];
        let slope = last - prev;
        (last + slope * horizon as f64).max(0.0)
    }

    /// Cost efficiency: ratio of actual utilization to total allocation.
    /// Returns (utilization_ratio, cost_efficiency_score).
    pub fn cost_efficiency(&self) -> (f64, f64) {
        let util = self.cluster_utilization();
        // Cost efficiency: 1.0 is perfect utilization, penalize under and over
        let score = if util <= 0.0 {
            0.0
        } else if util <= 1.0 {
            util // Higher utilization = better cost efficiency
        } else {
            // Overcommitted: penalize
            1.0 / util
        };
        (util, score)
    }

    /// Per-node signals for detailed analysis.
    pub fn node_signals(&self) -> Vec<(String, TernarySignal, f64)> {
        self.nodes
            .iter()
            .map(|n| (n.id.clone(), n.ternary_signal(), n.composite_utilization()))
            .collect()
    }
}

impl Default for CapacityPlanner {
    fn default() -> Self {
        Self::new()
    }
}

// ── Scale Recommendation ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ScaleRecommendation {
    pub signal: TernarySignal,
    pub recommendation: String,
    pub utilization: f64,
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn a100_profile() -> ResourceProfile {
        ResourceProfile::new(80.0, 108, 2039.0)
    }

    fn small_profile() -> ResourceProfile {
        ResourceProfile::new(24.0, 46, 900.0)
    }

    #[test]
    fn test_ternary_classification() {
        assert_eq!(TernarySignal::from_utilization(0.1), TernarySignal::Underutilized);
        assert_eq!(TernarySignal::from_utilization(0.39), TernarySignal::Underutilized);
        assert_eq!(TernarySignal::from_utilization(0.5), TernarySignal::Balanced);
        assert_eq!(TernarySignal::from_utilization(0.7), TernarySignal::Balanced);
        assert_eq!(TernarySignal::from_utilization(0.9), TernarySignal::Overloaded);
    }

    #[test]
    fn test_resource_profile_creation() {
        let p = a100_profile();
        assert_eq!(p.gpu_memory_gib, 80.0);
        assert_eq!(p.compute_units, 108);
        assert_eq!(p.bandwidth_gib_s, 2039.0);
    }

    #[test]
    fn test_node_utilization() {
        let mut node = Node::new("gpu-0", a100_profile());
        assert_eq!(node.memory_utilization(), 0.0);

        node.assign(Workload::new("wl-0", 40.0, 54, 1000.0));
        assert!((node.memory_utilization() - 0.5).abs() < 1e-9);
        assert!((node.compute_utilization() - 0.5).abs() < 1e-9);
        assert_eq!(node.ternary_signal(), TernarySignal::Balanced);
    }

    #[test]
    fn test_node_can_fit() {
        let mut node = Node::new("gpu-0", small_profile());
        assert!(node.can_fit(&Workload::new("wl", 24.0, 46, 900.0)));
        node.assign(Workload::new("wl-0", 20.0, 40, 800.0));
        // Fits remaining memory (4 GiB) but not bandwidth
        assert!(!node.can_fit(&Workload::new("wl-1", 4.0, 6, 200.0)));
    }

    #[test]
    fn test_bin_pack_best_fit_decreasing() {
        let mut planner = CapacityPlanner::new();
        planner.add_node(Node::new("n0", ResourceProfile::new(40.0, 100, 1000.0)));
        planner.add_node(Node::new("n1", ResourceProfile::new(40.0, 100, 1000.0)));

        let workloads = vec![
            Workload::new("w0", 25.0, 50, 500.0),
            Workload::new("w1", 20.0, 40, 400.0),
            Workload::new("w2", 15.0, 30, 300.0),
            Workload::new("w3", 10.0, 20, 200.0),
        ];

        let unplaced = planner.bin_pack_best_fit_decreasing(workloads);
        assert!(unplaced.is_empty());

        // w0(25) → n0 (remaining 15), w1(20) → n1 (remaining 20)
        // w2(15) → n0 fits exactly (remaining 0), w3(10) → n1 (remaining 10)
        assert_eq!(planner.nodes[0].workloads.len(), 2);
        assert_eq!(planner.nodes[1].workloads.len(), 2);
    }

    #[test]
    fn test_bin_pack_unplaced_when_full() {
        let mut planner = CapacityPlanner::new();
        planner.add_node(Node::new("n0", ResourceProfile::new(10.0, 10, 100.0)));

        let workloads = vec![
            Workload::new("w0", 10.0, 10, 100.0),
            Workload::new("w1", 5.0, 5, 50.0), // won't fit
        ];

        let unplaced = planner.bin_pack_best_fit_decreasing(workloads);
        assert_eq!(unplaced.len(), 1);
        assert_eq!(unplaced[0].id, "w1");
    }

    #[test]
    fn test_scale_recommendation() {
        let mut planner = CapacityPlanner::new();
        planner.add_node(Node::new("n0", a100_profile()));
        // Empty cluster → underutilized → scale-down
        let rec = planner.scale_recommendation();
        assert_eq!(rec.signal, TernarySignal::Underutilized);
        assert!(rec.recommendation.contains("scale-down"));
    }

    #[test]
    fn test_prediction_linear_extrapolation() {
        let history = vec![0.2, 0.3, 0.4, 0.5];
        // Last slope = 0.1, horizon = 3 → 0.5 + 0.3 = 0.8
        let predicted = CapacityPlanner::predict_utilization(&history, 3);
        assert!((predicted - 0.8).abs() < 1e-9);

        // Edge: empty history
        assert_eq!(CapacityPlanner::predict_utilization(&[], 5), 0.0);
        // Edge: single point
        assert_eq!(CapacityPlanner::predict_utilization(&[0.5], 10), 0.5);
    }

    #[test]
    fn test_cost_efficiency() {
        let mut planner = CapacityPlanner::new();
        let mut n0 = Node::new("n0", ResourceProfile::new(100.0, 100, 1000.0));
        n0.assign(Workload::new("w0", 75.0, 50, 500.0));
        planner.add_node(n0);

        let (util, score) = planner.cost_efficiency();
        assert!((util - 0.75).abs() < 1e-9);
        assert!((score - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_overloaded_cluster() {
        let mut planner = CapacityPlanner::new();
        let mut n0 = Node::new("n0", ResourceProfile::new(40.0, 100, 2000.0));
        // Overcommit memory
        n0.assign(Workload::new("w0", 35.0, 50, 500.0));
        n0.assign(Workload::new("w1", 35.0, 50, 500.0));
        planner.add_node(n0);

        // Memory util > 1.0, composite > 0.8 → overloaded
        let signals = planner.node_signals();
        assert_eq!(signals[0].1, TernarySignal::Overloaded);
    }

    #[test]
    fn test_aggregate_signal_balanced() {
        let mut planner = CapacityPlanner::new();
        let mut n0 = Node::new("n0", ResourceProfile::new(100.0, 100, 1000.0));
        n0.assign(Workload::new("w0", 60.0, 60, 600.0));
        planner.add_node(n0);

        assert_eq!(planner.aggregate_signal(), TernarySignal::Balanced);
        let (util, score) = planner.cost_efficiency();
        assert!((util - 0.6).abs() < 1e-9);
        assert!((score - 0.6).abs() < 1e-9);
    }
}
