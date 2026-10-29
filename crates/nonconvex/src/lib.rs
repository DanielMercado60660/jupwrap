//! jupwrap-nonconvex: local improvement of a feasible cycle
//! field under nonconvex grid costs. See DESIGN.md section 7.

use jupwrap_core::{total_cost, ArcCost, Dims, GridCosts, ImproveOptions, ImproveStats, McfError};
use std::collections::VecDeque;
use std::time::Instant;

const NONE: u32 = u32::MAX;
const DISABLED: i64 = i64::MAX;

/// Improve `k` (one entry per edge of `costs.dims`) in place by applying
/// circulations (equivalently integer shifts of connected pixel sets) that
/// strictly lower `total_cost(costs, k)`.
///
/// Postconditions: the dual divergence of `k` is unchanged (so the result is
/// still curl-consistent with the residues); the cost never increases; and,
/// unless a limit in `opts` stopped the search, no single-pixel shift
/// `n_p += 1` or `n_p -= 1` lowers the cost. Returns initial/final costs.
/// A pass cancels residual cycles to stationarity and then sweeps all pixels.
/// Both option limits use zero for unlimited; an interrupted search returns
/// the current feasible field with exact cost and move statistics.
/// `Err(InvalidInput)` if `k.len() != costs.dims.n_edges()`.
pub fn improve_grid(
    costs: &GridCosts,
    k: &mut [i32],
    opts: &ImproveOptions,
) -> Result<ImproveStats, McfError> {
    let started = Instant::now();
    if k.len() != costs.dims.n_edges() {
        return Err(McfError::InvalidInput(
            "k length != number of grid edges".into(),
        ));
    }
    if costs.kt < 0
        || costs.kt > (i32::MAX - 1) / 2
        || k.len().checked_mul(costs.width()) != Some(costs.table.len())
    {
        return Err(McfError::InvalidInput("invalid grid cost table".into()));
    }
    if !opts.time_limit_s.is_finite() || opts.time_limit_s < 0.0 {
        return Err(McfError::InvalidInput(
            "time_limit_s must be finite and nonnegative".into(),
        ));
    }
    if k.len() >= (NONE / 2) as usize || costs.dims.n_dual_nodes() >= NONE as usize {
        return Err(McfError::InvalidInput(
            "grid too large for residual indices".into(),
        ));
    }
    // With a singleton table, core's negative-tail distance -k must fit i32.
    if costs.kt == 0 && k.contains(&i32::MIN) {
        return Err(McfError::InvalidInput(
            "flow outside supported integer range".into(),
        ));
    }
    let limit = Limit {
        started,
        seconds: opts.time_limit_s,
    };
    let initial_cost = total_cost(costs, k);
    let mut stats = ImproveStats {
        initial_cost,
        final_cost: initial_cost,
        ..ImproveStats::default()
    };
    if k.is_empty() || limit.expired() {
        return Ok(stats);
    }
    let mut graph = Residual::new(costs.dims);
    let mut search = Search::new(costs.dims.n_dual_nodes(), k.len());
    while !limit.expired() && (opts.max_passes == 0 || stats.passes < opts.max_passes as u64) {
        stats.passes += 1;
        let before = stats.moves;
        // A full pass searches to residual stationarity, then sweeps pixels.
        // Rebuilding after a batch keeps all incremental weights current.
        loop {
            if limit.expired() {
                break;
            }
            graph.reweight(costs, k);
            let moves = search.cancel_batch(&graph, costs, k, &limit);
            stats.moves += moves;
            if moves == 0 {
                break;
            }
        }
        stats.moves += pixel_sweep(costs, k, &limit);
        if stats.moves == before {
            break;
        }
    }
    stats.final_cost = total_cost(costs, k);
    Ok(stats)
}

struct Limit {
    started: Instant,
    seconds: f64,
}

impl Limit {
    fn expired(&self) -> bool {
        self.seconds > 0.0 && self.started.elapsed().as_secs_f64() >= self.seconds
    }
}

/// Residual IDs 2e and 2e+1 mean increasing and decreasing k[e]. CSR
/// adjacency includes both directions; a disabled weight removes a direction.
struct Residual {
    endpoints: Vec<(u32, u32)>,
    offsets: Vec<usize>,
    outgoing: Vec<u32>,
    weight: Vec<i64>,
}

impl Residual {
    fn new(dims: Dims) -> Self {
        let mut endpoints = Vec::with_capacity(dims.n_edges());
        let mut offsets = vec![0; dims.n_dual_nodes() + 1];
        for e in 0..dims.n_edges() {
            let (u, v) = dims.edge_dual(e);
            endpoints.push((u as u32, v as u32));
            offsets[u + 1] += 1;
            offsets[v + 1] += 1;
        }
        for v in 1..offsets.len() {
            offsets[v] += offsets[v - 1];
        }
        let mut next = offsets.clone();
        let mut outgoing = vec![0; 2 * dims.n_edges()];
        for (e, &(u, v)) in endpoints.iter().enumerate() {
            outgoing[next[u as usize]] = (2 * e) as u32;
            next[u as usize] += 1;
            outgoing[next[v as usize]] = (2 * e + 1) as u32;
            next[v as usize] += 1;
        }
        Self {
            endpoints,
            offsets,
            outgoing,
            weight: vec![0; 2 * dims.n_edges()],
        }
    }

    fn from(&self, a: u32) -> usize {
        let (u, v) = self.endpoints[a as usize / 2];
        if a & 1 == 0 {
            u as usize
        } else {
            v as usize
        }
    }

    fn to(&self, a: u32) -> usize {
        self.from(a ^ 1)
    }

    fn reweight(&mut self, costs: &GridCosts, k: &[i32]) {
        for (e, &x) in k.iter().enumerate() {
            let c = costs.cost(e, x);
            let fw = stepped(x, 1, costs.kt).map_or(DISABLED, |next| costs.cost(e, next) - c);
            let bw = stepped(x, -1, costs.kt).map_or(DISABLED, |next| costs.cost(e, next) - c);
            self.weight[2 * e] = fw;
            self.weight[2 * e + 1] = bw;
            // Concavity can create a false two-arc negative cycle that
            // changes an edge in both directions. Retain the cheaper arc.
            if fw != DISABLED && bw != DISABLED && fw + bw < 0 {
                if fw <= bw {
                    self.weight[2 * e + 1] = DISABLED;
                } else {
                    self.weight[2 * e] = DISABLED;
                }
            }
        }
    }
}

struct Search {
    distance: Vec<i64>,
    predecessor: Vec<u32>,
    queued: Vec<bool>,
    queue: VecDeque<usize>,
    state: Vec<u8>,
    path: Vec<usize>,
    cycle: Vec<u32>,
    edge_seen: Vec<bool>,
}

impl Search {
    fn new(nodes: usize, edges: usize) -> Self {
        Self {
            distance: vec![0; nodes],
            predecessor: vec![NONE; nodes],
            queued: vec![true; nodes],
            queue: VecDeque::with_capacity(nodes),
            state: vec![0; nodes],
            path: Vec::new(),
            cycle: Vec::new(),
            edge_seen: vec![false; edges],
        }
    }

    /// SPFA from an implicit zero-cost source connected to every dual node.
    /// Periodic O(V) scans collect *all* predecessor cycles, avoiding a full
    /// Bellman--Ford computation per individual cancellation.
    fn cancel_batch(
        &mut self,
        graph: &Residual,
        costs: &GridCosts,
        k: &mut [i32],
        limit: &Limit,
    ) -> u64 {
        self.distance.fill(0);
        self.predecessor.fill(NONE);
        self.queued.fill(true);
        self.queue.clear();
        self.queue.extend(0..self.distance.len());
        let period = self.distance.len().max(1);
        let mut since_scan = 0;
        let mut arcs_scanned = 0usize;
        while let Some(u) = self.queue.pop_front() {
            self.queued[u] = false;
            for &a in &graph.outgoing[graph.offsets[u]..graph.offsets[u + 1]] {
                // Ground has one incident arc per boundary edge. Check work
                // by arcs as well as nodes so its degree cannot delay a stop.
                arcs_scanned += 1;
                if arcs_scanned & 1023 == 0 && limit.expired() {
                    return 0;
                }
                let w = graph.weight[a as usize];
                if w == DISABLED {
                    continue;
                }
                let v = graph.to(a);
                let candidate = self.distance[u] + w;
                if candidate < self.distance[v] {
                    self.distance[v] = candidate;
                    self.predecessor[v] = a;
                    if !self.queued[v] {
                        self.queued[v] = true;
                        self.queue.push_back(v);
                    }
                }
            }
            since_scan += 1;
            if since_scan & 1023 == 0 && limit.expired() {
                return 0;
            }
            if since_scan >= period {
                if limit.expired() {
                    return 0;
                }
                let moves = self.extract(graph, costs, k, limit);
                if moves != 0 {
                    return moves;
                }
                since_scan = 0;
            }
        }
        0
    }

    fn extract(
        &mut self,
        graph: &Residual,
        costs: &GridCosts,
        k: &mut [i32],
        limit: &Limit,
    ) -> u64 {
        self.state.fill(0);
        let mut moves = 0;
        for root in 0..self.state.len() {
            if root & 1023 == 0 && limit.expired() {
                break;
            }
            if self.state[root] != 0 {
                continue;
            }
            self.path.clear();
            let mut u = root;
            while self.state[u] == 0 && self.predecessor[u] != NONE {
                if self.path.len() & 1023 == 0 && limit.expired() {
                    return moves;
                }
                self.state[u] = 1;
                self.path.push(u);
                u = graph.from(self.predecessor[u]);
            }
            if self.state[u] == 1 {
                // Every directed cycle of this functional graph is node
                // disjoint from the others, and hence edge disjoint too.
                self.cycle.clear();
                let first = u;
                let mut change = 0i64;
                let mut unique = true;
                loop {
                    let a = self.predecessor[u];
                    let e = a as usize / 2;
                    unique &= !self.edge_seen[e];
                    self.edge_seen[e] = true;
                    self.cycle.push(a);
                    let step = if a & 1 == 0 { 1 } else { -1 };
                    change += costs.cost(e, k[e] + step) - costs.cost(e, k[e]);
                    u = graph.from(a);
                    if u == first {
                        break;
                    }
                }
                if unique && change < 0 {
                    for &a in &self.cycle {
                        k[a as usize / 2] += if a & 1 == 0 { 1 } else { -1 };
                    }
                    moves += 1;
                }
                for &a in &self.cycle {
                    self.edge_seen[a as usize / 2] = false;
                }
            }
            for &v in &self.path {
                self.state[v] = 2;
            }
        }
        moves
    }
}

/// Gauss--Seidel sweep over every primal pixel, including its boundary.
/// Pixel incidence is obtained from the shared grid helpers. Each accepted
/// shift is a circulation even in a one-row/one-column grid.
fn pixel_sweep(costs: &GridCosts, k: &mut [i32], limit: &Limit) -> u64 {
    let d = costs.dims;
    let mut moves = 0;
    for r in 0..d.nrow {
        for c in 0..d.ncol {
            if (r * d.ncol + c) & 1023 == 0 && limit.expired() {
                return moves;
            }
            let mut edges = [(0, 0i32); 4];
            let mut n = 0;
            if r > 0 {
                edges[n] = (d.edge_v(r - 1, c), 1);
                n += 1;
            }
            if r + 1 < d.nrow {
                edges[n] = (d.edge_v(r, c), -1);
                n += 1;
            }
            if c > 0 {
                edges[n] = (d.edge_h(r, c - 1), 1);
                n += 1;
            }
            if c + 1 < d.ncol {
                edges[n] = (d.edge_h(r, c), -1);
                n += 1;
            }
            let mut up = 0;
            let mut down = 0;
            let mut up_valid = true;
            let mut down_valid = true;
            for &(e, sign) in &edges[..n] {
                let old = costs.cost(e, k[e]);
                if let Some(x) = stepped(k[e], sign, costs.kt) {
                    up += costs.cost(e, x) - old;
                } else {
                    up_valid = false;
                }
                if let Some(x) = stepped(k[e], -sign, costs.kt) {
                    down += costs.cost(e, x) - old;
                } else {
                    down_valid = false;
                }
            }
            let direction = if up_valid && up < 0 && (!down_valid || up <= down) {
                1
            } else if down_valid && down < 0 {
                -1
            } else {
                continue;
            };
            for &(e, sign) in &edges[..n] {
                k[e] += direction * sign;
            }
            moves += 1;
        }
    }
    moves
}

fn stepped(x: i32, sign: i32, kt: i32) -> Option<i32> {
    x.checked_add(sign)
        .filter(|&next| kt != 0 || next != i32::MIN)
}

#[cfg(test)]
mod tests;
