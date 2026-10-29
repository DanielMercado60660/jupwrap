//! Epsilon scaling for an implicit unit-arc expansion (AMO, chapters 10/14).
//!
//! Each residual direction exposes only its next marginal cost. A push consumes
//! an entire admissible interval, found by binary search, rather than expanding
//! the convex arc or iterating over its capacity. Prices maintain c_p >= -eps.
//! At eps=1, costs scaled by n+1 exclude every negative simple residual cycle,
//! and hence certify the exact integer optimum.

use jupwrap_core::{ArcCost, McfError, Network};
use std::collections::VecDeque;

const INF: i64 = i64::MAX;

/// Monotone integer priority queue. Dijkstra's keys never decrease below the
/// last extracted key; radix buckets avoid heap comparisons on large grids.
struct RadixHeap {
    buckets: [Vec<(i64, usize)>; 64],
    occupied: u64,
    last: i64,
}

impl RadixHeap {
    fn new() -> Self {
        Self {
            buckets: std::array::from_fn(|_| Vec::new()),
            occupied: 0,
            last: 0,
        }
    }

    fn push(&mut self, key: i64, node: usize) {
        debug_assert!(key >= self.last);
        let b = 64 - ((key ^ self.last) as u64).leading_zeros() as usize;
        self.buckets[b].push((key, node));
        self.occupied |= 1u64 << b;
    }

    fn clear(&mut self) {
        for bucket in &mut self.buckets {
            bucket.clear();
        }
        self.occupied = 0;
        self.last = 0;
    }

    fn pop(&mut self) -> Option<(i64, usize)> {
        if self.occupied == 0 {
            return None;
        }
        if self.buckets[0].is_empty() {
            let b = self.occupied.trailing_zeros() as usize;
            self.last = self.buckets[b].iter().map(|&(key, _)| key).min().unwrap();
            self.occupied &= !(1u64 << b);
            while let Some((key, node)) = self.buckets[b].pop() {
                self.push(key, node);
            }
        }
        let result = self.buckets[0].pop();
        if self.buckets[0].is_empty() {
            self.occupied &= !1;
        }
        result
    }
}

fn arithmetic() -> McfError {
    McfError::Other("min-cost flow arithmetic exceeds i64 range".into())
}

/// Find a minimizer without scanning the bounds. Start near zero so the usual
/// grid case takes only a few cost calls, even with million-unit bounds.
fn minimizer(cost: &dyn ArcCost, a: usize, lower: i32, upper: i32) -> i32 {
    let x = 0.clamp(lower, upper);
    let c = cost.cost(a, x);
    let direction = if x < upper && cost.cost(a, x + 1) < c {
        1i64
    } else if x > lower && cost.cost(a, x - 1) < c {
        -1
    } else {
        return x;
    };
    let limit = if direction == 1 {
        i64::from(upper) - i64::from(x)
    } else {
        i64::from(x) - i64::from(lower)
    };
    let at = |s: i64| (i64::from(x) + direction * s) as i32;
    let decreasing = |s| cost.cost(a, at(s + 1)) < cost.cost(a, at(s));
    let mut lo = 0;
    let mut hi = 1;
    while hi < limit && decreasing(hi) {
        lo = hi + 1;
        hi = (hi * 2).min(limit);
    }
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if decreasing(mid) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    at(lo)
}

// Most graphs use four-byte adjacency entries. Retain the usize variant for
// arbitrary Network sizes rather than imposing a new public size restriction.
trait Index: Copy {
    fn encode(x: usize) -> Self;
    fn decode(self) -> usize;
}

impl Index for u32 {
    fn encode(x: usize) -> Self {
        x as u32
    }
    fn decode(self) -> usize {
        self as usize
    }
}

impl Index for usize {
    fn encode(x: usize) -> Self {
        x
    }
    fn decode(self) -> usize {
        self
    }
}

struct Solver<'a, I> {
    net: &'a Network,
    cost: &'a dyn ArcCost,
    scale: i64,
    x: Vec<i32>,
    excess: Vec<i64>,
    price: Vec<i64>,
    // CSR adjacency: direction 2*a is forward; 2*a+1 is backward.
    start: Vec<usize>,
    adj: Vec<I>,
    marginal: Vec<i64>,
    current: Vec<usize>,
    // Global updates are frequent on large grids. Retain their allocations,
    // including the radix buckets left populated by an early-stopped search.
    distance: Vec<i64>,
    heap: RadixHeap,
}

pub(super) fn solve(net: &Network, cost: &dyn ArcCost) -> Result<Vec<i32>, McfError> {
    if net.n_arcs() <= u32::MAX as usize / 2 {
        solve_indexed::<u32>(net, cost)
    } else {
        solve_indexed::<usize>(net, cost)
    }
}

fn solve_indexed<I: Index>(net: &Network, cost: &dyn ArcCost) -> Result<Vec<i32>, McfError> {
    let n = net.n_nodes;
    let m = net.n_arcs();
    let scale = i64::try_from(n)
        .map_err(|_| arithmetic())?
        .checked_add(1)
        .ok_or_else(arithmetic)?;
    let mut s = Solver {
        net,
        cost,
        scale,
        x: Vec::with_capacity(m),
        excess: net.supply.clone(),
        price: vec![0; n],
        start: vec![0; n + 1],
        adj: Vec::new(),
        marginal: vec![0; 2 * m],
        current: vec![0; n],
        distance: vec![INF; n],
        heap: RadixHeap::new(),
    };
    for a in 0..m {
        let x = minimizer(cost, a, net.lower[a], net.upper[a]);
        s.x.push(x);
        let u = net.tail[a] as usize;
        let v = net.head[a] as usize;
        if u != v {
            s.excess[u] = s.excess[u]
                .checked_sub(i64::from(x))
                .ok_or_else(arithmetic)?;
            s.excess[v] = s.excess[v]
                .checked_add(i64::from(x))
                .ok_or_else(arithmetic)?;
            s.start[u + 1] += 1;
            s.start[v + 1] += 1;
        }
    }
    if s.excess.iter().all(|&e| e == 0) {
        return Ok(s.x);
    }
    for a in 0..m {
        if net.tail[a] != net.head[a] {
            s.refresh(a)?;
        }
    }
    for u in 0..n {
        s.start[u + 1] += s.start[u];
    }
    s.adj = vec![I::encode(0); s.start[n]];
    s.current.copy_from_slice(&s.start[..n]);
    for a in 0..m {
        let u = net.tail[a] as usize;
        let v = net.head[a] as usize;
        if u != v {
            s.adj[s.current[u]] = I::encode(2 * a);
            s.current[u] += 1;
            s.adj[s.current[v]] = I::encode(2 * a + 1);
            s.current[v] += 1;
        }
    }
    // Initial arc minima are zero-optimal even though not yet feasible. The
    // first refinement can therefore start at any positive epsilon.
    // Power-of-two epsilons make every shortest-path length an exact shift.
    // Keep the n+1 cost multiplier (the exactness certificate) independent of
    // this performance choice. The initial flow is zero-optimal, so rounding
    // the first epsilon upward does not change any invariant.
    let mut epsilon = (scale as u64)
        .checked_mul(1024)
        .and_then(u64::checked_next_power_of_two)
        .and_then(|e| i64::try_from(e).ok())
        .ok_or_else(arithmetic)?;
    loop {
        s.refine(epsilon)?;
        if epsilon == 1 {
            return Ok(s.x);
        }
        epsilon = (epsilon / 16).max(1);
    }
}

impl<I: Index> Solver<'_, I> {
    #[inline]
    fn target(&self, d: usize) -> usize {
        if d & 1 == 0 {
            self.net.head[d / 2] as usize
        } else {
            self.net.tail[d / 2] as usize
        }
    }

    #[inline]
    fn capacity(&self, d: usize) -> i64 {
        let a = d / 2;
        if d & 1 == 0 {
            i64::from(self.net.upper[a]) - i64::from(self.x[a])
        } else {
            i64::from(self.x[a]) - i64::from(self.net.lower[a])
        }
    }

    fn refresh(&mut self, a: usize) -> Result<(), McfError> {
        let x = self.x[a];
        let c = self.cost.cost(a, x);
        for dir in 0..2 {
            let d = 2 * a + dir;
            if self.capacity(d) != 0 {
                let next = if dir == 0 { x + 1 } else { x - 1 };
                self.marginal[d] = i64::try_from(
                    (i128::from(self.cost.cost(a, next)) - i128::from(c)) * i128::from(self.scale),
                )
                .map_err(|_| arithmetic())?;
            }
        }
        Ok(())
    }

    #[inline]
    fn reduced(&self, u: usize, d: usize) -> Result<i64, McfError> {
        i64::try_from(
            i128::from(self.marginal[d]) + i128::from(self.price[u])
                - i128::from(self.price[self.target(d)]),
        )
        .map_err(|_| arithmetic())
    }

    // Largest prefix of at most `limit` marginal units with negative reduced
    // cost. Check the first two units before searching: quadratic grid costs
    // almost always permit only one unit.
    fn admissible(&self, u: usize, d: usize, limit: i64) -> i64 {
        let limit = limit.min(self.capacity(d));
        if limit <= 1 {
            return limit;
        }
        let a = d / 2;
        let step = if d & 1 == 0 { 1 } else { -1 };
        let negative = |k: i64| {
            let x = (i64::from(self.x[a]) + step * (k - 1)) as i32;
            let y = (i64::from(x) + step) as i32;
            let c = (i128::from(self.cost.cost(a, y)) - i128::from(self.cost.cost(a, x)))
                * i128::from(self.scale);
            c + i128::from(self.price[u]) - i128::from(self.price[self.target(d)]) < 0
        };
        if !negative(2) {
            return 1;
        }
        if negative(limit) {
            return limit;
        }
        let (mut lo, mut hi) = (2, limit);
        while lo + 1 < hi {
            let mid = lo + (hi - lo) / 2;
            if negative(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }

    fn push(&mut self, u: usize, d: usize, amount: i64) -> Result<(), McfError> {
        let v = self.target(d);
        let a = d / 2;
        let step = if d & 1 == 0 { amount } else { -amount };
        self.x[a] = (i64::from(self.x[a]) + step) as i32;
        self.excess[u] = self.excess[u].checked_sub(amount).ok_or_else(arithmetic)?;
        self.excess[v] = self.excess[v].checked_add(amount).ok_or_else(arithmetic)?;
        if amount == 1 {
            // Reversing this unit has exactly the negative marginal cost of
            // the unit just consumed. Reuse it rather than calling the opaque
            // cost function again. Only the next unit in the push direction
            // needs evaluation; at a bound even those calls are unnecessary.
            self.marginal[d ^ 1] = self.marginal[d].checked_neg().ok_or_else(arithmetic)?;
            if self.capacity(d) != 0 {
                let x = self.x[a];
                let next = if d & 1 == 0 { x + 1 } else { x - 1 };
                self.marginal[d] = i64::try_from(
                    (i128::from(self.cost.cost(a, next)) - i128::from(self.cost.cost(a, x)))
                        * i128::from(self.scale),
                )
                .map_err(|_| arithmetic())?;
            }
            Ok(())
        } else {
            self.refresh(a)
        }
    }

    /// Global price update: reversed multi-source Dijkstra from all deficits.
    /// Length floor(c_p/eps)+1 is nonnegative under epsilon optimality. Lowering
    /// each price by eps*distance preserves that invariant and provides a
    /// downhill route toward a deficit. Unreachable excess certifies a cut
    /// with insufficient capacity, independently of costs.
    fn global_update(&mut self, epsilon: i64) -> Result<(), McfError> {
        debug_assert!((epsilon as u64).is_power_of_two());
        let shift = epsilon.trailing_zeros();
        let n = self.net.n_nodes;
        self.distance.fill(INF);
        self.heap.clear();
        let mut remaining = 0;
        for (u, &e) in self.excess.iter().enumerate() {
            if e < 0 {
                self.distance[u] = 0;
                self.heap.push(0, u);
            }
            if e > 0 {
                remaining += 1;
            }
        }
        let mut farthest = 0;
        while let Some((du, u)) = self.heap.pop() {
            if du != self.distance[u] {
                continue;
            }
            farthest = du;
            if self.excess[u] > 0 {
                remaining -= 1;
                if remaining == 0 {
                    break;
                }
            }
            for i in self.start[u]..self.start[u + 1] {
                let outgoing = self.adj[i].decode();
                let d = outgoing ^ 1;
                if self.capacity(d) == 0 {
                    continue;
                }
                let v = self.target(outgoing);
                let rc = self.reduced(v, d)?;
                debug_assert!(rc >= -epsilon);
                let length = if rc < 0 {
                    0
                } else {
                    (rc >> shift).checked_add(1).ok_or_else(arithmetic)?
                };
                let dv = du.checked_add(length).ok_or_else(arithmetic)?;
                if dv < self.distance[v] {
                    self.distance[v] = dv;
                    self.heap.push(dv, v);
                }
            }
        }
        for (u, &du) in self.distance.iter().enumerate() {
            if remaining != 0 && du == INF && self.excess[u] > 0 {
                return Err(McfError::Infeasible);
            }
            // Unsettled labels are capped at the last settled distance. This
            // is also the correct update for nodes outside the search region.
            let delta = du.min(farthest);
            self.price[u] = self.price[u]
                .checked_sub(delta.checked_mul(epsilon).ok_or_else(arithmetic)?)
                .ok_or_else(arithmetic)?;
        }
        self.current.copy_from_slice(&self.start[..n]);
        Ok(())
    }

    fn refine(&mut self, epsilon: i64) -> Result<(), McfError> {
        // Restore zero optimality before discharging the new pseudoflow.
        for a in 0..self.net.n_arcs() {
            let u = self.net.tail[a] as usize;
            let v = self.net.head[a] as usize;
            if u == v {
                continue;
            }
            for dir in 0..2 {
                let d = 2 * a + dir;
                let from = if dir == 0 { u } else { v };
                if self.capacity(d) > 0 && self.reduced(from, d)? < 0 {
                    let amount = self.admissible(from, d, self.capacity(d));
                    self.push(from, d, amount)?;
                }
            }
        }
        if self.excess.iter().all(|&e| e == 0) {
            return Ok(());
        }
        self.global_update(epsilon)?;
        let mut active = VecDeque::new();
        for (u, &e) in self.excess.iter().enumerate() {
            if e > 0 {
                active.push_back(u);
            }
        }
        let mut work = 0;
        // Refresh after one adjacency scan's worth of local work. Letting
        // local relabels run longer creates expensive price gradients across
        // large sparse graphs; the global update repairs them in one search.
        let work_limit = self.adj.len().max(1);
        while let Some(u) = active.pop_front() {
            while self.excess[u] > 0 {
                let end = self.start[u + 1];
                while self.current[u] < end {
                    let d = self.adj[self.current[u]].decode();
                    work += 1;
                    if self.capacity(d) > 0 && self.reduced(u, d)? < 0 {
                        let v = self.target(d);
                        let was_active = self.excess[v] > 0;
                        let amount = self.admissible(u, d, self.excess[u]);
                        self.push(u, d, amount)?;
                        if !was_active && self.excess[v] > 0 {
                            active.push_back(v);
                        }
                        if self.excess[u] == 0 {
                            break;
                        }
                    }
                    self.current[u] += 1;
                }
                if self.excess[u] == 0 {
                    break;
                }
                if work >= work_limit {
                    self.global_update(epsilon)?;
                    work = 0;
                } else {
                    let mut least = INF;
                    for i in self.start[u]..end {
                        let d = self.adj[i].decode();
                        if self.capacity(d) > 0 {
                            least = least.min(self.reduced(u, d)?);
                        }
                    }
                    if least == INF {
                        return Err(McfError::Infeasible);
                    }
                    debug_assert!(least >= 0);
                    self.price[u] = self.price[u]
                        .checked_sub(least.checked_add(epsilon).ok_or_else(arithmetic)?)
                        .ok_or_else(arithmetic)?;
                    self.current[u] = self.start[u];
                    work += end - self.start[u];
                }
            }
        }
        Ok(())
    }
}
