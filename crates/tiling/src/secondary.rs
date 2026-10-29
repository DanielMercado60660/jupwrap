//! Integer offsets on a region graph. Each pair function is stored exactly as
//! discrete slope changes; no bound on the relative phase offset is imposed.
//! This also retains nonconvex deformation costs, including their tail rule.

use jupwrap_core::{ArcCost, GridCosts, UnwrapError, MIN_TAIL_SLOPE};
use std::collections::BTreeMap;

#[derive(Clone, Default)]
struct Function {
    left: i128,
    jumps: BTreeMap<i64, i128>,
}
impl Function {
    fn jump(&mut self, x: i64, delta: i128) {
        if delta != 0 {
            *self.jumps.entry(x).or_default() += delta;
        }
    }
}
pub(super) struct Secondary {
    nodes: usize,
    functions: BTreeMap<(usize, usize), Function>,
    boundaries: BTreeMap<(usize, usize), Function>,
}
impl Secondary {
    pub(super) fn new(nodes: usize) -> Self {
        Self {
            nodes,
            functions: BTreeMap::new(),
            boundaries: BTreeMap::new(),
        }
    }

    pub(super) fn overlap(&mut self, a: usize, b: usize, target: i64, weight: i64) {
        if a == b {
            return;
        }
        let (a, b, target) = if a < b {
            (a, b, target)
        } else {
            (b, a, -target)
        };
        let f = self.functions.entry((a, b)).or_default();
        f.left -= i128::from(weight);
        f.jump(target, 2 * i128::from(weight));
    }

    pub(super) fn boundary(&mut self, a: usize, b: usize, costs: &GridCosts, e: usize, k: i64) {
        let forward = a < b;
        let key = if forward { (a, b) } else { (b, a) };
        let f = self.boundaries.entry(key).or_default();
        let row = costs.row(e);
        let last = row.len() - 1;
        let value = |i: usize| i64::from(row[if forward { i } else { last - i }]);
        let lo = if forward {
            -i64::from(costs.kt) - k
        } else {
            k - i64::from(costs.kt)
        };
        let left = if last > 0 {
            (value(0) - value(1)).max(MIN_TAIL_SLOPE)
        } else {
            MIN_TAIL_SLOPE
        };
        let right = if last > 0 {
            (value(last) - value(last - 1)).max(MIN_TAIL_SLOPE)
        } else {
            MIN_TAIL_SLOPE
        };
        f.left -= i128::from(left);
        let mut slope = -left;
        for i in 0..last {
            let next = value(i + 1) - value(i);
            f.jump(lo + i as i64, i128::from(next - slope));
            slope = next;
        }
        f.jump(lo + last as i64, i128::from(right - slope));
    }

    pub(super) fn solve(mut self) -> Vec<i64> {
        let physical: Vec<Pair> = self
            .boundaries
            .into_iter()
            .map(|((a, b), f)| {
                let pair = Pair::new(a, b, f.clone());
                let combined = self.functions.entry((a, b)).or_default();
                combined.left += f.left;
                for (x, delta) in f.jumps {
                    combined.jump(x, delta);
                }
                pair
            })
            .collect();
        let edges: Vec<Pair> = self
            .functions
            .into_iter()
            .map(|((a, b), f)| Pair::new(a, b, f))
            .collect();
        let mut order: Vec<usize> = (0..edges.len()).collect();
        // Maximum confidence spanning forest. Pair IDs supply a stable tie break.
        order.sort_unstable_by(|&i, &j| {
            edges[j]
                .confidence
                .cmp(&edges[i].confidence)
                .then(i.cmp(&j))
        });
        let mut parent: Vec<usize> = (0..self.nodes).collect();
        let mut size = vec![1usize; self.nodes];
        let mut tree = vec![Vec::new(); self.nodes];
        let mut adjacent = vec![Vec::new(); self.nodes];
        for (i, e) in physical.iter().enumerate() {
            adjacent[e.a].push(i);
            adjacent[e.b].push(i);
        }
        for i in order {
            let e = &edges[i];
            let (mut a, mut b) = (root(&mut parent, e.a), root(&mut parent, e.b));
            if a == b {
                continue;
            }
            if size[a] < size[b] {
                std::mem::swap(&mut a, &mut b);
            }
            parent[b] = a;
            size[a] += size[b];
            tree[e.a].push((e.b, e.best));
            tree[e.b].push((e.a, -e.best));
        }
        let mut offsets = vec![0; self.nodes];
        let mut seen = vec![false; self.nodes];
        let mut anchor = vec![false; self.nodes];
        let mut preorder = Vec::with_capacity(self.nodes);
        let mut position = vec![0; self.nodes];
        let mut tree_parent: Vec<usize> = (0..self.nodes).collect();
        for r in 0..self.nodes {
            if seen[r] {
                continue;
            }
            seen[r] = true;
            anchor[r] = true;
            let mut stack = vec![r];
            while let Some(a) = stack.pop() {
                position[a] = preorder.len();
                preorder.push(a);
                for &(b, shift) in &tree[a] {
                    if !seen[b] {
                        seen[b] = true;
                        offsets[b] = offsets[a] + shift;
                        tree_parent[b] = a;
                        stack.push(b);
                    }
                }
            }
        }
        let mut end: Vec<usize> = position.iter().map(|&p| p + 1).collect();
        for &a in preorder.iter().rev() {
            let p = tree_parent[a];
            end[p] = end[p].max(end[a]);
        }
        // Overlaps establish a robust initial gauge. Once cores are selected,
        // refine their *actual* statistical cost. Retaining area-weighted
        // overlap penalties here would overwhelm boundary evidence and lock
        // inconsistent copies of a region together, especially at faults.
        let edges = physical;
        // Exact coordinate minimization over all slope breakpoints, rather
        // than just unit moves: large tile anchor differences cost no iterations.
        let mut candidates = Vec::new();
        loop {
            let mut changed = false;
            for a in 0..self.nodes {
                if anchor[a] {
                    continue;
                }
                candidates.clear();
                candidates.push(offsets[a]);
                for &i in &adjacent[a] {
                    let e = &edges[i];
                    for &(x, _, _) in &e.knots {
                        candidates.push(if a == e.a {
                            offsets[e.b] - x
                        } else {
                            offsets[e.a] + x
                        });
                    }
                }
                candidates.sort_unstable();
                candidates.dedup();
                let energy = |x| {
                    adjacent[a]
                        .iter()
                        .map(|&i| {
                            let e = &edges[i];
                            e.energy(if a == e.a {
                                offsets[e.b] - x
                            } else {
                                x - offsets[e.a]
                            })
                        })
                        .sum::<i128>()
                };
                let mut best = offsets[a];
                let mut cost = energy(best);
                for &x in &candidates {
                    let next = energy(x);
                    if next < cost {
                        best = x;
                        cost = next;
                    }
                }
                if best != offsets[a] {
                    offsets[a] = best;
                    changed = true;
                }
            }
            // Move a whole branch together: node moves alone can be blocked
            // by mutually consistent copies or by several weak boundaries.
            // Singleton branches were already covered by the node sweep.
            let mut cut = Vec::new();
            for &a in &preorder {
                let lo = position[a];
                let hi = end[a];
                if anchor[a] || hi - lo <= 1 {
                    continue;
                }
                cut.clear();
                candidates.clear();
                candidates.push(0);
                for (i, e) in edges.iter().enumerate() {
                    let inside_a = (lo..hi).contains(&position[e.a]);
                    let inside_b = (lo..hi).contains(&position[e.b]);
                    if inside_a == inside_b {
                        continue;
                    }
                    let sign = if inside_b { 1 } else { -1 };
                    let m = offsets[e.b] - offsets[e.a];
                    cut.push((i, m, sign));
                    for &(x, _, _) in &e.knots {
                        candidates.push((x - m) * sign);
                    }
                }
                candidates.sort_unstable();
                candidates.dedup();
                let energy = |shift| {
                    cut.iter()
                        .map(|&(i, m, sign)| edges[i].energy(m + sign * shift))
                        .sum::<i128>()
                };
                let mut best = 0;
                let mut cost = energy(0);
                for &shift in &candidates {
                    let next = energy(shift);
                    if next < cost {
                        best = shift;
                        cost = next;
                    }
                }
                if best != 0 {
                    for &b in &preorder[lo..hi] {
                        offsets[b] += best;
                    }
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        offsets
    }
}

struct Pair {
    a: usize,
    b: usize,
    left: i128,
    // breakpoint, prefix slope change, prefix (breakpoint * slope change)
    knots: Vec<(i64, i128, i128)>,
    best: i64,
    confidence: i128,
}
impl Pair {
    fn new(a: usize, b: usize, f: Function) -> Self {
        let mut knots = Vec::new();
        let (mut slope, mut moment) = (0, 0);
        for (x, delta) in f.jumps {
            if delta == 0 {
                continue;
            }
            slope += delta;
            moment += i128::from(x) * delta;
            knots.push((x, slope, moment));
        }
        let mut pair = Self {
            a,
            b,
            left: f.left,
            knots,
            best: 0,
            confidence: 0,
        };
        let mut best_cost = pair.energy(0);
        for &(x, _, _) in &pair.knots {
            let cost = pair.energy(x);
            if (cost, x.abs(), x) < (best_cost, pair.best.abs(), pair.best) {
                pair.best = x;
                best_cost = cost;
            }
        }
        // A nonconvex row can have a distant competing minimum. Include it
        // when estimating confidence, not only the two neighboring offsets.
        let alternative = pair
            .knots
            .iter()
            .filter(|&&(x, _, _)| x != pair.best)
            .map(|&(x, _, _)| pair.energy(x))
            .chain([pair.energy(pair.best - 1), pair.energy(pair.best + 1)])
            .min()
            .unwrap();
        pair.confidence = alternative - best_cost;
        pair
    }
    fn energy(&self, x: i64) -> i128 {
        let i = self.knots.partition_point(|&(k, _, _)| k <= x);
        let (slope, moment) = if i == 0 {
            (0, 0)
        } else {
            (self.knots[i - 1].1, self.knots[i - 1].2)
        };
        (self.left + slope) * i128::from(x) - moment
    }
}
fn root(parent: &mut [usize], mut a: usize) -> usize {
    while parent[a] != a {
        parent[a] = parent[parent[a]];
        a = parent[a];
    }
    a
}

/// Convex correction to an existing feasible field. Write k = base + delta:
/// delta is a zero-supply circulation, with shifted costs and bounds. Integrate
/// that circulation into pixel corrections and add them to the merged field.
/// This escapes multi-cycle deformation plateaus before true FromN refinement.
pub(super) fn refine_convex(
    psi: &[f32],
    costs: &GridCosts,
    n: &mut [i32],
) -> Result<(), UnwrapError> {
    let d = costs.dims;
    let base = jupwrap_grid::k_from_n(d, psi, n);
    let mut net = jupwrap_grid::dual_network(d, &vec![0; d.n_loops()]);
    for (e, &k) in base.iter().enumerate() {
        net.lower[e] = net.lower[e].saturating_sub(k);
        net.upper[e] = net.upper[e].saturating_sub(k);
    }
    struct Shifted<'a> {
        costs: &'a GridCosts,
        base: &'a [i32],
    }
    impl ArcCost for Shifted<'_> {
        fn cost(&self, e: usize, x: i32) -> i64 {
            self.costs.cost(e, self.base[e] + x)
        }
    }
    let change = jupwrap_mcf::solve_convex(&net, &Shifted { costs, base: &base })?;
    let delta = jupwrap_grid::integrate(d, &vec![0.0; d.npix()], &change)?;
    for (value, shift) in n.iter_mut().zip(delta) {
        *value = i32::try_from(i64::from(*value) + i64::from(shift))
            .map_err(|_| UnwrapError::InvalidInput("convex correction exceeds i32".into()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use jupwrap_core::{ArcCost, Dims};

    #[test]
    fn aggregated_function_matches_true_costs_in_both_orientations() {
        // A nonconvex row exercises negative slope jumps, asymmetry, and
        // both linear tails, far from zero relative tile offset.
        let costs = GridCosts {
            dims: Dims::new(1, 2),
            kt: 2,
            table: vec![700, 80, 0, 90, 180],
        };
        for (a, b) in [(0, 1), (1, 0)] {
            let mut graph = Secondary::new(2);
            graph.boundary(a, b, &costs, 0, 1007);
            graph.boundary(b, a, &costs, 0, -997);
            graph.overlap(0, 1, -1000, 17);
            let mut f = graph.boundaries.remove(&(0, 1)).unwrap();
            let overlap = graph.functions.remove(&(0, 1)).unwrap();
            f.left += overlap.left;
            for (x, delta) in overlap.jumps {
                f.jump(x, delta);
            }
            let pair = Pair::new(0, 1, f);
            let direct = |m: i64| {
                let sign = if a < b { 1 } else { -1 };
                i128::from(costs.cost(0, (1007 + sign * m) as i32))
                    + i128::from(costs.cost(0, (-997 - sign * m) as i32))
                    + 17 * i128::from((m + 1000).abs())
            };
            for m in [
                -5000, -1015, -1009, -1007, -1005, -1000, -999, -995, 0, 1234,
            ] {
                assert_eq!(pair.energy(m) - pair.energy(0), direct(m) - direct(0));
            }
        }
    }

    #[test]
    fn confidence_forest_resists_a_conflicting_weak_overlap() {
        let mut graph = Secondary::new(5);
        graph.overlap(0, 1, 1000, 100);
        graph.overlap(1, 2, -1200, 100);
        graph.overlap(1, 3, 20, 100);
        graph.overlap(2, 3, 1220, 100);
        graph.overlap(0, 2, 50, 1);
        assert_eq!(graph.solve(), [0, 1000, -200, 1020, 0]);
    }

    #[test]
    fn local_moves_use_all_boundaries_after_tree_initialization() {
        let mut graph = Secondary::new(4);
        boundary(&mut graph, 0, 1, 0, 10);
        boundary(&mut graph, 0, 2, 0, 9);
        boundary(&mut graph, 0, 3, 0, 9);
        boundary(&mut graph, 1, 2, 10, 8);
        boundary(&mut graph, 1, 3, 10, 8);
        assert_eq!(graph.solve(), [0, -10, 0, 0]);
    }

    #[test]
    fn forest_moves_release_mutually_blocked_region_copies() {
        let mut graph = Secondary::new(4);
        boundary(&mut graph, 0, 1, 0, 20);
        boundary(&mut graph, 1, 2, 0, 100);
        boundary(&mut graph, 0, 2, 10, 11);
        boundary(&mut graph, 0, 3, 0, 20);
        boundary(&mut graph, 2, 3, -10, 11);
        assert_eq!(graph.solve(), [0, 10, 10, 0]);
    }

    fn boundary(graph: &mut Secondary, a: usize, b: usize, target: i64, weight: i32) {
        let costs = GridCosts {
            dims: Dims::new(1, 2),
            kt: 16,
            table: (-16i32..=16).map(|k| weight * k.abs()).collect(),
        };
        graph.boundary(a, b, &costs, 0, -target);
    }
}
