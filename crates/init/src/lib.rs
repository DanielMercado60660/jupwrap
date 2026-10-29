//! jupwrap-init: minimum-spanning-tree initialization
//! (DESIGN.md section 8).

use jupwrap_core::{ArcCost, GridCosts};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MstInit {
    /// Feasible cycle field, one entry per edge.
    pub k: Vec<i32>,
    /// Edge indices of the spanning tree of the dual graph (n_dual_nodes - 1 edges).
    pub tree: Vec<u32>,
}

/// MST initialization: `k0` = per-edge table argmin, MST of the dual graph
/// under weights `w_e`, then subtree routing of the residual supplies.
///
/// Argmin ties prefer the smallest absolute flow, then the smaller signed flow.
/// Weights use [`GridCosts::cost`] even at table endpoints. Kruskal breaks equal
/// weights by edge index for determinism; parallel arcs remain distinct and
/// self-loops retain their minimizing flow without entering the tree.
/// All traversals, including union-find, are iterative.
pub fn mst_init(costs: &GridCosts, residues: &[i8]) -> MstInit {
    let dims = costs.dims;
    let n = dims.n_dual_nodes();
    let m = dims.n_edges();
    let root = dims.ground();
    assert_eq!(residues.len(), dims.n_loops(), "residue length");

    let mut need = vec![0_i64; n];
    for (v, &residue) in residues.iter().enumerate() {
        need[v] = -i64::from(residue);
        need[root] += i64::from(residue);
    }
    let mut k = Vec::with_capacity(m);
    let mut weighted = Vec::with_capacity(m);
    for e in 0..m {
        let mut best = 0;
        let mut best_cost = costs.cost(e, best);
        for x in -costs.kt..=costs.kt {
            let c = costs.cost(e, x);
            if c < best_cost || (c == best_cost && (x.abs(), x) < (best.abs(), best)) {
                best = x;
                best_cost = c;
            }
        }
        k.push(best);
        let (from, to) = dims.edge_dual(e);
        // Start from the supplies, then subtract out-minus-in divergence(k0).
        need[from] -= i64::from(best);
        need[to] += i64::from(best);
        if from != to {
            let weight = costs.cost(e, best + 1).min(costs.cost(e, best - 1)) - best_cost;
            weighted.push((weight, e));
        }
    }

    weighted.sort_unstable();
    let mut sets = DisjointSets::new(n);
    let mut tree = Vec::with_capacity(n - 1);
    for (_, e) in weighted {
        let (from, to) = dims.edge_dual(e);
        if sets.join(from, to) {
            tree.push(u32::try_from(e).expect("edge index exceeds u32"));
            if tree.len() == n - 1 {
                break;
            }
        }
    }
    assert_eq!(tree.len(), n - 1, "dual graph is disconnected");
    drop(sets);

    // Compact adjacency lists: each tree edge contributes two directed links.
    let mut first = vec![usize::MAX; n];
    let mut next = Vec::with_capacity(2 * tree.len());
    let mut links = Vec::with_capacity(2 * tree.len());
    for &edge in &tree {
        let (a, b) = dims.edge_dual(edge as usize);
        for (u, v) in [(a, b), (b, a)] {
            next.push(first[u]);
            links.push((v, edge));
            first[u] = links.len() - 1;
        }
    }
    let mut parent = vec![usize::MAX; n];
    let mut parent_edge = vec![0_u32; n];
    let mut order = Vec::with_capacity(n);
    parent[root] = root;
    order.push(root);
    let mut i = 0;
    while i < order.len() {
        let u = order[i];
        i += 1;
        let mut link = first[u];
        while link != usize::MAX {
            let (v, edge) = links[link];
            if parent[v] == usize::MAX {
                parent[v] = u;
                parent_edge[v] = edge;
                order.push(v);
            }
            link = next[link];
        }
    }
    // Parent-before-child order reversed is a postorder sufficient for sums.
    // Only parent links are needed below, so release the undirected adjacency.
    drop(first);
    drop(next);
    drop(links);
    for &u in order[1..].iter().rev() {
        let e = parent_edge[u] as usize;
        let (from, _) = dims.edge_dual(e);
        let delta = if from == u { need[u] } else { -need[u] };
        k[e] = i32::try_from(i64::from(k[e]) + delta).expect("routed flow exceeds i32");
        need[parent[u]] += need[u];
    }
    MstInit { k, tree }
}

struct DisjointSets {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl DisjointSets {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn join(&mut self, a: usize, b: usize) -> bool {
        let mut a = self.find(a);
        let mut b = self.find(b);
        if a == b {
            return false;
        }
        if self.rank[a] < self.rank[b] {
            std::mem::swap(&mut a, &mut b);
        }
        self.parent[b] = a;
        if self.rank[a] == self.rank[b] {
            self.rank[a] += 1;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jupwrap_core::Dims;

    #[test]
    fn self_loops_keep_signed_argmin_ties() {
        let costs = GridCosts {
            dims: Dims::new(1, 5),
            kt: 2,
            table: vec![
                7, 0, 5, 0, 7, // -1 and +1: prefer -1.
                0, 8, 9, 8, 0, // -2 and +2: prefer -2.
                0, 0, 0, 0, 0, // Flat row: prefer zero.
                5, 4, 3, 2, 1, // Unique minimum at the table boundary.
            ],
        };
        let out = mst_init(&costs, &[]);
        assert!(out.tree.is_empty());
        assert_eq!(out.k, [-1, -2, 0, 2]);
    }

    #[test]
    fn boundary_minimum_uses_extension_and_routes_against_arc() {
        let costs = GridCosts {
            dims: Dims::new(2, 2),
            kt: 1,
            table: vec![1000, 600, 0, 150, 0, 150, 300, 0, 300, 400, 0, 400],
        };
        // e0 has weight 100 from the extension, rather than 600 from its
        // in-table neighbor. Its k0=1 enters the loop, whose supply is +1;
        // routing must send two units back toward ground on this same edge.
        let out = mst_init(&costs, &[-1]);
        assert_eq!(out.tree, [0]);
        assert_eq!(out.k, [-1, 0, 0, 0]);
    }

    #[test]
    fn million_node_chain_routes_without_recursion() {
        let nloops = 1_000_000;
        let dims = Dims::new(2, nloops + 1);
        let mut table = Vec::with_capacity(3 * dims.n_edges());
        for e in 0..dims.n_edges() {
            let weight = if e > 0 && e < nloops {
                1 // Every link along the chain is mandatory in the MST.
            } else if e == 0 {
                2 // The sole cheapest connection to ground is at the left end.
            } else {
                3
            };
            table.extend_from_slice(&[weight, 0, weight]);
        }
        let costs = GridCosts { dims, kt: 1, table };
        let mut residues = vec![0; nloops];
        residues[nloops - 1] = 1;
        let out = mst_init(&costs, &residues);
        assert_eq!(out.tree.len(), nloops);
        assert!(out.k[..nloops].iter().all(|&k| k == 1));
        assert!(out.k[nloops..].iter().all(|&k| k == 0));
    }
}
