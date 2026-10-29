//! Judge tests for jupwrap-init (MST initialization). Goldens from judge/ref/flows.py
//! (judge/gen/gen_mst.py). The MST is not unique under ties, so the tests check:
//! spanning tree of the dual graph, total weight == minimum (Kruskal reference),
//! and k == k0 + subtree routing on the RETURNED tree (exact), plus feasibility.

use jupwrap_core::{ArcCost, Dims, GridCosts, Network};
use jupwrap_init::mst_init;
use jupwrap_testkit as tk;

fn net_of(d: Dims, res: &[i8]) -> Network {
    tk::grid_network(d, res, i32::MIN / 2, i32::MAX / 2)
}

fn find(p: &mut Vec<usize>, mut x: usize) -> usize {
    while p[x] != x {
        p[x] = p[p[x]];
        x = p[x];
    }
    x
}

/// Exact routing of `need` (sum 0) on the tree rooted at ground.
fn route(d: Dims, tree: &[u32], need: &[i64]) -> Vec<i64> {
    let n = d.n_dual_nodes();
    let mut adj: Vec<Vec<(usize, usize)>> = vec![vec![]; n];
    for &e in tree {
        let (f, t) = d.edge_dual(e as usize);
        adj[f].push((t, e as usize));
        adj[t].push((f, e as usize));
    }
    let root = d.ground();
    let mut order = vec![root];
    let mut seen = vec![false; n];
    let mut pe = vec![usize::MAX; n];
    seen[root] = true;
    let mut i = 0;
    while i < order.len() {
        let u = order[i];
        i += 1;
        for &(v, e) in &adj[u] {
            if !seen[v] {
                seen[v] = true;
                pe[v] = e;
                order.push(v);
            }
        }
    }
    assert!(seen.iter().all(|&s| s), "tree does not span the dual graph");
    let mut sub = need.to_vec();
    let mut delta = vec![0i64; d.n_edges()];
    for &u in order[1..].iter().rev() {
        let e = pe[u];
        let (f, t) = d.edge_dual(e);
        let s = sub[u];
        let p = if f == u {
            delta[e] += s;
            t
        } else {
            delta[e] -= s;
            f
        };
        sub[p] += s;
    }
    delta
}

fn check(costs: &GridCosts, res: &[i8], k0: Option<&[i32]>, want_w: i64) {
    let d = costs.dims;
    let r = mst_init(costs, res);
    assert_eq!(r.k.len(), d.n_edges());
    assert_eq!(r.tree.len(), d.n_dual_nodes() - 1, "tree size");
    // spanning, acyclic
    let mut p: Vec<usize> = (0..d.n_dual_nodes()).collect();
    for &e in &r.tree {
        let (f, t) = d.edge_dual(e as usize);
        let (a, b) = (find(&mut p, f), find(&mut p, t));
        assert_ne!(a, b, "tree contains a cycle (edge {e})");
        p[a] = b;
    }
    // k0 rule (golden, or recomputed for recipes): argmin, ties smallest |k| then smaller k
    let k0: Vec<i32> = match k0 {
        Some(v) => v.to_vec(),
        None => (0..d.n_edges())
            .map(|e| {
                let mut best = 0;
                for k in -costs.kt..=costs.kt {
                    let (c, cb) = (costs.cost(e, k), costs.cost(e, best));
                    if c < cb || (c == cb && (k.abs(), k) < (best.abs(), best)) {
                        best = k;
                    }
                }
                best
            })
            .collect(),
    };
    let w = |e: usize| costs.cost(e, k0[e] + 1).min(costs.cost(e, k0[e] - 1)) - costs.cost(e, k0[e]);
    let tw: i64 = r.tree.iter().map(|&e| w(e as usize)).sum();
    assert_eq!(tw, want_w, "tree weight {tw} != minimum {want_w}");
    let net = net_of(d, res);
    let div0 = net.divergence(&k0);
    let need: Vec<i64> = (0..d.n_dual_nodes()).map(|v| net.supply[v] - div0[v]).collect();
    let delta = route(d, &r.tree, &need);
    for e in 0..d.n_edges() {
        assert_eq!(r.k[e] as i64, k0[e] as i64 + delta[e], "edge {e}: k != k0 + subtree routing");
    }
    assert_eq!(net.divergence(&r.k), net.supply, "infeasible");
}

#[test]
fn mst_init_cases() {
    let dir = tk::golden_dir("mst");
    let s = tk::meta_file(&dir.join("summary.json"));
    let cs = tk::cases(&dir);
    assert_eq!(cs.len() as u64, s["n_cases"].as_u64().unwrap());
    for c in cs {
        let m = tk::meta(&c);
        let costs = tk::grid_costs(&c, "table");
        let res = tk::npy(&c, "residues").i8();
        let k0 = tk::npy(&c, "k0").i32();
        eprintln!("{}", c.display());
        check(&costs, &res, Some(&k0), m["mst_weight"].as_i64().unwrap());
    }
}

#[test]
fn mst_init_zero_residues_feasible() {
    let dir = tk::golden_dir("mst").join("case_04");
    let costs = tk::grid_costs(&dir, "table");
    let res = vec![0i8; costs.dims.n_loops()];
    let r = mst_init(&costs, &res);
    let net = net_of(costs.dims, &res);
    assert_eq!(net.divergence(&r.k), net.supply);
}

#[test]
fn mst_init_recipe_1024() {
    let dir = tk::golden_dir("mst");
    let r = tk::meta_file(&dir.join("recipe.json"));
    let (costs, res) = tk::quadgrid(1024, 1024, r["seed"].as_u64().unwrap(), r["B"].as_i64().unwrap() as i32, r["resid_permille"].as_u64().unwrap());
    let t0 = std::time::Instant::now();
    let out = mst_init(&costs, &res);
    let dt = tk::secs(t0);
    eprintln!("mst_init 1024^2: {dt:.2} s");
    assert!(dt <= r["time_limit_s"].as_f64().unwrap(), "too slow: {dt:.1} s");
    drop(out);
    check(&costs, &res, None, r["mst_weight"].as_i64().unwrap());
}
