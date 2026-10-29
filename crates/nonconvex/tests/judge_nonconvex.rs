//! Judge tests for jupwrap-nonconvex. Goldens from judge/gen/gen_nonconvex.py:
//! convex optima from OR-tools (exact), nonconvex optima from OR-tools CP-SAT
//! over pixel fields (exact, small grids), gap limits calibrated from a simple
//! reference cycle-canceling heuristic (summary.json).

use jupwrap_core::{total_cost, ArcCost, GridCosts, ImproveOptions, McfError};
use jupwrap_nonconvex::improve_grid;
use jupwrap_testkit as tk;
use std::path::PathBuf;

fn cases(kind: &str) -> Vec<PathBuf> {
    let dir = tk::golden_dir("nonconvex");
    let s = tk::meta_file(&dir.join("summary.json"));
    let all = tk::cases(&dir);
    assert_eq!(all.len() as u64, s["n_cases"].as_u64().unwrap());
    all.into_iter().filter(|c| tk::meta(c)["kind"] == kind).collect()
}

fn divergence(costs: &GridCosts, k: &[i32]) -> Vec<i64> {
    let d = costs.dims;
    let mut v = vec![0i64; d.n_dual_nodes()];
    for e in 0..d.n_edges() {
        let (f, t) = d.edge_dual(e);
        v[f] += k[e] as i64;
        v[t] -= k[e] as i64;
    }
    v
}

/// Best single-pixel shift (n_p += 1 or -= 1) cost change; must be >= 0.
fn best_pixel_move(costs: &GridCosts, k: &[i32]) -> (i64, usize) {
    let d = costs.dims;
    let mut up = vec![0i64; d.npix()];
    let mut dn = vec![0i64; d.npix()];
    for e in 0..d.n_edges() {
        let (p, q) = d.edge_endpoints(e);
        let c0 = costs.cost(e, k[e]);
        let cm = costs.cost(e, k[e] - 1) - c0;
        let cp = costs.cost(e, k[e] + 1) - c0;
        up[p] += cm;
        up[q] += cp;
        dn[p] += cp;
        dn[q] += cm;
    }
    let mut best = (0i64, usize::MAX);
    for p in 0..d.npix() {
        let m = up[p].min(dn[p]);
        if m < best.0 {
            best = (m, p);
        }
    }
    best
}

/// Runs improve_grid and checks invariants; returns (final cost, seconds).
fn run(c: &PathBuf) -> (i64, f64) {
    let m = tk::meta(c);
    let costs = tk::grid_costs(c, "table");
    let mut k = tk::npy(c, "k_start").i32();
    let start = total_cost(&costs, &k);
    assert_eq!(start, m["start_cost"].as_i64().unwrap(), "{}: golden start cost", c.display());
    let div0 = divergence(&costs, &k);
    let t0 = std::time::Instant::now();
    let st = improve_grid(&costs, &mut k, &ImproveOptions::default()).unwrap_or_else(|e| panic!("{}: {e}", c.display()));
    let dt = tk::secs(t0);
    let fin = total_cost(&costs, &k);
    assert_eq!(divergence(&costs, &k), div0, "{}: divergence changed (not a circulation)", c.display());
    assert_eq!(st.initial_cost, start, "{}: stats.initial_cost", c.display());
    assert_eq!(st.final_cost, fin, "{}: stats.final_cost", c.display());
    assert!(fin <= start, "{}: cost increased {start} -> {fin}", c.display());
    let (bm, p) = best_pixel_move(&costs, &k);
    assert!(bm >= 0, "{}: single-pixel move at pixel {p} lowers cost by {}", c.display(), -bm);
    (fin, dt)
}

#[test]
fn convex_instances_reach_global_optimum() {
    let cs = cases("convex");
    assert_eq!(cs.len(), 3);
    for c in cs {
        let (fin, dt) = run(&c);
        let opt = tk::meta(&c)["opt_cost"].as_i64().unwrap();
        eprintln!("{}: {fin} vs opt {opt} in {dt:.2} s", c.display());
        assert_eq!(fin, opt, "{}: convex costs must reach the exact optimum", c.display());
        assert!(dt < 60.0, "{}: too slow {dt:.1} s", c.display());
    }
}

fn near_optimum(kind: &str, expected: usize) {
    let s = tk::meta_file(&tk::golden_dir("nonconvex").join("summary.json"));
    let mean_lim = s[format!("{kind}_mean_gap_limit")].as_f64().unwrap();
    let max_lim = s[format!("{kind}_max_gap_limit")].as_f64().unwrap();
    let cs = cases(kind);
    assert_eq!(cs.len(), expected);
    let mut gaps = vec![];
    for c in &cs {
        let (fin, dt) = run(c);
        let opt = tk::meta(c)["opt_cost"].as_i64().unwrap();
        assert!(fin >= opt, "{}: below the proven optimum ({fin} < {opt}): bug", c.display());
        let g = (fin - opt) as f64 / opt as f64;
        eprintln!("{}: {fin} opt {opt} gap {g:.4} ({dt:.2} s)", c.display());
        assert!(g <= max_lim, "{}: gap {g:.4} > {max_lim}", c.display());
        assert!(dt < 20.0, "{}: too slow", c.display());
        gaps.push(g);
    }
    let mean = gaps.iter().sum::<f64>() / gaps.len() as f64;
    eprintln!("{kind}: mean gap {mean:.4} (limit {mean_lim})");
    assert!(mean <= mean_lim, "{kind}: mean gap {mean:.4} > {mean_lim}");
}

#[test]
fn defo_instances_near_optimum() {
    near_optimum("defo", 16);
}

#[test]
fn plateau_instances_near_optimum() {
    near_optimum("plateau", 8);
}

#[test]
fn medium_instances_invariants_and_time() {
    let cs = cases("medium");
    assert_eq!(cs.len(), 2);
    for c in cs {
        let (fin, dt) = run(&c);
        let lim = tk::meta(&c)["time_limit_s"].as_f64().unwrap();
        eprintln!("{}: final {fin} in {dt:.2} s", c.display());
        assert!(dt <= lim, "{}: {dt:.1} s > {lim} s", c.display());
    }
}

#[test]
fn rejects_bad_length() {
    let c = cases("convex").remove(0);
    let costs = tk::grid_costs(&c, "table");
    let mut k = vec![0i32; costs.dims.n_edges() - 1];
    assert!(matches!(improve_grid(&costs, &mut k, &ImproveOptions::default()), Err(McfError::InvalidInput(_))));
}
