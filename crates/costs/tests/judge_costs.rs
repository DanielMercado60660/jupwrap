//! Judge tests for jupwrap-costs. Goldens from judge/ref/costs.py
//! (judge/gen/gen_costs.py). Float quantities within 1e-12 / 1e-9; quantized
//! tables within +-1 unit (rounding ties); convex envelope exact.

use jupwrap_core::{wrap, ArcCost, CostMode, CostParams, Dims, GridCosts, COST_CAP, KT};
use jupwrap_costs::{build_costs, convex_envelope, effective_coherence, mean_gradients, pixel_sigma2};
use jupwrap_testkit as tk;
use std::path::PathBuf;

struct Case {
    name: String,
    dims: Dims,
    psi: Vec<f32>,
    corr: Vec<f32>,
    mask: Option<Vec<u8>>,
    params: CostParams,
    dir: PathBuf,
}

fn load() -> Vec<Case> {
    let dir = tk::golden_dir("costs");
    let s = tk::meta_file(&dir.join("summary.json"));
    let cs = tk::cases(&dir);
    assert_eq!(cs.len() as u64, s["n_cases"].as_u64().unwrap());
    cs.into_iter()
        .map(|c| {
            let m = tk::meta(&c);
            let mode = if m["mode"] == "smooth" { CostMode::Smooth } else { CostMode::Defo };
            let mut p = CostParams::new(mode, m["nlooks"].as_f64().unwrap());
            p.phase_grad_window = (m["window"][0].as_u64().unwrap() as usize, m["window"][1].as_u64().unwrap() as usize);
            p.sigma2_signal = m["sigma2_signal"].as_f64().unwrap();
            p.p_discontinuity = m["p_discontinuity"].as_f64().unwrap();
            Case {
                name: c.file_name().unwrap().to_string_lossy().to_string(),
                dims: tk::dims_of(&m),
                psi: tk::npy(&c, "psi").f32(),
                corr: tk::npy(&c, "corr").f32(),
                mask: if m["masked"].as_bool().unwrap() { Some(tk::npy(&c, "mask").u8()) } else { None },
                params: p,
                dir: c,
            }
        })
        .collect()
}

#[test]
fn effective_coherence_matches() {
    for c in load() {
        let want = tk::npy(&c.dir, "effcoh").f64();
        for (i, &r) in c.corr.iter().enumerate() {
            let got = effective_coherence(r, c.params.nlooks);
            assert!((got - want[i]).abs() <= 1e-12, "{} pixel {i}: corr {r} -> {got} vs {}", c.name, want[i]);
        }
    }
}

#[test]
fn pixel_sigma2_matches() {
    for c in load() {
        let want = tk::npy(&c.dir, "sigma2").f64();
        for (i, &r) in c.corr.iter().enumerate() {
            let got = pixel_sigma2(r, c.params.nlooks);
            assert!((got - want[i]).abs() <= 1e-12 * want[i].abs().max(1.0), "{} pixel {i}: {got} vs {}", c.name, want[i]);
        }
    }
}

#[test]
fn mean_gradients_match() {
    for c in load() {
        let want = tk::npy(&c.dir, "mu").f64();
        let check = tk::npy(&c.dir, "check").u8();
        let got = mean_gradients(c.dims, &c.psi, c.mask.as_deref(), c.params.phase_grad_window);
        assert_eq!(got.len(), c.dims.n_edges(), "{}", c.name);
        for e in 0..got.len() {
            if check[e] == 1 {
                assert!(wrap(got[e] - want[e]).abs() <= 1e-9, "{} edge {e}: {} vs {}", c.name, got[e], want[e]);
            }
        }
    }
}

fn check_tables(mode: CostMode) -> usize {
    let mut n = 0;
    for c in load().into_iter().filter(|c| c.params.mode == mode) {
        n += 1;
        let want = tk::npy(&c.dir, "table").i32();
        let check = tk::npy(&c.dir, "check").u8();
        let got = build_costs(c.dims, &c.psi, &c.corr, c.mask.as_deref(), &c.params);
        assert_eq!(got.dims, c.dims, "{}", c.name);
        assert_eq!(got.kt, KT, "{}", c.name);
        assert_eq!(got.table.len(), want.len(), "{}", c.name);
        let w = got.width();
        let mut exact = 0usize;
        let mut total = 0usize;
        for e in 0..c.dims.n_edges() {
            if check[e] == 0 {
                continue;
            }
            for j in 0..w {
                let (g, r) = (got.table[e * w + j], want[e * w + j]);
                assert!((g - r).abs() <= 1, "{} edge {e} k={}: {g} vs {r}", c.name, j as i32 - KT);
                total += 1;
                exact += (g == r) as usize;
            }
            if let Some(m) = &c.mask {
                let (p, q) = c.dims.edge_endpoints(e);
                if m[p] == 0 || m[q] == 0 {
                    assert!(got.row(e).iter().all(|&v| v == 0), "{} masked edge {e} must be all zero", c.name);
                }
            }
        }
        assert!(exact as f64 >= 0.999 * total as f64, "{}: only {exact}/{total} entries exact", c.name);
    }
    n
}

#[test]
fn smooth_tables_match() {
    assert_eq!(check_tables(CostMode::Smooth), 5);
}

#[test]
fn defo_tables_match() {
    assert_eq!(check_tables(CostMode::Defo), 4);
}

#[test]
fn convex_envelope_matches_on_model_tables() {
    for c in load() {
        let t = GridCosts { dims: c.dims, kt: KT, table: tk::npy(&c.dir, "table").i32() };
        let env = convex_envelope(&t);
        assert_eq!(env.dims, c.dims);
        assert_eq!(env.kt, KT);
        assert_eq!(env.table, tk::npy(&c.dir, "envelope").i32(), "{}", c.name);
    }
}

#[test]
fn convex_envelope_matches_on_random_rows() {
    let d = tk::golden_dir("costs").join("envelope_random");
    let rows = tk::npy(&d, "rows");
    let nrow = rows.shape[0];
    // pack rows as a 1 x (nrow+1) grid (nrow horizontal edges)
    let t = GridCosts { dims: Dims::new(1, nrow + 1), kt: KT, table: rows.i32() };
    let env = convex_envelope(&t);
    assert_eq!(env.table, tk::npy(&d, "envelope").i32());
    // convexity of the surrogate, including the fixed linear extension
    for e in 0..nrow {
        let incs: Vec<i64> = (-KT - 3..KT + 3).map(|k| env.cost(e, k + 1) - env.cost(e, k)).collect();
        assert!(incs.windows(2).all(|w| w[0] <= w[1]), "row {e} not convex");
    }
}

#[test]
fn build_costs_large_is_fast_and_normalized() {
    let d = Dims::new(2048, 2048);
    let mut g = tk::SplitMix(5);
    let mut u = || (g.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
    let psi: Vec<f32> = (0..d.npix()).map(|_| ((u() * 2.0 - 1.0) * std::f64::consts::PI) as f32).collect();
    let corr: Vec<f32> = (0..d.npix()).map(|_| u() as f32).collect();
    for mode in [CostMode::Smooth, CostMode::Defo] {
        let p = CostParams::new(mode, 9.0);
        let t0 = std::time::Instant::now();
        let t = build_costs(d, &psi, &corr, None, &p);
        let dt = tk::secs(t0);
        eprintln!("build_costs 2048^2 {mode:?}: {dt:.2} s");
        assert!(dt < 30.0, "too slow: {dt:.1} s");
        assert_eq!(t.table.len(), d.n_edges() * (2 * KT as usize + 1));
        for e in (0..d.n_edges()).step_by(997) {
            let r = t.row(e);
            assert_eq!(*r.iter().min().unwrap(), 0);
            assert!(r.iter().all(|&v| v >= 0 && (v as i64) <= COST_CAP));
        }
    }
}
