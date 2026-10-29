//! Judge tests (Rust part) for jupwrap-tiling. The end-to-end part is
//! judge/tests/test_e2e_tiled.py.

use jupwrap_core::{wrap, Dims, UnwrapInput, UnwrapParams, CostMode, InitMethod, TAU};
use jupwrap_testkit as tk;
use jupwrap_tiling::{tile_bounds, unwrap_tiled};

#[test]
fn tile_bounds_formula() {
    assert_eq!(tile_bounds(100, 90, (1, 1), (0, 0)), vec![(0, 100, 0, 90)]);
    assert_eq!(tile_bounds(100, 90, (1, 1), (16, 16)), vec![(0, 100, 0, 90)]);
    assert_eq!(
        tile_bounds(100, 90, (3, 2), (5, 7)),
        vec![(0, 38, 0, 52), (0, 38, 38, 90), (28, 71, 0, 52), (28, 71, 38, 90), (61, 100, 0, 52), (61, 100, 38, 90)]
    );
    let t = tile_bounds(7, 5, (7, 5), (0, 0));
    assert_eq!(t.len(), 35);
    assert_eq!(t[0], (0, 1, 0, 1));
    assert_eq!(t[34], (6, 7, 4, 5));
    assert_eq!(tile_bounds(33000, 512, (12, 1), (128, 0))[1], (2622, 5628, 0, 512));
}

fn synth(nrow: usize, ncol: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut g = tk::SplitMix(seed);
    let mut u = || (g.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
    let mut psi = vec![0f32; nrow * ncol];
    let mut corr = vec![0f32; nrow * ncol];
    for r in 0..nrow {
        for c in 0..ncol {
            let phi = 0.03 * ((c as f64 - ncol as f64 / 2.0).powi(2) / 4.0) + 0.5 * r as f64;
            psi[r * ncol + c] = wrap(phi + 0.6 * (u() - 0.5)) as f32;
            corr[r * ncol + c] = (0.4 + 0.5 * u()) as f32;
        }
    }
    (psi, corr)
}

#[test]
fn one_tile_equals_single_tile_pipeline() {
    let d = Dims::new(64, 72);
    let (psi, corr) = synth(64, 72, 3);
    let inp = UnwrapInput { dims: d, psi: &psi, corr: &corr, mask: None };
    let p = UnwrapParams::new(5.0, CostMode::Smooth, InitMethod::Mcf);
    let a = unwrap_tiled(&inp, &p).expect("tiled");
    let b = jupwrap_solve::unwrap_single(&inp, &p).expect("single");
    assert_eq!(a.n, b.n);
    assert_eq!(a.unw, b.unw);
    assert_eq!(a.conncomp, b.conncomp);
}

#[test]
fn small_tiled_is_congruent_and_complete() {
    let d = Dims::new(96, 80);
    let (psi, corr) = synth(96, 80, 4);
    let inp = UnwrapInput { dims: d, psi: &psi, corr: &corr, mask: None };
    for reopt in [true, false] {
        let mut p = UnwrapParams::new(5.0, CostMode::Defo, InitMethod::Mcf);
        p.ntiles = (2, 3);
        p.tile_overlap = (8, 8);
        p.min_region_size = 20;
        p.single_tile_reoptimize = reopt;
        let o = unwrap_tiled(&inp, &p).expect("tiled");
        assert_eq!(o.unw.len(), d.npix());
        assert_eq!(o.conncomp.len(), d.npix());
        assert_eq!(o.n.len(), d.npix());
        for i in 0..d.npix() {
            let x = o.unw[i] as f64 - psi[i] as f64;
            assert!((x - TAU * (x / TAU).round()).abs() < 1e-4, "pixel {i} not congruent");
            assert!(((o.unw[i] as f64 - psi[i] as f64) / TAU - o.n[i] as f64).abs() < 1e-3, "unw != psi + 2 pi n");
        }
    }
}
