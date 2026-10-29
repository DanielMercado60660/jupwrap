use super::*;
use jupwrap_core::{wrap, CostMode, TAU};

#[test]
fn seedless_regions_and_zero_overlap_cover_thin_and_tiny_images() {
    for (rows, cols) in [(1, 17), (17, 1), (3, 7), (1, 1)] {
        let dims = Dims::new(rows, cols);
        let truth: Vec<f64> = (0..dims.npix())
            .map(|p| 0.4 * (p / cols) as f64 + 0.7 * (p % cols) as f64)
            .collect();
        let psi: Vec<f32> = truth.iter().map(|&v| wrap(v) as f32).collect();
        let corr = vec![0.99; dims.npix()];
        let input = UnwrapInput {
            dims,
            psi: &psi,
            corr: &corr,
            mask: None,
        };
        let mut params = UnwrapParams::new(4.0, CostMode::Smooth, InitMethod::Mcf);
        params.ntiles = (19, 23);
        params.nproc = 0;
        params.single_tile_reoptimize = false;
        params.tile_cost_thresh = i64::MAX;
        params.min_region_size = 1000;
        let output = unwrap_tiled(&input, &params).unwrap();
        for (p, &v) in truth.iter().enumerate() {
            assert!((f64::from(output.unw[p]) - v).abs() < 1e-5);
            assert!(
                (f64::from(output.unw[p]) - f64::from(psi[p]) - TAU * f64::from(output.n[p])).abs()
                    < 1e-5
            );
        }
    }
}

#[test]
fn entirely_masked_tiles_need_no_secondary_nodes() {
    let dims = Dims::new(3, 7);
    let psi = vec![1.0; dims.npix()];
    let corr = vec![0.99; dims.npix()];
    let mask = vec![0; dims.npix()];
    let input = UnwrapInput {
        dims,
        psi: &psi,
        corr: &corr,
        mask: Some(&mask),
    };
    let mut params = UnwrapParams::new(4.0, CostMode::Smooth, InitMethod::Mcf);
    params.ntiles = (2, 3);
    params.tile_overlap = (usize::MAX, usize::MAX);
    let output = unwrap_tiled(&input, &params).unwrap();
    assert!(output.unw.iter().all(|&v| v == 0.0));
    assert!(output.conncomp.iter().all(|&v| v == 0));
    assert_eq!(output.stats.n_components, 0);
}

#[test]
fn convex_correction_descends_and_reaches_pixel_optimum() {
    for (rows, cols) in [(7, 9), (1, 11), (11, 1)] {
        let dims = Dims::new(rows, cols);
        let psi: Vec<f32> = (0..dims.npix())
            .map(|p| wrap(0.4 * (p / cols) as f64 + 0.7 * (p % cols) as f64) as f32)
            .collect();
        let corr = vec![0.99; dims.npix()];
        let params = UnwrapParams::new(4.0, CostMode::Smooth, InitMethod::Mcf);
        let costs = jupwrap_costs::build_costs(dims, &psi, &corr, None, &params.cost_params());
        let mut n: Vec<i32> = (0..dims.npix())
            .map(|p| if p % 3 == 0 { 3 } else { -4 })
            .collect();
        let previous = total_cost(&costs, &jupwrap_grid::k_from_n(dims, &psi, &n));
        secondary::refine_convex(&psi, &costs, &mut n).unwrap();
        let next = total_cost(&costs, &jupwrap_grid::k_from_n(dims, &psi, &n));
        assert!(next <= previous);
        let (_, _, _, optimum) = solve_grid(&psi, &costs, Start::Mcf, &params).unwrap();
        assert_eq!(next, optimum);
    }
}
