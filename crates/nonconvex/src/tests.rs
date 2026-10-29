use super::*;

fn quadratic(dims: Dims, centers: &[i32]) -> GridCosts {
    let mut table = Vec::new();
    for &center in centers {
        for x in -4i32..=4 {
            table.push(100 * (x - center).pow(2));
        }
    }
    GridCosts { dims, kt: 4, table }
}

fn divergence(dims: Dims, k: &[i32]) -> Vec<i64> {
    let mut div = vec![0; dims.n_dual_nodes()];
    for (e, &x) in k.iter().enumerate() {
        let (u, v) = dims.edge_dual(e);
        div[u] += i64::from(x);
        div[v] -= i64::from(x);
    }
    div
}

#[test]
fn thin_grids_optimize_self_loops() {
    for dims in [Dims::new(1, 5), Dims::new(5, 1)] {
        let centers = [-2, 1, 0, 2];
        let costs = quadratic(dims, &centers);
        let mut k = [0; 4];
        let stats = improve_grid(&costs, &mut k, &ImproveOptions::default()).unwrap();
        assert_eq!(k, centers);
        assert_eq!(stats.initial_cost, 900);
        assert_eq!(stats.final_cost, 0);
        assert_eq!(divergence(dims, &k), vec![0]);
    }
}

#[test]
fn boundary_circulation_and_pass_limit_have_exact_stats() {
    let dims = Dims::new(2, 2);
    let costs = quadratic(dims, &[-1, 0, -1, 0]);
    for max_passes in [0, 1] {
        let mut k = [0; 4];
        let stats = improve_grid(
            &costs,
            &mut k,
            &ImproveOptions {
                max_passes,
                time_limit_s: 0.0,
            },
        )
        .unwrap();
        assert_eq!(k, [-1, 0, -1, 0]);
        assert_eq!(divergence(dims, &k), vec![0, 0]);
        assert_eq!(stats.initial_cost, 200);
        assert_eq!(stats.final_cost, 0);
        assert_eq!(stats.moves, 1);
        assert_eq!(stats.passes, if max_passes == 0 { 2 } else { 1 });
    }
}

#[test]
fn expired_deadline_returns_unchanged_feasible_field() {
    let costs = quadratic(Dims::new(2, 2), &[-1, 0, -1, 0]);
    let mut k = [0; 4];
    let stats = improve_grid(
        &costs,
        &mut k,
        &ImproveOptions {
            max_passes: 0,
            time_limit_s: 1e-300,
        },
    )
    .unwrap();
    assert_eq!(k, [0; 4]);
    assert_eq!(
        stats,
        ImproveStats {
            initial_cost: 200,
            final_cost: 200,
            moves: 0,
            passes: 0
        }
    );
}

#[test]
fn concave_edge_does_not_make_a_false_negative_cycle() {
    let mut costs = quadratic(Dims::new(2, 2), &[0; 4]);
    costs.table[3] = 0;
    costs.table[4] = 10;
    costs.table[5] = 0;
    let mut k = [0; 4];
    let stats = improve_grid(&costs, &mut k, &ImproveOptions::default()).unwrap();
    assert_eq!(k, [0; 4]);
    assert_eq!(stats.final_cost, 10);
    assert_eq!(stats.moves, 0);
}

#[test]
fn small_convex_grids_match_exhaustive_pixel_fields() {
    let dims = Dims::new(2, 2);
    // Exhaustive oracle uses integer pixel fields, independently of the
    // residual search. Different centers and weights exercise ties too.
    for seed in 0..24 {
        let centers: Vec<_> = (0..4).map(|e| (seed / 3i32.pow(e) % 3) - 1).collect();
        let mut costs = quadratic(dims, &centers);
        for e in 0..4 {
            for entry in &mut costs.table[e * 9..(e + 1) * 9] {
                *entry *= 1 + ((seed as usize + 2 * e) % 5) as i32;
            }
        }
        let mut best = i64::MAX;
        for a in -4..=4 {
            for b in -4..=4 {
                for c in -4..=4 {
                    let n = [0, a, b, c];
                    let field: Vec<_> = (0..4)
                        .map(|e| {
                            let (p, q) = dims.edge_endpoints(e);
                            n[q] - n[p]
                        })
                        .collect();
                    best = best.min(total_cost(&costs, &field));
                }
            }
        }
        let mut k = [0; 4];
        let stats = improve_grid(&costs, &mut k, &ImproveOptions::default()).unwrap();
        assert_eq!(stats.final_cost, best, "seed {seed}");
        assert_eq!(divergence(dims, &k), vec![0, 0]);
    }
}

#[test]
fn empty_grid_is_already_stationary() {
    let costs = GridCosts {
        dims: Dims::new(1, 1),
        kt: 0,
        table: vec![],
    };
    assert_eq!(
        improve_grid(&costs, &mut [], &ImproveOptions::default()).unwrap(),
        ImproveStats::default()
    );
}
