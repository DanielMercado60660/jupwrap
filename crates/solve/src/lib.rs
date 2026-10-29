//! Single-tile unwrapping pipeline (DESIGN.md section 10).

use jupwrap_core::{
    min_component_size, total_cost, unwrapped_from_n, CostMode, GridCosts, InitMethod, UnwrapError,
    UnwrapInput, UnwrapOutput, UnwrapParams, UnwrapStats,
};
use std::time::Instant;

/// How to obtain the starting cycle field.
#[derive(Clone, Copy, Debug)]
pub enum Start<'a> {
    Mst,
    Mcf,
    /// Start from an existing pixel field.
    FromN(&'a [i32]),
}

struct Solved {
    k: Vec<i32>,
    n: Vec<i32>,
    init_cost: i64,
    final_cost: i64,
    n_residues: u64,
    seconds_init: f64,
    seconds_improve: f64,
}

fn solve(
    psi: &[f32],
    costs: &GridCosts,
    start: Start<'_>,
    params: &UnwrapParams,
) -> Result<Solved, UnwrapError> {
    let dims = costs.dims;
    if dims.nrow == 0
        || dims.ncol == 0
        || psi.len() != dims.npix()
        || costs.kt < 0
        || costs.kt > (i32::MAX - 1) / 2
        || dims.n_edges().checked_mul(costs.width()) != Some(costs.table.len())
    {
        return Err(UnwrapError::InvalidInput(
            "invalid grid dimensions, phase, or cost table".into(),
        ));
    }
    if let Start::FromN(n) = start {
        if n.len() != dims.npix() {
            return Err(UnwrapError::InvalidInput(
                "pixel field length != number of pixels".into(),
            ));
        }
        let counts = jupwrap_core::wrap_counts(dims, psi);
        for (e, &count) in counts.iter().enumerate() {
            let (p, q) = dims.edge_endpoints(e);
            if i32::try_from(i64::from(count) + i64::from(n[q]) - i64::from(n[p])).is_err() {
                return Err(UnwrapError::InvalidInput("cycle field exceeds i32".into()));
            }
        }
    }
    let started = Instant::now();
    let res = jupwrap_grid::residues(dims, psi);
    let n_residues = res.iter().filter(|&&r| r != 0).count() as u64;
    let mut k = match start {
        Start::FromN(n) => jupwrap_grid::k_from_n(dims, psi, n),
        Start::Mcf | Start::Mst => {
            let envelope = if params.cost == CostMode::Smooth {
                None
            } else {
                Some(jupwrap_costs::convex_envelope(costs))
            };
            let surrogate = envelope.as_ref().unwrap_or(costs);
            match start {
                Start::Mcf => {
                    let net = jupwrap_grid::dual_network(dims, &res);
                    jupwrap_mcf::solve_convex(&net, surrogate)?
                }
                Start::Mst => {
                    let mut k = jupwrap_init::mst_init(costs, &res).k;
                    jupwrap_nonconvex::improve_grid(surrogate, &mut k, &params.improve)?;
                    k
                }
                Start::FromN(_) => unreachable!(),
            }
        }
    };
    let init_cost = total_cost(costs, &k);
    let seconds_init = started.elapsed().as_secs_f64();
    let started = Instant::now();
    if !(params.cost == CostMode::Smooth && matches!(start, Start::Mcf)) {
        jupwrap_nonconvex::improve_grid(costs, &mut k, &params.improve)?;
    }
    let final_cost = total_cost(costs, &k);
    let seconds_improve = started.elapsed().as_secs_f64();
    let n = jupwrap_grid::integrate(dims, psi, &k)?;
    Ok(Solved {
        k,
        n,
        init_cost,
        final_cost,
        n_residues,
        seconds_init,
        seconds_improve,
    })
}

/// Solve one grid, returning cycles, integrated pixel field and true costs.
pub fn solve_grid(
    psi: &[f32],
    costs: &GridCosts,
    start: Start<'_>,
    params: &UnwrapParams,
) -> Result<(Vec<i32>, Vec<i32>, i64, i64), UnwrapError> {
    let solved = solve(psi, costs, start, params)?;
    Ok((solved.k, solved.n, solved.init_cost, solved.final_cost))
}

/// Full single-tile unwrap. Masked pixels receive zero phase and label.
pub fn unwrap_single(
    input: &UnwrapInput<'_>,
    params: &UnwrapParams,
) -> Result<UnwrapOutput, UnwrapError> {
    let total_started = Instant::now();
    input.validate()?;
    params.validate()?;
    let started = Instant::now();
    let costs = jupwrap_costs::build_costs(
        input.dims,
        input.psi,
        input.corr,
        input.mask,
        &params.cost_params(),
    );
    let seconds_costs = started.elapsed().as_secs_f64();
    let start = match params.init {
        InitMethod::Mcf => Start::Mcf,
        InitMethod::Mst => Start::Mst,
    };
    let solved = solve(input.psi, &costs, start, params)?;
    let mut unw = unwrapped_from_n(input.psi, &solved.n);
    if let Some(mask) = input.mask {
        for (value, &valid) in unw.iter_mut().zip(mask) {
            if valid == 0 {
                *value = 0.0;
            }
        }
    }
    let (conncomp, n_components) = jupwrap_conncomp::label_components(
        &costs,
        &solved.k,
        input.mask,
        params.conncomp_cost_thresh,
        min_component_size(params.min_conncomp_frac, input.dims.npix()),
    );
    Ok(UnwrapOutput {
        n: solved.n,
        unw,
        conncomp,
        stats: UnwrapStats {
            n_residues: solved.n_residues,
            init_cost: solved.init_cost,
            final_cost: solved.final_cost,
            n_components,
            n_tiles: 1,
            seconds_costs,
            seconds_init: solved.seconds_init,
            seconds_improve: solved.seconds_improve,
            seconds_total: total_started.elapsed().as_secs_f64(),
        },
    })
}
