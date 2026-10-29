//! Tiled unwrapping with reliable regions and a secondary statistical network.
use jupwrap_core::{
    min_component_size, total_cost, unwrapped_from_n, wrap_count, ArcCost, CostMode, Dims,
    GridCosts, InitMethod, UnwrapError, UnwrapInput, UnwrapOutput, UnwrapParams, UnwrapStats,
};
use jupwrap_solve::{solve_grid, Start};
use rayon::prelude::*;
use std::{collections::VecDeque, time::Instant};
mod secondary;
use secondary::Secondary;
type Bounds = (usize, usize, usize, usize);
const INVALID: u32 = u32::MAX;

#[cfg(test)]
mod tests;

/// Half-open extended tile bounds, in row-major tile order.
pub fn tile_bounds(
    nrow: usize,
    ncol: usize,
    ntiles: (usize, usize),
    overlap: (usize, usize),
) -> Vec<(usize, usize, usize, usize)> {
    let mut out = Vec::new();
    if ntiles.0 == 0 || ntiles.1 == 0 {
        return out;
    }
    for i in 0..ntiles.0 {
        for j in 0..ntiles.1 {
            out.push((
                split(i, nrow, ntiles.0).saturating_sub(overlap.0),
                split(i + 1, nrow, ntiles.0)
                    .saturating_add(overlap.0)
                    .min(nrow),
                split(j, ncol, ntiles.1).saturating_sub(overlap.1),
                split(j + 1, ncol, ntiles.1)
                    .saturating_add(overlap.1)
                    .min(ncol),
            ));
        }
    }
    out
}
fn split(i: usize, size: usize, count: usize) -> usize {
    ((i as u128 * size as u128) / count as u128) as usize
}
struct Tile {
    bounds: Bounds,
    dims: Dims,
    n: Vec<i32>,
    region: Vec<u32>,
    reliable: Vec<bool>,
    n_regions: u32,
}
impl Tile {
    fn pixel(&self, r: usize, c: usize) -> usize {
        self.dims.pix(r - self.bounds.0, c - self.bounds.2)
    }
}
fn extract<T: Copy>(data: &[T], dims: Dims, b: Bounds) -> Vec<T> {
    let mut out = Vec::with_capacity((b.1 - b.0) * (b.3 - b.2));
    for r in b.0..b.1 {
        out.extend_from_slice(&data[dims.pix(r, b.2)..dims.pix(r, b.3)]);
    }
    out
}
fn solve_tile(
    input: &UnwrapInput<'_>,
    costs: &GridCosts,
    bounds: Bounds,
    params: &UnwrapParams,
) -> Result<Tile, UnwrapError> {
    trace(&format!("tile {bounds:?}: start"));
    let (r0, r1, c0, c1) = bounds;
    let dims = Dims::new(r1 - r0, c1 - c0);
    let psi = extract(input.psi, input.dims, bounds);
    let mask = input.mask.map(|m| extract(m, input.dims, bounds));
    let w = costs.width();
    let mut table = Vec::with_capacity(dims.n_edges() * w);
    // Use full-image tables so gradient windows do not see artificial borders.
    for r in r0..r1 - 1 {
        let e = input.dims.edge_v(r, c0);
        table.extend_from_slice(&costs.table[e * w..(e + dims.ncol) * w]);
    }
    if dims.ncol > 1 {
        for r in r0..r1 {
            let e = input.dims.edge_h(r, c0);
            table.extend_from_slice(&costs.table[e * w..(e + dims.ncol - 1) * w]);
        }
    }
    let local = GridCosts {
        dims,
        kt: costs.kt,
        table,
    };
    let start = match params.init {
        InitMethod::Mcf => Start::Mcf,
        InitMethod::Mst => Start::Mst,
    };
    let (k, n, _, _) = solve_grid(&psi, &local, start, params)?;
    trace(&format!("tile {bounds:?}: solved"));
    let (mut region, mut n_regions) = jupwrap_conncomp::label_components(
        &local,
        &k,
        mask.as_deref(),
        params.tile_cost_thresh,
        params.min_region_size.max(1),
    );
    let reliable = region.iter().map(|&v| v != 0).collect();
    // Nearest-region fill preserves each uncertain pixel's local relative phase.
    // Masks are barriers; seedless valid components get fallback variables but
    // never supply reliable overlap evidence.
    let mut queue = VecDeque::new();
    for (p, label) in region.iter_mut().enumerate() {
        if mask.as_ref().is_some_and(|m| m[p] == 0) {
            *label = INVALID;
        } else if *label != 0 {
            queue.push_back(p);
        }
    }
    grow_regions(dims, &mut region, &mut queue);
    for p in 0..dims.npix() {
        if region[p] == 0 {
            n_regions = n_regions
                .checked_add(1)
                .filter(|&v| v != INVALID)
                .ok_or_else(|| UnwrapError::Other("too many tile regions".into()))?;
            region[p] = n_regions;
            queue.push_back(p);
            grow_regions(dims, &mut region, &mut queue);
        }
    }
    Ok(Tile {
        bounds,
        dims,
        n,
        region,
        reliable,
        n_regions,
    })
}
fn grow_regions(d: Dims, labels: &mut [u32], queue: &mut VecDeque<usize>) {
    while let Some(p) = queue.pop_front() {
        let (r, c) = (p / d.ncol, p % d.ncol);
        let mut visit = |q| {
            if labels[q] == 0 {
                labels[q] = labels[p];
                queue.push_back(q);
            }
        };
        if r > 0 {
            visit(p - d.ncol);
        }
        if c > 0 {
            visit(p - 1);
        }
        if c + 1 < d.ncol {
            visit(p + 1);
        }
        if r + 1 < d.nrow {
            visit(p + d.ncol);
        }
    }
}

/// Unwrap tiles in a private rayon pool, merge region offsets, and optionally
/// reoptimize the assembled field. Connected components always use the full grid.
pub fn unwrap_tiled(
    input: &UnwrapInput<'_>,
    params: &UnwrapParams,
) -> Result<UnwrapOutput, UnwrapError> {
    if params.ntiles == (1, 1) {
        return jupwrap_solve::unwrap_single(input, params);
    }
    let total_started = Instant::now();
    input.validate()?;
    params.validate()?;
    let d = input.dims;
    let nt = (params.ntiles.0.min(d.nrow), params.ntiles.1.min(d.ncol));
    let bounds = tile_bounds(d.nrow, d.ncol, nt, params.tile_overlap);
    let cores = tile_bounds(d.nrow, d.ncol, nt, (0, 0));
    let started = Instant::now();
    let costs =
        jupwrap_costs::build_costs(d, input.psi, input.corr, input.mask, &params.cost_params());
    let seconds_costs = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(params.nproc)
        .build()
        .map_err(|e| UnwrapError::Other(format!("tile thread pool: {e}")))?;
    // Indexed collection fixes region ordering independently of scheduling.
    let results: Vec<Result<Tile, UnwrapError>> = pool.install(|| {
        bounds
            .par_iter()
            .map(|&b| solve_tile(input, &costs, b, params))
            .collect()
    });
    let mut tiles = results.into_iter().collect::<Result<Vec<_>, _>>()?;
    let mut n_regions = 0u32;
    for tile in &mut tiles {
        let base = n_regions;
        n_regions = n_regions
            .checked_add(tile.n_regions)
            .filter(|&v| v < INVALID)
            .ok_or_else(|| UnwrapError::Other("too many regions".into()))?;
        for label in &mut tile.region {
            if *label != INVALID {
                *label = base + *label - 1;
            }
        }
    }
    let mut n = vec![0; d.npix()];
    let mut owner = vec![INVALID; d.npix()];
    let mut unreliable = vec![false; d.npix()];
    for (tile, &(r0, r1, c0, c1)) in tiles.iter().zip(&cores) {
        for r in r0..r1 {
            for c in c0..c1 {
                let (p, q) = (d.pix(r, c), tile.pixel(r, c));
                n[p] = tile.n[q];
                owner[p] = tile.region[q];
                unreliable[p] = !tile.reliable[q];
            }
        }
    }
    let mut graph = Secondary::new(n_regions as usize);
    // True boundary costs couple regions, including zero-overlap seams.
    for e in 0..d.n_edges() {
        let (p, q) = d.edge_endpoints(e);
        let (a, b) = (owner[p], owner[q]);
        if a != b && a != INVALID && b != INVALID {
            let k = wrap_count(input.psi[q] as f64 - input.psi[p] as f64) + i64::from(n[q])
                - i64::from(n[p]);
            graph.boundary(a as usize, b as usize, &costs, e, k);
        }
    }
    // Aggregate the entire reliable overlap disagreement distribution per
    // region pair as an L1 penalty in cost units, not a modal tile offset.
    for (i, a) in tiles.iter().enumerate() {
        for b in &tiles[i + 1..] {
            if b.bounds.0 >= a.bounds.1 {
                break;
            }
            let (r0, r1) = (a.bounds.0.max(b.bounds.0), a.bounds.1.min(b.bounds.1));
            let (c0, c1) = (a.bounds.2.max(b.bounds.2), a.bounds.3.min(b.bounds.3));
            for r in r0..r1 {
                for c in c0..c1 {
                    let (p, q) = (a.pixel(r, c), b.pixel(r, c));
                    if a.reliable[p] && b.reliable[q] {
                        graph.overlap(
                            a.region[p] as usize,
                            b.region[q] as usize,
                            i64::from(a.n[p]) - i64::from(b.n[q]),
                            params.tile_cost_thresh.max(1),
                        );
                    }
                }
            }
        }
    }
    drop(tiles);
    let offsets = graph.solve();
    trace("secondary graph solved");
    for (p, value) in n.iter_mut().enumerate() {
        if owner[p] != INVALID {
            *value = checked_cycle(i64::from(*value) + offsets[owner[p] as usize])?;
        }
    }
    drop(owner);
    fill_unreliable(input, &costs, &mut n, &unreliable);
    drop(unreliable);
    let anchor = i64::from(n[0]);
    for value in &mut n {
        *value = checked_cycle(i64::from(*value) - anchor)?;
    }
    let seconds_init = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let (k, n, init_cost, final_cost) = if params.single_tile_reoptimize {
        if params.cost == CostMode::Defo {
            // Plateau costs can trap a merged field several cycles from a
            // consistent large-scale branch. Reoptimize its convex surrogate
            // first, just as the single-tile pipeline uses that surrogate for
            // initialization. Both solves start from the assembled pixel field.
            trace("full-image surrogate reoptimization start");
            let surrogate = jupwrap_costs::convex_envelope(&costs);
            secondary::refine_convex(input.psi, &surrogate, &mut n)?;
            trace("full-image surrogate reoptimization done");
        }
        trace("full-image reoptimization start");
        solve_grid(input.psi, &costs, Start::FromN(&n), params)?
    } else {
        let k = jupwrap_grid::k_from_n(d, input.psi, &n);
        let cost = total_cost(&costs, &k);
        (k, n, cost, cost)
    };
    let seconds_improve = started.elapsed().as_secs_f64();
    trace("full-image optimization done");
    let (conncomp, n_components) = jupwrap_conncomp::label_components(
        &costs,
        &k,
        input.mask,
        params.conncomp_cost_thresh,
        min_component_size(params.min_conncomp_frac, d.npix()),
    );
    let n_residues = jupwrap_grid::residues(d, input.psi)
        .iter()
        .filter(|&&v| v != 0)
        .count() as u64;
    let mut unw = unwrapped_from_n(input.psi, &n);
    if let Some(mask) = input.mask {
        for (v, &valid) in unw.iter_mut().zip(mask) {
            if valid == 0 {
                *v = 0.0;
            }
        }
    }
    Ok(UnwrapOutput {
        n,
        unw,
        conncomp,
        stats: UnwrapStats {
            n_residues,
            init_cost,
            final_cost,
            n_components,
            n_tiles: bounds.len() as u32,
            seconds_costs,
            seconds_init,
            seconds_improve,
            seconds_total: total_started.elapsed().as_secs_f64(),
        },
    })
}
fn trace(message: &str) {
    if std::env::var_os("JUPWRAP_TILING_TRACE").is_some() {
        eprintln!("tiling: {message}");
    }
}
fn checked_cycle(n: i64) -> Result<i32, UnwrapError> {
    i32::try_from(n).map_err(|_| UnwrapError::InvalidInput("merged pixel field exceeds i32".into()))
}
fn fill_unreliable(input: &UnwrapInput<'_>, costs: &GridCosts, n: &mut [i32], unreliable: &[bool]) {
    let d = input.dims;
    for _ in 0..8 {
        let mut changed = false;
        for p in 0..d.npix() {
            if !unreliable[p] || input.mask.is_some_and(|m| m[p] == 0) {
                continue;
            }
            let (r, c) = (p / d.ncol, p % d.ncol);
            let mut edges = [(0, 0); 4];
            let mut len = 0;
            if r > 0 {
                edges[len] = (d.edge_v(r - 1, c), 1);
                len += 1;
            }
            if r + 1 < d.nrow {
                edges[len] = (d.edge_v(r, c), -1);
                len += 1;
            }
            if c > 0 {
                edges[len] = (d.edge_h(r, c - 1), 1);
                len += 1;
            }
            if c + 1 < d.ncol {
                edges[len] = (d.edge_h(r, c), -1);
                len += 1;
            }
            let (mut up, mut down) = (0, 0);
            for &(e, sign) in &edges[..len] {
                let (a, b) = d.edge_endpoints(e);
                let k = wrap_count(input.psi[b] as f64 - input.psi[a] as f64) + i64::from(n[b])
                    - i64::from(n[a]);
                if let Ok(k) = i32::try_from(k) {
                    let old = costs.cost(e, k);
                    up += k
                        .checked_add(sign)
                        .map_or(i64::MAX / 8, |v| costs.cost(e, v) - old);
                    down += k
                        .checked_sub(sign)
                        .map_or(i64::MAX / 8, |v| costs.cost(e, v) - old);
                }
            }
            let step = if up < 0 && up <= down {
                1
            } else if down < 0 {
                -1
            } else {
                0
            };
            if step != 0 {
                if let Some(v) = n[p].checked_add(step) {
                    n[p] = v;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
}
