//! jupwrap-grid: residues, grid dual network, integration.
//! Conventions: jupwrap-core docs and DESIGN.md section 3.

use jupwrap_core::{wrap_counts, Dims, Network, UnwrapError, K_BIG};

/// Residue of every loop, in loop order: `-(sum of signed wrap counts)` around
/// the loop, equal to `round(curl / TAU)` with
/// `curl = Dh(i,j) + Dv(i,j+1) - Dh(i+1,j) - Dv(i,j)`. Values are -1, 0 or +1.
/// Length `dims.n_loops()`.
/// Computed exactly from core integer wrap counts, without floating-point curls.
/// A one-row or one-column image has no residues.
pub fn residues(dims: Dims, psi: &[f32]) -> Vec<i8> {
    let counts = wrap_counts(dims, psi);
    let mut result = Vec::with_capacity(dims.n_loops());
    for r in 0..dims.nrow.saturating_sub(1) {
        for c in 0..dims.ncol.saturating_sub(1) {
            let curl = i64::from(counts[dims.edge_h(r, c)])
                + i64::from(counts[dims.edge_v(r, c + 1)])
                - i64::from(counts[dims.edge_h(r + 1, c)])
                - i64::from(counts[dims.edge_v(r, c)]);
            result.push((-curl) as i8);
        }
    }
    result
}

/// Dual network of the grid: `n_nodes = dims.n_dual_nodes()`, one arc per edge
/// in edge order with `(tail, head) = dims.edge_dual(e)`, bounds
/// `[-K_BIG, K_BIG]`, supply `-residue` on loops and `sum(residues)` on ground.
/// With no loops, the network has only ground and every arc is a self-loop.
pub fn dual_network(dims: Dims, residues: &[i8]) -> Network {
    assert_eq!(residues.len(), dims.n_loops(), "residue length");
    let mut tail = Vec::with_capacity(dims.n_edges());
    let mut head = Vec::with_capacity(dims.n_edges());
    for e in 0..dims.n_edges() {
        let (a, b) = dims.edge_dual(e);
        tail.push(a as u32);
        head.push(b as u32);
    }
    let mut supply: Vec<i64> = residues.iter().map(|&r| -i64::from(r)).collect();
    supply.push(residues.iter().map(|&r| i64::from(r)).sum());
    Network {
        n_nodes: dims.n_dual_nodes(),
        tail,
        head,
        lower: vec![-K_BIG; dims.n_edges()],
        upper: vec![K_BIG; dims.n_edges()],
        supply,
    }
}

/// Integrate a cycle field `k` (one entry per edge) into the pixel field `n`
/// with `n[0] = 0` and `n_q - n_p = k_e - c_e` for every edge.
/// Returns `Err(UnwrapError::NotCurlFree)` if `k` is not consistent
/// (some loop has `k`-curl != -residue), `Err(InvalidInput)` on bad lengths.
/// Also returns `InvalidInput` if the anchored pixel field cannot fit in i32.
pub fn integrate(dims: Dims, psi: &[f32], k: &[i32]) -> Result<Vec<i32>, UnwrapError> {
    if psi.len() != dims.npix() || k.len() != dims.n_edges() {
        return Err(UnwrapError::InvalidInput(
            "phase or cycle field length does not match dimensions".into(),
        ));
    }
    let counts = wrap_counts(dims, psi);
    // At loop (i,j), divergence(c) is c_h(i,j) + c_v(i,j+1)
    // - c_h(i+1,j) - c_v(i,j) = -residue(i,j). At ground it is
    // sum(residues), since every arc contributes zero total divergence.
    // Thus divergence(k - c) == 0 checks k against every prescribed supply,
    // including ground, before integrating along the spanning tree.
    let mut divergence = vec![0i64; dims.n_dual_nodes()];
    for e in 0..dims.n_edges() {
        let (a, b) = dims.edge_dual(e);
        let flow = i64::from(k[e]) - i64::from(counts[e]);
        divergence[a] += flow;
        divergence[b] -= flow;
    }
    if divergence.iter().any(|&d| d != 0) {
        return Err(UnwrapError::NotCurlFree);
    }
    drop(divergence);
    let mut n = vec![0i32; dims.npix()];
    for r in 0..dims.nrow {
        for c in 0..dims.ncol {
            if r == 0 && c == 0 {
                continue;
            }
            let (p, e) = if c == 0 {
                (dims.pix(r - 1, c), dims.edge_v(r - 1, c))
            } else {
                (dims.pix(r, c - 1), dims.edge_h(r, c - 1))
            };
            let value = i64::from(n[p]) + i64::from(k[e]) - i64::from(counts[e]);
            n[dims.pix(r, c)] = i32::try_from(value).map_err(|_| {
                UnwrapError::InvalidInput("integrated cycle field exceeds i32".into())
            })?;
        }
    }
    Ok(n)
}

/// Cycle field from a pixel field: `k_e = c_e + n_q - n_p`.
/// Panics on incorrect input lengths or if a cycle value cannot fit in i32.
pub fn k_from_n(dims: Dims, psi: &[f32], n: &[i32]) -> Vec<i32> {
    assert_eq!(n.len(), dims.npix(), "pixel field length");
    let counts = wrap_counts(dims, psi);
    counts
        .into_iter()
        .enumerate()
        .map(|(e, count)| {
            let (p, q) = dims.edge_endpoints(e);
            i32::try_from(i64::from(count) + i64::from(n[q]) - i64::from(n[p]))
                .expect("cycle field exceeds i32")
        })
        .collect()
}
