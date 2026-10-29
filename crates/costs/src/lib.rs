//! Statistical cost model and integer convex surrogate (DESIGN.md section 5).

use jupwrap_core::{
    wrapped_gradients, CostMode, CostParams, Dims, GridCosts, COST_CAP, COST_SCALE, KT, RHO_MAX,
    SIGMA2_MAX, TAU,
};

/// Effective coherence after de-biasing (DESIGN.md eq. C1).
pub fn effective_coherence(corr: f32, nlooks: f64) -> f64 {
    let mut r = if corr.is_nan() {
        0.0
    } else {
        (corr as f64).clamp(0.0, 1.0)
    };
    if nlooks > 1.0 {
        r = ((nlooks * r * r - 1.0) / (nlooks - 1.0)).max(0.0).sqrt();
    }
    r.min(RHO_MAX)
}

/// Pixel phase variance from coherence and looks (DESIGN.md eq. C2).
pub fn pixel_sigma2(corr: f32, nlooks: f64) -> f64 {
    let rho = effective_coherence(corr, nlooks);
    if rho > 0.0 {
        let rho2 = rho * rho;
        ((1.0 - rho2) / (2.0 * nlooks * rho2)).min(SIGMA2_MAX)
    } else {
        SIGMA2_MAX
    }
}

fn means(dims: Dims, gradients: &[f64], mask: Option<&[u8]>, window: (usize, usize)) -> Vec<f64> {
    let mut output = vec![0.0; dims.n_edges()];
    for (offset, rows, cols) in [
        (0, dims.nrow.saturating_sub(1), dims.ncol),
        (dims.nv(), dims.nrow, dims.ncol.saturating_sub(1)),
    ] {
        if rows == 0 || cols == 0 {
            continue;
        }
        let stride = cols + 1;
        let size = (rows + 1) * stride;
        // Each direction has its own summed-area tables, with a zero border.
        // Counts distinguish an empty masked window from a zero vector sum.
        let mut sin_sum = vec![0.0; size];
        let mut cos_sum = vec![0.0; size];
        let mut count = vec![0usize; size];
        for r in 0..rows {
            let (mut s, mut c, mut n) = (0.0, 0.0, 0usize);
            for col in 0..cols {
                let edge = offset + r * cols + col;
                let (p, q) = dims.edge_endpoints(edge);
                if mask.is_none_or(|m| m[p] != 0 && m[q] != 0) {
                    let (sn, cs) = gradients[edge].sin_cos();
                    s += sn;
                    c += cs;
                    n += 1;
                }
                let i = (r + 1) * stride + col + 1;
                sin_sum[i] = sin_sum[i - stride] + s;
                cos_sum[i] = cos_sum[i - stride] + c;
                count[i] = count[i - stride] + n;
            }
        }
        for r in 0..rows {
            let r0 = r.saturating_sub(window.0 / 2);
            let r1 = r.saturating_add(window.0 / 2).saturating_add(1).min(rows);
            for col in 0..cols {
                let c0 = col.saturating_sub(window.1 / 2);
                let c1 = col.saturating_add(window.1 / 2).saturating_add(1).min(cols);
                let (a, b, c, d) = (
                    r0 * stride + c0,
                    r0 * stride + c1,
                    r1 * stride + c0,
                    r1 * stride + c1,
                );
                let n = count[d] + count[a] - count[b] - count[c];
                if n > 0 {
                    let s = sin_sum[d] - sin_sum[b] - sin_sum[c] + sin_sum[a];
                    let co = cos_sum[d] - cos_sum[b] - cos_sum[c] + cos_sum[a];
                    output[offset + r * cols + col] = s.atan2(co);
                }
            }
        }
    }
    output
}

/// Window-mean wrapped gradient for every edge (DESIGN.md eq. C3), edge order.
pub fn mean_gradients(
    dims: Dims,
    psi: &[f32],
    mask: Option<&[u8]>,
    window: (usize, usize),
) -> Vec<f64> {
    means(dims, &wrapped_gradients(dims, psi), mask, window)
}

/// Quantized cost tables, `kt = KT` (DESIGN.md eqs. C4-C7).
pub fn build_costs(
    dims: Dims,
    psi: &[f32],
    corr: &[f32],
    mask: Option<&[u8]>,
    params: &CostParams,
) -> GridCosts {
    let gradients = wrapped_gradients(dims, psi);
    let mu = means(dims, &gradients, mask, params.phase_grad_window);
    let variance: Vec<f64> = corr
        .iter()
        .map(|&r| pixel_sigma2(r, params.nlooks))
        .collect();
    let width = (2 * KT + 1) as usize;
    let mut table = vec![0; dims.n_edges() * width];
    let log_uniform = params.p_discontinuity.ln() - (TAU * width as f64).ln();
    let log_core = (-params.p_discontinuity).ln_1p();
    for e in 0..dims.n_edges() {
        let (p, q) = dims.edge_endpoints(e);
        if mask.is_some_and(|m| m[p] == 0 || m[q] == 0) {
            continue;
        }
        let s2 = variance[p] + variance[q] + params.sigma2_signal;
        let log_normalizer = 0.5 * (TAU * s2).ln();
        let mut values = [0.0; (2 * KT + 1) as usize];
        let mut minimum = f64::INFINITY;
        for (i, value) in values.iter_mut().enumerate() {
            let x = gradients[e] + TAU * (i as i32 - KT) as f64 - mu[e];
            let gaussian = x * x / (2.0 * s2);
            *value = match params.mode {
                CostMode::Smooth => gaussian,
                CostMode::Defo => {
                    // Preserve C6's subtraction order and evaluate the mixture
                    // in log space even when the Gaussian density underflows.
                    let a = log_core - gaussian - log_normalizer;
                    let hi = a.max(log_uniform);
                    let lo = a.min(log_uniform);
                    -(hi + (lo - hi).exp().ln_1p())
                }
            };
            minimum = minimum.min(*value);
        }
        for (i, value) in values.iter().enumerate() {
            table[e * width + i] = (COST_SCALE * (value - minimum) + 0.5)
                .floor()
                .min(COST_CAP as f64) as i32;
        }
    }
    GridCosts {
        dims,
        kt: KT,
        table,
    }
}

/// Integer convex surrogate of every table row (DESIGN.md eq. C8).
pub fn convex_envelope(costs: &GridCosts) -> GridCosts {
    let width = costs.width();
    let mut result = costs.clone();
    let mut hull = Vec::with_capacity(width);
    let mut increments = vec![0i64; width.saturating_sub(1)];
    for (input, output) in costs
        .table
        .chunks_exact(width)
        .zip(result.table.chunks_exact_mut(width))
    {
        hull.clear();
        for j in 0..width {
            while hull.len() >= 2 {
                let h: usize = hull[hull.len() - 2];
                let i: usize = hull[hull.len() - 1];
                let cross = (i - h) as i64 * (input[j] as i64 - input[h] as i64)
                    - (input[i] as i64 - input[h] as i64) * (j - h) as i64;
                // Remove collinear points too, as specified by C8.
                if cross > 0 {
                    break;
                }
                hull.pop();
            }
            hull.push(j);
        }
        for pair in hull.windows(2) {
            let (h, j) = (pair[0], pair[1]);
            // Euclidean division floors negative slopes rather than truncating.
            let slope = (input[j] as i64 - input[h] as i64).div_euclid((j - h) as i64);
            increments[h..j].fill(slope);
        }
        // min_by_key returns the first minimum, including ties and plateaus.
        let anchor = input.iter().enumerate().min_by_key(|&(_, v)| v).unwrap().0;
        output[anchor] = input[anchor];
        for i in anchor..width - 1 {
            output[i + 1] = (output[i] as i64 + increments[i]) as i32;
        }
        for i in (0..anchor).rev() {
            output[i] = (output[i + 1] as i64 - increments[i]) as i32;
        }
    }
    result
}
