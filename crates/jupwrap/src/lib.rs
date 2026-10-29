//! jupwrap: facade. Dispatches to the single-tile
//! pipeline or the tiled unwrapper. DESIGN.md section 12.

pub use jupwrap_core::*;

/// Replace NaN coherence by 0 and clip to [0, 1].
pub fn sanitize_corr(corr: &[f32]) -> Vec<f32> {
    corr.iter()
        .map(|&c| if c.is_nan() { 0.0 } else { c.clamp(0.0, 1.0) })
        .collect()
}

/// Wrapped phase of an interleaved complex64 interferogram `[re0, im0, re1, im1, ...]`.
/// NaN samples give phase 0.
pub fn phase_from_interleaved(igram: &[f32]) -> Vec<f32> {
    igram.chunks_exact(2).map(|z| phase_of(z[0], z[1])).collect()
}

/// Unwrap a wrapped-phase image. `ntiles == (1, 1)` runs the single-tile
/// pipeline; anything else runs the tiled unwrapper.
pub fn unwrap(input: &UnwrapInput<'_>, params: &UnwrapParams) -> Result<UnwrapOutput, UnwrapError> {
    input.validate()?;
    params.validate()?;
    let corr = sanitize_corr(input.corr);
    let inp = UnwrapInput { dims: input.dims, psi: input.psi, corr: &corr, mask: input.mask };
    let t0 = std::time::Instant::now();
    let mut out = if params.ntiles == (1, 1) {
        jupwrap_solve::unwrap_single(&inp, params)?
    } else {
        jupwrap_tiling::unwrap_tiled(&inp, params)?
    };
    out.stats.seconds_total = t0.elapsed().as_secs_f64();
    Ok(out)
}

/// Unwrap an interleaved complex64 interferogram.
pub fn unwrap_complex(
    dims: Dims,
    igram: &[f32],
    corr: &[f32],
    mask: Option<&[u8]>,
    params: &UnwrapParams,
) -> Result<UnwrapOutput, UnwrapError> {
    if igram.len() != 2 * dims.npix() {
        return Err(UnwrapError::InvalidInput("igram length != 2*nrow*ncol".into()));
    }
    let psi = phase_from_interleaved(igram);
    unwrap(&UnwrapInput { dims, psi: &psi, corr, mask }, params)
}
