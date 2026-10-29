//! Python bindings for the complex-input facade.
use jupwrap::{CostMode, Dims, InitMethod, UnwrapError, UnwrapParams};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

fn floats(bytes: &[u8]) -> PyResult<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return Err(PyValueError::new_err("float buffer length must be divisible by 4"));
    }
    Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn solve<'py>(
    py: Python<'py>, igram: &[u8], corr: &[u8], mask: Option<&[u8]>,
    shape: (usize, usize), nlooks: f64, cost: &str, init: &str,
    min_conncomp_frac: f64, phase_grad_window: (usize, usize), ntiles: (usize, usize),
    tile_overlap: (usize, usize), nproc: usize, tile_cost_thresh: i64,
    min_region_size: usize, single_tile_reoptimize: bool, regrow_conncomps: bool,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
    let igram = floats(igram)?;
    let corr = floats(corr)?;
    let mask = mask.map(<[u8]>::to_vec);
    let cost = match cost {
        "smooth" => CostMode::Smooth,
        "defo" => CostMode::Defo,
        _ => return Err(PyValueError::new_err("cost must be 'smooth' or 'defo'")),
    };
    let init = match init {
        "mcf" => InitMethod::Mcf,
        "mst" => InitMethod::Mst,
        _ => return Err(PyValueError::new_err("init must be 'mcf' or 'mst'")),
    };
    let mut params = UnwrapParams::new(nlooks, cost, init);
    params.min_conncomp_frac = min_conncomp_frac;
    params.phase_grad_window = phase_grad_window;
    params.ntiles = ntiles;
    params.tile_overlap = tile_overlap;
    params.nproc = nproc;
    params.tile_cost_thresh = tile_cost_thresh;
    params.min_region_size = min_region_size;
    params.single_tile_reoptimize = single_tile_reoptimize;
    params.regrow_conncomps = regrow_conncomps;
    let result = py.detach(|| jupwrap::unwrap_complex(
        Dims::new(shape.0, shape.1), &igram, &corr, mask.as_deref(), &params,
    ));
    let output = result.map_err(|err| match err {
        UnwrapError::InvalidInput(message) => PyValueError::new_err(message),
        other => PyRuntimeError::new_err(other.to_string()),
    })?;
    let unw: Vec<u8> = output.unw.iter().flat_map(|v| v.to_le_bytes()).collect();
    let cc: Vec<u8> = output.conncomp.iter().flat_map(|v| v.to_le_bytes()).collect();
    Ok((PyBytes::new(py, &unw), PyBytes::new(py, &cc)))
}

#[pymodule]
fn _jupwrap(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    Ok(())
}
