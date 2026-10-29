"""Clean-room statistical-cost network-flow phase unwrapping."""

import operator
import numpy as np
from . import _jupwrap

__all__ = ["unwrap"]


def _pair(value, name, *, scalar=False, minimum=0):
    if scalar:
        try:
            value = (operator.index(value),) * 2
        except TypeError:
            pass
    try:
        a, b = value
        a, b = operator.index(a), operator.index(b)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a pair of integers") from None
    if a < minimum or b < minimum:
        raise ValueError(f"{name} values must be >= {minimum}")
    return a, b


def unwrap(igram, corr, nlooks, cost="smooth", init="mcf", *, mask=None,
           min_conncomp_frac=0.01, phase_grad_window=(7, 7), ntiles=(1, 1),
           tile_overlap=0, nproc=1, tile_cost_thresh=500, min_region_size=100,
           single_tile_reoptimize=True, regrow_conncomps=True, scratchdir=None,
           delete_scratch=True, unw=None, conncomp=None):
    """Unwrap a complex interferogram; supplied outputs are filled in place.

    Scratch arguments are accepted for compatibility; no scratch files are used.
    """
    igram = np.asarray(igram)
    if igram.ndim != 2:
        raise ValueError("igram must be two-dimensional")
    if not np.issubdtype(igram.dtype, np.complexfloating):
        raise TypeError("igram must have a complex dtype")
    corr = np.asarray(corr)
    if corr.shape != igram.shape:
        raise ValueError("corr must have the same shape as igram")
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != igram.shape:
            raise ValueError("mask must have the same shape as igram")
        if mask.dtype != np.bool_ and mask.dtype != np.uint8:
            raise TypeError("mask must have dtype bool or uint8")
        mask = np.ascontiguousarray(mask, dtype=np.uint8).tobytes()
    if cost not in ("smooth", "defo"):
        raise ValueError("cost must be 'smooth' or 'defo'")
    if init not in ("mcf", "mst"):
        raise ValueError("init must be 'mcf' or 'mst'")
    phase_grad_window = _pair(phase_grad_window, "phase_grad_window", minimum=1)
    ntiles = _pair(ntiles, "ntiles", minimum=1)
    tile_overlap = _pair(tile_overlap, "tile_overlap", scalar=True)
    for output, name in ((unw, "unw"), (conncomp, "conncomp")):
        if output is not None and output.shape != igram.shape:
            raise ValueError(f"{name} must have the same shape as igram")
    ubytes, cbytes = _jupwrap.solve(
        np.ascontiguousarray(igram, dtype="<c8").tobytes(),
        np.ascontiguousarray(corr, dtype="<f4").tobytes(), mask,
        igram.shape, nlooks, cost, init, min_conncomp_frac, phase_grad_window,
        ntiles, tile_overlap, nproc, tile_cost_thresh, min_region_size,
        single_tile_reoptimize, regrow_conncomps,
    )
    u = np.frombuffer(ubytes, dtype="<f4").reshape(igram.shape).copy()
    c = np.frombuffer(cbytes, dtype="<u4").reshape(igram.shape).copy()
    if unw is not None:
        unw[...] = u
        u = unw
    if conncomp is not None:
        conncomp[...] = c
        c = conncomp
    return u, c
