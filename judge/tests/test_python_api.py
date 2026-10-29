"""Tests for the Python package: the snaphu-py compatible `jupwrap.unwrap`.

The wheel is built by the test command (maturin, offline) into
<workspace>/target/wheels; this module extracts the newest wheel into a temp
dir and imports it from there (no install step). The CLI is also built so the
Python result can be compared bit-for-bit with the Rust facade."""
from __future__ import annotations

import importlib
import inspect
import sys
import zipfile

import numpy as np
import pytest

from conftest import ROOT, case, run_cli, score

TAU = 2 * np.pi


@pytest.fixture(scope="module")
def jw(tmp_path_factory):
    wheels = sorted((ROOT / "target" / "wheels").glob("jupwrap-*.whl"), key=lambda p: p.stat().st_mtime)
    assert wheels, "no wheel in target/wheels: build with maturin first"
    dest = tmp_path_factory.mktemp("wheel")
    with zipfile.ZipFile(wheels[-1]) as z:
        z.extractall(dest)
    sys.path.insert(0, str(dest))
    for m in [m for m in sys.modules if m == "jupwrap" or m.startswith("jupwrap.")]:
        del sys.modules[m]
    mod = importlib.import_module("jupwrap")
    assert str(dest) in str(mod.__file__), f"imported jupwrap from {mod.__file__}, not the built wheel"
    return mod


SNAPHU_PY_SIGNATURE = [
    ("igram", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
    ("corr", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
    ("nlooks", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
    ("cost", "POSITIONAL_OR_KEYWORD", "smooth"),
    ("init", "POSITIONAL_OR_KEYWORD", "mcf"),
    ("mask", "KEYWORD_ONLY", None),
    ("min_conncomp_frac", "KEYWORD_ONLY", 0.01),
    ("phase_grad_window", "KEYWORD_ONLY", (7, 7)),
    ("ntiles", "KEYWORD_ONLY", (1, 1)),
    ("tile_overlap", "KEYWORD_ONLY", 0),
    ("nproc", "KEYWORD_ONLY", 1),
    ("tile_cost_thresh", "KEYWORD_ONLY", 500),
    ("min_region_size", "KEYWORD_ONLY", 100),
    ("single_tile_reoptimize", "KEYWORD_ONLY", True),
    ("regrow_conncomps", "KEYWORD_ONLY", True),
    ("scratchdir", "KEYWORD_ONLY", None),
    ("delete_scratch", "KEYWORD_ONLY", True),
    ("unw", "KEYWORD_ONLY", None),
    ("conncomp", "KEYWORD_ONLY", None),
]


def test_signature_matches_snaphu_py(jw):
    sig = inspect.signature(jw.unwrap)
    got = [(p.name, p.kind.name, p.default) for p in sig.parameters.values()]
    assert got == SNAPHU_PY_SIGNATURE


def _ramp(n=512):
    y, x = np.ogrid[-3:3:n * 1j, -3:3:n * 1j]
    phi = np.pi * (x + y)
    return phi, np.exp(1j * phi).astype(np.complex64), np.ones((n, n), np.float32)


def test_readme_example(jw):
    phi, igram, corr = _ramp()
    out = jw.unwrap(igram, corr, nlooks=1.0, cost="smooth", init="mcf")
    assert isinstance(out, tuple) and len(out) == 2
    unw, conncomp = out
    assert isinstance(unw, np.ndarray) and unw.dtype == np.float32 and unw.shape == igram.shape
    assert isinstance(conncomp, np.ndarray) and conncomp.dtype == np.uint32 and conncomp.shape == igram.shape
    off = np.round(np.median((unw - phi) / TAU))
    assert np.abs(unw - phi - TAU * off).max() < 1e-3
    assert (conncomp == 1).all()


def test_matches_cli_bitwise(jw, tmp_path):
    cs = case("topo_512_l4")
    unw, cc = jw.unwrap(cs["igram"], cs["corr"], cs["nlooks"])
    u2, c2, _, _, _ = run_cli(tmp_path, cs["igram"], cs["corr"], cs["nlooks"])
    assert np.array_equal(unw, u2) and np.array_equal(cc, c2)
    fc, cong, _ = score(cs, unw)
    assert cong < 1e-3


def test_mask_bool_and_uint8(jw):
    cs = case("masked_topo_384")
    m8 = cs["mask"].astype(np.uint8)
    a = jw.unwrap(cs["igram"], cs["corr"], cs["nlooks"], mask=m8)
    b = jw.unwrap(cs["igram"], cs["corr"], cs["nlooks"], mask=m8.astype(bool))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert (a[0][m8 == 0] == 0).all() and (a[1][m8 == 0] == 0).all()


def test_nan_inputs_are_zeroed(jw):
    rng = np.random.default_rng(0)
    phi, igram, corr = _ramp(128)
    corr = np.full(igram.shape, 0.8, np.float32)
    ig = igram.copy()
    ig[rng.random(ig.shape) < 0.01] = np.nan
    co = corr.copy()
    co[rng.random(co.shape) < 0.01] = np.nan
    unw, cc = jw.unwrap(ig, co, 4.0)
    assert np.isfinite(unw).all()
    ig0 = np.where(np.isnan(ig), 0, ig).astype(np.complex64)
    co0 = np.where(np.isnan(co), 0, co).astype(np.float32)
    u2, c2 = jw.unwrap(ig0, co0, 4.0)
    assert np.array_equal(unw, u2) and np.array_equal(cc, c2)


def test_corr_is_clipped(jw):
    cs = case("topo_512_l4")
    a = jw.unwrap(cs["igram"], cs["corr"] * 1.5 - 0.1, cs["nlooks"])
    b = jw.unwrap(cs["igram"], np.clip(cs["corr"] * 1.5 - 0.1, 0, 1), cs["nlooks"])
    assert np.array_equal(a[0], b[0])


@pytest.mark.parametrize("bad,exc", [
    (dict(nlooks=0.5), ValueError),
    (dict(cost="topo"), ValueError),
    (dict(cost="bogus"), ValueError),
    (dict(init="bogus"), ValueError),
    (dict(corr="shape"), ValueError),
    (dict(mask="shape"), ValueError),
    (dict(igram="real"), TypeError),
    (dict(igram="3d"), ValueError),
    (dict(ntiles=(0, 1)), ValueError),
    (dict(min_conncomp_frac=1.5), ValueError),
])
def test_invalid_arguments(jw, bad, exc):
    phi, igram, corr = _ramp(64)
    kw = dict(nlooks=4.0)
    args = [igram, corr]
    for k, v in bad.items():
        if k == "corr":
            args[1] = corr[:-1]
        elif k == "mask":
            kw["mask"] = np.ones((3, 3), bool)
        elif k == "igram" and v == "real":
            args[0] = np.angle(igram)
        elif k == "igram" and v == "3d":
            args[0] = igram[None]
            args[1] = corr[None]
        else:
            kw[k] = v
    with pytest.raises(exc):
        jw.unwrap(*args, **kw)


def test_output_datasets(jw):
    cs = case("topo_512_l4")
    ref = jw.unwrap(cs["igram"], cs["corr"], cs["nlooks"])
    u = np.zeros(cs["igram"].shape, np.float32)
    c = np.zeros(cs["igram"].shape, np.uint32)
    out = jw.unwrap(cs["igram"], cs["corr"], cs["nlooks"], unw=u, conncomp=c)
    assert out[0] is u and out[1] is c
    assert np.array_equal(u, ref[0]) and np.array_equal(c, ref[1])
    u64 = np.zeros(cs["igram"].shape, np.float64)
    c64 = np.zeros(cs["igram"].shape, np.int64)
    jw.unwrap(cs["igram"], cs["corr"], cs["nlooks"], unw=u64, conncomp=c64)
    assert np.array_equal(u64.astype(np.float32), ref[0]) and np.array_equal(c64, ref[1].astype(np.int64))


def test_tiling_arguments(jw):
    cs = case("topo_512_l4")
    a = jw.unwrap(cs["igram"], cs["corr"], cs["nlooks"], ntiles=(2, 2), tile_overlap=32, nproc=2)
    b = jw.unwrap(cs["igram"], cs["corr"], cs["nlooks"], ntiles=(2, 2), tile_overlap=(32, 32), nproc=1)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    fc, cong, _ = score(cs, a[0])
    assert cong < 1e-3 and fc > 0.99


def test_dtype_and_layout_flexibility(jw):
    cs = case("topo_512_l4")
    ref = jw.unwrap(cs["igram"], cs["corr"], cs["nlooks"])
    big = np.zeros((512, 1024), np.complex128)
    big[:, ::2] = cs["igram"]
    view = big[:, ::2]
    assert not view.flags.c_contiguous
    out = jw.unwrap(view, cs["corr"].astype(np.float64), int(cs["nlooks"]))
    assert np.array_equal(out[0], ref[0]) and np.array_equal(out[1], ref[1])


def test_scratch_and_misc_arguments_accepted(jw, tmp_path):
    phi, igram, corr = _ramp(64)
    unw, cc = jw.unwrap(igram, corr, 2.0, scratchdir=tmp_path / "scratch", delete_scratch=False,
                        regrow_conncomps=False, tile_cost_thresh=300, min_region_size=50, phase_grad_window=(5, 5))
    off = np.round(np.median((unw - phi) / TAU))
    assert np.abs(unw - phi - TAU * off).max() < 1e-3


def test_min_conncomp_frac_and_modes(jw):
    cs = case("topo_512_l3_harsh")
    few = jw.unwrap(cs["igram"], cs["corr"], cs["nlooks"], min_conncomp_frac=0.2)[1]
    many = jw.unwrap(cs["igram"], cs["corr"], cs["nlooks"], min_conncomp_frac=0.0)[1]
    assert few.max() <= many.max()
    unw, cc = jw.unwrap(cs["igram"], cs["corr"], cs["nlooks"], cost="defo", init="mst")
    assert score(cs, unw)[1] < 1e-3
