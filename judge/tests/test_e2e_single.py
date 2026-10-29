"""End-to-end tests for the single-tile pipeline (jupwrap-solve, via jupwrap-cli).

Thresholds (DESIGN.md section 14): smooth mode is a convex problem, so a correct
pipeline reaches the reference optimum: its cost under the reference cost
tables must be <= ref_cost * (1 + 1e-4) + 100 (slack for +-1 table rounding) and
the fraction of correctly unwrapped truth-region pixels >= ref_frac - 0.002.
Defo mode is nonconvex: fraction >= ref_frac - 0.005 (init=mst: 0.008) and
cost <= 1.02 * ref_cost."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import EXPECTED, assert_conncomp_contract, case, record, run_cli, score
from ref import e2e_cases as E
from ref.scoring import cost_of_unw, ref_table

TAU = 2 * np.pi


def _run_scored(tmpd, name, init="mcf", **kw):
    cs = case(name)
    unw, cc, stats, wall, rss = run_cli(tmpd, cs["igram"], cs["corr"], cs["nlooks"], cs["cost"], init, cs["mask"], **kw)
    fc, cong, reg = score(cs, unw)
    d, psi, table = ref_table(cs)
    cost = cost_of_unw(d, psi, table, unw, cs["mask"])
    ex = EXPECTED[name]
    record(f"pipeline_{name}_{init}", frac=fc, ref_frac=ex["ref_frac"], cost=cost, ref_cost=ex["ref_cost"],
           congruence=cong, wall_s=wall, maxrss_bytes=rss, stats=stats)
    assert cong < 1e-3, f"{name}: unwrapped phase not congruent with wrapped phase (max {cong})"
    assert_conncomp_contract(cc, cs["mask"])
    if cs["mask"] is not None:
        assert (unw[cs["mask"] == 0] == 0).all(), "masked pixels must have unw 0"
    return cs, unw, cc, fc, cost, ex, wall


def test_clean_topography_exact(tmpd):
    cs, unw, cc, fc, cost, ex, _ = _run_scored(tmpd, "clean_topo_256")
    assert fc == 1.0, f"residue-free input must be recovered exactly up to 2*pi*k (got {fc})"
    assert cost == ex["ref_cost"]
    assert cc.max() == 1 and (cc == 1).all()


def test_readme_ramp_nlooks1(tmpd):
    phi, igram, corr = E.readme_ramp(512)
    unw, cc, stats, wall, rss = run_cli(tmpd, igram, corr, 1.0)
    off = np.round(np.median((unw - phi) / TAU))
    assert np.abs(unw - phi - TAU * off).max() < 1e-3


@pytest.mark.parametrize("name", ["topo_512_l4", "topo_512_l3_harsh", "masked_topo_384"])
def test_smooth_reaches_reference_optimum(tmpd, name):
    cs, unw, cc, fc, cost, ex, _ = _run_scored(tmpd, name)
    assert cost <= ex["ref_cost"] * (1 + 1e-4) + 100, f"{name}: cost {cost} > reference optimum {ex['ref_cost']}"
    assert fc >= ex["ref_frac"] - 0.002, f"{name}: fraction correct {fc:.5f} < {ex['ref_frac'] - 0.002:.5f}"


def test_smooth_mst_init_reaches_optimum(tmpd):
    cs, unw, cc, fc, cost, ex, _ = _run_scored(tmpd, "topo_512_l4", init="mst")
    assert cost <= ex["ref_cost"] * (1 + 1e-4) + 100, f"init=mst: cost {cost} > optimum {ex['ref_cost']}"
    assert fc >= ex["ref_frac"] - 0.002


@pytest.mark.parametrize("init", ["mcf", "mst"])
def test_defo_mode(tmpd, init):
    cs, unw, cc, fc, cost, ex, _ = _run_scored(tmpd, "defo_512", init=init)
    tol = 0.005 if init == "mcf" else 0.008
    assert fc >= ex["ref_frac"] - tol, f"defo/{init}: fraction {fc:.5f} < {ex['ref_frac'] - tol:.5f}"
    assert cost <= 1.02 * ex["ref_cost"], f"defo/{init}: cost {cost} > 1.02 * {ex['ref_cost']}"


def test_conncomp_coverage_high_coherence(tmpd):
    cs, unw, cc, fc, cost, ex, _ = _run_scored(tmpd, "masked_topo_384")
    reg = score(cs, unw)[2]
    cov = (cc[reg > 0] > 0).mean()
    assert cov >= 0.95, f"only {cov:.3f} of high-coherence truth pixels are in a component"


@pytest.mark.parametrize("shape", [(1, 40), (40, 1), (2, 2), (3, 3)])
def test_degenerate_shapes(tmpd, shape):
    rng = np.random.default_rng(3)
    igram = np.exp(1j * rng.uniform(-np.pi, np.pi, shape)).astype(np.complex64)
    corr = np.full(shape, 0.5, np.float32)
    unw, cc, stats, wall, rss = run_cli(tmpd, igram, corr, 5.0)
    d = unw.astype(np.float64) - np.angle(igram).astype(np.float32)
    assert np.abs(d - TAU * np.round(d / TAU)).max() < 1e-4


def test_runtime_2048(tmpd):
    cs, unw, cc, fc, cost, ex, wall = _run_scored(tmpd, "topo_2048")
    assert cost <= ex["ref_cost"] * (1 + 1e-4) + 100
    assert fc >= ex["ref_frac"] - 0.002
    assert wall <= 600, f"2048x2048 took {wall:.0f} s (limit 600 s)"
