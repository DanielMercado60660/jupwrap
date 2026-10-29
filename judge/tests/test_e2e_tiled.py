"""End-to-end tests for tiled unwrapping (jupwrap-tiling, via jupwrap-cli).

With single_tile_reoptimize (default True) the final full-image improvement
under convex smooth costs reaches the global optimum, so the smooth tiled result
must match the single-tile reference optimum (cost <= ref*(1+1e-4)+100).
Without reoptimization, fraction >= ref_frac - 0.01. Defo: fraction >=
ref_frac - 0.01 and cost <= 1.02*ref (defo accuracy does not track cost exactly). The 33000-row strip exercises sizes beyond
SNAPHU's 32k-per-side limit; tiling with 8 threads must take at most 0.7x the
single-tile wall time (or 120 s, whichever is larger), so tiling cannot be a
single-tile solve in disguise."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import EXPECTED, assert_conncomp_contract, case, record, run_cli, score
from ref.scoring import cost_of_unw, ref_table


def _tiled(tmpd, name, **kw):
    cs = case(name)
    unw, cc, stats, wall, rss = run_cli(tmpd, cs["igram"], cs["corr"], cs["nlooks"], cs["cost"], "mcf", cs["mask"], **kw)
    fc, cong, reg = score(cs, unw)
    d, psi, table = ref_table(cs)
    cost = cost_of_unw(d, psi, table, unw, cs["mask"])
    ex = EXPECTED.get(name, {})
    record(f"tiling_{name}_{'x'.join(map(str, kw.get('ntiles', (1, 1))))}_{kw.get('single_tile_reoptimize', True)}",
           frac=fc, ref_frac=ex.get("ref_frac"), cost=cost, ref_cost=ex.get("ref_cost"), congruence=cong,
           wall_s=wall, maxrss_bytes=rss, stats=stats)
    assert cong < 1e-3, f"{name}: not congruent ({cong})"
    assert_conncomp_contract(cc, cs["mask"])
    if cs["mask"] is not None:
        assert (unw[cs["mask"] == 0] == 0).all()
    return cs, unw, cc, fc, cost, ex, wall


def test_topo_2x2_reoptimized_reaches_optimum(tmpd):
    cs, unw, cc, fc, cost, ex, _ = _tiled(tmpd, "topo_1024_l4", ntiles=(2, 2), tile_overlap=(64, 64), nproc=4)
    assert cost <= ex["ref_cost"] * (1 + 1e-4) + 100, f"cost {cost} > optimum {ex['ref_cost']}"
    assert fc >= ex["ref_frac"] - 0.002


def test_topo_3x3_no_reoptimize(tmpd):
    cs, unw, cc, fc, cost, ex, _ = _tiled(tmpd, "topo_1024_l4", ntiles=(3, 3), tile_overlap=(128, 128), nproc=4,
                                          single_tile_reoptimize=False)
    assert fc >= ex["ref_frac"] - 0.01, f"fraction {fc:.5f} < {ex['ref_frac'] - 0.01:.5f}"


def test_defo_2x3(tmpd):
    cs, unw, cc, fc, cost, ex, _ = _tiled(tmpd, "defo_1024", ntiles=(2, 3), tile_overlap=(96, 96), nproc=6)
    assert fc >= ex["ref_frac"] - 0.01, f"fraction {fc:.5f} < {ex['ref_frac'] - 0.01:.5f}"
    assert cost <= 1.02 * ex["ref_cost"]


def test_masked_2x2(tmpd):
    cs, unw, cc, fc, cost, ex, _ = _tiled(tmpd, "masked_topo_768", ntiles=(2, 2), tile_overlap=(64, 64), nproc=4)
    assert fc >= ex["ref_frac"] - 0.002
    assert cost <= ex["ref_cost"] * (1 + 1e-4) + 100


def test_deterministic_across_nproc(tmpd):
    cs = case("topo_512_l4")
    a = run_cli(tmpd / "a", cs["igram"], cs["corr"], cs["nlooks"], "smooth", "mcf", None, ntiles=(2, 2),
                tile_overlap=(32, 32), nproc=1, single_tile_reoptimize=False)
    b = run_cli(tmpd / "b", cs["igram"], cs["corr"], cs["nlooks"], "smooth", "mcf", None, ntiles=(2, 2),
                tile_overlap=(32, 32), nproc=4, single_tile_reoptimize=False)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]), "output depends on nproc"


def test_strip_beyond_32k_rows(tmpd):
    cs = case("strip_33000x512")
    single = run_cli(tmpd / "s", cs["igram"], cs["corr"], cs["nlooks"], "smooth", "mcf", None)
    tiled = run_cli(tmpd / "t", cs["igram"], cs["corr"], cs["nlooks"], "smooth", "mcf", None, ntiles=(12, 1),
                    tile_overlap=(128, 0), nproc=8)
    fs, cs_, _ = score(cs, single[0])
    ft, ct, _ = score(cs, tiled[0])
    record("tiling_strip_33000x512", frac_single=fs, frac_tiled=ft, wall_single=single[3], wall_tiled=tiled[3],
           maxrss_bytes=tiled[4], stats_single=single[2], stats_tiled=tiled[2])
    assert cs_ < 1e-3 and ct < 1e-3
    assert ft >= 0.995 and fs >= 0.995, f"fractions single {fs:.5f} tiled {ft:.5f}"
    assert tiled[3] <= max(0.7 * single[3], 120.0), f"tiled {tiled[3]:.0f} s vs single {single[3]:.0f} s"
