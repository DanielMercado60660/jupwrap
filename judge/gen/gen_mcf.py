"""Generate goldens for jupwrap-mcf (reference = OR-tools + networkx optimum).

Run from judge/:  .venv/bin/python -m gen.gen_mcf
Writes golden/mcf/case_XX/{tail,head,lower,upper,supply,cost_offset,cost_values}.npy
and meta.json {n_nodes, status, opt_cost, kind}; and golden/mcf/recipes.json for
large grid instances regenerated in Rust from splitmix64 (judge/ref/recipe.py).
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

from ref.conventions import Dims, residues, supply, table_cost, wrap
from ref.costs import build_costs
from ref.flows import mcf_networkx, mcf_ortools
from ref.recipe import quadgrid

OUT = Path(__file__).resolve().parent.parent / "golden" / "mcf"


def convex_table(rng, lo, hi, kind="random"):
    n = hi - lo + 1
    if kind == "linear":
        c = int(rng.integers(-50, 51))
        x = np.arange(lo, hi + 1)
        return (c * x + int(rng.integers(0, 100))).tolist()
    inc = np.sort(rng.integers(-60, 80, size=n - 1))
    if kind == "flatmin":  # many ties
        inc = np.sort(np.concatenate([np.zeros((n - 1) // 2, np.int64), rng.integers(-5, 6, size=n - 1 - (n - 1) // 2)]))
    v = int(rng.integers(0, 50)) + np.concatenate([[0], np.cumsum(inc)])
    return v.astype(np.int64).tolist()


def random_instance(rng, n, m, kind="random", feasible=True, zero_supply=False, selfloops=0, parallel=0):
    tail = rng.integers(0, n, size=m)
    head = rng.integers(0, n, size=m)
    # ensure weak connectivity with a random spanning path
    perm = rng.permutation(n)
    tail[: n - 1] = perm[:-1]
    head[: n - 1] = perm[1:]
    flip = rng.random(n - 1) < 0.5
    tail[: n - 1][flip], head[: n - 1][flip] = head[: n - 1][flip].copy(), tail[: n - 1][flip].copy()
    tail = list(tail); head = list(head)
    for _ in range(selfloops):
        v = int(rng.integers(0, n)); tail.append(v); head.append(v)
    for _ in range(parallel):
        a = int(rng.integers(0, len(tail))); tail.append(tail[a]); head.append(head[a])
    m = len(tail)
    lower = rng.integers(-5, 3, size=m)
    upper = lower + rng.integers(0, 9, size=m)
    if zero_supply:  # zero flow feasible; the optimum is a nonzero circulation
        lower = np.minimum(lower, 0)
        upper = np.maximum(upper, 1)
    tables = [convex_table(rng, int(lower[a]), int(upper[a]), kind) for a in range(m)]
    if zero_supply:
        sup = np.zeros(n, np.int64)
    else:
        x = np.array([rng.integers(lower[a], upper[a] + 1) for a in range(m)])
        sup = np.zeros(n, np.int64)
        np.add.at(sup, np.array(tail), x)
        np.add.at(sup, np.array(head), -x)
        if not feasible:
            # push more supply than the total capacity out of one node
            out_cap = sum(int(upper[a]) for a in range(m) if tail[a] == 0 and head[a] != 0) - \
                sum(int(lower[a]) for a in range(m) if head[a] == 0 and tail[a] != 0)
            extra = out_cap - sup[0] + 1 + int(rng.integers(0, 5))
            sup[0] += extra
            sup[1 % n] -= extra
    return n, np.array(tail), np.array(head), lower, upper, sup, tables


def write_case(idx, kind, n, tail, head, lower, upper, sup, tables, check_nx=True):
    st, cost, _ = mcf_ortools(n, tail, head, lower, upper, sup, tables)
    if check_nx:
        st2, cost2 = mcf_networkx(n, tail, head, lower, upper, sup, tables)
        assert st == st2 and cost == cost2, (idx, st, cost, st2, cost2)
    d = OUT / f"case_{idx:02d}"
    d.mkdir(parents=True, exist_ok=True)
    lens = np.asarray(upper) - np.asarray(lower) + 1
    np.save(d / "tail.npy", np.asarray(tail, np.int32))
    np.save(d / "head.npy", np.asarray(head, np.int32))
    np.save(d / "lower.npy", np.asarray(lower, np.int32))
    np.save(d / "upper.npy", np.asarray(upper, np.int32))
    np.save(d / "supply.npy", np.asarray(sup, np.int64))
    np.save(d / "cost_offset.npy", np.concatenate([[0], np.cumsum(lens)]).astype(np.int64))
    flat = np.concatenate([np.asarray(t, np.int64) for t in tables]) if len(tables) else np.zeros(0, np.int64)
    np.save(d / "cost_values.npy", flat.astype(np.int64))
    meta = dict(n_nodes=int(n), n_arcs=int(len(tail)), status=st, opt_cost=cost, kind=kind)
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    print(idx, kind, n, len(tail), st, cost)


def grid_case(rng, R, C, B, mode="smooth"):
    """Grid dual from a synthetic noisy interferogram with statistical costs."""
    d = Dims(R, C)
    yy, xx = np.mgrid[0:R, 0:C]
    phi = 0.02 * (xx - C / 2) ** 2 / 4 + 0.6 * yy + 3 * np.sin(xx / 7.0)
    psi = wrap(phi + rng.normal(0, 0.9, (R, C))).astype(np.float32)
    corr = rng.uniform(0.2, 0.9, R * C).astype(np.float32)
    table = build_costs(d, psi, corr, None, mode, 4)
    res = residues(d, psi)
    f, t = d.dual_arcs()
    E = d.n_edges
    ks = np.arange(-B, B + 1)
    tabs = table_cost(np.repeat(table, len(ks), 0), np.tile(ks, E)).reshape(E, -1)
    return d.n_dual, f, t, np.full(E, -B), np.full(E, B), supply(d, res), tabs


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    rng = np.random.default_rng(20261028)
    idx = 0
    # (a) random convex
    for n, m in [(2, 3), (5, 12), (10, 30), (20, 60), (50, 150), (100, 400), (200, 900), (400, 1500)]:
        write_case(idx, "random", *random_instance(rng, n, m)); idx += 1
    # (b) linear costs
    for n, m in [(8, 20), (60, 200), (300, 1200)]:
        write_case(idx, "linear", *random_instance(rng, n, m, kind="linear")); idx += 1
    # (c) zero supplies, optimum is a circulation
    for n, m in [(6, 15), (40, 160), (150, 600)]:
        write_case(idx, "circulation", *random_instance(rng, n, m, zero_supply=True)); idx += 1
    # (d) self-loops, parallel arcs, ties
    for n, m in [(7, 14), (50, 120)]:
        write_case(idx, "selfloop_parallel", *random_instance(rng, n, m, selfloops=5, parallel=10)); idx += 1
    write_case(idx, "ties", *random_instance(rng, 80, 300, kind="flatmin")); idx += 1
    # (e) infeasible
    for n, m in [(5, 8), (30, 70), (100, 250)]:
        write_case(idx, "infeasible", *random_instance(rng, n, m, feasible=False)); idx += 1
    # (f) grid duals with statistical costs (realistic structure)
    for R, C, B in [(2, 2, 3), (1, 6, 2), (6, 1, 2), (16, 20, 4), (64, 64, 4), (120, 100, 5)]:
        write_case(idx, "grid_stat", *grid_case(rng, R, C, B)); idx += 1
    for R, C, B in [(256, 256, 4)]:
        write_case(idx, "grid_stat", *grid_case(rng, R, C, B), check_nx=False); idx += 1
    # large recipes (regenerated in Rust): optimum from OR-tools only
    recipes = []
    for R, C, seed, B, pm in [(128, 128, 11, 3, 100), (512, 512, 12, 3, 50),
                              (1024, 1024, 13, 3, 30), (300, 2000, 14, 4, 150)]:
        d, res, table = quadgrid(R, C, seed, B, pm)
        f, t = d.dual_arcs()
        E = d.n_edges
        t0 = time.time()
        st, cost, _ = mcf_ortools(d.n_dual, f, t, np.full(E, -B), np.full(E, B), supply(d, res), table)
        dt = time.time() - t0
        recipes.append(dict(nrow=R, ncol=C, seed=seed, B=B, resid_permille=pm, status=st, opt_cost=cost,
                            ortools_seconds=round(dt, 2), time_limit_s=round(30.0 + 5.0 * dt, 1), n_residues=int(np.count_nonzero(res))))
        print("recipe", R, C, st, cost, round(dt, 2), "s")
    (OUT / "recipes.json").write_text(json.dumps(recipes, indent=1))
    (OUT / "summary.json").write_text(json.dumps(dict(n_cases=idx), indent=1))


if __name__ == "__main__":
    main()
