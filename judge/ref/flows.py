"""Reference network algorithms: convex min-cost flow via OR-tools / networkx
(unit-arc expansion of convex piecewise-linear costs), MST initialization,
connected components, and a simple nonconvex local search.

General textbook algorithms only (Ahuja, Magnanti and Orlin, "Network Flows",
1993: ch. 14 convex cost flows by arc expansion; Kruskal MST).
"""
from __future__ import annotations

import numpy as np

from .conventions import Dims, divergence, supply, table_cost, KT


# ---------------------------------------------------------------------------
# Convex min-cost flow (exact) by unit-arc expansion
# ---------------------------------------------------------------------------

def _expand(n_nodes, tail, head, lower, upper, sup, tables):
    """tables[a] = cost(a, x) for x = lower[a] .. upper[a] (convex). Either a list
    of sequences or a 2-D array (all arcs the same length).
    Returns (base_cost, unit arcs T, H, C, arc_id, shifted supplies)."""
    tail = np.asarray(tail, np.int64)
    head = np.asarray(head, np.int64)
    lower = np.asarray(lower, np.int64)
    upper = np.asarray(upper, np.int64)
    s = np.asarray(sup, np.int64).copy()
    lens = upper - lower + 1
    if isinstance(tables, np.ndarray) and tables.ndim == 2:
        flat = tables.astype(np.int64).ravel()
        assert (lens == tables.shape[1]).all()
    else:
        flat = np.concatenate([np.asarray(tb, np.int64) for tb in tables]) if len(tables) else np.zeros(0, np.int64)
        assert all(len(tb) == l for tb, l in zip(tables, lens))
    off = np.concatenate([[0], np.cumsum(lens)])
    base = int(flat[off[:-1]].sum()) if len(lens) else 0
    inc = np.diff(flat)
    # valid increments: inside one arc's table
    arc_of = np.repeat(np.arange(len(lens)), lens)
    keep = arc_of[:-1] == arc_of[1:]
    inc_arc = arc_of[:-1][keep]
    inc = inc[keep]
    # convexity check per arc
    same = inc_arc[:-1] == inc_arc[1:]
    assert (np.diff(inc)[same] >= 0).all(), "non-convex arc table"
    np.add.at(s, tail, -lower * (tail != head))
    np.add.at(s, head, lower * (tail != head))
    return base, tail[inc_arc], head[inc_arc], inc, inc_arc, s


def mcf_ortools(n_nodes, tail, head, lower, upper, sup, tables, return_flow=False):
    """Exact optimum of a convex separable integer MCF. Returns (status, cost[, flow])."""
    from ortools.graph.python import min_cost_flow
    base, T, H, Cst, A, s = _expand(n_nodes, tail, head, lower, upper, sup, tables)
    # self-loop unit arcs: take every negative-cost unit (they never affect conservation)
    selfm = T == H
    self_gain = int(Cst[selfm & (Cst < 0)].sum())
    keep = ~selfm
    smcf = min_cost_flow.SimpleMinCostFlow()
    if keep.any():
        smcf.add_arcs_with_capacity_and_unit_cost(T[keep], H[keep], np.ones(keep.sum(), np.int64), Cst[keep])
    for v in range(n_nodes):
        smcf.set_node_supply(v, int(s[v]))
    if not keep.any():
        ok = (s == 0).all()
        if not ok:
            return "infeasible", None, None
        status = "optimal"
        cost = base + self_gain
        flow = np.asarray(lower, np.int64).copy()
    else:
        st = smcf.solve()
        if st != smcf.OPTIMAL:
            return ("infeasible" if st == smcf.INFEASIBLE else f"status{st}"), None, None
        status = "optimal"
        cost = base + self_gain + int(smcf.optimal_cost())
        flow = None
        if return_flow:
            fl = np.array([smcf.flow(i) for i in range(smcf.num_arcs())], np.int64)
            flow = np.asarray(lower, np.int64).copy()
            np.add.at(flow, A[keep], fl)
    if return_flow:
        selfarcs = np.nonzero(selfm & (Cst < 0))[0]
        np.add.at(flow, A[selfarcs], 1)
        return status, cost, flow
    return status, cost, None


def mcf_networkx(n_nodes, tail, head, lower, upper, sup, tables):
    """Independent check with networkx.network_simplex (demand = in - out)."""
    import networkx as nx
    base, T, H, Cst, A, s = _expand(n_nodes, tail, head, lower, upper, sup, tables)
    G = nx.MultiDiGraph()
    for v in range(n_nodes):
        G.add_node(v, demand=-int(s[v]))
    self_gain = 0
    for t, h, c in zip(T, H, Cst):
        if t == h:
            if c < 0:
                self_gain += int(c)
            continue
        G.add_edge(int(t), int(h), capacity=1, weight=int(c))
    try:
        cost, _ = nx.network_simplex(G)
    except nx.NetworkXUnfeasible:
        return "infeasible", None
    return "optimal", base + self_gain + int(cost)


def grid_mcf(dims: Dims, table, res, B=KT):
    """Exact convex optimum on the grid dual with k restricted to [-B, B]."""
    f, t = dims.dual_arcs()
    E = dims.n_edges
    table = np.asarray(table, np.int64)
    ks = np.arange(-B, B + 1)
    tabs = table_cost(np.repeat(table, len(ks), axis=0), np.tile(ks, E)).reshape(E, len(ks))
    st, cost, flow = mcf_ortools(dims.n_dual, f, t, np.full(E, -B), np.full(E, B), supply(dims, res), tabs, return_flow=True)
    assert st == "optimal", st
    return flow, cost


# ---------------------------------------------------------------------------
# MST initialization (DESIGN.md section 8)
# ---------------------------------------------------------------------------

def table_argmin(table):
    """k0 = argmin over [-kt, kt]; ties: smallest |k|, then smaller k."""
    table = np.asarray(table, np.int64)
    kt = (table.shape[1] - 1) // 2
    ks = np.arange(-kt, kt + 1)
    order = np.lexsort((ks, np.abs(ks)))  # preference order of columns
    t = table[:, order]
    j = np.argmin(t, axis=1)  # first minimum in preference order
    return ks[order][j].astype(np.int64)


def mst_weights(table, k0):
    c0 = table_cost(table, k0)
    return np.minimum(table_cost(table, k0 + 1), table_cost(table, k0 - 1)) - c0


class _DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        self.p[a] = b
        return True


def kruskal(n_nodes, f, t, w):
    order = np.lexsort((np.arange(len(w)), w))
    d = _DSU(n_nodes)
    tree = []
    for e in order:
        if d.union(int(f[e]), int(t[e])):
            tree.append(int(e))
    return np.array(tree, np.int64)


def route_on_tree(n_nodes, f, t, tree, root, need):
    """Flow on tree edges so that divergence == need (sum(need) == 0).
    Returns per-edge delta (only tree edges nonzero)."""
    adj = [[] for _ in range(n_nodes)]
    for e in tree:
        adj[f[e]].append((t[e], e))
        adj[t[e]].append((f[e], e))
    parent_edge = np.full(n_nodes, -1, np.int64)
    order = [root]
    seen = np.zeros(n_nodes, bool)
    seen[root] = True
    i = 0
    while i < len(order):
        u = order[i]; i += 1
        for v, e in adj[u]:
            if not seen[v]:
                seen[v] = True
                parent_edge[v] = e
                order.append(v)
    assert seen.all(), "tree does not span"
    sub = np.asarray(need, np.int64).copy()
    delta = np.zeros(len(f), np.int64)
    for u in reversed(order[1:]):
        e = parent_edge[u]
        s = sub[u]  # must leave subtree of u through e
        if f[e] == u:
            delta[e] += s
            p = t[e]
        else:
            delta[e] -= s
            p = f[e]
        sub[p] += s
    return delta


def mst_init(dims: Dims, table, res):
    f, t = dims.dual_arcs()
    k0 = table_argmin(table)
    w = mst_weights(table, k0)
    tree = kruskal(dims.n_dual, f, t, w)
    need = supply(dims, res) - divergence(dims, k0)
    k = k0 + route_on_tree(dims.n_dual, f, t, tree, dims.ground, need)
    return k, tree, int(w[tree].sum())


# ---------------------------------------------------------------------------
# Connected components (DESIGN.md section 9)
# ---------------------------------------------------------------------------

def label_components(dims: Dims, table, k, mask, thresh, min_size):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    p, q = dims.endpoints()
    k = np.asarray(k, np.int64)
    w = np.minimum(table_cost(table, k + 1), table_cost(table, k - 1)) - table_cost(table, k)
    m = np.ones(dims.npix, bool) if mask is None else (np.asarray(mask).ravel() != 0)
    good = m[p] & m[q] & (w >= thresh)
    N = dims.npix
    A = coo_matrix((np.ones(good.sum()), (p[good], q[good])), shape=(N, N))
    ncomp, lab = connected_components(A, directed=False)
    lab = lab.astype(np.int64)
    lab[~m] = -1
    labels = np.zeros(N, np.uint32)
    comps = []
    for cid in np.unique(lab[lab >= 0]):
        idx = np.nonzero(lab == cid)[0]
        if len(idx) >= min_size:
            comps.append((-len(idx), int(idx[0]), idx))
    comps.sort(key=lambda x: (x[0], x[1]))
    for i, (_, _, idx) in enumerate(comps):
        labels[idx] = i + 1
    return labels, len(comps)


# ---------------------------------------------------------------------------
# Nonconvex local search (calibration reference only)
# ---------------------------------------------------------------------------

def pixel_move_deltas(dims: Dims, table, k):
    """Cost change of shifting each pixel by +1 and by -1 (k_e -= 1 where p is tail, += 1 where head)."""
    p, q = dims.endpoints()
    k = np.asarray(k, np.int64)
    c0 = table_cost(table, k)
    cm = table_cost(table, k - 1) - c0  # edge where shifted pixel is tail (p) under +1
    cp = table_cost(table, k + 1) - c0  # edge where shifted pixel is head (q) under +1
    up = np.zeros(dims.npix, np.int64)
    np.add.at(up, p, cm)
    np.add.at(up, q, cp)
    dn = np.zeros(dims.npix, np.int64)
    np.add.at(dn, p, cp)
    np.add.at(dn, q, cm)
    return up, dn


def local_search_pixels(dims: Dims, table, k, max_iter=100000):
    """Greedy single-pixel moves until none improves (reference for calibration)."""
    k = np.asarray(k, np.int64).copy()
    p, q = dims.endpoints()
    for _ in range(max_iter):
        up, dn = pixel_move_deltas(dims, table, k)
        best = min(up.min(), dn.min())
        if best >= 0:
            break
        # apply a maximal independent set of improving moves (pixels not adjacent)
        cand = np.nonzero((up < 0) | (dn < 0))[0]
        gain = np.minimum(up, dn)[cand]
        cand = cand[np.argsort(gain, kind="stable")]
        taken = np.zeros(dims.npix, bool)
        blocked = np.zeros(dims.npix, bool)
        R, C = dims.nrow, dims.ncol
        for v in cand:
            if blocked[v]:
                continue
            taken[v] = True
            r, c = divmod(int(v), C)
            for rr, cc in ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < R and 0 <= cc < C:
                    blocked[rr * C + cc] = True
        s = np.zeros(dims.npix, np.int64)
        s[taken & (up <= dn)] = 1
        s[taken & (dn < up)] = -1
        k = k + s[q] - s[p]
    return k


def cycle_cancel(dims: Dims, table, k, max_cycles=100000):
    """Calibration reference for nonconvex improvement: repeatedly cancel a
    negative unit cycle in the residual dual graph (incremental costs; for
    locally concave edges only the cheaper direction is kept; parallel arcs
    reduced to the cheapest), then greedy single-pixel moves; until neither
    improves."""
    import networkx as nx
    f, t = dims.dual_arcs()
    k = np.asarray(k, np.int64).copy()
    S = dims.n_dual
    for _ in range(max_cycles):
        c0 = table_cost(table, k)
        fw = table_cost(table, k + 1) - c0
        bw = table_cost(table, k - 1) - c0
        best = {}
        for e in range(len(k)):
            u, v = int(f[e]), int(t[e])
            if u == v:
                continue
            arcs = [(u, v, int(fw[e]), e, +1), (v, u, int(bw[e]), e, -1)]
            if fw[e] + bw[e] < 0:
                arcs = [min(arcs, key=lambda a: a[2])]
            for a in arcs:
                key = (a[0], a[1])
                if key not in best or a[2] < best[key][2]:
                    best[key] = a
        G = nx.DiGraph()
        for (u, v), a in best.items():
            G.add_edge(u, v, weight=a[2])
        G.add_node(S)
        for u in range(S):
            G.add_edge(S, u, weight=0)
        try:
            cyc = nx.find_negative_cycle(G, S)
        except nx.NetworkXError:
            kk = local_search_pixels(dims, table, k)
            if total_cost(table, kk) < total_cost(table, k):
                k = kk
                continue
            break
        for u, v in zip(cyc[:-1], cyc[1:]):
            a = best[(u, v)]
            k[a[3]] += a[4]
    return k


def total_cost(table, k):
    return int(table_cost(table, k).sum())


def cpsat_opt(dims: Dims, table, psi, B=KT + 1, nb=6, time_limit=60.0):
    """Exact optimum over pixel fields n (|n| <= nb, n[0] = 0) with |k| <= B, via CP-SAT."""
    from ortools.sat.python import cp_model
    from .conventions import wrap_counts
    p, q = dims.endpoints()
    c = wrap_counts(dims, psi)
    E = dims.n_edges
    ks = np.arange(-B, B + 1)
    tabs = table_cost(np.repeat(np.asarray(table, np.int64), len(ks), 0), np.tile(ks, E)).reshape(E, -1)
    m = cp_model.CpModel()
    n = [m.new_int_var(-nb, nb, f"n{i}") for i in range(dims.npix)]
    m.add(n[0] == 0)
    obj = []
    for e in range(E):
        ke = m.new_int_var(0, 2 * B, f"k{e}")
        m.add(ke == int(c[e]) + B + n[q[e]] - n[p[e]])
        ce = m.new_int_var(int(tabs[e].min()), int(tabs[e].max()), f"c{e}")
        m.add_element(ke, [int(x) for x in tabs[e]], ce)
        obj.append(ce)
    m.minimize(sum(obj))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_workers = 8
    st = s.solve(m)
    nn = np.array([s.value(v) for v in n], np.int64)
    p_, q_ = p, q
    kk = c + nn[q_] - nn[p_]
    return (st == cp_model.OPTIMAL), int(s.objective_value), kk
