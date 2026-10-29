# jupwrap design: a clean-room SCF phase unwrapper in Rust

Design as of 2026-10-28. License Apache-2.0.

## 0. Ground rules

- **Clean room.** Contributors must not read, download or copy SNAPHU source code, its CS2 solver, or the `whirlwind` repository. SNAPHU is not used as a test oracle. The implementation is based only on this document, the published papers below, and textbook algorithms.
- **Allowed sources.** Chen and Zebker (2000) JOSA-A 17:401 "Network approaches to two-dimensional phase unwrapping" [CZ2000]; Chen and Zebker (2001) JOSA-A 18:338 "Two-dimensional phase unwrapping with use of statistical models for cost functions in nonlinear optimization" [CZ2001]; Chen and Zebker (2002) IEEE TGRS 40:1709 "Phase unwrapping for large SAR interferograms: statistical segmentation and generalized network models" [CZ2002]; textbook network-flow algorithms (Ahuja, Magnanti and Orlin, *Network Flows*, 1993 [AMO]: shortest paths, MST, min-cost flow, convex-cost flow by arc expansion, cycle canceling); standard InSAR statistics (Cramér–Rao phase variance, multilook sample coherence). The snaphu-py *public signature* only (its Python docstring), never its bundled C.
- **Papers were not available in full text** when this design was written (all three are paywalled; only abstracts were read). Every formula below is therefore an explicit, self-contained definition *in the spirit of* the cited paper section; no equation numbers are cited because none could be verified. See section 4 "Unverified reconstruction".
- The code implements exactly the signatures and conventions defined here. The tests check these definitions, not SNAPHU.

## 1. Pipeline

```
igram (complex64) --phase--> psi (f32, wrapped)      corr (f32), mask (u8, optional)
        |                                                   |
        +--> wrapped gradients, wrap counts (core)          |
        +--> residues on 2x2 loops (grid)                   |
        +--> statistical cost tables per edge (costs) <-----+
        |
        +--> initial cycle field k:  mcf: exact convex MCF on the convex surrogate (mcf)
        |                            mst: MST + subtree routing (mst-init), then convex canceling
        +--> nonconvex improvement under the true costs (nonconvex)
        +--> integrate k -> integer field n, unw = psi + 2*pi*n (grid)
        +--> connected components (conncomp)
large images: tiles -> per-tile solve -> reliable regions -> secondary problem -> merge
              -> optional full-image reoptimization (tiling)
```

Everything is a network-flow problem on the **dual graph** of the pixel grid (CZ2000): each primal edge (pixel pair) carries an integer cycle count `k_e`; each 2x2 loop is a dual node whose supply is minus its residue; the unwrapped gradients are curl-free exactly when `k` is a feasible flow. Costs are negative log-likelihoods of the unwrapped gradient (CZ2001), so the minimum-cost flow is an approximate MAP solution.

## 2. Crates and dependencies

| Crate (path) | Depends on | Purpose |
|---|---|---|
| `jupwrap-core` (`crates/core`) | none | conventions, `Dims`, `Network`, `ArcCost`, `GridCosts`, params, errors |
| `jupwrap-testkit` (`crates/testkit`) | core | `.npy` reader, golden helpers, splitmix64 recipes |
| `jupwrap-grid` (`crates/grid`) | core | residues, dual network, integration |
| `jupwrap-mcf` (`crates/mcf`) | core | exact convex min-cost flow |
| `jupwrap-costs` (`crates/costs`) | core | statistical cost model, convex surrogate |
| `jupwrap-init` (`crates/init`) | core | MST initialization |
| `jupwrap-conncomp` (`crates/conncomp`) | core | connected components |
| `jupwrap-nonconvex` (`crates/nonconvex`) | core | local improvement under nonconvex costs |
| `jupwrap-solve` (`crates/solve`) | all of the above | single-tile pipeline |
| `jupwrap-tiling` (`crates/tiling`) | solve (+ all) , rayon | tiled unwrapping |
| `jupwrap` (`crates/jupwrap`) | solve, tiling | facade: `unwrap`, `unwrap_complex` |
| `jupwrap-cli` (`crates/cli`) | jupwrap | harness binary for the end-to-end tests |
| `jupwrap-py` (`py/`: `py/src`, `py/python`) | jupwrap, pyo3 | Python package `jupwrap` |

External crates (pinned in the workspace, fetched; build with `--offline`): `rayon =1.12.0` (tiling only), `pyo3 =0.30.1` (abi3-py310, Python package only), `serde_json =1.0.154` (tests only). There are no other dependencies.

## 3. Core conventions (implemented in `crates/core/src/lib.rs`, mirrored in `judge/ref/conventions.py`)

### 3.1 Arrays
Row-major, `index = r*ncol + c`. Phase `psi` is f32 radians; all arithmetic is f64 on f32-cast values. Coherence `corr` is f32 (NaN means 0; clipped to [0,1]). Mask is u8 with 0 = invalid. Complex input is interleaved little-endian complex64.

### 3.2 Wrapping
`wrap_count(d) = floor(d/TAU + 0.5)`, `wrap(d) = d - TAU*wrap_count(d)`, range `[-pi, pi)`. Use the core functions; do not reimplement with `round()` (ties differ).

### 3.3 Edges, loops, dual arcs, flows
- Edge order: all vertical edges `p=(r,c) -> q=(r+1,c)` (index `r*ncol + c`), then all horizontal edges `p=(r,c) -> q=(r,c+1)` (index `nv + r*(ncol-1) + c`). `Dims::edge_endpoints(e) -> (p, q)`.
- Wrapped gradient `Delta_e = wrap(psi_q - psi_p)`, wrap count `c_e = wrap_count(psi_q - psi_p)`.
- Unwrapped gradient `Delta_e + TAU*k_e`, integer `k_e` = the flow on dual arc e.
- Loop `(i,j)` (top-left pixel), index `i*(ncol-1)+j`; ground node index `n_loops`.
- Dual arc of edge e from `Dims::edge_dual(e)`: horizontal `(r,c)`: from loop `(r,c)` (ground if `r = nrow-1`) to loop `(r-1,c)` (ground if `r = 0`); vertical `(r,c)`: from loop `(r,c-1)` (ground if `c = 0`) to loop `(r,c)` (ground if `c = ncol-1`). With `nrow == 1` or `ncol == 1` all arcs are ground self-loops.
- Residue of loop `(i,j)`: `-(c_h(i,j) + c_v(i,j+1) - c_h(i+1,j) - c_v(i,j))`, equal to `round(curl/TAU)` of the wrapped gradients; always -1, 0 or +1. Computed from integer wrap counts, so bit-exact across languages.
- Conservation (`out - in = supply`): loop supply `-residue`, ground supply `sum(residues)`.
- Integration: `n[0] = 0` and `n_q - n_p = k_e - c_e` for every edge (integer, exact); `unw = (psi as f64 + TAU*n) as f32`. `k` is consistent iff it is a feasible flow. `k_from_n`: `k_e = c_e + n_q - n_p`.
- Shifting pixel p by +1 (`n_p += 1`) changes `k_e` by -1 on edges where p is the tail and +1 where p is the head; such a shift is a unit circulation around p in the dual.

### 3.4 Worked 2x2 example (test `worked_example_2x2`)
`psi = [[0.0, 2.5], [-2.5, 1.0]]`. Edges: e0 = v(0,0) `d=-2.5, c=0`; e1 = v(0,1) `d=-1.5, c=0`; e2 = h(0,0) `d=2.5, c=0`; e3 = h(1,0) `d=3.5, c=1` (`Delta=-2.783`). Residue of loop 0 = `-(0 + 0 - 1 - 0) = +1`. Dual arcs `(tail, head)` with ground = 1: e0 `(1,0)`, e1 `(0,1)`, e2 `(0,1)`, e3 `(1,0)`; supply `[-1, +1]`. `k = [0,0,0,1]` integrates to `n = [0,0,0,0]` (h(1,0) is not wrapped); `k = [0,-1,0,0]` integrates to `n = [0,0,0,-1]`; `k = 0` is `NotCurlFree`.

### 3.5 Generic networks and costs
`Network { n_nodes, tail, head, lower, upper, supply }` (i32 flows, i64 supplies; self-loops and parallel arcs allowed). `ArcCost::cost(arc, x) -> i64` is separable and integer. `GridCosts { dims, kt, table }` holds `table[e*(2kt+1) + k+kt]` for `|k| <= kt` (`KT = 4`); beyond the table the extension is linear with slope `max(last increment, MIN_TAIL_SLOPE = 100)` (convexity-preserving). Grid networks use bounds `[-K_BIG, K_BIG]` (`K_BIG = 2^20`): effectively unbounded.

## 4. What comes from which paper, and what is unverified

| Component | Source (topic) | Our definition | Verified against paper text? |
|---|---|---|---|
| Residues, branch-cut-free dual formulation, ground node | CZ2000 network formulation; standard (Goldstein et al. 1988; Costantini 1998) | section 3 | No (abstract only); the formulation itself is textbook |
| MAP objective = sum of per-edge negative log pdfs of unwrapped gradients | CZ2001 MAP framework (abstract confirms MAP, wrapped phase + intensity + coherence, nonlinear network flow) | section 5 | Abstract only |
| Phase variance from coherence and looks | CZ2001 noise model topic; we use the Cramér–Rao form (Rodriguez and Martin 1992) | C2 | **Unverified** that CZ2001 uses this form |
| Coherence de-biasing | CZ2001 discusses coherence estimation (unverified); our moment correction `E[rho^2] = 1/L` at zero coherence | C1 | **Unverified** |
| Local mean phase gradient (window) | snaphu-py parameter `phase_grad_window` exists; idea of centring costs on local slope from CZ2001 (unverified) | C3 | **Unverified** |
| Smooth mode: Gaussian core only | our reading of a "smooth surface" model | C6 | **Unverified** |
| Defo mode: Gaussian core plus a constant tail for discontinuities (faults) | CZ2001 deformation model (abstract mentions differential interferogram of earthquake deformation) | C6 | **Unverified shape and constants** |
| Topo mode (layover, shadow, intensity, baseline geometry) | CZ2001 topography model | **not built in v1** (needs geometry; deferred) | n/a |
| MST initialization | CZ2000 first algorithm (shortest paths and spanning trees, improving residue-cut) | section 8 | Abstract only; ours is a plain MST on the dual graph |
| Nonlinear (nonconvex) network flow by local improvement | CZ2000 second algorithm (extends linear network flow to the nonlinear L0 case; NP-hardness) | section 7 | Abstract only |
| Tiling, reliable regions, secondary (generalized) network | CZ2002 (abstract: tiles unwrapped individually, divided into reliable regions, then a secondary problem) | section 11 | Abstract only |

"Matches the paper" therefore means, per mode: **smooth** = convex Gaussian NLL costs from coherence (CZ2001's MAP principle), solved exactly; **defo** = nonconvex NLL with a discontinuity plateau (CZ2001's deformation idea), solved by convex surrogate plus local improvement (CZ2000's nonlinear network flow idea); **topo** = not implemented. Shapes and constants are ours until someone with the papers checks section 5 against CZ2001 (QUESTIONS.md item 1).

## 5. Cost model (crate `jupwrap-costs`; reference `judge/ref/costs.py`)

Inputs: `psi`, `corr`, optional `mask`, `CostParams { mode, nlooks L, phase_grad_window (wr, wc), sigma2_signal, p_discontinuity }` (defaults in core: window (7,7), `sigma2_signal = 0.02 rad^2`, `p_discontinuity = 1e-4`).

- **C1 effective coherence** `effective_coherence(corr, L)`: `r = clip(nan->0, 0, 1)`; if `L > 1`: `r = sqrt(max(0, (L r^2 - 1)/(L - 1)))`; return `min(r, RHO_MAX = 0.999)`.
- **C2 pixel phase variance** `pixel_sigma2(corr, L)`: `rho = C1`; if `rho > 0`: `min((1 - rho^2)/(2 L rho^2), pi^2/3)`, else `pi^2/3`.
- **C3 mean gradient** `mean_gradients(dims, psi, mask, (wr, wc))`: for edge e of direction D at `(r,c)`, average over same-direction edges in rows `[r - wr/2, r + wr/2]`, cols `[c - wc/2, c + wc/2]` (integer halves, clipped to that direction's edge array), using only edges with both endpoints valid: `S = sum sin(Delta)`, `C = sum cos(Delta)`, `mu = atan2(S, C)`; `mu = 0` when the window has no valid edge.
- **C4 edge variance** `s2_e = sigma2_p + sigma2_q + sigma2_signal`.
- **C5 deviation** `x_k = Delta_e + TAU*k - mu_e`, `k in [-KT, KT]`.
- **C6 float cost**: smooth `c(k) = x^2 / (2 s2)`; defo `a = ln(1-P) - x^2/(2 s2) - 0.5 ln(TAU s2)`, `b = ln(P) - ln(TAU (2KT+1))`, `c(k) = -(max(a,b) + ln1p(exp(min(a,b) - max(a,b))))` (log-sum-exp of a Gaussian and a uniform density over the table span).
- **C7 quantization**: `table[k] = min(COST_CAP, floor(COST_SCALE*(c(k) - min_j c(j)) + 0.5))`, `COST_SCALE = 100` units per nat, `COST_CAP = 1e9`; edges with a masked endpoint get an all-zero row.
- **C8 convex surrogate** `convex_envelope`: per row `v[0..2KT]` (index `i = k + KT`): lower convex hull (monotone chain, pop on cross product `<= 0`, integer arithmetic); for every unit segment inside hull segment `[h, j]`, increment `inc_i = floor_div(v[j] - v[h], j - h)`; anchor `E[i*] = v[i*]` at `i* = argmin v` (smallest index on ties); `E[i+1] = E[i] + inc_i` to the right, `E[i-1] = E[i] - inc_{i-1}` to the left. Result is convex (identity on convex rows).

Smooth tables are convex by construction (second differences >= ~600 units, far above rounding). Defo tables are not.

Test tolerances: C1/C2 within 1e-12, C3 within 1e-9 (angular), tables within +-1 unit (rounding ties) with >= 99.9% exact, C8 exact.

## 6. Exact convex min-cost flow (crate `jupwrap-mcf`)

`solve_convex(net, cost) -> Result<Vec<i32>, McfError>`: feasible flow of minimum total cost for separable convex integer costs over `[lower, upper]`; `Infeasible` if none exists; `InvalidInput` if `net.validate()` fails. Never panics on those inputs. Per-arc minima need not be at 0 (zero-supply instances can have a nonzero optimal circulation). Self-loops: set each to its own cost minimizer.

Textbook route [AMO ch. 9, 10, 14]: start every arc at its own cost minimizer (all residual reduced costs are then >= 0), compute node imbalances, then successive shortest paths with Dijkstra and node potentials on the residual graph whose unit arcs cost `c(x+1)-c(x)` forward and `c(x-1)-c(x)` backward. Plain one-unit-per-Dijkstra SSP is correct but too slow for the timing gates (1024x1024 grid, ~30k residues). Suitable methods: primal-dual with many augmentations per Dijkstra (augment along every admissible path in the zero-reduced-cost subgraph, Dinic-style blocking flow), capacity/cost scaling, or network simplex. CS2 source is not used or imitated; cost-scaling from AMO or Goldberg's published paper is fine. Grid bounds are `+-2^20`: never iterate over a bound range; minimizers of convex grid tables lie inside `[-kt, kt]`, but generic `TableCost` arcs (bounded tables) must be handled too.

Tests: 27 explicit instances (random, linear, circulation, self-loops/parallel/ties, infeasible, grid duals with statistical costs) with optimum cost from OR-tools SimpleMinCostFlow, cross-checked by networkx network_simplex (unit-arc expansion of convex costs); 4 large splitmix64 "quadgrid" recipes up to 1024x1024 (time limit 30 s + 5x OR-tools time: 128^2 30.7 s, 512^2 49 s, 1024^2 233 s, 300x2000 109 s). Exact cost equality plus feasibility; flows may differ.

## 7. Nonconvex improvement (crate `jupwrap-nonconvex`)

`improve_grid(costs, k, opts) -> Result<ImproveStats, McfError>`: from a feasible `k`, apply circulations that strictly lower the total cost until none is found (or `opts` limits hit). Postconditions: divergence unchanged; cost non-increasing; no improving single-pixel shift; `stats.initial_cost`/`final_cost` exact.

This is CZ2000's "nonlinear network flow" idea in our own design: cycle canceling on the residual dual graph with **incremental** costs `fw = c(k+1) - c(k)`, `bw = c(k-1) - c(k)`. For a locally concave edge (`fw + bw < 0`) the two residual directions form a spurious negative 2-cycle; keep only the cheaper direction for that edge while searching. Recommended engine: Bellman–Ford/SPFA from all nodes (initial distance 0) with periodic predecessor-graph cycle extraction, cancel every disjoint negative cycle found (verify each cycle's true cost change before applying; skip cycles that reuse an edge), repeat; then greedy single-pixel moves; loop until neither improves. Our scratch implementation of exactly this solved the 200x240 test instance in 0.09 s, while a naive full Bellman–Ford per cycle took 1000 s. For convex costs this procedure reaches the global optimum (no negative cycle iff optimal, AMO ch. 14), and the tests check that.

Tests: convex instances from MST starts must reach the OR-tools optimum exactly; 16 defo and 8 plateau instances (7x7 to 12x12, starting from the convex-surrogate optimum) are compared with the CP-SAT global optimum: mean gap <= 3% (defo) / 11.06% (plateau), each <= 15.15% / 23.59% (1.5x the reference heuristic's mean gap, its max + 0.05); medium 128x128 and 200x240 instances within 20 s.

## 8. MST initialization (crate `jupwrap-init`)

`mst_init(costs, residues) -> MstInit { k, tree }`:
1. `k0_e = argmin_{|k| <= kt} cost(e, k)`, ties: smallest `|k|`, then smaller `k`.
2. Weight `w_e = min(cost(e, k0+1), cost(e, k0-1)) - cost(e, k0)` (uses the extension rule).
3. `tree` = a minimum spanning tree of the dual graph (all loops plus ground; parallel arcs and self-loops are allowed; self-loops never enter) under `w` — any MST under ties.
4. Residual need `r = supply - divergence(k0)`; root the tree at ground; for every tree edge from child u to its parent, the flow `s = sum of r over u's subtree` must leave the subtree: `k_e = k0_e + s` if the dual arc is oriented u -> parent, else `k0_e - s`. Non-tree edges keep `k0`.
Tests: spanning and acyclic, weight equals the Kruskal minimum, `k` equals that routing on the returned tree exactly, feasible; 1024x1024 within 20 s.

## 9. Connected components (crate `jupwrap-conncomp`)

`label_components(costs, k, mask, thresh, min_size) -> (labels, N)`: an edge is reliable iff both endpoints are valid and `min(cost(k+1), cost(k-1)) - cost(k) >= thresh`. 4-connected components over reliable edges; masked pixels and components with fewer than `min_size` pixels get 0; the rest get 1..N by decreasing size, ties by smaller first (row-major) pixel index. Defaults: `thresh = CONNCOMP_COST_THRESH = 300` units (3 nats; calibrated so decorrelated holes drop out while >= 99.9% of coherent pixels stay labelled), `min_size = min_component_size(min_conncomp_frac, npix) = max(1, ceil(frac*npix))`. Exact test; 4096x4096 within 20 s.

## 10. Single-tile pipeline (crate `jupwrap-solve`)

`solve_grid(psi, costs, start, params) -> (k, n, init_cost, final_cost)` and `unwrap_single(input, params) -> UnwrapOutput`:
1. `costs = build_costs(dims, psi, corr, mask, &params.cost_params())`; `surrogate = convex_envelope(costs)` (identity for smooth).
2. Residues and the dual network (grid).
3. Start: `Mcf` = `solve_convex(dual_network, surrogate)`; `Mst` = `mst_init(costs, res)` then `improve_grid(surrogate)` (which reaches the surrogate optimum because it is convex); `FromN(n)` = `k_from_n(psi, n)`.
4. `init_cost = total_cost(costs, k)`. Run `improve_grid(costs, k, &params.improve)` unless (mode is smooth and start is `Mcf`), where the start is already the global optimum.
5. `n = integrate(psi, k)`; `unw = psi + TAU*n` with 0 on masked pixels; `conncomp = label_components(costs, k, mask, params.conncomp_cost_thresh, min_component_size(params.min_conncomp_frac, npix))`; fill `UnwrapStats` (residue count, costs, timings, `n_tiles = 1`).

## 11. Tiling (crate `jupwrap-tiling`, CZ2002)

`tile_bounds(nrow, ncol, (R, C), (or, oc))`: core rows `[i*nrow/R, (i+1)*nrow/R)` (integer division), extended by `or` on both sides and clipped; same for columns; row-major order. `unwrap_tiled(input, params)`: must accept any size (no dimension cap), return exactly `unwrap_single` for `ntiles = (1,1)`, and be deterministic regardless of `nproc`.

Algorithm (our reading of CZ2002; the end-to-end tests define what is acceptable):
1. Costs once on the full image (so windows do not see tile edges); clamp `ntiles` to the image size.
2. Solve tiles in parallel (`rayon` pool with `nproc` threads, 0 = all) with `solve_grid` on the tile's sub-table (or tile-local costs).
3. Reliable regions per tile: 4-connected components of tile pixels over edges whose incremental cost (as in section 9) is `>= tile_cost_thresh`; regions smaller than `min_region_size` are unreliable.
4. Secondary problem: one integer offset per region; for every pair of regions that share overlap pixels or primal edges, an aggregated cost `G_AB(m)` (sum of edge costs across their common boundary as a function of the relative offset, plus overlap disagreement). Minimize `sum G_AB(m_B - m_A)` (small graph; spanning tree by confidence then local moves, or the CZ2002 generalized-network idea).
5. Assemble `n` over each tile's core area, fill unreliable pixels, then if `single_tile_reoptimize` run `solve_grid(..., Start::FromN(n))` on the full image (under convex smooth costs this reaches the global optimum).
6. Connected components on the full image (`regrow_conncomps = False` is accepted and behaves like `True`; documented deviation).
Tests (pytest via the CLI): 1024^2 2x2 with reoptimization must hit the single-tile optimum; 3x3 without reoptimization within 0.01 of reference accuracy; defo 2x3; masked 2x2; identical output for nproc 1 and 4; a 33000x512 strip (beyond SNAPHU's 32k limit) with 12x1 tiles on 8 threads at accuracy >= 0.995 and wall time <= max(0.7x single-tile, 120 s).

## 12. Facade and CLI

`jupwrap::unwrap(&UnwrapInput, &UnwrapParams)`: validates, sanitizes `corr` (NaN -> 0, clip), dispatches `ntiles == (1,1)` to `jupwrap_solve::unwrap_single`, else `jupwrap_tiling::unwrap_tiled`. `unwrap_complex` takes interleaved complex64 (NaN -> phase 0). `jupwrap-cli` (see `crates/cli/src/main.rs` header) reads raw files and writes `unw.f4`, `cc.u4`, `stats.json`.

## 13. Python API (`py/`)

`jupwrap.unwrap(igram, corr, nlooks, cost="smooth", init="mcf", *, mask=None, min_conncomp_frac=0.01, phase_grad_window=(7, 7), ntiles=(1, 1), tile_overlap=0, nproc=1, tile_cost_thresh=500, min_region_size=100, single_tile_reoptimize=True, regrow_conncomps=True, scratchdir=None, delete_scratch=True, unw=None, conncomp=None) -> (unw, conncomp)` — the snaphu-py 0.4 signature, parameter for parameter. Semantics: `igram` 2-D complex (any complex dtype, any strides; real -> `TypeError`; not 2-D -> `ValueError`); `corr` same shape (else `ValueError`); NaN in either -> 0; `nlooks >= 1`; `cost in {"smooth","defo"}` (`"topo"` and others -> `ValueError`); `init in {"mcf","mst"}`; `mask` bool or uint8 of the same shape, 0/False excluded; `tile_overlap` int or pair; `ntiles` >= 1; `min_conncomp_frac` in [0,1]; `scratchdir`/`delete_scratch` accepted and ignored (no scratch files). Returns `float32` unw and `uint32` conncomp (0 = no component); if `unw`/`conncomp` arrays are passed they are filled (with dtype casting) and returned as the same objects. Must produce bit-identical results to `jupwrap-cli` for the same inputs (both call the same facade). The Rust side is `py/src/lib.rs` (module `jupwrap._jupwrap`); it releases the GIL during the solve.

## 14. Tests, synthetic data, thresholds

- **Exact tests** (Rust, `cargo test`): grid, costs (tolerances above), mst-init, conncomp, mcf (optimal cost), nonconvex (convex optimum exact; nonconvex gap to CP-SAT optimum).
- **Synthetic interferograms** (`judge/ref/synth.py`): topography (power-law random field plus ridges, max gradient 1.0–2.0 rad/px), deformation (Mogi-like bowl, 4–40 fringes, tapered fault step of 2–3 cycles), coherence maps (smooth fields in [0.25,0.95], low-coherence holes rho = 0.08, a layover band, a masked water body). Noise: L independent looks of a correlated circular complex Gaussian pair, `igram = exp(i phi) * sum s1 conj(s2)`, sample coherence passed as `corr`. Inputs are regenerated from seeds; `judge/golden/e2e/expected.json` stores input hashes and reference numbers.
- **Reference pipeline** (`judge/ref/pipeline.py`): exact OR-tools optimum of the smooth costs (or of the defo surrogate, then greedy pixel moves).
- **Metrics**: truth regions = 4-connected pixels with true rho >= 0.5, outside holes and mask, >= 0.2% of the image. Per region, offset = modal cycle error; a pixel is correct iff `|unw - phi - TAU m| < pi`. Congruence: `max |wrap(unw - psi)| < 1e-3` on valid pixels. Cost: the cycle field implied by the output, evaluated under the reference tables over valid edges.
- **Thresholds and why**: smooth is convex, so a correct pipeline reaches the reference optimum: cost `<= ref*(1+1e-4) + 100` (slack for +-1 table rounding) and fraction `>= ref - 0.002` (ties between equal-cost optima). Defo is nonconvex and accuracy does not track cost exactly: fraction `>= ref - 0.005` (mst 0.008; tiled 0.01) and cost `<= 1.02 ref`. Residue-free and README-ramp inputs must be exact. Runtime gates are loose (2048^2 single tile <= 600 s); wall time and peak RSS are always recorded in `judge/results/*.json`. The thresholds were confirmed achievable by an independent implementation during design (except the largest timing gates).
- Reference numbers (e.g. topo 512 L=4: ref fraction 0.9974 vs 0.9963 for an unweighted MCF and 0.387 for path integration) show the cost model matters but modestly on these synthetics; the cost gate is what discriminates.

## 15. Numerical and performance notes

- Use i64 for path lengths, potentials and cost sums; tables are i32.
- Never iterate over `[-K_BIG, K_BIG]`.
- Memory budget: tables are 36 B per edge (2 edges per pixel); aim for < 200 B/pixel total in a single-tile solve.
- Determinism: same inputs give the same outputs regardless of thread count.
- No `#![deny(warnings)]`; no new dependencies.

## 16. Contributor notes

- Build and test offline: `cargo ... --offline`; Python via `judge/.venv/bin/python`.
- Clean room: no SNAPHU/CS2/whirlwind source, ever; no SNAPHU binary.
