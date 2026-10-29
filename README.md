# jupwrap

Statistical-cost network-flow (SCF) phase unwrapping for 2-D InSAR interferograms, in Rust with a Python API.

> **Experimental.** The cost-model formulas (C1–C6 in [DESIGN.md](DESIGN.md) §5) are the authors' own reconstruction from the abstracts of Chen and Zebker (2000, 2001, 2002). They have **not** been verified against the papers' full text. jupwrap has been tested **only on synthetic interferograms**, not validated on real data.

## About

jupwrap reimplements the SCF method of Chen and Zebker, the method behind [SNAPHU](https://web.stanford.edu/group/radar/softwareandlinks/sw/snaphu/). It was written from the published papers' abstracts and textbook network-flow algorithms. It uses no SNAPHU source code and no CS2 solver, and SNAPHU was not used as a test oracle.

The Python package has a snaphu-py-compatible `unwrap()` signature (the parameter names and defaults of [snaphu-py](https://github.com/isce-framework/snaphu-py) 0.4). Results are not expected to match SNAPHU's.

## What it does

- **Cost modes:** `cost="smooth"` (Gaussian; convex, solved exactly) and `cost="defo"` (Gaussian plus a uniform discontinuity tail; nonconvex, solved on a convex surrogate then improved by cycle canceling). `"topo"` is not implemented.
- **Initialization:** `init="mcf"` (exact convex min-cost flow, own epsilon-scaling solver) or `init="mst"` (minimum spanning tree, then improvement).
- **Tiling:** `ntiles`, `tile_overlap`, `nproc`. Tiles are solved in parallel, merged through reliable regions and a secondary network problem, then optionally reoptimized on the full image. No dimension cap; output does not depend on `nproc`.
- **Connected components:** labels 1..N by decreasing size; 0 = masked, unreliable, or below `min_conncomp_frac`.
- **Masks:** bool or uint8, 0 = excluded; masked pixels get `unw = 0`. NaNs are treated as 0; `corr` is clipped to [0, 1].

Differences from snaphu-py behaviour: `tile_cost_thresh` is in 1/100 nat (500 = 5 nats); `regrow_conncomps=False` behaves like `True`; `scratchdir` and `delete_scratch` are ignored.

## Build from source

Nothing is published to PyPI. Requires Rust ≥ 1.85, Python ≥ 3.10 and maturin ≥ 1.5, < 2.

```sh
git clone https://github.com/DanielMercado60660/jupwrap.git
cd jupwrap
pip install "maturin>=1.5,<2"
maturin build --release -m py/Cargo.toml
pip install target/wheels/jupwrap-*.whl
```

Use maturin for the Python package: plain `cargo build --workspace` fails to link the PyO3 crate on macOS.

Tested on macOS (Apple silicon) only. If you build and test it on Linux or another platform, please [open an issue](https://github.com/DanielMercado60660/jupwrap/issues) with the result.
<!-- TODO(Daniel): test on a Linux VM. -->

## Usage

```python
import numpy as np
import jupwrap

y, x = np.ogrid[-3:3:512j, -3:3:512j]
phi = np.pi * (x + y)                         # synthetic phase ramp
igram = np.exp(1j * phi).astype(np.complex64)
corr = np.ones(igram.shape, dtype=np.float32)

unw, conncomp = jupwrap.unwrap(igram, corr, nlooks=1.0, cost="smooth", init="mcf")
# unw: float32 radians; conncomp: uint32 labels

unw, conncomp = jupwrap.unwrap(igram, corr, 1.0, ntiles=(2, 2), tile_overlap=64, nproc=4)
```

Passing `unw=` or `conncomp=` arrays fills them in place.

## How it was tested

All inputs are synthetic, regenerated from seeds (`judge/ref/synth.py`): topography with ridges, a deformation bowl with a fault step, coherence maps with decorrelated holes, a layover band and a masked water body, plus multilook noise. The reference is the exact OR-tools optimum of **jupwrap's own cost tables** (defo: of its convex surrogate, then greedy pixel moves), so the tests check the solver, not the cost model.

- **Components** (`crates/*/tests`): residues and integration; cost tables against a Python reference; min-cost flow optimum against OR-tools and networkx; MST initialization; connected components; nonconvex improvement against the CP-SAT optimum on small grids.
- **End to end** (`judge/tests`, via a test harness binary): exact recovery of residue-free input; smooth mode reaches the reference optimum; defo within tolerance; degenerate shapes; tiling with and without reoptimization; identical output for 1 and 4 threads; a 33000 × 512 strip, beyond SNAPHU's 32 000-pixel limit.
- **Python API:** signature matches snaphu-py; output bit-identical to the harness; masks, NaNs, invalid arguments, strided inputs.

To run them (the Python tests need [uv](https://docs.astral.sh/uv/) and take about 12 minutes):

```bash
cargo test --release                      # Rust suites
cargo build --release -p jupwrap-cli      # harness binary used by judge/tests
cd judge
uv sync && uv pip install "maturin>=1.5,<2"
uv run maturin build --release -m ../py/Cargo.toml
uv pip install ../target/wheels/jupwrap-*.whl
.venv/bin/python -m pytest tests
```

Results from one run during the 2026-10-28 build, on an Apple M5 Max. "Fraction correct" = share of high-coherence truth pixels within π of the truth after removing each region's constant 2π offset.

| Case | Mode | Fraction correct (jupwrap / reference) | Wall time |
|---|---|---|---|
| topo 512², 4 looks | smooth | 0.9974 / 0.9974 | 2.6 s |
| defo 512² | defo | 0.9964 / 0.9953 | 10.4 s |
| topo 2048² | smooth | 0.999997 / 0.999997 | 14.8 s |
| defo 1024², 2 × 3 tiles | defo | 0.9738 / 0.9783 | 202 s |
| strip 33000 × 512, 12 × 1 tiles, 8 threads | smooth | 0.99999 / — (no reference) | 33 s (single tile: 446 s) |

## Limitations and roadmap

- Verify C1–C6 against the full text of the papers ([QUESTIONS.md](QUESTIONS.md) item 1).
- Validate on real interferograms with a trusted reference (item 7). TODO(Daniel): choose datasets.
- No `topo` mode; tiled defo ran slightly below the reference.
- `corr` is assumed to be windowed sample coherence with `nlooks` effective looks.

## References

TODO(Daniel): check full titles; add author initials, DOIs, and the erratum to Chen and Zebker (2000).

- Chen and Zebker (2000). Network approaches to two-dimensional phase unwrapping. *J. Opt. Soc. Am. A* 17, 401.
- Chen and Zebker (2001). Two-dimensional phase unwrapping with use of statistical models for cost functions in nonlinear optimization. *J. Opt. Soc. Am. A* 18, 338.
- Chen and Zebker (2002). Phase unwrapping for large SAR interferograms: statistical segmentation and generalized network models. *IEEE Trans. Geosci. Remote Sens.* 40, 1709.
- Ahuja, Magnanti and Orlin (1993). *Network Flows*.

## License

Apache-2.0; see [LICENSE](LICENSE).
