# Open questions for InSAR practitioners

Design questions that need input from people who work with InSAR data.

1. **The papers.** Can you (or a library) supply PDFs of Chen and Zebker 2000 (JOSA-A 17:401, plus its erratum), 2001 (JOSA-A 18:338) and 2002 (IEEE TGRS 40:1709)? Only abstracts were reachable, so DESIGN.md section 5 (coherence de-biasing, phase variance, mean-gradient centring, the smooth model, the defo plateau and its constants) is our own reconstruction. Someone with the text should check it against the paper; we would then adjust the formulas and regenerate the goldens.
2. **Coherence input.** Is the `corr` you pass a windowed sample coherence with `nlooks` equal to the effective number of looks in that window? Our de-biasing (`rho^2 = (L r^2 - 1)/(L - 1)`) assumes that.
3. **Defo discontinuities.** How many cycles do fault offsets in your data jump across one pixel? Our tables span +-4 cycles exactly, with linear tails beyond. What discontinuity probability feels right (we use 1e-4 per edge)?
4. **Units of `tile_cost_thresh`.** Ours is in 1/100 nat, so 500 means "5 nats", which is not SNAPHU's scale. Do Dolphin or your scripts pass explicit values that should keep their meaning, or are defaults fine?
5. **Masked pixels and connected components.** We write `unw = 0` and `conncomp = 0` on masked pixels and drop decorrelated holes from all components (threshold 3 nats). Does downstream code (Dolphin, MintPy) rely on different values or on SNAPHU's component sizes? We treat `regrow_conncomps=False` like `True`; does anyone need the per-tile labels?
6. **`init="mst"`.** Is it used in practice? Ours converges to the same optimum as `"mcf"` on smooth costs, only slower.
7. **Real data for integration.** Which 2–3 public interferograms (with a trusted reference: GNSS, published products, PhUnBench if its terms allow) should the integration test use? Testing so far is synthetic only.

## Broader questions

About unwrapping in practice, to decide where jupwrap should go next.

1. Which unwrapper do you run in production today (SNAPHU, spurt, PHASS, ICU, other), with which mode and tiling, and why?
2. What measured result would make you swap SNAPHU for a new unwrapper? For example ≥99.5% of pixels agreeing on N real pairs, 10× speed, or no tile seams.
3. Which SNAPHU features are essential: defo, smooth or topo modes; tiling with regrow; single-tile reoptimize; connected components; MST vs MCF initialization?
4. Is it acceptable to grade a clean-room unwrapper against the SNAPHU *binary* as a black-box oracle, or should grading use only synthetics and published results? (A legal check may be needed.)
5. Is 2-D spatial unwrapping still the bottleneck worth optimizing, or is the field moving to temporal/3D (spurt, EMCF) or GPU convex methods (FAUST-ADMM) fast enough that a faster SCF unwrapper is obsolete on arrival?
6. What scene sizes and memory ceilings do you hit (full-frame Sentinel-1 at full resolution, NISAR), and is a GPU-only tool acceptable to your users?
7. Which datasets do you trust as ground truth for grading (PhUnBench, Ridgecrest 2019, DISP-S1 GNSS validation sites, others)?
8. Does NISAR L-band (larger, more coherent scenes and stronger ionosphere) change which step hurts most?
