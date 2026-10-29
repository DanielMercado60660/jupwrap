//! jupwrap-core: shared conventions for the clean-room SCF phase unwrapper.
//!
//! Every convention here is mirrored
//! exactly by the Python reference in `judge/ref/conventions.py`.
//!
//! Summary (see DESIGN.md section 3 for the worked 2x2 example):
//! * Images are row-major, `index = r * ncol + c`, `Dims { nrow, ncol }`.
//! * Wrapped phase `psi` is f32; all arithmetic is f64 on f32-cast values.
//! * `wrap(d) = d - TAU * floor(d / TAU + 0.5)`, range `[-pi, pi)`.
//! * Primal edges: all vertical edges first (`p=(r,c) -> q=(r+1,c)`,
//!   index `r*ncol + c`), then horizontal edges (`p=(r,c) -> q=(r,c+1)`,
//!   index `nv + r*(ncol-1) + c`).
//! * Unwrapped gradient on edge e: `Delta_e + TAU * k_e` with integer `k_e`.
//! * Unwrapped phase `phi = psi + TAU * n` with integer pixel field `n`;
//!   `n_q - n_p = k_e - c_e` where `c_e = wrap_count(psi_q - psi_p)`.
//! * Dual nodes: loop (face) `(i,j)` for `i < nrow-1, j < ncol-1`, index
//!   `i*(ncol-1)+j`, plus one ground node with index `n_loops`.
//! * Flow on dual arc e equals `k_e`; arc orientation from [`Dims::edge_dual`].
//! * Supply (out - in) of loop u is `-residue(u)`; ground supply is `sum(residues)`.

use std::f64::consts::PI;

pub const TAU: f64 = std::f64::consts::TAU;

/// Cost units per nat (negative log-likelihood) in quantized cost tables.
pub const COST_SCALE: f64 = 100.0;
/// Clamp for quantized table entries (keeps every entry inside i32).
pub const COST_CAP: i64 = 1_000_000_000;
/// Half-width of grid cost tables: tables hold k in [-KT, KT].
pub const KT: i32 = 4;
/// Minimum slope (cost units per cycle) of the linear extension beyond the table.
pub const MIN_TAIL_SLOPE: i64 = 100;
/// Flow bound used on grid dual networks (effectively unbounded).
pub const K_BIG: i32 = 1 << 20;
/// Coherence cap applied after de-biasing.
pub const RHO_MAX: f64 = 0.999;
/// Default signal (model) variance added to every edge, rad^2, smooth mode.
pub const SIGMA2_SIGNAL_SMOOTH: f64 = 0.02;
/// Default signal (model) variance added to every edge, rad^2, defo mode.
pub const SIGMA2_SIGNAL_DEFO: f64 = 0.02;
/// Default probability that an edge crosses a deformation discontinuity (defo mode).
pub const P_DISCONTINUITY_DEFO: f64 = 1e-4;
/// Default connected-component reliability threshold, cost units.
pub const CONNCOMP_COST_THRESH: i64 = 300;

// ---------------------------------------------------------------------------
// Wrapping
// ---------------------------------------------------------------------------

/// Number of whole cycles removed by [`wrap`]: `floor(d / TAU + 0.5)`.
#[inline]
pub fn wrap_count(d: f64) -> i64 {
    (d / TAU + 0.5).floor() as i64
}

/// Wrap to `[-pi, pi)`: `d - TAU * floor(d / TAU + 0.5)`.
#[inline]
pub fn wrap(d: f64) -> f64 {
    d - TAU * (d / TAU + 0.5).floor()
}

/// Phase of a complex sample, `atan2(im, re)` in f64 then cast to f32.
/// Not bit-judged against Python (libm differences); NaN inputs give 0.
#[inline]
pub fn phase_of(re: f32, im: f32) -> f32 {
    if re.is_nan() || im.is_nan() {
        return 0.0;
    }
    (im as f64).atan2(re as f64) as f32
}

// ---------------------------------------------------------------------------
// Grid indexing and dual orientation
// ---------------------------------------------------------------------------

/// Edge direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeDir {
    /// p = (r, c), q = (r + 1, c)
    Vertical,
    /// p = (r, c), q = (r, c + 1)
    Horizontal,
}

/// Image dimensions. All arrays are row-major.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Dims {
    pub nrow: usize,
    pub ncol: usize,
}

impl Dims {
    pub fn new(nrow: usize, ncol: usize) -> Self {
        Dims { nrow, ncol }
    }
    #[inline]
    pub fn npix(&self) -> usize {
        self.nrow * self.ncol
    }
    #[inline]
    pub fn pix(&self, r: usize, c: usize) -> usize {
        r * self.ncol + c
    }
    /// Number of vertical edges `(nrow-1)*ncol`.
    #[inline]
    pub fn nv(&self) -> usize {
        self.nrow.saturating_sub(1) * self.ncol
    }
    /// Number of horizontal edges `nrow*(ncol-1)`.
    #[inline]
    pub fn nh(&self) -> usize {
        self.nrow * self.ncol.saturating_sub(1)
    }
    #[inline]
    pub fn n_edges(&self) -> usize {
        self.nv() + self.nh()
    }
    /// Number of loops (faces) `(nrow-1)*(ncol-1)`.
    #[inline]
    pub fn n_loops(&self) -> usize {
        self.nrow.saturating_sub(1) * self.ncol.saturating_sub(1)
    }
    /// Dual node index of the ground (outer boundary) node.
    #[inline]
    pub fn ground(&self) -> usize {
        self.n_loops()
    }
    #[inline]
    pub fn n_dual_nodes(&self) -> usize {
        self.n_loops() + 1
    }
    /// Index of the vertical edge `(r,c) -> (r+1,c)`, `r < nrow-1`.
    #[inline]
    pub fn edge_v(&self, r: usize, c: usize) -> usize {
        r * self.ncol + c
    }
    /// Index of the horizontal edge `(r,c) -> (r,c+1)`, `c < ncol-1`.
    #[inline]
    pub fn edge_h(&self, r: usize, c: usize) -> usize {
        self.nv() + r * (self.ncol - 1) + c
    }
    /// Loop (face) index of the loop with top-left pixel `(i,j)`.
    #[inline]
    pub fn loop_index(&self, i: usize, j: usize) -> usize {
        i * (self.ncol - 1) + j
    }
    /// Direction and top-left pixel `(r, c)` of edge `e`.
    #[inline]
    pub fn edge_rc(&self, e: usize) -> (EdgeDir, usize, usize) {
        let nv = self.nv();
        if e < nv {
            (EdgeDir::Vertical, e / self.ncol, e % self.ncol)
        } else {
            let h = e - nv;
            let w = self.ncol - 1;
            (EdgeDir::Horizontal, h / w, h % w)
        }
    }
    /// Pixel endpoints `(p, q)` of edge `e`; the gradient is `psi[q] - psi[p]`.
    #[inline]
    pub fn edge_endpoints(&self, e: usize) -> (usize, usize) {
        match self.edge_rc(e) {
            (EdgeDir::Vertical, r, c) => (self.pix(r, c), self.pix(r + 1, c)),
            (EdgeDir::Horizontal, r, c) => (self.pix(r, c), self.pix(r, c + 1)),
        }
    }
    /// Dual arc `(from, to)` of edge `e`. Flow `k_e` leaves `from` and enters `to`.
    ///
    /// * Horizontal edge (r,c): from loop (r,c) [ground if r == nrow-1]
    ///   to loop (r-1,c) [ground if r == 0].
    /// * Vertical edge (r,c): from loop (r,c-1) [ground if c == 0]
    ///   to loop (r,c) [ground if c == ncol-1].
    ///
    /// When `nrow == 1` or `ncol == 1` there are no loops and every arc is a
    /// ground self-loop.
    #[inline]
    pub fn edge_dual(&self, e: usize) -> (usize, usize) {
        let g = self.ground();
        match self.edge_rc(e) {
            (EdgeDir::Horizontal, r, c) => {
                let from = if r + 1 >= self.nrow { g } else { self.loop_index(r, c) };
                let to = if r == 0 { g } else { self.loop_index(r - 1, c) };
                (from, to)
            }
            (EdgeDir::Vertical, r, c) => {
                let from = if c == 0 { g } else { self.loop_index(r, c - 1) };
                let to = if c + 1 >= self.ncol { g } else { self.loop_index(r, c) };
                (from, to)
            }
        }
    }
}

/// Wrapped gradients `Delta_e = wrap(psi[q] - psi[p])` for every edge, in edge order.
pub fn wrapped_gradients(dims: Dims, psi: &[f32]) -> Vec<f64> {
    assert_eq!(psi.len(), dims.npix(), "psi length");
    (0..dims.n_edges())
        .map(|e| {
            let (p, q) = dims.edge_endpoints(e);
            wrap(psi[q] as f64 - psi[p] as f64)
        })
        .collect()
}

/// Wrap counts `c_e = wrap_count(psi[q] - psi[p])` for every edge, in edge order.
pub fn wrap_counts(dims: Dims, psi: &[f32]) -> Vec<i32> {
    assert_eq!(psi.len(), dims.npix(), "psi length");
    (0..dims.n_edges())
        .map(|e| {
            let (p, q) = dims.edge_endpoints(e);
            wrap_count(psi[q] as f64 - psi[p] as f64) as i32
        })
        .collect()
}

/// `phi = (psi as f64 + TAU * n) as f32` per pixel.
pub fn unwrapped_from_n(psi: &[f32], n: &[i32]) -> Vec<f32> {
    assert_eq!(psi.len(), n.len());
    psi.iter()
        .zip(n)
        .map(|(&p, &m)| (p as f64 + TAU * m as f64) as f32)
        .collect()
}

/// Minimum connected-component size from snaphu-py's `min_conncomp_frac`:
/// `max(1, ceil(frac * npix))`.
pub fn min_component_size(frac: f64, npix: usize) -> usize {
    let s = (frac * npix as f64).ceil();
    if s.is_finite() && s >= 1.0 {
        s as usize
    } else {
        1
    }
}

// ---------------------------------------------------------------------------
// Generic networks and costs
// ---------------------------------------------------------------------------

/// A directed network with integer flow bounds and node supplies.
///
/// Flow conservation: for every node `v`,
/// `sum(flow on arcs with tail v) - sum(flow on arcs with head v) == supply[v]`.
/// Self-loops (`tail == head`) are allowed and do not affect conservation.
/// Parallel arcs are allowed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Network {
    pub n_nodes: usize,
    pub tail: Vec<u32>,
    pub head: Vec<u32>,
    pub lower: Vec<i32>,
    pub upper: Vec<i32>,
    pub supply: Vec<i64>,
}

impl Network {
    #[inline]
    pub fn n_arcs(&self) -> usize {
        self.tail.len()
    }

    /// Structural validation: lengths, node indices, `lower <= upper`, supplies sum to 0.
    pub fn validate(&self) -> Result<(), McfError> {
        let m = self.tail.len();
        if self.head.len() != m || self.lower.len() != m || self.upper.len() != m {
            return Err(McfError::InvalidInput("arc array lengths differ".into()));
        }
        if self.supply.len() != self.n_nodes {
            return Err(McfError::InvalidInput("supply length != n_nodes".into()));
        }
        for a in 0..m {
            if self.tail[a] as usize >= self.n_nodes || self.head[a] as usize >= self.n_nodes {
                return Err(McfError::InvalidInput(format!("arc {a} endpoint out of range")));
            }
            if self.lower[a] > self.upper[a] {
                return Err(McfError::InvalidInput(format!("arc {a} lower > upper")));
            }
        }
        if self.supply.iter().sum::<i64>() != 0 {
            return Err(McfError::InvalidInput("supplies do not sum to zero".into()));
        }
        Ok(())
    }

    /// Divergence `out - in` of `flow` at every node.
    pub fn divergence(&self, flow: &[i32]) -> Vec<i64> {
        assert_eq!(flow.len(), self.n_arcs());
        let mut d = vec![0i64; self.n_nodes];
        for a in 0..self.n_arcs() {
            d[self.tail[a] as usize] += flow[a] as i64;
            d[self.head[a] as usize] -= flow[a] as i64;
        }
        d
    }

    /// True iff `flow` respects bounds and conservation.
    pub fn is_feasible(&self, flow: &[i32]) -> bool {
        flow.len() == self.n_arcs()
            && (0..self.n_arcs()).all(|a| self.lower[a] <= flow[a] && flow[a] <= self.upper[a])
            && self.divergence(flow) == self.supply
    }
}

/// Integer, separable arc cost: `cost(a, x)` for flow `x` on arc `a`.
///
/// Callers only evaluate `x` inside the arc's bounds. "Convex" means
/// `cost(a,x+1) - cost(a,x)` is nondecreasing in `x` over the bounds.
pub trait ArcCost: Sync {
    fn cost(&self, arc: usize, x: i32) -> i64;
}

/// Total cost `sum_a cost(a, flow[a])`.
pub fn total_cost(cost: &dyn ArcCost, flow: &[i32]) -> i64 {
    flow.iter().enumerate().map(|(a, &x)| cost.cost(a, x)).sum()
}

/// Explicit per-arc cost tables (used by the MCF judge instances).
/// Arc `a` stores `cost(a, x)` for `x = base[a] .. base[a] + len_a - 1` at
/// `values[offset[a] .. offset[a+1]]`. Evaluating outside the table panics.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TableCost {
    pub base: Vec<i32>,
    pub offset: Vec<usize>,
    pub values: Vec<i64>,
}

impl ArcCost for TableCost {
    #[inline]
    fn cost(&self, arc: usize, x: i32) -> i64 {
        let i = (x - self.base[arc]) as usize;
        let lo = self.offset[arc];
        let hi = self.offset[arc + 1];
        assert!(lo + i < hi, "TableCost: flow {x} outside table of arc {arc}");
        self.values[lo + i]
    }
}

/// Quantized statistical costs on grid edges.
///
/// `table[e * (2*kt+1) + (k + kt) as usize]` is the cost of `k_e = k` for
/// `|k| <= kt`. Outside the table the cost is extended linearly (fixed rule):
/// for `k > kt`: `c(kt) + (k - kt) * max(c(kt) - c(kt-1), MIN_TAIL_SLOPE)`;
/// for `k < -kt`: `c(-kt) + (-kt - k) * max(c(-kt) - c(-kt+1), MIN_TAIL_SLOPE)`.
/// The extension preserves convexity of convex tables. Arc index == edge index.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GridCosts {
    pub dims: Dims,
    pub kt: i32,
    pub table: Vec<i32>,
}

impl Default for Dims {
    fn default() -> Self {
        Dims { nrow: 0, ncol: 0 }
    }
}

impl GridCosts {
    #[inline]
    pub fn width(&self) -> usize {
        (2 * self.kt + 1) as usize
    }
    /// Table entry for `|k| <= kt`.
    #[inline]
    pub fn entry(&self, e: usize, k: i32) -> i64 {
        self.table[e * self.width() + (k + self.kt) as usize] as i64
    }
    /// Table row of edge `e`, index `k + kt`.
    #[inline]
    pub fn row(&self, e: usize) -> &[i32] {
        let w = self.width();
        &self.table[e * w..(e + 1) * w]
    }
}

impl ArcCost for GridCosts {
    #[inline]
    fn cost(&self, e: usize, k: i32) -> i64 {
        let kt = self.kt;
        if k > kt {
            let c = self.entry(e, kt);
            let slope = if kt > -kt { (c - self.entry(e, kt - 1)).max(MIN_TAIL_SLOPE) } else { MIN_TAIL_SLOPE };
            c.saturating_add(slope.saturating_mul((k - kt) as i64))
        } else if k < -kt {
            let c = self.entry(e, -kt);
            let slope = if kt > -kt { (c - self.entry(e, -kt + 1)).max(MIN_TAIL_SLOPE) } else { MIN_TAIL_SLOPE };
            c.saturating_add(slope.saturating_mul((-kt - k) as i64))
        } else {
            self.entry(e, k)
        }
    }
}

// ---------------------------------------------------------------------------
// Parameters, results, errors
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CostMode {
    Defo,
    Smooth,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InitMethod {
    Mst,
    Mcf,
}

/// Parameters of the statistical cost model (DESIGN.md section 5).
#[derive(Clone, Debug, PartialEq)]
pub struct CostParams {
    pub mode: CostMode,
    /// Effective number of looks L (>= 1).
    pub nlooks: f64,
    /// Window (rows, cols) for the mean phase gradient; half-width = w / 2.
    pub phase_grad_window: (usize, usize),
    /// Signal variance added to each edge (rad^2).
    pub sigma2_signal: f64,
    /// Discontinuity probability (defo mode only).
    pub p_discontinuity: f64,
}

impl CostParams {
    pub fn new(mode: CostMode, nlooks: f64) -> Self {
        CostParams {
            mode,
            nlooks,
            phase_grad_window: (7, 7),
            sigma2_signal: match mode {
                CostMode::Smooth => SIGMA2_SIGNAL_SMOOTH,
                CostMode::Defo => SIGMA2_SIGNAL_DEFO,
            },
            p_discontinuity: P_DISCONTINUITY_DEFO,
        }
    }
}

/// Options for nonconvex local improvement.
#[derive(Clone, Debug, PartialEq)]
pub struct ImproveOptions {
    /// Upper bound on full improvement passes (0 = no limit).
    pub max_passes: usize,
    /// Wall-clock limit in seconds (0.0 = no limit). When it expires the
    /// current (feasible, non-worse) solution is returned.
    pub time_limit_s: f64,
}

impl Default for ImproveOptions {
    fn default() -> Self {
        ImproveOptions { max_passes: 0, time_limit_s: 0.0 }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ImproveStats {
    pub initial_cost: i64,
    pub final_cost: i64,
    /// Number of improving moves (cycles / region shifts) applied.
    pub moves: u64,
    pub passes: u64,
}

/// All user-facing unwrapping parameters (mirrors snaphu-py's `unwrap`).
#[derive(Clone, Debug, PartialEq)]
pub struct UnwrapParams {
    pub nlooks: f64,
    pub cost: CostMode,
    pub init: InitMethod,
    pub min_conncomp_frac: f64,
    pub phase_grad_window: (usize, usize),
    pub ntiles: (usize, usize),
    pub tile_overlap: (usize, usize),
    /// Worker threads for tiles; 0 = all available.
    pub nproc: usize,
    /// Reliable-region threshold for tiling, cost units.
    pub tile_cost_thresh: i64,
    pub min_region_size: usize,
    pub single_tile_reoptimize: bool,
    pub regrow_conncomps: bool,
    /// Connected-component reliability threshold, cost units.
    pub conncomp_cost_thresh: i64,
    pub sigma2_signal: Option<f64>,
    pub p_discontinuity: Option<f64>,
    pub improve: ImproveOptions,
}

impl UnwrapParams {
    pub fn new(nlooks: f64, cost: CostMode, init: InitMethod) -> Self {
        UnwrapParams {
            nlooks,
            cost,
            init,
            min_conncomp_frac: 0.01,
            phase_grad_window: (7, 7),
            ntiles: (1, 1),
            tile_overlap: (0, 0),
            nproc: 1,
            tile_cost_thresh: 500,
            min_region_size: 100,
            single_tile_reoptimize: true,
            regrow_conncomps: true,
            conncomp_cost_thresh: CONNCOMP_COST_THRESH,
            sigma2_signal: None,
            p_discontinuity: None,
            improve: ImproveOptions::default(),
        }
    }

    pub fn cost_params(&self) -> CostParams {
        let mut cp = CostParams::new(self.cost, self.nlooks);
        cp.phase_grad_window = self.phase_grad_window;
        if let Some(s) = self.sigma2_signal {
            cp.sigma2_signal = s;
        }
        if let Some(p) = self.p_discontinuity {
            cp.p_discontinuity = p;
        }
        cp
    }

    pub fn validate(&self) -> Result<(), UnwrapError> {
        if !(self.nlooks >= 1.0) || !self.nlooks.is_finite() {
            return Err(UnwrapError::InvalidInput("nlooks must be >= 1".into()));
        }
        if self.ntiles.0 < 1 || self.ntiles.1 < 1 {
            return Err(UnwrapError::InvalidInput("ntiles must be >= 1".into()));
        }
        if !(0.0..=1.0).contains(&self.min_conncomp_frac) {
            return Err(UnwrapError::InvalidInput("min_conncomp_frac must be in [0,1]".into()));
        }
        if self.phase_grad_window.0 < 1 || self.phase_grad_window.1 < 1 {
            return Err(UnwrapError::InvalidInput("phase_grad_window must be >= 1".into()));
        }
        Ok(())
    }
}

/// Inputs to the unwrapper. `psi` is wrapped phase (f32, radians), `corr` the
/// sample coherence (NaN treated as 0, clipped to [0,1] by the cost model),
/// `mask` optional valid-pixel mask (0 = invalid).
#[derive(Clone, Copy, Debug)]
pub struct UnwrapInput<'a> {
    pub dims: Dims,
    pub psi: &'a [f32],
    pub corr: &'a [f32],
    pub mask: Option<&'a [u8]>,
}

impl<'a> UnwrapInput<'a> {
    pub fn validate(&self) -> Result<(), UnwrapError> {
        let n = self.dims.npix();
        if self.dims.nrow == 0 || self.dims.ncol == 0 {
            return Err(UnwrapError::InvalidInput("empty image".into()));
        }
        if self.psi.len() != n || self.corr.len() != n {
            return Err(UnwrapError::InvalidInput("array length != nrow*ncol".into()));
        }
        if let Some(m) = self.mask {
            if m.len() != n {
                return Err(UnwrapError::InvalidInput("mask length != nrow*ncol".into()));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct UnwrapStats {
    pub n_residues: u64,
    pub init_cost: i64,
    pub final_cost: i64,
    pub n_components: u32,
    pub n_tiles: u32,
    pub seconds_costs: f64,
    pub seconds_init: f64,
    pub seconds_improve: f64,
    pub seconds_total: f64,
}

/// Output: integer cycle field `n`, unwrapped phase `unw = psi + TAU*n`
/// (0 where masked), connected components (0 = none, 1.. by decreasing size).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct UnwrapOutput {
    pub n: Vec<i32>,
    pub unw: Vec<f32>,
    pub conncomp: Vec<u32>,
    pub stats: UnwrapStats,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum McfError {
    InvalidInput(String),
    Infeasible,
    Other(String),
}

impl std::fmt::Display for McfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            McfError::InvalidInput(s) => write!(f, "invalid input: {s}"),
            McfError::Infeasible => write!(f, "infeasible network"),
            McfError::Other(s) => write!(f, "{s}"),
        }
    }
}
impl std::error::Error for McfError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnwrapError {
    InvalidInput(String),
    Solver(McfError),
    NotCurlFree,
    Other(String),
}

impl From<McfError> for UnwrapError {
    fn from(e: McfError) -> Self {
        UnwrapError::Solver(e)
    }
}

impl std::fmt::Display for UnwrapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnwrapError::InvalidInput(s) => write!(f, "invalid input: {s}"),
            UnwrapError::Solver(e) => write!(f, "solver: {e}"),
            UnwrapError::NotCurlFree => write!(f, "cycle field is not curl-free"),
            UnwrapError::Other(s) => write!(f, "{s}"),
        }
    }
}
impl std::error::Error for UnwrapError {}

/// Variance cap for a pixel's phase (uniform phase on [-pi, pi)).
pub const SIGMA2_MAX: f64 = PI * PI / 3.0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_range() {
        for &d in &[-7.0, -PI, -3.0, 0.0, 3.0, PI, 7.0, 2.0 * PI] {
            let w = wrap(d);
            assert!(w >= -PI && w < PI, "{d} -> {w}");
            assert!((d - w - TAU * wrap_count(d) as f64).abs() < 1e-12);
        }
    }

    #[test]
    fn dual_orientation_2x3() {
        let d = Dims::new(2, 3);
        assert_eq!(d.n_edges(), 3 + 4);
        assert_eq!(d.n_loops(), 2);
        let g = d.ground();
        // vertical edges (0,0),(0,1),(0,2)
        assert_eq!(d.edge_dual(0), (g, 0));
        assert_eq!(d.edge_dual(1), (0, 1));
        assert_eq!(d.edge_dual(2), (1, g));
        // horizontal (0,0),(0,1) top row: from loop to ground
        assert_eq!(d.edge_dual(3), (0, g));
        assert_eq!(d.edge_dual(4), (1, g));
        // horizontal (1,0),(1,1) bottom row: from ground to loop
        assert_eq!(d.edge_dual(5), (g, 0));
        assert_eq!(d.edge_dual(6), (g, 1));
    }

    #[test]
    fn grid_costs_extension() {
        let gc = GridCosts { dims: Dims::new(1, 2), kt: 1, table: vec![5, 0, 3] };
        assert_eq!(gc.cost(0, 0), 0);
        assert_eq!(gc.cost(0, 2), 3 + 100);
        assert_eq!(gc.cost(0, -3), 5 + 2 * 100);
    }
}
