//! jupwrap-testkit: test support for the judge tests.
//! Minimal `.npy` reader (C order, little endian) and golden-file helpers.

use jupwrap_core::{Dims, GridCosts, Network, TableCost};
use std::path::{Path, PathBuf};

/// `<workspace>/judge/golden/<sub>`.
pub fn golden_dir(sub: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../judge/golden").join(sub)
}

/// A loaded `.npy` array: dtype descr (e.g. "<f4"), shape, raw bytes.
pub struct Npy {
    pub descr: String,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

pub fn read_npy(path: &Path) -> Npy {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert!(b.len() >= 10 && &b[..6] == b"\x93NUMPY", "{}: not an npy file", path.display());
    let major = b[6];
    let (hlen, hstart) = if major == 1 {
        (u16::from_le_bytes([b[8], b[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12)
    };
    let header = std::str::from_utf8(&b[hstart..hstart + hlen]).expect("npy header utf8");
    let dfield = field(header, "descr");
    let q = dfield.chars().next().expect("descr quote");
    let descr = dfield[1..1 + dfield[1..].find(q).expect("descr end")].to_string();
    let fortran = field(header, "fortran_order");
    assert!(fortran.starts_with("False"), "fortran order not supported");
    let shape_s = field(header, "shape");
    let inner = shape_s.trim().trim_start_matches('(');
    let inner = &inner[..inner.find(')').unwrap()];
    let shape: Vec<usize> = inner
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect();
    Npy { descr, shape, data: b[hstart + hlen..].to_vec() }
}

fn field<'a>(header: &'a str, key: &str) -> &'a str {
    let pat = format!("'{key}':");
    let i = header.find(&pat).unwrap_or_else(|| panic!("npy header missing {key}"));
    header[i + pat.len()..].trim_start()
}

impl Npy {
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn check(&self, d: &[&str]) {
        assert!(d.contains(&self.descr.as_str()), "dtype {} not in {:?}", self.descr, d);
    }
    pub fn f32(&self) -> Vec<f32> {
        self.check(&["<f4"]);
        self.data.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    }
    pub fn f64(&self) -> Vec<f64> {
        self.check(&["<f8"]);
        self.data.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
    }
    pub fn i64(&self) -> Vec<i64> {
        match self.descr.as_str() {
            "<i8" => self.data.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect(),
            "<i4" => self.data.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64).collect(),
            "<u4" => self.data.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap()) as i64).collect(),
            "<i2" => self.data.chunks_exact(2).map(|c| i16::from_le_bytes(c.try_into().unwrap()) as i64).collect(),
            "|i1" => self.data.iter().map(|&v| v as i8 as i64).collect(),
            "|u1" | "|b1" => self.data.iter().map(|&v| v as i64).collect(),
            d => panic!("unsupported integer dtype {d}"),
        }
    }
    pub fn i32(&self) -> Vec<i32> {
        self.i64().into_iter().map(|v| v as i32).collect()
    }
    pub fn i8(&self) -> Vec<i8> {
        self.i64().into_iter().map(|v| v as i8).collect()
    }
    pub fn u8(&self) -> Vec<u8> {
        self.check(&["|u1", "|b1"]);
        self.data.clone()
    }
    pub fn u32(&self) -> Vec<u32> {
        self.i64().into_iter().map(|v| v as u32).collect()
    }
}

/// Read `dir/name.npy`.
pub fn npy(dir: &Path, name: &str) -> Npy {
    read_npy(&dir.join(format!("{name}.npy")))
}

/// Read `dir/meta.json`.
pub fn meta(dir: &Path) -> serde_json::Value {
    let p = dir.join("meta.json");
    let s = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    serde_json::from_str(&s).expect("meta.json")
}

/// Sorted case directories `dir/case_*`.
pub fn cases(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.is_dir() && p.file_name().unwrap().to_string_lossy().starts_with("case_"))
        .collect();
    v.sort();
    v
}

/// Dims from meta.json fields "nrow", "ncol".
pub fn dims_of(meta: &serde_json::Value) -> Dims {
    Dims::new(meta["nrow"].as_u64().unwrap() as usize, meta["ncol"].as_u64().unwrap() as usize)
}

/// GridCosts from `table.npy` (shape [n_edges, 2*kt+1], int32) and meta dims.
pub fn grid_costs(dir: &Path, table_name: &str) -> GridCosts {
    let m = meta(dir);
    let t = npy(dir, table_name);
    assert_eq!(t.shape.len(), 2);
    let kt = ((t.shape[1] - 1) / 2) as i32;
    GridCosts { dims: dims_of(&m), kt, table: t.i32() }
}

/// Generic network + table costs from a case directory written by
/// `judge/gen/gen_mcf.py`: tail, head, lower, upper (int32), supply (int64),
/// cost_offset (int64, n_arcs+1), cost_values (int64). Table of arc a covers
/// x = lower[a] ..= upper[a].
pub fn network_case(dir: &Path) -> (Network, TableCost) {
    let m = meta(dir);
    let tail: Vec<u32> = npy(dir, "tail").i64().into_iter().map(|v| v as u32).collect();
    let head: Vec<u32> = npy(dir, "head").i64().into_iter().map(|v| v as u32).collect();
    let lower = npy(dir, "lower").i32();
    let upper = npy(dir, "upper").i32();
    let supply = npy(dir, "supply").i64();
    let offset: Vec<usize> = npy(dir, "cost_offset").i64().into_iter().map(|v| v as usize).collect();
    let values = npy(dir, "cost_values").i64();
    let net = Network {
        n_nodes: m["n_nodes"].as_u64().unwrap() as usize,
        tail,
        head,
        lower: lower.clone(),
        upper,
        supply,
    };
    (net, TableCost { base: lower, offset, values })
}

/// Seconds since `t0` as f64.
pub fn secs(t0: std::time::Instant) -> f64 {
    t0.elapsed().as_secs_f64()
}

// ---------------------------------------------------------------------------
// Deterministic recipes shared bit-for-bit with judge/ref/recipe.py
// ---------------------------------------------------------------------------

/// splitmix64 generator.
pub struct SplitMix(pub u64);

impl SplitMix {
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

/// Recipe "quadgrid" (see judge/ref/recipe.py): residues per loop and a
/// GridCosts with half-width `b` and rows `a * (k - u)^2`.
pub fn quadgrid(nrow: usize, ncol: usize, seed: u64, b: i32, resid_permille: u64) -> (GridCosts, Vec<i8>) {
    let dims = Dims::new(nrow, ncol);
    let mut g = SplitMix(seed);
    let res: Vec<i8> = (0..dims.n_loops())
        .map(|_| {
            let r = g.next_u64();
            if r % 1000 >= resid_permille {
                0
            } else if (r >> 32) & 1 == 1 {
                1
            } else {
                -1
            }
        })
        .collect();
    let w = (2 * b + 1) as usize;
    let mut table = Vec::with_capacity(dims.n_edges() * w);
    for _ in 0..dims.n_edges() {
        let r1 = g.next_u64();
        let r2 = g.next_u64();
        let a = 1 + (r1 % 2000) as i64;
        let u = match r2 % 10 {
            0 => -1i64,
            1 => 1,
            _ => 0,
        };
        for k in -b..=b {
            let d = k as i64 - u;
            table.push((a * d * d) as i32);
        }
    }
    (GridCosts { dims, kt: b, table }, res)
}

/// Grid dual network (core orientation) with bounds `[lo, hi]` and supplies from residues.
pub fn grid_network(dims: Dims, res: &[i8], lo: i32, hi: i32) -> Network {
    let e = dims.n_edges();
    let mut tail = Vec::with_capacity(e);
    let mut head = Vec::with_capacity(e);
    for i in 0..e {
        let (f, t) = dims.edge_dual(i);
        tail.push(f as u32);
        head.push(t as u32);
    }
    let mut supply: Vec<i64> = res.iter().map(|&r| -(r as i64)).collect();
    supply.push(res.iter().map(|&r| r as i64).sum());
    Network { n_nodes: dims.n_dual_nodes(), tail, head, lower: vec![lo; e], upper: vec![hi; e], supply }
}

/// Read any JSON file.
pub fn meta_file(path: &Path) -> serde_json::Value {
    let s = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&s).expect("json")
}
