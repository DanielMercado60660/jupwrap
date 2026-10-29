//! jupwrap-cli: harness binary used by the end-to-end tests.
//!
//! jupwrap-cli --igram F.c8 | --phase F.f4   --corr F.f4 --nrow R --ncol C
//!             --nlooks L [--cost smooth|defo] [--init mcf|mst] [--mask F.u1]
//!             [--ntiles R C] [--tile-overlap R C] [--nproc N]
//!             [--tile-cost-thresh T] [--min-region-size S]
//!             [--single-tile-reoptimize 0|1] [--regrow-conncomps 0|1]
//!             [--min-conncomp-frac F] [--phase-grad-window R C]
//!             [--conncomp-cost-thresh T]
//!             --unw OUT.f4 --conncomp OUT.u4 [--stats OUT.json]
//!
//! Raw little-endian row-major files: .c8 = complex64 (re, im interleaved),
//! .f4 = float32, .u1 = uint8, .u4 = uint32. Exit code 0 on success, 2 on
//! usage errors, 1 on unwrapping errors.

use jupwrap::{CostMode, Dims, InitMethod, UnwrapParams};
use std::collections::HashMap;
use std::process::exit;

fn die(msg: &str) -> ! {
    eprintln!("jupwrap-cli: {msg}");
    exit(2)
}

fn read_f32(path: &str, n: usize) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| die(&format!("read {path}: {e}")));
    if b.len() != 4 * n {
        die(&format!("{path}: expected {} bytes, got {}", 4 * n, b.len()));
    }
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn write_bytes(path: &str, b: &[u8]) {
    std::fs::write(path, b).unwrap_or_else(|e| die(&format!("write {path}: {e}")));
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let two = ["--ntiles", "--tile-overlap", "--phase-grad-window"];
    let mut kv: HashMap<String, Vec<String>> = HashMap::new();
    let mut i = 0;
    while i < args.len() {
        let k = args[i].clone();
        if !k.starts_with("--") {
            die(&format!("unexpected argument {k}"));
        }
        let n = if two.contains(&k.as_str()) { 2 } else { 1 };
        let vals: Vec<String> = args[i + 1..(i + 1 + n).min(args.len())].to_vec();
        if vals.len() != n {
            die(&format!("missing value for {k}"));
        }
        kv.insert(k, vals);
        i += 1 + n;
    }
    let get = |k: &str| kv.get(k).map(|v| v[0].clone());
    let req = |k: &str| get(k).unwrap_or_else(|| die(&format!("{k} is required")));
    let num = |s: String, k: &str| -> f64 { s.parse::<f64>().unwrap_or_else(|_| die(&format!("bad number for {k}"))) };
    let int = |s: &str, k: &str| -> usize { s.parse::<usize>().unwrap_or_else(|_| die(&format!("bad integer for {k}"))) };
    let pair = |k: &str| kv.get(k).map(|v| (int(&v[0], k), int(&v[1], k)));

    let nrow = int(&req("--nrow"), "--nrow");
    let ncol = int(&req("--ncol"), "--ncol");
    let dims = Dims::new(nrow, ncol);
    let npix = dims.npix();
    let nlooks = num(req("--nlooks"), "--nlooks");
    let cost = match get("--cost").as_deref().unwrap_or("smooth") {
        "smooth" => CostMode::Smooth,
        "defo" => CostMode::Defo,
        other => die(&format!("unsupported cost mode {other} (smooth|defo)")),
    };
    let init = match get("--init").as_deref().unwrap_or("mcf") {
        "mcf" => InitMethod::Mcf,
        "mst" => InitMethod::Mst,
        other => die(&format!("unsupported init {other} (mcf|mst)")),
    };
    let mut p = UnwrapParams::new(nlooks, cost, init);
    if let Some(v) = pair("--ntiles") { p.ntiles = v; }
    if let Some(v) = pair("--tile-overlap") { p.tile_overlap = v; }
    if let Some(v) = pair("--phase-grad-window") { p.phase_grad_window = v; }
    if let Some(v) = get("--nproc") { p.nproc = int(&v, "--nproc"); }
    if let Some(v) = get("--tile-cost-thresh") { p.tile_cost_thresh = num(v, "--tile-cost-thresh") as i64; }
    if let Some(v) = get("--min-region-size") { p.min_region_size = int(&v, "--min-region-size"); }
    if let Some(v) = get("--single-tile-reoptimize") { p.single_tile_reoptimize = v != "0"; }
    if let Some(v) = get("--regrow-conncomps") { p.regrow_conncomps = v != "0"; }
    if let Some(v) = get("--min-conncomp-frac") { p.min_conncomp_frac = num(v, "--min-conncomp-frac"); }
    if let Some(v) = get("--conncomp-cost-thresh") { p.conncomp_cost_thresh = num(v, "--conncomp-cost-thresh") as i64; }

    let corr = read_f32(&req("--corr"), npix);
    let mask: Option<Vec<u8>> = get("--mask").map(|m| {
        let b = std::fs::read(&m).unwrap_or_else(|e| die(&format!("read {m}: {e}")));
        if b.len() != npix {
            die(&format!("{m}: expected {npix} bytes"));
        }
        b
    });
    let res = if let Some(f) = get("--igram") {
        let ig = read_f32(&f, 2 * npix);
        jupwrap::unwrap_complex(dims, &ig, &corr, mask.as_deref(), &p)
    } else {
        let psi = read_f32(&req("--phase"), npix);
        jupwrap::unwrap(&jupwrap::UnwrapInput { dims, psi: &psi, corr: &corr, mask: mask.as_deref() }, &p)
    };
    let out = match res {
        Ok(o) => o,
        Err(e) => {
            eprintln!("jupwrap-cli: unwrap failed: {e}");
            exit(1)
        }
    };
    let unw: Vec<u8> = out.unw.iter().flat_map(|v| v.to_le_bytes()).collect();
    let cc: Vec<u8> = out.conncomp.iter().flat_map(|v| v.to_le_bytes()).collect();
    write_bytes(&req("--unw"), &unw);
    write_bytes(&req("--conncomp"), &cc);
    if let Some(sp) = get("--stats") {
        let s = &out.stats;
        let json = format!(
            "{{\"n_residues\": {}, \"init_cost\": {}, \"final_cost\": {}, \"n_components\": {}, \"n_tiles\": {}, \"seconds_costs\": {}, \"seconds_init\": {}, \"seconds_improve\": {}, \"seconds_total\": {}}}\n",
            s.n_residues, s.init_cost, s.final_cost, s.n_components, s.n_tiles,
            s.seconds_costs, s.seconds_init, s.seconds_improve, s.seconds_total
        );
        write_bytes(&sp, json.as_bytes());
    }
}
