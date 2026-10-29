//! Judge tests for jupwrap-conncomp. Goldens from judge/ref/flows.py
//! (judge/gen/gen_conncomp.py). Exact label equality.

use jupwrap_conncomp::label_components;
use jupwrap_core::Dims;
use jupwrap_testkit as tk;

#[test]
fn labels_match_reference() {
    let dir = tk::golden_dir("conncomp");
    let s = tk::meta_file(&dir.join("summary.json"));
    let cs = tk::cases(&dir);
    assert_eq!(cs.len() as u64, s["n_cases"].as_u64().unwrap());
    for c in cs {
        let m = tk::meta(&c);
        let costs = tk::grid_costs(&c, "table");
        let k = tk::npy(&c, "k").i32();
        let mask = if m["masked"].as_bool().unwrap() { Some(tk::npy(&c, "mask").u8()) } else { None };
        let (lab, n) = label_components(&costs, &k, mask.as_deref(), m["thresh"].as_i64().unwrap(), m["min_size"].as_u64().unwrap() as usize);
        assert_eq!(n as u64, m["n_components"].as_u64().unwrap(), "{}: component count", c.display());
        let want = tk::npy(&c, "labels").u32();
        let bad = lab.iter().zip(&want).filter(|(a, b)| a != b).count();
        assert_eq!(bad, 0, "{}: {bad} labels differ", c.display());
    }
}

#[test]
fn fully_masked_has_no_components() {
    let c = tk::golden_dir("conncomp").join("case_02");
    let costs = tk::grid_costs(&c, "table");
    let k = tk::npy(&c, "k").i32();
    let mask = vec![0u8; costs.dims.npix()];
    let (lab, n) = label_components(&costs, &k, Some(&mask), 0, 1);
    assert_eq!(n, 0);
    assert!(lab.iter().all(|&v| v == 0));
}

#[test]
fn large_grid_is_fast() {
    let (costs, _res) = tk::quadgrid(4096, 4096, 31, 2, 0);
    let d: Dims = costs.dims;
    let k = vec![0i32; d.n_edges()];
    let t0 = std::time::Instant::now();
    let (lab, n) = label_components(&costs, &k, None, 500, 100);
    let dt = tk::secs(t0);
    eprintln!("conncomp 4096^2: {dt:.2} s, {n} components");
    assert_eq!(lab.len(), d.npix());
    assert!(n >= 1);
    assert!(dt < 20.0, "too slow: {dt:.1} s");
}
