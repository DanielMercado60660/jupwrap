//! Judge tests for jupwrap-grid. Goldens from judge/ref/conventions.py
//! (judge/gen/gen_grid.py). All checks are exact (integer).

use jupwrap_core::{Dims, UnwrapError, K_BIG};
use jupwrap_grid::{dual_network, integrate, k_from_n, residues};
use jupwrap_testkit as tk;
use std::path::PathBuf;

fn all_cases() -> Vec<PathBuf> {
    let dir = tk::golden_dir("grid");
    let s = tk::meta_file(&dir.join("summary.json"));
    let c = tk::cases(&dir);
    assert_eq!(c.len() as u64, s["n_cases"].as_u64().unwrap());
    c
}

#[test]
fn residues_match_reference() {
    for c in all_cases() {
        let d = tk::dims_of(&tk::meta(&c));
        let psi = tk::npy(&c, "psi").f32();
        assert_eq!(residues(d, &psi), tk::npy(&c, "residues").i8(), "{}", c.display());
    }
}

#[test]
fn dual_network_matches_reference() {
    for c in all_cases() {
        let d = tk::dims_of(&tk::meta(&c));
        let res = tk::npy(&c, "residues").i8();
        let net = dual_network(d, &res);
        assert_eq!(net.n_nodes, d.n_dual_nodes(), "{}", c.display());
        let tail: Vec<u32> = tk::npy(&c, "tail").i64().into_iter().map(|v| v as u32).collect();
        let head: Vec<u32> = tk::npy(&c, "head").i64().into_iter().map(|v| v as u32).collect();
        assert_eq!(net.tail, tail, "tail {}", c.display());
        assert_eq!(net.head, head, "head {}", c.display());
        assert_eq!(net.supply, tk::npy(&c, "supply").i64(), "supply {}", c.display());
        assert!(net.lower.iter().all(|&v| v == -K_BIG) && net.lower.len() == d.n_edges());
        assert!(net.upper.iter().all(|&v| v == K_BIG) && net.upper.len() == d.n_edges());
        net.validate().unwrap();
    }
}

#[test]
fn integrate_matches_reference() {
    for c in all_cases() {
        let d = tk::dims_of(&tk::meta(&c));
        let psi = tk::npy(&c, "psi").f32();
        let k = tk::npy(&c, "k").i32();
        let n = integrate(d, &psi, &k).unwrap_or_else(|e| panic!("{}: {e}", c.display()));
        assert_eq!(n, tk::npy(&c, "n").i32(), "{}", c.display());
    }
}

#[test]
fn integrate_rejects_inconsistent_k() {
    let mut checked = 0;
    for c in all_cases() {
        let m = tk::meta(&c);
        if !m["has_bad"].as_bool().unwrap() {
            continue;
        }
        let d = tk::dims_of(&m);
        let psi = tk::npy(&c, "psi").f32();
        let kbad = tk::npy(&c, "kbad").i32();
        assert_eq!(integrate(d, &psi, &kbad), Err(UnwrapError::NotCurlFree), "{}", c.display());
        checked += 1;
    }
    assert_eq!(checked, 6);
}

#[test]
fn integrate_rejects_bad_lengths() {
    let d = Dims::new(3, 3);
    let psi = vec![0.0f32; 9];
    assert!(matches!(integrate(d, &psi, &[0; 5]), Err(UnwrapError::InvalidInput(_))));
    assert!(matches!(integrate(d, &psi[..8], &[0; 12]), Err(UnwrapError::InvalidInput(_))));
}

#[test]
fn k_from_n_matches_reference() {
    for c in all_cases() {
        let d = tk::dims_of(&tk::meta(&c));
        let psi = tk::npy(&c, "psi").f32();
        let n = tk::npy(&c, "nrand").i32();
        assert_eq!(k_from_n(d, &psi, &n), tk::npy(&c, "kn").i32(), "{}", c.display());
    }
}

#[test]
fn worked_example_2x2() {
    // DESIGN.md section 3.4
    let d = Dims::new(2, 2);
    let psi = [0.0f32, 2.5, -2.5, 1.0];
    assert_eq!(residues(d, &psi), vec![1i8]);
    let net = dual_network(d, &[1]);
    assert_eq!(net.tail, vec![1, 0, 0, 1]);
    assert_eq!(net.head, vec![0, 1, 1, 0]);
    assert_eq!(net.supply, vec![-1, 1]);
    assert_eq!(integrate(d, &psi, &[0, 0, 0, 1]), Ok(vec![0, 0, 0, 0]));
    assert_eq!(integrate(d, &psi, &[0, -1, 0, 0]), Ok(vec![0, 0, 0, -1]));
    assert_eq!(integrate(d, &psi, &[0, 0, 0, 0]), Err(UnwrapError::NotCurlFree));
}

#[test]
fn roundtrip_large_2048() {
    let d = Dims::new(2048, 2048);
    let mut g = tk::SplitMix(99);
    let psi: Vec<f32> = (0..d.npix())
        .map(|_| ((g.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * std::f64::consts::TAU - std::f64::consts::PI) as f32)
        .collect();
    let n: Vec<i32> = (0..d.npix()).map(|_| (g.next_u64() % 41) as i32 - 20).collect();
    let t0 = std::time::Instant::now();
    let res = residues(d, &psi);
    let k = k_from_n(d, &psi, &n);
    let net = dual_network(d, &res);
    let n2 = integrate(d, &psi, &k).expect("consistent");
    let dt = tk::secs(t0);
    assert!(net.is_feasible(&k), "k from n must satisfy conservation");
    assert!(n2.iter().zip(&n).all(|(a, b)| *a == b - n[0]));
    eprintln!("roundtrip 2048^2: {dt:.2} s");
    assert!(dt < 20.0, "too slow: {dt:.1} s");
}
