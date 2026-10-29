//! Judge tests for jupwrap-mcf. Optimal costs from OR-tools SimpleMinCostFlow,
//! cross-checked with networkx.network_simplex (judge/gen/gen_mcf.py).
//! Flows may differ from the reference when the optimum is not unique, so
//! only feasibility and the exact optimal cost are checked.

use jupwrap_core::{total_cost, McfError, Network, TableCost};
use jupwrap_mcf::solve_convex;
use jupwrap_testkit as tk;

fn run_kind(kinds: &[&str]) -> usize {
    let dir = tk::golden_dir("mcf");
    let summary = tk::meta_file(&dir.join("summary.json"));
    let all = tk::cases(&dir);
    assert_eq!(all.len() as u64, summary["n_cases"].as_u64().unwrap(), "golden case count");
    let mut n = 0;
    for c in all {
        let m = tk::meta(&c);
        let kind = m["kind"].as_str().unwrap();
        if !kinds.contains(&kind) {
            continue;
        }
        n += 1;
        let (net, cost) = tk::network_case(&c);
        let res = solve_convex(&net, &cost);
        let name = c.file_name().unwrap().to_string_lossy().to_string();
        match m["status"].as_str().unwrap() {
            "optimal" => {
                let flow = res.unwrap_or_else(|e| panic!("{name} ({kind}): expected optimal, got error {e}"));
                assert!(net.is_feasible(&flow), "{name} ({kind}): returned flow infeasible");
                let got = total_cost(&cost, &flow);
                let want = m["opt_cost"].as_i64().unwrap();
                assert_eq!(got, want, "{name} ({kind}): cost {got} != optimum {want}");
            }
            "infeasible" => {
                assert_eq!(res, Err(McfError::Infeasible), "{name} ({kind}): expected Infeasible");
            }
            s => panic!("unknown status {s}"),
        }
    }
    n
}

#[test]
fn random_convex_instances() {
    assert_eq!(run_kind(&["random"]), 8);
}

#[test]
fn linear_cost_instances() {
    assert_eq!(run_kind(&["linear"]), 3);
}

#[test]
fn zero_supply_circulation_instances() {
    assert_eq!(run_kind(&["circulation"]), 3);
}

#[test]
fn selfloops_parallel_arcs_and_ties() {
    assert_eq!(run_kind(&["selfloop_parallel", "ties"]), 3);
}

#[test]
fn infeasible_instances_return_error() {
    assert_eq!(run_kind(&["infeasible"]), 3);
}

#[test]
fn grid_dual_statistical_cost_instances() {
    assert_eq!(run_kind(&["grid_stat"]), 7);
}

#[test]
fn invalid_inputs_return_error() {
    let cost = TableCost { base: vec![0], offset: vec![0, 2], values: vec![0, 1] };
    let mut net = Network { n_nodes: 2, tail: vec![0], head: vec![1], lower: vec![0], upper: vec![1], supply: vec![1, 0] };
    assert!(matches!(solve_convex(&net, &cost), Err(McfError::InvalidInput(_))), "supplies not summing to zero");
    net.supply = vec![1, -1];
    net.lower = vec![2];
    net.upper = vec![1];
    let cost2 = TableCost { base: vec![2], offset: vec![0, 1], values: vec![0] };
    assert!(matches!(solve_convex(&net, &cost2), Err(McfError::InvalidInput(_))), "lower > upper");
    net.lower = vec![0];
    net.supply = vec![1, -1, 0];
    assert!(matches!(solve_convex(&net, &cost), Err(McfError::InvalidInput(_))), "supply length");
    net.supply = vec![1, -1];
    net.head = vec![5];
    assert!(matches!(solve_convex(&net, &cost), Err(McfError::InvalidInput(_))), "node index out of range");
}

#[test]
fn trivial_networks() {
    let empty = Network { n_nodes: 1, supply: vec![0], ..Default::default() };
    let none = TableCost { base: vec![], offset: vec![0], values: vec![] };
    assert_eq!(solve_convex(&empty, &none), Ok(vec![]));
    // single arc, forced flow 3 with lower bound 3
    let net = Network { n_nodes: 2, tail: vec![0], head: vec![1], lower: vec![3], upper: vec![5], supply: vec![3, -3] };
    let c = TableCost { base: vec![3], offset: vec![0, 3], values: vec![9, 1, 0] };
    assert_eq!(solve_convex(&net, &c), Ok(vec![3]));
    // two parallel arcs with opposite orientation forming a profitable circulation
    let net = Network { n_nodes: 2, tail: vec![0, 1], head: vec![1, 0], lower: vec![-2, -2], upper: vec![2, 2], supply: vec![0, 0] };
    let c = TableCost { base: vec![-2, -2], offset: vec![0, 5, 10], values: vec![4, 1, 0, -1, -2, 0, 0, 0, 0, 0] };
    let f = solve_convex(&net, &c).unwrap();
    assert!(net.is_feasible(&f));
    assert_eq!(total_cost(&c, &f), -2);
}

fn recipe(i: usize) {
    let dir = tk::golden_dir("mcf");
    let recs = tk::meta_file(&dir.join("recipes.json"));
    let r = &recs[i];
    let (nrow, ncol) = (r["nrow"].as_u64().unwrap() as usize, r["ncol"].as_u64().unwrap() as usize);
    let b = r["B"].as_i64().unwrap() as i32;
    let (costs, res) = tk::quadgrid(nrow, ncol, r["seed"].as_u64().unwrap(), b, r["resid_permille"].as_u64().unwrap());
    let net = tk::grid_network(costs.dims, &res, -b, b);
    let t0 = std::time::Instant::now();
    let flow = solve_convex(&net, &costs).expect("recipe instance is feasible");
    let dt = tk::secs(t0);
    assert!(net.is_feasible(&flow), "infeasible flow");
    let got = total_cost(&costs, &flow);
    assert_eq!(got, r["opt_cost"].as_i64().unwrap(), "recipe {i}: cost != optimum");
    let limit = r["time_limit_s"].as_f64().unwrap();
    eprintln!("recipe {i} {nrow}x{ncol}: {dt:.2} s (limit {limit} s, OR-tools {} s)", r["ortools_seconds"]);
    assert!(dt <= limit, "recipe {i}: {dt:.1} s > limit {limit} s");
}

#[test]
fn recipe_grid_128() {
    recipe(0);
}

#[test]
fn recipe_grid_512() {
    recipe(1);
}

#[test]
fn recipe_grid_1024() {
    recipe(2);
}

#[test]
fn recipe_grid_300x2000() {
    recipe(3);
}
