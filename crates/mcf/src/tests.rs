use super::*;

struct Quadratic {
    weight: Vec<i64>,
    center: Vec<i64>,
    slope: Vec<i64>,
}

impl ArcCost for Quadratic {
    fn cost(&self, a: usize, x: i32) -> i64 {
        let z = i64::from(x) - self.center[a];
        self.weight[a] * z * z + self.slope[a] * i64::from(x)
    }
}

#[test]
fn exhaustive_small_convex_networks() {
    // Independent enumeration also covers infeasibility, nonzero lower bounds,
    // disconnected components, loops, parallel arcs, and flat/linear costs.
    let mut state = 17u64;
    let mut random = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for case in 0..600 {
        let n = 1 + random() as usize % 4;
        let m = random() as usize % 7;
        let mut net = Network {
            n_nodes: n,
            supply: vec![0; n],
            ..Network::default()
        };
        let mut cost = Quadratic {
            weight: Vec::new(),
            center: Vec::new(),
            slope: Vec::new(),
        };
        for _ in 0..m {
            net.tail.push((random() as usize % n) as u32);
            net.head.push((random() as usize % n) as u32);
            let lo = (random() % 5) as i32 - 2;
            net.lower.push(lo);
            net.upper.push(lo + (random() % 4) as i32);
            cost.weight.push((random() % 5) as i64);
            cost.center.push((random() % 7) as i64 - 3);
            cost.slope.push((random() % 11) as i64 - 5);
        }
        for u in 1..n {
            net.supply[u] = (random() % 9) as i64 - 4;
            net.supply[0] -= net.supply[u];
        }
        let mut flow = net.lower.clone();
        let mut best = None;
        loop {
            if net.is_feasible(&flow) {
                let c = jupwrap_core::total_cost(&cost, &flow);
                best = Some(best.map_or(c, |old: i64| old.min(c)));
            }
            let mut a = 0;
            while a < m && flow[a] == net.upper[a] {
                flow[a] = net.lower[a];
                a += 1;
            }
            if a == m {
                break;
            }
            flow[a] += 1;
        }
        let result = solve_convex(&net, &cost);
        if let Some(want) = best {
            let flow = result.unwrap_or_else(|e| panic!("case {case}: {e}"));
            assert!(net.is_feasible(&flow), "case {case}");
            assert_eq!(jupwrap_core::total_cost(&cost, &flow), want, "case {case}");
        } else {
            assert_eq!(result, Err(McfError::Infeasible), "case {case}");
        }
    }
}

#[test]
fn full_i32_capacity_is_pushed_in_bulk() {
    struct Linear;
    impl ArcCost for Linear {
        fn cost(&self, _: usize, x: i32) -> i64 {
            -i64::from(x)
        }
    }
    let net = Network {
        n_nodes: 2,
        tail: vec![0],
        head: vec![1],
        lower: vec![i32::MIN],
        upper: vec![i32::MAX],
        supply: vec![i64::from(i32::MIN), -i64::from(i32::MIN)],
    };
    assert_eq!(solve_convex(&net, &Linear), Ok(vec![i32::MIN]));
}

#[test]
fn distant_arc_minimum_and_fixed_selfloop() {
    let net = Network {
        n_nodes: 1,
        tail: vec![0, 0],
        head: vec![0, 0],
        lower: vec![-1_000_000, -7],
        upper: vec![1_000_000, -7],
        supply: vec![0],
    };
    let cost = Quadratic {
        weight: vec![1, 1],
        center: vec![123_456, 0],
        slope: vec![0, 0],
    };
    assert_eq!(solve_convex(&net, &cost), Ok(vec![123_456, -7]));
}

#[test]
fn overflowing_supply_prefix_is_validated_without_panicking() {
    struct UnusedCost;
    impl ArcCost for UnusedCost {
        fn cost(&self, _: usize, _: i32) -> i64 {
            panic!("invalid or arcless input must not evaluate costs")
        }
    }
    let mut net = Network {
        n_nodes: 4,
        supply: vec![i64::MAX, 1, -i64::MAX, -1],
        ..Network::default()
    };
    // The mathematical sum is zero although the second prefix exceeds i64.
    assert_eq!(solve_convex(&net, &UnusedCost), Err(McfError::Infeasible));
    net.supply[3] = 0;
    assert!(matches!(
        solve_convex(&net, &UnusedCost),
        Err(McfError::InvalidInput(_))
    ));
    // A mathematical sum of 2^64 must not wrap to zero in a release build.
    net.supply = vec![i64::MAX, i64::MAX, 1, 1];
    assert!(matches!(
        solve_convex(&net, &UnusedCost),
        Err(McfError::InvalidInput(_))
    ));
    // Structural checks retain precedence over the exceptional supply check.
    net.head.push(0);
    assert_eq!(
        solve_convex(&net, &UnusedCost),
        Err(McfError::InvalidInput("arc array lengths differ".into()))
    );
}
