//! jupwrap-mcf: exact min-cost flow for convex separable integer costs.
//! See DESIGN.md section 6.

use jupwrap_core::{ArcCost, McfError, Network};

mod scaling;

#[cfg(test)]
mod tests;

/// Minimum-cost feasible flow.
///
/// Preconditions: `net.validate()` passes; `cost(a, x)` is convex in `x` over
/// `[net.lower[a], net.upper[a]]` for every arc. Self-loops and parallel arcs
/// may occur. The per-arc cost minimum need not be at `x = 0` (the optimum may
/// contain circulations even when all supplies are zero).
///
/// Returns a flow `x` with `net.is_feasible(&x)` and minimum `total_cost`;
/// `Err(McfError::Infeasible)` if no feasible flow exists;
/// `Err(McfError::InvalidInput)` if `net.validate()` fails. Never panics on
/// those inputs.
pub fn solve_convex(net: &Network, cost: &dyn ArcCost) -> Result<Vec<i32>, McfError> {
    validate(net)?;
    scaling::solve(net, cost)
}

/// Core's validator sums supplies in i64. Keep its structural checks and error
/// ordering, but avoid debug panics (and release wrapping) when that sum would
/// overflow. The exceptional path validates an otherwise identical network
/// with zeroed supplies, then checks the actual sum exactly.
fn validate(net: &Network) -> Result<(), McfError> {
    if net
        .supply
        .iter()
        .try_fold(0i64, |sum, &supply| sum.checked_add(supply))
        .is_some()
    {
        return net.validate();
    }
    let mut structural = net.clone();
    structural.supply.fill(0);
    structural.validate()?;
    if net.supply.iter().map(|&s| i128::from(s)).sum::<i128>() != 0 {
        return Err(McfError::InvalidInput("supplies do not sum to zero".into()));
    }
    Ok(())
}
