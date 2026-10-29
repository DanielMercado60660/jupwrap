use super::label_components;
use jupwrap_core::{Dims, GridCosts, K_BIG};

#[test]
fn reliability_and_minimum_size_include_equality() {
    let costs = GridCosts {
        dims: Dims::new(1, 3),
        kt: 1,
        table: vec![300, 0, 700, 299, 0, 900],
    };
    assert_eq!(
        label_components(&costs, &[0, 0], None, 300, 1),
        (vec![1, 1, 2], 2)
    );
    assert_eq!(
        label_components(&costs, &[0, 0], None, 300, 2),
        (vec![1, 1, 0], 1)
    );
    assert_eq!(
        label_components(&costs, &[0, 0], None, 300, 3),
        (vec![0, 0, 0], 0)
    );
    assert_eq!(
        label_components(&costs, &[0, 0], None, 301, 1),
        (vec![1, 2, 3], 3)
    );
}

#[test]
fn reliability_uses_both_linear_tails() {
    for (table, sign) in [(vec![800, 400, 0], 1), (vec![0, 400, 800], -1)] {
        let costs = GridCosts {
            dims: Dims::new(2, 1),
            kt: 1,
            table,
        };
        // At the table boundary, the outward increment is the minimum tail
        // slope (100), while the inward increment is 400.
        assert_eq!(
            label_components(&costs, &[sign], None, 100, 1),
            (vec![1, 1], 1)
        );
        assert_eq!(
            label_components(&costs, &[sign], None, 101, 1),
            (vec![1, 2], 2)
        );
        for flow in [2 * sign, K_BIG * sign] {
            assert_eq!(
                label_components(&costs, &[flow], None, -100, 1),
                (vec![1, 1], 1)
            );
            assert_eq!(
                label_components(&costs, &[flow], None, -99, 1),
                (vec![1, 2], 2)
            );
        }
    }
    let costs = GridCosts {
        dims: Dims::new(1, 2),
        kt: 0,
        table: vec![0],
    };
    assert_eq!(
        label_components(&costs, &[0], None, 100, 1),
        (vec![1, 1], 1)
    );
}

#[test]
fn sizes_ties_and_nonzero_masks() {
    let dims = Dims::new(2, 4);
    let mut costs = GridCosts {
        dims,
        kt: 1,
        table: vec![0; dims.n_edges() * 3],
    };
    // Components {0, 1, 5} and {2, 6, 7} tie in size. The first one's
    // union-find root is 1, whereas its first row-major pixel is 0.
    for e in [1, 2, 4, 9] {
        costs.table[e * 3..e * 3 + 3].copy_from_slice(&[7, 0, 7]);
    }
    let flow = vec![0; dims.n_edges()];
    let mask = [255, 2, 1, 0, 1, 1, 1, 1];
    assert_eq!(
        label_components(&costs, &flow, Some(&mask), 7, 1),
        (vec![1, 1, 2, 0, 3, 1, 2, 2], 3)
    );
    assert_eq!(
        label_components(&costs, &flow, Some(&mask), 7, 3),
        (vec![1, 1, 2, 0, 0, 1, 2, 2], 2)
    );
    assert_eq!(
        label_components(&costs, &flow, Some(&mask), 7, 4),
        (vec![0; 8], 0)
    );
    assert_eq!(
        label_components(&costs, &flow, Some(&mask), 0, 8),
        (vec![0; 8], 0)
    );
}

#[test]
fn masked_pixels_never_connect_diagonal_neighbors() {
    let costs = GridCosts {
        dims: Dims::new(2, 2),
        kt: 1,
        table: vec![0; 12],
    };
    let mask = [1, 0, 0, 255];
    assert_eq!(
        label_components(&costs, &[0; 4], Some(&mask), 0, 0),
        (vec![1, 0, 0, 2], 2)
    );
    assert_eq!(
        label_components(&costs, &[0; 4], Some(&[0; 4]), 0, 0),
        (vec![0; 4], 0)
    );
    assert_eq!(
        label_components(&GridCosts::default(), &[], None, 0, 0),
        (vec![], 0)
    );
}
