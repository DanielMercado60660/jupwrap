//! Connected components over reliable grid edges (DESIGN.md section 9).

use jupwrap_core::{ArcCost, GridCosts};

/// Label pixels by decreasing component size, breaking ties by first pixel.
/// Masked pixels and components smaller than `min_size` receive zero.
///
/// Only four-neighbor edges with two valid endpoints and
/// `min(cost(k + 1), cost(k - 1)) - cost(k) >= thresh` connect pixels.
/// Costs outside the stored table use the core `GridCosts::cost` extension.
pub fn label_components(
    costs: &GridCosts,
    k: &[i32],
    mask: Option<&[u8]>,
    thresh: i64,
    min_size: usize,
) -> (Vec<u32>, u32) {
    let dims = costs.dims;
    let npix = dims.npix();
    assert_eq!(k.len(), dims.n_edges(), "cycle field length");
    if let Some(mask) = mask {
        assert_eq!(mask.len(), npix, "mask length");
    }
    let nvalid = mask.map_or(npix, |m| m.iter().filter(|&&v| v != 0).count());
    if nvalid == 0 || min_size > nvalid {
        return (vec![0; npix], 0);
    }
    let mut parent: Vec<usize> = (0..npix).collect();
    let mut sizes = vec![1usize; npix];

    for (e, &flow) in k.iter().enumerate() {
        let (p, q) = dims.edge_endpoints(e);
        if mask.is_some_and(|m| m[p] == 0 || m[q] == 0) {
            continue;
        }
        let current = costs.cost(e, flow);
        let neighbor = costs.cost(e, flow + 1).min(costs.cost(e, flow - 1));
        if neighbor - current >= thresh {
            let mut a = root(&mut parent, p);
            let mut b = root(&mut parent, q);
            if a != b {
                if sizes[a] < sizes[b] {
                    std::mem::swap(&mut a, &mut b);
                }
                parent[b] = a;
                sizes[a] += sizes[b];
            }
        }
    }

    // Row-major traversal discovers each component's first pixel. Sizes are
    // no longer needed after discovery, so taking them also marks roots seen.
    let mut components = Vec::new();
    for p in 0..npix {
        if mask.is_some_and(|m| m[p] == 0) {
            continue;
        }
        let r = root(&mut parent, p);
        let size = std::mem::take(&mut sizes[r]);
        if size != 0 && size >= min_size {
            components.push((size, p, r));
        }
    }
    drop(sizes);
    components.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    let count =
        u32::try_from(components.len()).expect("too many connected components for u32 labels");
    let mut labels = vec![0u32; npix];
    for (i, &(_, _, r)) in components.iter().enumerate() {
        labels[r] = (i + 1) as u32;
    }
    drop(components);
    // The output doubles as the root-to-label map: writing a pixel cannot
    // change another component's root, and each root keeps its own label.
    for p in 0..npix {
        if mask.is_none_or(|m| m[p] != 0) {
            labels[p] = labels[root(&mut parent, p)];
        }
    }
    (labels, count)
}

#[inline]
fn root(parent: &mut [usize], mut p: usize) -> usize {
    while parent[p] != p {
        parent[p] = parent[parent[p]];
        p = parent[p];
    }
    p
}

#[cfg(test)]
mod tests;
