//! Path simplification — Ramer–Douglas–Peucker on closed loops.
//!
//! Stage 4 of the owned pipeline: the contour tracer emits staircase polygons
//! along pixel edges; RDP drops vertices within `epsilon` of a straight chord,
//! turning long stairs into clean diagonals and shrinking the path data. (The
//! later curve-fitting stage will replace these polylines with Béziers.)

type Pt = (i32, i32);

/// Squared perpendicular distance from `p` to the segment `a`–`b`.
fn perp_dist2(p: Pt, a: Pt, b: Pt) -> f64 {
    let (px, py) = (p.0 as f64, p.1 as f64);
    let (ax, ay) = (a.0 as f64, a.1 as f64);
    let (bx, by) = (b.0 as f64, b.1 as f64);
    let dx = bx - ax;
    let dy = by - ay;
    let len2 = dx * dx + dy * dy;
    if len2 == 0.0 {
        let ex = px - ax;
        let ey = py - ay;
        return ex * ex + ey * ey;
    }
    let cross = (px - ax) * dy - (py - ay) * dx;
    cross * cross / len2
}

fn rdp(pts: &[Pt], first: usize, last: usize, eps2: f64, keep: &mut [bool]) {
    if last <= first + 1 {
        return;
    }
    let mut idx = first;
    let mut max_d = 0.0;
    for i in (first + 1)..last {
        let d = perp_dist2(pts[i], pts[first], pts[last]);
        if d > max_d {
            max_d = d;
            idx = i;
        }
    }
    if max_d > eps2 {
        keep[idx] = true;
        rdp(pts, first, idx, eps2, keep);
        rdp(pts, idx, last, eps2, keep);
    }
}

/// Simplify a closed loop with tolerance `epsilon`. Returns the kept vertices
/// (first point preserved as an anchor). A non-positive epsilon is a no-op.
pub fn simplify_closed(loop_pts: &[Pt], epsilon: f64) -> Vec<Pt> {
    let n = loop_pts.len();
    if epsilon <= 0.0 || n <= 3 {
        return loop_pts.to_vec();
    }
    // Treat the closed loop as an open polyline that returns to its anchor.
    let mut pts = loop_pts.to_vec();
    pts.push(loop_pts[0]);
    let m = pts.len();
    let mut keep = vec![false; m];
    keep[0] = true;
    keep[m - 1] = true;
    rdp(&pts, 0, m - 1, epsilon * epsilon, &mut keep);
    let mut out = Vec::new();
    for i in 0..(m - 1) {
        if keep[i] {
            out.push(pts[i]);
        }
    }
    out
}

/// Squared perpendicular distance from float `p` to the segment `a`–`b`.
fn perp_dist2_f(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let dx = b.0 - a.0;
    let dy = b.1 - a.1;
    let len2 = dx * dx + dy * dy;
    if len2 == 0.0 {
        let ex = p.0 - a.0;
        let ey = p.1 - a.1;
        return ex * ex + ey * ey;
    }
    let cross = (p.0 - a.0) * dy - (p.1 - a.1) * dx;
    cross * cross / len2
}

/// Float variant of [`simplify_closed`] — used after edge-snapping, where the
/// contour is sub-pixel. `force_keep[i]` (if provided) marks vertices that must
/// survive (shared junctions), so adjacent regions agree on the boundary.
pub fn simplify_closed_f(loop_pts: &[(f64, f64)], epsilon: f64) -> Vec<(f64, f64)> {
    simplify_closed_f_keep(loop_pts, epsilon, &[])
}

pub fn simplify_closed_f_keep(
    loop_pts: &[(f64, f64)],
    epsilon: f64,
    force_keep: &[bool],
) -> Vec<(f64, f64)> {
    let n = loop_pts.len();
    if n < 3 {
        return loop_pts.to_vec();
    }
    let mut forced: Vec<usize> = (0..n)
        .filter(|&i| force_keep.get(i).copied().unwrap_or(false))
        .collect();
    if forced.is_empty() {
        forced.push(lexmin_index_f(loop_pts)); // canonical anchor
    }
    let keep = split_rdp(n, &forced, epsilon, |a, b, c| {
        perp_dist2_f(loop_pts[a], loop_pts[b], loop_pts[c])
    });
    (0..n).filter(|&i| keep[i]).map(|i| loop_pts[i]).collect()
}

/// Integer [`simplify_closed`] that force-keeps any vertex in `keep_set` (shared
/// junctions) and **splits** the RDP at them, so the boundary between two
/// junctions simplifies identically from whichever region traces it.
pub fn simplify_closed_keep(
    loop_pts: &[Pt],
    epsilon: f64,
    keep_set: &std::collections::HashSet<Pt>,
) -> Vec<Pt> {
    let n = loop_pts.len();
    if n < 3 {
        return loop_pts.to_vec();
    }
    let mut forced: Vec<usize> = (0..n).filter(|&i| keep_set.contains(&loop_pts[i])).collect();
    if forced.is_empty() {
        forced.push(lexmin_index_i(loop_pts)); // canonical anchor
    }
    let keep = split_rdp(n, &forced, epsilon, |a, b, c| {
        perp_dist2(loop_pts[a], loop_pts[b], loop_pts[c])
    });
    (0..n).filter(|&i| keep[i]).map(|i| loop_pts[i]).collect()
}

/// Run RDP independently on each circular interval of `0..n` delimited by the
/// `forced` indices (which are always kept). `dist(i, a, b)` is the squared
/// perpendicular distance from point `i` to the segment `a`–`b`. This makes
/// simplification a deterministic function of the boundary's geometry, so two
/// regions sharing it produce the same vertices.
fn split_rdp(
    n: usize,
    forced: &[usize],
    epsilon: f64,
    dist: impl Fn(usize, usize, usize) -> f64,
) -> Vec<bool> {
    let mut anchors = forced.to_vec();
    anchors.sort_unstable();
    anchors.dedup();
    let mut keep = vec![false; n];
    for &a in &anchors {
        keep[a] = true;
    }
    if epsilon <= 0.0 {
        return keep;
    }
    let eps2 = epsilon * epsilon;
    let k = anchors.len();
    for idx in 0..k {
        let a = anchors[idx];
        let b = anchors[(idx + 1) % k];
        // Build the circular index run a..=b and RDP it as an open polyline.
        // A do-while so the single-anchor case (a == b) wraps the *whole* loop
        // (a, a+1, …, a) instead of degenerating to [a] and collapsing the loop.
        let mut run = vec![a];
        let mut i = a;
        loop {
            i = (i + 1) % n;
            run.push(i);
            if i == b {
                break;
            }
        }
        rdp_run(&run, 0, run.len() - 1, eps2, &dist, &mut keep);
    }
    keep
}

fn rdp_run(
    run: &[usize],
    first: usize,
    last: usize,
    eps2: f64,
    dist: &impl Fn(usize, usize, usize) -> f64,
    keep: &mut [bool],
) {
    if last <= first + 1 {
        return;
    }
    let (a, b) = (run[first], run[last]);
    let mut max_d = 0.0;
    let mut mi = first;
    for (i, &ri) in run.iter().enumerate().take(last).skip(first + 1) {
        let d = dist(ri, a, b);
        if d > max_d {
            max_d = d;
            mi = i;
        }
    }
    if max_d > eps2 {
        keep[run[mi]] = true;
        rdp_run(run, first, mi, eps2, dist, keep);
        rdp_run(run, mi, last, eps2, dist, keep);
    }
}

fn lexmin_index_f(pts: &[(f64, f64)]) -> usize {
    let mut best = 0;
    for i in 1..pts.len() {
        if pts[i] < pts[best] {
            best = i;
        }
    }
    best
}

fn lexmin_index_i(pts: &[Pt]) -> usize {
    let mut best = 0;
    for i in 1..pts.len() {
        if pts[i] < pts[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collapses_a_staircase_into_a_diagonal() {
        // A staircase from (0,0) to (4,4) closed back along the top edge.
        let loop_pts = vec![
            (0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (4, 3), (4, 4), (0, 4),
        ];
        let s = simplify_closed(&loop_pts, 1.5);
        // The stair should collapse to roughly the corners of a triangle.
        assert!(s.len() < loop_pts.len());
        assert!(s.len() <= 4, "expected a near-triangle, got {}", s.len());
    }

    #[test]
    fn noop_on_small_or_zero_epsilon() {
        let loop_pts = vec![(0, 0), (4, 0), (4, 4), (0, 4)];
        assert_eq!(simplify_closed(&loop_pts, 0.0), loop_pts);
        assert_eq!(simplify_closed(&loop_pts, -1.0), loop_pts);
    }

    #[test]
    fn keeps_real_corners() {
        // A clean square should keep its 4 corners.
        let sq = vec![(0, 0), (10, 0), (10, 10), (0, 10)];
        let s = simplify_closed(&sq, 1.0);
        assert_eq!(s.len(), 4);
    }

    #[test]
    fn float_keep_single_anchor_does_not_collapse_loop() {
        // A loop with no forced points falls back to a single (lexmin) anchor.
        // The circular run must wrap the whole loop, not degenerate to [anchor].
        let sq: Vec<(f64, f64)> = vec![
            (0.0, 0.0), (5.0, 0.0), (10.0, 0.0), (10.0, 5.0),
            (10.0, 10.0), (5.0, 10.0), (0.0, 10.0), (0.0, 5.0),
        ];
        let s = simplify_closed_f_keep(&sq, 1.0, &[]);
        // Should collapse the collinear midpoints to the 4 corners — not vanish.
        assert!(s.len() >= 4, "loop collapsed to {} points", s.len());
        assert!(s.len() <= 5);
    }

    #[test]
    fn float_keep_pins_forced_points() {
        let sq: Vec<(f64, f64)> =
            vec![(0.0, 0.0), (5.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        // Force-keep the collinear midpoint (5,0) at index 1.
        let keep = [false, true, false, false, false];
        let s = simplify_closed_f_keep(&sq, 1.0, &keep);
        assert!(s.contains(&(5.0, 0.0)), "forced midpoint must survive");
    }
}
