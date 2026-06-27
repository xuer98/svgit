//! Curve fitting — stage 6 of the owned pipeline.
//!
//! Fits cubic Béziers to each contour loop using Schneider's algorithm
//! ("An Algorithm for Automatically Fitting Digitized Curves", Graphics Gems):
//! parameterize the points by chord length, solve a least-squares fit for the
//! two interior control points, and recursively subdivide at the worst-fitting
//! point until every span is within tolerance. Corner detection splits the loop
//! at sharp turns first, so genuine corners stay crisp instead of being rounded.

use crate::svg::{Seg, Subpath};

type V = (f64, f64);

#[inline]
fn add(a: V, b: V) -> V {
    (a.0 + b.0, a.1 + b.1)
}
#[inline]
fn sub(a: V, b: V) -> V {
    (a.0 - b.0, a.1 - b.1)
}
#[inline]
fn scale(a: V, s: f64) -> V {
    (a.0 * s, a.1 * s)
}
#[inline]
fn dot(a: V, b: V) -> f64 {
    a.0 * b.0 + a.1 * b.1
}
#[inline]
fn dist(a: V, b: V) -> f64 {
    sub(a, b).0.hypot(sub(a, b).1)
}
#[inline]
fn normalize(a: V) -> V {
    let n = (a.0 * a.0 + a.1 * a.1).sqrt();
    if n == 0.0 {
        (0.0, 0.0)
    } else {
        (a.0 / n, a.1 / n)
    }
}

/// Evaluate a Bézier curve at `t` via de Casteljau. Degree is at most 3 here
/// (cubic and its derivatives), so this evaluates on the stack — no allocation.
fn bezier(ctrl: &[V], t: f64) -> V {
    let n = ctrl.len();
    let mut p = [(0.0, 0.0); 4];
    p[..n].copy_from_slice(ctrl);
    for k in 1..n {
        for i in 0..(n - k) {
            p[i] = add(scale(p[i], 1.0 - t), scale(p[i + 1], t));
        }
    }
    p[0]
}

/// Fit one loop of integer corners into a closed subpath of cubics.
/// `corner_deg` is the turn angle above which a vertex is treated as a corner;
/// `error` is the fit tolerance in pixels.
pub fn fit_loop(loop_pts: &[(i32, i32)], corner_deg: f64, error: f64) -> Subpath {
    let mut pts: Vec<V> = loop_pts.iter().map(|&(x, y)| (x as f64, y as f64)).collect();
    // Drop consecutive duplicate vertices so corner detection and tangent
    // estimation never see a zero-length edge.
    pts.dedup();
    if pts.len() > 1 && pts.first() == pts.last() {
        pts.pop();
    }
    let corners = detect_corners(&pts, corner_deg);
    fit_loop_pts(&pts, &corners, error)
}

/// Indices of `pts` where the turn between the incoming and outgoing edge
/// exceeds `corner_deg` degrees. Direction-symmetric, so a boundary traced from
/// either side yields the same corners (keeps shared region edges consistent).
pub fn detect_corners(pts: &[V], corner_deg: f64) -> Vec<usize> {
    let n = pts.len();
    if n < 3 {
        return Vec::new();
    }
    let cos_thresh = corner_deg.to_radians().cos();
    let mut corners = Vec::new();
    for i in 0..n {
        let prev = pts[(i + n - 1) % n];
        let cur = pts[i];
        let next = pts[(i + 1) % n];
        let v1 = normalize(sub(cur, prev));
        let v2 = normalize(sub(next, cur));
        if dot(v1, v2) < cos_thresh {
            corners.push(i);
        }
    }
    corners
}

/// Fit a closed loop given as float points plus precomputed corner indices
/// (ascending, `< pts.len()`). The curve is cut into open spans at the corners
/// — each fit independently — so genuine corners stay crisp. Used by both the
/// integer [`fit_loop`] and the edge-refined tracer.
pub fn fit_loop_pts(pts: &[V], corners: &[usize], error: f64) -> Subpath {
    let n = pts.len();
    if n < 4 {
        // Degenerate: emit a polygon.
        return Subpath {
            start: pts.first().copied().unwrap_or((0.0, 0.0)),
            segs: pts.get(1..).unwrap_or(&[]).iter().map(|&p| Seg::Line(p)).collect(),
        };
    }

    let mut cubics: Vec<[V; 3]> = Vec::new();
    let start;
    if corners.is_empty() {
        // Smooth closed loop: break at vertex 0 and fit it as one open span.
        start = pts[0];
        let mut span: Vec<V> = pts.to_vec();
        span.push(pts[0]);
        fit_span(&span, error, &mut cubics);
    } else {
        start = pts[corners[0]];
        let c = corners.len();
        for k in 0..c {
            let a = corners[k];
            let b = corners[(k + 1) % c];
            let mut span = Vec::new();
            let mut i = a;
            loop {
                span.push(pts[i]);
                if i == b {
                    break;
                }
                i = (i + 1) % n;
            }
            fit_span(&span, error, &mut cubics);
        }
    }

    Subpath {
        start,
        segs: cubics
            .into_iter()
            .map(|c| Seg::Cubic(c[0], c[1], c[2]))
            .collect(),
    }
}

/// Fit an open span of points (>= 2) into one or more cubics.
fn fit_span(d: &[V], error: f64, out: &mut Vec<[V; 3]>) {
    let n = d.len();
    if n < 2 {
        return;
    }
    let t_left = normalize(sub(d[1], d[0]));
    let t_right = normalize(sub(d[n - 2], d[n - 1]));
    fit_cubic(d, t_left, t_right, error.max(1e-6), 0, out);
}

fn line_cubic(p0: V, p3: V) -> [V; 3] {
    let c1 = add(p0, scale(sub(p3, p0), 1.0 / 3.0));
    let c2 = add(p0, scale(sub(p3, p0), 2.0 / 3.0));
    [c1, c2, p3]
}

/// Recursion depth cap — bails out to a straight segment rather than risking a
/// stack overflow on a pathological span.
const MAX_FIT_DEPTH: usize = 48;

fn fit_cubic(d: &[V], t1: V, t2: V, error: f64, depth: usize, out: &mut Vec<[V; 3]>) {
    let n = d.len();
    if n == 2 || depth >= MAX_FIT_DEPTH {
        out.push(line_cubic(d[0], d[n - 1]));
        return;
    }

    let mut u = chord_param(d);
    let mut bez = generate_bezier(d, &u, t1, t2);
    let (mut max_err, mut split) = compute_max_error(d, &u, &bez);

    if max_err < error {
        out.push([bez[1], bez[2], bez[3]]);
        return;
    }

    // Close enough: try reparameterizing a few times before giving up.
    if max_err < error * 4.0 {
        for _ in 0..16 {
            reparameterize(d, &mut u, &bez);
            bez = generate_bezier(d, &u, t1, t2);
            let (e, s) = compute_max_error(d, &u, &bez);
            max_err = e;
            split = s;
            if max_err < error {
                out.push([bez[1], bez[2], bez[3]]);
                return;
            }
        }
    }

    // Split at the worst point and recurse with a tangent estimated there.
    let split = split.clamp(1, n - 2);
    let center = normalize(sub(d[split - 1], d[split + 1]));
    fit_cubic(&d[..=split], t1, center, error, depth + 1, out);
    fit_cubic(&d[split..], (-center.0, -center.1), t2, error, depth + 1, out);
}

fn chord_param(d: &[V]) -> Vec<f64> {
    let n = d.len();
    let mut u = vec![0.0; n];
    for i in 1..n {
        u[i] = u[i - 1] + dist(d[i], d[i - 1]);
    }
    let total = u[n - 1];
    if total > 0.0 {
        for x in u.iter_mut() {
            *x /= total;
        }
    }
    u
}

fn b0(u: f64) -> f64 {
    let t = 1.0 - u;
    t * t * t
}
fn b1(u: f64) -> f64 {
    let t = 1.0 - u;
    3.0 * u * t * t
}
fn b2(u: f64) -> f64 {
    let t = 1.0 - u;
    3.0 * u * u * t
}
fn b3(u: f64) -> f64 {
    u * u * u
}

fn generate_bezier(d: &[V], u: &[f64], t1: V, t2: V) -> [V; 4] {
    let n = d.len();
    let p0 = d[0];
    let p3 = d[n - 1];
    let mut c = [[0.0f64; 2]; 2];
    let mut x = [0.0f64; 2];
    for i in 0..n {
        let a0 = scale(t1, b1(u[i]));
        let a1 = scale(t2, b2(u[i]));
        c[0][0] += dot(a0, a0);
        c[0][1] += dot(a0, a1);
        c[1][0] += dot(a0, a1);
        c[1][1] += dot(a1, a1);
        let fixed = add(
            add(scale(p0, b0(u[i])), scale(p0, b1(u[i]))),
            add(scale(p3, b2(u[i])), scale(p3, b3(u[i]))),
        );
        let tmp = sub(d[i], fixed);
        x[0] += dot(a0, tmp);
        x[1] += dot(a1, tmp);
    }

    let det_c = c[0][0] * c[1][1] - c[1][0] * c[0][1];
    let det_x_c1 = x[0] * c[1][1] - x[1] * c[0][1];
    let det_c0_x = c[0][0] * x[1] - c[1][0] * x[0];

    let (mut alpha_l, mut alpha_r) = if det_c.abs() < 1e-12 {
        (0.0, 0.0)
    } else {
        (det_x_c1 / det_c, det_c0_x / det_c)
    };

    let seg_len = dist(p0, p3);
    let eps = 1e-6 * seg_len;
    // Reject non-finite or runaway handles: a near-singular least-squares system
    // can blow the control points far off-canvas even when det passes the
    // threshold, so cap the handle length and fall back to Wu/Barsky.
    let max_alpha = 4.0 * seg_len;
    if !alpha_l.is_finite()
        || !alpha_r.is_finite()
        || alpha_l < eps
        || alpha_r < eps
        || alpha_l > max_alpha
        || alpha_r > max_alpha
    {
        // Fall back to the Wu/Barsky heuristic: place handles a third of the
        // chord length along the end tangents.
        let third = seg_len / 3.0;
        alpha_l = third;
        alpha_r = third;
        return [p0, add(p0, scale(t1, alpha_l)), add(p3, scale(t2, alpha_r)), p3];
    }
    [p0, add(p0, scale(t1, alpha_l)), add(p3, scale(t2, alpha_r)), p3]
}

fn compute_max_error(d: &[V], u: &[f64], bez: &[V; 4]) -> (f64, usize) {
    let n = d.len();
    let mut max_d = 0.0;
    let mut split = n / 2;
    for i in 1..(n - 1) {
        let p = bezier(bez, u[i]);
        let e = dist(p, d[i]);
        if e >= max_d {
            max_d = e;
            split = i;
        }
    }
    (max_d, split)
}

fn reparameterize(d: &[V], u: &mut [f64], bez: &[V; 4]) {
    for i in 0..d.len() {
        u[i] = newton(bez, d[i], u[i]);
    }
}

fn newton(bez: &[V; 4], p: V, u: f64) -> f64 {
    // First and second derivative control points.
    let q1: [V; 3] = [
        scale(sub(bez[1], bez[0]), 3.0),
        scale(sub(bez[2], bez[1]), 3.0),
        scale(sub(bez[3], bez[2]), 3.0),
    ];
    let q2: [V; 2] = [scale(sub(q1[1], q1[0]), 2.0), scale(sub(q1[2], q1[1]), 2.0)];
    let qu = bezier(bez, u);
    let q1u = bezier(&q1, u);
    let q2u = bezier(&q2, u);
    let num = dot(sub(qu, p), q1u);
    let den = dot(q1u, q1u) + dot(sub(qu, p), q2u);
    if den.abs() < 1e-12 {
        u
    } else {
        // Keep the parameter in [0,1]; Newton can otherwise overshoot and feed
        // out-of-range values into the next least-squares fit.
        (u - num / den).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval_subpath_points(sp: &Subpath, per: usize) -> Vec<V> {
        // Sample the curve densely for error checking.
        let mut pts = Vec::new();
        let mut cur = sp.start;
        for seg in &sp.segs {
            match seg {
                Seg::Line(e) => {
                    pts.push(cur);
                    cur = *e;
                }
                Seg::Cubic(c1, c2, e) => {
                    let ctrl = [cur, *c1, *c2, *e];
                    for k in 0..per {
                        pts.push(bezier(&ctrl, k as f64 / per as f64));
                    }
                    cur = *e;
                }
            }
        }
        pts.push(cur);
        pts
    }

    #[test]
    fn fits_a_straight_run_with_low_error() {
        let line: Vec<(i32, i32)> = (0..=10).map(|i| (i, 0)).collect();
        let mut out = Vec::new();
        let d: Vec<V> = line.iter().map(|&(x, y)| (x as f64, y as f64)).collect();
        fit_span(&d, 1.0, &mut out);
        assert_eq!(out.len(), 1, "a straight run should be one cubic");
    }

    #[test]
    fn fits_a_circle_smoothly_within_tolerance() {
        // Sample a circle; fit_loop should reproduce it closely.
        let r = 50.0;
        let pts: Vec<(i32, i32)> = (0..48)
            .map(|i| {
                let a = i as f64 / 48.0 * std::f64::consts::TAU;
                ((r * a.cos() + r) as i32, (r * a.sin() + r) as i32)
            })
            .collect();
        let sp = fit_loop(&pts, 80.0, 2.0);
        // Every sampled curve point must lie near radius r from the center.
        let center = (r, r);
        let max_dev = eval_subpath_points(&sp, 8)
            .iter()
            .map(|&p| (dist(p, center) - r).abs())
            .fold(0.0, f64::max);
        assert!(max_dev < 3.0, "circle deviation too large: {max_dev}");
    }

    #[test]
    fn square_keeps_four_corners() {
        // A 10x10 square has four 90-degree corners; each should be detected,
        // yielding (near-)straight cubics between them.
        let sq = vec![(0, 0), (10, 0), (10, 10), (0, 10)];
        let sp = fit_loop(&sq, 80.0, 1.0);
        assert_eq!(sp.segs.len(), 4, "four edges between four corners");
    }

    #[test]
    fn degenerate_and_duplicate_input_stays_bounded() {
        // Duplicate vertices + a long near-collinear span must not panic, and
        // (regression for the near-singular least-squares blow-up) every emitted
        // control point must stay finite and near the input bounds.
        let pts = vec![
            (0, 0), (0, 0), (10, 0), (20, 0), (30, 1), (40, 0), (50, 0), (50, 0), (25, 8),
        ];
        let sp = fit_loop(&pts, 80.0, 2.0);
        let ok = |p: (f64, f64)| {
            p.0.is_finite() && p.1.is_finite() && p.0.abs() < 1000.0 && p.1.abs() < 1000.0
        };
        assert!(ok(sp.start));
        for seg in &sp.segs {
            match seg {
                Seg::Line(e) => assert!(ok(*e)),
                Seg::Cubic(c1, c2, e) => {
                    assert!(ok(*c1) && ok(*c2) && ok(*e), "control point exploded: {c1:?} {c2:?} {e:?}");
                }
            }
        }
    }
}
