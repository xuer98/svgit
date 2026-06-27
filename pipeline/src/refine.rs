//! Edge-guided contour refinement (Level 3).
//!
//! Given a CNN edge-probability map (produced by `svgit-edgenet`), this **snaps**
//! the dense contour onto the nearest true-edge ridge along the edge gradient —
//! de-staircasing the quantization boundary — then re-simplifies and detects
//! corners on the cleaned-up curve, so curve-fitting breaks tangents where the
//! real (edge-aligned) geometry turns sharply.
//!
//! Snapping the *dense* contour (not the simplified vertices) is what keeps the
//! result smooth: each pixel-edge vertex moves a little toward the ridge, then
//! RDP collapses the smooth snapped curve. The snap is a pure function of vertex
//! position and the edge map (gradient ascent, bounded displacement, ridge-
//! strength-gated); because the owned tracer draws each color region's boundary
//! independently, that determinism keeps a shared boundary coincident from both
//! sides. Corner detection is likewise position-based and direction-symmetric.

use std::collections::HashSet;

use crate::curvefit::detect_corners;
use crate::simplify::simplify_closed_f_keep;

type V = (f64, f64);

#[derive(Debug, Clone)]
pub struct RefineConfig {
    /// Max distance (px) a vertex may move toward an edge ridge.
    pub snap_radius: f64,
    /// Minimum edge strength (0..1) required to snap toward a ridge.
    pub edge_threshold: f32,
    /// Turn-angle corner threshold (degrees), shared with curve fitting.
    pub corner_threshold: f64,
}

impl Default for RefineConfig {
    fn default() -> Self {
        Self {
            snap_radius: 2.0,
            edge_threshold: 0.25,
            corner_threshold: 80.0,
        }
    }
}

/// Edge map + dimensions + config, used to refine each contour loop.
pub struct Refiner<'a> {
    edge: &'a [f32],
    width: usize,
    height: usize,
    cfg: RefineConfig,
}

impl<'a> Refiner<'a> {
    pub fn new(edge: &'a [f32], width: usize, height: usize, cfg: RefineConfig) -> Self {
        Self {
            edge,
            width,
            height,
            cfg,
        }
    }

    #[inline]
    fn sample(&self, x: f64, y: f64) -> f32 {
        bilinear(self.edge, self.width, self.height, x, y)
    }

    /// Edge-map gradient at (x,y) via central differences (points uphill).
    fn gradient(&self, x: f64, y: f64) -> V {
        let gx = (self.sample(x + 1.0, y) - self.sample(x - 1.0, y)) as f64 * 0.5;
        let gy = (self.sample(x, y + 1.0) - self.sample(x, y - 1.0)) as f64 * 0.5;
        (gx, gy)
    }

    /// Move a vertex toward the local edge ridge along the gradient, bounded by
    /// `snap_radius` and gated on ridge strength. Pure in `p`.
    fn snap(&self, p: V) -> V {
        let g = self.gradient(p.0, p.1);
        let gl = (g.0 * g.0 + g.1 * g.1).sqrt();
        if gl < 1e-4 {
            return p; // flat region: no edge direction
        }
        let (dx, dy) = (g.0 / gl, g.1 / gl);
        let r = self.cfg.snap_radius;
        let steps = (r * 4.0).round() as i32; // 0.25 px granularity
        let mut best_v = self.sample(p.0, p.1);
        let mut best_t = 0.0;
        for s in -steps..=steps {
            let t = s as f64 * 0.25;
            if t.abs() > r {
                continue;
            }
            let v = self.sample(p.0 + dx * t, p.1 + dy * t);
            if v > best_v {
                best_v = v;
                best_t = t;
            }
        }
        if best_v >= self.cfg.edge_threshold && best_t != 0.0 {
            (p.0 + dx * best_t, p.1 + dy * best_t)
        } else {
            p // no strong ridge nearby — leave the vertex where it is
        }
    }

    /// Refine a *dense* integer contour loop into snapped float points (simplified
    /// at `simplify_eps`) + corner indices (ascending), ready for
    /// [`crate::curvefit::fit_loop_pts`].
    ///
    /// Evolves the contour as a short, bounded **snake**: each iteration nudges
    /// every vertex toward the local edge ridge *and* toward the midpoint of its
    /// neighbours. The smoothing term is what makes this work on texture —
    /// without it, competing fur/stripe ridges zigzag the boundary. Total
    /// displacement is capped at `snap_radius`, preserving topology and keeping
    /// the (per-region) shared boundaries from drifting apart.
    pub fn refine_loop(
        &self,
        loop_pts: &[(i32, i32)],
        simplify_eps: f64,
        junctions: &HashSet<(i32, i32)>,
    ) -> (Vec<V>, Vec<usize>) {
        let n = loop_pts.len();
        let orig: Vec<V> = loop_pts.iter().map(|&(x, y)| (x as f64, y as f64)).collect();
        if n < 4 {
            return (orig, Vec::new());
        }
        // Junction vertices (3+ regions meet) are pinned: never moved by the
        // snake and always kept by simplification, so every region that shares
        // them agrees on the boundary — no gaps.
        let pinned: Vec<bool> = loop_pts.iter().map(|p| junctions.contains(p)).collect();

        const ITERS: usize = 5;
        const EDGE_W: f64 = 0.45; // pull toward the edge ridge
        const SMOOTH_W: f64 = 0.30; // pull toward neighbour midpoint
        let max_disp = self.cfg.snap_radius;

        let mut pts = orig.clone();
        for _ in 0..ITERS {
            let mut next = pts.clone();
            for i in 0..n {
                if pinned[i] {
                    continue; // junction stays at its lattice position
                }
                let p = pts[i];
                let prev = pts[(i + n - 1) % n];
                let nx = pts[(i + 1) % n];
                let smooth = ((prev.0 + nx.0) * 0.5, (prev.1 + nx.1) * 0.5);
                let ridge = self.snap(p); // ridge position, or p if none nearby
                let mut t = (
                    p.0 + EDGE_W * (ridge.0 - p.0) + SMOOTH_W * (smooth.0 - p.0),
                    p.1 + EDGE_W * (ridge.1 - p.1) + SMOOTH_W * (smooth.1 - p.1),
                );
                // Clamp the cumulative move from the original lattice position.
                let (dx, dy) = (t.0 - orig[i].0, t.1 - orig[i].1);
                let d = (dx * dx + dy * dy).sqrt();
                if d > max_disp {
                    let s = max_disp / d;
                    t = (orig[i].0 + dx * s, orig[i].1 + dy * s);
                }
                next[i] = t;
            }
            pts = next;
        }

        let mut pts = simplify_closed_f_keep(&pts, simplify_eps, &pinned);
        // Drop near-duplicates, but never a pinned junction (the only points at
        // exact integer coords) — removing one would reopen a tiling gap.
        pts.dedup_by(|a, b| close(*a, *b) && !is_int(*a));
        if pts.len() > 1 && close(pts[0], pts[pts.len() - 1]) && !is_int(pts[pts.len() - 1]) {
            pts.pop();
        }
        if pts.len() < 3 {
            return (pts, Vec::new());
        }
        let corners = corners_with_junctions(&pts, self.cfg.corner_threshold, junctions);
        (pts, corners)
    }
}

/// Corner indices for curve fitting: the turn-angle corners, plus every pinned
/// junction. Breaking the curve span at junctions is what makes a shared edge
/// fit identically from both regions — without it the span blends in each
/// region's *other* edges and the two curves diverge (gaps).
pub fn corners_with_junctions(
    pts: &[V],
    corner_deg: f64,
    junctions: &HashSet<(i32, i32)>,
) -> Vec<usize> {
    let mut set: HashSet<usize> = detect_corners(pts, corner_deg).into_iter().collect();
    for (i, p) in pts.iter().enumerate() {
        // Junction vertices are pinned, so they sit at exact integer coords.
        if p.0.fract() == 0.0
            && p.1.fract() == 0.0
            && junctions.contains(&(p.0 as i32, p.1 as i32))
        {
            set.insert(i);
        }
    }
    // A perfectly smooth closed loop (no corners) would be fit as one span
    // starting at pts[0] — non-canonical, so two regions sharing it diverge at
    // the seam. Break at the lexicographically-smallest point instead, which is
    // the same geometric landmark from either side.
    if set.is_empty() && !pts.is_empty() {
        set.insert(lexmin_index(pts));
    }
    let mut out: Vec<usize> = set.into_iter().collect();
    out.sort_unstable();
    out
}

/// Index of the lexicographically-smallest point — a traversal-independent seam.
pub fn lexmin_index(pts: &[V]) -> usize {
    let mut best = 0;
    for i in 1..pts.len() {
        if pts[i] < pts[best] {
            best = i;
        }
    }
    best
}

fn close(a: V, b: V) -> bool {
    (a.0 - b.0).abs() < 1e-3 && (a.1 - b.1).abs() < 1e-3
}

/// A pinned junction is the only point left at exact integer coordinates.
fn is_int(p: V) -> bool {
    p.0.fract() == 0.0 && p.1.fract() == 0.0
}

/// Bilinear sample of a single-channel map, clamped at the borders.
fn bilinear(map: &[f32], w: usize, h: usize, x: f64, y: f64) -> f32 {
    if w == 0 || h == 0 {
        return 0.0;
    }
    let xc = x.clamp(0.0, (w - 1) as f64);
    let yc = y.clamp(0.0, (h - 1) as f64);
    let x0 = xc.floor() as usize;
    let y0 = yc.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let tx = (xc - x0 as f64) as f32;
    let ty = (yc - y0 as f64) as f32;
    let a = map[y0 * w + x0];
    let b = map[y0 * w + x1];
    let c = map[y1 * w + x0];
    let d = map[y1 * w + x1];
    let top = a + (b - a) * tx;
    let bot = c + (d - c) * tx;
    top + (bot - top) * ty
}

#[cfg(test)]
mod tests {
    use super::*;

    // A 7-wide edge map with a vertical ridge at column 4.
    fn ridge() -> (Vec<f32>, usize, usize) {
        let (w, h) = (7usize, 7usize);
        let mut m = vec![0f32; w * h];
        for y in 0..h {
            m[y * w + 4] = 1.0;
            m[y * w + 3] = 0.4;
            m[y * w + 5] = 0.4;
        }
        (m, w, h)
    }

    #[test]
    fn snaps_vertex_onto_ridge() {
        let (m, w, h) = ridge();
        let r = Refiner::new(&m, w, h, RefineConfig::default());
        // A vertex one pixel off the ridge should move toward column 4.
        let p = r.snap((3.0, 3.0));
        assert!((p.0 - 4.0).abs() < 0.5, "expected snap toward x=4, got {p:?}");
        assert!((p.1 - 3.0).abs() < 1e-6, "y should be unchanged on a vertical ridge");
    }

    #[test]
    fn leaves_vertex_in_flat_region() {
        let m = vec![0.0f32; 49]; // no edges
        let r = Refiner::new(&m, 7, 7, RefineConfig::default());
        let p = r.snap((3.0, 3.0));
        assert_eq!(p, (3.0, 3.0));
    }

    #[test]
    fn refine_loop_keeps_point_count_stable_and_sorted_corners() {
        let (m, w, h) = ridge();
        let r = Refiner::new(&m, w, h, RefineConfig::default());
        let loop_pts = vec![(1, 1), (5, 1), (5, 5), (1, 5)];
        let (pts, corners) = r.refine_loop(&loop_pts, 0.0, &HashSet::new());
        assert!(pts.len() <= 4 && !pts.is_empty());
        assert!(corners.windows(2).all(|w| w[0] < w[1]), "corners ascending");
        assert!(corners.iter().all(|&c| c < pts.len()));
    }
}
