//! Gradient detection — linear & radial fills for the flat tracer.
//!
//! The tracer segments the *quantized* image, so within a region every pixel is
//! one flat palette color. But a region (or a run of adjacent regions that
//! quantization split into bands) often corresponds to a smoothly varying area
//! of the *original* image — a gradient fill. This module:
//!
//!   1. accumulates additive least-squares **moments** per component over the
//!      original colors;
//!   2. **merges** adjacent components whose union is well modeled by one linear
//!      gradient (validated straight from the summed moments — no re-scan), so a
//!      ramp banded into N flat regions becomes one shared gradient;
//!   3. fits each group's gradient — **linear** (affine color field, axis = the
//!      principal eigenvector of the channel slopes) or, when that fails, a
//!      **radial** model (color affine in distance from the group centroid) —
//!      and picks whichever fits tighter.
//!
//! [`fit_all`] returns one optional [`Fill`] per component (the same gradient is
//! handed to every member of a merged group; the SVG layer dedups it). A region
//! that's too small, too flat, or poorly modeled stays solid (`None`).

use std::collections::HashSet;

use crate::segment::Segmentation;
use crate::svg::{Fill, LinearGradient, RadialGradient};

#[derive(Debug, Clone)]
pub struct GradientConfig {
    /// Don't emit a gradient for a (merged) region smaller than this.
    pub min_area: u32,
    /// Minimum color span (max per-channel change across the region, in 0..255)
    /// for a gradient to be worth emitting — below this the region is "flat".
    pub min_delta: f32,
    /// Maximum RMS residual (0..255, over pixels & channels) for a fit to be
    /// accepted. Above this the region is textured / non-(linear|radial).
    pub max_residual: f32,
}

impl Default for GradientConfig {
    fn default() -> Self {
        Self {
            min_area: 64,
            min_delta: 16.0,
            max_residual: 12.0,
        }
    }
}

/// Smallest per-channel color std (0..255) a merge candidate must have, to stop
/// perfectly-flat same-color regions from chaining into giant useless groups.
/// Well below a single quantization band's intrinsic spread, so it never blocks a
/// real gradient.
const MERGE_MIN_STD: f64 = 2.0;

/// Additive least-squares moments over a set of pixels (positions + colors). Sums
/// combine by addition, so a merged region's fit comes straight from `a + b`.
#[derive(Clone, Default)]
struct Moments {
    n: f64,
    sx: f64,
    sy: f64,
    sxx: f64,
    sxy: f64,
    syy: f64,
    sv: [f64; 3],
    svx: [f64; 3],
    svy: [f64; 3],
    svv: [f64; 3],
}

impl Moments {
    fn add_pixel(&mut self, fx: f64, fy: f64, rgb: [f64; 3]) {
        self.n += 1.0;
        self.sx += fx;
        self.sy += fy;
        self.sxx += fx * fx;
        self.sxy += fx * fy;
        self.syy += fy * fy;
        for (c, &v) in rgb.iter().enumerate() {
            self.sv[c] += v;
            self.svx[c] += v * fx;
            self.svy[c] += v * fy;
            self.svv[c] += v * v;
        }
    }

    fn merge(&mut self, o: &Moments) {
        self.n += o.n;
        self.sx += o.sx;
        self.sy += o.sy;
        self.sxx += o.sxx;
        self.sxy += o.sxy;
        self.syy += o.syy;
        for c in 0..3 {
            self.sv[c] += o.sv[c];
            self.svx[c] += o.svx[c];
            self.svy[c] += o.svy[c];
            self.svv[c] += o.svv[c];
        }
    }

    /// Largest per-channel color standard deviation (0..255).
    fn color_std(&self) -> f64 {
        let n = self.n.max(1.0);
        let mut mx = 0f64;
        for c in 0..3 {
            let var = (self.svv[c] / n - (self.sv[c] / n).powi(2)).max(0.0);
            mx = mx.max(var.sqrt());
        }
        mx
    }

    fn centroid(&self) -> (f64, f64) {
        let n = self.n.max(1.0);
        (self.sx / n, self.sy / n)
    }
}

/// An affine color fit `c ≈ a + b·x + d·y` plus the gradient axis and its RMS
/// residual — everything derivable from moments alone.
struct LinFit {
    coef: [[f64; 3]; 3], // coef[channel] = [a, b, d]
    ux: f64,
    uy: f64,
    rmse: f64,
}

/// Fit the affine color model from moments. `None` if spatially degenerate.
fn fit_linear(m: &Moments) -> Option<LinFit> {
    if m.n < 3.0 {
        return None;
    }
    let mat = [
        [m.n, m.sx, m.sy],
        [m.sx, m.sxx, m.sxy],
        [m.sy, m.sxy, m.syy],
    ];
    let inv = invert3(mat)?;
    let mut coef = [[0f64; 3]; 3];
    let mut rss = 0f64;
    for (c, cf) in coef.iter_mut().enumerate() {
        let rhs = [m.sv[c], m.svx[c], m.svy[c]];
        let beta = [
            inv[0][0] * rhs[0] + inv[0][1] * rhs[1] + inv[0][2] * rhs[2],
            inv[1][0] * rhs[0] + inv[1][1] * rhs[1] + inv[1][2] * rhs[2],
            inv[2][0] * rhs[0] + inv[2][1] * rhs[1] + inv[2][2] * rhs[2],
        ];
        *cf = beta;
        // RSS = Σc² − βᵀ·rhs (residual sum of squares from normal-equation moments).
        rss += (m.svv[c] - (beta[0] * rhs[0] + beta[1] * rhs[1] + beta[2] * rhs[2])).max(0.0);
    }
    let rmse = (rss / (m.n * 3.0)).sqrt();
    let (mut s00, mut s01, mut s11) = (0f64, 0f64, 0f64);
    for cf in &coef {
        let (b, d) = (cf[1], cf[2]);
        s00 += b * b;
        s01 += b * d;
        s11 += d * d;
    }
    let (ux, uy) = principal_eigvec(s00, s01, s11)?;
    Some(LinFit { coef, ux, uy, rmse })
}

/// Per-root accumulators that need a pixel pass (min/max ranges; radial sums),
/// because — unlike the moments — they aren't additive.
#[derive(Clone)]
struct RangeAcc {
    tmin: f64,
    tmax: f64,
    sr: f64,
    srr: f64,
    scr: [f64; 3],
    rmin: f64,
    rmax: f64,
}

impl Default for RangeAcc {
    fn default() -> Self {
        Self {
            tmin: f64::INFINITY,
            tmax: f64::NEG_INFINITY,
            sr: 0.0,
            srr: 0.0,
            scr: [0.0; 3],
            rmin: f64::INFINITY,
            rmax: f64::NEG_INFINITY,
        }
    }
}

/// Union-find over components (with path compression + union by rank).
struct Dsu {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl Dsu {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }
    fn find(&mut self, x: usize) -> usize {
        let mut r = x;
        while self.parent[r] != r {
            r = self.parent[r];
        }
        let mut c = x;
        while self.parent[c] != r {
            let nx = self.parent[c];
            self.parent[c] = r;
            c = nx;
        }
        r
    }
    /// Union two roots, returning the new root.
    fn union(&mut self, ra: usize, rb: usize) -> usize {
        if ra == rb {
            return ra;
        }
        match self.rank[ra].cmp(&self.rank[rb]) {
            std::cmp::Ordering::Less => {
                self.parent[ra] = rb;
                rb
            }
            std::cmp::Ordering::Greater => {
                self.parent[rb] = ra;
                ra
            }
            std::cmp::Ordering::Equal => {
                self.parent[rb] = ra;
                self.rank[ra] += 1;
                ra
            }
        }
    }
}

/// Fit a gradient ([`Fill::Linear`] or [`Fill::Radial`]) for every component of
/// `seg` against the `original` (pre-quantization) colors, merging adjacent
/// components that share one gradient. Returns one entry per component: `Some`
/// gradient fill, or `None` to keep the flat color. `original` is RGBA with the
/// same dimensions as `seg`.
pub fn fit_all(seg: &Segmentation, original: &[u8], cfg: &GradientConfig) -> Vec<Option<Fill>> {
    let nc = seg.num_components;
    let mut out = vec![None; nc];
    let (w, h) = (seg.width, seg.height);
    if nc == 0 || original.len() < w * h * 4 {
        return out;
    }

    // --- pass 1: per-component moments + 4-neighbour adjacency (opaque only) ---
    let opaque = |c: usize| seg.component_color[c] != 0;
    let mut mom = vec![Moments::default(); nc];
    let mut adj: HashSet<(u32, u32)> = HashSet::new();
    for y in 0..h {
        for x in 0..w {
            let p = y * w + x;
            let c = seg.labels[p] as usize;
            if !opaque(c) {
                continue;
            }
            let rgb = [
                original[p * 4] as f64,
                original[p * 4 + 1] as f64,
                original[p * 4 + 2] as f64,
            ];
            mom[c].add_pixel(x as f64, y as f64, rgb);
            let mut link = |q: usize| {
                let qc = seg.labels[q] as usize;
                if qc != c && opaque(qc) {
                    let (a, b) = (c.min(qc) as u32, c.max(qc) as u32);
                    adj.insert((a, b));
                }
            };
            if x + 1 < w {
                link(p + 1);
            }
            if y + 1 < h {
                link(p + w);
            }
        }
    }

    // --- merge adjacent components whose union fits one linear gradient ---
    let mut adj: Vec<(u32, u32)> = adj.into_iter().collect();
    adj.sort_unstable(); // determinism
    let mut dsu = Dsu::new(nc);
    let max_rmse = cfg.max_residual as f64;
    loop {
        let mut changed = false;
        for &(a, b) in &adj {
            let ra = dsu.find(a as usize);
            let rb = dsu.find(b as usize);
            if ra == rb {
                continue;
            }
            let mut u = mom[ra].clone();
            u.merge(&mom[rb]);
            if u.color_std() < MERGE_MIN_STD {
                continue; // both ~flat & same color — nothing to gain
            }
            match fit_linear(&u) {
                Some(lf) if lf.rmse <= max_rmse => {
                    let r = dsu.union(ra, rb);
                    mom[r] = u;
                    changed = true;
                }
                _ => {}
            }
        }
        if !changed {
            break;
        }
    }

    // --- per-root linear fit + centroid (for the range pass) ---
    let mut root_lin: Vec<Option<LinFit>> = (0..nc).map(|_| None).collect();
    let mut root_cxy = vec![(0f64, 0f64); nc];
    for c in 0..nc {
        if !opaque(c) || dsu.find(c) != c {
            continue;
        }
        // Centroid is always available (radial needs it even when the linear axis
        // is degenerate, e.g. a symmetric radial field).
        root_cxy[c] = mom[c].centroid();
        root_lin[c] = fit_linear(&mom[c]);
    }

    // --- pass 2: per-root projection range (linear) + radial sums ---
    let mut acc = vec![RangeAcc::default(); nc];
    for y in 0..h {
        for x in 0..w {
            let p = y * w + x;
            let c = seg.labels[p] as usize;
            if !opaque(c) {
                continue;
            }
            let r = dsu.find(c);
            let (fx, fy) = (x as f64, y as f64);
            let a = &mut acc[r];
            // Linear projection range — only when the root has a (non-degenerate) axis.
            if let Some(lf) = &root_lin[r] {
                let t = lf.ux * fx + lf.uy * fy;
                a.tmin = a.tmin.min(t);
                a.tmax = a.tmax.max(t);
            }
            // Radial sums — always (centroid-based), so a radial-only region still
            // gets its candidate.
            let (cx, cy) = root_cxy[r];
            let dist = ((fx - cx).powi(2) + (fy - cy).powi(2)).sqrt();
            a.sr += dist;
            a.srr += dist * dist;
            a.rmin = a.rmin.min(dist);
            a.rmax = a.rmax.max(dist);
            for (ch, scr) in a.scr.iter_mut().enumerate() {
                *scr += original[p * 4 + ch] as f64 * dist;
            }
        }
    }

    // --- decide each root's fill, then hand it to every member ---
    let mut root_fill: Vec<Option<Fill>> = (0..nc).map(|_| None).collect();
    for c in 0..nc {
        if !opaque(c) || dsu.find(c) != c {
            continue;
        }
        if (mom[c].n as u32) < cfg.min_area {
            continue;
        }
        let lin = root_lin[c]
            .as_ref()
            .and_then(|lf| build_linear(lf, &mom[c], &acc[c], cfg));
        let rad = build_radial(&mom[c], &acc[c], cfg, root_cxy[c]);
        root_fill[c] = match (lin, rad) {
            (Some((fl, rl)), Some((fr, rr))) => Some(if rl <= rr { fl } else { fr }),
            (Some((fl, _)), None) => Some(fl),
            (None, Some((fr, _))) => Some(fr),
            (None, None) => None,
        };
    }

    // --- radial-band merge: concentric radial bands → one shared radial ---
    // The band-merge above is linear-only, so a radial gradient quantized into
    // rings stays N separate radials with faint seams. Group adjacent radial
    // roots whose centers coincide, then **re-scan** each group from its common
    // center to fit one shared radial (deduped to a single <defs> entry).
    //
    // The re-scan is essential and the radial sums are NOT reused: r-sums (Σr,
    // Σr², Σcr) involve a non-translatable sqrt, so summing per-root sums taken
    // from different centroids would fit a biased model the residual can't catch.
    // Grouping is pure centroid proximity (concentric bands coincide); the
    // re-scanned fit then soundly accepts or rejects each group.
    let is_radial = |rf: &Option<Fill>| matches!(rf, Some(Fill::Radial(_)));
    let center_tol = (w.min(h) as f64 * 0.05).max(3.0);
    let redges: Vec<(usize, usize)> = adj
        .iter()
        .filter_map(|&(a, b)| {
            let (ra, rb) = (dsu.find(a as usize), dsu.find(b as usize));
            let close = (root_cxy[ra].0 - root_cxy[rb].0).hypot(root_cxy[ra].1 - root_cxy[rb].1)
                <= center_tol;
            if ra != rb && is_radial(&root_fill[ra]) && is_radial(&root_fill[rb]) && close {
                Some((ra, rb))
            } else {
                None
            }
        })
        .collect();
    if !redges.is_empty() {
        // Group radial roots by adjacency + centroid proximity (no fit in the loop).
        let mut rdsu = Dsu::new(nc);
        for &(a, b) in &redges {
            let (ra, rb) = (rdsu.find(a), rdsu.find(b));
            rdsu.union(ra, rb);
        }
        // Per-group combined moments (additive ⇒ exact group centroid) + a pixel
        // RE-SCAN of radial sums measured from that common center.
        let mut gmom = vec![Moments::default(); nc];
        for (c, rf) in root_fill.iter().enumerate() {
            if is_radial(rf) {
                let mc = mom[c].clone();
                gmom[rdsu.find(c)].merge(&mc);
            }
        }
        let mut gcenter = vec![(0f64, 0f64); nc];
        for (c, gc) in gcenter.iter_mut().enumerate() {
            if rdsu.find(c) == c {
                *gc = gmom[c].centroid();
            }
        }
        let mut gacc = vec![RangeAcc::default(); nc];
        for y in 0..h {
            for x in 0..w {
                let p = y * w + x;
                let cc = seg.labels[p] as usize;
                if !opaque(cc) || !is_radial(&root_fill[dsu.find(cc)]) {
                    continue;
                }
                let rr = rdsu.find(cc);
                let (cx, cy) = gcenter[rr];
                let dist = ((x as f64 - cx).powi(2) + (y as f64 - cy).powi(2)).sqrt();
                let a = &mut gacc[rr];
                a.sr += dist;
                a.srr += dist * dist;
                a.rmin = a.rmin.min(dist);
                a.rmax = a.rmax.max(dist);
                for (ch, scr) in a.scr.iter_mut().enumerate() {
                    *scr += original[p * 4 + ch] as f64 * dist;
                }
            }
        }
        // Fit each group from the correctly-centered sums; share it if it holds.
        let mut shared: Vec<Option<Fill>> = (0..nc).map(|_| None).collect();
        for (c, sh) in shared.iter_mut().enumerate() {
            if rdsu.find(c) == c && (gmom[c].n as u32) >= cfg.min_area {
                *sh = build_radial(&gmom[c], &gacc[c], cfg, gcenter[c]).map(|(f, _)| f);
            }
        }
        let rroot: Vec<usize> = (0..nc).map(|c| rdsu.find(c)).collect();
        for (c, rf) in root_fill.iter_mut().enumerate() {
            if matches!(rf, Some(Fill::Radial(_))) {
                if let Some(f) = &shared[rroot[c]] {
                    *rf = Some(f.clone());
                }
            }
        }
    }

    for c in 0..nc {
        if !opaque(c) {
            continue;
        }
        out[c] = root_fill[dsu.find(c)].clone();
    }
    out
}

fn clamp_color(v: [f64; 3]) -> [u8; 3] {
    [
        v[0].round().clamp(0.0, 255.0) as u8,
        v[1].round().clamp(0.0, 255.0) as u8,
        v[2].round().clamp(0.0, 255.0) as u8,
    ]
}

/// Build a linear-gradient fill (and its RMS residual) from a root's fit + range,
/// or `None` if the residual/span/delta don't clear the thresholds.
fn build_linear(
    lf: &LinFit,
    m: &Moments,
    a: &RangeAcc,
    cfg: &GradientConfig,
) -> Option<(Fill, f64)> {
    if lf.rmse > cfg.max_residual as f64 {
        return None;
    }
    let span = a.tmax - a.tmin;
    if span < 1e-3 {
        return None;
    }
    let mut max_delta = 0f64;
    for cf in &lf.coef {
        let slope_t = cf[1] * lf.ux + cf[2] * lf.uy;
        max_delta = max_delta.max((slope_t * span).abs());
    }
    if (max_delta as f32) < cfg.min_delta {
        return None;
    }
    let (cx, cy) = m.centroid();
    let tc = lf.ux * cx + lf.uy * cy;
    let p1 = (cx + (a.tmin - tc) * lf.ux, cy + (a.tmin - tc) * lf.uy);
    let p2 = (cx + (a.tmax - tc) * lf.ux, cy + (a.tmax - tc) * lf.uy);
    let eval = |pt: (f64, f64)| -> [u8; 3] {
        clamp_color([
            lf.coef[0][0] + lf.coef[0][1] * pt.0 + lf.coef[0][2] * pt.1,
            lf.coef[1][0] + lf.coef[1][1] * pt.0 + lf.coef[1][2] * pt.1,
            lf.coef[2][0] + lf.coef[2][1] * pt.0 + lf.coef[2][2] * pt.1,
        ])
    };
    Some((
        Fill::Linear(LinearGradient {
            x1: p1.0,
            y1: p1.1,
            x2: p2.0,
            y2: p2.1,
            stops: vec![(0.0, eval(p1)), (1.0, eval(p2))],
        }),
        lf.rmse,
    ))
}

/// Build a radial-gradient fill (color affine in distance from the centroid) and
/// its RMS residual, or `None` if it doesn't clear the thresholds. The center is
/// the group centroid, so off-center radials simply won't fit (→ stay solid).
fn build_radial(
    m: &Moments,
    a: &RangeAcc,
    cfg: &GradientConfig,
    center: (f64, f64),
) -> Option<(Fill, f64)> {
    let n = m.n;
    let rmax = a.rmax;
    if n < 3.0 || rmax < 1e-3 {
        return None;
    }
    // Per-channel 1-D fit c ≈ p + s·r via the 2×2 normal equations
    // [[n, Σr],[Σr, Σr²]]·[p,s] = [Σc, Σcr].
    let det = n * a.srr - a.sr * a.sr;
    if det.abs() < 1e-6 {
        return None;
    }
    let mut p = [0f64; 3];
    let mut s = [0f64; 3];
    let mut rss = 0f64;
    for c in 0..3 {
        let pc = (a.srr * m.sv[c] - a.sr * a.scr[c]) / det;
        let sc = (-a.sr * m.sv[c] + n * a.scr[c]) / det;
        p[c] = pc;
        s[c] = sc;
        rss += (m.svv[c] - pc * m.sv[c] - sc * a.scr[c]).max(0.0);
    }
    let rmse = (rss / (n * 3.0)).sqrt();
    if rmse > cfg.max_residual as f64 {
        return None;
    }
    let vis_span = rmax - a.rmin.max(0.0);
    let mut max_delta = 0f64;
    for &sc in &s {
        max_delta = max_delta.max((sc * vis_span).abs());
    }
    if (max_delta as f32) < cfg.min_delta {
        return None;
    }
    // Stops map offset 0→center (r=0) and 1→rim (r=rmax); the affine model is
    // exact between, so a pixel at radius r reads p + s·r as intended.
    let col0 = clamp_color(p);
    let col1 = clamp_color([p[0] + s[0] * rmax, p[1] + s[1] * rmax, p[2] + s[2] * rmax]);
    Some((
        Fill::Radial(RadialGradient {
            cx: center.0,
            cy: center.1,
            r: rmax,
            stops: vec![(0.0, col0), (1.0, col1)],
        }),
        rmse,
    ))
}

/// Invert a 3×3 matrix; `None` if (near) singular.
fn invert3(m: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let a = m;
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if det.abs() < 1e-6 {
        return None;
    }
    let inv_det = 1.0 / det;
    let mut o = [[0f64; 3]; 3];
    o[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) * inv_det;
    o[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * inv_det;
    o[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * inv_det;
    o[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) * inv_det;
    o[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * inv_det;
    o[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * inv_det;
    o[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) * inv_det;
    o[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * inv_det;
    o[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * inv_det;
    Some(o)
}

/// Unit principal eigenvector of the symmetric 2×2 `[[a,b],[b,d]]`; `None` if the
/// largest eigenvalue is ~0 (no directional variation).
fn principal_eigvec(a: f64, b: f64, d: f64) -> Option<(f64, f64)> {
    let tr = (a + d) * 0.5;
    let diff = (a - d) * 0.5;
    let disc = (diff * diff + b * b).sqrt();
    let lambda = tr + disc; // larger eigenvalue
    if lambda < 1e-6 {
        return None;
    }
    let (vx, vy) = if b.abs() > 1e-9 {
        (b, lambda - a)
    } else if a >= d {
        (1.0, 0.0)
    } else {
        (0.0, 1.0)
    };
    let len = (vx * vx + vy * vy).sqrt();
    if len < 1e-12 {
        return None;
    }
    Some((vx / len, vy / len))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a segmentation from an explicit per-pixel component-label map.
    fn seg_from(labels: Vec<u32>, w: usize, h: usize, colors: &[u32]) -> Segmentation {
        let num = colors.len();
        let mut area = vec![0u32; num];
        for &l in &labels {
            area[l as usize] += 1;
        }
        Segmentation {
            width: w,
            height: h,
            labels,
            num_components: num,
            component_color: colors.to_vec(),
            component_area: area,
        }
    }

    fn one_region(w: usize, h: usize) -> Segmentation {
        seg_from(vec![0u32; w * h], w, h, &[1])
    }

    fn ramp_rgba(w: usize, h: usize, gray: bool) -> Vec<u8> {
        let mut px = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let p = (y * w + x) * 4;
                let v = ((x * 255) / (w - 1)) as u8;
                px[p] = v;
                if gray {
                    px[p + 1] = v;
                    px[p + 2] = v;
                }
                px[p + 3] = 255;
            }
        }
        px
    }

    fn as_linear(f: &Fill) -> &LinearGradient {
        match f {
            Fill::Linear(g) => g,
            _ => panic!("expected a linear gradient"),
        }
    }

    #[test]
    fn fits_a_horizontal_ramp() {
        let (w, h) = (32usize, 16usize);
        let px = ramp_rgba(w, h, false);
        let out = fit_all(&one_region(w, h), &px, &GradientConfig::default());
        let g = as_linear(out[0].as_ref().expect("ramp should fit"));
        assert!((g.y1 - g.y2).abs() < 1e-6, "axis horizontal");
        assert!((g.x1 - g.x2).abs() > w as f64 * 0.5, "axis spans width");
        let reds = [g.stops[0].1[0], g.stops[1].1[0]];
        assert!(reds.iter().any(|&r| r < 30) && reds.iter().any(|&r| r > 225), "red spans 0..255: {reds:?}");
    }

    #[test]
    fn flat_region_is_not_a_gradient() {
        let (w, h) = (20usize, 20usize);
        let mut px = vec![0u8; w * h * 4];
        for i in 0..w * h {
            px[i * 4] = 120;
            px[i * 4 + 1] = 80;
            px[i * 4 + 2] = 40;
            px[i * 4 + 3] = 255;
        }
        let out = fit_all(&one_region(w, h), &px, &GradientConfig::default());
        assert!(out[0].is_none(), "flat → solid");
    }

    #[test]
    fn small_region_is_skipped() {
        let (w, h) = (4usize, 4usize);
        let px = ramp_rgba(w, h, false);
        let out = fit_all(&one_region(w, h), &px, &GradientConfig::default());
        assert!(out[0].is_none(), "16 px < min_area");
    }

    #[test]
    fn diagonal_ramp_axis_is_diagonal() {
        let (w, h) = (24usize, 24usize);
        let mut px = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let p = (y * w + x) * 4;
                let v = ((x + y) * 255 / (w + h - 2)) as u8;
                px[p] = v;
                px[p + 1] = v;
                px[p + 2] = v;
                px[p + 3] = 255;
            }
        }
        let out = fit_all(&one_region(w, h), &px, &GradientConfig::default());
        let g = as_linear(out[0].as_ref().expect("diagonal ramp should fit"));
        let dx = (g.x2 - g.x1).abs();
        let dy = (g.y2 - g.y1).abs();
        assert!(dx > 1.0 && dy > 1.0 && (dx / dy - 1.0).abs() < 0.4, "≈45°: dx={dx} dy={dy}");
    }

    #[test]
    fn merges_two_bands_into_one_shared_gradient() {
        // 32×16, split into two components at x=16 (as if quantized into 2 bands),
        // but the original is one continuous red ramp. The two bands must merge
        // into a single shared gradient spanning the full width.
        let (w, h) = (32usize, 16usize);
        let mut labels = vec![0u32; w * h];
        for y in 0..h {
            for x in 0..w {
                labels[y * w + x] = if x < w / 2 { 0 } else { 1 };
            }
        }
        let seg = seg_from(labels, w, h, &[1, 2]);
        let px = ramp_rgba(w, h, false);
        let out = fit_all(&seg, &px, &GradientConfig::default());
        let g0 = as_linear(out[0].as_ref().expect("band 0 gradient"));
        let g1 = as_linear(out[1].as_ref().expect("band 1 gradient"));
        // Same shared gradient: identical endpoints spanning ~full width.
        assert!((g0.x1 - g1.x1).abs() < 1e-9 && (g0.x2 - g1.x2).abs() < 1e-9, "shared gradient");
        assert!((g0.x2 - g0.x1).abs() > w as f64 * 0.5, "spans the whole ramp, not one band");
    }

    #[test]
    fn fits_a_radial_cone() {
        // Brightness falls off with distance from the center — a symmetric cone.
        // A linear plane fits it flat (no axis slope → rejected), so the radial
        // model must win.
        let (w, h) = (28usize, 28usize);
        let (cx, cy) = (13.5f64, 13.5f64);
        let mut px = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let p = (y * w + x) * 4;
                let r = ((x as f64 - cx).powi(2) + (y as f64 - cy).powi(2)).sqrt();
                let v = (255.0 - 11.0 * r).clamp(0.0, 255.0) as u8;
                px[p] = v;
                px[p + 1] = v;
                px[p + 2] = v;
                px[p + 3] = 255;
            }
        }
        let out = fit_all(&one_region(w, h), &px, &GradientConfig::default());
        match out[0].as_ref().expect("cone should fit a gradient") {
            Fill::Radial(g) => {
                assert!((g.cx - cx).abs() < 2.0 && (g.cy - cy).abs() < 2.0, "center ≈ middle");
                assert!(g.r > w as f64 * 0.3, "radius spans the region");
            }
            Fill::Linear(_) => panic!("a symmetric cone should be radial, not linear"),
            Fill::Solid(_) => unreachable!(),
        }
    }

    #[test]
    fn merges_concentric_radial_bands() {
        // A radial gradient quantized into an inner disk (comp 0) and an outer
        // ring (comp 1). Both fit radial on their own; radial-band merging must
        // hand them ONE shared radial (identical center + radius).
        let (w, h) = (40usize, 40usize);
        let (cx, cy) = (19.5f64, 19.5f64);
        let split = 12.0;
        let mut labels = vec![0u32; w * h];
        let mut px = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let r = ((x as f64 - cx).powi(2) + (y as f64 - cy).powi(2)).sqrt();
                let p = (y * w + x) * 4;
                let v = (240.0 - 7.0 * r).clamp(0.0, 255.0) as u8;
                px[p] = v;
                px[p + 1] = v;
                px[p + 2] = v;
                px[p + 3] = 255;
                labels[y * w + x] = if r < split { 0 } else { 1 };
            }
        }
        let seg = seg_from(labels, w, h, &[1, 2]);
        let out = fit_all(&seg, &px, &GradientConfig::default());
        let radial = |f: &Fill| match f {
            Fill::Radial(g) => (g.cx, g.cy, g.r),
            _ => panic!("expected radial"),
        };
        let g0 = radial(out[0].as_ref().expect("inner radial"));
        let g1 = radial(out[1].as_ref().expect("outer radial"));
        assert!(
            (g0.0 - g1.0).abs() < 1e-9 && (g0.1 - g1.1).abs() < 1e-9 && (g0.2 - g1.2).abs() < 1e-9,
            "both bands must share one radial: {g0:?} vs {g1:?}"
        );
        assert!((g0.0 - cx).abs() < 2.0 && (g0.1 - cy).abs() < 2.0, "center ≈ middle");
    }

    #[test]
    fn distinct_radials_do_not_merge() {
        // Two side-by-side radial cones with DIFFERENT centers must NOT collapse
        // into one shared radial — the centroid-proximity gate keeps them apart,
        // and even if grouped the re-scanned fit would reject the union.
        let (w, h) = (60usize, 30usize);
        let centers = [(15.0f64, 15.0f64), (45.0f64, 15.0f64)];
        let mut labels = vec![0u32; w * h];
        let mut px = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let comp = if x < 30 { 0 } else { 1 };
                let (cx, cy) = centers[comp];
                let r = ((x as f64 - cx).powi(2) + (y as f64 - cy).powi(2)).sqrt();
                let v = (240.0 - 8.0 * r).clamp(0.0, 255.0) as u8;
                let p = (y * w + x) * 4;
                px[p] = v;
                px[p + 1] = v;
                px[p + 2] = v;
                px[p + 3] = 255;
                labels[y * w + x] = comp as u32;
            }
        }
        let seg = seg_from(labels, w, h, &[1, 2]);
        let out = fit_all(&seg, &px, &GradientConfig::default());
        let cx = |f: &Fill| match f {
            Fill::Radial(g) => g.cx,
            _ => panic!("expected radial"),
        };
        let c0 = cx(out[0].as_ref().expect("left radial"));
        let c1 = cx(out[1].as_ref().expect("right radial"));
        assert!((c0 - c1).abs() > 20.0, "distinct radials keep distinct centers: {c0} vs {c1}");
    }

    #[test]
    fn unrelated_neighbors_do_not_merge() {
        // Two adjacent flat regions of very different colors must NOT merge (a
        // step is not a linear gradient) and neither is a gradient on its own.
        let (w, h) = (32usize, 16usize);
        let mut labels = vec![0u32; w * h];
        let mut px = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let p = (y * w + x) * 4;
                if x < w / 2 {
                    labels[y * w + x] = 0;
                    px[p] = 230; // red
                } else {
                    labels[y * w + x] = 1;
                    px[p + 2] = 230; // blue
                }
                px[p + 3] = 255;
            }
        }
        let seg = seg_from(labels, w, h, &[1, 2]);
        let out = fit_all(&seg, &px, &GradientConfig::default());
        assert!(out[0].is_none() && out[1].is_none(), "flat step → both stay solid");
    }
}
