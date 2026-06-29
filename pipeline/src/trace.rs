//! Owned tracer — orchestrates the Level-2 pipeline end to end:
//! quantized raster → palette indices → connected-components segmentation →
//! contour extraction → RDP simplification → layered polygon SVG.
//!
//! The input is expected to already be color-reduced (see [`crate::quantize`]);
//! the tracer derives its palette from the distinct colors present, so feeding
//! it a full-color photo would produce one region per unique color.

use std::collections::{HashMap, HashSet};

use crate::contour::contours_of;
use crate::curvefit;
use crate::gradient::{fit_all, GradientConfig};
use crate::refine::Refiner;
use crate::segment::{segment, Segmentation};
use crate::simplify::simplify_closed;
use crate::svg::{polygon_subpath, polygon_subpath_f, to_svg, to_svg_layered, Fill, Region};

/// Lattice points where 3+ distinct regions (including "outside") meet. Pinning
/// these in simplification keeps adjacent regions' shared boundaries aligned —
/// without it, per-region RDP drops junctions inconsistently and leaves gaps.
fn junction_set(labels: &[u32], w: usize, h: usize) -> HashSet<(i32, i32)> {
    let mut set = HashSet::new();
    let lab = |px: i64, py: i64| -> u32 {
        if px < 0 || py < 0 || px >= w as i64 || py >= h as i64 {
            u32::MAX // "outside" counts as its own region
        } else {
            labels[py as usize * w + px as usize]
        }
    };
    for ly in 0..=h as i64 {
        for lx in 0..=w as i64 {
            let mut vals = [
                lab(lx - 1, ly - 1),
                lab(lx, ly - 1),
                lab(lx - 1, ly),
                lab(lx, ly),
            ];
            vals.sort_unstable();
            let distinct = 1
                + (vals[0] != vals[1]) as usize
                + (vals[1] != vals[2]) as usize
                + (vals[2] != vals[3]) as usize;
            if distinct >= 3 {
                set.insert((lx as i32, ly as i32));
            }
        }
    }
    set
}

#[derive(Debug, Clone)]
pub struct TraceConfig {
    /// Pixels with alpha at or below this are treated as transparent (not drawn).
    pub alpha_threshold: u8,
    /// Merge regions smaller than this many pixels into their largest neighbour.
    pub min_area: u32,
    /// RDP simplification tolerance in pixels (0 = keep exact staircase edges).
    pub simplify: f64,
    /// Emit the largest region as a background rect instead of a full polygon.
    pub background: bool,
    /// Fit cubic Béziers to the contours instead of emitting polygons.
    pub curve: bool,
    /// Corner-detection threshold (degrees) for curve fitting; sharper turns
    /// stay as crisp corners.
    pub corner_threshold: f64,
    /// Curve-fit error tolerance in pixels.
    pub curve_error: f64,
    /// Detect smooth color ramps in regions and emit `<linearGradient>` fills
    /// instead of flat colors. Requires the original (pre-quantization) pixels to
    /// be passed to [`trace_rgba`]; ignored otherwise.
    pub gradient: bool,
}

impl Default for TraceConfig {
    fn default() -> Self {
        Self {
            alpha_threshold: 0,
            min_area: 4,
            simplify: 1.2,
            background: true,
            curve: false,
            corner_threshold: 80.0,
            curve_error: 2.0,
            gradient: false,
        }
    }
}

/// Build palette indices from an RGBA buffer: 0 = transparent, 1.. = the
/// distinct opaque colors in first-seen order (the returned palette).
fn palette_indices(pixels: &[u8], n: usize, alpha_threshold: u8) -> (Vec<u32>, Vec<[u8; 3]>) {
    let mut idx = vec![0u32; n];
    let mut palette: Vec<[u8; 3]> = Vec::new();
    let mut map: HashMap<u32, u32> = HashMap::new();
    for i in 0..n {
        let a = pixels[i * 4 + 3];
        if a <= alpha_threshold {
            continue; // idx stays 0
        }
        let (r, g, b) = (pixels[i * 4], pixels[i * 4 + 1], pixels[i * 4 + 2]);
        let key = (r as u32) << 16 | (g as u32) << 8 | b as u32;
        let ci = *map.entry(key).or_insert_with(|| {
            palette.push([r, g, b]);
            palette.len() as u32 // first opaque color -> 1
        });
        idx[i] = ci;
    }
    (idx, palette)
}

/// Trace every opaque component of a segmentation into a [`Region`], optionally
/// skipping one component (the background rect). Shared by the flat and layered
/// tracers.
fn regions_of(
    seg: &Segmentation,
    palette: &[[u8; 3]],
    cfg: &TraceConfig,
    skip: Option<usize>,
    refine: Option<&Refiner>,
    fills: Option<&[Option<Fill>]>,
    group_rep: Option<&[u32]>,
) -> Vec<Region> {
    let bboxes = seg.bboxes();
    // Shared junctions, pinned in simplification so adjacent regions tile gaplessly.
    let junctions = junction_set(&seg.labels, seg.width, seg.height);
    let mut regions: Vec<Region> = Vec::new();
    for c in 0..seg.num_components {
        if seg.component_color[c] == 0 || Some(c) == skip {
            continue;
        }
        // Geometry-union: a component that's been folded into a gradient group
        // (its rep is another component) has no pixels of its own under the
        // relabeled map — only the group's representative draws the union.
        if group_rep.is_some_and(|g| g[c] as usize != c) {
            continue;
        }
        let color = palette[(seg.component_color[c] - 1) as usize];
        // Use the precomputed per-component gradient fill when present; fall back
        // to the flat palette color otherwise.
        let fill = fills
            .and_then(|f| f.get(c).cloned().flatten())
            .unwrap_or(Fill::Solid(color));
        let raw_loops = contours_of(&seg.labels, seg.width, seg.height, c as u32, bboxes[c]);
        let mut subpaths = Vec::with_capacity(raw_loops.len());
        for lp in raw_loops {
            let sub = if let Some(r) = refine {
                // Edge-guided: snap the dense contour onto true edges, simplify,
                // then curve-fit (or emit the snapped float polygon).
                let (pts, corners) = r.refine_loop(&lp, cfg.simplify, &junctions);
                if pts.len() < 3 {
                    continue;
                }
                if cfg.curve && pts.len() >= 4 {
                    curvefit::fit_loop_pts(&pts, &corners, cfg.curve_error)
                } else {
                    polygon_subpath_f(&pts)
                }
            } else {
                // Simplify, but never let RDP collapse a real loop to nothing —
                // fall back to the exact contour so small holes/regions survive.
                let simp = simplify_closed(&lp, cfg.simplify);
                let poly = if simp.len() >= 3 { simp } else { lp };
                if poly.len() < 3 {
                    continue;
                }
                if cfg.curve && poly.len() >= 4 {
                    curvefit::fit_loop(&poly, cfg.corner_threshold, cfg.curve_error)
                } else {
                    polygon_subpath(&poly)
                }
            };
            subpaths.push(sub);
        }
        if !subpaths.is_empty() {
            regions.push(Region { fill, subpaths });
        }
    }
    regions
}

/// Optionally fit per-component gradient fills against the original
/// (pre-quantization) pixels, then apply the geometry-union relabel in place.
///
/// When `cfg.gradient` is on and `original` covers the image, [`fit_all`] fits a
/// gradient per component (merging adjacent quantization bands) and returns a
/// per-component group representative; we relabel `seg.labels` so each gradient
/// group traces as ONE union shape (only its outer boundary), and hand the fills
/// and reps to [`regions_of`]. Returns `(None, None)` when gradients are off or
/// no usable originals were supplied — the caller then paints flat solids.
///
/// Shared by the flat ([`trace_rgba`]) and layered ([`trace_layered`]) tracers;
/// in the layered case it runs once per object layer, so a smoothly-shaded
/// object becomes a gradient fill instead of a flat patch.
fn fit_gradients(
    seg: &mut Segmentation,
    cfg: &TraceConfig,
    original: Option<&[u8]>,
) -> (Option<Vec<Option<Fill>>>, Option<Vec<u32>>) {
    if !cfg.gradient {
        return (None, None);
    }
    let n = seg.width * seg.height;
    let Some(orig) = original.filter(|o| o.len() >= n * 4) else {
        return (None, None);
    };
    let (fills, group_rep) = fit_all(seg, orig, &GradientConfig::default());
    for lab in seg.labels.iter_mut() {
        *lab = group_rep[*lab as usize];
    }
    (Some(fills), Some(group_rep))
}

/// Trace an (already quantized) RGBA buffer into a flat-color SVG document.
/// Pass a [`Refiner`] to snap contours onto a CNN edge map before fitting.
pub fn trace_rgba(
    pixels: &[u8],
    width: usize,
    height: usize,
    cfg: &TraceConfig,
    original: Option<&[u8]>,
    refine: Option<&Refiner>,
) -> String {
    let n = width * height;
    if n == 0 || pixels.len() < n * 4 {
        return to_svg(width, height, None, &[]);
    }

    let (idx, palette) = palette_indices(pixels, n, cfg.alpha_threshold);

    // --- segment + merge speckles ---
    let mut seg = segment(&idx, width, height);
    seg.merge_small(cfg.min_area);

    // Fit gradients against the original colors (merging bands + relabeling for
    // the geometry-union) when enabled; `(None, None)` keeps regions flat.
    let (fills, group_rep) = fit_gradients(&mut seg, cfg, original);

    // --- choose a background region (largest opaque) to lay down as a rect ---
    // Skipped in gradient mode: the largest region may itself be a gradient, so
    // we trace it as a path rather than flattening it to a solid rect.
    let mut bg_region: Option<usize> = None;
    if cfg.background && fills.is_none() {
        let mut best_area = 0u32;
        for c in 0..seg.num_components {
            if seg.component_color[c] != 0 && seg.component_area[c] > best_area {
                best_area = seg.component_area[c];
                bg_region = Some(c);
            }
        }
    }
    let background = bg_region.map(|c| palette[(seg.component_color[c] - 1) as usize]);

    let regions = regions_of(
        &seg,
        &palette,
        cfg,
        bg_region,
        refine,
        fills.as_deref(),
        group_rep.as_deref(),
    );
    to_svg(width, height, background, &regions)
}

/// Trace an (already quantized) RGBA buffer into a *layered* SVG, one
/// `<g data-object="…">` group per object. `instance_id` (length `width*height`)
/// assigns each pixel to a layer: 0 is the background, 1.. are the objects (e.g.
/// from ML instance masks, resolved to a single id per pixel by the caller). The
/// layer count is taken from the largest id present. Each layer is segmented and
/// traced independently, then color-merged, so an object built from several
/// quantized colors becomes one group of color paths. Layers are emitted
/// background-first.
///
/// Pass the `original` (pre-quantization) pixels with `cfg.gradient` set to fit
/// gradient fills per object — a smoothly-shaded object then becomes one
/// `<linearGradient>`/`<radialGradient>` fill (deduped across layers in the
/// shared `<defs>`) instead of a flat patch. `None` keeps every layer flat.
pub fn trace_layered(
    pixels: &[u8],
    width: usize,
    height: usize,
    instance_id: &[u32],
    cfg: &TraceConfig,
    original: Option<&[u8]>,
    refine: Option<&Refiner>,
) -> String {
    let n = width * height;
    if n == 0 || pixels.len() < n * 4 || instance_id.len() < n {
        return to_svg_layered(width, height, &[]);
    }

    let (idx, palette) = palette_indices(pixels, n, cfg.alpha_threshold);

    // Layers run 0..=max id present; absent/fully-occluded ids yield empty masks
    // and are skipped below, so the count is derivable from the map itself.
    let num_instances = instance_id.iter().copied().max().unwrap_or(0);
    let mut layers: Vec<(String, Vec<Region>)> = Vec::new();
    let mut midx = vec![0u32; n];
    for layer in 0..=num_instances {
        // Mask the palette indices down to just this layer's pixels.
        let mut any = false;
        for p in 0..n {
            if instance_id[p] == layer {
                midx[p] = idx[p];
                any |= idx[p] != 0;
            } else {
                midx[p] = 0;
            }
        }
        if !any {
            continue; // layer is empty or fully transparent
        }
        let mut seg = segment(&midx, width, height);
        seg.merge_small(cfg.min_area);
        // Per-layer gradient fitting: each object's region(s) refit against the
        // original colors (geometry-union applies within the layer). Gradients
        // never cross object boundaries — fitting is scoped to this layer's mask.
        let (fills, group_rep) = fit_gradients(&mut seg, cfg, original);
        let regions = regions_of(
            &seg,
            &palette,
            cfg,
            None,
            refine,
            fills.as_deref(),
            group_rep.as_deref(),
        );
        if regions.is_empty() {
            continue;
        }
        let label = if layer == 0 {
            "background".to_string()
        } else {
            format!("object-{layer}")
        };
        layers.push((label, regions));
    }

    to_svg_layered(width, height, &layers)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgba(colors: &[[u8; 4]]) -> Vec<u8> {
        colors.iter().flat_map(|c| c.iter().copied()).collect()
    }

    #[test]
    fn two_flat_regions_produce_valid_svg() {
        // 2x2: top row red, bottom row blue.
        let px = rgba(&[
            [255, 0, 0, 255],
            [255, 0, 0, 255],
            [0, 0, 255, 255],
            [0, 0, 255, 255],
        ]);
        let svg = trace_rgba(&px, 2, 2, &TraceConfig { min_area: 0, simplify: 0.0, ..Default::default() }, None, None);
        assert!(svg.starts_with("<?xml"));
        assert!(svg.contains("<svg"));
        assert!(svg.ends_with("</svg>\n"));
        // Background rect (one color) + one path (the other region).
        assert!(svg.contains("<rect"));
        assert_eq!(svg.matches("<path").count(), 1);
        // Both colors appear.
        assert!(svg.contains("#ff0000"));
        assert!(svg.contains("#0000ff"));
    }

    #[test]
    fn transparent_pixels_are_not_drawn() {
        // One opaque green, three transparent.
        let px = rgba(&[
            [0, 255, 0, 255],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]);
        let svg = trace_rgba(&px, 2, 2, &TraceConfig { background: false, min_area: 0, simplify: 0.0, ..Default::default() }, None, None);
        assert_eq!(svg.matches("<path").count(), 1);
        assert!(svg.contains("#00ff00"));
        assert!(!svg.contains("<rect"));
    }

    #[test]
    fn same_color_regions_merge_into_one_path() {
        // 6x1: red red green red green red. The two disconnected green cells
        // share a color and must merge into ONE <path> (two subpaths); red is
        // the background rect, so the red cells are dropped entirely.
        let r = [200, 0, 0, 255];
        let g = [0, 200, 0, 255];
        let px = rgba(&[r, r, g, r, g, r]);
        let svg = trace_rgba(&px, 6, 1, &TraceConfig { min_area: 0, simplify: 0.0, ..Default::default() }, None, None);
        assert!(svg.contains("<rect")); // red background
        assert_eq!(svg.matches("<path").count(), 1, "the two greens merge to one path");
        assert_eq!(svg.matches('M').count(), 2, "one path, two subpaths");
        assert!(svg.contains("#00c800")); // green
    }

    #[test]
    fn empty_image_is_valid_svg() {
        let svg = trace_rgba(&[], 0, 0, &TraceConfig::default(), None, None);
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn gradient_region_emits_lineargradient() {
        // 16×16 region that quantized to ONE flat color but whose *original*
        // pixels ramp red 0→255 across x — the exact case gradient detection is
        // for. `pixels` (quantized) is uniform → one region; `original` is the
        // ramp the gradient is fit from.
        let (w, h) = (16usize, 16usize);
        let mut quant = vec![0u8; w * h * 4];
        let mut orig = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let p = (y * w + x) * 4;
                quant[p] = 128; // flat red after quantization
                quant[p + 3] = 255;
                orig[p] = ((x * 255) / (w - 1)) as u8; // red ramp in the original
                orig[p + 3] = 255;
            }
        }
        let cfg = TraceConfig { min_area: 0, simplify: 0.0, gradient: true, ..Default::default() };
        let svg = trace_rgba(&quant, w, h, &cfg, Some(&orig), None);
        assert!(svg.contains("<defs>"), "defs block present");
        assert!(svg.contains("<linearGradient id=\"g0\""), "a gradient is defined");
        assert!(svg.contains("fill=\"url(#g0)\""), "the path references the gradient");
        assert!(!svg.contains("<rect"), "gradient mode drops the background rect");

        // With gradient off, the same region is a flat solid (no gradient).
        let cfg_flat = TraceConfig { min_area: 0, simplify: 0.0, gradient: false, ..Default::default() };
        let flat = trace_rgba(&quant, w, h, &cfg_flat, Some(&orig), None);
        assert!(!flat.contains("linearGradient"), "no gradient when disabled");
        assert!(flat.contains("#800000"), "flat red fill instead");
    }

    #[test]
    fn layered_groups_by_object() {
        // 4x1: [red, red, blue, green]. Background = pixels 0..1 (red),
        // object-1 = pixel 2 (blue), object-2 = pixel 3 (green).
        let r = [200, 0, 0, 255];
        let b = [0, 0, 200, 255];
        let g = [0, 200, 0, 255];
        let px = rgba(&[r, r, b, g]);
        let inst = vec![0u32, 0, 1, 2];
        let cfg = TraceConfig { min_area: 0, simplify: 0.0, ..Default::default() };
        let svg = trace_layered(&px, 4, 1, &inst, &cfg, None, None);

        assert!(svg.starts_with("<?xml"));
        assert!(svg.ends_with("</svg>\n"));
        assert!(!svg.contains("<rect"), "layered output uses groups, not a bg rect");
        // Three groups: background + two objects.
        assert_eq!(svg.matches("<g ").count(), 3);
        assert!(svg.contains("data-object=\"background\""));
        assert!(svg.contains("data-object=\"object-1\""));
        assert!(svg.contains("data-object=\"object-2\""));
        // Each distinct color is present.
        assert!(svg.contains("#c80000")); // red bg
        assert!(svg.contains("#0000c8")); // blue object
        assert!(svg.contains("#00c800")); // green object
    }

    #[test]
    fn layered_empty_is_valid_svg() {
        let svg = trace_layered(&[], 0, 0, &[], &TraceConfig::default(), None, None);
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn layered_object_emits_gradient() {
        // 16×16: the top half (rows 0..8) is one object that quantized to a flat
        // red but whose ORIGINAL pixels ramp red 0→255 across x; the bottom half
        // is a flat-blue background layer. With gradients on, the object becomes a
        // <linearGradient> fill inside its own group while the background — a
        // separate layer — stays a flat solid (gradients don't cross objects).
        let (w, h) = (16usize, 16usize);
        let mut quant = vec![0u8; w * h * 4];
        let mut orig = vec![0u8; w * h * 4];
        let mut inst = vec![0u32; w * h];
        for y in 0..h {
            for x in 0..w {
                let p = (y * w + x) * 4;
                if y < 8 {
                    inst[y * w + x] = 1; // object layer
                    quant[p] = 128; // flat red after quantization
                    quant[p + 3] = 255;
                    orig[p] = ((x * 255) / (w - 1)) as u8; // red ramp in the original
                    orig[p + 3] = 255;
                } else {
                    quant[p + 2] = 200; // flat blue background
                    quant[p + 3] = 255;
                    orig[p + 2] = 200;
                    orig[p + 3] = 255;
                }
            }
        }
        let cfg = TraceConfig { min_area: 0, simplify: 0.0, gradient: true, ..Default::default() };
        let svg = trace_layered(&quant, w, h, &inst, &cfg, Some(&orig), None);
        assert!(svg.contains("<defs>"), "defs block present");
        assert!(svg.contains("<linearGradient"), "the object fit a gradient");
        assert!(svg.contains("fill=\"url(#g0)\""), "the object path references the gradient");
        assert!(svg.contains("data-object=\"object-1\""));
        assert!(svg.contains("data-object=\"background\""));
        assert!(svg.contains("#0000c8"), "the background layer stays flat blue");

        // With gradients off the object is a flat fill again — no gradient def.
        let cfg_flat = TraceConfig { min_area: 0, simplify: 0.0, gradient: false, ..Default::default() };
        let flat = trace_layered(&quant, w, h, &inst, &cfg_flat, Some(&orig), None);
        assert!(!flat.contains("linearGradient"), "no gradient when disabled");
        assert!(flat.contains("#800000"), "flat red object instead");
    }
}
