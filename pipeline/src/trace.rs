//! Owned tracer — orchestrates the Level-2 pipeline end to end:
//! quantized raster → palette indices → connected-components segmentation →
//! contour extraction → RDP simplification → layered polygon SVG.
//!
//! The input is expected to already be color-reduced (see [`crate::quantize`]);
//! the tracer derives its palette from the distinct colors present, so feeding
//! it a full-color photo would produce one region per unique color.

use std::collections::HashMap;

use crate::contour::contours_of;
use crate::curvefit;
use crate::segment::segment;
use crate::simplify::simplify_closed;
use crate::svg::{polygon_subpath, to_svg, Region};

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
        }
    }
}

/// Trace an (already quantized) RGBA buffer into a flat-color SVG document.
pub fn trace_rgba(pixels: &[u8], width: usize, height: usize, cfg: &TraceConfig) -> String {
    let n = width * height;
    if n == 0 || pixels.len() < n * 4 {
        return to_svg(width, height, None, &[]);
    }

    // --- build palette indices: 0 = transparent, 1.. = opaque colors ---
    let mut idx = vec![0u32; n];
    let mut palette: Vec<[u8; 3]> = Vec::new();
    let mut map: HashMap<u32, u32> = HashMap::new();
    for i in 0..n {
        let a = pixels[i * 4 + 3];
        if a <= cfg.alpha_threshold {
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

    // --- segment + merge speckles ---
    let mut seg = segment(&idx, width, height);
    seg.merge_small(cfg.min_area);
    let bboxes = seg.bboxes();

    // --- choose a background region (largest opaque) to lay down as a rect ---
    let mut bg_region: Option<usize> = None;
    if cfg.background {
        let mut best_area = 0u32;
        for c in 0..seg.num_components {
            if seg.component_color[c] != 0 && seg.component_area[c] > best_area {
                best_area = seg.component_area[c];
                bg_region = Some(c);
            }
        }
    }
    let background = bg_region.map(|c| palette[(seg.component_color[c] - 1) as usize]);

    // --- trace every opaque region (except the background) ---
    let mut regions: Vec<Region> = Vec::new();
    for c in 0..seg.num_components {
        if seg.component_color[c] == 0 || Some(c) == bg_region {
            continue;
        }
        let color = palette[(seg.component_color[c] - 1) as usize];
        let raw_loops = contours_of(&seg.labels, width, height, c as u32, bboxes[c]);
        let mut subpaths = Vec::with_capacity(raw_loops.len());
        for lp in raw_loops {
            // Simplify, but never let RDP collapse a real loop to nothing —
            // fall back to the exact contour so small holes/regions survive.
            let simp = simplify_closed(&lp, cfg.simplify);
            let poly = if simp.len() >= 3 { simp } else { lp };
            if poly.len() < 3 {
                continue;
            }
            let sub = if cfg.curve && poly.len() >= 4 {
                curvefit::fit_loop(&poly, cfg.corner_threshold, cfg.curve_error)
            } else {
                polygon_subpath(&poly)
            };
            subpaths.push(sub);
        }
        if !subpaths.is_empty() {
            regions.push(Region { color, subpaths });
        }
    }

    to_svg(width, height, background, &regions)
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
        let svg = trace_rgba(&px, 2, 2, &TraceConfig { min_area: 0, simplify: 0.0, ..Default::default() });
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
        let svg = trace_rgba(&px, 2, 2, &TraceConfig { background: false, min_area: 0, simplify: 0.0, ..Default::default() });
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
        let svg = trace_rgba(&px, 6, 1, &TraceConfig { min_area: 0, simplify: 0.0, ..Default::default() });
        assert!(svg.contains("<rect")); // red background
        assert_eq!(svg.matches("<path").count(), 1, "the two greens merge to one path");
        assert_eq!(svg.matches('M').count(), 2, "one path, two subpaths");
        assert!(svg.contains("#00c800")); // green
    }

    #[test]
    fn empty_image_is_valid_svg() {
        let svg = trace_rgba(&[], 0, 0, &TraceConfig::default());
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
    }
}
