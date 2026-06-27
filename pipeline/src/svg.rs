//! SVG serialization — stage 5 of the owned pipeline, plus the geometry types
//! the contour/curve-fit stages produce.
//!
//! Emits one `<path>` per region (holes are extra subpaths in the same path,
//! rendered with the nonzero fill rule thanks to their opposite winding). An
//! optional background `<rect>` lets the caller drop the single largest region,
//! which the others then stack on top of — keeping the path count low.

use std::collections::BTreeMap;
use std::fmt::Write;

type Pt = (i32, i32);
pub type Ptf = (f64, f64);

/// A single path segment. Polygon output uses only `Line`; curve fitting emits
/// `Cubic` (two control points + endpoint; the start point is implicit).
pub enum Seg {
    Line(Ptf),
    Cubic(Ptf, Ptf, Ptf),
}

/// A closed subpath: a start point followed by segments back around to it.
pub struct Subpath {
    pub start: Ptf,
    pub segs: Vec<Seg>,
}

pub struct Region {
    pub color: [u8; 3],
    /// Outer subpath plus any hole subpaths.
    pub subpaths: Vec<Subpath>,
}

/// Build a closed polygon subpath (all line segments) from integer corners.
pub fn polygon_subpath(pts: &[Pt]) -> Subpath {
    let start = (pts[0].0 as f64, pts[0].1 as f64);
    let segs = pts[1..]
        .iter()
        .map(|&p| Seg::Line((p.0 as f64, p.1 as f64)))
        .collect();
    Subpath { start, segs }
}

/// Build a closed polygon subpath from float corners (e.g. edge-snapped points).
pub fn polygon_subpath_f(pts: &[Ptf]) -> Subpath {
    let start = pts.first().copied().unwrap_or((0.0, 0.0));
    let segs = pts.get(1..).unwrap_or(&[]).iter().map(|&p| Seg::Line(p)).collect();
    Subpath { start, segs }
}

fn hex(c: [u8; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])
}

/// Format a coordinate with up to 2 decimals, trimming trailing zeros.
fn fnum(v: f64) -> String {
    if !v.is_finite() {
        return "0".to_string(); // never emit NaN/inf into path data
    }
    let r = (v * 100.0).round() / 100.0;
    if r == r.trunc() {
        format!("{}", r as i64)
    } else {
        let s = format!("{r:.2}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

fn write_subpath(d: &mut String, sp: &Subpath) {
    if sp.segs.len() < 2 {
        return;
    }
    let _ = write!(d, "M{} {}", fnum(sp.start.0), fnum(sp.start.1));
    for seg in &sp.segs {
        match seg {
            Seg::Line(p) => {
                let _ = write!(d, "L{} {}", fnum(p.0), fnum(p.1));
            }
            Seg::Cubic(c1, c2, e) => {
                let _ = write!(
                    d,
                    "C{} {} {} {} {} {}",
                    fnum(c1.0),
                    fnum(c1.1),
                    fnum(c2.0),
                    fnum(c2.1),
                    fnum(e.0),
                    fnum(e.1)
                );
            }
        }
    }
    d.push('Z');
}

/// Build a full SVG document from regions, optionally laying the largest region
/// down as a background rectangle.
pub fn to_svg(width: usize, height: usize, background: Option<[u8; 3]>, regions: &[Region]) -> String {
    let mut s = String::with_capacity(1024 + regions.len() * 96);
    s.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    s.push_str("<!-- Generator: svgit owned pipeline -->\n");
    let _ = writeln!(
        s,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">"
    );
    if let Some(bg) = background {
        let _ = writeln!(
            s,
            "<rect width=\"{width}\" height=\"{height}\" fill=\"{}\"/>",
            hex(bg)
        );
    }
    // Minify: merge every subpath of the same fill color into a single
    // `<path>`. The owned tracer's regions tile the canvas (no overlap), so
    // z-order is irrelevant and this is lossless. A region whose color matches
    // the background rect is fully covered by it and is dropped entirely. The
    // result has at most one path per color. BTreeMap keeps the output ordered
    // (deterministic) by color.
    let mut by_color: BTreeMap<[u8; 3], String> = BTreeMap::new();
    for r in regions {
        if Some(r.color) == background {
            continue; // already painted by the background rect
        }
        let d = by_color.entry(r.color).or_default();
        for sp in &r.subpaths {
            write_subpath(d, sp);
        }
    }
    for (color, d) in &by_color {
        if d.is_empty() {
            continue;
        }
        let _ = writeln!(
            s,
            "<path d=\"{}\" fill=\"{}\" fill-rule=\"nonzero\"/>",
            d,
            hex(*color)
        );
    }
    s.push_str("</svg>\n");
    s
}

/// Escape the few characters that aren't legal inside a double-quoted XML
/// attribute value. Layer labels are svgit-generated and simple, so this is
/// defensive rather than load-bearing.
fn attr_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Build a full SVG document where each layer becomes a `<g data-object="...">`
/// group, emitted bottom-first. Within a layer, same-color subpaths merge into
/// one `<path>` (deterministic via BTreeMap) exactly as [`to_svg`] does — so an
/// object made of several quantized colors stays a single group of color paths.
pub fn to_svg_layered(width: usize, height: usize, layers: &[(String, Vec<Region>)]) -> String {
    let mut s = String::with_capacity(2048);
    s.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    s.push_str("<!-- Generator: svgit owned pipeline (layered) -->\n");
    let _ = writeln!(
        s,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">"
    );
    for (label, regions) in layers {
        let mut by_color: BTreeMap<[u8; 3], String> = BTreeMap::new();
        for r in regions {
            let d = by_color.entry(r.color).or_default();
            for sp in &r.subpaths {
                write_subpath(d, sp);
            }
        }
        if by_color.values().all(|d| d.is_empty()) {
            continue;
        }
        let _ = writeln!(s, "<g data-object=\"{}\">", attr_escape(label));
        for (color, d) in &by_color {
            if d.is_empty() {
                continue;
            }
            let _ = writeln!(
                s,
                "<path d=\"{}\" fill=\"{}\" fill-rule=\"nonzero\"/>",
                d,
                hex(*color)
            );
        }
        s.push_str("</g>\n");
    }
    s.push_str("</svg>\n");
    s
}
