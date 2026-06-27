//! End-to-end DexiNed test. Ignored by default (needs the ~47 MB weights + a
//! real image). Writes `bg-edges.png` (the edge probability map) for inspection.
//!
//!   SVGIT_TEST_IMAGE=/path/photo.jpg SVGIT_MODEL_DIR=$PWD/models \
//!     cargo test -p svgit-edgenet --test inference -- --ignored --nocapture

use svgit_edgenet::{default_model_path, edge_map};

#[test]
#[ignore = "needs dexined weights + SVGIT_TEST_IMAGE"]
fn produces_edge_map() {
    let img_path = match std::env::var("SVGIT_TEST_IMAGE") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("SVGIT_TEST_IMAGE unset — skipping");
            return;
        }
    };
    let model = default_model_path();
    assert!(model.exists(), "model missing at {}", model.display());

    let decoded = image::open(&img_path).expect("decode").to_rgba8();
    let (w, h) = (decoded.width() as usize, decoded.height() as usize);
    let rgba = decoded.into_raw();

    let edges = edge_map(&rgba, w, h, &model).expect("edge_map should succeed");
    assert_eq!(edges.len(), w * h);

    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    for &v in &edges {
        assert!(v.is_finite() && (0.0..=1.0).contains(&v), "edge value out of range: {v}");
        min = min.min(v);
        max = max.max(v);
        sum += v as f64;
    }
    let mean = sum / edges.len() as f64;
    eprintln!("edges: min={min:.3} max={max:.3} mean={mean:.3}");
    // A real photo must produce variation (some strong edges, mostly low).
    assert!(max > 0.5, "no strong edges detected (max={max})");
    assert!(mean < 0.6, "edge map implausibly saturated (mean={mean})");

    let buf: Vec<u8> = edges.iter().map(|&v| (v * 255.0).round() as u8).collect();
    let out_dir = std::env::var("SVGIT_TEST_OUT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir());
    let path = out_dir.join("bg-edges.png");
    image::save_buffer(&path, &buf, w as u32, h as u32, image::ColorType::L8).expect("write edges");
    eprintln!("wrote {}", path.display());
}
