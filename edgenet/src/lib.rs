//! svgit-edgenet — CNN edge detection (DexiNed) for contour refinement.
//!
//! Produces a per-pixel edge-probability map in `[0, 1]` at the original image
//! resolution. The owned tracer's refinement stage ([`svgit_pipeline`]) snaps
//! contour vertices onto this map's ridges and derives corner breaks from it.
//!
//! The heavy `ort`/onnxruntime dependency lives here, never in `svgit-pipeline`.
//!
//! Model: "Informative Drawings" (line-art). Preprocessing is simple — RGB,
//! scaled to `[0, 1]` (no mean subtraction), NCHW. The fully-convolutional net
//! runs at the fed resolution; we feed the original size capped to a long edge.
//! Its output is a *drawing* (dark lines on white paper), so we invert it into an
//! edge-probability map (1 = edge).

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

/// Longest input edge fed to the net (capped for speed); the model is fully
/// convolutional, so this just trades detail for latency.
const MAX_EDGE: usize = 768;

pub const MODEL_FILENAME: &str = "lineart.onnx";

/// `$SVGIT_MODEL_DIR/lineart.onnx`, defaulting to `./models/lineart.onnx`.
pub fn default_model_path() -> PathBuf {
    let dir = std::env::var_os("SVGIT_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models"));
    dir.join(MODEL_FILENAME)
}

static SESSION: OnceLock<Mutex<Session>> = OnceLock::new();

fn session(model_path: &Path) -> Result<&'static Mutex<Session>, String> {
    if let Some(s) = SESSION.get() {
        return Ok(s);
    }
    if !model_path.exists() {
        return Err(format!(
            "edge model not found at {} — run scripts/fetch-models.sh (or set SVGIT_MODEL_DIR)",
            model_path.display()
        ));
    }
    let built = Session::builder()
        .map_err(|e| format!("ort session builder: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| format!("ort optimization level: {e}"))?
        .with_intra_threads(4)
        .map_err(|e| format!("ort thread config: {e}"))?
        .commit_from_file(model_path)
        .map_err(|e| format!("loading {}: {e}", model_path.display()))?;
    let _ = SESSION.set(Mutex::new(built));
    Ok(SESSION.get().expect("session just set"))
}

/// Compute a `width*height` edge-probability map in `[0, 1]` from an RGBA buffer.
pub fn edge_map(
    rgba: &[u8],
    width: usize,
    height: usize,
    model_path: &Path,
) -> Result<Vec<f32>, String> {
    let n = width.checked_mul(height).ok_or("image dimensions overflow")?;
    let n4 = n.checked_mul(4).ok_or("image dimensions overflow")?;
    if n == 0 || rgba.len() < n4 {
        return Err("empty or truncated RGBA buffer".to_string());
    }

    let lock = session(model_path)?;
    let mut sess = lock.lock().map_err(|_| "model session poisoned".to_string())?;
    // DexiNed has a fixed, non-square input [1,3,H,W] (the OpenCV export is
    // 480×640). Read H and W from the model; fall back to a square default.
    let ishape: Vec<i64> = sess
        .inputs
        .first()
        .and_then(|i| i.input_type.tensor_shape())
        .map(|s| s.to_vec())
        .unwrap_or_default();
    let dim = |i: usize| ishape.get(i).copied().filter(|&d| d > 0).map(|d| d as usize);
    // Fixed model dims win; otherwise feed the original size capped to MAX_EDGE
    // and rounded to a multiple of 8 (keeps the conv up/down-sampling aligned).
    let round8 = |v: usize| (v.max(8) / 8 * 8).max(8);
    let (in_h, in_w) = match (dim(2), dim(3)) {
        (Some(h), Some(w)) => (h, w),
        _ => {
            let scale = (MAX_EDGE as f32 / width.max(height) as f32).min(1.0);
            (
                round8((height as f32 * scale) as usize),
                round8((width as f32 * scale) as usize),
            )
        }
    };

    // --- resize RGB → in_w×in_h, build /255 NCHW tensor (RGB) ---
    let mut rgb = vec![0u8; n * 3];
    for i in 0..n {
        rgb[i * 3] = rgba[i * 4];
        rgb[i * 3 + 1] = rgba[i * 4 + 1];
        rgb[i * 3 + 2] = rgba[i * 4 + 2];
    }
    let src = image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(width as u32, height as u32, rgb)
        .ok_or("could not wrap RGB buffer")?;
    let small = image::imageops::resize(
        &src,
        in_w as u32,
        in_h as u32,
        image::imageops::FilterType::Triangle,
    );
    let sp = small.as_raw();
    let hw = in_w * in_h;
    let mut input = vec![0f32; 3 * hw];
    for p in 0..hw {
        input[p] = sp[p * 3] as f32 / 255.0; // R plane
        input[hw + p] = sp[p * 3 + 1] as f32 / 255.0; // G plane
        input[2 * hw + p] = sp[p * 3 + 2] as f32 / 255.0; // B plane
    }

    // --- run, take the fused (last) single-channel rank-4 map ---
    let (eshape, edata) = {
        let in_name = sess
            .inputs
            .first()
            .map(|i| i.name.clone())
            .unwrap_or_else(|| "input".to_string());
        let tensor = Tensor::from_array((vec![1i64, 3, in_h as i64, in_w as i64], input))
            .map_err(|e| format!("building input tensor: {e}"))?;
        let outputs = sess
            .run(ort::inputs![in_name => tensor])
            .map_err(|e| format!("edge-net inference: {e}"))?;
        // Prefer the last rank-4 single-channel output (the edge/line map).
        let mut picked: Option<(Vec<i64>, Vec<f32>)> = None;
        for i in 0..outputs.len() {
            let (shape, data) = outputs[i]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("reading edge output {i}: {e}"))?;
            if shape.len() == 4 {
                picked = Some((shape.to_vec(), data.to_vec()));
            }
        }
        picked.ok_or("model produced no rank-4 edge map")?
    };
    drop(sess);

    let eh = eshape[2] as usize;
    let ew = eshape[3] as usize;
    // Reject degenerate shapes: a zero dim makes eh*ew == 0, which would slip
    // past a `len < eh*ew` check and yield an all-zero (useless) edge map.
    if eh == 0 || ew == 0 || edata.len() < eh * ew {
        return Err(format!("unexpected edge output shape {eshape:?}"));
    }
    let map = &edata[..eh * ew];

    // The line-art net outputs a drawing in [0,1] with dark lines on white
    // paper, so the edge probability is the inverse. (If a future model emits
    // logits instead, sigmoid first.)
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in map {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    let is_prob = lo >= -1e-3 && hi <= 1.0 + 1e-3;
    let small_map: Vec<f32> = if is_prob {
        map.iter().map(|&v| 1.0 - v.clamp(0.0, 1.0)).collect()
    } else {
        map.iter().map(|&v| 1.0 - sigmoid(v)).collect()
    };

    // Bilinear resample to the original resolution.
    Ok(resample(&small_map, ew, eh, width, height))
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Bilinear resample a single-channel map from (sw,sh) to (dw,dh).
fn resample(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    let mut out = vec![0f32; dw * dh];
    if sw == 0 || sh == 0 {
        return out;
    }
    let fx = sw as f32 / dw as f32;
    let fy = sh as f32 / dh as f32;
    for y in 0..dh {
        let syf = (y as f32 + 0.5) * fy - 0.5;
        let sy = syf.floor();
        let ty = (syf - sy).clamp(0.0, 1.0);
        let y0 = (sy as isize).clamp(0, sh as isize - 1) as usize;
        let y1 = ((sy as isize) + 1).clamp(0, sh as isize - 1) as usize;
        for x in 0..dw {
            let sxf = (x as f32 + 0.5) * fx - 0.5;
            let sx = sxf.floor();
            let tx = (sxf - sx).clamp(0.0, 1.0);
            let x0 = (sx as isize).clamp(0, sw as isize - 1) as usize;
            let x1 = ((sx as isize) + 1).clamp(0, sw as isize - 1) as usize;
            let a = src[y0 * sw + x0];
            let b = src[y0 * sw + x1];
            let c = src[y1 * sw + x0];
            let d = src[y1 * sw + x1];
            let top = a + (b - a) * tx;
            let bot = c + (d - c) * tx;
            out[y * dw + x] = top + (bot - top) * ty;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_monotone() {
        assert!(sigmoid(-10.0) < 0.01);
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
    }

    #[test]
    fn resample_upscales_constant() {
        let src = vec![0.7f32; 4]; // 2x2 constant
        let out = resample(&src, 2, 2, 6, 6);
        assert_eq!(out.len(), 36);
        assert!(out.iter().all(|&v| (v - 0.7).abs() < 1e-5));
    }

    #[test]
    fn resample_identity() {
        let src = vec![0.1, 0.2, 0.3, 0.4];
        let out = resample(&src, 2, 2, 2, 2);
        for (a, b) in src.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn edge_map_rejects_truncated() {
        let m = Path::new("/nonexistent/dexined.onnx");
        assert!(edge_map(&[0, 0, 0], 4, 4, m).is_err());
    }
}
