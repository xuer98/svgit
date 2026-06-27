//! svgit — a thin HTTP service wrapping the vtracer core.
//!
//! `GET  /`            serves a single-page parameter UI.
//! `POST /api/convert` takes a multipart form (`image` file + tracing params)
//!                     and returns the traced `image/svg+xml` document.

use axum::{
    body::Bytes,
    extract::{DefaultBodyLimit, Multipart},
    http::{header, StatusCode},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Router,
};
use std::io::Cursor;
use std::net::SocketAddr;
use std::sync::OnceLock;
use svgit_pipeline::{QuantizeConfig, TraceConfig};
use tokio::sync::Semaphore;
use vtracer::{ColorImage, ColorMode, Config, Hierarchical, PathSimplifyMode, Preset};

/// Max upload size. Raster scans can be large, so allow generously more than
/// axum's 2 MB default. Bounds the *compressed* upload only.
const MAX_UPLOAD_BYTES: usize = 32 * 1024 * 1024;

/// Max decoded raster size. The upload limit bounds compressed bytes, not the
/// decoded pixel buffer — a tiny file can declare enormous dimensions
/// (a decompression bomb). We reject anything past this before allocating.
const MAX_PIXELS: u64 = 25_000_000; // ~5000×5000

/// Bounds how many CPU-bound conversions run at once, so a burst of requests
/// can't saturate every core or exhaust the blocking-thread pool.
fn convert_slots() -> &'static Semaphore {
    static SLOTS: OnceLock<Semaphore> = OnceLock::new();
    SLOTS.get_or_init(|| {
        let n = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        Semaphore::new(n)
    })
}

const INDEX_HTML: &str = include_str!("../static/index.html");

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/", get(index))
        .route("/api/convert", post(convert))
        .layer(DefaultBodyLimit::max(MAX_UPLOAD_BYTES));

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);
    let addr = SocketAddr::from(([127, 0, 0, 1], port));

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .unwrap_or_else(|e| panic!("failed to bind {addr}: {e}"));
    println!("svgit service listening on http://{addr}");

    axum::serve(listener, app)
        .await
        .expect("server error");
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

/// Raw, unvalidated parameters as received from the form. Every field is
/// optional; missing fields fall back to the core defaults (or the chosen
/// preset).
#[derive(Default)]
struct RawParams {
    preset: Option<String>,
    engine: Option<String>,
    quantize: Option<String>,
    colors: Option<usize>,
    simplify: Option<f64>,
    min_region: Option<u32>,
    curve: Option<String>,
    curve_corner: Option<f64>,
    curve_error: Option<f64>,
    color_mode: Option<String>,
    hierarchical: Option<String>,
    mode: Option<String>,
    filter_speckle: Option<usize>,
    color_precision: Option<i32>,
    layer_difference: Option<i32>,
    corner_threshold: Option<i32>,
    length_threshold: Option<f64>,
    splice_threshold: Option<i32>,
    max_iterations: Option<usize>,
    path_precision: Option<u32>,
}

async fn convert(mut multipart: Multipart) -> Result<Response, AppError> {
    let mut image_bytes: Option<Bytes> = None;
    let mut p = RawParams::default();

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::bad(format!("malformed multipart body: {e}")))?
    {
        let name = field.name().unwrap_or("").to_string();

        if name == "image" {
            let bytes = field
                .bytes()
                .await
                .map_err(|e| AppError::bad(format!("could not read uploaded image: {e}")))?;
            image_bytes = Some(bytes);
            continue;
        }

        let value = field
            .text()
            .await
            .map_err(|e| AppError::bad(format!("could not read field `{name}`: {e}")))?;
        let value = value.trim();
        if value.is_empty() {
            continue;
        }

        match name.as_str() {
            "preset" => p.preset = Some(value.to_string()),
            "engine" => p.engine = Some(value.to_string()),
            "quantize" => p.quantize = Some(value.to_string()),
            "colors" => p.colors = value.parse().ok(),
            "simplify" => p.simplify = value.parse().ok().filter(|v: &f64| v.is_finite()),
            "min_region" => p.min_region = value.parse().ok(),
            "curve" => p.curve = Some(value.to_string()),
            "curve_corner" => p.curve_corner = value.parse().ok().filter(|v: &f64| v.is_finite()),
            "curve_error" => p.curve_error = value.parse().ok().filter(|v: &f64| v.is_finite()),
            "color_mode" => p.color_mode = Some(value.to_string()),
            "hierarchical" => p.hierarchical = Some(value.to_string()),
            "mode" => p.mode = Some(value.to_string()),
            "filter_speckle" => p.filter_speckle = value.parse().ok(),
            "color_precision" => p.color_precision = value.parse().ok(),
            "layer_difference" => p.layer_difference = value.parse().ok(),
            "corner_threshold" => p.corner_threshold = value.parse().ok(),
            "length_threshold" => {
                p.length_threshold = value.parse().ok().filter(|v: &f64| v.is_finite())
            }
            "splice_threshold" => p.splice_threshold = value.parse().ok(),
            "max_iterations" => p.max_iterations = value.parse().ok(),
            "path_precision" => p.path_precision = value.parse().ok(),
            _ => {} // ignore unknown fields
        }
    }

    let bytes = image_bytes.ok_or_else(|| AppError::bad("missing `image` field"))?;
    let config = build_config(&p);
    let quantize_on = matches!(p.quantize.as_deref(), Some("on") | Some("true") | Some("1"));
    let num_colors = p.colors.unwrap_or(16).clamp(2, 256);
    let engine_owned = matches!(p.engine.as_deref(), Some("owned"));
    let simplify_eps = p.simplify.unwrap_or(1.2).clamp(0.0, 10.0);
    let min_region = p.min_region.unwrap_or(4).min(4096);
    let curve_on = matches!(p.curve.as_deref(), Some("on") | Some("true") | Some("1"));
    let curve_corner = p.curve_corner.unwrap_or(80.0).clamp(0.0, 180.0);
    let curve_err = p.curve_error.unwrap_or(2.0).clamp(0.1, 20.0);

    // Limit concurrent CPU-bound conversions. Held until the response is built.
    let _permit = convert_slots()
        .acquire()
        .await
        .map_err(|_| AppError::internal("converter pool closed"))?;

    // Conversion is CPU-bound and synchronous; keep it off the async runtime.
    let svg = tokio::task::spawn_blocking(move || -> Result<String, String> {
        // Reject oversized rasters *before* decoding, so a decompression bomb
        // can't allocate gigabytes. Reading dimensions only parses the header.
        let (w, h) = image::io::Reader::new(Cursor::new(&bytes))
            .with_guessed_format()
            .map_err(|e| format!("could not read image: {e}"))?
            .into_dimensions()
            .map_err(|e| format!("could not read image dimensions: {e}"))?;
        if w == 0 || h == 0 {
            return Err("image has a zero dimension".to_string());
        }
        if (w as u64) * (h as u64) > MAX_PIXELS {
            return Err(format!(
                "image too large: {w}×{h} exceeds the {} megapixel limit",
                MAX_PIXELS / 1_000_000
            ));
        }

        let rgba = image::load_from_memory(&bytes)
            .map_err(|e| format!("could not decode image: {e}"))?
            .to_rgba8();
        let mut raw = rgba.into_raw();

        // Owned engine: quantize to N colors then run the fully-owned flat
        // tracer (segmentation → contours → simplify → SVG). No VTracer.
        if engine_owned {
            raw = svgit_pipeline::quantize_rgba(
                raw,
                &QuantizeConfig {
                    num_colors,
                    ..Default::default()
                },
            );
            return Ok(svgit_pipeline::trace_rgba(
                &raw,
                w as usize,
                h as usize,
                &TraceConfig {
                    alpha_threshold: 0,
                    min_area: min_region,
                    simplify: simplify_eps,
                    background: true,
                    curve: curve_on,
                    corner_threshold: curve_corner,
                    curve_error: curve_err,
                },
            ));
        }

        // VTracer engine: optionally pre-quantize to N colors (LAB k-means)
        // before handing the flattened raster to the VTracer core.
        if quantize_on {
            raw = svgit_pipeline::quantize_rgba(
                raw,
                &QuantizeConfig {
                    num_colors,
                    ..Default::default()
                },
            );
        }

        let img = ColorImage {
            pixels: raw,
            width: w as usize,
            height: h as usize,
        };
        let svg = vtracer::convert(img, config)?;
        Ok(svg.to_string())
    })
    .await
    .map_err(|e| AppError::internal(format!("conversion task failed: {e}")))?
    .map_err(AppError::bad)?;

    Ok((
        [(header::CONTENT_TYPE, "image/svg+xml; charset=utf-8")],
        svg,
    )
        .into_response())
}

/// Turn raw form params into a validated [`Config`]. Out-of-range values are
/// clamped (rather than rejected) so a UI slider can never produce a 400.
fn build_config(p: &RawParams) -> Config {
    let mut config = match p.preset.as_deref() {
        Some("bw") => Config::from_preset(Preset::Bw),
        Some("poster") => Config::from_preset(Preset::Poster),
        Some("photo") => Config::from_preset(Preset::Photo),
        _ => Config::default(),
    };

    if let Some(v) = p.color_mode.as_deref() {
        config.color_mode = match v {
            "binary" | "bw" | "BW" => ColorMode::Binary,
            _ => ColorMode::Color,
        };
    }
    if let Some(v) = p.hierarchical.as_deref() {
        config.hierarchical = match v {
            "cutout" => Hierarchical::Cutout,
            _ => Hierarchical::Stacked,
        };
    }
    if let Some(v) = p.mode.as_deref() {
        config.mode = match v {
            "pixel" | "none" => PathSimplifyMode::None,
            "polygon" => PathSimplifyMode::Polygon,
            _ => PathSimplifyMode::Spline,
        };
    }

    if let Some(v) = p.filter_speckle {
        config.filter_speckle = v.min(16);
    }
    if let Some(v) = p.color_precision {
        config.color_precision = v.clamp(1, 8);
    }
    if let Some(v) = p.layer_difference {
        config.layer_difference = v.clamp(0, 255);
    }
    if let Some(v) = p.corner_threshold {
        config.corner_threshold = v.clamp(0, 180);
    }
    if let Some(v) = p.length_threshold {
        config.length_threshold = v.clamp(3.5, 10.0);
    }
    if let Some(v) = p.splice_threshold {
        config.splice_threshold = v.clamp(0, 180);
    }
    if let Some(v) = p.max_iterations {
        config.max_iterations = v.clamp(1, 20);
    }
    if let Some(v) = p.path_precision {
        config.path_precision = Some(v.min(8));
    }

    config
}

/// A simple error type that renders as an HTTP status + plain-text message.
struct AppError {
    code: StatusCode,
    msg: String,
}

impl AppError {
    fn bad(msg: impl Into<String>) -> Self {
        Self {
            code: StatusCode::BAD_REQUEST,
            msg: msg.into(),
        }
    }
    fn internal(msg: impl Into<String>) -> Self {
        Self {
            code: StatusCode::INTERNAL_SERVER_ERROR,
            msg: msg.into(),
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (self.code, self.msg).into_response()
    }
}
