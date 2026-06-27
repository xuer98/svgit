<div align="center">

  <h1>SvgIt</h1>

  <p>
    <strong>Raster → vector graphics, vectorizer.ai-style</strong>
  </p>

</div>

## Introduction

**SvgIt** turns raster images (PNG, JPG, …) into clean, compact SVG. It started as a
fork of [visioncortex/vtracer](https://github.com/visioncortex/vtracer) and grew a
fully-owned tracing pipeline plus an optional ML layer (background removal,
object segmentation, edge refinement), all wrapped in a small HTTP service with a
live-preview UI.

The project is built in three levels:

- **Level 1** — a thin [axum](https://github.com/tokio-rs/axum) service wrapping the
  VTracer core, with a parameter UI for live tuning.
- **Level 2** — a dependency-free, fully-owned classical pipeline: LAB color
  quantization → region segmentation → contour tracing → RDP simplify → Schneider
  curve-fit → layering → minified SVG. Selectable as the **Owned** engine; produces
  exactly-N-color output.
- **Level 3** — an ML layer (PyTorch → ONNX, run embedded in Rust via
  [`ort`](https://github.com/pykeio/ort)): salient-object background removal,
  FastSAM "segment everything", and CNN edge/corner refinement.

## Quick start (the service)

```sh
cargo run -p svgit-service
```

Then open **http://127.0.0.1:8080** and drag, paste, or upload an image — it
converts live as you adjust parameters.

```sh
PORT=8090 cargo run -p svgit-service   # bind a different port
```

> The first build downloads a prebuilt onnxruntime (via `ort`) and needs network
> access. Subsequent builds are fast. Add `--release` for much faster ML inference.

The **VTracer** and **Owned** engines work with no extra setup. The ML features
need model weights — see below.

### Engines

| Engine | What it does |
|--------|--------------|
| **VTracer** | The upstream vtracer core — stacked color clustering + curve tracing. |
| **Owned** | The dependency-free Level-2 pipeline; exact N-color flat output. Supports background removal and edge refinement. |
| **Segment** | FastSAM "segment everything" → layered SVG, one `<g>` per detected object. |

## ML layer & models

The weights are large and license-encumbered, so they're git-ignored and fetched
on demand:

```sh
./scripts/fetch-models.sh             # all models, into ./models
SVGIT_MODEL_DIR=/path ./scripts/fetch-models.sh
```

| Model | Size | Powers |
|-------|------|--------|
| `u2netp.onnx` | ~4.6 MB | Remove background · **Fast** |
| `isnet-general-use.onnx` | ~178 MB | Remove background · **High** (sharper fine detail) |
| `lineart.onnx` | ~17 MB | **Refine edges** (owned engine contour snapping) |
| `FastSAM-x.onnx` | ~289 MB | **Segment** engine |

Until a model is present, its feature returns a clear "model not found" error; every
other feature still works. The core tracer needs no models at all.

## Command-line app (vtracer)

The original VTracer CLI is still in the workspace:

```sh
cargo run -p vtracer -- --input input.jpg --output output.svg
```

It's also published independently on [crates.io/vtracer](https://crates.io/crates/vtracer)
(`cargo install vtracer`) and as a [Python package](https://pypi.org/project/vtracer/)
(`pip install vtracer`).

```
OPTIONS:
        --colormode <color_mode>                 True color image `color` (default) or Binary image `bw`
    -p, --color_precision <color_precision>      Number of significant bits to use in an RGB channel
    -c, --corner_threshold <corner_threshold>    Minimum momentary angle (degree) to be considered a corner
    -f, --filter_speckle <filter_speckle>        Discard patches smaller than X px in size
    -g, --gradient_step <gradient_step>          Color difference between gradient layers
        --hierarchical <hierarchical>            `stacked` (default) or `cutout` (color mode only)
    -i, --input <input>                          Path to input raster image
    -m, --mode <mode>                            Curve fitting mode `pixel`, `polygon`, `spline`
    -o, --output <output>                        Path to output vector graphics
        --path_precision <path_precision>        Number of decimal places to use in path string
        --preset <preset>                        Use one of the preset configs `bw`, `poster`, `photo`
    -l, --segment_length <segment_length>        Subdivide-smooth until all segments are shorter than this
    -s, --splice_threshold <splice_threshold>    Minimum angle displacement (degree) to splice a spline
```

## Workspace layout

| Crate | Role |
|-------|------|
| `cmdapp` (`vtracer`) | Upstream VTracer CLI. |
| `webapp` | VTracer WASM build for the browser. |
| `service` (`svgit-service`) | The axum HTTP service + live-preview UI. |
| `pipeline` (`svgit-pipeline`) | Owned, dependency-free classical tracer. |
| `bgremove` (`svgit-bgremove`) | ONNX background removal (u2netp / ISNet). |
| `objseg` (`svgit-objseg`) | FastSAM object segmentation. |
| `edgenet` (`svgit-edgenet`) | Line-art CNN edge map for contour refinement. |

## Credits

SvgIt is built on [VTracer](https://github.com/visioncortex/vtracer) by
[VisionCortex](https://www.visioncortex.org/). See the
[tracing](https://www.visioncortex.org/vtracer-docs) and
[clustering](https://www.visioncortex.org/impression-docs) algorithm write-ups for
the foundations it builds on.
