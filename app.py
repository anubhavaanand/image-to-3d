"""
Production App — Image to 3D Model
====================================
Upload any photo → background removed → 3D model generated → download GLB.

Engines available:
  • TripoSR  (<0.5s, ~5 GB VRAM) — fastest, good for quick previews
  • TRELLIS  (~10s,  ~16-24 GB VRAM) — highest quality mesh
  • LGM      (~5s,  ~10 GB VRAM) — Gaussian output (PLY)

Run locally:
    pip install gradio
    python app.py

Deploy to HuggingFace Spaces:
    1. Create a new Space (Gradio SDK, GPU T4 or A100)
    2. Push this file + requirements.txt to the Space repo
    3. Done — it auto-runs on HF infrastructure

HuggingFace Space URL (after deploy):
    https://huggingface.co/spaces/YOUR_USERNAME/image-to-3d
"""

from __future__ import annotations

import os
import sys
import time
import tempfile
import traceback
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

# Add project src to path when running standalone
sys.path.insert(0, str(Path(__file__).parent))
from src.preprocessing.background_removal import BackgroundRemover
from src.engines.triposr_engine import TriposrEngine
from src.engines.trellis_engine import TrellisEngine
from src.engines.lgm_engine import LGMEngine
from src.export.glb_exporter import export_glb
from src.export.ply_exporter import export_ply

# ---------------------------------------------------------------------------
# Global state — engines loaded lazily on first use to save startup time
# ---------------------------------------------------------------------------

_engines: dict = {}
_bg_remover: BackgroundRemover | None = None

DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
OUTPUT_DIR = Path(tempfile.gettempdir()) / "image_to_3d_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def _get_bg_remover() -> BackgroundRemover:
    global _bg_remover
    if _bg_remover is None:
        _bg_remover = BackgroundRemover()
    return _bg_remover


def _get_engine(name: str):
    global _engines
    if name not in _engines:
        if name == "TripoSR":
            e = TriposrEngine(device=DEVICE)
        elif name == "TRELLIS":
            e = TrellisEngine(device=DEVICE, output_format="mesh", slat_steps=12, sparse_steps=12)
        elif name == "LGM":
            e = LGMEngine(device=DEVICE)
        else:
            raise ValueError(f"Unknown engine: {name}")
        e.load_model()
        _engines[name] = e
    return _engines[name]


# ---------------------------------------------------------------------------
# Core inference function
# ---------------------------------------------------------------------------

def generate_3d(
    image: Image.Image,
    engine_name: str,
    remove_bg: bool,
) -> tuple[str | None, str, Image.Image | None]:
    """
    Returns
    -------
    (glb_path_or_none, status_message, preview_image_or_none)
    """
    if image is None:
        return None, "⚠️ Please upload an image first.", None

    try:
        t0 = time.time()

        # 1. Background removal
        status = "Removing background…"
        if remove_bg:
            remover = _get_bg_remover()
            image = remover.remove(image)
        else:
            # Ensure white background if RGBA
            if image.mode == "RGBA":
                white = Image.new("RGB", image.size, "WHITE")
                white.paste(image, mask=image.split()[3])
                image = white

        # 2. Run engine
        status = f"Running {engine_name}…"
        engine = _get_engine(engine_name)
        result = engine.predict(image)
        latency = time.time() - t0

        # 3. Export
        out_file = OUTPUT_DIR / f"output_{int(time.time()*1000)}"

        if result.mesh is not None:
            glb_path = str(out_file.with_suffix(".glb"))
            export_glb(result.mesh, glb_path)
            vertex_count = len(result.mesh.vertices)
            face_count = len(result.mesh.faces)
            status_msg = (
                f"✅ Done in {latency:.1f}s\n"
                f"Engine: {engine_name}\n"
                f"Vertices: {vertex_count:,}  |  Faces: {face_count:,}\n"
                f"Format: GLB mesh"
            )
            # Quick 3-view thumbnail
            preview = _render_preview(result.mesh)
            return glb_path, status_msg, preview

        elif result.gaussians is not None:
            ply_path = str(out_file.with_suffix(".ply"))
            export_ply(result.gaussians, ply_path)
            g_count = len(result.gaussians.positions)
            status_msg = (
                f"✅ Done in {latency:.1f}s\n"
                f"Engine: {engine_name}\n"
                f"Gaussians: {g_count:,}\n"
                f"Format: PLY point cloud\n"
                f"Note: Use Gaussian viewer (e.g. SuperSplat) to open .ply files."
            )
            return ply_path, status_msg, None

        else:
            return None, "❌ Engine returned no output.", None

    except Exception as exc:
        tb = traceback.format_exc()
        return None, f"❌ Error: {exc}\n\n{tb}", None


def _render_preview(mesh) -> Image.Image | None:
    """Generate a simple 3-view preview image using trimesh."""
    try:
        import trimesh

        tm = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            vertex_colors=(
                (mesh.vertex_colors * 255).astype(np.uint8)
                if mesh.vertex_colors is not None else None
            ),
            process=False,
        )
        # trimesh scene rendering (headless, no display needed)
        scene = tm.scene()
        frames = []
        for azim in [0, 90, 270]:
            scene.set_camera(angles=[0.3, 0, np.radians(azim)], distance=2.0)
            try:
                png = scene.save_image(resolution=(256, 256), visible=True)
                from io import BytesIO
                frames.append(Image.open(BytesIO(png)).convert("RGB"))
            except Exception:
                pass

        if not frames:
            return None

        # Stitch frames side by side
        total_w = sum(f.width for f in frames)
        combined = Image.new("RGB", (total_w, frames[0].height), "WHITE")
        x = 0
        for f in frames:
            combined.paste(f, (x, 0))
            x += f.width
        return combined
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

DESCRIPTION = """
# 🎯 Image → 3D Model

Upload any photo of an object and get a downloadable 3D model.

| Engine | Speed | Quality | VRAM | Best for |
|--------|-------|---------|------|----------|
| **TripoSR** | <0.5s | Good | ~5 GB | Quick previews, real-time |
| **TRELLIS** | ~10s | Excellent | ~16-24 GB | Production assets, clean meshes |
| **LGM** | ~5s | Very Good | ~10 GB | Detailed Gaussian outputs |

> **Tip:** For best results, use a clean image with the object centred and well-lit.  
> Background removal is applied automatically.
"""

EXAMPLES = [
    ["examples/rubber_duck.jpg", "TripoSR", True],
    ["examples/sneaker.jpg", "TRELLIS", True],
    ["examples/vase.jpg", "LGM", True],
]


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Image → 3D Model", theme=gr.themes.Soft()) as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Input Image",
                    type="pil",
                    height=320,
                )
                engine_choice = gr.Radio(
                    choices=["TripoSR", "TRELLIS", "LGM"],
                    value="TripoSR",
                    label="3D Engine",
                )
                remove_bg_toggle = gr.Checkbox(
                    value=True,
                    label="Auto-remove background",
                )
                run_btn = gr.Button("🚀 Generate 3D Model", variant="primary", size="lg")

            with gr.Column(scale=1):
                status_box = gr.Textbox(
                    label="Status",
                    lines=5,
                    interactive=False,
                    placeholder="Status will appear here after generation…",
                )
                preview_img = gr.Image(
                    label="3-View Preview",
                    type="pil",
                    interactive=False,
                    height=200,
                )
                output_file = gr.File(
                    label="⬇️ Download 3D Model (.glb / .ply)",
                    interactive=False,
                )

        run_btn.click(
            fn=generate_3d,
            inputs=[input_image, engine_choice, remove_bg_toggle],
            outputs=[output_file, status_box, preview_img],
            show_progress=True,
        )

        gr.Markdown("""
---
### 📖 How to view your 3D model
- **GLB files**: Open in [gltf.report](https://gltf.report), [3D Viewer (Windows)](https://apps.microsoft.com/detail/9nblggh42ths), or drag into [Blender](https://blender.org)
- **PLY files** (Gaussians): Open in [SuperSplat](https://supersplat.xyz) or [Gaussian Splatting Viewer](https://antimatter15.com/splat/)

### 📄 Paper
This app is the production demo for:  
*Triplanes, Gaussians, or Structured Latents? A Comparative Evaluation of Feed-Forward Paradigms for Single-Image 3D Reconstruction*
        """)

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Create example directory if it doesn't exist
    Path("examples").mkdir(exist_ok=True)

    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=True,         # generates a public gradio.live link
        show_error=True,
    )
