"""
TripoSR Engine — real inference wrapper.

Model:  stabilityai/TripoSR  (HuggingFace)
Arch:   DINOv1 encoder → 16-layer Image-to-Triplane transformer →
        40-ch triplane (64×64) → NeRF MLP → Marching Cubes @ 256³
VRAM:   ~4–6 GB (fp16)
Speed:  <0.5 s on A100 / ~1–2 s on T4

Install:
    pip install git+https://github.com/VAST-AI-Research/TripoSR.git
"""

from __future__ import annotations

import time
import numpy as np
from PIL import Image
from typing import Dict, Any

from .base import AbstractEngine, ReconstructionResult, Mesh3D


class TriposrEngine(AbstractEngine):
    """
    Thin wrapper around the official TripoSR pipeline.

    Usage
    -----
    engine = TriposrEngine(device="cuda")
    engine.load_model()
    result = engine.predict(pil_image)          # returns ReconstructionResult
    """

    def __init__(self, device: str = "cuda", chunk_size: int = 8192):
        self.device = device
        self.chunk_size = chunk_size   # NeRF MLP chunk size; reduce if OOM
        self.model = None

    # ------------------------------------------------------------------
    # AbstractEngine interface
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Download weights from HF hub and move to device."""
        try:
            import torch
            from tsr.system import TSR  # installed via TripoSR repo
            from tsr.utils import resize_foreground

            print("[TripoSR] Loading model from stabilityai/TripoSR …")
            self.model = TSR.from_pretrained(
                "stabilityai/TripoSR",
                config_name="config.yaml",
                weight_name="model.ckpt",
            )
            self.model.renderer.set_chunk_size(self.chunk_size)
            self.model.to(self.device)
            self._resize_foreground = resize_foreground
            print("[TripoSR] Model loaded.")
        except ImportError:
            raise ImportError(
                "TripoSR is not installed. "
                "Run: pip install git+https://github.com/VAST-AI-Research/TripoSR.git"
            )

    def predict(self, image: Image.Image) -> ReconstructionResult:
        """
        Run full TripoSR pipeline on a PIL image.

        The image must already have background removed (RGBA).
        This engine applies TripoSR-specific foreground resizing and
        gray-background compositing before inference.

        Returns
        -------
        ReconstructionResult with .mesh populated (Mesh3D).
        """
        import torch

        if self.model is None:
            self.load_model()

        t0 = time.time()

        # Model-specific preprocessing: resize foreground to 85% of image size
        image = self._resize_foreground(image, 0.85)

        # Convert to numpy float32 and composite onto gray background
        img_np = np.array(image).astype(np.float32) / 255.0
        img_np = img_np[..., :3] * img_np[..., 3:4] + (1.0 - img_np[..., 3:4]) * 0.5
        image_input = Image.fromarray((img_np * 255.0).astype(np.uint8))

        with torch.no_grad():
            scene_codes = self.model([image_input], device=self.device)
            meshes = self.model.extract_mesh(
                scene_codes,
                False,  # skip vertex colors — MC vertices yield near-uniform grey via sigmoid
                resolution=256,
            )

        mesh_obj = meshes[0]  # trimesh.Trimesh

        latency = time.time() - t0
        print(f"[TripoSR] Inference done in {latency:.2f}s")

        # Bake UV texture from the triplane NeRF decoder (queries at dense atlas
        # texels instead of sparse MC vertices, restoring full color fidelity).
        # Falls back to mesh vertex colors (near-grey sigmoid) if xatlas/moderngl unavailable.
        vertex_colors = None
        try:
            from tsr.bake_texture import bake_texture as _bake_texture
            texture = _bake_texture(mesh_obj, self.model, scene_codes[0], 512)
            if texture is not None:
                # Sample UV texture at vertex positions via barycentric lookup
                from scipy.spatial import cKDTree
                uv = texture["uvs"]
                colors = texture["colors"]  # (H, W, 4)
                # Build vertex colors by querying the rendered atlas at each vertex's UV
                # Use the atlas vmapping to get per-vertex UV coords
                vmapping = texture["vmapping"]
                vertex_uvs = uv[vmapping]  # (n_verts, 2) in [0, 1]
                h, w = colors.shape[:2]
                u = np.clip((vertex_uvs[:, 0] * (w - 1)).astype(int), 0, w - 1)
                v = np.clip(((1.0 - vertex_uvs[:, 1]) * (h - 1)).astype(int), 0, h - 1)
                rgba = colors[v, u]  # (n_verts, 4)
                vertex_colors = rgba[:, :3].astype(np.float32) / 255.0
                # Clip to valid range (sigmoid output)
                vertex_colors = np.clip(vertex_colors, 0.0, 1.0)
                print(f"[TripoSR] Baked UV texture: {len(vertex_colors)} verts with color")
        except Exception as e:
            print(f"[TripoSR] Texture baking unavailable ({e}), falling back to "
                  f"mesh vertex colors (near-grey via sigmoid)")
            # Fallback: re-extract with vertex colors. The MC vertices use sigmoid
            # activation which gives near-uniform grey, but that's better than
            # None (which produces completely black renders in pyrender).
            try:
                fallback_meshes = self.model.extract_mesh(
                    scene_codes,
                    True,  # has_vertex_color=True — gets sigmoid-based greys
                    resolution=256,
                )
                mesh_obj = fallback_meshes[0] if isinstance(fallback_meshes, list) else fallback_meshes
                # Extract whatever vertex colors the fallback mesh has
                if mesh_obj.visual.vertex_colors is not None:
                    vc_arr = np.array(mesh_obj.visual.vertex_colors, dtype=np.float32)
                    if vc_arr.ndim == 2 and vc_arr.shape[0] == len(mesh_obj.vertices):
                        vertex_colors = vc_arr[:, :3] / 255.0
                    else:
                        vertex_colors = None
            except Exception as fallback_e:
                print(f"[TripoSR] Fallback also failed: {fallback_e}")
                vertex_colors = None

        result_mesh = Mesh3D(
            vertices=np.array(mesh_obj.vertices, dtype=np.float32),
            faces=np.array(mesh_obj.faces, dtype=np.int32),
            vertex_colors=vertex_colors,
        )

        torch.cuda.empty_cache()

        return ReconstructionResult(
            mesh=result_mesh,
            metadata={
                "engine": "TripoSR",
                "latency_s": round(latency, 3),
                "vertex_count": len(result_mesh.vertices),
                "face_count": len(result_mesh.faces),
            },
        )

    def get_engine_info(self) -> Dict[str, Any]:
        return {
            "name": "TripoSR",
            "paper": "arXiv:2403.02151",
            "hf_repo": "stabilityai/TripoSR",
            "latency_approx": "<0.5s (A100) / ~1-2s (T4)",
            "vram_approx": "4–6 GB (fp16)",
            "output_format": "mesh (Marching Cubes @ 256³)",
            "triplane_resolution": "64×64",
            "triplane_channels": 40,
        }
