"""
TripoSR Engine — audited wrapper for VAST-AI-Research/TripoSR (stabilityai/TripoSR).

Matches official run.py / gradio_app.py:
1. True RGBA foreground crop via `resize_foreground(rgba, 0.85)` + 0.5 mid-grey bg.
2. Multi-depth inward-normal triplane color queries (`[0.0, -0.015, -0.030]`) to
   capture the saturated interior NeRF color rather than the boundary blend.
3. Applies TripoSR's canonical orientation transform (`to_gradio_3d_orientation`:
   Rx(-90) then Ry(+90)) so the output mesh is upright (+Y up) in OpenGL space.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
from PIL import Image

from .base import AbstractEngine, ReconstructionResult, Mesh3D


class TriposrEngine(AbstractEngine):
    def __init__(self, device: str = "cuda", chunk_size: int = 8192, resolution: int = 256):
        self.device = device
        self.chunk_size = chunk_size
        self.resolution = resolution
        self.model = None
        self._resize_foreground = None

    def _ensure_tsr_imports(self) -> None:
        try:
            from tsr.system import TSR  # noqa: F401
        except ImportError:
            for c in [
                Path("/kaggle/working/TripoSR"),
                Path(__file__).resolve().parents[2] / "TripoSR",
                Path(__file__).resolve().parents[3] / "TripoSR",
            ]:
                if (c / "tsr").is_dir() and str(c) not in sys.path:
                    sys.path.insert(0, str(c))
                    break

    def load_model(self) -> None:
        self._ensure_tsr_imports()
        from tsr.system import TSR
        from tsr.utils import resize_foreground

        print("[TripoSR] Loading stabilityai/TripoSR …")
        self.model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        self.model.renderer.set_chunk_size(self.chunk_size)
        self.model.to(self.device)
        self._resize_foreground = resize_foreground
        print("[TripoSR] Model loaded.")

    def _query_interior_vertex_colors(self, mesh_obj, scene_code) -> np.ndarray | None:
        import torch

        verts = np.asarray(mesh_obj.vertices, dtype=np.float32)
        if len(verts) == 0:
            return None
        normals = np.asarray(mesh_obj.vertex_normals, dtype=np.float32)
        if normals.shape != verts.shape:
            normals = np.zeros_like(verts)

        radius = float(self.model.renderer.cfg.radius)
        offsets = [0.0, -0.015, -0.030]
        weights = [0.25, 0.45, 0.30]
        color_accum = np.zeros((len(verts), 3), dtype=np.float32)

        with torch.no_grad():
            for off, w in zip(offsets, weights):
                pts = np.clip(verts + off * normals, -radius, radius)
                pts_t = torch.from_numpy(pts).float().to(self.device)
                queried = self.model.renderer.query_triplane(
                    self.model.decoder, pts_t, scene_code
                )
                c = queried["color"].detach().cpu().numpy().astype(np.float32)
                color_accum += w * np.clip(c, 0.0, 1.0)

        return np.clip(color_accum, 0.0, 1.0)

    def predict(self, image: Image.Image) -> ReconstructionResult:
        import torch
        import trimesh

        if self.model is None:
            self.load_model()

        t0 = time.perf_counter()

        rgba = image.convert("RGBA")
        arr = np.array(rgba)
        if arr[..., 3].min() > 250:
            non_white = np.any(arr[..., :3] < 248, axis=-1)
            if non_white.any():
                arr[..., 3] = non_white.astype(np.uint8) * 255
                rgba = Image.fromarray(arr, mode="RGBA")

        fg = self._resize_foreground(rgba, 0.85)
        img_np = np.array(fg).astype(np.float32) / 255.0
        alpha = img_np[..., 3:4]
        img_rgb = img_np[..., :3] * alpha + (1.0 - alpha) * 0.5
        image_input = Image.fromarray((np.clip(img_rgb, 0.0, 1.0) * 255.0).astype(np.uint8))

        with torch.no_grad():
            scene_codes = self.model([image_input], device=self.device)
            meshes = self.model.extract_mesh(
                scene_codes, True, resolution=self.resolution
            )

        mesh_obj = meshes[0]
        # Query colors in native triplane coordinates BEFORE rotating mesh to OpenGL Y-up
        vertex_colors = None
        try:
            vertex_colors = self._query_interior_vertex_colors(mesh_obj, scene_codes[0])
        except Exception as e:
            print(f"[TripoSR] Interior color fallback ({e})")
            if getattr(mesh_obj.visual, "vertex_colors", None) is not None:
                vc = np.asarray(mesh_obj.visual.vertex_colors, dtype=np.float32)
                if vc.ndim == 2 and len(vc) == len(mesh_obj.vertices):
                    vertex_colors = vc[:, :3] / (255.0 if vc.max() > 1.0 else 1.0)

        # Apply TripoSR's official to_gradio_3d_orientation (tsr/utils.py:471):
        # maps (+X front, +Y right, +Z up) -> (+X right, +Y up, +Z front)
        rot_x = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        rot_y = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
        mesh_obj.apply_transform(rot_y @ rot_x)

        latency = time.perf_counter() - t0
        print(f"[TripoSR] Inference done in {latency:.2f}s ({len(mesh_obj.vertices)} verts)")

        result_mesh = Mesh3D(
            vertices=np.asarray(mesh_obj.vertices, dtype=np.float32),
            faces=np.asarray(mesh_obj.faces, dtype=np.int64),
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
                "has_vertex_colors": vertex_colors is not None,
            },
        )

    def get_engine_info(self) -> Dict[str, Any]:
        return {
            "name": "TripoSR",
            "paper": "arXiv:2403.02151",
            "hf_repo": "stabilityai/TripoSR",
            "output_format": "mesh (Marching Cubes @ 256^3)",
        }
