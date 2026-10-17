"""
SPAR3D Engine — Stable Point-Aware Reconstruction of 3D Objects (arXiv:2501.04689).

Model:  stabilityai/stable-point-aware-3d (HuggingFace)
Arch:   Point diffusion (512 albedo points) -> Triplane transformer -> DMTet mesh + PBR
VRAM:   ~7-10.5 GB (low_vram_mode=True fits 15.6 GB T4 comfortably)
Speed:  ~0.7 s on A100 / ~1.5-2.5 s on T4
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from PIL import Image

from .base import AbstractEngine, ReconstructionResult, Mesh3D


class Spar3dEngine(AbstractEngine):
    """Wrapper around Stability AI's official SPAR3D pipeline."""

    def __init__(self, device: str = "cuda", texture_resolution: int = 1024, low_vram_mode: bool = True):
        self.device = device
        self.texture_resolution = texture_resolution
        self.low_vram_mode = low_vram_mode
        self.model = None

    def _locate_repo(self) -> Path:
        candidates = [
            Path("/kaggle/working/stable-point-aware-3d"),
            Path(__file__).resolve().parents[2] / "stable-point-aware-3d",
            Path(__file__).resolve().parents[3] / "stable-point-aware-3d",
        ]
        for c in candidates:
            if (c / "spar3d").is_dir():
                return c
        raise RuntimeError(
            "stable-point-aware-3d repo not found. Clone https://github.com/Stability-AI/stable-point-aware-3d."
        )

    def load_model(self) -> None:
        repo_dir = self._locate_repo()
        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))
        from spar3d.system import SPAR3D

        self.model = SPAR3D.from_pretrained(
            "stabilityai/stable-point-aware-3d",
            config_name="config.yaml",
            weight_name="model.safetensors",
            low_vram_mode=self.low_vram_mode,
        )
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, image: Image.Image) -> ReconstructionResult:
        if self.model is None:
            self.load_model()

        rgba = image.convert("RGBA")
        t0 = time.perf_counter()
        with torch.cuda.device(self.device):
            mesh_out, _ = self.model.run_image(
                rgba,
                bake_resolution=self.texture_resolution,
            )
        latency = time.perf_counter() - t0

        tm = mesh_out[0] if isinstance(mesh_out, (list, tuple)) else mesh_out
        vcols = None
        try:
            from ..evaluation.benchmark_runner import bake_texture_to_vertices, vertex_colors_rgb
            tm.visual = bake_texture_to_vertices(tm)
            vcols = vertex_colors_rgb(tm)
        except Exception:
            pass

        mesh = Mesh3D(
            vertices=np.asarray(tm.vertices, dtype=np.float32),
            faces=np.asarray(tm.faces, dtype=np.int64),
            vertex_colors=vcols,
        )
        torch.cuda.empty_cache()
        return ReconstructionResult(
            mesh=mesh,
            metadata={
                "engine": "SPAR3D",
                "latency_s": round(latency, 3),
                "vertex_count": len(mesh.vertices),
                "face_count": len(mesh.faces),
                "has_vertex_colors": vcols is not None,
            },
        )

    def get_engine_info(self) -> Dict[str, Any]:
        return {
            "name": "SPAR3D",
            "paper": "arXiv:2501.04689",
            "hf_repo": "stabilityai/stable-point-aware-3d",
            "latency_approx": "~0.7s (A100) / ~1.8s (T4)",
            "vram_approx": "~7-10 GB (low_vram_mode=True)",
            "output_format": "mesh (Point Diffusion + DMTet + PBR)",
        }
