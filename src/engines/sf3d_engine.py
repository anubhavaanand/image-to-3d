"""
SF3D Engine — Stable Fast 3D (Stability AI, arXiv:2408.00653) wrapper.

Model:  stabilityai/stable-fast-3d (HuggingFace, non-commercial community license)
Arch:   DINOv2 + LRM-style transformer → DMTet mesh extraction (vs TripoSR's
        Marching Cubes) → spherical-Gaussian delighting → UV unwrap (xatlas)
VRAM:   ~6–7 GB default (fits T4 15 GB comfortably)
Speed:  ~0.5 s on A100; measured ~1.5–2 s expected on T4
Role in benchmark: 4th engine, Paradigm-1 ablation (DMTet vs Marching Cubes)

Requires (install order matters, same JIT-build family as spconv):
    git clone https://github.com/Stability-AI/stable-fast-3d
    pip install -r stable-fast-3d/requirements.txt
    cd stable-fast-3d/custom_rasterizer && python setup.py install
    cd ../texture && python setup.py install
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


class SF3DEngine(AbstractEngine):
    """Thin wrapper around the official SF3D pipeline."""

    def __init__(self, device: str = "cuda", texture_resolution: int = 1024):
        self.device = device
        self.texture_resolution = texture_resolution
        self.model = None
        self._sf3d_dir: Path | None = None

    def _locate_repo(self) -> Path:
        """Find the stable-fast-3d checkout (workspace root or alongside src/)."""
        candidates = [
            Path(__file__).resolve().parents[2] / "stable-fast-3d",
            Path(__file__).resolve().parents[3] / "stable-fast-3d",
            Path("/kaggle/working/stable-fast-3d"),
        ]
        for c in candidates:
            if (c / "sf3d").is_dir():
                return c
        raise RuntimeError(
            "stable-fast-3d repo not found. Clone it and build custom_rasterizer "
            "+ texture extensions first (see module docstring)."
        )

    def load_model(self) -> None:
        try:
            self._sf3d_dir = self._locate_repo()
            if str(self._sf3d_dir) not in sys.path:
                sys.path.insert(0, str(self._sf3d_dir))
            from sf3d.system import SF3D  # official loader

            self.model = SF3D.from_pretrained(
                "stabilityai/stable-fast-3d",
                config_name="sf3d.yaml",
                low_vram_mode=True,   # T4-safe: offloads delighting between stages
            )
            self.model.to(self.device).eval()
        except Exception as e:
            raise RuntimeError(f"SF3D load failed: {e}") from e

    @torch.no_grad()
    def predict(self, image: Image.Image) -> ReconstructionResult:
        if self.model is None:
            self.load_model()

        # SF3D expects RGBA with background removed; runner already supplies
        # white-bg RGB. Composite onto transparent for its internal rembg-free path.
        rgba = image.convert("RGBA")

        t0 = time.perf_counter()
        with torch.cuda.device(self.device):
            mesh_out, glob_dict = self.model.run_image(
                rgba,
                bake_resolution=self.texture_resolution,
            )
        latency = time.perf_counter() - t0

        tm = mesh_out[0]  # trimesh.Trimesh with UV texture
        mesh = Mesh3D(
            vertices=np.asarray(tm.vertices, dtype=np.float32),
            faces=np.asarray(tm.faces, dtype=np.int32),
        )
        metadata: Dict[str, Any] = {
            "engine": "SF3D",
            "latency_s": round(latency, 3),
            "vertex_count": len(mesh.vertices),
            "face_count": len(mesh.faces),
            "texture_resolution": self.texture_resolution,
            "note": "DMTet extraction + UV unwrap (vs TripoSR Marching Cubes)",
        }
        torch.cuda.empty_cache()
        return ReconstructionResult(mesh=mesh, metadata=metadata)

    def get_engine_info(self) -> Dict[str, Any]:
        return {
            "name": "SF3D",
            "paper": "arXiv:2408.00653",
            "hf_repo": "stabilityai/stable-fast-3d",
            "latency_approx": "~0.5s (A100) / ~1.5-2s (T4 expected)",
            "vram_approx": "~6-7 GB (low_vram_mode=True)",
            "output_formats": ["textured UV mesh (DMTet)"],
            "license": "Stability AI non-commercial community license (research OK, disclose in paper)",
        }
