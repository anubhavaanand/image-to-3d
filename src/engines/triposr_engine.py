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

            print("[TripoSR] Loading model from stabilityai/TripoSR …")
            self.model = TSR.from_pretrained(
                "stabilityai/TripoSR",
                config_name="config.yaml",
                weight_name="model.ckpt",
            )
            self.model.renderer.set_chunk_size(self.chunk_size)
            self.model.to(self.device)
            print("[TripoSR] Model loaded.")
        except ImportError:
            raise ImportError(
                "TripoSR is not installed. "
                "Run: pip install git+https://github.com/VAST-AI-Research/TripoSR.git"
            )

    def predict(self, image: Image.Image) -> ReconstructionResult:
        """
        Run full TripoSR pipeline on a PIL image.

        The image must already have background removed and be composited
        on a white background (use BackgroundRemover first).

        Returns
        -------
        ReconstructionResult with .mesh populated (Mesh3D).
        """
        import torch

        if self.model is None:
            self.load_model()

        t0 = time.time()

        # Resize to 512×512 as expected by the model
        image = image.resize((512, 512), Image.LANCZOS)

        with torch.no_grad():
            # scene_codes shape: (1, C)  — the triplane latent
            scene_codes = self.model([image], device=self.device)

            # Render meshes via Marching Cubes at resolution 256
            meshes = self.model.extract_mesh(
                scene_codes,
                resolution=256,
                threshold=25.0,
            )

        mesh_obj = meshes[0]  # trimesh.Trimesh

        latency = time.time() - t0
        print(f"[TripoSR] Inference done in {latency:.2f}s")

        result_mesh = Mesh3D(
            vertices=np.array(mesh_obj.vertices, dtype=np.float32),
            faces=np.array(mesh_obj.faces, dtype=np.int32),
            vertex_colors=(
                np.array(mesh_obj.visual.vertex_colors[:, :3], dtype=np.float32) / 255.0
                if mesh_obj.visual is not None and hasattr(mesh_obj.visual, "vertex_colors")
                   and mesh_obj.visual.vertex_colors is not None
                else None
            ),
        )

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
