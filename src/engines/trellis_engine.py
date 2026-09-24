"""
TRELLIS Engine — real inference wrapper.

Model:  JeffreyXiang/TRELLIS-image-large  (HuggingFace)
Arch:   DINOv2 visual features → Sparse VAE → 2-stage rectified-flow
        transformers (G_S structure + G_L latent) → FlexiCubes mesh decoder
        OR 3D Gaussian decoder OR Radiance Field decoder
VRAM:   ~16–24 GB (fp16, image-large model)
Speed:  ~10 s on A100

Install:
    pip install git+https://github.com/microsoft/TRELLIS.git
    # Also needs: spconv, flash-attn, diffusers, transformers
"""

from __future__ import annotations

import time
import numpy as np
from PIL import Image
from typing import Dict, Any, Literal

from .base import AbstractEngine, ReconstructionResult, Mesh3D, GaussianCloud


OutputFormat = Literal["mesh", "gaussian", "radiance_field"]


class TrellisEngine(AbstractEngine):
    """
    Thin wrapper around the official TRELLIS TrellisImageTo3DPipeline.

    Usage
    -----
    engine = TrellisEngine(device="cuda", output_format="mesh")
    engine.load_model()
    result = engine.predict(pil_image)
    """

    def __init__(
        self,
        device: str = "cuda",
        output_format: OutputFormat = "mesh",
        # Sampling hyper-parameters (defaults match paper)
        slat_steps: int = 12,          # ODE steps for structured latent stage
        sparse_steps: int = 12,        # ODE steps for sparse structure stage
        cfg_strength: float = 7.5,     # classifier-free guidance scale
    ):
        self.device = device
        self.output_format = output_format
        self.slat_steps = slat_steps
        self.sparse_steps = sparse_steps
        self.cfg_strength = cfg_strength
        self.pipeline = None

    # ------------------------------------------------------------------
    # AbstractEngine interface
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Download TRELLIS-image-large from HF and move to device."""
        try:
            from trellis.pipelines import TrellisImageTo3DPipeline

            print("[TRELLIS] Loading TrellisImageTo3DPipeline …")
            self.pipeline = TrellisImageTo3DPipeline.from_pretrained(
                "JeffreyXiang/TRELLIS-image-large"
            )
            self.pipeline = self.pipeline.to(self.device)
            print("[TRELLIS] Pipeline loaded.")
        except ImportError:
            raise ImportError(
                "TRELLIS is not installed. "
                "Run: pip install git+https://github.com/microsoft/TRELLIS.git"
            )

    def predict(self, image: Image.Image) -> ReconstructionResult:
        """
        Run full TRELLIS pipeline on a PIL image.

        The image must already have background removed (RGBA or white-bg RGB).

        Returns
        -------
        ReconstructionResult with .mesh populated (for 'mesh' output_format)
        or .gaussians populated (for 'gaussian' output_format).
        """
        if self.pipeline is None:
            self.load_model()

        t0 = time.time()

        # TRELLIS expects RGBA or will auto-remove background
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        outputs = self.pipeline.run(
            image,
            seed=42,
            formats=[self.output_format],
            sparse_structure_sampler_params={
                "steps": self.sparse_steps,
                "cfg_strength": self.cfg_strength,
            },
            slat_sampler_params={
                "steps": self.slat_steps,
                "cfg_strength": self.cfg_strength,
            },
        )

        latency = time.time() - t0
        print(f"[TRELLIS] Inference done in {latency:.2f}s")

        metadata = {
            "engine": "TRELLIS",
            "output_format": self.output_format,
            "latency_s": round(latency, 3),
            "slat_steps": self.slat_steps,
            "sparse_steps": self.sparse_steps,
        }

        # ---- Mesh output (FlexiCubes) ---------------------------------
        if self.output_format == "mesh":
            raw = outputs["mesh"][0]   # trimesh.Trimesh

            # TRELLIS meshes come with texture; extract vertex colors if present
            if hasattr(raw.visual, "vertex_colors") and raw.visual.vertex_colors is not None:
                vc = np.array(raw.visual.vertex_colors[:, :3], dtype=np.float32) / 255.0
            else:
                vc = None

            mesh = Mesh3D(
                vertices=np.array(raw.vertices, dtype=np.float32),
                faces=np.array(raw.faces, dtype=np.int32),
                vertex_colors=vc,
            )
            metadata.update({"vertex_count": len(mesh.vertices), "face_count": len(mesh.faces)})
            return ReconstructionResult(mesh=mesh, metadata=metadata)

        # ---- 3D Gaussian output ---------------------------------------
        elif self.output_format == "gaussian":
            gs = outputs["gaussian"][0]  # TRELLIS Gaussian object

            gaussians = GaussianCloud(
                positions=gs.get_xyz.detach().cpu().numpy().astype(np.float32),
                scales=gs.get_scaling.detach().cpu().numpy().astype(np.float32),
                rotations=gs.get_rotation.detach().cpu().numpy().astype(np.float32),
                opacities=gs.get_opacity.detach().cpu().numpy().astype(np.float32),
                sh_coeffs=gs.get_features[:, 0, :3].detach().cpu().numpy().astype(np.float32),
            )
            metadata.update({"gaussian_count": len(gaussians.positions)})
            return ReconstructionResult(gaussians=gaussians, metadata=metadata)

        else:
            raise NotImplementedError(
                f"output_format='{self.output_format}' not yet handled in this wrapper. "
                "Use 'mesh' or 'gaussian'."
            )

    def get_engine_info(self) -> Dict[str, Any]:
        return {
            "name": "TRELLIS",
            "paper": "arXiv:2412.01506",
            "hf_repo": "JeffreyXiang/TRELLIS-image-large",
            "latency_approx": "~10s (A100, 12 ODE steps)",
            "vram_approx": "~16–24 GB (fp16)",
            "output_formats": ["mesh (FlexiCubes @ 256³)", "gaussian (SLAT)", "radiance_field"],
            "grid_resolution": "64³",
            "active_voxels_approx": "~20 000",
            "model_params": "2B (XL) / 1.1B (L) / 342M (B)",
        }
