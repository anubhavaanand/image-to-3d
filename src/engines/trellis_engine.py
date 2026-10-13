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
"""

from __future__ import annotations

import os
import time
import gc
from typing import Dict, Any, Literal

import numpy as np
import torch
from PIL import Image

from .base import AbstractEngine, ReconstructionResult, Mesh3D, GaussianCloud


os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("ATTN_BACKEND", "xformers")

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
        slat_steps: int = 12,
        sparse_steps: int = 12,
        cfg_strength: float = 7.5,
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
            from huggingface_hub import snapshot_download

            # Load from a LOCAL snapshot: Pipeline.from_pretrained resolves each model
            # via f"{path}/{name}" and only falls back to hf_hub on failure - and that
            # fallback masks the real error behind a bogus 404 (repo_id="ckpts/...").
            # A local dir makes every model resolve as a local file, deterministically.
            print("[TRELLIS] Fetching microsoft/TRELLIS-image-large snapshot …")
            local_dir = snapshot_download("microsoft/TRELLIS-image-large")
            print(f"[TRELLIS] Loading pipeline from local snapshot: {local_dir}")
            _pipeline = TrellisImageTo3DPipeline.from_pretrained(local_dir)
            _moved = _pipeline.to(self.device)
            # On some TRELLIS versions .to() returns None instead of self when a
            # sub-module can't be moved; fall back to the original pipeline object
            self.pipeline = _moved if _moved is not None else _pipeline
            print("[TRELLIS] Pipeline loaded.")
        except ImportError as e:
            # Don't mask the real import error with HF_TOKEN blame
            # The real cause is typically missing: spconv, nvdiffrast, FlexiCubes, diffoctreerast, utils3d
            raise ImportError(
                f"TRELLIS import failed: {e}. "
                "Missing dependencies: spconv-cu126 (py3.13-compatible wheel), nvdiffrast, xformers, "
                "FlexiCubes (git submodule), diffoctreerast (pip), utils3d (pinned commit). "
                "See notebook cell 8 for complete install sequence."
            )

    def predict(self, image: Image.Image) -> ReconstructionResult:
        """
        Run full TRELLIS pipeline on a PIL image.

        The image must already have background removed (RGBA).
        TRELLIS handles its own preprocessing internally.

        Returns
        -------
        ReconstructionResult with .mesh populated (for 'mesh' output_format)
        or .gaussians populated (for 'gaussian' output_format).
        """
        if self.pipeline is None:
            self.load_model()

        t0 = time.time()

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
            preprocess_image=True,
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

        if self.output_format == "mesh":
            raw = outputs["mesh"][0]
            mesh = Mesh3D(
                vertices=raw.vertices.cpu().numpy().astype(np.float32),
                faces=raw.faces.cpu().numpy().astype(np.int32),
            )
            metadata.update({
                "vertex_count": len(mesh.vertices),
                "face_count": len(mesh.faces),
            })
            torch.cuda.empty_cache()
            gc.collect()
            return ReconstructionResult(mesh=mesh, metadata=metadata)

        if self.output_format == "gaussian":
            gs = outputs["gaussian"][0]
            gaussians = GaussianCloud(
                positions=gs.get_xyz.detach().cpu().numpy().astype(np.float32),
                scales=gs.get_scaling.detach().cpu().numpy().astype(np.float32),
                rotations=gs.get_rotation.detach().cpu().numpy().astype(np.float32),
                opacities=gs.get_opacity.detach().cpu().numpy().astype(np.float32),
                sh_coeffs=gs.get_features[:, 0, :3].detach().cpu().numpy().astype(np.float32),
            )
            metadata.update({"gaussian_count": len(gaussians.positions)})
            torch.cuda.empty_cache()
            gc.collect()
            return ReconstructionResult(gaussians=gaussians, metadata=metadata)

        raise NotImplementedError(
            f"output_format='{self.output_format}' not yet handled in this wrapper. "
            "Use 'mesh' or 'gaussian'."
        )

    def get_engine_info(self) -> Dict[str, Any]:
        return {
            "name": "TRELLIS",
            "paper": "arXiv:2412.01506",
            "hf_repo": "microsoft/TRELLIS-image-large",
            "latency_approx": "~10s (A100, 12 ODE steps)",
            "vram_approx": "~16–24 GB (fp16)",
            "output_formats": ["mesh (FlexiCubes @ 256³)", "gaussian (SLAT)", "radiance_field"],
            "grid_resolution": "64³",
            "active_voxels_approx": "~20 000",
            "model_params": "2B (XL) / 1.1B (L) / 342M (B)",
        }
