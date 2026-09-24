"""
LGM Engine — real inference wrapper.

Model:  ashawkey/LGM  (HuggingFace)
Arch:   MVDream/ImageDream multi-view diffusion (4 views, 256×256) →
        Asymmetric U-Net (6 down + 5 up + cross-view self-attention) →
        65 536 3D Gaussians (14-channel per Gaussian) →
        optional NeRF→mesh conversion (~1 min, separate step)
VRAM:   ~10 GB (fp16, multi-view diffusion + U-Net)
Speed:  ~4 s diffusion + ~1 s U-Net = ~5 s total

Install:
    pip install git+https://github.com/3DTopia/LGM.git
    # Also needs: diffusers, accelerate, kiui (pip install kiui)
"""

from __future__ import annotations

import time
import numpy as np
from PIL import Image
from typing import Dict, Any

from .base import AbstractEngine, ReconstructionResult, Mesh3D, GaussianCloud


class LGMEngine(AbstractEngine):
    """
    Thin wrapper around the official LGM inference pipeline.

    Usage
    -----
    engine = LGMEngine(device="cuda")
    engine.load_model()
    result = engine.predict(pil_image)
    """

    def __init__(
        self,
        device: str = "cuda",
        # Multi-view diffusion hyper-parameters
        mv_guidance_scale: float = 5.0,
        mv_num_inference_steps: int = 30,
        elevation: float = 0.0,
    ):
        self.device = device
        self.mv_guidance_scale = mv_guidance_scale
        self.mv_num_inference_steps = mv_num_inference_steps
        self.elevation = elevation

        self.model = None
        self.mv_pipeline = None

    # ------------------------------------------------------------------
    # AbstractEngine interface
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load LGM U-Net and ImageDream multi-view diffusion pipeline."""
        try:
            import torch
            from lgm.models.lgm import LGM
            from diffusers import DDIMScheduler
            from mvdream.pipeline_mvdream import MVDreamPipeline  # ImageDream shares interface

            print("[LGM] Loading U-Net from ashawkey/LGM …")
            self.model = LGM.from_pretrained("ashawkey/LGM").to(self.device)
            self.model.eval()

            print("[LGM] Loading ImageDream multi-view diffusion …")
            self.mv_pipeline = MVDreamPipeline.from_pretrained(
                "ashawkey/imagedream-ipmv-diffusers",
                torch_dtype=torch.float16,
            ).to(self.device)
            self.mv_pipeline.scheduler = DDIMScheduler.from_config(
                self.mv_pipeline.scheduler.config
            )
            print("[LGM] Models loaded.")
        except ImportError as e:
            raise ImportError(
                f"LGM dependencies not installed ({e}). "
                "Run: pip install git+https://github.com/3DTopia/LGM.git kiui"
            )

    def predict(self, image: Image.Image) -> ReconstructionResult:
        """
        Run full LGM pipeline on a PIL image.

        Step 1 — Multi-view diffusion: generates 4 consistent orthogonal views.
        Step 2 — U-Net: predicts 65 536 3D Gaussians from the 4 views.

        Returns
        -------
        ReconstructionResult with .gaussians populated (GaussianCloud).
        Note: Mesh extraction is a separate, optional ~1 min step.
        """
        import torch

        if self.model is None or self.mv_pipeline is None:
            self.load_model()

        t0 = time.time()

        # ------ Step 1: multi-view generation -------------------------
        # Resize to 256 for diffusion model
        image_256 = image.resize((256, 256), Image.LANCZOS)

        mv_images = self.mv_pipeline(
            "",                              # text prompt left empty (image-only mode)
            image=image_256,
            guidance_scale=self.mv_guidance_scale,
            num_inference_steps=self.mv_num_inference_steps,
            elevation=self.elevation,
            output_type="pil",
        ).images                             # list of 4 PIL images

        t_mv = time.time()
        print(f"[LGM] Multi-view diffusion: {t_mv - t0:.2f}s")

        # ------ Step 2: Gaussian prediction ---------------------------
        # Stack views to (4, H, W, 3) → normalise
        mv_np = np.stack([np.array(img) for img in mv_images], axis=0)  # (4,256,256,3)

        with torch.no_grad():
            gaussians_raw = self.model.forward_gaussians(
                images=torch.from_numpy(mv_np).float().permute(0, 3, 1, 2)
                      .unsqueeze(0).to(self.device) / 255.0,  # (1,4,3,256,256)
            )  # returns kiui.Gaussians or dict with keys pos/scale/rot/opacity/sh

        t_unet = time.time()
        print(f"[LGM] U-Net Gaussian prediction: {t_unet - t_mv:.2f}s")

        # Normalise output depending on LGM version's return type
        if hasattr(gaussians_raw, "pos"):
            pos  = gaussians_raw.pos.squeeze(0).cpu().numpy().astype(np.float32)
            scl  = gaussians_raw.scale.squeeze(0).cpu().numpy().astype(np.float32)
            rot  = gaussians_raw.rotation.squeeze(0).cpu().numpy().astype(np.float32)
            opa  = gaussians_raw.opacity.squeeze(0).cpu().numpy().astype(np.float32)
            sh   = gaussians_raw.rgbs.squeeze(0).cpu().numpy().astype(np.float32)
        else:
            # dict-style return
            pos  = gaussians_raw["pos"].squeeze(0).cpu().numpy().astype(np.float32)
            scl  = gaussians_raw["scale"].squeeze(0).cpu().numpy().astype(np.float32)
            rot  = gaussians_raw["rotation"].squeeze(0).cpu().numpy().astype(np.float32)
            opa  = gaussians_raw["opacity"].squeeze(0).cpu().numpy().astype(np.float32)
            sh   = gaussians_raw["rgbs"].squeeze(0).cpu().numpy().astype(np.float32)

        latency = time.time() - t0
        print(f"[LGM] Total inference: {latency:.2f}s")

        gaussians = GaussianCloud(
            positions=pos,
            scales=scl,
            rotations=rot,
            opacities=opa,
            sh_coeffs=sh,
        )

        return ReconstructionResult(
            gaussians=gaussians,
            metadata={
                "engine": "LGM",
                "latency_s": round(latency, 3),
                "latency_mv_s": round(t_mv - t0, 3),
                "latency_unet_s": round(t_unet - t_mv, 3),
                "gaussian_count": len(pos),
            },
        )

    def get_engine_info(self) -> Dict[str, Any]:
        return {
            "name": "LGM",
            "paper": "arXiv:2402.05054",
            "hf_repo": "ashawkey/LGM",
            "latency_approx": "~4s (diffusion) + ~1s (U-Net) = ~5s total",
            "vram_approx": "~10 GB (fp16)",
            "output_format": "GaussianCloud (65 536 Gaussians, 14-ch)",
            "backbone": "Asymmetric U-Net (6 down / 5 up + cross-view attention)",
            "mv_views": 4,
            "mv_resolution": "256×256",
        }
