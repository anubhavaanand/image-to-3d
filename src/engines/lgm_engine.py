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
    1. git clone https://github.com/3DTopia/LGM.git
    2. pip install -r LGM/requirements.txt
    3. git clone https://github.com/ashawkey/diff-gaussian-rasterization.git && pip install ./diff-gaussian-rasterization
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
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
        self.rays_embeddings = None
        self.opt = None

    # ------------------------------------------------------------------
    # AbstractEngine interface
    # ------------------------------------------------------------------

    def _ensure_lgm_imports(self):
        """Ensure LGM repo is importable. The repo is not pip-installable."""
        try:
            from core.options import Options  # noqa: F401
            from core.models import LGM  # noqa: F401
        except ImportError:
            lgm_path = Path(__file__).resolve().parents[2] / "LGM"
            if lgm_path.exists() and str(lgm_path) not in sys.path:
                sys.path.insert(0, str(lgm_path))
            from core.options import Options  # noqa: F401
            from core.models import LGM  # noqa: F401

    def load_model(self) -> None:
        """Load LGM U-Net and ImageDream multi-view diffusion pipeline."""
        try:
            import torch
            from diffusers import DDIMScheduler
            from mvdream.pipeline_mvdream import MVDreamPipeline

            self._ensure_lgm_imports()
            from core.options import Options
            from core.models import LGM

            print("[LGM] Loading U-Net from ashawkey/LGM …")
            self.opt = Options()
            self.model = LGM(self.opt)
            ckpt_path = hf_hub_download(
                repo_id="ashawkey/LGM",
                filename="model_fp16_fixrot.safetensors",
            )
            ckpt = load_file(ckpt_path, device="cpu")
            self.model.load_state_dict(ckpt, strict=False)
            self.model = self.model.half().to(self.device).eval()

            print("[LGM] Loading ImageDream multi-view diffusion …")
            self.mv_pipeline = MVDreamPipeline.from_pretrained(
                "ashawkey/imagedream-ipmv-diffusers",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            ).to(self.device)
            self.mv_pipeline.scheduler = DDIMScheduler.from_config(
                self.mv_pipeline.scheduler.config
            )

            self.rays_embeddings = self.model.prepare_default_rays(self.device)
            print("[LGM] Models loaded.")
        except ImportError as e:
            raise ImportError(
                f"LGM dependencies not installed ({e}). "
                "Clone LGM repo: git clone https://github.com/3DTopia/LGM.git"
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
        import torch.nn.functional as F
        import kiui

        if self.model is None or self.mv_pipeline is None:
            self.load_model()

        t0 = time.time()

        # ------ Step 1: multi-view generation -------------------------
        image_np = np.array(image)
        mask = image_np[..., -1] > 0
        image_np = kiui.op.recenter(image_np, mask, border_ratio=0.2)
        image_np = image_np.astype(np.float32) / 255.0

        if image_np.shape[-1] == 4:
            image_np = image_np[..., :3] * image_np[..., 3:4] + (1.0 - image_np[..., 3:4])

        mv_image = self.mv_pipeline(
            "",
            image_np,
            guidance_scale=self.mv_guidance_scale,
            num_inference_steps=self.mv_num_inference_steps,
            elevation=self.elevation,
        )
        mv_image = np.stack([mv_image[1], mv_image[2], mv_image[3], mv_image[0]], axis=0)

        t_mv = time.time()
        print(f"[LGM] Multi-view diffusion: {t_mv - t0:.2f}s")

        # ------ Step 2: Gaussian prediction ---------------------------
        input_image = torch.from_numpy(mv_image).permute(0, 3, 1, 2).float().to(self.device)
        input_image = F.interpolate(
            input_image,
            size=(self.opt.input_size, self.opt.input_size),
            mode="bilinear",
            align_corners=False,
        )
        input_image = torch.nn.functional.normalize(
            input_image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        input_image = torch.cat([input_image, self.rays_embeddings], dim=1).unsqueeze(0)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                gaussians = self.model.forward_gaussians(input_image)

        t_unet = time.time()
        print(f"[LGM] U-Net Gaussian prediction: {t_unet - t_mv:.2f}s")

        g = gaussians.squeeze(0).cpu().numpy().astype(np.float32)
        pos = g[..., 0:3]
        opa = g[..., 3:4]
        scl = g[..., 4:7]
        rot = g[..., 7:11]
        sh = g[..., 11:14]

        latency = time.time() - t0
        print(f"[LGM] Total inference: {latency:.2f}s")

        gaussians_out = GaussianCloud(
            positions=pos,
            scales=scl,
            rotations=rot,
            opacities=opa,
            sh_coeffs=sh,
        )

        torch.cuda.empty_cache()

        return ReconstructionResult(
            gaussians=gaussians_out,
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
