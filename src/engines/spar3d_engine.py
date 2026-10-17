"""
SPAR3D Engine — audited wrapper for Stability-AI/stable-point-aware-3d (arXiv:2501.04689).

Fixes applied from repo_4_spar3d.txt audit:
1. Calls `spar3d.utils.foreground_crop(rgba, crop_ratio=1.3, newsize=(512, 512))`
   before `model.run_image()` (matching run.py:128 and gradio_app.py:432).
2. Uses `low_vram_mode=True` (~7 GB VRAM on T4 via 3-stage lazy module loading).
3. Samples per-vertex colors from `PBRMaterial.baseColorTexture` using SPAR3D's
   native top-left UV convention (`spar3d/models/mesh.py:297`), so `Mesh3D.vertex_colors`
   are sharp and saturated.
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
    def __init__(
        self,
        device: str = "cuda",
        texture_resolution: int = 1024,
        low_vram_mode: bool = True,
        foreground_ratio: float = 1.3,
    ):
        self.device = device
        self.texture_resolution = texture_resolution
        self.low_vram_mode = low_vram_mode
        self.foreground_ratio = foreground_ratio
        self.model = None
        self._foreground_crop = None

    def _locate_repo(self) -> Path:
        for c in [
            Path("/kaggle/working/stable-point-aware-3d"),
            Path(__file__).resolve().parents[2] / "stable-point-aware-3d",
            Path(__file__).resolve().parents[3] / "stable-point-aware-3d",
            Path(__file__).resolve().parents[3] / "SPAR3D",
        ]:
            if (c / "spar3d").is_dir():
                return c
        raise RuntimeError(
            "stable-point-aware-3d not found. Clone https://github.com/Stability-AI/stable-point-aware-3d."
        )

    def load_model(self) -> None:
        repo_dir = self._locate_repo()
        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))

        # Provide a lightweight stub if transparent_background is not installed
        # (since our BackgroundRemover already supplies RGBA masks)
        if "transparent_background" not in sys.modules:
            try:
                import transparent_background  # noqa: F401
            except ImportError:
                import types
                tb_mod = types.ModuleType("transparent_background")
                class _DummyRemover:
                    def __init__(self, *a, **kw): pass
                    def process(self, img, *a, **kw): return img.convert("RGBA")
                tb_mod.Remover = _DummyRemover
                sys.modules["transparent_background"] = tb_mod

        from spar3d.system import SPAR3D
        from spar3d.utils import foreground_crop

        print(f"[SPAR3D] Loading stabilityai/stable-point-aware-3d (low_vram_mode={self.low_vram_mode}) …")
        self.model = SPAR3D.from_pretrained(
            "stabilityai/stable-point-aware-3d",
            config_name="config.yaml",
            weight_name="model.safetensors",
            low_vram_mode=self.low_vram_mode,
        )
        self.model.to(self.device).eval()
        self._foreground_crop = foreground_crop
        print("[SPAR3D] Model loaded.")

    @staticmethod
    def _extract_spar3d_vertex_colors(tm) -> np.ndarray | None:
        """
        Extract per-vertex RGB colors in [0, 1] from SPAR3D's TextureVisuals
        (PBRMaterial.baseColorTexture). Note: in spar3d/models/mesh.py:297,
        SPAR3D does NOT flip v to 1-v when baking `baseColorTexture`, so row
        index y is `v * (H - 1)` directly.
        """
        try:
            uv = getattr(tm.visual, "uv", None)
            mat = getattr(tm.visual, "material", None)
            if uv is None or mat is None:
                return None
            tex_img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
            if tex_img is None:
                return None
            rgb = np.asarray(tex_img.convert("RGB"), dtype=np.float32) / 255.0
            h, w = rgb.shape[:2]
            u = np.clip(np.asarray(uv[:, 0], dtype=np.float64), 0.0, 1.0)
            v = np.clip(np.asarray(uv[:, 1], dtype=np.float64), 0.0, 1.0)
            x = np.clip(np.round(u * (w - 1)).astype(int), 0, w - 1)
            y = np.clip(np.round(v * (h - 1)).astype(int), 0, h - 1)
            return np.clip(rgb[y, x], 0.0, 1.0).astype(np.float32)
        except Exception as e:
            print(f"[SPAR3D] Vertex color extraction warning: {e}")
            return None

    @torch.no_grad()
    def predict(self, image: Image.Image) -> ReconstructionResult:
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

        # Crop foreground with SPAR3D's official 1.3 padding ratio (run.py:128)
        try:
            rgba = self._foreground_crop(
                rgba, crop_ratio=self.foreground_ratio, newsize=(512, 512)
            )
        except Exception as crop_err:
            print(f"[SPAR3D] foreground_crop fallback ({crop_err})")

        torch.manual_seed(42)
        np.random.seed(42)

        mesh_out, _ = self.model.run_image(
            rgba,
            bake_resolution=self.texture_resolution,
            remesh="none",
            estimate_illumination=False,
        )
        latency = time.perf_counter() - t0

        tm = mesh_out[0] if isinstance(mesh_out, (list, tuple)) else mesh_out
        vcols = self._extract_spar3d_vertex_colors(tm)

        print(f"[SPAR3D] Inference done in {latency:.2f}s ({len(tm.vertices)} verts)")

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
            "output_format": "mesh (Point Diffusion + DMTet + PBR)",
        }
