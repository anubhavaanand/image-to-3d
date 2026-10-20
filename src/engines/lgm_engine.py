"""
LGM Engine — audited wrapper for 3DTopia/LGM (ashawkey/LGM).

Fixes applied from repo_2_lgm.txt & repo_1_triposr.txt audit:
1. True RGBA foreground mask for `kiui.op.recenter(image_np, mask, border_ratio=0.2)`.
2. Filters transparent background splats (`opacity > 0.05`) and 1% outliers BEFORE
   computing the 256^3 Marching Cubes bounding box.
3. Uses `skimage.measure.marching_cubes` directly on the CPU density grid so vertex
   axes stay strictly `(x, y, z)` (avoiding `torchmcubes`'s `[2, 1, 0]` mirror flip!).
4. Cleans floating islands via `tm.split(only_watertight=False)` and bakes 3-channel
   RGB colors from k=4 nearest foreground Gaussians.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from .base import AbstractEngine, ReconstructionResult, Mesh3D, GaussianCloud


def _install_lgm_xformers_shim() -> None:
    try:
        import xformers.ops  # noqa: F401
        return
    except Exception:
        pass
    import types
    import torch.nn.functional as _F

    def _memory_efficient_attention(q, k, v, attn_bias=None, op=None, **_kw):
        if q.ndim == 3:
            out = _F.scaled_dot_product_attention(
                q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), attn_mask=attn_bias
            )
            return out.squeeze(1)
        return _F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)

    _xm = types.ModuleType("xformers")
    _xops = types.ModuleType("xformers.ops")
    _xops.memory_efficient_attention = _memory_efficient_attention
    _xm.ops = _xops
    sys.modules["xformers"] = _xm
    sys.modules["xformers.ops"] = _xops
    print("[LGM] Installed PyTorch SDPA shim for xformers.ops.memory_efficient_attention ✅")


class LGMEngine(AbstractEngine):
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

    def _ensure_lgm_imports(self) -> None:
        for repo_name in ("LGM", "mvdream"):
            for c in [
                Path(f"/kaggle/working/{repo_name}"),
                Path(__file__).resolve().parents[2] / repo_name,
                Path(__file__).resolve().parents[3] / repo_name,
            ]:
                if c.is_dir() and str(c) not in sys.path:
                    sys.path.insert(0, str(c))
                    break

    def load_model(self, big: bool = True) -> None:
        _install_lgm_xformers_shim()
        self._ensure_lgm_imports()
        from diffusers import DDIMScheduler
        from mvdream.pipeline_mvdream import MVDreamPipeline
        from core.options import Options, config_defaults
        from core.models import LGM

        print("[LGM] Loading U-Net from ashawkey/LGM …")
        self.opt = config_defaults["big"] if big else Options()
        self.model = LGM(self.opt)
        ckpt_path = hf_hub_download(
            repo_id="ashawkey/LGM", filename="model_fp16_fixrot.safetensors"
        )
        ckpt = load_file(ckpt_path, device="cpu")
        model_sd = self.model.state_dict()
        filtered = {
            k: v for k, v in ckpt.items()
            if k in model_sd and model_sd[k].shape == v.shape
        }
        self.model.load_state_dict(filtered, strict=False)
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
        print("[LGM] Model loaded.")

    def predict(self, image: Image.Image, extract_mesh: bool = True) -> ReconstructionResult:
        import torch.nn.functional as F
        import kiui

        if self.model is None or self.mv_pipeline is None:
            self.load_model(big=True)

        t0 = time.perf_counter()

        rgba = image.convert("RGBA")
        image_np = np.array(rgba)
        mask = image_np[..., -1] > 10
        if mask.mean() > 0.98:
            non_white = np.any(image_np[..., :3] < 248, axis=-1)
            if non_white.any():
                mask = non_white

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
        t_mv = time.perf_counter()

        input_image = torch.from_numpy(mv_image).permute(0, 3, 1, 2).float().to(self.device)
        input_image = F.interpolate(
            input_image,
            size=(self.opt.input_size, self.opt.input_size),
            mode="bilinear",
            align_corners=False,
        )
        mean = torch.tensor([0.485, 0.456, 0.406], device=input_image.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=input_image.device).view(1, 3, 1, 1)
        input_image = (input_image - mean) / std
        input_image = torch.cat([input_image, self.rays_embeddings], dim=1).unsqueeze(0)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                gaussians = self.model.forward_gaussians(input_image)

        t_unet = time.perf_counter()

        g = gaussians.squeeze(0).detach().cpu().numpy().astype(np.float32)
        pos = g[..., 0:3]
        opa = g[..., 3:4]
        scl = g[..., 4:7]
        rot = g[..., 7:11]
        rgb = np.clip(g[..., 11:14], 0.0, 1.0)

        gaussians_out = GaussianCloud(
            positions=pos,
            scales=scl,
            rotations=rot,
            opacities=opa,
            sh_coeffs=rgb,
        )

        mesh_out = self._gaussians_to_mesh(gaussians_out) if extract_mesh else None
        latency = time.perf_counter() - t0
        torch.cuda.empty_cache()

        return ReconstructionResult(
            gaussians=gaussians_out,
            mesh=mesh_out,
            metadata={
                "engine": "LGM",
                "latency_s": round(latency, 3),
                "latency_mv_s": round(t_mv - t0, 3),
                "latency_unet_s": round(t_unet - t_mv, 3),
                "gaussian_count": len(pos),
                "vertex_count": len(mesh_out.vertices) if mesh_out is not None else 0,
                "face_count": len(mesh_out.faces) if mesh_out is not None else 0,
                "has_vertex_colors": (mesh_out is not None and mesh_out.vertex_colors is not None),
            },
        )

    def _gaussians_to_mesh(
        self, gaussians: GaussianCloud, resolution: int = 256
    ) -> Optional[Mesh3D]:
        try:
            import trimesh
            from scipy.spatial import cKDTree
            from skimage import measure

            pos_all = np.asarray(gaussians.positions, dtype=np.float32)
            opa_all = np.asarray(gaussians.opacities, dtype=np.float32).reshape(-1)
            rgb_all = np.clip(np.asarray(gaussians.sh_coeffs, dtype=np.float32), 0.0, 1.0)

            valid = opa_all > 0.05
            if valid.sum() < 200:
                valid = opa_all > 0.01
            if valid.sum() < 50:
                valid = np.ones_like(opa_all, dtype=bool)

            pos_fg = pos_all[valid]
            opa_fg = opa_all[valid]
            rgb_fg = rgb_all[valid]

            med = np.median(pos_fg, axis=0)
            dists = np.linalg.norm(pos_fg - med, axis=1)
            inlier = dists <= np.percentile(dists, 99.0)
            if inlier.sum() >= 50:
                pos_fg = pos_fg[inlier]
                opa_fg = opa_fg[inlier]
                rgb_fg = rgb_fg[inlier]

            pos_t = torch.from_numpy(pos_fg).float()
            opa_t = torch.from_numpy(opa_fg).float()

            pos_min, pos_max = pos_t.min(0)[0], pos_t.max(0)[0]
            pos_center = (pos_min + pos_max) / 2.0
            pos_scale = ((pos_max - pos_min).max() / 2.0) * 1.15
            pos_norm = (pos_t - pos_center) / (pos_scale + 1e-8)

            grid = torch.zeros((resolution, resolution, resolution), dtype=torch.float32)
            indices = ((pos_norm + 1.0) / 2.0 * (resolution - 1)).long().clamp(0, resolution - 1)
            grid.index_put_(
                (indices[:, 0], indices[:, 1], indices[:, 2]),
                opa_t,
                accumulate=True,
            )

            _k, _sigma = 9, 1.5
            _ax = torch.arange(_k, dtype=torch.float32) - (_k - 1) / 2.0
            _g = torch.exp(-(_ax ** 2) / (2.0 * (_sigma ** 2)))
            _g = (_g / _g.sum()).view(1, 1, _k, 1, 1)
            _pad = _k // 2
            g4 = grid.unsqueeze(0).unsqueeze(0)
            g4 = torch.nn.functional.conv3d(g4, _g, padding=(_pad, 0, 0))
            g4 = torch.nn.functional.conv3d(g4, _g.view(1, 1, 1, _k, 1), padding=(0, _pad, 0))
            g4 = torch.nn.functional.conv3d(g4, _g.view(1, 1, 1, 1, _k), padding=(0, 0, _pad))
            grid_np = g4[0, 0].cpu().numpy()

            nonzero = grid_np[grid_np > 1e-4]
            if len(nonzero) == 0:
                return None
            iso = float(np.percentile(nonzero, 35))

            # Always use skimage.measure.marching_cubes on CPU grid_np so vertex
            # coordinates stay in (ix, iy, iz) = (x, y, z) order without axis reversal!
            verts, faces, _, _ = measure.marching_cubes(grid_np, iso)
            verts = verts.astype(np.float32)
            faces = faces.astype(np.int64)

            if len(verts) == 0 or len(faces) == 0:
                return None

            verts = (verts / (resolution - 1) * 2.0 - 1.0) * pos_scale.numpy() + pos_center.numpy()

            tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            try:
                parts = tm.split(only_watertight=False)
                if len(parts) > 1:
                    max_faces = max(len(p.faces) for p in parts)
                    kept = [p for p in parts if len(p.faces) >= max(20, int(max_faces * 0.05))]
                    if kept:
                        tm = trimesh.util.concatenate(kept)
            except Exception:
                pass

            verts_final = np.asarray(tm.vertices, dtype=np.float32)
            faces_final = np.asarray(tm.faces, dtype=np.int64)

            tree = cKDTree(pos_fg)
            k = min(4, len(pos_fg))
            dists_k, idx_k = tree.query(verts_final, k=k)
            if k == 1 or idx_k.ndim == 1:
                vertex_colors = np.clip(rgb_fg[idx_k], 0.0, 1.0)
            else:
                weights = 1.0 / (dists_k + 1e-6)
                weights /= weights.sum(axis=1, keepdims=True)
                vertex_colors = np.clip(
                    (rgb_fg[idx_k] * weights[..., None]).sum(axis=1), 0.0, 1.0
                )

            return Mesh3D(
                vertices=verts_final,
                faces=faces_final,
                vertex_colors=vertex_colors.astype(np.float32),
            )
        except Exception as e:
            print(f"[LGM] Mesh extraction error: {e}")
            return None

    def get_engine_info(self) -> Dict[str, Any]:
        return {
            "name": "LGM",
            "paper": "arXiv:2402.05054",
            "hf_repo": "ashawkey/LGM",
            "output_format": "GaussianCloud (65,536 splats) + MC Mesh3D",
        }
