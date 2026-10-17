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

    def load_model(self, big: bool = True) -> None:
        """Load LGM U-Net and ImageDream multi-view diffusion pipeline.
        
        Args:
            big: If True, use Options(big=True) for 65,536 Gaussians (matches paper).
                 If False, use Options() for 16,384 Gaussians (faster, lower quality).
        """
        try:
            import torch
            from diffusers import DDIMScheduler
            from mvdream.pipeline_mvdream import MVDreamPipeline

            self._ensure_lgm_imports()
            from core.options import Options, config_defaults
            from core.models import LGM

            print("[LGM] Loading U-Net from ashawkey/LGM …")
            # LGM has no 'big' flag - it ships named presets; 'big' = splat_size 128 / output_size 512
            # (+ one extra up-decoder), which is the architecture of the released checkpoint.
            self.opt = config_defaults['big'] if big else Options()
            self.model = LGM(self.opt)
            ckpt_path = hf_hub_download(
                repo_id="ashawkey/LGM",
                filename="model_fp16_fixrot.safetensors",
            )
            ckpt = load_file(ckpt_path, device="cpu")
            # Safe load: filter out keys with mismatched shapes (128 vs 256 channel bug)
            _model_sd = self.model.state_dict()
            _filtered = {}
            _skipped = []
            for k, v in ckpt.items():
                if k in _model_sd and _model_sd[k].shape == v.shape:
                    _filtered[k] = v
                else:
                    _skipped.append(k)
            if _skipped:
                print(f"[LGM] Skipped {len(_skipped)} keys with shape mismatches: {_skipped[:8]}...")
            self.model.load_state_dict(_filtered, strict=False)
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
            print(f"[LGM] Models loaded (big={big}, splat_size={self.opt.splat_size}, gaussians={self.opt.splat_size**2 * 4}).")
        except ImportError as e:
            raise ImportError(
                f"LGM dependencies not installed ({e}). "
                "Clone LGM repo: git clone https://github.com/3DTopia/LGM.git"
            )

    def predict(self, image: Image.Image, extract_mesh: bool = True) -> ReconstructionResult:
        """
        Run full LGM pipeline on a PIL image.

        Step 1 — Multi-view diffusion: generates 4 consistent orthogonal views.
        Step 2 — U-Net: predicts 65 536 3D Gaussians from the 4 views.
        Step 3 (optional) — Gaussian→Mesh conversion via density grid + marching cubes.

        Args:
            image: Input PIL image (RGBA)
            extract_mesh: If True, run Gaussian→mesh conversion (~30-60s extra)

        Returns
        -------
        ReconstructionResult with .gaussians populated, and .mesh if extract_mesh=True.
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
        _mean = torch.tensor([0.485, 0.456, 0.406], device=input_image.device).view(1, 3, 1, 1)
        _std = torch.tensor([0.229, 0.224, 0.225], device=input_image.device).view(1, 3, 1, 1)
        input_image = (input_image - _mean) / _std

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

        gaussians_out = GaussianCloud(
            positions=pos,
            scales=scl,
            rotations=rot,
            opacities=opa,
            sh_coeffs=sh,
        )

        # ------ Step 3 (optional): Gaussian → Mesh conversion ---------
        mesh_out = None
        if extract_mesh:
            print("[LGM] Extracting mesh from Gaussians...")
            mesh_out = self._gaussians_to_mesh(gaussians_out)
            if mesh_out is not None:
                print(f"[LGM] Mesh extracted: {len(mesh_out.vertices)} verts, {len(mesh_out.faces)} faces")
            else:
                print("[LGM] Mesh extraction failed")

        latency = time.time() - t0
        print(f"[LGM] Total inference: {latency:.2f}s")

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
            },
        )

    def _gaussians_to_mesh(self, gaussians: GaussianCloud, resolution: int = 256, density_threshold: float = 0.3) -> Optional[Mesh3D]:
        """
        Convert Gaussian splats to mesh via density grid + marching cubes.

        Simple approach: splat Gaussians into a 3D density grid, then run marching cubes.
        This approximates the LGM paper's Instant-NGP distillation but is much faster.
        """
        try:
            import torch
            import numpy as np
            
            # Try torchmcubes first (faster on GPU), fall back to skimage
            try:
                import torchmcubes
                has_torchmcubes = True
            except ImportError:
                from skimage import measure
                has_torchmcubes = False
            
            # Normalize Gaussian positions to [-1, 1] grid
            pos = torch.from_numpy(gaussians.positions).float()
            # reshape(-1): opacities arrive as [N, 1]; a 2-D value column breaks
            # index_put_ broadcasting against the [N] index columns
            opa = torch.from_numpy(gaussians.opacities).float().reshape(-1)
            scl = torch.from_numpy(gaussians.scales).float()
            
            # Center and scale to unit cube
            pos_min, pos_max = pos.min(0)[0], pos.max(0)[0]
            pos_center = (pos_min + pos_max) / 2
            pos_scale = (pos_max - pos_min).max() / 2
            pos_norm = (pos - pos_center) / (pos_scale + 1e-8)
            
            # Create density grid by splatting Gaussians
            # Simple approach: each Gaussian contributes to nearby voxels
            grid = torch.zeros((resolution, resolution, resolution), dtype=torch.float32)
            
            # Convert normalized positions to grid indices
            indices = ((pos_norm + 1) / 2 * (resolution - 1)).long()
            indices = indices.clamp(0, resolution - 1)
            
            # Splat opacities at Gaussian centers (vectorized index_add)
            indices = ((pos_norm + 1) / 2 * (resolution - 1)).long()
            indices = indices.clamp(0, resolution - 1)
            grid.index_put_(
                (indices[:, 0], indices[:, 1], indices[:, 2]), opa, accumulate=True
            )

            # Spread each point into a smooth blob via separable Gaussian smoothing.
            # Raw point-splatting yields a hard, single-voxel shell that marching
            # cubes extracts as disconnected fragments (good CD, dotted renders);
            # a continuous field produces a watertight surface.
            _k, _sigma = 7, 0.8
            _ax = torch.arange(_k, dtype=torch.float32) - (_k - 1) / 2
            _g = torch.exp(-_ax ** 2 / (2 * _sigma ** 2))
            _g = (_g / _g.sum()).view(1, 1, _k, 1, 1)
            _pad = _k // 2
            g4 = grid.unsqueeze(0).unsqueeze(0)
            g4 = torch.nn.functional.conv3d(g4, _g, padding=(_pad, 0, 0))
            g4 = torch.nn.functional.conv3d(g4, _g.view(1, 1, 1, _k, 1), padding=(0, _pad, 0))
            g4 = torch.nn.functional.conv3d(g4, _g.view(1, 1, 1, 1, _k), padding=(0, 0, _pad))
            grid = g4[0, 0]

            # Adaptive iso-level: use percentile of grid values instead of
            # a fraction of grid.max(). The 0.9*grid.max() approach discards
            # ~95% of the surface because it only captures the single densest
            # voxel cluster. A 40th percentile of non-zero grid cells captures
            # the full Gaussian cloud including shoulders, not just peaks.
            grid_np = grid.cpu().numpy()
            nonzero = grid_np[grid_np > 1e-4]
            if len(nonzero) > 0:
                iso = float(np.percentile(nonzero, 40))
            else:
                iso = max(1e-4, density_threshold * 0.5 * float(grid.max()))

            # Apply marching cubes
            if has_torchmcubes:
                verts, faces = torchmcubes.marching_cubes(grid, iso)
                verts = verts.cpu().numpy().astype(np.float32)
                faces = faces.cpu().numpy().astype(np.int64)
            else:
                grid_np = grid.cpu().numpy()
                verts, faces, _, _ = measure.marching_cubes(grid_np, iso)
                verts = verts.astype(np.float32)
                faces = faces.astype(np.int64)
            
            if len(verts) == 0:
                return None
            
            # Convert grid indices back to world coordinates
            verts = (verts / (resolution - 1) * 2 - 1) * pos_scale.numpy() + pos_center.numpy()
            
            # Bake vertex colors from RGB coefficients.
            # CRITICAL: sh_coeffs stores already-converted RGB [N, 3], NOT raw SH
            # coefficients. The SH→RGB transform (0.28209… * sh + 0.5) is applied
            # at PLY load time in LGM/core/gs.py:188. Indexing [:, 0] takes only
            # the R channel, producing 1-D output that trimesh broadcasts to
            # uniform grey [102, 102, 102]. Fix: use all 3 channels directly.
            vertex_colors = None
            if gaussians.sh_coeffs is not None:
                vertex_colors_rgb = gaussians.sh_coeffs  # [N, 3] — already RGB
                # Simple nearest-neighbor assignment (crude but works)
                from scipy.spatial import cKDTree
                tree = cKDTree(gaussians.positions)
                _, idx = tree.query(verts)
                vertex_colors = np.clip(vertex_colors_rgb[idx], 0.0, 1.0)

            # Create mesh
            import trimesh
            tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

            # Set vertex colors BEFORE floater removal so colors survive the
            # vertex-index remapping
            if vertex_colors is not None:
                tm.visual.vertex_colors = (vertex_colors * 255).astype(np.uint8)

            # Remove tiny disconnected floaters (stray Gaussian splats that
            # produce isolated mesh islands). Without this, a single outlier
            # vertex inflates the bounding radius and normal_mesh_to_unit_sphere
            # shrinks the main object to a sub-pixel speck.
            if len(tm.faces) > 0:
                try:
                    from trimesh.graph import connected_components
                    components = connected_components(tm, min_length=0)
                    face_per_comp = np.array(components)
                    comp_sizes = np.bincount(face_per_comp, minlength=face_per_comp.max() + 1)
                    total_faces = len(tm.faces)
                    # Keep components with >= 5% of total faces
                    keep_thresh = max(1, int(total_faces * 0.05))
                    keep_comps = np.where(comp_sizes >= keep_thresh)[0]
                    mask = np.isin(face_per_comp, keep_comps)
                    faces_keep = tm.faces[mask]
                    verts_keep = np.unique(faces_keep)
                    # Build vertex remap
                    remap = np.full(len(tm.vertices), -1)
                    remap[verts_keep] = np.arange(len(verts_keep))
                    faces_remapped = remap[faces_keep]
                    verts_new = tm.vertices[verts_keep]
                    if vertex_colors is not None:
                        vc_new = tm.visual.vertex_colors[verts_keep]
                    else:
                        vc_new = None
                    tm = trimesh.Trimesh(
                        vertices=verts_new, faces=faces_remapped,
                        vertex_colors=vc_new if vc_new is not None else None,
                        process=False,
                    )
                except Exception:
                    pass  # fallback: keep original mesh if floater removal fails
            
            # Extract vertex colors from tm (post-floater-removal if that ran)
            # so the color array matches the vertex count exactly
            final_vertex_colors = None
            if vertex_colors is not None and hasattr(tm.visual, 'vertex_colors'):
                vc_arr = np.array(tm.visual.vertex_colors, dtype=np.float32)
                if vc_arr.ndim == 2 and vc_arr.shape[0] == len(tm.vertices):
                    final_vertex_colors = vc_arr[:, :3] / 255.0
            
            return Mesh3D(
                vertices=np.array(tm.vertices, dtype=np.float32),
                faces=np.array(tm.faces, dtype=np.int64),
                vertex_colors=final_vertex_colors,
            )
        except Exception as e:
            print(f"[LGM] Mesh extraction error: {e}")
            return None

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
