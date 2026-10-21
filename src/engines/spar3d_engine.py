"""
SPAR3D Engine — audited wrapper for Stability-AI/stable-point-aware-3d (arXiv:2501.04689).
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path
from typing import Dict, Any

import numpy as np
import packaging
import torch
from PIL import Image
from scipy.spatial import cKDTree

from .base import AbstractEngine, ReconstructionResult, Mesh3D


def _install_spar3d_compat_shims() -> None:
    if "pkg_resources" not in sys.modules:
        pkgr = types.ModuleType("pkg_resources")
        pkgr.packaging = packaging
        sys.modules["pkg_resources"] = pkgr
    elif not hasattr(sys.modules["pkg_resources"], "packaging"):
        sys.modules["pkg_resources"].packaging = packaging

    try:
        import transformers.pytorch_utils as _tpu
        if not hasattr(_tpu, "find_pruneable_heads_and_indices"):
            def _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
                heads = set(heads) - already_pruned_heads
                mask = torch.ones(n_heads, head_size)
                for head in heads:
                    head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                    mask[head] = 0
                mask = mask.view(-1).contiguous().eq(1)
                index = torch.arange(len(mask))[mask].long()
                return heads, index
            _tpu.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices

        if not hasattr(_tpu, "prune_linear_layer"):
            def _prune_linear_layer(layer, index, dim=0):
                index = index.to(layer.weight.device)
                W = layer.weight.index_select(dim, index).clone().detach()
                b = (layer.bias.clone().detach() if dim == 1 else layer.bias[index].clone().detach()) if layer.bias is not None else None
                new_size = list(layer.weight.size())
                new_size[dim] = len(index)
                new_layer = torch.nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
                new_layer.weight.requires_grad = False
                new_layer.weight.copy_(W.contiguous())
                new_layer.weight.requires_grad = True
                if layer.bias is not None:
                    new_layer.bias.requires_grad = False
                    new_layer.bias.copy_(b.contiguous())
                    new_layer.bias.requires_grad = True
                return new_layer
            _tpu.prune_linear_layer = _prune_linear_layer

        from transformers.modeling_utils import PreTrainedModel
        if not hasattr(PreTrainedModel, "get_head_mask"):
            def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
                if head_mask is not None:
                    head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
                    if is_attention_chunked is True:
                        head_mask = head_mask.unsqueeze(-1)
                else:
                    head_mask = [None] * num_hidden_layers
                return head_mask
            PreTrainedModel.get_head_mask = _get_head_mask
    except Exception:
        pass

    if "transparent_background" not in sys.modules:
        try:
            import transparent_background  # noqa: F401
        except ImportError:
            tb_mod = types.ModuleType("transparent_background")
            class _DummyRemover:
                def __init__(self, *a, **kw): pass
                def process(self, img, *a, **kw): return img.convert("RGBA")
            tb_mod.Remover = _DummyRemover
            sys.modules["transparent_background"] = tb_mod


class Spar3dEngine(AbstractEngine):
    def __init__(
        self,
        device: str = "cuda",
        texture_resolution: int = 1024,
        low_vram_mode: bool = False,
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
        _install_spar3d_compat_shims()
        repo_dir = self._locate_repo()
        for p in [repo_dir, Path("/kaggle/working/CLIP"), Path("/kaggle/working/AlphaCLIP")]:
            if p.is_dir() and str(p) not in sys.path:
                sys.path.insert(0, str(p))

        from spar3d.system import SPAR3D
        from spar3d.utils import foreground_crop

        print(f"[SPAR3D] Loading model (low_vram_mode={self.low_vram_mode}) …")
        try:
            self.model = SPAR3D.from_pretrained(
                "stabilityai/stable-point-aware-3d",
                config_name="config.yaml",
                weight_name="model.safetensors",
                low_vram_mode=self.low_vram_mode,
            )
        except Exception as e:
            print(f"[SPAR3D] Primary repo gated ({e}); falling back to public mirror zimengxiong/Modelr-SPAR3D …")
            self.model = SPAR3D.from_pretrained(
                "zimengxiong/Modelr-SPAR3D",
                config_name="config.yaml",
                weight_name="model.safetensors",
                low_vram_mode=self.low_vram_mode,
            )
        self.model.to(self.device).eval()
        self._foreground_crop = foreground_crop
        print("[SPAR3D] Model loaded.")

    @staticmethod
    def _extract_spar3d_vertex_colors(tm, verts: np.ndarray) -> np.ndarray | None:
        """
        Extract per-vertex RGB colors from SPAR3D's baked PBRMaterial.baseColorTexture
        and repair any unbaked black UV-seam texels via 3D k-NN interpolation.
        """
        try:
            vis = getattr(tm, "visual", None)
            if vis is None:
                return None
            uv = getattr(vis, "uv", None)
            mat = getattr(vis, "material", None)
            tex_img = None
            if mat is not None:
                tex_img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)

            vcols = None
            if uv is not None and tex_img is not None and len(uv) == len(verts):
                rgb = np.asarray(tex_img.convert("RGB"), dtype=np.float32) / 255.0
                h, w = rgb.shape[:2]
                u = np.clip(np.nan_to_num(np.asarray(uv[:, 0], dtype=np.float64), nan=0.5), 0.0, 1.0)
                v = np.clip(np.nan_to_num(np.asarray(uv[:, 1], dtype=np.float64), nan=0.5), 0.0, 1.0)
                # SPAR3D's baker_kernel.cu writes UV v = 1.0 - row/height (OpenGL bottom-left origin).
                # Bilinearly sample at x = u * (w - 1), y = (1.0 - v) * (h - 1).
                fx = u * (w - 1)
                fy = (1.0 - v) * (h - 1)
                x0 = np.clip(np.floor(fx).astype(int), 0, w - 1)
                x1 = np.clip(x0 + 1, 0, w - 1)
                y0 = np.clip(np.floor(fy).astype(int), 0, h - 1)
                y1 = np.clip(y0 + 1, 0, h - 1)
                wx = (fx - x0)[..., None].astype(np.float32)
                wy = (fy - y0)[..., None].astype(np.float32)
                c00 = rgb[y0, x0]
                c10 = rgb[y0, x1]
                c01 = rgb[y1, x0]
                c11 = rgb[y1, x1]
                vcols = np.clip(
                    (1.0 - wy) * ((1.0 - wx) * c00 + wx * c10)
                    + wy * ((1.0 - wx) * c01 + wx * c11),
                    0.0,
                    1.0,
                ).astype(np.float32)
            else:
                vc_vis = vis.to_color()
                if getattr(vc_vis, "vertex_colors", None) is not None:
                    vc = np.asarray(vc_vis.vertex_colors, dtype=np.float32)
                    if len(vc) == len(verts):
                        vcols = vc[:, :3] / (255.0 if vc.max() > 1.05 else 1.0)

            if vcols is not None and len(vcols) == len(verts):
                # Repair unbaked black UV chart padding texels (brightness < 0.04) via 3D k-NN
                valid = vcols.max(axis=-1) >= 0.04
                if valid.any() and (~valid).any():
                    tree = cKDTree(verts[valid])
                    _, idx = tree.query(verts[~valid], k=min(4, int(valid.sum())))
                    if idx.ndim == 1:
                        vcols[~valid] = vcols[valid][idx]
                    else:
                        vcols[~valid] = vcols[valid][idx].mean(axis=1)
            return vcols
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
        # Undo spar3d/system.py:917's `tmesh.invert()` so triangle winding & normals face OUTWARD!
        try:
            tm.invert()
        except Exception:
            pass

        verts = np.asarray(tm.vertices, dtype=np.float32)
        faces = np.asarray(tm.faces, dtype=np.int64)
        vcols = self._extract_spar3d_vertex_colors(tm, verts)

        # Note: spar3d/system.py:705-707 & 915 ALREADY applies output_rotation = Ry(90) @ Rx(-90)
        # to tmesh inside run_image(), placing vertices in OpenGL Y-up (+Y up, +Z front).
        # Do NOT apply a second Rx(-90) rotation here.

        print(f"[SPAR3D] Inference done in {latency:.2f}s ({len(verts)} verts)")

        mesh = Mesh3D(
            vertices=verts,
            faces=faces,
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
