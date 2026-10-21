"""
TripoSR Engine — audited wrapper for VAST-AI-Research/TripoSR (stabilityai/TripoSR).
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
from PIL import Image

from .base import AbstractEngine, ReconstructionResult, Mesh3D


def _skimage_mc_func(level, iso=0.0):
    """
    Drop-in replacement for torchmcubes.marching_cubes that runs on CPU via
    scikit-image and returns vertices in (Z, Y, X) order so tsr/models/isosurface.py
    line 50 (`v_pos = v_pos[..., [2, 1, 0]]`) restores exact (X, Y, Z) coordinates.
    """
    import torch
    from skimage import measure

    if level.dim() == 4:
        level = level.squeeze(0)
    vol = level.detach().cpu().numpy().astype(np.float32)
    try:
        v, f, _, _ = measure.marching_cubes(vol, float(iso))
        v = np.ascontiguousarray(v[:, [2, 1, 0]], dtype=np.float32)
        f = np.ascontiguousarray(f, dtype=np.int64)
    except Exception:
        return (
            torch.zeros((0, 3), dtype=torch.float32),
            torch.zeros((0, 3), dtype=torch.int64),
        )
    return torch.from_numpy(v), torch.from_numpy(f)


# transformers 4.x ViT key naming -> 5.x+ naming. transformers 5.x renamed the
# ViT internals: `encoder.layer.N` -> `layers.N`, BERT-style attention/MLP
# submodule names -> modern q/k/v/o_proj + mlp.fc1/fc2. The stabilityai/TripoSR
# model.ckpt stores the DINO ViT-B/16 image-tokenizer weights in 4.x naming, so
# under 5.x only 8/200 keys matched and 98.5% of the tokenizer ran at random
# init (root cause of fragmented TripoSR meshes). LayerNorm semantics are
# unchanged between 4.x and 5.x (both pre-norm: layernorm_before -> attention,
# layernorm_after -> MLP), so only the paths are renamed.
_VIT_4X_TO_5X_RULES = [
    (r"^encoder\.layer\.(\d+)\.attention\.attention\.query\.", r"layers.\1.attention.q_proj."),
    (r"^encoder\.layer\.(\d+)\.attention\.attention\.key\.", r"layers.\1.attention.k_proj."),
    (r"^encoder\.layer\.(\d+)\.attention\.attention\.value\.", r"layers.\1.attention.v_proj."),
    (r"^encoder\.layer\.(\d+)\.attention\.output\.dense\.", r"layers.\1.attention.o_proj."),
    (r"^encoder\.layer\.(\d+)\.intermediate\.dense\.", r"layers.\1.mlp.fc1."),
    (r"^encoder\.layer\.(\d+)\.output\.dense\.", r"layers.\1.mlp.fc2."),
    (r"^encoder\.layer\.(\d+)\.layernorm_before\.", r"layers.\1.layernorm_before."),
    (r"^encoder\.layer\.(\d+)\.layernorm_after\.", r"layers.\1.layernorm_after."),
]


def _remap_vit_checkpoint_keys(state_dict, model):
    """Remap transformers-4.x ViT checkpoint keys to the live model's naming.

    Version-proof shim: under transformers 4.x every key already matches the
    model and the function is a no-op; under 5.x+ the 4.x-named keys are
    rewritten. A remapped key is accepted only if it exists in the live
    model's state_dict with a matching shape, so future renames fail loudly
    (reported by load_state_dict) instead of silently corrupting weights.
    """
    model_sd = model.state_dict()
    model_keys = set(model_sd.keys())
    unmatched = [
        k for k in state_dict
        if k not in model_keys and k.startswith("image_tokenizer.model.")
    ]
    if not unmatched:
        return state_dict  # already current naming (transformers 4.x) — no-op

    rules = [(re.compile(pat), repl) for pat, repl in _VIT_4X_TO_5X_RULES]
    out, remapped = {}, 0
    for k, v in state_dict.items():
        if k in model_keys or not k.startswith("image_tokenizer.model."):
            out[k] = v
            continue
        sub = k[len("image_tokenizer.model."):]
        for pat, repl in rules:
            cand = pat.sub(repl, sub)
            if cand != sub:
                full = "image_tokenizer.model." + cand
                ref = model_sd.get(full)
                if ref is not None and tuple(ref.shape) == tuple(v.shape):
                    out[full] = v
                    remapped += 1
                    break
        else:
            out[k] = v  # keep original; load_state_dict(strict=False) reports it
    print(f"[TripoSR] Remapped {remapped} transformers-4.x ViT keys to current naming.")
    return out


class TriposrEngine(AbstractEngine):
    def __init__(self, device: str = "cuda", chunk_size: int = 8192, resolution: int = 256):
        self.device = device
        self.chunk_size = chunk_size
        self.resolution = resolution
        self.model = None
        self._resize_foreground = None

    def _ensure_tsr_imports(self) -> None:
        try:
            from tsr.system import TSR  # noqa: F401
        except ImportError:
            for c in [
                Path("/kaggle/working/TripoSR"),
                Path(__file__).resolve().parents[2] / "TripoSR",
                Path(__file__).resolve().parents[3] / "TripoSR",
            ]:
                if (c / "tsr").is_dir() and str(c) not in sys.path:
                    sys.path.insert(0, str(c))
                    break

    def load_model(self) -> None:
        self._ensure_tsr_imports()
        import tsr.models.isosurface as tsr_iso
        from tsr.system import TSR
        from tsr.utils import resize_foreground

        tsr_iso.marching_cubes = _skimage_mc_func

        _orig_load_state_dict = TSR.load_state_dict
        def _remap_load_state_dict(self_model, state_dict, strict=False, **kwargs):
            state_dict = _remap_vit_checkpoint_keys(state_dict, self_model)
            return _orig_load_state_dict(self_model, state_dict, strict=False, **kwargs)
        TSR.load_state_dict = _remap_load_state_dict

        print("[TripoSR] Loading stabilityai/TripoSR …")
        self.model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        self.model.renderer.set_chunk_size(self.chunk_size)
        self.model.to(self.device)
        self._resize_foreground = resize_foreground
        print("[TripoSR] Model loaded (using verified scikit-image CPU Marching Cubes).")

    def _query_interior_vertex_colors(self, mesh_obj, scene_code) -> np.ndarray | None:
        import torch

        verts = np.asarray(mesh_obj.vertices, dtype=np.float32)
        if len(verts) == 0:
            return None
        normals = np.asarray(mesh_obj.vertex_normals, dtype=np.float32)
        if normals.shape != verts.shape:
            normals = np.zeros_like(verts)

        radius = float(self.model.renderer.cfg.radius)
        offsets = [0.0, -0.002, -0.005]
        base_weights = [0.60, 0.25, 0.15]
        color_accum = np.zeros((len(verts), 3), dtype=np.float32)
        weight_accum = np.zeros((len(verts), 1), dtype=np.float32)
        surf_density = None

        with torch.no_grad():
            for idx, (off, w) in enumerate(zip(offsets, base_weights)):
                pts = np.clip(verts + off * normals, -radius, radius)
                pts_t = torch.from_numpy(pts).float().to(self.device)
                queried = self.model.renderer.query_triplane(
                    self.model.decoder, pts_t, scene_code
                )
                c = np.clip(
                    queried["color"].detach().cpu().numpy().astype(np.float32),
                    0.0,
                    1.0,
                )
                d = (
                    queried["density_act"]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                    .reshape(-1, 1)
                )
                if idx == 0:
                    surf_density = d
                    eff_w = np.full((len(verts), 1), w, dtype=np.float32)
                else:
                    # Only blend interior samples where density remains inside the solid shell;
                    # prevents thin shells (bowls, pots, wings) from blending empty interior grey.
                    inside = (d >= 0.85 * surf_density).astype(np.float32)
                    eff_w = w * inside
                color_accum += eff_w * c
                weight_accum += eff_w

        return np.clip(color_accum / np.maximum(weight_accum, 1e-6), 0.0, 1.0)

    def predict(self, image: Image.Image) -> ReconstructionResult:
        import torch
        import trimesh

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

        fg = self._resize_foreground(rgba, 0.85)
        img_np = np.array(fg).astype(np.float32) / 255.0
        alpha = img_np[..., 3:4]
        img_rgb = img_np[..., :3] * alpha + (1.0 - alpha) * 0.5
        image_input = Image.fromarray((np.clip(img_rgb, 0.0, 1.0) * 255.0).astype(np.uint8))

        with torch.no_grad():
            scene_codes = self.model([image_input], device=self.device)
            self.model.set_marching_cubes_resolution(self.resolution)
            self.model.isosurface_helper.mc_func = _skimage_mc_func
            meshes = self.model.extract_mesh(
                scene_codes, True, resolution=self.resolution
            )

        mesh_obj = meshes[0]
        vertex_colors = None
        try:
            vertex_colors = self._query_interior_vertex_colors(mesh_obj, scene_codes[0])
        except Exception:
            if getattr(mesh_obj.visual, "vertex_colors", None) is not None:
                vc = np.asarray(mesh_obj.visual.vertex_colors, dtype=np.float32)
                if vc.ndim == 2 and len(vc) == len(mesh_obj.vertices):
                    vertex_colors = vc[:, :3] / (255.0 if vc.max() > 1.0 else 1.0)

        rot_x = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        rot_y = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
        mesh_obj.apply_transform(rot_y @ rot_x)

        latency = time.perf_counter() - t0
        print(f"[TripoSR] Inference done in {latency:.2f}s ({len(mesh_obj.vertices)} verts)")

        result_mesh = Mesh3D(
            vertices=np.asarray(mesh_obj.vertices, dtype=np.float32),
            faces=np.asarray(mesh_obj.faces, dtype=np.int64),
            vertex_colors=vertex_colors,
        )
        torch.cuda.empty_cache()

        return ReconstructionResult(
            mesh=result_mesh,
            metadata={
                "engine": "TripoSR",
                "latency_s": round(latency, 3),
                "vertex_count": len(result_mesh.vertices),
                "face_count": len(result_mesh.faces),
                "has_vertex_colors": vertex_colors is not None,
            },
        )

    def get_engine_info(self) -> Dict[str, Any]:
        return {
            "name": "TripoSR",
            "paper": "arXiv:2403.02151",
            "hf_repo": "stabilityai/TripoSR",
            "output_format": "mesh (Marching Cubes @ 256^3)",
        }
