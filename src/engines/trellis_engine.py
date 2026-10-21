"""
TRELLIS Engine — audited wrapper for microsoft/TRELLIS-image-large.

Fixes applied from repo_3_trellis.txt audit:
1. Built-in T4 single-GPU memory hygiene (`staged_offload=True`): keeps unused
   GS/RF decoders on CPU, runs under `@torch.no_grad()`, and offloads flow models
   + DINOv2 to CPU before 256^3 FlexiCubes decoding.
2. Uses `raw.vertices` (fixes `raw.vertex` typo) and preserves 3-channel RGB
   per-vertex colors from `raw.vertex_attrs[:, :3]`.
3. Rotates from TRELLIS's Blender Z-up coordinate system (`Rx(-90)`) to standard
   OpenGL Y-up coordinates.
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Literal

import numpy as np
import torch
from PIL import Image

from .base import AbstractEngine, ReconstructionResult, Mesh3D, GaussianCloud

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("ATTN_BACKEND", "xformers")

OutputFormat = Literal["mesh", "gaussian", "radiance_field"]


class TrellisEngine(AbstractEngine):
    def __init__(
        self,
        device: str = "cuda",
        output_format: OutputFormat = "mesh",
        slat_steps: int = 12,
        sparse_steps: int = 12,
        sparse_cfg_strength: float = 7.5,
        slat_cfg_strength: float = 3.0,
        cfg_strength: float | None = None,
        staged_offload: bool = True,
    ):
        self.device = device
        self.dev = torch.device(device if ":" in device else f"{device}:0")
        self.output_format = output_format
        self.slat_steps = slat_steps
        self.sparse_steps = sparse_steps
        self.sparse_cfg_strength = float(cfg_strength) if cfg_strength is not None else float(sparse_cfg_strength)
        self.slat_cfg_strength = float(slat_cfg_strength)
        self.cfg_strength = self.sparse_cfg_strength
        self.staged_offload = staged_offload
        self.pipeline = None

    def _ensure_trellis_imports(self) -> None:
        try:
            import trellis  # noqa: F401
        except ImportError:
            for c in [
                Path("/kaggle/working/TRELLIS"),
                Path(__file__).resolve().parents[2] / "TRELLIS",
                Path(__file__).resolve().parents[3] / "TRELLIS",
            ]:
                if (c / "trellis").is_dir() and str(c) not in sys.path:
                    sys.path.insert(0, str(c))
                    break

    def attach(self, pipeline, dev=None) -> None:
        self.pipeline = pipeline
        if dev is not None:
            self.dev = torch.device(dev)

    def load_model(self) -> None:
        self._ensure_trellis_imports()
        from trellis.pipelines import TrellisImageTo3DPipeline
        from huggingface_hub import snapshot_download

        print("[TRELLIS] Fetching microsoft/TRELLIS-image-large snapshot …")
        local_dir = snapshot_download("microsoft/TRELLIS-image-large")
        pipe = TrellisImageTo3DPipeline.from_pretrained(local_dir)

        if self.staged_offload:
            for name, mod in pipe.models.items():
                target = "cpu" if name in ("slat_decoder_gs", "slat_decoder_rf") else self.dev
                mod.to(target)
            self.pipeline = pipe
        else:
            moved = pipe.to(self.device)
            self.pipeline = moved if moved is not None else pipe

        torch.cuda.empty_cache()
        print("[TRELLIS] Pipeline loaded.")

    @torch.no_grad()
    def predict(self, image: Image.Image) -> ReconstructionResult:
        if self.pipeline is None:
            self.load_model()

        p = self.pipeline
        rgba = image.convert("RGBA")
        arr = np.array(rgba)
        if arr[..., 3].min() > 250:
            non_white = np.any(arr[..., :3] < 248, axis=-1)
            if non_white.any():
                arr[..., 3] = non_white.astype(np.uint8) * 255
                rgba = Image.fromarray(arr, mode="RGBA")

        t0 = time.perf_counter()
        decode_mode = self.output_format

        if self.staged_offload and self.output_format == "mesh":
            img = p.preprocess_image(rgba)
            cond = p.get_cond([img])
            torch.manual_seed(42)
            coords = p.sample_sparse_structure(
                cond, 1, {"steps": self.sparse_steps, "cfg_strength": self.sparse_cfg_strength}
            )
            slat = p.sample_slat(
                cond, coords, {"steps": self.slat_steps, "cfg_strength": self.slat_cfg_strength}
            )
            del cond, coords
            gc.collect()
            torch.cuda.empty_cache()

            offload_names = (
                "sparse_structure_flow_model",
                "slat_flow_model",
                "image_cond_model",
                "sparse_structure_decoder",
            )
            for n in offload_names:
                if n in p.models:
                    p.models[n].to("cpu")
            torch.cuda.empty_cache()
            gc.collect()

            out_gs_color = None
            try:
                out_mesh = p.models["slat_decoder_mesh"](slat)
                outputs = {"mesh": out_mesh}
                if "slat_decoder_gs" in p.models:
                    try:
                        p.models["slat_decoder_mesh"].to("cpu")
                        torch.cuda.empty_cache()
                        p.models["slat_decoder_gs"].to(self.dev)
                        with torch.cuda.device(self.dev):
                            out_gs_color = p.models["slat_decoder_gs"](slat)
                    except Exception as gs_err:
                        print(f"[TRELLIS] GS color decoder fallback ({gs_err})")
                    finally:
                        p.models["slat_decoder_gs"].to("cpu")
                        p.models["slat_decoder_mesh"].to(self.dev)
                        torch.cuda.empty_cache()
            except torch.OutOfMemoryError:
                print("[TRELLIS] mesh decode OOM -> gaussian fallback")
                torch.cuda.empty_cache()
                gc.collect()
                p.models["slat_decoder_gs"].to(self.dev)
                with torch.cuda.device(self.dev):
                    out_gs = p.models["slat_decoder_gs"](slat)
                p.models["slat_decoder_gs"].to("cpu")
                outputs = {"gaussian": out_gs}
                decode_mode = "gaussian"
            finally:
                for n in offload_names:
                    if n in p.models:
                        p.models[n].to(self.dev)
                torch.cuda.empty_cache()
        else:
            outputs = p.run(
                rgba,
                seed=42,
                formats=[self.output_format],
                sparse_structure_sampler_params={
                    "steps": self.sparse_steps,
                    "cfg_strength": self.sparse_cfg_strength,
                },
                slat_sampler_params={
                    "steps": self.slat_steps,
                    "cfg_strength": self.slat_cfg_strength,
                },
                preprocess_image=True,
            )
            out_gs_color = outputs.get("gaussian")

        latency = time.perf_counter() - t0
        print(f"[TRELLIS] Inference done in {latency:.2f}s (mode={decode_mode})")

        metadata = {
            "engine": "TRELLIS",
            "output_format": decode_mode,
            "latency_s": round(latency, 3),
            "slat_steps": self.slat_steps,
            "sparse_steps": self.sparse_steps,
            "sparse_cfg_strength": self.sparse_cfg_strength,
            "slat_cfg_strength": self.slat_cfg_strength,
        }

        if decode_mode == "mesh":
            raw = outputs["mesh"][0]
            verts_raw = raw.vertices.detach().cpu().numpy().astype(np.float32)
            faces = raw.faces.detach().cpu().numpy().astype(np.int32)

            vcols = None
            attrs = getattr(raw, "vertex_attrs", None)
            if attrs is not None:
                try:
                    vcols = np.clip(
                        attrs.detach().cpu().numpy().astype(np.float32)[:, :3],
                        0.0,
                        1.0,
                    )
                except Exception as ce:
                    print(f"[TRELLIS] vertex color warning: {ce}")

            # Transfer high-fidelity appearance colors from SLatGaussianDecoder in the shared
            # pre-rotated [-0.5, 0.5]^3 coordinate frame via opacity-weighted 3D k-NN.
            if out_gs_color is not None and len(out_gs_color) > 0 and len(verts_raw) > 0:
                try:
                    from scipy.spatial import cKDTree

                    gs0 = out_gs_color[0]
                    gs_xyz = gs0.get_xyz.detach().cpu().numpy().astype(np.float32)
                    gs_opa = gs0.get_opacity.detach().cpu().numpy().astype(np.float32).reshape(-1)
                    sh_c0 = 0.28209479177387814
                    gs_rgb = np.clip(
                        gs0._features_dc.detach().cpu().numpy().astype(np.float32)[:, 0, :3] * sh_c0 + 0.5,
                        0.0,
                        1.0,
                    )
                    mask = gs_opa > 0.10
                    if int(mask.sum()) >= 64:
                        gs_xyz, gs_opa, gs_rgb = gs_xyz[mask], gs_opa[mask], gs_rgb[mask]
                    tree = cKDTree(gs_xyz)
                    k = min(8, len(gs_xyz))
                    dists, idx = tree.query(verts_raw, k=k)
                    if k == 1 or idx.ndim == 1:
                        gs_vcols = gs_rgb[idx]
                    else:
                        w = (gs_opa[idx] ** 2) / (dists ** 2 + 1e-6)
                        w /= w.sum(axis=1, keepdims=True)
                        gs_vcols = (gs_rgb[idx] * w[..., None]).sum(axis=1).astype(np.float32)
                    vcols = np.clip(gs_vcols, 0.0, 1.0).astype(np.float32)
                except Exception as gce:
                    print(f"[TRELLIS] GS color transfer warning: {gce}")

            # Convert Blender Z-up (x, y, z) -> OpenGL Y-up (x, z, -y) via Rx(-90)
            rx_neg90 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float32)
            verts = verts_raw @ rx_neg90.T

            mesh = Mesh3D(vertices=verts, faces=faces, vertex_colors=vcols)
            metadata.update({
                "vertex_count": len(mesh.vertices),
                "face_count": len(mesh.faces),
                "has_vertex_colors": vcols is not None,
            })
            torch.cuda.empty_cache()
            gc.collect()
            return ReconstructionResult(mesh=mesh, metadata=metadata)

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

    def get_engine_info(self) -> Dict[str, Any]:
        return {
            "name": "TRELLIS",
            "paper": "arXiv:2412.01506",
            "hf_repo": "microsoft/TRELLIS-image-large",
            "output_formats": ["mesh (FlexiCubes @ 256^3)", "gaussian (SLAT)"],
        }
