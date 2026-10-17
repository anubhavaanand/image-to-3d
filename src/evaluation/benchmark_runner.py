"""
Unified Benchmark Runner — ground-truth-referenced evaluation on GSO.

Protocol highlights:
1. Uniform 10,000-point surface sampling, centroid centering, and unit-sphere
   normalization (with 99.5th-percentile radius protection against stray floaters).
2. Canonical 3D orientation search (36 candidate poses: 3 principal axis-up
   orientations x 12 yaw angles) + 3-iteration Kabsch ICP refinement using
   symmetric mean Chamfer Distance (mean(d1) + mean(d2)).
3. Applies the optimal 3D rigid transform directly to `pred_mesh.vertices`
   while preserving `pred_mesh.vertex_colors` so novel-view renders
   (`*_novel.png`) and PSNR/LPIPS are evaluated in the exact same canonical
   pose as `gt_novel.png`.
4. Matched 2x-supersampled white-background rendering with even ambient fill
   (`ambient_light=[0.65, 0.65, 0.65]`) + directional key light (`intensity=1.8`).
5. Caps exported `*_mesh.obj` QA files to the first 10 objects (`save_mesh_limit=10`)
   so 100-object archives stay compact (<300 MB).
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import time
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image

from ..engines.base import AbstractEngine, Mesh3D
from ..evaluation.chamfer_distance import (
    compute_chamfer_distance,
    compute_chamfer_distance_symmetric,
)
from ..evaluation.f_score import compute_f_score
from ..evaluation.psnr_lpips import compute_psnr, compute_lpips
from ..preprocessing.background_removal import BackgroundRemover


# ---------------------------------------------------------------------------
# GSO dataset loader
# ---------------------------------------------------------------------------

def load_gso_samples(
    dataset_root: str,
    num_samples: int = 100,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Load GSO samples with 10K unit-sphere-normalized ground-truth surface points."""
    import trimesh

    root = Path(dataset_root)
    object_dirs = sorted([d for d in root.iterdir() if d.is_dir()])

    rng = np.random.default_rng(seed)
    if len(object_dirs) > num_samples:
        indices = rng.choice(len(object_dirs), size=num_samples, replace=False)
        object_dirs = [object_dirs[i] for i in sorted(indices)]

    samples = []
    for obj_dir in object_dirs:
        img_dir = obj_dir / "images"
        front_image_path = img_dir / "000.png"
        if not front_image_path.exists():
            candidates = sorted(img_dir.glob("*.png"))
            if not candidates:
                continue
            front_image_path = candidates[0]

        mesh_path = obj_dir / "meshes" / "model.obj"
        if not mesh_path.exists():
            mesh_candidates = (
                list((obj_dir / "meshes").glob("*.obj"))
                + list((obj_dir / "meshes").glob("*.ply"))
            )
            if not mesh_candidates:
                continue
            mesh_path = mesh_candidates[0]

        try:
            gt_mesh = trimesh.load(str(mesh_path), force="mesh")
            gt_points, _ = trimesh.sample.sample_surface(gt_mesh, 10_000)
            gt_points = np.asarray(gt_points, dtype=np.float32)

            center = gt_points.mean(axis=0)
            scale = np.linalg.norm(gt_points - center, axis=1).max()
            gt_points = (gt_points - center) / (scale + 1e-8)
        except Exception:
            continue

        samples.append({
            "id": obj_dir.name,
            "image_path": str(front_image_path),
            "mesh_path": str(mesh_path),
            "gt_points": gt_points,
        })

    print(f"[BenchmarkRunner] Loaded {len(samples)} GSO samples.")
    return samples


# ---------------------------------------------------------------------------
# Normalization & point-cloud helpers
# ---------------------------------------------------------------------------

def normalize_mesh_to_unit_sphere(mesh: Mesh3D, use_percentile: bool = True) -> Mesh3D:
    """Center a Mesh3D at its vertex centroid and scale to the unit sphere."""
    if mesh is None or len(mesh.vertices) == 0:
        return mesh
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    c = verts.mean(axis=0)
    radii = np.linalg.norm(verts - c, axis=1)
    s = float(np.percentile(radii, 99.5)) if use_percentile else float(radii.max())
    if s < 1e-8:
        s = float(radii.max())
    verts_norm = (verts - c) / (s + 1e-8)
    vcols = (
        np.asarray(mesh.vertex_colors, dtype=np.float32).copy()
        if getattr(mesh, "vertex_colors", None) is not None
        else None
    )
    return Mesh3D(
        vertices=verts_norm,
        faces=np.asarray(mesh.faces, dtype=np.int64),
        vertex_colors=vcols,
        uv_coords=getattr(mesh, "uv_coords", None),
        texture_map=getattr(mesh, "texture_map", None),
    )


def mesh_to_pointcloud(mesh: Optional[Mesh3D], n_points: int = 10_000) -> Optional[np.ndarray]:
    """Sample n_points from a Mesh3D surface and normalize to the unit sphere."""
    import trimesh

    if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return None
    try:
        norm_mesh = normalize_mesh_to_unit_sphere(mesh, use_percentile=True)
        tm = trimesh.Trimesh(
            vertices=norm_mesh.vertices,
            faces=norm_mesh.faces,
            process=False,
        )
        pts, _ = trimesh.sample.sample_surface(tm, n_points)
        pts = np.asarray(pts, dtype=np.float32)
        center = pts.mean(axis=0)
        scale = np.percentile(np.linalg.norm(pts - center, axis=1), 99.5)
        return (pts - center) / (scale + 1e-8)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Canonical 3D rotation search + Kabsch ICP alignment
# ---------------------------------------------------------------------------

def _candidate_rotations(n_yaw: int = 12) -> List[np.ndarray]:
    """Generate 36 candidate 3D rotations (3 principal axis-up orientations x 12 yaw angles)."""
    base_orientations = [
        np.eye(3, dtype=np.float32),
        np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32),
        np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32),
    ]
    rots = []
    for base in base_orientations:
        for i in range(n_yaw):
            angle = 2.0 * np.pi * i / n_yaw
            ca, sa = np.cos(angle), np.sin(angle)
            rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=np.float32)
            rots.append(rz @ base)
    return rots


def align_mesh_to_gt(
    pred_mesh: Mesh3D,
    gt_points: np.ndarray,
    n_rotations: int = 12,
    icp_iters: int = 3,
):
    """
    Align predicted Mesh3D to normalized GT point cloud using multi-axis yaw search
    + Kabsch ICP refinement under symmetric mean Chamfer Distance.

    Returns
    -------
    (aligned_mesh: Mesh3D, best_cd: float, best_pts: np.ndarray)
    """
    try:
        import trimesh
        from scipy.spatial import cKDTree

        norm_mesh = normalize_mesh_to_unit_sphere(pred_mesh, use_percentile=True)
        tm = trimesh.Trimesh(
            vertices=norm_mesh.vertices,
            faces=norm_mesh.faces,
            process=False,
        )
        pred_pts, _ = trimesh.sample.sample_surface(tm, 10_000)
        pred_pts = np.asarray(pred_pts, dtype=np.float32)

        p_center = pred_pts.mean(axis=0)
        p_scale = float(np.percentile(np.linalg.norm(pred_pts - p_center, axis=1), 99.5))
        if p_scale < 1e-8:
            p_scale = float(np.linalg.norm(pred_pts - p_center, axis=1).max())
        pred_pts_norm = (pred_pts - p_center) / (p_scale + 1e-8)

        tree_gt = cKDTree(gt_points)
        candidates = _candidate_rotations(n_yaw=n_rotations)

        best_cd = float("inf")
        best_R_total = np.eye(3, dtype=np.float32)
        best_t_total = np.zeros(3, dtype=np.float32)
        best_pts = pred_pts_norm

        for R_init in candidates:
            R_curr = R_init.copy()
            t_curr = np.zeros(3, dtype=np.float32)
            pts_curr = pred_pts_norm @ R_init.T

            for _ in range(icp_iters):
                _, idx = tree_gt.query(pts_curr)
                matched_gt = gt_points[idx]

                c_pred = pts_curr.mean(axis=0)
                c_gt = matched_gt.mean(axis=0)
                H = (pts_curr - c_pred).T @ (matched_gt - c_gt)
                U, _, Vt = np.linalg.svd(H)
                R_step = (Vt.T @ U.T).astype(np.float32)
                if np.linalg.det(R_step) < 0:
                    Vt[-1, :] *= -1
                    R_step = (Vt.T @ U.T).astype(np.float32)

                t_step = c_gt - (c_pred @ R_step.T)
                pts_curr = pts_curr @ R_step.T + t_step
                R_curr = R_step @ R_curr
                t_curr = t_curr @ R_step.T + t_step

            d1, _ = tree_gt.query(pts_curr)
            tree_pred = cKDTree(pts_curr)
            d2, _ = tree_pred.query(gt_points)
            cd = float(np.mean(d1) + np.mean(d2))

            if cd < best_cd:
                best_cd = cd
                best_R_total = R_curr
                best_t_total = t_curr
                best_pts = pts_curr

        verts_norm = (np.asarray(norm_mesh.vertices, dtype=np.float32) - p_center) / (p_scale + 1e-8)
        aligned_verts = verts_norm @ best_R_total.T + best_t_total

        vcols = (
            np.asarray(pred_mesh.vertex_colors, dtype=np.float32).copy()
            if getattr(pred_mesh, "vertex_colors", None) is not None
            else None
        )

        aligned_mesh = Mesh3D(
            vertices=aligned_verts.astype(np.float32),
            faces=np.asarray(pred_mesh.faces, dtype=np.int64),
            vertex_colors=vcols,
            uv_coords=getattr(pred_mesh, "uv_coords", None),
            texture_map=getattr(pred_mesh, "texture_map", None),
        )
        return aligned_mesh, best_cd, best_pts
    except Exception:
        return None, float("inf"), None


def align_mesh_to_gt_simple(pred_mesh: Mesh3D, gt_points: np.ndarray, n_rotations: int = 12):
    """Backwards-compatible alias for align_mesh_to_gt."""
    return align_mesh_to_gt(pred_mesh, gt_points, n_rotations=n_rotations, icp_iters=1)


# ---------------------------------------------------------------------------
# UV texture -> vertex color helpers for GSO ground-truth meshes
# ---------------------------------------------------------------------------

def bake_texture_to_vertices(tm):
    """Sample a trimesh object's UV texture map into per-vertex ColorVisuals."""
    try:
        import trimesh
        from PIL import Image as _PILImage

        uv = getattr(tm.visual, "uv", None)
        material = getattr(tm.visual, "material", None)
        img = getattr(material, "image", None) if material is not None else None
        if uv is None or img is None:
            return tm.visual
        if not isinstance(img, _PILImage.Image):
            img = _PILImage.fromarray(np.array(img))
        img = np.array(img.convert("RGB"))
        h, w = img.shape[:2]
        u = np.mod(np.asarray(uv[:, 0], dtype=np.float64), 1.0)
        v = 1.0 - np.mod(np.asarray(uv[:, 1], dtype=np.float64), 1.0)
        x = np.clip((u * (w - 1)).round().astype(int), 0, w - 1)
        y = np.clip((v * (h - 1)).round().astype(int), 0, h - 1)
        vcol = img[y, x].astype(np.float32) / 255.0
        return trimesh.visual.ColorVisuals(tm, vertex_colors=vcol)
    except Exception:
        return tm.visual


def vertex_colors_rgb(tm) -> Optional[np.ndarray]:
    """Extract per-vertex RGB float32 colors in [0, 1] from a trimesh.Trimesh."""
    try:
        if getattr(tm.visual, "kind", None) != "vertex":
            return None
        vc = np.asarray(tm.visual.vertex_colors)
        if vc.ndim != 2 or vc.shape[0] != len(tm.vertices):
            return None
        return vc[:, :3].astype(np.float32) / 255.0
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Matched novel-view renderer (2x supersampled, balanced ambient + key light)
# ---------------------------------------------------------------------------

def render_novel_view_same_bg(
    mesh: Mesh3D,
    azimuth_deg: float = 45.0,
    resolution: int = 256,
    background_color: tuple = (1.0, 1.0, 1.0),
) -> Optional[np.ndarray]:
    """Render a Mesh3D at `azimuth_deg` on a solid background with balanced lighting."""
    try:
        import trimesh
        import pyrender

        vcols_uint8 = None
        if getattr(mesh, "vertex_colors", None) is not None:
            vc = np.asarray(mesh.vertex_colors, dtype=np.float32)
            if vc.max() <= 1.05:
                vc = np.clip(vc, 0.0, 1.0) * 255.0
            vcols_uint8 = np.clip(vc, 0, 255).astype(np.uint8)

        tm = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            vertex_colors=vcols_uint8,
            process=False,
        )
        scene = pyrender.Scene(
            bg_color=np.array([*background_color, 1.0], dtype=np.float32),
            ambient_light=np.array([0.65, 0.65, 0.65], dtype=np.float32),
        )
        scene.add(pyrender.Mesh.from_trimesh(tm))

        az = np.radians(azimuth_deg)
        camera_pose = np.array([
            [np.cos(az),  0.0, np.sin(az), 2.0 * np.sin(az)],
            [0.0,         1.0, 0.0,        0.0             ],
            [-np.sin(az), 0.0, np.cos(az), 2.0 * np.cos(az)],
            [0.0,         0.0, 0.0,        1.0             ],
        ], dtype=np.float32)
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        scene.add(camera, pose=camera_pose)
        scene.add(
            pyrender.DirectionalLight(color=np.array([1.0, 1.0, 1.0]), intensity=1.8),
            pose=camera_pose,
        )

        renderer = pyrender.OffscreenRenderer(resolution * 2, resolution * 2)
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
        renderer.delete()

        color = np.array(
            Image.fromarray(color).resize((resolution, resolution), Image.LANCZOS)
        )
        return color
    except Exception:
        return None


def render_novel_view(mesh: Mesh3D, azimuth_deg: float = 45.0, resolution: int = 256):
    """Alias for render_novel_view_same_bg with white background."""
    return render_novel_view_same_bg(
        mesh, azimuth_deg=azimuth_deg, resolution=resolution, background_color=(1.0, 1.0, 1.0)
    )


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Uniform evaluation runner across all feed-forward 3D engines."""

    def __init__(
        self,
        dataset_path: str,
        engines: List[AbstractEngine],
        num_samples: int = 100,
        remove_background: bool = True,
        compute_render_metrics: bool = True,
        output_dir: str = "./outputs/benchmark",
        save_mesh_limit: int = 10,
    ):
        self.dataset_path = dataset_path
        self.engines = engines
        self.num_samples = num_samples
        self.remove_background = remove_background
        self.compute_render_metrics = compute_render_metrics
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_mesh_limit = save_mesh_limit

        self._bg_remover = BackgroundRemover() if remove_background else None
        self.results: List[Dict[str, Any]] = []

    def load_dataset(self) -> List[Dict[str, Any]]:
        return load_gso_samples(self.dataset_path, num_samples=self.num_samples)

    def _flush_partial(self, engine_name: str) -> None:
        try:
            rows = [r for r in self.results if r.get("engine") == engine_name]
            if not rows:
                return
            tmp = self.output_dir / f".partial_{engine_name}.json"
            tmp.write_text(json.dumps(rows, indent=1, default=str))
            os.replace(tmp, self.output_dir / f"partial_{engine_name}.json")
            print(f"  [checkpoint] {engine_name}: {len(rows)} objects saved")
        except Exception as e:
            print(f"  [checkpoint] warning: {e}")

    def _render_gt_novel(self, sample: Dict[str, Any]) -> Optional[np.ndarray]:
        """Render the ground-truth mesh under the exact same camera and lighting."""
        import trimesh

        gt_mesh_path = Path(sample["image_path"]).parent.parent / "meshes" / "model.obj"
        if not gt_mesh_path.exists() and "mesh_path" in sample:
            gt_mesh_path = Path(sample["mesh_path"])
        if gt_mesh_path.exists():
            try:
                gt_tm = trimesh.load(str(gt_mesh_path), force="mesh")
                gt_tm.visual = bake_texture_to_vertices(gt_tm)
                pts = np.asarray(gt_tm.vertices, dtype=np.float32)
                c = pts.mean(axis=0)
                s = np.linalg.norm(pts - c, axis=1).max()
                gt_mesh_obj = Mesh3D(
                    vertices=(pts - c) / (s + 1e-8),
                    faces=np.asarray(gt_tm.faces, dtype=np.int64),
                    vertex_colors=vertex_colors_rgb(gt_tm),
                )
                return render_novel_view_same_bg(
                    gt_mesh_obj, azimuth_deg=45.0, background_color=(1.0, 1.0, 1.0)
                )
            except Exception:
                pass
        gt_render_path = Path(sample["image_path"]).parent / "001.png"
        if gt_render_path.exists():
            return np.array(Image.open(gt_render_path).convert("RGB").resize((256, 256)))
        return None

    def run(self) -> None:
        samples = self.load_dataset()

        for engine in self.engines:
            info = engine.get_engine_info()
            engine_name = info["name"]
            print(f"\n{'=' * 60}")
            print(f"  Evaluating: {engine_name}")
            print(f"{'=' * 60}")

            for idx, sample in enumerate(samples):
                sample_id = sample["id"]
                print(f"  [{engine_name}] ({idx + 1}/{len(samples)}) {sample_id} ... ", end="", flush=True)

                try:
                    image = Image.open(sample["image_path"]).convert("RGBA")
                    if self._bg_remover is not None:
                        image = self._bg_remover.remove(image)

                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                    t0 = time.perf_counter()
                    result = engine.predict(image)
                    latency = time.perf_counter() - t0
                    vram_peak_gb = (
                        round(torch.cuda.max_memory_allocated() / 1e9, 2)
                        if torch.cuda.is_available()
                        else None
                    )

                    geo_metrics: Dict[str, Any] = {}
                    aligned_for_render = False
                    pred_pts = mesh_to_pointcloud(result.mesh)
                    if pred_pts is not None and result.mesh is not None:
                        gt_pts = sample["gt_points"]
                        aligned_mesh, best_cd, aligned_pts_norm = align_mesh_to_gt(
                            result.mesh, gt_pts, n_rotations=12, icp_iters=3
                        )
                        if (
                            aligned_mesh is not None
                            and best_cd != float("inf")
                            and aligned_pts_norm is not None
                        ):
                            geo_metrics["chamfer_distance"] = float(best_cd)
                            fs = compute_f_score(
                                aligned_pts_norm, gt_pts, thresholds=[0.1, 0.2, 0.5]
                            )
                            geo_metrics.update(fs)
                            result.mesh = aligned_mesh
                            aligned_for_render = True
                        else:
                            geo_metrics["chamfer_distance"] = float(
                                compute_chamfer_distance_symmetric(pred_pts, gt_pts)
                            )
                            fs = compute_f_score(pred_pts, gt_pts, thresholds=[0.1, 0.2, 0.5])
                            geo_metrics.update(fs)
                    else:
                        geo_metrics = {
                            "chamfer_distance": float("nan"),
                            "f_score_0.1": float("nan"),
                            "f_score_0.2": float("nan"),
                            "f_score_0.5": float("nan"),
                        }

                    render_metrics: Dict[str, Any] = {}
                    pred_render_to_save = None
                    gt_render_to_save = None
                    if self.compute_render_metrics and result.mesh is not None:
                        try:
                            pred_norm = (
                                result.mesh
                                if aligned_for_render
                                else normalize_mesh_to_unit_sphere(result.mesh)
                            )
                            pred_render = render_novel_view_same_bg(
                                pred_norm, azimuth_deg=45.0, background_color=(1.0, 1.0, 1.0)
                            )
                            if idx < self.save_mesh_limit:
                                try:
                                    import trimesh as _tm

                                    qm = self.output_dir / "qualitative" / sample_id
                                    qm.mkdir(parents=True, exist_ok=True)
                                    vc_u8 = (
                                        (np.clip(pred_norm.vertex_colors, 0.0, 1.0) * 255).astype(
                                            np.uint8
                                        )
                                        if pred_norm.vertex_colors is not None
                                        else None
                                    )
                                    _m = _tm.Trimesh(
                                        vertices=pred_norm.vertices,
                                        faces=pred_norm.faces,
                                        vertex_colors=vc_u8,
                                        process=False,
                                    )
                                    _m.export(str(qm / f"{engine_name}_mesh.obj"))
                                except Exception:
                                    pass

                            if pred_render is not None:
                                pred_render_to_save = pred_render
                                gt_render = self._render_gt_novel(sample)
                                if gt_render is not None:
                                    gt_render_to_save = gt_render
                                    render_metrics["psnr"] = float(
                                        compute_psnr(pred_render, gt_render)
                                    )

                                    def to_lpips_tensor(img_np):
                                        img_np = np.ascontiguousarray(img_np)
                                        t = torch.from_numpy(img_np).float() / 127.5 - 1.0
                                        return t.permute(2, 0, 1).unsqueeze(0)

                                    render_metrics["lpips"] = float(
                                        compute_lpips(
                                            to_lpips_tensor(pred_render),
                                            to_lpips_tensor(gt_render),
                                        )
                                    )
                        except Exception as rm_err:
                            print(f"  [render] warning: {rm_err}")

                    try:
                        qual_dir = self.output_dir / "qualitative" / sample_id
                        qual_dir.mkdir(parents=True, exist_ok=True)
                        input_src = Path(sample["image_path"])
                        if not (qual_dir / "input.png").exists() and input_src.exists():
                            shutil.copy(input_src, qual_dir / "input.png")
                        if gt_render_to_save is not None:
                            Image.fromarray(gt_render_to_save).save(qual_dir / "gt_novel.png")
                        elif not (qual_dir / "gt_novel.png").exists():
                            gt_novel_src = input_src.parent / "001.png"
                            if gt_novel_src.exists():
                                shutil.copy(gt_novel_src, qual_dir / "gt_novel.png")
                        if pred_render_to_save is not None:
                            Image.fromarray(pred_render_to_save).save(
                                qual_dir / f"{engine_name}_novel.png"
                            )
                    except Exception as qual_exc:
                        print(f"  [qualitative] warning: {qual_exc}")

                    record = {
                        "engine": engine_name,
                        "sample_id": sample_id,
                        "latency_s": round(latency, 4),
                        "vram_peak_gb": vram_peak_gb,
                        **geo_metrics,
                        **render_metrics,
                        "metadata": result.metadata or {},
                    }
                    self.results.append(record)
                    vram_str = f"  VRAM={vram_peak_gb:.2f}GB" if vram_peak_gb else ""
                    psnr_val = render_metrics.get("psnr", float("nan"))
                    psnr_str = f"  PSNR={psnr_val:.2f}dB" if not np.isnan(psnr_val) else ""
                    print(
                        f"CD={geo_metrics.get('chamfer_distance', float('nan')):.4f}  "
                        f"F@0.2={geo_metrics.get('f_score_0.2', float('nan')):.3f}"
                        f"{psnr_str}  t={latency:.2f}s{vram_str}"
                    )

                except Exception as exc:
                    print(f"FAILED: {exc}")
                    traceback.print_exc()
                    self.results.append({
                        "engine": engine_name,
                        "sample_id": sample_id,
                        "error": str(exc),
                    })

                done = sum(1 for r in self.results if r.get("engine") == engine_name)
                if done % 10 == 0:
                    self._flush_partial(engine_name)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_results_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)

    def get_summary_df(self) -> pd.DataFrame:
        df = self.get_results_df()
        if df.empty:
            return pd.DataFrame()
        numeric_cols = [
            "latency_s",
            "vram_peak_gb",
            "chamfer_distance",
            "f_score_0.1",
            "f_score_0.2",
            "f_score_0.5",
            "psnr",
            "lpips",
        ]
        existing = [c for c in numeric_cols if c in df.columns]
        if not existing:
            return pd.DataFrame()
        return df.groupby("engine")[existing].agg(["mean", "std"]).round(4)

    def export_latex_table(self) -> str:
        summary = self.get_summary_df()
        if summary.empty:
            return "% No results yet."
        summary.columns = [f"{m}_{s}" for m, s in summary.columns]
        summary = summary.reset_index()

        lines = [
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{Unified Benchmark on GSO}",
            r"\label{tab:unified_benchmark}",
            r"\begin{tabular}{lcccccc}",
            r"\toprule",
            r"Method & CD$\downarrow$ & F@0.1$\uparrow$ & F@0.2$\uparrow$ & "
            r"PSNR$\uparrow$ & LPIPS$\downarrow$ & Latency(s)$\downarrow$ \\",
            r"\midrule",
        ]
        for _, row in summary.iterrows():
            def fmt(col):
                mc = f"{col}_mean"
                return f"{row[mc]:.3f}" if mc in row and not pd.isna(row[mc]) else "--"

            lines.append(
                f"{row['engine']} & {fmt('chamfer_distance')} & {fmt('f_score_0.1')} & "
                f"{fmt('f_score_0.2')} & {fmt('psnr')} & {fmt('lpips')} & {fmt('latency_s')} \\"
            )
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        return "\n".join(lines)

    def save_results(self, path: str) -> None:
        records = []
        for r in self.results:
            rec = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in r.items()
                if k != "metadata"
            }
            rec["metadata"] = r.get("metadata", {})
            records.append(rec)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"[BenchmarkRunner] Results saved to {path}")
