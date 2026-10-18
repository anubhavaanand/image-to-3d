"""
Unified Benchmark Runner — executes engines over GSO samples and computes
3D geometric + 2D perceptual + hardware efficiency metrics.
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image

from ..engines.base import AbstractEngine, ReconstructionResult, Mesh3D
from ..preprocessing.background_removal import BackgroundRemover
from .metrics import (
    compute_chamfer_distance,
    compute_f_score,
    compute_psnr,
    compute_lpips,
    normalize_point_cloud,
)

GSO_ZUP_TO_YUP_4X4 = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


@dataclass
class GSOSample:
    sample_id: str
    image_path: Path
    gt_mesh_path: Optional[Path] = None
    gt_novel_Views: List[Path] = field(default_factory=list)
    gt_points: Optional[np.ndarray] = None


def bake_texture_to_vertices(tm, mesh_path: Optional[str | Path] = None):
    """
    Sample UV texture into per-vertex ColorVisuals.
    If trimesh failed to load `mat.image` (e.g. GSO `materials/textures/texture.png`
    relative path mismatch), explicitly locate `texture.png` on disk so GSO meshes
    never fall back to flat grey.
    """
    import trimesh

    vis = getattr(tm, "visual", None)
    if vis is None:
        return None
    if getattr(vis, "kind", None) == "vertex" and getattr(vis, "vertex_colors", None) is not None:
        return vis

    uv = getattr(vis, "uv", None)
    mat = getattr(vis, "material", None)
    img = None
    is_spar3d_top_left = False
    if mat is not None:
        if getattr(mat, "image", None) is not None:
            img = mat.image
        elif getattr(mat, "baseColorTexture", None) is not None:
            img = mat.baseColorTexture
            is_spar3d_top_left = True

    if img is None and mesh_path is not None:
        mp = Path(mesh_path)
        search_roots = [mp.parent, mp.parent.parent]
        for sr in search_roots:
            if not sr.exists():
                continue
            for cand in (
                list(sr.glob("texture.png"))
                + list(sr.glob("texture.jpg"))
                + list(sr.rglob("texture.png"))
                + list(sr.rglob("*.png"))
                + list(sr.rglob("*.jpg"))
            ):
                if "novel" not in cand.name and "000" not in cand.name and "001" not in cand.name and "input" not in cand.name:
                    try:
                        img = Image.open(cand)
                        img.load()
                        break
                    except Exception:
                        pass
            if img is not None:
                break

    if uv is None or len(uv) != len(tm.vertices) or img is None:
        return tm.visual.to_color()

    im = np.asarray(img.convert("RGBA"))
    h, w = im.shape[:2]
    u = np.clip(np.nan_to_num(uv[:, 0], nan=0.5), 0.0, 1.0)
    v = np.clip(np.nan_to_num(uv[:, 1], nan=0.5), 0.0, 1.0)
    x = np.clip((u * (w - 1)).round().astype(int), 0, w - 1)
    y_norm = v if is_spar3d_top_left else (1.0 - v)
    y = np.clip((y_norm * (h - 1)).round().astype(int), 0, h - 1)
    return trimesh.visual.ColorVisuals(mesh=tm, vertex_colors=im[y, x])


def load_gso_samples(
    dataset_path: str | Path,
    num_samples: int = 100,
    num_gt_points: int = 10000,
) -> List[GSOSample]:
    import trimesh

    root = Path(dataset_path)
    if not root.exists():
        print(f"[GSO Loader] Dataset path not found: {root}")
        return []

    samples: List[GSOSample] = []
    obj_dirs = sorted([d for d in root.iterdir() if d.is_dir()])[:num_samples]

    for obj_dir in obj_dirs:
        if (obj_dir / "images" / "000.png").exists():
            image_path = obj_dir / "images" / "000.png"
            novel_views = [obj_dir / "images" / "001.png"] if (obj_dir / "images" / "001.png").exists() else []
        else:
            img_candidates = (
                list((obj_dir / "images").glob("*.png"))
                + list((obj_dir / "thumbnails").glob("*.jpg"))
                + list((obj_dir / "thumbnails").glob("*.png"))
                + list(obj_dir.glob("*.png"))
                + list(obj_dir.glob("*.jpg"))
            )
            if not img_candidates:
                continue
            image_path = sorted(img_candidates)[0]
            novel_views = sorted(img_candidates)[1:5]

        mesh_candidates = (
            list((obj_dir / "meshes").glob("*.obj"))
            + list((obj_dir / "meshes").glob("*.ply"))
            + list(obj_dir.glob("*.obj"))
            + list(obj_dir.glob("*.ply"))
        )
        gt_mesh_path = mesh_candidates[0] if mesh_candidates else None

        gt_points = None
        if gt_mesh_path and gt_mesh_path.exists():
            try:
                mesh = trimesh.load(str(gt_mesh_path), force="mesh")
                mesh.apply_transform(GSO_ZUP_TO_YUP_4X4)
                pts, _ = trimesh.sample.sample_surface(mesh, num_gt_points)
                gt_points = normalize_point_cloud(np.array(pts, dtype=np.float32))
            except Exception as e:
                print(f"[GSO Loader] Failed to sample mesh {gt_mesh_path}: {e}")

        samples.append(
            GSOSample(
                sample_id=obj_dir.name,
                image_path=image_path,
                gt_mesh_path=gt_mesh_path,
                gt_novel_Views=novel_views,
                gt_points=gt_points,
            )
        )

    print(f"[GSO Loader] Loaded {len(samples)} samples from {root}.")
    return samples


def _normalize_vertices_robust(verts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    v = np.asarray(verts, dtype=np.float32)
    if len(v) == 0:
        return v, np.zeros(3, dtype=np.float32), 1.0
    bbox_center = 0.5 * (v.min(axis=0) + v.max(axis=0))
    vc = v - bbox_center
    radii = np.linalg.norm(vc, axis=1)
    scale = float(np.percentile(radii, 99.5))
    if scale < 1e-8:
        scale = float(radii.max()) if len(radii) > 0 else 1.0
    return vc / (scale + 1e-8), bbox_center, scale


def _candidate_rotations(n_yaw: int = 12) -> List[np.ndarray]:
    base_orientations = [
        np.eye(3, dtype=np.float32),
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32),
        np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32),
        np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32),
        np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32),
        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32),
    ]
    rots = []
    for base in base_orientations:
        for i in range(n_yaw):
            angle = 2.0 * np.pi * i / n_yaw
            ca, sa = np.cos(angle), np.sin(angle)
            ry = np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]], dtype=np.float32)
            rots.append(ry @ base)
    return rots


def align_mesh_to_gt(
    mesh: Mesh3D,
    gt_points: np.ndarray,
    num_points: int = 10000,
    n_iter: int = 3,
) -> Tuple[Mesh3D, np.ndarray]:
    import trimesh
    from scipy.spatial import cKDTree

    norm_verts, _, _ = _normalize_vertices_robust(mesh.vertices)
    tm_norm = trimesh.Trimesh(vertices=norm_verts, faces=mesh.faces, process=False)
    pred_pts, _ = trimesh.sample.sample_surface(tm_norm, num_points)
    pred_pts = normalize_point_cloud(np.array(pred_pts, dtype=np.float32))

    gt = np.asarray(gt_points, dtype=np.float32)
    sub_n = min(2048, len(pred_pts), len(gt))
    rng = np.random.default_rng(0)
    p_sub = pred_pts[rng.choice(len(pred_pts), sub_n, replace=False)]
    g_sub = gt[rng.choice(len(gt), sub_n, replace=False)]
    gt_tree = cKDTree(g_sub)

    best_cd = float("inf")
    best_R = np.eye(3, dtype=np.float32)
    best_t = np.zeros(3, dtype=np.float32)

    for R_init in _candidate_rotations(12):
        R_total = R_init.copy()
        t_total = np.zeros(3, dtype=np.float32)
        pts = p_sub @ R_total.T

        for _ in range(n_iter):
            _, idx = gt_tree.query(pts, k=1, workers=-1)
            matched = g_sub[idx]
            p_mean = pts.mean(axis=0)
            m_mean = matched.mean(axis=0)
            H = (pts - p_mean).T @ (matched - m_mean)
            U, _, Vt = np.linalg.svd(H)
            d = np.linalg.det(Vt.T @ U.T)
            D = np.diag([1.0, 1.0, 1.0 if d > 0 else -1.0]).astype(np.float32)
            R_step = (Vt.T @ D @ U.T).astype(np.float32)
            t_step = m_mean - (p_mean @ R_step.T)

            pts = (pts @ R_step.T) + t_step
            R_total = R_step @ R_total
            t_total = (t_total @ R_step.T) + t_step

        d_p2g, _ = gt_tree.query(pts, k=1, workers=-1)
        p_tree = cKDTree(pts)
        d_g2p, _ = p_tree.query(g_sub, k=1, workers=-1)
        cd = float(np.mean(d_p2g) + np.mean(d_g2p))
        if cd < best_cd:
            best_cd = cd
            best_R = R_total
            best_t = t_total

    aligned_pts = (pred_pts @ best_R.T) + best_t
    aligned_verts = (norm_verts @ best_R.T) + best_t
    aligned_mesh = Mesh3D(
        vertices=aligned_verts.astype(np.float32),
        faces=mesh.faces,
        vertex_normals=None,
        vertex_colors=mesh.vertex_colors,
        uvs=mesh.uvs,
        texture_image=mesh.texture_image,
    )
    return aligned_mesh, aligned_pts.astype(np.float32)


def make_camera_pose(azimuth_deg: float, elevation_deg: float = 15.0, distance: float = 2.0) -> np.ndarray:
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    cam_pos = np.array(
        [
            distance * np.cos(el) * np.sin(az),
            distance * np.sin(el),
            distance * np.cos(el) * np.cos(az),
        ],
        dtype=np.float32,
    )
    forward = -cam_pos / (np.linalg.norm(cam_pos) + 1e-8)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right) + 1e-8
    up = np.cross(right, forward)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = cam_pos
    return pose


def render_novel_view_same_bg(
    mesh: Mesh3D,
    azimuth_deg: float = 65.0,
    elevation_deg: float = 15.0,
    distance: float = 2.0,
    resolution: int = 256,
    white_bg: bool = True,
    already_normalized: bool = False,
) -> Optional[np.ndarray]:
    try:
        import trimesh
        import pyrender

        raw_v = np.asarray(mesh.vertices, dtype=np.float32)
        max_r = float(np.linalg.norm(raw_v - raw_v.mean(axis=0), axis=1).max()) if len(raw_v) > 0 else 1.0
        # Always normalize if already_normalized is False OR if mesh is still in raw meters (max_r < 0.6)
        if not already_normalized or max_r < 0.6 or max_r > 1.6:
            verts, _, _ = _normalize_vertices_robust(raw_v)
        else:
            verts = raw_v

        vcols = None
        if mesh.vertex_colors is not None and len(mesh.vertex_colors) == len(verts):
            vc = np.asarray(mesh.vertex_colors, dtype=np.float32)
            if vc.max() <= 1.05:
                vc = np.clip(vc * 255.0, 0.0, 255.0)
            vcols = np.clip(vc, 0.0, 255.0).astype(np.uint8)
            if vcols.shape[1] == 3:
                alpha = np.full((len(vcols), 1), 255, dtype=np.uint8)
                vcols = np.concatenate([vcols, alpha], axis=1)

        tm = trimesh.Trimesh(
            vertices=verts,
            faces=mesh.faces,
            vertex_colors=vcols,
            process=False,
        )
        try:
            tm.fix_normals()
        except Exception:
            pass

        scene = pyrender.Scene(
            ambient_light=np.array([0.72, 0.72, 0.72]),
            bg_color=np.array([1.0, 1.0, 1.0, 1.0] if white_bg else [0.0, 0.0, 0.0, 1.0]),
        )
        scene.add(pyrender.Mesh.from_trimesh(tm, smooth=False))

        camera_pose = make_camera_pose(azimuth_deg, elevation_deg, distance)
        camera = pyrender.PerspectiveCamera(yfov=np.radians(45.0))
        scene.add(camera, pose=camera_pose)

        light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.35)
        scene.add(light, pose=camera_pose)

        r = pyrender.OffscreenRenderer(resolution * 2, resolution * 2)
        color, _ = r.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
        r.delete()
        color = np.array(
            Image.fromarray(color).resize((resolution, resolution), Image.LANCZOS)
        )
        return color
    except Exception as e:
        print(f"[Renderer] pyrender failed ({e}). Skipping 2D perceptual metrics.")
        return None


render_novel_view = render_novel_view_same_bg


class BenchmarkRunner:
    def __init__(
        self,
        dataset_path: str | Path,
        engines: List[AbstractEngine],
        num_samples: int = 100,
        num_points: int = 10000,
        output_dir: str | Path = "outputs",
    ):
        self.dataset_path = Path(dataset_path)
        self.engines = engines
        self.num_samples = num_samples
        self.num_points = num_points
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bg_remover = BackgroundRemover()
        self.results: List[Dict[str, Any]] = []

    @staticmethod
    def _render_gt_novel(
        mesh_path: Path,
        azimuth_deg: float = 65.0,
        elevation_deg: float = 15.0,
        resolution: int = 256,
    ) -> Optional[np.ndarray]:
        try:
            import trimesh

            tm = trimesh.load(str(mesh_path), force="mesh")
            if tm is None or len(tm.faces) == 0:
                return None
            tm.apply_transform(GSO_ZUP_TO_YUP_4X4)
            tm.visual = bake_texture_to_vertices(tm, mesh_path=mesh_path)
            vcols = None
            if getattr(tm.visual, "vertex_colors", None) is not None:
                vcols = np.asarray(tm.visual.vertex_colors)[:, :3]
            gt_mesh = Mesh3D(
                vertices=np.asarray(tm.vertices, dtype=np.float32),
                faces=np.asarray(tm.faces, dtype=np.int64),
                vertex_colors=vcols,
            )
            return render_novel_view_same_bg(
                gt_mesh,
                azimuth_deg=azimuth_deg,
                elevation_deg=elevation_deg,
                resolution=resolution,
                white_bg=True,
                already_normalized=False,
            )
        except Exception as e:
            print(f"[GT Renderer] failed ({e})")
            return None

    def run(self) -> pd.DataFrame:
        samples = load_gso_samples(
            self.dataset_path,
            num_samples=self.num_samples,
            num_gt_points=self.num_points,
        )
        if not samples:
            raise RuntimeError(f"No valid GSO samples found in {self.dataset_path}")

        for engine in self.engines:
            info = engine.get_engine_info()
            engine_name = info.get("name", engine.__class__.__name__)
            print(f"\n{'='*60}\n  Evaluating: {engine_name}\n{'='*60}")
            engine.load_model()

            for idx, sample in enumerate(samples):
                print(
                    f"  [{engine_name}] ({idx+1}/{len(samples)}) {sample.sample_id} ...",
                    end=" ",
                    flush=True,
                )
                try:
                    raw_img = Image.open(sample.image_path).convert("RGBA")
                    clean_img = self.bg_remover.remove(raw_img)

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()

                    t_start = time.perf_counter()
                    recon: ReconstructionResult = engine.predict(clean_img)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    latency = time.perf_counter() - t_start

                    vram_gb = (
                        torch.cuda.max_memory_allocated() / (1024**3)
                        if torch.cuda.is_available()
                        else 0.0
                    )

                    row: Dict[str, Any] = {
                        "engine": engine_name,
                        "sample_id": sample.sample_id,
                        "latency_s": round(latency, 4),
                        "vram_peak_gb": round(vram_gb, 2),
                    }

                    aligned_for_render: Optional[Mesh3D] = None
                    if recon.mesh is not None and sample.gt_points is not None:
                        aligned_for_render, pred_pts = align_mesh_to_gt(
                            recon.mesh, sample.gt_points, self.num_points
                        )
                        row["chamfer_distance"] = compute_chamfer_distance(
                            pred_pts, sample.gt_points
                        )
                        f_scores = compute_f_score(pred_pts, sample.gt_points)
                        row.update(f_scores)
                    else:
                        row["chamfer_distance"] = np.nan
                        row["f_score_0.1"] = np.nan
                        row["f_score_0.2"] = np.nan
                        row["f_score_0.5"] = np.nan

                    render_target = (
                        aligned_for_render if aligned_for_render is not None else recon.mesh
                    )
                    if render_target is not None:
                        pred_render = render_novel_view_same_bg(
                            render_target,
                            azimuth_deg=65.0,
                            elevation_deg=15.0,
                            resolution=256,
                            white_bg=True,
                            already_normalized=(aligned_for_render is not None),
                        )
                        gt_render = None
                        if sample.gt_mesh_path and sample.gt_mesh_path.exists():
                            gt_render = self._render_gt_novel(
                                sample.gt_mesh_path,
                                azimuth_deg=65.0,
                                elevation_deg=15.0,
                                resolution=256,
                            )
                        if gt_render is None and sample.gt_novel_Views:
                            gt_img = (
                                Image.open(sample.gt_novel_Views[0])
                                .convert("RGB")
                                .resize((256, 256))
                            )
                            gt_render = np.array(gt_img)

                        if pred_render is not None and gt_render is not None:
                            row["psnr"] = compute_psnr(pred_render, gt_render)
                            row["lpips"] = compute_lpips(pred_render, gt_render)
                        else:
                            row["psnr"] = np.nan
                            row["lpips"] = np.nan

                        if idx < 10 and pred_render is not None:
                            qdir = self.output_dir / "qualitative" / sample.sample_id
                            qdir.mkdir(parents=True, exist_ok=True)
                            Image.fromarray(pred_render).save(qdir / f"{engine_name}_novel.png")
                            if gt_render is not None and not (qdir / "gt_novel.png").exists():
                                Image.fromarray(gt_render).save(qdir / "gt_novel.png")
                            if not (qdir / "input.png").exists():
                                clean_img.save(qdir / "input.png")
                            try:
                                import trimesh as _tm
                                vc_out = None
                                if render_target.vertex_colors is not None:
                                    vc = np.asarray(render_target.vertex_colors, dtype=np.float32)
                                    vc_out = (np.clip(vc, 0.0, 1.0) * 255.0).astype(np.uint8)
                                _tm.Trimesh(
                                    vertices=render_target.vertices,
                                    faces=render_target.faces,
                                    vertex_colors=vc_out,
                                    process=False,
                                ).export(str(qdir / f"{engine_name}_mesh.obj"))
                            except Exception:
                                pass
                    else:
                        row["psnr"] = np.nan
                        row["lpips"] = np.nan

                    row["metadata"] = recon.metadata
                    self.results.append(row)
                    print(
                        f"CD={row['chamfer_distance']:.4f}  "
                        f"F@0.2={row.get('f_score_0.2', float('nan')):.3f}  "
                        f"PSNR={row['psnr']:.2f}dB  "
                        f"t={latency:.2f}s  VRAM={vram_gb:.2f}GB"
                    )
                except Exception as e:
                    print(f"FAILED: {e}")
                    self.results.append(
                        {
                            "engine": engine_name,
                            "sample_id": sample.sample_id,
                            "error": str(e),
                        }
                    )

            try:
                self.save_results(f"{engine_name.lower()}_results.json")
            except Exception:
                pass

            engine.unload_model()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        return self.get_results_df()

    def get_results_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)

    def get_summary_df(self) -> pd.DataFrame:
        df = self.get_results_df()
        if df.empty:
            return df
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

    def save_results(self, filename: str = "benchmark_results.json") -> Path:
        out_path = Path(filename) if Path(filename).is_absolute() else (self.output_dir / filename)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"[BenchmarkRunner] Results saved to {out_path}")
        return out_path
