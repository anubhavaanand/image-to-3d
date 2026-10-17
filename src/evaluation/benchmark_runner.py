"""
Real benchmark runner — uniform evaluation of all three engines on GSO dataset.

Loads images + ground-truth meshes from GSO, runs each engine, computes
Chamfer Distance, F-Score@{0.1,0.2,0.5}, PSNR, LPIPS, and wall-clock latency
with a single shared protocol so results are directly comparable.
"""

from __future__ import annotations

import gc
import os
import json
import shutil
import time
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch

from ..engines.base import AbstractEngine, Mesh3D
from ..evaluation.chamfer_distance import compute_chamfer_distance, compute_chamfer_distance_symmetric
from ..evaluation.f_score import compute_f_score
from ..evaluation.psnr_lpips import compute_psnr, compute_lpips
from ..preprocessing.background_removal import BackgroundRemover


# ---------------------------------------------------------------------------
# GSO dataset helper
# ---------------------------------------------------------------------------

def load_gso_samples(
    dataset_root: str,
    num_samples: int = 100,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Load samples from Google Scanned Objects (GSO) dataset.

    Expected directory layout (standard GSO structure):
        <dataset_root>/
            <object_id>/
                images/           ← rendered RGBA views
                    000.png       ← front view (azimuth=0, elevation=0)
                    ...
                meshes/
                    model.obj     ← ground-truth mesh

    Downloads
    ---------
    GSO is available at:
      https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research
    A curated 100-object subset used by TripoSR is linked in their README.

    Parameters
    ----------
    dataset_root : str
        Path to the root of the GSO dataset.
    num_samples : int
        How many objects to evaluate (randomly sampled, reproducible via seed).
    seed : int
        Random seed for reproducible object selection.
    """
    import trimesh

    root = Path(dataset_root)
    object_dirs = sorted([d for d in root.iterdir() if d.is_dir()])

    rng = np.random.default_rng(seed)
    if len(object_dirs) > num_samples:
        indices = rng.choice(len(object_dirs), size=num_samples, replace=False)
        object_dirs = [object_dirs[i] for i in sorted(indices)]

    samples = []
    for obj_dir in object_dirs:
        # Find front-view image
        img_dir = obj_dir / "images"
        front_image_path = img_dir / "000.png"
        if not front_image_path.exists():
            # Try alternative naming
            candidates = sorted(img_dir.glob("*.png"))
            if not candidates:
                continue
            front_image_path = candidates[0]

        # Find ground-truth mesh
        mesh_path = obj_dir / "meshes" / "model.obj"
        if not mesh_path.exists():
            mesh_candidates = list((obj_dir / "meshes").glob("*.obj")) + \
                              list((obj_dir / "meshes").glob("*.ply"))
            if not mesh_candidates:
                continue
            mesh_path = mesh_candidates[0]

        try:
            gt_mesh = trimesh.load(str(mesh_path), force="mesh")
            # Sample 10K points from the ground-truth surface (standard protocol)
            gt_points, _ = trimesh.sample.sample_surface(gt_mesh, 10_000)
            gt_points = np.array(gt_points, dtype=np.float32)

            # Normalise GT to unit sphere centred at origin (centroid, not bbox)
            center = gt_points.mean(0)
            scale = np.linalg.norm(gt_points - center, axis=1).max()
            gt_points = (gt_points - center) / (scale + 1e-8)

        except Exception:
            continue

        samples.append({
            "id": obj_dir.name,
            "image_path": str(front_image_path),
            "gt_points": gt_points,
        })

    print(f"[BenchmarkRunner] Loaded {len(samples)} GSO samples.")
    return samples


# ---------------------------------------------------------------------------
# Mesh → point cloud helper
# ---------------------------------------------------------------------------

def mesh_to_pointcloud(mesh, n_points: int = 10_000) -> Optional[np.ndarray]:
    """Sample n_points from a Mesh3D surface and normalise to unit sphere."""
    import trimesh

    if mesh is None:
        return None

    try:
        tm = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            process=False,
        )
        pts, _ = trimesh.sample.sample_surface(tm, n_points)
        pts = np.array(pts, dtype=np.float32)

        center = pts.mean(0)
        scale = np.linalg.norm(pts - center, axis=1).max()
        pts = (pts - center) / (scale + 1e-8)
        return pts
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Novel-view rendering helper for PSNR / LPIPS
# ---------------------------------------------------------------------------

def render_novel_view(mesh, azimuth_deg: float = 45.0, resolution: int = 256):
    """
    Render a simple novel view of a Mesh3D for PSNR/LPIPS evaluation.
    Returns a (H, W, 3) uint8 numpy array.
    """
    try:
        import trimesh
        import pyrender
        import numpy as np

        tm = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            vertex_colors=(
                (mesh.vertex_colors * 255).astype(np.uint8)
                if mesh.vertex_colors is not None else None
            ),
            process=False,
        )
        scene = pyrender.Scene(
            bg_color=np.array([1.0, 1.0, 1.0, 1.0]),
            ambient_light=np.array([0.65, 0.65, 0.65]),
        )
        # Ambient light on Scene + directional light replaces the old approach
        # of intensity=3.0 with default [0,0,0] ambient. Ambient=[0.65,0.65,0.65]
        # provides even fill to prevent darkening while the directional light
        # (1.0,1.0,1.0 @ 1.8) adds natural shading on visible surfaces.
        scene.add(pyrender.Mesh.from_trimesh(tm))

        az = np.radians(azimuth_deg)
        camera_pose = np.array([
            [np.cos(az), 0, np.sin(az), 2.0 * np.sin(az)],
            [0,          1, 0,          0                ],
            [-np.sin(az),0, np.cos(az), 2.0 * np.cos(az)],
            [0,          0, 0,          1                ],
        ])
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        scene.add(camera, pose=camera_pose)
        scene.add(pyrender.DirectionalLight(color=np.array([1.0, 1.0, 1.0]), intensity=1.8))

        renderer = pyrender.OffscreenRenderer(resolution * 2, resolution * 2)
        # SKIP_CULL_FACES: double-sided rendering without a material override
        # (marching-cubes meshes often have inconsistent winding)
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
        renderer.delete()
        # 2x supersampling: MC meshes carry sub-pixel triangles at 256 px, which
        # rasterize as a dot field; downscaling a 2x render gives a clean image
        from PIL import Image as _PILImage
        color = np.array(
            _PILImage.fromarray(color).resize((resolution, resolution), _PILImage.LANCZOS)
        )
        return color  # (H, W, 3) uint8
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Rotation search + ICP alignment (TripoSR evaluation protocol)
# ---------------------------------------------------------------------------

def align_mesh_to_gt(pred_mesh, gt_points, n_rotations: int = 12):
    """
    Align predicted mesh to GT using rotation search + ICP.
    
    TripoSR evaluation protocol: search over yaw rotations (0-360 deg in steps),
    run ICP for each, pick best alignment by Chamfer distance.
    
    Args:
        pred_mesh: Mesh3D (prediction)
        gt_points: (N, 3) normalized GT point cloud
        n_rotations: Number of yaw angles to search (default 12 = 30 deg steps)
        
    Returns:
        aligned_mesh: Mesh3D with best alignment
        best_cd: float, best Chamfer distance found
    """
    try:
        import trimesh
        from scipy.spatial import cKDTree
        
        # Convert to trimesh
        tm = trimesh.Trimesh(
            vertices=pred_mesh.vertices,
            faces=pred_mesh.faces,
            vertex_colors=pred_mesh.vertex_colors,
            process=False,
        )
        
        # Sample points from mesh
        pred_pts, _ = trimesh.sample.sample_surface(tm, 10_000)
        pred_pts = np.array(pred_pts, dtype=np.float32)
        
        # Normalize pred to unit sphere (same as GT) — use centroid not bbox center
        center = pred_pts.mean(0)
        scale = np.linalg.norm(pred_pts - center, axis=1).max()
        pred_pts_norm = (pred_pts - center) / (scale + 1e-8)
        
        best_cd = float('inf')
        best_rotation = None
        best_pts = None
        best_yaw_angle = None
        
        # Rotation search around yaw axis
        for i in range(n_rotations):
            angle = 2 * np.pi * i / n_rotations
            rot_z = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle),  np.cos(angle), 0],
                [0,              0,             1]
            ], dtype=np.float32)
            
            rotated_pts = pred_pts_norm @ rot_z.T
            
            # ICP refinement (1 iteration, simple closest-point)
            tree_gt = cKDTree(gt_points)
            _, idx = tree_gt.query(rotated_pts)
            matched_gt = gt_points[idx]
            
            # Compute rigid transform (Kabsch algorithm)
            centroid_pred = rotated_pts.mean(axis=0)
            centroid_gt = matched_gt.mean(axis=0)
            pred_centered = rotated_pts - centroid_pred
            gt_centered = matched_gt - centroid_gt
            
            H = pred_centered.T @ gt_centered
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            
            # Ensure proper rotation (no reflection)
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            # Apply transform
            aligned_pts = (rotated_pts - centroid_pred) @ R.T + centroid_gt
            
            # Chamfer distance
            tree_gt = cKDTree(gt_points)
            d1, _ = tree_gt.query(aligned_pts)
            tree_pred = cKDTree(aligned_pts)
            d2, _ = tree_pred.query(gt_points)
            cd = np.mean(d1**2) + np.mean(d2**2)
            
            if cd < best_cd:
                best_cd = cd
                best_rotation = R
                best_pts = aligned_pts
                best_yaw_angle = angle
        
        # Apply full transform to mesh vertices:
        # 1. Normalize to unit sphere (centered + scaled)
        # 2. Apply best yaw rotation
        # 3. Apply Kabsch rotation
        # 4. Translate to GT centroid
        if best_rotation is not None and best_pts is not None and best_yaw_angle is not None:
            gt_center = gt_points.mean(axis=0)
            rot_z = np.array([
                [np.cos(best_yaw_angle), -np.sin(best_yaw_angle), 0],
                [np.sin(best_yaw_angle),  np.cos(best_yaw_angle), 0],
                [0,                       0,                      1]
            ], dtype=np.float32)
            
            # Apply: normalize → yaw → Kabsch → translate to GT centroid
            mesh_verts = np.array(tm.vertices, dtype=np.float32)
            # Step 1: normalize to unit sphere (same as pred_pts_norm)
            mesh_verts = (mesh_verts - center) / (scale + 1e-8)
            # Step 2: apply yaw rotation
            mesh_verts = mesh_verts @ rot_z.T
            # Step 3: apply Kabsch rotation
            mesh_verts = mesh_verts @ best_rotation.T
            # Step 4: translate to GT centroid
            mesh_verts = mesh_verts + gt_center
            
            tm.vertices = mesh_verts
        
        # Return as Mesh3D so render_novel_view_same_bg can access .vertex_colors
        vertex_colors = None
        if hasattr(tm.visual, 'vertex_colors') and tm.visual.vertex_colors is not None:
            vc_arr = np.array(tm.visual.vertex_colors)
            if vc_arr.ndim == 2 and vc_arr.shape[0] == len(tm.vertices):
                vertex_colors = vc_arr[:, :3].astype(np.float32) / 255.0
        
        return Mesh3D(
            vertices=np.array(tm.vertices, dtype=np.float32),
            faces=np.array(tm.faces, dtype=np.int64),
            vertex_colors=vertex_colors,
        ), best_cd, best_pts
        
    except Exception:
        return None, float('inf'), None


def align_mesh_to_gt_simple(pred_mesh, gt_points, n_rotations: int = 12):
    """
    Simpler rotation search without ICP - just yaw rotation search.
    Matches TripoSR paper's evaluation protocol.
    
    Returns (aligned_mesh, best_cd, aligned_points_normalized) where:
    - aligned_mesh: mesh with best rotation applied (vertices in original scale)
    - best_cd: symmetric mean-distance Chamfer distance (TripoSR convention)
    - aligned_points_normalized: 10K points already normalized to unit sphere, 
      so caller can skip re-normalization
    """
    try:
        import trimesh
        from scipy.spatial import cKDTree
        
        tm = trimesh.Trimesh(
            vertices=pred_mesh.vertices,
            faces=pred_mesh.faces,
            vertex_colors=pred_mesh.vertex_colors,
            process=False,
        )
        
        pred_pts, _ = trimesh.sample.sample_surface(tm, 10_000)
        pred_pts = np.array(pred_pts, dtype=np.float32)
        
        # Normalize once — use centroid not bbox center
        center = pred_pts.mean(0)
        scale = np.linalg.norm(pred_pts - center, axis=1).max()
        pred_pts_norm = (pred_pts - center) / (scale + 1e-8)
        
        best_cd = float('inf')
        best_rot = None
        best_pts = None
        
        for i in range(n_rotations):
            angle = 2 * np.pi * i / n_rotations
            rot_z = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle),  np.cos(angle), 0],
                [0,              0,             1]
            ], dtype=np.float32)
            
            rotated = pred_pts_norm @ rot_z.T
            
            # Query ROTATED points (not unrotated!)
            tree = cKDTree(gt_points)
            d1, _ = tree.query(rotated)
            tree_pred = cKDTree(rotated)
            d2, _ = tree_pred.query(gt_points)
            
            # Symmetric mean-distance CD (TripoSR convention), NOT squared
            cd = np.mean(d1) + np.mean(d2)
            
            if cd < best_cd:
                best_cd = cd
                best_rot = angle
                best_pts = rotated
        
        # Apply best rotation to mesh
        if best_rot is not None:
            rot_z = np.array([
                [np.cos(best_rot), -np.sin(best_rot), 0],
                [np.sin(best_rot),  np.cos(best_rot), 0],
                [0,                0,               1]
            ], dtype=np.float32)
            tm.vertices = (tm.vertices - center) @ rot_z.T + center
        
        return tm, best_cd, best_pts
        
    except Exception:
        return None, float('inf'), None


# ---------------------------------------------------------------------------
# Texture baking: GSO meshes carry UV textures; trimesh's to_color() returns
# flat gray for them, which made every input / GT render colorless.
# ---------------------------------------------------------------------------

def bake_texture_to_vertices(tm):
    """Sample the mesh's UV texture into per-vertex colors.

    Returns a ColorVisuals (kind='vertex') on success, else the original
    visual untouched so callers can fall back to their previous behavior.
    """
    try:
        import numpy as np
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


def vertex_colors_rgb(tm):
    """Per-vertex RGB in [0,1] if the mesh has vertex colors, else None."""
    import numpy as np
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
# Same-background rendering helper (for fair PSNR)
# ---------------------------------------------------------------------------

def normalize_mesh_to_unit_sphere(mesh):
    """Return a copy of a Mesh3D translated to origin and scaled so max radius = 1.
    Must be applied to BOTH prediction and GT before rendering: raw model outputs
    come at arbitrary scales (e.g. GSO raw meshes ~0.19 units extent vs the render
    camera's ~1.15-unit visible half-height -> predictions rendered as sub-pixel specks).
    
    Uses surface-area-weighted centroid (mean of vertices) rather than bounding-box
    center, so asymmetric objects are balanced around the origin and fill the frame.
    
    Uses 99.5th percentile radius instead of max — prevents a single stray
    Gaussian-splat floater from inflating the scale and shrinking the object
    to a sub-pixel speck.
    """
    import numpy as np
    verts = np.array(mesh.vertices, dtype=np.float32)
    if len(verts) == 0:
        return mesh
    c = verts.mean(0)  # centroid (not bbox center) for better centering
    radii = np.linalg.norm(verts - c, axis=1)
    s = float(np.percentile(radii, 99.5))  # robust to floaters
    verts = (verts - c) / (s + 1e-8)
    return type(mesh)(
        vertices=verts,
        faces=np.array(mesh.faces),
        vertex_colors=(np.array(mesh.vertex_colors, dtype=np.float32)
                       if getattr(mesh, "vertex_colors", None) is not None else None),
    )


def render_novel_view_same_bg(mesh, azimuth_deg: float = 45.0, resolution: int = 256,
                               background_color: tuple = (1.0, 1.0, 1.0)):
    """
    Render novel view with specified background color.
    For fair PSNR, both prediction and GT should use the SAME background.
    
    Args:
        mesh: Mesh3D
        azimuth_deg: Camera azimuth angle
        resolution: Output resolution
        background_color: RGB tuple in [0,1] range (default white)
        
    Returns:
        (H, W, 3) uint8 numpy array
    """
    try:
        import trimesh
        import pyrender
        import numpy as np
        
        tm = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            vertex_colors=(
                (mesh.vertex_colors * 255).astype(np.uint8)
                if mesh.vertex_colors is not None else None
            ),
            process=False,
        )
        scene = pyrender.Scene(
            bg_color=np.array([*background_color, 1.0]),
            ambient_light=np.array([0.65, 0.65, 0.65]),
        )
        # Ambient light on Scene (not a separate AmbientLight) + directional light.
        # Old approach: intensity=3.0 with default [0,0,0] ambient → bright TRELLIS
        # objects lost 1.08-1.24 dB PSNR from 5x dimming. New: ambient=[0.65,0.65,0.65]
        # for even fill + directional [1,1,1]@1.8 for natural highlights.
        scene.add(pyrender.Mesh.from_trimesh(tm))

        az = np.radians(azimuth_deg)
        camera_pose = np.array([
            [np.cos(az), 0, np.sin(az), 2.0 * np.sin(az)],
            [0,          1, 0,          0                ],
            [-np.sin(az), 0, np.cos(az), 2.0 * np.cos(az)],
            [0,          0, 0,          1                ],
        ])
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        scene.add(camera, pose=camera_pose)
        scene.add(pyrender.DirectionalLight(color=np.array([1.0, 1.0, 1.0]), intensity=1.8))
        
        renderer = pyrender.OffscreenRenderer(resolution * 2, resolution * 2)
        # SKIP_CULL_FACES: double-sided rendering without a material override
        # (marching-cubes meshes often have inconsistent winding)
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
        renderer.delete()
        # 2x supersampling: MC meshes carry sub-pixel triangles at 256 px, which
        # rasterize as a dot field; downscaling a 2x render gives a clean image
        from PIL import Image as _PILImage
        color = np.array(
            _PILImage.fromarray(color).resize((resolution, resolution), _PILImage.LANCZOS)
        )
        return color
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """
    Runs all engines on the same dataset with the same metric protocol.

    Example
    -------
    runner = BenchmarkRunner(
        dataset_path="./data/gso",
        engines=[TriposrEngine(), TrellisEngine(), LGMEngine()],
        num_samples=100,
    )
    runner.run()
    df = runner.get_results_df()
    print(runner.export_latex_table())
    runner.save_results("./results/benchmark_results.json")
    """

    def __init__(
        self,
        dataset_path: str,
        engines: List[AbstractEngine],
        num_samples: int = 100,
        remove_background: bool = True,
        compute_render_metrics: bool = True,
        output_dir: str = "./outputs/benchmark",
    ):
        self.dataset_path = dataset_path
        self.engines = engines
        self.num_samples = num_samples
        self.remove_background = remove_background
        self.compute_render_metrics = compute_render_metrics
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._bg_remover = BackgroundRemover() if remove_background else None
        self.results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    def load_dataset(self) -> List[Dict[str, Any]]:
        return load_gso_samples(self.dataset_path, num_samples=self.num_samples)

    # ------------------------------------------------------------------
    def _flush_partial(self, engine_name: str) -> None:
        """Checkpoint results every N objects so a mid-run crash never loses completed work."""
        try:
            rows = [r for r in self.results if r.get("engine") == engine_name]
            if not rows:
                return
            tmp = self.output_dir / f".partial_{engine_name}.json"
            tmp.write_text(json.dumps(rows, indent=1, default=str))
            os.replace(tmp, self.output_dir / f"partial_{engine_name}.json")
            print(f"  [checkpoint] {engine_name}: {len(rows)} objects saved")
        except Exception as _e:
            print(f"  [checkpoint] failed (non-fatal): {_e}")

    def run(self) -> None:
        """Run all engines over all samples. Results accumulated in self.results."""
        samples = self.load_dataset()

        for engine in self.engines:
            info = engine.get_engine_info()
            engine_name = info["name"]
            print(f"\n{'='*60}")
            print(f"  Evaluating: {engine_name}")
            print(f"{'='*60}")

            for idx, sample in enumerate(samples):
                sample_id = sample["id"]
                print(f"  [{engine_name}] {sample_id} … ", end="", flush=True)

                try:
                    # ---- load & preprocess image ----------------------
                    image = Image.open(sample["image_path"]).convert("RGBA")
                    if self._bg_remover is not None:
                        image = self._bg_remover.remove(image)
                    else:
                        # Composite RGBA onto white RGBA
                        white = Image.new("RGBA", image.size, "WHITE")
                        white.paste(image, mask=image.split()[3])
                        image = white

                    # ---- inference ------------------------------------
                    # Peak VRAM: reset stats right before predict so
                    # max_memory_allocated covers this object's inference
                    # (steady-state after the first object = model weights
                    # included — the honest deployment number).
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                    t0 = time.perf_counter()
                    result = engine.predict(image)
                    latency = time.perf_counter() - t0
                    vram_peak_gb = (
                        round(torch.cuda.max_memory_allocated() / 1e9, 2)
                        if torch.cuda.is_available() else None
                    )

                    # ---- geometry metrics (requires mesh) -------------
                    geo_metrics: Dict[str, Any] = {}
                    aligned_for_render = False
                    pred_pts = mesh_to_pointcloud(result.mesh)
                    if pred_pts is not None:
                        gt_pts = sample["gt_points"]

                        # Apply canonical rotation search + ICP alignment for ALL
                        # engines (SPAR3D Section 4.1 protocol). The previous code
                        # only aligned TripoSR; TRELLIS and LGM also benefit because
                        # mesh orientation is arbitrary in 3D space.
                        if result.mesh is not None:
                            aligned_mesh, best_cd, aligned_pts_norm = align_mesh_to_gt(
                                result.mesh, gt_pts, n_rotations=12
                            )
                            if aligned_mesh is not None and best_cd != float('inf') and aligned_pts_norm is not None:
                                # Use symmetric mean-distance CD (alignment result)
                                geo_metrics["chamfer_distance"] = float(best_cd)
                                # Use pre-normalized aligned points for F-score
                                fs = compute_f_score(aligned_pts_norm, gt_pts, thresholds=[0.1, 0.2, 0.5])
                                geo_metrics.update(fs)
                                # Store aligned mesh for rendering so PSNR/LPIPS
                                # uses the canonical-orientation prediction.
                                # The aligned mesh is already normalized to the GT
                                # unit sphere, so rendering should NOT re-normalize.
                                result.mesh = aligned_mesh
                                aligned_for_render = True
                            else:
                                # Fallback to symmetric CD without alignment
                                geo_metrics["chamfer_distance"] = float(
                                    compute_chamfer_distance_symmetric(pred_pts, gt_pts)
                                )
                                fs = compute_f_score(pred_pts, gt_pts, thresholds=[0.1, 0.2, 0.5])
                                geo_metrics.update(fs)
                        else:
                            # No mesh for alignment — use un-aligned metrics
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

                    # ---- rendering metrics (PSNR / LPIPS) ------------
                    render_metrics: Dict[str, Any] = {}
                    pred_render_to_save = None
                    if self.compute_render_metrics:
                        if result.mesh is not None:
                            try:
                                # Normalize the PREDICTION exactly like the GT (unit sphere)
                                # before rendering - raw engine outputs have arbitrary scale
                                # and would otherwise render as sub-pixel specks.
                                # BUT: if alignment already placed the mesh in GT unit-sphere
                                # space, skip re-normalization to preserve the canonical
                                # orientation that matches gt_novel.png.
                                if aligned_for_render:
                                    pred_norm = result.mesh
                                else:
                                    pred_norm = normalize_mesh_to_unit_sphere(result.mesh)
                                pred_render = render_novel_view_same_bg(
                                    pred_norm, azimuth_deg=45, background_color=(1.0, 1.0, 1.0)
                                )
                                # persist the normalized prediction mesh for offline QA/re-render
                                try:
                                    import trimesh as _tm
                                    _qm = self.output_dir / "qualitative" / sample_id
                                    _qm.mkdir(parents=True, exist_ok=True)
                                    _m = _tm.Trimesh(
                                        vertices=pred_norm.vertices, faces=pred_norm.faces,
                                        vertex_colors=(pred_norm.vertex_colors * 255).astype(np.uint8)
                                        if pred_norm.vertex_colors is not None else None,
                                        process=False)
                                    # Cap .obj saves: ~26 MB/object, so the 100-object
                                    # full run would ship ~2.7 GB compressed of meshes.
                                    # Keep metrics+renders for all; persist .obj QA mesh
                                    # for only the first 10 objects (idx < 10).
                                    if idx < 10:
                                        _m.export(str(_qm / f"{engine_name}_mesh.obj"))
                                except Exception:
                                    pass
                                if pred_render is not None:
                                    pred_render_to_save = pred_render
                                    gt_render_path = (
                                        Path(sample["image_path"]).parent / "001.png"
                                    )
                                    if gt_render_path.exists():
                                        # Render GT with same white background for fair comparison
                                        # Load GT mesh and render with white background
                                        gt_mesh_path = (
                                            Path(sample["image_path"]).parent.parent / "meshes" / "model.obj"
                                        )
                                        gt_render = None
                                        if gt_mesh_path.exists():
                                            import trimesh
                                            gt_mesh = trimesh.load(str(gt_mesh_path), force="mesh")
                                            # Bake UV texture to vertex colors (to_color()
                                            # returns flat gray for GSO's textured meshes)
                                            gt_mesh.visual = bake_texture_to_vertices(gt_mesh)
                                            # Normalize GT mesh (centroid-based, same as load_gso_samples)
                                            pts = np.array(gt_mesh.vertices, dtype=np.float32)
                                            c = pts.mean(0)
                                            s = np.linalg.norm(pts - c, axis=1).max()
                                            gt_mesh.apply_translation(-c)
                                            gt_mesh.apply_scale(1.0 / (s + 1e-8))
                                            gt_render = render_novel_view_same_bg(
                                                type('obj', (object,), {
                                                    'vertices': gt_mesh.vertices,
                                                    'faces': gt_mesh.faces,
                                                    'vertex_colors': vertex_colors_rgb(gt_mesh)
                                                }), azimuth_deg=45, background_color=(1.0, 1.0, 1.0)
                                            )
                                        if gt_render is None:
                                            # Fallback: use existing GT render image (but note it has black bg)
                                            gt_render = np.array(
                                                Image.open(gt_render_path).convert("RGB")
                                                      .resize((256, 256))
                                            )
                                        render_metrics["psnr"] = float(
                                            compute_psnr(pred_render, gt_render)
                                        )
                                        def to_lpips_tensor(img_np):
                                            img_np = np.ascontiguousarray(img_np)  # pyrender returns negative strides - torch refuses them
                                            t = torch.from_numpy(img_np).float() / 127.5 - 1.0
                                            return t.permute(2, 0, 1).unsqueeze(0)
                                        render_metrics["lpips"] = float(
                                            compute_lpips(
                                                to_lpips_tensor(pred_render),
                                                to_lpips_tensor(gt_render),
                                            )
                                        )
                            except Exception as _rm_err:
                                print(f"  [render] metrics failed (non-fatal): {_rm_err}")
                        elif result.gaussians is not None:
                            render_metrics = {
                                "psnr": float("nan"),
                                "lpips": float("nan"),
                            }

                    # ---- qualitative assets (never crash a metrics run) ----
                    # input.png + gt_novel.png are shared by all engines;
                    # {engine}_novel.png is this engine's novel-view render.
                    try:
                        qual_dir = self.output_dir / "qualitative" / sample_id
                        qual_dir.mkdir(parents=True, exist_ok=True)
                        input_src = Path(sample["image_path"])
                        if not (qual_dir / "input.png").exists() and input_src.exists():
                            shutil.copy(input_src, qual_dir / "input.png")
                        gt_novel_src = input_src.parent / "001.png"
                        if not (qual_dir / "gt_novel.png").exists() and gt_novel_src.exists():
                            shutil.copy(gt_novel_src, qual_dir / "gt_novel.png")
                        if pred_render_to_save is not None:
                            Image.fromarray(pred_render_to_save).save(
                                qual_dir / f"{engine_name}_novel.png"
                            )
                    except Exception as _qual_exc:
                        print(f"  [qualitative] save failed: {_qual_exc}")

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
                    _vram_str = f"  VRAM={vram_peak_gb:.2f}GB" if vram_peak_gb else ""
                    print(f"CD={geo_metrics.get('chamfer_distance', 'n/a'):.4f}  "
                          f"F@0.1={geo_metrics.get('f_score_0.1', 'n/a'):.3f}  "
                          f"t={latency:.2f}s{_vram_str}")

                except Exception as exc:
                    print(f"FAILED: {exc}")
                    traceback.print_exc()
                    self.results.append({
                        "engine": engine_name,
                        "sample_id": sample_id,
                        "error": str(exc),
                    })

                # checkpoint every 10 objects (crash insurance for the 100-object run)
                _done = sum(1 for r in self.results if r.get("engine") == engine_name)
                if _done % 10 == 0:
                    self._flush_partial(engine_name)

            gc.collect()
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    def get_results_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)

    def get_summary_df(self) -> pd.DataFrame:
        """Per-engine mean ± std of all numeric metrics."""
        df = self.get_results_df()
        if df.empty:
            print("[BenchmarkRunner] No results to summarize.")
            return pd.DataFrame()
        numeric_cols = [
            "latency_s", "chamfer_distance",
            "f_score_0.1", "f_score_0.2", "f_score_0.5",
            "psnr", "lpips",
        ]
        existing = [c for c in numeric_cols if c in df.columns]
        if not existing:
            print(f"[BenchmarkRunner] No numeric metrics in results. Columns: {list(df.columns)}")
            if "error" in df.columns:
                for _, row in df.iterrows():
                    print(f"  ERROR [{row.get('engine','?')} / {row.get('sample_id','?')}]: {str(row['error'])[:200]}")
            return pd.DataFrame()
        return (
            df.groupby("engine")[existing]
              .agg(["mean", "std"])
              .round(4)
        )

    def export_latex_table(self) -> str:
        """Produce a ready-to-paste IEEE LaTeX table from summary stats."""
        summary = self.get_summary_df()
        if summary.empty:
            return "% No results yet."

        # Flatten multi-level columns
        summary.columns = [f"{m}_{s}" for m, s in summary.columns]
        summary = summary.reset_index()

        lines = [
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{Unified Benchmark — TripoSR vs. LGM vs. TRELLIS on GSO (100 objects)}",
            r"\label{tab:unified_benchmark}",
            r"\begin{tabular}{lcccccc}",
            r"\toprule",
            r"Method & CD$\downarrow$ & F@0.1$\uparrow$ & F@0.2$\uparrow$ & "
            r"PSNR$\uparrow$ & LPIPS$\downarrow$ & Latency(s)$\downarrow$ \\",
            r"\midrule",
        ]
        for _, row in summary.iterrows():
            def fmt(col):
                mean_col = f"{col}_mean"
                if mean_col in row and not pd.isna(row[mean_col]):
                    return f"{row[mean_col]:.3f}"
                return "--"

            lines.append(
                f"{row['engine']} & "
                f"{fmt('chamfer_distance')} & "
                f"{fmt('f_score_0.1')} & "
                f"{fmt('f_score_0.2')} & "
                f"{fmt('psnr')} & "
                f"{fmt('lpips')} & "
                f"{fmt('latency_s')} \\\\"
            )

        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        return "\n".join(lines)

    def save_results(self, path: str) -> None:
        """Save full per-sample results to JSON."""
        records = []
        for r in self.results:
            rec = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in r.items() if k != "metadata"}
            rec["metadata"] = r.get("metadata", {})
            records.append(rec)
        with open(path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"[BenchmarkRunner] Results saved to {path}")
