"""
Real benchmark runner — uniform evaluation of all three engines on GSO dataset.

Loads images + ground-truth meshes from GSO, runs each engine, computes
Chamfer Distance, F-Score@{0.1,0.2,0.5}, PSNR, LPIPS, and wall-clock latency
with a single shared protocol so results are directly comparable.
"""

from __future__ import annotations

import os
import json
import time
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from PIL import Image

from ..engines.base import AbstractEngine
from ..evaluation.chamfer_distance import compute_chamfer_distance
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

            # Normalise GT to unit sphere centred at origin
            center = (gt_points.max(0) + gt_points.min(0)) / 2
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

        center = (pts.max(0) + pts.min(0)) / 2
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
        scene = pyrender.Scene()
        scene.add(pyrender.Mesh.from_trimesh(tm))

        az = np.radians(azimuth_deg)
        camera_pose = np.array([
            [np.cos(az), 0, np.sin(az), 1.5 * np.sin(az)],
            [0,          1, 0,          0                ],
            [-np.sin(az),0, np.cos(az), 1.5 * np.cos(az)],
            [0,          0, 0,          1                ],
        ])
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        scene.add(camera, pose=camera_pose)
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0))

        renderer = pyrender.OffscreenRenderer(resolution, resolution)
        color, _ = renderer.render(scene)
        renderer.delete()
        return color  # (H, W, 3) uint8
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
    def run(self) -> None:
        """Run all engines over all samples. Results accumulated in self.results."""
        samples = self.load_dataset()

        for engine in self.engines:
            info = engine.get_engine_info()
            engine_name = info["name"]
            print(f"\n{'='*60}")
            print(f"  Evaluating: {engine_name}")
            print(f"{'='*60}")

            for sample in samples:
                sample_id = sample["id"]
                print(f"  [{engine_name}] {sample_id} … ", end="", flush=True)

                try:
                    # ---- load & preprocess image ----------------------
                    image = Image.open(sample["image_path"]).convert("RGBA")
                    if self._bg_remover is not None:
                        image = self._bg_remover.remove(image)
                    else:
                        # Composite RGBA onto white
                        white = Image.new("RGB", image.size, "WHITE")
                        white.paste(image, mask=image.split()[3])
                        image = white

                    # ---- inference ------------------------------------
                    t0 = time.perf_counter()
                    result = engine.predict(image)
                    latency = time.perf_counter() - t0

                    # ---- geometry metrics (requires mesh) -------------
                    geo_metrics: Dict[str, Any] = {}
                    pred_pts = mesh_to_pointcloud(result.mesh)
                    if pred_pts is not None:
                        gt_pts = sample["gt_points"]
                        geo_metrics["chamfer_distance"] = float(
                            compute_chamfer_distance(pred_pts, gt_pts)
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
                    if self.compute_render_metrics and result.mesh is not None:
                        try:
                            import torch
                            pred_render = render_novel_view(result.mesh, azimuth_deg=45)
                            if pred_render is not None:
                                # For PSNR we need a GT render at the same angle.
                                # We use the next available rendered view from GSO.
                                gt_render_path = (
                                    Path(sample["image_path"]).parent / "001.png"
                                )
                                if gt_render_path.exists():
                                    gt_render = np.array(
                                        Image.open(gt_render_path).convert("RGB")
                                              .resize((256, 256))
                                    )
                                    render_metrics["psnr"] = float(
                                        compute_psnr(pred_render, gt_render)
                                    )
                                    # LPIPS expects tensors in [-1,1] NCHW
                                    def to_lpips_tensor(img_np):
                                        t = torch.from_numpy(img_np).float() / 127.5 - 1.0
                                        return t.permute(2, 0, 1).unsqueeze(0)
                                    render_metrics["lpips"] = float(
                                        compute_lpips(
                                            to_lpips_tensor(pred_render),
                                            to_lpips_tensor(gt_render),
                                        )
                                    )
                        except Exception:
                            pass

                    record = {
                        "engine": engine_name,
                        "sample_id": sample_id,
                        "latency_s": round(latency, 4),
                        **geo_metrics,
                        **render_metrics,
                        "metadata": result.metadata or {},
                    }
                    self.results.append(record)
                    print(f"CD={geo_metrics.get('chamfer_distance', 'n/a'):.4f}  "
                          f"F@0.1={geo_metrics.get('f_score_0.1', 'n/a'):.3f}  "
                          f"t={latency:.2f}s")

                except Exception as exc:
                    print(f"FAILED: {exc}")
                    traceback.print_exc()
                    self.results.append({
                        "engine": engine_name,
                        "sample_id": sample_id,
                        "error": str(exc),
                    })

    # ------------------------------------------------------------------
    def get_results_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)

    def get_summary_df(self) -> pd.DataFrame:
        """Per-engine mean ± std of all numeric metrics."""
        df = self.get_results_df()
        numeric_cols = [
            "latency_s", "chamfer_distance",
            "f_score_0.1", "f_score_0.2", "f_score_0.5",
            "psnr", "lpips",
        ]
        existing = [c for c in numeric_cols if c in df.columns]
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
