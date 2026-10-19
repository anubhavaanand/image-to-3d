import numpy as np
from .chamfer_distance import compute_chamfer_distance
from .f_score import compute_f_score
from .psnr_lpips import compute_psnr, compute_lpips


def normalize_point_cloud(points: np.ndarray) -> np.ndarray:
    """Center point cloud and normalize by 99.5th-percentile radius to fit unit sphere."""
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) == 0:
        return pts
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    radii = np.linalg.norm(centered, axis=1)
    scale = float(np.percentile(radii, 99.5))
    if scale < 1e-8:
        scale = float(radii.max()) if len(radii) > 0 else 1.0
    return (centered / (scale + 1e-8)).astype(np.float32)


__all__ = [
    "compute_chamfer_distance",
    "compute_f_score",
    "compute_psnr",
    "compute_lpips",
    "normalize_point_cloud",
]
