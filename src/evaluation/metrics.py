"""
Metrics module for 3D evaluation.
Exports functions for computing Chamfer Distance, F-Score, PSNR, and LPIPS.
"""

from .chamfer_distance import compute_chamfer_distance
from .f_score import compute_f_score
from .psnr_lpips import compute_psnr, compute_lpips
import numpy as np


def normalize_point_cloud(points: np.ndarray) -> np.ndarray:
    """
    Normalize a point cloud to fit within a unit sphere centered at the origin.
    
    Args:
        points: Point cloud of shape (N, 3)
        
    Returns:
        Normalized point cloud of shape (N, 3)
    """
    if len(points) == 0:
        return points
    
    # Center the point cloud at the origin
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    
    # Scale to fit within a unit sphere
    max_distance = np.max(np.linalg.norm(centered_points, axis=1))
    if max_distance > 0:
        normalized_points = centered_points / max_distance
    else:
        normalized_points = centered_points
        
    return normalized_points.astype(np.float32)


__all__ = [
    "compute_chamfer_distance",
    "compute_f_score",
    "compute_psnr",
    "compute_lpips",
    "normalize_point_cloud",
]
