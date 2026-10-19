import numpy as np
from scipy.spatial import cKDTree


def compute_chamfer_distance(pc1: np.ndarray, pc2: np.ndarray, squared: bool = False) -> float:
    """Computes symmetric mean L2 Chamfer Distance between two point clouds."""
    p1 = np.asarray(pc1, dtype=np.float32)
    p2 = np.asarray(pc2, dtype=np.float32)
    tree1 = cKDTree(p1)
    tree2 = cKDTree(p2)
    d1, _ = tree1.query(p2, k=1, workers=-1)
    d2, _ = tree2.query(p1, k=1, workers=-1)
    if squared:
        return float(np.mean(d1 ** 2) + np.mean(d2 ** 2))
    return float(np.mean(d1) + np.mean(d2))


def compute_chamfer_distance_symmetric(pc1: np.ndarray, pc2: np.ndarray) -> float:
    return compute_chamfer_distance(pc1, pc2, squared=False)
