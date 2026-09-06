import numpy as np
from scipy.spatial import cKDTree

def compute_chamfer_distance(pc1: np.ndarray, pc2: np.ndarray) -> float:
    """Computes the Chamfer Distance between two point clouds."""
    tree1 = cKDTree(pc1)
    tree2 = cKDTree(pc2)
    
    d1, _ = tree1.query(pc2)
    d2, _ = tree2.query(pc1)
    
    return np.mean(d1**2) + np.mean(d2**2)
