import numpy as np
from scipy.spatial import cKDTree

def compute_chamfer_distance(pc1: np.ndarray, pc2: np.ndarray, squared: bool = True) -> float:
    """
    Computes the Chamfer Distance between two point clouds.
    
    Args:
        pc1, pc2: Point clouds of shape (N, 3) and (M, 3)
        squared: If True, returns squared Chamfer distance (sum of squared distances).
                 If False, returns symmetric mean-distance Chamfer (mean of distances).
                 
    Returns:
        float: Chamfer distance
        
    Note:
        - squared=True returns the standard squared Chamfer (used in many ML papers)
        - squared=False returns symmetric mean-distance Chamfer (more interpretable, 
          matches TripoSR paper's CD convention)
    """
    tree1 = cKDTree(pc1)
    tree2 = cKDTree(pc2)
    
    d1, _ = tree1.query(pc2)
    d2, _ = tree2.query(pc1)
    
    if squared:
        return np.mean(d1**2) + np.mean(d2**2)
    else:
        # Symmetric mean-distance Chamfer (more interpretable, matches TripoSR paper)
        return np.mean(d1) + np.mean(d2)

def compute_chamfer_distance_symmetric(pc1: np.ndarray, pc2: np.ndarray) -> float:
    """
    Convenience wrapper for symmetric mean-distance Chamfer Distance.
    Matches TripoSR paper convention.
    """
    return compute_chamfer_distance(pc1, pc2, squared=False)
