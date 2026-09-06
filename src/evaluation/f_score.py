import numpy as np
from scipy.spatial import cKDTree

def compute_f_score(pc1: np.ndarray, pc2: np.ndarray, thresholds=[0.1, 0.2, 0.5]) -> dict:
    """Computes the F-Score at specified thresholds."""
    tree1 = cKDTree(pc1)
    tree2 = cKDTree(pc2)
    
    d1, _ = tree1.query(pc2)
    d2, _ = tree2.query(pc1)
    
    results = {}
    for th in thresholds:
        precision = np.mean(d1 < th)
        recall = np.mean(d2 < th)
        
        if precision + recall > 0:
            f_score = 2 * (precision * recall) / (precision + recall)
        else:
            f_score = 0.0
            
        results[f'f_score_{th}'] = f_score
        
    return results
