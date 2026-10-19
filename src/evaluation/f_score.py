import numpy as np
from scipy.spatial import cKDTree


def compute_f_score(pc1: np.ndarray, pc2: np.ndarray, thresholds=(0.1, 0.2, 0.5)) -> dict:
    """Computes F-Score at thresholds tau in {0.1, 0.2, 0.5} on unit-sphere point clouds."""
    p1 = np.asarray(pc1, dtype=np.float32)
    p2 = np.asarray(pc2, dtype=np.float32)
    tree1 = cKDTree(p1)
    tree2 = cKDTree(p2)
    d1, _ = tree1.query(p2, k=1, workers=-1)
    d2, _ = tree2.query(p1, k=1, workers=-1)
    results = {}
    for th in thresholds:
        precision = float(np.mean(d2 < th))
        recall = float(np.mean(d1 < th))
        if precision + recall > 0:
            f_val = 2.0 * (precision * recall) / (precision + recall)
        else:
            f_val = 0.0
        results[f"f_score_{th}"] = float(f_val)
    return results
