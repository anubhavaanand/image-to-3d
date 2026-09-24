from __future__ import annotations

import numpy as np
from pathlib import Path
from ..engines.base import GaussianCloud


def export_ply(gaussians: GaussianCloud, filepath: str) -> None:
    """
    Exports a GaussianCloud to a standard 3DGS PLY format.
    Compatible with 3DGS visualizers (e.g. SuperSplat, MeshLab, WebGL 3DGS).
    """
    if gaussians is None:
        raise ValueError("GaussianCloud is None, cannot export.")

    pos = np.asarray(gaussians.positions, dtype=np.float32)
    n = len(pos)

    # Scale attributes
    if gaussians.scales is not None:
        scales = np.asarray(gaussians.scales, dtype=np.float32)
        if scales.ndim == 1:
            scales = np.tile(scales[:, None], (1, 3))
    else:
        scales = np.zeros((n, 3), dtype=np.float32)

    # Rotation attributes (quaternion)
    if gaussians.rotations is not None:
        rot = np.asarray(gaussians.rotations, dtype=np.float32)
    else:
        rot = np.zeros((n, 4), dtype=np.float32)

    # Opacity
    if gaussians.opacities is not None:
        opacities = np.asarray(gaussians.opacities, dtype=np.float32)
        if opacities.ndim == 1:
            opacities = opacities[:, None]
    else:
        opacities = np.ones((n, 1), dtype=np.float32)

    # SH DC colors
    if gaussians.sh_coeffs is not None:
        sh = np.asarray(gaussians.sh_coeffs, dtype=np.float32)
        if sh.ndim == 3:
            sh = sh[:, 0, :3]
    else:
        sh = np.zeros((n, 3), dtype=np.float32)

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "end_header\n",
    ]

    with open(filepath, "w") as f:
        f.write("\n".join(header))
        for i in range(n):
            f.write(
                f"{pos[i, 0]:.6f} {pos[i, 1]:.6f} {pos[i, 2]:.6f} "
                f"0.0 0.0 0.0 "
                f"{sh[i, 0]:.6f} {sh[i, 1]:.6f} {sh[i, 2]:.6f} "
                f"{opacities[i, 0]:.6f} "
                f"{scales[i, 0]:.6f} {scales[i, 1]:.6f} {scales[i, 2]:.6f} "
                f"{rot[i, 0]:.6f} {rot[i, 1]:.6f} {rot[i, 2]:.6f} {rot[i, 3]:.6f}\n"
            )

