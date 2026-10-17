"""Engines module with lazy imports to avoid heavyweight engine loads at import time."""

from .base import AbstractEngine, Mesh3D, GaussianCloud, ReconstructionResult

__all__ = [
    "AbstractEngine",
    "Mesh3D",
    "GaussianCloud",
    "ReconstructionResult",
    "TriposrEngine",
    "LGMEngine",
    "TrellisEngine",
    "Spar3dEngine",
    "SF3DEngine",
]

def __getattr__(name):
    if name == "TriposrEngine":
        from .triposr_engine import TriposrEngine
        return TriposrEngine
    if name == "LGMEngine":
        from .lgm_engine import LGMEngine
        return LGMEngine
    if name == "TrellisEngine":
        from .trellis_engine import TrellisEngine
        return TrellisEngine
    if name == "Spar3dEngine":
        from .spar3d_engine import Spar3dEngine
        return Spar3dEngine
    if name == "SF3DEngine":
        from .sf3d_engine import SF3DEngine
        return SF3DEngine
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
