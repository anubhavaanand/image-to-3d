from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
from PIL import Image

@dataclass
class Mesh3D:
    vertices: np.ndarray
    faces: np.ndarray
    vertex_colors: Optional[np.ndarray] = None
    uv_coords: Optional[np.ndarray] = None
    texture_map: Optional[Image.Image] = None

@dataclass
class GaussianCloud:
    positions: np.ndarray
    scales: np.ndarray
    rotations: np.ndarray
    opacities: np.ndarray
    sh_coeffs: np.ndarray

@dataclass
class ReconstructionResult:
    mesh: Optional[Mesh3D] = None
    gaussians: Optional[GaussianCloud] = None
    metadata: Optional[Dict[str, Any]] = None

class AbstractEngine(ABC):
    @abstractmethod
    def load_model(self) -> None:
        pass
        
    @abstractmethod
    def predict(self, image: Image.Image) -> ReconstructionResult:
        pass
        
    @abstractmethod
    def get_engine_info(self) -> Dict[str, Any]:
        pass
