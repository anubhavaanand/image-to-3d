from enum import Enum
from typing import Tuple, List
from pydantic import BaseModel, Field

class EngineType(str, Enum):
    TRIPOSR = "TRIPOSR"
    LGM = "LGM"
    TRELLIS = "TRELLIS"

class PreprocessingConfig(BaseModel):
    background_removal_model: str = Field(default="birefnet-general")
    target_size: Tuple[int, int] = Field(default=(512, 512))

class EngineConfig(BaseModel):
    engine_type: EngineType = Field(default=EngineType.TRIPOSR)
    device: str = Field(default="cuda")
    dtype: str = Field(default="float16")
    checkpoint_path: str | None = Field(default=None)

class ExportConfig(BaseModel):
    format: str = Field(default="glb", pattern="^(glb|ply|obj)$")
    output_dir: str = Field(default="./outputs")

class BenchmarkConfig(BaseModel):
    dataset_path: str = Field(default="./data")
    metrics: List[str] = Field(default_factory=lambda: ["chamfer_distance", "f_score", "psnr", "lpips"])
    num_samples: int = Field(default=100)

class AppConfig(BaseModel):
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
