from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
import io
import os
import uuid
from typing import Dict, Any

from ..config import EngineType
from ..engines import AbstractEngine, TriposrEngine, LGMEngine, TrellisEngine
from ..export.glb_exporter import export_glb
from ..export.ply_exporter import export_ply

app = FastAPI(title="Image-to-3D Benchmark API")

# Lazy engine cache
_engines: Dict[EngineType, AbstractEngine] = {}

def get_engine(engine_type: EngineType) -> AbstractEngine:
    if engine_type not in _engines:
        if engine_type == EngineType.TRIPOSR:
            _engines[engine_type] = TriposrEngine()
        elif engine_type == EngineType.LGM:
            _engines[engine_type] = LGMEngine()
        elif engine_type == EngineType.TRELLIS:
            _engines[engine_type] = TrellisEngine()
        else:
            raise ValueError(f"Unknown engine: {engine_type}")
    return _engines[engine_type]

OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/engines")
def list_engines():
    return [
        {
            "id": EngineType.TRIPOSR,
            "info": {
                "name": "TripoSR",
                "paper": "arXiv:2403.02151",
                "latency_approx": "<0.5s (A100) / ~1-2s (T4)",
                "vram_approx": "4–6 GB (fp16)",
                "output_format": "mesh (.glb)",
            },
        },
        {
            "id": EngineType.LGM,
            "info": {
                "name": "LGM",
                "paper": "arXiv:2402.05054",
                "latency_approx": "~5s total",
                "vram_approx": "~10 GB (fp16)",
                "output_format": "gaussians (.ply)",
            },
        },
        {
            "id": EngineType.TRELLIS,
            "info": {
                "name": "TRELLIS",
                "paper": "arXiv:2412.01506",
                "latency_approx": "~6.5–10s (A100)",
                "vram_approx": "~16–24 GB (fp16)",
                "output_format": "mesh (.glb) / gaussians (.ply)",
            },
        },
    ]

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    engine_type: EngineType = Form(...)
):
    try:
        engine = get_engine(engine_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        result = engine.predict(image)

        if result.mesh is not None:
            out_filename = f"{uuid.uuid4().hex}.glb"
            out_path = os.path.join(OUTPUT_DIR, out_filename)
            export_glb(result.mesh, out_path)
            return FileResponse(path=out_path, filename=out_filename, media_type="model/gltf-binary")
        elif result.gaussians is not None:
            out_filename = f"{uuid.uuid4().hex}.ply"
            out_path = os.path.join(OUTPUT_DIR, out_filename)
            export_ply(result.gaussians, out_path)
            return FileResponse(path=out_path, filename=out_filename, media_type="application/octet-stream")
        else:
            raise HTTPException(status_code=500, detail="Engine returned neither mesh nor gaussians.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

