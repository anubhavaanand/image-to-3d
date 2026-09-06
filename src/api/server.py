from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
import io
import os
import uuid
from typing import List

from ..config import EngineType
from ..engines.triposr_engine import TriposrEngine
from ..engines.lgm_engine import LGMEngine
from ..engines.trellis_engine import TrellisEngine
from ..export.glb_exporter import export_glb

app = FastAPI(title="Image-to-3D Benchmark API")

# Initialize engines
engines = {
    EngineType.TRIPOSR: TriposrEngine(),
    EngineType.LGM: LGMEngine(),
    EngineType.TRELLIS: TrellisEngine()
}

OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.get("/health")
def health_check():
    return {"status": "healthy"}
    
@app.get("/engines")
def list_engines():
    return [{"id": k, "info": v.get_engine_info()} for k, v in engines.items()]

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    engine_type: EngineType = Form(...)
):
    if engine_type not in engines:
        raise HTTPException(status_code=400, detail="Invalid engine type")
        
    engine = engines[engine_type]
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        result = engine.predict(image)
        
        out_filename = f"{uuid.uuid4().hex}.glb"
        out_path = os.path.join(OUTPUT_DIR, out_filename)
        
        if result.mesh is not None:
            export_glb(result.mesh, out_path)
            return FileResponse(path=out_path, filename=out_filename, media_type="model/gltf-binary")
        else:
            # Handle Gaussians
            return {"message": "Model generated Gaussians, which are currently only supported via PLY export locally.", "metadata": result.metadata}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
