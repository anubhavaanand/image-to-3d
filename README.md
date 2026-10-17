# Image-to-3D Unified Benchmark

A standardized, ground-truth-referenced benchmark evaluating feed-forward single-image 3D reconstruction paradigms on Google Scanned Objects (GSO):

- **TripoSR** (*Triplane NeRF*, `arXiv:2403.02151`)
- **LGM** (*Multi-View 3D Gaussian Splatting*, `arXiv:2402.05054`)
- **TRELLIS** (*Structured 3D Latents + Rectified Flow*, `arXiv:2412.01506`)
- **SPAR3D** (*Point-Aware Diffusion + DMTet PBR Mesh*, `arXiv:2501.04689`)

## Repository Structure

```text
image-to-3d-benchmark/
├── AGENTS.md                  # Full technical context & protocol specification
├── PROJECT_STATUS.md          # Current benchmark status & run log
├── app.py                     # Gradio interactive web application
├── handpicked.txt             # Curated 100-object GSO evaluation catalog
├── kaggle/
│   └── kernel.ipynb           # Canonical Kaggle T4 benchmark notebook
└── src/
    ├── config.py              # Pydantic configuration schemas
    ├── preprocessing/
    │   └── background_removal.py
    ├── engines/
    │   ├── __init__.py
    │   ├── base.py
    │   ├── triposr_engine.py
    │   ├── lgm_engine.py
    │   ├── trellis_engine.py
    │   ├── spar3d_engine.py
    │   └── sf3d_engine.py
    ├── evaluation/
    │   ├── __init__.py
    │   ├── benchmark_runner.py
    │   ├── chamfer_distance.py
    │   ├── f_score.py
    │   └── psnr_lpips.py
    ├── export/
    │   ├── __init__.py
    │   ├── glb_exporter.py
    │   └── ply_exporter.py
    └── api/
        ├── __init__.py
        └── server.py
```

## Installation

```bash
pip install -e .
pip install pydantic numpy scipy pandas torch pyrender trimesh rembg
```

## Usage

```bash
python -c "
from src.engines.triposr_engine import TriposrEngine
from src.evaluation.benchmark_runner import BenchmarkRunner

runner = BenchmarkRunner(
    dataset_path='./data/gso',
    engines=[TriposrEngine()],
    num_samples=10,
)
runner.run()
print(runner.get_summary_df())
"
```

## License

This benchmark evaluates models with non-commercial licenses (SF3D, LGM).
See each model card for licensing details.
