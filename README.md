# Image-to-3D Benchmark Pipeline

A complete, production-ready benchmarking pipeline for evaluating state-of-the-art Image-to-3D generation models.

## Architecture

```text
+---------------+      +-------------------+      +-------------------+
|               |      |                   |      |                   |
|  Input Image  | +--> |  Preprocessing    | +--> |    3D Engines     |
|               |      |  (Background      |      |  - TripoSR        |
+---------------+      |   Removal)        |      |  - LGM            |
                       |                   |      |  - TRELLIS        |
                       +-------------------+      +--------+----------+
                                                           |
                                                           v
                                                  +-------------------+
                                                  |                   |
                                                  |    Evaluation     |
                                                  |  - Chamfer Dist.  |
                                                  |  - F-Score        |
                                                  |  - PSNR / LPIPS   |
                                                  |                   |
                                                  +--------+----------+
                                                           |
                                                           v
                                                  +-------------------+
                                                  |                   |
                                                  |    Results /      |
                                                  |    Export         |
                                                  |  - GLB/PLY        |
                                                  |  - LaTeX Table    |
                                                  +-------------------+
```

## Quick Start

1. Install requirements:
   ```bash
   pip install -e .
   ```

2. Run the API Server:
   ```bash
   uvicorn src.api.server:app --reload
   ```

## API Usage

List available engines:
```bash
curl http://localhost:8000/engines
```

Run prediction:
```bash
curl -X POST -F "file=@sample.png" -F "engine_type=TRIPOSR" http://localhost:8000/predict -o result.glb
```

## Citation

```bibtex
@misc{image23d_benchmark_2023,
  title={Image-to-3D Benchmark Pipeline},
  author={Anand, Anubhav},
  year={2023}
}
```
