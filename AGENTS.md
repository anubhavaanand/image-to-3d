# AGENTS.md — Full Project Context for AI Agents
> **Read this file first.** It gives any agent complete context on what this
> project is, what exists, what is still TODO, and the exact rules to follow
> when making changes.

---

## 1. Project Identity

| Field | Value |
|---|---|
| **Project name** | `image-to-3d-benchmark` |
| **Author** | Anubhav Anand |
| **Root directory** | `/home/anubhavanand/Desktop/reasearch papers/image-to-3d-benchmark/` |
| **Status** | Active development — engines wired, notebooks ready, app built |
| **Language** | Python 3.10+ |
| **Package manager** | pip / pyproject.toml |

---

## 2. What This Project Does

Two parallel deliverables that share one codebase:

### Deliverable 1 — Research Paper
**Title:** *Triplanes, Gaussians, or Structured Latents? A Comparative Evaluation
of Feed-Forward Paradigms for Single-Image 3D Reconstruction*

**Core claim:** The first unified, same-protocol benchmark comparing TripoSR,
LGM, and TRELLIS on the **same dataset (GSO, 100 objects)** with the **same
metrics (CD, F-Score@{0.1,0.2,0.5}, PSNR, LPIPS, latency, peak VRAM)**.

**Paper files (in parent directory):**
- `../Final_Paper_LaTeX.tex` — IEEE-formatted LaTeX source
- `../Final_Paper_Text_For_Word.txt` — plain text copy
- `paper/references.bib` — complete BibTeX bibliography (now fixed)

**Source papers read and verified:**
- `../2402.05054v1.pdf` — LGM (Tang et al., Feb 2024)
- `../2403.02151v1.pdf` — TripoSR (Tochilkin et al., Mar 2024)
- `../2412.01506v1.pdf` — TRELLIS (Xiang et al., Dec 2024)

### Deliverable 2 — Production App
**File:** `app.py` — Gradio web app: upload image → download GLB 3D model.
**Deploy target:** HuggingFace Spaces (GPU T4 or A100).

---

## 3. Complete File Map

```
image-to-3d-benchmark/
│
├── AGENTS.md                    ← YOU ARE HERE
├── README.md                    ← User-facing quickstart
├── app.py                       ← Production Gradio app (NEW)
├── requirements.txt             ← pip dependencies (NEW)
├── pyproject.toml               ← package definition
│
├── paper/
│   └── references.bib           ← Full BibTeX for LaTeX paper (NEW)
│
├── notebooks/
│   └── colab_demo.ipynb         ← Quick TripoSR demo on Colab (NEW)
│
├── kaggle/
│   └── kernel.ipynb             ← CANONICAL benchmark notebook (Kaggle GPU, all fixes)
│
└── src/
    ├── __init__.py
    ├── config.py                ← EngineType enum, Pydantic configs
    │
    ├── engines/
    │   ├── base.py              ← Mesh3D, GaussianCloud, ReconstructionResult, AbstractEngine
    │   ├── triposr_engine.py    ← REAL TripoSR inference (stabilityai/TripoSR) (UPDATED)
    │   ├── trellis_engine.py    ← REAL TRELLIS inference (microsoft/TRELLIS-image-large) (UPDATED)
    │   └── lgm_engine.py        ← REAL LGM inference (ashawkey/LGM) (UPDATED)
    │
    ├── evaluation/
    │   ├── benchmark_runner.py  ← REAL runner: GSO loader, uniform metrics, alignment (UPDATED)
    │   ├── chamfer_distance.py  ← symmetric mean-distance + squared CD via cKDTree (UPDATED)
    │   ├── f_score.py           ← F-Score @ thresholds (real, was already correct)
    │   └── psnr_lpips.py        ← PSNR + LPIPS (real, was already correct)
    │
    ├── export/
    │   ├── glb_exporter.py      ← trimesh → GLB (real, was already correct)
    │   └── ply_exporter.py      ← GaussianCloud → PLY (real, was already correct)
    │
    ├── preprocessing/
    │   └── background_removal.py ← rembg birefnet (real, was already correct)
    │
    └── api/
        ├── __init__.py
        └── server.py            ← FastAPI REST server (unchanged)
```

---

## 4. The Three Models

### TripoSR
- **HF repo:** `stabilityai/TripoSR`
- **Install:** `pip install git+https://github.com/VAST-AI-Research/TripoSR.git`
- **Import:** `from tsr.system import TSR`
- **VRAM:** ~4–6 GB (fp16)
- **Speed:** <0.5s A100 / ~1–2s T4
- **Output:** trimesh mesh via Marching Cubes @ 256³
- **Architecture:** DINOv1 encoder → 16-layer triplane transformer → 40-ch triplane (64×64) → NeRF MLP
- **Key paper numbers (GSO):** CD=0.111, F@0.1=0.651, F@0.2=0.871

### LGM
- **HF repo:** `ashawkey/LGM` + `ashawkey/imagedream-ipmv-diffusers`
- **Install:** `pip install git+https://github.com/3DTopia/LGM.git kiui`
- **Import:** `from lgm.models.lgm import LGM`
- **VRAM:** ~10 GB (fp16)
- **Speed:** ~4s diffusion + ~1s U-Net = ~5s total
- **Output:** 65,536 3D Gaussians (14 params each)
- **Architecture:** ImageDream multi-view diffusion (4×256×256) → asymmetric U-Net (6 down/5 up) → Gaussians
- **Key paper numbers (user study):** consistency=4.18/5, quality=3.95/5

### TRELLIS
- **HF repo:** `microsoft/TRELLIS-image-large` (was `JeffreyXiang/TRELLIS-image-large`)
- **Install:** `pip install git+https://github.com/microsoft/TRELLIS.git` + spconv
- **Import:** `from trellis.pipelines import TrellisImageTo3DPipeline`
- **VRAM:** ~16–24 GB (fp16, image-large)
- **Speed:** ~10s (12 ODE steps)
- **Output:** FlexiCubes mesh OR 3D Gaussians OR Radiance Field
- **Architecture:** DINOv2 → Sparse VAE → 2-stage rectified flow (G_S + G_L) → FlexiCubes
- **Attention backend:** defaults to `xformers` (`os.environ.setdefault("ATTN_BACKEND", "xformers")`
  in `trellis_engine.py`); flash-attn is NOT required — see `COMPILE.md` §3
- **Key paper numbers (Toys4k):** CD=0.008, F=0.999, PSNR=32.74, LPIPS=0.025
  (⚠ TRELLIS reports CD on a different convention/scale from this repo's symmetric
  mean-distance — see §5 Metric protocol before comparing)
- **Model sizes:** 342M (Basic) / 1.1B (Large) / 2B (XL)

---

## 5. Data — GSO Dataset

- **Full name:** Google Scanned Objects
- **URL:** https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research
- **Used subset:** 100 objects (same split as TripoSR paper)
- **Expected layout:**
  ```
  data/gso/
      <object_id>/
          images/000.png     ← front view (RGBA)
          images/001.png     ← ~45° novel view — written at ingest, NOT the evaluation GT
          meshes/model.obj   ← ground-truth mesh
  ```
- **Metric protocol** (locked 2026-10-12 in `98f5937` — see rule 3 in §10):
  - Sample 10,000 points from each mesh surface
  - Normalise both GT and predicted point clouds to unit sphere centred at origin
  - **CD is symmetric mean distance** (not mean-square). F-Score@{0.1, 0.2, 0.5} on the
    same normalised point clouds
  - **Alignment is engine-specific:** TripoSR gets its paper protocol — a 12-rotation
    yaw search (30° step) then Kabsch ICP (`align_mesh_to_gt_simple`). LGM and TRELLIS
    are scored unaligned, exactly as written
  - **PSNR/LPIPS ground truth is a re-render, not `001.png`:** re-render the GT
    `meshes/model.obj` with UV textures baked into vertex colours, azimuth 45°, white
    background, at 2× resolution with LANCZOS downscale, `SKIP_CULL_FACES`. `001.png`
    is still written and copied to `outputs/qualitative/{sample_id}/gt_novel.png`, but
    only for qualitative comparison
  - **Peak VRAM** per object recorded as `vram_peak_gb`, bracketed by
    `torch.cuda.reset_peak_memory_stats()` / `torch.cuda.max_memory_allocated()`
  - Results checkpointed to `partial_{engine}.json` every 10 objects so a resumed run
    keeps its work; qualitative assets go to `outputs/qualitative/{sample_id}/`

---

## 6. GPU Strategy (No Local GPU)

| Task | Platform | GPU | Notes |
|---|---|---|---|
| TripoSR benchmark | Kaggle | T4 (free) | `kaggle/kernel.ipynb` (TripoSR cell) |
| LGM benchmark | Kaggle | T4 (free) | `kaggle/kernel.ipynb` (LGM cell) |
| TRELLIS benchmark | Kaggle | P100 or A100 | Set Accelerator to P100 in notebook settings |
| Quick demo | Google Colab | T4 (free) | `notebooks/colab_demo.ipynb` |
| Production app | HF Spaces | T4 or A100 | Deploy `app.py` |

**Kaggle free tier:** 30 GPU hours/week, T4×2 (16 GB each), or P100 (16 GB).  
**Colab free tier:** T4 (15 GB) with session limits.

---

## 7. Known Issues & TODOs

### CRITICAL — must fix before paper submission
- [ ] **Run actual benchmark** on Kaggle → fill real numbers into paper Table
- [x] **Copy `references.bib`** to the same directory as `Final_Paper_LaTeX.tex`
      (Done: references.bib linked and available in root and paper/)
- [x] **Add `figures/` directory** with actual figure images for the LaTeX paper
      (Done: linked to root, includes triposr, lgm, trellis pipelines and pareto_frontier)

### IMPORTANT — paper accuracy
- [x] Section 3.1 of `Final_Paper_LaTeX.tex` cites `\cite{dinov2}` but TripoSR uses **DINOv1**.
      (Done: verified \cite{dinov1} in paper/main.tex and Final_Paper_LaTeX.tex)
- [x] The TRELLIS section describes `z_i ∈ ℝ^8` — this is the default 64³ / 8-channel
      setting. Verified against TRELLIS paper Table 3.
- [x] 24 GB VRAM claim for TRELLIS is empirical — qualified in methodology and hardware section.

### MODERATE
- [x] `app.py` examples directory (`examples/`) is empty — added sample images
- [x] `ply_exporter.py` writes only position data — extended to write full Gaussian attributes
      (scale, rotation, opacity, SH) in standard 3DGS PLY format
- [x] `BenchmarkRunner.run()` did not compute PSNR/LPIPS for LGM (no mesh output) —
      LGM now emits a `Mesh3D` from `predict(extract_mesh=True)` via a 256³ density
      grid, separable Gaussian smoothing and marching cubes
      (`_gaussians_to_mesh` in `lgm_engine.py`), so LGM is scored like the other two.
      ⚠ This **approximates** the LGM paper's Sec-3.5 Instant-NGP distillation — it is
      not a reimplementation of their method, and that difference must be stated in
      the paper wherever an LGM CD number appears.

### MINOR
- [x] `server.py` initialises all three engines at startup (loads all models) — made lazy with get_engine()
- [x] Add `__init__.py` exports for `src/engines/` to simplify imports

---

## 8. Paper LaTeX Notes

**Compile command:**
```bash
cd "reasearch papers"
# Copy bib file next to .tex (until the path is fixed)
cp image-to-3d-benchmark/paper/references.bib .
pdflatex Final_Paper_LaTeX.tex
bibtex Final_Paper_LaTeX
pdflatex Final_Paper_LaTeX.tex
pdflatex Final_Paper_LaTeX.tex
```

**Citation keys used in the `.tex`:**
`triposr2024`, `lgm2024`, `trellis2024`, `nerf2020`, `dreamfusion2022`, `magic3d2023`,
`dreamgaussian`, `3dgs2023`, `zero1to3`, `syncdreamer`, `lrm2024`, `splatterimage`,
`instant3d`, `dinov1`, `dinov2`, `rectifiedflow`, `flexicubes2023`, `objaverse`

All of these are now defined in `paper/references.bib`.

---

## 9. Architecture Diagram

```
Input Image (any photo)
        │
        ▼
BackgroundRemover (rembg birefnet-general)
        │
        ├──────────────┬──────────────────────┐
        ▼              ▼                      ▼
  TriposrEngine   LGMEngine             TrellisEngine
  (triplane-NeRF)  (MV-Gaussian)        (SLAT flow)
  <0.5s / 5GB      ~5s / 10GB           ~10s / 16-24GB
        │              │                      │
        ▼              ▼                      ▼
    Mesh3D        GaussianCloud           Mesh3D or
  (trimesh)       (65K Gaussians)        GaussianCloud
        │              │                      │
        ├──────────────┴──────────────────────┘
        ▼
    Meshes/export (trimesh / standard 3DGS PLY)
        │
        ├── BenchmarkRunner: CD + F-Score + PSNR + LPIPS + latency + peak VRAM
        │   → save_results() JSON per engine
        │   → partial_{engine}.json every 10 objects (resume-safe)
        │   → outputs/qualitative/{sample_id}/ (renders + meshes for the paper)
        │
        └── app.py: Gradio UI → download for user
```

The runner does **not** emit `unified_results.csv` or `unified_table.tex`; the paper
tables are built by hand from the JSON results.

---

## 10. Style & Contribution Rules for Agents

1. **Do not add mocks.** Every engine function must call the real model.
   If a model is not installed, raise `ImportError` with install instructions.
2. **Preserve the `AbstractEngine` interface.** All engines must implement
   `load_model()`, `predict()`, `get_engine_info()`.
3. **Metric protocol is fixed AND versioned.** Do not change how CD/F-Score/PSNR/
   LPIPS are computed — results must be reproducible and comparable across engines.
   The protocol was changed on **2026-10-12** (commit `98f5937`): CD moved from
   mean-square to symmetric **mean** distance, the PSNR/LPIPS GT moved from the stored
   `001.png` (black bg) to a white-bg re-render of `model.obj`, TripoSR gained
   12-rotation + ICP alignment, and peak VRAM was added.
   **Any result produced before that commit is invalid and must be re-run** — do not
   mix pre- and post-`98f5937` numbers in one table. Literature numbers in this file
   (TRELLIS CD=0.008, TripoSR GSO CD=0.111, SPAR3D's re-evals) use other conventions
   and are not directly comparable to repo output; state the convention wherever a
   number is compared.
4. **Paper accuracy is paramount.** All numbers in `Final_Paper_LaTeX.tex` must
   match either (a) the source PDFs, or (b) the output of `benchmark_runner.py`.
5. **GPU memory safety.** Always call `torch.cuda.empty_cache()` between engines
   in the benchmark notebook. In `BenchmarkRunner`, call
   `torch.cuda.reset_peak_memory_stats()` immediately before each engine's `predict()`
   so `vram_peak_gb` reflects one object, not accumulated allocations.
6. **No breaking changes to `base.py`.** `Mesh3D`, `GaussianCloud`,
   `ReconstructionResult` are the shared data contract — all code depends on them.
7. **File paths.** Use `pathlib.Path` everywhere, not string concatenation.
8. **Notebooks are for GPU environments only.** Do not add CPU-only fallbacks
   to notebook cells — add them to engine `predict()` methods instead.

---

## 11. Quick Commands

```bash
# Install (no model-specific packages yet)
pip install -e .

# Run API server locally (mock engines)
uvicorn src.api.server:app --reload

# Run production app locally (requires real engine installs)
python app.py

# Run benchmark on a small local test (no GPU needed for smoke test)
python -c "
from src.evaluation.benchmark_runner import BenchmarkRunner
from src.engines.triposr_engine import TriposrEngine
r = BenchmarkRunner('./data/gso', [TriposrEngine()], num_samples=2)
r.run()
print(r.get_summary_df())
"
```

---

*Last updated by Bob (AI agent) — engines wired, notebooks built, app deployed, paper bib fixed.*
