# AGENTS.md — Authoritative Project Context for AI Agents

## 1. Project Identity
- **Repository:** `image-to-3d-benchmark` (`https://github.com/anubhavaanand/image-to-3d`)
- **Author:** Anubhav Anand (Amity University Uttar Pradesh)
- **Workspace Root:** `/home/anubhavanand/Desktop/reasearch papers/`
- **Repo Root:** `/home/anubhavanand/Desktop/reasearch papers/image-to-3d-benchmark/`
- **Paper Location:** `/home/anubhavanand/Desktop/reasearch papers/paper/` (local-only by design; excluded via `.gitignore`)

## 2. Core Architecture & Evaluated Paradigms
1. **TripoSR (`src/engines/triposr_engine.py`)**: Feed-forward DINOv1 -> 16-layer Triplane Transformer -> NeRF MLP -> 256^3 Marching Cubes mesh with interior-normal multi-sample color queries.
2. **LGM (`src/engines/lgm_engine.py`)**: ImageDream 4-view diffusion -> Asymmetric U-Net -> 65,536 3D Gaussians -> foreground-filtered (`opa_all > 0.10`) 256^3 Gaussian-smoothed Marching Cubes at the 48th-percentile iso with a 7-tap Gaussian (`_k = 7, _sigma = 1.10`), coloured via k = 6 opacity-weighted core splats (`core_mask = opa_fg > 0.35`, falling back to `> 0.20` when fewer than 100 core splats survive) so halo fog never donates blue/teal.
3. **TRELLIS (`src/engines/trellis_engine.py`)**: DINOv2 -> Sparse Structure Flow (G_S) + SLat Flow (G_L) at 8 + 8 steps (CFG 7.5 / 3.0) -> 256^3 FlexiCubes decoder with built-in T4 staged CPU/GPU offload (staged_offload=True) and 3-channel vertex_attrs[:, :3] colors.
4. **SPAR3D (`src/engines/spar3d_engine.py`)**: Point diffusion (512 albedo points) -> Triplane transformer -> DMTet mesh + PBR material generation (Run 2 isolated kernel).
5. **SF3D (`src/engines/sf3d_engine.py`)**: DINOv2 + LRM-style transformer -> DMTet mesh (Paradigm-1 ablation: DMTet vs Marching Cubes).

## 3. Locked Evaluation Protocol (`src/evaluation/benchmark_runner.py`)
- **Preprocessing (`src/preprocessing/background_removal.py`)**: Segments foreground via `rembg` (`isnet-general-use`) and returns a true `RGBA` image (alpha=0 on background, alpha=255 on foreground) so each engine's native foreground cropper (`resize_foreground`, `kiui.op.recenter`, `preprocess_image`) and background compositor (0.5 grey for TripoSR, white for LGM/TRELLIS) operate accurately.
- **3D Geometry Metrics**: 10,000 surface points sampled uniformly, centered at centroid, normalized to the unit sphere (99.5th-percentile radius). Aligned via 72 candidate 3D rotations (6 principal axis-up orientations x 12 yaw steps) + 3-iteration Kabsch ICP under **symmetric mean Chamfer Distance** (CD = mean(d1) + mean(d2)) and F-Score@{0.1, 0.2, 0.5}.
- **2D Perceptual Metrics (PSNR & LPIPS)**: The optimal 3D alignment transform (R, t) is applied directly to `pred_mesh.vertices` before rendering. Both predicted and GT meshes are rendered at 65-degree azimuth / 15-degree elevation, 2x supersampled (512 -> 256 Lanczos), white background, with three-point studio lighting (ambient 0.45, key 0.85, fill 0.35, rim 0.25, distance 2.45, yfov 49.1) and linear-to-sRGB encoding.
- **Disk Hygiene**: the final runs use `BenchmarkRunner(save_mesh_limit=100)`, exporting raw `*_mesh.obj` for every object, while PNG renders and JSON metrics are written for every evaluated sample. Budget disk accordingly at 100 objects x 4 engines.
