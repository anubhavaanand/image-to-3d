# AGENTS.md — Authoritative Project Context for AI Agents

## 1. Project Identity
- **Repository:** `image-to-3d-benchmark` (`https://github.com/anubhavaanand/image-to-3d`)
- **Author:** Anubhav Anand (Amity University Uttar Pradesh)
- **Workspace Root:** `/home/anubhavanand/Desktop/reasearch papers/`
- **Repo Root:** `/home/anubhavanand/Desktop/reasearch papers/image-to-3d-benchmark/`
- **Paper Location:** `/home/anubhavanand/Desktop/reasearch papers/paper/` (local-only by design; excluded via `.gitignore`)

## 2. Core Architecture & Evaluated Paradigms
1. **TripoSR (`src/engines/triposr_engine.py`)**: Feed-forward DINOv1 -> 16-layer Triplane Transformer -> NeRF MLP -> 256^3 Marching Cubes mesh with interior-normal multi-sample color queries.
2. **LGM (`src/engines/lgm_engine.py`)**: ImageDream 4-view diffusion -> Asymmetric U-Net -> 65,536 3D Gaussians -> foreground-filtered (alpha > 0.05) 256^3 Gaussian-smoothed Marching Cubes + k=4 inverse-distance 3-channel RGB vertex coloring.
3. **TRELLIS (`src/engines/trellis_engine.py`)**: DINOv2 -> Sparse Structure Flow (G_S) + SLat Flow (G_L) -> 256^3 FlexiCubes decoder with built-in T4 staged CPU/GPU offload (staged_offload=True) and 3-channel vertex_attrs[:, :3] colors.
4. **SPAR3D (`src/engines/spar3d_engine.py`)**: Point diffusion (512 albedo points) -> Triplane transformer -> DMTet mesh + PBR material generation (Run 2 isolated kernel).
5. **SF3D (`src/engines/sf3d_engine.py`)**: DINOv2 + LRM-style transformer -> DMTet mesh (Paradigm-1 ablation: DMTet vs Marching Cubes).

## 3. Locked Evaluation Protocol (`src/evaluation/benchmark_runner.py`)
- **Preprocessing (`src/preprocessing/background_removal.py`)**: Segments foreground via `rembg` (`isnet-general-use`) and returns a true `RGBA` image (alpha=0 on background, alpha=255 on foreground) so each engine's native foreground cropper (`resize_foreground`, `kiui.op.recenter`, `preprocess_image`) and background compositor (0.5 grey for TripoSR, white for LGM/TRELLIS) operate accurately.
- **3D Geometry Metrics**: 10,000 surface points sampled uniformly, centered at centroid, normalized to the unit sphere (99.5th-percentile radius). Aligned via 36 candidate 3D rotations (3 principal axis-up orientations x 12 yaw steps) + 3-iteration Kabsch ICP under **symmetric mean Chamfer Distance** (CD = mean(d1) + mean(d2)) and F-Score@{0.1, 0.2, 0.5}.
- **2D Perceptual Metrics (PSNR & LPIPS)**: The optimal 3D alignment transform (R, t) is applied directly to `pred_mesh.vertices` before rendering. Both predicted and GT meshes are rendered at 45-degree azimuth, 2x supersampled (512 -> 256 Lanczos), white background, `ambient_light=[0.65, 0.65, 0.65]`, and directional key light `intensity=1.8`.
- **Disk Hygiene**: Raw `*_mesh.obj` exports are capped at `save_mesh_limit=10` (idx < 10) while PNG renders and JSON metrics are saved for all 100 objects.
