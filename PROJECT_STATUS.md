# PROJECT STATUS & HANDOFF (Updated 2026-10-17)

## 1. Current Milestone Summary
- **v54 Smoke Run (Oct 16, 2026)**: Completed 10/10 objects across all 3 core engines (TripoSR, LGM, TRELLIS) on a single 15.6 GB Tesla T4 with zero OOMs (analysis/v54_smoke_results/).
- **Post-v54 Forensic Audit & Codebase Rewrite (Oct 17, 2026)**:
  1. Fixed BackgroundRemover.remove() wiping out the RGBA alpha mask (which previously prevented TripoSR and LGM from cropping/centering the foreground and forced TripoSR onto a white background instead of its training 0.5 mid-grey background).
  2. Fixed LGM mesh extraction (alpha > 0.05 foreground filtering before bounding-box normalization, sigma=1.5 smoothing, 35th-percentile Marching Cubes iso-level, tm.split() floater removal, and k=4 inverse-distance 3-channel RGB coloring).
  3. Fixed TripoSR vertex coloring via multi-depth inward-normal triplane NeRF queries.
  4. Unified 36-pose + 3-step Kabsch ICP alignment under symmetric mean CD for all engines, rotating pred_mesh.vertices prior to novel-view rendering.
  5. Integrated staged_offload=True directly into TrellisEngine and added Spar3dEngine (src/engines/spar3d_engine.py).
  6. Cleaned workspace clutter into _archive_cleanup/ and synchronized kaggle/kernel.ipynb with kaggle_push/kernel.ipynb.

## 2. Next Execution Steps
- **Run 1 (Core 3 Engines — TripoSR, LGM, TRELLIS)**: Push kaggle_push/kernel.ipynb (with rebuilt image-to-3d-benchmark-code.tar.gz) to verify the vibrant colored renders on 10 objects, then run all 100 curated objects in handpicked.txt.
- **Run 2 (4th Engine — SPAR3D)**: Run Spar3dEngine in an isolated second kernel (setuptools==69.5.1 + uv_unwrapper) and merge spar3d_results.json.
