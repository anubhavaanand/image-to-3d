# IEEE Conference Paper: Development Plan

## 1. Paper Overview & Objectives
**Working Title:** "Triplanes, Gaussians, or Structured Latents? A Comparative Evaluation of Feed-Forward Paradigms for Single-Image 3D Reconstruction"
**Target:** Standard IEEE Conference (e.g., CVPR, ICCV, or ICRA formatting)
**Format:** 2-Column, 10pt font, standard IEEE `\documentclass[conference]{IEEEtran}`

**Core Objective:** Present the first unified, empirical comparison of the three leading feed-forward 3D generative architectures (TripoSR, LGM, TRELLIS) under a strictly controlled benchmark protocol, analyzing the Pareto frontier of speed (latency/VRAM) vs. quality (Chamfer Distance, F-Score).

---

## 2. Structural Outline

### I. Introduction (1 Page)
*   **Context:** The shift from slow, optimization-based methods (SDS, DreamFusion) to fast, feed-forward architectures.
*   **Problem:** Fragmented evaluation protocols in original papers make cross-architecture comparisons difficult.
*   **Contributions:** 
    1. A unified empirical benchmark on the GSO dataset.
    2. Architectural analysis of Triplanes vs. Gaussians vs. Structured Latents.
    3. Open-source release of the modular benchmarking pipeline.

### II. Related Work (0.75 Pages)
*   **Optimization-Based Methods:** NeRFs, Magic3D, DreamGaussian.
*   **Feed-Forward Paradigms:** LRM, Instant3D, Splatter Image.
*   **Foundation Models & Multi-View:** Zero-1-to-3, SyncDreamer.

### III. Methodology & Architectures (1.5 Pages)
*   *Heavy mathematical breakdown of the three paradigms:*
*   **A. Triplane Neural Radiance Fields (TripoSR):** DINOv1 ViT encoder → transformer decoder → $3 \times 64 \times 64 \times 40$ triplane → NeRF MLP → Marching Cubes.
*   **B. Multi-View Gaussian Splatting (LGM):** Asymmetric U-Net → Plücker ray embeddings → 65,536 3D Gaussians.
*   **C. Structured Latent Flow Transformers (TRELLIS):** Sparse VAE ($64^3$ voxel grid) → Rectified Flow Transformers ($G_S$ and $G_L$) → FlexiCubes extraction.

### IV. Experimental Setup & Results (1.5 Pages)
*   **Protocol:** 120 curated objects from the Google Scanned Objects (GSO) dataset. Metrics: Chamfer Distance, F-Score (@0.1, @0.2), PSNR, LPIPS. Hardware: NVIDIA T4 / A100.
*   **Quantitative Results:** The `unified_table.tex` output from our Kaggle pipeline.
*   **Computational Efficiency:** Latency vs. VRAM tradeoffs (the Pareto frontier graph).
*   **Visual Comparisons:** Side-by-side renders of failure modes (e.g., backside blurring, baked-in lighting).

### V. Discussion & Conclusion (0.5 Pages)
*   **Trade-offs:** When to use which paradigm (e.g., TripoSR for real-time edge devices, TRELLIS for high-fidelity offline assets).
*   **Future Work:** Unifying flow-matching with triplane structures.

---

## 3. Next Steps & Execution Plan

1.  **Extract Kaggle Output:** Wait for the V2 Kaggle run to complete and extract `unified_table.tex` and `pareto_frontier.png`.
2.  **Integrate Results:** Inject the empirical data into Section IV of `paper/main.tex`.
3.  **Incorporate Professor's Feedback:** Review your professor's specific requests (additions/removals) and restructure the narrative focus accordingly.
4.  **Drafting & Compilation:** Let the Agent draft the deep-dive mathematical sections (Section III) and compile the final PDF using `pdflatex` to ensure zero compilation errors and perfect column balancing.
