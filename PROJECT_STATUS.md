# PROJECT STATUS & SESSION HANDOFF
**IEEE Image-to-3D Benchmark Paper — Anubhav Anand (Amity University UP / CST-UP grant)**
Last updated: 2026-10-07. Written so any new session (or collaborator) can resume without re-reading this conversation.

---

## 1. GOAL & POSITIONING (final, agreed)

**Paper:** "A Unified Benchmark of Feed-Forward Paradigms for Single-Image 3D Reconstruction" (draft: `paper/main.tex`, compiles to 7 pages).

**Positioning (one sentence):** *The first evaluation to place the three feed-forward image-to-3D paradigms — triplane regression (TripoSR), Gaussian splatting (LGM), and structured-latent flow (TRELLIS) — under one ground-truth-referenced protocol, measuring what each reconstructs AND what it costs in time and memory.*

- Claim type: **measurement gap, not method gap**. Never claim a new model.
- Pillar 1 — protocol fragmentation PROVEN: same model + GSO dataset has CD 0.111 (TripoSR paper) → 0.145 (SPAR3D re-eval) → 0.247 (InstantMesh re-eval) = 2.2× spread; quantified in the paper's new Table `tab:protocol_fragmentation`.
- Pillar 2 — gap located: same-protocol comparisons exist in the regression line (SPAR3D, InstantMesh re-evals) but NEVER include structured-latent systems; preference platforms (3D Arena: 123K votes) rank all three models but have zero ground truth and zero efficiency metrics; NO paper anywhere reports VRAM.
- Pillar 3 — 2025 convergence framing: TripoSG, Hunyuan3D 2.x, Direct3D-S2, Sparc3D, LATTICE all adopted TRELLIS's recipe (sparse structured latents + rectified flow) → this paper is the retrospective audit of the trade-offs the dominant paradigm inherited. Claims-audit ammo: LGM measured 41 s vs ~5 s claimed; InstantMesh 36.1 s vs ~10 s claimed (SPAR3D's re-measurements, unspecified GPU).
- Explicitly NOT claimed: new architecture/dataset, human-preference superiority, cross-dataset Pareto proof.
- Genre precedent: Tatarchenko et al., CVPR 2019, "What Do Single-View 3D Reconstruction Networks Learn?"
- Venue reality: synthesis-only version → INDICON/ICACCS tier; **with own Kaggle data → ICIP/WACV/3DV plausible**.

## 2. STATE SUMMARY

| Item | Status |
|---|---|
| Read all 14 papers (3 core + 11 new) | ✅ done (facts in memory file `factcheck-triposr-lgm-trellis.md` + project memory) |
| Positioning analysis | ✅ done (this file §1) |
| `paper/main.tex` rewritten (Related Work: 2025 convergence + benchmarks subsections; protocol-fragmentation table; claims-audit paragraph; TRELLIS facts fixed: ~10 s, F 0.9999 @ r=0.05; abstract cleaned) | ✅ done |
| `paper/references.bib` rebuilt (12 new entries incl. TripoSG=ICML'25, InstantMesh=ICLR'25, TRELLIS=CVPR'25, SPAR3D=2501.**04689**; ZeroShape=CVPR'24 etc.) | ✅ done |
| Compile verified on user's machine | ✅ done — 7 pages, 34 refs resolved; one 5pt overfull table fixed; user recompiles once (`pdflatex main`) |
| Benchmark notebook fixed | ✅ `kaggle_kernel_tmp/kaggle_benchmark_fixed.ipynb` (GPU hard-check, EGL rendering, xformers backend for TRELLIS, engine-isolated cells, GSO mirror ingest + handpicked.txt support + self-rendering) |
| GSO data source verified | ✅ HF mirror `suvadityamuk/google-scanned-objects-raw` (1,046 official per-object zips; meshes+textures, no renders → notebook self-renders) |
| Object catalog for handpicking | ✅ `gso_catalog.txt` (workspace root) |
| Visual picker (local HTML gallery → handpicked.txt) | ⏳ building in background (`build_picker.py` → `gso_picker/picker.html`; log: `picker_build.log`) |
| Kaggle smoke run (10 objects) | ⏳ NEXT — standing trigger: user says "run the benchmark" |
| Full run (100 objects, 6 engines) + rebuild paper tables/Pareto from own data | ⏳ pending |
| Repo push of paper + notebook to github.com/anubhavaanand/image-to-3d | ⏳ in progress (this session) |

## 3. BENCHMARK PLAN

**6 engines on Kaggle free T4 (16 GB), all feasible:**
TripoSR (~1–2 s/obj) · SPAR3D (~0.7–1 s, 10.5 GB) · LGM (~10–20 s) · InstantMesh (~30 s, needs test) · TRELLIS @12 steps (~30–90 s, tight) · TripoSG 1.5B optional 6th.
Full 100-object run ≈ 3–4 GPU-hours + ~40 min setup vs ~30 GPU-h/week free quota. Smoke test ≈ 25 min.
Cite-only (not tested): Hunyuan3D 2.0/2.1, Direct3D-S2, Sparc3D, LATTICE.

**Runner gaps to fix BEFORE the paper-grade run** (in `image-to-3d-benchmark/src/evaluation/benchmark_runner.py`):
1. **Record peak VRAM** per object (`torch.cuda.max_memory_allocated`) — currently only latency+metrics; VRAM is a headline claim.
2. **Rotation/ICP alignment** before CD/F-Score (TripoSR's protocol: brute-force rotation + ICP) — otherwise numbers inflated vs literature.
3. **CD convention**: repo uses raw mean-squared CD; papers use 10K points ×100 scaling — pick ONE and state it in the paper.
4. **LGM mesh conversion**: LGM outputs Gaussians → v1 gives latency only; implement their Sec-3.5 conversion (Instant-NGP distill → MC) for a subset, or benchmark splat-render metrics.
5. GT novel-view vs prediction background mismatch (input white bg, evaluator renders black bg) — acceptable v1, document it.

**Kaggle driving:** API key at `~/.kaggle/kaggle.json`; CLI at `paper/.venv/bin/kaggle`. Can push kernel / poll status / pull output from any session. User's Antigravity-CLI MCP config is an alternative, not required.

**Accounts:** AWS $100 credit "doesn't work" — UNDIAGNOSED (likely AWS-Educate restrictions or 0-vCPU quota on G instances; need exact error text). Azure for Students ($100, no card) = fallback, needs GPU quota approval. CST-UP budget line: Rs 12,000 for cloud GPU.

## 4. DATASET (GSO) WORKFLOW

1. User HANDPICKS ~100 objects (criteria: no trivial primitives, no multi-object sets, ~10 categories × 10, include thin-structure hard cases; optionally flag ODOP-relevant objects for the grant).
   - Explore: `gso_catalog.txt` (1,046 names) or official scans at research.google blog / app.gazebosim.org (old collection links 404 since the Fuel reorg).
   - Visual picker being generated: `gso_picker/picker.html` (checkbox gallery, downloads handpicked.txt directly).
2. Deliver `handpicked.txt` (one object name per line): drag-drop into Kaggle notebook editor (Upload → File → lands at /kaggle/working/) OR bundle at CLI push.
3. Notebook cell 4 reads it; else seeded random sample (seed 42). Downloads zips from the HF mirror, extracts meshes, renders 000.png (input, white bg, 512) + 001.png (GT novel view, black bg, 256) with the evaluator's own camera.

## 5. FILE MAP (workspace = `~/Desktop/reasearch papers/`)

- `paper/main.tex`, `paper/references.bib`, `paper/main.pdf` — **authoritative paper** (compiles; IEEEtran.cls/.bst present)
- `kaggle_kernel_tmp/kaggle_benchmark_fixed.ipynb` — fixed benchmark notebook (import into Kaggle or push via CLI)
- `gso_catalog.txt` — 1,046-object catalog; `handpicked.txt` ← user creates this
- `build_picker.py` → `gso_picker/picker.html` + `thumbs/` + `manifest.json` — visual picker
- `image-to-3d-benchmark/` — benchmark repo (origin: github.com/anubhavaanand/image-to-3d); `src/evaluation/benchmark_runner.py` = the protocol; note: repo's `paper/` subfolder is a STALE copy, workspace `paper/` is authoritative
- `new papers/` — 11 PDFs (TripoSG 2502.06608, Hunyuan3D 2.0 2501.12202, Hunyuan3D 2.1 2506.15442, Direct3D-S2 2505.17412 = `document.pdf`, Sparc3D 2505.14521, LATTICE 2512.03052, SPAR3D **2501.04689**, InstantMesh 2404.07191, 3DGen-Bench 2503.21745, 3D Arena 2506.18787, WorldScore 2504.00983). Wrong physics PDF 2501.04777 deleted.
- Root PDFs: TripoSR 2403.02151, LGM 2402.05054, TRELLIS 2412.01506
- `cst_up_proposal.tex` — grant proposal (Rs 20,000; ODOP handicrafts framing; paper = its academic base). Paper #2 = application/heritage paper (deferred).
- Persistent memory: `~/.zcode/cli/memories/projects/reasearch-papers-fc4daa5ec638f428/memory/` (auto-loaded next session).

## 6. RESUME INSTRUCTIONS (new session)

1. Read this file + memory index (auto-loaded).
2. If `handpicked.txt` exists → push the smoke run via kaggle CLI (notebook + file), 10 objects (`NUM_SAMPLES=10`), report results.
3. If not → check `gso_picker/picker.html` exists, have user pick.
4. After smoke passes: fix runner gaps §3.1–3.3, run full 100, pull JSONs, rebuild Tables + Pareto figure from real data, update paper (replace literature tables), recompile.
5. Standing user triggers: "run the benchmark" = start Kaggle run; user reports AWS error = diagnose credit.

## 7. KEY FACTS TO NEVER RE-DERIVE

- TRELLIS: ~10 s (50 steps, CFG 3), F-Score 0.9999 (r=0.05, 500 Toys4k instances), CD 0.0083, PSNR 32.74, LPIPS 0.025; 2B params; 64×A100, 400K steps, 500,777 assets. FlexiCubes decode at 256³.
- LGM: NO geometric metrics in own paper (user study only: 4.18/3.95); 65,536 Gaussians; 32×A100 ~4 days; ~10 GB VRAM.
- TripoSR: GSO CD 0.111 / F@0.1 0.651 / F@0.2 0.871 / F@0.5 0.980 (10K pts, rot+ICP); OmniObject3D CD 0.102. <0.5 s on A100.
- SPAR3D Table 1 (GSO, ~250 obj, ICP): TripoSR 0.145/0.501/0.2s · LGM 0.196/0.356/41.0s · InstantMesh 0.135/0.545/36.1s · SF3D 0.137/0.540/0.3s · SPAR3D 0.120/0.584/0.7s.
- 3D Arena ELO: TRELLIS-mesh 1306 (#5), TRELLIS-splat 1384 (#2), LGM 1100 (#16), TripoSR 1089 (#17); splat>mesh +16.6; TRELLIS splat vs own mesh +78; textured +144.1.
- Hunyuan3D param counts (from LATTICE appendix only): mini 0.6B, 2.0 = 1.1B, 2.1 = 3B.
