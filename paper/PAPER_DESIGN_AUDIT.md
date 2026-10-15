# PAPER DESIGN AUDIT
## What the 14 published papers contain that our draft does not — a complete apparatus gap analysis

**Draft under audit:** `paper/main.tex` (334 lines, "A Unified Benchmark of Feed-Forward Paradigms for Single-Image 3D Reconstruction", compiled 7 pp.)
**Audit corpus:** 14 published papers, read in full (see §1). Every claim below is anchored to a specific paper, section, table or figure number.
**Purpose:** this document is the checklist for redesigning the paper professionally. Nothing here requires the 100-object run to *read*; items marked **[RUN]** need the run to *execute*.

---

## 1. The corpus — what each paper is and what it contributes to our design

| # | Paper | arXiv / venue | Type | What it contributes to our paper |
|---|---|---|---|---|
| 1 | **TripoSR** (Tochilkin et al., Stability AI + Tripo AI) | 2403.02151, technical report | Model paper (triplane) | The triplane paradigm; the GSO protocol origin (300 objects, 10K pts, rotation search + ICP); Table 2 = the "original" TripoSR numbers our fragmentation table anchors on |
| 2 | **LGM** (Tang et al., PKU/NTU/Shanghai AI Lab) | 2402.05054, ECCV 2024 Oral | Model paper (Gaussians) | The Gaussian paradigm; the *only* paper with per-stage timing labels (~4s diffusion, ~1s UNet, ~1m meshing) and the *only* one reporting inference VRAM (~10 GB); user-study table |
| 3 | **TRELLIS** (Xiang et al., Microsoft Research) | 2412.01506, CVPR 2025 | Model paper (structured latents) | The structured-latent paradigm; the most complete metric-definition apparatus (Eq. 13–14, r=0.05, 100K pts); LGM appears as a *row in its Table 2* — citable cross-paradigm data |
| 4 | **SPAR3D** (Huang et al., Stability AI) | 2501.04689 | Re-evaluating model paper | Re-evaluates 7 models incl. TripoSR, LGM, InstantMesh under one protocol **with a Time(s) column** (LGM 41.0 s, InstantMesh 36.1 s) — our strongest fragmentation evidence |
| 5 | **InstantMesh** (Xu et al., Tencent ARC) | 2404.07191, ICLR 2025 | Re-evaluating model paper | Re-evaluates TripoSR/LGM/CRM/SV3D on GSO + OmniObject3D (CD 0.247 / 0.345 — the third fragmentation anchor); capability-mismatch handling (SV3D gets "–") |
| 6 | **Hunyuan3D 2.0** (Tencent) | 2501.12202 | Model paper + user study | The 2025 convergence exemplar; VAE token-budget equalization; "300 unselected results" anti-cherry-picking language |
| 7 | **TripoSG** (Li et al., VAST) | 2502.06608 | Model paper (4B rectified flow) | Convergence exemplar; data-quality>quantity finding; anti-cherry-picking + no-post-processing statement; demo-version disclosure |
| 8 | **3DGen-Bench** (Zhang et al.) | 2503.21745 | Benchmark platform paper | The model of benchmark-paper hygiene: 22-model zoo, 68k votes, paid annotators with compensation disclosed, licensing section, Kendall's-τ validation |
| 9 | **WorldScore** (Duan et al., Stanford) | 2504.00983 | Benchmark platform paper | The gold standard for instrument validation: 400-participant 2AFC, agreement vs upper bound (0.637 vs 0.772), per-model Table S1 with named GPUs (H100/L40S) and per-instance wall-clock |
| 10 | **Sparc3D** (Li et al., NTU/Imperial) | 2505.14521 | Model paper (sparse VAE) | Convergence exemplar; re-evaluates TRELLIS/Craftsman/Dora VAEs with CD/ANC/F1 on a curated hard set + "Wild" set |
| 11 | **Hunyuan3D 2.1** (Tencent) | 2506.15442 | Tutorial-style model paper | Production-PBR convergence exemplar; ULIP/Uni3D evaluation convention (8,192 surface points) |
| 12 | **3D Arena** (Ebert, Hugging Face) | 2506.18787 | Preference platform paper | ELO leaderboard covering **all three of our models** (TripoSR 1089, LGM 1100, TRELLIS 1306, InstantMesh 1278); splat +16.6 / texture +144.1 ELO effects; binomial-test fraud detection as QA model |
| 13 | **LATTICE** (Lai et al., CUHK + Tencent) | 2512.03052 | Model paper (VoxSet) | Convergence exemplar; test-time token scaling; F1 at r=0.001 points-to-surface convention |
| 14 | **Direct3D-S2** (Wu et al., NJU/DreamTech/Oxford) | 2505.17412 (local file `document.pdf`) | Model paper (SSA) | Convergence exemplar; 8×A100 headline; runtime-vs-token-count plot (Fig. 7) — the efficiency-plot style to copy |

**Reading of the field:** three model papers define the paradigms; two re-evaluating papers (SPAR3D, InstantMesh) prove cross-paper numbers are irreconcilable; two benchmark platforms (3DGen-Bench, WorldScore) define what a *credible* evaluation artifact looks like; one platform paper (3D Arena) quantifies what crowds prefer without ground truth. Our paper sits exactly in the unclaimed intersection.

---

## 2. The expected apparatus — what a credible paper in this space contains

Synthesized from all 14. This is the checklist a reviewer who knows this literature will apply.

**A. Page-1 apparatus.** A teaser/hero figure coupling visual quality with one headline number on **named hardware** (TripoSR Fig. 1: "under 0.5 seconds"; LGM Fig. 1: "in 5 seconds"; TRELLIS Fig. 1: "about 10 seconds"). All three model papers do this; both benchmark papers instead open with a *positioning* figure (WorldScore Fig. 1 shows an existing benchmark confusing two models 80.28 vs 83.08 that WorldScore separates 70.70 vs 46.28 — a "blind spot" teaser; 3DGen-Bench Fig. 1 is an infographic of the dataset).

**B. Quantitative tables vs named baselines on held-out public data.** TripoSR Tables 2–3 (GSO, OmniObject3D); TRELLIS Tables 1–2 (reconstruction + generation on Toys4k); SPAR3D Tables 1–2 (7 baselines re-evaluated on GSO + OmniObject3D with 8 metric columns incl. Time); InstantMesh Tables 2–4. LGM is the outlier — its only table is a 3-row user study — and our paper should say so explicitly (it is part of the fragmentation story).

**C. Exact metric definitions.** TRELLIS is the model to copy: symmetric CD (Eq. 13) and F-score (Eq. 14) written as equations, threshold r=0.05 stated, **100K points** sampled from depth-unprojected surfaces over 100 uniform views. TripoSR states 10K points; InstantMesh 16K points at FS@0.2; LATTICE uses points-to-surface F1 at r=0.001. Point count and threshold vary by 10× across the field — this variance IS our Table 1.

**D. Alignment procedure.** TripoSR (Sec. 3): "brute-force search approach to align the predictions with the ground truth shapes. We linearly search the rotation angle by optimizing for the lowest CD and further employ ICP." SPAR3D (Sec. 4.1): "brute-force search in rotations … both normalized before the brute-force alignment … refined with ICP." TRELLIS discloses no alignment. **Our runner has neither normalization guarantees nor ICP** — our numbers would be inflated relative to every protocol that does align.

**E. Evaluation-set construction.** Every credible paper states curation rules and counts: TripoSR ~300 objects ("manually filter … exclude simple-shaped objects (box, sphere, cylinder)"), SPAR3D ~250 (same rule, following TripoSR), InstantMesh GSO-300 + OmniObject3D-130 (first 5 per category, 28 categories), TRELLIS 500/1,250-instance Toys4k subsets + Table 8 composition, 3DGen-Bench 1,020 prompts (510 text / 510 image, stratified 9:1:2), WorldScore 3,000 examples (Table S4). **Our draft never states the handpicked-100 selection criteria.**

**F. Training-data curation + licensing.** LGM's appendix A prints the full filter word-list; TRELLIS's Table 8 + Fig. 8 gives per-source thresholds and score histograms; TripoSR names the CC-BY subset. This apparatus doubles as licensing disclosure.

**G. Architecture disclosure to rebuild-level.** TripoSR Table 1 (every module config incl. learning rate); TRELLIS Table 6 (#layers/#dim/#heads/#params for all 14 networks); LGM Sec. 4.1 + Fig. 3.

**H. Hardware disclosure.** TripoSR: "A100 GPU" (inference), no training hardware in-paper; LGM: 32×A100-80G, 4 days; TRELLIS: 64×A100-40G, 400K steps; InstantMesh: 8×H800; TripoSG: 160×A100, 3 weeks; Direct3D-S2: 8×A100 (headline); WorldScore Table S1: H100 + L40S per model. **SPAR3D's hygiene failure is instructive: its Time(s) column discloses no GPU at all** — exactly the gap our paper exists to exploit, so our own hardware disclosure must be impeccable (T4, container, torch/CUDA versions, driver).

**I. Inference hyperparameters.** LGM: guidance 5 (ImageDream) / 7.5 (MVDream), 30 steps, DDM scheduler, azimuths (0,90,180,270), elevation 0, background removal + white composite. TRELLIS: CFG 3, 50 steps. SPAR3D: DDIM. **Our notebook's actual settings (TRELLIS 12+12 steps, etc.) must be stated per engine.**

**J. Speed reporting convention.** Total latency everywhere; **per-stage breakdown only in LGM** (Fig. 2 labels "Multi-view Generation ~4s / Gaussian Generation ~1s / Meshing ~1m"; Fig. 4 "Gaussian Splatting 15s → NeRF (Instant-NGP) 45s → Mesh + HashGrid Texture 5s"). A benchmark paper should generalize LGM's practice to all engines.

**K. Memory/VRAM.** Only LGM reports it ("about 10 GB … friendly for deployment"). **No paper reports measured peak VRAM per model on the same rig — this is our headline novelty and needs a methodology section** (allocator warm-up, `torch.cuda.max_memory_allocated` semantics, reserved vs allocated).

**L. Ablations.** TRELLIS: 5 ablation tables (latent size, generation paradigm, model size, timestep sampling, network configs). LGM: qualitative ablation figure (views / augmentation / resolution). TripoSR: none. A benchmark paper's equivalent: **protocol-sensitivity analysis** (same predictions scored under two CD conventions / with and without ICP) — cheap, and it converts our Pillar 1 from literature citation to measured demonstration. **[RUN]**

**M. User study.** LGM Table 1 (20 volunteers, 30 images, 1–5); TRELLIS Fig. 6 + supp. Table 9 (104 participants, 2,701 trials, full selection percentages + UI screenshot); Hunyuan3D "50 participants, 300 unselected results"; Direct3D-S2 (40 participants, 75 unfiltered meshes); LATTICE (~500 questions × 3 raters). Our free-tier constraint makes a large study unrealistic — the honest substitute is the 3D Arena ELO correlation (see §6, F9).

**N. Limitations & failure cases.** LGM Sec. 4.5 + appendix Fig. 11 (failure rows *labeled by category*: "Low resolution / 3D Inconsistency / Large Elevation"); TRELLIS supp. E; InstantMesh conclusion paragraph; Direct3D-S2 Sec. 8; Sparc3D conclusion. TripoSR is the cautionary outlier with none. **Our draft has a failure-mode taxonomy table but no failure figure and no separate Limitations section.**

**O. Release & data-availability statements.** TripoSR (MIT + 3 URLs), InstantMesh ("we release all the code, weights, and demo"), TRELLIS ("code, model, and data available"), 3DGen-Bench (HF dataset + Croissant metadata + A.2 Licensing Terms), WorldScore (site + leaderboard). **Our draft has none of these, though the notebook/dataset/code already exist.**

**P. Anti-cherry-picking statements.** TripoSG: "there are no duplicates among these cases, and the generated models have not undergone any post-processing (such as smoothing or removing floaters)." Hunyuan3D: "300 unselected results." 3DGen-Bench: cross-annotation audits (52.6%), conflict ratios (2%), error rates (3%). **Adopt the sentence pattern verbatim in our qualitative figure captions.**

**Q. Instrument validation.** WorldScore validates its metrics against a 400-participant 2AFC study (agreement 0.637, upper bound 0.772, per-bucket analysis) *before* presenting results. 3DGen-Bench reports Kendall's τ between automatic scores and human Elo. **A benchmark paper treats the evaluation protocol itself as the artifact under test — the single biggest craft difference between our draft and the platform papers.**

---

## 3. Item-by-item gap analysis

Legend: ✅ present · ⚠️ present but defective · ❌ absent · **[RUN]** requires the 100-object run.

| # | Apparatus item | Exemplar evidence | Draft status | Verdict |
|---|---|---|---|---|
| G1 | Own measured experiments | SPAR3D T1–T2, InstantMesh T2–T4 (all re-run baselines) | L141: "rely on standardized benchmark results aggregated from the original studies" | ❌ **The fatal gap.** Every results table is transcluded |
| G2 | Dataset section (selection criteria, counts, license) | TripoSR §3, SPAR3D §4.1, TRELLIS T8, WorldScore T-S4 | None | ❌ |
| G3 | Unified-protocol section (inputs, GT processing, alignment, point counts) | TRELLIS §C.1 Eq. 13–14; SPAR3D §4.1 | None — metrics named but never defined | ❌ |
| G4 | Alignment procedure (rotation search + ICP) | TripoSR §3; SPAR3D §4.1 | Not in draft; **not in runner either** | ❌ **[RUN]** |
| G5 | Metric equations (CD form, F-score def.) | TRELLIS Eq. 13–14; SPAR3D Eq. 3 | None | ❌ |
| G6 | Hardware/environment disclosure | WorldScore T-S1 (named GPUs per model); LGM/TRELLIS training rigs | Draft claims A100 for transcluded numbers; **our own rig (T4) never stated** | ❌ |
| G7 | Inference hyperparameters per engine | LGM §4.1; TRELLIS §4 | Not stated (TRELLIS 12-step setting lives only in the notebook) | ❌ |
| G8 | Per-stage timing breakdown | LGM Figs. 2/4 (the only exemplar) | Prose only, and only for transcluded numbers | ❌ **[RUN]** |
| G9 | Peak-VRAM measurement + methodology | LGM ~10 GB (only exemplar, no method given) | Claimed per-model VRAM in T-unified is literature folklore (4–6 / ~10 / ~16 GB), unsourced per-rig | ❌ **[RUN]** — headline novelty |
| G10 | Statistical treatment (std / error bars / per-object) | Absent in all 14 (TRELLIS closest via user-study percentages) | Runner computes mean±std; paper shows none | ❌ — **a publishable novelty if we add per-object distributions** |
| G11 | Teaser/hero figure | All 3 model papers Fig. 1; WorldScore Fig. 1 | None | ❌ |
| G12 | Qualitative comparison grid (own renders, all engines) | TripoSR Fig. 3; LGM Figs. 5–7; TRELLIS Fig. 5/15; SPAR3D Fig. 5 | None (draft has zero images of any system output) | ❌ **[RUN]** |
| G13 | Failure-case figure (categorized rows) | LGM Fig. 11 (3 labeled categories) | Taxonomy **table** only (T-failure) | ❌ **[RUN]** |
| G14 | Protocol-sensitivity / ablation-of-protocol | TRELLIS T3–T5 (5 ablation tables); WorldScore T-S7 robustness | None | ❌ **[RUN]** |
| G15 | Protocol-fragmentation table | — (our novelty) | T1 present, numbers verified against PDFs | ✅ strongest existing asset |
| G16 | Efficiency re-measurement claims audit | SPAR3D Time column; draft §4.3 | Prose (LGM 41 s = 8× claim; InstantMesh 36.1 s = 3.6×) — verified against SPAR3D T1 | ✅ promote to a proper table |
| G17 | Pareto/efficiency plot from measured data | TripoSR Fig. 2 (log-time scatter, per-method) | `generate_pareto.py` plots **hardcoded literature numbers on three incompatible protocols** (GSO F-score, Toys4k F-score, LGM user-study/5 proxy) | ⚠️ **must be rebuilt from our JSONs** **[RUN]** |
| G18 | Pipeline diagrams of the *paradigms* | LGM Figs. 2–3; TRELLIS Figs. 2–3 (round boxes, stage coloring, timing labels); SPAR3D Fig. 2 | 3 figures exist but are **Kroki.io web-service renderings of Mermaid text** — third-party style, no data shapes, no timing labels | ⚠️ redraw |
| G19 | Architecture diagram of *our benchmark system* | WorldScore Fig. 2; 3DGen-Bench Fig. 1 (the pipeline IS the contribution) | None | ❌ |
| G20 | User study | LGM T1; TRELLIS T9; Hunyuan; Direct3D-S2; LATTICE | None; free tier makes own study unrealistic | ⚠️ substitute: 3D Arena ELO correlation (see §6-F9) |
| G21 | Limitations section (separate) | LGM §4.5; TRELLIS supp. E; InstantMesh §5; Direct3D-S2 §8 | Folded into Discussion | ❌ |
| G22 | Failure-mode taxonomy table | — (our design) | T-failure present | ✅ keep, add quantified column **[RUN]** |
| G23 | Deployment-recommendation table | — (our design) | T-deployment present | ✅ keep |
| G24 | Reproducibility / artifact statement | TripoSR (MIT+URLs); InstantMesh; TRELLIS; WorldScore | None — yet the code, dataset and notebook all exist | ❌ trivial to add |
| G25 | Licensing / ethics statement | 3DGen-Bench §A.2 (the model); TripoSR MIT | None | ❌ |
| G26 | Data availability statement | 3DGen-Bench (Croissant); WorldScore | None | ❌ |
| G27 | Anti-cherry-picking caption language | TripoSG; Hunyuan3D; 3DGen-Bench | None | ❌ |
| G28 | Acknowledgments + funding/grant | TripoSR; LGM (long list); WorldScore (ONR/NSF) | Guide acknowledged ✅; **CST-UP grant not acknowledged** | ⚠️ add grant line |
| G29 | Supplementary/appendix with own tables | TRELLIS pp. 14–26 (T6–T9, F8–F20); LGM appendix | None | ⚠️ plan for it |
| G30 | Bib integrity | — | `trellis2024` key carries a 2025/CVPR entry; `instantmesh2024` key carries ICLR 2025; **ULIP/Uni3D/OpenShape uncited** (needed if we adopt their metrics or cite Hunyuan-style eval); Dora uncited (Sparc3D/LATTICE compare against it) | ⚠️ |
| G31 | Contribution list formatting | 3DGen-Bench: 3 bolded numbered; TripoSG: numbered advances + bullets; WorldScore: 4 bullets | 3 bullets, not numbered/bolded | ⚠️ cosmetic |
| G32 | Empty/placeholder assets | — | `paper/tables/` empty; `IEEE_Conference_Template.docx` and `IEEE_Template.docx` are **0 bytes** | ⚠️ cleanup |

---

## 4. Factual-accuracy flags found during the audit (fix before submission)

These were found by checking draft sentences against the PDFs. They *individually* justify the paper's own thesis — manual cross-paper transcription is error-prone — and one of them is already a fabricated-looking number.

1. **F1 — `main.tex` L90, "trained on 176 NVIDIA A100 GPUs for 5 days": NOT in the TripoSR paper.** Pages 2–4 checked: the paper discloses no training hardware or duration at all (only "A100 GPU" for the 0.5 s inference claim, and the 128×128 patch supervision, which is real). Remove or re-source. *(This is exactly the class of error our paper warns about.)*
2. **F2 — T-trellis-toys4k (L232–235) transcription errors vs TRELLIS Table 1:** LN3Diff CD printed **0.0299** (PDF: **0.01099**), LN3Diff F printed **0.9649** (PDF: **0.9699**), CLAY CD printed **0.0124** (PDF: **0.01024**). Three wrong cells in one 4-row table — photographed evidence for §3-G1.
3. **F3 — LGM mesh-conversion stage attribution (L106).** Draft: "Instant-NGP distillation (~15 s), Marching Cubes (~5 s), refinement via nvdiffrast (~45 s), xatlas (~5 s)". LGM Fig. 4 labels: "Gaussian Splatting 15s → NeRF (Instant-NGP) 45s → Mesh + HashGrid Texture 5s". Same ~1 min total, different stage→time mapping. Re-verify Fig. 4 and correct; when we measure the conversion ourselves the draft numbers become irrelevant anyway. **[RUN]**
4. **F4 — Abstract metric inconsistency (L141 vs tables):** §4.1 says "F-Score (τ ∈ {0.1, 0.2})" while every table reports F@0.5 too. State all three thresholds or fix the text.
5. **F5 — T-unified mixes datasets in one column set** (GSO\* vs Toys4k†) with TRELLIS CD 0.0083 sitting next to TripoSR 0.111\* — visually a 13× gap that is mostly *dataset*, not method. The table is honest (footnoted) but a reviewer will read it as a ranking; restructure as two panels or move it after the fragmentation table with an explicit "not comparable" caption.
6. **F6 — SPAR3D arXiv id:** web sources disagree (2501.06806 vs 2501.04689); the local PDF (`2501.04689v1_SPAR3D.pdf`, sidebar "8 Jan 2025") is authoritative — `references.bib` already carries 2501.04689. Keep, and cite the PDF not the web.
7. **F7 — "Structured Latent Flow Transformers" (title of §3.3):** TRELLIS uses *rectified flow* (the draft's own abstract says "rectified linear probability paths", which is a slip for *rectified-flow probability paths* — check L129/L322). Terminology should be exactly "rectified flow" everywhere.
8. **F8 — TripoSR has no architecture figure** in its own paper (confirmed: only Figs. 1–3, none architectural). Our Fig. `triposr_pipeline` is therefore *our* reconstruction, not the paper's — the caption should not imply otherwise, and it must match TripoSR Table 1 exactly (it currently does: ViT-B/16, 16-layer decoder, 64×64×40 ✓).
9. **F9 — TRELLIS F-score convention:** draft table caption says "F-Score at radius r=0.05" ✓ correct, but the draft nowhere explains that this r=0.05 is **not** the same quantity as TripoSR's FS@0.5 — different normalizations make them incommensurable. One sentence in the protocol section fixes it.

---

## 5. Figures & diagrams blueprint (the complete set, with drawing specs)

Order = order of appearance in the rewritten paper. "(style ref)" points at the exemplar to imitate. Everything marked **[RUN]** needs the 100-object run's outputs; the rest can be produced now.

**F1 — Teaser / hero (page 1). (style ref: WorldScore Fig. 1 blind-spot teaser; TripoSR Fig. 1)**
Layout: one input image column on the left; three reconstruction renders (TripoSR / LGM / TRELLIS, same object, two views each) to the right; underneath, a one-row "the numbers differ by protocol" strip: `CD 0.111 | 0.145 | 0.247 — same model, three papers`. Headline caption: *"Same image, three feed-forward paradigms — and three irreconcilable numbers for the same model."* Message-first like WorldScore, not gallery-first like TripoSR. **[RUN]** for the renders (can placeholder with our own smoke-run renders if available).

**F2 — Benchmark-system architecture (our contribution diagram). (style ref: WorldScore Fig. 2; TRELLIS Fig. 2 two-band layout)**
Two horizontal bands, rounded module boxes, stage-tinted backgrounds, data shapes annotated on arrows:
- Band 1 "Unified Protocol": `GSO mirror (1,046 objs)` → `handpicked-100 ingest` → `unit-sphere GT normalization + 10K-pt sampling` → `self-rendered 512² white-bg inputs`
- Band 2 "Engines → Metrics": three parallel engine boxes (TripoSR triplane-NeRF / LGM MV-Gaussians / TRELLIS SLAT-flow, each with its output glyph: mesh / splats / SLAT→FlexiCubes mesh) → `rotation search + ICP` → `CD · F@{0.1,0.2,0.5} · PSNR · LPIPS · latency · peak-VRAM` → `unified CSV/JSON`.
Draw this ourselves (TikZ or draw.io export PDF) — never a web-service render. This figure does not exist in the draft and is the one a reviewer will look for first.

**F3–F5 — Paradigm pipeline diagrams (replace the Kroki PNGs). (style ref: LGM Fig. 2 for timing labels; TRELLIS Fig. 3 for block stacks)**
Keep one figure per paradigm but redraw to exemplar standard: rounded pastel module boxes, **per-stage timing labels under the arrows** (LGM's "~4s / ~1s / ~1m" pattern — with *our measured* T4 timings, which is a contribution over the original figures: same diagram, free-tier numbers), tensor shapes on arrows (TripoSR: 1024×768 tokens → 3×64×64×40 triplane; LGM: 4×256² → 65,536×14 Gaussians; TRELLIS: ~20K SLAT tokens, 8-dim). **[RUN]** for the measured timings.

**F6 — Protocol-fragmentation dot plot (upgrade of Table 1). (style ref: TripoSR Fig. 2 log-axis scatter)**
X = CD (log), one row of dots per (model × evaluation source): TripoSR at 0.111 / 0.145 / 0.247 and LGM at 0.196 / 0.345, colored by source paper, connected by horizontal range lines annotated "2.2×". Can be built **today** — all numbers are verified literature values.

**F7 — Main results: per-object distributions. (novel — no exemplar does this)**
Box/violin per engine per metric over the 100 objects (CD, F@0.1, PSNR, LPIPS). None of the 14 papers show dispersion at all; this is the cheapest genuine statistical novelty in the paper. **[RUN]**

**F8 — Efficiency: latency breakdown + Pareto, measured only. (style ref: Direct3D-S2 Fig. 7; TripoSR Fig. 2)**
(a) Stacked-bar per-stage latency per engine (TripoSR: encode+triplane+MC; LGM: MVDream+UNet+mesh-conversion — the conversion bar will dominate and *is the finding*; TRELLIS: structure+SLAT+decode) on the T4; (b) Pareto scatter: latency vs F@0.1, bubble = measured peak VRAM. Delete the hardcoded `pareto_frontier.png` entirely. **[RUN]** — note: `lgm_engine.py` (2026-10-10 22:06 revision) already emits `latency_mv_s` / `latency_unet_s` per object in its result metadata, so the LGM stage split needs no new instrumentation, only the mesh-conversion stage and the TripoSR/TRELLIS equivalents.

**F9 — Preference vs geometry correlation. (novel; built from 3D Arena Table 1)**
Scatter: x = 3D Arena ELO (TripoSR 1089, LGM 1100, TRELLIS 1306, InstantMesh 1278, SPAR3D 1144 — five points, all models we can measure), y = our measured F@0.1 / CD. Caption makes the paper's meta-point: crowd preference and ground-truth geometry **disagree** (Arena's own splat +16.6 / texture +144.1 ELO effects are the mechanism). Five points is honest; do not fit a line to n=5, show the rank disagreement. **[RUN]** for y-axis.

**F10 — Qualitative comparison grid. (style ref: TripoSR Fig. 3 — including its texture-normalization fairness device)**
Rows = engines + GT; columns = 6 handpicked objects (2 per broad category); each cell two views; textureless renders for any engine whose textures are unreliable (TripoSR's w/o-texture row is the precedent for the fairness note). Anti-cherry-picking caption in TripoSG's language: *"no duplicates; no post-processing (no smoothing, no floater removal)."* **[RUN]**

**F11 — Failure cases. (style ref: LGM Fig. 11 — rows labeled by failure category)**
Rows = "back-side blurring (triplane)", "floater artifacts / multi-view inconsistency (Gaussians)", "baked-in shading (SLAT)"; columns = input → output zoom. Maps 1:1 onto the draft's T-failure rows so table and figure reinforce each other. **[RUN]**

**F12 — (Supplementary) dataset-overview figure. (style ref: TRELLIS Fig. 8 histograms; WorldScore Fig. 3)**
Handpicked-100 category composition + the GSO mirror stats; renders of a sample of the 100 with their GSO ids. Feeds reproducibility. **[RUN]** (or partial now from the catalog).

---

## 6. Tables blueprint

**Writable today (literature-verified):**
- **T1 Protocol fragmentation** (exists; extend with SPAR3D's full columns): add the Time(s) column values — TripoSR 0.2 s, SPAR3D 0.7 s, InstantMesh 36.1 s, LGM 41.0 s — and InstantMesh's protocol row ("300 obj, 16K pts, FS@0.2 only, orbit views"). Verified against SPAR3D T1 and InstantMesh T2.
- **T2 "Published efficiency claims vs re-measured"** (promote draft §4.3 prose): columns Model | Published claim (verbatim, + hardware if stated) | SPAR3D re-measurement | ratio. Rows: TripoSR "<0.5 s, A100" / 0.2 s / 0.4×; LGM "5 s, unstated GPU" / 41.0 s / 8.2×; InstantMesh "within 10 s" / 36.1 s / 3.6×. The unstated-GPU cells get an explicit "—" — that absence is the finding.
- **T3 Cross-paradigm generation metrics, citable as-is:** TRELLIS Table 2 image-to-3D block already contains **LGM as a row** (CLIP 83.97, FD_incep 26.31, KD_dinov2 38.27, FD_point 15.90) next to TRELLIS-L (85.77 / 9.35 / 0.72 / 2.03) on Toys4k. Cite as "the only existing same-protocol cross-paradigm numbers, on Toys4k, without ground-truth geometry" — it motivates our GSO run.
- **T4 3D Arena leaderboard extract** (rows for our models + InstantMesh/SPAR3D/SF3D, with Votes and Win-Rate columns) — the preference-channel anchor for F9.

**Requires the 100-object run — column specs now, fill later:**
- **T5 Main results** (the paper's centerpiece): Method | CD↓ | F@0.1↑ | F@0.2↑ | F@0.5↑ | PSNR↑ | LPIPS↓ | Latency(s)↓ | Peak-VRAM(GB)↓ — mean **and std** per cell (std over 100 objects; no paper in the corpus reports std — say so in the caption). LGM's geometry cells: report with the mesh-conversion step included (its cost is its problem) and footnote the latency split.
- **T6 Per-stage latency** (feeds F8): Engine | stage₁ | stage₂ | … | total, measured on T4.
- **T7 Protocol sensitivity**: same predictions, {our CD squared / field mean-distance CD} × {with / without ICP} — a 2×2 per engine. Converts Pillar 1 from citation to measurement.
- **T8 Failure taxonomy** (exists; add quantified columns: median hole count / floater count per paradigm) **[RUN]**.

---

## 7. Evaluation-hygiene checklist (from the re-evaluating papers — our compliance status)

The 10-point apparatus distilled from SPAR3D, InstantMesh, 3DGen-Bench and WorldScore, checked against our current notebook/runner:

1. **Same inputs to every method** — ✅ compliant by design (self-rendered 512² white-bg set fed to all three engines; mirrors SPAR3D "evaluate the produced meshes under the same protocol").
2. **Official implementations, deviations documented** — ⚠️ we clone official repos, but the TRELLIS 12-step deviation from the 50-step default and the `load_state_dict(strict=False)` sed on TripoSR are undocumented. State both, with the reason (T4 VRAM).
3. **One coordinate frame + exact alignment recipe** — ❌ runner has no normalization guarantee and no ICP. Adopt TripoSR's recipe verbatim (linear rotation search minimizing CD, then ICP) and print the search parameters we use. **[RUN]**
4. **Curated test set, rules disclosed** — ⚠️ the handpicked-100 exists but its selection rationale is nowhere in the draft. Write the criteria (category coverage, no simple primitives — TripoSR's rule —, license permissibility) into §Dataset.
5. **Capability mismatches flagged, not forced** — ⚠️ draft's T-unified leaves LGM cells "--" ✓ good pattern, but PSNR/LPIPS for Gaussian outputs needs an explicit "rendered via splat rasterization, not mesh" footnote (InstantMesh's SV3D "–" is the model).
6. **One timing rig, disclosed** — ⚠️ all engines will run on the same T4 ✓, but the draft must state container image, torch/CUDA, driver, and that all timings are same-rig (the SPAR3D omission is the counterexample to avoid).
7. **Anti-cherry-picking statement** — ❌ add TripoSG's sentence pattern to the qualitative figure captions.
8. **Instrument validated against humans** — ⚠️ own user study unrealistic on the free tier; substitute the 3D Arena correlation (F9) and frame it as *cross-instrument* validation: our GT ranking vs 123k-vote ranking.
9. **Separate own-ablations from re-evaluations** — ✅ by construction (we only re-evaluate our three engines under one protocol; T7 is the protocol ablation).
10. **Release everything** — ❌ statement absent; artifacts exist (dataset `image-to-3d-code`, notebook, runner). Add an Artifacts paragraph with URLs + licenses (GSO CC-BY 4.0; TripoSR MIT; LGM Apache-2.0; TRELLIS MIT — verify each from the repos before submission).

**Supplement — community benchmark-writing standards (web-verified 2026-10-10):** the NeurIPS Datasets & Benchmarks review criteria, the Pineau et al. reproducibility checklist, and protocol-design guidance (ACM/NIST response, GLUE-style protocol standardization) independently corroborate §2's apparatus and add five requirements the 14-paper corpus alone does not surface:

- **S1 Seeds/repeats + confidence intervals.** TRELLIS flow sampling is stochastic — the seed must be disclosed and, ideally, k repeats per object reported with CIs (the 100-object std gives this almost for free; k=1 with n=100 objects is defensible if stated).
- **S2 "One command to reproduce the leaderboard."** (Papers-with-Code convention.) Our artifact statement should say exactly that: *one Kaggle kernel run reproduces every number in every table* — and the notebook genuinely does this.
- **S3 Explicit scope statement** — what the score does *not* evidence (per the "don't mistake a score for evidence" guidance): our F@0.1 ranking does not measure human preference, texture quality, or downstream usability. The deployment table partially plays this role; make the scope sentence explicit.
- **S4 Dataset documentation ("datasheets").** GSO provenance + the handpicked-100 construction (criteria, counts, license per GSO's CC-BY) documented as a first-class section — §III already plans this.
- **S5 Versioned artifacts.** Dataset version pinning (`image-to-3d-code` v2, 49 files) named in the paper; a maintenance note ("kernels run on Kaggle's rolling image; torch/CUDA versions recorded per run in the log") pre-empts the "results won't reproduce next year" review.

---

## 8. Metric-convention decisions (must be fixed and stated)

| Decision | Field's options (evidence) | Our runner today | Recommendation |
|---|---|---|---|
| CD formulation | Squared vs mean-distance; symmetric two-term (TRELLIS Eq. 13); ×100 scaling (TRELLIS T2 "KD ×100"; LATTICE ×10⁴) | `mean(d1²)+mean(d2²)` — squared, **nonstandard**, inflates vs literature by construction | Switch to the field's symmetric mean-distance CD, report BOTH in T7, and state the choice in §Protocol. Without this our CD column is incomparable with every number in T1. |
| F-score thresholds | FS@{0.1,0.2,0.5} on unit-normalized meshes (TripoSR, SPAR3D); FS@0.2 only (InstantMesh); r=0.05 on [0,1]-normalized (TRELLIS); r=0.001 points-to-surface (LATTICE) | Absolute 0.1/0.2/0.5 — unit-sensitive unless meshes are normalized first | Normalize GT+pred to unit sphere (TripoSR/SPAR3D recipe), keep FS@{0.1,0.2,0.5}, state normalization explicitly |
| Point count | 10K (TripoSR) / 16K (InstantMesh) / 100K (TRELLIS) | 10K for GT (`sample_surface(gt_mesh, 10_000)`) | 10K, matching TripoSR (the protocol our T1 anchors on); note TRELLIS uses 100K |
| Alignment | Rotation search + ICP (TripoSR, SPAR3D) / none disclosed (TRELLIS, InstantMesh: "align coordinate system" only) | **None** | Implement rotation search + ICP; without it, cross-paradigm comparison is meaningless for view-centric outputs (LGM reconstructs in view space — InstantMesh's Fig. 3 elevation footnote is the cautionary tale) |
| LGM geometry | Gaussians are not meshes; LGM itself reports no geometry metrics (only user study) | `mesh_to_pointcloud` returns None → NaN metrics | Implement the Gaussian→mesh conversion (LGM §3.5; ~1 min/obj on their rig — measure ours), count it in LGM's latency, and keep splat-native PSNR/LPIPS as a secondary row **[RUN]** |
| Input images | White-bg composite (LGM); rembg (TripoSG); MVDream-style HDRI test renders (SPAR3D) | RGBA→white composite, self-rendered | State it; also state the GT novel-view background mismatch (white vs evaluator black) as a known limitation |

---

## 9. Rewrite blueprint for `main.tex` — **ICIP format: 5 pages content + 1 references-only page** (web-verified 2026-10-10 from the ICIP 2025 kit; ICIP 2027 kit not yet published)

The original blueprint assumed 8 pages — **wrong for ICIP**. The current draft compiles to 7 pages, so even before own data is added it must lose ~2 pages. The trade that makes it fit: the four transcluded literature tables (T-triposr-gso, T-triposr-omni, T-trellis-toys4k, T-lgm-user ≈ 1 page) are **deleted from the main text** — they move to the arXiv/supplementary version — and their one-line conclusions stay in prose. Own-data tables take the freed space.

| § | Content (5-page budget) | Pages | Disposition |
|---|---|---|---|
| I | Intro + F1 teaser + 3 numbered contributions | 0.8 | Keep, tighten |
| II | Related work — compress 6 subsections to 4 paragraphs (optimization+multi-view merged; feed-forward; 2025 convergence; evaluation landscape) | 0.6 | Compress |
| III | **Unified Protocol** (spine): dataset criteria, input/GT pipeline, metric equations, alignment, hardware/env disclosure, per-engine settings + deviations | 1.0 | New |
| IV | Paradigms: 3 compressed subsections (~0.2 col each) + redrawn F3–F5 as one 2-row figure or moved to supplementary | 0.6 | Compress hard; full detail → arXiv version |
| V | Results: T5 main (mean±std), **one** efficiency figure (F8), T7 protocol sensitivity, F9 ELO correlation | 1.3 | Own data; F7 distributions or F10 qualitative → supplementary if tight |
| VI | Discussion: failure taxonomy (T8), deployment table (keep), explicit Limitations | 0.6 | Split |
| VII | Conclusion + Artifacts/licensing + Acknowledgments (CST-UP grant) | 0.1 | New statements |
| p6 | References only | 1.0 | ~35 refs |

**Anything cut lands in one of two places:** an arXiv preprint version (full 8-page apparatus: T1 extension, T2 claims audit, T3 TRELLIS-LGM row, T4 Arena extract, F6, F10, F11, F12, full paradigm detail) and/or an ICIP supplementary upload. Strategy: write the 8-page version first as the master, then compress — never write the 5-page one directly, or the protocol section (the paper's spine) gets gutted.

---

## 10. Write-now vs blocked-on-run

**Writable immediately (no run needed):** T1 extension, T2, T3, T4, F6, §III protocol text, §II additions, all bib fixes, G24–G28 statements, F2 system diagram, corrected F3–F5 specs, Limitations section, F1/F5/F7/F8/F9/F10/F11 column-and-caption scaffolding with placeholders.

**Blocked on the 100-object run:** T5/T6/T7/T8 data cells, F1/F7/F8/F9/F10/F11 render data, measured VRAM, measured per-stage timings. **Runner prerequisites (other agent's lane):** peak-VRAM instrumentation (`torch.cuda.max_memory_allocated` + reset per object), rotation-search+ICP, CD convention switch, LGM Gaussian→mesh conversion, results-JSON robustness (already hardened v23–v24).

---

## 11. Venue & timeline (deadlines web-verified 2026-10-10)

| Venue | Deadline | Status |
|---|---|---|
| CVPR 2027 | Registration Nov 10, 2026; submission Nov 16, 2026 | **Not reachable with evidence** — the 100-object run has not produced a single successful engine pass yet; do not plan on it |
| 3DV 2027 | Aug 28, 2026 | Passed |
| **ICIP 2027 (target)** | **Mar 31, 2027** (conference Nov 29–Dec 3, Singapore) | ~24 weeks out. **Format: 5 pages + 1 refs-only page** (ICIP 2025 kit; 2027 kit not yet posted — re-check 2027.ieeeicip.org). Backward plan: Oct–Nov = runner upgrades + 100-object run + T5–T7; Dec = 8-page master draft + figures; Jan = compress to 5+1 + guide review (Dr. Paliwal) + freeze; Feb = polish, anti-cherry-picking captions, artifact statements; Mar = compile, page-fit, submit |
| Fallback | INDICON/ICACCS tier | If ICIP slips |

---

## 12. Cleanup candidates (need user's OK, not done)

- `paper/IEEE_Conference_Template.docx` and `paper/IEEE_Template.docx` — both **0 bytes**, dead weight.
- `paper/tables/` — empty directory.
- `paper/download_diagrams.py` + `paper/architecture_diagrams.md` — obsolete once F3–F5 are redrawn (keep as history or delete).
- `generate_pareto.py` — obsolete once F8 exists (it cannot be salvaged: its inputs are three incompatible protocols).
- `paper/main.pdf`, `*.aux/log/out/bbl` build artifacts — safe to clean before any repo push.
