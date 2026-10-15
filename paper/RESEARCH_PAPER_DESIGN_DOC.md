# RESEARCH PAPER DESIGN DOC — Master plan for the IEEE paper
**"A Unified Benchmark of Feed-Forward Paradigms for Single-Image 3D Reconstruction"**
**Anubhav Anand — Amity University UP / CST-UP grant. Compiled 2026-10-15.**

This document consolidates EVERYTHING needed to write the final paper: (a) the verified 14-paper
corpus comparison, (b) our own measured evidence and its provenance, (c) the novelty check against
the 2026 literature, (d) the section-by-section writing blueprint, and (e) every factual fix the
text must incorporate. It supersedes scattered notes; `PAPER_DESIGN_AUDIT.md` remains the detailed
apparatus gap-analysis it builds on.

**Provenance symbols:** `[V28]` = own measurement, `kaggle_output_v28/` (post-`98f5937` protocol,
Oct-13 kernel run, runner md5 `569f2faf79`); `[FC]` = fact-checked from the 3 arXiv PDFs
(memory `factcheck-triposr-lgm-trellis.md`); `[AUD]` = `paper/PAPER_DESIGN_AUDIT.md`;
`[WEB]` = web-verified 2026-10-15; `[MEM]` = project memory file.

---

## 1. VERIFIED EVIDENCE INVENTORY (what the paper may cite as OURS)

### 1.1 Own measurements — the 10-object smoke pass [V28]

Rig: Kaggle free tier, Tesla T4 (15.6 GB), torch 2.11.0+cu128, py3.13, 10 handpicked GSO objects
(seed-42 protocol, white-bg 512² inputs, GT re-rendered white-bg, 10K surface points, symmetric
mean CD, F@{0.1,0.2,0.5}, per-object peak VRAM via `torch.cuda.max_memory_allocated`).

| Engine | n | Latency (s) | Peak VRAM (GB) | CD ↓ | F@0.1 | F@0.2 | F@0.5 | PSNR ↑ | LPIPS ↓ |
|---|---|---|---|---|---|---|---|---|---|
| TripoSR | 9/10 ¹ | 3.61 ± 0.28 | 2.50 | 0.282 ± 0.132 | 0.451 | 0.702 | 0.994 | 10.82 ± 1.92 | 0.412 ± 0.117 |
| LGM | 10/10 | 8.39 ± 0.25 | 5.57 | 0.474 ± 0.167 | 0.124 | 0.379 | 0.877 | 10.36 ± 2.03 | 0.285 ± 0.082 |
| TRELLIS | 0/10 ² | — | — | — | — | — | — | — | — |

¹ `3D_Dollhouse_Sofa` failed post-reconstruction (degenerate tensor reshape in MC extraction).
² All 10 objects: SLAT sampling OK, FlexiCubes mesh decode CUDA-OOM at 14.17/14.56 GiB on the
shared-GPU run (flexicubes.py:71 + sparse/norm.py:19 zeros_like). **Isolated 2×T4 re-run in
flight** (`trellis-smoke-2gpu`, decode offloaded to the idle second GPU — verified idle in the
Oct-14 interactive session: GPU-1 at 3 MiB while GPU-0 hit 14.6/15 GiB) [MEM].

**Key headline facts from V28 (each is a paper finding):**
- **VRAM ordering**: TripoSR 2.5 GB < LGM 5.6 GB — both *below* their published/ folklore numbers
  (LGM "~10 GB" claim not reproduced; measured 5.57 GB on T4 with the big 65,536-Gaussian config —
  config verified: "Skipped 35 keys" warning gone) [V28][MEM].
- **Latency ordering flips the narrative**: TripoSR 3.6 s on T4 is 7× its "<0.5 s (A100)" claim —
  published latency is hardware-relative, exactly the paper's thesis [V28][FC].
- **LGM geometry is scorable** (Gaussian→mesh conversion works): CD 0.474 vs TripoSR 0.282 —
  the first same-protocol geometry numbers for LGM on GSO (its own paper reports none) [V28][FC].
- **TRELLIS cannot complete on a single 15 GB T4 in mesh mode** — the efficiency column the field
  never reports (VRAM) is now measured and decisive [V28][MEM].
- LGM is **better on perceptual metrics** (LPIPS 0.285 vs 0.412) while **worse on geometry**
  (CD 0.474 vs 0.282) — the preference-vs-geometry divergence, measured locally, consistent with
  3D Arena ELO behavior [V28][FC].

### 1.2 Literature values verified against the PDFs [FC][AUD]

- TripoSR GSO (~300 obj, 10K pts, rot-search+ICP, CD×100): 0.111 / F 0.651 / 0.871 / 0.980;
  OmniObject3D 0.102 / 0.677. Baselines: One-2-3-45 0.227, ZeroShape 0.160, TGS 0.122, OpenLRM 0.180.
- SPAR3D re-eval (~250 obj, ICP): TripoSR 0.145/0.501/0.2 s · LGM 0.196/0.356/41.0 s ·
  InstantMesh 0.135/0.545/36.1 s · SF3D 0.137/0.540/0.3 s · SPAR3D 0.120/0.584/0.7 s.
- InstantMesh re-eval: TripoSR CD 0.247 (GSO), 0.345 (OmniObject3D).
- TRELLIS Toys4k (500 inst, r=0.05, 100K pts): PSNR 32.74, LPIPS 0.025, CD 0.0083, F 0.9999.
  **Correct the draft's T-trellis table: LN3Diff CD 0.01099 (not 0.0299), F 0.9699 (not 0.9649);
  CLAY CD 0.01024 (not 0.0124)** [AUD-F2].
- LGM own paper: NO geometry metrics; user study only (20 volunteers: 4.18/3.95).
- Protocol-fragmentation headline: TripoSR GSO CD = 0.111 / 0.145 / 0.247 across three papers = 2.2×.
- Architecture facts (for §3): TripoSR DINOv1 ViT-B/16 @512², 16-layer triplane transformer
  1024d/16h, triplane 64×64×40, NeRF MLP 64w×10L, MC mesh. LGM MVDream/ImageDream → 4×256² →
  asymmetric U-Net → 65,536×14 Gaussians, Plücker conditioning; meshing Instant-NGP 15 s → MC 45 s
  → refine → xatlas (~1 min) — **fix draft's stage→time mapping** [AUD-F3]. TRELLIS SLAT 64³,
  ~20K active voxels, C=8; SS-flow 975M + SLAT-flow 1073M (2B total); FlexiCubes @256³;
  trained 64×A100-40G, 400K steps, 500,777 assets.
- **Training-hardware fabrication to delete**: draft L90 "176×A100 5 days" for TripoSR is NOT in
  the paper [AUD-F1].

### 1.3 Verified repo/config internals (for the protocol section's honesty) [FC][MEM]

- LGM: HF checkpoint = `big` config; loading into `small` silently drops 35 keys under
  `strict=False` (the v33 16,384-Gaussian bug — now fixed and verified in V28).
- TRELLIS loader: `Pipeline.from_pretrained` masks first-attempt errors behind a bogus 404
  fallback; canonical usage = `snapshot_download("microsoft/TRELLIS-image-large")` → local dir.
- TRELLIS ckpts ship **fp16 already**; the OOM driver is fp32 FlexiCubes dense grids
  (256³×57 ≈ 3.8 GB) + resident flow weights, NOT weight dtype (reviewer-verified from source).
- spconv JIT ninja failure on py3.13 is **non-fatal** (sparse convs ran; sampling completed).

---

## 2. NOVELTY CHECK vs the 2026 literature [WEB, checked 2026-10-15]

- arXiv 2507.14501 "Advances in Feed-Forward 3D Reconstruction and View Synthesis" (Jul 2025) is a
  *survey* of datasets/protocols — cites the gap, does not fill it with own measurements.
- Uni3R (CVPR 2026), Any4D, TRELLIS.2 (Dec 2025, needs ≥24 GB), Hunyuan3D 2.x, TripoSG: all are
  **method papers**; none re-measures the three paradigms under one GT-referenced protocol with
  per-object VRAM.
- No competing paper found that reports measured peak VRAM per engine on identical hardware.
  **The measurement-gap positioning remains unclaimed.** TRELLIS.2's existence *strengthens* the
  retrospective-audit framing (the field converged even harder on structured latents).
- Action for the paper: cite 2507.14501 (survey) as independent confirmation that protocol
  fragmentation is recognized but unquantified.

---

## 3. THE CORPUS COMPARISON (condensed matrix; full detail in PAPER_DESIGN_AUDIT §1–2)

| Paper | Type | Paradigm | Eval protocol | Geometry GT? | VRAM? | What we take |
|---|---|---|---|---|---|---|
| TripoSR 2403.02151 | model | triplane | GSO ~300, 10K pts, rot+ICP, CD×100 | yes | no | protocol origin; T2 anchors |
| LGM 2402.05054 | model | Gaussians | user study only (20 ppl) | **no** | ~10 GB claim | per-stage timing labels; VRAM claim to re-test |
| TRELLIS 2412.01506 | model | SLAT+flow | Toys4k 500, 100K pts, r=0.05 | yes | no | metric equations (Eq.13–14) as model |
| SPAR3D 2501.04689 | re-eval | point+GS | GSO ~250, ICP, Time col, **no GPU named** | yes | no | fragmentation evidence; hygiene counterexample |
| InstantMesh 2404.07191 | re-eval | LRM+flexi | GSO 300 + OmniObject3D 130, 16K pts | yes | no | third fragmentation anchor (0.247) |
| Hunyuan3D 2.0/2.1 | model | vec-set flow | user study + Uni3D/ULIP | partial | no | convergence exemplar; anti-cherry-pick language |
| TripoSG 2502.06608 | model | 4B rect-flow | GSO-style | yes | no | convergence exemplar; "no post-processing" caption pattern |
| Sparc3D 2505.14521 | model | sparse VAE | curated hard set + Wild | yes | no | VAE re-eval methodology |
| Direct3D-S2 2505.17412 | model | SSA | runtime-vs-token plot (Fig.7) | yes | no | efficiency-plot style to copy |
| LATTICE 2512.03052 | model | VoxSet | F1 r=0.001 pts-to-surface | yes | no | threshold-chaos evidence |
| 3DGen-Bench 2503.21745 | bench | — | 22 models, 68K votes, Kendall τ | pref | no | benchmark hygiene model |
| WorldScore 2504.00983 | bench | — | 3,000 examples, 2AFC-400 validation | pref | **per-model named GPUs** | instrument-validation gold standard |
| 3D Arena 2506.18787 | bench | — | ELO, 123K votes, zero GT | pref | no | preference channel for F9 correlation |

**The unclaimed intersection (the paper's thesis):** same-protocol geometry comparisons exist in
the regression line but never include structured-latent systems; preference platforms rank all
three but have zero GT and zero efficiency metrics; **nobody reports measured VRAM per engine on
one rig.** We do all three.

---

## 4. CLAIMS → EVIDENCE MAP (every contribution is falsifiable from named artifacts)

**C1 — "Protocol fragmentation is real and large" (2.2× CD spread for the same model+dataset).**
Evidence: T1 (0.111/0.145/0.247, verified [FC]) + our T7 protocol-sensitivity ablation (same
predictions re-scored under squared vs mean CD, with/without ICP) [V28+100-run].

**C2 — "First same-protocol, GT-referenced cross-paradigm measurement including structured
latents, with per-engine peak VRAM."** Evidence: V28 table §1.1 (TripoSR/LGM complete; TRELLIS
per its re-run outcome) + 100-object run.

**C3 — "Published efficiency claims do not survive re-measurement; latency is hardware-relative
and VRAM is the missing axis."** Evidence: TripoSR <0.5 s (A100) vs 3.61 s (T4); LGM ~5 s claim vs
8.39 s measured (and ~10 GB claim vs 5.57 GB measured); SPAR3D's LGM 41 s [FC] — three
independent measurements of one claim, none agreeing.

**C4 — "Preference and geometry disagree" (deployment is multi-objective).** Evidence: our LPIPS
vs CD inversion (LGM better perceptual, worse geometric) [V28] + 3D Arena ELO ranks (TRELLIS
1306 > LGM 1100 > TripoSR 1089) [FC] + TRELLIS OOM as the deployment-binding constraint [V28].

**Anti-claims (never state):** new architecture/dataset; cross-dataset Pareto proof; human-preference
superiority; TRELLIS superiority at 50 steps (we measure 8-step T4-feasible configs and say so).

---

## 5. PAPER BLUEPRINT — 8-page master first, then ICIP 5+1 compression

Target: **ICIP 2027 Datasets & Benchmarks-style track, deadline ~Mar 31 2027** (5 pages + 1 refs).
Fallback: INDICON/ICACCS. Master = 8 pp → compress per PAPER_DESIGN_AUDIT §9.

| § | Content | Source for every sentence |
|---|---|---|
| I Introduction | Problem: three paradigms, irreconcilable numbers; F1 teaser; contributions = C1–C4 (numbered, bolded) | §1.1, §1.2, §3, §4 here |
| II Related work | 4 paragraphs: feed-forward models; 2025 convergence (TripoSG/Hunyuan/Direct3D-S2/Sparc3D/LATTICE all adopted structured latents); evaluation landscape (SPAR3D/InstantMesh re-evals; 3DGen-Bench/WorldScore/3D Arena); the unclaimed intersection | §3 matrix [FC][AUD] |
| III Unified Protocol (the spine) | Dataset: handpicked-100 GSO, criteria (no primitives, no multi-object sets, 10 categories, thin-structure hard cases), CC-BY license. Inputs: white-bg 512² self-rendered. GT: unit-sphere norm, 10K pts. Metric EQUATIONS (symmetric mean CD; F@τ) copied in TRELLIS Eq.13–14 style. Alignment: 12-step yaw + Kabsch ICP for TripoSR only — **asymmetry disclosed**. Hardware/env: T4 15.6 GB, torch 2.11.0+cu128, py3.13, driver, container. Per-engine settings + deviations (TRELLIS 8+8 steps vs 50 default, reason = T4 VRAM; LGM big config, 65,536 Gaussians; TripoSR official repo) | [V28][FC][AUD §7] |
| IV Paradigms | 3 compressed subsections; pipeline figures redrawn with OUR measured stage timings | [FC] arch facts; [V28] timings |
| V Results | T5 main table (mean±std, n, failures) from V28/100-run; efficiency figure (stacked per-stage latency + Pareto, bubble=VRAM); T7 protocol sensitivity; ELO-correlation scatter; qualitative grid + failure figure with anti-cherry-picking captions | [V28] + 100-run |
| VI Discussion | Failure taxonomy (TripoSR: back-side blur, degenerate-MC failure; LGM: multi-view inconsistency, conversion cost; TRELLIS: OOM class, baked lighting); **Limitations** (separate): 10-obj smoke stats, asymmetric alignment, LGM conversion ≈ paper's Sec-3.5 not exact, single GPU class, white-bg GT choice | [V28][MEM][AUD] |
| VII Conclusion + Artifacts | Artifacts: GitHub repo, Kaggle dataset `image-to-3d-code`, result JSONs (Zenodo DOI before submission). Licensing: GSO CC-BY; TripoSR MIT; LGM Apache-2.0; TRELLIS MIT (verify each). Acknowledgments: Dr. Paliwal guide + **CST-UP grant Rs 12,000** + **IEEE-mandated AI-assistance disclosure** | [AUD G24–G28][WEB-IEEE] |

**Figures:** F1 teaser · F2 benchmark-system diagram (our contribution; TikZ, never web-rendered) ·
F3–F5 paradigm pipelines with measured timings · F6 fragmentation dot plot (buildable today) ·
F7 per-object distributions (box/violin — no paper in the corpus shows dispersion) · F8 latency
breakdown + Pareto · F9 ELO-vs-geometry scatter · F10 qualitative grid · F11 failure cases ·
F12 dataset overview (supp).

**Tables:** T1 fragmentation (exists, extend with Time column) · T2 claims-audit (promote §4.3
prose) · T3 TRELLIS-T2 cross-paradigm block (LGM row) · T4 Arena extract · T5 main results ·
T6 per-stage latency · T7 protocol sensitivity 2×2 · T8 failure taxonomy quantified.

---

## 6. MANDATORY TEXT FIXES (from the audits — apply while writing)

1. Delete "176×A100 5 days" (TripoSR) — not in the paper [AUD-F1].
2. Fix TRELLIS-Toys4k table cells: LN3Diff 0.01099/0.9699; CLAY 0.01024 [AUD-F2].
3. LGM meshing stages: Instant-NGP 15 s → MC 45 s (draft had them swapped) [AUD-F3].
4. Terminology: "rectified flow" everywhere; never "rectified linear probability paths" [AUD-F7].
5. One sentence stating r=0.05 (TRELLIS) ≠ FS@0.5 (TripoSR) — incommensurable normalizations [AUD-F9].
6. Abstract must carry OUR numbers (or none) — no "<0.5 s"/"~5 s" unqualified [GAP-5].
7. Every table: n, units, protocol, GPU named [AUD §2-H].
8. Captions of qualitative figures: TripoSG's "no duplicates; no post-processing" pattern [AUD-P].
9. All v33-derived numbers already purged — current main.tex interim numbers must now be replaced
   by V28 values (TripoSR CD 0.282 not 0.093; VRAM 2.50/5.57 not "4–6"/"~10"; LGM geometry no †) —
   pending TRELLIS re-run to do this in one edit.
10. `\cite{kaggle_verdict}`: replace with a real artifact citation (the V28 kernel log / the
    trellis-smoke run) or a footnote to the released logs.

## 7. WRITING ORDER (dependency-true)

1. **Now (no new data needed):** §II related work, §III protocol, T1/T2/T3/T4, F2/F6, all bib
   fixes, artifacts/licensing/ack text, F3–F5 redraw specs.
2. **On TRELLIS smoke completion (in flight):** finalize V28 table; rewrite §IV/V + abstract +
   Pareto from V28; single edit replacing all interim numbers (fix list §6.9–6.10).
3. **On 100-object run:** T5/T6/T7/T8 final cells, F7/F8/F9/F10/F11 data, significance tests.
4. **Compression:** 8-page master → 5+1 ICIP (move T1-ext/T2/T3/T4 + F10/F11 to arXiv version).
5. **Compliance gate:** figures ≥300 dpi or vector; zero undefined refs; overfull <3 pt; PDF
   eXpress; double-anonymize (ICIP) — repo link "anonymized for review", self-citations 3rd person.

## 8. OPEN ITEMS

- TRELLIS 10-object re-run (isolated 2×T4, decode offload) — status checked live; results land in
  `kaggle_output_v29/` (or later) and close the last V28 gap.
- 100-object paper-grade run — gated only on the smoke pass; quota ~30 GPU-h/week is sufficient
  (≈3–4 GPU-h needed for TripoSR+LGM; TRELLIS needs its verdict first).
- User study: out of scope; F9 ELO correlation is the substitute (state it in Limitations).
- Verify each repo license before the artifacts paragraph is frozen.
