# Results

This directory holds the measured benchmark data. Keep the canonical files; treat the rest as run history.

## Canonical files

| File | Rows | What it is |
|---|---:|---|
| `unified_results_4engines.csv` | 40 | Canonical-10 run: 10 GSO objects × 4 engines, with latency, peak VRAM, CD, F-Score, PSNR, LPIPS, and mesh statistics |
| `unified_results_4engines_100obj.csv` | 400 | Full run: 100 objects × 4 engines (the paper's n = 100 block) |
| `unified_table_4engines.tex` | — | LaTeX table generated from the canonical-10 CSV |
| `unified_table_4engines_100obj.tex` | — | LaTeX table generated from the n = 100 CSV |

Both CSVs are produced by `scripts/rebuild_figures.py` / the Kaggle kernels and are the source of every number in the manuscript.

## Reproducing

The canonical CSVs were built from the final Kaggle run outputs, which live outside this repository:

```bash
python scripts/merge_batches.py ../kaggle_runs/final_batch1_100obj/outputs \
                                 ../kaggle_runs/final_batch2_100obj/outputs
python scripts/rebuild_figures.py
```

`merge_batches.py` collects `*_results.json` files, de-duplicates by `(engine, sample_id)`, and writes the unified CSV/TeX plus the Pareto figure. The `results/batch*/` directories kept in git are partial historical copies and do not contain all four engines for every run.

## Run history (not tracked)

`batch1_v17/`, `batch2_v24/`, `srgb_b1_*`, `srgb_gt/`, `srgb_v33_spar3d/`, `srgb_v34_spar3d/` contain intermediate JSON and render dumps from protocol debugging (several GB of PNGs). They are ignored by git and kept locally for audit only — do not cite numbers from them.

## Columns

`engine`, `sample_id`, `latency_s`, `vram_peak_gb`, `chamfer_distance`, `f_score_0.1`, `f_score_0.2`, `f_score_0.5`, `psnr`, `lpips`, `metadata`, `vertex_count`, `face_count`, `mesh_size_mb`.
