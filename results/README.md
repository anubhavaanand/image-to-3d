# Results

This directory holds the measured benchmark data. Keep the canonical files; treat the rest as run history.

## Canonical files

| File | Rows | What it is |
|---|---:|---|
| `unified_results_4engines.csv` | 40 | Canonical-10 subset: 10 GSO objects × 4 engines, filtered from the 400-row run. Not 40 additional evaluations. |
| `unified_results_4engines_100obj.csv` | 400 | Full run: 100 objects × 4 engines (the paper's n = 100 block) |
| `unified_table_4engines.tex` | — | LaTeX table generated from the canonical-10 CSV |
| `unified_table_4engines_100obj.tex` | — | LaTeX table generated from the n = 100 CSV |

Both CSVs are derived from the same final 400 evaluations; the 40-row file is a filtered canonical-10 subset. They are the source of every number in the manuscript.

## Provenance of the n = 100 block

Both final kernels ran at code commit `09131e61568d98401e71861bc27fdce6637b2f20` (each output directory contains a `code_sha.txt`). The run directories are named `batch1_100obj_v35` and `batch2_100obj_v35`; both produced 100 rows per engine (400 unique evaluations, 0 failures). The 10-object block in the paper is the canonical subset of those 400 rows, not a separate 40-evaluation run.

## Reproducing

The canonical CSVs were built from the final Kaggle run outputs, which live outside this repository:

```bash
python scripts/merge_batches.py ../kaggle_runs/batch1_100obj_v35/outputs \
                                 ../kaggle_runs/batch2_100obj_v35/outputs
python scripts/rebuild_figures.py
```

`merge_batches.py` collects `*_results.json` files, de-duplicates by `(engine, sample_id)`, and writes the unified CSV/TeX plus the Pareto figure. The canonical `results/` directory now contains only the final data and the derived canonical-10 subset; superseded run dumps were removed on 2026-10-31.

## Columns

`engine`, `sample_id`, `latency_s`, `vram_peak_gb`, `chamfer_distance`, `f_score_0.1`, `f_score_0.2`, `f_score_0.5`, `psnr`, `lpips`, `metadata`, `vertex_count`, `face_count`, `mesh_size_mb`.
