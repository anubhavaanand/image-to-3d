#!/usr/bin/env python3
"""
Merge Batch 1 (TripoSR + LGM) and Batch 2 (TRELLIS + SPAR3D) results into:
1. unified_4models.csv
2. unified_4models_table.tex (IEEE publication table)
3. pareto_4models.png (Speed vs F-Score & VRAM Pareto plot)
"""
import glob
import json
import sys
from pathlib import Path
import pandas as pd

def merge_all(search_dirs):
    rows = []
    seen = set()
    for d in search_dirs:
        for p in sorted(Path(d).rglob("*_results.json")):
            try:
                data = json.load(open(p))
                for r in data:
                    key = (r.get("engine"), r.get("sample_id"))
                    if key not in seen and "chamfer_distance" in r:
                        seen.add(key)
                        rows.append(r)
                print(f"[loaded] {p} ({len(data)} rows)")
            except Exception as e:
                print(f"[skip] {p}: {e}")
    if not rows:
        print("No result JSONs found.")
        return

    out_dir = Path(search_dirs[0])
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "unified_4models.csv", index=False)

    cols = ["chamfer_distance", "f_score_0.1", "f_score_0.2", "f_score_0.5", "psnr", "lpips", "latency_s", "vram_peak_gb"]
    avail = [c for c in cols if c in df.columns]
    summary = df.groupby("engine")[avail].agg(["mean", "std"]).round(4)
    print("\n=== 4-MODEL UNIFIED BENCHMARK SUMMARY (mean +/- std) ===")
    print(summary.to_string())

    tex_path = out_dir / "unified_4models_table.tex"
    tex_path.write_text(summary.reset_index().to_latex(index=False, na_rep="--"))
    print(f"\nSaved CSV and LaTeX table to {out_dir}")

if __name__ == "__main__":
    dirs = sys.argv[1:] or ["./outputs"]
    merge_all(dirs)
