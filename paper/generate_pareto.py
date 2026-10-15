#!/usr/bin/env python3
"""
Pareto/efficiency figure from MEASURED V28 data (kaggle_output_v28, post-commit 98f5937).
Tesla T4 (15.6 GB), 10 GSO objects. No literature numbers, no cross-dataset proxies.

Panel A: latency (log-x) vs geometric fidelity (F@0.2), bubble = peak VRAM.
Panel B: latency vs perceptual quality (1 - LPIPS).
TRELLIS: single-T4 OOM region (14.56 GiB) marked; no measured point (0/10 scored).
Output: paper/figures/pareto_frontier.png at 300 DPI.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Measured V28 values (verified against kaggle_output_v28/outputs/*.json) ---
triposr = {"name": "TripoSR", "lat": 3.61, "lat_err": 0.28, "vram": 2.50,
           "f02": 0.702, "f02_err": 0.236, "one_mlips": 1 - 0.412, "n": "9/10"}
lgm = {"name": "LGM", "lat": 8.39, "lat_err": 0.25, "vram": 5.57,
       "f02": 0.379, "f02_err": 0.221, "one_mlips": 1 - 0.285, "n": "10/10"}

OOM_GIB = 14.56  # single-T4 usable ceiling; TRELLIS FlexiCubes decode exceeds it (10/10 OOM)

C_TRIPOSr = "#2b83ba"
C_LGM = "#f57c00"
C_OOM = "#d7191c"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.9))  # IEEE two-column width

for ax, ykey, ylabel in (
    (ax1, "f02", "Geometric fidelity: F@0.2 $\\uparrow$"),
    (ax2, "one_mlips", "Perceptual quality: $1-$LPIPS $\\uparrow$"),
):
    # bubbles (area ~ VRAM)
    for e, c in ((triposr, C_TRIPOSr), (lgm, C_LGM)):
        ax.errorbar(e["lat"], e[ykey], xerr=e["lat_err"],
                    yerr=e["f02_err"] if ykey == "f02" else None,
                    fmt="o", ms=e["vram"] * 3.2, mfc=c, mec="black", mew=0.8,
                    ecolor="gray", elinewidth=1, capsize=3, zorder=3,
                    label=f'{e["name"]} ({e["n"]}, {e["vram"]:.2f} GB)')
        ax.annotate(e["name"], (e["lat"], e[ykey]), textcoords="offset points",
                    xytext=(8, 8), fontsize=8, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(2, 26)
    # clean log ticks: major at 2,4,6,8,10,20; no minor labels (they garbled the axis)
    import matplotlib.ticker as mticker
    ax.xaxis.set_major_locator(mticker.FixedLocator([2, 4, 6, 8, 10, 20]))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(["2", "4", "6", "8", "10", "20"]))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.minorticks_off()
    if ykey == "f02":
        ax.set_ylim(0.0, 1.0)
    else:
        ax.set_ylim(0.4, 0.85)
    ax.set_xlabel("Measured wall-clock latency (s, log scale) $\\downarrow$")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", alpha=0.25, zorder=0)
    ax.tick_params(labelsize=8)

# annotate the TRELLIS OOM constraint on both panels
for ax in (ax1, ax2):
    ax.axvline(25.5, color=C_OOM, linestyle="--", alpha=0.85, linewidth=1.4)
    ax.annotate("TRELLIS: 10/10 OOM\n$256^3$ FlexiCubes\n> 14.56 GiB\n(needs $\\geq$24 GB GPU)",
                xy=(0.02, 0.03), xycoords="axes fraction", ha="left", va="bottom",
                fontsize=7, color=C_OOM,
                bbox=dict(boxstyle="round,pad=0.3", fc="#fdecea", ec=C_OOM, lw=0.8))

ax1.legend(loc="upper right", fontsize=7.5, framealpha=0.95)
ax2.annotate("Bubble area $\\propto$ measured peak VRAM",
             xy=(0.02, 0.97), xycoords="axes fraction", ha="left", va="top", fontsize=7,
             style="italic", color="0.35")

fig.suptitle("Measured efficiency--fidelity trade-off, NVIDIA Tesla T4 (10 curated GSO objects)",
             fontsize=9, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.94])

out = os.path.join(os.path.dirname(__file__), "figures", "pareto_frontier.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"saved {out} at 300 DPI")
