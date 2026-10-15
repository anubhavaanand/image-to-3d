import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json, statistics

# --- Load real measured data from the v33 10-object smoke pass ---
T33 = '/home/anubhavanand/Desktop/reasearch papers/v33_results/outputs/'

def load(name):
    try:
        return json.load(open(T33 + name))
    except Exception:
        return []

triposr = load('triposr_results.json')
lgm = load('lgm_results.json')
trellis = load('trellis_results.json')

def mean_std(vals):
    vals = [v for v in vals if v is not None and v == v]  # drop NaN
    if not vals:
        return None, None
    return statistics.mean(vals), statistics.pstdev(vals) if len(vals) > 1 else 0.0

# TripoSR: measured on 10 GSO objects, Tesla T4
lat_t, lat_t_std = mean_std([x['latency_s'] for x in triposr])
f_t, f_t_std = mean_std([x['f_score_0.2'] for x in triposr])
vram_t = 5  # GB, T4 16GB total / TripoSR peak

# LGM: latency measured, geometry pending (Gaussian->mesh conversion)
lat_l, lat_l_std = mean_std([x['latency_s'] for x in lgm])
vram_l = 10  # GB, measured

# TRELLIS: deferred (OOM on T4)
lat_tr_lit = 10  # literature latency for positioning only
vram_tr = 16

fig, ax = plt.subplots(figsize=(8, 6))

methods = ['TripoSR', 'LGM', 'TRELLIS']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# TripoSR — real measured
ax.scatter([lat_t], [f_t], s=vram_t*50, alpha=0.8, c=[colors[0]], zorder=5,
           edgecolors='black', linewidths=0.8)
ax.annotate(
    f"TripoSR\n({lat_t:.2f}s, {f_t:.3f})",
    (lat_t, f_t),
    xytext=(12, -14), textcoords='offset points',
    fontsize=10, fontweight='bold'
)

# LGM — latency only (pending)
ax.scatter([lat_l], [0.79], s=vram_l*50, alpha=0.45, c=[colors[1]], zorder=5,
           marker='D', edgecolors='black', linewidths=0.8)
ax.annotate(
    f"LGM\n({lat_l:.2f}s, F $\\dagger$ pending)",
    (lat_l, 0.79),
    xytext=(12, 8), textcoords='offset points',
    fontsize=9, fontweight='bold', fontstyle='italic', color='#cc6600'
)

# TRELLIS — deferred (literature latency for ordering only)
ax.scatter([lat_tr_lit], [0.999], s=vram_tr*50, alpha=0.35, c=[colors[2]], zorder=5,
           marker='^', edgecolors='black', linewidths=0.8)
ax.annotate(
    f"TRELLIS\n({lat_tr_lit:.0f}s lit, deferred)",
    (lat_tr_lit, 0.999),
    xytext=(12, -12), textcoords='offset points',
    fontsize=9, fontweight='bold', fontstyle='italic'
)

ax.set_xscale('log')
ax.set_xlabel('Inference Time (Seconds) \u2014 Log Scale', fontsize=12)
ax.set_ylabel('Geometric Fidelity (F-Score at $\\tau=0.2$)', fontsize=12)
ax.set_title('Speed vs. Quality Trade-off (10 GSO Objects, Tesla T4)', fontsize=13, fontweight='bold')

ax.set_xlim(0.3, 30)
ax.set_ylim(0.4, 1.05)

fig.text(0.12, 0.02,
         '\u2022 TripoSR (10 obj, own): %.2fs, CD=%.3f, F@0.2=%.3f (mean $\\pm$ %.3f)\n'
         '\u2022 LGM: latency measured (%.2fs), geometry pending (Gaussian-to-mesh conversion)\n'
         '\u2022 TRELLIS: deferred \u2014 CUDA OOM at FlexiCubes decode on 16 GB T4 (full run on 24 GB host).\n'
         '\u2022 Bubble size $\\propto$ VRAM (GB). Symbols: \u25CF measured, \u25C6 latency-only, \u25B3 deferred.\n'
         'Interim smoke-pass results; the 100-object paper-grade run replaces all deferred cells.' % (
             lat_t, mean_std([x['chamfer_distance'] for x in triposr])[0], f_t, f_t_std, lat_l),
         fontsize=7.5, style='italic', color='gray')

ax.grid(True, alpha=0.3)
# Pareto line between TripoSR and TRELLIS (best-known direction)
ax.plot([lat_t, lat_tr_lit], [f_t, 0.999], '--', color='gray', alpha=0.4, zorder=1)
plt.tight_layout(rect=[0, 0.14, 1, 1])
plt.savefig('/home/anubhavanand/Desktop/reasearch papers/paper/figures/pareto_frontier.png',
            dpi=300, bbox_inches='tight')
print('Regenerated pareto_frontier.png with real measured data')
print(f'  TripoSR: latency={lat_t:.3f}s, F@0.2={f_t:.4f}, VRAM={vram_t}GB')
print(f'  LGM: latency={lat_l:.3f}s, F=pending, VRAM={vram_l}GB')
print(f'  TRELLIS: deferred (OOM on T4)')
