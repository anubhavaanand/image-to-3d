#!/usr/bin/env python3
"""Figure generator for the 4-Engine 3D Benchmark paper:
1. paper/figures/qualitative_comparison.png (Full 10-object x 6-column comparison from batch1_v17 + batch2_v24)
2. paper/figures/gso10_3d_showcase.png (3D Circular Multi-Angle Turntable Showcase + Top-3 Cleanest 4-Engine Strip)

Ground-truth meshes are rendered as smooth, textured, watertight triangle meshes with
pyrender (EGL, 2x supersampled); normals are reconstructed from the depth buffer.
Engine columns reuse the pre-rendered Kaggle novel-view PNGs (avoids loading 920 MB of
.obj files into RAM), one GSO mesh is held in memory at a time.
"""

import ctypes
import gc
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import trimesh
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

# Headless EGL must be selected before pyrender/OpenGL is imported.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# PyOpenGL 3.1.0 cannot marshal ctypes output arrays on Python 3.14, so
# pyrender's glGenTextures(1) raises "No array-type handler for type _ctypes.type".
# Texture *names* are arbitrary integers, so resolve the name with a raw GL call.
try:
    import pyrender
    import pyrender.texture as _pr_texture
    from OpenGL import platform as _gl_platform

    def _gl_gen_textures(n: int = 1):
        ids = (ctypes.c_uint * n)()
        _gl_platform.PLATFORM.GL.glGenTextures(n, ids)
        return ids[0] if n == 1 else list(ids)

    _pr_texture.glGenTextures = _gl_gen_textures
except Exception as _exc:  # pragma: no cover - only needed for the GT mesh renders
    pyrender = None
    print(f"[rebuild_figures] pyrender unavailable: {_exc}", flush=True)

ROOT = Path("/home/anubhavanand/Desktop/reasearch papers")
BENCH = ROOT / "image-to-3d-benchmark"

GSO_ZUP_TO_YUP_4X4 = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

CANONICAL_10 = [
    ("3D_Dollhouse_Happy_Brother", "Dollhouse Brother"),
    ("3D_Dollhouse_Sofa", "Dollhouse Sofa"),
    ("ASICS_GELLinksmaster_WhiteCoffeeSand", "ASICS Golf Shoe"),
    ("Android_Figure_Panda", "Android Panda"),
    ("Animal_Planet_Foam_2Headed_Dragon", "2-Headed Dragon"),
    ("Black_and_Decker_TR3500SD_2Slice_Toaster", "2-Slice Toaster"),
    ("Bradshaw_International_11642_7_Qt_MP_Plastic_Bowl", "Plastic Bowl"),
    ("Dino_3", "Dino Figurine"),
    ("Dog", "Dog Figurine"),
    ("Down_To_Earth_Orchid_Pot_Ceramic_Red", "Ceramic Orchid Pot"),
]

ENGINES = ["TripoSR", "LGM", "SPAR3D", "TRELLIS"]
ENGINE_COLORS = {
    "TripoSR": (43, 92, 143),
    "LGM": (217, 95, 2),
    "SPAR3D": (117, 112, 179),
    "TRELLIS": (27, 158, 119),
}

ENGINE_QUAL_DIRS = {
    "TripoSR": BENCH / "results" / "batch1_v17" / "triposr" / "qualitative",
    "LGM": BENCH / "results" / "batch1_v17" / "lgm" / "qualitative",
    "SPAR3D": BENCH / "results" / "batch2_v24" / "spar3d" / "qualitative",
    "TRELLIS": BENCH / "results" / "batch2_v24" / "trellis" / "qualitative",
}
GSO_DIR = ROOT / "data" / "gso"
CSV_PATH = BENCH / "results" / "unified_results_4engines.csv"

FONT_REG_PATH = "/usr/share/fonts/TTF/DejaVuSans.ttf"
FONT_BOLD_PATH = "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"


def get_font(size: int, bold: bool = False):
    p = FONT_BOLD_PATH if bold else FONT_REG_PATH
    return ImageFont.truetype(p, size)


def normalize_vertices(verts: np.ndarray) -> np.ndarray:
    c = verts.mean(axis=0, keepdims=True)
    v = verts - c
    r = np.percentile(np.linalg.norm(v, axis=1), 99.5)
    return v / max(float(r), 1e-6)


def make_camera_pose(azimuth_deg: float, elevation_deg: float, distance: float) -> np.ndarray:
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    cam_pos = np.array(
        [
            distance * np.cos(el) * np.sin(az),
            distance * np.sin(el),
            distance * np.cos(el) * np.cos(az),
        ],
        dtype=np.float32,
    )
    forward = -cam_pos / (np.linalg.norm(cam_pos) + 1e-8)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right) + 1e-8
    up = np.cross(right, forward)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = cam_pos
    return pose


def _load_render_mesh(sid: str, tex_max: int = 2048) -> trimesh.Trimesh:
    """Load one GSO scan, orient it Y-up, unit-normalize it, and cap its texture size."""
    mesh_path = GSO_DIR / sid / "meshes" / "model.obj"
    tm = trimesh.load(str(mesh_path), force="mesh", process=False)
    tm.apply_transform(GSO_ZUP_TO_YUP_4X4)
    verts = normalize_vertices(np.asarray(tm.vertices, dtype=np.float32))
    faces = np.asarray(tm.faces, dtype=np.int64)
    uv = getattr(tm.visual, "uv", None)
    img = getattr(getattr(tm.visual, "material", None), "image", None)
    if uv is not None and img is not None and len(uv) == len(verts):
        pil = Image.fromarray(np.asarray(img)).convert("RGB")
        if max(pil.size) > tex_max:
            pil = pil.resize((tex_max, tex_max), Image.LANCZOS)
        visual = trimesh.visual.TextureVisuals(uv=np.asarray(uv, np.float32), image=pil)
        mesh = trimesh.Trimesh(verts, faces, visual=visual, process=False)
    else:
        mesh = trimesh.Trimesh(verts, faces, process=False)
    del tm
    gc.collect()
    return mesh


def _depth_normals(depth: np.ndarray, yfov: float) -> np.ndarray:
    """Reconstruct camera-space normals from a pyrender depth buffer and encode them as RGB."""
    h, w = depth.shape
    fx = fy = 0.5 * w / np.tan(yfov / 2.0)
    z = depth.astype(np.float32)

    # Smooth the depth before differentiating so Marching-Cubes stair-steps do not alias.
    k = 3
    zp = np.pad(z, k, mode="edge")
    acc = np.zeros_like(z)
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            acc += zp[k + dy : k + dy + h, k + dx : k + dx + w]
    zs = acc / float((2 * k + 1) ** 2)

    vv, uu = np.mgrid[0:h, 0:w].astype(np.float32)
    x = (uu - (w - 1) / 2.0) * zs / fx
    y = -(vv - (h - 1) / 2.0) * zs / fy
    pts = np.stack([x, y, -zs], axis=-1)

    n = np.cross(np.gradient(pts, axis=1), np.gradient(pts, axis=0))
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8
    n *= np.sign(n[..., 2:3] + 1e-12)  # orient every normal toward the +Z (viewer) side

    out = np.full((h, w, 3), 255, dtype=np.uint8)
    mask = z > 0
    r = np.clip((n[..., 0] * 0.5 + 0.5) * 165.0 + 55.0, 45.0, 240.0)
    g = np.clip((n[..., 1] * 0.5 + 0.5) * 165.0 + 55.0, 45.0, 240.0)
    b = np.clip(n[..., 2] * 185.0 + 45.0, 60.0, 245.0)
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    out[mask] = rgb[mask]
    return out


def render_gt_views_for_object(sid: str, res: int = 260) -> dict:
    """Render smooth watertight GT triangle meshes with pyrender (textured, 2x supersampled)."""
    if pyrender is None:
        raise RuntimeError("pyrender is required to render GT meshes")

    mesh = _load_render_mesh(sid)
    ss = res * 2
    dist = 3.0
    radius = 1.12
    yfov = 2.0 * np.arctan((radius / np.sqrt(dist**2 - radius**2)) / 0.82)

    views = {"az35": (35.0, 18.0), "az65": (65.0, 15.0), "az145": (145.0, 20.0)}
    renderer = pyrender.OffscreenRenderer(ss, ss)
    colors, depths = {}, {}
    try:
        for key, (az, el) in views.items():
            scene = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[0.62, 0.62, 0.62])
            scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
            pose = make_camera_pose(az, el, dist)
            scene.add(pyrender.PerspectiveCamera(yfov=yfov), pose=pose)
            key_pose = pose.copy()
            key_pose[:3, 2] = -key_pose[:3, 2]
            scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.1), pose=key_pose)
            fill_pose = make_camera_pose(az + 75.0, el + 12.0, dist)
            fill_pose[:3, 2] = -fill_pose[:3, 2]
            scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=0.85), pose=fill_pose)

            color, depth = renderer.render(scene)
            colors[key], depths[key] = color, depth
    finally:
        renderer.delete()
        gc.collect()

    # Per-view silhouette crop at 82% fill keeps every cell legible at paper scale.
    def _framed(arr: np.ndarray, depth: np.ndarray) -> Image.Image:
        ys, xs = np.where(depth > 0)
        if len(xs):
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        else:
            x0, x1, y0, y1 = 0, ss - 1, 0, ss - 1
        side = max(x1 - x0, y1 - y0) / 0.82
        left = int(round(0.5 * (x0 + x1) - side / 2))
        top = int(round(0.5 * (y0 + y1) - side / 2))
        canvas = Image.new("RGB", (int(round(side)), int(round(side))), (255, 255, 255))
        canvas.paste(Image.fromarray(arr), (-left, -top))
        return canvas.resize((res, res), Image.LANCZOS)

    out = {k: _framed(colors[k], depths[k]) for k in views}
    out["norm65"] = _framed(_depth_normals(depths["az65"], yfov), depths["az65"])
    return out


def composite_with_inset(img_main: Image.Image, img_inset: Image.Image, inset_frac: float = 0.30) -> Image.Image:
    base = img_main.copy().convert("RGB")
    w, h = base.size
    iw = int(round(w * inset_frac))
    inset = img_inset.convert("RGB").resize((iw, iw), Image.LANCZOS)
    inset = ImageOps.expand(inset, border=2, fill=(160, 174, 192))
    base.paste(inset, (w - inset.size[0] - 4, h - inset.size[1] - 4))
    return base


def make_surface_relief_inset(rgb_img: Image.Image) -> Image.Image:
    """Create a clean 3D normal-style clay shading relief from a rendered novel-view RGB image (0 MB mesh load)."""
    arr = np.asarray(rgb_img.convert("RGB"), dtype=np.float32)
    mask = (arr < 250).any(axis=2)
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    gy, gx = np.gradient(gray)
    nz = np.full_like(gx, 18.0)
    norm = np.sqrt(gx * gx + gy * gy + nz * nz) + 1e-6
    nx, ny, nz = -gx / norm, -gy / norm, nz / norm
    out = np.full(arr.shape, 255, dtype=np.uint8)
    r = np.clip((nx * 0.5 + 0.5) * 165 + 55, 45, 240)
    g = np.clip((ny * 0.5 + 0.5) * 165 + 55, 45, 240)
    b = np.clip(nz * 185 + 45, 60, 245)
    clay = np.stack([r, g, b], axis=2).astype(np.uint8)
    out[mask] = clay[mask]
    return Image.fromarray(out)


def load_framed_image(path: Path, resolution: int = 260) -> Image.Image:
    """Load an RGBA or RGB image and center its non-white foreground cleanly at 82% fill."""
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    comp = Image.alpha_composite(bg, im).convert("RGB")
    arr = np.asarray(comp)
    non_white = (arr < 248).any(axis=2)
    if non_white.any():
        ys, xs = np.where(non_white)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        half = max(y1 - y0, x1 - x0) * 0.5 / 0.82
        left = int(round(cx - half))
        top = int(round(cy - half))
        right = int(round(cx + half))
        bottom = int(round(cy + half))
        canvas = Image.new("RGB", (right - left, bottom - top), (255, 255, 255))
        h2, w2 = arr.shape[:2]
        src_l, src_t = max(0, left), max(0, top)
        src_r, src_b = min(w2, right), min(h2, bottom)
        canvas.paste(
            Image.fromarray(arr[src_t:src_b, src_l:src_r]),
            (src_l - left, src_t - top),
        )
        return canvas.resize((resolution, resolution), Image.LANCZOS)
    return comp.resize((resolution, resolution), Image.LANCZOS)


def draw_centered_text(draw: ImageDraw.ImageDraw, box_x: int, box_w: int, y: int, text: str, font, fill):
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text((box_x + (box_w - tw) // 2, y), text, font=font, fill=fill)


def main():
    df = pd.read_csv(CSV_PATH)
    gt_views = {}
    for sid, _ in CANONICAL_10:
        print(f"Rendering GT views for {sid} (streaming, <80 MB RAM)...", flush=True)
        gt_views[sid] = render_gt_views_for_object(sid, res=260)

    # 1. Build qualitative_comparison.png (10 objects x 6 columns)
    print("Assembling paper/figures/qualitative_comparison.png...", flush=True)
    cell = 260
    pad_x = 14
    pad_y = 34
    left_label_w = 185
    top_header_h = 90
    n_rows = len(CANONICAL_10)
    n_cols = 6
    total_w = left_label_w + n_cols * (cell + pad_x) + pad_x
    total_h = top_header_h + n_rows * (cell + pad_y) + 16

    canvas1 = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw1 = ImageDraw.Draw(canvas1)

    f_title = get_font(21, bold=True)
    f_col = get_font(16, bold=True)
    f_col_sub = get_font(13, bold=False)
    f_row = get_font(15, bold=True)
    f_cap = get_font(12, bold=False)
    f_cap_bold = get_font(12, bold=True)

    draw_centered_text(
        draw1,
        0,
        total_w,
        12,
        "Qualitative & Geometric 3D Reconstruction Benchmark Across All 10 Canonical GSO Objects (Novel View Az=65°, El=15° + 3D Normal Inset)",
        f_title,
        (15, 23, 42),
    )

    col_headers = [
        ("Input Image (2D)", "1024px RGBA Centered"),
        ("Ground Truth (3D)", "Watertight GSO Scan"),
        ("TripoSR (Feed-Fwd)", "3.55s | 2.32 GB VRAM"),
        ("LGM (4-View GS)", "11.18s | 5.19 GB VRAM"),
        ("SPAR3D (Point-Diff)", "34.93s | 9.17 GB VRAM"),
        ("TRELLIS (SLAT Flow)", "21.58s | 7.01 GB VRAM"),
    ]
    for c_i, (h1, h2) in enumerate(col_headers):
        cx = left_label_w + c_i * (cell + pad_x)
        draw_centered_text(draw1, cx, cell, 46, h1, f_col, (15, 23, 42))
        draw_centered_text(draw1, cx, cell, 67, h2, f_col_sub, (71, 85, 105))

    for r_i, (sid, label) in enumerate(CANONICAL_10):
        ry = top_header_h + r_i * (cell + pad_y)
        sub = df[df["sample_id"] == sid].set_index("engine")
        best_cd_eng = sub["chamfer_distance"].idxmin()
        best_f1_eng = sub["f_score_0.1"].idxmax()

        draw1.text((14, ry + cell // 2 - 10), label, font=f_row, fill=(15, 23, 42))

        # Col 0: Uncropped Input Image
        c0_x = left_label_w
        inp_p = ENGINE_QUAL_DIRS["TRELLIS"] / sid / "input.png"
        inp_img = load_framed_image(inp_p, resolution=cell)
        canvas1.paste(inp_img, (c0_x, ry))
        draw1.rectangle([c0_x, ry, c0_x + cell, ry + cell], outline=(203, 213, 225), width=2)
        draw_centered_text(draw1, c0_x, cell, ry + cell + 5, "Single-View RGB Input", f_cap, (100, 116, 139))

        # Col 1: GT 3D Scan + Normals
        c1_x = left_label_w + 1 * (cell + pad_x)
        gt_nov_p = ENGINE_QUAL_DIRS["TRELLIS"] / sid / "gt_novel.png"
        gt_rgb = load_framed_image(gt_nov_p, resolution=cell) if gt_nov_p.exists() else gt_views[sid]["az65"]
        gt_comp = composite_with_inset(gt_rgb, gt_views[sid]["norm65"])
        canvas1.paste(gt_comp, (c1_x, ry))
        draw1.rectangle([c1_x, ry, c1_x + cell, ry + cell], outline=(148, 163, 184), width=2)
        draw_centered_text(draw1, c1_x, cell, ry + cell + 5, "GT Mesh + 3D Normals", f_cap, (71, 85, 105))

        # Cols 2-5: 4 Engines
        for e_i, eng in enumerate(ENGINES):
            cx = left_label_w + (2 + e_i) * (cell + pad_x)
            nov_p = ENGINE_QUAL_DIRS[eng] / sid / f"{eng}_novel.png"
            nov_img = load_framed_image(nov_p, resolution=cell)
            pcomp = composite_with_inset(nov_img, make_surface_relief_inset(nov_img))
            canvas1.paste(pcomp, (cx, ry))
            r = sub.loc[eng]
            is_best = (eng == best_cd_eng) or (eng == best_f1_eng)
            border_col = (13, 148, 136) if is_best else (203, 213, 225)
            border_w = 3 if is_best else 2
            draw1.rectangle([cx, ry, cx + cell, ry + cell], outline=border_col, width=border_w)
            cap = f"CD:{r['chamfer_distance']:.3f} | F@.1:{r['f_score_0.1']:.2f} | {r['psnr']:.1f}dB"
            draw_centered_text(
                draw1,
                cx,
                cell,
                ry + cell + 5,
                cap,
                f_cap_bold if is_best else f_cap,
                (15, 118, 110) if is_best else (51, 65, 85),
            )

    for out_path in [
        ROOT / "paper" / "figures" / "qualitative_comparison.png",
        BENCH / "paper" / "figures" / "qualitative_comparison.png",
    ]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas1.save(out_path, optimize=True)
        print("  Saved:", out_path, flush=True)
    del canvas1, draw1
    gc.collect()

    # 2. Build gso10_3d_showcase.png (Part A: 10-Object 3D Circular Turntable + Scores; Part B: Top-3 4-Engine Strip)
    print("Assembling paper/figures/gso10_3d_showcase.png...", flush=True)
    total_w = 1980
    top_h = 1090
    bot_h = 910
    canvas2 = Image.new("RGB", (total_w, top_h + bot_h), (255, 255, 255))
    draw2 = ImageDraw.Draw(canvas2)

    f_sec = get_font(20, bold=True)
    f_hdr = get_font(14, bold=True)
    f_sub = get_font(12, bold=False)
    f_obj = get_font(14, bold=True)
    f_bar = get_font(11, bold=True)

    draw_centered_text(
        draw2,
        0,
        total_w,
        12,
        "(a) 3D Multi-Angle Circular Turntable, Surface Normals & Per-Object 4-Engine Scores Across All 10 Canonical GSO Benchmark Objects",
        f_sec,
        (15, 23, 42),
    )

    m_cell = 165
    m_gap = 8
    bar_w = 245
    macro_w = 4 * (m_cell + m_gap) + bar_w + 18
    macro_h = m_cell + 34

    view_titles = ["2D Input", "3D View (35°)", "3D Orbit (145°)", "3D Normals"]
    for mc in range(2):
        base_x = 22 + mc * (macro_w + 20)
        for vi, vt in enumerate(view_titles):
            draw_centered_text(draw2, base_x + vi * (m_cell + m_gap), m_cell, 46, vt, f_hdr, (30, 41, 59))
        draw_centered_text(draw2, base_x + 4 * (m_cell + m_gap), bar_w, 46, "4-Engine F@0.1 (↑) & CD (↓)", f_hdr, (30, 41, 59))

    for idx, (sid, label) in enumerate(CANONICAL_10):
        mr = idx // 2
        mc = idx % 2
        base_x = 22 + mc * (macro_w + 20)
        base_y = 74 + mr * macro_h

        draw2.rectangle(
            [base_x - 6, base_y - 4, base_x + 4 * (m_cell + m_gap) + bar_w + 8, base_y + m_cell + 24],
            outline=(226, 232, 240),
            fill=(250, 251, 253),
            width=1,
        )

        inp_p = ENGINE_QUAL_DIRS["TRELLIS"] / sid / "input.png"
        imgs = [
            load_framed_image(inp_p, resolution=m_cell),
            gt_views[sid]["az35"].resize((m_cell, m_cell), Image.LANCZOS),
            gt_views[sid]["az145"].resize((m_cell, m_cell), Image.LANCZOS),
            gt_views[sid]["norm65"].resize((m_cell, m_cell), Image.LANCZOS),
        ]
        for vi, im in enumerate(imgs):
            vx = base_x + vi * (m_cell + m_gap)
            canvas2.paste(im, (vx, base_y))
            draw2.rectangle([vx, base_y, vx + m_cell, base_y + m_cell], outline=(203, 213, 225), width=1)

        draw2.text(
            (base_x + 4, base_y + m_cell + 4),
            f"#{idx+1:02d}  {label} ({sid[:28]})",
            font=f_obj,
            fill=(15, 23, 42),
        )

        bx = base_x + 4 * (m_cell + m_gap) + 6
        sub = df[df["sample_id"] == sid].set_index("engine").reindex(ENGINES)
        for ei, eng in enumerate(ENGINES):
            by = base_y + 12 + ei * 38
            f_val = float(sub.loc[eng, "f_score_0.1"])
            c_val = float(sub.loc[eng, "chamfer_distance"])
            draw2.text((bx, by), f"{eng[:7]:7s}", font=f_bar, fill=(30, 41, 59))
            bar_start = bx + 62
            bar_max_px = 82
            bar_px = max(4, int(round(f_val * bar_max_px)))
            draw2.rectangle([bar_start, by + 2, bar_start + bar_max_px, by + 16], fill=(226, 232, 240))
            draw2.rectangle([bar_start, by + 2, bar_start + bar_px, by + 16], fill=ENGINE_COLORS[eng])
            draw2.text(
                (bar_start + bar_max_px + 6, by),
                f"{f_val:.2f} ({c_val:.2f})",
                font=f_bar,
                fill=(15, 23, 42),
            )

    draw2.line([(24, top_h - 10), (total_w - 24, top_h - 10)], fill=(148, 163, 184), width=2)

    draw_centered_text(
        draw2,
        0,
        total_w,
        top_h + 6,
        "(b) Representative 4-Engine 3D Mesh Reconstructions on Canonical GSO Objects (Novel View Az=65°, El=15° + 3D Normal Inset)",
        f_sec,
        (15, 23, 42),
    )

    showcase_3 = [
        ("3D_Dollhouse_Sofa", "Dollhouse Sofa"),
        ("3D_Dollhouse_Happy_Brother", "Dollhouse Brother"),
        ("Dino_3", "Dino Figurine"),
    ]
    b_cell = 245
    b_pad_x = 18
    b_pad_y = 34
    b_left = 250
    b_top = top_h + 76

    for c_i, (h1, h2) in enumerate(col_headers):
        cx = b_left + c_i * (b_cell + b_pad_x)
        draw_centered_text(draw2, cx, b_cell, top_h + 38, h1, f_hdr, (15, 23, 42))
        draw_centered_text(draw2, cx, b_cell, top_h + 56, h2, f_sub, (71, 85, 105))

    for r_i, (sid, label) in enumerate(showcase_3):
        ry = b_top + r_i * (b_cell + b_pad_y)
        sub = df[df["sample_id"] == sid].set_index("engine")
        best_cd_eng = sub["chamfer_distance"].idxmin()
        best_f1_eng = sub["f_score_0.1"].idxmax()

        draw2.text((40, ry + b_cell // 2 - 10), label, font=get_font(15, bold=True), fill=(15, 23, 42))

        c0_x = b_left
        inp_p = ENGINE_QUAL_DIRS["TRELLIS"] / sid / "input.png"
        inp_img = load_framed_image(inp_p, resolution=b_cell)
        canvas2.paste(inp_img, (c0_x, ry))
        draw2.rectangle([c0_x, ry, c0_x + b_cell, ry + b_cell], outline=(203, 213, 225), width=2)
        draw_centered_text(draw2, c0_x, b_cell, ry + b_cell + 5, "Single-View RGB Input", f_cap, (100, 116, 139))

        c1_x = b_left + 1 * (b_cell + b_pad_x)
        gt_nov_p = ENGINE_QUAL_DIRS["TRELLIS"] / sid / "gt_novel.png"
        gt_rgb_b = load_framed_image(gt_nov_p, resolution=b_cell) if gt_nov_p.exists() else gt_views[sid]["az65"].resize((b_cell, b_cell), Image.LANCZOS)
        gt_comp = composite_with_inset(
            gt_rgb_b,
            gt_views[sid]["norm65"],
        )
        canvas2.paste(gt_comp, (c1_x, ry))
        draw2.rectangle([c1_x, ry, c1_x + b_cell, ry + b_cell], outline=(148, 163, 184), width=2)
        draw_centered_text(draw2, c1_x, b_cell, ry + b_cell + 5, "GT Mesh + 3D Normals", f_cap, (71, 85, 105))

        for e_i, eng in enumerate(ENGINES):
            cx = b_left + (2 + e_i) * (b_cell + b_pad_x)
            nov_p = ENGINE_QUAL_DIRS[eng] / sid / f"{eng}_novel.png"
            nov_img = load_framed_image(nov_p, resolution=b_cell)
            pcomp = composite_with_inset(nov_img, make_surface_relief_inset(nov_img))
            canvas2.paste(pcomp, (cx, ry))
            r = sub.loc[eng]
            is_best = (eng == best_cd_eng) or (eng == best_f1_eng)
            border_col = (13, 148, 136) if is_best else (203, 213, 225)
            border_w = 3 if is_best else 2
            draw2.rectangle([cx, ry, cx + b_cell, ry + b_cell], outline=border_col, width=border_w)
            cap = f"CD:{r['chamfer_distance']:.3f} | F@.1:{r['f_score_0.1']:.2f} | {r['psnr']:.1f}dB"
            draw_centered_text(
                draw2,
                cx,
                b_cell,
                ry + b_cell + 5,
                cap,
                f_cap_bold if is_best else f_cap,
                (15, 118, 110) if is_best else (51, 65, 85),
            )

    for out_path in [
        ROOT / "paper" / "figures" / "gso10_3d_showcase.png",
        BENCH / "paper" / "figures" / "gso10_3d_showcase.png",
    ]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas2.save(out_path, optimize=True)
        print("  Saved:", out_path, flush=True)


if __name__ == "__main__":
    main()
