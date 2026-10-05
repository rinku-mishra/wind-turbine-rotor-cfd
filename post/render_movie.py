#!/usr/bin/env python3
"""Rotor animation: light background, velocity cut planes, vortex surfaces.

Usage:
    python3 post/render_movie.py <case_dir> [--size 1080x1080] [--loops 4] [--fps 30]
                                    [--out <case>/rotor.mp4] [--max-frames N]

Reads postProcessing/animationSurfaces/<t>/{midPlane,hubPlane,vortices}.vtp of the last
revolution and ONE rotor.vtp: the rotor is a rigid body, so later positions are the first
one rotated by -omega*(t - t0) about x (checked to 1e-7 m against the sampled surfaces).

Look: light studio background, two cut-away planes through the rotor axis coloured by
axial velocity (blue = slowed, white = free stream, red = faster), Q-criterion vortex
surfaces in amber, brushed-metal rotor, minimal text.
"""
import argparse
import glob
import io
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from rotor_performance import read_params  # noqa: E402

pv.OFF_SCREEN = True
FONT = "Inter" if any("Inter" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"

BG_TOP, BG_BOT = "#ffffff", "#dde3e9"
INK, INK2 = "#1d252c", "#5a6670"
VORTEX = "#f0a24e"
BLADE = "#aab3bb"
# Downstream end of what is shown. After 6 revolutions the start-up wake front is still
# moving through x > ~1.4 m, so frames there do not repeat from one revolution to the next
# and the loop would visibly jump. Up to 3R (1.35 m) the flow is periodic.
XMAX = 1.35

# u/U from 0.3 to 1.3, white at 1.0 (free stream): blue = air slowed by the rotor
U0, U1 = 0.3, 1.3
_n = lambda v: (v - U0) / (U1 - U0)
CMAP = LinearSegmentedColormap.from_list("uxU", [
    (_n(0.30), "#0b2d5c"), (_n(0.50), "#1f5fa8"), (_n(0.72), "#6fa3d2"), (_n(0.90), "#cfe0ef"),
    (_n(1.00), "#f7f7f5"), (_n(1.12), "#f2b48f"), (_n(1.30), "#c0392b")])


def frames(case):
    base = os.path.join(case, "postProcessing", "animationSurfaces")
    out = []
    for d in glob.glob(os.path.join(base, "*")):
        n = os.path.basename(d)
        if re.match(r"^[\d.eE+-]+$", n) and os.path.exists(os.path.join(d, "midPlane.vtp")):
            out.append((float(n), d))
    return sorted(out)


def slice_mesh(path, U, bounds):
    m = pv.read(path)
    m.point_data["ux"] = m.point_data["U"][:, 0] / U
    return m.clip_box(bounds=bounds, invert=False)


def overlay(W, H, p):
    """Title + colour bar as one transparent layer (identical on every frame)."""
    s = min(W, H) / 1080
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.text(56 * s, 70 * s, "Airflow through a rotating wind-turbine rotor", fontsize=27 * s, color=INK,
            fontweight="semibold", family=FONT, va="baseline")
    ax.text(56 * s, 108 * s, "OpenFOAM  ·  sliding-mesh URANS  ·  k-ω SST", fontsize=17 * s, color=INK2,
            family=FONT, va="baseline")
    # colour bar, bottom left
    cb = fig.add_axes([56 * s / W, 58 * s / H, 380 * s / W, 14 * s / H])
    cb.imshow(np.linspace(0, 1, 512)[None, :], aspect="auto", cmap=CMAP, extent=[U0, U1, 0, 1])
    cb.set_yticks([]); cb.set_xticks([0.4, 0.7, 1.0, 1.3])
    cb.tick_params(colors=INK2, labelsize=12 * s, length=3 * s)
    for t in cb.get_xticklabels(): t.set_family(FONT)
    for sp in cb.spines.values(): sp.set_visible(False)
    fig.text(56 * s / W, (58 + 14 + 14) * s / H, "Axial velocity  u / U∞", fontsize=13.5 * s, color=INK,
             family=FONT)
    buf = io.BytesIO(); fig.savefig(buf, format="png", transparent=True, dpi=100); plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--size", default="1080x1080")
    ap.add_argument("--loops", type=int, default=4)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--frame-index", type=int, default=-1, help="render only this frame (test)")
    a = ap.parse_args()

    case = os.path.abspath(a.case)
    a.out = a.out or os.path.join(case, "rotor.mp4")
    W, H = (int(v) for v in a.size.split("x"))
    p = read_params(case)
    U, om = p["Uinf"], p["omega"]
    F = frames(case)
    F = [f for f in F if f[0] < F[0][0] + p["period"] - 1e-9]
    n = len(F)
    total = n * a.loops if a.max_frames == 0 else min(a.max_frames, n * a.loops)
    ks = [a.frame_index] if a.frame_index >= 0 else range(total)
    print(f"{n} frames per revolution, rendering {len(ks)}")

    t0, d0 = F[0]
    rotor0 = pv.read(os.path.join(d0, "rotor.vtp"))
    wide = W / H > 1.3
    mid_b = (-0.8, XMAX, -0.01, 0.01, -0.80, 0.0)     # vertical plane, lower half
    hub_b = (-0.8, XMAX, -0.95, 0.0, -0.01, 0.01)     # horizontal plane, far half
    over = overlay(W, H, p)

    fdir = os.path.join(os.path.dirname(os.path.abspath(a.out)), "frames_" + os.path.splitext(os.path.basename(a.out))[0])
    if a.frame_index < 0:
        shutil.rmtree(fdir, ignore_errors=True)
    os.makedirs(fdir, exist_ok=True)

    pl = pv.Plotter(off_screen=True, window_size=(W, H), lighting="none")
    pl.set_background(BG_BOT, top=BG_TOP)
    for pos, inten in (((-3, 3, 4), 0.75), ((2, -2, 3), 0.35), ((-4, -1, -1), 0.25)):
        pl.add_light(pv.Light(position=pos, focal_point=(0.5, 0, 0), intensity=inten, light_type="scene light"))
    try:
        pl.enable_anti_aliasing("ssaa")
    except Exception:
        pass

    for k in ks:
        tk, d = F[k % n]
        pl.clear_actors()
        for name, b in (("midPlane", mid_b), ("hubPlane", hub_b)):
            m = slice_mesh(os.path.join(d, name + ".vtp"), U, b)
            pl.add_mesh(m, scalars="ux", cmap=CMAP, clim=(U0, U1), show_scalar_bar=False, lighting=False)
            e = m.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False,
                                        non_manifold_edges=False)
            pl.add_mesh(e, color="#8e9aa4", line_width=1.2 * min(W, H) / 1080, lighting=False)
        v = pv.read(os.path.join(d, "vortices.vtp")).clip_box(bounds=(0.045, XMAX, -0.7, 0.7, -0.7, 0.7),
                                                               invert=False)
        if v.n_points:
            pl.add_mesh(v, color=VORTEX, opacity=0.5, smooth_shading=True, specular=0.35, specular_power=20,
                        ambient=0.25, diffuse=0.75)
        r = rotor0.copy()
        r.rotate_x(-np.degrees(om * (tk - t0)), point=(0, 0, 0), inplace=True)
        pl.add_mesh(r, color=BLADE, smooth_shading=True, split_sharp_edges=True, specular=0.7,
                    specular_power=40, ambient=0.3, diffuse=0.8)

        frac = k / max(total - 1, 1)
        az = np.radians(34 + 18 * frac)            # from upstream toward +y
        el = np.radians(21 - 4 * frac)
        focal = np.array([0.22, 0.0, -0.16]) if not wide else np.array([0.30, 0.0, -0.12])
        dist = 3.7 if not wide else 3.1
        cam = focal + dist * np.array([-np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        pl.camera_position = [tuple(cam), tuple(focal), (0, 0, 1)]
        pl.camera.view_angle = 30
        img = Image.fromarray(pl.screenshot(return_img=True)).convert("RGBA")
        img.alpha_composite(over)
        img.convert("RGB").save(os.path.join(fdir, f"f{k:04d}.png"))
        if k % 30 == 0:
            print(f"  frame {k + 1}/{total}", flush=True)
    pl.close()
    if a.frame_index >= 0:
        print("test frame:", os.path.join(fdir, f"f{a.frame_index:04d}.png"))
        return
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i",
                    os.path.join(fdir, "f%04d.png"), "-c:v", "libx264", "-profile:v", "high", "-preset", "slow",
                    "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out], check=True)
    Image.open(os.path.join(fdir, f"f{n // 2:04d}.png")).convert("RGB").save(
        os.path.splitext(a.out)[0] + "_poster.jpg", quality=92)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
