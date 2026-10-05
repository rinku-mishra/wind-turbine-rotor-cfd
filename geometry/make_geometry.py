#!/usr/bin/env python3
"""Generate the rotor surfaces for snappyHexMesh.

Writes into case/constant/triSurface/:
    blades.stl        three blades (solid name 'blades')
    hub.stl           spinner + short nacelle body of revolution ('hub')
    rotatingZone.stl  closed cylinder bounding the rotating AMI zone ('AMI')

Frame (identical to the OpenFOAM case):
    x  = streamwise (wind blows toward +x), rotor plane at x = 0
    z  = up, y = lateral, rotor centre at the origin
    rotor turns counter-clockwise seen from upstream -> omega vector = -omega * x_hat

Blade at azimuth 0 points along +z and moves toward +y. In each section the chord
runs from the leading edge to the trailing edge along (sin t, -cos t) in (x, y) and the
suction side faces (cos t, sin t), so lift has a +y (driving) and +x (thrust) part.

Only numpy + scipy needed. Run:  python3 geometry/make_geometry.py
"""
import csv
import json
import os

import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "case", "constant", "triSurface")

P = json.load(open(os.path.join(HERE, "params.json")))


# ----------------------------------------------------------------------------- airfoil
def load_airfoil(path, n=70):
    """Return x (cosine spaced, 0..1), y_upper, y_lower in chord units."""
    pts = np.loadtxt(path, skiprows=1)
    ile = np.argmin(pts[:, 0])
    up = pts[: ile + 1][::-1]   # LE -> TE
    lo = pts[ile:]              # LE -> TE
    t = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1.0 - np.cos(t))
    yu = CubicSpline(up[:, 0], up[:, 1])(x)
    yl = CubicSpline(lo[:, 0], lo[:, 1])(x)
    yu[0] = yl[0] = 0.5 * (yu[0] + yl[0])
    # finite trailing edge (milled blades are not knife sharp); thickness in chord units
    te = P["trailing_edge_thickness_chord"]
    yu += 0.5 * te * x
    yl -= 0.5 * te * x
    return x, yu, yl


def section_local(x, yu, yl, chord, blend, d_root):
    """Section outline in metres, relative to the pitch axis.

    Returns (n_ring, 2) points (s = along chord, n = toward suction side),
    ordered TE(upper) -> LE -> TE(lower), and the strip pairing for the tip cap.
    blend = 1 -> airfoil, 0 -> circle of diameter d_root centred on the pitch axis.
    """
    a = P["pitch_axis_chord"]
    # airfoil
    su = chord * (x - a); nu = chord * yu
    sl = chord * (x - a); nl = chord * yl
    # circle with the same parametrisation (x = (1-cos t)/2 -> angle t)
    t = np.arccos(1.0 - 2.0 * x)
    sc = d_root * (x - 0.5)
    nc = 0.5 * d_root * np.sin(t)
    su = blend * su + (1 - blend) * sc
    sl = blend * sl + (1 - blend) * sc
    nu = blend * nu + (1 - blend) * nc
    nl = blend * nl + (1 - blend) * (-nc)
    # ring: upper TE..LE (reverse), lower LE+1..TE-1
    s = np.concatenate([su[::-1], sl[1:-1]])
    n = np.concatenate([nu[::-1], nl[1:-1]])
    return np.column_stack([s, n])


def blade_sections():
    rows = [r for r in csv.reader(l for l in open(os.path.join(HERE, "blade_table.csv"))
                                  if not l.startswith("#"))]
    hdr, rows = rows[0], rows[1:]
    tab = np.array([[float(v) for v in r[:3]] for r in rows])
    r_t, c_t, tw_t = tab.T
    R = P["tip_radius"]
    r_air = P["airfoil_start_radius"]      # full S826 from here outward
    r_cyl = P["root_cylinder_end_radius"]  # pure cylinder up to here
    d_root = P["root_cylinder_diameter"]

    # airfoil-region interpolants (only S826 rows)
    m = r_t >= r_air - 1e-9
    f_c = PchipInterpolator(r_t[m], c_t[m], extrapolate=True)
    f_tw = PchipInterpolator(r_t[m], tw_t[m], extrapolate=True)

    # radial stations: root inside hub, cylinder, transition, airfoil (clustered at tip)
    r0 = P["root_start_radius"]
    st = list(np.linspace(r0, r_cyl, 4))
    st += list(np.linspace(r_cyl, r_air, 8)[1:])
    u = np.linspace(0, 1, P["n_span_sections"])
    st += list(r_air + (R - r_air) * np.sin(0.5 * np.pi * u)[1:])
    st = np.array(st)

    secs = []
    for r in st:
        if r <= r_cyl:
            b, c, tw = 0.0, d_root, f_tw(r_air)
        elif r < r_air:
            q = (r - r_cyl) / (r_air - r_cyl)
            b = 0.5 - 0.5 * np.cos(np.pi * q)          # smooth 0 -> 1
            c, tw = f_c(r_air), f_tw(r_air)
        else:
            b, c, tw = 1.0, float(f_c(r)), float(f_tw(r))
        secs.append((r, b, c, tw))
    return secs


def loft_blade(azimuth_deg):
    x, yu, yl = load_airfoil(os.path.join(HERE, "S826.dat"), P["n_chord_points"])
    secs = blade_sections()
    rings = []
    for r, b, c, tw in secs:
        loc = section_local(x, yu, yl, c, b, P["root_cylinder_diameter"])
        th = np.radians(tw)
        xi = np.array([np.sin(th), -np.cos(th)])    # LE -> TE in (x, y)
        eta = np.array([np.cos(th), np.sin(th)])    # suction side
        xy = np.outer(loc[:, 0], xi) + np.outer(loc[:, 1], eta)
        rings.append(np.column_stack([xy[:, 0], xy[:, 1], np.full(len(xy), r)]))
    rings = np.array(rings)                 # (nsec, nring, 3)
    ns, nr, _ = rings.shape
    V = rings.reshape(-1, 3)
    F = []
    for i in range(ns - 1):
        for j in range(nr):
            a = i * nr + j; b2 = i * nr + (j + 1) % nr
            c2 = (i + 1) * nr + j; d = (i + 1) * nr + (j + 1) % nr
            F += [(a, b2, d), (a, d, c2)]
    # caps: pair upper[k] with lower[k] (same chord station) -> strip
    npt = P["n_chord_points"]
    up_idx = lambda k: npt - 1 - k                 # k = 0 LE ... npt-1 TE  (ring index)
    lo_idx = lambda k: (npt - 1 + k) if 0 < k < npt - 1 else (npt - 1 if k == 0 else 0)
    for i, flip in ((0, True), (ns - 1, False)):
        base = i * nr
        for k in range(npt - 1):
            u0, u1 = base + up_idx(k), base + up_idx(k + 1)
            l0, l1 = base + lo_idx(k), base + lo_idx(k + 1)
            tris = [(u0, l0, l1), (u0, l1, u1)]
            for t in tris:
                if t[0] == t[1] or t[1] == t[2] or t[0] == t[2]:
                    continue
                F.append(t[::-1] if flip else t)
    F = np.array(F)
    # rotate about x by azimuth (blade 0 along +z, positive azimuth toward +y,
    # i.e. the sense of rotation)
    ph = np.radians(azimuth_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(ph), np.sin(ph)], [0, -np.sin(ph), np.cos(ph)]])
    # a point on +z rotated by +ph moves toward +y: (0,0,1)->(0, sin, cos)
    V = V @ Rx.T
    return V, F


def orient_outward(V, F):
    """Flip all faces if signed volume is negative (closed surface)."""
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    vol = np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0
    return (F[:, ::-1] if vol < 0 else F), abs(vol)


def revolve(profile_xr, n_theta=96):
    """Body of revolution about x from a (x, r) polyline with r=0 at both ends."""
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    pts, F = [], []
    inner = profile_xr[1:-1]
    for x, r in inner:
        for t in th:
            pts.append((x, r * np.sin(t), r * np.cos(t)))
    pts.append((profile_xr[0][0], 0, 0)); i_nose = len(pts) - 1
    pts.append((profile_xr[-1][0], 0, 0)); i_tail = len(pts) - 1
    m = len(inner)
    for i in range(m - 1):
        for j in range(n_theta):
            a = i * n_theta + j; b = i * n_theta + (j + 1) % n_theta
            c = (i + 1) * n_theta + j; d = (i + 1) * n_theta + (j + 1) % n_theta
            F += [(a, b, d), (a, d, c)]
    for j in range(n_theta):
        F.append((i_nose, (j + 1) % n_theta, j))
        a = (m - 1) * n_theta
        F.append((i_tail, a + j, a + (j + 1) % n_theta))
    return np.array(pts, float), np.array(F)


def hub_surface():
    h = P["hub"]
    rh, xn, x1, x2, xt = h["radius"], h["nose_x"], h["cyl_start_x"], h["cyl_end_x"], h["tail_x"]
    prof = []
    for t in np.linspace(0, 0.5 * np.pi, 24):          # elliptic nose
        prof.append((x1 - (x1 - xn) * np.cos(t), rh * np.sin(t)))
    for x in np.linspace(x1, x2, 12)[1:]:
        prof.append((x, rh))
    for t in np.linspace(0, 0.5 * np.pi, 16)[1:]:      # elliptic tail
        prof.append((x2 + (xt - x2) * np.sin(t), rh * np.cos(t)))
    prof[-1] = (xt, 0.0)
    return revolve(np.array(prof), 96)


def cylinder_surface(radius, x0, x1, n_theta=128, n_x=6, n_r=6):
    """Closed cylinder about x (side + two discs), returns V, F."""
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    V, F = [], []
    def ring(x, r):
        s = len(V)
        for t in th:
            V.append((x, r * np.sin(t), r * np.cos(t)))
        return s
    def connect(sa, sb):
        for j in range(n_theta):
            a = sa + j; b = sa + (j + 1) % n_theta; c = sb + j; d = sb + (j + 1) % n_theta
            F.extend([(a, b, d), (a, d, c)])
    # front disc rings (center -> radius), side rings, back disc rings
    rr = np.linspace(radius / n_r, radius, n_r)
    c0 = len(V); V.append((x0, 0, 0))
    front = [ring(x0, r) for r in rr]
    for j in range(n_theta):
        F.append((c0, front[0] + (j + 1) % n_theta, front[0] + j))
    for a, b in zip(front[:-1], front[1:]):
        connect(a, b)
    side = [front[-1]] + [ring(x, radius) for x in np.linspace(x0, x1, n_x)[1:]]
    for a, b in zip(side[:-1], side[1:]):
        connect(a, b)
    back = [side[-1]] + [ring(x1, r) for r in rr[::-1][1:]]
    for a, b in zip(back[:-1], back[1:]):
        connect(a, b)
    c1 = len(V); V.append((x1, 0, 0))
    for j in range(n_theta):
        F.append((c1, back[-1] + j, back[-1] + (j + 1) % n_theta))
    return np.array(V, float), np.array(F)


def write_stl(path, solid, parts):
    with open(path, "w") as f:
        f.write(f"solid {solid}\n")
        for V, F in parts:
            v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
            n = np.cross(v1 - v0, v2 - v0)
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-300
            for k in range(len(F)):
                f.write(f" facet normal {n[k,0]:.6e} {n[k,1]:.6e} {n[k,2]:.6e}\n  outer loop\n")
                for v in (v0[k], v1[k], v2[k]):
                    f.write(f"   vertex {v[0]:.7e} {v[1]:.7e} {v[2]:.7e}\n")
                f.write("  endloop\n endfacet\n")
        f.write(f"endsolid {solid}\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    nb = P["n_blades"]
    blades, vol = [], 0.0
    for k in range(nb):
        V, F = loft_blade(360.0 * k / nb)
        F, v = orient_outward(V, F)
        blades.append((V, F)); vol += v
    write_stl(os.path.join(OUT, "blades.stl"), "blades", blades)
    Vh, Fh = hub_surface(); Fh, vh = orient_outward(Vh, Fh)
    write_stl(os.path.join(OUT, "hub.stl"), "hub", [(Vh, Fh)])
    z = P["rotating_zone"]
    Vz, Fz = cylinder_surface(z["radius"], z["x_min"], z["x_max"])
    Fz, vz = orient_outward(Vz, Fz)
    write_stl(os.path.join(OUT, "rotatingZone.stl"), "AMI", [(Vz, Fz)])

    allV = np.vstack([b[0] for b in blades])
    rmax = np.sqrt(allV[:, 1] ** 2 + allV[:, 2] ** 2).max()
    summary = {
        "tip_radius_m": float(rmax),
        "x_extent_blades_m": [float(allV[:, 0].min()), float(allV[:, 0].max())],
        "blade_volume_m3_each": vol / nb,
        "hub_volume_m3": vh,
        "n_triangles": {"blades": int(sum(len(b[1]) for b in blades)), "hub": int(len(Fh)),
                        "rotatingZone": int(len(Fz))},
    }
    json.dump(summary, open(os.path.join(HERE, "geometry_summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
