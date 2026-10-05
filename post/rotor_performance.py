#!/usr/bin/env python3
"""Rotor performance + induction-zone analysis for the sliding-mesh (AMI) rotor case.

Usage:  python3 post/rotor_performance.py <case_dir>

Reads   postProcessing/rotorLoads/0/{force,moment}.dat   (every time step)
        postProcessing/upstreamProbes/0/U                (every time step)
        postProcessing/inductionLines/<t>/*.csv          (mean fields, write times)
Writes  <case>/postProcessing/summary/
            telemetry.csv     per-step loads and upstream velocities
            summary.json      revolution-averaged Cp, Ct, induction factor
            summary.md        human-readable table
            fig_loads.png     thrust / torque / Cp / Ct history
            fig_induction.png upstream slow-down vs actuator-disc theory

Sign conventions: wind along +x, rotor axis -x (counter-clockwise from upstream).
Thrust T = +Fx (pushes rotor downwind). Aerodynamic torque Q = -Mx (drives rotor).
Only numpy + matplotlib.
"""
import glob
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

C_CFD, C_THEORY = "#2a78d6", "#52514e"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style():
    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "lines.linewidth": 2.0, "legend.frameon": False,
        "figure.dpi": 130, "savefig.bbox": "tight",
    })


def read_params(case):
    txt = open(os.path.join(case, "system", "caseParameters")).read()
    txt = re.sub(r"//.*", "", txt)
    get = lambda k: float(re.search(rf"^\s*{k}\s+([-\d.eE+]+)\s*;", txt, re.M).group(1))
    p = {k: get(k) for k in ("Uinf", "R", "tsr", "rhoRef")}
    p["omega"] = p["tsr"] * p["Uinf"] / p["R"]
    p["period"] = 2 * np.pi / p["omega"]
    mp = os.path.join(case, "system", "meshParameters")
    if os.path.exists(mp):
        t = re.sub(r"//.*", "", open(mp).read())
        m = re.search(r"avgFromRev\s+([\d.]+)", t); p["avgFromRev"] = float(m.group(1)) if m else 2
        m = re.search(r"preset\s+(\w+)", t); p["preset"] = m.group(1) if m else "?"
    else:
        p["avgFromRev"], p["preset"] = 2, "?"
    return p


def read_dat(path):
    rows = []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        rows.append([float(v) for v in line.replace("(", " ").replace(")", " ").split()])
    return np.array(rows)


def latest_dir(case, fo):
    """postProcessing/<fo>/<t>/ may contain restarts: concatenate in time order."""
    ds = sorted(glob.glob(os.path.join(case, "postProcessing", fo, "*")),
                key=lambda d: float(os.path.basename(d)) if re.match(r"^[\d.eE+-]+$", os.path.basename(d)) else -1)
    return [d for d in ds if re.match(r"^[\d.eE+-]+$", os.path.basename(d))]


def load_series(case, fo, fname):
    parts = [read_dat(os.path.join(d, fname)) for d in latest_dir(case, fo)
             if os.path.exists(os.path.join(d, fname))]
    a = np.vstack(parts)
    _, idx = np.unique(a[:, 0], return_index=True)   # drop restart overlaps
    return a[idx]


def actuator_disc_axis(x, a, R):
    """Axial velocity on the axis upstream of a uniformly loaded actuator disc."""
    return 1.0 - a * (1.0 + x / np.sqrt(x ** 2 + R ** 2))


def main(case):
    style()
    case = os.path.abspath(case)
    out = os.path.join(case, "postProcessing", "summary")
    os.makedirs(out, exist_ok=True)
    p = read_params(case)
    U, R, rho, om = p["Uinf"], p["R"], p["rhoRef"], p["omega"]
    A = np.pi * R ** 2
    q = 0.5 * rho * U ** 2

    F = load_series(case, "rotorLoads", "force.dat")
    M = load_series(case, "rotorLoads", "moment.dat")
    n = min(len(F), len(M)); F, M = F[:n], M[:n]
    t = F[:, 0]
    T = F[:, 1]                     # thrust [N]
    Q = -M[:, 1]                    # torque [N m]
    P = Q * om                      # power [W]
    Ct = T / (q * A); Cp = P / (q * U * A)
    rev = t / p["period"]

    # upstream probes (time series)
    probes = load_series(case, "upstreamProbes", "U")
    pt = probes[:, 0]
    ux = probes[:, 1::3]            # columns: x components of each probe
    ux_i = np.column_stack([np.interp(t, pt, ux[:, k]) for k in range(ux.shape[1])])

    # telemetry table for the animation
    hdr = ("time_s,rev,azimuth_deg,thrust_N,torque_Nm,power_W,Cp,Ct,"
           "u_axis_m2R,u_axis_m1R,u_axis_m05R,u_axis_m025R,"
           "u_r07_m2R,u_r07_m1R,u_r07_m05R,u_r07_m025R,u_r07_p2R,u_r07_p4R")
    np.savetxt(os.path.join(out, "telemetry.csv"),
               np.column_stack([t, rev, np.degrees(om * t) % 360, T, Q, P, Cp, Ct, ux_i / U]),
               delimiter=",", header=hdr, comments="", fmt="%.6g")

    # revolution averages
    nrev = int(np.floor(rev[-1] + 1e-3))   # tolerant to rounding of the last time
    per_rev = []
    for k in range(nrev):
        m = (rev > k) & (rev <= k + 1)
        if m.sum() > 3:
            per_rev.append({"rev": k + 1, "Cp": float(Cp[m].mean()), "Ct": float(Ct[m].mean()),
                            "thrust_N": float(T[m].mean()), "torque_Nm": float(Q[m].mean())})
    avg = (rev > p["avgFromRev"]) & (rev <= nrev)
    if avg.sum() < 3:
        avg = rev > rev[-1] - 1
    Cp_m, Ct_m = float(Cp[avg].mean()), float(Ct[avg].mean())
    a_mom = float(0.5 * (1 - np.sqrt(max(1 - Ct_m, 0)))) if Ct_m < 1 else float("nan")

    # mean induction lines (latest write time with data)
    lines = {}
    for d in reversed(latest_dir(case, "inductionLines")):
        fs = glob.glob(os.path.join(d, "*.csv"))
        if fs:
            for f in fs:
                name = os.path.basename(f).split("_")[0]
                arr = np.genfromtxt(f, delimiter=",", names=True)
                lines[name] = arr
            lines["_time"] = float(os.path.basename(d))
            break

    summary = {
        "preset": p["preset"], "U_inf": U, "R": R, "tsr": p["tsr"], "omega_rad_s": om,
        "rpm": om * 60 / (2 * np.pi), "revolutions_simulated": float(rev[-1]),
        "averaging_window_rev": [float(rev[avg][0]), float(rev[avg][-1])],
        "Cp": Cp_m, "Ct": Ct_m, "thrust_N": float(T[avg].mean()),
        "torque_Nm": float(Q[avg].mean()), "power_W": float(P[avg].mean()),
        "thrust_ripple_pct": float(100 * T[avg].std() / abs(T[avg].mean())),
        "a_from_Ct_momentum": a_mom,
        "per_revolution": per_rev
    }
    if "axisUpstream" in lines:
        L = lines["axisUpstream"]
        x, ux0 = L["x"], L["UMean_0"]
        for xr in (-2.0, -1.0, -0.5):
            summary[f"u_axis_x{xr:+.1f}R"] = float(np.interp(xr * R, x, ux0) / U)
        summary["induction_lines_time"] = lines["_time"]
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)

    # ---------- figures
    fig, ax = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for a_, y, lab in ((ax[0, 0], T, "Thrust T [N]"), (ax[0, 1], Q, "Torque Q [N m]"),
                       (ax[1, 0], Cp, "Power coefficient $C_P$"), (ax[1, 1], Ct, "Thrust coefficient $C_T$")):
        a_.plot(rev, y, color=C_CFD, lw=1.4)
        a_.axvspan(rev[avg][0], rev[avg][-1], color=GRID, alpha=0.5, lw=0)
        a_.set_ylabel(lab)
    for a_ in ax[1]:
        a_.set_xlabel("Revolutions")
    for a_, k in ((ax[1, 0], "Cp"), (ax[1, 1], "Ct")):
        lo = min(0, np.percentile([Cp, Ct][k == "Ct"], 2))
        a_.set_ylim(lo, 1.3 * np.percentile([Cp, Ct][k == "Ct"], 99))
    for a_, y in ((ax[0, 0], T), (ax[0, 1], Q)):
        a_.set_ylim(min(0, np.percentile(y, 2)), 1.3 * np.percentile(y, 99))
    fig.suptitle(f"Model rotor, TSR {p['tsr']:.1f}, U = {U:.0f} m/s ({p['preset']} mesh): "
                 f"$C_P$ = {Cp_m:.3f}, $C_T$ = {Ct_m:.3f} (shaded = averaging window)", fontsize=11)
    fig.savefig(os.path.join(out, "fig_loads.png")); plt.close(fig)

    if "axisUpstream" in lines:
        L = lines["axisUpstream"]
        xs = np.linspace(-4 * R, -0.01, 300)
        fig, axx = plt.subplots(figsize=(7, 4))
        axx.plot(L["x"] / R, L["UMean_0"] / U, color=C_CFD, label="CFD, time-mean on the axis")
        if np.isfinite(a_mom):
            axx.plot(xs / R, actuator_disc_axis(xs, a_mom, R), color=C_THEORY, ls="--",
                     label=f"actuator disc, a = {a_mom:.3f} (from CFD $C_T$)")
        axx.axvline(0, color=INK2, lw=0.8)
        axx.set_xlabel("x / R   (rotor plane at 0, wind from the left)")
        axx.set_ylabel("$u_x / U_\\infty$")
        axx.set_xlim(-4, 0.05)
        axx.legend(loc="lower left")
        axx.set_title("The air slows down before it reaches the blades", fontsize=11)
        fig.savefig(os.path.join(out, "fig_induction.png")); plt.close(fig)

    md = [f"# Rotor performance ({p['preset']} mesh)", "",
          f"TSR {p['tsr']:.2f}, U = {U} m/s, {om*60/2/np.pi:.0f} rpm, averaged over revolutions "
          f"{rev[avg][0]:.2f}-{rev[avg][-1]:.2f}", "",
          "| quantity | CFD |", "|---|---|",
          f"| Cp | {Cp_m:.3f} |",
          f"| Ct | {Ct_m:.3f} |",
          f"| thrust [N] | {summary['thrust_N']:.2f} |",
          f"| torque [N m] | {summary['torque_Nm']:.3f} |",
          f"| power [W] | {summary['power_W']:.1f} |", ""]
    for k in ("u_axis_x-2.0R", "u_axis_x-1.0R", "u_axis_x-0.5R"):
        if k in summary:
            md.append(f"- mean axial velocity on the axis at {k[7:]}: {summary[k]:.3f} U")
    md.append(f"- axial induction from momentum theory (a = (1 - sqrt(1 - Ct))/2): {a_mom:.3f}")
    md.append("")
    md.append("| revolution | Cp | Ct |"); md.append("|---|---|---|")
    for r in per_rev:
        md.append(f"| {r['rev']} | {r['Cp']:.3f} | {r['Ct']:.3f} |")
    open(os.path.join(out, "summary.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
