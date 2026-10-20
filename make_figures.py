"""make_figures.py - redraw the figures of the paper (Figures 1-8) from the result files.

  Fig. 1     V2I architecture schematic                                (make_fig1_architecture.py, no data)
  Fig. 2, 3  garage layouts and an example LOS/NLOS visibility graph   (needs nothing)
  Fig. 4     old vs final solver                                       (old_vs_realistic_comparison.json)
  Fig. 5     GDOP heatmap of the best GA layout, Garage 1              (best_layouts.json)
  Fig. 6     PSO vs GA convergence, single run, seed 42                (runs both optimisers, ~30 s)
  Fig. 7     median error across radio technologies                    (task_2_5_results.json, uwb_rssi_comparison_results.json,
                                                                        accuracy_validation_both_garage1.json)
  Fig. 8     minimum anchor count versus target accuracy (95% level)   (accuracy_driven_design_p95_garage1.json, ..._garage2.json)

Run it inside the directory that holds the result files (python <path>/make_figures.py, as run_all.py does) or give --resdir.
Figures are written to <resdir>/figures/ as PNG (300 dpi) and PDF. A figure whose input file is missing is skipped with a message.
Usage: python make_figures.py [--resdir DIR] [--only 1 2 3 4 5 6 7 8]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from shapely.geometry import Point

from garage_model import build_scenario, GARAGE_1, GARAGE_2
from visibility import build_visibility_matrix
from gdop import gdop_2d

BLUE, ORANGE, GRAY, INK = "#0072B2", "#D55E00", "#8C8C8C", "#222222"   # Okabe-Ito based palette
LOS_C, NLOS_C, CAR_FILL, CAR_EDGE = "#1B8A5A", "#C0392B", "#E8C99B", "#B8863B"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#444444", "axes.linewidth": 0.9, "axes.spines.top": False, "axes.spines.right": False,
                     "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": INK, "text.color": INK, "figure.dpi": 100, "savefig.dpi": 300})
FIGDIR = Path("figures")


def save(fig, name):
    FIGDIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"): fig.savefig(FIGDIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig); print(f"  wrote {FIGDIR / name}.png / .pdf", flush=True)


def need(*files):
    missing = [f for f in files if not Path(f).exists()]
    if missing: print(f"  skipped: missing {', '.join(missing)}")
    return not missing


def draw_obstacles(ax, cfg, sc):
    ax.add_patch(mpatches.Rectangle((0, 0), cfg.width, cfg.length, fill=False, edgecolor="black", linewidth=2.2))
    for col in sc["columns"]:
        x, y = col.exterior.xy; ax.fill(x, y, color="#4D4D4D")
    for car in sc["parked_cars"]:
        x, y = car["polygon"].exterior.xy; ax.fill(x, y, facecolor=CAR_FILL, edgecolor=CAR_EDGE, linewidth=0.5)
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")


def fig_layout(cfg, name, example_point):
    sc = build_scenario(cfg); cand, samp = sc["candidate_AN_points"], sc["vehicle_sample_points"]
    vis = build_visibility_matrix(cand, samp, sc["columns"], sc["parked_cars"], cfg.an_height, cfg.tag_height)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ax = axes[0]; draw_obstacles(ax, cfg, sc)
    ax.scatter(cand[:, 0], cand[:, 1], s=40, color=BLUE, marker="^", zorder=3)
    ax.set_title("(a) Obstacles and candidate anchors")
    ax.legend(handles=[mpatches.Patch(color="#4D4D4D", label="Columns"), mpatches.Patch(facecolor=CAR_FILL, edgecolor=CAR_EDGE, label="Parked cars"),
                       Line2D([], [], marker="^", color=BLUE, linestyle="", markersize=9, label=f"Candidates (N={len(cand)})")],
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, fontsize=9)
    ax = axes[1]; draw_obstacles(ax, cfg, sc)
    k = int(np.argmin(np.sum((samp - np.array(example_point)) ** 2, axis=1))); p = samp[k]
    for i, a in enumerate(cand):
        ax.plot([p[0], a[0]], [p[1], a[1]], color=LOS_C if vis[i, k] else NLOS_C, linewidth=0.8, alpha=0.55, zorder=1)
    ax.scatter(cand[:, 0], cand[:, 1], s=40, color=BLUE, marker="^", zorder=3); ax.scatter(*p, s=120, color="black", zorder=5)
    ax.set_title(f"(b) LOS/NLOS from one position ({int(vis[:, k].sum())}/{len(cand)} LOS)")
    ax.legend(handles=[Line2D([], [], color=LOS_C, linewidth=2, label="LOS"), Line2D([], [], color=NLOS_C, linewidth=2, label="NLOS"),
                       Line2D([], [], marker="o", color="black", linestyle="", markersize=9, label="Vehicle position")],
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(); save(fig, name)


def fig4():
    if not need("old_vs_realistic_comparison.json"): return
    R = json.load(open("old_vs_realistic_comparison.json"))
    keys = [("best_pso_idx", "PSO layout"), ("best_ga_idx", "GA layout")]
    panels = [("median_cm", "Median error (cm)", False, lambda v: f"{v:.1f}"), ("max_cm", "Maximum error (cm, log)", True, lambda v: f"{v:.1e}" if v >= 1e5 else f"{v:,.0f}"),
              ("corr", "GDOP–error Pearson r", False, lambda v: f"{v:.2f}")]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2)); w = 0.36
    for ax, (m, title, log, fmt) in zip(axes, panels):
        for j, (cfgname, color, lab) in enumerate((("old", GRAY, "Original solver"), ("realistic", BLUE, "Final solver"))):
            vals = [R[k][cfgname][m] for k, _ in keys]; xs = np.arange(2) + (j - 0.5) * w
            ax.bar(xs, vals, w, color=color, label=lab)
            for x, v in zip(xs, vals): ax.text(x, v * (1.4 if log else 1.0) + (0 if log else 0.01 * max(1, abs(v))), fmt(v), ha="center", va="bottom", fontsize=9)
        if log: ax.set_yscale("log"); ax.set_ylim(1, 1e12)
        ax.set_xticks(range(2)); ax.set_xticklabels([l for _, l in keys]); ax.set_title(title, fontsize=11)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03)); fig.tight_layout(rect=(0, 0.05, 1, 1)); save(fig, "fig4_old_vs_final_solver")


def fig5():
    if not need("best_layouts.json"): return
    cfg = GARAGE_1; sc = build_scenario(cfg); cand = sc["candidate_AN_points"]; columns, cars = sc["columns"], sc["parked_cars"]
    idx = np.array(json.load(open("best_layouts.json"))["best_ga_idx"]); lc = cand[idx]
    xs = np.arange(0.5, cfg.width, 1.0); ys = np.arange(0.5, cfg.length, 1.0); pts = np.array([[x, y] for x in xs for y in ys])
    inside = np.array([any(Point(x, y).within(c) for c in columns) or any(Point(x, y).within(c["polygon"]) for c in cars) for x, y in pts])
    vis = build_visibility_matrix(lc, pts, columns, cars, cfg.an_height, cfg.tag_height)
    g = np.full(len(pts), np.nan); n = np.where(inside, -1, vis.sum(axis=0))
    for k in range(len(pts)):
        if not inside[k] and n[k] >= 2: g[k] = gdop_2d(lc[vis[:, k]], pts[k])
    G = g.reshape(len(xs), len(ys)).T; N = n.reshape(len(xs), len(ys)).T
    low = (N >= 0) & (N < 4); Z = np.clip(np.where(low | (N < 0), np.nan, G), 0.1, 20)
    fig, ax = plt.subplots(figsize=(9.2, 5.6)); ext = [0, cfg.width, 0, cfg.length]
    ax.set_facecolor("white")
    ax.imshow(np.where(low, 1.0, np.nan), origin="lower", extent=ext, cmap="gray_r", vmin=0, vmax=2.2, aspect="equal", interpolation="nearest")
    im = ax.imshow(Z, origin="lower", extent=ext, cmap="viridis_r", norm=LogNorm(vmin=0.3, vmax=10), aspect="equal", interpolation="nearest")
    for c in columns: x, y = c.exterior.xy; ax.fill(x, y, color="#222222")
    for car in cars: x, y = car["polygon"].exterior.xy; ax.plot(x, y, color=CAR_EDGE, linewidth=0.7, alpha=0.8)
    ax.scatter(lc[:, 0], lc[:, 1], s=80, color=ORANGE, marker="^", edgecolor="white", linewidth=1, zorder=5)
    ax.set_xlim(0, cfg.width + 3); ax.set_ylim(0, cfg.length); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    cb = fig.colorbar(im, ax=ax, shrink=0.95, pad=0.04); cb.set_label("GDOP (log scale)")
    ax.legend(handles=[Line2D([], [], marker="^", color=ORANGE, markeredgecolor="white", linestyle="", markersize=9, label="Selected anchors"),
                       mpatches.Patch(facecolor="#BDBDBD", edgecolor="#888888", label="Fewer than $n_{\\min}$ = 4 LOS anchors")],
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False, fontsize=10)
    valid = N >= 0
    print(f"  coverage of valid floor area with >= 4 LOS anchors: {100 * (~low[valid]).mean():.1f}%")
    save(fig, "fig5_gdop_heatmap_ga_garage1")


def fig6(seed=42):
    from pso import binary_pso
    from ga import genetic_algorithm
    cfg = GARAGE_1; sc = build_scenario(cfg); cand, samp = sc["candidate_AN_points"], sc["vehicle_sample_points"]
    vis = build_visibility_matrix(cand, samp, sc["columns"], sc["parked_cars"], cfg.an_height, cfg.tag_height)
    _, Fp, hp, _ = binary_pso(cand, samp, vis, M=16, swarm_size=40, iters=150, seed=seed)
    _, Fg, hg, _ = genetic_algorithm(cand, samp, vis, M=16, pop_size=40, generations=150, seed=seed)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(np.arange(len(hp)) * 40, hp, color=ORANGE, linewidth=2.2, label=f"PSO (final F = {Fp:.1f})")
    ax.plot(np.arange(len(hg)) * 40, hg, color=BLUE, linewidth=2.2, label=f"GA (final F = {Fg:.1f})")
    ax.set_yscale("log"); ax.set_xlabel("Fitness evaluations"); ax.set_ylabel("Best-so-far F(x)"); ax.grid(alpha=0.25, which="both"); ax.legend(frameon=False)
    print(f"  seed {seed}: PSO final F = {Fp:.2f}, GA final F = {Fg:.2f}")
    fig.tight_layout(); save(fig, "fig6_convergence_garage1")


def fig7():
    if not need("task_2_5_results.json", "uwb_rssi_comparison_results.json", "accuracy_validation_both_garage1.json"): return
    T = json.load(open("task_2_5_results.json")); U = json.load(open("uwb_rssi_comparison_results.json")); V = json.load(open("accuracy_validation_both_garage1.json"))
    tdoa_m = float(np.median([p["rmse_cm"] for p in V["best_ga_idx"]])) / 100
    bars = [("UWB-TDoA\n(native)", tdoa_m, BLUE, f"{tdoa_m * 100:.1f} cm"),
            ("UWB hardware,\nRSSI mode", U["tier3_same_deployment"]["uwb_rssi"]["median_m"], GRAY, None),
            # Table 8 takes the dense BLE median from task_2_5_technology_comparison.py and the dense WiFi median from uwb_rssi_comparison.py;
            # the two scripts run the same model with different random draws (differences in the third significant digit only).
            ("BLE-RSSI\n16 AP", T["tier3_uwb_matched"]["ble"]["median_m"], GRAY, None),
            ("WiFi-RSSI\n16 AP", U["tier3_same_deployment"]["wifi_rssi"]["median_m"], GRAY, None),
            ("UWB-RSSI\n4 corner APs", U["tier1_calibration"]["uwb_rssi"]["median_m"], GRAY, None)]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for i, (lab, v, c, txt) in enumerate(bars):
        ax.bar(i, v, 0.62, color=c); ax.text(i, v * 1.12, txt or f"{v:.2f} m", ha="center", va="bottom", fontsize=9.5)
    ax.set_yscale("log"); ax.set_ylim(0.05, 30); ax.set_xticks(range(len(bars))); ax.set_xticklabels([b[0] for b in bars], fontsize=9)
    ax.set_ylabel("Median positioning error (m)"); ax.grid(axis="y", which="both", alpha=0.25)
    fig.tight_layout(); save(fig, "fig7_technology_comparison")


def reached(T, t):
    """Minimum M if the target is met in the Monte Carlo validation (at least 95% of points within the target), otherwise None.
    Garage 1, 10 cm: the GDOP criterion is met at M = 48 but only about 80% of points are within 10 cm in validation, so the target is not reached."""
    e = T.get(t)
    if not e or e["min_M"] is None: return None
    return e["min_M"] if e.get("confirm", {}).get("frac_points_within_target", 0) >= 0.95 else None


def fig8():
    if not need("accuracy_driven_design_p95_garage1.json", "accuracy_driven_design_p95_garage2.json"): return
    g1 = json.load(open("accuracy_driven_design_p95_garage1.json"))["targets"]; g2 = json.load(open("accuracy_driven_design_p95_garage2.json"))["targets"]
    targets = ["10.0", "15.0", "20.0"]; w = 0.36; fig, ax = plt.subplots(figsize=(6.4, 4.0)); M_CAP = 60
    for j, (T, color, lab) in enumerate(((g1, ORANGE, "Garage 1"), (g2, BLUE, "Garage 2"))):
        for i, t in enumerate(targets):
            m = reached(T, t); x = i + (j - 0.5) * w
            if m is None:
                ax.bar(x, M_CAP, w, facecolor="white", edgecolor=color, hatch="//", linewidth=1.2)
                ax.text(x, M_CAP + 2, f"not reached\n(M ≤ {M_CAP})", ha="center", va="bottom", fontsize=8.5, color="#444444")
            else:
                ax.bar(x, m, w, color=color); ax.text(x, m + 0.8, str(m), ha="center", va="bottom", fontsize=10)
    ax.bar(0, 0, color=ORANGE, label="Garage 1"); ax.bar(0, 0, color=BLUE, label="Garage 2")
    ax.set_xticks(range(3)); ax.set_xticklabels([f"{int(float(t))} cm" for t in targets]); ax.set_ylim(0, 80)
    ax.set_xlabel("Target accuracy (95% of points)"); ax.set_ylabel("Minimum anchor count M"); ax.legend(frameon=False, loc="upper right")
    fig.tight_layout(); save(fig, "fig8_min_anchors_vs_accuracy")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--resdir", default="."); ap.add_argument("--only", nargs="*", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8])
    a = ap.parse_args(); import os; os.chdir(a.resdir)  # fig 1 and the figures/ folder are written relative to the result directory
    for n in a.only:
        print(f"Figure {n}", flush=True)
        {1: lambda: __import__("runpy").run_path(str(Path(__file__).resolve().parent / "make_fig1_architecture.py")),
         2: lambda: fig_layout(GARAGE_1, "fig2_garage1_layout", (25, 15)), 3: lambda: fig_layout(GARAGE_2, "fig3_garage2_layout", (15, 10)),
         4: fig4, 5: fig5, 6: fig6, 7: fig7, 8: fig8}[n]()
