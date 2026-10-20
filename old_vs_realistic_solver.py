"""
old_vs_realistic_solver.py

Paired comparison of the OLD solver configuration (naive centroid initialisation,
unconstrained Levenberg-Marquardt, flat 10 cm noise) against the REALISTIC
configuration (Chan-Ho initialisation, bounded trust-region-reflective,
distance-dependent noise). Same layout, same points, same random seed (999);
only the solver configuration differs.

Paper: Table 4 Panel B and Figure 4 (make_old_vs_realistic_chart.py).
Needs best_layouts.json (extract_best_layouts.py).  Output: old_vs_realistic_comparison.json
"""

import numpy as np
import json
from scipy.optimize import least_squares

from garage_model import build_scenario, GARAGE_1
from visibility import build_visibility_matrix
from gdop import gdop_2d
from tdoa_solver import ranges, chan_ho_initial_estimate

N_TRIALS = 60


def simulate_measurements_flat(anchor_xy, p_true, sigma_flat, rng, dz):
    """OLD noise model: single flat sigma applied directly to the
    range-difference (not per-anchor, not distance-dependent)."""
    r = ranges(anchor_xy, p_true, dz)
    D_true = r[1:] - r[0]
    noise = rng.normal(0, sigma_flat, size=len(D_true))
    return D_true + noise


def simulate_measurements_distdep(anchor_xy, p_true, sigma_floor, sigma_coeff, rng, dz):
    """NEW noise model: distance-dependent, per-anchor."""
    r = ranges(anchor_xy, p_true, dz)
    sigma_per_anchor = sigma_floor + sigma_coeff * r
    noise = rng.normal(0, sigma_per_anchor)
    r_noisy = r + noise
    return r_noisy[1:] - r_noisy[0]


def solve_old(anchor_xy, D_measured, dz):
    """OLD solver: naive centroid init, unconstrained LM."""
    p_init = anchor_xy.mean(axis=0)

    def residuals(p_xy):
        r = ranges(anchor_xy, p_xy, dz)
        return D_measured - (r[1:] - r[0])

    result = least_squares(residuals, p_init, method='lm')
    return result.x, result.success


def solve_realistic(anchor_xy, D_measured, dz):
    """NEW solver: Chan-Ho init, bounded TRF."""
    p_init = chan_ho_initial_estimate(anchor_xy, D_measured)
    margin = 20.0
    lo = anchor_xy.min(axis=0) - margin
    hi = anchor_xy.max(axis=0) + margin
    p_init = np.clip(p_init, lo, hi)

    def residuals(p_xy):
        r = ranges(anchor_xy, p_xy, dz)
        return D_measured - (r[1:] - r[0])

    result = least_squares(residuals, p_init, method='trf', bounds=(lo, hi))
    return result.x, result.success


def evaluate_layout(layout_idx, candidates, samples, vis, cfg, seed, config):
    """config: 'old' or 'realistic'"""
    layout_coords = candidates[layout_idx]
    layout_vis = vis[layout_idx, :]
    dz = cfg.tag_height - cfg.an_height
    rng = np.random.default_rng(seed)

    results = []
    for k in range(samples.shape[0]):
        active = layout_vis[:, k]
        if active.sum() < 3:
            continue
        active_coords = layout_coords[active]
        p_true = samples[k]
        g = gdop_2d(active_coords, p_true)
        if not np.isfinite(g):
            continue

        errors = np.empty(N_TRIALS)
        for t in range(N_TRIALS):
            if config == 'old':
                D = simulate_measurements_flat(active_coords, p_true, 0.10, rng, dz)  # old flat 10cm
                p_hat, ok = solve_old(active_coords, D, dz)
            else:
                D = simulate_measurements_distdep(active_coords, p_true, cfg.sigma_tdoa_floor,
                                                    cfg.sigma_tdoa_coeff, rng, dz)
                p_hat, ok = solve_realistic(active_coords, D, dz)
            errors[t] = np.linalg.norm(p_hat - p_true)

        rmse = np.sqrt(np.mean(errors ** 2))
        results.append({"n_active": int(active.sum()), "gdop": float(g),
                         "rmse_cm": float(rmse * 100), "median_cm": float(np.median(errors) * 100)})
    return results


if __name__ == "__main__":
    scenario = build_scenario(GARAGE_1)
    candidates = scenario["candidate_AN_points"]
    samples = scenario["vehicle_sample_points"]
    vis = build_visibility_matrix(candidates, samples, scenario["columns"], scenario["parked_cars"],
                                   GARAGE_1.an_height, GARAGE_1.tag_height)
    layouts = json.load(open("best_layouts.json"))

    report = {}
    for key, label in [("best_pso_idx", "PSO"), ("best_ga_idx", "GA")]:
        layout_idx = np.array(layouts[key])
        print(f"=== {label} layout ===")
        old_results = evaluate_layout(layout_idx, candidates, samples, vis, GARAGE_1, seed=999, config='old')
        real_results = evaluate_layout(layout_idx, candidates, samples, vis, GARAGE_1, seed=999, config='realistic')

        old_rmse = np.array([r["rmse_cm"] for r in old_results])
        real_rmse = np.array([r["rmse_cm"] for r in real_results])
        old_gdop = np.array([r["gdop"] for r in old_results])
        real_gdop = np.array([r["gdop"] for r in real_results])

        print(f"  OLD (naive init, unconstrained LM, flat 10cm noise):")
        print(f"    median={np.median(old_rmse):.1f}cm mean={old_rmse.mean():.1f}cm max={old_rmse.max():.1f}cm "
              f"corr(GDOP,err)={np.corrcoef(old_gdop, old_rmse)[0,1]:.3f}")
        print(f"  REALISTIC (Chan-Ho init, bounded TRF, distance-dependent noise):")
        print(f"    median={np.median(real_rmse):.1f}cm mean={real_rmse.mean():.1f}cm max={real_rmse.max():.1f}cm "
              f"corr(GDOP,err)={np.corrcoef(real_gdop, real_rmse)[0,1]:.3f}")

        report[key] = {
            "old": {"median_cm": float(np.median(old_rmse)), "mean_cm": float(old_rmse.mean()),
                    "max_cm": float(old_rmse.max()), "corr": float(np.corrcoef(old_gdop, old_rmse)[0, 1])},
            "realistic": {"median_cm": float(np.median(real_rmse)), "mean_cm": float(real_rmse.mean()),
                          "max_cm": float(real_rmse.max()), "corr": float(np.corrcoef(real_gdop, real_rmse)[0, 1])},
        }

    json.dump(report, open("old_vs_realistic_comparison.json", "w"), indent=2)
    print("\nSaved old_vs_realistic_comparison.json")
