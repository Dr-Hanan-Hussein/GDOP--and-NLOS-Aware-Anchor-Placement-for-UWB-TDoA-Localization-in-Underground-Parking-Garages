"""
angle_penalty_experiment.py

Tests whether the GDOP-invisible redundant-anchor failure mode (Section 5.2) can be reduced by penalising near-collinear
anchor directions in the fitness function. Three phases:

  PHASE 1 (diagnostic): does the condition number kappa = sqrt(lambda_max / lambda_min) of the full active-anchor Jacobian H
  (the same H used for GDOP) separate known failure points from healthy points?
  Needs accuracy_validation_both_garage1.json and best_layouts.json (run validate_accuracy.py and extract_best_layouts.py first).

  PHASE 2 (diagnostic): the minimum pairwise angle between any two active anchors' direction vectors at a point, tested the same way.

  PHASE 3 (experiment): adds a penalty ANGLE_PENALTY_WEIGHT * (ANGLE_THRESHOLD_DEG - theta_min) when theta_min < ANGLE_THRESHOLD_DEG
  (30 * (5 deg - theta)), re-optimises Garage 1 with it (GA, M = 16) and validates each resulting layout by Monte Carlo TDoA solving,
  for three seeds (2026, 3033, 4044).

Paper: Section 5.2 and Table 4 Panel A. The unmodified-GA baseline printed at the end (3.2% / 1.6% / 13.5 cm) is the hard-coded
reference value of Table 4 Panel A.  Output: angle_penalty_validation.json
"""

import time
import json
import numpy as np

import ga
from fitness import DEFAULT_PARAMS, is_connected, min_pairwise_distance
from gdop import gdop_all_points, gdop_2d
from garage_model import build_scenario, GARAGE_1
from visibility import build_visibility_matrix
from tdoa_solver import monte_carlo_trial
from accuracy_driven_design import calibrate_slope

ANGLE_THRESHOLD_DEG = 5.0
ANGLE_PENALTY_WEIGHT = 30.0
SWEEP_GENERATIONS = 150
POP_SIZE = 40
N_TRIALS_VALIDATE = 50


# ----------------------------------------------------------------------
# Shared geometry helpers
# ----------------------------------------------------------------------
def condition_number(anchor_xy, point_xy):
    """kappa = sqrt(lambda_max/lambda_min) of the same H^T H used for GDOP."""
    diffs = anchor_xy - point_xy
    dists = np.linalg.norm(diffs, axis=1)
    dists[dists == 0] = 1e-6
    u = diffs / dists[:, None]
    u_bar = u.mean(axis=0)
    centered = u - u_bar
    Mmat = centered.T @ centered
    eigs = np.clip(np.linalg.eigvalsh(Mmat), 1e-12, None)
    return np.sqrt(eigs.max() / eigs.min())


def worst_pair_score(anchor_xy, point_xy):
    """Smallest pairwise angle (deg) between any two active anchors'
    direction vectors, and that pair's distance ratio."""
    diffs = anchor_xy - point_xy
    dists = np.linalg.norm(diffs, axis=1)
    dists[dists == 0] = 1e-6
    u = diffs / dists[:, None]
    n = len(anchor_xy)
    worst_angle_deg = 180.0
    worst_ratio = 1.0
    for i in range(n):
        for j in range(i + 1, n):
            cos_ang = np.clip(np.dot(u[i], u[j]), -1, 1)
            ang = np.degrees(np.arccos(cos_ang))
            ratio = max(dists[i], dists[j]) / min(dists[i], dists[j])
            if ang < worst_angle_deg:
                worst_angle_deg = ang
                worst_ratio = ratio
    return worst_angle_deg, worst_ratio


def worst_pair_angle_per_point(anchor_xy, sample_xy, active_mask):
    """Per-point version used inside the fitness penalty: for
    each sample point k, the smallest pairwise angle (degrees) between any
    two active anchors. 180 deg (no penalty) if fewer than 2 active."""
    K = sample_xy.shape[0]
    out = np.full(K, 180.0)
    for k in range(K):
        active = np.where(active_mask[:, k])[0]
        if len(active) < 2:
            continue
        diffs = anchor_xy[active] - sample_xy[k]
        dists = np.linalg.norm(diffs, axis=1)
        dists[dists == 0] = 1e-6
        u = diffs / dists[:, None]
        cos_mat = np.clip(u @ u.T, -1, 1)
        np.fill_diagonal(cos_mat, -1)
        out[k] = np.degrees(np.arccos(cos_mat.max()))
    return out


# ----------------------------------------------------------------------
# PHASE 1 + 2: diagnostics against already-known failure points
# ----------------------------------------------------------------------
def run_diagnostics(candidates, samples, vis, a_slope):
    layouts = json.load(open("best_layouts.json"))
    val = json.load(open("accuracy_validation_both_garage1.json"))

    for key, label in [("best_pso_idx", "PSO"), ("best_ga_idx", "GA")]:
        layout_idx = np.array(layouts[key])
        layout_coords = candidates[layout_idx]
        layout_vis = vis[layout_idx, :]
        pts = val[key]
        print(f"\n--- {label} layout: PHASE 1 (full-Jacobian condition number) ---")
        conds_bad, conds_good = [], []
        for p in pts:
            px, py = p["point"]
            k = np.argmin(np.sum((samples - np.array([px, py])) ** 2, axis=1))
            active = layout_vis[:, k]
            coords = layout_coords[active]
            cond = condition_number(coords, np.array([px, py]))
            predicted = a_slope * p["gdop"]
            is_bad = (p["gdop"] <= 3.0) and (p["rmse_cm"] > max(2 * predicted, 100))
            (conds_bad if is_bad else conds_good).append(cond)
        conds_bad, conds_good = np.array(conds_bad), np.array(conds_good)
        print(f"  n_bad={len(conds_bad)}  n_good={len(conds_good)}")
        if len(conds_bad):
            print(f"  BAD:  mean={conds_bad.mean():.2f} min={conds_bad.min():.2f} max={conds_bad.max():.2f}")
        print(f"  GOOD: mean={conds_good.mean():.2f} min={conds_good.min():.2f} max={conds_good.max():.2f}")

        print(f"--- {label} layout: PHASE 2 (worst-pairwise-angle) ---")
        good_angles = []
        for p in pts:
            px, py = p["point"]
            k = np.argmin(np.sum((samples - np.array([px, py])) ** 2, axis=1))
            active = layout_vis[:, k]
            coords = layout_coords[active]
            ang, ratio = worst_pair_score(coords, np.array([px, py]))
            predicted = a_slope * p["gdop"]
            is_bad = (p["gdop"] <= 3.0) and (p["rmse_cm"] > max(2 * predicted, 100))
            if is_bad:
                print(f"  BAD  gdop={p['gdop']:.2f} rmse={p['rmse_cm']:.0f}cm  "
                      f"worst_angle={ang:5.1f} deg  dist_ratio={ratio:.2f}")
            else:
                good_angles.append(ang)
        good_angles = np.array(good_angles)
        print(f"  GOOD worst_angle: mean={good_angles.mean():.1f} median={np.median(good_angles):.1f} "
              f"min={good_angles.min():.1f} (n={len(good_angles)})")


# ----------------------------------------------------------------------
# PHASE 3: fitness-function penalty + real optimization + validation
# ----------------------------------------------------------------------
def evaluate_with_angle_penalty(selected_idx, candidates, samples, vis_matrix, params=None, verbose=False):
    p = {**DEFAULT_PARAMS, **(params or {})}
    selected_idx = np.asarray(selected_idx)
    diag = {"status": None}
    if len(selected_idx) > p["M_max"]:
        diag["status"] = "infeasible: exceeds M_max"
        return np.inf, diag
    sel_coords = candidates[selected_idx]
    if not is_connected(sel_coords, p["R_sync"]):
        diag["status"] = "infeasible: sync graph disconnected"
        return np.inf, diag
    if min_pairwise_distance(sel_coords) < p["d_min"]:
        diag["status"] = "infeasible: anchors too close (violates d_min)"
        return np.inf, diag

    active_mask = np.zeros_like(vis_matrix, dtype=bool)
    active_mask[selected_idx, :] = vis_matrix[selected_idx, :]
    n_active = active_mask.sum(axis=0)
    covered = n_active >= p["min_los_anchors"]
    coverage_frac = covered.mean()
    if coverage_frac == 0.0:
        diag["status"] = "infeasible: 0% coverage"
        return np.inf, diag

    covered_idx = np.where(covered)[0]
    gdop = gdop_all_points(candidates, samples[covered_idx], active_mask[:, covered_idx])
    f1 = gdop.mean()
    f2 = gdop.max()
    f3 = len(selected_idx)
    F = p["w1"] * f1 + p["w2"] * f2 + p["w3"] * f3
    if f2 > p["GDOP_max"]:
        F += p["gdop_penalty_weight"] * (f2 - p["GDOP_max"])
    if coverage_frac < p["coverage_threshold"]:
        F += p["coverage_penalty_weight"] * (p["coverage_threshold"] - coverage_frac)

    worst_angles = worst_pair_angle_per_point(candidates, samples[covered_idx], active_mask[:, covered_idx])
    min_angle_found = worst_angles.min()
    if min_angle_found < ANGLE_THRESHOLD_DEG:
        F += ANGLE_PENALTY_WEIGHT * (ANGLE_THRESHOLD_DEG - min_angle_found)

    diag.update({"status": "feasible", "coverage_frac": coverage_frac, "f1_mean_gdop": f1,
                 "f2_max_gdop": f2, "f3_n_anchors": f3, "min_pair_angle_deg": float(min_angle_found), "F": F})
    if verbose:
        print(diag)
    return F, diag


def run_mitigation_experiment(candidates, samples, vis, cfg, a_slope, seeds=(2026, 3033, 4044)):
    ga.evaluate = evaluate_with_angle_penalty
    fp = {**DEFAULT_PARAMS, "M_max": 16}
    dz = cfg.tag_height - cfg.an_height
    all_results = []
    for seed in seeds:
        print(f"\nRunning GA with angle-redundancy penalty (threshold={ANGLE_THRESHOLD_DEG} deg, "
              f"weight={ANGLE_PENALTY_WEIGHT}), M=16, seed={seed}, {SWEEP_GENERATIONS} generations...")
        t0 = time.time()
        idx, F, hist, n_evals = ga.genetic_algorithm(candidates, samples, vis, M=16, pop_size=POP_SIZE,
                                                      generations=SWEEP_GENERATIONS, seed=seed, fitness_params=fp)
        _, diag = evaluate_with_angle_penalty(idx, candidates, samples, vis, fp)
        print(f"Done in {time.time()-t0:.0f}s. F={F:.2f}  worst-case GDOP={diag['f2_max_gdop']:.2f}  "
              f"min pairwise angle found={diag['min_pair_angle_deg']:.1f} deg")

        rng = np.random.default_rng(123)
        layout_coords = candidates[idx]
        layout_vis = vis[idx, :]
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
            errors, sr = monte_carlo_trial(active_coords, p_true, cfg.sigma_tdoa_floor, cfg.sigma_tdoa_coeff,
                                            N_TRIALS_VALIDATE, rng, dz)
            rmse = np.sqrt(np.mean(errors ** 2))
            results.append({"point": p_true.tolist(), "n_active": int(active.sum()), "gdop": float(g),
                             "rmse_cm": float(rmse * 100)})

        gdop_arr = np.array([r["gdop"] for r in results])
        rmse_arr = np.array([r["rmse_cm"] for r in results])
        well_cond = gdop_arr <= 3.0
        predicted = a_slope * gdop_arr[well_cond]
        actual = rmse_arr[well_cond]
        exceeds_2x = (actual > 2 * predicted).sum()
        catastrophic = (actual > 100).sum()
        n_wc = well_cond.sum()
        print(f"  seed {seed}: {exceeds_2x}/{n_wc} ({100*exceeds_2x/n_wc:.1f}%) exceed 2x predicted; "
              f"{catastrophic}/{n_wc} ({100*catastrophic/n_wc:.1f}%) exceed 100cm; "
              f"median={np.median(rmse_arr):.1f}cm")
        all_results.append({"seed": seed, "layout_idx": idx.tolist(), "F": float(F),
                             "pct_exceeds_2x": 100 * exceeds_2x / n_wc, "pct_exceeds_100cm": 100 * catastrophic / n_wc,
                             "median_error_cm": float(np.median(rmse_arr)), "validation": results})
    json.dump(all_results, open("angle_penalty_validation.json", "w"), indent=2)
    print("\nBaseline (unmodified GA, no penalty): 3.2% / 1.6% exceed thresholds, median 13.5cm")
    return all_results


if __name__ == "__main__":
    cfg = GARAGE_1
    scenario = build_scenario(cfg)
    candidates = scenario["candidate_AN_points"]
    samples = scenario["vehicle_sample_points"]
    vis = build_visibility_matrix(candidates, samples, scenario["columns"], scenario["parked_cars"],
                                   cfg.an_height, cfg.tag_height)
    a_slope, _ = calibrate_slope("accuracy_validation_both_garage1.json")

    run_diagnostics(candidates, samples, vis, a_slope)
    run_mitigation_experiment(candidates, samples, vis, cfg, a_slope)
