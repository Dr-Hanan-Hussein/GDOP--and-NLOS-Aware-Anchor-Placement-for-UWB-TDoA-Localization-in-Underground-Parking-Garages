"""
task_2_5_technology_comparison.py

WiFi/BLE versus UWB benchmark in Garage 1, three tiers:
  Tier 1: calibration check (open geometry, 4 corner APs, 5 test points)
  Tier 2: sparse technology-typical AP deployment (4 WiFi / 6 BLE APs) in the NLOS-heavy garage
  Tier 3: same 16-anchor infrastructure as the optimised UWB layout (UWB-matched, dense)

Paper: Section 5.4, Table 8 and Figure 7 (rows WiFi/BLE dense, WiFi/BLE sparse, open geometry).
Needs best_layouts.json (extract_best_layouts.py) for Tier 3.  Seed 2024.  Output: task_2_5_results.json
"""

import numpy as np
import json
from garage_model import build_scenario, GARAGE_1
from visibility import build_visibility_matrix
from rssi_solver import monte_carlo_trial_rssi

WIFI_PARAMS = {"A": -40, "n": 3.5, "sigma_shadow_db": 6.0}
BLE_PARAMS = {"A": -59, "n": 3.0, "sigma_shadow_db": 4.0}
NLOS_PENALTY_DB = 8.0
N_TRIALS = 150


def pick_nearest(candidates, targets):
    idx = np.array([np.argmin(np.linalg.norm(candidates - t, axis=1)) for t in targets])
    return np.unique(idx)


def run_tier(ap_coords_all, ap_vis_all, samples, params, label, apply_nlos, rng, subsample=1):
    """Only access points with line of sight to the tag are used when apply_nlos=True;
    points with fewer than three such APs are infeasible and skipped."""
    rmses = []
    skipped = 0
    for k in range(0, len(samples), subsample):
        p_true = samples[k]
        if apply_nlos:
            los_mask = ap_vis_all[:, k]
            if los_mask.sum() < 3:
                skipped += 1
                continue
            ap_coords = ap_coords_all[los_mask]
        else:
            ap_coords = ap_coords_all
        errors, _ = monte_carlo_trial_rssi(ap_coords, p_true, params["A"], params["n"],
                                             params["sigma_shadow_db"], N_TRIALS, rng)
        rmses.append(float(np.sqrt(np.mean(errors ** 2))))

    n_total = len(range(0, len(samples), subsample))
    if len(rmses) == 0:
        print(f"  {label}: INFEASIBLE - 0/{n_total} points had >=3 simultaneous LOS APs")
        return {"feasible_fraction": 0.0, "n_points_evaluated": 0}

    rmses = np.array(rmses)
    feasible_frac = len(rmses) / n_total
    print(f"  {label}: mean={rmses.mean():.1f}m, median={np.median(rmses):.1f}m, max={rmses.max():.1f}m"
          f"  (feasible at {feasible_frac*100:.0f}% of points, {skipped} skipped)")
    return {"mean_m": float(rmses.mean()), "median_m": float(np.median(rmses)), "max_m": float(rmses.max()),
            "feasible_fraction": float(feasible_frac)}


if __name__ == "__main__":
    rng = np.random.default_rng(2024)
    cfg = GARAGE_1
    scenario = build_scenario(cfg)
    candidates = scenario["candidate_AN_points"]
    samples = scenario["vehicle_sample_points"]
    vis = build_visibility_matrix(candidates, samples, scenario["columns"], scenario["parked_cars"],
                                   cfg.an_height, cfg.tag_height)
    report = {}

    print("TIER 1: Calibration (open geometry)")
    open_aps = np.array([[2, 2], [48, 2], [2, 28], [48, 28]], dtype=float)
    open_pts = np.array([[25, 15], [10, 8], [40, 22], [15, 25], [35, 5]])
    r1w = run_tier(open_aps, np.ones((4, 5), dtype=bool), open_pts, WIFI_PARAMS, "WiFi", False, rng)
    r1b = run_tier(open_aps, np.ones((4, 5), dtype=bool), open_pts, BLE_PARAMS, "BLE", False, rng)
    report["tier1_calibration"] = {"wifi": r1w, "ble": r1b}

    print("\nTIER 2: Sparse technology-typical deployment in the NLOS-heavy garage")
    wifi_idx = pick_nearest(candidates, np.array([[2, 2], [48, 2], [2, 28], [48, 28]]))
    ble_idx = pick_nearest(candidates, np.array([[2, 2], [48, 2], [2, 28], [48, 28], [25, 2], [25, 28]]))
    r2w = run_tier(candidates[wifi_idx], vis[wifi_idx, :], samples, WIFI_PARAMS, f"WiFi ({len(wifi_idx)} APs)", True, rng, subsample=1)
    r2b = run_tier(candidates[ble_idx], vis[ble_idx, :], samples, BLE_PARAMS, f"BLE ({len(ble_idx)} APs)", True, rng, subsample=1)
    report["tier2_sparse"] = {"wifi": r2w, "ble": r2b}

    print("\nTIER 3: Same infrastructure as the optimised UWB layout")
    try:
        layouts = json.load(open("best_layouts.json"))
        uwb_idx = np.array(layouts["best_ga_idx"])
        r3w = run_tier(candidates[uwb_idx], vis[uwb_idx, :], samples, WIFI_PARAMS, f"WiFi ({len(uwb_idx)} APs, UWB-matched)", True, rng, subsample=1)
        r3b = run_tier(candidates[uwb_idx], vis[uwb_idx, :], samples, BLE_PARAMS, f"BLE ({len(uwb_idx)} APs, UWB-matched)", True, rng, subsample=1)
        report["tier3_uwb_matched"] = {"wifi": r3w, "ble": r3b}
    except FileNotFoundError:
        print("  (best_layouts.json not found - run compare.py and extract_best_layouts.py first for Tier 3)")

    json.dump(report, open("task_2_5_results.json", "w"), indent=2)
    print("\nSaved task_2_5_results.json")
