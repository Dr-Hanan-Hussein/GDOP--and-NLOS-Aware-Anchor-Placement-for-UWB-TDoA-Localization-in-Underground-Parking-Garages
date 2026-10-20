"""
uwb_rssi_comparison.py

Applies the RSSI positioning model to UWB hardware (not its native TDoA mode),
to separate the effect of the positioning method (TDoA versus RSSI) from the
effect of the radio hardware.

Paper: Section 5.4, Table 8 (rows "UWB hardware, RSSI mode" and "UWB-RSSI, open/ideal geometry").
Needs best_layouts.json (extract_best_layouts.py).  Output: uwb_rssi_comparison_results.json
The WiFi/BLE rows of Table 8 come from task_2_5_technology_comparison.py; the UWB-TDoA row from validate_accuracy.py.

PARAMETER NOTE. sigma_shadow_db for UWB-RSSI (2.0 dB, with A = -45 dBm and n = 3.0) is an ENGINEERING ESTIMATE, not a measured
value: UWB systems normally operate in ToA/TDoA mode, so no directly comparable published RSSI-only shadowing figure was used.
It is set at roughly one third of the 6 dB used for WiFi, reflecting the lower multipath fading of wideband UWB, and is flagged in
the paper as requiring a hardware calibration campaign.
"""

import numpy as np
import json

from garage_model import build_scenario, GARAGE_1
from visibility import build_visibility_matrix
from rssi_solver import monte_carlo_trial_rssi

UWB_RSSI_PARAMS = {"A": -45, "n": 3.0, "sigma_shadow_db": 2.0}  # engineering estimate, see docstring
WIFI_PARAMS = {"A": -40, "n": 3.5, "sigma_shadow_db": 6.0}
BLE_PARAMS = {"A": -59, "n": 3.0, "sigma_shadow_db": 4.0}
NLOS_PENALTY_DB = 8.0  # kept identical across all three technologies
N_TRIALS = 150


def run_tier(ap_coords_all, ap_vis_all, samples, params, label, apply_nlos, rng, subsample=1):
    """LOS-only access points are used when apply_nlos=True (see task_2_5_technology_comparison.py,
    run_tier docstring): a flat NLOS penalty applied to all APs would produce an unrealistic systematic bias."""
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
        return {"feasible_fraction": 0.0}
    rmses = np.array(rmses)
    print(f"  {label}: mean={rmses.mean():.2f}m, median={np.median(rmses):.2f}m, max={rmses.max():.2f}m"
          f"  (feasible {len(rmses)/n_total*100:.0f}%)")
    return {"mean_m": float(rmses.mean()), "median_m": float(np.median(rmses)), "max_m": float(rmses.max())}


if __name__ == "__main__":
    rng = np.random.default_rng(2024)
    cfg = GARAGE_1
    scenario = build_scenario(cfg)
    candidates = scenario["candidate_AN_points"]
    samples = scenario["vehicle_sample_points"]
    vis = build_visibility_matrix(candidates, samples, scenario["columns"], scenario["parked_cars"],
                                   cfg.an_height, cfg.tag_height)
    report = {}

    print("TIER 1: Calibration check (open geometry, no NLOS) - sanity check before the garage run")
    open_aps = np.array([[2, 2], [48, 2], [2, 28], [48, 28]], dtype=float)
    open_pts = np.array([[25, 15], [10, 8], [40, 22], [15, 25], [35, 5]])
    r1u = run_tier(open_aps, np.ones((4, 5), dtype=bool), open_pts, UWB_RSSI_PARAMS,
                    "UWB-RSSI (engineering estimate)", False, rng)
    r1w = run_tier(open_aps, np.ones((4, 5), dtype=bool), open_pts, WIFI_PARAMS, "WiFi (for reference)", False, rng)
    report["tier1_calibration"] = {"uwb_rssi": r1u, "wifi": r1w}

    print("\nTIER 3: Same 16-anchor UWB-optimized layout, all technologies")
    layouts = json.load(open("best_layouts.json"))
    uwb_idx = np.array(layouts["best_ga_idx"])
    r3u = run_tier(candidates[uwb_idx], vis[uwb_idx, :], samples, UWB_RSSI_PARAMS,
                    "UWB hardware + RSSI (not its native TDoA mode)", True, rng, subsample=1)
    r3w = run_tier(candidates[uwb_idx], vis[uwb_idx, :], samples, WIFI_PARAMS,
                    "WiFi-RSSI (native mode)", True, rng, subsample=1)
    r3b = run_tier(candidates[uwb_idx], vis[uwb_idx, :], samples, BLE_PARAMS,
                    "BLE-RSSI (native mode)", True, rng, subsample=1)
    report["tier3_same_deployment"] = {"uwb_rssi": r3u, "wifi_rssi": r3w, "ble_rssi": r3b}
    report["uwb_tdoa_reference"] = {"median_cm": 13.5, "note": "UWB-TDoA native mode (Table 8), from validate_accuracy.py with the realistic solver: Chan-Ho init, bounded TRF, distance-dependent noise. Hard-coded reference value."}

    json.dump(report, open("uwb_rssi_comparison_results.json", "w"), indent=2)
    print("\nSaved uwb_rssi_comparison_results.json")
