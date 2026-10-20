"""rssi_sensitivity.py - sensitivity of the WiFi/BLE/UWB-RSSI vs UWB-TDoA comparison to path-loss exponent n and shadowing sigma.
Uses the regenerated Garage-1 best-GA layout, LOS-only APs (>=3), same protocol as uwb_rssi_comparison.py (N_TRIALS=150, subsample points)."""
import json, numpy as np
from garage_model import build_scenario, GARAGE_1
from visibility import build_visibility_matrix
from rssi_solver import monte_carlo_trial_rssi

cfg = GARAGE_1
sc = build_scenario(cfg); cand, samp = sc["candidate_AN_points"], sc["vehicle_sample_points"]
vis = build_visibility_matrix(cand, samp, sc["columns"], sc["parked_cars"], cfg.an_height, cfg.tag_height)
idx = np.array(json.load(open("best_layouts.json"))["best_ga_idx"])
lc, lv = cand[idx], vis[idx, :]
SUB = 2; NT = 100
pts = [k for k in range(0, len(samp), SUB) if lv[:, k].sum() >= 3]
print("points:", len(pts), "of", len(range(0, len(samp), SUB)))

def median_rmse(A, n, sig, seed):
    rng = np.random.default_rng(seed); r = []
    for k in pts:
        e, _ = monte_carlo_trial_rssi(lc[lv[:, k]], samp[k], A, n, sig, NT, rng)
        r.append(np.sqrt(np.mean(e ** 2)))
    r = np.array(r); return float(np.median(r)), float(np.mean(r))

res = {}
for n in (2.0, 2.5, 3.0, 3.5, 4.0):
    for sig in (1.0, 2.0, 4.0, 6.0, 8.0):
        med, mean = median_rmse(-45, n, sig, 7)
        res[f"n{n}_s{sig}"] = dict(n=n, sigma_db=sig, median_m=med, mean_m=mean)
        print(f"n={n} sigma={sig}dB -> median={med:.2f} m  mean={mean:.2f} m", flush=True)
# critical sigma at which median RSSI error would equal UWB-TDoA median (~0.135 m) - scan low sigma at n=3.0
for sig in (0.05, 0.1, 0.2, 0.3, 0.5):
    med, mean = median_rmse(-45, 3.0, sig, 7)
    res[f"n3.0_s{sig}"] = dict(n=3.0, sigma_db=sig, median_m=med, mean_m=mean)
    print(f"n=3.0 sigma={sig}dB -> median={med:.3f} m", flush=True)
json.dump(res, open("rssi_sensitivity_results.json", "w"), indent=2)
