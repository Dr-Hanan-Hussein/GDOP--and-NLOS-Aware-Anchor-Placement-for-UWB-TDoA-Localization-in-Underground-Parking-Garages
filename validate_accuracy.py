"""validate_accuracy.py - empirical validation of GDOP against Monte Carlo positioning error for the best PSO and GA layouts
(50 trials per point, seed 123). Paper: Table 4 Panel C, slope a in Section 3.3.
Usage: python validate_accuracy.py both <1|2> <best_layouts.json>   ->   accuracy_validation_both_garage<1|2>.json"""

import json
import numpy as np
from garage_model import build_scenario, GARAGE_1, GARAGE_2
from visibility import build_visibility_matrix
from gdop import gdop_2d
from tdoa_solver import monte_carlo_trial

N_TRIALS_PER_POINT = 50


def evaluate_layout_accuracy(layout_idx, candidates, samples, vis, cfg, rng, label=""):
    layout_idx = np.array(layout_idx)
    layout_coords = candidates[layout_idx]
    layout_vis = vis[layout_idx, :]
    dz = cfg.tag_height - cfg.an_height

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
        errors, sr = monte_carlo_trial(active_coords, p_true, cfg.sigma_tdoa_floor,
                                        cfg.sigma_tdoa_coeff, N_TRIALS_PER_POINT, rng, dz)
        rmse = np.sqrt(np.mean(errors ** 2))
        results.append({"point": p_true.tolist(), "n_active": int(active.sum()),
                         "gdop": float(g), "rmse_cm": float(rmse * 100)})
    print(f"{label}: evaluated {len(results)}/{samples.shape[0]} points")
    return results


if __name__ == "__main__":
    import sys
    rng = np.random.default_rng(123)
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    garage_choice = sys.argv[2] if len(sys.argv) > 2 else "1"
    layouts_file = sys.argv[3] if len(sys.argv) > 3 else "best_layouts.json"

    cfg = GARAGE_1 if garage_choice == "1" else GARAGE_2
    scenario = build_scenario(cfg)
    candidates = scenario["candidate_AN_points"]
    samples = scenario["vehicle_sample_points"]
    vis = build_visibility_matrix(candidates, samples, scenario["columns"], scenario["parked_cars"],
                                   cfg.an_height, cfg.tag_height)
    layouts = json.load(open(layouts_file))

    targets = [("best_pso_idx", "Best PSO"), ("best_ga_idx", "Best GA")]
    if which == "pso": targets = targets[:1]
    elif which == "ga": targets = targets[1:]

    all_results = {}
    for key, label in targets:
        results = evaluate_layout_accuracy(layouts[key], candidates, samples, vis, cfg, rng, label)
        all_results[key] = results
        rmses = np.array([r["rmse_cm"] for r in results])
        gdops = np.array([r["gdop"] for r in results])
        print(f"  RMSE: mean={rmses.mean():.1f}cm median={np.median(rmses):.1f}cm max={rmses.max():.1f}cm")
        print(f"  Correlation(GDOP,RMSE): r={np.corrcoef(gdops,rmses)[0,1]:.3f}")

    out = f"accuracy_validation_{which}_garage{garage_choice}.json"
    json.dump(all_results, open(out, "w"), indent=2)
    print(f"Saved {out}")
