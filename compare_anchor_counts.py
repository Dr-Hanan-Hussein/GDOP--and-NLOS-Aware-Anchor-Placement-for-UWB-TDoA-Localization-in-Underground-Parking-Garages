"""
compare_anchor_counts.py

Systematic n = 3, 4, 5, 6 anchor-count reliability sweep with the enhanced solver
(Chan-Ho initialisation, bounded TRF, distance-dependent noise).

Paper: Section 5.1 and Table 3 (TDoA reliability versus active anchor count, Garage 1).
Seed 555.  Output: anchor_count_sweep_enhanced.json
"""

import numpy as np
import json

from garage_model import build_scenario, GARAGE_1
from visibility import build_visibility_matrix
from gdop import gdop_2d
from tdoa_solver import simulate_measurements, solve_position

N_TRIALS = 80
N_REPEATS_PER_POINT = 2


def run_sweep(cfg, n_values, n_points, seed=555):
    rng = np.random.default_rng(seed)
    dz = cfg.tag_height - cfg.an_height

    scenario = build_scenario(cfg)
    candidates = scenario["candidate_AN_points"]
    samples = scenario["vehicle_sample_points"]
    vis = build_visibility_matrix(candidates, samples, scenario["columns"], scenario["parked_cars"],
                                   cfg.an_height, cfg.tag_height)

    max_n = max(n_values)
    visible_counts = vis.sum(axis=0)
    eligible_points = np.where(visible_counts >= max_n)[0]
    chosen_points = rng.choice(eligible_points, size=min(n_points, len(eligible_points)), replace=False)

    results = {n: {"rmse_cm": [], "fail_100cm": [], "gdop": []} for n in n_values}

    for pt_idx in chosen_points:
        p_true = samples[pt_idx]
        visible_candidates = np.where(vis[:, pt_idx])[0]
        for n in n_values:
            for _ in range(N_REPEATS_PER_POINT):
                subset = rng.choice(visible_candidates, size=n, replace=False)
                coords = candidates[subset]
                g = gdop_2d(coords, p_true)

                errors = np.empty(N_TRIALS)
                for t in range(N_TRIALS):
                    D = simulate_measurements(coords, p_true, cfg.sigma_tdoa_floor,
                                               cfg.sigma_tdoa_coeff, rng, dz)
                    p_hat, ok = solve_position(coords, D, dz)
                    errors[t] = np.linalg.norm(p_hat - p_true)

                rmse = np.sqrt(np.mean(errors ** 2)) * 100
                fail_rate = (errors * 100 > 100).mean()
                results[n]["rmse_cm"].append(float(rmse))
                results[n]["fail_100cm"].append(float(fail_rate))
                results[n]["gdop"].append(float(g) if np.isfinite(g) else None)

    return results, len(chosen_points)


if __name__ == "__main__":
    import time
    t0 = time.time()
    n_values = [3, 4, 5, 6]
    results, n_points = run_sweep(GARAGE_1, n_values, n_points=20)
    elapsed = time.time() - t0

    print(f"Swept n={n_values} across {n_points} points, enhanced solver, {elapsed:.1f}s\n")
    print(f"{'n anchors':<12}{'median RMSE':<14}{'p90':<12}{'fail>100cm':<14}{'median GDOP':<12}")
    for n in n_values:
        rmses = np.array(results[n]["rmse_cm"])
        fails = np.array(results[n]["fail_100cm"])
        gdops = np.array([g for g in results[n]["gdop"] if g is not None])
        print(f"{n:<12}{np.median(rmses):<14.1f}{np.percentile(rmses,90):<12.1f}"
              f"{np.mean(fails)*100:<14.1f}{np.median(gdops):<12.2f}")

    for n in [3, 4]:
        rmses = np.array(results[n]["rmse_cm"])
        gdops = np.array([g if g is not None else np.inf for g in results[n]["gdop"]])
        low, high = gdops <= 3.0, gdops > 3.0
        print(f"\nn={n}: GDOP<=3.0 (n={low.sum()}): fail>100cm={100*(rmses[low]>100).mean() if low.sum() else 0:.1f}%")
        print(f"n={n}: GDOP>3.0  (n={high.sum()}): fail>100cm={100*(rmses[high]>100).mean() if high.sum() else 0:.1f}%")

    json.dump(results, open("anchor_count_sweep_enhanced.json", "w"), indent=2)
    print("\nSaved anchor_count_sweep_enhanced.json")
