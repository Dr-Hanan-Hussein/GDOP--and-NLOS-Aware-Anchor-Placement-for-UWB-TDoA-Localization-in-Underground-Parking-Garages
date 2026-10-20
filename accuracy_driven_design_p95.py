"""
accuracy_driven_design_p95.py

Accuracy-driven design, 95%-service-level (P95) form: the accuracy target is relaxed from "the single worst covered
point must meet the target" (f2 = max GDOP in fitness.py) to "95% of covered points must meet the target"
(f2 = P95 of GDOP) - the same 95% philosophy the framework already uses for the coverage constraint C1, applied to
the accuracy constraint C2.

Paper: Table 9 (Garage 1, targets 15 and 20 cm; the 10 cm target is run by accuracy_driven_design_p95_10cm.py, because with the
strict criterion Garage 1 has a worst-case GDOP floor near 16 cm, Section 5.5).
Needs accuracy_validation_both_garage1.json.  Output: accuracy_driven_design_p95_garage1.json

Implementation note: ga.py and pso.py both do `from fitness import
evaluate` at import time, so patching fitness.evaluate afterward would
NOT affect them. Instead we monkey-patch the name bound inside each
already-imported module (ga.evaluate / pso.evaluate) to point at the
percentile-based evaluate_percentile function below, for the duration of
this script only. Nothing in fitness.py, ga.py, or pso.py is edited, so
every previously-validated result (compare.py, compare_anchor_counts.py,
etc.) remains untouched and reproducible.
"""

import json
import time
import numpy as np

import ga
import pso
from fitness import DEFAULT_PARAMS, is_connected, min_pairwise_distance, evaluate as evaluate_strict
from gdop import gdop_all_points
from garage_model import build_scenario, GARAGE_1
from visibility import build_visibility_matrix
from tdoa_solver import monte_carlo_trial

PERCENTILE = 95
TARGETS_CM = [15.0, 20.0]   # 10 cm: accuracy_driven_design_p95_10cm.py
M_GRID = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
SWEEP_GENERATIONS = 80
FULL_GENERATIONS = 150
POP_SIZE = 40
N_TRIALS_VALIDATE = 50


def evaluate_percentile(selected_idx, candidates, samples, vis_matrix, params=None, verbose=False):
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
    finite = gdop[np.isfinite(gdop)]

    f1 = float(finite.mean()) if len(finite) else np.inf
    f2 = float(np.percentile(finite, PERCENTILE)) if len(finite) else np.inf   # <- the only real change
    f3 = len(selected_idx)
    F = p["w1"] * f1 + p["w2"] * f2 + p["w3"] * f3
    if f2 > p["GDOP_max"]:
        F += p["gdop_penalty_weight"] * (f2 - p["GDOP_max"])
    if coverage_frac < p["coverage_threshold"]:
        F += p["coverage_penalty_weight"] * (p["coverage_threshold"] - coverage_frac)

    diag.update({"status": "feasible", "coverage_frac": coverage_frac, "f1_mean_gdop": f1,
                 "f2_p95_gdop": f2, "f3_n_anchors": f3, "F": F,
                 "n_degenerate_excluded": int((~np.isfinite(gdop)).sum())})
    if verbose:
        print(diag)
    return F, diag


# monkey-patch: redirect ga.py's and pso.py's internal evaluate() calls
ga.evaluate = evaluate_percentile
pso.evaluate = evaluate_percentile


def sweep_M(candidates, samples, vis, gdop_max_target, m_grid, seed=2026):
    results = {}
    for M in m_grid:
        fp = {**DEFAULT_PARAMS, "M_max": M, "GDOP_max": gdop_max_target}
        idx, F, hist, n_evals = ga.genetic_algorithm(candidates, samples, vis, M=M, pop_size=POP_SIZE,
                                                      generations=SWEEP_GENERATIONS, seed=seed, fitness_params=fp)
        _, diag = evaluate_percentile(idx, candidates, samples, vis, fp)
        ok = diag.get("status") == "feasible" and diag.get("coverage_frac", 0) >= 0.95 \
            and diag.get("f2_p95_gdop", np.inf) <= gdop_max_target
        results[M] = {"coverage_frac": diag.get("coverage_frac", 0.0),
                      "f2_p95_gdop": diag.get("f2_p95_gdop", np.inf),
                      "f1_mean_gdop": diag.get("f1_mean_gdop", np.inf), "F": F,
                      "status": diag["status"], "meets_target": bool(ok)}
        print(f"    M={M:3d}  coverage={results[M]['coverage_frac']*100:5.1f}%  "
              f"P95-GDOP={results[M]['f2_p95_gdop']:.2f}  (target<={gdop_max_target:.2f})  "
              f"{'OK' if ok else '--'}")
    feasible_Ms = [M for M in m_grid if results[M]["meets_target"]]
    return results, (min(feasible_Ms) if feasible_Ms else None)


def confirm_layout(candidates, samples, vis, M, gdop_max_target, cfg, target_cm, seed=2026):
    fp = {**DEFAULT_PARAMS, "M_max": M, "GDOP_max": gdop_max_target}
    idx_ga, F_ga, _, _ = ga.genetic_algorithm(candidates, samples, vis, M=M, pop_size=POP_SIZE,
                                               generations=FULL_GENERATIONS, seed=seed, fitness_params=fp)
    idx_pso, F_pso, _, _ = pso.binary_pso(candidates, samples, vis, M=M, swarm_size=POP_SIZE,
                                           iters=FULL_GENERATIONS, seed=seed, fitness_params=fp)
    winner_idx, winner_alg = (idx_ga, "GA") if F_ga <= F_pso else (idx_pso, "PSO")

    dz = cfg.tag_height - cfg.an_height
    rng = np.random.default_rng(4242)
    layout_coords = candidates[winner_idx]
    layout_vis = vis[winner_idx, :]
    real_errors_cm, n_evaluated = [], 0
    for k in range(samples.shape[0]):
        active = layout_vis[:, k]
        if active.sum() < 3:
            continue
        p_true = samples[k]
        errors, _ = monte_carlo_trial(layout_coords[active], p_true, cfg.sigma_tdoa_floor,
                                       cfg.sigma_tdoa_coeff, N_TRIALS_VALIDATE, rng, dz)
        real_errors_cm.append(float(np.median(errors) * 100))
        n_evaluated += 1
    real_errors_cm = np.array(real_errors_cm)

    return {"M": M, "winner_alg": winner_alg, "F_GA": float(F_ga), "F_PSO": float(F_pso),
            "layout_idx": winner_idx.tolist(), "n_points_evaluated": n_evaluated,
            "n_points_total": int(samples.shape[0]),
            "real_median_error_cm": float(np.median(real_errors_cm)),
            "real_p95_error_cm": float(np.percentile(real_errors_cm, 95)),
            "real_max_error_cm": float(real_errors_cm.max()),
            "frac_points_within_target": float((real_errors_cm <= target_cm).mean())}


if __name__ == "__main__":
    cfg = GARAGE_1
    scenario = build_scenario(cfg)
    candidates = scenario["candidate_AN_points"]
    samples = scenario["vehicle_sample_points"]
    vis = build_visibility_matrix(candidates, samples, scenario["columns"], scenario["parked_cars"],
                                   cfg.an_height, cfg.tag_height)

    from accuracy_driven_design import calibrate_slope
    a_slope, slopes = calibrate_slope("accuracy_validation_both_garage1.json")
    print(f"Calibrated slope a={a_slope:.2f} cm/GDOP  (P{PERCENTILE} form, not strict worst-case)")

    all_results = {"garage": cfg.name, "slope_a": a_slope, "percentile": PERCENTILE, "targets": {}}
    for target_cm in TARGETS_CM:
        gdop_max_target = target_cm / a_slope
        print(f"\n--- Target {target_cm:.0f}cm (P{PERCENTILE})  ->  GDOP_max={gdop_max_target:.3f} ---")
        t0 = time.time()
        sweep_results, min_M = sweep_M(candidates, samples, vis, gdop_max_target, M_GRID)
        print(f"  sweep took {time.time()-t0:.1f}s. Minimal feasible M = {min_M}")
        entry = {"gdop_max_target": gdop_max_target, "sweep": sweep_results, "min_M": min_M}
        if min_M is not None:
            t0 = time.time()
            confirm = confirm_layout(candidates, samples, vis, min_M, gdop_max_target, cfg, target_cm)
            entry["confirm"] = confirm
            print(f"  confirmation ({confirm['winner_alg']} won) took {time.time()-t0:.1f}s")
            print(f"  REAL validation: median={confirm['real_median_error_cm']:.1f}cm  "
                  f"p95={confirm['real_p95_error_cm']:.1f}cm  max={confirm['real_max_error_cm']:.1f}cm  "
                  f"fraction<={target_cm:.0f}cm={confirm['frac_points_within_target']*100:.1f}%")
        else:
            print(f"  Still infeasible up to M={M_GRID[-1]}")
        all_results["targets"][target_cm] = entry

    json.dump(all_results, open("accuracy_driven_design_p95_garage1.json", "w"), indent=2)
    print("\nSaved accuracy_driven_design_p95_garage1.json")
