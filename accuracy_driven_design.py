"""
accuracy_driven_design.py

Accuracy-driven design, strict (worst-case) form: given a target accuracy, find the
minimum number of anchors M for which a layout reliably achieves it. This inverts the
base formulation, which fixes M_max and treats accuracy as a soft penalty: here the
accuracy target is the constraint and M is minimised.
Paper: Section 3.3 (formulation) and Section 5.5 ("Strict form").
Needs accuracy_validation_both_garage1.json (validate_accuracy.py).  Output: accuracy_driven_design_garage1.json

APPROACH
--------
Running the real Monte-Carlo TDoA solver inside every fitness evaluation
of a 40-pop/150-gen GA search (6000+ evaluations) is computationally
infeasible. Instead we reuse the already-validated GDOP-accuracy proxy
(gdop.py + fitness.py's existing soft C2 GDOP_max mechanism), but
recalibrate GDOP_max itself from the target accuracy, using a slope
freshly fit against real Monte-Carlo TDoA data (see calibrate_slope
below and accuracy_validation_both_garage1.json): RMSE_cm ~= a * GDOP,
a ~ 11 cm per GDOP unit in Garage 1's n_active>=4 "target zone" (fit
independently for both the PSO and GA best layouts already validated,
which agreed closely: 10.87 and 11.22).

For each target accuracy:
  1. GDOP_max_target = target_cm / a  (worst-case GDOP that keeps the
     worst-case real error at or under the accuracy target).
  2. Coarse GA sweep (80 generations, for speed) over a grid of M values,
     with fitness_params={"GDOP_max": GDOP_max_target, "M_max": M},
     recording achieved coverage_frac and worst-case GDOP (f2) for each M.
  3. Minimum feasible M = smallest M in the grid meeting BOTH
     coverage_frac >= 0.95 (C1) and f2 <= GDOP_max_target (C2).
  4. At that M, run a full-quality confirmation (150 gens/iters, GA AND
     PSO, the standard protocol of Table 5) and validate the
     winning layout with real Monte-Carlo TDoA solving (as in
     validate_accuracy.py) to report the ACTUAL achieved accuracy
     distribution - not just the GDOP proxy - including what fraction
     of points truly meet the target.
"""

import json
import time
import numpy as np

from garage_model import build_scenario, GARAGE_1, GARAGE_2
from visibility import build_visibility_matrix
from gdop import gdop_2d
from fitness import evaluate, DEFAULT_PARAMS
from ga import genetic_algorithm
from pso import binary_pso
from tdoa_solver import monte_carlo_trial

TARGETS_CM = [10.0, 15.0, 20.0]
M_GRID = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
SWEEP_GENERATIONS = 80
FULL_GENERATIONS = 150
POP_SIZE = 40
N_TRIALS_VALIDATE = 50


def calibrate_slope(accuracy_json_path):
    """Fit RMSE_cm ~= a * GDOP (through origin, trimming top-5% outliers)
    on the n_active>=4 'target zone' points, using both already-validated
    best layouts, and return the (conservative = larger) of the two
    slopes so the resulting GDOP_max targets aren't optimistic."""
    d = json.load(open(accuracy_json_path))
    slopes = []
    for key in ["best_pso_idx", "best_ga_idx"]:
        pts = d[key]
        gdop = np.array([p["gdop"] for p in pts if p["n_active"] >= 4])
        rmse = np.array([p["rmse_cm"] for p in pts if p["n_active"] >= 4])
        cutoff = np.percentile(rmse, 95)
        mask = rmse <= cutoff
        a = np.sum(gdop[mask] * rmse[mask]) / np.sum(gdop[mask] ** 2)
        slopes.append(a)
    return max(slopes), slopes


def sweep_M(candidates, samples, vis, gdop_max_target, m_grid, seed=2026):
    """Coarse GA-only sweep across M values at a fixed GDOP_max target.
    Returns per-M diagnostics and the minimal feasible M (or None)."""
    results = {}
    for M in m_grid:
        fp = {**DEFAULT_PARAMS, "M_max": M, "GDOP_max": gdop_max_target}
        idx, F, hist, n_evals = genetic_algorithm(candidates, samples, vis, M=M,
                                                   pop_size=POP_SIZE, generations=SWEEP_GENERATIONS,
                                                   seed=seed, fitness_params=fp)
        _, diag = evaluate(idx, candidates, samples, vis, fp)
        results[M] = {"coverage_frac": diag.get("coverage_frac", 0.0),
                      "f2_max_gdop": diag.get("f2_max_gdop", np.inf),
                      "f1_mean_gdop": diag.get("f1_mean_gdop", np.inf),
                      "F": F, "status": diag["status"]}
        ok = diag.get("status") == "feasible" and diag.get("coverage_frac", 0) >= 0.95 \
            and diag.get("f2_max_gdop", np.inf) <= gdop_max_target
        results[M]["meets_target"] = bool(ok)
        print(f"    M={M:3d}  coverage={results[M]['coverage_frac']*100:5.1f}%  "
              f"worst-GDOP={results[M]['f2_max_gdop']:.2f}  (target<={gdop_max_target:.2f})  "
              f"{'OK' if ok else '--'}")

    feasible_Ms = [M for M in m_grid if results[M]["meets_target"]]
    min_M = min(feasible_Ms) if feasible_Ms else None
    return results, min_M


def confirm_layout(candidates, samples, vis, M, gdop_max_target, cfg, target_cm, seed=2026):
    """Full-quality GA + PSO run at the chosen M, then real Monte-Carlo
    TDoA validation of the winning (lower-F) layout."""
    fp = {**DEFAULT_PARAMS, "M_max": M, "GDOP_max": gdop_max_target}

    idx_ga, F_ga, _, _ = genetic_algorithm(candidates, samples, vis, M=M, pop_size=POP_SIZE,
                                            generations=FULL_GENERATIONS, seed=seed, fitness_params=fp)
    idx_pso, F_pso, _, _ = binary_pso(candidates, samples, vis, M=M, swarm_size=POP_SIZE,
                                       iters=FULL_GENERATIONS, seed=seed, fitness_params=fp)

    winner_idx, winner_alg, winner_F = (idx_ga, "GA", F_ga) if F_ga <= F_pso else (idx_pso, "PSO", F_pso)

    dz = cfg.tag_height - cfg.an_height
    rng = np.random.default_rng(4242)
    layout_coords = candidates[winner_idx]
    layout_vis = vis[winner_idx, :]
    real_errors_cm = []
    n_evaluated = 0
    for k in range(samples.shape[0]):
        active = layout_vis[:, k]
        if active.sum() < 3:
            continue
        active_coords = layout_coords[active]
        p_true = samples[k]
        g = gdop_2d(active_coords, p_true)
        if not np.isfinite(g):
            continue
        errors, _ = monte_carlo_trial(active_coords, p_true, cfg.sigma_tdoa_floor,
                                       cfg.sigma_tdoa_coeff, N_TRIALS_VALIDATE, rng, dz)
        real_errors_cm.append(float(np.median(errors) * 100))
        n_evaluated += 1

    real_errors_cm = np.array(real_errors_cm)
    return {
        "M": M, "winner_alg": winner_alg, "F_GA": float(F_ga), "F_PSO": float(F_pso),
        "layout_idx": winner_idx.tolist(),
        "n_points_evaluated": n_evaluated,
        "n_points_total": int(samples.shape[0]),
        "real_median_error_cm": float(np.median(real_errors_cm)),
        "real_p95_error_cm": float(np.percentile(real_errors_cm, 95)),
        "real_max_error_cm": float(real_errors_cm.max()),
        "frac_points_within_target": float((real_errors_cm <= target_cm).mean()),
    }


def run_for_garage(cfg, tag, out_prefix):
    print(f"\n{'='*70}\nAccuracy-driven design (strict form) for {cfg.name}\n{'='*70}")
    scenario = build_scenario(cfg)
    candidates = scenario["candidate_AN_points"]
    samples = scenario["vehicle_sample_points"]
    vis = build_visibility_matrix(candidates, samples, scenario["columns"], scenario["parked_cars"],
                                   cfg.an_height, cfg.tag_height)

    a_slope, both_slopes = calibrate_slope(f"accuracy_validation_both_{tag}.json")
    print(f"Calibrated slope (conservative, max of PSO/GA fits): a={a_slope:.2f} cm/GDOP "
          f"(individual fits: {[round(s,2) for s in both_slopes]})")

    all_results = {"garage": cfg.name, "slope_a": a_slope, "targets": {}}
    for target_cm in TARGETS_CM:
        gdop_max_target = target_cm / a_slope
        print(f"\n--- Target accuracy: {target_cm:.0f} cm  ->  GDOP_max = {gdop_max_target:.3f} ---")
        t0 = time.time()
        sweep_results, min_M = sweep_M(candidates, samples, vis, gdop_max_target, M_GRID)
        print(f"  sweep took {time.time()-t0:.1f}s. Minimal feasible M = {min_M}")

        target_entry = {"gdop_max_target": gdop_max_target, "sweep": sweep_results, "min_M": min_M}

        if min_M is not None:
            t0 = time.time()
            confirm = confirm_layout(candidates, samples, vis, min_M, gdop_max_target, cfg, target_cm)
            target_entry["confirm"] = confirm
            print(f"  confirmation ({confirm['winner_alg']} won, F={confirm['F_GA'] if confirm['winner_alg']=='GA' else confirm['F_PSO']:.2f}) "
                  f"took {time.time()-t0:.1f}s")
            print(f"  REAL Monte-Carlo validation: median={confirm['real_median_error_cm']:.1f}cm  "
                  f"p95={confirm['real_p95_error_cm']:.1f}cm  max={confirm['real_max_error_cm']:.1f}cm  "
                  f"fraction<={target_cm:.0f}cm={confirm['frac_points_within_target']*100:.1f}%  "
                  f"(n={confirm['n_points_evaluated']}/{confirm['n_points_total']} points)")
        else:
            print(f"  No M up to {M_GRID[-1]} met the target at this calibration - would need a wider grid.")

        all_results["targets"][target_cm] = target_entry

    out_path = f"{out_prefix}.json"
    json.dump(all_results, open(out_path, "w"), indent=2)
    print(f"\nSaved {out_path}")
    return all_results


if __name__ == "__main__":
    run_for_garage(GARAGE_1, "garage1", "accuracy_driven_design_garage1")
