import json
import time
import numpy as np
import ga, pso
from accuracy_driven_design_p95 import evaluate_percentile, sweep_M, confirm_layout
from accuracy_driven_design import calibrate_slope
from garage_model import build_scenario, GARAGE_2
from visibility import build_visibility_matrix

ga.evaluate = evaluate_percentile
pso.evaluate = evaluate_percentile

cfg = GARAGE_2
scenario = build_scenario(cfg)
candidates = scenario["candidate_AN_points"]
samples = scenario["vehicle_sample_points"]
vis = build_visibility_matrix(candidates, samples, scenario["columns"], scenario["parked_cars"],
                               cfg.an_height, cfg.tag_height)

a_slope, slopes = calibrate_slope("accuracy_validation_both_garage2.json")
print(f"Garage 2 calibrated slope a={a_slope:.2f} cm/GDOP (individual: {[round(s,2) for s in slopes]})")

TARGETS_CM = [10.0, 15.0, 20.0]
M_GRID = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

all_results = {"garage": cfg.name, "slope_a": a_slope, "percentile": 95, "targets": {}}
for target_cm in TARGETS_CM:
    gdop_max_target = target_cm / a_slope
    print(f"\n--- Target {target_cm:.0f}cm (P95) -> GDOP_max={gdop_max_target:.3f} ---")
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
        print(f"  Still infeasible up to M={M_GRID[-1]} ({len(candidates)} candidates total)")
    all_results["targets"][target_cm] = entry

json.dump(all_results, open("accuracy_driven_design_p95_garage2.json", "w"), indent=2)
print("\nSaved accuracy_driven_design_p95_garage2.json")
