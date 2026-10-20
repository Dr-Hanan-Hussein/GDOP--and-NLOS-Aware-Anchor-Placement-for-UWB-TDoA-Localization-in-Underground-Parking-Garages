"""
accuracy_driven_design_p95_10cm.py

Extends the P95 accuracy-driven sweep of Garage 1 (accuracy_driven_design_p95.py, targets 15 and 20 cm) to the 10 cm target over a wider
anchor-count grid (M = 16-52), and folds the result into accuracy_driven_design_p95_garage1.json.

Paper: Table 9, Garage 1, 10 cm row. The sweep finds M = 48 under the P95 GDOP criterion, but the Monte Carlo validation printed at the end puts only
about 80% of points within 10 cm (< 95%), so the target is counted as not achieved; check_10cm_m60.py repeats the validation at M = 56 and 60.
Run after accuracy_driven_design_p95.py.
"""
import json
import time
import numpy as np
import ga, pso
from fitness import DEFAULT_PARAMS
from garage_model import build_scenario, GARAGE_1
from visibility import build_visibility_matrix
from accuracy_driven_design import calibrate_slope
from accuracy_driven_design_p95 import evaluate_percentile, sweep_M, confirm_layout

ga.evaluate = evaluate_percentile
pso.evaluate = evaluate_percentile

cfg = GARAGE_1
scenario = build_scenario(cfg)
candidates = scenario["candidate_AN_points"]
samples = scenario["vehicle_sample_points"]
vis = build_visibility_matrix(candidates, samples, scenario["columns"], scenario["parked_cars"],
                               cfg.an_height, cfg.tag_height)

a_slope, _ = calibrate_slope("accuracy_validation_both_garage1.json")
target_cm = 10.0
gdop_max_target = target_cm / a_slope
M_GRID_10 = [16, 20, 24, 28, 32, 36, 40, 44, 48, 52]

print(f"10cm target (P95), GDOP_max={gdop_max_target:.3f}")
t0 = time.time()
sweep_results, min_M = sweep_M(candidates, samples, vis, gdop_max_target, M_GRID_10)
print(f"sweep took {time.time()-t0:.1f}s. Minimal feasible M = {min_M}")

entry = {"gdop_max_target": gdop_max_target, "sweep": sweep_results, "min_M": min_M}
if min_M is not None:
    t0 = time.time()
    confirm = confirm_layout(candidates, samples, vis, min_M, gdop_max_target, cfg, target_cm)
    entry["confirm"] = confirm
    print(f"confirmation ({confirm['winner_alg']} won) took {time.time()-t0:.1f}s")
    print(f"REAL validation: median={confirm['real_median_error_cm']:.1f}cm  "
          f"p95={confirm['real_p95_error_cm']:.1f}cm  max={confirm['real_max_error_cm']:.1f}cm  "
          f"fraction<=10cm={confirm['frac_points_within_target']*100:.1f}%")
else:
    print(f"Still infeasible up to M={M_GRID_10[-1]}")

# fold into the existing P95 results file
existing = json.load(open("accuracy_driven_design_p95_garage1.json"))
existing["targets"][target_cm] = entry
json.dump(existing, open("accuracy_driven_design_p95_garage1.json", "w"), indent=2)
print("Updated accuracy_driven_design_p95_garage1.json")
