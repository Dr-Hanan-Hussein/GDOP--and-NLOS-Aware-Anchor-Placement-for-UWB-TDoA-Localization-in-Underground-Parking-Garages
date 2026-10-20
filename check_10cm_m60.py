"""
check_10cm_m60.py

Confirmation check for the Garage 1 10 cm target (Table 9): validates layouts found with the P95 criterion at M = 56 and M = 60 by Monte Carlo
and prints the share of points within 10 cm (about 84% and 85%, below the 95% service level, so the 10 cm target is not reached in Garage 1). Run after validate_accuracy.py (needs accuracy_validation_both_garage1.json).
"""
import json
import ga, pso
from accuracy_driven_design_p95 import evaluate_percentile, confirm_layout
from accuracy_driven_design import calibrate_slope
from garage_model import build_scenario, GARAGE_1
from visibility import build_visibility_matrix

ga.evaluate = evaluate_percentile
pso.evaluate = evaluate_percentile

cfg = GARAGE_1
scenario = build_scenario(cfg)
candidates = scenario["candidate_AN_points"]
samples = scenario["vehicle_sample_points"]
vis = build_visibility_matrix(candidates, samples, scenario["columns"], scenario["parked_cars"],
                               cfg.an_height, cfg.tag_height)
a_slope, _ = calibrate_slope("accuracy_validation_both_garage1.json")
gdop_max_target = 10.0 / a_slope

for M in [56, 60]:
    c = confirm_layout(candidates, samples, vis, M, gdop_max_target, cfg, 10.0)
    print(f"M={M}: median={c['real_median_error_cm']:.1f}cm p95={c['real_p95_error_cm']:.1f}cm "
          f"max={c['real_max_error_cm']:.1f}cm frac<=10cm={c['frac_points_within_target']*100:.1f}%")
