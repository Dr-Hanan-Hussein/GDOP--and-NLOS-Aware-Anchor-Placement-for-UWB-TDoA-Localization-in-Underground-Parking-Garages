import json
import numpy as np
from garage_model import build_scenario, GARAGE_1
from visibility import build_visibility_matrix
from accuracy_driven_design import sweep_M, confirm_layout, calibrate_slope

cfg = GARAGE_1
scenario = build_scenario(cfg)
candidates = scenario["candidate_AN_points"]
samples = scenario["vehicle_sample_points"]
vis = build_visibility_matrix(candidates, samples, scenario["columns"], scenario["parked_cars"],
                               cfg.an_height, cfg.tag_height)

a_slope, _ = calibrate_slope("accuracy_validation_both_garage1.json")
EXT_GRID = [32, 36, 40, 44, 48, 52, 56, 60]

results = json.load(open("accuracy_driven_design_garage1.json"))

for target_cm in [10.0, 15.0]:
    gdop_max_target = target_cm / a_slope
    print(f"\n--- EXTENDED sweep, target {target_cm:.0f}cm -> GDOP_max={gdop_max_target:.3f} ---")
    sweep_results, min_M = sweep_M(candidates, samples, vis, gdop_max_target, EXT_GRID)
    entry = results["targets"][str(target_cm)] if str(target_cm) in results["targets"] else results["targets"][target_cm]
    entry["extended_sweep"] = sweep_results
    entry["extended_min_M"] = min_M
    if min_M is not None:
        confirm = confirm_layout(candidates, samples, vis, min_M, gdop_max_target, cfg, target_cm)
        entry["confirm"] = confirm
        print(f"  CONFIRMED M={min_M}: real median={confirm['real_median_error_cm']:.1f}cm "
              f"p95={confirm['real_p95_error_cm']:.1f}cm max={confirm['real_max_error_cm']:.1f}cm "
              f"frac<={target_cm:.0f}cm={confirm['frac_points_within_target']*100:.1f}%")
    else:
        print(f"  Still infeasible up to M={EXT_GRID[-1]} ({len(candidates)} candidates total available)")

json.dump(results, open("accuracy_driven_design_garage1.json", "w"), indent=2)
print("\nUpdated accuracy_driven_design_garage1.json")
