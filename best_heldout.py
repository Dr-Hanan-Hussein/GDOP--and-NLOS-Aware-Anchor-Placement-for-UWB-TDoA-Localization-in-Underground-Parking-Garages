"""best_heldout.py - held-out occupancy performance of the single best (lowest-F) layout of each algorithm, plus SA best and SA-robust best.

Paper: Table 7 (generalisation to 30 unseen parked-car arrangements).
Needs: best_layouts.json, best_layouts_g2.json (extract_best_layouts.py) and sa_study_g1.json, sa_study_g2.json (sa_study.py).
Output: best_heldout.json
"""
import json, sys, numpy as np
from occupancy_study import vis_for, metrics, TEST_SEEDS
from garage_model import build_scenario, GARAGE_1, GARAGE_2
from fitness import evaluate
out = {}
for g in ("1", "2"):
    cfg, M, tag, suf = (GARAGE_1, 16, "g1", "") if g == "1" else (GARAGE_2, 8, "g2", "_g2")
    sc = build_scenario(cfg); cand, samp = sc["candidate_AN_points"], sc["vehicle_sample_points"]
    L = json.load(open("best_layouts.json" if g == "1" else "best_layouts_g2.json")); S = json.load(open(f"sa_study_{tag}.json"))
    vt = [vis_for(cfg, sc, s) for s in TEST_SEEDS]; res = {}
    for name, idx in (("PSO_best", L["best_pso_idx"]), ("GA_best", L["best_ga_idx"]), ("SA_single_best", S["SA_single_best"]["layout"]), ("SA_robust_best", S["SA_robust_best"]["layout"])):
        rows = [metrics(idx, cand, samp, v, M) for v in vt]
        res[name] = dict(feasible_frac=float(np.mean([r["feasible"] for r in rows])), median_cov=float(np.median([r["cov"] for r in rows])),
                         min_cov=float(min(r["cov"] for r in rows)), median_maxgdop=float(np.median([r["maxgdop"] for r in rows if r["maxgdop"] is not None])),
                         p90_maxgdop=float(np.percentile([r["maxgdop"] for r in rows if r["maxgdop"] is not None], 90)))
        print(f"G{g} {name}: {res[name]}", flush=True)
    out[f"g{g}"] = res
json.dump(out, open("best_heldout.json", "w"), indent=2)
