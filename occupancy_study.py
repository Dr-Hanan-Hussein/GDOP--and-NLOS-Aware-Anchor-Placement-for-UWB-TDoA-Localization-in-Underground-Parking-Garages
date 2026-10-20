"""occupancy_study.py - does an anchor layout optimised for ONE parked-car realisation generalise to unseen realisations?

Paper: Section 5.3 and Table 7.
Per garage: (a) the 20 single-occupancy GA layouts and 20 PSO layouts from compare.py, evaluated on K held-out occupancies;
(b) robust GA trained on 5 occupancies (mean F), 5 seeds; (c) oracle GA re-optimised on each of 10 held-out occupancies.
Metric per (layout, occupancy): finite F, coverage_frac (>=4 LOS anchors), max GDOP, and 'feasible' = coverage>=0.95 and max GDOP<=3 (no soft penalty).
Seeds: held-out occupancies 2001-2030, training occupancies 3001-3005.
Needs comparison_curves.json / comparison_curves_g2.json (compare.py).  Output: occupancy_study_g1.json, occupancy_study_g2.json
"""
import sys, json, dataclasses, time
import numpy as np
import ga as ga_mod
from garage_model import build_scenario, build_parked_cars, GARAGE_1, GARAGE_2
from visibility import build_visibility_matrix
from fitness import evaluate
from ga import genetic_algorithm

K_TEST = 30; TEST_SEEDS = list(range(2001, 2001 + K_TEST)); TRAIN_SEEDS = list(range(3001, 3006)); N_ORACLE = 10

def vis_for(cfg, sc, seed):
    c2 = dataclasses.replace(cfg, random_seed=seed)
    cars = build_parked_cars(c2, sc["columns"])
    return build_visibility_matrix(sc["candidate_AN_points"], sc["vehicle_sample_points"], sc["columns"], cars, cfg.an_height, cfg.tag_height)

def metrics(idx, cand, samp, vis, M):
    F, d = evaluate(np.array(idx), cand, samp, vis, {"M_max": M})
    ok = d["status"] == "feasible"
    return dict(F=float(F) if np.isfinite(F) else None, cov=float(d.get("coverage_frac", 0.0)) if ok else 0.0,
                maxgdop=float(d["f2_max_gdop"]) if ok else None,
                feasible=bool(ok and d["coverage_frac"] >= 0.95 and d["f2_max_gdop"] <= 3.0))

def summarize(rows):
    F = np.array([r["F"] if r["F"] is not None else np.inf for r in rows]); cov = np.array([r["cov"] for r in rows])
    fe = np.array([r["feasible"] for r in rows])
    return dict(n=len(rows), feasible_frac=float(fe.mean()), median_F=float(np.median(F)), p90_F=float(np.percentile(F, 90)),
                median_cov=float(np.median(cov)), min_cov=float(cov.min()))

def run(g):
    cfg, M, tag = (GARAGE_1, 16, "g1") if g == "1" else (GARAGE_2, 8, "g2")
    suffix = "" if g == "1" else "_g2"
    sc = build_scenario(cfg); cand, samp = sc["candidate_AN_points"], sc["vehicle_sample_points"]
    t0 = time.time()
    vis_orig = build_visibility_matrix(cand, samp, sc["columns"], sc["parked_cars"], cfg.an_height, cfg.tag_height)
    vis_test = [vis_for(cfg, sc, s) for s in TEST_SEEDS]
    vis_train = [vis_for(cfg, sc, s) for s in TRAIN_SEEDS]
    print(f"[{tag}] visibility built {time.time()-t0:.0f}s", flush=True)
    curves = json.load(open(f"comparison_curves{suffix}.json"))
    out = {"garage": cfg.name, "M": M, "test_seeds": TEST_SEEDS}
    for name in ("PSO", "GA"):
        layouts = curves[f"Idx_{name}"]
        rows_in = [metrics(l, cand, samp, vis_orig, M) for l in layouts]
        rows_out = [[metrics(l, cand, samp, v, M) for v in vis_test] for l in layouts]
        flat = [r for rr in rows_out for r in rr]
        out[f"single_{name}"] = dict(in_sample=summarize(rows_in), held_out=summarize(flat),
                                     per_layout_feasible_frac=[float(np.mean([r["feasible"] for r in rr])) for rr in rows_out])
        print(f"[{tag}] single-occupancy {name}: in-sample {out[f'single_{name}']['in_sample']}\n      held-out {out[f'single_{name}']['held_out']}", flush=True)
    # robust GA
    orig_eval = ga_mod.evaluate
    ga_mod.evaluate = lambda idx, c, s, v, p=None: (float(np.mean([evaluate(idx, c, s, vv, p)[0] for vv in vis_train])), {})
    robust = []
    for s in range(5):
        idx, F, _, _ = genetic_algorithm(cand, samp, vis_orig, M=M, pop_size=40, generations=150, seed=1000 + s, fitness_params={"M_max": M})
        robust.append(idx.tolist()); print(f"[{tag}] robust GA seed {s}: mean-train F={F:.3f} ({time.time()-t0:.0f}s)", flush=True)
    ga_mod.evaluate = orig_eval
    rows_out = [[metrics(l, cand, samp, v, M) for v in vis_test] for l in robust]
    out["robust_GA"] = dict(held_out=summarize([r for rr in rows_out for r in rr]),
                            per_layout_feasible_frac=[float(np.mean([r["feasible"] for r in rr])) for rr in rows_out], layouts=robust)
    print(f"[{tag}] robust GA held-out {out['robust_GA']['held_out']}", flush=True)
    # oracle
    ora = []
    for k in range(N_ORACLE):
        idx, F, _, _ = genetic_algorithm(cand, samp, vis_test[k], M=M, pop_size=40, generations=150, seed=1000 + k, fitness_params={"M_max": M})
        ora.append(metrics(idx, cand, samp, vis_test[k], M))
    out["oracle_GA"] = summarize(ora)
    # same-occupancy comparison for the first N_ORACLE test occupancies: single-occ GA layouts
    out["single_GA_on_oracle_occupancies"] = summarize([metrics(l, cand, samp, vis_test[k], M) for l in curves["Idx_GA"] for k in range(N_ORACLE)])
    print(f"[{tag}] oracle {out['oracle_GA']}\n      single GA same occupancies {out['single_GA_on_oracle_occupancies']}", flush=True)
    json.dump(out, open(f"occupancy_study_{tag}.json", "w"), indent=2)

if __name__ == "__main__":
    for g in sys.argv[1:] or ["1", "2"]: run(g)
