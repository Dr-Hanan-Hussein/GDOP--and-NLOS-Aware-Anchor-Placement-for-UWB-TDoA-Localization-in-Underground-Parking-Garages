"""second_root.py - for the worst Monte-Carlo points, determine whether failures are (a) optimiser divergence/local minima or (b) a genuine second ML root.
For each failed trial (err>1 m): compare cost at estimate vs cost at TRUE position (same noisy data) and cost at best of multi-start global search."""
import json, numpy as np
from scipy.optimize import least_squares
from garage_model import build_scenario, GARAGE_1
from visibility import build_visibility_matrix
from tdoa_solver import ranges, simulate_measurements, solve_position

cfg = GARAGE_1; dz = cfg.tag_height - cfg.an_height
sc = build_scenario(cfg); cand, samp = sc["candidate_AN_points"], sc["vehicle_sample_points"]
vis = build_visibility_matrix(cand, samp, sc["columns"], sc["parked_cars"], cfg.an_height, cfg.tag_height)
L = json.load(open("best_layouts.json")); V = json.load(open("accuracy_validation_both_garage1.json"))
def cost(a, D, p):
    r = ranges(a, p, dz); return float(np.sum((D - (r[1:] - r[0])) ** 2))
out = {}
for key in ("best_pso_idx", "best_ga_idx"):
    res = V[key]; mc = np.array([x["rmse_cm"] for x in res]); worst = np.argsort(-mc)[:8]
    lc = cand[np.array(L[key])]; lv = vis[np.array(L[key]), :]
    stats = []
    for i in worst:
        p = np.array(res[i]["point"]); k = int(np.argmin(np.linalg.norm(samp - p, axis=1)))
        a = lc[lv[:, k]]; rng = np.random.default_rng(5)
        nfail = 0; lower_than_truth = 0; ms_fix = 0
        for t in range(200):
            D = simulate_measurements(a, p, cfg.sigma_tdoa_floor, cfg.sigma_tdoa_coeff, rng, dz)
            ph, _ = solve_position(a, D, dz)
            if np.linalg.norm(ph - p) > 1.0:
                nfail += 1
                if cost(a, D, ph) < cost(a, D, p): lower_than_truth += 1   # estimate is a better ML fit than truth => genuine ambiguity
                # multistart from 25 grid points
                best = None
                lo, hi = a.min(0) - 20, a.max(0) + 20
                for s in range(25):
                    x0 = rng.uniform(lo, hi)
                    q = least_squares(lambda x: D - (ranges(a, x, dz)[1:] - ranges(a, x, dz)[0]), x0, method="trf", bounds=(lo, hi))
                    if best is None or q.cost < best.cost: best = q
                if np.linalg.norm(best.x - p) < 1.0: ms_fix += 1
        stats.append(dict(point=p.tolist(), n_active=int(a.shape[0]), gdop=res[i]["gdop"], mc_rmse_cm=float(mc[i]),
                          fail_frac=nfail / 200, est_better_fit_than_truth=lower_than_truth, multistart_recovers=ms_fix))
        print(key, p.round(1), f"n={a.shape[0]} fail={nfail}/200 est_better_than_truth={lower_than_truth} multistart_recovers={ms_fix}", flush=True)
    out[key] = stats
json.dump(out, open("second_root_results.json", "w"), indent=2)
