"""sa_study.py - simulated annealing layouts (single-occupancy and 5-occupancy robust), held-out occupancy evaluation, and Monte-Carlo accuracy validation
of the best SA layout. Same 6,040-evaluation budget per SA run as PSO/GA (robust SA: 6,040 iterations, each scored as the mean F over 5 occupancies).

Paper: Sections 5.3-5.5 (SA layouts, Tables 6-7, SA rows of Table 10).
Needs occupancy_study.py and baselines.py (imports).  Output: sa_study_g1.json, sa_study_g2.json
Note: the 'failure point' flag (n_failure_points, frac_failure) uses the Garage 1 slope a = 11.2 cm/GDOP for both garages; the paper does not use it for Garage 2.
"""
import sys, json, time
import numpy as np
from scipy.stats import spearmanr
from occupancy_study import vis_for, metrics, summarize, TEST_SEEDS, TRAIN_SEEDS
from baselines import BUDGET
from garage_model import build_scenario, GARAGE_1, GARAGE_2
from visibility import build_visibility_matrix
from fitness import evaluate
from gdop import gdop_2d
from tdoa_solver import monte_carlo_trial

def sa(cand, M, seed, Ffun, T0=5.0, T1=0.01):
    rng = np.random.default_rng(seed); N = len(cand); cur = rng.choice(N, M, replace=False); Fc = Ffun(cur); best, bi = Fc, cur.copy(); big = 1e6
    for t in range(BUDGET - 1):
        T = T0 * (T1 / T0) ** (t / (BUDGET - 2)); nb = cur.copy(); nb[rng.integers(M)] = rng.choice(np.setdiff1d(np.arange(N), cur)); Fn = Ffun(nb)
        a, b = (Fc if np.isfinite(Fc) else big), (Fn if np.isfinite(Fn) else big)
        if b <= a or rng.random() < np.exp(-(b - a) / T):
            cur, Fc = nb, Fn
            if Fc < best: best, bi = Fc, cur.copy()
    return bi, best

def run(g):
    cfg, M, tag = (GARAGE_1, 16, "g1") if g == "1" else (GARAGE_2, 8, "g2")
    sc = build_scenario(cfg); cand, samp = sc["candidate_AN_points"], sc["vehicle_sample_points"]; t0 = time.time()
    vis0 = build_visibility_matrix(cand, samp, sc["columns"], sc["parked_cars"], cfg.an_height, cfg.tag_height)
    vtest = [vis_for(cfg, sc, s) for s in TEST_SEEDS]; vtrain = [vis_for(cfg, sc, s) for s in TRAIN_SEEDS]
    F1 = lambda v: (lambda idx: evaluate(idx, cand, samp, v, {"M_max": M})[0])
    Fmean = lambda idx: float(np.mean([evaluate(idx, cand, samp, v, {"M_max": M})[0] for v in vtrain]))
    out = {"garage": cfg.name, "M": M}
    single = [sa(cand, M, 1000 + s, F1(vis0)) for s in range(20)]
    out["SA_single_layouts"] = [l.tolist() for l, _ in single]; out["SA_single_F"] = [float(f) for _, f in single]
    rows_in = [metrics(l, cand, samp, vis0, M) for l, _ in single]
    rows_out = [[metrics(l, cand, samp, v, M) for v in vtest] for l, _ in single]
    out["SA_single"] = dict(in_sample=summarize(rows_in), held_out=summarize([r for rr in rows_out for r in rr]))
    print(f"[{tag}] SA single in-sample {out['SA_single']['in_sample']}\n    held-out {out['SA_single']['held_out']} ({time.time()-t0:.0f}s)", flush=True)
    rob = [sa(cand, M, 1000 + s, Fmean) for s in range(10)]
    out["SA_robust_layouts"] = [l.tolist() for l, _ in rob]
    rows_out = [[metrics(l, cand, samp, v, M) for v in vtest] for l, _ in rob]
    out["SA_robust"] = dict(held_out=summarize([r for rr in rows_out for r in rr]), per_layout_feasible_frac=[float(np.mean([r["feasible"] for r in rr])) for rr in rows_out])
    print(f"[{tag}] SA robust held-out {out['SA_robust']['held_out']} ({time.time()-t0:.0f}s)", flush=True)
    # Monte-Carlo accuracy validation of best single SA layout and best robust SA layout (on the original occupancy)
    dz = cfg.tag_height - cfg.an_height; rng = np.random.default_rng(123)
    for name, layout in (("SA_single_best", single[int(np.argmin([f for _, f in single]))][0]), ("SA_robust_best", rob[int(np.argmin([evaluate(l, cand, samp, vis0, {'M_max': M})[0] for l, _ in rob]))][0])):
        lc, lv = cand[layout], vis0[layout]; G, R, N4 = [], [], []
        for k in range(len(samp)):
            if lv[:, k].sum() < 3: continue
            gd = gdop_2d(lc[lv[:, k]], samp[k])
            if not np.isfinite(gd): continue
            e, _ = monte_carlo_trial(lc[lv[:, k]], samp[k], cfg.sigma_tdoa_floor, cfg.sigma_tdoa_coeff, 50, rng, dz)
            G.append(gd); R.append(np.sqrt(np.mean(e ** 2)) * 100); N4.append(int(lv[:, k].sum()))
        G, R = np.array(G), np.array(R); fail = (G <= 3.0) & (R > 100) & (R > 2 * 11.2 * G)
        out[name] = dict(layout=layout.tolist(), n_points=len(G), median_cm=float(np.median(R)), p95_cm=float(np.percentile(R, 95)), max_cm=float(R.max()),
                         pearson=float(np.corrcoef(G, R)[0, 1]), spearman=float(spearmanr(G, R)[0]), n_failure_points=int(fail.sum()),
                         frac_failure=float(fail.mean()), F=float(evaluate(layout, cand, samp, vis0, {"M_max": M})[0]))
        print(f"[{tag}] {name}: {out[name]['n_points']} pts, median={out[name]['median_cm']:.1f} p95={out[name]['p95_cm']:.1f} max={out[name]['max_cm']:.0f} cm, "
              f"Spearman={out[name]['spearman']:.2f}, failure pts={out[name]['n_failure_points']}, F={out[name]['F']:.2f}", flush=True)
    json.dump(out, open(f"sa_study_{tag}.json", "w"), indent=2)

if __name__ == "__main__":
    for g in sys.argv[1:] or ["1", "2"]: run(g)
