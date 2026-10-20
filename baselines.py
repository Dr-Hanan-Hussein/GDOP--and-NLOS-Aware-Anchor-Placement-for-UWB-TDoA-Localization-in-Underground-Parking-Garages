"""baselines.py - random search, greedy construction and simulated annealing under the same 6,040-evaluation budget (greedy uses fewer).

Paper: Table 6 (metaheuristics and baselines, 20 seeds).
Usage:  python baselines.py 1 2                       (all four methods, both garages)
        python baselines.py 1 2 --methods greedyF     (selected methods only; merges into the existing output)
Needs comparison_report.json / comparison_report_g2.json from compare.py (PSO and GA reference values).
Output: baselines_g1.json, baselines_g2.json
"""
import sys, json, time
import numpy as np
from scipy.stats import mannwhitneyu
from garage_model import build_scenario, GARAGE_1, GARAGE_2
from visibility import build_visibility_matrix
from fitness import evaluate

BUDGET = 40 * 151  # 6040, as PSO/GA

def setup(g):
    cfg, M, suf = (GARAGE_1, 16, "") if g == "1" else (GARAGE_2, 8, "_g2")
    sc = build_scenario(cfg); cand, samp = sc["candidate_AN_points"], sc["vehicle_sample_points"]
    vis = build_visibility_matrix(cand, samp, sc["columns"], sc["parked_cars"], cfg.an_height, cfg.tag_height)
    return cfg, M, suf, cand, samp, vis

def Fval(idx, cand, samp, vis, M):
    return evaluate(np.array(idx), cand, samp, vis, {"M_max": M})

def random_search(cand, samp, vis, M, seed):
    rng = np.random.default_rng(seed); best = np.inf; bi = None
    for _ in range(BUDGET):
        idx = rng.choice(len(cand), M, replace=False); F, _ = Fval(idx, cand, samp, vis, M)
        if F < best or bi is None: best, bi = F, idx
    return bi, best

def sim_anneal(cand, samp, vis, M, seed, T0=5.0, T1=0.01):
    rng = np.random.default_rng(seed); N = len(cand)
    cur = rng.choice(N, M, replace=False); Fc, _ = Fval(cur, cand, samp, vis, M); best, bi = Fc, cur.copy()
    big = 1e6
    for t in range(BUDGET - 1):
        T = T0 * (T1 / T0) ** (t / (BUDGET - 2))
        nb = cur.copy(); out = rng.integers(M); pool = np.setdiff1d(np.arange(N), cur); nb[out] = rng.choice(pool)
        Fn, _ = Fval(nb, cand, samp, vis, M)
        a, b = (Fc if np.isfinite(Fc) else big), (Fn if np.isfinite(Fn) else big)
        if b <= a or rng.random() < np.exp(-(b - a) / T):
            cur, Fc = nb, Fn
            if Fc < best: best, bi = Fc, cur.copy()
    return bi, best

def greedy(cand, samp, vis, M, seed, R_sync=15.0, d_min=4.0):
    """Coverage-driven greedy: random first anchor, then repeatedly add the candidate (respecting d_min and sync range)
    that most increases mean min(n_LOS,4)/4; the final layout is scored with the full objective."""
    rng = np.random.default_rng(seed); N = len(cand); sel = [int(rng.integers(N))]
    while len(sel) < M:
        best, bc = -1, None
        for c in range(N):
            if c in sel: continue
            dd = np.linalg.norm(cand[sel] - cand[c], axis=1)
            if dd.min() < d_min or dd.min() > R_sync: continue
            cs = np.minimum(vis[sel + [c], :].sum(axis=0), 4).mean() / 4
            if cs > best + 1e-12: best, bc = cs, c
        if bc is None: return np.array(sel), np.inf
        sel.append(bc)
    F, _ = Fval(sel, cand, samp, vis, M); return np.array(sel), F

def greedy_F(cand, samp, vis, M, seed, R_sync=15.0, d_min=4.0):
    """Objective-aware greedy: random first anchor, then add the candidate that minimises the full objective F
    (coverage-based surrogate while F is infinite or coverage is still below target)."""
    rng = np.random.default_rng(seed); N = len(cand); sel = [int(rng.integers(N))]
    while len(sel) < M:
        best, bc = np.inf, None
        for c in range(N):
            if c in sel: continue
            dd = np.linalg.norm(cand[sel] - cand[c], axis=1)
            if dd.min() < d_min or dd.min() > R_sync: continue
            trial = sel + [c]; F, d = Fval(trial, cand, samp, vis, M)
            cs = np.minimum(vis[trial, :].sum(axis=0), 4).mean() / 4
            score = F if (np.isfinite(F) and d.get("coverage_frac", 0) >= 0.95) else 1e4 + (1 - cs) * 1e3
            if score < best: best, bc = score, c
        if bc is None: return np.array(sel), np.inf
        sel.append(bc)
    F, _ = Fval(sel, cand, samp, vis, M); return np.array(sel), F

def run(g, S=20, methods=None):
    cfg, M, suf, cand, samp, vis = setup(g)
    prev = {}
    try: prev = json.load(open(f"baselines_g{g}.json"))
    except Exception: pass
    rep = json.load(open(f"comparison_report{suf}.json")); out = {**prev, "garage": cfg.name, "M": M}
    ref = {"PSO": np.array(rep["F_PSO_all"]), "GA": np.array(rep["F_GA_all"])}
    for k in ref: out[k] = {"median": float(np.median(ref[k])), "mean": float(ref[k].mean())}
    for name, fn in [m for m in (("greedy", greedy), ("greedyF", greedy_F), ("random", random_search), ("SA", sim_anneal)) if methods is None or m[0] in methods]:
        t0 = time.time(); Fs = []
        for s in range(S): Fs.append(fn(cand, samp, vis, M, 1000 + s)[1])
        Fs = np.array(Fs); fin = np.isfinite(Fs)
        d = {"finite_runs": int(fin.sum()), "median": float(np.median(Fs)), "mean_finite": float(Fs[fin].mean()) if fin.any() else None,
             "best": float(Fs.min()), "F_all": [float(x) if np.isfinite(x) else None for x in Fs], "time_s": time.time() - t0}
        for k in ("GA", "PSO"):
            U, p = mannwhitneyu(Fs, ref[k], alternative="two-sided"); d[f"vs_{k}"] = {"p": float(p), "r": float(1 - 2 * U / (S * S))}
        out[name] = d
        print(f"[G{g}] {name}: median={d['median']:.2f} mean_finite={d['mean_finite']} best={d['best']:.2f} finite={d['finite_runs']}/{S} vsGA p={d['vs_GA']['p']:.4f} vsPSO p={d['vs_PSO']['p']:.4f} ({d['time_s']:.0f}s)", flush=True)
        json.dump(out, open(f"baselines_g{g}.json", "w"), indent=2)
    print(f"[G{g}] reference PSO median={out['PSO']['median']:.2f}, GA median={out['GA']['median']:.2f}", flush=True)

if __name__ == "__main__":
    methods = None
    if "--methods" in sys.argv:
        i = sys.argv.index("--methods"); methods = sys.argv[i + 1].split(","); del sys.argv[i:i + 2]
    for g in sys.argv[1:] or ["1", "2"]: run(g, methods=methods)
