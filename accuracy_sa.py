"""accuracy_sa.py - Table-7-style minimum anchor count (percentile form, P95 GDOP <= eps/a, coverage >= 95%) found with simulated annealing
instead of GA. Sweep budget 3,240 evaluations per (M, seed), seeds 2026 (as the GA sweep) plus 2027, 2028. Min M reported for seed 2026 and best-of-3.
Layouts at min M are validated by Monte Carlo (50 trials/point, per-point median, as in accuracy_driven_design.py)."""
import sys, json, time
import numpy as np
from garage_model import build_scenario, GARAGE_1, GARAGE_2
from visibility import build_visibility_matrix
from fitness import DEFAULT_PARAMS, is_connected, min_pairwise_distance
from gdop import gdop_all_points
from tdoa_solver import monte_carlo_trial

BUD = 40 * 81

def eval_p95(idx, cand, samp, vis, p):
    idx = np.asarray(idx)
    if len(idx) > p["M_max"] or not is_connected(cand[idx], p["R_sync"]) or min_pairwise_distance(cand[idx]) < p["d_min"]: return np.inf, None
    act = np.zeros_like(vis, dtype=bool); act[idx, :] = vis[idx, :]; n = act.sum(axis=0); cov = n >= p["min_los_anchors"]; cf = cov.mean()
    if cf == 0: return np.inf, None
    ci = np.where(cov)[0]; g = gdop_all_points(cand, samp[ci], act[:, ci]); g = g[np.isfinite(g)]
    f1 = g.mean() if len(g) else np.inf; f2 = np.percentile(g, 95) if len(g) else np.inf
    F = p["w1"] * f1 + p["w2"] * f2 + p["w3"] * len(idx)
    if f2 > p["GDOP_max"]: F += p["gdop_penalty_weight"] * (f2 - p["GDOP_max"])
    if cf < p["coverage_threshold"]: F += p["coverage_penalty_weight"] * (p["coverage_threshold"] - cf)
    return F, (cf, f2)

def sa(cand, samp, vis, M, seed, p, T0=5.0, T1=0.01):
    rng = np.random.default_rng(seed); N = len(cand); cur = rng.choice(N, M, replace=False); Fc, dc = eval_p95(cur, cand, samp, vis, p); best, bi, bd = Fc, cur.copy(), dc; big = 1e6
    for t in range(BUD - 1):
        T = T0 * (T1 / T0) ** (t / (BUD - 2)); nb = cur.copy(); nb[rng.integers(M)] = rng.choice(np.setdiff1d(np.arange(N), cur)); Fn, dn = eval_p95(nb, cand, samp, vis, p)
        a, b = (Fc if np.isfinite(Fc) else big), (Fn if np.isfinite(Fn) else big)
        if b <= a or rng.random() < np.exp(-(b - a) / T):
            cur, Fc, dc = nb, Fn, dn
            if Fc < best: best, bi, bd = Fc, cur.copy(), dc
    return bi, best, bd

def validate(layout, cand, samp, vis, cfg, target_cm):
    dz = cfg.tag_height - cfg.an_height; rng = np.random.default_rng(4242); lc, lv = cand[layout], vis[layout]; e = []
    for k in range(len(samp)):
        if lv[:, k].sum() < 3: continue
        er, _ = monte_carlo_trial(lc[lv[:, k]], samp[k], cfg.sigma_tdoa_floor, cfg.sigma_tdoa_coeff, 50, rng, dz); e.append(np.median(er) * 100)
    e = np.array(e); return dict(median_cm=float(np.median(e)), p95_cm=float(np.percentile(e, 95)), max_cm=float(e.max()), frac_within=float((e <= target_cm).mean()), n=len(e))

def run(g):
    cfg, a, targets = (GARAGE_1, 11.2, [10.0, 15.0, 20.0]) if g == "1" else (GARAGE_2, 7.3, [10.0, 15.0, 20.0])
    sc = build_scenario(cfg); cand, samp = sc["candidate_AN_points"], sc["vehicle_sample_points"]
    vis = build_visibility_matrix(cand, samp, sc["columns"], sc["parked_cars"], cfg.an_height, cfg.tag_height); out = {"garage": cfg.name, "a": a, "targets": {}}
    for tgt in targets:
        grid = list(range(6, 31, 2)) if g == "1" and tgt < 15 else None
        grid = (list(range(6, 31, 2)) + list(range(34, 61, 4))) if (g == "1" and tgt <= 10) else (list(range(6, 31, 2)) if g == "1" else [4, 5, 6, 7, 8, 9, 10, 12])
        p = {**DEFAULT_PARAMS, "GDOP_max": tgt / a}; res = {}; t0 = time.time()
        for M in grid:
            p["M_max"] = M; best = None
            runs = [sa(cand, samp, vis, M, s, p) for s in (2026, 2027, 2028)]
            ok = [(r[2] is not None and r[2][0] >= 0.95 and r[2][1] <= tgt / a) for r in runs]
            res[M] = dict(ok_seed2026=bool(ok[0]), ok_any=bool(any(ok)), cov=[None if r[2] is None else float(r[2][0]) for r in runs], p95gdop=[None if r[2] is None else float(r[2][1]) for r in runs])
            res[M]["_layout"] = (runs[0] if ok[0] else runs[int(np.argmax(ok))])[0].tolist() if any(ok) else None
            print(f"[G{g}] {tgt:.0f}cm M={M}: seed2026 {'OK' if ok[0] else '--'}, any {'OK' if any(ok) else '--'}, P95GDOP={[None if r[2] is None else round(r[2][1],2) for r in runs]} ({time.time()-t0:.0f}s)", flush=True)
            if any(ok) and ok[0]: break  # smallest M for seed 2026 found; continue only to find best-of-3 if smaller already covered earlier
        mS = next((M for M in grid if M in res and res[M]["ok_seed2026"]), None); mA = next((M for M in grid if M in res and res[M]["ok_any"]), None)
        ent = dict(gdop_max=tgt / a, min_M_seed2026=mS, min_M_best_of_3=mA, sweep={str(k): {kk: vv for kk, vv in v.items() if kk != "_layout"} for k, v in res.items()})
        if mA is not None:
            ent["validation"] = validate(np.array(res[mA]["_layout"]), cand, samp, vis, cfg, tgt)
        print(f"[G{g}] TARGET {tgt:.0f} cm: min M seed2026={mS}, best-of-3={mA}, validation={ent.get('validation')}", flush=True)
        out["targets"][str(tgt)] = ent; json.dump(out, open(f"accuracy_sa_g{g}.json", "w"), indent=2)

if __name__ == "__main__":
    for g in sys.argv[1:] or ["1", "2"]: run(g)
