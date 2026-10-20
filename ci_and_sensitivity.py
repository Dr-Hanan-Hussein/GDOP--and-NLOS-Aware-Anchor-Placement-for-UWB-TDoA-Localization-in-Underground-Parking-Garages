"""ci_and_sensitivity.py
(1) Bootstrap 95% CIs (points resampled) for Spearman rho(GDOP,RMSE) and the GDOP->RMSE slope a (calibrate_slope definition: n_active>=4, top-5% trimmed, through origin; per layout).
(2) Sensitivity of the validated GA layouts (no hardware): noise scale, undetected-NLOS contamination, tag height, car height, occupancy."""
import json, sys, dataclasses, time
import numpy as np
from scipy.stats import spearmanr
from garage_model import build_scenario, build_parked_cars, GARAGE_1, GARAGE_2
from visibility import build_visibility_matrix
from gdop import gdop_2d
from tdoa_solver import ranges, solve_position, simulate_measurements
from fitness import evaluate

rng0 = np.random.default_rng(2027)
def slope(g, r, n):
    m = n >= 4; g, r = g[m], r[m]; mask = r <= np.percentile(r, 95)
    return float(np.sum(g[mask] * r[mask]) / np.sum(g[mask] ** 2))

def bootstrap(garage):
    V = json.load(open(f"accuracy_validation_both_garage{garage}.json")); out = {}
    for key in ("best_pso_idx", "best_ga_idx"):
        pts = V[key]; g = np.array([p["gdop"] for p in pts]); r = np.array([p["rmse_cm"] for p in pts]); n = np.array([p["n_active"] for p in pts])
        sp, sl = [], []
        for _ in range(2000):
            i = rng0.integers(0, len(g), len(g)); sp.append(spearmanr(g[i], r[i])[0]); sl.append(slope(g[i], r[i], n[i]))
        out[key] = dict(spearman=float(spearmanr(g, r)[0]), spearman_ci=np.percentile(sp, [2.5, 97.5]).tolist(),
                        slope=slope(g, r, n), slope_ci=np.nanpercentile(sl, [2.5, 97.5]).tolist(), n_points=len(g))
        print(f"G{garage} {key}: Spearman {out[key]['spearman']:.3f} CI {np.round(out[key]['spearman_ci'],3)}; slope {out[key]['slope']:.2f} cm/GDOP CI {np.round(out[key]['slope_ci'],2)}", flush=True)
    return out

def mc_rmse(coords, p, cfg, floor_s, coeff_s, nlos_p, bias_rng, nt, rng, dz):
    errs = np.empty(nt)
    for t in range(nt):
        r = ranges(coords, p, dz); sig = (cfg.sigma_tdoa_floor * floor_s + cfg.sigma_tdoa_coeff * coeff_s * r)
        rn = r + rng.normal(0, sig)
        if nlos_p > 0:
            hit = rng.random(len(r)) < nlos_p; rn = rn + hit * rng.uniform(*bias_rng, size=len(r))
        D = rn[1:] - rn[0]; ph, _ = solve_position(coords, D, dz); errs[t] = np.linalg.norm(ph - p)
    return errs

def sens(garage):
    cfg, M, suf = (GARAGE_1, 16, "") if garage == "1" else (GARAGE_2, 8, "_g2")
    sc = build_scenario(cfg); cand, samp = sc["candidate_AN_points"], sc["vehicle_sample_points"]
    vis = build_visibility_matrix(cand, samp, sc["columns"], sc["parked_cars"], cfg.an_height, cfg.tag_height)
    L = json.load(open("best_layouts.json" if garage == "1" else "best_layouts_g2.json")); idx = np.array(L["best_ga_idx"]); lc, lv = cand[idx], vis[idx]
    dz = cfg.tag_height - cfg.an_height; res = {"garage": cfg.name, "noise": {}, "nlos": {}, "geometry": {}}
    pts = [k for k in range(0, len(samp), 2) if lv[:, k].sum() >= 4 and np.isfinite(gdop_2d(lc[lv[:, k]], samp[k]))]
    print(f"[G{garage}] sensitivity points: {len(pts)}", flush=True)
    def summarize(floor_s, coeff_s, nlos_p, bias, nt=40, seed=11):
        rng = np.random.default_rng(seed); e = []
        for k in pts:
            er = mc_rmse(lc[lv[:, k]], samp[k], cfg, floor_s, coeff_s, nlos_p, bias, nt, rng, dz); e.append(np.sqrt(np.mean(er ** 2)) * 100)
        e = np.array(e); return dict(median_cm=float(np.median(e)), p95_cm=float(np.percentile(e, 95)), frac_gt100=float((e > 100).mean()))
    for s in (0.5, 1.0, 1.5, 2.0):
        res["noise"][str(s)] = summarize(s, s, 0.0, (0, 0)); print(f"[G{garage}] noise x{s}: {res['noise'][str(s)]}", flush=True)
    for p in (0.0, 0.05, 0.1, 0.2):
        res["nlos"][str(p)] = summarize(1.0, 1.0, p, (0.2, 1.0)); print(f"[G{garage}] undetected NLOS prob {p} (bias U[0.2,1.0] m): {res['nlos'][str(p)]}", flush=True)
    # geometry: re-evaluate the FIXED layout under altered car height / tag height / occupancy
    for name, kw in (("base", {}), ("tag_0.3", {"tag_height": 0.3}), ("tag_1.0", {"tag_height": 1.0}), ("tag_1.5", {"tag_height": 1.5}),
                     ("car_1.4", {"car_height": 1.4}), ("car_1.8", {"car_height": 1.8}), ("car_2.0", {"car_height": 2.0}),
                     ("occ_0.6", {"occupancy_rate": 0.6}), ("occ_0.95", {"occupancy_rate": 0.95})):
        c2 = dataclasses.replace(cfg, **kw); cars = build_parked_cars(c2, sc["columns"])
        v2 = build_visibility_matrix(cand, samp, sc["columns"], cars, c2.an_height, c2.tag_height)
        F, d = evaluate(idx, cand, samp, v2, {"M_max": M}); ok = d["status"] == "feasible"
        res["geometry"][name] = dict(F=float(F) if np.isfinite(F) else None, cov=float(d.get("coverage_frac", 0)) if ok else 0.0,
                                     maxgdop=float(d["f2_max_gdop"]) if ok else None)
        print(f"[G{garage}] geometry {name}: {res['geometry'][name]}", flush=True)
    return res

if __name__ == "__main__":
    out = {}
    for g in sys.argv[1:] or ["1", "2"]:
        out[f"g{g}"] = {"bootstrap": bootstrap(g), "sensitivity": sens(g)}
        json.dump(out, open("ci_sensitivity_results.json", "w"), indent=2)
