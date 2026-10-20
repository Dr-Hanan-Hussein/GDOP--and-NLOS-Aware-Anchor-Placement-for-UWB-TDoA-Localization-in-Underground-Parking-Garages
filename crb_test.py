"""crb_test.py - true TDoA Cramer-Rao bound vs GDOP vs Monte Carlo RMSE (both garages, PSO/GA best layouts)."""
import json, sys
import numpy as np
from garage_model import build_scenario, GARAGE_1, GARAGE_2
from visibility import build_visibility_matrix
from gdop import gdop_2d
from tdoa_solver import ranges

def crb_rmse(anchor_xy, p, cfg):
    dz = cfg.tag_height - cfg.an_height
    r = ranges(anchor_xy, p, dz)
    sig = cfg.sigma_tdoa_floor + cfg.sigma_tdoa_coeff * r
    u = (p - anchor_xy) / r[:, None]            # d r_i / d p  (xy)
    J = u[1:] - u[0]
    S = np.diag(sig[1:] ** 2) + sig[0] ** 2 * np.ones((len(r) - 1, len(r) - 1))
    F = J.T @ np.linalg.solve(S, J)
    return float(np.sqrt(np.trace(np.linalg.inv(F)))), float(sig.mean())

out = {}
for g, cfg, lay, val in [("1", GARAGE_1, "best_layouts.json", "accuracy_validation_both_garage1.json"),
                         ("2", GARAGE_2, "best_layouts_g2.json", "accuracy_validation_both_garage2.json")]:
    sc = build_scenario(cfg)
    cand, samp = sc["candidate_AN_points"], sc["vehicle_sample_points"]
    L = json.load(open(lay)); V = json.load(open(val))
    for key in ("best_pso_idx", "best_ga_idx"):
        res = V[key]
        gd = np.array([x["gdop"] for x in res]); mc = np.array([x["rmse_cm"] for x in res])
        # recompute active anchors for each point
        vis = build_visibility_matrix(cand, samp, sc["columns"], sc["parked_cars"], cfg.an_height, cfg.tag_height)
        lc = cand[np.array(L[key])]; lv = vis[np.array(L[key]), :]
        crb = []; keep = []
        for x in res:
            k = int(np.argmin(np.linalg.norm(samp - np.array(x["point"]), axis=1)))
            act = lv[:, k]; crb.append(crb_rmse(lc[act], samp[k], cfg)[0] * 100)
        crb = np.array(crb)
        a_g = np.median(mc / gd)
        a_c = np.median(mc / crb)
        r_g = np.corrcoef(gd, mc)[0, 1]; r_c = np.corrcoef(crb, mc)[0, 1]
        from scipy.stats import spearmanr
        s_g = spearmanr(gd, mc)[0]; s_c = spearmanr(crb, mc)[0]
        ratio = mc / crb
        top = np.argsort(-mc)[:5]
        print(f"G{g} {key}: n={len(res)}  Pearson GDOP={r_g:.3f} CRB={r_c:.3f} | Spearman GDOP={s_g:.3f} CRB={s_c:.3f}")
        print(f"   median MC/CRB={np.median(ratio):.2f}  p95={np.percentile(ratio,95):.2f}  max={ratio.max():.1f}; median CRB={np.median(crb):.1f}cm  median MC={np.median(mc):.1f}cm")
        print("   worst MC points: " + "; ".join(f"MC={mc[i]:.0f} CRB={crb[i]:.1f} GDOP={gd[i]:.2f} n={res[i]['n_active']}" for i in top))
        out[f"g{g}_{key}"] = dict(pearson_gdop=r_g, pearson_crb=r_c, spearman_gdop=s_g, spearman_crb=s_c,
            median_ratio=float(np.median(ratio)), p95_ratio=float(np.percentile(ratio, 95)), max_ratio=float(ratio.max()),
            median_crb_cm=float(np.median(crb)), median_mc_cm=float(np.median(mc)),
            n_ratio_gt_2=int((ratio > 2).sum()), n=len(res))
json.dump(out, open("crb_results.json", "w"), indent=2)
