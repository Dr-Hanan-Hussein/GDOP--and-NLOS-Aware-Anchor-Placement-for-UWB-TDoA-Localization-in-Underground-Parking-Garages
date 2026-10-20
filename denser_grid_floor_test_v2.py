"""
denser_grid_floor_test_v2.py

Checks whether the strict worst-case GDOP floor in Garage 1 (Section 5.5, "Strict form") is an artefact of the 86-point candidate grid.

The GA's random initialisation/crossover almost never satisfies the d_min = 4 m spacing constraint (C5) on a 2 m grid (357 candidates),
so this test uses a d_min-aware greedy-random layout sampler (500 random restarts, C5 respected by construction), applied IDENTICALLY
to the original 86-candidate grid and the denser 357-candidate grid. At M = 52-60 the best worst-case GDOP of the two grids is within
about 3%; the sampler is a weaker search than the GA, so this is suggestive rather than conclusive evidence of a geometric limit.

Seed 2026.  Output: denser_grid_floor_test_v2.json
"""

import time
import json
import numpy as np
from shapely.geometry import Point

from garage_model import build_scenario, GARAGE_1
from visibility import build_visibility_matrix
from fitness import DEFAULT_PARAMS, evaluate

M_GRID = [16, 24, 32, 40, 44, 48, 52, 56, 60]
N_RESTARTS = 500
SEED = 2026

cfg = GARAGE_1
scenario = build_scenario(cfg)
samples = scenario["vehicle_sample_points"]
columns = scenario["columns"]
cars = scenario["parked_cars"]
orig_candidates = scenario["candidate_AN_points"]
orig_vis = build_visibility_matrix(orig_candidates, samples, columns, cars, cfg.an_height, cfg.tag_height)


def build_dense_candidates(cfg, columns, spacing):
    cands = []
    x_nodes = np.arange(cfg.wall_margin, cfg.width - cfg.wall_margin + 1e-6, spacing)
    y_nodes = np.arange(cfg.wall_margin, cfg.length - cfg.wall_margin + 1e-6, spacing)
    for x in x_nodes:
        for y in y_nodes:
            pt = Point(x, y)
            if not any(pt.distance(c) < 0.3 for c in columns):
                cands.append((x, y))
    return np.array(cands)


dense_candidates = build_dense_candidates(cfg, columns, spacing=2.0)
dense_vis = build_visibility_matrix(dense_candidates, samples, columns, cars, cfg.an_height, cfg.tag_height)
print(f"Original: {len(orig_candidates)} candidates. Dense: {len(dense_candidates)} candidates.")


def greedy_spaced_layout(coords, M, d_min, rng, max_tries=300):
    """Randomly order candidates, greedily accept ones that keep >= d_min
    from all already-accepted ones. Retries with a fresh random order on
    dead-end. Returns an index array of length M, or None if it never
    manages to reach M within max_tries restarts."""
    N = len(coords)
    for _ in range(max_tries):
        order = rng.permutation(N)
        selected = []
        for idx in order:
            if not selected:
                selected.append(idx)
            else:
                d = np.linalg.norm(coords[selected] - coords[idx], axis=1).min()
                if d >= d_min:
                    selected.append(idx)
            if len(selected) == M:
                return np.array(selected)
    return None


def best_random_search(candidates, vis, M, n_restarts, seed, label):
    rng = np.random.default_rng(seed)
    best_F, best_diag = np.inf, None
    n_feasible = 0
    for _ in range(n_restarts):
        idx = greedy_spaced_layout(candidates, M, DEFAULT_PARAMS["d_min"], rng)
        if idx is None:
            continue
        fp = {**DEFAULT_PARAMS, "M_max": M}
        F, diag = evaluate(idx, candidates, samples, vis, fp)
        if diag["status"] == "feasible":
            n_feasible += 1
            if F < best_F:
                best_F, best_diag = F, diag
    return best_F, best_diag, n_feasible


results = {"original": {}, "dense": {}}
for label, cands, vis in [("original", orig_candidates, orig_vis), ("dense", dense_candidates, dense_vis)]:
    for M in M_GRID:
        t0 = time.time()
        F, diag, n_feas = best_random_search(cands, vis, M, N_RESTARTS, SEED, label)
        f2 = diag["f2_max_gdop"] if diag else float("inf")
        cov = diag["coverage_frac"] if diag else 0.0
        results[label][M] = {"f2_max_gdop": f2, "coverage_frac": cov, "n_feasible_found": n_feas, "F": F}
        print(f"  [{label:8s}] M={M:3d}  worst-case GDOP={f2:7.3f}  coverage={cov*100:5.1f}%  "
              f"feasible found={n_feas}/{N_RESTARTS}  ({time.time()-t0:.0f}s)", flush=True)

print("\n=== COMPARISON (best-of-500 random spacing-respecting restarts, identical method both grids) ===")
print(f"{'M':>4}  {'Original (86 cand.)':>20}  {'Denser (357 cand.)':>20}  {'Improvement':>12}")
for M in M_GRID:
    o = results["original"][M]["f2_max_gdop"]
    d = results["dense"][M]["f2_max_gdop"]
    imp = (o - d) / o * 100 if np.isfinite(o) and np.isfinite(d) and o > 0 else float("nan")
    print(f"{M:4d}  {o:20.3f}  {d:20.3f}  {imp:11.1f}%")

json.dump(results, open("denser_grid_floor_test_v2.json", "w"), indent=2)
print("\nSaved denser_grid_floor_test_v2.json")
