"""
fitness.py
Algorithm 1 (Evaluate): shared fitness function for PSO and GA. Hard
constraints (C3 budget, C4 sync connectivity, C5 spacing); soft
penalties for (C1) coverage and (C2) GDOP quality.

NOTE on scope: a distance-ratio-based constraint was investigated and deliberately NOT added: a redundant
anchor can still cause ghost-solution (second-root) failures at n = 4 despite an acceptable distance ratio and GDOP
(Section 5.2). The objective therefore uses n_min = 4 and GDOP_max only.
"""

import numpy as np
from gdop import gdop_all_points

DEFAULT_PARAMS = {
    "M_max": 16,
    "R_sync": 15.0,
    "d_min": 4.0,
    "min_los_anchors": 4,   # four, not three (Section 5.1, Table 3)
    "coverage_threshold": 0.95,
    "coverage_penalty_weight": 200.0,
    "GDOP_max": 3.0,
    "gdop_penalty_weight": 50.0,
    "w1": 1.0, "w2": 1.0, "w3": 0.05,
}


def is_connected(coords, R_sync):
    n = len(coords)
    if n <= 1:
        return True
    dists = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    adj = dists <= R_sync
    np.fill_diagonal(adj, False)
    visited = np.zeros(n, dtype=bool)
    stack = [0]
    visited[0] = True
    while stack:
        node = stack.pop()
        neighbors = np.where(adj[node] & ~visited)[0]
        visited[neighbors] = True
        stack.extend(neighbors.tolist())
    return visited.all()


def min_pairwise_distance(coords):
    n = len(coords)
    if n <= 1:
        return np.inf
    dists = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    return dists.min()


def evaluate(selected_idx, candidates, samples, vis_matrix, params=None, verbose=False):
    p = {**DEFAULT_PARAMS, **(params or {})}
    selected_idx = np.asarray(selected_idx)
    diag = {"status": None}

    if len(selected_idx) > p["M_max"]:
        diag["status"] = "infeasible: exceeds M_max"
        return np.inf, diag

    sel_coords = candidates[selected_idx]

    if not is_connected(sel_coords, p["R_sync"]):
        diag["status"] = "infeasible: sync graph disconnected"
        return np.inf, diag

    if min_pairwise_distance(sel_coords) < p["d_min"]:
        diag["status"] = "infeasible: anchors too close (violates d_min)"
        return np.inf, diag

    active_mask = np.zeros_like(vis_matrix, dtype=bool)
    active_mask[selected_idx, :] = vis_matrix[selected_idx, :]
    n_active = active_mask.sum(axis=0)
    covered = n_active >= p["min_los_anchors"]
    coverage_frac = covered.mean()

    if coverage_frac == 0.0:
        diag["status"] = "infeasible: 0% coverage"
        return np.inf, diag

    covered_idx = np.where(covered)[0]
    gdop = gdop_all_points(candidates, samples[covered_idx], active_mask[:, covered_idx])

    f1 = gdop.mean()
    f2 = gdop.max()
    f3 = len(selected_idx)
    F = p["w1"] * f1 + p["w2"] * f2 + p["w3"] * f3

    if f2 > p["GDOP_max"]:
        F += p["gdop_penalty_weight"] * (f2 - p["GDOP_max"])
    if coverage_frac < p["coverage_threshold"]:
        F += p["coverage_penalty_weight"] * (p["coverage_threshold"] - coverage_frac)

    diag.update({"status": "feasible", "coverage_frac": coverage_frac,
                 "f1_mean_gdop": f1, "f2_max_gdop": f2, "f3_n_anchors": f3, "F": F})
    if verbose:
        print(diag)
    return F, diag


if __name__ == "__main__":
    from garage_model import build_scenario, GARAGE_1
    from visibility import build_visibility_matrix

    scenario = build_scenario(GARAGE_1)
    candidates = scenario["candidate_AN_points"]
    samples = scenario["vehicle_sample_points"]
    vis = build_visibility_matrix(candidates, samples, scenario["columns"], scenario["parked_cars"],
                                   GARAGE_1.an_height, GARAGE_1.tag_height)
    rng = np.random.default_rng(7)

    idx = rng.choice(len(candidates), size=16, replace=False)
    F, diag = evaluate(idx, candidates, samples, vis)
    print("Random 16-anchor placement:", diag)

    best_F, best_diag = np.inf, None
    for _ in range(100):
        idx = rng.choice(len(candidates), size=16, replace=False)
        F, diag = evaluate(idx, candidates, samples, vis)
        if F < best_F:
            best_F, best_diag = F, diag
    print("Best of 100 random:", best_diag)
