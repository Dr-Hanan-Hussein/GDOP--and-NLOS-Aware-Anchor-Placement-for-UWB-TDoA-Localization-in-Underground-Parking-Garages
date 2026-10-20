"""
pso.py
Algorithm 2 (Binary PSO). Top-M selection by sigmoid-transformed
probability automatically satisfies the cardinality constraint (C3/C6).
"""

import numpy as np
from fitness import evaluate


def binary_pso(candidates, samples, vis, M=16, swarm_size=40, iters=150,
                w_start=0.9, w_end=0.4, c1=2.0, c2=2.0, seed=None,
                fitness_params=None, verbose=False):
    rng = np.random.default_rng(seed)
    N = len(candidates)

    velocities = rng.uniform(-2, 2, size=(swarm_size, N))
    positions = np.zeros((swarm_size, N), dtype=bool)
    for s in range(swarm_size):
        idx = rng.choice(N, size=M, replace=False)
        positions[s, idx] = True

    def sel_idx(pos_row):
        return np.where(pos_row)[0]

    pbest = positions.copy()
    pbest_fitness = np.empty(swarm_size)
    for s in range(swarm_size):
        pbest_fitness[s], _ = evaluate(sel_idx(positions[s]), candidates, samples, vis, fitness_params)
    n_evals = swarm_size

    g_idx = np.argmin(pbest_fitness)
    gbest = pbest[g_idx].copy()
    gbest_fitness = pbest_fitness[g_idx]
    history = [gbest_fitness]

    for t in range(iters):
        w = w_start - (w_start - w_end) * t / iters
        for s in range(swarm_size):
            r1 = rng.random(N)
            r2 = rng.random(N)
            velocities[s] = (w * velocities[s]
                              + c1 * r1 * (pbest[s].astype(float) - positions[s].astype(float))
                              + c2 * r2 * (gbest.astype(float) - positions[s].astype(float)))
            velocities[s] = np.clip(velocities[s], -6, 6)
            prob = 1.0 / (1.0 + np.exp(-velocities[s]))
            top_idx = np.argpartition(-prob, M)[:M]
            new_pos = np.zeros(N, dtype=bool)
            new_pos[top_idx] = True
            positions[s] = new_pos

            F, _ = evaluate(top_idx, candidates, samples, vis, fitness_params)
            n_evals += 1
            if F < pbest_fitness[s]:
                pbest[s] = new_pos
                pbest_fitness[s] = F

        g_idx = np.argmin(pbest_fitness)
        if pbest_fitness[g_idx] < gbest_fitness:
            gbest = pbest[g_idx].copy()
            gbest_fitness = pbest_fitness[g_idx]
        history.append(gbest_fitness)
        if verbose and (t % 20 == 0 or t == iters - 1):
            print(f"  iter {t:4d}  best F = {gbest_fitness:.4f}")

    return sel_idx(gbest), gbest_fitness, history, n_evals


if __name__ == "__main__":
    import time
    from garage_model import build_scenario, GARAGE_1
    from visibility import build_visibility_matrix

    scenario = build_scenario(GARAGE_1)
    candidates = scenario["candidate_AN_points"]
    samples = scenario["vehicle_sample_points"]
    vis = build_visibility_matrix(candidates, samples, scenario["columns"], scenario["parked_cars"],
                                   GARAGE_1.an_height, GARAGE_1.tag_height)
    t0 = time.time()
    idx, F, history, n_evals = binary_pso(candidates, samples, vis, M=16, swarm_size=40, iters=150, seed=42)
    print(f"PSO: F={F:.2f}, time={time.time()-t0:.1f}s, n_evals={n_evals}")
