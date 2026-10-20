"""
ga.py
Algorithm 3 (Genetic Algorithm). Same binary chromosome representation
as pso.py's particle encoding, for a direct, apples-to-apples comparison.
"""

import numpy as np
from fitness import evaluate


def _random_chromosome(N, M, rng):
    chrom = np.zeros(N, dtype=bool)
    idx = rng.choice(N, size=M, replace=False)
    chrom[idx] = True
    return chrom


def _repair_cardinality(chrom, M, rng):
    selected = np.where(chrom)[0]
    n = len(selected)
    if n == M:
        return chrom
    chrom = chrom.copy()
    if n > M:
        drop = rng.choice(selected, size=n - M, replace=False)
        chrom[drop] = False
    else:
        unselected = np.where(~chrom)[0]
        add = rng.choice(unselected, size=M - n, replace=False)
        chrom[add] = True
    return chrom


def _tournament_select(pop, fitness, k, rng):
    idxs = rng.choice(len(pop), size=k, replace=False)
    best = idxs[np.argmin(fitness[idxs])]
    return pop[best]


def _uniform_crossover(p1, p2, M, rng):
    mask = rng.random(len(p1)) < 0.5
    child = np.where(mask, p1, p2)
    return _repair_cardinality(child, M, rng)


def _mutate(chrom, M, rate, rng):
    if rng.random() > rate:
        return chrom
    chrom = chrom.copy()
    selected = np.where(chrom)[0]
    unselected = np.where(~chrom)[0]
    if len(selected) == 0 or len(unselected) == 0:
        return chrom
    chrom[rng.choice(selected)] = False
    chrom[rng.choice(unselected)] = True
    return chrom


def genetic_algorithm(candidates, samples, vis, M=16, pop_size=40, generations=150,
                       tournament_k=3, mutation_rate=0.03, elitism=1,
                       seed=None, fitness_params=None, verbose=False):
    rng = np.random.default_rng(seed)
    N = len(candidates)

    pop = [_random_chromosome(N, M, rng) for _ in range(pop_size)]
    fitness = np.empty(pop_size)
    for i, chrom in enumerate(pop):
        fitness[i], _ = evaluate(np.where(chrom)[0], candidates, samples, vis, fitness_params)
    n_evals = pop_size

    best_idx = np.argmin(fitness)
    best_chrom = pop[best_idx].copy()
    best_fitness = fitness[best_idx]
    history = [best_fitness]

    for g in range(generations):
        order = np.argsort(fitness)
        new_pop = [pop[order[i]].copy() for i in range(elitism)]
        while len(new_pop) < pop_size:
            parent1 = _tournament_select(pop, fitness, tournament_k, rng)
            parent2 = _tournament_select(pop, fitness, tournament_k, rng)
            child = _uniform_crossover(parent1, parent2, M, rng)
            child = _mutate(child, M, mutation_rate, rng)
            new_pop.append(child)

        pop = new_pop
        fitness = np.empty(pop_size)
        for i, chrom in enumerate(pop):
            fitness[i], _ = evaluate(np.where(chrom)[0], candidates, samples, vis, fitness_params)
        n_evals += pop_size

        gen_best_idx = np.argmin(fitness)
        if fitness[gen_best_idx] < best_fitness:
            best_fitness = fitness[gen_best_idx]
            best_chrom = pop[gen_best_idx].copy()
        history.append(best_fitness)
        if verbose and (g % 20 == 0 or g == generations - 1):
            print(f"  gen {g:4d}  best F = {best_fitness:.4f}")

    return np.where(best_chrom)[0], best_fitness, history, n_evals


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
    idx, F, history, n_evals = genetic_algorithm(candidates, samples, vis, M=16, pop_size=40, generations=150, seed=42)
    print(f"GA: F={F:.2f}, time={time.time()-t0:.1f}s, n_evals={n_evals}")
