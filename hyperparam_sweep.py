"""
hyperparam_sweep.py

Tuning check for the PSO-vs-GA comparison (Table 5, Garage 1; Section 5.3): is the gap an artefact of the default hyperparameters?

The budget is held FIXED (swarm/population 40, 150 iterations/generations) across every configuration, matching the equal-budget
principle; only each algorithm's own behavioural parameters are varied.

Step 1: sweep six PSO and six GA configurations, 5 seeds each (1000-1004), Garage 1 (M = 16).
Step 2: re-run the best-median configuration of each algorithm over 20 seeds (1000-1019) and compare (Mann-Whitney U).

Output: hyperparam_sweep_results.json, hyperparam_tuned_comparison.json
(the "original_*" values in the summary are the Table 5 reference values, hard-coded).
"""

import time
import json
import numpy as np
from scipy.stats import mannwhitneyu

from garage_model import build_scenario, GARAGE_1
from visibility import build_visibility_matrix
from pso import binary_pso
from ga import genetic_algorithm

scenario = build_scenario(GARAGE_1)
candidates = scenario["candidate_AN_points"]
samples = scenario["vehicle_sample_points"]
vis = build_visibility_matrix(candidates, samples, scenario["columns"], scenario["parked_cars"],
                               GARAGE_1.an_height, GARAGE_1.tag_height)

M = 16
S_SWEEP = 5
SEED_BASE = 1000

PSO_CONFIGS = {
    "default (w0.9-0.4, c1=2.0, c2=2.0)": dict(w_start=0.9, w_end=0.4, c1=2.0, c2=2.0),
    "high-explore (w0.95-0.2, c1=2.5, c2=1.5)": dict(w_start=0.95, w_end=0.2, c1=2.5, c2=1.5),
    "low-inertia (w0.7-0.2, c1=1.5, c2=1.5)": dict(w_start=0.7, w_end=0.2, c1=1.5, c2=1.5),
    "balanced (w0.8-0.3, c1=1.8, c2=1.8)": dict(w_start=0.8, w_end=0.3, c1=1.8, c2=1.8),
    "social-dominant (c1=1.0, c2=2.5)": dict(w_start=0.9, w_end=0.4, c1=1.0, c2=2.5),
    "cognitive-dominant (c1=2.5, c2=1.0)": dict(w_start=0.9, w_end=0.4, c1=2.5, c2=1.0),
}

GA_CONFIGS = {
    "default (mut=0.03, k=3, elit=1)": dict(mutation_rate=0.03, tournament_k=3, elitism=1),
    "high-mutation (mut=0.10, k=3, elit=1)": dict(mutation_rate=0.10, tournament_k=3, elitism=1),
    "low-mutation (mut=0.01, k=3, elit=1)": dict(mutation_rate=0.01, tournament_k=3, elitism=1),
    "high-pressure (mut=0.03, k=5, elit=2)": dict(mutation_rate=0.03, tournament_k=5, elitism=2),
    "low-pressure (mut=0.03, k=2, elit=1)": dict(mutation_rate=0.03, tournament_k=2, elitism=1),
    "high-elitism (mut=0.05, k=3, elit=4)": dict(mutation_rate=0.05, tournament_k=3, elitism=4),
}

results = {"pso_sweep": {}, "ga_sweep": {}}

print("=" * 70)
print("STEP 1a: PSO hyperparameter sweep (S=5 seeds each)")
print("=" * 70)
for name, params in PSO_CONFIGS.items():
    Fs = []
    t0 = time.time()
    for s in range(S_SWEEP):
        idx, F, _, _ = binary_pso(candidates, samples, vis, M=M, swarm_size=40, iters=150,
                                   seed=SEED_BASE + s, **params)
        Fs.append(F)
    Fs = np.array(Fs)
    results["pso_sweep"][name] = {"mean": float(Fs.mean()), "median": float(np.median(Fs)),
                                   "std": float(Fs.std(ddof=1)), "all": Fs.tolist()}
    print(f"  {name:45s} mean={Fs.mean():7.2f} median={np.median(Fs):7.2f} std={Fs.std(ddof=1):6.2f}  ({time.time()-t0:.0f}s)", flush=True)

print("\n" + "=" * 70)
print("STEP 1b: GA hyperparameter sweep (S=5 seeds each)")
print("=" * 70)
for name, params in GA_CONFIGS.items():
    Fs = []
    t0 = time.time()
    for s in range(S_SWEEP):
        idx, F, _, _ = genetic_algorithm(candidates, samples, vis, M=M, pop_size=40, generations=150,
                                          seed=SEED_BASE + s, **params)
        Fs.append(F)
    Fs = np.array(Fs)
    results["ga_sweep"][name] = {"mean": float(Fs.mean()), "median": float(np.median(Fs)),
                                  "std": float(Fs.std(ddof=1)), "all": Fs.tolist()}
    print(f"  {name:45s} mean={Fs.mean():7.2f} median={np.median(Fs):7.2f} std={Fs.std(ddof=1):6.2f}  ({time.time()-t0:.0f}s)", flush=True)

best_pso_name = min(results["pso_sweep"], key=lambda k: results["pso_sweep"][k]["median"])
best_ga_name = min(results["ga_sweep"], key=lambda k: results["ga_sweep"][k]["median"])
print(f"\nBest PSO config: {best_pso_name}")
print(f"Best GA config:  {best_ga_name}")

json.dump(results, open("hyperparam_sweep_results.json", "w"), indent=2)

print("\n" + "=" * 70)
print("STEP 2: Full S=20 head-to-head with BEST-TUNED configs (vs. original)")
print("=" * 70)
S_FULL = 20
best_pso_params = PSO_CONFIGS[best_pso_name]
best_ga_params = GA_CONFIGS[best_ga_name]

F_PSO_tuned, F_GA_tuned = [], []
for s in range(S_FULL):
    seed = SEED_BASE + s
    _, Fp, _, _ = binary_pso(candidates, samples, vis, M=M, swarm_size=40, iters=150, seed=seed, **best_pso_params)
    _, Fg, _, _ = genetic_algorithm(candidates, samples, vis, M=M, pop_size=40, generations=150, seed=seed, **best_ga_params)
    F_PSO_tuned.append(Fp)
    F_GA_tuned.append(Fg)
    print(f"  seed {s+1:2d}/{S_FULL}: PSO(tuned) F={Fp:8.3f}   GA(tuned) F={Fg:8.3f}", flush=True)

F_PSO_tuned = np.array(F_PSO_tuned)
F_GA_tuned = np.array(F_GA_tuned)
U, pval = mannwhitneyu(F_PSO_tuned, F_GA_tuned, alternative="two-sided")
r = 1 - (2 * U) / (len(F_PSO_tuned) * len(F_GA_tuned))

summary = {
    "best_pso_config": best_pso_name, "best_ga_config": best_ga_name,
    "tuned_mean_PSO": float(F_PSO_tuned.mean()), "tuned_median_PSO": float(np.median(F_PSO_tuned)),
    "tuned_mean_GA": float(F_GA_tuned.mean()), "tuned_median_GA": float(np.median(F_GA_tuned)),
    "tuned_U": float(U), "tuned_p": float(pval), "tuned_r": float(r),
    "original_mean_PSO": 58.21, "original_mean_GA": 27.98, "original_p": 0.0051, "original_r": -0.52,
}
print("\n--- SUMMARY ---")
print(json.dumps(summary, indent=2))
json.dump(summary, open("hyperparam_tuned_comparison.json", "w"), indent=2)
