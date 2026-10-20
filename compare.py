"""
compare.py
Algorithm 4: multi-seed statistical comparison of PSO vs GA. Equal
fitness-evaluation budget; Wilcoxon rank-sum test (Mann-Whitney U) for
INDEPENDENT samples (not paired - PSO/GA consume randomness differently
even under a shared seed label).
"""

import time
import json
import numpy as np
from scipy.stats import mannwhitneyu

from garage_model import build_scenario, GARAGE_1
from visibility import build_visibility_matrix
from pso import binary_pso
from ga import genetic_algorithm


def run_comparison(cfg=GARAGE_1, fitness_params=None, S=20, M=16, swarm_pop=40,
                    budget_iters=150, seed_base=1000):
    scenario = build_scenario(cfg)
    candidates = scenario["candidate_AN_points"]
    samples = scenario["vehicle_sample_points"]
    vis = build_visibility_matrix(candidates, samples, scenario["columns"], scenario["parked_cars"],
                                   cfg.an_height, cfg.tag_height)

    F_PSO, F_GA, T_PSO, T_GA, Idx_PSO, Idx_GA = [], [], [], [], [], []
    for s in range(S):
        seed = seed_base + s
        t0 = time.time()
        idx_p, F_p, _, _ = binary_pso(candidates, samples, vis, M=M, swarm_size=swarm_pop,
                                        iters=budget_iters, seed=seed, fitness_params=fitness_params)
        t_p = time.time() - t0
        t0 = time.time()
        idx_g, F_g, _, _ = genetic_algorithm(candidates, samples, vis, M=M, pop_size=swarm_pop,
                                               generations=budget_iters, seed=seed, fitness_params=fitness_params)
        t_g = time.time() - t0

        F_PSO.append(F_p); F_GA.append(F_g)
        T_PSO.append(t_p); T_GA.append(t_g)
        Idx_PSO.append(idx_p.tolist()); Idx_GA.append(idx_g.tolist())
        print(f"seed {s+1:2d}/{S}: PSO F={F_p:8.3f} ({t_p:.1f}s)   GA F={F_g:8.3f} ({t_g:.1f}s)")

    F_PSO = np.array(F_PSO); F_GA = np.array(F_GA)
    U, pval = mannwhitneyu(F_PSO, F_GA, alternative="two-sided")
    r = 1 - (2 * U) / (len(F_PSO) * len(F_GA))

    report = {
        "garage": cfg.name, "M": M, "S": S,
        "mean_PSO": float(F_PSO.mean()), "std_PSO": float(F_PSO.std(ddof=1)), "median_PSO": float(np.median(F_PSO)),
        "mean_GA": float(F_GA.mean()), "std_GA": float(F_GA.std(ddof=1)), "median_GA": float(np.median(F_GA)),
        "mean_runtime_PSO": float(np.mean(T_PSO)), "mean_runtime_GA": float(np.mean(T_GA)),
        "mannwhitney_U": float(U), "p_value": float(pval), "rank_biserial_r": float(r),
        "F_PSO_all": F_PSO.tolist(), "F_GA_all": F_GA.tolist(),
    }
    if pval < 0.05 and np.median(F_PSO) < np.median(F_GA):
        report["winner"] = "PSO (statistically significant)"
    elif pval < 0.05 and np.median(F_GA) < np.median(F_PSO):
        report["winner"] = "GA (statistically significant)"
    else:
        report["winner"] = "no statistically significant difference"

    return report, {"Idx_PSO": Idx_PSO, "Idx_GA": Idx_GA}


if __name__ == "__main__":
    import sys
    from garage_model import GARAGE_1, GARAGE_2

    S = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    seed_base = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    out_suffix = sys.argv[3] if len(sys.argv) > 3 else ""
    garage_choice = sys.argv[4] if len(sys.argv) > 4 else "1"
    M_override = int(sys.argv[5]) if len(sys.argv) > 5 else None

    cfg = GARAGE_1 if garage_choice == "1" else GARAGE_2
    M = M_override if M_override is not None else 16
    fitness_params = {"M_max": M} if M_override is not None else None

    print(f"PSO vs GA on {cfg.name}: S={S}, seed_base={seed_base}, M={M}\n")
    t0 = time.time()
    report, curves = run_comparison(cfg=cfg, fitness_params=fitness_params, S=S, M=M, seed_base=seed_base)
    print(f"\nTotal: {time.time()-t0:.1f}s")
    print(f"PSO: mean={report['mean_PSO']:.2f} median={report['median_PSO']:.2f}")
    print(f"GA:  mean={report['mean_GA']:.2f} median={report['median_GA']:.2f}")
    print(f"U={report['mannwhitney_U']:.1f}, p={report['p_value']:.4f}, r={report['rank_biserial_r']:.3f}")
    print(f"Winner: {report['winner']}")

    json.dump(report, open(f"comparison_report{out_suffix}.json", "w"), indent=2)
    json.dump(curves, open(f"comparison_curves{out_suffix}.json", "w"))
    print(f"Saved comparison_report{out_suffix}.json / comparison_curves{out_suffix}.json")
