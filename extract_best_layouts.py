"""
extract_best_layouts.py

Selects, for each algorithm, the seed with the lowest (best) F(x) from the multi-seed run of compare.py (which saves every seed's
layout, Idx_PSO / Idx_GA, and its F) and writes the two layouts as {"best_pso_idx": [...], "best_ga_idx": [...]}, the format read by
validate_accuracy.py, old_vs_realistic_solver.py, the technology comparison scripts, ci_and_sensitivity.py and make_figures.py.

Usage: python extract_best_layouts.py <comparison_report.json> <comparison_curves.json> <best_layouts.json>
"""

import json
import numpy as np
import sys


def extract(report_path, curves_path, out_path):
    report = json.load(open(report_path))
    curves = json.load(open(curves_path))

    F_PSO = np.array(report["F_PSO_all"])
    F_GA = np.array(report["F_GA_all"])
    best_pso_seed = int(np.argmin(F_PSO))
    best_ga_seed = int(np.argmin(F_GA))

    best_pso_idx = curves["Idx_PSO"][best_pso_seed]
    best_ga_idx = curves["Idx_GA"][best_ga_seed]

    out = {
        "best_pso_idx": best_pso_idx,
        "best_ga_idx": best_ga_idx,
        "best_pso_F": float(F_PSO[best_pso_seed]),
        "best_ga_F": float(F_GA[best_ga_seed]),
        "best_pso_seed_rank": best_pso_seed,
        "best_ga_seed_rank": best_ga_seed,
        "source_report": report_path,
        "garage": report["garage"],
        "M": report["M"],
    }
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"{report['garage']}: best PSO F={out['best_pso_F']:.3f} (n={len(best_pso_idx)}), "
          f"best GA F={out['best_ga_F']:.3f} (n={len(best_ga_idx)}) -> {out_path}")


if __name__ == "__main__":
    report_path = sys.argv[1] if len(sys.argv) > 1 else "comparison_report.json"
    curves_path = sys.argv[2] if len(sys.argv) > 2 else "comparison_curves.json"
    out_path = sys.argv[3] if len(sys.argv) > 3 else "best_layouts.json"
    extract(report_path, curves_path, out_path)
