"""smoke_test.py - one-minute check that this environment reproduces the paper's pipeline.

Checks (about 1 minute on one core):
  1. software versions against those the results were produced with;
  2. garage geometry: Garage 1 has 86 candidate anchors and 260 evaluation points, Garage 2 has 55 and 79; the example vehicle position of Figure 2 sees 34 of 86 anchors;
  3. the noise-free TDoA solver recovers a known position;
  4. one PSO run and one GA run (seed 1000, Table 5 protocol) in each garage return exactly the objective values stored in results/.

PSO and GA outputs depend on NumPy's random stream (paper, Section 6.4). If check 4 fails on a different NumPy/SciPy version, the numbers it prints
show how far the runs have drifted; the qualitative conclusions are not affected by a single draw, but exact table values may differ.
Usage: python smoke_test.py          (exit status 0 = all checks passed)
"""
import json, sys, time
from pathlib import Path
import numpy as np, scipy, shapely

from garage_model import build_scenario, GARAGE_1, GARAGE_2
from visibility import build_visibility_matrix
from pso import binary_pso
from ga import genetic_algorithm
from tdoa_solver import simulate_measurements, solve_position

HERE = Path(__file__).resolve().parent
TESTED = dict(numpy="2.5.3", scipy="1.18.1", shapely="2.2.0")
fails = []


def check(ok, msg):
    print(("  PASS  " if ok else "  FAIL  ") + msg, flush=True)
    if not ok: fails.append(msg)


print("1. software versions (results were produced with Python 3.13, NumPy 2.5.3, SciPy 1.18.1, Shapely 2.2.0)")
for name, v in (("numpy", np.__version__), ("scipy", scipy.__version__), ("shapely", shapely.__version__)):
    print(f"  {name} {v}" + ("" if v == TESTED[name] else f"   (differs from {TESTED[name]}; exact reproduction of random runs is not guaranteed)"))

print("2. geometry")
sc1 = build_scenario(GARAGE_1); cand1, samp1 = sc1["candidate_AN_points"], sc1["vehicle_sample_points"]
vis1 = build_visibility_matrix(cand1, samp1, sc1["columns"], sc1["parked_cars"], GARAGE_1.an_height, GARAGE_1.tag_height)
check(len(cand1) == 86 and len(samp1) == 260, f"Garage 1: {len(cand1)} candidate anchors, {len(samp1)} evaluation points (expected 86 and 260)")
sc2 = build_scenario(GARAGE_2)
check(len(sc2["candidate_AN_points"]) == 55 and len(sc2["vehicle_sample_points"]) == 79,
      f"Garage 2: {len(sc2['candidate_AN_points'])} candidate anchors, {len(sc2['vehicle_sample_points'])} evaluation points (expected 55 and 79)")
k = int(np.argmin(np.sum((samp1 - np.array([25, 15])) ** 2, axis=1)))
check(int(vis1[:, k].sum()) == 34, f"Figure 2 example position sees {int(vis1[:, k].sum())}/86 anchors (expected 34)")

print("3. noise-free TDoA solver")
anchors = cand1[np.array(json.load(open(HERE / "results" / "best_layouts.json"))["best_ga_idx"])][:6]
p_true = np.array([20.0, 18.0]); dz = GARAGE_1.tag_height - GARAGE_1.an_height
D = simulate_measurements(anchors, p_true, 0.0, 0.0, np.random.default_rng(0), dz); p_hat, _ = solve_position(anchors, D, dz)
check(np.linalg.norm(p_hat - p_true) < 1e-4, f"recovered position error {np.linalg.norm(p_hat - p_true):.2e} m")

print("4. one PSO and one GA run per garage (seed 1000) against results/comparison_report*.json")
t0 = time.time()
for cfg, M, suffix, name in ((GARAGE_1, 16, "", "Garage 1"), (GARAGE_2, 8, "_g2", "Garage 2")):
    rep = json.load(open(HERE / "results" / f"comparison_report{suffix}.json"))
    sc = build_scenario(cfg); cand, samp = sc["candidate_AN_points"], sc["vehicle_sample_points"]
    vis = build_visibility_matrix(cand, samp, sc["columns"], sc["parked_cars"], cfg.an_height, cfg.tag_height)
    fp = {"M_max": M} if M != 16 else None
    _, Fp, _, _ = binary_pso(cand, samp, vis, M=M, swarm_size=40, iters=150, seed=1000, fitness_params=fp)
    _, Fg, _, _ = genetic_algorithm(cand, samp, vis, M=M, pop_size=40, generations=150, seed=1000, fitness_params=fp)
    check(abs(Fp - rep["F_PSO_all"][0]) < 1e-6, f"{name} PSO F = {Fp:.6f} (stored {rep['F_PSO_all'][0]:.6f})")
    check(abs(Fg - rep["F_GA_all"][0]) < 1e-6, f"{name} GA  F = {Fg:.6f} (stored {rep['F_GA_all'][0]:.6f})")
print(f"  ({time.time() - t0:.0f} s)")

print("\nALL CHECKS PASSED" if not fails else f"\n{len(fails)} CHECK(S) FAILED")
sys.exit(1 if fails else 0)
