# Replication package: GDOP- and NLOS-aware anchor placement for UWB-TDoA localization in underground parking garages

Code, fixed random seeds and result files that reproduce every table and figure of

> H. H. Hussein, *GDOP- and NLOS-Aware Anchor Placement for UWB-TDoA Localization in Underground Parking Garages*, Vehicular Communications (under review).

All results are **simulation-based**: there is no measured data. The "data" of this package are the two garage configurations (in `garage_model.py`), the random seeds, and the result files in `results/`.

**Citation.** See `CITATION.cff`, or cite the archived version:
`[dataset] H.H. Hussein, Replication package for GDOP- and NLOS-aware anchor placement for UWB-TDoA localization in underground parking garages, Zenodo, version 1.0.0, 2026. https://doi.org/10.5281/zenodo.XXXXXXX`

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate        # Python 3.13 recommended
pip install -r requirements.txt
python smoke_test.py                                      # about 1 minute; must print ALL CHECKS PASSED
python run_all.py --outdir results --figures              # redraw Figures 2-8 from the shipped results (about 1 minute)
python run_all.py --list                                  # every step, the paper item it produces, its dependencies
python run_all.py --outdir my_run --jobs 4                # recompute everything from the seeds (see run times below)
```

Python 3.13, NumPy 2.5.3, SciPy 1.18.1, Shapely 2.2.0 and Matplotlib 3.11.2 were used. PSO, GA and SA runs depend on NumPy's random stream, so exact table values are guaranteed only for these versions (paper, Section 6.4); with other versions expect the same conclusions but possibly different last digits. `smoke_test.py` tells you which case you are in.

## Contents

| File | Purpose |
|---|---|
| `garage_model.py` | Parametric geometry of Garage 1 (50 x 30 m, reference) and Garage 2 (30 x 20 m, parameters from the field study cited as [7]); columns, parked cars, candidate anchors, evaluation points (Tables 1-2, Section 4) |
| `visibility.py` | Height-aware LOS/NLOS model: full-height columns block, 1.5 m cars block only when the ray is below car height |
| `gdop.py` | Reference-invariant (centroid) GDOP, vectorised (Eqs. 10-12) |
| `fitness.py` | Objective F(x) and constraints C1-C6 (Table 1) |
| `pso.py`, `ga.py` | Binary PSO and GA (Table 2 defaults, equal budget of 6,040 evaluations) |
| `baselines.py`, `sa_study.py` | Random search, coverage greedy, objective-aware greedy, simulated annealing (same budget); robust SA |
| `tdoa_solver.py` | TDoA solver (Chan-Ho start, bounded trust-region-reflective), distance-dependent noise, Monte Carlo trials |
| `rssi_solver.py` | Log-distance RSSI multilateration for the WiFi/BLE/UWB-RSSI benchmark |
| `run_all.py` | Runs every experiment in dependency order (`--jobs N` for parallel), logs run times and software versions |
| `make_figures.py`, `make_fig1_architecture.py` | Figures 1-8 |
| `smoke_test.py` | One-minute check of the environment against the shipped results |
| `results/` | Result files (JSON) of the runs reported in the paper, and `figures/` |

## From paper items to scripts

| Paper item | Script(s) | Result file(s) |
|---|---|---|
| Fig. 1 (architecture) | `make_fig1_architecture.py` | `figures/fig1_architecture.png` |
| Figs. 2-3 (garages, LOS/NLOS) | `make_figures.py` | `figures/fig2_*`, `fig3_*` |
| Tables 1-2 (constraints, parameters) | definitions in `garage_model.py`, `fitness.py`, `pso.py`, `ga.py` | - |
| Table 3 (reliability versus anchor count) | `compare_anchor_counts.py` | `anchor_count_sweep_enhanced.json` |
| Table 4 Panel A (angle penalty) | `angle_penalty_experiment.py` | `angle_penalty_validation.json` |
| Table 4 Panel B, Fig. 4 (old vs final solver) | `old_vs_realistic_solver.py` | `old_vs_realistic_comparison.json` |
| Table 4 Panel C and notes (CRB, Spearman, bootstrap CIs) | `validate_accuracy.py`, `crb_test.py`, `ci_and_sensitivity.py` | `accuracy_validation_both_garage{1,2}.json`, `crb_results.json`, `ci_sensitivity_results.json` |
| Section 5.2 (second-root failures) | `second_root.py` | `second_root_results.json` |
| Table 5, Fig. 5, Fig. 6 (PSO vs GA, heatmap, convergence) | `compare.py`, `extract_best_layouts.py`, `make_figures.py` | `comparison_report{,_g2}.json`, `comparison_curves{,_g2}.json`, `best_layouts{,_g2}.json` |
| Section 5.3 tuning check | `hyperparam_sweep.py` | `hyperparam_sweep_results.json`, `hyperparam_tuned_comparison.json` |
| Table 6 (methods and baselines) | `compare.py` (PSO, GA), `baselines.py` (random, greedy, SA), `occupancy_study.py` and `sa_study.py` (runs meeting C1-C2) | `baselines_g{1,2}.json`, `occupancy_study_g{1,2}.json`, `sa_study_g{1,2}.json` |
| Table 7 (30 unseen arrangements) | `occupancy_study.py`, `sa_study.py`, `best_heldout.py` | `occupancy_study_g{1,2}.json`, `sa_study_g{1,2}.json`, `best_heldout.json` |
| Table 8, Fig. 7 (technology benchmark) | `task_2_5_technology_comparison.py` (WiFi, BLE), `uwb_rssi_comparison.py` (UWB-RSSI), `validate_accuracy.py` (UWB-TDoA 13.5 cm) | `task_2_5_results.json`, `uwb_rssi_comparison_results.json` |
| Section 5.4 path-loss grid | `rssi_sensitivity.py` | `rssi_sensitivity_results.json` |
| Table 9, Fig. 8 (minimum anchors, P95 form, GA/PSO) | `accuracy_driven_design_p95.py` (Garage 1, 15 and 20 cm), `accuracy_driven_design_p95_10cm.py`, `check_10cm_m60.py` (Garage 1, 10 cm), `accuracy_driven_design_p95_garage2.py` | `accuracy_driven_design_p95_garage{1,2}.json` |
| Table 10 (minimum anchors with SA) | `accuracy_sa.py` | `accuracy_sa_g{1,2}.json` |
| Section 5.5 "Strict form" | `accuracy_driven_design.py`, `accuracy_driven_design_extend.py` (optional, M up to 60), `denser_grid_floor_test_v2.py` | `accuracy_driven_design_garage1.json`, `denser_grid_floor_test_v2.json` |
| Table 11, Section 5.6 (noise, NLOS bias, geometry) | `ci_and_sensitivity.py` | `ci_sensitivity_results.json` |
| Table 12 (literature comparison) | values taken from the cited papers | - |

## Seeds

| Use | Seeds |
|---|---|
| Parked-car arrangement (Garage 1, Garage 2) | 42, 100 |
| PSO, GA, SA runs (20 per method) | 1000-1019 |
| Held-out parked-car arrangements (30) / training arrangements for robust layouts (5) | 2001-2030 / 3001-3005 |
| Monte Carlo validation of best layouts | 123 |
| Old-vs-final solver pairing, anchor-count sweep | 999, 555 |
| Technology benchmark | 2024 |
| Angle-penalty re-optimisation, accuracy-driven sweeps | 2026, 3033, 4044 / 2026-2028 |

## Run times

Measured on one core of a 2-core cloud sandbox (Python 3.13, NumPy 2.5.3). Steps are independent apart from the dependencies listed by `python run_all.py --list`, so `--jobs N` divides the wall time roughly by N. Per-step logs and the environment record are in `results/logs/` and `results/run_info.json`.

| Step | Paper item | Time |
|---|---|---|
| `compare_g1` | Table 5, Garage 1 (PSO vs GA, 20 seeds) | 8 min |
| `compare_g2` | Table 5, Garage 2 (M = 8) | 2 min |
| `extract_g1` | best layouts, Garage 1 | 1 s |
| `extract_g2` | best layouts, Garage 2 | 1 s |
| `validate_g1` | Monte Carlo validation of best layouts, Garage 1 (Table 4 Panel C, slope a) | 50 s |
| `validate_g2` | Monte Carlo validation of best layouts, Garage 2 (Table 4 Panel C, slope a) | 15 s |
| `old_vs_realistic` | Table 4 Panel B, Figure 4 (old vs realistic solver) | 85 s |
| `anchor_counts` | Table 3 (reliability versus anchor count) | 33 s |
| `crb` | Table 4 Panel C (CRB and Monte Carlo) | 7 s |
| `second_root` | Section 5.2 (second-root failures) | 39 s |
| `angle_penalty` | Table 4 Panel A (angle penalty) | 4 min |
| `ci_sensitivity` | Table 4 notes (bootstrap CIs), Table 11, Section 5.6 | 2 min |
| `baselines` | Table 6 (random, greedy, SA baselines) | 7 min |
| `hyperparam` | Section 5.3 tuning check | 18 min |
| `occupancy` | Table 7 (single-arrangement GA/PSO layouts, robust GA) | 9 min |
| `sa_study` | Tables 6-7 (SA and robust SA layouts) | 13 min |
| `best_heldout` | Table 7 (best layout of each method, 30 held-out arrangements) | 49 s |
| `rssi_grid` | Section 5.4 (path-loss grid) | 12 min |
| `technology_wifi_ble` | Table 8 and Figure 7 (WiFi/BLE rows) | 2 min |
| `technology_uwb_rssi` | Table 8 and Figure 7 (UWB-RSSI rows) | 3 min |
| `accuracy_strict_g1` | Section 5.5, strict form, Garage 1 | 5 min |
| `accuracy_strict_g1_ext` | Section 5.5, strict form, Garage 1, extended grid (M up to 60) | 2 min |
| `accuracy_p95_g1` | Table 9, Garage 1, 15 and 20 cm | 3 min |
| `accuracy_p95_g1_10cm` | Table 9, Garage 1, 10 cm (GDOP criterion met at M = 48, validation below 95%: not achieved) | 2 min |
| `accuracy_p95_g1_10cm_m60` | Table 9, Garage 1, 10 cm at M = 56 and 60 | 2 min |
| `accuracy_p95_g2` | Table 9, Garage 2 | 86 s |
| `accuracy_sa` | Table 10 (SA, both garages) | 54 min |
| `denser_grid` | Section 5.5, strict form (357-candidate grid) | 62 s |
| **total** | | **2.6 h** single-core, roughly 1.3 h with `--jobs 2` |

## Notes and limitations

* Results are simulation results; the propagation parameters of the UWB-RSSI model are an engineering estimate (`uwb_rssi_comparison.py`), and the WiFi/BLE parameters are assumed values (paper, Section 5.4).
* A few reference constants are hard-coded and documented where they occur: the UWB-TDoA reference median of 13.5 cm in `uwb_rssi_comparison.py`, the unmodified-GA baseline printed by `angle_penalty_experiment.py`, the Table 5 reference values in `hyperparam_sweep.py`, the slopes a = 11.2 and 7.3 cm/GDOP in `accuracy_sa.py`, and the Garage 1 slope used by the failure-point flag of `sa_study.py` (not used for Garage 2 in the paper).
* `accuracy_driven_design_extend.py` and `accuracy_driven_design_p95_10cm.py` update the JSON file written by the script they extend; run them after it.
* Table 8: the dense WiFi median (7.66 m) comes from `uwb_rssi_comparison.py` and the dense BLE median (5.85 m) from `task_2_5_technology_comparison.py`; the two scripts run the same model with different random draws, so the other script gives 7.62 m (WiFi) and 5.86 m (BLE).
* Figures 2-8 are redrawn by `make_figures.py` from the result files; their appearance may differ slightly from the typeset figures of the article, their data do not.
* The shipped `results/` were produced by running the scripts through `run_all.py` with the software versions above. The long steps were interrupted and resumed (`--resume`) several times because of sandbox time limits; each step was nevertheless run to completion from its seeds, and fourteen of them (`compare_g1/g2`, `extract_g1/g2`, `validate_g1/g2`, `crb`, `second_root`, `ci_sensitivity`, `accuracy_strict_g1`, `accuracy_strict_g1_ext`, `rssi_grid`, `accuracy_p95_g2`) were also run a second time from scratch with identical result files. Run `--resume` to skip steps whose result files exist.

## License

MIT (see `LICENSE`).
