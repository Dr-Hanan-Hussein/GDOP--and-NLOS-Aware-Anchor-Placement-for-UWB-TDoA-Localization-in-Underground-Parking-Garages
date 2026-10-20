"""run_all.py - regenerate every result of the paper from fixed seeds, in dependency order.

Each step is an independent script that reads and writes JSON files in the output directory. Steps run in dependency order;
with --jobs N independent steps run in parallel. Console output of every step is kept in <outdir>/logs/<step>.log and a summary
(software versions, return codes, run times) in <outdir>/run_all_log.json.

Usage
  python run_all.py --list                      print the steps, the paper items they produce, and their dependencies
  python run_all.py                             run everything (hours; use --jobs 2 or more if you have the cores)
  python run_all.py compare_g1 extract_g1       run selected steps (by id or by number from --list)
  python run_all.py --outdir my_run --jobs 4    choose the output directory (default: outputs) and the number of parallel steps
  python run_all.py --resume                    skip steps whose declared output files already exist
  python run_all.py --figures                   only redraw the figures from the result files in --outdir

Precomputed results are shipped in results/; run with  --outdir results --figures  to redraw the figures without recomputing anything.
"""
import argparse, json, os, platform, shlex, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# id, paper item, script and arguments, steps that must finish first, files the step writes
S = lambda sid, paper, cmd, after=(), makes=(): dict(id=sid, paper=paper, cmd=cmd, after=list(after), makes=list(makes))
STEPS = [
    S("compare_g1", "Table 5, Garage 1 (PSO vs GA, 20 seeds)", ["compare.py", "20", "1000", "", "1"],
      makes=["comparison_report.json", "comparison_curves.json"]),
    S("compare_g2", "Table 5, Garage 2 (M = 8)", ["compare.py", "20", "1000", "_g2", "2", "8"],
      makes=["comparison_report_g2.json", "comparison_curves_g2.json"]),
    S("extract_g1", "best layouts, Garage 1", ["extract_best_layouts.py", "comparison_report.json", "comparison_curves.json", "best_layouts.json"],
      ["compare_g1"], ["best_layouts.json"]),
    S("extract_g2", "best layouts, Garage 2", ["extract_best_layouts.py", "comparison_report_g2.json", "comparison_curves_g2.json", "best_layouts_g2.json"],
      ["compare_g2"], ["best_layouts_g2.json"]),
    S("validate_g1", "Monte Carlo validation of best layouts, Garage 1 (Table 4 Panel C, slope a)", ["validate_accuracy.py", "both", "1", "best_layouts.json"],
      ["extract_g1"], ["accuracy_validation_both_garage1.json"]),
    S("validate_g2", "Monte Carlo validation of best layouts, Garage 2 (Table 4 Panel C, slope a)", ["validate_accuracy.py", "both", "2", "best_layouts_g2.json"],
      ["extract_g2"], ["accuracy_validation_both_garage2.json"]),
    S("old_vs_realistic", "Table 4 Panel B, Figure 4 (old vs realistic solver)", ["old_vs_realistic_solver.py"],
      ["extract_g1"], ["old_vs_realistic_comparison.json"]),
    S("anchor_counts", "Table 3 (reliability versus anchor count)", ["compare_anchor_counts.py"], [], ["anchor_count_sweep_enhanced.json"]),
    S("crb", "Table 4 Panel C (CRB and Monte Carlo)", ["crb_test.py"], ["validate_g1", "validate_g2"], ["crb_results.json"]),
    S("second_root", "Section 5.2 (second-root failures)", ["second_root.py"], ["validate_g1"], ["second_root_results.json"]),
    S("angle_penalty", "Table 4 Panel A (angle penalty)", ["angle_penalty_experiment.py"], ["validate_g1"], ["angle_penalty_validation.json"]),
    S("ci_sensitivity", "Table 4 notes (bootstrap CIs), Table 11, Section 5.6", ["ci_and_sensitivity.py", "1", "2"],
      ["validate_g1", "validate_g2"], ["ci_sensitivity_results.json"]),
    S("baselines", "Table 6 (random, greedy, SA baselines)", ["baselines.py", "1", "2"],
      ["compare_g1", "compare_g2"], ["baselines_g1.json", "baselines_g2.json"]),
    S("hyperparam", "Section 5.3 tuning check", ["hyperparam_sweep.py"], [], ["hyperparam_sweep_results.json", "hyperparam_tuned_comparison.json"]),
    S("occupancy", "Table 7 (single-arrangement GA/PSO layouts, robust GA)", ["occupancy_study.py", "1", "2"],
      ["compare_g1", "compare_g2"], ["occupancy_study_g1.json", "occupancy_study_g2.json"]),
    S("sa_study", "Tables 6-7 (SA and robust SA layouts)", ["sa_study.py", "1", "2"], [], ["sa_study_g1.json", "sa_study_g2.json"]),
    S("best_heldout", "Table 7 (best layout of each method, 30 held-out arrangements)", ["best_heldout.py"],
      ["extract_g1", "extract_g2", "sa_study"], ["best_heldout.json"]),
    S("rssi_grid", "Section 5.4 (path-loss grid)", ["rssi_sensitivity.py"], ["extract_g1"], ["rssi_sensitivity_results.json"]),
    S("technology_wifi_ble", "Table 8 and Figure 7 (WiFi/BLE rows)", ["task_2_5_technology_comparison.py"], ["extract_g1"], ["task_2_5_results.json"]),
    S("technology_uwb_rssi", "Table 8 and Figure 7 (UWB-RSSI rows)", ["uwb_rssi_comparison.py"], ["extract_g1"], ["uwb_rssi_comparison_results.json"]),
    S("accuracy_strict_g1", "Section 5.5, strict form, Garage 1", ["accuracy_driven_design.py"], ["validate_g1"], ["accuracy_driven_design_garage1.json"]),
    S("accuracy_strict_g1_ext", "Section 5.5, strict form, Garage 1, extended grid (M up to 60)", ["accuracy_driven_design_extend.py"],
      ["accuracy_strict_g1"], []),  # modifies accuracy_driven_design_garage1.json in place
    S("accuracy_p95_g1", "Table 9, Garage 1, 15 and 20 cm", ["accuracy_driven_design_p95.py"], ["validate_g1"], ["accuracy_driven_design_p95_garage1.json"]),
    S("accuracy_p95_g1_10cm", "Table 9, Garage 1, 10 cm (GDOP criterion met at M = 48, validation below 95%: not achieved)", ["accuracy_driven_design_p95_10cm.py"], ["accuracy_p95_g1"], []),  # modifies accuracy_driven_design_p95_garage1.json in place
    S("accuracy_p95_g1_10cm_m60", "Table 9, Garage 1, 10 cm at M = 56 and 60", ["check_10cm_m60.py"], ["validate_g1"], []),
    S("accuracy_p95_g2", "Table 9, Garage 2", ["accuracy_driven_design_p95_garage2.py"], ["validate_g2"], ["accuracy_driven_design_p95_garage2.json"]),
    S("accuracy_sa", "Table 10 (SA, both garages)", ["accuracy_sa.py", "1", "2"], [], ["accuracy_sa_g1.json", "accuracy_sa_g2.json"]),
    S("denser_grid", "Section 5.5, strict form (357-candidate grid)", ["denser_grid_floor_test_v2.py"], [], ["denser_grid_floor_test_v2.json"]),
    S("figures", "Figures 2-8", ["make_figures.py"],
      ["extract_g1", "old_vs_realistic", "technology_wifi_ble", "technology_uwb_rssi", "accuracy_p95_g1_10cm", "accuracy_p95_g2", "validate_g1"], []),
]
BY_ID = {s["id"]: s for s in STEPS}


def env_info():
    import numpy, scipy, shapely
    try: import matplotlib; mpl = matplotlib.__version__
    except ImportError: mpl = None
    return dict(python=platform.python_version(), numpy=numpy.__version__, scipy=scipy.__version__, shapely=shapely.__version__,
                matplotlib=mpl, platform=platform.platform(), cpu_count=os.cpu_count())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="*", help="step ids or numbers (default: all)")
    ap.add_argument("--list", action="store_true"); ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--jobs", type=int, default=1); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--figures", action="store_true", help="only run the figure step")
    a = ap.parse_args()
    if a.list:
        for i, s in enumerate(STEPS, 1):
            print(f"{i:2d}. {s['id']:26s} {s['paper']}\n      {shlex.join(s['cmd']):70s} after: {', '.join(s['after']) or '-'}")
        return 0
    sel = ["figures"] if a.figures else [(STEPS[int(x) - 1]["id"] if x.isdigit() else x) for x in a.steps] or [s["id"] for s in STEPS]
    for x in sel:
        if x not in BY_ID: sys.exit(f"unknown step: {x}")
    out = Path(a.outdir).resolve(); (out / "logs").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if a.jobs > 1: env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    log = {"environment": env_info(), "steps": {}}
    done, running, failed = set(), {}, set()
    pending = [x for x in sel if not (a.resume and STEPS_done(BY_ID[x], out))]
    for x in sel:
        if x not in pending: done.add(x); log["steps"][x] = "skipped (outputs exist)"
    def ready(x):
        return all((d in done) or (d not in sel) for d in BY_ID[x]["after"])  # prerequisites outside the selection are assumed to exist already
    t_start = time.time()
    while pending or running:
        for x in [p for p in pending if ready(p)]:
            if len(running) >= a.jobs: break
            s = BY_ID[x]; pending.remove(x); lf = open(out / "logs" / f"{x}.log", "w")
            print(f"[start] {x}: {s['paper']}", flush=True)
            proc = subprocess.Popen([sys.executable, "-u", str(HERE / s["cmd"][0])] + s["cmd"][1:], cwd=out, stdout=lf, stderr=subprocess.STDOUT, env=env)
            running[x] = (proc, lf, time.time())
        for x, (proc, lf, t0) in list(running.items()):
            rc = proc.poll()
            if rc is None: continue
            lf.close(); del running[x]; secs = round(time.time() - t0)
            log["steps"][x] = dict(cmd=BY_ID[x]["cmd"], returncode=rc, seconds=secs); (done if rc == 0 else failed).add(x)
            print(f"[{'done' if rc == 0 else 'FAILED'}] {x} ({secs}s)", flush=True)
            json.dump(log, open(out / "run_all_log.json", "w"), indent=2)
        if not running and pending and not any(ready(p) for p in pending):
            for p in pending: log["steps"][p] = "blocked (a prerequisite failed or its output is missing)"; print(f"[blocked] {p}")
            break
        time.sleep(0.5)
    log["total_seconds"] = round(time.time() - t_start); json.dump(log, open(out / "run_all_log.json", "w"), indent=2)
    return 1 if failed or any(isinstance(v, str) and v.startswith("blocked") for v in log["steps"].values()) else 0


def STEPS_done(s, out):
    return bool(s["makes"]) and all((out / f).exists() for f in s["makes"])


if __name__ == "__main__":
    sys.exit(main())
