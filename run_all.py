"""Run the whole pipeline in order, stopping at the first failure.

    python run_all.py                 # everything (download skipped if files exist)
    python run_all.py --from 01_clustering   # resume from a step
    DATA_DIR=data_synth REPORT_DIR=report_synth python run_all.py --skip-download   # smoke test

Each step writes a log to report/logs/<step>.log.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    "data_prep/00_download.py",
    "data_prep/01_build_features.py",
    "data_prep/02_build_sequences.py",
    "data_prep/03_build_leakage_audit.py",
    "analysis/01_eda.py",
    "mining/01_clustering.py",
    "mining/02_association_rules.py",
    "mining/03_classification.py",
    "mining/04_bonus.py",
    "mining/05_sequential.py",
    "mining/06_validation.py",
    "mining/07_stability.py",
    "mining/08_business_value.py",
    "mining/09_model_improvement.py",
    "mining/10_metrics_summary.py",
    "mining/11_clustering_variants.py",
    "mining/12_ordering_sensitivity.py",
    "mining/13_drift_ablation.py",
    "report/make_diagrams.py",
    "report/build_report.py",
]


def main():
    args = sys.argv[1:]
    steps = STEPS[1:] if "--skip-download" in args else STEPS
    if "--from" in args:
        name = args[args.index("--from") + 1]
        steps = steps[next(i for i, s in enumerate(steps) if name in s):]
    logs = Path(os.environ.get("REPORT_DIR", ROOT / "report")) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    for s in steps:
        t = time.time()
        print(f"==> {s}", flush=True)
        log = logs / (Path(s).stem + ".log")
        with open(log, "w", encoding="utf-8") as f:
            r = subprocess.run([sys.executable, str(ROOT / s)], cwd=ROOT, env=env, stdout=f,
                               stderr=subprocess.STDOUT)
        print(f"    {'ok' if r.returncode == 0 else 'FAILED'} in {time.time() - t:.0f}s (log: {log})")
        if r.returncode:
            print(log.read_text(encoding="utf-8")[-3000:])
            sys.exit(r.returncode)


if __name__ == "__main__":
    main()
