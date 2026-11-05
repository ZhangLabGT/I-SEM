#!/usr/bin/env bash
# Times the shipped I-SEM on all four datasets, each on an idle GPU.
# Waits until nvidia-smi lists no compute process for 120 s, then fits while holding ../gpu.lock (every other GPU
# job takes the same lock, so none can start mid-timing). A fit that saw another process
# before or after is kept as {ds}.shared.N.json and retried, up to 5 tries per dataset.
# Run from the repo root: nohup bash scripts/timing/run_idle.sh > .../timing/run_idle.log 2>&1 &
set -u
T=scripts/timing
LOCK=scripts/gpu.lock
idle() { [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
for ds in hepg2 rpe1 jurkat k562; do
  for try in 1 2 3 4 5; do
    quiet=0
    until [ $quiet -ge 4 ]; do if idle; then quiet=$((quiet + 1)); else quiet=0; fi; sleep 30; done
    echo "$(date -Is) start $ds try $try"
    flock "$LOCK" bash scripts/experiments/run_env.sh python -u "$T/fit_idle.py" "$ds" > "$T/$ds.log" 2>&1
    echo "$(date -Is) end $ds exit=$?"
    if python3 -c "import json,sys; sys.exit(0 if json.load(open('$T/$ds.json'))['idle'] else 1)" 2>/dev/null; then break; fi
    [ -f "$T/$ds.json" ] && mv "$T/$ds.json" "$T/$ds.shared.$try.json"
  done
done
echo "$(date -Is) ALL DONE"
