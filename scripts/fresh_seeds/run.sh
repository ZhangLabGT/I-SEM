#!/usr/bin/env bash
# Repo-local envs; every temp file and cache stays in this folder (run_env.sh would write under .envs/).
set -euo pipefail
F=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); export REPO_ROOT=$(cd "$F/../.." && pwd)
export PYTHONDONTWRITEBYTECODE=1 PATH="$REPO_ROOT/.envs/isem/bin:$PATH" TMPDIR="$F/tmp" XDG_CACHE_HOME="$F/tmp/xdg"
export TORCHINDUCTOR_CACHE_DIR="$F/tmp/xdg/ti" TRITON_CACHE_DIR="$F/tmp/xdg/tr" MPLCONFIGDIR="$F/tmp/xdg/mpl" NUMBA_CACHE_DIR="$F/tmp/xdg/nb"
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4} OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
mkdir -p "$F/tmp/xdg/ti" "$F/tmp/xdg/tr" "$F/tmp/xdg/mpl" "$F/tmp/xdg/nb" "$F/logs" "$F/results"
cd "$F"
rscript() { local R="$REPO_ROOT/.envs/inspre/bin"; env -u R_HOME PATH="$R:$PATH" R_LIBS="$REPO_ROOT/.envs/Rlib" R_LIBS_USER= R_LIBS_SITE= \
  R_PROFILE_USER=/dev/null R_ENVIRON_USER=/dev/null OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$R/Rscript" "$@"; }
case "${1:-}" in
  simulate) rscript simulate.R "$F/data" > logs/simulate.log 2>&1; WEAK_MODE=SWEEP rscript sweep.R "$F" > logs/sweep_data.log 2>&1 ;;
  ratio)    WEAK_MODE=RATIO rscript sweep.R "$F" > logs/ratio.log 2>&1; tail -1 logs/ratio.log ;;
  analyze)  python -u analyze.py ;;
  *)        python -u run.py "$@" ;;
esac
