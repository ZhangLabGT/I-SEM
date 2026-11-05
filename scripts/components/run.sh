#!/usr/bin/env bash
# Repo-local envs; every temp file and cache stays in this folder.
set -euo pipefail
F=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); export REPO_ROOT=$(cd "$F/../.." && pwd)
export PYTHONDONTWRITEBYTECODE=1 PATH="$REPO_ROOT/.envs/isem/bin:$PATH" TMPDIR="$F/tmp" XDG_CACHE_HOME="$F/tmp/xdg"
export TORCHINDUCTOR_CACHE_DIR="$F/tmp/xdg/ti" TRITON_CACHE_DIR="$F/tmp/xdg/tr" MPLCONFIGDIR="$F/tmp/xdg/mpl" NUMBA_CACHE_DIR="$F/tmp/xdg/nb"
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2} OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
mkdir -p "$F/tmp/xdg/ti" "$F/tmp/xdg/tr" "$F/tmp/xdg/mpl" "$F/tmp/xdg/nb" "$F/logs" "$F/results"
cd "$F"
exec python -u "$@"
