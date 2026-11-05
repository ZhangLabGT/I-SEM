#!/usr/bin/env bash
# Every temp file and cache stays in this folder.
set -euo pipefail
F=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); export REPO_ROOT=$(cd "$F/../.." && pwd)
export PYTHONDONTWRITEBYTECODE=1 PATH="$REPO_ROOT/.envs/isem/bin:$PATH" TMPDIR="$F/tmp" XDG_CACHE_HOME="$F/tmp/xdg"
export TORCHINDUCTOR_CACHE_DIR="$F/tmp/xdg/ti" TRITON_CACHE_DIR="$F/tmp/xdg/tr" MPLCONFIGDIR="$F/tmp/xdg/mpl" NUMBA_CACHE_DIR="$F/tmp/xdg/nb"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
mkdir -p "$F/tmp/xdg" "$F/logs" "$F/results" "$F/cache"
cd "$F"
exec python -u div.py "$@"
