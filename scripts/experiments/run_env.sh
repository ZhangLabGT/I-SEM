#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONDONTWRITEBYTECODE=1
export PATH="$EXPERIMENT_ROOT/.envs/isem/bin:$PATH"   # repo-local env (scripts/envs/build.sh isem)
export XDG_CACHE_HOME="$EXPERIMENT_ROOT/.envs/.cache"
export TMPDIR="$EXPERIMENT_ROOT/.envs/.cache/tmp"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export CONDA_PKGS_DIRS="$XDG_CACHE_HOME/conda"
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$CONDA_PKGS_DIRS"
cd "$EXPERIMENT_ROOT"
exec "$@"
