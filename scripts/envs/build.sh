#!/usr/bin/env bash
# Build the repo-local environments into .envs/<name> from the pinned specs in scripts/envs/.
#   bash scripts/envs/build.sh isem inspre golem lingam igsp rice (default: all six)
# isem    Python: I-SEM, NOTEARS, experiments, tests          (isem.requirements.txt)
# inspre  R: inspre @ scripts/baselines/inspre/upstream + ADAPRE deps  (inspre.lock.txt = the env of the stored runs)
# golem   Python + TensorFlow 2.15 (GPU): GOLEM                (golem.requirements.txt)
# lingam  R + pcalg: LiNGAM                                    (lingam.lock.txt)
# igsp    Python: causaldag 0.1a163 (IGSP)                      (igsp.requirements.txt)
# rice    Python + CuPy 14.2 (CUDA 12.8 libs, GPU): RICE        (rice.requirements.txt)
# Every cache and temp file stays under .envs/; R envs are isolated from any user R library.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD
export CONDA_PKGS_DIRS=$ROOT/.envs/.cache/conda PIP_CACHE_DIR=$ROOT/.envs/.cache/pip TMPDIR=$ROOT/.envs/.cache/tmp
export R_LIBS_USER="" R_LIBS="" R_LIBS_SITE="" R_PROFILE_USER=/dev/null R_ENVIRON_USER=/dev/null
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR"

# The authors' code at the commits used in the paper.
clone() { [ -d "scripts/baselines/$1/upstream" ] || git clone -q "https://github.com/$2" "scripts/baselines/$1/upstream"; git -C "scripts/baselines/$1/upstream" switch -q --detach "$3"; }
clone inspre brielin/inspre 7016697
clone notears xunzheng/notears 4a9ab19
clone golem ignavierng/golem 9e99e61

for name in "${@:-isem inspre golem lingam igsp rice}"; do
  for n in $name; do
    export E=$ROOT/.envs/$n
    echo "=== $n -> $E"
    case $n in
      isem)
        [ -x "$E/bin/python" ] || mamba create -y -p "$E" -c conda-forge python=3.10.13 pip
        "$E/bin/pip" install -r requirements.txt
        "$E/bin/python" -c "import torch, triton, numpy; print('isem ok: torch', torch.__version__, 'cuda', torch.cuda.is_available())" ;;
      inspre)
        [ -x "$E/bin/Rscript" ] || mamba create -y -p "$E" --file scripts/envs/inspre.lock.txt
        PATH="$E/bin:$PATH" "$E/bin/R" CMD INSTALL -l "$E/lib/R/library" scripts/baselines/inspre/upstream   # env compilers
        "$E/bin/Rscript" -e 'stopifnot(startsWith(find.package("inspre"), Sys.getenv("E"))); library(inspre); library(caret); library(hdf5r); cat("inspre ok:", as.character(packageVersion("inspre")), "\n")'
        # The simulators need mc2d 0.1-22 (scripts/experiments/e9_author_sims/simulate_authors.R).
        mkdir -p .envs/Rlib
        PATH="$E/bin:$PATH" "$E/bin/Rscript" -e '.libPaths(c(".envs/Rlib", .libPaths())); if (!requireNamespace("mc2d", quietly = TRUE)) { install.packages("mvtnorm", lib = ".envs/Rlib", repos = "https://cloud.r-project.org"); install.packages("https://cran.r-project.org/src/contrib/Archive/mc2d/mc2d_0.1-22.tar.gz", lib = ".envs/Rlib", repos = NULL, type = "source") }; stopifnot(packageVersion("mc2d") == "0.1.22")' ;;
      golem)
        [ -x "$E/bin/python" ] || mamba create -y -p "$E" -c conda-forge python=3.10.22 pip
        "$E/bin/pip" install -r scripts/envs/golem.requirements.txt
        "$E/bin/python" -c "import tensorflow as tf, igraph; print('golem ok: tf', tf.__version__)" ;;
      lingam)
        [ -x "$E/bin/Rscript" ] || mamba create -y -p "$E" --file scripts/envs/lingam.lock.txt
        "$E/bin/Rscript" -e 'library(pcalg); cat("lingam ok: pcalg", as.character(packageVersion("pcalg")), "\n")' ;;
      igsp)
        [ -x "$E/bin/python" ] || mamba create -y -p "$E" -c conda-forge python=3.10.22 pip
        PYTHONNOUSERSITE=1 "$E/bin/pip" install -r scripts/envs/igsp.requirements.txt
        PYTHONNOUSERSITE=1 "$E/bin/python" -s -W ignore -c "from causaldag import igsp; import causaldag, inspect; print('igsp ok:', igsp.__module__, inspect.signature(igsp))" ;;
      rice)
        [ -x "$E/bin/python" ] || mamba create -y -p "$E" -c conda-forge python=3.10.22 pip
        PYTHONNOUSERSITE=1 "$E/bin/pip" install -r scripts/envs/rice.requirements.txt
        PYTHONNOUSERSITE=1 CUPY_CACHE_DIR=$ROOT/.envs/.cache/cupy "$E/bin/python" -s -c "import cupy, cupyx.scipy.special, igraph, castle.metrics, anndata; print('rice ok: cupy', cupy.__version__, 'eigvals', hasattr(cupy.linalg, 'eigvals'), 'anndata', anndata.__version__)" ;;
      *) echo "unknown env: $n" >&2; exit 2 ;;
    esac
  done
done
