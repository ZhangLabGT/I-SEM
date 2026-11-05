#!/usr/bin/env bash
# Link every main simulation's inspre_dag_path.npz into res/, where the scorer reads it.
set -euo pipefail
R=$(cd "$(dirname "$0")/../.." && pwd); O=$R/scripts/results/config_tests; T=$R/scripts/sims_gpu/res
for f in $(cd "$O" && find sims -name inspre_dag_path.npz); do d=$T/$(dirname "$f"); mkdir -p "$d"; ln -sfn "$O/$f" "$d/inspre_dag_path.npz"; done
