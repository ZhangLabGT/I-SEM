#!/usr/bin/env bash
# Stands in for Rscript in e8.fit / run_sims.inspre_path; R's compilers on PATH.
F=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); R=$(cd "$F/../.." && pwd)/.envs/inspre/bin
export PATH="$R:$PATH" DIVISOR_HERE="$F"
exec "$R/Rscript" "$F/shim.R" "$@"
