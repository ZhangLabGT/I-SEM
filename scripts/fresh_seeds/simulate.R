#!/usr/bin/env Rscript
# Loads the repo's simulate_authors.R unedited from disk; in memory changes only the seed loops (seed 1 = byte check,
# seeds 12-21 = confirmatory) and this_dir (derived from --file). Every setting, call and write_rep stays as released.
# usage: Rscript simulate.R <out_dir>
orig <- file.path(Sys.getenv("REPO_ROOT"), "scripts/experiments/e9_author_sims/simulate_authors.R")
SEEDS <- c(1L, 12:21)
src <- readLines(orig)
swap <- function(src, pat, new) {
  i <- which(src == pat); if (length(i) != 1L) stop("expected exactly one line: ", pat); src[i] <- new; src
}
src <- swap(src, "    for (iter in seq_len(N_ITER)) {", "    for (iter in SEEDS) {")
src <- swap(src, "  for (seed in seq_len(N_ITER)) {", "  for (seed in SEEDS) {")
i <- grep("^this_dir <- ", src); stopifnot(length(i) == 1L)
src[i] <- sprintf('this_dir <- "%s"', dirname(orig))
eval(parse(text = src), envir = globalenv())
