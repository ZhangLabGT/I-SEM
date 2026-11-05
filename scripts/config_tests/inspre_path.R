#!/usr/bin/env Rscript
# E9's inspre_DAG fit (scripts/experiments/e2_inspre.R, sourced unchanged for its inspre_cv_fix), the same call in the
# same order and seed, saving the WHOLE lambda path instead of only which.min(eps_hat_G). The E9 cache keeps only the
# selected G, and config C1 compares along the path. run_sims.py checks that the path's which.min slice reproduces
# E9's cached inspre_dag.npz.
#
# usage: inspre_path.R <expr.bin float32 row-major> <n_rows> <targets.txt> <genes.txt> <out prefix> <seed> <e2_inspre.R>
# writes <out>.Gpath.bin (float32, K x K x L column-major, inspre's G = exposure x outcome, NOT transposed) and
# <out>.json (kept genes, lambda, eps_hat_G, which.min index).

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 7L) stop("usage: inspre_path.R <expr.bin> <n_rows> <targets> <genes> <out> <seed> <e2_inspre.R>")
source(args[[7]])            # defines inspre_cv_fix(); its main block runs only as a script
genes <- readLines(args[[4]], warn = FALSE)
targets <- readLines(args[[3]], warn = FALSE)
n_rows <- as.integer(args[[2]])
X <- matrix(readBin(args[[1]], "numeric", n = n_rows * length(genes), size = 4L), nrow = n_rows, byrow = TRUE)
colnames(X) <- genes
set.seed(as.integer(args[[6]]))

suppressPackageStartupMessages({library(inspre); library(magrittr); library(purrr)})
inspre_cv_fix()
fit <- inspre::fit_inspre_from_X(X, targets, cv_folds = 5, verbose = 1, rho = 10,
                                 constraint = "UV", DAG = TRUE, ncores = 8)
kept <- rownames(fit$G_hat[, , 1])
writeBin(as.numeric(fit$G_hat), paste0(args[[5]], ".Gpath.bin"), size = 4L)
jsonlite::write_json(list(kept_genes = kept, K = length(kept), L = dim(fit$G_hat)[3],
                          lambda = as.numeric(fit$lambda), eps_hat_G = as.numeric(fit$eps_hat_G),
                          best_index = which.min(fit$eps_hat_G), inspre_version = as.character(packageVersion("inspre"))),
                     paste0(args[[5]], ".json"), auto_unbox = TRUE, digits = NA)
