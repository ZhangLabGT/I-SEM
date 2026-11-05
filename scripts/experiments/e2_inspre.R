#!/usr/bin/env Rscript
# E2 inspre exactly as the authors run it in their own cyclic simulations
# (inspre Rmd/simulate_*_cycles_*.Rmd at the pinned commit 7016697f):
#   fit_inspre_from_X(Y, targets, cv_folds = 5, verbose = 1, rho = 10, constraint = "UV", DAG = ...)
#   index <- which.min(eps_hat_G);  G_hat <- G path at that index
# DAG=FALSE is "inspre", DAG=TRUE is their "inspre_DAG".
#
# Upstream bug fix (only change to the authors' code): commit 840665b renamed the G path from
# res$R_hat to res$G_hat in fit_inspre_from_R but not in predict_inspre / predict_inspre_G_h5X,
# which still index res$R_hat[,,i] (now the 2-D input), so every cv_folds > 0 fit crashes with
# "incorrect number of dimensions". inspre_cv_fix() rewrites that one token back to G_hat.
#
# usage: e2_inspre.R <expr.bin float32 row-major, z-scored vs control> <n_rows>
#                    <targets.txt> <genes.txt> <out.bin> <DAG TRUE|FALSE> <seed>
# writes <out.bin>: |G| at the selected lambda, float32, column-major, in the I-SEM orientation
# W[i, j] = effect of j on i (inspre's G is exposure x outcome), zero for genes inspre's filter
# drops; <out.bin>.json: metadata.

inspre_cv_fix <- function() {
  for (fn in c("predict_inspre", "predict_inspre_G_h5X")) {
    f <- utils::getFromNamespace(fn, "inspre")
    src <- deparse(body(f))
    n_hit <- sum(grepl("res$R_hat[, , ", src, fixed = TRUE))
    if (n_hit == 0L) stop("inspre_cv_fix: pattern not found in ", fn, " (package changed?)")
    body(f) <- parse(text = gsub("res$R_hat[, , ", "res$G_hat[, , ", src, fixed = TRUE))[[1]]
    utils::assignInNamespace(fn, f, "inspre")
  }
}

if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) != 7L) stop("usage: e2_inspre.R <expr.bin> <n_rows> <targets> <genes> <out.bin> <DAG> <seed>")
  genes <- readLines(args[[4]], warn = FALSE)
  targets <- readLines(args[[3]], warn = FALSE)
  n_rows <- as.integer(args[[2]])
  X <- matrix(readBin(args[[1]], "numeric", n = n_rows * length(genes), size = 4L),
              nrow = n_rows, byrow = TRUE)
  colnames(X) <- genes
  dag <- as.logical(args[[6]])
  set.seed(as.integer(args[[7]]))

  suppressPackageStartupMessages({library(inspre); library(magrittr); library(purrr)})
  inspre_cv_fix()
  fit <- inspre::fit_inspre_from_X(X, targets, cv_folds = 5, verbose = 1, rho = 10,
                                   constraint = "UV", DAG = dag, ncores = 8)
  best <- which.min(fit$eps_hat_G)
  G <- fit$G_hat[, , best]
  kept <- rownames(G)
  if (is.null(kept) || !identical(kept, colnames(G))) stop("G_hat lacks matching dimnames")
  S <- matrix(0, length(genes), length(genes), dimnames = list(genes, genes))
  S[kept, kept] <- abs(t(G))
  diag(S) <- 0
  writeBin(as.numeric(S), args[[5]], size = 4L)
  jsonlite::write_json(list(DAG = dag, selected_lambda = fit$lambda[[best]], best_index = best,
                            lambda_path = as.numeric(fit$lambda), eps_hat_G = as.numeric(fit$eps_hat_G),
                            n_kept = length(kept), inspre_version = as.character(packageVersion("inspre")),
                            call = "fit_inspre_from_X(cv_folds=5, rho=10, constraint='UV'); which.min(eps_hat_G)",
                            patch = "predict_inspre*/res$R_hat[,,i] -> res$G_hat[,,i] (upstream 840665b rename)"),
                       paste0(args[[5]], ".json"), auto_unbox = TRUE)
}
