#!/usr/bin/env Rscript
# E8: one ADAPRE / inspre fit on a cells x genes matrix, writing the whole lambda path (selection is done
# in e8_adapre.py, so every rule is computed from the same fit).
#
# methods
#   adapre          ADAPRE as released (keleslab/DEMO_RECOMB_2026 @ SHA in meta): source R/adapre_inspre_core.R
#                   + sourceCpp(src/adapre_inspre_cpp.cpp), then the demo's call
#                   fit_inspre_from_X(X, targets, weighted = TRUE, constraint = "UV", adaptive_lambda = TRUE)
#   adapre_noadapt  the same released code with adaptive_lambda = FALSE (isolates the adaptive penalty)
#   inspre_default  inspre::fit_inspre_from_X(X, targets, weighted = TRUE) at its package defaults (pinned
#                   7016697f), the demo's baseline call; inspre_cv_fix() from e2_inspre.R as in our other runs
# All three also get cv_folds (for the D_hat / eps_hat_G selection rules), ncores, its, solve_its.
#
# Pre-filter (only adaptation to the call): filter_tce() is run to a fixed point on the full data before
# the fit and X is restricted to the surviving targets. R_hat[t, g] and SE[t, g] depend only on the
# (t, g) pair, so the fit is the one the authors' code would do; without it (1) the C++ silently drops
# the adaptive penalty when length(beta_obs) != nrow(R_hat) (adapre_inspre_cpp.cpp: `beta_vec.size() == D`)
# and (2) cv_folds > 0 fails (xi_mat is sized by ncol(X) before filtering, upstream and fork alike).
#
# usage: adapre_fit.R <expr.bin float32 row-major, z-scored vs control> <n_rows> <targets.txt> <genes.txt>
#                     <out_prefix> <method> <seed> <ncores> <its> <solve_its> <cv_folds>
# writes <out_prefix>.Gpath.bin  float32 column-major [K, K, L], W orientation W[i, j] = effect of j on i
#                                (= t(G), inspre/ADAPRE G is exposure x outcome), signed, diagonal zeroed;
#        <out_prefix>.meta.json  kept genes, lambda, train_error, D_hat, eps_hat_G/I, beta_obs, timings.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 11L) stop("usage: adapre_fit.R <expr.bin> <n_rows> <targets> <genes> <out_prefix> <method> <seed> <ncores> <its> <solve_its> <cv_folds>")
method <- args[[6]]
seed <- as.integer(args[[7]]); ncores <- as.integer(args[[8]])
its <- as.integer(args[[9]]); solve_its <- as.integer(args[[10]]); cv_folds <- as.integer(args[[11]])
if (!method %in% c("adapre", "adapre_noadapt", "inspre_default")) stop("unknown method ", method)

this_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])))
study <- normalizePath(file.path(this_dir, "..", ".."))
src_dir <- file.path(study, "baselines", "adapre", "upstream")

suppressPackageStartupMessages({library(dplyr); library(purrr); library(magrittr)})
pkg_dir <- normalizePath(find.package("inspre"))
if (!grepl("/.envs/inspre/", pkg_dir, fixed = TRUE)) stop("inspre not from .envs/inspre: ", pkg_dir)

genes <- readLines(args[[4]], warn = FALSE)
targets <- readLines(args[[3]], warn = FALSE)
n_rows <- as.integer(args[[2]])
X <- matrix(readBin(args[[1]], "numeric", n = n_rows * length(genes), size = 4L), nrow = n_rows, byrow = TRUE)
colnames(X) <- genes
if (length(targets) != n_rows) stop("targets/rows mismatch")

# Both code bases share multiple_iv_reg/filter_tce verbatim (fork lines 46-77, 595-616); use the fork's copy
# for the pre-filter in every method so the gene set is decided by one implementation.
core <- new.env()
sys.source(file.path(src_dir, "R", "adapre_inspre_core.R"), envir = core)
keep <- intersect(targets, genes)
repeat {
  obs <- targets %in% c(keep, "control")
  Xo <- X[obs, keep, drop = FALSE]; to <- targets[obs]   # subset once, not per target (2 GB copies on K562)
  ie <- purrr::map(keep, ~ core$multiple_iv_reg(.x, Xo, to)) %>% purrr::list_rbind()
  R <- data.matrix(dplyr::select(ie, ends_with("beta_hat"))); SE <- data.matrix(dplyr::select(ie, ends_with("se_hat")))
  colnames(R) <- colnames(SE) <- keep; rownames(R) <- rownames(SE) <- ie$target
  f <- core$filter_tce(R, SE)
  if (!identical(rownames(f$R), colnames(f$R))) stop("filter_tce kept different rows and columns")
  if (length(rownames(f$R)) == length(keep)) break
  keep <- keep[keep %in% rownames(f$R)]
}
n_prefilter_dropped <- length(intersect(targets, genes)) - length(keep)
obs <- targets %in% c(keep, "control")
X <- X[obs, keep, drop = FALSE]; targets <- targets[obs]
beta_obs <- setNames(ie$beta_obs, ie$target)[keep]
cat(sprintf("[e8] method=%s kept %d genes (pre-filter dropped %d), %d cells, its=%d solve_its=%d cv=%d ncores=%d\n",
            method, length(keep), n_prefilter_dropped, nrow(X), its, solve_its, cv_folds, ncores))

set.seed(seed)
t0 <- Sys.time()
if (method == "inspre_default") {
  suppressPackageStartupMessages(library(inspre))
  source(file.path(study, "experiments", "e2_inspre.R"))   # defines inspre_cv_fix() only (main guarded)
  inspre_cv_fix()
  fit <- inspre::fit_inspre_from_X(X = X, targets = targets, weighted = TRUE, verbose = 1, cv_folds = cv_folds,
                                   its = its, solve_its = solve_its, ncores = ncores)
} else {
  # Order as in run_adapre_demo.R: source the core, then sourceCpp (defines fit_U_UV_const etc. globally).
  source(file.path(src_dir, "R", "adapre_inspre_core.R"))
  cache <- Sys.getenv("E8_RCPP_CACHE", tempdir())
  Rcpp::sourceCpp(file.path(src_dir, "src", "adapre_inspre_cpp.cpp"), cacheDir = cache)
  if (length(beta_obs) != ncol(X)) stop("beta_obs length != D: the adaptive penalty would be silently skipped")
  fit <- fit_inspre_from_X(X = X, targets = targets, weighted = TRUE, verbose = 1, cv_folds = cv_folds,
                           constraint = "UV", adaptive_lambda = (method == "adapre"),
                           its = its, solve_its = solve_its, ncores = ncores)
}
fit_seconds <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

G <- fit$G_hat
if (length(dim(G)) != 3L) stop("G_hat is not a lambda path")
kept <- rownames(G)
if (is.null(kept) || !identical(kept, colnames(G)) || !identical(kept, keep)) stop("G_hat dimnames do not match the pre-filtered genes")
W_path <- aperm(G, c(2, 1, 3))
for (l in seq_len(dim(W_path)[3])) diag(W_path[, , l]) <- 0
# G = I - V / diag(V): the solver's lasso penalizes diag(V) too, so a zeroed V_pp makes row p of G non-finite.
# Recorded per lambda (not fixed here); e8_adapre.py decides how a selected lambda with such genes is scored.
zero_diag_V <- lapply(seq_len(dim(fit$V)[3]), function(l) kept[abs(diag(fit$V[, , l])) < 1e-12])
nonfinite_per_lambda <- apply(W_path, 3, function(M) sum(!is.finite(M)))
writeBin(as.numeric(W_path), paste0(args[[5]], ".Gpath.bin"), size = 4L, endian = "little")

num <- function(x) if (is.null(x)) NULL else as.numeric(x)
sha <- tryCatch(system2("git", c("-C", src_dir, "rev-parse", "HEAD"), stdout = TRUE), error = function(e) "unknown")
jsonlite::write_json(list(
  method = method, kept_genes = kept, K = length(kept), L = dim(W_path)[3],
  n_prefilter_dropped = n_prefilter_dropped, n_cells = nrow(X),
  lambda = num(fit$lambda), train_error = num(fit$train_error), D_hat = num(fit$D_hat),
  eps_hat_G = num(fit$eps_hat_G), eps_hat_I = num(fit$eps_hat_I), rho_used = num(fit$rho),
  beta_obs = as.numeric(beta_obs),
  nonfinite_per_lambda = as.integer(nonfinite_per_lambda), zero_diag_V_genes_per_lambda = zero_diag_V,
  settings = list(weighted = TRUE, constraint = "UV", adaptive_lambda = (method == "adapre"), cv_folds = cv_folds,
                  its = its, solve_its = solve_its, ncores = ncores, seed = seed,
                  defaults_kept = "rho=100, nlambda=20, lambda_min_ratio=0.01, mu=10, tau=1.5, min_nz=0.01, DAG=FALSE, filter=TRUE, max_med_ratio=NULL, warm_start=FALSE"),
  adapre_src_sha = sha, inspre_version = as.character(utils::packageVersion("inspre")),
  inspre_sha = "7016697f87db64d7a1ac9bc384dd048298470425",
  R_version = R.version.string,
  pkg_versions = lapply(c(Rcpp = "Rcpp", RcppEigen = "RcppEigen", dplyr = "dplyr", purrr = "purrr", caret = "caret"),
                        function(p) as.character(utils::packageVersion(p))),
  fit_seconds = fit_seconds),
  paste0(args[[5]], ".meta.json"), auto_unbox = TRUE, digits = NA, na = "string")
cat(sprintf("[e8] done %s K=%d L=%d fit=%.1fs\n", method, length(kept), dim(W_path)[3], fit_seconds))
