#!/usr/bin/env Rscript
# Official INSPRE GWPS-style fit (mirrors Rmd/run_gwps_analysis.R at pinned commit 7016697f).
# usage: run_inspre_official.R <panel.h5 genes x cells> <guide_ids.txt> <targets.txt> <genes.txt>
#                              <out_prefix> <seed> <ncores>
# writes <out_prefix>.Gpath.bin (float32, column-major, array [K,K,L], W orientation
# W[i,j] = effect of j on i = t(inspre G), BEFORE thresholding) and <out_prefix>.meta.json.
# Lambda rule (primary): D_bar = cummax of D_hat from largest to smallest lambda (Methods:
# D_bar_l = sup_{l' >= l} D_hat_l'); choose the smallest lambda with D_bar <= 0.01; if none, the
# largest lambda (flag). which.min(eps_hat_G) is recorded as an alternative only.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 7L) stop("usage: run_inspre_official.R <panel.h5> <guides> <targets> <genes> <out_prefix> <seed> <ncores>")
h5_path <- args[[1]]; out_prefix <- args[[5]]
seed <- as.integer(args[[6]]); ncores <- as.integer(args[[7]])

suppressPackageStartupMessages({library(hdf5r); library(inspre); library(dplyr); library(purrr)})
pkg_dir <- normalizePath(find.package("inspre"))
if (!grepl("/.envs/inspre/", pkg_dir, fixed = TRUE)) stop("inspre not from .envs/inspre: ", pkg_dir)
this_file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])
source(file.path(dirname(normalizePath(this_file)), "e2_inspre.R"))  # defines inspre_cv_fix() only
inspre_cv_fix()

X_ids <- readLines(args[[2]], warn = FALSE)
X_targets <- readLines(args[[3]], warn = FALSE)
X_vars <- readLines(args[[4]], warn = FALSE)
if (length(X_ids) != length(X_targets)) stop("guide/target metadata length mismatch")
if (anyDuplicated(X_vars)) stop("duplicate panel genes")

hfile <- H5File$new(h5_path, mode = "r")
X <- hfile[["X"]]
if (!identical(as.integer(X$dims), c(length(X_vars), length(X_ids)))) stop("X shape mismatch")

control_ids <- X_ids == "control"
n_ntc <- sum(control_ids)
if (n_ntc < 2L) stop("fewer than two non-targeting control cells")
X_ntc <- X[, control_ids]
rownames(X_ntc) <- X_vars
colnames(X_ntc) <- X_ids[control_ids]

# Guide effects for every (guide, on-panel target) pair, as in the authors' script.
genes_guides <- unique(data.frame(guide = X_ids[!control_ids], target = X_targets[!control_ids],
                                  stringsAsFactors = FALSE))
genes_guides <- genes_guides[genes_guides$target %in% X_vars, , drop = FALSE]
if (anyDuplicated(genes_guides$guide)) stop("a guide id maps to more than one target")
cat(sprintf("[inspre-official] computing effects for %d guides\n", nrow(genes_guides)))
guide_effects <- purrr::map2(genes_guides$guide, genes_guides$target,
                             ~ inspre::calc_inst_effect_h5X(.x, .y, X, X_ntc, X_ids, X_vars)) %>%
  purrr::list_rbind() %>%
  mutate(Z = inst_cor / cor_se, p = pt(Z, df = n - 2), p_adj = p.adjust(p, method = "fdr")) %>%
  arrange(target, p_adj) %>% filter(!duplicated(target))
keep_guides <- filter(guide_effects, inst_beta < -0.75, n > n_ntc + 50)
if (nrow(keep_guides) < 2L) stop(sprintf("guide filter kept only %d targets", nrow(keep_guides)))
keep_guides <- keep_guides[match(X_vars[X_vars %in% keep_guides$target], keep_guides$target), ]
targets <- as.list(keep_guides$target)
names(targets) <- keep_guides$inst_id
cat(sprintf("[inspre-official] kept %d / %d targets; controls=%d seed=%d ncores=%d\n",
            nrow(keep_guides), nrow(guide_effects), n_ntc, seed, ncores))

set.seed(seed)
t0 <- Sys.time()
fit <- fit_inspre_from_h5X(X, X_ntc, X_ids, X_vars, targets, max_med_ratio = 50, cv_folds = 5,
                           ncores = ncores, DAG = TRUE, min_nz = 0.015, lambda_min_ratio = 0.1,
                           nlambda = 10)
fit_seconds <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
if (is.null(fit$D_hat) || is.null(fit$eps_hat_G)) stop("CV outputs (D_hat/eps_hat_G) missing")

# Second upstream bug at 7016697f: fit_inspre_from_h5X ends with `G_hat <- R_hat` (a leftover from the
# rename), overwriting the 3-D G path with the 2-D observed R. Rebuild G from V with the identical formula
# fit_inspre_from_R uses: G[,,l] = I - V[,,l] / diag(V[,,l]).
if (length(dim(fit$G_hat)) != 2L || length(dim(fit$V)) != 3L) stop("upstream layout changed; revisit")
G_path <- array(0, dim = dim(fit$V))
for (l in seq_len(dim(fit$V)[3])) G_path[, , l] <- diag(nrow(fit$V)) - fit$V[, , l] / diag(fit$V[, , l])
dimnames(G_path) <- list(rownames(fit$R_hat), colnames(fit$R_hat), fit$lambda)
# h5X path: rows are named by guide id (names(targets)), columns by gene.
kept <- colnames(G_path)
if (is.null(kept) || !identical(unname(unlist(targets[rownames(G_path)])), kept)) stop("G_hat rows (guide->target) and columns differ")
rownames(G_path) <- kept
L <- length(fit$lambda)
# inspre G is exposure x outcome: predict_inspre_G_h5X uses X_hat = t(G) %*% X, so G[j,i] = effect of j on i.
W_path <- aperm(G_path, c(2, 1, 3))
for (l in seq_len(L)) diag(W_path[, , l]) <- 0
writeBin(as.numeric(W_path), paste0(out_prefix, ".Gpath.bin"), size = 4L, endian = "little")

ord <- order(fit$lambda, decreasing = TRUE)
dbar <- cummax(fit$D_hat[ord])
ok <- which(dbar <= 0.01)
fallback <- length(ok) == 0L
chosen <- if (fallback) ord[1] else ord[max(ok)]
alt <- which.min(fit$eps_hat_G)

jsonlite::write_json(list(
  kept_genes = kept,
  guide_filter_table = as.data.frame(guide_effects %>% mutate(kept = inst_id %in% keep_guides$inst_id) %>%
      select(guide = inst_id, target, inst_beta, n, p, p_adj, kept)),
  n_control = n_ntc,
  selection = list(
    rule = "smallest lambda with D_bar <= 0.01; D_bar = cummax(D_hat) from largest to smallest lambda",
    threshold = 0.01, fallback_largest_lambda = fallback,
    chosen_index = chosen, chosen_lambda = fit$lambda[[chosen]],
    D_bar_in_lambda_desc_order = as.numeric(dbar), lambda_desc_order_index = as.integer(ord),
    lambda = as.numeric(fit$lambda), D_hat = as.numeric(fit$D_hat),
    eps_hat_G = as.numeric(fit$eps_hat_G), eps_hat_I = as.numeric(fit$eps_hat_I),
    alt_which_min_eps_hat_G_index = alt, alt_lambda = fit$lambda[[alt]]),
  settings = list(DAG = TRUE, cv_folds = 5L, max_med_ratio = 50, lambda_min_ratio = 0.1,
                  nlambda = 10L, min_nz = 0.015, ncores = ncores, rho_default = 100,
                  post_threshold_abs_G_lt = 0.015,
                  guide_filter = "best guide/target by BH p_adj (one-sided pt); inst_beta < -0.75; n > n_ntc + 50"),
  inspre_version = as.character(utils::packageVersion("inspre")),
  inspre_sha = "7016697f87db64d7a1ac9bc384dd048298470425",
  inspre_cv_fix = "predict_inspre*: res$R_hat[,,i] -> res$G_hat[,,i] (upstream 840665b rename)",
  inspre_h5X_fix = "fit_inspre_h5X overwrites 3-D G_hat with 2-D R_hat; G rebuilt from V as I - V/diag(V)",
  fit_seconds = fit_seconds, seed = seed, K = length(kept), L = L),
  paste0(out_prefix, ".meta.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA, na = "string")
cat(sprintf("[inspre-official] chosen idx=%d lambda=%.6g fallback=%s alt(min eps)=%d kept=%d fit=%.0fs\n",
            chosen, fit$lambda[[chosen]], fallback, alt, length(kept), fit_seconds))
hfile$close_all()
