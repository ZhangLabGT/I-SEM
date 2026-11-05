#!/usr/bin/env Rscript
# Our extension of the authors' simulator, labelled as such. generate_dataset at the locked setting: the setting's own
# assignment lines and call are parsed by simulate_authors.R (loaded unedited); only int_beta and N_int are overridden.
# RATIO = the authors' multiple_iv_reg / filter_tce against the true R. SWEEP = writes datasets for the method fits.
# usage: WEAK_MODE=RATIO|SWEEP Rscript sweep.R <folder>
orig <- file.path(Sys.getenv("REPO_ROOT"), "scripts/experiments/e9_author_sims/simulate_authors.R")
src <- readLines(orig)
i <- grep("^this_dir <- ", src); stopifnot(length(i) == 1L); src[i] <- sprintf('this_dir <- "%s"', dirname(orig))
for (x in c("run_adapre()", "run_inspre()")) { j <- which(src == x); stopifnot(length(j) == 1L); src[j] <- "" }
eval(parse(text = src), envir = globalenv())
suppressPackageStartupMessages({library(parallel); library(purrr); library(dplyr)})
MODE <- Sys.getenv("WEAK_MODE"); stopifnot(MODE %in% c("RATIO", "SWEEP"))
set7 <- Filter(function(s) s$setting == "rand_cycles_confound_7", inspre_settings())[[1]]

gen <- function(d, n, seed, mixed = FALSE) {
  env <- new.env(parent = globalenv())
  for (a in set7$assign) eval(parse(text = a), envir = env)
  set.seed(seed)
  env$N_int <- as.integer(n)
  if (mixed) {                              # per-gene int_beta: the generator multiplies it into the design column-wise
    D <- env$D; C <- env$C
    b <- rep(-2, D + C); b[sample(D, 2)] <- -0.25
    env$int_beta <- b
  } else env$int_beta <- -d
  invisible(capture.output(eval(parse(text = set7$call), envir = env)))
  list(ds = get("dataset", envir = env), env = env)
}

ratio <- function(cell, rep, seed) {
  g <- gen(cell$d, cell$n, seed, cell$mixed); ds <- g$ds
  X <- ds$Y; targets <- ds$targets; R <- ds$R; D <- ncol(X); colnames(X) <- paste0("V", 1:D)
  rownames(R) <- colnames(R) <- colnames(X)
  iv <- purrr::map(colnames(X), ~ inspre:::multiple_iv_reg(.x, X, targets)) %>% purrr::list_rbind()
  Rh <- data.matrix(dplyr::select(iv, ends_with("beta_hat"))); SE <- data.matrix(dplyr::select(iv, ends_with("se_hat")))
  rownames(Rh) <- rownames(SE) <- iv$target; colnames(Rh) <- colnames(SE) <- colnames(X)
  flt <- tryCatch(inspre:::filter_tce(Rh, SE), error = function(e) NULL)
  K <- if (is.null(flt)) 0L else nrow(flt$R)
  masked <- if (is.null(flt)) NA else mean(is.na(flt$R))
  off <- !diag(D); ctl <- targets == "control"; sdc <- apply(X[ctl, ], 2, sd)
  b <- rep_len(g$env$int_beta, D)
  kd <- do.call(rbind, lapply(seq_len(D), function(j) {
    k <- targets == colnames(X)[j]
    sh <- mean(X[k, j]) - mean(X[ctl, j])
    e <- abs(Rh[j, -j] - R[j, -j])
    data.frame(cell = cell$name, rep = rep, gene = j, int_beta = b[j], n = sum(k), shift_sd = sh / sdc[j],
               t = sh / sqrt(var(X[k, j]) / sum(k) + var(X[ctl, j]) / sum(ctl)), flip = sign(sh) != sign(b[j]),
               err_med = median(e), err_p99 = quantile(e, .99, names = FALSE), err_max = max(e),
               frac_abs_gt2 = mean(abs(Rh[j, -j]) > 2), frac_se_gt1 = mean(SE[j, -j] > 1))
  }))
  e <- abs(Rh - R)[off]
  list(kd = kd, ds = data.frame(cell = cell$name, d = cell$d, n_kd = cell$n, mixed = cell$mixed, rep = rep,
       err_med = median(e), err_p99 = quantile(e, .99, names = FALSE), err_max = max(e),
       frac_abs_gt2 = mean(abs(Rh[off]) > 2), frac_se_gt1 = mean(SE[off] > 1), flip_frac = mean(kd$flip),
       K_after_filter = K, masked_frac_after_filter = masked, cor_R = cor(Rh[off], R[off]), cor_tR = cor(Rh[off], t(R)[off])))
}

folder <- args[[1]]
if (MODE == "RATIO") {
  grid <- expand.grid(d = c(0.1, 0.25, 0.5, 1, 2), n = c(25, 100, 400, 1600))
  cells <- lapply(seq_len(nrow(grid)), function(i)
    list(name = sprintf("d%g_n%d", grid$d[i], grid$n[i]), d = grid$d[i], n = grid$n[i], mixed = FALSE))
  cells <- c(cells, list(list(name = "mixed", d = NA_real_, n = 100, mixed = TRUE)))
  jobs <- list(); for (ci in seq_along(cells)) for (r in 1:20) jobs[[length(jobs) + 1L]] <- list(c = cells[[ci]], r = r, s = 300000L + 1000L * ci + r)
  res <- mclapply(jobs, function(j) tryCatch(ratio(j$c, j$r, j$s), error = function(e) list(err = paste(j$c$name, j$r, conditionMessage(e)))),
                  mc.cores = 64, mc.preschedule = FALSE)
  good <- Filter(function(x) is.null(x$err) && !inherits(x, "try-error"), res)
  cat(sprintf("ratio test: %d jobs, %d failed\n", length(res), length(res) - length(good)))
  write.csv(do.call(rbind, lapply(good, `[[`, "ds")), file.path(folder, "results/ratio_datasets.csv"), row.names = FALSE)
  write.csv(do.call(rbind, lapply(good, `[[`, "kd")), file.path(folder, "results/ratio_knockdowns.csv"), row.names = FALSE)
} else {
  cells <- list(c(2, 25), c(1, 100), c(0.5, 400), c(0.25, 1600), c(0.25, 400), c(0.3, 100), c(0.25, 100), c(0.25, 25))
  cells <- c(lapply(cells, function(x) list(name = sprintf("d%g_n%d", x[1], x[2]), d = x[1], n = x[2], mixed = FALSE)),
             list(list(name = "mixed", d = NA_real_, n = 100, mixed = TRUE)))
  jobs <- list(); for (ci in seq_along(cells)) for (r in 1:10) jobs[[length(jobs) + 1L]] <- list(c = cells[[ci]], r = r, s = 400000L + 1000L * ci + r)
  res <- mclapply(jobs, function(j) {
    g <- gen(j$c$d, j$c$n, j$s, j$c$mixed); ds <- g$ds; env <- g$env
    params <- mget(c("D", "N_int", "N_cont", "int_beta", "graph", "v", "p", "DAG", "C"), envir = env)
    write_rep(file.path(folder, "data", "sweep", j$c$name, sprintf("rep%d", j$r)), ds$Y, ds$targets, ds$G,
              list(sim = "sweep", setting = j$c$name, rep = j$r, seed = j$s, params = params, truth_thr = params$v / 4,
                   truth_rule = "abs_ge", truth_source = "inspre calc_metrics(eps = v/4)"))
    "ok"
  }, mc.cores = 32, mc.preschedule = FALSE)
  cat(sprintf("sweep data: %d datasets, %d ok\n", length(res), sum(unlist(res) == "ok")))
}
