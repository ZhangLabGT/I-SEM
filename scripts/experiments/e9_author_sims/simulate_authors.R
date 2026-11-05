#!/usr/bin/env Rscript
# E9: datasets from the authors' own simulators, run as released. No generator, setting or call is edited.
#
#   inspre  inspre::generate_dataset (inspre 1.0.1 @ 7016697f, .envs/inspre). Every setting chunk of the
#           8 Rmd/simulate_*.Rmd files at that commit (8 chunks each, 64 settings), n_iter = 10 each, as their
#           run_sim() does. The chunk's own assignment lines and its own `dataset <- generate_dataset(...)`
#           line are parsed from the Rmd and evaluated verbatim (nothing transcribed by hand).
#           Seeds: the Rmds set none (one rslurm job per iter); set.seed(iter) before each call, iter = 1..10.
#   adapre  generate_dataset_two_int from keleslab/DEMO_RECOMB_2026 @ c9e3b1f (R/simulate_two_interventions.R,
#           scripts/baselines/adapre/upstream). The release has one setting, the demo's (run_adapre_demo.R); its
#           assignment expressions and its `sim <- generate_dataset_two_int(...)` expression are parsed from
#           the demo and evaluated verbatim after set.seed(seed), seed = 1..10. Both outputs are written:
#           set1 (homogeneous int_beta) and set2 (heterogeneous; the one the demo fits).
#           seed 1 must be identical() to the authors' committed output/sim_data.RData; stops otherwise.
# Truth threshold = the authors' own: inspre plot_*.Rmd scores with calc_metrics(eps = v / 4) (|G| < eps -> 0);
# the ADAPRE demo with edge_thr = 0.001 (|G| > thr is an edge).
#
# mc2d is pinned to 0.1-22 (.envs/Rlib, CRAN archive, 2022-11-20). generate_network/generate_dataset draw
# edge and confounder weights with mc2d::rpert. Since mc2d 0.2.0 (2023-07-17) rpert returns NA when mode < min;
# the confounder draw rpert(min = 0.01, mode = v^log(D)) has mode < min for every random-graph setting (0.3^log(50)
# = 0.009), so on mc2d >= 0.2.0 all 16 random + confounder settings fail (I - G singular). The inspre authors
# report those settings, which needs the earlier behaviour (a Beta draw with shape1 slightly below 1).
# For mode >= min both versions make the same rbeta call (checked: ADAPRE seed 1 reproduces either way).
#
# usage: simulate_authors.R <out_dir>     (needs mc2d on .libPaths(); e9_author_sims.py sets R_LIBS)
# writes <out_dir>/<sim>/<setting>/rep<k>/
#   Y.bin        float32, row-major cells x genes, exactly the generator's Y (control-z-scored by the generator)
#   targets.txt  "control" or V<j>, one per row
#   G.bin        float64, column-major D x D, the generator's true G (exposure x outcome, G[i, j]: i -> j)
#   meta.json    setting, seed, the evaluated parameters, truth threshold, provenance
# A rep whose meta.json exists is skipped (resume).

suppressPackageStartupMessages({library(inspre); library(jsonlite)})
pkg_dir <- normalizePath(find.package("inspre"))
if (!grepl("/.envs/inspre/", pkg_dir, fixed = TRUE)) stop("inspre not from .envs/inspre: ", pkg_dir)
if (!requireNamespace("mc2d", quietly = TRUE) || packageVersion("mc2d") != "0.1.22")
  stop("need mc2d 0.1-22 on .libPaths() (see header)")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("usage: simulate_authors.R <out_dir>")
out_dir <- args[[1]]
this_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])))
study <- normalizePath(file.path(this_dir, "..", ".."))
inspre_src <- file.path(study, "baselines", "inspre", "upstream")
adapre_src <- file.path(study, "baselines", "adapre", "upstream")
N_ITER <- 10L

write_rep <- function(d, Y, targets, G, meta) {
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
  writeBin(as.numeric(t(Y)), file.path(d, "Y.bin"), size = 4L)
  writeLines(targets, file.path(d, "targets.txt"))
  writeBin(as.numeric(G), file.path(d, "G.bin"), size = 8L)
  off <- !diag(nrow(G))
  thr_edge <- if (meta$truth_rule == "abs_ge") abs(G) >= meta$truth_thr else abs(G) > meta$truth_thr
  meta <- c(meta, list(n_rows = nrow(Y), D = ncol(Y), genes = colnames(Y),
                       n_true_edges = sum(thr_edge & off),
                       n_subthreshold_nonzero = sum(!thr_edge & abs(G) > 1e-8 & off),
                       spectral_radius = max(Mod(eigen(G, only.values = TRUE)$values)),
                       r_version = R.version.string, inspre_version = as.character(packageVersion("inspre")),
                       mc2d_version = as.character(packageVersion("mc2d"))))
  write_json(meta, file.path(d, "meta.json"), auto_unbox = TRUE, digits = NA)   # last: marks the rep complete
}

# ------------------------------------------------------------------------------------------------ inspre
# Each setting chunk: header comments ("# 1", "# Cycles, ...", "# High density, ..."), assignments, then
# `dataset <- generate_dataset(...)`. The run_sim() call passes the same named arguments.
inspre_settings <- function() {
  out <- list()
  for (f in sort(list.files(file.path(inspre_src, "Rmd"), pattern = "^simulate_.*\\.Rmd$", full.names = TRUE))) {
    lines <- readLines(f, warn = FALSE)
    starts <- grep("^```\\{r\\}", lines); ends <- grep("^```\\s*$", lines)
    for (s in starts) {
      e <- min(ends[ends > s])
      chunk <- lines[(s + 1):(e - 1)]
      call_i <- grep("^dataset <- generate_dataset\\(", chunk)
      if (length(call_i) != 1L) next
      head <- chunk[seq_len(call_i - 1)]
      comments <- trimws(sub("^#", "", grep("^#", head, value = TRUE)))
      number <- as.integer(comments[[1]])
      stem <- sub("^simulate_", "", sub("\\.Rmd$", "", basename(f)))
      out[[length(out) + 1L]] <- list(
        setting = sprintf("%s_%d", stem, number), rmd = basename(f), chunk = number,
        description = comments[-1], assign = grep("^[A-Za-z_][A-Za-z0-9_.]* <- ", head, value = TRUE),
        call = chunk[[call_i]])
    }
  }
  out
}

run_inspre <- function() {
  sets <- inspre_settings()
  if (length(sets) != 64L) stop("expected 64 setting chunks, parsed ", length(sets))
  for (s in sets) {
    env <- new.env(parent = globalenv())
    for (a in s$assign) eval(parse(text = a), envir = env)
    params <- mget(c("n_iter", "D", "N_int", "N_cont", "int_beta", "graph", "v", "p", "DAG", "C"), envir = env)
    if (params$n_iter != N_ITER) stop(s$setting, ": n_iter != 10")
    for (iter in seq_len(N_ITER)) {
      d <- file.path(out_dir, "inspre", s$setting, sprintf("rep%d", iter))
      if (file.exists(file.path(d, "meta.json"))) next
      set.seed(iter)
      eval(parse(text = s$call), envir = env)              # the authors' line: dataset <- generate_dataset(...)
      ds <- get("dataset", envir = env)
      write_rep(d, ds$Y, ds$targets, ds$G, list(
        sim = "inspre", setting = s$setting, rep = iter, seed = iter, params = params,
        rmd = s$rmd, chunk = s$chunk, description = s$description, call = s$call,
        authors_job_name = paste(params$graph, params$DAG, params$p, params$v, params$int_beta, params$C, sep = "_"),
        truth_thr = params$v / 4, truth_rule = "abs_ge",
        truth_source = "inspre Rmd/plot_*.Rmd: get_metrics(dir_name, n_iter, v/4) -> calc_metrics(eps = v/4)"))
    }
    cat(sprintf("inspre %-28s done (%s)\n", s$setting, paste(s$description, collapse = "; ")))
  }
}

# ------------------------------------------------------------------------------------------------ ADAPRE
run_adapre <- function() {
  source(file.path(adapre_src, "R", "simulate_two_interventions.R"))   # as the demo does, after library(inspre)
  exprs <- parse(file.path(adapre_src, "run_adapre_demo.R"))
  is_assign <- function(x, names) is.call(x) && identical(x[[1]], as.name("<-")) && is.name(x[[2]]) &&
    as.character(x[[2]]) %in% names
  setting_names <- c("D", "N_int", "N_cont", "int_beta_base", "int_beta_hetero", "graph", "v", "p", "DAG", "C")
  assigns <- Filter(function(x) is_assign(x, setting_names), as.list(exprs))
  sim_call <- Filter(function(x) is_assign(x, "sim"), as.list(exprs))
  if (length(assigns) != length(setting_names) || length(sim_call) != 1L) stop("demo parse: unexpected structure")
  seed_calls <- Filter(function(x) is.call(x) && identical(x[[1]], as.name("set.seed")), as.list(exprs))
  if (length(seed_calls) != 1L || seed_calls[[1]][[2]] != 1) stop("demo parse: expected set.seed(1)")
  env <- new.env(parent = globalenv())
  for (a in assigns) eval(a, envir = env)
  params <- mget(setting_names, envir = env)

  load(file.path(adapre_src, "output", "sim_data.RData"), envir = (authors <- new.env()))
  for (seed in seq_len(N_ITER)) {
    set.seed(seed)
    eval(sim_call[[1]], envir = env)                    # the authors' expression: sim <- generate_dataset_two_int(...)
    sim <- get("sim", envir = env)
    repro <- NA
    if (seed == 1L) {
      # The random draws (G_orig, targets) must be bit-identical; derived fields go through solve() and differ
      # by LAPACK rounding across BLAS builds (measured 3e-14 on this host), so they are held to 1e-10.
      num <- function(s) unlist(lapply(c("set1", "set2"), function(k) unlist(s[[k]][c("Y", "R", "G", "int_beta")])))
      maxdiff <- max(abs(num(sim) - num(authors$sim)))
      repro <- identical(sim$G_orig, authors$sim$G_orig) && identical(sim$set2$targets, authors$sim$set2$targets) &&
        maxdiff < 1e-10
      if (!repro) stop("seed 1 does not reproduce the authors' output/sim_data.RData (max |diff| ", maxdiff, ")")
      repro <- list(G_orig_identical = TRUE, targets_identical = TRUE, max_abs_diff_Y_R_G_int_beta = maxdiff)
      cat(sprintf("adapre seed 1 reproduces the authors' output/sim_data.RData (G_orig identical, max |diff| %.2g)\n", maxdiff))
    }
    for (set in c("set1", "set2")) {
      d <- file.path(out_dir, "adapre", paste0("demo_", set), sprintf("rep%d", seed))
      if (file.exists(file.path(d, "meta.json"))) next
      x <- sim[[set]]
      write_rep(d, x$Y, x$targets, x$G, list(
        sim = "adapre", setting = paste0("demo_", set), rep = seed, seed = seed, params = params,
        int_beta = as.numeric(x$int_beta), call = paste(deparse(sim_call[[1]]), collapse = " "),
        description = if (set == "set1") "homogeneous int_beta = int_beta_base" else
          "heterogeneous int_beta = int_beta_hetero (the demo's dataset)",
        reproduces_authors_sim_data = repro,
        truth_thr = 0.001, truth_rule = "abs_gt", truth_source = "run_adapre_demo.R: edge_thr <- 0.001 (abs(G) > thr)"))
    }
  }
  cat("adapre done\n")
}

run_adapre()
run_inspre()
