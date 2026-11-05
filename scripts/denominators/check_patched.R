# Assert-only: arm observed is identical() to both authors' functions; the swap arms do what they say.
# usage: check_patched.R <upstream adapre core.R>
suppressPackageStartupMessages(library(inspre))
args <- commandArgs(trailingOnly = TRUE)
source(file.path(Sys.getenv("DIVISOR_HERE"), "patched.R"))
core <- new.env(); sys.source(args[[1]], envir = core)
set.seed(1); g <- paste0("V", 1:6)
X <- matrix(rnorm(6 * 300), 300, 6, dimnames = list(NULL, g)); tg <- c(rep("control", 240), rep(g, each = 10))
X[tg == "V2", "V2"] <- X[tg == "V2", "V2"] - 1
ob <- make_patched("observed")
for (t in g) {
  a <- ob(t, X, tg)
  stopifnot(identical(a, inspre:::multiple_iv_reg(t, X, tg)), identical(a, core$multiple_iv_reg(t, X, tg)))
}
div <- setNames(rep(-2, 6), g); R <- matrix(0.25, 6, 6, dimnames = list(g, g))
a <- make_patched("true_divisor", div)("V2", X, tg)
stopifnot(a$beta_obs == -2, a$V2_beta_hat == 1, all.equal(a$V1_beta_hat, mean(X[tg == "V2", "V1"]) / -2))
b <- make_patched("true_R", div, R)("V2", X, tg)
stopifnot(b$beta_obs == -2, b$V2_beta_hat == 1, b$V1_beta_hat == 0.25, is.finite(b$V1_se_hat))
cat("check_patched OK\n")
