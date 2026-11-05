# Runs a repo R driver (inspre_path.R, adapre_fit.R) byte-unchanged with multiple_iv_reg overridden at run time.
# usage: shim.R <driver.R> <driver args...>; env DIVISOR_ARM, DIVISOR_TRUTH, DIVISOR_HERE. Nothing under upstream/ changes.
.a <- base::commandArgs(trailingOnly = TRUE)
.script <- .a[[1]]; .args <- .a[-1]
source(file.path(Sys.getenv("DIVISOR_HERE"), "patched.R"))
.arm <- Sys.getenv("DIVISOR_ARM")
.truth <- if (.arm == "observed") list() else read_truth(Sys.getenv("DIVISOR_TRUTH"))
.patched <- make_patched(.arm, .truth$div, .truth$R)
commandArgs <- function(trailingOnly = FALSE) if (trailingOnly) .args else paste0("--file=", .script)
source <- function(file, ...) {
  base::source(file, ...)
  if (grepl("e2_inspre\\.R$", file)) {                 # the fix the driver calls right after library(inspre)
    .fix <- get("inspre_cv_fix", envir = globalenv())
    assign("inspre_cv_fix", function() {
      .fix(); utils::assignInNamespace("multiple_iv_reg", .patched, "inspre"); cat("[shim] inspre arm", .arm, "\n")
    }, envir = globalenv())
  }
  if (grepl("adapre_inspre_core\\.R$", file)) {        # the fit
    assign("multiple_iv_reg", .patched, envir = globalenv()); cat("[shim] adapre fit arm", .arm, "\n")
  }
  invisible()
}
sys.source <- function(file, envir, ...) {             # adapre_fit.R's pre-filter env
  base::sys.source(file, envir = envir, ...)
  assign("multiple_iv_reg", .patched, envir = envir); cat("[shim] adapre pre-filter arm", .arm, "\n")
  invisible()
}
base::source(.script, local = globalenv())
