# multiple_iv_reg with the divisor made an argument. The body is the authors' (inspre R/inspre.R:574, identical in ADAPRE's
# core); only the two marked lines differ, and arm "observed" keeps their expressions verbatim (check_patched.R asserts it).
# truth_div: named true own-gene shift. truth_R: true ratio matrix, exposure x outcome, gene dimnames.
make_patched <- function(arm, truth_div = NULL, truth_R = NULL) {
  if (!arm %in% c("observed", "true_divisor", "true_R")) stop("unknown arm ", arm)
  function(target, .X, .targets) {
    inst_obs <- .targets == target
    control_obs <- .targets == "control"
    n_target <- sum(inst_obs)
    n_control <- sum(control_obs)
    n_total <- n_target + n_control
    X_target <- .X[inst_obs, ]
    X_control <- .X[control_obs, ]
    this_feature <- which(colnames(.X) == target)
    X_exp_target <- X_target[, this_feature]
    X_exp_control <- X_control[, this_feature]
    sums_target <- colSums(X_target)
    if (arm == "observed") {                 # marked: the authors' two lines
      beta_inst_obs <- sum(X_exp_target)/n_target
      beta_hat <- sums_target/sums_target[this_feature]
    } else {                                 # marked: the swap; R_jj = 1 as the observed ratio has by construction
      beta_inst_obs <- truth_div[[target]]
      beta_hat <- if (arm == "true_divisor") (sums_target/n_target)/beta_inst_obs else truth_R[target, colnames(.X)]
      beta_hat[this_feature] <- 1
    }
    resid_sum <- colSums((X_target - outer(X_exp_target, beta_hat))**2) +
      colSums((X_control - outer(X_exp_control, beta_hat))**2)
    var_resid <- resid_sum/(n_total*n_target*beta_inst_obs**2)
    se_hat <- sqrt(var_resid)
    names(beta_hat) <- paste(names(beta_hat), "beta_hat", sep="_")
    names(se_hat) <- paste(names(se_hat), "se_hat", sep="_")
    return(as.data.frame(c(list(target = target, beta_obs = beta_inst_obs), as.list(c(beta_hat, se_hat)))))
  }
}

read_truth <- function(dir) {
  genes <- readLines(file.path(dir, "genes.txt"), warn = FALSE)
  D <- length(genes)
  list(div = setNames(readBin(file.path(dir, "div.bin"), "numeric", n = D, size = 8L), genes),
       R = matrix(readBin(file.path(dir, "R.bin"), "numeric", n = D * D, size = 8L), D, D, dimnames = list(genes, genes)))
}
