# ICA-LiNGAM (Shimizu et al. 2006) via pcalg::lingam, called as the inspre authors' simulation Rmds call it:
#   lingam_res <- pcalg::lingam(dataset$Y); G_hat <- t(lingam_res$Bpruned)
# Runs inside .envs/lingam (scripts/envs/lingam.lock.txt). Writes Bpruned unchanged: Bpruned[i, j] is the edge j -> i,
# which is already this repo's orientation (W[i, j] = effect of j on i), plus a JSON meta file.
# usage: Rscript fit.R <X.bin float32 row-major> <n_rows> <n_cols> <out.bin float64 column-major> <seed>
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5L) stop("usage: fit.R <X.bin> <n_rows> <n_cols> <out.bin> <seed>")
n <- as.integer(args[[2]]); d <- as.integer(args[[3]])
X <- matrix(readBin(args[[1]], "numeric", n = n * d, size = 4), nrow = n, ncol = d, byrow = TRUE)
set.seed(as.integer(args[[5]]))   # fastICA's random initial unmixing matrix
t0 <- Sys.time()
res <- pcalg::lingam(X)
secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
writeBin(as.vector(res$Bpruned), args[[4]], size = 8)
writeLines(jsonlite::toJSON(list(method = "ICA-LiNGAM", implementation = "pcalg::lingam",
                                 pcalg = as.character(packageVersion("pcalg")), n = n, d = d,
                                 seed = as.integer(args[[5]]), wall_seconds = secs,
                                 orientation = "Bpruned[i,j] = edge j -> i (as returned)"),
                            auto_unbox = TRUE), paste0(args[[4]], ".json"))
