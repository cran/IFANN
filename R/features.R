# Feature construction, neural network fitting and prediction helpers.
#
# For a target y[t] and lag order L the inputs are
#   * the lagged values y[t-1], ..., y[t-L] (min-max scaled), and
#   * the IFCM membership and non-membership degrees of each lagged value.
# Only past values enter the inputs, and the IFCM centres and the scaling
# constants are estimated from the fitting part of the series, so no
# information from the period being predicted leaks into the model.


.minmax <- function(x) {
  lo <- min(x)
  hi <- max(x)
  if (hi - lo < .Machine$double.eps) hi <- lo + 1
  list(lo = lo, hi = hi)
}

.scale01 <- function(x, s) (x - s$lo) / (s$hi - s$lo)

.unscale01 <- function(z, s) z * (s$hi - s$lo) + s$lo


# Everything estimated from the fitting window y_fit: IFCM centres and scaling
.prepare <- function(y_fit, Class, m, alpha, max_iter, tol) {
  cl <- .ifcm(y_fit, Class = Class, m = m, alpha = alpha,
              max_iter = max_iter, tol = tol)
  list(
    centers = cl$centers,
    m = m,
    alpha = alpha,
    scaling = .minmax(y_fit),
    ifcm = cl
  )
}


# Feature matrix for the targets y[idx] using lag order L
.feature_matrix <- function(y, idx, L, prep) {
  n <- length(idx)
  lag_idx <- outer(idx, seq_len(L), "-")
  lagmat <- matrix(y[lag_idx], nrow = n, ncol = L)

  d <- .if_degrees(as.vector(lagmat), prep$centers, prep$m, prep$alpha)
  K <- length(prep$centers)

  mu_blocks <- vector("list", L)
  nu_blocks <- vector("list", L)
  for (k in seq_len(L)) {
    rows <- ((k - 1L) * n + 1L):(k * n)
    mu_blocks[[k]] <- d$mu[rows, , drop = FALSE]
    nu_blocks[[k]] <- d$nu[rows, , drop = FALSE]
  }

  X <- cbind(.scale01(lagmat, prep$scaling),
             do.call(cbind, mu_blocks),
             do.call(cbind, nu_blocks))

  colnames(X) <- c(
    paste0("Lag", seq_len(L)),
    paste0("Mu_L", rep(seq_len(L), each = K), "_C", rep(seq_len(K), L)),
    paste0("Nu_L", rep(seq_len(L), each = K), "_C", rep(seq_len(K), L))
  )
  X
}


# Number of weights of a p-size-1 network with skip-layer = FALSE
.n_weights <- function(p, size) (p + 1L) * size + size + 1L


# Fit `Repeats` single-hidden-layer networks (different random starts) on
# the targets y[idx]; predictions are averaged over the networks.
.fit_ann <- function(y, idx, L, prep, params, Maxit, Repeats) {
  X <- .feature_matrix(y, idx, L, prep)
  z <- .scale01(y[idx], prep$scaling)
  size <- as.integer(params$Size)
  nets <- lapply(seq_len(Repeats), function(r) {
    nnet::nnet(
      x = X, y = z,
      size = size,
      decay = params$Decay,
      maxit = Maxit,
      linout = TRUE,
      trace = FALSE,
      MaxNWts = .n_weights(ncol(X), size) + 10L
    )
  })
  list(nets = nets, n_inputs = ncol(X), size = size, decay = params$Decay)
}

.predict_scaled <- function(fit, X) {
  P <- vapply(fit$nets, function(net) as.numeric(stats::predict(net, X)),
              numeric(nrow(X)))
  if (is.matrix(P)) rowMeans(P) else mean(P)
}


# Predict the targets y[idx] (idx increasing and contiguous for "recursive").
#   "one-step"  : every prediction uses the observed past values
#   "recursive" : predictions inside idx replace the observations as lags
.predict_ann <- function(fit, y, idx, L, prep, type = "one-step") {
  if (type == "one-step") {
    X <- .feature_matrix(y, idx, L, prep)
    return(.unscale01(.predict_scaled(fit, X), prep$scaling))
  }
  yy <- y
  out <- numeric(length(idx))
  for (i in seq_along(idx)) {
    t <- idx[i]
    X <- .feature_matrix(yy, t, L, prep)
    p <- .unscale01(.predict_scaled(fit, X), prep$scaling)
    yy[t] <- p
    out[i] <- p
  }
  out
}


.metrics <- function(actual, pred) {
  e <- actual - pred
  c(
    RMSE = sqrt(mean(e^2)),
    MAE = mean(abs(e)),
    MAPE = if (any(actual == 0)) NA_real_ else mean(abs(e / actual)) * 100
  )
}
