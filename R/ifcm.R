# Intuitionistic fuzzy c-means (IFCM) clustering for a univariate series.
#
# Non-membership degrees come from the Yager-type intuitionistic fuzzy
# generator  nu = (1 - mu^alpha)^(1/alpha),  0 < alpha <= 1, so that the
# hesitation degree pi = 1 - mu - nu is non-negative. alpha = 1 gives
# nu = 1 - mu (ordinary fuzzy sets, no hesitation). Following Chaira (2011),
# cluster centres are updated with the hesitation-augmented memberships
# mu* = mu + pi and the algorithm is iterated until the memberships converge.


# Yager-type non-membership degrees
.yager_nonmembership <- function(mu, alpha) {
  (1 - mu^alpha)^(1 / alpha)
}


# Fuzzy c-means membership degrees of the values x with respect to centres
.fcm_membership <- function(x, centers, m) {
  D <- abs(outer(x, centers, "-"))
  zero <- D < 1e-12
  U <- D^(-2 / (m - 1))
  U <- U / rowSums(U)
  hit <- rowSums(zero) > 0
  if (any(hit)) {
    Z <- zero[hit, , drop = FALSE] * 1
    U[hit, ] <- Z / rowSums(Z)
  }
  U
}


# Fit IFCM to the numeric vector x.
#   Class    number of clusters (>= 2)
#   m        fuzzifier (> 1)
#   alpha    Yager generator exponent in (0, 1]
# The membership matrix is initialised at random; the caller controls the
# random number generator (e.g. set.seed()).
.ifcm <- function(x, Class, m = 2, alpha = 0.85, max_iter = 100, tol = 1e-5) {

  x <- as.numeric(x)
  n <- length(x)

  if (Class < 2) {
    stop("Class must be greater than or equal to 2")
  }
  if (length(unique(x)) < Class) {
    stop("The training data need at least 'Class' distinct values")
  }

  U <- matrix(stats::runif(n * Class), nrow = n, ncol = Class)
  U <- U / rowSums(U)

  converged <- FALSE
  iter <- 0L
  centers <- numeric(Class)

  for (iter in seq_len(max_iter)) {
    nu <- .yager_nonmembership(U, alpha)
    W <- (1 - nu)^m                       # (mu + pi)^m, since mu + pi = 1 - nu
    centers <- colSums(W * x) / colSums(W)
    U_new <- .fcm_membership(x, centers, m)
    delta <- max(abs(U_new - U))
    U <- U_new
    if (delta < tol) {
      converged <- TRUE
      break
    }
  }

  # Order clusters by centre so that feature columns have a stable meaning
  o <- order(centers)
  centers <- centers[o]
  U <- U[, o, drop = FALSE]
  nu <- .yager_nonmembership(U, alpha)

  list(
    centers = centers,
    membership = U,
    nonmembership = nu,
    hesitation = 1 - U - nu,
    iterations = iter,
    converged = converged,
    m = m,
    alpha = alpha
  )
}


# Membership and non-membership degrees of new values given fitted centres
.if_degrees <- function(x, centers, m, alpha) {
  mu <- .fcm_membership(as.numeric(x), centers, m)
  list(mu = mu, nu = .yager_nonmembership(mu, alpha))
}
