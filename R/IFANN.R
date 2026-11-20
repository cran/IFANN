#' Intuitionistic Fuzzy Artificial Neural Network for Time Series
#'
#' Fits an intuitionistic fuzzy artificial neural network (IFANN) model to a
#' univariate time series. The lagged values of the series and their
#' intuitionistic fuzzy membership and non-membership degrees (from
#' intuitionistic fuzzy c-means clustering) are used as inputs of a
#' feed-forward neural network with one hidden layer. The lag order and the
#' network hyper-parameters can each be set manually or tuned automatically
#' by grid search.
#'
#' @details
#' \strong{Model.} The first \code{round(length(Data) * Split)} observations
#' form the training set and the rest the test set. Intuitionistic fuzzy
#' c-means clustering (Chaira, 2011) with \code{Class} clusters is fitted to
#' the training observations only. Non-membership degrees come from the
#' Yager-type generator \eqn{\nu = (1 - \mu^\alpha)^{1/\alpha}}, so the
#' hesitation degree \eqn{\pi = 1 - \mu - \nu} is non-negative for
#' \eqn{0 < \alpha \le 1}; \code{Alpha = 1} gives ordinary fuzzy sets. For a
#' target \eqn{y_t} and lag order \eqn{L} the inputs are
#' \eqn{y_{t-1}, \dots, y_{t-L}} (min-max scaled with training statistics)
#' and the membership and non-membership degrees of each lagged value in
#' every cluster, i.e. \eqn{p = L(1 + 2\,Class)} inputs. Only past values
#' enter the inputs, so the test-set forecasts use no information from the
#' values being forecast.
#'
#' \strong{Network.} The network has \eqn{p} inputs, one hidden layer of
#' \code{Size} logistic neurons and a linear output, and is fitted to the
#' target scaled to [0, 1] with \code{\link[nnet]{nnet}} (BFGS with weight
#' decay \code{Decay}, at most \code{Maxit} iterations). Because the result
#' depends on the random starting weights, \code{Repeats} networks are
#' fitted from different starts and their predictions are averaged.
#' \code{Maxit} and \code{Repeats} are fixed settings used for every model,
#' including during tuning.
#'
#' \strong{Tuning.} \code{Lag_Tuning} and \code{ANN_Tuning} are set
#' independently:
#' \itemize{
#'   \item \code{"manual"}: the single values \code{Lag} and / or
#'     \code{Size}, \code{Decay} are used as given.
#'   \item \code{"auto"}: the parameters are tuned by grid search. Each
#'     search range is given as \code{c(min, max, step)} and expands to
#'     \code{seq(min, max, by = step)}; for example
#'     \code{Lag_Range = c(1, 12, 1)} gives the lags 1 to 12 and
#'     \code{Size_Range = c(2, 8, 2)} gives 2, 4, 6, 8 hidden neurons. If
#'     \code{max} is not reached exactly by the steps, the last value is the
#'     largest one below \code{max}. Use \code{min = max} for a single value.
#' }
#' When both are \code{"auto"} every combination of lag and network
#' parameters is evaluated. With \code{Verbose = TRUE} (default) the search
#' prints the values of every parameter, the number of combinations, and a
#' progress bar with the best score so far, the elapsed time and the
#' estimated time remaining.
#'
#' \strong{Validation.} Candidates are compared on the last
#' \code{Val_Split} share of the training set. With \code{"holdout"} the
#' model is fitted on the observations before that block and evaluated on
#' the block. With \code{"tscv"} the block is divided into \code{Folds}
#' consecutive parts and each part is predicted from a model fitted on all
#' earlier observations (expanding window). The clustering and scaling are
#' re-estimated on each fitting window. After tuning, the model is refitted
#' on the whole training set with the best parameters.
#'
#' \strong{Output.} With \code{Verbose = TRUE} a summary of the selected
#' parameters and the accuracy is shown as a message at the end (silence it
#' with \code{Verbose = FALSE} or \code{suppressMessages()}), and the result is
#' returned invisibly; assign it (e.g. \code{Model <- IFANN(...)}) to keep
#' it.
#'
#' \strong{Reproducibility.} The clustering starts from a random membership
#' matrix and the network weights are initialised at random, so call
#' \code{set.seed()} before \code{IFANN()} to obtain reproducible results.
#'
#' @param Data Numeric time series (vector or \code{ts}) without missing
#'   values.
#' @param Class Number of intuitionistic fuzzy clusters (at least 2).
#' @param Split Proportion of the data used for training, strictly between 0
#'   and 1.
#' @param Lag_Tuning \code{"auto"} (default) to tune the lag order over
#'   \code{Lag_Range}, or \code{"manual"} to use \code{Lag}.
#' @param Lag Lag order used when \code{Lag_Tuning = "manual"} (a single
#'   positive integer).
#' @param Lag_Range Lag search range \code{c(min, max, step)} for
#'   \code{Lag_Tuning = "auto"} (positive integers). Lags too large for the
#'   data are dropped with a warning.
#' @param ANN_Tuning \code{"auto"} (default) to tune \code{Size} and
#'   \code{Decay}, or \code{"manual"} to use the single values given.
#' @param Size,Decay Number of hidden neurons (a positive integer) and
#'   weight decay (non-negative), used when \code{ANN_Tuning = "manual"}.
#' @param Size_Range,Decay_Range Search ranges \code{c(min, max, step)} for
#'   \code{ANN_Tuning = "auto"}. \code{Size} values must be positive
#'   integers and \code{Decay} values non-negative.
#' @param Maxit Maximum number of training iterations of each network.
#' @param Repeats Number of networks fitted from different random starting
#'   weights; their predictions are averaged.
#' @param Validation \code{"holdout"} (default) or \code{"tscv"}
#'   (expanding-window time-series cross-validation).
#' @param Val_Split Share of the training set used for validation, strictly
#'   between 0 and 1.
#' @param Folds Number of folds for \code{Validation = "tscv"}.
#' @param Metric Criterion minimised during tuning: \code{"RMSE"} (default),
#'   \code{"MAE"} or \code{"MAPE"}.
#' @param Forecast_Type \code{"one-step"} (default): every test value is
#'   predicted from the observed previous values. \code{"recursive"}: the
#'   test period is forecast multi-step ahead, feeding predictions back as
#'   lags. The same type is used during validation.
#' @param m Fuzzifier of the clustering (greater than 1).
#' @param Alpha Exponent of the Yager non-membership generator, in (0, 1].
#' @param Max_Iter,Tol Maximum number of iterations and convergence
#'   tolerance of the clustering.
#' @param Verbose Logical; show the grid-search progress and the result
#'   summary as messages (default \code{TRUE}).
#'
#' @return A list (returned invisibly) with components
#' \describe{
#'   \item{\code{Accuracy}}{Matrix of RMSE, MAE and MAPE (\%) for the
#'     training and test sets.}
#'   \item{\code{Train_Fitted}}{Fitted values for the training targets
#'     (the first \code{Lag} training observations have no fitted value).}
#'   \item{\code{Test_Prediction}}{Predictions for the test set.}
#'   \item{\code{Train_Actual}, \code{Test_Actual}}{The corresponding
#'     observed values.}
#'   \item{\code{Best_Params}}{List with \code{Lag}, \code{Size} and
#'     \code{Decay} of the final model.}
#'   \item{\code{Architecture}}{Number of inputs, hidden neurons and
#'     outputs, number of weights per network, \code{Maxit}, \code{Repeats}
#'     and \code{Converged}, the number of networks whose optimiser stopped
#'     before reaching \code{Maxit}.}
#'   \item{\code{Tuning}}{\code{NULL} when nothing was tuned; otherwise a
#'     list with the validation scheme, metric, number of combinations,
#'     best validation score, elapsed time and \code{History} (a data frame
#'     with the score of every combination).}
#'   \item{\code{Networks}}{List of the \code{Repeats} fitted
#'     \code{\link[nnet]{nnet}} objects of the final model (fitted to the
#'     target scaled to [0, 1]).}
#'   \item{\code{Scaling}}{Minimum and maximum of the training data used for
#'     the [0, 1] scaling.}
#'   \item{\code{IFCM}}{The intuitionistic fuzzy clustering of the training
#'     set (centres, membership, non-membership and hesitation degrees).}
#'   \item{\code{Settings}}{Other settings used.}
#' }
#'
#' @references
#' Yolcu, O.C. and Yolcu, U. (2023). A novel intuitionistic fuzzy time
#' series prediction model with cascaded structure for financial time
#' series. \emph{Expert Systems with Applications}, 215, 119336.
#' \doi{10.1016/j.eswa.2022.119336}
#'
#' Chaira, T. (2011). A novel intuitionistic fuzzy C means clustering
#' algorithm and its application to medical images. \emph{Applied Soft
#' Computing}, 11(2), 1711--1717.
#'
#' Venables, W.N. and Ripley, B.D. (2002). \emph{Modern Applied Statistics
#' with S}. Fourth edition. Springer, New York.
#'
#' @examples
#' set.seed(123)
#' dat <- 50 + 10 * sin(2 * pi * (1:120) / 12) + rnorm(120)
#'
#' # Everything manual: lag 12, 4 hidden neurons, weight decay 0.05
#' Model1 <- IFANN(dat, Class = 3, Split = 0.8,
#'                 Lag_Tuning = "manual", Lag = 12,
#'                 ANN_Tuning = "manual", Size = 4, Decay = 0.05,
#'                 Repeats = 3)
#' Model1$Accuracy
#'
#' # Manual lag, network parameters tuned on a very small grid
#' Model2 <- IFANN(dat, Class = 3, Split = 0.8,
#'                 Lag_Tuning = "manual", Lag = 12,
#'                 Size_Range = c(2, 4, 2),           # 2, 4
#'                 Decay_Range = c(0.01, 0.09, 0.08), # 0.01, 0.09
#'                 Repeats = 2, Verbose = FALSE)
#' Model2$Best_Params
#'
#' \donttest{
#' # Lag and network parameters tuned jointly (small grid)
#' Model3 <- IFANN(dat, Class = 3, Split = 0.8,
#'                 Lag_Range = c(2, 12, 5),           # 2, 7, 12
#'                 Size_Range = c(2, 4, 2),           # 2, 4
#'                 Decay_Range = c(0.05, 0.05, 0),    # 0.05
#'                 Repeats = 2, Verbose = FALSE)
#' head(Model3$Tuning$History)
#' }
#'
#' @importFrom nnet nnet
#' @importFrom stats predict runif
#' @importFrom utils modifyList
#' @export
IFANN <- function(Data, Class = 3, Split = 0.8,
                  Lag_Tuning = c("auto", "manual"),
                  Lag = 1,
                  Lag_Range = c(min = 1, max = 12, step = 1),
                  ANN_Tuning = c("auto", "manual"),
                  Size = 4, Decay = 0.01,
                  Size_Range = c(min = 2, max = 8, step = 2),
                  Decay_Range = c(min = 0.01, max = 0.09, step = 0.04),
                  Maxit = 200, Repeats = 5,
                  Validation = c("holdout", "tscv"),
                  Val_Split = 0.2, Folds = 3,
                  Metric = c("RMSE", "MAE", "MAPE"),
                  Forecast_Type = c("one-step", "recursive"),
                  m = 2, Alpha = 0.85, Max_Iter = 100, Tol = 1e-5,
                  Verbose = TRUE) {

  call <- match.call()

  # ---- argument checks ---------------------------------------------------
  Lag_Tuning <- match.arg(Lag_Tuning)
  ANN_Tuning <- match.arg(ANN_Tuning)
  Validation <- match.arg(Validation)
  Metric <- match.arg(Metric)
  Forecast_Type <- match.arg(Forecast_Type)

  if (!is.numeric(Data)) stop("Data must be numeric")
  if (anyNA(Data)) stop("Data must not contain missing values")
  y <- as.numeric(Data)
  n <- length(y)

  .is_count <- function(x, min = 1) {
    is.numeric(x) && length(x) == 1L && !is.na(x) && x >= min && x == round(x)
  }

  if (!.is_count(Class, 2)) stop("Class must be an integer >= 2")
  Class <- as.integer(Class)
  if (!is.numeric(Split) || length(Split) != 1L || Split <= 0 || Split >= 1) {
    stop("Split must be between 0 and 1")
  }
  if (!is.numeric(m) || length(m) != 1L || m <= 1) stop("m must be greater than 1")
  if (!is.numeric(Alpha) || length(Alpha) != 1L || Alpha <= 0 || Alpha > 1) {
    stop("Alpha must be in (0, 1]")
  }
  if (!is.numeric(Val_Split) || length(Val_Split) != 1L || Val_Split <= 0 || Val_Split >= 1) {
    stop("Val_Split must be between 0 and 1")
  }
  if (!.is_count(Folds, 1)) stop("Folds must be a positive integer")
  if (!.is_count(Maxit, 1)) stop("Maxit must be a positive integer")
  if (!.is_count(Repeats, 1)) stop("Repeats must be a positive integer")
  Maxit <- as.integer(Maxit)
  Repeats <- as.integer(Repeats)
  if (!is.logical(Verbose) || length(Verbose) != 1L || is.na(Verbose)) {
    stop("Verbose must be TRUE or FALSE")
  }

  ntrain <- as.integer(round(n * Split))
  if (ntrain < 1 || ntrain >= n) {
    stop("Data is too short: training and test sets must both be non-empty")
  }

  if (Lag_Tuning == "manual") {
    if (!.is_count(Lag, 1)) stop("Lag must be a single positive integer")
    Lag <- as.integer(Lag)
  } else {
    lag_values <- .range_values(Lag_Range, "Lag_Range", integer = TRUE)
  }

  if (ANN_Tuning == "manual") {
    if (!.is_count(Size, 1)) stop("Size must be a single positive integer")
    if (!is.numeric(Decay) || length(Decay) != 1L || is.na(Decay) || Decay < 0) {
      stop("Decay must be a single non-negative number")
    }
    Size <- as.integer(Size)
  } else {
    size_values <- .range_values(Size_Range, "Size_Range", integer = TRUE)
    decay_values <- .range_values(Decay_Range, "Decay_Range", allow_zero = TRUE)
  }

  if (Metric == "MAPE" && any(y[seq_len(ntrain)] == 0)) {
    stop("Metric = \"MAPE\" cannot be used when the training data contain zeros")
  }

  tuning_needed <- Lag_Tuning == "auto" || ANN_Tuning == "auto"
  min_rows <- 5L

  # ---- grid-search tuning ------------------------------------------------
  tuning <- NULL
  best <- list(
    Lag = if (Lag_Tuning == "manual") Lag else NA_integer_,
    Size = Size,
    Decay = Decay
  )

  if (tuning_needed) {
    folds <- .make_folds(ntrain, Validation, Val_Split, Folds)
    first_end <- min(vapply(folds, `[[`, integer(1), "fit_end"))
    max_lag <- first_end - min_rows

    if (Lag_Tuning == "auto") {
      keep <- lag_values[lag_values <= max_lag]
      if (length(keep) == 0L) {
        stop("The training set is too short for tuning even with lag ",
             min(lag_values), "; use more data, a larger Split or a smaller Val_Split")
      }
      if (length(keep) < length(lag_values)) {
        warning("Lags larger than ", max_lag,
                " are too long for the validation window and were dropped from Lag_Range")
      }
      lag_values <- keep
    } else if (Lag > max_lag) {
      stop("Lag = ", Lag, " is too large for tuning on this training set (maximum ",
           max(max_lag, 0), ")")
    }

    preps <- lapply(folds, function(fo) {
      .prepare(y[seq_len(fo$fit_end)], Class, m, Alpha, Max_Iter, Tol)
    })

    fixed <- list()
    if (Lag_Tuning == "manual") fixed$Lag <- Lag
    if (ANN_Tuning == "manual") {
      fixed$Size <- Size
      fixed$Decay <- Decay
    }

    gv <- list()
    if (Lag_Tuning == "auto") gv$Lag <- lag_values
    if (ANN_Tuning == "auto") {
      gv$Size <- size_values
      gv$Decay <- decay_values
    }

    objective <- .make_objective(y, folds, preps, fixed, Metric, Forecast_Type,
                                 Maxit, Repeats)
    header <- function() {
      .grid_header(gv, fixed, folds, ntrain, Validation, Metric,
                   list(Repeats = Repeats, Maxit = Maxit))
    }

    t0 <- Sys.time()
    history <- .grid_search(gv, objective, Verbose, Metric, header)
    elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

    if (!any(is.finite(history$Score))) {
      stop("Every combination failed during tuning; check the search ranges")
    }
    b <- which.min(history$Score)
    for (nm in names(gv)) best[[nm]] <- history[[nm]][b]
    for (nm in names(fixed)) best[[nm]] <- fixed[[nm]]

    tuning <- list(
      Method = "grid",
      Validation = Validation,
      Metric = Metric,
      Combinations = nrow(history),
      Best_Score = history$Score[b],
      Elapsed_Seconds = elapsed,
      History = history
    )
  }

  best$Lag <- as.integer(best$Lag)
  best$Size <- as.integer(best$Size)
  L <- best$Lag
  if (ntrain - L < min_rows) {
    stop("Lag = ", L, " leaves fewer than ", min_rows,
         " training observations; use a smaller lag or more data")
  }

  # ---- final model -------------------------------------------------------
  prep <- .prepare(y[seq_len(ntrain)], Class, m, Alpha, Max_Iter, Tol)
  train_idx <- (L + 1L):ntrain
  test_idx <- (ntrain + 1L):n

  fit <- .fit_ann(y, train_idx, L, prep, best, Maxit, Repeats)

  train_fitted <- .predict_ann(fit, y, train_idx, L, prep, "one-step")
  test_pred <- .predict_ann(fit, y, test_idx, L, prep, Forecast_Type)

  acc <- rbind(
    "Training set" = .metrics(y[train_idx], train_fitted),
    "Test set" = .metrics(y[test_idx], test_pred)
  )

  converged <- vapply(fit$nets, function(net) net$convergence == 0L, logical(1))

  result <- list(
    Accuracy = acc,
    Train_Fitted = train_fitted,
    Test_Prediction = test_pred,
    Train_Actual = y[train_idx],
    Test_Actual = y[test_idx],
    Best_Params = best[c("Lag", "Size", "Decay")],
    Architecture = list(
      Inputs = fit$n_inputs,
      Hidden = fit$size,
      Outputs = 1L,
      Weights = .n_weights(fit$n_inputs, fit$size),
      Maxit = Maxit,
      Repeats = Repeats,
      Converged = sum(converged)
    ),
    Tuning = tuning,
    Networks = fit$nets,
    Scaling = c(min = prep$scaling$lo, max = prep$scaling$hi),
    IFCM = prep$ifcm,
    Settings = list(
      Class = Class, Split = Split,
      Lag_Tuning = Lag_Tuning, ANN_Tuning = ANN_Tuning,
      Forecast_Type = Forecast_Type, m = m, Alpha = Alpha,
      n_train = ntrain, n_test = n - ntrain
    ),
    Call = call
  )

  if (Verbose) .show_summary(result)

  invisible(result)
}


# Summary shown at the end of IFANN() when Verbose = TRUE (internal).
# It is sent as a message, so it can be silenced with suppressMessages().
.show_summary <- function(x) {
  s <- x$Settings
  a <- x$Architecture
  bp <- x$Best_Params
  lines <- c(
    "Intuitionistic Fuzzy Artificial Neural Network (IFANN)",
    "",
    paste0("Clusters: ", s$Class, "   Train/Test: ", s$n_train, " / ", s$n_test,
           "   Forecast: ", s$Forecast_Type),
    sprintf("Network: %d-%d-1 (%d weights), %d network%s averaged, Maxit = %d (%d of %d stopped at Maxit)",
            a$Inputs, a$Hidden, a$Weights, a$Repeats,
            if (a$Repeats == 1L) "" else "s", a$Maxit,
            a$Repeats - a$Converged, a$Repeats),
    paste0("Lag tuning: ", s$Lag_Tuning, "   ANN tuning: ", s$ANN_Tuning)
  )
  if (!is.null(x$Tuning)) {
    lines <- c(lines, paste0(
      "Grid search: ", format(x$Tuning$Combinations, big.mark = ","),
      " combinations   Validation: ", x$Tuning$Validation, "   ",
      x$Tuning$Metric, " = ", format(x$Tuning$Best_Score, digits = 5)))
  }
  lines <- c(
    lines, "", "Selected parameters:",
    paste0("  Lag   = ", bp$Lag),
    paste0("  Size  = ", bp$Size, " hidden neurons"),
    paste0("  Decay = ", format(bp$Decay, digits = 5)),
    "", "Accuracy:",
    utils::capture.output(print(round(x$Accuracy, 4)))
  )
  message(paste(lines, collapse = "\n"))
  invisible(NULL)
}
