sim <- function(n = 100, seed = 1) {
  set.seed(seed)
  50 + 10 * sin(2 * pi * seq_len(n) / 12) + rnorm(n)
}

# small, fast networks for the tests
fast <- list(Size_Range = c(2, 2, 0), Decay_Range = c(0.05, 0.05, 0),
             Maxit = 50, Repeats = 2)

# ---- exports ---------------------------------------------------------------

test_that("IFANN is the only exported function and no print method is registered", {
  expect_identical(getNamespaceExports("IFANN"), "IFANN")
  expect_null(getS3method("print", "IFANN", optional = TRUE))
})

# ---- input validation ------------------------------------------------------

test_that("IFANN validates its inputs", {
  x <- sim()
  expect_error(IFANN(letters[1:20]), "Data must be numeric")
  expect_error(IFANN(c(x[1:49], NA)), "missing values")
  expect_error(IFANN(x, Class = 1), "Class must be")
  expect_error(IFANN(x, Split = 1), "Split must be between 0 and 1")
  expect_error(IFANN(x, Split = 0), "Split must be between 0 and 1")
  expect_error(IFANN(c(1, 2, 3), Class = 2, Split = 0.99), "too short")
  expect_error(IFANN(x, Lag_Tuning = "manual", Lag = 0), "Lag must be")
  expect_error(IFANN(x, Lag_Tuning = "manual", Lag = 2.5), "Lag must be")
  expect_error(IFANN(x, ANN_Tuning = "manual", Size = 0), "Size must be")
  expect_error(IFANN(x, ANN_Tuning = "manual", Size = 2.5), "Size must be")
  expect_error(IFANN(x, ANN_Tuning = "manual", Decay = -1), "Decay must be")
  expect_error(IFANN(x, Maxit = 0), "Maxit must be")
  expect_error(IFANN(x, Repeats = 0), "Repeats must be")
  expect_error(IFANN(x, Alpha = 1.5), "Alpha must be")
  expect_error(IFANN(x, m = 1), "m must be")
  expect_error(IFANN(c(0, x), Metric = "MAPE"), "zeros")
  expect_error(IFANN(x, Verbose = NA), "Verbose")
})

test_that("search ranges must be c(min, max, step)", {
  x <- sim()
  expect_error(IFANN(x, Lag_Range = 1:12), "c\\(min, max, step\\)")
  expect_error(IFANN(x, Lag_Range = c(0, 5, 1)), "positive integers")
  expect_error(IFANN(x, Size_Range = c(1, 5, 0.5)), "positive integers")
  expect_error(IFANN(x, Size_Range = c(8, 2, 2)), "max must be")
  expect_error(IFANN(x, Decay_Range = c(0.01, 0.1, 0)), "step must be positive")
  expect_error(IFANN(x, Decay_Range = c(-0.1, 0.1, 0.1)), "non-negative")
})

test_that("ranges expand like seq(min, max, by = step)", {
  rv <- IFANN:::.range_values
  expect_equal(rv(c(1, 12, 1), "L", integer = TRUE), 1:12)
  expect_equal(rv(c(2, 8, 2), "S", integer = TRUE), c(2L, 4L, 6L, 8L))
  expect_equal(rv(c(1, 10, 4), "S", integer = TRUE), c(1L, 5L, 9L))   # max not hit
  expect_equal(rv(c(0.01, 0.09, 0.04), "D", allow_zero = TRUE), c(0.01, 0.05, 0.09))
  expect_equal(rv(c(0, 0.2, 0.1), "D", allow_zero = TRUE), c(0, 0.1, 0.2))
  expect_equal(rv(c(5, 5, 0), "S", integer = TRUE), 5L)              # single value
  expect_equal(rv(c(step = 2, min = 1, max = 5), "S", integer = TRUE), c(1L, 3L, 5L))
})

# ---- clustering ------------------------------------------------------------

test_that("IFCM returns valid intuitionistic fuzzy degrees", {
  set.seed(42)
  res <- IFANN:::.ifcm(rnorm(60, 60, 10), Class = 3, alpha = 0.85)
  expect_equal(dim(res$membership), c(60L, 3L))
  expect_equal(unname(rowSums(res$membership)), rep(1, 60), tolerance = 1e-8)
  expect_true(all(res$hesitation >= -1e-12))
  expect_true(any(res$hesitation > 1e-3))
  expect_false(is.unsorted(res$centers))
  expect_true(res$converged)
})

test_that("Alpha = 1 gives ordinary fuzzy sets", {
  set.seed(3)
  res <- IFANN:::.ifcm(rnorm(40), Class = 2, alpha = 1)
  expect_equal(res$nonmembership, 1 - res$membership)
})

# ---- features: no look-ahead -----------------------------------------------

test_that("features of a target use only past values", {
  set.seed(9)
  y <- rnorm(40, 10)
  prep <- IFANN:::.prepare(y[1:30], 2, 2, 0.85, 100, 1e-5)
  X1 <- IFANN:::.feature_matrix(y, 31:35, 3, prep)
  y2 <- y
  y2[31:35] <- 999
  X2 <- IFANN:::.feature_matrix(y2, 31, 3, prep)
  expect_equal(X1[1, ], X2[1, ])
  expect_equal(ncol(X1), 3 * (1 + 2 * 2))
})

test_that("predictions are the average of the repeated networks", {
  set.seed(10)
  y <- sim(60)
  prep <- IFANN:::.prepare(y[1:50], 2, 2, 0.85, 100, 1e-5)
  fit <- IFANN:::.fit_ann(y, 4:50, 3, prep, list(Size = 2, Decay = 0.01), 50, 3)
  expect_length(fit$nets, 3)
  X <- IFANN:::.feature_matrix(y, 51:55, 3, prep)
  each <- sapply(fit$nets, function(net) as.numeric(predict(net, X)))
  expect_equal(IFANN:::.predict_scaled(fit, X), rowMeans(each))
  expect_equal(IFANN:::.predict_scaled(fit, X[1, , drop = FALSE]), mean(each[1, ]))
})

# ---- model fitting ---------------------------------------------------------

test_that("fully manual model: shapes, invisible return, silent when Verbose = FALSE", {
  x <- sim(100)
  set.seed(7)
  expect_silent(
    res <- IFANN(x, Class = 2, Split = 0.8,
                 Lag_Tuning = "manual", Lag = 4,
                 ANN_Tuning = "manual", Size = 3, Decay = 0.02,
                 Maxit = 60, Repeats = 2, Verbose = FALSE)
  )
  expect_type(res, "list")
  expect_null(attr(res, "class"))
  expect_null(res$Tuning)
  expect_length(res$Train_Fitted, 80 - 4)
  expect_length(res$Test_Prediction, 20)
  expect_equal(res$Test_Actual, x[81:100])
  expect_true(all(is.finite(res$Test_Prediction)))
  expect_equal(dim(res$Accuracy), c(2L, 3L))
  expect_equal(res$Best_Params, list(Lag = 4L, Size = 3L, Decay = 0.02))
  expect_length(res$Networks, 2)
  expect_s3_class(res$Networks[[1]], "nnet")
  expect_equal(res$Architecture$Inputs, 4L * (1L + 2L * 2L))
  expect_equal(res$Architecture$Weights, (20L + 1L) * 3L + 3L + 1L)
  expect_equal(length(res$Networks[[1]]$wts), res$Architecture$Weights)

  expect_invisible(IFANN(x, Class = 2, Lag_Tuning = "manual", Lag = 2,
                         ANN_Tuning = "manual", Size = 2, Maxit = 20, Repeats = 1,
                         Verbose = FALSE))
})

test_that("the summary is shown as a message when Verbose = TRUE", {
  x <- sim(100)
  set.seed(8)
  msgs <- testthat::capture_messages(
    IFANN(x, Class = 2, Lag_Tuning = "manual", Lag = 3,
          ANN_Tuning = "manual", Size = 2, Maxit = 30, Repeats = 2)
  )
  txt <- paste(msgs, collapse = "")
  expect_match(txt, "Intuitionistic Fuzzy Artificial Neural Network")
  expect_match(txt, "Network: 15-2-1")
  expect_match(txt, "Selected parameters")
  expect_match(txt, "Test set")
})

test_that("grid search tunes lag and network parameters jointly", {
  x <- sim(100)
  set.seed(2)
  res <- IFANN(x, Class = 2,
               Lag_Range = c(1, 11, 5),          # 1, 6, 11
               Size_Range = c(2, 4, 2),          # 2, 4
               Decay_Range = c(0, 0.1, 0.1),     # 0, 0.1
               Maxit = 50, Repeats = 2, Verbose = FALSE)
  h <- res$Tuning$History
  expect_equal(nrow(h), 3 * 2 * 2)
  expect_equal(res$Tuning$Combinations, 12)
  expect_equal(sort(unique(h$Lag)), c(1L, 6L, 11L))
  expect_equal(res$Tuning$Best_Score, min(h$Score))
  b <- h[which.min(h$Score), ]
  expect_equal(res$Best_Params$Lag, as.integer(b$Lag))
  expect_equal(res$Best_Params$Size, as.integer(b$Size))
  expect_equal(res$Best_Params$Decay, b$Decay)
})

test_that("the progress display shows the grid and the number of combinations", {
  x <- sim(100)
  set.seed(2)
  msgs <- testthat::capture_messages(
    res <- do.call(IFANN, c(list(Data = x, Class = 2, Lag_Range = c(1, 3, 1)), fast))
  )
  txt <- paste(msgs, collapse = "")
  expect_match(txt, "IFANN grid search")
  expect_match(txt, "3 x 1 x 1 = 3 combinations")
  expect_match(txt, "Networks: 2 averaged per fit, up to 50 iterations each")
  expect_match(txt, "3/3")
  expect_match(txt, "100%")
  expect_match(txt, "Done: 3 combinations")
  # messages can be silenced and nothing goes to standard output
  expect_silent(suppressMessages(
    do.call(IFANN, c(list(Data = x, Class = 2, Lag_Range = c(1, 3, 1)), fast))))
})

test_that("manual lag with tuned network, and tuned lag with manual network", {
  x <- sim(100)
  set.seed(3)
  r1 <- IFANN(x, Class = 2, Lag_Tuning = "manual", Lag = 12,
              Size_Range = c(2, 3, 1), Decay_Range = c(0.05, 0.05, 0),
              Maxit = 40, Repeats = 1, Verbose = FALSE)
  expect_equal(r1$Best_Params$Lag, 12L)
  expect_false("Lag" %in% names(r1$Tuning$History))

  r2 <- IFANN(x, Class = 2, ANN_Tuning = "manual", Size = 3, Decay = 0.02,
              Lag_Range = c(1, 4, 1), Maxit = 40, Repeats = 1, Verbose = FALSE)
  expect_equal(r2$Best_Params$Size, 3L)
  expect_equal(r2$Best_Params$Decay, 0.02)
  expect_equal(names(r2$Tuning$History), c("Combination", "Lag", "Score"))
})

test_that("results are reproducible after set.seed()", {
  skip_on_cran()
  x <- sim(90)
  args <- c(list(Data = x, Class = 2, Lag_Range = c(1, 4, 1), Verbose = FALSE), fast)
  set.seed(99)
  r1 <- do.call(IFANN, args)
  set.seed(99)
  r2 <- do.call(IFANN, args)
  expect_equal(r1$Test_Prediction, r2$Test_Prediction)
  expect_equal(r1$Tuning$History, r2$Tuning$History)
})

test_that("time-series cross-validation and recursive forecasts", {
  skip_on_cran()
  x <- sim(100)
  set.seed(5)
  res <- do.call(IFANN, c(list(Data = x, Class = 2, Validation = "tscv", Folds = 3,
                               Forecast_Type = "recursive",
                               Lag_Range = c(2, 12, 10), Verbose = FALSE), fast))
  expect_length(res$Test_Prediction, 20)
  expect_true(all(is.finite(res$Test_Prediction)))
  expect_equal(res$Tuning$Validation, "tscv")
})

test_that("too long lags are dropped with a warning", {
  x <- sim(40)
  set.seed(6)
  expect_warning(
    res <- IFANN(x, Class = 2, ANN_Tuning = "manual", Size = 2, Maxit = 30,
                 Repeats = 1, Lag_Range = c(2, 30, 28), Verbose = FALSE),
    "dropped"
  )
  expect_equal(res$Best_Params$Lag, 2L)
})
