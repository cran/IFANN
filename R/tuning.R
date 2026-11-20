# Grid-search tuning: search ranges, validation folds, objective, progress.


# Expand a c(min, max, step) range into the grid values.
# Names min/max/step are optional; without names the order is min, max, step.
#   integer    values must be positive integers
#   allow_zero values must be non-negative (otherwise positive)
.range_values <- function(r, name, integer = FALSE, allow_zero = FALSE) {
  if (!is.numeric(r) || length(r) != 3L || anyNA(r)) {
    stop(name, " must be a numeric vector c(min, max, step)")
  }
  if (!is.null(names(r)) && all(c("min", "max", "step") %in% names(r))) {
    r <- r[c("min", "max", "step")]
  }
  lo <- r[[1]]
  hi <- r[[2]]
  st <- r[[3]]

  if (hi < lo) stop(name, ": max must be greater than or equal to min")
  if (lo == hi) {
    v <- lo
  } else {
    if (!(st > 0)) stop(name, ": step must be positive")
    v <- seq(lo, hi, by = st)
  }
  v <- unique(round(v, 10))

  if (integer) {
    if (any(v != round(v)) || lo < 1) {
      stop(name, ": values must be positive integers (use an integer min and step)")
    }
    v <- as.integer(v)
  } else if (allow_zero) {
    if (lo < 0) stop(name, ": values must be non-negative")
  } else if (!(lo > 0)) {
    stop(name, ": values must be positive")
  }
  v
}


# Validation folds inside the training window (time ordered).
# Each fold: fit on targets <= fit_end, validate on targets in idx.
.make_folds <- function(ntrain, Validation, Val_Split, Folds) {
  ntrain <- as.integer(ntrain)
  n_val <- as.integer(max(1, round(ntrain * Val_Split)))
  val_start <- ntrain - n_val + 1L
  if (Validation == "holdout") {
    return(list(list(fit_end = val_start - 1L, idx = val_start:ntrain)))
  }
  Folds <- min(Folds, n_val)
  blocks <- split(val_start:ntrain, cut(seq_len(n_val), Folds, labels = FALSE))
  lapply(unname(blocks), function(b) list(fit_end = as.integer(min(b) - 1L), idx = b))
}


# Objective: validation error of a parameter set (pooled over folds)
.make_objective <- function(y, folds, preps, fixed, Metric, Forecast_Type,
                            Maxit, Repeats) {
  function(vals) {
    p <- utils::modifyList(fixed, vals)
    L <- as.integer(p$Lag)
    res <- tryCatch({
      act <- numeric(0)
      prd <- numeric(0)
      for (f in seq_along(folds)) {
        fo <- folds[[f]]
        fit <- .fit_ann(y, (L + 1L):fo$fit_end, L, preps[[f]], p, Maxit, Repeats)
        prd <- c(prd, .predict_ann(fit, y, fo$idx, L, preps[[f]], Forecast_Type))
        act <- c(act, y[fo$idx])
      }
      .metrics(act, prd)[[Metric]]
    }, error = function(e) Inf)
    if (is.na(res)) Inf else res
  }
}


# ---- progress display ------------------------------------------------------

.fmt_time <- function(s) {
  if (!is.finite(s)) return("--")
  if (s < 60) return(sprintf("%.1fs", s))
  if (s < 3600) return(sprintf("%dm %02ds", as.integer(s %/% 60), as.integer(round(s %% 60))))
  sprintf("%dh %02dm", as.integer(s %/% 3600), as.integer((s %% 3600) %/% 60))
}

.fmt_values <- function(v) {
  f <- if (is.numeric(v)) {
    format(v, digits = 6, trim = TRUE, scientific = FALSE, drop0trailing = TRUE)
  } else {
    as.character(v)
  }
  if (length(f) <= 8) return(paste(f, collapse = ", "))
  paste(c(f[1:3], "...", f[(length(f) - 1):length(f)]), collapse = ", ")
}

.fmt_num <- function(x) {
  if (!is.finite(x)) return("--")
  format(signif(x, 5), scientific = FALSE, drop0trailing = TRUE)
}

.grid_header <- function(grid_vals, fixed, folds, ntrain, Validation, Metric, extra) {
  total <- prod(lengths(grid_vals))
  all_names <- c("Lag", "Size", "Decay")
  w <- max(nchar(c(all_names, "Total")))
  message("\nIFANN grid search\n", appendLF = FALSE)
  for (nm in all_names) {
    if (!is.null(grid_vals[[nm]])) {
      v <- grid_vals[[nm]]
      message(sprintf("  %-*s : %s  (%d value%s)\n", w, nm, .fmt_values(v),
                  length(v), if (length(v) == 1L) "" else "s"), appendLF = FALSE)
    } else if (!is.null(fixed[[nm]])) {
      message(sprintf("  %-*s : %s  (manual)\n", w, nm, .fmt_values(fixed[[nm]])), appendLF = FALSE)
    }
  }
  message(sprintf("  %-*s : %s%s combination%s\n", w, "Total",
              if (length(grid_vals) > 1L) {
                paste0(paste(lengths(grid_vals), collapse = " x "), " = ")
              } else "",
              format(total, big.mark = ","), if (total == 1) "" else "s"), appendLF = FALSE)
  n_val <- sum(vapply(folds, function(f) length(f$idx), integer(1)))
  if (Validation == "holdout") {
    message(sprintf("  Validation: hold-out, last %d of %d training observations, metric %s\n",
                n_val, ntrain, Metric), appendLF = FALSE)
  } else {
    message(sprintf("  Validation: time-series CV, %d folds over the last %d of %d training observations, metric %s\n",
                length(folds), n_val, ntrain, Metric), appendLF = FALSE)
  }
  message(sprintf("  Networks: %d averaged per fit, up to %d iterations each\n\n",
              extra$Repeats, extra$Maxit), appendLF = FALSE)
}

.progress_line <- function(i, total, best, Metric, start) {
  el <- as.numeric(difftime(Sys.time(), start, units = "secs"))
  eta <- if (i > 0) el / i * (total - i) else NA_real_
  width <- 30L
  filled <- as.integer(round(width * i / total))
  bar <- paste0(strrep("=", filled), if (filled < width) ">" else "",
                strrep(" ", max(0L, width - filled - 1L)))
  message(sprintf("\r  [%s] %3d%%  %s/%s  best %s = %s  elapsed %s  ETA %s   ",
              bar, as.integer(floor(100 * i / total)),
              format(i, big.mark = ","), format(total, big.mark = ","),
              Metric, .fmt_num(best), .fmt_time(el), .fmt_time(eta)), appendLF = FALSE)
}


.grid_search <- function(grid_vals, objective, verbose = TRUE, Metric = "RMSE",
                         header = NULL) {
  grid <- expand.grid(grid_vals, KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
  total <- nrow(grid)
  scores <- numeric(total)
  best <- Inf
  start <- Sys.time()
  every <- max(1L, total %/% 200L)
  last_draw <- start

  if (verbose) {
    if (is.function(header)) header()
    .progress_line(0L, total, best, Metric, start)
  }

  for (i in seq_len(total)) {
    scores[i] <- objective(as.list(grid[i, , drop = FALSE]))
    if (scores[i] < best) best <- scores[i]
    if (verbose) {
      now <- Sys.time()
      if (i == total || i %% every == 0L ||
          as.numeric(difftime(now, last_draw, units = "secs")) > 0.5) {
        .progress_line(i, total, best, Metric, start)
        last_draw <- now
      }
    }
  }

  if (verbose) {
    message("")
    el <- as.numeric(difftime(Sys.time(), start, units = "secs"))
    if (is.finite(best)) {
      b <- which.min(scores)
      pr <- vapply(names(grid), function(nm) {
        paste0(nm, " = ", .fmt_num(grid[[nm]][b]))
      }, character(1))
      message(sprintf("  Done: %s combinations in %s. Best %s = %s at %s\n\n",
                  format(total, big.mark = ","), .fmt_time(el), Metric,
                  .fmt_num(best), paste(pr, collapse = ", ")),
              appendLF = FALSE)
    } else {
      message(sprintf("  Done: %s combinations in %s. No candidate could be fitted.\n\n",
                  format(total, big.mark = ","), .fmt_time(el)),
              appendLF = FALSE)
    }
  }

  cbind(Combination = seq_len(total), grid, Score = scores)
}
