#!/usr/bin/env Rscript

# Diel--phenology scale audit for the frozen carbon-only K = 4 HMM.
#
# Purpose: determine whether the half-hourly hidden states contain phenological
# information beyond diel timing and irradiance. The HMM is never refitted here:
# saved posterior probabilities and canonical Viterbi labels are treated as
# immutable inputs. Outputs provide Figure 2 inputs and the half-hourly and
# daily light/dark cross-year gains shown in Figure S2.

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(patchwork)
  library(scales)
  library(splines)
})

set.seed(20260812)

args <- commandArgs(trailingOnly = FALSE)
script_arg <- grep("^--file=", args, value = TRUE)
root <- if (length(script_arg)) {
  normalizePath(file.path(dirname(sub("^--file=", "", script_arg[1])), "../.."), mustWork = TRUE)
} else normalizePath(".", mustWork = TRUE)
input_k4 <- file.path(root, "data/processed/stjones_hmm_k4_primary_v1.csv.gz")
input_contract <- file.path(root, "data/processed/stjones_halfhourly_contract_v1.csv.gz")
input_fingerprints <- file.path(root, "outputs/complete_dataset/state_model_final/primary_k4_state_fingerprints_final.csv")
out_dir <- file.path(root, "outputs/complete_dataset/diel_phenology_audit")
fig_dir <- file.path(root, "figures/diagnostics/diel_phenology_audit")
report_path <- file.path(root, "reports/complete_dataset/diel_phenology_scale_audit.md")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

states <- paste0("C", 1:4)
post_cols <- paste0("posterior_", states)
hard_cols <- paste0("hard_", states)
phases_display <- c("Greenup", "Maturity", "Senescence", "Dormancy")
phases_model <- c("Dormancy", "Greenup", "Maturity", "Senescence")
years <- 2016:2021
state_cols <- c(C1 = "#0072B2", C2 = "#E69F00", C3 = "#009E73", C4 = "#CC79A7")

read_gz <- function(path) {
  con <- gzfile(path, open = "rt")
  on.exit(close(con))
  as.data.table(read.csv(con, check.names = FALSE, stringsAsFactors = FALSE))
}

renorm <- function(x, eps = 1e-9) {
  x <- pmin(pmax(as.matrix(x), eps), 1 - eps)
  rs <- rowSums(x)
  x / rs
}

entropy_vec <- function(p) {
  p <- p[is.finite(p) & p > 0]
  if (!length(p)) return(0)
  -sum(p * log(p))
}

contingency_metrics <- function(x, y, scope) {
  tab <- table(x, y)
  n <- sum(tab)
  prob <- tab / n
  rp <- rowSums(prob); cp <- colSums(prob)
  expected_prob <- outer(rp, cp)
  ok <- prob > 0 & expected_prob > 0
  mi <- sum(prob[ok] * log(prob[ok] / expected_prob[ok]))
  hx <- entropy_vec(rp); hy <- entropy_vec(cp)
  expected <- outer(rowSums(tab), colSums(tab)) / n
  valid <- expected > 0
  chi <- sum((tab[valid] - expected[valid])^2 / expected[valid])
  data.table(
    scope = scope,
    n_halfhours = n,
    normalized_mutual_information = if (hx > 0 && hy > 0) mi / sqrt(hx * hy) else NA_real_,
    cramers_v = sqrt(chi / (n * min(nrow(tab) - 1, ncol(tab) - 1)))
  )
}

metric_by_date <- function(observed, predicted, dates) {
  predicted <- renorm(predicted)
  observed <- as.matrix(observed)
  row_brier <- rowSums((predicted - observed)^2)
  row_mse <- rowMeans((predicted - observed)^2)
  row_log <- -rowSums(observed * log(pmax(predicted, 1e-9)))
  obs_dom <- states[max.col(observed, ties.method = "first")]
  pred_dom <- states[max.col(predicted, ties.method = "first")]
  z <- data.table(
    date = as.IDate(dates), brier = row_brier, mse = row_mse,
    log_loss = row_log, correct = as.numeric(obs_dom == pred_dom)
  )
  by_date <- z[, .(
    brier = mean(brier), mse = mean(mse), log_loss = mean(log_loss),
    dominant_accuracy = mean(correct), n_rows = .N
  ), by = date]
  list(
    summary = data.table(
      brier = mean(by_date$brier),
      log_loss = mean(by_date$log_loss),
      rmse = sqrt(mean(by_date$mse)),
      dominant_accuracy = mean(by_date$dominant_accuracy),
      n_test_rows = nrow(z), n_test_dates = nrow(by_date)
    ),
    by_date = by_date
  )
}

fit_predict_statewise <- function(train, test, response_cols, rhs) {
  pred <- matrix(NA_real_, nrow(test), length(states))
  for (j in seq_along(states)) {
    f <- as.formula(paste(response_cols[j], "~", rhs))
    fit <- suppressWarnings(glm(f, data = train, family = quasibinomial(link = "logit"),
                                control = glm.control(maxit = 60)))
    value <- suppressWarnings(predict(fit, newdata = test, type = "response"))
    if (any(!is.finite(value))) {
      value[!is.finite(value)] <- mean(train[[response_cols[j]]], na.rm = TRUE)
    }
    pred[, j] <- value
  }
  renorm(pred)
}

model_specs <- data.table(
  model_id = c("A_constant", "B_hour", "C_diel_PAR", "C_diel_PAR_common", "D_diel_PAR_phase",
               "E_diel_PAR_GCC", "F_diel_PAR_phase_GCC"),
  model_label = c("Constant", "Hour", "Diel + PAR", "Diel + PAR (common)", "Diel + PAR + phase",
                  "Diel + PAR + GCC", "Diel + PAR + phase + GCC"),
  rhs = c(
    "1",
    "hour_s1 + hour_c1 + hour_s2 + hour_c2",
    "hour_s1 + hour_c1 + hour_s2 + hour_c2 + ns(par_log, df = 4)",
    "hour_s1 + hour_c1 + hour_s2 + hour_c2 + ns(par_log, df = 4)",
    "hour_s1 + hour_c1 + hour_s2 + hour_c2 + ns(par_log, df = 4) + phenological_phase",
    "hour_s1 + hour_c1 + hour_s2 + hour_c2 + ns(par_log, df = 4) + ns(GCC, df = 4)",
    "hour_s1 + hour_c1 + hour_s2 + hour_c2 + ns(par_log, df = 4) + phenological_phase + ns(GCC, df = 4)"
  ),
  population = c("PAR", "PAR", "PAR", "PAR_GCC", "PAR", "PAR_GCC", "PAR_GCC")
)

daily_specs <- data.table(
  model_id = c("A_constant", "A_constant_common", "B_phase", "C_GCC", "D_phase_GCC"),
  model_label = c("Constant", "Constant (common)", "Phase", "GCC", "Phase + GCC"),
  rhs = c("1", "1", "phenological_phase", "ns(GCC, df = 4)",
          "phenological_phase + ns(GCC, df = 4)"),
  population = c("all", "GCC", "all", "GCC", "GCC")
)

theme_audit <- function(base_size = 9) {
  theme_classic(base_size = base_size, base_family = "Arial") +
    theme(
      axis.title = element_text(size = base_size + 0.5),
      axis.text = element_text(size = base_size),
      strip.background = element_rect(fill = "grey94", colour = NA),
      strip.text = element_text(face = "bold"),
      legend.position = "bottom", legend.title = element_blank(),
      panel.grid.major.y = element_line(colour = "grey91", linewidth = 0.25),
      panel.grid.minor = element_blank(),
      plot.tag = element_text(face = "bold", size = base_size + 2),
      plot.margin = margin(5, 6, 5, 6)
    )
}

save_plot <- function(plot, stem, width, height) {
  pdf_device <- function(filename, width, height, ...) {
    if (identical(Sys.info()[["sysname"]], "Darwin")) {
      quartz(type = "pdf", file = filename, width = width, height = height,
             family = "Arial", ...)
    } else {
      cairo_pdf(filename, width = width, height = height, family = "Arial", ...)
    }
  }
  png_device <- function(filename, width, height, ...) {
    if (identical(Sys.info()[["sysname"]], "Darwin")) {
      png(filename, width = width, height = height, units = "in", res = 400,
          type = "quartz", ...)
    } else {
      png(filename, width = width, height = height, units = "in", res = 400,
          type = "cairo", ...)
    }
  }
  ggsave(file.path(fig_dir, paste0(stem, ".png")), plot, width = width, height = height,
         units = "in", device = png_device, bg = "white")
  ggsave(file.path(fig_dir, paste0(stem, ".pdf")), plot, width = width, height = height,
         units = "in", device = pdf_device, bg = "white")
}

# Frozen-input verification and analysis table -------------------------------
fingerprints <- fread(input_fingerprints)
stopifnot(identical(fingerprints$canonical_state_id, states))
k4 <- read_gz(input_k4)
contract <- read_gz(input_contract)
stopifnot(nrow(k4) == 63156L, all(abs(rowSums(as.matrix(k4[, ..post_cols])) - 1) < 1e-6))
stopifnot(all(tolower(as.character(k4$primary_hmm_eligible)) == "true"))
stopifnot(!anyDuplicated(k4$source_row), !anyDuplicated(contract$source_row))

d <- merge(
  k4,
  contract[, .(source_row, photosynthetically_active_radiation, GCC, phenological_phase)],
  by = "source_row", all.x = TRUE
)
d[, date := as.IDate(date)]
d[, year := as.integer(format(date, "%Y"))]
d[, day_of_year := as.integer(format(date, "%j"))]
d[, timestamp_parsed := as.POSIXct(timestamp_start, format = "%Y-%m-%dT%H:%M:%S", tz = "UTC")]
d[, hour_decimal := as.integer(format(timestamp_parsed, "%H")) + as.integer(format(timestamp_parsed, "%M")) / 60]
d[, hour_bin_2h := floor(hour_decimal / 2) * 2 + 1]
d[, par_log := log1p(photosynthetically_active_radiation)]
d[, light_regime := fifelse(photosynthetically_active_radiation > 0, "Light", "Dark")]
d[!is.finite(photosynthetically_active_radiation), light_regime := NA_character_]
d[, phenological_phase := factor(phenological_phase, levels = phases_model)]
d[, `:=`(
  hour_s1 = sin(2 * pi * hour_decimal / 24), hour_c1 = cos(2 * pi * hour_decimal / 24),
  hour_s2 = sin(4 * pi * hour_decimal / 24), hour_c2 = cos(4 * pi * hour_decimal / 24),
  doy_s1 = sin(2 * pi * (day_of_year - 1) / 365.25), doy_c1 = cos(2 * pi * (day_of_year - 1) / 365.25),
  doy_s2 = sin(4 * pi * (day_of_year - 1) / 365.25), doy_c2 = cos(4 * pi * (day_of_year - 1) / 365.25),
  doy_s3 = sin(6 * pi * (day_of_year - 1) / 365.25), doy_c3 = cos(6 * pi * (day_of_year - 1) / 365.25)
)]
for (s in states) d[, (paste0("hard_", s)) := as.integer(canonical_state_id == s)]
stopifnot(setequal(unique(d$year), years), !anyNA(d$hour_decimal))

# Descriptive diel structure -------------------------------------------------
post_long <- melt(
  d[is.finite(photosynthetically_active_radiation)],
  id.vars = c("date", "year", "phenological_phase", "hour_bin_2h", "light_regime"),
  measure.vars = patterns(posterior = "^posterior_C", hard = "^hard_C"),
  variable.name = "state_index"
)
post_long[, canonical_state_id := states[state_index]]
post_long[, phenological_phase := factor(as.character(phenological_phase), levels = phases_display)]

hour_phase <- post_long[, .(
  mean_posterior = mean(posterior), hard_state_fraction = mean(hard),
  n_halfhours = .N, n_dates = uniqueN(date), n_years = uniqueN(year)
), by = .(phenological_phase, hour_bin_2h, canonical_state_id)]
fwrite(hour_phase, file.path(out_dir, "state_by_hour_phase.csv"))

light_phase <- post_long[, .(
  mean_posterior = mean(posterior), hard_state_fraction = mean(hard),
  n_halfhours = .N, n_dates = uniqueN(date), n_years = uniqueN(year)
), by = .(phenological_phase, light_regime, canonical_state_id)]
fwrite(light_phase, file.path(out_dir, "state_by_light_phase.csv"))

assoc <- rbindlist(c(
  list(contingency_metrics(d[!is.na(light_regime)]$light_regime,
                           d[!is.na(light_regime)]$canonical_state_id, "all_phases")),
  lapply(phases_display, function(ph) {
    x <- d[!is.na(light_regime) & as.character(phenological_phase) == ph]
    contingency_metrics(x$light_regime, x$canonical_state_id, ph)
  })
))
fwrite(assoc, file.path(out_dir, "light_dark_state_association.csv"))

# Daily posterior compositions, retaining Light and Dark strata -------------
daily_parts <- list(); dpi <- 1L
for (reg in c("Whole day", "Light", "Dark")) {
  x <- if (reg == "Whole day") d else d[light_regime == reg]
  z <- x[, c(
    list(
      year = first(year), day_of_year = first(day_of_year),
      phenological_phase = as.character(first(phenological_phase)),
      GCC = mean(GCC, na.rm = TRUE), n_halfhours = .N,
      n_GCC = sum(is.finite(GCC)), n_PAR = sum(is.finite(photosynthetically_active_radiation))
    ),
    lapply(.SD[, post_cols, with = FALSE], mean), lapply(.SD[, hard_cols, with = FALSE], mean)
  ), by = date]
  z[!is.finite(GCC), GCC := NA_real_]
  z[, light_regime := reg]
  daily_parts[[dpi]] <- z; dpi <- dpi + 1L
}
daily <- rbindlist(daily_parts, fill = TRUE)
setcolorder(daily, c("date", "year", "day_of_year", "phenological_phase", "GCC",
                    "light_regime", "n_halfhours", "n_GCC", "n_PAR", post_cols, hard_cols))
daily[, phenological_phase := factor(phenological_phase, levels = phases_model)]
daily[, `:=`(
  doy_s1 = sin(2 * pi * (day_of_year - 1) / 365.25), doy_c1 = cos(2 * pi * (day_of_year - 1) / 365.25),
  doy_s2 = sin(4 * pi * (day_of_year - 1) / 365.25), doy_c2 = cos(4 * pi * (day_of_year - 1) / 365.25),
  doy_s3 = sin(6 * pi * (day_of_year - 1) / 365.25), doy_c3 = cos(6 * pi * (day_of_year - 1) / 365.25)
)]
daily[, eligible_primary := fifelse(light_regime == "Whole day", n_halfhours >= 8, n_halfhours >= 4)]
daily[, eligible_strict := fifelse(light_regime == "Whole day", n_halfhours >= 16, n_halfhours >= 8)]
fwrite(daily, file.path(out_dir, "daily_state_compositions.csv"))

population_summary <- rbindlist(list(
  d[, .(population = "half_hour_all", n_rows = .N, n_dates = uniqueN(date), n_years = uniqueN(year))],
  d[is.finite(photosynthetically_active_radiation),
    .(population = "half_hour_PAR", n_rows = .N, n_dates = uniqueN(date), n_years = uniqueN(year))],
  d[is.finite(photosynthetically_active_radiation) & is.finite(GCC),
    .(population = "half_hour_PAR_GCC", n_rows = .N, n_dates = uniqueN(date), n_years = uniqueN(year))],
  daily[eligible_primary == TRUE, .(n_rows = .N, n_dates = uniqueN(date), n_years = uniqueN(year)),
        by = .(population = paste0("daily_primary_", gsub(" ", "_", light_regime)))],
  daily[eligible_strict == TRUE, .(n_rows = .N, n_dates = uniqueN(date), n_years = uniqueN(year)),
        by = .(population = paste0("daily_strict_", gsub(" ", "_", light_regime)))]
), fill = TRUE)
fwrite(population_summary, file.path(out_dir, "analysis_population_summary.csv"))

# Leave-one-year-out half-hourly model hierarchy -----------------------------
half_rows <- list(); hri <- 1L
for (response_type in c("posterior", "hard")) {
  response_cols <- if (response_type == "posterior") post_cols else hard_cols
        specs <- if (response_type == "posterior") model_specs else model_specs[model_id %in% c(
    "C_diel_PAR", "C_diel_PAR_common", "D_diel_PAR_phase", "E_diel_PAR_GCC",
    "F_diel_PAR_phase_GCC"
  )]
  for (i in seq_len(nrow(specs))) {
    spec <- specs[i]
    pop <- if (spec$population == "PAR") {
      d[is.finite(photosynthetically_active_radiation)]
    } else {
      d[is.finite(photosynthetically_active_radiation) & is.finite(GCC)]
    }
    for (yr in years) {
      train <- pop[year != yr]
      test <- pop[year == yr]
      pred <- fit_predict_statewise(train, test, response_cols, spec$rhs)
      met <- metric_by_date(as.matrix(test[, ..response_cols]), pred, test$date)$summary
      met[, `:=`(
        response_type = response_type, model_id = spec$model_id,
        model_label = spec$model_label, population = spec$population,
        held_out_year = yr
      )]
      half_rows[[hri]] <- met; hri <- hri + 1L
    }
  }
}
half_metrics <- rbindlist(half_rows, fill = TRUE)
setcolorder(half_metrics, c("response_type", "population", "model_id", "model_label",
                           "held_out_year", "n_test_rows", "n_test_dates", "brier",
                           "log_loss", "rmse", "dominant_accuracy"))
fwrite(half_metrics, file.path(out_dir, "halfhourly_loyo_metrics.csv"))

half_targets <- half_metrics[model_id %in% c("D_diel_PAR_phase", "E_diel_PAR_GCC",
                                             "F_diel_PAR_phase_GCC")]
half_targets[, baseline_model_id := fifelse(population == "PAR", "C_diel_PAR", "C_diel_PAR_common")]
half_baselines <- half_metrics[model_id %in% c("C_diel_PAR", "C_diel_PAR_common"),
                               .(response_type, held_out_year, baseline_model_id = model_id,
                                 baseline_brier = brier)]
half_gains <- merge(half_targets, half_baselines,
                    by = c("response_type", "held_out_year", "baseline_model_id"), all.x = TRUE)
half_gains[, relative_brier_gain_percent := 100 * (baseline_brier - brier) / baseline_brier]
fwrite(half_gains, file.path(out_dir, "halfhourly_relative_gains.csv"))

# Leave-one-year-out daily models within Light and Dark ----------------------
daily_rows <- list(); dri <- 1L
for (threshold in c("primary", "strict")) {
  eligible_col <- paste0("eligible_", threshold)
  for (reg in c("Light", "Dark")) {
    base <- daily[light_regime == reg & get(eligible_col) == TRUE]
    for (i in seq_len(nrow(daily_specs))) {
      spec <- daily_specs[i]
      pop <- if (spec$population == "GCC") base[is.finite(GCC)] else base
      for (yr in years) {
        train <- pop[year != yr]
        test <- pop[year == yr]
        pred <- fit_predict_statewise(train, test, post_cols, spec$rhs)
        met <- metric_by_date(as.matrix(test[, ..post_cols]), pred, test$date)$summary
        met[, `:=`(
          threshold = threshold, light_regime = reg, model_id = spec$model_id,
          model_label = spec$model_label, population = spec$population,
          held_out_year = yr
        )]
        daily_rows[[dri]] <- met; dri <- dri + 1L
      }
    }
  }
}
daily_metrics <- rbindlist(daily_rows, fill = TRUE)
setcolorder(daily_metrics, c("threshold", "light_regime", "population", "model_id",
                            "model_label", "held_out_year", "n_test_rows", "n_test_dates",
                            "brier", "log_loss", "rmse", "dominant_accuracy"))
fwrite(daily_metrics, file.path(out_dir, "daily_light_dark_loyo_metrics.csv"))

daily_targets <- daily_metrics[model_id %in% c("B_phase", "C_GCC", "D_phase_GCC")]
daily_targets[, baseline_model_id := fifelse(population == "GCC", "A_constant_common", "A_constant")]
daily_baselines <- daily_metrics[model_id %in% c("A_constant", "A_constant_common"),
                                 .(threshold, light_regime, held_out_year,
                                   baseline_model_id = model_id, baseline_brier = brier)]
daily_gains <- merge(daily_targets, daily_baselines,
                     by = c("threshold", "light_regime", "held_out_year", "baseline_model_id"), all.x = TRUE)
daily_gains[, relative_brier_gain_percent := 100 * (baseline_brier - brier) / baseline_brier]
fwrite(daily_gains, file.path(out_dir, "daily_relative_gains.csv"))

# PAR > 10 sensitivity for the light/dark definition -------------------------
d10 <- copy(d[is.finite(photosynthetically_active_radiation)])
d10[, light_regime := fifelse(photosynthetically_active_radiation > 10, "Light", "Dark")]
daily10_parts <- list()
for (reg in c("Light", "Dark")) {
  z <- d10[light_regime == reg, c(
    list(
      year = first(year), day_of_year = first(day_of_year),
      phenological_phase = as.character(first(phenological_phase)),
      GCC = mean(GCC, na.rm = TRUE), n_halfhours = .N
    ),
    lapply(.SD[, post_cols, with = FALSE], mean)
  ), by = date]
  z[!is.finite(GCC), GCC := NA_real_]
  z[, light_regime := reg]
  daily10_parts[[reg]] <- z
}
daily10 <- rbindlist(daily10_parts)
daily10[, phenological_phase := factor(phenological_phase, levels = phases_model)]
daily10[, `:=`(
  doy_s1 = sin(2 * pi * (day_of_year - 1) / 365.25), doy_c1 = cos(2 * pi * (day_of_year - 1) / 365.25),
  doy_s2 = sin(4 * pi * (day_of_year - 1) / 365.25), doy_c2 = cos(4 * pi * (day_of_year - 1) / 365.25),
  doy_s3 = sin(6 * pi * (day_of_year - 1) / 365.25), doy_c3 = cos(6 * pi * (day_of_year - 1) / 365.25)
)]
daily10 <- daily10[n_halfhours >= 4 & is.finite(GCC)]
fwrite(daily10, file.path(out_dir, "daily_state_compositions_PAR10_sensitivity.csv"))

par10_rows <- list(); p10i <- 1L
par10_specs <- daily_specs[model_id %in% c("A_constant_common", "D_phase_GCC")]
for (reg in c("Light", "Dark")) {
  pop <- daily10[light_regime == reg]
  for (i in seq_len(nrow(par10_specs))) {
    spec <- par10_specs[i]
    for (yr in years) {
      train <- pop[year != yr]
      test <- pop[year == yr]
      pred <- fit_predict_statewise(train, test, post_cols, spec$rhs)
      met <- metric_by_date(as.matrix(test[, ..post_cols]), pred, test$date)$summary
      met[, `:=`(light_regime = reg, model_id = spec$model_id,
                 model_label = spec$model_label, held_out_year = yr)]
      par10_rows[[p10i]] <- met; p10i <- p10i + 1L
    }
  }
}
par10_metrics <- rbindlist(par10_rows)
par10_baseline <- par10_metrics[model_id == "A_constant_common",
                                .(light_regime, held_out_year, baseline_brier = brier)]
par10_gains <- merge(par10_metrics[model_id == "D_phase_GCC"],
                     par10_baseline, by = c("light_regime", "held_out_year"))
par10_gains[, relative_brier_gain_percent := 100 * (baseline_brier - brier) / baseline_brier]
fwrite(par10_metrics, file.path(out_dir, "daily_PAR10_loyo_metrics.csv"))
fwrite(par10_gains, file.path(out_dir, "daily_PAR10_relative_gains.csv"))

# Prespecified interpretation rules -----------------------------------------
main_half <- half_gains[response_type == "posterior" & model_id == "F_diel_PAR_phase_GCC"]
half_mean_gain <- mean(main_half$relative_brier_gain_percent)
half_positive_years <- sum(main_half$relative_brier_gain_percent > 0)
daily_main <- daily_gains[threshold == "primary" & model_id == "D_phase_GCC"]
daily_decision <- daily_main[, .(
  mean_gain_percent = mean(relative_brier_gain_percent),
  positive_years = sum(relative_brier_gain_percent > 0)
), by = light_regime]
par10_decision <- par10_gains[model_id == "D_phase_GCC", .(
  mean_gain_percent = mean(relative_brier_gain_percent),
  positive_years = sum(relative_brier_gain_percent > 0)
), by = light_regime]
best_daily_gain <- max(daily_decision$mean_gain_percent)
best_daily_positive <- max(daily_decision[mean_gain_percent == best_daily_gain]$positive_years)

decision <- if (half_mean_gain >= 5 && half_positive_years >= 5 &&
                best_daily_gain >= 5 && best_daily_positive >= 5) {
  "RETAIN_MULTISCALE_HMM"
} else if ((half_mean_gain >= 3 && half_positive_years >= 4) ||
           (best_daily_gain >= 3 && best_daily_positive >= 4)) {
  "RETAIN_WITH_NARROW_CLAIM"
} else {
  "REDESIGN_DAILY_FUNCTIONAL_HMM"
}

decision_summary <- data.table(
  decision = decision,
  halfhour_full_gain_percent = half_mean_gain,
  halfhour_positive_years_of_6 = half_positive_years,
  light_daily_full_gain_percent = daily_decision[light_regime == "Light"]$mean_gain_percent,
  light_daily_positive_years_of_6 = daily_decision[light_regime == "Light"]$positive_years,
  dark_daily_full_gain_percent = daily_decision[light_regime == "Dark"]$mean_gain_percent,
  dark_daily_positive_years_of_6 = daily_decision[light_regime == "Dark"]$positive_years,
  PAR10_light_daily_full_gain_percent = par10_decision[light_regime == "Light"]$mean_gain_percent,
  PAR10_light_daily_positive_years_of_6 = par10_decision[light_regime == "Light"]$positive_years,
  PAR10_dark_daily_full_gain_percent = par10_decision[light_regime == "Dark"]$mean_gain_percent,
  PAR10_dark_daily_positive_years_of_6 = par10_decision[light_regime == "Dark"]$positive_years,
  rule_note = paste(
    "Project decision rule, not a universal statistical threshold:",
    "retain multiscale at >=5% mean Brier gain and positive in >=5/6 years at both half-hour and one daily stratum;",
    "narrow at >=3% and >=4/6 in either analysis; otherwise redesign a daily functional HMM."
  )
)
fwrite(decision_summary, file.path(out_dir, "decision_summary.csv"))

# Diagnostic figures ---------------------------------------------------------
p_a <- ggplot(hour_phase, aes(hour_bin_2h, canonical_state_id, fill = mean_posterior)) +
  geom_tile(colour = "white", linewidth = 0.25) +
  facet_wrap(~ phenological_phase, nrow = 1) +
  scale_x_continuous(breaks = c(1, 7, 13, 19), labels = c("00", "06", "12", "18"),
                     expand = c(0, 0)) +
  scale_fill_viridis_c(option = "C", limits = c(0, max(hour_phase$mean_posterior)),
                       labels = label_number(accuracy = 0.1)) +
  labs(x = "Hour of day", y = "Carbon state", fill = "Mean posterior") +
  theme_audit() + theme(panel.grid = element_blank(), legend.position = "right")

daily_plot_data <- melt(
  daily[light_regime %in% c("Light", "Dark") & eligible_primary == TRUE],
  id.vars = c("date", "phenological_phase", "light_regime"), measure.vars = post_cols,
  variable.name = "posterior_name", value.name = "posterior"
)
daily_plot_data[, canonical_state_id := sub("posterior_", "", posterior_name)]
daily_plot_data[, phenological_phase := factor(as.character(phenological_phase), levels = phases_display)]
daily_plot_data <- daily_plot_data[, .(mean_daily_occupancy = mean(posterior)),
                                   by = .(light_regime, phenological_phase, canonical_state_id)]
p_b <- ggplot(daily_plot_data,
              aes(phenological_phase, mean_daily_occupancy, fill = canonical_state_id)) +
  geom_col(width = 0.72, colour = "white", linewidth = 0.2) +
  facet_wrap(~ light_regime, nrow = 1) +
  scale_fill_manual(values = state_cols) +
  scale_y_continuous(labels = percent_format(accuracy = 1), expand = expansion(mult = c(0, 0.03))) +
  labs(x = NULL, y = "Mean daily state occupancy") +
  theme_audit() + theme(axis.text.x = element_text(angle = 25, hjust = 1))

half_plot <- half_metrics[
  response_type == "posterior" &
    ((population == "PAR" & model_id %in% c("A_constant", "B_hour", "C_diel_PAR", "D_diel_PAR_phase")) |
     (population == "PAR_GCC" & model_id %in% c("C_diel_PAR_common", "E_diel_PAR_GCC",
                                                 "F_diel_PAR_phase_GCC")))
]
half_plot[, model_label := factor(model_label, levels = model_specs$model_label)]
half_mean <- half_plot[, .(mean_brier = mean(brier)), by = .(model_label)]
p_c <- ggplot(half_plot, aes(model_label, brier)) +
  geom_point(aes(group = held_out_year), colour = "grey58", alpha = 0.7, size = 1.5,
             position = position_jitter(width = 0.08, height = 0)) +
  geom_point(data = half_mean, aes(model_label, mean_brier), shape = 18, size = 3,
             colour = "#0072B2") +
  coord_flip() +
  labs(x = NULL, y = "LOYO Brier score (lower is better)") +
  theme_audit() + theme(panel.grid.major.y = element_blank(), panel.grid.major.x = element_line(colour = "grey91"))

daily_plot <- daily_metrics[threshold == "primary" &
                             model_id %in% c("A_constant", "A_constant_common", "B_phase", "C_GCC",
                                              "D_phase_GCC")]
daily_plot[, model_label := factor(model_label, levels = daily_specs$model_label)]
daily_mean <- daily_plot[, .(mean_brier = mean(brier)), by = .(light_regime, model_label)]
p_d <- ggplot(daily_plot, aes(model_label, brier)) +
  geom_point(colour = "grey58", alpha = 0.7, size = 1.4,
             position = position_jitter(width = 0.08, height = 0)) +
  geom_point(data = daily_mean, aes(model_label, mean_brier), shape = 18, size = 2.8,
             colour = "#0072B2") +
  facet_wrap(~ light_regime, scales = "free_x") +
  coord_flip() +
  labs(x = NULL, y = "Daily-composition LOYO Brier score") +
  theme_audit() + theme(panel.grid.major.y = element_blank(), panel.grid.major.x = element_line(colour = "grey91"))

overview <- (p_a + plot_annotation(tag_levels = "A")) /
  ((p_b + p_c + p_d) + plot_layout(widths = c(1.05, 1, 1.05)) + plot_annotation(tag_levels = "B")) +
  plot_layout(heights = c(0.88, 1.12))
save_plot(overview, "diel_phenology_audit_overview", 13.2, 8.6)
save_plot(p_a, "figure_A_state_hour_phase", 10.5, 3.1)
save_plot(p_b, "figure_B_light_dark_phase_composition", 7.0, 3.5)
save_plot(p_c, "figure_C_halfhour_predictive_hierarchy", 6.5, 4.0)
save_plot(p_d, "figure_D_daily_light_dark_hierarchy", 7.5, 4.0)

# Reproducible narrative report ---------------------------------------------
fmt <- function(x, digits = 2) format(round(x, digits), nsmall = digits, trim = TRUE)
par_assoc <- assoc[scope == "all_phases"]
hard_compare <- half_gains[response_type == "hard" & model_id == "F_diel_PAR_phase_GCC",
                           .(mean_gain = mean(relative_brier_gain_percent), positive_years = sum(relative_brier_gain_percent > 0))]
strict_daily <- daily_gains[threshold == "strict" & model_id == "D_phase_GCC",
                            .(mean_gain = mean(relative_brier_gain_percent),
                              positive_years = sum(relative_brier_gain_percent > 0)), by = light_regime]

report <- c(
  "# Diel--phenology scale audit",
  "",
  "## Purpose and safeguards",
  "",
  paste0("This audit asks whether the frozen half-hourly K = 4 carbon states retain seasonal information after accounting for diel timing and PAR. ",
         "It uses the saved posterior probabilities and canonical Viterbi labels for all ", format(nrow(d), big.mark = ","),
         " primary observations; no hidden-state model was refitted, selected, or relabeled."),
  "",
  "The inferential unit for cross-validation is the date: row-level losses are first averaged within date, then dates receive equal weight within each held-out year. This prevents well-sampled days from dominating.",
  "",
  "## What the audit shows",
  "",
  paste0("Light versus dark is associated with the hard state labels (NMI = ", fmt(par_assoc$normalized_mutual_information, 3),
         "; Cramer's V = ", fmt(par_assoc$cramers_v, 3), "), confirming that diel irradiance is an important organizing dimension."),
  paste0("At the half-hour scale, adding phenophase and GCC to the diel + PAR baseline changed posterior-composition Brier score by ",
         fmt(half_mean_gain), "% on average and improved prediction in ", half_positive_years, " of 6 held-out years."),
  paste0("Using hard Viterbi labels instead of posteriors gave a corresponding mean gain of ", fmt(hard_compare$mean_gain),
         "% and improvement in ", hard_compare$positive_years, " of 6 years."),
  paste0("After aggregation within the light period, phase + GCC improved over a constant composition by ",
         fmt(daily_decision[light_regime == "Light"]$mean_gain_percent), "% on average (",
         daily_decision[light_regime == "Light"]$positive_years, "/6 years). Within the dark period, the gain was ",
         fmt(daily_decision[light_regime == "Dark"]$mean_gain_percent), "% (",
         daily_decision[light_regime == "Dark"]$positive_years, "/6 years)."),
  paste0("Under the stricter coverage rule, the corresponding gains were ",
         paste0(strict_daily$light_regime, " ", fmt(strict_daily$mean_gain), "% (", strict_daily$positive_years, "/6)", collapse = "; "), "."),
  paste0("When the light threshold was changed from PAR > 0 to PAR > 10, the phase + GCC gains remained ",
         paste0(par10_decision$light_regime, " ", fmt(par10_decision$mean_gain_percent), "% (",
                par10_decision$positive_years, "/6)", collapse = "; "), "."),
  "",
  "## Prespecified decision",
  "",
  paste0("**", decision, "**"),
  "",
  "The project rule was defined before examining these results: retain the multiscale interpretation when the phenology model improves Brier score by at least 5% on average, is positive in at least 5/6 years at the half-hour scale, and meets the same standard in at least one light/dark daily stratum. Use a narrower claim at at least 3% and 4/6 years in either analysis; otherwise redesign around a daily functional HMM. These are transparent project decision thresholds, not universal statistical cutoffs.",
  "",
  "## Ecological interpretation",
  "",
  "The half-hourly HMM should be interpreted first as a model of recurring ecosystem carbon-exchange regimes at the native flux timescale. Phenology is supported only to the extent that it predicts changes in the daily mixture of those regimes after diel structure has been separated. The light/dark analysis is therefore the bridge between short-term carbon dynamics and seasonal canopy development; it avoids claiming that individual half-hour transitions are phenological transitions.",
  "",
  "## Outputs",
  "",
  "- `state_by_hour_phase.csv`: posterior and hard-state occurrence in 2-hour bins by phenophase.",
  "- `daily_state_compositions.csv`: whole-day, light, and dark posterior compositions with coverage flags.",
  "- `halfhourly_loyo_metrics.csv`: date-balanced leave-one-year-out hierarchy.",
  "- `daily_light_dark_loyo_metrics.csv`: light/dark daily-composition hierarchy.",
  "- `daily_PAR10_loyo_metrics.csv`: sensitivity to defining light as PAR > 10.",
  "- `decision_summary.csv`: prespecified multiscale decision assessment.",
  "- `diel_phenology_audit_overview.pdf/.png`: diagnostic synthesis figure."
)
writeLines(report, report_path)

cat("Diel--phenology audit complete.\n")
cat("Decision:", decision, "\n")
cat("Half-hour full-model gain:", fmt(half_mean_gain), "% in", half_positive_years, "of 6 years\n")
cat("Report:", report_path, "\n")
