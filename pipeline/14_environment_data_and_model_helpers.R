# Shared preparation for the final phase-specific environmental analysis.
#
# This file deliberately contains data preparation and generic modelling helpers
# only. It never fits an HMM, changes state labels, or writes a figure. Keeping
# these functions separate allows the final Figure 3 script to be reproduced
# from frozen analytical inputs without redistributing raw network data.

suppressPackageStartupMessages(library(data.table))

# Read compressed CSVs without relying on data.table's platform-specific gzip
# command discovery.
read_environment_gz <- function(path) {
  as.data.table(read.csv(gzfile(path), stringsAsFactors = FALSE, check.names = FALSE))
}

# Independent quasibinomial models do not guarantee that the four predictions
# sum to one. Clamp numerical edge cases and renormalize before scoring.
renormalize_state_probabilities <- function(probabilities, epsilon = 1e-9) {
  probabilities <- pmax(as.matrix(probabilities), epsilon)
  probabilities / rowSums(probabilities)
}

prepare_environment_data <- function(root) {
  # Perform all joins and validation once so Figure 3 fitting and prediction
  # operate on identical core and water-level subsets.
  states <- paste0("C", 1:4)
  posterior_columns <- paste0("posterior_", states)
  years <- 2016:2021
  phase_levels <- c("Dormancy", "Greenup", "Maturity", "Senescence")

  hmm <- read_environment_gz(file.path(root, "data/processed/stjones_hmm_k4_primary_v1.csv.gz"))
  contract <- read_environment_gz(file.path(root, "data/processed/stjones_halfhourly_contract_v1.csv.gz"))
  pressure <- read_environment_gz(file.path(root, "data/processed/stjones_barometric_pressure_v1.csv.gz"))
  stopifnot(nrow(hmm) == 63156L, !anyDuplicated(hmm$source_row), !anyDuplicated(contract$source_row),
            !anyDuplicated(pressure$source_row), nrow(pressure) == nrow(contract))

  contract_columns <- c(
    "source_row", "phenological_phase",
    "photosynthetically_active_radiation", "vapor_pressure_deficit",
    "air_temperature", "precipitation", "salinity", "wind_speed",
    "recorded_water_level"
  )
  data <- merge(
    hmm[, c("source_row", "timestamp_start", "date", posterior_columns), with = FALSE],
    contract[, ..contract_columns], by = "source_row", all.x = TRUE, sort = FALSE
  )
  data <- merge(data, pressure, by = "source_row", all.x = TRUE, sort = FALSE)
  stopifnot(nrow(data) == 63156L)
  stopifnot(sum(is.finite(data$barometric_pressure)) == 62556L)
  stopifnot(all(range(data$barometric_pressure, na.rm = TRUE) > c(900, 1000)))
  stopifnot(all(range(data$barometric_pressure, na.rm = TRUE) < c(1050, 1100)))

  data[, year := as.integer(substr(date, 1, 4))]
  data[, phenological_phase := factor(phenological_phase, levels = phase_levels)]
  data[, hour_decimal := as.integer(substr(timestamp_start, 12, 13)) +
                         as.integer(substr(timestamp_start, 15, 16)) / 60]
  for (harmonic in 1:3) {
    data[, (paste0("hour_sin", harmonic)) := sin(2 * pi * harmonic * hour_decimal / 24)]
    data[, (paste0("hour_cos", harmonic)) := cos(2 * pi * harmonic * hour_decimal / 24)]
  }
  data[, par_log := log1p(photosynthetically_active_radiation)]
  data[, precipitation_log := log1p(precipitation)]
  data[, water_level_relative := {
    value <- rep(NA_real_, .N)
    observed <- is.finite(recorded_water_level)
    if (sum(observed) > 1L) {
      value[observed] <- (frank(recorded_water_level[observed], ties.method = "average") - 1) /
        (sum(observed) - 1)
    }
    value
  }, by = year]

  core_predictors <- c(
    "par_log", "vapor_pressure_deficit", "air_temperature",
    "precipitation_log", "salinity", "wind_speed", "barometric_pressure"
  )
  focal_predictors <- c("par_log", "air_temperature", "vapor_pressure_deficit", "salinity", "barometric_pressure")
  extended_predictors <- c(core_predictors, "water_level_relative")
  harmonic_terms <- as.vector(rbind(paste0("hour_sin", 1:3), paste0("hour_cos", 1:3)))
  required <- c("date", "year", "phenological_phase", harmonic_terms, posterior_columns)
  core <- data[complete.cases(data[, c(required, core_predictors), with = FALSE])]
  extended <- data[complete.cases(data[, c(required, extended_predictors), with = FALSE])]
  stopifnot(setequal(unique(core$year), years), setequal(unique(extended$year), years))

  list(
    states = states, posterior_columns = posterior_columns, years = years,
    phase_levels = phase_levels, core = core, extended = extended,
    core_predictors = core_predictors, focal_predictors = focal_predictors,
    extended_predictors = extended_predictors, harmonic_terms = harmonic_terms,
    all_hmm_rows = nrow(data), all_hmm_dates = uniqueN(data$date)
  )
}

scale_environment_fold <- function(training, test, predictors) {
  parameters <- rbindlist(lapply(predictors, function(variable) {
    data.table(canonical_variable = variable, mean = mean(training[[variable]]), sd = sd(training[[variable]]))
  }))
  stopifnot(all(is.finite(parameters$sd)), all(parameters$sd > 0))
  for (variable in predictors) {
    mean_value <- parameters[canonical_variable == variable]$mean
    sd_value <- parameters[canonical_variable == variable]$sd
    training[, (paste0("z_", variable)) := (get(variable) - mean_value) / sd_value]
    test[, (paste0("z_", variable)) := (get(variable) - mean_value) / sd_value]
  }
  list(training = training, test = test, parameters = parameters)
}

apply_environment_scaling <- function(data, predictors, parameters) {
  data <- copy(data)
  for (variable in predictors) {
    mean_value <- parameters[canonical_variable == variable]$mean
    sd_value <- parameters[canonical_variable == variable]$sd
    data[, (paste0("z_", variable)) := (get(variable) - mean_value) / sd_value]
  }
  data
}

date_balanced_brier <- function(observed, predicted, dates) {
  loss <- rowSums((as.matrix(observed) - predicted)^2)
  mean(data.table(date = dates, loss = loss)[, mean(loss), by = date]$V1)
}

date_balanced_state_change <- function(change, dates, states) {
  data <- as.data.table(change)
  setnames(data, states)
  data[, date := dates]
  as.numeric(data[, lapply(.SD, mean), by = date, .SDcols = states][, lapply(.SD, mean), .SDcols = states])
}
