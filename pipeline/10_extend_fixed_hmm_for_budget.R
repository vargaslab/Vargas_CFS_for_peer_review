#!/usr/bin/env Rscript

# Apply the fixed, already-fitted canonical K4 HMM parameters to complete
# gap-filled carbon-flux records for annual-budget attribution. No parameter is
# estimated and the existing observed frozen state table is never overwritten.
#
# The implementation first verifies exact reproduction on observed intervals,
# then applies the locked parameters to the MDS-completed sequence. Its compact
# state-by-phase budget output is consumed by Figures 4 and Table S4.

args <- commandArgs(trailingOnly = FALSE)
script_arg <- grep("^--file=", args, value = TRUE)
root <- if (length(script_arg)) {
  normalizePath(file.path(dirname(sub("^--file=", "", script_arg[1])), "../.."), mustWork = TRUE)
} else normalizePath(".", mustWork = TRUE)

suppressPackageStartupMessages({
  library(data.table)
  library(jsonlite)
})

raw_path <- file.path(root, "data/raw/stjones_season.csv")
contract_path <- file.path(root, "data/processed/stjones_halfhourly_contract_v1.csv.gz")
hmm_path <- file.path(root, "data/processed/stjones_hmm_k4_primary_v1.csv.gz")
model_metadata_path <- file.path(
  root, "outputs/complete_dataset/hmm_primary/best_by_structure/full/K4/fit_metadata.json"
)
standardization_path <- file.path(
  root, "outputs/complete_dataset/hmm_primary/standardization_parameters.json"
)
output_dir <- file.path(
  root, "outputs/complete_dataset/manuscript/annual_carbon_balance_decomposition"
)
publication_input_dir <- file.path(root, "data/publication_inputs")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(publication_input_dir, recursive = TRUE, showWarnings = FALSE)

states <- paste0("C", 1:4)
phase_levels <- c("Greenup", "Maturity", "Senescence", "Dormancy")
sink_years <- c(2016L, 2019L)
source_years <- c(2017L, 2018L, 2020L, 2021L)
interval_seconds <- 1800
carbon_molar_mass <- 12.0107

read_gz <- function(path) {
  as.data.table(read.csv(gzfile(path), stringsAsFactors = FALSE, check.names = FALSE))
}
as_flag <- function(x) {
  if (is.logical(x)) return(x)
  tolower(as.character(x)) == "true"
}
log_sum_exp <- function(x) {
  m <- max(x)
  if (!is.finite(m)) return(m)
  m + log(sum(exp(x - m)))
}
adjusted_rand <- function(left, right) {
  tab <- table(left, right)
  choose2 <- function(x) x * (x - 1) / 2
  n <- sum(tab)
  index <- sum(choose2(tab))
  row_sum <- sum(choose2(rowSums(tab)))
  col_sum <- sum(choose2(colSums(tab)))
  total <- choose2(n)
  expected <- row_sum * col_sum / total
  maximum <- (row_sum + col_sum) / 2
  if (maximum == expected) return(1)
  (index - expected) / (maximum - expected)
}

gaussian_log_emission <- function(x, means, covariances) {
  n <- nrow(x)
  k <- nrow(means)
  d <- ncol(x)
  out <- matrix(NA_real_, nrow = n, ncol = k)
  for (state in seq_len(k)) {
    covariance <- covariances[state, , ]
    inverse <- solve(covariance)
    determinant <- determinant(covariance, logarithm = TRUE)
    if (determinant$sign <= 0) stop("Non-positive covariance determinant")
    centered <- sweep(x, 2, means[state, ], "-")
    quadratic <- rowSums((centered %*% inverse) * centered)
    out[, state] <- -0.5 * (
      d * log(2 * pi) + as.numeric(determinant$modulus) + quadratic
    )
  }
  out
}

decode_fixed_hmm <- function(x, lengths, start_probability, transition, means, covariances) {
  stopifnot(sum(lengths) == nrow(x), all(lengths > 0))
  log_emission <- gaussian_log_emission(x, means, covariances)
  log_transition <- log(transition)
  log_start <- log(start_probability)
  k <- ncol(log_emission)
  posterior <- matrix(NA_real_, nrow = nrow(x), ncol = k)
  viterbi <- integer(nrow(x))
  log_likelihoods <- numeric(length(lengths))
  offset <- 0L

  for (sequence_index in seq_along(lengths)) {
    length_i <- lengths[sequence_index]
    rows <- offset + seq_len(length_i)
    emission <- log_emission[rows, , drop = FALSE]
    alpha <- matrix(NA_real_, nrow = length_i, ncol = k)
    beta <- matrix(0, nrow = length_i, ncol = k)
    delta <- matrix(NA_real_, nrow = length_i, ncol = k)
    psi <- matrix(0L, nrow = length_i, ncol = k)

    alpha[1, ] <- log_start + emission[1, ]
    delta[1, ] <- alpha[1, ]
    if (length_i > 1) {
      for (time in 2:length_i) {
        for (destination in seq_len(k)) {
          candidates <- alpha[time - 1, ] + log_transition[, destination]
          alpha[time, destination] <- emission[time, destination] + log_sum_exp(candidates)
          viterbi_candidates <- delta[time - 1, ] + log_transition[, destination]
          psi[time, destination] <- which.max(viterbi_candidates)
          delta[time, destination] <- emission[time, destination] + max(viterbi_candidates)
        }
      }
      for (time in (length_i - 1):1) {
        for (origin in seq_len(k)) {
          beta[time, origin] <- log_sum_exp(
            log_transition[origin, ] + emission[time + 1, ] + beta[time + 1, ]
          )
        }
      }
    }

    log_likelihood <- log_sum_exp(alpha[length_i, ])
    probabilities <- exp(alpha + beta - log_likelihood)
    probabilities <- probabilities / rowSums(probabilities)
    posterior[rows, ] <- probabilities
    path <- integer(length_i)
    path[length_i] <- which.max(delta[length_i, ])
    if (length_i > 1) {
      for (time in (length_i - 1):1) path[time] <- psi[time + 1, path[time + 1]]
    }
    viterbi[rows] <- path
    log_likelihoods[sequence_index] <- log_likelihood
    offset <- offset + length_i
  }

  list(
    posterior = posterior,
    viterbi = viterbi,
    log_likelihood = sum(log_likelihoods),
    sequence_log_likelihood = log_likelihoods
  )
}

sequence_lengths <- function(timestamp_start, extra_break = NULL) {
  time <- as.POSIXct(timestamp_start, format = "%Y-%m-%dT%H:%M:%S", tz = "UTC")
  breaks <- c(TRUE, diff(as.numeric(time)) != interval_seconds)
  if (!is.null(extra_break)) breaks <- breaks | c(TRUE, diff(extra_break) != 0)
  sequence <- cumsum(breaks)
  as.integer(tabulate(sequence))
}

model <- read_json(model_metadata_path, simplifyVector = TRUE)
standardization <- read_json(standardization_path, simplifyVector = TRUE)$parameters
stopifnot(model$K == 4, model$covariance_type == "full", model$converged)
means <- as.matrix(model$emission_means_standardized)
covariances <- model$emission_covariances_standardized
transition <- as.matrix(model$transition_matrix)
start_probability <- as.numeric(model$start_probabilities)
stopifnot(
  max(abs(rowSums(transition) - 1)) < 1e-10,
  abs(sum(start_probability) - 1) < 1e-10
)

# Reproduce the frozen inference first; failure here invalidates extension.
hmm <- read_gz(hmm_path)
hmm[, timestamp_start := as.character(timestamp_start)]
setorder(hmm, timestamp_start, source_row)
validation_x <- cbind(
  (hmm$NEE - standardization$NEE$mean) / standardization$NEE$sample_standard_deviation,
  (hmm$CH4 - standardization$CH4$mean) / standardization$CH4$sample_standard_deviation
)
validation_lengths <- as.integer(hmm[, .N, by = sequence_id]$N)
validation <- decode_fixed_hmm(
  validation_x, validation_lengths, start_probability, transition, means, covariances
)
stored_viterbi <- as.integer(sub("C", "", hmm$canonical_state_id))
stored_posterior <- as.matrix(hmm[, paste0("posterior_C", 1:4), with = FALSE])
implementation_validation <- data.table(
  n_observations = nrow(hmm),
  viterbi_exact_agreement_percent = 100 * mean(validation$viterbi == stored_viterbi),
  adjusted_rand_index = adjusted_rand(validation$viterbi, stored_viterbi),
  posterior_mean_absolute_difference = mean(abs(validation$posterior - stored_posterior)),
  posterior_maximum_absolute_difference = max(abs(validation$posterior - stored_posterior)),
  stored_mean_maximum_posterior = mean(apply(stored_posterior, 1, max)),
  reproduced_mean_maximum_posterior = mean(apply(validation$posterior, 1, max))
)
if (
  implementation_validation$viterbi_exact_agreement_percent < 99.99 ||
    implementation_validation$posterior_maximum_absolute_difference > 1e-5
) stop("Fixed-parameter R decoder failed to reproduce frozen inference")

# Construct the canonical complete budget table.
raw <- fread(
  raw_path,
  select = c(
    "NEE_f", "CH4_f", "NEE_C_sum", "CH4_C_sum"
  ),
  showProgress = FALSE
)
raw[, source_row := .I]
contract <- read_gz(contract_path)
contract[, timestamp_start := as.character(timestamp_start)]
full <- raw[contract[, .(
  source_row, timestamp_start, timestamp_end, date, phenological_phase
)], on = "source_row"]
setorder(full, timestamp_start, source_row)
full[, year := as.integer(substr(timestamp_start, 1, 4))]
full[, annual_balance_group := fifelse(
  year %in% sink_years, "sink",
  fifelse(year %in% source_years, "source", NA_character_)
)]
full[, NEE_complete := fifelse(
  is.na(NEE_f), NEE_C_sum * 1e6 / (interval_seconds * carbon_molar_mass), NEE_f
)]
full[, CH4_MDS_complete := fifelse(
  is.na(CH4_f), CH4_C_sum * 1e9 / (interval_seconds * carbon_molar_mass), CH4_f
)]
stopifnot(
  !anyNA(full$NEE_complete), !anyNA(full$CH4_MDS_complete),
  !anyNA(full$phenological_phase)
)

full_lengths <- sequence_lengths(full$timestamp_start)
make_standardized <- function(nee, ch4) cbind(
  (nee - standardization$NEE$mean) / standardization$NEE$sample_standard_deviation,
  (ch4 - standardization$CH4$mean) / standardization$CH4$sample_standard_deviation
)
decode_mds <- decode_fixed_hmm(
  make_standardized(full$NEE_complete, full$CH4_MDS_complete),
  full_lengths, start_probability, transition, means, covariances
)

for (state in seq_along(states)) {
  full[, (paste0("posterior_MDS_", states[state])) := decode_mds$posterior[, state]]
}
full[, state_MDS := states[decode_mds$viterbi]]

# Validate extended assignments on the observed frozen population without
# replacing the authoritative stored inference.
comparison <- hmm[, .(
  source_row, frozen_state = canonical_state_id,
  frozen_C1 = posterior_C1, frozen_C2 = posterior_C2,
  frozen_C3 = posterior_C3, frozen_C4 = posterior_C4
)][full, on = "source_row"]
comparison <- comparison[!is.na(frozen_state)]
predicted_posterior <- as.matrix(comparison[, paste0("posterior_MDS_", states), with = FALSE])
frozen_posterior <- as.matrix(comparison[, paste0("frozen_", states), with = FALSE])
comparison[, `:=`(
  predicted_state = state_MDS,
  posterior_absolute_difference = rowMeans(abs(predicted_posterior - frozen_posterior)),
  maximum_posterior_extended = apply(predicted_posterior, 1, max)
)]
extension_validation <- comparison[, .(
  n_observed_frozen = .N,
  exact_state_agreement_percent = 100 * mean(predicted_state == frozen_state),
  adjusted_rand_index = adjusted_rand(predicted_state, frozen_state),
  posterior_mean_absolute_difference = mean(posterior_absolute_difference),
  mean_maximum_posterior_extended = mean(maximum_posterior_extended)
), by = year][, branch := "MDS"]
setcolorder(extension_validation, c("branch", "year", setdiff(names(extension_validation), c("branch", "year"))))
setorder(extension_validation, year)

extended_export <- full[, c(
  "source_row", "timestamp_start", "timestamp_end", "date", "year",
  "annual_balance_group", "phenological_phase", "state_MDS",
  paste0("posterior_MDS_", states)
), with = FALSE]
fwrite(
  extended_export,
  file.path(output_dir, "fixed_k4_gapfilled_state_attribution.csv.gz"),
  compress = "gzip"
)

make_budget_long <- function(probability_prefix, ch4_column, branch_label) {
  rows <- lapply(states, function(state) {
    probability <- full[[paste0(probability_prefix, state)]]
    data.table(
      source_row = full$source_row,
      year = full$year,
      annual_balance_group = full$annual_balance_group,
      phenological_phase = full$phenological_phase,
      canonical_state_id = state,
      posterior_probability = probability,
      NEE_C_contribution = probability * full$NEE_C_sum,
      CH4_C_contribution = probability * full[[ch4_column]]
    )
  })
  out <- rbindlist(rows)
  out[, net_C_contribution := NEE_C_contribution + CH4_C_contribution]
  out[, attribution_branch := branch_label]
  out
}

budget_primary <- make_budget_long(
  "posterior_MDS_", "CH4_C_sum", "MDS_state_MDS_budget_primary"
)
all_budget <- budget_primary

annual_state <- all_budget[, .(
  posterior_halfhours = sum(posterior_probability),
  posterior_occupancy = mean(posterior_probability),
  NEE_C_g_m2_y = sum(NEE_C_contribution),
  CH4_C_g_m2_y = sum(CH4_C_contribution),
  net_C_g_m2_y = sum(net_C_contribution)
), by = .(
  attribution_branch, year, annual_balance_group, canonical_state_id
)]
annual_state_phase <- all_budget[, .(
  posterior_halfhours = sum(posterior_probability),
  posterior_occupancy = mean(posterior_probability),
  NEE_C_g_m2 = sum(NEE_C_contribution),
  CH4_C_g_m2 = sum(CH4_C_contribution),
  net_C_g_m2 = sum(net_C_contribution)
), by = .(
  attribution_branch, year, annual_balance_group,
  phenological_phase, canonical_state_id
)]

reconciliation <- annual_state[, .(
  attributed_NEE_C = sum(NEE_C_g_m2_y),
  attributed_CH4_C = sum(CH4_C_g_m2_y),
  attributed_net_C = sum(net_C_g_m2_y),
  total_posterior_occupancy = sum(posterior_occupancy)
), by = .(attribution_branch, year, annual_balance_group)]
full_totals <- full[, .(
  complete_NEE_C = sum(NEE_C_sum),
  complete_CH4_C_MDS = sum(CH4_C_sum)
), by = .(year, annual_balance_group)]
reconciliation <- full_totals[reconciliation, on = c("year", "annual_balance_group")]
reconciliation[, expected_CH4_C := complete_CH4_C_MDS]
reconciliation[, `:=`(
  NEE_reconciliation_error = attributed_NEE_C - complete_NEE_C,
  CH4_reconciliation_error = attributed_CH4_C - expected_CH4_C,
  net_reconciliation_error = attributed_net_C - complete_NEE_C - expected_CH4_C
)]

fwrite(implementation_validation, file.path(output_dir, "fixed_decoder_validation.csv"))
fwrite(extension_validation, file.path(output_dir, "gapfilled_extension_validation_by_year.csv"))
fwrite(annual_state, file.path(output_dir, "annual_budget_by_state_all_branches.csv"))
fwrite(annual_state_phase, file.path(output_dir, "annual_budget_by_state_phase_all_branches.csv"))
fwrite(
  annual_state_phase,
  file.path(publication_input_dir, "annual_budget_by_state_phase_all_branches.csv")
)
fwrite(reconciliation, file.path(output_dir, "annual_state_budget_reconciliation.csv"))

metadata <- list(
  analysis = "Fixed-K4 extension for complete annual carbon-budget attribution",
  hmm_refitted = FALSE,
  model_parameters_changed = FALSE,
  existing_frozen_states_overwritten = FALSE,
  fixed_fit_id = model$fit_id,
  implementation_validation = as.list(implementation_validation[1]),
  full_sequence_definition = "canonical intervals split only where interval-start gaps differ from 30 minutes",
  primary_attribution = list(
    state_probabilities = "fixed K4 applied to complete MDS NEE and CH4 fluxes",
    CO2_budget = "NEE_C_sum",
    CH4_budget = "CH4_C_sum",
    rationale = "the primary state extension and greenhouse-gas accounting use the same internally matched MDS-completed NEE and CH4 series"
  ),
  sensitivities = list(),
  interpretation_boundary = paste(
    "Extended probabilities are a gap-filled budget-attribution sensitivity,",
    "not replacements for the frozen observed-state sequence."
  )
)
write_json(
  metadata, file.path(output_dir, "state_attribution_metadata.json"),
  pretty = TRUE, auto_unbox = TRUE
)

cat("Frozen decoder agreement:",
    sprintf("%.4f%%", implementation_validation$viterbi_exact_agreement_percent), "\n")
cat("Created:", file.path(output_dir, "annual_budget_by_state_all_branches.csv"), "\n")
cat("Created:", file.path(output_dir, "annual_budget_by_state_phase_all_branches.csv"), "\n")
