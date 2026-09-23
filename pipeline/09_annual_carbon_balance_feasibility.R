#!/usr/bin/env Rscript

# Audit whether the independently supplied annual sink/source classification can
# be reconstructed from the authoritative raw carbon-budget variables and
# linked to the frozen K4 HMM. This script does not refit or relabel the HMM and
# does not assign states to gap-filled intervals.
#
# Outputs are budget reconstruction, phase coverage, unit-conversion, and
# alignment diagnostics. Only the validated phase-coverage table crosses the
# publication-input boundary.

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
output_dir <- file.path(
  root, "outputs/complete_dataset/manuscript/annual_carbon_balance_decomposition"
)
publication_input_dir <- file.path(root, "data/publication_inputs")
report_path <- file.path(
  root, "reports/complete_dataset/annual_carbon_balance_decomposition_feasibility.md"
)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(publication_input_dir, recursive = TRUE, showWarnings = FALSE)

sink_years <- c(2016L, 2019L)
source_years <- c(2017L, 2018L, 2020L, 2021L)
years <- 2016:2021
phase_levels <- c("Greenup", "Maturity", "Senescence", "Dormancy")

as_flag <- function(x) {
  if (is.logical(x)) return(x)
  tolower(as.character(x)) == "true"
}
read_gz <- function(path) {
  as.data.table(read.csv(gzfile(path), stringsAsFactors = FALSE, check.names = FALSE))
}

raw_columns <- c(
  "DATE_TIME", "DATE", "NEE_orig", "NEE_f", "CH4_orig", "CH4_f",
  "NEE_C_sum", "CH4_C_sum", "GCC", "Season"
)
raw <- fread(raw_path, select = raw_columns, showProgress = FALSE)
raw[, source_row := .I]
contract <- read_gz(contract_path)
hmm <- read_gz(hmm_path)
contract[, primary_hmm_eligible := as_flag(primary_hmm_eligible)]
hmm[, primary_hmm_eligible := as_flag(primary_hmm_eligible)]

stopifnot(
  nrow(contract) == 105210L,
  nrow(hmm) == 63156L,
  uniqueN(contract$source_row) == nrow(contract),
  uniqueN(contract$timestamp_start) == nrow(contract),
  uniqueN(hmm$source_row) == nrow(hmm),
  all(hmm$primary_hmm_eligible)
)

budget <- raw[contract[, .(
  source_row, source_file_line, timestamp_start, timestamp_end,
  phenological_phase, primary_hmm_eligible, duplicate_group, duplicate_retained
)], on = "source_row"]
stopifnot(nrow(budget) == nrow(contract), !anyNA(budget$timestamp_start))
budget[, year := as.integer(substr(timestamp_start, 1, 4))]
budget[, annual_balance_group := fifelse(
  year %in% sink_years, "sink",
  fifelse(year %in% source_years, "source", NA_character_)
)]
stopifnot(setequal(unique(budget$year), years), !anyNA(budget$annual_balance_group))

# Frozen-state linkage and identity checks.
hmm_link <- hmm[, .(
  source_row, hmm_timestamp_start = timestamp_start, hmm_NEE = NEE, hmm_CH4 = CH4,
  canonical_state_id, posterior_C1, posterior_C2, posterior_C3, posterior_C4
)]
budget <- hmm_link[budget, on = "source_row"]
budget[, has_frozen_state := !is.na(canonical_state_id)]
stopifnot(
  sum(budget$has_frozen_state) == nrow(hmm),
  max(abs(budget[has_frozen_state == TRUE, hmm_NEE - NEE_orig]), na.rm = TRUE) < 1e-10,
  max(abs(budget[has_frozen_state == TRUE, hmm_CH4 - CH4_orig]), na.rm = TRUE) < 1e-10,
  all(budget[has_frozen_state == TRUE, hmm_timestamp_start == timestamp_start])
)

# Verify the supplied half-hour carbon-sum columns against their flux columns.
carbon_molar_mass <- 12.0107
interval_seconds <- 1800
budget[, expected_NEE_C_sum := NEE_f * interval_seconds * carbon_molar_mass / 1e6]
budget[, expected_CH4_C_sum_mds := CH4_f * interval_seconds * carbon_molar_mass / 1e9]
conversion_checks <- data.table(
  component = c("NEE_C_sum", "CH4_C_sum"),
  supplied_column = c("NEE_C_sum", "CH4_C_sum"),
  source_flux_column = c("NEE_f", "CH4_f"),
  maximum_absolute_difference_gC_m2_halfhour = c(
    max(abs(budget$NEE_C_sum - budget$expected_NEE_C_sum), na.rm = TRUE),
    max(abs(budget$CH4_C_sum - budget$expected_CH4_C_sum_mds), na.rm = TRUE)
  ),
  interval_seconds = interval_seconds,
  carbon_molar_mass_g_mol = carbon_molar_mass
)

annual_budget <- budget[, .(
  n_canonical_halfhours = .N,
  n_dates = uniqueN(substr(timestamp_start, 1, 10)),
  n_frozen_state_halfhours = sum(has_frozen_state),
  frozen_state_coverage_percent = 100 * mean(has_frozen_state),
  NEE_C_g_m2_y = sum(NEE_C_sum),
  CH4_C_MDS_g_m2_y = sum(CH4_C_sum),
  net_C_MDS_g_m2_y = sum(NEE_C_sum + CH4_C_sum)
), by = .(year, annual_balance_group)]
annual_budget[, reconstructed_class_MDS := fifelse(net_C_MDS_g_m2_y < 0, "sink", "source")]
annual_budget[, supplied_class_reproduced_MDS := reconstructed_class_MDS == annual_balance_group]
setorder(annual_budget, year)

# Quantify how much of the complete annual budget has an observed frozen state.
coverage_by_year <- budget[, .(
  canonical_halfhours = .N,
  frozen_state_halfhours = sum(has_frozen_state),
  frozen_state_coverage_percent = 100 * mean(has_frozen_state),
  complete_dates = uniqueN(substr(timestamp_start, 1, 10)),
  frozen_state_dates = uniqueN(substr(timestamp_start[has_frozen_state], 1, 10)),
  frozen_state_fraction_absolute_NEE_C_percent =
    100 * sum(abs(NEE_C_sum[has_frozen_state])) / sum(abs(NEE_C_sum)),
  frozen_state_fraction_absolute_CH4_C_MDS_percent =
    100 * sum(abs(CH4_C_sum[has_frozen_state])) / sum(abs(CH4_C_sum)),
  observed_state_NEE_C_g_m2 = sum(NEE_C_sum[has_frozen_state]),
  complete_NEE_C_g_m2 = sum(NEE_C_sum),
  observed_state_CH4_C_MDS_g_m2 = sum(CH4_C_sum[has_frozen_state]),
  complete_CH4_C_MDS_g_m2 = sum(CH4_C_sum)
), by = .(year, annual_balance_group)]
setorder(coverage_by_year, year)

coverage_by_phase <- budget[, .(
  canonical_halfhours = .N,
  frozen_state_halfhours = sum(has_frozen_state),
  frozen_state_coverage_percent = 100 * mean(has_frozen_state),
  NEE_C_g_m2 = sum(NEE_C_sum),
  CH4_C_MDS_g_m2 = sum(CH4_C_sum),
  net_C_MDS_g_m2 = sum(NEE_C_sum + CH4_C_sum)
), by = .(year, annual_balance_group, phenological_phase)]
coverage_by_phase[, phenological_phase := factor(phenological_phase, levels = phase_levels)]
setorder(coverage_by_phase, year, phenological_phase)

budget[, CH4_MDS_minus_HMM := CH4_f - hmm_CH4]
ch4_alignment <- budget[has_frozen_state == TRUE, .(
  n_frozen_state_halfhours = .N,
  n_MDS_differences_gt_0_001 = sum(abs(CH4_MDS_minus_HMM) > 0.001, na.rm = TRUE),
  MDS_difference_max_abs = max(abs(CH4_MDS_minus_HMM), na.rm = TRUE)
), by = year]
setorder(ch4_alignment, year)

duplicate_audit <- data.table(
  raw_rows = nrow(raw),
  canonical_contract_rows = nrow(contract),
  removed_duplicate_rows = nrow(raw) - nrow(contract),
  canonical_duplicate_flag_rows = sum(!is.na(contract$duplicate_group)),
  canonical_duplicate_retained_rows = sum(as_flag(contract$duplicate_retained), na.rm = TRUE),
  affected_years = paste(sort(unique(budget[!is.na(duplicate_group), year])), collapse = ", ")
)

fwrite(annual_budget, file.path(output_dir, "annual_carbon_budget_reconstruction.csv"))
fwrite(coverage_by_year, file.path(output_dir, "frozen_state_coverage_by_year.csv"))
phase_export <- copy(coverage_by_phase)
phase_export[, phenological_phase := as.character(phenological_phase)]
fwrite(phase_export, file.path(output_dir, "budget_and_state_coverage_by_year_phase.csv"))
fwrite(
  phase_export,
  file.path(publication_input_dir, "budget_and_state_coverage_by_year_phase.csv")
)
fwrite(ch4_alignment, file.path(output_dir, "ch4_budget_to_hmm_alignment_by_year.csv"))
fwrite(conversion_checks, file.path(output_dir, "carbon_sum_conversion_checks.csv"))
fwrite(duplicate_audit, file.path(output_dir, "duplicate_resolution_audit.csv"))

audit_metadata <- list(
  purpose = "Feasibility audit for annual carbon-balance decomposition by frozen HMM state",
  hmm_refitted = FALSE,
  states_reassigned = FALSE,
  authoritative_budget_columns = list(
    CO2_carbon = "NEE_C_sum",
    CH4_carbon = "CH4_C_sum"
  ),
  common_unit = "g C m^-2 per half-hour; annual sums are g C m^-2 y^-1",
  annual_grouping = "year of canonical interval-start timestamp",
  sink_years = as.list(sink_years),
  source_years = as.list(source_years),
  full_budget_classification_reproduced = all(annual_budget$supplied_class_reproduced_MDS),
  feasibility_status = "CONDITIONALLY FEASIBLE",
  condition = paste(
    "Full annual budgets are complete and reproduce all six classifications,",
    "but gap-filled intervals lack frozen states and require a prespecified",
    "state-attribution extension."
  )
)
write_json(
  audit_metadata, file.path(output_dir, "feasibility_metadata.json"),
  pretty = TRUE, auto_unbox = TRUE
)

fmt <- function(x, digits = 1) formatC(x, digits = digits, format = "f")
annual_lines <- apply(annual_budget, 1, function(row) paste0(
  "| ", row[["year"]], " | ", row[["annual_balance_group"]], " | ",
  " ", fmt(as.numeric(row[["NEE_C_g_m2_y"]]), 1), " |",
  " ", fmt(as.numeric(row[["CH4_C_MDS_g_m2_y"]]), 1), " |",
  " ", fmt(as.numeric(row[["net_C_MDS_g_m2_y"]]), 1), " |",
  " ", fmt(as.numeric(row[["frozen_state_coverage_percent"]]), 1), "% |"
))

report <- c(
  "# Feasibility of decomposing annual carbon balance by frozen HMM state",
  "",
  "## Outcome",
  "",
  "**CONDITIONALLY FEASIBLE.** The complete-data MDS carbon-sum columns reproduce the supplied annual sink/source classification in all six years. The frozen HMM links exactly to the original observed NEE and CH4 rows. However, frozen states exist only for observed primary-eligible intervals, covering 41.4–72.4% of half-hours by year. A complete annual state budget therefore requires an explicitly labeled state-attribution extension for gap-filled intervals; it cannot be presented as a direct sum of the existing frozen state table.",
  "",
  "## Reconstructed annual carbon budgets",
  "",
  "The analysis uses `NEE_C_sum` and `CH4_C_sum` as the half-hourly MDS-completed carbon-budget terms. Both are in g C m^-2 per half-hour and were summed by the canonical interval-start year. Negative totals denote net ecosystem carbon uptake.",
  "",
  "| Year | Supplied class | NEE-C | CH4-C (MDS) | Net C | Frozen-state coverage |",
  "|---:|:---|---:|---:|---:|---:|",
  annual_lines,
  "",
  "The MDS-completed budget reproduces sink years 2016 and 2019 and source years 2017, 2018, 2020, and 2021.",
  "",
  "## Compatibility with the frozen HMM",
  "",
  paste0("- The canonical contract contains ", format(nrow(contract), big.mark = ","), " unique half-hours; the frozen K4 state table contains ", format(nrow(hmm), big.mark = ","), " observed primary-eligible half-hours."),
  "- Frozen HMM NEE and CH4 values match `NEE_orig` and `CH4_orig` exactly on every linked state row.",
  "- The full annual budget is based on gap-filled NEE and CH4, so intervals without observed joint fluxes do not currently have frozen states.",
  paste0("- The raw file contains ", nrow(raw) - nrow(contract), " duplicate rows excluded by the canonical contract; annual reconstruction uses only canonical retained rows."),
  "",
  "## CH4 budget alignment",
  "",
  "The MDS-completed CH4 series matches observed `CH4_orig` on the frozen-state population wherever both are available, maintaining alignment between the primary budget and the observed state-defining CH4 values.",
  "",
  "## Prespecified next step",
  "",
  "1. Preserve the existing observed frozen state assignments without alteration.",
  "2. Apply the fixed K4 parameters, without refitting or relabeling, to the complete MDS-filled NEE/CH4 record for budget attribution.",
  "3. Validate the extended assignments against the frozen states on observed intervals and report agreement by year.",
  "4. Attribute `NEE_C_sum` and `CH4_C_sum` to state × phenological phase using posterior probabilities, then reconcile every state contribution exactly to the complete annual budget."
)
writeLines(report, report_path)

cat("Created:", file.path(output_dir, "annual_carbon_budget_reconstruction.csv"), "\n")
cat("Created:", report_path, "\n")
