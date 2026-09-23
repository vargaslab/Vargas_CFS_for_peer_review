#!/usr/bin/env Rscript

# Sensitivity audit for Figure 4.
#
# Inputs: the same frozen K4 annual state-by-phase attribution used in Figure 4.
# Outputs: GWP-horizon and leave-one-year-out ratio tables used in Table S4.
# Scientific guardrail: no HMM fitting or state relabelling occurs. The audit
# repeats descriptive duration/occupancy/intensity accounting for GWP100 and
# GWP20 using the primary MDS-state/MDS-budget attribution.

suppressPackageStartupMessages({
  library(data.table)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = FALSE)
script_arg <- grep("^--file=", args, value = TRUE)
root <- if (length(script_arg)) {
  normalizePath(file.path(dirname(sub("^--file=", "", script_arg[1])), "../.."), mustWork = TRUE)
} else normalizePath(".", mustWork = TRUE)

input_dir <- file.path(root, "data/publication_inputs")
output_dir <- file.path(root, "outputs/complete_dataset/manuscript/figure_4_state_reorganization_sensitivity")
report_dir <- file.path(root, "reports/complete_dataset")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(report_dir, recursive = TRUE, showWarnings = FALSE)

phases <- c("Greenup", "Maturity", "Senescence", "Dormancy")
branches <- "MDS_state_MDS_budget_primary"
metrics <- c("GWP100", "GWP20")
primary_branch <- branches[1]
carbon_to_co2 <- 44 / 12
carbon_to_ch4 <- 16 / 12
gwp <- c(GWP100 = 27.0, GWP20 = 79.7)

state_phase <- fread(file.path(input_dir, "annual_budget_by_state_phase_all_branches.csv"))
phase_coverage <- fread(file.path(input_dir, "budget_and_state_coverage_by_year_phase.csv"))
phase_hours <- phase_coverage[, .(phase_halfhours = canonical_halfhours[1]),
                              by = .(year, phenological_phase)]
z_all <- merge(state_phase, phase_hours, by = c("year", "phenological_phase"), all.x = TRUE)
stopifnot(!anyNA(z_all$phase_halfhours), all(z_all$posterior_halfhours > 0))

permutations <- list(c(1, 2, 3), c(1, 3, 2), c(2, 1, 3),
                     c(2, 3, 1), c(3, 1, 2), c(3, 2, 1))
shapley_three <- function(reference_values, year_values) {
  contribution <- numeric(3)
  for (order in permutations) {
    current <- reference_values
    for (index in order) {
      before <- prod(current)
      current[index] <- year_values[index]
      contribution[index] <- contribution[index] + prod(current) - before
    }
  }
  contribution / length(permutations)
}

metric_value <- function(dt, metric) {
  multiplier <- gwp[[metric]]
  dt$NEE_C_g_m2 * carbon_to_co2 + dt$CH4_C_g_m2 * carbon_to_ch4 * multiplier
}

decompose <- function(branch, metric, omitted_year = NA_integer_) {
  z <- copy(z_all[attribution_branch == branch])
  if (!is.na(omitted_year)) z <- z[year != omitted_year]
  z[, response_value := metric_value(.SD, metric)]
  z[, `:=`(
    H = phase_halfhours,
    O = posterior_halfhours / phase_halfhours,
    I = response_value / posterior_halfhours
  )]
  reference <- z[, .(H = mean(H), O = mean(O), I = mean(I)),
                 by = .(phenological_phase, canonical_state_id)]
  z <- merge(z, reference, by = c("phenological_phase", "canonical_state_id"),
             suffixes = c("_year", "_reference"))
  by_state <- rbindlist(lapply(seq_len(nrow(z)), function(i) {
    row <- z[i]
    x <- shapley_three(
      c(row$H_reference, row$O_reference, row$I_reference),
      c(row$H_year, row$O_year, row$I_year)
    )
    data.table(
      year = row$year, phenological_phase = row$phenological_phase,
      duration = x[1], occupancy = x[2], intensity = x[3]
    )
  }))
  out <- by_state[, lapply(.SD, sum), by = .(year, phenological_phase),
                  .SDcols = c("duration", "occupancy", "intensity")]
  out[, state_reorganization := occupancy + intensity]
  out
}

ratio_rows <- list()
loyo_rows <- list()
index <- 1L
loyo_index <- 1L
for (branch in branches) {
  for (metric in metrics) {
    d <- decompose(branch, metric)
    ratio_rows[[index]] <- d[, .(
      median_absolute_duration = median(abs(duration)),
      median_absolute_state_reorganization = median(abs(state_reorganization)),
      state_to_duration_ratio = median(abs(state_reorganization)) / median(abs(duration))
    ), by = phenological_phase][, `:=`(attribution_branch = branch, metric = metric)]
    index <- index + 1L

    for (omitted in sort(unique(d$year))) {
      dl <- decompose(branch, metric, omitted)
      loyo_rows[[loyo_index]] <- dl[, .(
        state_to_duration_ratio = median(abs(state_reorganization)) / median(abs(duration))
      ), by = phenological_phase][, `:=`(
        attribution_branch = branch, metric = metric, omitted_year = omitted
      )]
      loyo_index <- loyo_index + 1L
    }
  }
}

ratios <- rbindlist(ratio_rows, use.names = TRUE)
loyo <- rbindlist(loyo_rows, use.names = TRUE)
loyo_range <- loyo[, .(
  minimum_loyo_ratio = min(state_to_duration_ratio),
  maximum_loyo_ratio = max(state_to_duration_ratio),
  loyo_cases_ratio_above_one = sum(state_to_duration_ratio > 1),
  n_loyo_cases = .N
), by = .(attribution_branch, metric, phenological_phase)]
ratios <- merge(ratios, loyo_range,
                by = c("attribution_branch", "metric", "phenological_phase"), all.x = TRUE)
ratios[, phenological_phase := factor(phenological_phase, levels = phases)]
setorder(ratios, attribution_branch, metric, phenological_phase)

fwrite(ratios, file.path(output_dir, "state_to_duration_ratio_sensitivity.csv"))
fwrite(loyo, file.path(output_dir, "state_to_duration_ratio_leave_one_year_out.csv"))

primary <- ratios[attribution_branch == primary_branch & metric == "GWP100"]
prespecified <- ratios
all_above_one <- all(prespecified$state_to_duration_ratio > 1)
all_loyo_above_one <- all(prespecified$loyo_cases_ratio_above_one == prespecified$n_loyo_cases)

metadata <- list(
  analysis = "Figure 4 state-reorganization sensitivity",
  hmm_refitted = FALSE,
  state_labels_reassigned = FALSE,
  prespecified_comparisons = list(
    primary_GWP100 = "MDS-state/MDS-budget primary branch, GWP100",
    alternative_metric = "GWP20"
  ),
  all_prespecified_full_sample_ratios_above_one = all_above_one,
  all_prespecified_leave_one_year_out_ratios_above_one = all_loyo_above_one,
  interpretation = "Ratios quantify relative descriptive contribution magnitude and are not causal effect sizes."
)
write_json(metadata, file.path(output_dir, "sensitivity_metadata.json"), pretty = TRUE, auto_unbox = TRUE)

fmt <- function(x) formatC(x, digits = 1, format = "f")
lines <- c(
  "# Figure 4 sensitivity: state reorganization versus phenophase duration",
  "",
  "The fixed K4 attribution was not refitted or relabelled. Ratios compare the median absolute carbon-flux-state-reorganization contribution with the median absolute phenophase-duration contribution across years.",
  "",
  "## Primary GWP100 result",
  "",
  "| Phenophase | Full-sample ratio | Leave-one-year-out range | LOO cases > 1 |",
  "|---|---:|---:|---:|",
  vapply(seq_len(nrow(primary)), function(i) sprintf(
    "| %s | %s× | %s–%s× | %d/%d |",
    as.character(primary$phenological_phase[i]), fmt(primary$state_to_duration_ratio[i]),
    fmt(primary$minimum_loyo_ratio[i]), fmt(primary$maximum_loyo_ratio[i]),
    primary$loyo_cases_ratio_above_one[i], primary$n_loyo_cases[i]
  ), character(1)),
  "",
  "## Prespecified robustness conclusion",
  "",
  if (all_above_one) {
    "Every full-sample ratio remained above one across GWP100 and GWP20 under the primary MDS-state/MDS-budget attribution."
  } else {
    "At least one prespecified full-sample sensitivity produced a ratio at or below one; the dominance claim must be qualified."
  },
  if (all_loyo_above_one) {
    "Every leave-one-year-out ratio also remained above one across both GWP horizons."
  } else {
    "At least one leave-one-year-out ratio was at or below one, so the robustness conclusion must be qualified."
  },
  "",
  "These are descriptive robustness checks, not inferential tests or evidence that HMM states are causal mechanisms."
)
writeLines(lines, file.path(report_dir, "figure_4_state_reorganization_sensitivity.md"))
cat("Created Figure 4 sensitivity outputs in", output_dir, "\n")
