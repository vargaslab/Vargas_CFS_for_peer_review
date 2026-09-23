#!/usr/bin/env Rscript

# Main Figure 3: phase-specific environmental context of frozen K4 carbon-flux
# states.
#
# Inputs: the shared environmental preparation and frozen K4 posterior
# probabilities.
# Outputs: Figure_3_phase_specific_environmental_context.{pdf,png}, complete
# conditional-effect estimates, and leave-one-year-out prediction summaries.
# Scientific guardrail: models describe conditional predictive contrasts after
# diel timing and phenophase; they are not causal effects and never alter HMM
# assignments. The dedicated helper is reused to avoid duplicating data
# cleaning, barometric-pressure validation, and water-level normalization.

args <- commandArgs(trailingOnly = FALSE)
script_arg <- grep("^--file=", args, value = TRUE)
root <- if (length(script_arg)) {
  normalizePath(file.path(dirname(sub("^--file=", "", script_arg[1])), "../.."), mustWork = TRUE)
} else normalizePath(".", mustWork = TRUE)

sys.source(file.path(root, "scripts/pipeline/14_environment_data_and_model_helpers.R"), envir = environment())

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(scales)
  library(jsonlite)
})

figure_dir <- file.path(root, "figures/manuscript/main")
output_dir <- file.path(root, "outputs/complete_dataset/manuscript/figure_3_phase_specific_environment")
report_dir <- file.path(root, "reports/complete_dataset")
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

environment_data <- prepare_environment_data(root)
states <- environment_data$states
post_cols <- environment_data$posterior_columns
years <- environment_data$years
phases <- environment_data$phase_levels
core <- copy(environment_data$core)
extended <- copy(environment_data$extended)
core_predictors <- environment_data$core_predictors
focal_predictors <- environment_data$focal_predictors
extended_predictors <- environment_data$extended_predictors
harmonic_terms <- environment_data$harmonic_terms
font_family <- "sans"

theme_journal <- function() {
  theme_classic(base_family = font_family, base_size = 7.5) +
    theme(
      plot.background = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA),
      axis.line = element_line(colour = "#242424", linewidth = 0.35),
      axis.ticks = element_line(colour = "#242424", linewidth = 0.30),
      axis.ticks.length = unit(1.3, "mm"),
      axis.text = element_text(colour = "#242424", size = 7),
      axis.title = element_text(colour = "#111111", size = 8),
      legend.title = element_text(size = 7, colour = "#111111"),
      legend.text = element_text(size = 7, colour = "#242424"),
      plot.tag = element_text(family = font_family, face = "bold", size = 10.5),
      plot.tag.position = c(0.004, 0.995), plot.margin = margin(2, 3, 2, 2, unit = "mm")
    )
}
pdf_device <- function(filename, width, height, ...) {
  if (identical(Sys.info()[["sysname"]], "Darwin")) {
    quartz(type = "pdf", file = filename, width = width, height = height, family = font_family, ...)
  } else cairo_pdf(filename, width = width, height = height, family = font_family, ...)
}
png_device <- function(filename, width, height, ...) {
  if (identical(Sys.info()[["sysname"]], "Darwin")) {
    png(filename, width = width, height = height, units = "in", res = 600, type = "quartz", ...)
  } else png(filename, width = width, height = height, units = "in", res = 600, type = "cairo", ...)
}

fit_phase_models <- function(train, environmental_predictors = character()) {
  environment_terms <- if (length(environmental_predictors)) {
    paste0("z_", environmental_predictors)
  } else character()
  rhs <- c(harmonic_terms, environment_terms)
  lapply(post_cols, function(response) {
    glm(
      as.formula(paste(response, "~", paste(rhs, collapse = " + "))),
      data = train, family = quasibinomial("logit"), control = glm.control(maxit = 75)
    )
  })
}
predict_phase_states <- function(models, newdata) {
  pred <- vapply(models, function(model) {
    predict(model, newdata = newdata, type = "response")
  }, numeric(nrow(newdata)))
  renormalize_state_probabilities(pred)
}
date_mean_change <- function(change, dates) {
  x <- as.data.table(change)
  setnames(x, states)
  x[, date := dates]
  as.numeric(x[, lapply(.SD, mean), by = date, .SDcols = states][, lapply(.SD, mean), .SDcols = states])
}

effect_rows <- list()
gain_rows <- list()
effect_index <- 1L
gain_index <- 1L
for (phase in phases) {
  for (held_out_year in years) {
    train <- copy(core[year != held_out_year & phenological_phase == phase])
    test <- copy(core[year == held_out_year & phenological_phase == phase])
    stopifnot(nrow(train) > 100L, nrow(test) > 20L)
    scaled <- scale_environment_fold(train, test, core_predictors)
    baseline_models <- fit_phase_models(scaled$training)
    environment_models <- fit_phase_models(scaled$training, core_predictors)
    observed <- as.matrix(test[, ..post_cols])
    pred_baseline <- predict_phase_states(baseline_models, scaled$test)
    pred_environment <- predict_phase_states(environment_models, scaled$test)
    gain_rows[[gain_index]] <- data.table(
      held_out_year = held_out_year, phenological_phase = phase,
      comparison_id = "environment_beyond_diel_timing",
      baseline_brier = date_balanced_brier(observed, pred_baseline, test$date),
      augmented_brier = date_balanced_brier(observed, pred_environment, test$date),
      n_halfhours = nrow(test), n_dates = uniqueN(test$date)
    )
    gain_index <- gain_index + 1L

    for (v in focal_predictors) {
      q <- quantile(train[[v]], c(0.25, 0.75), names = FALSE)
      low <- copy(test); high <- copy(test)
      low[, (v) := q[1]]
      high[, (v) := q[2]]
      low <- apply_environment_scaling(low, core_predictors, scaled$parameters)
      high <- apply_environment_scaling(high, core_predictors, scaled$parameters)
      change <- 100 * (predict_phase_states(environment_models, high) -
                       predict_phase_states(environment_models, low))
      average_change <- date_mean_change(change, test$date)
      for (j in seq_along(states)) {
        effect_rows[[effect_index]] <- data.table(
          held_out_year = held_out_year, phenological_phase = phase,
          canonical_variable = v, canonical_state_id = states[j],
          change_percentage_points = average_change[j],
          low_training_quartile = q[1], high_training_quartile = q[2],
          population = "primary_environment"
        )
        effect_index <- effect_index + 1L
      }
    }

    train_w <- copy(extended[year != held_out_year & phenological_phase == phase])
    test_w <- copy(extended[year == held_out_year & phenological_phase == phase])
    stopifnot(nrow(train_w) > 100L, nrow(test_w) > 20L)
    scaled_w <- scale_environment_fold(train_w, test_w, extended_predictors)
    environment_subset_models <- fit_phase_models(scaled_w$training, core_predictors)
    water_models <- fit_phase_models(scaled_w$training, extended_predictors)
    observed_w <- as.matrix(test_w[, ..post_cols])
    pred_environment_subset <- predict_phase_states(environment_subset_models, scaled_w$test)
    pred_water <- predict_phase_states(water_models, scaled_w$test)
    gain_rows[[gain_index]] <- data.table(
      held_out_year = held_out_year, phenological_phase = phase,
      comparison_id = "water_beyond_environment",
      baseline_brier = date_balanced_brier(observed_w, pred_environment_subset, test_w$date),
      augmented_brier = date_balanced_brier(observed_w, pred_water, test_w$date),
      n_halfhours = nrow(test_w), n_dates = uniqueN(test_w$date)
    )
    gain_index <- gain_index + 1L

    q_water <- quantile(train_w$water_level_relative, c(0.25, 0.75), names = FALSE)
    low_w <- copy(test_w); high_w <- copy(test_w)
    low_w[, water_level_relative := q_water[1]]
    high_w[, water_level_relative := q_water[2]]
    low_w <- apply_environment_scaling(low_w, extended_predictors, scaled_w$parameters)
    high_w <- apply_environment_scaling(high_w, extended_predictors, scaled_w$parameters)
    change_w <- 100 * (predict_phase_states(water_models, high_w) -
                       predict_phase_states(water_models, low_w))
    average_change_w <- date_mean_change(change_w, test_w$date)
    for (j in seq_along(states)) {
      effect_rows[[effect_index]] <- data.table(
        held_out_year = held_out_year, phenological_phase = phase,
        canonical_variable = "water_level_relative", canonical_state_id = states[j],
        change_percentage_points = average_change_w[j],
        low_training_quartile = q_water[1], high_training_quartile = q_water[2],
        population = "hydrologic_extension"
      )
      effect_index <- effect_index + 1L
    }
  }
}

effects <- rbindlist(effect_rows)
gains <- rbindlist(gain_rows)
gains[, relative_brier_gain_percent := 100 * (baseline_brier - augmented_brier) / baseline_brier]
effect_summary <- effects[, .(
  mean_change_percentage_points = mean(change_percentage_points),
  minimum_change_percentage_points = min(change_percentage_points),
  maximum_change_percentage_points = max(change_percentage_points),
  same_sign_years = max(sum(change_percentage_points > 0), sum(change_percentage_points < 0)),
  n_years = .N
), by = .(population, phenological_phase, canonical_variable, canonical_state_id)]
gain_summary <- gains[, .(
  mean_gain_percent = mean(relative_brier_gain_percent),
  minimum_gain_percent = min(relative_brier_gain_percent),
  maximum_gain_percent = max(relative_brier_gain_percent),
  positive_years = sum(relative_brier_gain_percent > 0), n_years = .N
), by = .(comparison_id, phenological_phase)]

variable_labels <- c(
  par_log = "PAR", air_temperature = "Tair",
  vapor_pressure_deficit = "VPD", salinity = "Salinity",
  barometric_pressure = "BP", water_level_relative = "WL"
)
effect_summary[, phenological_phase := factor(phenological_phase, levels = phases)]
effect_summary[, canonical_state_id := factor(canonical_state_id, levels = rev(states))]
effect_summary[, variable_label := factor(variable_labels[canonical_variable],
                                            levels = unname(variable_labels))]
effect_summary[, stable := same_sign_years >= 5L]
effect_summary[, display_value := sprintf("%+.1f", mean_change_percentage_points)]
effect_limit <- max(5, ceiling(max(abs(effect_summary$mean_change_percentage_points)) / 5) * 5)
effect_summary[, text_colour := fifelse(
  abs(mean_change_percentage_points) >= 0.55 * effect_limit, "white", "#222222"
)]

p <- ggplot(effect_summary, aes(variable_label, canonical_state_id,
                                fill = mean_change_percentage_points)) +
  geom_tile(colour = "white", linewidth = 0.75) +
  geom_text(data = effect_summary[stable == FALSE],
            aes(label = display_value, colour = text_colour), family = font_family,
            fontface = "plain", size = 2.35, show.legend = FALSE) +
  geom_text(data = effect_summary[stable == TRUE],
            aes(label = display_value, colour = text_colour), family = font_family,
            fontface = "bold", size = 2.35, show.legend = FALSE) +
  scale_colour_identity() +
  scale_fill_gradient2(
    low = "#2166AC", mid = "#F7F7F7", high = "#B2182B", midpoint = 0,
    limits = c(-effect_limit, effect_limit), oob = squish,
    breaks = pretty(c(-effect_limit, effect_limit), n = 5),
    labels = label_number(accuracy = 1)
  ) +
  facet_wrap(~ phenological_phase, ncol = 2) +
  labs(x = NULL, y = "Ecosystem-carbon-flux state",
       fill = "Change in predicted\nstate occurrence\n(percentage points)") +
  guides(fill = guide_colourbar(barheight = unit(34, "mm"), barwidth = unit(5.2, "mm"))) +
  theme_journal() +
  theme(
    axis.line = element_blank(), axis.ticks = element_blank(),
    axis.text.x = element_text(size = 7, lineheight = 0.9),
    axis.text.y = element_text(size = 7.5), legend.position = "right",
    strip.background = element_rect(fill = "#F1F1F1", colour = "#242424", linewidth = 0.35),
    panel.spacing = unit(3.1, "mm"), plot.margin = margin(2, 2, 2, 2, unit = "mm")
  )

pdf_path <- file.path(figure_dir, "Figure_3_phase_specific_environmental_context.pdf")
png_path <- file.path(figure_dir, "Figure_3_phase_specific_environmental_context.png")
ggsave(pdf_path, p, width = 180, height = 122, units = "mm", device = pdf_device, bg = "white")
ggsave(png_path, p, width = 180, height = 122, units = "mm", device = png_device, bg = "white")

effect_export <- copy(effect_summary)
effect_export[, `:=`(phenological_phase = as.character(phenological_phase),
                     canonical_state_id = as.character(canonical_state_id),
                     variable_label = as.character(variable_label))]
gain_export <- copy(gain_summary)
gain_export[, phenological_phase := as.character(phenological_phase)]
fwrite(effects, file.path(output_dir, "cross_fitted_phase_specific_environmental_effects_by_year.csv"))
fwrite(effect_export, file.path(output_dir, "phase_specific_environmental_fingerprint_summary.csv"))
fwrite(gains, file.path(output_dir, "leave_one_year_out_phase_specific_predictive_gains.csv"))
fwrite(gain_export, file.path(output_dir, "phase_specific_predictive_gain_summary.csv"))
fwrite(data.table(
  population = c("all_HMM_eligible", "primary_environment", "hydrologic_extension"),
  n_halfhours = c(environment_data$all_hmm_rows, nrow(core), nrow(extended)),
  n_dates = c(environment_data$all_hmm_dates, uniqueN(core$date), uniqueN(extended$date)),
  n_years = c(length(years), uniqueN(core$year), uniqueN(extended$year))
), file.path(output_dir, "analysis_population_summary.csv"))

metadata <- list(
  figure = "Figure 3",
  title = "Phenophase-specific environmental context of recurring carbon-flux states",
  hmm_refitted = FALSE, state_labels_reassigned = FALSE,
  response = "frozen K4 posterior state probabilities",
  model = "separate phase-specific quasibinomial posterior models with three cyclic Fourier harmonics of hour of day and additive environmental predictors",
  validation = "leave one complete year out within each phenological phase; scaling and percentile contrasts estimated in training years",
  conditional_effect = "average held-out-year change in predicted posterior occupancy from the phase-specific training-year 25th to 75th percentile",
  primary_environment = list(
    variables = list("log1p(PAR)", "VPD", "air temperature", "log1p(half-hour precipitation)",
                     "salinity", "wind speed", "BP_MET barometric pressure"),
    common_population_halfhours = nrow(core)
  ),
  hydrologic_extension = list(
    variable = "within-year empirical percentile of observed Level_YSI, 0-1",
    common_population_halfhours = nrow(extended), missing_values_imputed = FALSE,
    interpretation = "relative recorded water level only; not inundation depth or tidal height"
  ),
  stable_cell_rule = "same effect direction in at least five of six held-out years",
  interpretation_boundary = "conditional predictive associations with state occurrence, not causal effects or next-half-hour transition predictors",
  figure_width_mm = 180, figure_height_mm = 122, png_dpi = 600
)
write_json(metadata, file.path(output_dir, "figure_3_metadata.json"), pretty = TRUE, auto_unbox = TRUE)

caption <- paste0(
  "# Figure 3. Phenophase-specific environmental context of recurring carbon-flux states\n\n",
  "Cross-fitted average change in predicted occurrence of each frozen joint CO2–CH4 state when the indicated environmental variable increases from its phase-specific training-year 25th to 75th percentile. Each model was fitted separately within the named PhenoCam-derived phenophase and retained observed diel timing and the other environmental conditions. Values are percentage-point changes averaged equally across six held-out years; bold values retain the same direction in at least five years. PAR, Tair, VPD, salinity, and BP were estimated on the primary common environmental population. WL is a separate within-year 0–1 normalization of observed Level_YSI on its matched subset and denotes relative recorded gauge level, not inundation depth or tidal height. Precipitation and wind speed were adjustment variables but are not displayed. Values are conditional predictive associations with state occurrence, not causal effects or predictors of next-half-hour transitions. The HMM was not refitted or relabelled."
)
writeLines(caption, file.path(report_dir, "figure_3_phase_specific_environmental_context_caption.md"))
cat("Created:", pdf_path, "\n")
cat("Created:", png_path, "\n")
