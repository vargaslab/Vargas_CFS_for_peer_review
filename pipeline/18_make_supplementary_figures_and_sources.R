#!/usr/bin/env Rscript

# Canonical supplementary package for the Communications Earth & Environment
# submission.
#
# Inputs: frozen HMM selection/stability outputs, phenology audit outputs,
# Figure 3 cross-year results, and Figure 4 sensitivity results.
# Outputs: Figures S1-S3 plus the machine-readable CSV source tables used to
# build Tables S1-S4. The workbook itself is formatted by script 19.
# Scientific guardrail: this script uses frozen reanalysis outputs only; it
# does not fit, refit, select, or relabel an HMM.

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(patchwork)
  library(scales)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = FALSE)
script_arg <- grep("^--file=", args, value = TRUE)
root <- if (length(script_arg)) {
  normalizePath(file.path(dirname(sub("^--file=", "", script_arg[1])), "../.."), mustWork = TRUE)
} else normalizePath(".", mustWork = TRUE)

out_dir <- file.path(root, "outputs/complete_dataset/manuscript/canonical_supplementary_package")
fig_dir <- file.path(root, "figures/manuscript/supplementary")
report_dir <- file.path(root, "reports/complete_dataset")
input_dir <- file.path(root, "data/publication_inputs")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(report_dir, recursive = TRUE, showWarnings = FALSE)

font_family <- "Arial"
state_cols <- c(C1 = "#0072B2", C2 = "#E69F00", C3 = "#009E73", C4 = "#CC79A7")
phase_cols <- c(Dormancy = "#6B6B6B", Greenup = "#009E73", Maturity = "#0072B2", Senescence = "#D55E00")
phases <- c("Greenup", "Maturity", "Senescence", "Dormancy")
year_cols <- setNames(c("#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#E6AB02"), as.character(2016:2021))

theme_journal <- function() {
  theme_classic(base_family = font_family, base_size = 7.5) +
    theme(
      plot.background = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA),
      axis.line = element_line(colour = "#242424", linewidth = 0.35),
      axis.ticks = element_line(colour = "#242424", linewidth = 0.30),
      axis.ticks.length = unit(1.2, "mm"),
      axis.text = element_text(colour = "#242424", size = 7),
      axis.title = element_text(colour = "#111111", size = 8),
      legend.title = element_text(size = 7),
      legend.text = element_text(size = 7),
      plot.tag = element_text(family = font_family, face = "bold", size = 11),
      plot.margin = margin(3, 4, 3, 3, unit = "mm")
    )
}
pdf_device <- function(filename, width, height, ...) {
  if (identical(Sys.info()[["sysname"]], "Darwin")) {
    quartz(type = "pdf", file = filename, width = width, height = height,
           family = font_family, ...)
  } else cairo_pdf(filename, width = width, height = height, family = font_family, ...)
}
png_device <- function(filename, width, height, ...) {
  if (identical(Sys.info()[["sysname"]], "Darwin")) {
    png(filename, width = width, height = height, units = "in", res = 600,
        type = "quartz", ...)
  } else png(filename, width = width, height = height, units = "in", res = 600,
             type = "cairo", ...)
}
save_figure <- function(p, stem, height_mm, width_mm = 180) {
  ggsave(file.path(fig_dir, paste0(stem, ".pdf")), p, width = width_mm, height = height_mm,
         units = "mm", device = pdf_device, bg = "white")
  ggsave(file.path(fig_dir, paste0(stem, ".png")), p, width = width_mm, height = height_mm,
         units = "mm", device = png_device, bg = "white")
}

# ----- Figure S1: model choice and broad-state reproducibility -----
model_selection <- fread(file.path(input_dir, "model_selection.csv"))[covariance_type == "full" & K %in% 2:6]
stability <- fread(file.path(input_dir, "stability_summary_by_k.csv"))[K %in% 4:6]
loyo <- fread(file.path(input_dir, "leave_one_year_out_summary.csv"))[K == 4]
setorder(model_selection, K); setorder(stability, K); setorder(loyo, omitted_year)

p_s1a <- ggplot(model_selection, aes(x = factor(K), y = BIC / 1000, fill = factor(K))) +
  geom_col(width = 0.66, colour = "white", linewidth = 0.25) +
  geom_text(aes(label = sprintf("%.1f", BIC / 1000)), vjust = -0.38, size = 2.35, family = font_family) +
  scale_fill_manual(values = c("2" = "#BDBDBD", "3" = "#BDBDBD", "4" = "#0072B2", "5" = "#BDBDBD", "6" = "#D55E00"), guide = "none") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.08))) +
  labs(x = "Number of states", y = "BIC (×10³)", tag = "A") +
  theme_journal() + theme(panel.grid.major.y = element_line(colour = "#EAEAEA", linewidth = 0.25))

stability_long <- melt(stability[, .(K, pairwise_ARI_mean, winner_recovery_ARI_ge_0_95)],
                       id.vars = "K", variable.name = "metric", value.name = "value")
stability_long[, metric := factor(metric,
  levels = c("pairwise_ARI_mean", "winner_recovery_ARI_ge_0_95"),
  labels = c("Pairwise ARI", "Representative recovery"))]
p_s1b <- ggplot(stability_long, aes(x = K, y = value, colour = metric, shape = metric)) +
  geom_line(linewidth = 0.45) + geom_point(size = 2.45, stroke = 0.65) +
  scale_x_continuous(breaks = 4:6) +
  scale_y_continuous(limits = c(0, 1.05), breaks = c(0, 0.25, 0.50, 0.75, 1.00),
                     labels = label_number(accuracy = 0.01)) +
  scale_colour_manual(values = c("Pairwise ARI" = "#0072B2", "Representative recovery" = "#D55E00")) +
  labs(x = "Number of states", y = "Reproducibility", colour = NULL, shape = NULL, tag = "B") +
  theme_journal() + theme(panel.grid.major.y = element_line(colour = "#EAEAEA", linewidth = 0.25), legend.position = "bottom")

p_s1c <- ggplot(loyo, aes(x = factor(omitted_year), y = omitted_year_assignment_ARI)) +
  geom_hline(yintercept = 0.90, linetype = "dashed", colour = "#777777", linewidth = 0.35) +
  geom_point(shape = 21, size = 3.0, stroke = 0.65, fill = "#0072B2", colour = "#202020") +
  geom_text(aes(label = sprintf("%.3f", omitted_year_assignment_ARI)), vjust = -0.8, size = 2.25, family = font_family) +
  scale_y_continuous(limits = c(0.84, 1.01), breaks = c(0.85, 0.90, 0.95, 1.00), labels = label_number(accuracy = 0.01)) +
  labs(x = "Omitted year", y = "Held-out assignment ARI", tag = "C") +
  theme_journal() + theme(panel.grid.major.y = element_line(colour = "#EAEAEA", linewidth = 0.25))

figure_s1 <- p_s1a + p_s1b + p_s1c + plot_layout(ncol = 3, widths = c(1, 1.12, 1.03))
save_figure(figure_s1, "Figure_S1_HMM_selection_and_reproducibility", 65)

# ----- Figure S2: cross-year predictive contribution of phenology -----
half <- fread(file.path(input_dir, "halfhourly_relative_gains.csv"))[
  response_type == "posterior" & model_id == "F_diel_PAR_phase_GCC",
  .(held_out_year, relative_brier_gain_percent)
]
half[, scale := "Half-hourly"]
daily <- fread(file.path(input_dir, "daily_relative_gains.csv"))[
  threshold == "primary" & model_id == "D_phase_GCC",
  .(held_out_year, light_regime, relative_brier_gain_percent)
]
daily[, scale := fifelse(light_regime == "Light", "Daily light", "Daily dark")]
seasonal_gain <- rbindlist(list(half[, .(held_out_year, scale, relative_brier_gain_percent)],
                                daily[, .(held_out_year, scale, relative_brier_gain_percent)]))
seasonal_gain[, scale := factor(scale, levels = c("Half-hourly", "Daily light", "Daily dark"))]

p_s2 <- ggplot(seasonal_gain, aes(x = scale, y = relative_brier_gain_percent, colour = scale)) +
  geom_hline(yintercept = 0, colour = "#777777", linewidth = 0.35) +
  geom_point(position = position_jitter(width = 0.08, height = 0), size = 2.05) +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 3.0, fill = "white", colour = "#111111", stroke = 0.7) +
  scale_colour_manual(values = c("Half-hourly" = "#6B6B6B", "Daily light" = "#E69F00", "Daily dark" = "#0072B2"), guide = "none") +
  labs(x = NULL, y = "Brier improvement from\nphenophase + GCC (%)") +
  theme_journal() + theme(panel.grid.major.y = element_line(colour = "#EAEAEA", linewidth = 0.25))

save_figure(p_s2, "Figure_S2_phenology_predictive_gain", 75, width_mm = 95)

# ----- Figure S3: cross-year environmental validation -----
env_gain <- fread(file.path(root, "outputs/complete_dataset/manuscript/figure_3_phase_specific_environment/leave_one_year_out_phase_specific_predictive_gains.csv"))
env_gain[, comparison := fifelse(comparison_id == "environment_beyond_diel_timing",
                                 "Environmental context beyond diel timing", "WL beyond environmental context")]
env_gain[, phenological_phase := factor(phenological_phase, levels = names(phase_cols))]
env_gain[, held_out_year := factor(held_out_year, levels = 2016:2021)]

p_s3 <- ggplot(env_gain, aes(x = relative_brier_gain_percent, y = phenological_phase, colour = held_out_year)) +
  geom_vline(xintercept = 0, colour = "#777777", linewidth = 0.35) +
  geom_point(position = position_jitter(height = 0.11, width = 0), size = 2.0) +
  stat_summary(aes(group = phenological_phase), fun = mean, geom = "point", shape = 23, size = 3.2,
               fill = "white", colour = "#111111", stroke = 0.75) +
  facet_wrap(~comparison, nrow = 1, scales = "free_x") +
  scale_colour_manual(values = year_cols, name = "Held-out year") +
  labs(x = "Held-out-year Brier improvement (%)", y = "Phenological phase") +
  theme_journal() + theme(panel.grid.major.x = element_line(colour = "#EAEAEA", linewidth = 0.25),
                          strip.background = element_rect(fill = "#F1F1F1", colour = "#242424", linewidth = 0.35),
                          strip.text = element_text(size = 7.5, face = "bold"), legend.position = "bottom")
save_figure(p_s3, "Figure_S3_environmental_cross_year_validation", 75)

# ----- Table input files: the spreadsheet builder formats these for publication -----
# AIC/BIC and posterior summaries come from the common candidate-model fit.
# Stability metrics use the final available start ensemble for each K. In
# particular, the expanded K6 audit (13 attempts, 9 converged) supersedes the
# initial five-start screen (2 converged) for reproducibility reporting.
stability_for_table <- rbindlist(list(
  model_selection[!K %in% stability$K, .(
    K,
    stability_ensemble = "initial candidate ensemble",
    attempted_starts = successful_fit_count,
    converged_starts = converged_fit_count,
    mean_pairwise_ARI = pairwise_ari_mean,
    representative_recovery = winner_recovery_rate_ari_ge_0_95,
    centroid_SD_RMS = centroid_sd_rms
  )],
  stability[, .(
    K,
    stability_ensemble = fifelse(K == 6L, "expanded audit (13 starts)",
                                 "focused audit (10 starts)"),
    attempted_starts,
    converged_starts,
    mean_pairwise_ARI = pairwise_ARI_mean,
    representative_recovery = winner_recovery_ARI_ge_0_95,
    centroid_SD_RMS
  )]
), use.names = TRUE)

table_s1_model <- merge(
  model_selection[, .(
    K, AIC, BIC,
    mean_posterior_certainty = mean_maximum_posterior,
    minimum_state_occupancy
  )],
  stability_for_table,
  by = "K", all.x = TRUE
)
setorder(table_s1_model, K)
table_s1_model <- table_s1_model[, .(
  K, AIC, BIC, mean_posterior_certainty, minimum_state_occupancy,
  converged_attempted_starts = sprintf("%d/%d", converged_starts, attempted_starts),
  mean_pairwise_ARI,
  representative_recovery_percent = 100 * representative_recovery,
  centroid_variability_RMS_SD = centroid_SD_RMS
)]

table_s2_states <- fread(file.path(input_dir, "primary_k4_state_fingerprints_final.csv"))[
  , .(
    CFS = canonical_state_id,
    carbon_flux_signature = canonical_state_label,
    mean_NEE = NEE_mean,
    mean_CH4_flux = CH4_mean,
    occupancy,
    posterior_certainty,
    self_transition_probability,
    nearest_centroid_distance = nearest_centroid_distance_standardized
  )
]

table_s3_loyo <- loyo[, .(
  omitted_year,
  held_out_observations = test_n,
  converged_starts,
  start_stability_ARI,
  held_out_assignment_ARI = omitted_year_assignment_ARI,
  same_state_fraction = omitted_year_same_state_fraction
)]

ratio_sensitivity <- fread(file.path(root, "outputs/complete_dataset/manuscript/figure_4_state_reorganization_sensitivity/state_to_duration_ratio_sensitivity.csv"))
ratio_sensitivity[, `:=`(
  metric_order = match(metric, c("GWP100", "GWP20")),
  phase_order = match(as.character(phenological_phase), phases)
)]
setorder(ratio_sensitivity, metric_order, phase_order)
table_s4_sensitivity <- ratio_sensitivity[, .(
  GWP_horizon = metric,
  phenophase = as.character(phenological_phase),
  full_sample_ratio = state_to_duration_ratio,
  leave_one_year_out_range = sprintf("%.2f–%.2f", minimum_loyo_ratio, maximum_loyo_ratio)
)]

fwrite(table_s1_model, file.path(out_dir, "Table_S1_candidate_model_selection.csv"))
fwrite(table_s2_states, file.path(out_dir, "Table_S2_CFS_definitions.csv"))
fwrite(table_s3_loyo, file.path(out_dir, "Table_S3_K4_leave_one_year_out.csv"))
fwrite(table_s4_sensitivity, file.path(out_dir, "Table_S4_GHG_sensitivity_summary.csv"))

write_json(list(
  Table_S1 = table_s1_model,
  Table_S2 = table_s2_states,
  Table_S3 = table_s3_loyo,
  Table_S4 = table_s4_sensitivity
), file.path(out_dir, "supplementary_tables_source.json"), dataframe = "rows", pretty = TRUE,
auto_unbox = TRUE, na = "null")

captions <- c(
  "# Canonical supplementary-figure captions",
  "",
  "## Figure S1. Selection and reproducibility of the hidden carbon-flux representation",
  "(A) Bayesian information criterion (BIC) for full-covariance Gaussian HMMs with two to six states; lower values indicate improved within-sample fit. (B) Exact partition reproducibility for the focused four-, five-, and six-state fits. Pairwise adjusted Rand index (ARI) measures agreement across converged starts, and representative recovery is the fraction of converged starts that recovered the representative partition at ARI ≥ 0.95. (C) Agreement between the frozen primary K4 assignment and K4 models retrained after omitting each year. The dashed line marks ARI = 0.90. BIC favored K6, whereas K4 provided the reproducible broad representation used in the main figures. No model was refit for any main-text result.",
  "",
  "## Figure S2. Cross-year predictive contribution of phenology to carbon-flux-state composition",
  "Leave-one-year-out relative improvement in posterior-composition Brier score when phenophase and green chromatic coordinate (GCC) are added to the corresponding baseline at the half-hourly scale and for daily light and dark periods. Points represent held-out years and open diamonds show six-year means. Positive values indicate improved prediction after adding phenophase and GCC. This audit used frozen K4 posterior probabilities and did not refit or relabel the HMM.",
  "",
  "## Figure S3. Cross-year validation of environmental differentiation within phenophases",
  "Points are complete held-out years and open diamonds are mean relative Brier-score improvements. The left panel shows improvement from adding the primary environmental context to cyclic diel timing within each phenophase. The right panel shows the incremental contribution of normalized recorded water level (WL) on its matched subset after the primary environmental context. WL denotes relative recorded gauge level, not inundation depth or tidal height. The HMM was not refit or relabelled."
)
writeLines(captions, file.path(report_dir, "canonical_supplementary_figure_captions.md"))

metadata <- list(
  scope = "Canonical three-figure supplementary package for Communications Earth & Environment",
  HMM_refitted = FALSE,
  state_labels_reassigned = FALSE,
  figures = c("Figure S1", "Figure S2", "Figure S3"),
  tables = c(
    "Table S1: candidate-model selection and stability",
    "Table S2: canonical carbon-flux-state definitions",
    "Table S3: K4 leave-one-year-out generalizability",
    "Table S4: greenhouse-gas accounting sensitivity"
  )
)
write_json(metadata, file.path(out_dir, "canonical_supplementary_package_metadata.json"), pretty = TRUE, auto_unbox = TRUE)
cat("Created canonical supplementary figures and table inputs in", out_dir, "\n")
