#!/usr/bin/env Rscript

# Main Figure 2: the scale bridge from recurring half-hourly HMM regimes to
# phenophase-specific daily state composition across all study years.
#
# Inputs: frozen K4 posterior probabilities, the phenology data contract, and
# daily state compositions from the diel-phenology audit.
# Outputs: Figure_2_all_years_phenology_scale_bridge.{pdf,png} and panel CSVs.
# Scientific guardrail: the frozen K4 HMM is never refit or relabelled; yearly
# summaries are balanced before calculating the multiannual display.

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

state_file <- file.path(root, "data/processed/stjones_hmm_k4_primary_v1.csv.gz")
contract_file <- file.path(root, "data/processed/stjones_halfhourly_contract_v1.csv.gz")
daily_file <- file.path(root, "data/publication_inputs/daily_state_compositions.csv")
figure_dir <- file.path(root, "figures/manuscript/main")
output_dir <- file.path(root, "outputs/complete_dataset/manuscript/figure_2_all_years_phenology_scale_bridge")
report_dir <- file.path(root, "reports/complete_dataset")
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

states <- c("C1", "C2", "C3", "C4")
phases <- c("Dormancy", "Greenup", "Maturity", "Senescence")
state_colours <- c(C1 = "#0072B2", C2 = "#E69F00", C3 = "#009E73", C4 = "#CC79A7")
font_family <- "Arial"

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
      strip.background = element_rect(fill = "#F1F1F1", colour = NA),
      strip.text = element_text(colour = "#111111", face = "bold", size = 7.3,
                                margin = margin(1.3, 1.2, 1.3, 1.2, unit = "mm")),
      legend.title = element_text(size = 7, colour = "#111111"),
      legend.text = element_text(size = 7, colour = "#242424"),
      legend.key.height = unit(3.2, "mm"),
      legend.key.width = unit(4.5, "mm"),
      plot.tag = element_text(family = font_family, face = "bold", size = 10.5,
                              colour = "#111111"),
      plot.tag.position = c(0.004, 0.995),
      plot.margin = margin(2, 3, 2, 2, unit = "mm")
    )
}

# ---- Panel A: phase-specific diel composition ----
hmm <- as.data.table(read.csv(gzfile(state_file), stringsAsFactors = FALSE))
contract <- as.data.table(read.csv(gzfile(contract_file), stringsAsFactors = FALSE))
stopifnot(nrow(hmm) == 63156L, !anyDuplicated(hmm$source_row), !anyDuplicated(contract$source_row))

x <- merge(
  hmm[, c("source_row", "timestamp_start", paste0("posterior_", states)), with = FALSE],
  contract[, .(source_row, phenological_phase)],
  by = "source_row", all.x = TRUE, sort = FALSE
)
stopifnot(!anyNA(x$phenological_phase), setequal(unique(x$phenological_phase), phases))
x[, year := as.integer(substr(timestamp_start, 1, 4))]
x[, hour_decimal := as.integer(substr(timestamp_start, 12, 13)) +
                     as.integer(substr(timestamp_start, 15, 16)) / 60]
x[, hour_bin_2h := floor(hour_decimal / 2) * 2 + 1]

x_long <- melt(
  x,
  id.vars = c("source_row", "year", "phenological_phase", "hour_bin_2h"),
  measure.vars = paste0("posterior_", states),
  variable.name = "state_variable", value.name = "posterior_probability"
)
x_long[, canonical_state_id := sub("posterior_", "", state_variable)]

panel_a_year <- x_long[, .(
  mean_posterior = mean(posterior_probability),
  n_halfhours = .N
), by = .(year, phenological_phase, hour_bin_2h, canonical_state_id)]
panel_a <- panel_a_year[, .(
  mean_posterior = mean(mean_posterior),
  n_years = uniqueN(year),
  n_halfhours = sum(n_halfhours)
), by = .(phenological_phase, hour_bin_2h, canonical_state_id)]
stopifnot(all(abs(panel_a[, .(sum_posterior = sum(mean_posterior)),
                           by = .(phenological_phase, hour_bin_2h)]$sum_posterior - 1) < 1e-8))
panel_a[, phenological_phase := factor(phenological_phase, levels = phases)]
panel_a[, canonical_state_id := factor(canonical_state_id, levels = rev(states))]

p_a <- ggplot(panel_a, aes(hour_bin_2h, mean_posterior, fill = canonical_state_id)) +
  geom_col(width = 1.82, colour = "white", linewidth = 0.17) +
  facet_wrap(~ phenological_phase, nrow = 1) +
  scale_fill_manual(values = state_colours, breaks = states, drop = FALSE) +
  scale_x_continuous(
    breaks = c(1, 7, 13, 19), labels = c("00", "06", "12", "18"),
    expand = c(0, 0)
  ) +
  scale_y_continuous(
    breaks = c(0, 0.5, 1), labels = percent_format(accuracy = 1),
    expand = c(0, 0)
  ) +
  coord_cartesian(ylim = c(0, 1)) +
  labs(
    x = "Hour of day",
    y = "Year-balanced posterior\nstate composition",
    fill = "Ecosystem-carbon-flux state",
    tag = "A"
  ) +
  theme_journal() +
  theme(
    panel.spacing.x = unit(2.1, "mm"),
    legend.position = "none",
    legend.direction = "horizontal",
    legend.margin = margin(1, 0, 0, 0, unit = "mm"),
    plot.margin = margin(2, 3, 2, 2, unit = "mm")
  )

# ---- Panel B: daily integration, retained separately for light and dark ----
daily <- fread(daily_file)
daily <- daily[light_regime %in% c("Light", "Dark") & eligible_primary == TRUE]
daily_long <- melt(
  daily,
  id.vars = c("date", "year", "phenological_phase", "light_regime"),
  measure.vars = paste0("posterior_", states),
  variable.name = "state_variable", value.name = "daily_posterior_composition"
)
daily_long[, canonical_state_id := sub("posterior_", "", state_variable)]

panel_b_year <- daily_long[, .(
  mean_daily_composition = mean(daily_posterior_composition),
  n_dates = uniqueN(date)
), by = .(year, phenological_phase, light_regime, canonical_state_id)]
panel_b <- panel_b_year[, .(
  mean_daily_composition = mean(mean_daily_composition),
  n_years = uniqueN(year),
  n_dates = sum(n_dates)
), by = .(phenological_phase, light_regime, canonical_state_id)]
stopifnot(all(abs(panel_b[, .(sum_posterior = sum(mean_daily_composition)),
                           by = .(phenological_phase, light_regime)]$sum_posterior - 1) < 1e-8))
stopifnot(all(panel_b$n_years == 6L))
panel_b[, phenological_phase := factor(phenological_phase, levels = phases)]
panel_b[, light_regime := factor(light_regime, levels = c("Light", "Dark"))]
panel_b[, canonical_state_id := factor(canonical_state_id, levels = rev(states))]

p_b <- ggplot(panel_b, aes(phenological_phase, mean_daily_composition, fill = canonical_state_id)) +
  geom_col(width = 0.72, colour = "white", linewidth = 0.25) +
  facet_wrap(~ light_regime, nrow = 1) +
  scale_fill_manual(values = state_colours, breaks = states, drop = FALSE) +
  scale_y_continuous(
    breaks = c(0, 0.5, 1), labels = percent_format(accuracy = 1),
    expand = c(0, 0)
  ) +
  coord_cartesian(ylim = c(0, 1)) +
  labs(
    x = "Phenological phase",
    y = "Year-balanced daily\nstate composition",
    fill = "Ecosystem-carbon-flux state",
    tag = "B"
  ) +
  theme_journal() +
  theme(
    axis.text.x = element_text(angle = 22, hjust = 1),
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.margin = margin(1, 0, 0, 0, unit = "mm"),
    panel.spacing.x = unit(3.0, "mm"),
    plot.margin = margin(3, 3, 2, 2, unit = "mm")
  )

figure_2 <- p_a / p_b + plot_layout(heights = c(1.05, 0.95))

pdf_path <- file.path(figure_dir, "Figure_2_all_years_phenology_scale_bridge.pdf")
png_path <- file.path(figure_dir, "Figure_2_all_years_phenology_scale_bridge.png")
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
ggsave(pdf_path, figure_2, width = 180, height = 128, units = "mm", device = pdf_device, bg = "white")
ggsave(png_path, figure_2, width = 180, height = 128, units = "mm", device = png_device, bg = "white")

panel_a_export <- copy(panel_a)
panel_a_export[, `:=`(
  phenological_phase = as.character(phenological_phase),
  canonical_state_id = as.character(canonical_state_id)
)]
panel_b_export <- copy(panel_b)
panel_b_export[, `:=`(
  phenological_phase = as.character(phenological_phase),
  light_regime = as.character(light_regime),
  canonical_state_id = as.character(canonical_state_id)
)]
fwrite(panel_a_year, file.path(output_dir, "panel_A_year_specific_hour_phase.csv"))
fwrite(panel_a_export, file.path(output_dir, "panel_A_year_balanced_hour_phase.csv"))
fwrite(panel_b_year, file.path(output_dir, "panel_B_year_specific_daily_composition.csv"))
fwrite(panel_b_export, file.path(output_dir, "panel_B_year_balanced_daily_composition.csv"))

metadata <- list(
  figure = "Figure 2",
  title = "Diel carbon regimes aggregate into phenophase-specific daily composition",
  hmm_refitted = FALSE,
  state_labels_reassigned = FALSE,
  state_source = "frozen K4 posterior probabilities",
  panel_A = list(
    population = "all 63,156 HMM-eligible half-hours across 2016-2021",
    aggregation = "mean posterior within year x phenological phase x 2-hour bin, followed by equal-weight mean across six years"
  ),
  panel_B = list(
    population = "daily light and dark records passing primary coverage: at least four eligible half-hours per regime",
    aggregation = "mean daily posterior composition within year x phenological phase x light regime, followed by equal-weight mean across six years",
    light_definition = "PAR > 0"
  ),
  interpretation_boundary = "Diel carbon-exchange regimes are summarized by phenophase; a half-hour state change is not a phenological transition.",
  state_palette = as.list(state_colours),
  figure_width_mm = 180,
  figure_height_mm = 128,
  png_dpi = 600
)
write_json(metadata, file.path(output_dir, "figure_2_metadata.json"), pretty = TRUE, auto_unbox = TRUE)

caption <- paste0(
  "# Figure 2. Diel carbon regimes aggregate into phenophase-specific daily composition\n\n",
  "**(A)** Year-balanced posterior composition of the frozen four-state HMM across 2-h bins of the diel cycle, shown separately for each PhenoCam-derived phenophase. Posterior probabilities were first averaged within year, phase, time bin, and state and then averaged equally across the six years. ",
  "**(B)** The same state mixtures after daily aggregation, shown separately for light (PAR > 0) and dark periods. Daily records required at least four eligible half-hours within the corresponding light or dark period, and annual means received equal weight. Multiple joint CO2–CH4 states recur within every phenophase, but their relative occurrence changes through the annual cycle. The HMM was not refitted or relabelled."
)
writeLines(caption, file.path(report_dir, "figure_2_all_years_phenology_scale_bridge_caption.md"))

cat("Created:", pdf_path, "\n")
cat("Created:", png_path, "\n")
