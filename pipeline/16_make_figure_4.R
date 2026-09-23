#!/usr/bin/env Rscript

# Main Figure 4: compare the interannual contribution of phenophase duration
# with that of within-phase carbon-flux-state reorganization.
#
# Inputs: frozen-K4 annual state-by-phase budget attribution retained as a
# compact publication input.
# Outputs: Figure_4_state_reorganization_vs_duration.{pdf,png} and numerical
# accounting tables in outputs/complete_dataset/manuscript/figure_4_state_reorganization.
# Scientific guardrail: this is descriptive fixed-state accounting. It neither
# refits/relabels the HMM nor estimates causal drivers of annual carbon balance.

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(jsonlite)
  library(scales)
})

args <- commandArgs(trailingOnly = FALSE)
script_arg <- grep("^--file=", args, value = TRUE)
root <- if (length(script_arg)) {
  normalizePath(file.path(dirname(sub("^--file=", "", script_arg[1])), "../.."), mustWork = TRUE)
} else normalizePath(".", mustWork = TRUE)

input_dir <- file.path(root, "data/publication_inputs")
output_dir <- file.path(root, "outputs/complete_dataset/manuscript/figure_4_state_reorganization")
figure_dir <- file.path(root, "figures/manuscript/main")
report_dir <- file.path(root, "reports/complete_dataset")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(report_dir, recursive = TRUE, showWarnings = FALSE)

states <- paste0("C", 1:4)
phases <- c("Greenup", "Maturity", "Senescence", "Dormancy")
primary_branch <- "MDS_state_MDS_budget_primary"
gwp100_nonfossil_ch4 <- 27.0
carbon_to_co2 <- 44 / 12
carbon_to_ch4 <- 16 / 12
font_family <- "Arial"
phase_colours <- c(
  Greenup = "#009E73", Maturity = "#0072B2",
  Senescence = "#D55E00", Dormancy = "#6B6B6B"
)

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
      strip.text = element_text(size = 8.2, face = "bold", colour = "#222222"),
      plot.subtitle = element_text(size = 7.2, colour = "#4A4A4A", hjust = 0.5),
      plot.margin = margin(3, 4, 4, 3, unit = "mm")
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

state_phase <- fread(file.path(input_dir, "annual_budget_by_state_phase_all_branches.csv"))[
  attribution_branch == primary_branch
]
phase_coverage <- fread(file.path(input_dir, "budget_and_state_coverage_by_year_phase.csv"))
phase_hours <- phase_coverage[, .(phase_halfhours = canonical_halfhours[1]),
                              by = .(year, phenological_phase)]
z <- merge(state_phase, phase_hours, by = c("year", "phenological_phase"), all.x = TRUE)
stopifnot(nrow(z) == 24L * 4L, !anyNA(z$phase_halfhours), all(z$posterior_halfhours > 0))

z[, state_gwp100_gco2e_m2 :=
    NEE_C_g_m2 * carbon_to_co2 + CH4_C_g_m2 * carbon_to_ch4 * gwp100_nonfossil_ch4]
z[, `:=`(
  H = phase_halfhours,
  O = posterior_halfhours / phase_halfhours,
  I = state_gwp100_gco2e_m2 / posterior_halfhours
)]
reference <- z[, .(H = mean(H), O = mean(O), I = mean(I)),
               by = .(phenological_phase, canonical_state_id)]
z <- merge(z, reference, by = c("phenological_phase", "canonical_state_id"),
           suffixes = c("_year", "_reference"))

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

by_state <- rbindlist(lapply(seq_len(nrow(z)), function(i) {
  row <- z[i]
  x <- shapley_three(
    c(row$H_reference, row$O_reference, row$I_reference),
    c(row$H_year, row$O_year, row$I_year)
  )
  data.table(
    year = row$year,
    phenological_phase = row$phenological_phase,
    canonical_state_id = row$canonical_state_id,
    duration = x[1], occupancy = x[2], intensity = x[3]
  )
}))
contributions <- by_state[, lapply(.SD, sum),
                          by = .(year, phenological_phase),
                          .SDcols = c("duration", "occupancy", "intensity")]
contributions[, `:=`(
  carbon_flux_state_reorganization = occupancy + intensity,
  total_phase_deviation = duration + occupancy + intensity,
  phenological_phase = factor(phenological_phase, levels = phases),
  year_f = factor(year, levels = rev(sort(unique(year))))
)]

# One compact magnitude summary per phenophase. Median absolute contributions
# retain the effect-size comparison while avoiding cancellation of opposite signs.
magnitude_summary <- contributions[, .(
  median_absolute_duration = median(abs(duration)),
  median_absolute_state_reorganization = median(abs(carbon_flux_state_reorganization))
), by = phenological_phase]
magnitude_summary[, state_to_duration_ratio :=
                    median_absolute_state_reorganization / median_absolute_duration]
magnitude_summary[, phase_order := match(as.character(phenological_phase), phases)]
setorder(magnitude_summary, phase_order)
facet_labels <- setNames(
  as.character(magnitude_summary$phenological_phase),
  as.character(magnitude_summary$phenological_phase)
)
contributions[, phase_facet_label := factor(
  facet_labels[as.character(phenological_phase)],
  levels = unname(facet_labels[phases])
)]

plot_limit <- max(abs(c(contributions$duration, contributions$carbon_flux_state_reorganization)))
plot_limit <- max(200, ceiling(plot_limit / 100) * 100)
plot_breaks <- pretty(c(-plot_limit, plot_limit), n = 5)
magnitude_annotation <- copy(magnitude_summary)[, .(
  phase_facet_label = factor(
    facet_labels[as.character(phenological_phase)],
    levels = unname(facet_labels[phases])
  ),
  year_f = factor(min(as.integer(as.character(contributions$year))),
                  levels = rev(sort(unique(contributions$year)))),
  x = 0.94 * plot_limit,
  label = sprintf("State / duration\n%.1f×", state_to_duration_ratio)
)]

figure_4 <- ggplot(contributions, aes(y = year_f)) +
  geom_vline(xintercept = 0, colour = "#5B5B5B", linewidth = 0.38) +
  geom_segment(
    aes(x = duration, xend = carbon_flux_state_reorganization,
        yend = year_f), colour = "#B8B8B8", linewidth = 0.72
  ) +
  geom_point(
    aes(x = duration), shape = 21, size = 3.1, stroke = 0.72,
    fill = "white", colour = "#555555"
  ) +
  geom_point(
    aes(x = carbon_flux_state_reorganization, fill = phenological_phase),
    shape = 21, size = 3.45, stroke = 0.72, colour = "#202020", show.legend = FALSE
  ) +
  geom_text(
    data = magnitude_annotation, aes(x = x, y = year_f, label = label),
    inherit.aes = FALSE, hjust = 1, vjust = -0.35, family = font_family,
    fontface = "bold", size = 2.55, lineheight = 0.94, colour = "#444444"
  ) +
  facet_wrap(~phase_facet_label, ncol = 2, drop = FALSE) +
  scale_fill_manual(values = phase_colours, drop = FALSE) +
  scale_x_continuous(limits = c(-plot_limit, plot_limit), breaks = plot_breaks,
                     labels = label_number(accuracy = 1),
                     expand = expansion(mult = 0.01)) +
  labs(
    x = expression("Contribution to phase-integrated " * GWP[100] * " deviation (g CO"[2] * "e " * m^{-2} * " " * y^{-1} * ")"),
    y = "Year"
  ) +
  theme_journal() +
  theme(
    panel.grid.major.x = element_line(colour = "#E8E8E8", linewidth = 0.25),
    panel.grid.major.y = element_line(colour = "#F0F0F0", linewidth = 0.22),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(colour = "#4A4A4A", fill = NA, linewidth = 0.35),
    axis.line = element_blank(),
    strip.background = element_rect(fill = "#ECECEC", colour = "#4A4A4A", linewidth = 0.35),
    strip.text = element_text(margin = margin(1.6, 0, 1.6, 0, unit = "mm")),
    panel.spacing = unit(4.2, "mm")
  )

pdf_path <- file.path(figure_dir, "Figure_4_state_reorganization_vs_duration.pdf")
png_path <- file.path(figure_dir, "Figure_4_state_reorganization_vs_duration.png")
ggsave(pdf_path, figure_4, width = 180, height = 128, units = "mm", device = pdf_device, bg = "white")
ggsave(png_path, figure_4, width = 180, height = 128, units = "mm", device = png_device, bg = "white")

fwrite(by_state, file.path(output_dir, "shapley_contributions_by_state_phase_gwp100.csv"))
fwrite(contributions, file.path(output_dir, "phase_duration_vs_state_reorganization_gwp100.csv"))
fwrite(magnitude_summary[, !"phase_order"], file.path(output_dir, "median_absolute_contribution_ratios_gwp100.csv"))
metadata <- list(
  figure = "Figure 4",
  title = "Carbon-flux-state reorganization exceeds phenophase-duration contributions",
  hmm_refitted = FALSE,
  state_labels_reassigned = FALSE,
  data_source = "Existing fixed-K4 budget-attribution output; this script recalculates displayed summaries only.",
  primary_attribution = "MDS-derived state probabilities with MDS-filled NEE and CH4 budgets",
  metric = "GWP100 using AR6 non-fossil CH4 factor 27.0",
  reference = "Equal-year, phase- and state-specific six-year mean",
  identity = "phase duration x state occupancy x within-state joint CO2-CH4 intensity",
  comparison = "phenophase duration contribution versus state-occupancy plus within-state-intensity contribution",
  facet_annotation = "median absolute state-reorganization contribution divided by median absolute duration contribution across six years",
  interpretation = "The contribution decomposition is descriptive, not causal attribution."
)
write_json(metadata, file.path(output_dir, "figure_4_metadata.json"), pretty = TRUE, auto_unbox = TRUE)

caption <- paste0(
  "# Figure 4. Carbon-flux-state reorganization exceeds the contribution of phenophase duration to interannual greenhouse-gas variation\n\n",
  "Each row is one year and each facet a phenophase. Open circles show the Shapley contribution of phenophase duration to the deviation in phase-integrated GWP100 balance from an equal-year six-year reference. Filled circles show the combined contribution of carbon-flux-state occupancy and within-state joint CO2-CH4 flux intensity. Connecting lines link the two contributions within the same year and phenophase. Values in the upper-right corner are the ratio of the median absolute state-reorganization contribution to the median absolute duration contribution across the six years; values greater than one indicate a larger typical contribution from state reorganization. GWP100 uses the AR6 non-fossil CH4 factor of 27.0. The fixed K4 model was applied without refitting to MDS-completed NEE and CH4 series for state extension and greenhouse-gas accounting. The decomposition describes interannual accounting under this fixed-state representation and does not identify causal mechanisms."
)
writeLines(caption, file.path(report_dir, "figure_4_state_reorganization_vs_duration_caption.md"))
cat("Created:", pdf_path, "\n")
cat("Created:", png_path, "\n")
