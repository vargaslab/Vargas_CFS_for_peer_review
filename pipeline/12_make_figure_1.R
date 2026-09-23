#!/usr/bin/env Rscript

# Main Figure 1: seasonal flux context, the multiannual carbon-state
# fingerprint, and the joint NEE-CH4 state space.
#
# Inputs: frozen K4 state assignments and the half-hourly data contract.
# Outputs: Figure_1_all_years_carbon_fingerprint.{pdf,png} plus one CSV file
# per panel in outputs/complete_dataset/manuscript/figure_1_all_years_carbon_fingerprint.
# Scientific guardrail: the HMM is never refitted or relabelled here. Daily
# summaries and state probabilities are averaged with equal weight per year.

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

hmm_file <- file.path(root, "data/processed/stjones_hmm_k4_primary_v1.csv.gz")
contract_file <- file.path(root, "data/processed/stjones_halfhourly_contract_v1.csv.gz")
figure_dir <- file.path(root, "figures/manuscript/main")
output_dir <- file.path(root, "outputs/complete_dataset/manuscript/figure_1_all_years_carbon_fingerprint")
report_dir <- file.path(root, "reports/complete_dataset")
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

states <- c("C1", "C2", "C3", "C4")
phase_levels <- c("Dormancy", "Greenup", "Maturity", "Senescence")
font_family <- "Arial"
state_colours <- c(C1 = "#0072B2", C2 = "#E69F00", C3 = "#009E73", C4 = "#CC79A7")
phase_colours <- c(
  Dormancy = "#B8B8B8", Greenup = "#5AAE61",
  Maturity = "#238443", Senescence = "#D95F0E"
)

theme_nature <- function() {
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
      legend.key.height = unit(3, "mm"),
      legend.key.width = unit(4.2, "mm"),
      plot.tag = element_text(family = font_family, face = "bold", size = 10.5,
                              colour = "#111111"),
      plot.tag.position = c(0.004, 0.995),
      plot.margin = margin(2, 3, 2, 2, unit = "mm")
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
  } else {
    png(filename, width = width, height = height, units = "in", res = 600,
        type = "cairo", ...)
  }
}
read_gz <- function(path) as.data.table(read.csv(gzfile(path), stringsAsFactors = FALSE))

hmm <- read_gz(hmm_file)
contract <- read_gz(contract_file)
stopifnot(nrow(hmm) == 63156L, !anyDuplicated(hmm$source_row),
          !anyDuplicated(contract$source_row))
hmm[, year := as.integer(substr(date, 1, 4))]
hmm[, doy := as.integer(format(as.Date(date), "%j"))]
hmm[, hour_decimal := as.integer(substr(timestamp_start, 12, 13)) +
                    as.integer(substr(timestamp_start, 15, 16)) / 60]
hmm[, canonical_state_id := factor(canonical_state_id, levels = states)]
contract[, year := as.integer(substr(date, 1, 4))]
contract[, doy := as.integer(format(as.Date(date), "%j"))]

# Select the example year solely by the count of jointly observed fluxes.
year_coverage <- hmm[, .(
  joint_observed_halfhours = .N,
  potential_halfhours = as.integer(format(as.Date(paste0(year[1], "-12-31")), "%j")) * 48L,
  observed_dates = uniqueN(date),
  mean_maximum_posterior = mean(max_posterior)
), by = year]
year_coverage[, coverage_percent := 100 * joint_observed_halfhours / potential_halfhours]
setorder(year_coverage, -joint_observed_halfhours, year)
example_year <- year_coverage$year[1]
stopifnot(example_year == 2016L)
year_coverage[, selection_rank := .I]

# Figure 1 summarizes all years with equal year weights rather than displaying
# one potentially unrepresentative annual realization.
last_doy <- 366L
analysis_years <- sort(unique(hmm$year))

# ---- Panel A: year-balanced daily observed flux context ----
daily_year <- hmm[, .(
  n_joint_halfhours = .N,
  NEE_daily_mean = mean(NEE),
  CH4_daily_mean = mean(CH4)
), by = .(year, doy)]
# A daily mean is only used when at least one quarter of its half-hours has a
# joint observation. Years are then weighted equally within each calendar day.
daily_year[n_joint_halfhours < 12L, `:=`(NEE_daily_mean = NA_real_, CH4_daily_mean = NA_real_)]
daily <- daily_year[is.finite(NEE_daily_mean) & is.finite(CH4_daily_mean), .(
  n_years = .N,
  NEE_daily_mean = mean(NEE_daily_mean),
  NEE_q25 = quantile(NEE_daily_mean, 0.25),
  NEE_q75 = quantile(NEE_daily_mean, 0.75),
  CH4_daily_mean = mean(CH4_daily_mean),
  CH4_q25 = quantile(CH4_daily_mean, 0.25),
  CH4_q75 = quantile(CH4_daily_mean, 0.75)
), by = doy]
daily <- merge(data.table(doy = seq_len(last_doy)), daily, by = "doy", all.x = TRUE, sort = TRUE)
rolling_mean <- function(x, width = 7L, minimum_n = 4L) {
  n <- length(x)
  half_width <- floor(width / 2)
  vapply(seq_len(n), function(i) {
    values <- x[max(1L, i - half_width):min(n, i + half_width)]
    if (sum(is.finite(values)) < minimum_n) NA_real_ else mean(values, na.rm = TRUE)
  }, numeric(1))
}
daily[, `:=`(
  NEE_7d_mean = rolling_mean(NEE_daily_mean),
  CH4_7d_mean = rolling_mean(CH4_daily_mean)
)]
nee_limits <- c(-15, 8)
ch4_limits <- c(-25, 250)
ch4_to_nee <- function(x) nee_limits[1] + (x - ch4_limits[1]) *
  diff(nee_limits) / diff(ch4_limits)
nee_to_ch4 <- function(x) ch4_limits[1] + (x - nee_limits[1]) *
  diff(ch4_limits) / diff(nee_limits)
daily[, `:=`(
  CH4_daily_scaled = ch4_to_nee(CH4_daily_mean),
  CH4_7d_scaled = ch4_to_nee(CH4_7d_mean),
  CH4_q25_scaled = ch4_to_nee(CH4_q25),
  CH4_q75_scaled = ch4_to_nee(CH4_q75)
)]

p_a <- ggplot(daily, aes(doy)) +
  geom_hline(yintercept = 0, colour = "#8A8A8A", linewidth = 0.32, linetype = "22") +
  geom_ribbon(aes(ymin = NEE_q25, ymax = NEE_q75), fill = alpha("#0072B2", 0.13),
              colour = NA, na.rm = TRUE) +
  geom_ribbon(aes(ymin = CH4_q25_scaled, ymax = CH4_q75_scaled), fill = alpha("#E69F00", 0.13),
              colour = NA, na.rm = TRUE) +
  geom_line(aes(y = NEE_daily_mean), colour = alpha("#0072B2", 0.30), linewidth = 0.25,
            na.rm = TRUE) +
  geom_line(aes(y = CH4_daily_scaled), colour = alpha("#E69F00", 0.30), linewidth = 0.25,
            na.rm = TRUE) +
  geom_line(aes(y = NEE_7d_mean), colour = "#0072B2", linewidth = 0.78, na.rm = TRUE) +
  geom_line(aes(y = CH4_7d_scaled), colour = "#E69F00", linewidth = 0.78, na.rm = TRUE) +
  scale_x_continuous(breaks = seq(1, last_doy, by = 60), expand = expansion(mult = c(0, 0))) +
  scale_y_continuous(
    limits = nee_limits, breaks = c(-15, -10, -5, 0, 5),
    sec.axis = sec_axis(~ nee_to_ch4(.), name = expression(
      "CH"[4] * " flux (nmol CH"[4] * " " * m^{-2} * " " * s^{-1} * ")"
    ), breaks = c(0, 50, 100, 150, 200, 250))
  ) +
  labs(
    x = NULL,
    y = expression("NEE (" * mu * "mol CO"[2] * " " * m^{-2} * " " * s^{-1} * ")"),
    tag = "A"
  ) +
  theme_nature() +
  theme(
    axis.title.y = element_text(colour = "#0072B2"),
    axis.text.y = element_text(colour = "#0072B2"),
    axis.title.y.right = element_text(colour = "#C57A00"),
    axis.text.y.right = element_text(colour = "#C57A00"),
    axis.line.y.right = element_line(colour = "#C57A00", linewidth = 0.35),
    axis.ticks.y.right = element_line(colour = "#C57A00", linewidth = 0.30),
    panel.grid.major = element_line(colour = "#E8E8E8", linewidth = 0.25),
    panel.grid.minor = element_blank(),
    axis.text.x = element_blank(), axis.ticks.x = element_blank(),
    plot.margin = margin(2, 7, 0, 2, unit = "mm")
  )

# ---- Panel B: year-balanced posterior fingerprint and mean phenophase dates ----
phase_runs_by_year <- rbindlist(lapply(analysis_years, function(y) {
  x <- unique(contract[year == y, .(doy, phenological_phase)])
  setorder(x, doy)
  x[, phase_run := rleid(phenological_phase)]
  x[, .(year = y, phase = phenological_phase[1], start_doy = min(doy), end_doy = max(doy)),
    by = phase_run]
}))
transition_dates <- phase_runs_by_year[phase != "Dormancy", .(start_doy = start_doy[1]),
                                        by = .(year, phase)]
late_dormancy <- phase_runs_by_year[phase == "Dormancy" & start_doy > 1,
                                    .(dormancy_start_doy = start_doy[1]), by = year]
transition_dates <- dcast(transition_dates, year ~ phase, value.var = "start_doy")
transition_dates <- merge(transition_dates, late_dormancy, by = "year", all.x = TRUE)
transition_summary <- melt(
  transition_dates, id.vars = "year", variable.name = "transition", value.name = "doy"
)[, .(mean_doy = mean(doy), sd_doy = sd(doy), minimum_doy = min(doy), maximum_doy = max(doy)),
  by = transition]
mean_boundaries <- transition_summary[, setNames(mean_doy, transition)]
phase_daily <- data.table(doy = seq_len(last_doy))
phase_daily[, phenological_phase := fifelse(
  doy < mean_boundaries[["Greenup"]] | doy >= mean_boundaries[["dormancy_start_doy"]], "Dormancy",
  fifelse(doy < mean_boundaries[["Maturity"]], "Greenup",
          fifelse(doy < mean_boundaries[["Senescence"]], "Maturity", "Senescence"))
)]
phase_daily[, phenological_phase := factor(phenological_phase, levels = phase_levels)]
phase_daily[, phase_run := rleid(phenological_phase)]
phase_runs <- phase_daily[!is.na(phenological_phase), .(
  middle_doy = round((min(doy) + max(doy)) / 2)
), by = .(phase_run, phenological_phase)]
p_phase <- ggplot(phase_daily, aes(doy, 1, fill = phenological_phase)) +
  geom_tile(width = 1, height = 1) +
  geom_text(
    data = phase_runs,
    aes(middle_doy, 1, label = phenological_phase), inherit.aes = FALSE,
    family = font_family, size = 2.35, colour = "#111111"
  ) +
  scale_fill_manual(values = phase_colours, drop = FALSE) +
  scale_x_continuous(limits = c(0.5, last_doy + 0.5), expand = c(0, 0)) +
  theme_void(base_family = font_family) +
  theme(legend.position = "none", plot.margin = margin(0, 7, 0, 2, unit = "mm"))

fingerprint <- hmm[, .(
  n_years = uniqueN(year),
  posterior_C1 = mean(posterior_C1), posterior_C2 = mean(posterior_C2),
  posterior_C3 = mean(posterior_C3), posterior_C4 = mean(posterior_C4)
), by = .(doy, hour_decimal)]
fingerprint <- merge(CJ(doy = seq_len(last_doy), hour_decimal = seq(0, 23.5, by = 0.5)),
                     fingerprint, by = c("doy", "hour_decimal"), all.x = TRUE, sort = TRUE)
posterior_cols <- paste0("posterior_", states)
fingerprint[, mean_posterior_support := do.call(pmax, c(.SD, na.rm = TRUE)), .SDcols = posterior_cols]
fingerprint[!is.finite(mean_posterior_support), mean_posterior_support := NA_real_]

# Visual-only continuity fill. Posterior probabilities are linearly interpolated
# along the calendar at each half-hour; leading/trailing gaps use the nearest
# available day. No fitted or interpolated values enter any analysis.
linear_calendar_fill <- function(x) {
  observed <- which(is.finite(x))
  if (!length(observed)) return(rep(NA_real_, length(x)))
  if (length(observed) == 1L) return(rep(x[observed], length(x)))
  approx(x = observed, y = x[observed], xout = seq_along(x), method = "linear", rule = 2)$y
}
visual_posterior_cols <- paste0("visual_", states)
fingerprint[, (visual_posterior_cols) := lapply(.SD, linear_calendar_fill),
            by = hour_decimal, .SDcols = posterior_cols]
fingerprint[, visual_posterior_sum := rowSums(.SD), .SDcols = visual_posterior_cols]
for (column in visual_posterior_cols) {
  fingerprint[, (column) := get(column) / visual_posterior_sum]
}
visual_probability_matrix <- as.matrix(fingerprint[, ..visual_posterior_cols])
fingerprint[, visual_interpolated := is.na(n_years) | n_years < 3L]
fingerprint[, dominant_state := states[max.col(visual_probability_matrix, ties.method = "first")]]
fingerprint[, display_state := factor(dominant_state, levels = states)]
fingerprint[, visual_posterior_support := do.call(pmax, .SD), .SDcols = visual_posterior_cols]
fingerprint[, support_alpha := pmax(0.25, pmin(1, visual_posterior_support))]
fingerprint[visual_interpolated == TRUE, support_alpha := pmin(support_alpha, 0.52)]
p_b <- ggplot(fingerprint, aes(doy, hour_decimal, fill = display_state, alpha = support_alpha)) +
  geom_tile(width = 1, height = 0.5) +
  scale_fill_manual(values = state_colours, drop = FALSE) +
  scale_alpha_continuous(range = c(0.30, 1), limits = c(0.25, 1), guide = "none") +
  scale_x_continuous(
    limits = c(0.5, last_doy + 0.5), breaks = seq(1, last_doy, by = 60),
    expand = c(0, 0)
  ) +
  scale_y_continuous(
    limits = c(-0.25, 23.75), breaks = c(0, 6, 12, 18, 23.5),
    labels = c("00:00", "06:00", "12:00", "18:00", "24:00"), expand = c(0, 0)
  ) +
  labs(x = "Day of year", y = "Hour of day",
       fill = "Ecosystem-carbon-flux state", tag = "B") +
  theme_nature() +
  guides(fill = guide_legend(nrow = 1, byrow = TRUE, override.aes = list(alpha = 1))) +
  theme(
    legend.position = "bottom", legend.box.just = "left",
    legend.box.margin = margin(-1, 0, 0, 0, unit = "mm"), panel.grid = element_blank(),
    plot.margin = margin(0, 7, 2, 2, unit = "mm")
  )

# ---- Panel C: all-years joint response-space geometry ----
state_summary <- hmm[, .(
  n_halfhours = .N,
  state_percentage = 100 * .N / nrow(hmm),
  NEE_mean = mean(NEE), CH4_mean = mean(CH4),
  mean_maximum_posterior = mean(max_posterior)
), by = canonical_state_id]
state_summary[, canonical_state_id := factor(canonical_state_id, levels = states)]
state_descriptions <- c(
  C1 = "atop(CO[2]*' uptake', 'moderate '*CH[4])",
  C2 = "atop('near-neutral '*CO[2], 'low '*CH[4])",
  C3 = "atop(CO[2]*' release', 'moderate '*CH[4])",
  C4 = "atop(CO[2]*' release', 'high '*CH[4])"
)
label_positions <- data.table(
  canonical_state_id = factor(states, levels = states),
  label_x = c(-24, -7, 16, 21),
  label_y = c(270, -58, 80, 510)
)
state_summary <- merge(state_summary, label_positions, by = "canonical_state_id", all.x = TRUE)
state_summary[, display_label := sprintf(
  "atop('%s  %.1f%%', %s)", as.character(canonical_state_id), state_percentage,
  state_descriptions[as.character(canonical_state_id)]
)]
p_c <- ggplot(hmm, aes(NEE, CH4, colour = canonical_state_id)) +
  geom_hline(yintercept = 0, linewidth = 0.28, colour = "#9A9A9A", linetype = "22") +
  geom_vline(xintercept = 0, linewidth = 0.28, colour = "#9A9A9A", linetype = "22") +
  geom_point(shape = 16, size = 0.48, alpha = 0.080, stroke = 0) +
  geom_segment(
    data = state_summary,
    aes(x = NEE_mean, y = CH4_mean, xend = label_x, yend = label_y,
        colour = canonical_state_id),
    inherit.aes = FALSE, linewidth = 0.34
  ) +
  geom_point(
    data = state_summary,
    aes(x = NEE_mean, y = CH4_mean, fill = canonical_state_id),
    inherit.aes = FALSE, shape = 21, colour = "white", stroke = 0.65, size = 3.05
  ) +
  geom_label(
    data = state_summary,
    aes(x = label_x, y = label_y, label = display_label, colour = canonical_state_id),
    inherit.aes = FALSE, family = font_family, fontface = "bold", size = 2.65,
    lineheight = 0.94, hjust = 0.5, parse = TRUE,
    fill = alpha("white", 0.90), label.padding = unit(0.15, "mm"),
    label.r = unit(0, "mm"), linewidth = 0
  ) +
  annotate(
    "text", x = -34, y = 635, hjust = 0, vjust = 1,
    label = sprintf("All eligible half-hours  (n = %s)", comma(nrow(hmm))),
    family = font_family, size = 2.35, colour = "#4A4A4A"
  ) +
  scale_colour_manual(values = state_colours[states], drop = FALSE) +
  scale_fill_manual(values = state_colours[states], drop = FALSE) +
  scale_x_continuous(breaks = seq(-30, 40, by = 10), expand = expansion(mult = c(0, 0))) +
  scale_y_continuous(breaks = seq(-100, 600, by = 100), expand = expansion(mult = c(0, 0))) +
  coord_cartesian(xlim = c(-35, 40), ylim = c(-150, 650), clip = "on") +
  labs(
    x = expression("NEE (" * mu * "mol CO"[2] * " " * m^{-2} * " " * s^{-1} * ")"),
    y = expression("CH"[4] * " flux (nmol CH"[4] * " " * m^{-2} * " " * s^{-1} * ")"),
    tag = "C"
  ) +
  theme_nature() +
  theme(legend.position = "none", panel.grid = element_blank(),
        plot.margin = margin(2, 3, 2, 2, unit = "mm"))

figure_1 <- p_a / (p_phase / p_b + plot_layout(heights = c(0.12, 1))) / p_c +
  plot_layout(heights = c(0.88, 1.18, 1.04))
pdf_path <- file.path(figure_dir, "Figure_1_all_years_carbon_fingerprint.pdf")
png_path <- file.path(figure_dir, "Figure_1_all_years_carbon_fingerprint.png")
ggsave(pdf_path, figure_1, width = 180, height = 178, units = "mm",
       device = pdf_device, bg = "white")
ggsave(png_path, figure_1, width = 180, height = 178, units = "mm",
       device = png_device, bg = "white")

fwrite(year_coverage, file.path(output_dir, "example_year_selection_coverage.csv"))
fwrite(daily, file.path(output_dir, "panel_A_daily_observed_fluxes.csv"))
fwrite(fingerprint, file.path(output_dir, "panel_B_halfhourly_state_fingerprint.csv"))
fwrite(state_summary, file.path(output_dir, "panel_C_all_years_state_summary.csv"))
fwrite(transition_dates, file.path(output_dir, "panel_B_phenophase_transition_dates_by_year.csv"))
fwrite(transition_summary, file.path(output_dir, "panel_B_phenophase_transition_date_summary.csv"))

metadata <- list(
  figure = "Figure 1",
  title = "Seasonal fingerprint and joint state space of ecosystem carbon exchange",
  hmm_refitted = FALSE,
  state_labels_reassigned = FALSE,
  abandoned_single_year_example = list(
    objective_coverage_rule = "maximum count of jointly observed NEE-CH4 half-hours",
    selected_year_under_that_rule = example_year,
    reason_not_used = "all-years seasonal composite better represents interannual state variation"
  ),
  panel_A = list(
    summary = "daily means first computed within year; then years weighted equally within day of year",
    minimum_joint_halfhours_per_year_day = 12,
    bands = "interquartile range across contributing years"
  ),
  panel_B = list(
    state_display = "dominant state from year-balanced mean posterior probabilities",
    minimum_years_per_day_hour_cell = 3,
    saturation = "mean posterior support for displayed dominant state",
    visual_gap_fill = "linear interpolation of posterior probabilities along day of year within each half-hour; nearest observed value at calendar ends; visualization only",
    phenology = "mean annual PhenoCam transition dates across six years"
  ),
  panel_C = list(population = "all 63156 frozen-HMM-eligible half-hours across 2016-2021",
                 point_layer = "all eligible observations; no sampling"),
  interpretation_boundary = "State changes are flux-scale regimes, not individual phenological transitions.",
  figure_width_mm = 180, figure_height_mm = 178, png_dpi = 600,
  pdf = sub(paste0("^", root, "/"), "", pdf_path),
  png = sub(paste0("^", root, "/"), "", png_path)
)
write_json(metadata, file.path(output_dir, "figure_1_metadata.json"), pretty = TRUE, auto_unbox = TRUE)

caption <- paste0(
  "# Figure 1. Seasonal fingerprint and joint state space of ecosystem carbon exchange\n\n",
  "**(A)** Year-balanced seasonal dynamics of observed NEE and CH4 flux from 2016–2021. Daily means were calculated separately within each year from days with at least 12 jointly observed half-hours and then averaged with equal weight across contributing years at each day of year. Thin lines show daily means, bold lines centered 7-day means, and shaded bands the interquartile range among years. Negative NEE indicates net ecosystem CO2 uptake. ",
  "**(B)** Year-balanced half-hourly fingerprint of the frozen four-state HMM. Posterior probabilities were averaged equally among contributing years within each day-of-year × hour-of-day cell; color identifies the state with the largest mean posterior probability and saturation gives its posterior support. Cells represented by fewer than three years were linearly interpolated along day of year at the same hour for visual continuity only and were excluded from quantitative analyses. The phenophase strip uses mean PhenoCam transition dates across the six years. ",
  "**(C)** Joint NEE–CH4 response space for all 63,156 eligible half-hours. Colors denote unchanged canonical carbon-flux states; large symbols are hard-state means and labels give all-years prevalence. The HMM resolves recurrent flux-scale regimes, so individual state changes in the fingerprint are not interpreted as phenological transitions."
)
writeLines(caption, file.path(report_dir, "figure_1_all_years_carbon_fingerprint_caption.md"))
cat("Created:", pdf_path, "\n")
cat("Created:", png_path, "\n")
