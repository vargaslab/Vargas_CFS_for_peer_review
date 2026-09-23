#!/usr/bin/env Rscript

# Rebuild the canonical publication package from frozen project outputs.
#
# This is intentionally a small orchestration script rather than a second
# analysis implementation. Each downstream script remains independently
# runnable and documents its scientific calculation. This runner provides the
# single reproducible command used for the manuscript deliverables.
#
# Usage:
#   Rscript scripts/pipeline/20_reproduce_publication.R             # all deliverables
#   Rscript scripts/pipeline/20_reproduce_publication.R --main      # Figures 1–4 only
#   Rscript scripts/pipeline/20_reproduce_publication.R --supplement # Figures S1–S3 only
#   Rscript scripts/pipeline/20_reproduce_publication.R --tables    # Tables S1–S4 only
#   Rscript scripts/pipeline/20_reproduce_publication.R --validate  # validate existing files

args <- commandArgs(trailingOnly = TRUE)
valid_modes <- c("--main", "--supplement", "--tables", "--validate")
if (length(args) > 1L || (length(args) == 1L && !args %in% valid_modes)) {
  stop("Use zero arguments or one of: ", paste(valid_modes, collapse = ", "), call. = FALSE)
}

script_file <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (!length(script_file)) stop("Run this file with Rscript.", call. = FALSE)
root <- normalizePath(file.path(dirname(sub("^--file=", "", script_file[1])), "../.."), mustWork = TRUE)
mode <- if (length(args)) args else "--all"

required_inputs <- c(
  "data/processed/stjones_hmm_k4_primary_v1.csv.gz",
  "data/processed/stjones_halfhourly_contract_v1.csv.gz",
  "data/processed/stjones_barometric_pressure_v1.csv.gz",
  "data/publication_inputs/annual_budget_by_state_phase_all_branches.csv",
  "data/publication_inputs/budget_and_state_coverage_by_year_phase.csv",
  "data/publication_inputs/daily_state_compositions.csv",
  "data/publication_inputs/model_selection.csv",
  "data/publication_inputs/stability_summary_by_k.csv",
  "data/publication_inputs/leave_one_year_out_summary.csv",
  "data/publication_inputs/primary_k4_state_fingerprints_final.csv",
  "data/publication_inputs/halfhourly_relative_gains.csv",
  "data/publication_inputs/daily_relative_gains.csv"
)
missing_inputs <- required_inputs[!file.exists(file.path(root, required_inputs))]
if (length(missing_inputs)) {
  stop("Required frozen inputs are missing:\n", paste(missing_inputs, collapse = "\n"), call. = FALSE)
}

run_r <- function(script) {
  # Run every stage from an absolute path while preserving the caller's working
  # directory independence. system2 returns the child status without a shell.
  path <- file.path(root, "scripts", "pipeline", script)
  message("\n[publication] Running ", script)
  status <- system2("Rscript", path, stdout = "", stderr = "")
  if (status != 0L) stop("Failed: ", script, call. = FALSE)
}
run_python <- function(script) {
  # The active Conda environment is preferred, but PYTHON provides an explicit
  # override for orchestrated or non-interactive runs.
  python_candidates <- unique(c(
    Sys.getenv("PYTHON", unset = ""),
    Sys.which("python3"),
    Sys.which("python")
  ))
  python_candidates <- python_candidates[nzchar(python_candidates)]
  if (!length(python_candidates)) {
    stop("Python with xlsxwriter is required to build Supplementary_Tables.xlsx. Create the environment in environment.yml first.", call. = FALSE)
  }
  has_xlsxwriter <- vapply(python_candidates, function(candidate) {
    status <- suppressWarnings(system2(
      candidate, c("-c", shQuote("import xlsxwriter")),
      stdout = FALSE, stderr = FALSE
    ))
    identical(status, 0L)
  }, logical(1))
  python_candidates <- python_candidates[has_xlsxwriter]
  if (!length(python_candidates)) {
    stop(
      "No discovered Python interpreter can import xlsxwriter. Activate environment.yml or set PYTHON to a compatible interpreter.",
      call. = FALSE
    )
  }
  path <- file.path(root, "scripts", "pipeline", script)
  message("\n[publication] Running ", script)
  status <- system2(python_candidates[1], path, stdout = "", stderr = "")
  if (status != 0L) stop("Failed: ", script, call. = FALSE)
}

expected_files <- function() {
  main_stems <- c(
    "Figure_1_all_years_carbon_fingerprint",
    "Figure_2_all_years_phenology_scale_bridge",
    "Figure_3_phase_specific_environmental_context",
    "Figure_4_state_reorganization_vs_duration"
  )
  supplement_stems <- c(
    "Figure_S1_HMM_selection_and_reproducibility",
    "Figure_S2_phenology_predictive_gain",
    "Figure_S3_environmental_cross_year_validation"
  )
  main_figures <- unlist(lapply(main_stems, function(stem) {
    file.path(root, "figures/manuscript/main", paste0(stem, c(".pdf", ".png")))
  }))
  supplement_figures <- unlist(lapply(supplement_stems, function(stem) {
    file.path(root, "figures/manuscript/supplementary", paste0(stem, c(".pdf", ".png")))
  }))

  source_files <- c(
    "figure_1_all_years_carbon_fingerprint/panel_A_daily_observed_fluxes.csv",
    "figure_1_all_years_carbon_fingerprint/panel_B_halfhourly_state_fingerprint.csv",
    "figure_1_all_years_carbon_fingerprint/panel_C_all_years_state_summary.csv",
    "figure_2_all_years_phenology_scale_bridge/panel_A_year_balanced_hour_phase.csv",
    "figure_2_all_years_phenology_scale_bridge/panel_B_year_balanced_daily_composition.csv",
    "figure_3_phase_specific_environment/cross_fitted_phase_specific_environmental_effects_by_year.csv",
    "figure_3_phase_specific_environment/leave_one_year_out_phase_specific_predictive_gains.csv",
    "figure_3_phase_specific_environment/phase_specific_environmental_fingerprint_summary.csv",
    "figure_4_state_reorganization/shapley_contributions_by_state_phase_gwp100.csv",
    "figure_4_state_reorganization/phase_duration_vs_state_reorganization_gwp100.csv",
    "figure_4_state_reorganization/median_absolute_contribution_ratios_gwp100.csv",
    "figure_4_state_reorganization_sensitivity/state_to_duration_ratio_sensitivity.csv",
    "figure_4_state_reorganization_sensitivity/state_to_duration_ratio_leave_one_year_out.csv",
    "canonical_supplementary_package/Table_S1_candidate_model_selection.csv",
    "canonical_supplementary_package/Table_S2_CFS_definitions.csv",
    "canonical_supplementary_package/Table_S3_K4_leave_one_year_out.csv",
    "canonical_supplementary_package/Table_S4_GHG_sensitivity_summary.csv",
    "canonical_supplementary_package/supplementary_tables_source.json"
  )

  c(
    main_figures,
    supplement_figures,
    file.path(root, "outputs/complete_dataset/manuscript", source_files),
    file.path(root, "outputs/complete_dataset/manuscript/canonical_supplementary_package/Supplementary_Tables.xlsx")
  )
}
validate_outputs <- function() {
  files <- expected_files()
  missing <- files[!file.exists(files)]
  # Size gates catch interrupted or empty writes without pretending to validate
  # the scientific content, which is covered by the regression tests.
  minimum_size <- ifelse(
    grepl("\\.xlsx$", files, ignore.case = TRUE), 5000L,
    ifelse(grepl("\\.png$", files, ignore.case = TRUE), 10000L,
           ifelse(grepl("\\.pdf$", files, ignore.case = TRUE), 1000L, 20L))
  )
  too_small <- files[file.exists(files) & file.info(files)$size < minimum_size[file.exists(files)]]
  if (length(missing) || length(too_small)) {
    message("\nPublication artefact validation failed.")
    if (length(missing)) message("Missing:\n", paste(missing, collapse = "\n"))
    if (length(too_small)) message("Suspiciously small:\n", paste(too_small, collapse = "\n"))
    stop("Rebuild the affected deliverables before submission.", call. = FALSE)
  }
  message("\nPublication artefact validation passed: ", length(files), " required files are present.")
}

if (mode %in% c("--all", "--main")) {
  for (script in c(
    "12_make_figure_1.R",
    "13_make_figure_2.R",
    "15_make_figure_3.R",
    "16_make_figure_4.R",
    "17_audit_figure_4_sensitivity.R"
  )) run_r(script)
}
if (mode %in% c("--all", "--supplement", "--tables")) {
  run_r("18_make_supplementary_figures_and_sources.R")
}
if (mode %in% c("--all", "--tables")) run_python("19_build_supplementary_tables.py")
if (mode %in% c("--all", "--validate")) validate_outputs()
