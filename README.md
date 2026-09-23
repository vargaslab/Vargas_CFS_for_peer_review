# Reproducible analysis and publication workflow

This directory contains the canonical manuscript workflow. The submitted figures
and tables are rebuilt from frozen publication inputs and a **frozen four-state
HMM** fitted to joint half-hourly NEE and CH4. Publication scripts must never
refit, select, or relabel that HMM.

## Run the publication package

From the repository root:

```bash
Rscript scripts/pipeline/20_reproduce_publication.R
```

Optional, focused commands are:

```bash
Rscript scripts/pipeline/20_reproduce_publication.R --main
Rscript scripts/pipeline/20_reproduce_publication.R --supplement
Rscript scripts/pipeline/20_reproduce_publication.R --tables
Rscript scripts/pipeline/20_reproduce_publication.R --validate
```

The full command regenerates the four main figures, three supplementary figures, their numerical source files, and the Excel workbook of Tables S1–S4. It validates that the expected artefacts were created. It does **not** delete figures or outputs.

For the controlled-data reconstruction used to create the frozen inputs, see
[RAW_TO_PUBLICATION.md](RAW_TO_PUBLICATION.md). It requires the registered
local integrated source file and is not part of the public peer-review run.

```bash
python scripts/pipeline/00_reproduce_from_raw.py --dry-run
python scripts/pipeline/00_reproduce_from_raw.py --stage all --hmm-jobs 4
```

## Canonical publication scripts

| Order | Script | Purpose | Primary outputs |
|---:|---|---|---|
| 12 | `pipeline/12_make_figure_1.R` | Seasonal flux context, carbon-state fingerprint, and all-year state space. | Main Figure 1 and panel data |
| 13 | `pipeline/13_make_figure_2.R` | Links diel state mixtures to phenophases across years. | Main Figure 2 and panel data |
| 15 | `pipeline/15_make_figure_3.R` | Estimates phase-specific conditional environmental contrasts and held-out-year validation. | Main Figure 3 and numerical results |
| 16 | `pipeline/16_make_figure_4.R` | Descriptive fixed-state accounting of duration versus state reorganization. | Main Figure 4 and panel data |
| 17 | `pipeline/17_audit_figure_4_sensitivity.R` | Endpoint, attribution-branch, and leave-one-year-out sensitivity audit. | Figure 4 sensitivity tables |
| 18 | `pipeline/18_make_supplementary_figures_and_sources.R` | Builds Figures S1–S3 and all source CSV files for Tables S1–S4. | Supplementary figures and source tables |
| 19 | `pipeline/19_build_supplementary_tables.py` | Formats the four supplementary tables as one publication-ready workbook. | `Supplementary_Tables.xlsx` |

## Scope

`pipeline/` contains the complete documented workflow. Figure 3 uses
`pipeline/14_environment_data_and_model_helpers.R` and its frozen
barometric-pressure input, so publication reproduction does not require the
raw data file.

The authoritative script-to-manuscript map is in
[`SCRIPT_MANIFEST.md`](SCRIPT_MANIFEST.md). Superseded scripts have been moved
to `archive/superseded_workflows/` and are not part of this active directory.

## Reproducibility principles

1. Inputs are read by explicit, repository-relative paths.
2. The primary HMM is always `data/processed/stjones_hmm_k4_primary_v1.csv.gz`.
3. Main-figure summaries use equal weighting among years where stated; this prevents dense years from dominating multiannual summaries.
4. Environmental results are conditional predictive contrasts with leave-one-year-out validation; they are not causal effects.
5. Figure 4 is fixed-state accounting; its sensitivity results must accompany interpretation of the state-reorganization ratio.
6. PDF and 600-dpi PNG versions are exported for every figure. The Excel workbook is built only from the canonical CSV source tables.

## Software

R packages: `data.table`, `ggplot2`, `patchwork`, `scales`, and `jsonlite`.
The table workbook is built with Python and the public `xlsxwriter` package;
both are declared in `environment.yml`.
