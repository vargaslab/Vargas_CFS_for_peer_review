# Script manifest for the final manuscript

This manifest defines the scripts that reproduce the results, figures, and
supplementary tables presented in `manuscript/Vargas_HMM_08262026.docx`.
Only `scripts/pipeline/` is active. Superseded manuscript builders and analyses
that were evaluated but not retained are stored under `archive/` and are not
part of the future GitHub reproduction package.

## Public reproduction from frozen inputs

Run:

```bash
Rscript scripts/pipeline/20_reproduce_publication.R
```

| Order | Script | Manuscript output |
|---:|---|---|
| 12 | `12_make_figure_1.R` | Figure 1 and its numerical source files |
| 13 | `13_make_figure_2.R` | Figure 2 and daily CFS-composition summaries |
| 14 | `14_environment_data_and_model_helpers.R` | Shared validated preparation for Figure 3 |
| 15 | `15_make_figure_3.R` | Figure 3, environmental contrasts, and held-out-year gains |
| 16 | `16_make_figure_4.R` | Figure 4 and primary GWP100 Shapley accounting |
| 17 | `17_audit_figure_4_sensitivity.R` | GWP20 and leave-one-year-out results for Table S4 |
| 18 | `18_make_supplementary_figures_and_sources.R` | Figures S1-S3 and source CSVs for Tables S1-S4 |
| 19 | `19_build_supplementary_tables.py` | Publication-ready `Supplementary_Tables.xlsx` |
| 20 | `20_reproduce_publication.R` | Orchestrates and validates all publication deliverables |

## Controlled reconstruction from the registered raw table

Run `python scripts/pipeline/00_reproduce_from_raw.py --dry-run` to inspect the
full sequence. The raw table is not intended for GitHub distribution.

| Order | Script | Role in the final evidence chain |
|---:|---|---|
| 00 | `00_reproduce_from_raw.py` | Runs the complete dependency-ordered workflow |
| 01 | `01_audit_and_freeze_data_contract.py` | Validates source fields, QC rules, units, and frozen contracts |
| 02 | `02_fit_primary_hmm.py` | Fits candidate carbon-only HMMs used in Figure S1 and Table S1 |
| 03 | `03_temporal_scale_diagnostic.py` | Decision-support diagnostic used by the stability audit |
| 04 | `04_hmm_state_stability.py` | Produces start-stability and K6 sensitivity evidence |
| 05 | `05_hmm_qc_robustness.py` | Evaluates the data-perturbation robustness described in Methods |
| 06 | `06_hmm_generalizability.py` | Produces leave-one-year-out evidence for Figure S1 and Table S3 |
| 07 | `07_freeze_canonical_state_model.py` | Freezes canonical K4 assignments and state definitions |
| 08 | `08_diel_phenology_scale_audit.R` | Produces the Figure S2 predictive gains and Figure 2 inputs |
| 09 | `09_annual_carbon_balance_feasibility.R` | Validates phase budgets and coverage used by Figures 4 and Table S4 |
| 10 | `10_extend_fixed_hmm_for_budget.R` | Applies fixed K4 parameters to MDS-completed budget records |
| 11 | `11_freeze_publication_inputs.py` | Validates and freezes the nine compact inputs consumed by scripts 12-20 |

Scripts 03 and 05 do not create standalone manuscript figures, but they remain
active because later state-selection and generalizability steps depend on them
and the Methods explicitly reports temporal and data-perturbation diagnostics.

## Archived work

See `archive/superseded_workflows/README.md`. Archived scripts are preserved
for provenance only and must not be invoked by the active pipeline.
