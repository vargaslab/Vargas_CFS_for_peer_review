# Code accompanying Vargas (manuscript under review)

This repository provides analysis and figure-generation code accompanying the manuscript:

**Beyond canopy phenology: carbon-flux states reveal functional organization of ecosystem CO2-CH4 exchange**

Rodrigo Vargas.

## Repository status

This repository was prepared to support peer review. It is not the final archival release. The code and documentation may be updated in response to peer-review comments.

After manuscript acceptance, a versioned archival release is expected to be deposited in a permanent repository and assigned a DOI.

## Scope

The repository contains the active Python and R scripts used to audit the data contract, fit and evaluate hidden Markov models (HMMs), freeze the canonical state model, analyze diel and phenological structure, estimate annual greenhouse-gas balances, and generate the main and supplementary figures and tables.

Only the files from the study's active `scripts/pipeline/` directory are distributed here. Original observations, processed datasets, fitted-model objects, frozen publication inputs, rendered figures, manuscript files, and intermediate results are not included.

This code-only repository therefore documents the complete computational workflow, but it is not a self-contained package for reproducing the manuscript's numerical results without the omitted inputs and decision records.

## Repository structure

| **Files** | **Contents** |
| --- | --- |
| `00_reproduce_from_raw.py` | Top-level orchestration of the workflow from the integrated raw dataset |
| `01_audit_and_freeze_data_contract.py`-`07_freeze_canonical_state_model.py` | Data-contract audit, primary HMM fitting, temporal-scale diagnostics, state-stability and quality-control analyses, generalizability tests, and canonical-model selection |
| `08_diel_phenology_scale_audit.R`-`11_freeze_publication_inputs.py` | Diel and phenological analyses, annual carbon-balance feasibility checks, extension of the fixed HMM for budget calculations, and freezing of compact publication inputs |
| `12_make_figure_1.R`-`13_make_figure_2.R` | Generation of main Figures 1 and 2 |
| `14_environment_data_and_model_helpers.R` | Shared environmental-data preparation and statistical-model helpers |
| `15_make_figure_3.R` | Generation of main Figure 3 |
| `16_make_figure_4.R`-`17_audit_figure_4_sensitivity.R` | Generation of main Figure 4 and its sensitivity analysis, including results summarized in Supplementary Table S4 |
| `18_make_supplementary_figures_and_sources.R`-`19_build_supplementary_tables.py` | Generation of supplementary figures, source data, and Supplementary Tables S1-S4 |
| `20_reproduce_publication.R` | Publication-output runner and validation checks |

## Workflow orientation

The active workflow is organized in the following order:

1. Audit the integrated dataset and freeze its analysis contract.
2. Fit candidate HMMs and evaluate temporal resolution, state stability, quality-control sensitivity, and generalizability.
3. Freeze the selected four-state canonical HMM.
4. Analyze diel and phenological state structure and prepare annual greenhouse-gas accounting inputs.
5. Freeze the compact inputs used for publication figures and tables.
6. Generate and validate the main figures, supplementary figures, and supplementary tables.

In the original project directory, the workflow can be inspected or run with:

```bash
python scripts/pipeline/00_reproduce_from_raw.py --dry-run
python scripts/pipeline/00_reproduce_from_raw.py --stage all --hmm-jobs 4
Rscript scripts/pipeline/20_reproduce_publication.R
```

These commands assume the original project layout and the required `config/`, `data/`, `outputs/`, and decision-report files. They will not execute successfully from this pipeline-only repository unless those inputs are restored in the expected locations.

## Reproducibility limitations

The raw-workflow runner requires the integrated US-StJ observational dataset, data-contract records, author decisions, and locked model-selection reports from the original project. The publication runner also requires frozen processed inputs and compact publication-input files produced earlier in the workflow.

The repository does not include an automated downloader or a complete reconstruction procedure for the source AmeriFlux and PhenoCam records. Consequently, reviewers can only inspect the analytical logic and figure-generation code.

A final archival release may pair this code with a separately deposited, DOI-linked set of permitted derived inputs.

## Software dependencies

The analyses were developed with Python 3.12.13 and R 4.6.0.

The Python workflow uses:

`hmmlearn`, `joblib`, `matplotlib`, `numpy`, `pandas`, `scikit-learn`, `scipy`, and `xlsxwriter`.

The R workflow uses:

`data.table`, `ggplot2`, `jsonlite`, `patchwork`, and `scales`, together with the recommended R package `splines`.

Package and system requirements may vary by operating system. Exact package versions should be recorded in the final archival release.

## Data availability

Raw and derived data are not included in this repository. The analysis uses observations from the St. Jones Reserve salt-marsh site (US-StJ) for 2016-2021.

AmeriFlux site information and data access: [US-StJ site page](https://ameriflux.lbl.gov/sites/siteinfo/US-StJ)

PhenoCam imagery and vegetation-color products: [St. Jones PhenoCam page](https://phenocam.nau.edu/webcam/sites/stjones/)

Use of source data remains subject to the terms, acknowledgments, and citation requirements of the respective data providers.

## Manuscript citation

The associated manuscript is currently under review:

> Vargas, R. *Beyond canopy phenology: carbon-flux states reveal functional organization of ecosystem CO2-CH4 exchange*. Manuscript under review.

A final journal citation and permanent repository DOI will be added after acceptance.

## License

This repository does not yet include a software license. Until a license is added, the code remains copyrighted and reuse requires permission from the author. A software license and archival terms will be specified before public release.
