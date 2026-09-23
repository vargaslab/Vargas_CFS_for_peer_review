# Raw-data-to-publication workflow

`pipeline/00_reproduce_from_raw.py` is the controlled-data workflow used to
create the frozen inputs for the canonical manuscript. It starts with the
registered local integrated source file `data/raw/stjones_season.csv` and
rebuilds every calculation required by the active main figures, supplementary
figures, and supplementary tables.

The public repository deliberately does not include that raw file, nor does it
provide an automated download-and-assembly workflow from AmeriFlux and
PhenoCam. Use `20_reproduce_publication.R` for the peer-review reproduction
package; see [DATA_AVAILABILITY.md](../DATA_AVAILABILITY.md) for the
authoritative source networks.

```bash
python scripts/pipeline/00_reproduce_from_raw.py --dry-run
python scripts/pipeline/00_reproduce_from_raw.py --stage all --hmm-jobs 4
```

## Stages

| Stage | What it rebuilds | Scripts |
|---|---|---|
| `contract` | Frozen half-hourly flux, PhenoCam, and environmental data contracts from the raw CSV. | 01 |
| `hmm` | Candidate carbon-only HMMs, temporal/QC/stability/generalizability diagnostics, and frozen canonical K4 assignments. | 02–07 |
| `downstream` | Diel–phenology evidence, fixed-state annual carbon-accounting inputs, and the validated publication-input freeze. | 08–11 |
| `publication` | Main Figures 1–4, Figures S1–S3, and Tables S1–S4. | 12–20 |

The default `--stage all` runs these stages in order. Script 11 copies nine
validated source outputs into `data/publication_inputs/` and writes a checksum
manifest, ensuring that scripts 12–20 cannot reuse stale compact inputs. The
workflow contains only calculations required to reproduce the canonical
publication package or the documented state-model decision.

## The model-selection decision is intentionally locked

The raw CSV and all transformations are automated. The workflow also verifies two versioned scientific-decision records before freezing the primary state table:

- `reports/complete_dataset/primary_state_model_decision_partA_clean.md`
- `reports/complete_dataset/historical_hmm_implementation_comparison.md`

This is deliberate. The four-state representation is a documented inference from candidate fit, stability, and generalizability evidence—not a plotting preference. Script 04 checks the decision record checksum and then freezes its canonical labels. If the raw data, package versions, or model-selection evidence change, the workflow fails rather than silently changing the state definition or manuscript conclusions.

## Expected compute time and safe use

The HMM stage fits multiple deterministic starts across K=2–6, plus temporal, QC, and leave-one-year-out diagnostics. It is the computationally intensive portion of the workflow. Begin with `--dry-run`, choose `--hmm-jobs` appropriate to the machine, and run the complete pipeline in a clean copy of the repository when preparing an archival release.

The pipeline never edits the raw CSV. It overwrites only generated analysis outputs and canonical figure/table deliverables.
