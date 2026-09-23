# Canonical analysis pipeline

These scripts are the only active workflow for reproducing the manuscript. Start with `00_reproduce_from_raw.py` for a complete rebuild or `20_reproduce_publication.R` when the frozen publication inputs already exist. Script 11 validates and refreshes the compact boundary between those workflows.

`14_environment_data_and_model_helpers.R` provides the validated environmental preparation used by the final Figure 3 script. It performs no model fitting or figure export itself.

See `../SCRIPT_MANIFEST.md` for the manuscript-output map and the rationale for
retaining each decision-support script.
