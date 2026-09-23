#!/usr/bin/env python3
"""Reproduce the canonical manuscript from the raw St Jones data.

This is the top-level orchestration entry point. It runs the existing,
independently documented analysis scripts in dependency order:

    raw CSV -> frozen data contract -> carbon-only HMM -> frozen K4 states
    -> phenology/diel and budget analyses -> frozen publication inputs
    -> publication figures and tables.

The primary state count is deliberately *not* selected by this runner. The
versioned K4 scientific-decision lock is verified immediately before state
freezing. This makes the decision auditable and prevents a software update or
accidental change in model-selection rules from silently changing the paper.

Examples
--------
python scripts/pipeline/00_reproduce_from_raw.py --dry-run
python scripts/pipeline/00_reproduce_from_raw.py --stage contract
python scripts/pipeline/00_reproduce_from_raw.py --stage hmm --hmm-jobs 4
python scripts/pipeline/00_reproduce_from_raw.py --stage all --hmm-jobs 4

The full HMM diagnostics are computationally intensive. Use ``--dry-run`` to
inspect exactly what will run before starting a complete rebuild.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "config" / "reproducibility_manifest_v1.json"
RAW_DATA = ROOT / "data" / "raw" / "stjones_season.csv"
DECISION_LOCK = ROOT / "reports" / "complete_dataset" / "primary_state_model_decision_partA_clean.md"
LEGACY_LOCK = ROOT / "reports" / "complete_dataset" / "historical_hmm_implementation_comparison.md"
AUTHOR_DECISIONS = ROOT / "config" / "data_contract_author_decisions.json"


@dataclass(frozen=True)
class Step:
    """One executable analysis step and the stage that owns it."""

    stage: str
    script: str
    command: tuple[str, ...]
    description: str


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest without loading large files in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--stage",
        choices=("all", "contract", "hmm", "downstream", "publication"),
        default="all",
        help="Run all stages, or one named stage (default: all).",
    )
    parser.add_argument("--hmm-jobs", type=int, default=4, help="Parallel workers for the primary HMM grid.")
    parser.add_argument("--dry-run", action="store_true", help="Print the checked inputs and commands without running them.")
    args = parser.parse_args()
    if args.hmm_jobs < 1:
        parser.error("--hmm-jobs must be at least one.")
    return args


def verify_project_inputs() -> None:
    """Fail early when raw data or the versioned analysis decisions are absent."""
    required = (MANIFEST, RAW_DATA, DECISION_LOCK, LEGACY_LOCK, AUTHOR_DECISIONS)
    missing = [path.relative_to(ROOT).as_posix() for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Required raw-data or decision-lock inputs are missing:\n" + "\n".join(missing))

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected_raw_hash = manifest["primary_source"]["sha256"]
    observed_raw_hash = sha256(RAW_DATA)
    if observed_raw_hash != expected_raw_hash:
        raise RuntimeError(
            "Raw-data checksum mismatch. Expected "
            f"{expected_raw_hash}, observed {observed_raw_hash}. "
            "Do not rebuild the manuscript with an unregistered raw-data version."
        )


def step_definitions(hmm_jobs: int) -> list[Step]:
    """Define the canonical DAG in a readable, inspectable order."""
    python = sys.executable
    rscript = shutil.which("Rscript")
    if not rscript:
        raise RuntimeError("Rscript is not available on PATH; install R before running this workflow.")

    steps = [
        Step("contract", "01_audit_and_freeze_data_contract.py", (python, "scripts/pipeline/01_audit_and_freeze_data_contract.py", "--freeze"),
             "Validate raw schema/QC decisions and write the frozen half-hourly and daily contracts."),
        Step("hmm", "02_fit_primary_hmm.py", (python, "scripts/pipeline/02_fit_primary_hmm.py", "--n-jobs", str(hmm_jobs)),
             "Fit deterministic K=2–6 carbon-only Gaussian-HMM candidates to NEE and CH4."),
        Step("hmm", "03_temporal_scale_diagnostic.py", (python, "scripts/pipeline/03_temporal_scale_diagnostic.py", "--jobs", str(hmm_jobs)),
             "Evaluate temporal-resolution sensitivity using frozen-contract aggregates."),
        Step("hmm", "04_hmm_state_stability.py", (python, "scripts/pipeline/04_hmm_state_stability.py",),
             "Assess broad-state and higher-resolution start stability."),
        Step("hmm", "05_hmm_qc_robustness.py", (python, "scripts/pipeline/05_hmm_qc_robustness.py",),
             "Evaluate prespecified QC sensitivity without using ecological covariates in HMM inference."),
        Step("hmm", "06_hmm_generalizability.py", (python, "scripts/pipeline/06_hmm_generalizability.py",),
             "Evaluate leave-one-year-out and deletion generalizability."),
        Step("hmm", "07_freeze_canonical_state_model.py", (python, "scripts/pipeline/07_freeze_canonical_state_model.py",),
             "Verify the K4 scientific-decision lock and create immutable canonical state assignments."),
        Step("downstream", "08_diel_phenology_scale_audit.R", (rscript, "scripts/pipeline/08_diel_phenology_scale_audit.R"),
             "Quantify phenology and GCC information beyond diel timing using frozen K4 posteriors."),
        Step("downstream", "09_annual_carbon_balance_feasibility.R", (rscript, "scripts/pipeline/09_annual_carbon_balance_feasibility.R"),
             "Audit annual carbon-budget variables and align them with observed frozen states."),
        Step("downstream", "10_extend_fixed_hmm_for_budget.R", (rscript, "scripts/pipeline/10_extend_fixed_hmm_for_budget.R"),
             "Apply, but do not refit, fixed K4 parameters to gap-filled budget records."),
        Step("downstream", "11_freeze_publication_inputs.py", (python, "scripts/pipeline/11_freeze_publication_inputs.py"),
             "Validate and freeze the compact inputs consumed by the publication-only workflow."),
        Step("publication", "20_reproduce_publication.R", (rscript, "scripts/pipeline/20_reproduce_publication.R"),
             "Render canonical Figures 1–4, Figures S1–S3, and Tables S1–S4."),
    ]
    return steps


def selected_steps(steps: list[Step], stage: str) -> list[Step]:
    if stage == "all":
        return steps
    return [step for step in steps if step.stage == stage]


def run_step(step: Step) -> None:
    """Run one process from project root and stop at the first failing stage."""
    print(f"\n[raw-to-publication] {step.script}\n  {step.description}", flush=True)
    environment = os.environ.copy()
    # The R publication runner delegates Table S1-S4 creation back to Python.
    # Reuse this interpreter so a complete rebuild cannot accidentally switch
    # to another Python lacking the declared workbook dependency.
    if step.script == "20_reproduce_publication.R":
        environment.setdefault("PYTHON", sys.executable)
    subprocess.run(step.command, cwd=ROOT, check=True, env=environment)


def main() -> None:
    args = parse_args()
    verify_project_inputs()
    steps = selected_steps(step_definitions(args.hmm_jobs), args.stage)
    if not steps:
        raise RuntimeError(f"No steps selected for stage {args.stage!r}.")

    print("Verified raw-data checksum and versioned decision locks.")
    for number, step in enumerate(steps, start=1):
        print(f"{number:>2}. {step.stage:<11} {step.script} — {step.description}")
    if args.dry_run:
        print("\nDry run complete; no files were created or changed.")
        return
    for step in steps:
        run_step(step)
    print("\nRaw-to-publication workflow completed successfully.")


if __name__ == "__main__":
    main()
