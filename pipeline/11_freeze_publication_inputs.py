#!/usr/bin/env python3
"""Freeze the compact inputs consumed by the publication-only workflow.

The controlled-data pipeline writes detailed analysis outputs under
``outputs/complete_dataset``.  The public reproduction package intentionally
ships only the compact CSVs required by scripts 12--20.  This step validates
those source outputs and copies them byte-for-byte into
``data/publication_inputs`` so a raw-data rebuild cannot silently reuse stale
publication inputs.

Use ``--check`` to verify an existing frozen input set without rewriting it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DESTINATION = ROOT / "data" / "publication_inputs"
MANIFEST = DESTINATION / "publication_input_manifest.json"


@dataclass(frozen=True)
class InputSpec:
    name: str
    source: str
    rows: int
    required_columns: tuple[str, ...]


SPECS = (
    InputSpec(
        "model_selection.csv",
        "outputs/complete_dataset/hmm_primary/model_selection.csv",
        10,
        ("K", "covariance_type", "AIC", "BIC", "mean_maximum_posterior"),
    ),
    InputSpec(
        "stability_summary_by_k.csv",
        "outputs/complete_dataset/hmm_state_stability/stability_summary_by_k.csv",
        3,
        ("K", "attempted_starts", "converged_starts", "pairwise_ARI_mean"),
    ),
    InputSpec(
        "leave_one_year_out_summary.csv",
        "outputs/complete_dataset/hmm_generalizability/leave_one_year_out_summary.csv",
        12,
        ("omitted_year", "K", "test_n", "start_stability_ARI", "omitted_year_assignment_ARI"),
    ),
    InputSpec(
        "primary_k4_state_fingerprints_final.csv",
        "outputs/complete_dataset/state_model_final/primary_k4_state_fingerprints_final.csv",
        4,
        ("canonical_state_id", "NEE_mean", "CH4_mean", "occupancy", "self_transition_probability"),
    ),
    InputSpec(
        "daily_state_compositions.csv",
        "outputs/complete_dataset/diel_phenology_audit/daily_state_compositions.csv",
        6114,
        ("date", "year", "phenological_phase", "light_regime", "posterior_C1", "eligible_primary"),
    ),
    InputSpec(
        "halfhourly_relative_gains.csv",
        "outputs/complete_dataset/diel_phenology_audit/halfhourly_relative_gains.csv",
        36,
        ("response_type", "held_out_year", "model_id", "relative_brier_gain_percent"),
    ),
    InputSpec(
        "daily_relative_gains.csv",
        "outputs/complete_dataset/diel_phenology_audit/daily_relative_gains.csv",
        72,
        ("threshold", "light_regime", "held_out_year", "model_id", "relative_brier_gain_percent"),
    ),
    InputSpec(
        "budget_and_state_coverage_by_year_phase.csv",
        "outputs/complete_dataset/manuscript/annual_carbon_balance_decomposition/budget_and_state_coverage_by_year_phase.csv",
        24,
        ("year", "phenological_phase", "canonical_halfhours", "NEE_C_g_m2", "CH4_C_MDS_g_m2"),
    ),
    InputSpec(
        "annual_budget_by_state_phase_all_branches.csv",
        "outputs/complete_dataset/manuscript/annual_carbon_balance_decomposition/annual_budget_by_state_phase_all_branches.csv",
        96,
        ("attribution_branch", "year", "phenological_phase", "canonical_state_id", "NEE_C_g_m2", "CH4_C_g_m2"),
    ),
)


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for a potentially large input."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_csv(path: Path, spec: InputSpec) -> int:
    """Validate the required schema and locked row count for one CSV."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = set(spec.required_columns) - columns
        if missing:
            raise RuntimeError(f"{path}: missing required columns {sorted(missing)}")
        row_count = sum(1 for _ in reader)
    if row_count != spec.rows:
        raise RuntimeError(f"{path}: expected {spec.rows} rows, found {row_count}")
    return row_count


def expected_manifest() -> dict[str, object]:
    """Build deterministic provenance records from detailed source outputs."""
    files = []
    for spec in SPECS:
        source = ROOT / spec.source
        row_count = validate_csv(source, spec)
        files.append(
            {
                "file": spec.name,
                "source": spec.source,
                "rows": row_count,
                "sha256": sha256(source),
            }
        )
    return {
        "manifest_version": "1.0.0",
        "purpose": "Compact frozen inputs consumed by scripts 12-20",
        "generated_by": "scripts/pipeline/11_freeze_publication_inputs.py",
        "files": files,
    }


def freeze(manifest: dict[str, object]) -> None:
    """Atomically replace each compact CSV, then write its provenance manifest."""
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for spec in SPECS:
        source = ROOT / spec.source
        target = DESTINATION / spec.name
        temporary = target.with_name(target.name + ".tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(target)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def check(manifest: dict[str, object]) -> None:
    """Require compact inputs and their manifest to match source outputs exactly."""
    for spec in SPECS:
        source = ROOT / spec.source
        target = DESTINATION / spec.name
        validate_csv(target, spec)
        if sha256(source) != sha256(target):
            raise RuntimeError(f"Frozen publication input is stale: {target.relative_to(ROOT)}")
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)
    observed = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if observed != manifest:
        raise RuntimeError(f"Publication-input manifest is stale: {MANIFEST.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate without rewriting frozen inputs.")
    args = parser.parse_args()
    manifest = expected_manifest()
    if args.check:
        check(manifest)
        print("Publication inputs are complete and byte-identical to their source outputs.")
    else:
        freeze(manifest)
        check(manifest)
        print(f"Frozen {len(SPECS)} publication inputs in {DESTINATION.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
