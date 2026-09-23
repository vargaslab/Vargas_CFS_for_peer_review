#!/usr/bin/env python3
"""Freeze the completed clean K=4 decision into downstream canonical artifacts.

This script performs deterministic joins and transcriptions from saved clean
outputs. It does not fit a model, recalculate state count, or use phenology,
GCC, season, legacy state IDs, or environmental variables in state inference.

The decision-record checksum is a scientific lock: if the documented K4
decision changes, this script fails instead of silently producing differently
labelled states. Outputs are the canonical state table, K4 fingerprints, and
the explicitly secondary K6 component summary.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CLEAN_REPORT = ROOT / "reports" / "complete_dataset" / "primary_state_model_decision_partA_clean.md"
LEGACY_REPORT = ROOT / "reports" / "complete_dataset" / "historical_hmm_implementation_comparison.md"
EXPECTED_CLEAN_SHA256 = "76d395d65d14c74807fd8fb4d430f5022b35efe115acbe8a76c515d0a11da081"

CONTRACT = ROOT / "data" / "processed" / "stjones_halfhourly_contract_v1.csv.gz"
K4_DIR = ROOT / "outputs" / "complete_dataset" / "hmm_primary" / "best_by_structure" / "full" / "K4"
K4_FINGERPRINTS = ROOT / "outputs" / "complete_dataset" / "primary_state_decision_clean" / "primary_state_fingerprints.csv"
PRIMARY_SUMMARY = ROOT / "outputs" / "complete_dataset" / "hmm_primary" / "candidate_state_summary.csv"
STANDARDIZATION = ROOT / "outputs" / "complete_dataset" / "hmm_primary" / "standardization_parameters.json"
K6_STABILITY = ROOT / "outputs" / "complete_dataset" / "hmm_state_stability" / "k6_component_stability.csv"
K6_HIERARCHY = ROOT / "outputs" / "complete_dataset" / "hmm_state_stability" / "higher_k_split_characterization.csv"
K6_METADATA = ROOT / "outputs" / "complete_dataset" / "hmm_primary" / "best_by_structure" / "full" / "K6" / "fit_metadata.json"

PRIMARY_OUTPUT = ROOT / "data" / "processed" / "stjones_hmm_k4_primary_v1.csv.gz"
FINAL_OUTPUT_DIR = ROOT / "outputs" / "complete_dataset" / "state_model_final"
FINAL_FINGERPRINTS = FINAL_OUTPUT_DIR / "primary_k4_state_fingerprints_final.csv"
FINAL_K6 = FINAL_OUTPUT_DIR / "k6_higher_resolution_sensitivity.csv"

CANONICAL_IDS = {state: f"C{state}" for state in range(1, 5)}
ORDERING_RULE = (
    "preserve validated clean order K4-S1..K4-S4: ascending NEE emission mean "
    "(most negative to most positive), with CH4 emission mean as secondary tie-breaker"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_prerequisites() -> None:
    actual = sha256(CLEAN_REPORT)
    if actual != EXPECTED_CLEAN_SHA256:
        raise RuntimeError(f"Clean decision checksum mismatch: {actual}")
    clean_text = CLEAN_REPORT.read_text(encoding="utf-8")
    if "CLEAN PRIMARY STATE MODEL: K = 4" not in clean_text:
        raise RuntimeError("Clean primary K=4 decision is absent")
    if "HIGHER-RESOLUTION SENSITIVITY: K = 6" not in clean_text:
        raise RuntimeError("Clean K=6 sensitivity decision is absent")
    legacy_text = LEGACY_REPORT.read_text(encoding="utf-8")
    if "LEGACY REPRODUCIBILITY CLASSIFICATION:\nINDEPENDENT IMPLEMENTATION SUPPORT" not in legacy_text:
        raise RuntimeError("Legacy reproducibility classification prerequisite failed")
    if "EFFECT ON CONFIDENCE IN CLEAN K=4 DECISION:\nINCREASES CONFIDENCE" not in legacy_text:
        raise RuntimeError("Legacy confidence-effect prerequisite failed")


def build_primary_assignments() -> pd.DataFrame:
    contract_columns = [
        "source_row", "timestamp_start", "timestamp_end", "date", "NEE", "CH4",
        "primary_hmm_eligible",
    ]
    contract = pd.read_csv(CONTRACT, compression="gzip", usecols=contract_columns)
    primary = contract.loc[contract["primary_hmm_eligible"]].copy()
    assignments = pd.read_csv(K4_DIR / "viterbi_assignments.csv.gz")
    posterior = pd.read_csv(K4_DIR / "posterior_probabilities.csv.gz")
    if len(primary) != 63_156 or len(assignments) != len(primary) or len(posterior) != len(primary):
        raise RuntimeError("Primary K4 row counts do not reconcile")

    assignment_columns = [
        "source_row", "timestamp_start", "sequence_id", "state_index", "state_label",
        "maximum_posterior",
    ]
    posterior_columns = [
        "source_row", "timestamp_start", "sequence_id",
        "posterior_K4_S1", "posterior_K4_S2", "posterior_K4_S3", "posterior_K4_S4",
    ]
    inference = assignments[assignment_columns].merge(
        posterior[posterior_columns],
        on=["source_row", "timestamp_start", "sequence_id"],
        how="inner",
        validate="one_to_one",
    )
    final = primary.merge(
        inference.drop(columns="timestamp_start"),
        on="source_row",
        how="inner",
        validate="one_to_one",
    )
    if len(final) != len(primary):
        raise RuntimeError("Primary contract and K4 inference join is incomplete")

    expected_labels = final["state_index"].map(lambda value: f"K4-S{int(value)}")
    if not expected_labels.eq(final["state_label"]).all():
        raise RuntimeError("Saved state labels do not match saved state indices")
    final["canonical_state_id"] = final["state_index"].map(CANONICAL_IDS)
    label_map = pd.read_csv(K4_FINGERPRINTS).set_index("state_id")["descriptive_label"]
    final["canonical_state_label"] = final["state_label"].map(label_map)
    final["viterbi_state_original"] = final["state_label"]
    final["viterbi_state_original_index"] = final["state_index"].astype(int)

    probability_columns = [f"posterior_K4_S{state}" for state in range(1, 5)]
    probabilities = final[probability_columns].to_numpy(dtype=float)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=2e-6)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(probabilities > 0, probabilities * np.log(probabilities), 0.0)
    final["posterior_entropy"] = -terms.sum(axis=1)
    saved_max = final["maximum_posterior"].to_numpy(dtype=float)
    np.testing.assert_allclose(saved_max, probabilities.max(axis=1), rtol=0, atol=2e-6)

    final = final.rename(columns={
        "posterior_K4_S1": "posterior_C1",
        "posterior_K4_S2": "posterior_C2",
        "posterior_K4_S3": "posterior_C3",
        "posterior_K4_S4": "posterior_C4",
        "maximum_posterior": "max_posterior",
    })
    output_columns = [
        "source_row", "timestamp_start", "timestamp_end", "date", "NEE", "CH4",
        "primary_hmm_eligible", "sequence_id", "canonical_state_id",
        "canonical_state_label", "viterbi_state_original",
        "viterbi_state_original_index", "posterior_C1", "posterior_C2",
        "posterior_C3", "posterior_C4", "max_posterior", "posterior_entropy",
    ]
    final = final.loc[:, output_columns].sort_values(
        ["timestamp_start", "source_row"], kind="mergesort"
    ).reset_index(drop=True)
    final.to_csv(
        PRIMARY_OUTPUT,
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
        float_format="%.15g",
    )
    return final


def build_final_fingerprints() -> pd.DataFrame:
    source = pd.read_csv(K4_FINGERPRINTS)
    source["state_number"] = source["state_id"].str.extract(r"S(\d+)$").astype(int)
    source = source.sort_values("state_number").reset_index(drop=True)
    if not source["NEE_mean"].is_monotonic_increasing:
        raise RuntimeError("Validated K4 ordering is not ascending by NEE emission mean")
    final = pd.DataFrame({
        "canonical_state_id": source["state_number"].map(CANONICAL_IDS),
        "canonical_state_label": source["descriptive_label"],
        "original_hmm_state": source["state_id"],
        "NEE_mean": source["NEE_mean"],
        "NEE_unit": source["NEE_unit"],
        "CH4_mean": source["CH4_mean"],
        "CH4_unit": source["CH4_unit"],
        "occupancy": source["occupancy"],
        "occupancy_count": source["occupancy_count"].astype(int),
        "posterior_certainty": source["posterior_certainty"],
        "posterior_certainty_q10": source["posterior_certainty_q10"],
        "posterior_certainty_q50": source["posterior_certainty_q50"],
        "posterior_certainty_q90": source["posterior_certainty_q90"],
        "self_transition_probability": source["self_transition_probability"],
        "nearest_centroid_distance_standardized": source["nearest_centroid_distance_standardized"],
        "ordering_rule": ORDERING_RULE,
    })
    final.to_csv(FINAL_FINGERPRINTS, index=False, float_format="%.10f")
    return final


def build_k6_sensitivity() -> pd.DataFrame:
    summary = pd.read_csv(PRIMARY_SUMMARY)
    summary = summary.loc[(summary["K"] == 6) & (summary["covariance_type"] == "full")].copy()
    summary = summary.sort_values("state_index").reset_index(drop=True)
    if len(summary) != 6:
        raise RuntimeError("Representative full-covariance K6 summary is incomplete")
    parameters = json.loads(STANDARDIZATION.read_text(encoding="utf-8"))["parameters"]
    summary["NEE_mean"] = (
        summary["emission_mean_NEE_standardized"] * parameters["NEE"]["sample_standard_deviation"]
        + parameters["NEE"]["mean"]
    )
    summary["CH4_mean"] = (
        summary["emission_mean_CH4_standardized"] * parameters["CH4"]["sample_standard_deviation"]
        + parameters["CH4"]["mean"]
    )
    stability = pd.read_csv(K6_STABILITY).rename(columns={"reference_state": "state_label"})
    hierarchy = pd.read_csv(K6_HIERARCHY)
    hierarchy = hierarchy.loc[hierarchy["mapping"].eq("K4_to_K6")].rename(
        columns={"child_state": "state_label", "parent_state": "validated_broad_k4_parent"}
    )
    metadata = json.loads(K6_METADATA.read_text(encoding="utf-8"))
    final = summary.merge(
        stability[[
            "state_label", "centroid_SD_norm", "membership_Jaccard_mean",
            "membership_Jaccard_min", "classification",
        ]],
        on="state_label",
        how="left",
        validate="one_to_one",
    ).merge(
        hierarchy[[
            "state_label", "validated_broad_k4_parent", "parent_purity",
            "primary_split_axis", "tail_or_extreme_response",
        ]],
        on="state_label",
        how="left",
        validate="one_to_one",
    )
    parent_number = final["validated_broad_k4_parent"].str.extract(r"S(\d+)$")[0].astype(int)
    final["canonical_k4_parent"] = parent_number.map(CANONICAL_IDS)
    final["representative_fit_id"] = metadata["fit_id"]
    final["sensitivity_role"] = "HIGHER-RESOLUTION SENSITIVITY ONLY; not a primary state"
    final["hierarchy_note"] = (
        "Dominant K4 parent in the representative saved fit; mapping is recognizable but not invariant across starts"
    )
    final = final.rename(columns={
        "state_label": "original_k6_component",
        "state_index": "original_state_index",
        "occupancy_proportion": "occupancy",
        "mean_assigned_state_posterior": "posterior_certainty",
        "transition_persistence": "self_transition_probability",
        "classification": "component_stability_classification",
        "parent_purity": "parent_purity_representative_fit",
        "membership_Jaccard_mean": "cross_start_membership_Jaccard_mean",
        "membership_Jaccard_min": "cross_start_membership_Jaccard_min",
        "centroid_SD_norm": "cross_start_centroid_SD_norm",
    })
    final["NEE_unit"] = parameters["NEE"]["unit"]
    final["CH4_unit"] = parameters["CH4"]["unit"]
    columns = [
        "representative_fit_id", "original_k6_component", "original_state_index",
        "NEE_mean", "NEE_unit", "CH4_mean", "CH4_unit", "occupancy",
        "posterior_certainty", "self_transition_probability",
        "component_stability_classification", "validated_broad_k4_parent",
        "canonical_k4_parent", "parent_purity_representative_fit",
        "primary_split_axis", "tail_or_extreme_response",
        "cross_start_membership_Jaccard_mean", "cross_start_membership_Jaccard_min",
        "cross_start_centroid_SD_norm", "sensitivity_role", "hierarchy_note",
    ]
    final = final.loc[:, columns]
    final.to_csv(FINAL_K6, index=False, float_format="%.10f")
    return final


def main() -> None:
    verify_prerequisites()
    FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    primary = build_primary_assignments()
    fingerprints = build_final_fingerprints()
    k6 = build_k6_sensitivity()
    print(f"Primary downstream rows: {len(primary):,}")
    print(f"Canonical primary states: {len(fingerprints)}")
    print(f"K6 sensitivity components: {len(k6)}")
    print(f"Wrote: {PRIMARY_OUTPUT.relative_to(ROOT)}")
    print(f"Wrote: {FINAL_FINGERPRINTS.relative_to(ROOT)}")
    print(f"Wrote: {FINAL_K6.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
