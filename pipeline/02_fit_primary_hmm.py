#!/usr/bin/env python3
"""Fit the clean primary Gaussian-HMM candidate set from the frozen v1 contract.

This script is intentionally isolated from all legacy HMM artifacts. It reads
only the frozen contracts and canonical half-hourly table, and it uses only NEE
and CH4 as HMM emissions.

The script fits deterministic multi-start candidate grids, saves every fit with
traceable metadata, and exports the model-selection summaries used by Figure S1
and Table S1. Phenology and environmental covariates are prohibited from HMM
inference by construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import warnings
from collections import Counter
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "saltmarsh_hmm_v1_matplotlib")
)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from hmmlearn import __version__ as hmmlearn_version
from hmmlearn.hmm import GaussianHMM
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


ROOT = Path(__file__).resolve().parents[2]
UMBRELLA_PATH = ROOT / "config" / "data_contract_v1.json"
CORE_PATH = ROOT / "config" / "data_contract_core_v1.json"
EXTENSION_PATH = ROOT / "config" / "data_contract_phenology_environment_v1.json"
DATA_PATH = ROOT / "data" / "processed" / "stjones_halfhourly_contract_v1.csv.gz"
OUTPUT_DIR = ROOT / "outputs" / "complete_dataset" / "hmm_primary"
FIGURE_DIR = ROOT / "figures" / "diagnostics" / "hmm_primary"
REPORT_PATH = ROOT / "reports" / "complete_dataset" / "hmm_primary_report.md"

EMISSIONS = ("NEE", "CH4")
CANONICAL_EMISSION_NAMES = ("nee", "ch4")
K_VALUES = tuple(range(2, 7))
COVARIANCE_TYPES = ("full", "diag")
EXPECTED_PRIMARY_N = 63_156
EXPECTED_INTERVAL = timedelta(minutes=30)
SHORT_SEQUENCE_THRESHOLD = 48
BASE_SEED = 731_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-starts", type=int, default=10)
    parser.add_argument("--k6-starts", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--tol", type=float, default=0.01)
    parser.add_argument("--backend", choices=["loky", "threading"], default="loky")
    parser.add_argument(
        "--single-fit", default="",
        help="Run one resumable fit as K,covariance,start_index and exit",
    )
    args = parser.parse_args()
    if args.n_starts < 10:
        raise SystemExit("K=2–5 require at least 10 deterministic starts.")
    if args.k6_starts < 5:
        raise SystemExit("K=6 requires no fewer than 5 deterministic starts.")
    if args.n_jobs < 1:
        raise SystemExit("--n-jobs must be positive")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=json_default) + "\n", encoding="utf-8")


def load_and_validate_contract() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], pd.DataFrame]:
    umbrella = json.loads(UMBRELLA_PATH.read_text(encoding="utf-8"))
    core = json.loads(CORE_PATH.read_text(encoding="utf-8"))
    extension = json.loads(EXTENSION_PATH.read_text(encoding="utf-8"))
    statuses = {
        "umbrella": umbrella.get("contract_status"),
        "core": core.get("contract_status"),
        "extension": extension.get("contract_status"),
    }
    if set(statuses.values()) != {"FROZEN"}:
        raise RuntimeError(f"All applicable contracts must be FROZEN: {statuses}")
    if core.get("hmm_response_variables") != list(CANONICAL_EMISSION_NAMES):
        raise RuntimeError(f"Frozen HMM emissions are not exactly {CANONICAL_EMISSION_NAMES}")
    if core.get("units") != {
        "nee": "micromol CO2 m^-2 s^-1", "ch4": "nmol CH4 m^-2 s^-1"
    }:
        raise RuntimeError(f"Unexpected frozen flux units: {core.get('units')}")
    if "negative NEE" not in core["sign_conventions"]["nee"]:
        raise RuntimeError("Frozen NEE sign convention is not preserved")
    if core["timestamp_convention"]["date_time_role"] != "interval_end":
        raise RuntimeError("DATE_TIME must be the frozen interval-end timestamp")
    if core["timestamp_convention"]["interval_minutes"] != 30:
        raise RuntimeError("Frozen interval duration must be 30 minutes")

    required = {
        "source_row", "timestamp_start", "timestamp_end", "NEE", "CH4",
        "nee_unit", "ch4_unit", "primary_hmm_eligible", "duplicate_group",
        "duplicate_retained",
    }
    data = pd.read_csv(DATA_PATH, compression="gzip", low_memory=False)
    missing = sorted(required - set(data.columns))
    if missing:
        raise RuntimeError(f"Canonical half-hourly table is missing: {missing}")
    if int(data["primary_hmm_eligible"].sum()) != EXPECTED_PRIMARY_N:
        raise RuntimeError("Frozen primary_hmm_eligible count is not 63,156")
    if set(data["nee_unit"].dropna()) != {core["units"]["nee"]}:
        raise RuntimeError("Canonical NEE unit differs from the frozen contract")
    if set(data["ch4_unit"].dropna()) != {core["units"]["ch4"]}:
        raise RuntimeError("Canonical CH4 unit differs from the frozen contract")
    data["timestamp_start"] = pd.to_datetime(data["timestamp_start"], errors="raise")
    data["timestamp_end"] = pd.to_datetime(data["timestamp_end"], errors="raise")
    if not ((data["timestamp_end"] - data["timestamp_start"]) == EXPECTED_INTERVAL).all():
        raise RuntimeError("Canonical timestamp intervals are not uniformly 30 minutes")
    if data["timestamp_start"].duplicated().any() or data["timestamp_end"].duplicated().any():
        raise RuntimeError("Canonical timestamps are not unique after duplicate handling")
    if int(data["duplicate_retained"].sum()) != core["duplicate_handling"]["groups"]:
        raise RuntimeError("Canonical duplicate-resolution flags disagree with the frozen contract")
    eligible = data.loc[data["primary_hmm_eligible"]].copy()
    if not np.isfinite(eligible.loc[:, EMISSIONS].to_numpy(dtype=float)).all():
        raise RuntimeError("Primary emissions contain nonfinite values")
    return umbrella, core, extension, data


def prepare_primary_data(data: pd.DataFrame, core: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray, list[int], pd.DataFrame, dict[str, Any]]:
    primary = data.loc[data["primary_hmm_eligible"], ["source_row", "timestamp_start", "timestamp_end", *EMISSIONS]].copy()
    primary = primary.sort_values(["timestamp_start", "source_row"], kind="mergesort").reset_index(drop=True)
    breaks = primary["timestamp_start"].diff().ne(EXPECTED_INTERVAL)
    breaks.iloc[0] = True
    primary["sequence_id"] = breaks.cumsum().astype(int)
    sequence_summary = primary.groupby("sequence_id", sort=True).agg(
        sequence_start=("timestamp_start", "min"),
        sequence_end=("timestamp_end", "max"),
        sequence_length=("source_row", "size"),
        first_source_row=("source_row", "min"),
        last_source_row=("source_row", "max"),
    ).reset_index()
    sequence_summary["singleton"] = sequence_summary["sequence_length"].eq(1)
    sequence_summary["short_lt_48_half_hours"] = sequence_summary["sequence_length"].lt(SHORT_SEQUENCE_THRESHOLD)
    lengths = sequence_summary["sequence_length"].astype(int).tolist()
    if sum(lengths) != EXPECTED_PRIMARY_N:
        raise RuntimeError("Sequence lengths do not reconcile to the frozen primary sample")
    if any(length <= 0 for length in lengths):
        raise RuntimeError("Sequence construction produced an empty sequence")

    raw_x = primary.loc[:, EMISSIONS].to_numpy(dtype=float)
    means = raw_x.mean(axis=0)
    sds = raw_x.std(axis=0, ddof=1)
    if np.any(~np.isfinite(sds)) or np.any(sds <= 0):
        raise RuntimeError("Invalid primary standardization parameters")
    x = (raw_x - means) / sds
    parameters = {
        "fit_input_columns": list(EMISSIONS),
        "canonical_emission_names": list(CANONICAL_EMISSION_NAMES),
        "n_observations": len(primary),
        "calculation_population": "primary_hmm_eligible == TRUE only",
        "standard_deviation_definition": "sample standard deviation (ddof=1)",
        "parameters": {
            "NEE": {"mean": means[0], "sample_standard_deviation": sds[0], "unit": core["units"]["nee"]},
            "CH4": {"mean": means[1], "sample_standard_deviation": sds[1], "unit": core["units"]["ch4"]},
        },
        "nee_sign_preserved": True,
        "co2_uptake_transform_applied": False,
    }
    primary["NEE_standardized"] = x[:, 0]
    primary["CH4_standardized"] = x[:, 1]
    return primary, x, lengths, sequence_summary, parameters


def parameter_count(k: int, d: int, covariance_type: str) -> int:
    start = k - 1
    transitions = k * (k - 1)
    means = k * d
    covariance = k * d if covariance_type == "diag" else k * d * (d + 1) // 2
    return start + transitions + means + covariance


def canonical_order(means: np.ndarray) -> np.ndarray:
    return np.lexsort((means[:, 1], means[:, 0]))


def covariance_payload(model: GaussianHMM) -> np.ndarray:
    covars = np.asarray(model.covars_, dtype=float)
    return covars


def covariance_geometry(covariance: np.ndarray) -> dict[str, float]:
    eigenvalues = np.linalg.eigvalsh(covariance)
    minimum = max(float(eigenvalues.min()), np.finfo(float).tiny)
    denominator = math.sqrt(max(covariance[0, 0] * covariance[1, 1], np.finfo(float).tiny))
    return {
        "eigenvalue_min": float(eigenvalues.min()),
        "eigenvalue_max": float(eigenvalues.max()),
        "eigenvalue_ratio": float(eigenvalues.max() / minimum),
        "correlation": float(covariance[0, 1] / denominator),
    }


def fit_one(k: int, covariance_type: str, start_index: int, seed: int, x: np.ndarray, lengths: list[int], max_iter: int, tol: float) -> dict[str, Any]:
    fit_id = f"K{k}_{covariance_type}_start{start_index:02d}_seed{seed}"
    fit_dir = OUTPUT_DIR / "fits" / f"K{k}_{covariance_type}" / f"start{start_index:02d}_seed{seed}"
    fit_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "fit_id": fit_id, "K": k, "covariance_type": covariance_type,
        "start_index": start_index, "seed": seed, "success": False,
        "converged": False, "iterations": 0, "log_likelihood": np.nan,
        "parameter_count": parameter_count(k, x.shape[1], covariance_type),
        "AIC": np.nan, "BIC": np.nan, "error": "", "warnings": "",
        "artifact_directory": str(fit_dir.relative_to(ROOT)),
    }
    try:
        model = GaussianHMM(
            n_components=k,
            covariance_type=covariance_type,
            n_iter=max_iter,
            tol=tol,
            min_covar=1e-6,
            random_state=seed,
            init_params="stmc",
            params="stmc",
            implementation="scaling",
            verbose=False,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("once")
            model.fit(x, lengths)
            log_likelihood = float(model.score(x, lengths))
            posterior = model.predict_proba(x, lengths)
            viterbi = model.predict(x, lengths)
        monitor_history = [float(value) for value in model.monitor_.history]
        final_delta = monitor_history[-1] - monitor_history[-2] if len(monitor_history) >= 2 else np.nan
        iterations = int(model.monitor_.iter)
        hmmlearn_converged = bool(model.monitor_.converged)
        monotonic_final_step = not math.isfinite(final_delta) or final_delta >= -1e-3
        reached_iteration_limit = iterations >= max_iter
        # hmmlearn's monitor reports ``converged`` when the iteration limit is
        # reached, even if the likelihood increment still exceeds tolerance.
        # Treat that termination as nonconverged for candidate selection.
        converged = hmmlearn_converged and monotonic_final_step and not reached_iteration_limit
        order = canonical_order(np.asarray(model.means_))
        raw_to_canonical = np.empty(k, dtype=int)
        raw_to_canonical[order] = np.arange(k)
        posterior_canonical = posterior[:, order]
        viterbi_canonical = raw_to_canonical[viterbi]
        max_posterior = posterior_canonical.max(axis=1)
        occupancies = np.bincount(viterbi_canonical, minlength=k)
        n_parameters = result["parameter_count"]
        aic = 2 * n_parameters - 2 * log_likelihood
        bic = math.log(len(x)) * n_parameters - 2 * log_likelihood
        covariance = covariance_payload(model)[order]
        means = np.asarray(model.means_)[order]
        transition = np.asarray(model.transmat_)[np.ix_(order, order)]
        startprob = np.asarray(model.startprob_)[order]

        joblib.dump(model, fit_dir / "model.joblib", compress=3)
        np.savez_compressed(
            fit_dir / "inference.npz",
            posterior_probabilities=posterior_canonical.astype(np.float32),
            viterbi_state_index=viterbi_canonical.astype(np.int16),
            maximum_posterior=max_posterior.astype(np.float32),
        )
        warning_counts = Counter(
            (item.category.__name__, str(item.message)) for item in caught
        )
        warning_summary = [
            {"category": category, "message": message, "count": count}
            for (category, message), count in warning_counts.most_common(20)
        ]
        metadata = {
            **result,
            "success": True, "converged": converged,
            "hmmlearn_monitor_converged": hmmlearn_converged,
            "final_log_likelihood_delta": final_delta,
            "monotonic_final_step": monotonic_final_step,
            "reached_iteration_limit": reached_iteration_limit,
            "termination_reason": "tolerance_reached" if converged else (
                "maximum_iterations_reached" if reached_iteration_limit else "nonmonotonic_or_failed_convergence"
            ),
            "iterations": iterations,
            "log_likelihood": log_likelihood, "AIC": aic, "BIC": bic,
            "state_labels": [f"K{k}-S{index}" for index in range(1, k + 1)],
            "raw_model_state_to_canonical_index": raw_to_canonical.tolist(),
            "canonical_order_raw_state_indices": order.tolist(),
            "state_occupancy_counts": occupancies.tolist(),
            "state_occupancy_proportions": (occupancies / len(x)).tolist(),
            "transition_matrix": transition.tolist(),
            "start_probabilities": startprob.tolist(),
            "emission_means_standardized": means.tolist(),
            "emission_covariances_standardized": covariance.tolist(),
            "mean_maximum_posterior": float(max_posterior.mean()),
            "maximum_posterior_quantiles": {
                str(q): float(np.quantile(max_posterior, q))
                for q in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
            },
            "posterior_row_sum_max_abs_error": float(np.max(np.abs(posterior_canonical.sum(axis=1) - 1))),
            "warning_count": len(caught),
            "unique_warning_count": len(warning_counts),
            "warning_summary_top_20": warning_summary,
            "warning_summary_truncated": len(warning_counts) > 20,
            "monitor_history": monitor_history,
        }
        write_json(fit_dir / "fit_metadata.json", metadata)
        result.update(metadata)
        result["viterbi_array"] = viterbi_canonical
        result["means_array"] = means
        result["covariance_array"] = covariance
        result["transition_array"] = transition
        result["posterior_path"] = str((fit_dir / "inference.npz").relative_to(ROOT))
        result["model_path"] = str((fit_dir / "model.joblib").relative_to(ROOT))
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        write_json(fit_dir / "fit_metadata.json", result)
    return result


def load_saved_fit(
    k: int, covariance_type: str, start_index: int, seed: int, max_iter: int
) -> dict[str, Any] | None:
    fit_dir = OUTPUT_DIR / "fits" / f"K{k}_{covariance_type}" / f"start{start_index:02d}_seed{seed}"
    metadata_path = fit_dir / "fit_metadata.json"
    if not metadata_path.is_file():
        return None
    result = json.loads(metadata_path.read_text(encoding="utf-8"))
    if result.get("seed") != seed or result.get("K") != k or result.get("covariance_type") != covariance_type:
        raise RuntimeError(f"Saved fit metadata does not match requested deterministic fit: {fit_dir}")
    if result.get("success"):
        reached_iteration_limit = int(result["iterations"]) >= max_iter
        strict_converged = bool(
            result.get("hmmlearn_monitor_converged", result.get("converged", False))
            and result.get("monotonic_final_step", True)
            and not reached_iteration_limit
        )
        termination_reason = "tolerance_reached" if strict_converged else (
            "maximum_iterations_reached" if reached_iteration_limit else "nonmonotonic_or_failed_convergence"
        )
        if (
            result.get("converged") != strict_converged
            or result.get("reached_iteration_limit") != reached_iteration_limit
            or result.get("termination_reason") != termination_reason
        ):
            result["converged"] = strict_converged
            result["reached_iteration_limit"] = reached_iteration_limit
            result["termination_reason"] = termination_reason
            write_json(metadata_path, result)
    if result.get("success"):
        inference_path = fit_dir / "inference.npz"
        model_path = fit_dir / "model.joblib"
        if not inference_path.is_file() or not model_path.is_file():
            return None
        inference = np.load(inference_path)
        result["viterbi_array"] = inference["viterbi_state_index"].astype(int)
        result["means_array"] = np.asarray(result["emission_means_standardized"], dtype=float)
        result["covariance_array"] = np.asarray(result["emission_covariances_standardized"], dtype=float)
        result["transition_array"] = np.asarray(result["transition_matrix"], dtype=float)
        result["posterior_path"] = str(inference_path.relative_to(ROOT))
        result["model_path"] = str(model_path.relative_to(ROOT))
    return result


def alignment_and_stability(group: list[dict[str, Any]], winner: dict[str, Any]) -> dict[str, Any]:
    successful = [fit for fit in group if fit["success"] and fit["converged"]]
    if not successful:
        return {
            "successful_converged_starts": 0, "pairwise_ari_mean": np.nan,
            "pairwise_ari_min": np.nan, "pairwise_nmi_mean": np.nan,
            "winner_ari_mean": np.nan, "winner_recovery_rate_ari_ge_0_95": np.nan,
            "centroid_sd_rms": np.nan, "centroid_max_sd": np.nan,
        }
    reference_means = winner["means_array"]
    aligned_centroids = []
    aligned_assignments = []
    for fit in successful:
        row, col = linear_sum_assignment(
            np.linalg.norm(fit["means_array"][:, None, :] - reference_means[None, :, :], axis=2)
        )
        mapping = np.empty(len(row), dtype=int)
        mapping[row] = col
        aligned_centroids.append(fit["means_array"][np.argsort(mapping)])
        aligned_assignments.append(mapping[fit["viterbi_array"]])
    aris, nmis = [], []
    for left, right in combinations(aligned_assignments, 2):
        aris.append(adjusted_rand_score(left, right))
        nmis.append(normalized_mutual_info_score(left, right))
    winner_index = next(
        index for index, fit in enumerate(successful)
        if fit["fit_id"] == winner["fit_id"]
    )
    winner_assignment = aligned_assignments[winner_index]
    winner_aris = [adjusted_rand_score(winner_assignment, item) for item in aligned_assignments]
    centroid_stack = np.stack(aligned_centroids)
    centroid_sd = centroid_stack.std(axis=0, ddof=1) if len(successful) > 1 else np.zeros_like(reference_means)
    return {
        "successful_converged_starts": len(successful),
        "pairwise_ari_mean": float(np.mean(aris)) if aris else 1.0,
        "pairwise_ari_min": float(np.min(aris)) if aris else 1.0,
        "pairwise_nmi_mean": float(np.mean(nmis)) if nmis else 1.0,
        "winner_ari_mean": float(np.mean(winner_aris)),
        "winner_recovery_rate_ari_ge_0_95": float(np.mean(np.asarray(winner_aris) >= 0.95)),
        "centroid_sd_rms": float(np.sqrt(np.mean(centroid_sd ** 2))),
        "centroid_max_sd": float(np.max(centroid_sd)),
    }


def best_fit_diagnostics(fit: dict[str, Any], posterior: np.ndarray, viterbi: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    k = fit["K"]
    means = fit["means_array"]
    covariances = fit["covariance_array"]
    transition = fit["transition_array"]
    max_posterior = posterior.max(axis=1)
    separations = []
    pair_rows = []
    for left, right in combinations(range(k), 2):
        distance = float(np.linalg.norm(means[left] - means[right]))
        separations.append(distance)
        pair_rows.append({
            "K": k, "covariance_type": fit["covariance_type"],
            "state_a": f"K{k}-S{left + 1}", "state_b": f"K{k}-S{right + 1}",
            "euclidean_centroid_separation": distance,
        })
    occupancies = np.bincount(viterbi, minlength=k)
    state_rows = []
    for state in range(k):
        assigned = viterbi == state
        state_posterior = posterior[assigned, state]
        geometry = covariance_geometry(covariances[state])
        occupancy = int(assigned.sum())
        proportion = occupancy / len(viterbi)
        min_separation = min(
            float(np.linalg.norm(means[state] - means[other]))
            for other in range(k) if other != state
        )
        centroid_norm = float(np.linalg.norm(means[state]))
        posterior_mean = float(state_posterior.mean()) if occupancy else 0.0
        posterior_q10 = float(np.quantile(state_posterior, 0.10)) if occupancy else 0.0
        posterior_q50 = float(np.quantile(state_posterior, 0.50)) if occupancy else 0.0
        posterior_q90 = float(np.quantile(state_posterior, 0.90)) if occupancy else 0.0
        state_rows.append({
            "K": k, "covariance_type": fit["covariance_type"],
            "state_index": state + 1, "state_label": f"K{k}-S{state + 1}",
            "occupancy_count": occupancy, "occupancy_proportion": proportion,
            "mean_assigned_state_posterior": posterior_mean,
            "assigned_state_posterior_q10": posterior_q10,
            "assigned_state_posterior_q50": posterior_q50,
            "assigned_state_posterior_q90": posterior_q90,
            "emission_mean_NEE_standardized": float(means[state, 0]),
            "emission_mean_CH4_standardized": float(means[state, 1]),
            "centroid_norm_standardized": centroid_norm,
            "nearest_centroid_distance": min_separation,
            "transition_persistence": float(transition[state, state]),
            **geometry,
            "very_small_state_flag": proportion < 0.01,
            "poorly_separated_state_flag": min_separation < 0.75,
            "low_certainty_state_flag": posterior_mean < 0.80,
            "extreme_response_tail_flag": proportion < 0.05 and centroid_norm >= 2.0,
        })
    aggregate = {
        "minimum_state_occupancy": float(np.min(occupancies / len(viterbi))),
        "mean_maximum_posterior": float(max_posterior.mean()),
        "fraction_max_posterior_lt_0_60": float(np.mean(max_posterior < 0.60)),
        "fraction_max_posterior_lt_0_70": float(np.mean(max_posterior < 0.70)),
        "fraction_max_posterior_lt_0_80": float(np.mean(max_posterior < 0.80)),
        "fraction_max_posterior_lt_0_90": float(np.mean(max_posterior < 0.90)),
        "minimum_centroid_separation": float(min(separations)),
        "mean_transition_persistence": float(np.diag(transition).mean()),
        "very_small_state_count": int(sum(row["very_small_state_flag"] for row in state_rows)),
        "poorly_separated_state_count": int(sum(row["poorly_separated_state_flag"] for row in state_rows)),
        "low_certainty_state_count": int(sum(row["low_certainty_state_flag"] for row in state_rows)),
        "extreme_tail_state_count": int(sum(row["extreme_response_tail_flag"] for row in state_rows)),
    }
    return state_rows, aggregate, pair_rows


def save_best_structure_outputs(fit: dict[str, Any], primary: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    source_dir = ROOT / fit["artifact_directory"]
    inference = np.load(source_dir / "inference.npz")
    posterior = inference["posterior_probabilities"].astype(float)
    viterbi = inference["viterbi_state_index"].astype(int)
    target_dir = OUTPUT_DIR / "best_by_structure" / fit["covariance_type"] / f"K{fit['K']}"
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_dir / "model.joblib", target_dir / "model.joblib")
    shutil.copy2(source_dir / "fit_metadata.json", target_dir / "fit_metadata.json")
    shutil.copy2(source_dir / "inference.npz", target_dir / "inference.npz")
    assignments = primary[["source_row", "timestamp_start", "timestamp_end", "sequence_id"]].copy()
    assignments["state_index"] = viterbi + 1
    assignments["state_label"] = [f"K{fit['K']}-S{value + 1}" for value in viterbi]
    assignments["maximum_posterior"] = posterior.max(axis=1)
    assignments.to_csv(target_dir / "viterbi_assignments.csv.gz", index=False, compression="gzip")
    posterior_frame = primary[["source_row", "timestamp_start", "sequence_id"]].copy()
    for state in range(fit["K"]):
        posterior_frame[f"posterior_K{fit['K']}_S{state + 1}"] = posterior[:, state]
    posterior_frame.to_csv(target_dir / "posterior_probabilities.csv.gz", index=False, compression="gzip")
    return posterior, viterbi


def diagnostic_figures(model_selection: pd.DataFrame, state_summary: pd.DataFrame, best_covariance: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    colors = {"full": "#1f77b4", "diag": "#d95f02"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for covariance in COVARIANCE_TYPES:
        data = model_selection.loc[model_selection["covariance_type"].eq(covariance)].sort_values("K")
        axes[0].plot(data["K"], data["AIC"], marker="o", label=covariance, color=colors[covariance])
        axes[1].plot(data["K"], data["BIC"], marker="o", label=covariance, color=colors[covariance])
    for axis, metric in zip(axes, ["AIC", "BIC"]):
        axis.set(xlabel="Number of states (K)", ylabel=metric, xticks=K_VALUES)
        axis.grid(alpha=0.25)
    axes[0].legend(title="Covariance")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_01_aic_bic_vs_k.png", dpi=180)
    plt.close(fig)

    for filename, ylabel, column in [
        ("figure_02_occupancy_vs_k.png", "Minimum state occupancy", "minimum_state_occupancy"),
        ("figure_03_posterior_certainty_vs_k.png", "Mean maximum posterior", "mean_maximum_posterior"),
        ("figure_04_minimum_centroid_separation_vs_k.png", "Minimum centroid separation", "minimum_centroid_separation"),
    ]:
        fig, axis = plt.subplots(figsize=(6.5, 4.2))
        for covariance in COVARIANCE_TYPES:
            data = model_selection.loc[model_selection["covariance_type"].eq(covariance)].sort_values("K")
            axis.plot(data["K"], data[column], marker="o", label=covariance, color=colors[covariance])
        axis.set(xlabel="Number of states (K)", ylabel=ylabel, xticks=K_VALUES)
        axis.grid(alpha=0.25)
        axis.legend(title="Covariance")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / filename, dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for covariance in COVARIANCE_TYPES:
        data = model_selection.loc[model_selection["covariance_type"].eq(covariance)].sort_values("K")
        axes[0].plot(data["K"], data["converged_fraction"], marker="o", label=covariance, color=colors[covariance])
        axes[1].plot(data["K"], data["pairwise_ari_mean"], marker="o", label=covariance, color=colors[covariance])
    axes[0].set(xlabel="Number of states (K)", ylabel="Converged-start fraction", xticks=K_VALUES, ylim=(-0.02, 1.02))
    axes[1].set(xlabel="Number of states (K)", ylabel="Mean pairwise ARI", xticks=K_VALUES, ylim=(-0.02, 1.02))
    axes[0].legend(title="Covariance")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_05_convergence_stability_across_starts.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), sharex=True, sharey=True)
    axes = axes.ravel()
    for index, k in enumerate(K_VALUES):
        subset = state_summary.loc[
            state_summary["K"].eq(k) & state_summary["covariance_type"].eq(best_covariance)
        ]
        axes[index].scatter(
            subset["emission_mean_NEE_standardized"], subset["emission_mean_CH4_standardized"],
            s=30 + 1000 * subset["occupancy_proportion"], c=np.arange(len(subset)), cmap="viridis", edgecolor="black",
        )
        for row in subset.itertuples(index=False):
            axes[index].annotate(row.state_label, (row.emission_mean_NEE_standardized, row.emission_mean_CH4_standardized), xytext=(4, 4), textcoords="offset points", fontsize=8)
        axes[index].axhline(0, color="grey", linewidth=0.6)
        axes[index].axvline(0, color="grey", linewidth=0.6)
        axes[index].set_title(f"K={k}")
        axes[index].grid(alpha=0.2)
    axes[-1].axis("off")
    fig.supxlabel("Standardized NEE centroid")
    fig.supylabel("Standardized CH4 centroid")
    fig.suptitle(f"Neutral emission centroids ({best_covariance} covariance)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_06_nee_ch4_centroids_by_k.png", dpi=180)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy().replace({np.nan: ""})
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join("---" for _ in display.columns) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def write_report(model_selection: pd.DataFrame, state_summary: pd.DataFrame, sequence_summary: pd.DataFrame, standardization: dict[str, Any], best_covariance: str, run_metadata: dict[str, Any]) -> None:
    bic_best = model_selection.loc[model_selection["BIC"].idxmin()]
    aic_best = model_selection.loc[model_selection["AIC"].idxmin()]
    best_cov = model_selection.loc[model_selection["covariance_type"].eq(best_covariance)].sort_values("K")
    flagged = state_summary.loc[state_summary["covariance_type"].eq(best_covariance)].groupby("K").agg(
        small_states=("very_small_state_flag", "sum"),
        poorly_separated_states=("poorly_separated_state_flag", "sum"),
        extreme_tail_states=("extreme_response_tail_flag", "sum"),
    ).reset_index()
    subdivision_ks = flagged.loc[
        flagged[["small_states", "poorly_separated_states", "extreme_tail_states"]].sum(axis=1).gt(0), "K"
    ].tolist()
    if subdivision_ks:
        additional_state_text = (
            "At least some higher-K solutions contain small, closely spaced, or extreme-tail states "
            f"(flagged K values: {subdivision_ks}); the extra states therefore include subdivisions/tails rather than uniformly broad new signatures."
        )
    else:
        additional_state_text = "No candidate state crossed the prespecified small-state, separation, or extreme-tail flags; added states appear broad by these diagnostics."
    stable = bool(bic_best["winner_recovery_rate_ari_ge_0_95"] >= 0.70 and bic_best["pairwise_ari_mean"] >= 0.90)
    parsimonious = (
        "There is statistical support for a parsimonious lower-dimensional representation because the BIC optimum is at "
        f"K={int(bic_best['K'])}. This is evidence about compression, not a final ecological state-count decision."
        if int(bic_best["K"]) <= 3 else
        "The BIC optimum is not at the lowest-dimensional end of the candidate set; a parsimonious lower-K representation remains possible but is not preferred by BIC alone."
    )
    sequence_lengths = sequence_summary["sequence_length"]
    short_observations = int(sequence_summary.loc[sequence_summary["short_lt_48_half_hours"], "sequence_length"].sum())
    table = model_selection[[
        "K", "covariance_type", "log_likelihood", "AIC", "BIC", "converged_fraction",
        "pairwise_ari_mean", "winner_recovery_rate_ari_ge_0_95", "minimum_state_occupancy",
        "mean_maximum_posterior", "minimum_centroid_separation",
    ]].copy()
    for column in table.select_dtypes(include="number").columns:
        if column != "K":
            table[column] = table[column].map(lambda value: f"{value:.4f}")
    report = f"""# Clean primary HMM reanalysis v1

## Scope and gate

This analysis used only the frozen v1 contracts and `data/processed/stjones_halfhourly_contract_v1.csv.gz`. It did not read or use legacy HMM assignments, fitted models, posterior probabilities, state labels, model-selection results, or downstream outputs. No phenology or environmental variable entered fitting. The emissions were only canonical NEE and CH4, with the original NEE sign preserved.

## Primary data and sequences

- Primary observations: **{standardization['n_observations']:,}**.
- Independent sequences: **{len(sequence_summary):,}**.
- Singleton sequences: **{int(sequence_summary['singleton'].sum()):,}**.
- Median sequence length: **{sequence_lengths.median():.0f} half-hours**.
- Maximum sequence length: **{int(sequence_lengths.max()):,} half-hours**.
- Observations in sequences shorter than {SHORT_SEQUENCE_THRESHOLD} half-hours: **{short_observations:,} ({short_observations / standardization['n_observations']:.1%})**.
- Short sequences were retained.

Standardization used only primary-eligible rows: NEE mean `{standardization['parameters']['NEE']['mean']:.8g}`, SD `{standardization['parameters']['NEE']['sample_standard_deviation']:.8g}`; CH4 mean `{standardization['parameters']['CH4']['mean']:.8g}`, SD `{standardization['parameters']['CH4']['sample_standard_deviation']:.8g}`.

## Candidate comparison

{markdown_table(table)}

## Direct answers

- **BIC minimum:** K={int(bic_best['K'])}, `{bic_best['covariance_type']}` covariance (BIC `{bic_best['BIC']:.2f}`).
- **AIC minimum:** K={int(aic_best['K'])}, `{aic_best['covariance_type']}` covariance (AIC `{aic_best['AIC']:.2f}`).
- **Stability of the BIC optimum:** {'stable by the prespecified recovery criteria' if stable else 'not consistently stable by the prespecified recovery criteria'}; mean pairwise ARI `{bic_best['pairwise_ari_mean']:.3f}`, winner recovery rate `{bic_best['winner_recovery_rate_ari_ge_0_95']:.1%}`.
- **Additional states:** {additional_state_text}
- **Parsimonious representation:** {parsimonious}

The covariance structure with the lowest global BIC was `{best_covariance}`. This determines which covariance family receives the complete K=2–6 best-model export; it does not choose the manuscript state count. Best alternatives from the other covariance family are also preserved.

## Classification and state-quality cautions

State-quality flags are diagnostic thresholds, not ecological labels: occupancy <1%, nearest-centroid distance <0.75 standardized units, mean assigned-state posterior <0.80, and an extreme-tail flag requiring occupancy <5% plus centroid norm ≥2.0. States retain neutral labels such as `K3-S1`.

Before a manuscript state count is chosen, test sensitivity to response-tail influence, covariance specification, fragmented/short sequences, primary versus QC sensitivity datasets, posterior uncertainty, and whether apparently split centroids reproduce under resampling or time-block validation. Phenology and environmental associations should be evaluated only after a state count is chosen independently of those variables.

## Reproducibility

- Deterministic starts: 10 per K/covariance structure for K=2–5 and 5 per structure for K=6. K=6 was reduced to the allowed minimum after exact-sequence timing demonstrated computationally prohibitive wall time; no sequence or observation was removed.
- HMM implementation: hmmlearn GaussianHMM {run_metadata['package_versions']['hmmlearn']}.
- Fit command: `{run_metadata['command']}`.
- All per-start seeds, convergence, diagnostics, models, posterior matrices, and Viterbi assignments are under `outputs/complete_dataset/hmm_primary/`.
- Diagnostic figures are under `figures/diagnostics/hmm_primary/`.
- No final manuscript figure was created.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    umbrella, core, extension, data = load_and_validate_contract()
    primary, x, lengths, sequence_summary, standardization = prepare_primary_data(data, core)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    primary[["source_row", "timestamp_start", "timestamp_end", "sequence_id"]].to_csv(
        OUTPUT_DIR / "primary_observation_index.csv.gz", index=False, compression="gzip"
    )
    sequence_summary.to_csv(OUTPUT_DIR / "sequence_summary.csv", index=False)
    write_json(OUTPUT_DIR / "standardization_parameters.json", standardization)

    if args.single_fit:
        try:
            k_text, covariance_type, start_text = args.single_fit.split(",")
            k, start_index = int(k_text), int(start_text)
        except ValueError as exc:
            raise SystemExit("--single-fit must be K,covariance,start_index") from exc
        if k not in K_VALUES or covariance_type not in COVARIANCE_TYPES:
            raise SystemExit("Invalid --single-fit candidate")
        covariance_index = COVARIANCE_TYPES.index(covariance_type)
        seed = BASE_SEED + k * 1_000 + covariance_index * 100 + start_index
        result = fit_one(k, covariance_type, start_index, seed, x, lengths, args.max_iter, args.tol)
        print(json.dumps({
            "fit_id": result["fit_id"], "success": result["success"],
            "converged": result["converged"], "iterations": result["iterations"],
            "log_likelihood": result["log_likelihood"], "error": result["error"],
        }, default=json_default))
        return

    tasks = []
    results = []
    for k in K_VALUES:
        for covariance_index, covariance_type in enumerate(COVARIANCE_TYPES):
            target_starts = args.k6_starts if k == 6 else args.n_starts
            for start_index in range(1, target_starts + 1):
                seed = BASE_SEED + k * 1_000 + covariance_index * 100 + start_index
                saved = load_saved_fit(k, covariance_type, start_index, seed, args.max_iter)
                if saved is None:
                    tasks.append((k, covariance_type, start_index, seed))
                else:
                    results.append(saved)
    if tasks:
        new_results = Parallel(n_jobs=min(args.n_jobs, len(tasks)), backend=args.backend, verbose=10)(
            delayed(fit_one)(k, covariance_type, start_index, seed, x, lengths, args.max_iter, args.tol)
            for k, covariance_type, start_index, seed in tasks
        )
        results.extend(new_results)

    flat_diagnostics = []
    for result in results:
        flat_diagnostics.append({key: value for key, value in result.items() if not key.endswith("_array")})
    diagnostics = pd.DataFrame(flat_diagnostics)
    structures = []
    state_rows_all = []
    pair_rows_all = []
    best_fits = []
    for k in K_VALUES:
        for covariance_type in COVARIANCE_TYPES:
            group = [result for result in results if result["K"] == k and result["covariance_type"] == covariance_type]
            successful = [result for result in group if result["success"] and result["converged"]]
            if not successful:
                raise RuntimeError(f"No converged fits for K={k}, covariance={covariance_type}")
            winner = max(successful, key=lambda item: item["log_likelihood"])
            best_fits.append(winner)
            stability = alignment_and_stability(group, winner)
            posterior, viterbi = save_best_structure_outputs(winner, primary)
            state_rows, aggregate, pair_rows = best_fit_diagnostics(winner, posterior, viterbi)
            state_rows_all.extend(state_rows)
            pair_rows_all.extend(pair_rows)
            structures.append({
                "K": k, "covariance_type": covariance_type,
                "best_fit_id": winner["fit_id"], "best_seed": winner["seed"],
                "successful_fit_count": int(sum(result["success"] for result in group)),
                "converged_fit_count": len(successful),
                "converged_fraction": len(successful) / len(group),
                "iterations": winner["iterations"], "log_likelihood": winner["log_likelihood"],
                "parameter_count": winner["parameter_count"], "AIC": winner["AIC"], "BIC": winner["BIC"],
                **stability, **aggregate,
                "unstable_solution_flag": stability["pairwise_ari_mean"] < 0.90 or stability["winner_recovery_rate_ari_ge_0_95"] < 0.70,
                "best_model_path": str((OUTPUT_DIR / "best_by_structure" / covariance_type / f"K{k}" / "model.joblib").relative_to(ROOT)),
                "best_posterior_path": str((OUTPUT_DIR / "best_by_structure" / covariance_type / f"K{k}" / "posterior_probabilities.csv.gz").relative_to(ROOT)),
                "best_viterbi_path": str((OUTPUT_DIR / "best_by_structure" / covariance_type / f"K{k}" / "viterbi_assignments.csv.gz").relative_to(ROOT)),
            })

    model_selection = pd.DataFrame(structures).sort_values(["covariance_type", "K"], ignore_index=True)
    model_selection["AIC_rank"] = model_selection["AIC"].rank(method="min").astype(int)
    model_selection["BIC_rank"] = model_selection["BIC"].rank(method="min").astype(int)
    best_covariance = str(model_selection.loc[model_selection["BIC"].idxmin(), "covariance_type"])
    model_selection["best_supported_covariance"] = model_selection["covariance_type"].eq(best_covariance)
    state_summary = pd.DataFrame(state_rows_all).sort_values(["covariance_type", "K", "state_index"], ignore_index=True)
    pair_summary = pd.DataFrame(pair_rows_all).sort_values(["covariance_type", "K", "state_a", "state_b"], ignore_index=True)
    diagnostics.drop(columns=[column for column in diagnostics.columns if column in {"monitor_history"}], errors="ignore").to_csv(
        OUTPUT_DIR / "all_fit_diagnostics.csv", index=False
    )
    model_selection.to_csv(OUTPUT_DIR / "model_selection.csv", index=False)
    state_summary.to_csv(OUTPUT_DIR / "candidate_state_summary.csv", index=False)
    pair_summary.to_csv(OUTPUT_DIR / "candidate_pairwise_centroid_separation.csv", index=False)

    run_metadata = {
        "analysis": "clean primary Gaussian HMM candidate evaluation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python scripts/pipeline/02_fit_primary_hmm.py",
        "input_data": str(DATA_PATH.relative_to(ROOT)),
        "input_sha256": sha256(DATA_PATH),
        "contract_files": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [UMBRELLA_PATH, CORE_PATH, EXTENSION_PATH]
        },
        "contract_statuses": {
            "umbrella": umbrella["contract_status"], "core": core["contract_status"],
            "extension": extension["contract_status"],
        },
        "fit_input_columns": list(EMISSIONS),
        "excluded_from_hmm_inference": [
            column for column in data.columns if column not in {
                "source_row", "timestamp_start", "timestamp_end", "primary_hmm_eligible", *EMISSIONS
            }
        ],
        "primary_observations": len(primary), "sequence_count": len(sequence_summary),
        "n_starts": {"K2_to_K5_per_covariance": args.n_starts, "K6_per_covariance": args.k6_starts},
        "k6_start_reduction_reason": "Exact 12,282-sequence timing made 10 starts computationally prohibitive; the prompt-authorized minimum of 5 was used without removing observations or changing sequence definitions.",
        "resumed_saved_fit_count": len(results) - len(tasks) if tasks else len(results),
        "new_fit_count": len(tasks), "max_iter": args.max_iter, "tolerance": args.tol,
        "n_jobs": args.n_jobs, "parallel_backend": args.backend,
        "hmmlearn_implementation": "scaling", "K_values": list(K_VALUES),
        "covariance_types": list(COVARIANCE_TYPES), "best_supported_covariance": best_covariance,
        "state_alignment": "Hungarian assignment minimizing Euclidean distance between standardized emission centroids and the winning fit",
        "package_versions": {
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
            "hmmlearn": hmmlearn_version, "joblib": joblib.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "legacy_hmm_artifacts_read": False,
        "phenology_or_environment_used_in_fit": False,
        "final_manuscript_state_count_selected": False,
    }
    write_json(OUTPUT_DIR / "run_metadata.json", run_metadata)
    diagnostic_figures(model_selection, state_summary, best_covariance)
    write_report(model_selection, state_summary, sequence_summary, standardization, best_covariance, run_metadata)
    print(f"Primary observations: {len(primary):,}")
    print(f"Sequences: {len(sequence_summary):,}")
    print(f"Best-supported covariance by global BIC: {best_covariance}")
    print(f"BIC minimum: K={int(model_selection.loc[model_selection['BIC'].idxmin(), 'K'])}")
    print(f"AIC minimum: K={int(model_selection.loc[model_selection['AIC'].idxmin(), 'K'])}")
    print(f"Report: {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
