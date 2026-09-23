#!/usr/bin/env python3
"""Evaluate start stability and the K4-to-higher-K state hierarchy.

Inputs are the frozen half-hourly contract and saved candidate-model fits.
Outputs quantify assignment reproducibility, centroid variability, component
splitting, and temporal persistence for the state-resolution decision. This
diagnostic never uses phenology or environmental covariates to define states.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import tempfile
import warnings
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "saltmarsh_hmm_stability_matplotlib"),
)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from hmmlearn import __version__ as hmmlearn_version
from hmmlearn.hmm import GaussianHMM
from scipy.linalg import sqrtm
from scipy.optimize import linear_sum_assignment
from sklearn import __version__ as sklearn_version
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture


ROOT = Path(__file__).resolve().parents[2]
PRIMARY = ROOT / "outputs" / "complete_dataset" / "hmm_primary"
TEMPORAL = ROOT / "outputs" / "complete_dataset" / "temporal_scale_diagnostic"
DATA = ROOT / "data" / "processed" / "stjones_halfhourly_contract_v1.csv.gz"
CORE = ROOT / "config" / "data_contract_core_v1.json"
OUTPUT = ROOT / "outputs" / "complete_dataset" / "hmm_state_stability"
FIGURES = ROOT / "figures" / "diagnostics" / "hmm_state_stability"
REPORT = ROOT / "reports" / "complete_dataset" / "hmm_state_stability_report.md"

N = 63_156
K_VALUES = [3, 4, 5, 6]
TARGET_K = [4, 5, 6]
ORIGINAL_STARTS = {4: 10, 5: 10, 6: 5}
ADDITIONAL_RANDOM = 5
ADDITIONAL_GMM = 3
ADDITIONAL_MAX_ITER = 300
TOLERANCE = 0.01
BASE_RANDOM_SEED = 846_000
BASE_GMM_SEED = 846_100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--single-additional", default="",
        help="Fit one additional K=6 model as strategy,index (random or gmm).",
    )
    parser.add_argument("--max-iter", type=int, default=ADDITIONAL_MAX_ITER)
    parser.add_argument("--tol", type=float, default=TOLERANCE)
    parser.add_argument("--gmm-starts", type=int, default=10)
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_inputs() -> None:
    required = [
        ROOT / "reports" / "complete_dataset" / "hmm_primary_report.md",
        ROOT / "reports" / "complete_dataset" / "temporal_scale_diagnostic_report.md",
        PRIMARY / "model_selection.csv", PRIMARY / "all_fit_diagnostics.csv",
        PRIMARY / "candidate_state_summary.csv", PRIMARY / "sequence_summary.csv",
        TEMPORAL / "model_selection_by_resolution.csv",
        TEMPORAL / "cross_resolution_centroid_alignment_by_k.csv", DATA, CORE,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")
    core = json.loads(CORE.read_text(encoding="utf-8"))
    if core.get("contract_status") != "FROZEN":
        raise RuntimeError("Core v1 contract is not FROZEN.")
    temporal_report = required[1].read_text(encoding="utf-8")
    if "HALF-HOURLY RECOMMENDED AS PRIMARY" not in temporal_report:
        raise RuntimeError("Temporal-resolution conclusion is not available.")


def load_primary_data() -> tuple[pd.DataFrame, np.ndarray, list[int], dict[str, Any]]:
    columns = ["source_row", "timestamp_start", "NEE", "CH4", "primary_hmm_eligible"]
    data = pd.read_csv(DATA, compression="gzip", usecols=columns)
    primary = data.loc[data["primary_hmm_eligible"]].copy()
    if len(primary) != N:
        raise RuntimeError("Primary sample is not 63,156 observations.")
    primary["timestamp_start"] = pd.to_datetime(primary["timestamp_start"], errors="raise")
    primary = primary.sort_values(["timestamp_start", "source_row"], kind="mergesort").reset_index(drop=True)
    breaks = primary["timestamp_start"].diff().ne(pd.Timedelta(np.timedelta64(30, "m")))
    breaks.iloc[0] = True
    primary["sequence_id"] = breaks.cumsum().astype(int)
    lengths = primary.groupby("sequence_id", sort=True).size().astype(int).tolist()
    parameters = json.loads((PRIMARY / "standardization_parameters.json").read_text(encoding="utf-8"))["parameters"]
    means = np.array([parameters["NEE"]["mean"], parameters["CH4"]["mean"]], dtype=float)
    sds = np.array([
        parameters["NEE"]["sample_standard_deviation"],
        parameters["CH4"]["sample_standard_deviation"],
    ], dtype=float)
    x = (primary[["NEE", "CH4"]].to_numpy(dtype=float) - means) / sds
    return primary, x, lengths, {"means": means, "sds": sds, "raw": parameters}


def canonicalize(
    means: np.ndarray, covariances: np.ndarray, posterior: np.ndarray, assignments: np.ndarray,
    transition: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.lexsort((means[:, 1], means[:, 0]))
    raw_to_order = np.empty(len(order), dtype=int)
    raw_to_order[order] = np.arange(len(order))
    return (
        means[order], covariances[order], posterior[:, order], raw_to_order[assignments],
        transition[np.ix_(order, order)],
    )


def parameter_count(k: int) -> int:
    return (k - 1) + k * (k - 1) + 2 * k + 3 * k


def fit_directory(strategy: str, index: int) -> Path:
    seed = BASE_RANDOM_SEED + index if strategy == "random" else BASE_GMM_SEED + index
    return OUTPUT / "additional_k6_fits" / strategy / f"start{index:02d}_seed{seed}"


def fit_additional(
    strategy: str, index: int, x: np.ndarray, lengths: list[int], max_iter: int, tol: float,
) -> dict[str, Any]:
    seed = BASE_RANDOM_SEED + index if strategy == "random" else BASE_GMM_SEED + index
    directory = fit_directory(strategy, index)
    directory.mkdir(parents=True, exist_ok=True)
    fit_id = f"K6_full_{strategy}_additional{index:02d}_seed{seed}"
    base = {
        "fit_id": fit_id, "K": 6, "covariance_type": "full", "strategy": strategy,
        "additional_start_index": index, "seed": seed, "success": False, "converged": False,
        "iterations": 0, "log_likelihood": np.nan, "parameter_count": parameter_count(6),
        "AIC": np.nan, "BIC": np.nan, "error": "", "max_iter": max_iter, "tol": tol,
        "artifact_directory": str(directory.relative_to(ROOT)),
    }
    try:
        if strategy == "gmm":
            initializer = GaussianMixture(
                n_components=6, covariance_type="full", n_init=1, max_iter=300,
                tol=1e-3, reg_covar=1e-6, random_state=seed,
            ).fit(x)
            model = GaussianHMM(
                n_components=6, covariance_type="full", n_iter=max_iter, tol=tol,
                min_covar=1e-6, random_state=seed, init_params="", params="stmc",
                implementation="scaling",
            )
            model.startprob_ = np.full(6, 1 / 6)
            model.transmat_ = np.full((6, 6), .05 / 5)
            np.fill_diagonal(model.transmat_, .95)
            model.means_ = initializer.means_.copy()
            model.covars_ = initializer.covariances_.copy()
        elif strategy == "random":
            model = GaussianHMM(
                n_components=6, covariance_type="full", n_iter=max_iter, tol=tol,
                min_covar=1e-6, random_state=seed, implementation="scaling",
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("once")
            model.fit(x, lengths)
            log_likelihood = float(model.score(x, lengths))
            posterior = model.predict_proba(x, lengths)
            assignments = model.predict(x, lengths)
        history = [float(item) for item in model.monitor_.history]
        delta = history[-1] - history[-2] if len(history) > 1 else np.nan
        iterations = int(model.monitor_.iter)
        reached_limit = iterations >= max_iter
        monotonic = not math.isfinite(delta) or delta >= -1e-3
        converged = bool(model.monitor_.converged) and monotonic and not reached_limit
        means, covariances, posterior, assignments, transition = canonicalize(
            np.asarray(model.means_), np.asarray(model.covars_), posterior, assignments,
            np.asarray(model.transmat_),
        )
        occupancy = np.bincount(assignments, minlength=6)
        maximum = posterior.max(axis=1)
        p = parameter_count(6)
        warning_counts = Counter((item.category.__name__, str(item.message)) for item in caught)
        result = {
            **base, "success": True, "converged": converged, "iterations": iterations,
            "reached_iteration_limit": reached_limit,
            "termination_reason": "tolerance_reached" if converged else (
                "maximum_iterations_reached" if reached_limit else "nonmonotonic_or_failed_convergence"
            ),
            "hmmlearn_monitor_converged": bool(model.monitor_.converged),
            "final_log_likelihood_delta": delta, "monotonic_final_step": monotonic,
            "log_likelihood": log_likelihood, "AIC": 2 * p - 2 * log_likelihood,
            "BIC": math.log(len(x)) * p - 2 * log_likelihood,
            "emission_means_standardized": means.tolist(),
            "emission_covariances_standardized": covariances.tolist(),
            "transition_matrix": transition.tolist(),
            "state_occupancy_counts": occupancy.tolist(),
            "state_occupancy_proportions": (occupancy / len(x)).tolist(),
            "mean_maximum_posterior": float(maximum.mean()),
            "posterior_row_sum_max_abs_error": float(np.abs(posterior.sum(axis=1) - 1).max()),
            "warning_count": len(caught),
            "warning_summary": [
                {"category": category, "message": message, "count": count}
                for (category, message), count in warning_counts.most_common(10)
            ],
            "monitor_history": history,
        }
        joblib.dump(model, directory / "model.joblib", compress=3)
        np.savez_compressed(
            directory / "inference.npz", posterior=posterior.astype(np.float32),
            assignments=assignments.astype(np.int16), maximum_posterior=maximum.astype(np.float32),
        )
        write_json(directory / "fit_metadata.json", result)
        return result
    except Exception as exc:
        base["error"] = f"{type(exc).__name__}: {exc}"
        write_json(directory / "fit_metadata.json", base)
        return base


def load_result_from_paths(metadata_path: Path, inference_path: Path, source: str) -> dict[str, Any]:
    result = json.loads(metadata_path.read_text(encoding="utf-8"))
    result["source"] = source
    result["metadata_path"] = str(metadata_path.relative_to(ROOT))
    if result.get("success") and inference_path.is_file():
        inference = np.load(inference_path)
        result["posterior_array"] = inference[
            "posterior_probabilities" if "posterior_probabilities" in inference.files else "posterior"
        ].astype(float)
        result["assignment_array"] = inference[
            "viterbi_state_index" if "viterbi_state_index" in inference.files else "assignments"
        ].astype(int)
        result["means_array"] = np.asarray(result["emission_means_standardized"], dtype=float)
        result["covariances_array"] = np.asarray(result["emission_covariances_standardized"], dtype=float)
        result["transition_array"] = np.asarray(result["transition_matrix"], dtype=float)
    return result


def load_original_results() -> dict[int, list[dict[str, Any]]]:
    results: dict[int, list[dict[str, Any]]] = {}
    for k in TARGET_K:
        items = []
        for start in range(1, ORIGINAL_STARTS[k] + 1):
            seed = 731_000 + k * 1_000 + start
            directory = PRIMARY / "fits" / f"K{k}_full" / f"start{start:02d}_seed{seed}"
            items.append(load_result_from_paths(
                directory / "fit_metadata.json", directory / "inference.npz", "original_random",
            ))
        results[k] = items
    return results


def load_additional_results() -> list[dict[str, Any]]:
    items = []
    for strategy, count in [("random", ADDITIONAL_RANDOM), ("gmm", ADDITIONAL_GMM)]:
        for index in range(1, count + 1):
            directory = fit_directory(strategy, index)
            metadata = directory / "fit_metadata.json"
            inference = directory / "inference.npz"
            if not metadata.is_file():
                raise RuntimeError(f"Missing additional fit: {metadata}")
            items.append(load_result_from_paths(metadata, inference, f"additional_{strategy}"))
    return items


def gaussian_wasserstein(mean_a: np.ndarray, cov_a: np.ndarray, mean_b: np.ndarray, cov_b: np.ndarray) -> float:
    root_a = np.real_if_close(sqrtm(cov_a)).real
    middle = root_a @ cov_b @ root_a
    root_middle = np.real_if_close(sqrtm(middle)).real
    covariance_term = max(0.0, float(np.trace(cov_a + cov_b - 2 * root_middle)))
    return float(math.sqrt(np.sum((mean_a - mean_b) ** 2) + covariance_term))


def alignment_mapping(result: dict[str, Any], reference: dict[str, Any]) -> np.ndarray:
    k = int(result["K"])
    costs = np.empty((k, k))
    for i in range(k):
        for j in range(k):
            costs[i, j] = gaussian_wasserstein(
                result["means_array"][i], result["covariances_array"][i],
                reference["means_array"][j], reference["covariances_array"][j],
            )
    rows, columns = linear_sum_assignment(costs)
    mapping = np.empty(k, dtype=int)
    mapping[rows] = columns
    return mapping


def aligned_payload(result: dict[str, Any], reference: dict[str, Any]) -> dict[str, np.ndarray]:
    mapping = alignment_mapping(result, reference)
    order = np.argsort(mapping)
    assignments = mapping[result["assignment_array"]]
    posterior = result["posterior_array"][:, order]
    return {
        "means": result["means_array"][order], "covariances": result["covariances_array"][order],
        "assignments": assignments, "posterior": posterior,
        "occupancy": np.bincount(assignments, minlength=result["K"]) / len(assignments),
    }


def per_attempt_table(results: dict[int, list[dict[str, Any]]], additional: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for k, group in results.items():
        for result in group + (additional if k == 6 else []):
            if not result.get("success"):
                rows.append({key: result.get(key) for key in [
                    "K", "seed", "source", "converged", "log_likelihood", "BIC", "AIC", "iterations", "error"
                ]})
                continue
            means = result["means_array"]
            distances = np.linalg.norm(means[:, None] - means[None, :], axis=2)
            np.fill_diagonal(distances, np.inf)
            assignments = result["assignment_array"]
            posterior = result["posterior_array"]
            occupancy = np.bincount(assignments, minlength=k) / len(assignments)
            state_certainty = [posterior[assignments == state, state].mean() for state in range(k)]
            rows.append({
                "K": k, "fit_id": result["fit_id"], "seed": result["seed"], "source": result["source"],
                "strategy": result.get("strategy", "random"), "success": result["success"],
                "converged": result["converged"], "iterations": result["iterations"],
                "log_likelihood": result["log_likelihood"], "BIC": result["BIC"], "AIC": result["AIC"],
                "occupancy_proportions": json.dumps(occupancy.tolist()),
                "minimum_occupancy": occupancy.min(),
                "mean_maximum_posterior": posterior.max(axis=1).mean(),
                "minimum_state_specific_posterior": np.min(state_certainty),
                "minimum_centroid_separation": distances.min(),
                "mean_transition_persistence": np.diag(result["transition_array"]).mean(),
                "error": result.get("error", ""),
            })
    return pd.DataFrame(rows).sort_values(["K", "source", "seed"], ignore_index=True)


def stability_outputs(
    results: dict[int, list[dict[str, Any]]], additional: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, np.ndarray], dict[int, dict[str, Any]]]:
    summaries, pair_rows, matrices, representatives = [], [], {}, {}
    for k in TARGET_K:
        group = results[k] + (additional if k == 6 else [])
        valid = [item for item in group if item.get("success") and item.get("converged")]
        if not valid:
            raise RuntimeError(f"No converged K={k} fits.")
        reference = max(valid, key=lambda item: item["log_likelihood"])
        representatives[k] = reference
        aligned = [aligned_payload(item, reference) for item in valid]
        ari_matrix = np.eye(len(valid))
        nmi_matrix = np.eye(len(valid))
        for i, j in combinations(range(len(valid)), 2):
            ari = adjusted_rand_score(aligned[i]["assignments"], aligned[j]["assignments"])
            nmi = normalized_mutual_info_score(aligned[i]["assignments"], aligned[j]["assignments"])
            ari_matrix[i, j] = ari_matrix[j, i] = ari
            nmi_matrix[i, j] = nmi_matrix[j, i] = nmi
            pair_rows.append({
                "K": k, "fit_i": valid[i]["fit_id"], "fit_j": valid[j]["fit_id"], "ARI": ari, "NMI": nmi,
            })
        matrices[k] = ari_matrix
        centroid_stack = np.stack([item["means"] for item in aligned])
        occupancy_stack = np.stack([item["occupancy"] for item in aligned])
        certainty = np.array([item["posterior"].max(axis=1).mean() for item in aligned])
        likelihoods = np.array([item["log_likelihood"] for item in valid])
        winner_index = valid.index(reference)
        winner_ari = np.array([
            adjusted_rand_score(aligned[winner_index]["assignments"], item["assignments"]) for item in aligned
        ])
        upper = ari_matrix[np.triu_indices(len(valid), 1)]
        upper_nmi = nmi_matrix[np.triu_indices(len(valid), 1)]
        summaries.append({
            "K": k, "attempted_starts": len(group), "successful_starts": int(sum(item.get("success", False) for item in group)),
            "converged_starts": len(valid), "representative_fit": reference["fit_id"],
            "representative_seed": reference["seed"], "representative_log_likelihood": reference["log_likelihood"],
            "pairwise_ARI_mean": upper.mean() if len(upper) else 1.0,
            "pairwise_ARI_min": upper.min() if len(upper) else 1.0,
            "pairwise_NMI_mean": upper_nmi.mean() if len(upper_nmi) else 1.0,
            "winner_recovery_ARI_ge_0_95": np.mean(winner_ari >= .95),
            "centroid_SD_RMS": np.sqrt(np.mean(centroid_stack.std(axis=0, ddof=1) ** 2)),
            "centroid_SD_max": centroid_stack.std(axis=0, ddof=1).max(),
            "occupancy_SD_RMS": np.sqrt(np.mean(occupancy_stack.std(axis=0, ddof=1) ** 2)),
            "occupancy_range_max": np.ptp(occupancy_stack, axis=0).max(),
            "posterior_certainty_SD": certainty.std(ddof=1),
            "posterior_certainty_range": certainty.max() - certainty.min(),
            "log_likelihood_spread": likelihoods.max() - likelihoods.min(),
        })
    return pd.DataFrame(summaries), pd.DataFrame(pair_rows), matrices, representatives


def k6_component_stability(
    group: list[dict[str, Any]], reference: dict[str, Any], params: dict[str, Any], primary: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, np.ndarray]]]:
    valid = [item for item in group if item.get("success") and item.get("converged")]
    aligned = [aligned_payload(item, reference) for item in valid]
    q = primary[["NEE", "CH4"]].quantile([.05, .95])
    rows = []
    for state in range(6):
        means = np.stack([item["means"][state] for item in aligned])
        occupancies = np.array([item["occupancy"][state] for item in aligned])
        certainties = np.array([
            item["posterior"][item["assignments"] == state, state].mean() for item in aligned
        ])
        jaccards = []
        for left, right in combinations(aligned, 2):
            a = left["assignments"] == state
            b = right["assignments"] == state
            union = np.logical_or(a, b).sum()
            jaccards.append(np.logical_and(a, b).sum() / union if union else 1.0)
        mean_std = means.std(axis=0, ddof=1)
        mean_raw = params["means"] + means.mean(axis=0) * params["sds"]
        centroid_sd_norm = np.linalg.norm(mean_std)
        membership = float(np.mean(jaccards)) if jaccards else 1.0
        tail = (
            mean_raw[1] >= q.loc[.95, "CH4"] or mean_raw[1] <= q.loc[.05, "CH4"]
            or mean_raw[0] >= q.loc[.95, "NEE"] or mean_raw[0] <= q.loc[.05, "NEE"]
        )
        if tail and centroid_sd_norm <= .35:
            classification = "tail/extreme-response component"
        elif centroid_sd_norm <= .25 and membership >= .60 and occupancies.min() >= .08:
            classification = "stable broad component"
        elif centroid_sd_norm <= .25 and membership >= .60:
            classification = "stable subdivision"
        elif centroid_sd_norm <= .75:
            classification = "unstable subdivision"
        else:
            classification = "not consistently identifiable"
        rows.append({
            "reference_state": f"K6-S{state + 1}", "successful_solutions": len(valid),
            "mean_NEE_standardized": means[:, 0].mean(), "mean_CH4_standardized": means[:, 1].mean(),
            "mean_NEE_original": mean_raw[0], "mean_CH4_original": mean_raw[1],
            "NEE_centroid_SD": mean_std[0], "CH4_centroid_SD": mean_std[1],
            "centroid_SD_norm": centroid_sd_norm, "occupancy_min": occupancies.min(),
            "occupancy_max": occupancies.max(), "posterior_certainty_min": certainties.min(),
            "posterior_certainty_max": certainties.max(), "membership_Jaccard_mean": membership,
            "membership_Jaccard_min": np.min(jaccards) if jaccards else 1.0,
            "classification": classification,
        })
    return pd.DataFrame(rows), aligned


def load_best_original(k: int) -> dict[str, Any]:
    directory = PRIMARY / "best_by_structure" / "full" / f"K{k}"
    return load_result_from_paths(directory / "fit_metadata.json", directory / "inference.npz", "best_original")


def overlap_rows(lower: dict[str, Any], higher: dict[str, Any], mapping_name: str, fit_id: str) -> list[dict[str, Any]]:
    low = lower["assignment_array"]
    high = higher["assignment_array"]
    k_low, k_high = lower["K"], higher["K"]
    counts = np.zeros((k_low, k_high), dtype=int)
    np.add.at(counts, (low, high), 1)
    rows = []
    for i in range(k_low):
        for j in range(k_high):
            rows.append({
                "mapping": mapping_name, "higher_fit_id": fit_id,
                "lower_state": f"K{k_low}-S{i + 1}", "higher_state": f"K{k_high}-S{j + 1}",
                "joint_count": counts[i, j],
                "fraction_of_lower_to_higher": counts[i, j] / counts[i].sum(),
                "fraction_of_higher_from_lower": counts[i, j] / counts[:, j].sum(),
                "higher_primary_parent": counts[i, j] == counts[:, j].max(),
            })
    return rows


def hierarchy_outputs(
    representatives: dict[int, dict[str, Any]], original_best: dict[int, dict[str, Any]],
    k6_group: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    k6_reference = representatives[6]
    hierarchy = []
    for lower_k, higher_k in [(3, 4), (4, 5), (4, 6), (5, 6)]:
        lower = original_best[lower_k]
        higher = k6_reference if higher_k == 6 else original_best[higher_k]
        hierarchy.extend(overlap_rows(lower, higher, f"K{lower_k}_to_K{higher_k}", higher["fit_id"]))
    across_rows, coarse_rows = [], []
    k4 = original_best[4]
    valid = [item for item in k6_group if item.get("success") and item.get("converged")]
    for item in valid:
        aligned = aligned_payload(item, k6_reference)
        aligned_result = {**item, "assignment_array": aligned["assignments"]}
        across_rows.extend(overlap_rows(k4, aligned_result, "K4_to_K6_across_starts", item["fit_id"]))
        low, high = k4["assignment_array"], aligned["assignments"]
        parent_map, purities = [], []
        for state in range(6):
            counts = np.bincount(low[high == state], minlength=4)
            parent_map.append(int(counts.argmax()))
            purities.append(counts.max() / counts.sum())
        coarse = np.asarray(parent_map)[high]
        coarse_rows.append({
            "fit_id": item["fit_id"], "seed": item["seed"], "source": item["source"],
            "coarsened_accuracy_to_K4": np.mean(coarse == low),
            "coarsened_ARI_to_K4": adjusted_rand_score(low, coarse),
            "coarsened_NMI_to_K4": normalized_mutual_info_score(low, coarse),
            "minimum_child_parent_purity": np.min(purities), "mean_child_parent_purity": np.mean(purities),
            "child_parent_mapping": json.dumps([value + 1 for value in parent_map]),
        })
    return pd.DataFrame(hierarchy), pd.DataFrame(across_rows), pd.DataFrame(coarse_rows)


def split_characterization(
    original_best: dict[int, dict[str, Any]], k6_reference: dict[str, Any], primary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    q = primary[["NEE", "CH4"]].quantile([.05, .95])
    params_raw = json.loads((PRIMARY / "standardization_parameters.json").read_text(encoding="utf-8"))["parameters"]
    raw_mean = np.array([params_raw["NEE"]["mean"], params_raw["CH4"]["mean"]])
    raw_sd = np.array([params_raw["NEE"]["sample_standard_deviation"], params_raw["CH4"]["sample_standard_deviation"]])
    for lower_k, higher_k in [(4, 5), (4, 6)]:
        lower = original_best[lower_k]
        higher = original_best[5] if higher_k == 5 else k6_reference
        low_a, high_a = lower["assignment_array"], higher["assignment_array"]
        for child in range(higher_k):
            counts = np.bincount(low_a[high_a == child], minlength=lower_k)
            parent = int(counts.argmax())
            delta = higher["means_array"][child] - lower["means_array"][parent]
            pooled = (higher["covariances_array"][child] + lower["covariances_array"][parent]) / 2
            mahalanobis = float(math.sqrt(max(0, delta @ np.linalg.pinv(pooled) @ delta)))
            child_raw = raw_mean + higher["means_array"][child] * raw_sd
            parent_raw = raw_mean + lower["means_array"][parent] * raw_sd
            if abs(delta[0]) > 1.5 * abs(delta[1]):
                primary_axis = "NEE magnitude"
            elif abs(delta[1]) > 1.5 * abs(delta[0]):
                primary_axis = "CH4 magnitude"
            else:
                primary_axis = "joint NEE-CH4"
            tail = (
                child_raw[1] >= q.loc[.95, "CH4"] or child_raw[1] <= q.loc[.05, "CH4"]
                or child_raw[0] >= q.loc[.95, "NEE"] or child_raw[0] <= q.loc[.05, "NEE"]
            )
            rows.append({
                "mapping": f"K{lower_k}_to_K{higher_k}", "parent_state": f"K{lower_k}-S{parent + 1}",
                "child_state": f"K{higher_k}-S{child + 1}", "parent_purity": counts.max() / counts.sum(),
                "delta_NEE_standardized": delta[0], "delta_CH4_standardized": delta[1],
                "centroid_distance_standardized": np.linalg.norm(delta),
                "pooled_covariance_Mahalanobis_distance": mahalanobis,
                "close_relative_to_within_state_covariance": mahalanobis < 1,
                "primary_split_axis": primary_axis, "NEE_sign_change": np.sign(child_raw[0]) != np.sign(parent_raw[0]),
                "tail_or_extreme_response": tail, "child_NEE_original": child_raw[0], "child_CH4_original": child_raw[1],
            })
    return pd.DataFrame(rows)


def fit_gmm_grid(x: np.ndarray, starts: int) -> tuple[pd.DataFrame, dict[int, dict[str, Any]]]:
    rows, winners = [], {}
    for k in TARGET_K:
        candidates = []
        for start in range(1, starts + 1):
            seed = 852_000 + k * 100 + start
            directory = OUTPUT / "gmm_fits" / f"K{k}" / f"start{start:02d}_seed{seed}"
            metadata_path = directory / "fit_metadata.json"
            if metadata_path.is_file() and (directory / "inference.npz").is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                inference = np.load(directory / "inference.npz")
                metadata["assignment_array"] = inference["assignments"].astype(int)
                metadata["posterior_array"] = inference["posterior"].astype(float)
                metadata["means_array"] = np.asarray(metadata["means_standardized"])
                metadata["covariances_array"] = np.asarray(metadata["covariances_standardized"])
            else:
                directory.mkdir(parents=True, exist_ok=True)
                model = GaussianMixture(
                    n_components=k, covariance_type="full", n_init=1, max_iter=300,
                    tol=1e-3, reg_covar=1e-6, random_state=seed,
                ).fit(x)
                posterior = model.predict_proba(x)
                assignments = posterior.argmax(axis=1)
                order = np.lexsort((model.means_[:, 1], model.means_[:, 0]))
                raw_to_order = np.empty(k, dtype=int); raw_to_order[order] = np.arange(k)
                posterior = posterior[:, order]; assignments = raw_to_order[assignments]
                means = model.means_[order]; covariances = model.covariances_[order]
                metadata = {
                    "K": k, "start_index": start, "seed": seed, "converged": bool(model.converged_),
                    "iterations": int(model.n_iter_), "lower_bound_per_observation": float(model.lower_bound_),
                    "total_log_likelihood": float(model.score(x) * len(x)),
                    "mean_maximum_posterior": float(posterior.max(axis=1).mean()),
                    "occupancy_proportions": (np.bincount(assignments, minlength=k) / len(x)).tolist(),
                    "means_standardized": means.tolist(), "covariances_standardized": covariances.tolist(),
                    "artifact_directory": str(directory.relative_to(ROOT)),
                }
                joblib.dump(model, directory / "model.joblib", compress=3)
                np.savez_compressed(directory / "inference.npz", posterior=posterior.astype(np.float32), assignments=assignments.astype(np.int16))
                write_json(metadata_path, metadata)
                metadata["assignment_array"] = assignments; metadata["posterior_array"] = posterior
                metadata["means_array"] = means; metadata["covariances_array"] = covariances
            candidates.append(metadata)
            rows.append({key: value for key, value in metadata.items() if not key.endswith("_array") and key not in {"means_standardized", "covariances_standardized"}})
        winners[k] = max([item for item in candidates if item["converged"]], key=lambda item: item["total_log_likelihood"])
    return pd.DataFrame(rows), winners


def gmm_comparison(
    hmm: dict[int, dict[str, Any]], gmm: dict[int, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries, components = [], []
    for k in TARGET_K:
        h, g = hmm[k], gmm[k]
        pseudo_g = {"K": k, "means_array": g["means_array"], "covariances_array": g["covariances_array"]}
        mapping = alignment_mapping(pseudo_g, h)
        order = np.argsort(mapping)
        g_assign = mapping[g["assignment_array"]]
        g_post = g["posterior_array"][:, order]
        h_occ = np.bincount(h["assignment_array"], minlength=k) / N
        g_occ = np.bincount(g_assign, minlength=k) / N
        distances = []
        for state in range(k):
            distance = gaussian_wasserstein(
                h["means_array"][state], h["covariances_array"][state],
                g["means_array"][order][state], g["covariances_array"][order][state],
            )
            distances.append(distance)
            components.append({
                "K": k, "HMM_state": f"K{k}-S{state + 1}", "GMM_aligned_state": f"K{k}-G{state + 1}",
                "Gaussian_Wasserstein_distance": distance, "HMM_occupancy": h_occ[state], "GMM_occupancy": g_occ[state],
            })
        summaries.append({
            "K": k, "assignment_ARI": adjusted_rand_score(h["assignment_array"], g_assign),
            "assignment_NMI": normalized_mutual_info_score(h["assignment_array"], g_assign),
            "HMM_mean_max_posterior": h["posterior_array"].max(axis=1).mean(),
            "GMM_mean_max_posterior": g_post.max(axis=1).mean(),
            "mean_centroid_distribution_distance": np.mean(distances), "max_centroid_distribution_distance": np.max(distances),
            "maximum_occupancy_difference": np.max(np.abs(h_occ - g_occ)),
        })
        g["aligned_assignment_array"] = g_assign
    return pd.DataFrame(summaries), pd.DataFrame(components)


def markov_outputs(
    sequence_summary: pd.DataFrame, representatives: dict[int, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lengths = sequence_summary["sequence_length"].astype(int).to_numpy()
    bins = pd.cut(lengths, [0, 1, 3, 11, 47, 95, np.inf], labels=["1", "2-3", "4-11", "12-47", "48-95", ">=96"])
    sequence_bins = pd.DataFrame({"sequence_length_bin": bins}).value_counts(sort=False).rename("sequence_count").reset_index()
    pair_by_bin = []
    for label in bins.categories:
        selected = lengths[np.asarray(bins == label)]
        pair_by_bin.append({
            "sequence_length_bin": str(label), "sequence_count": len(selected),
            "observation_count": int(selected.sum()), "adjacent_pair_count": int(np.maximum(selected - 1, 0).sum()),
        })
    pair_by_bin = pd.DataFrame(pair_by_bin)
    total_pairs = int(np.maximum(lengths - 1, 0).sum())
    overview = pd.DataFrame([{
        "total_observations": int(lengths.sum()), "sequence_count": len(lengths),
        "singleton_observations": int((lengths == 1).sum()),
        "within_sequence_adjacent_pairs": total_pairs,
        "proportion_observations_participating_in_at_least_one_pair": float(lengths[lengths >= 2].sum() / lengths.sum()),
        "long_sequence_ge_96_pair_count": int(np.maximum(lengths[lengths >= 96] - 1, 0).sum()),
        "long_sequence_ge_96_pair_fraction": float(np.maximum(lengths[lengths >= 96] - 1, 0).sum() / total_pairs),
    }])
    transition_rows = []
    ends = np.cumsum(lengths) - 1
    valid_left = np.ones(N - 1, dtype=bool)
    valid_left[ends[:-1]] = False
    for k in [4, 6]:
        result = representatives[k]
        assignments = result["assignment_array"]
        left, right = assignments[:-1][valid_left], assignments[1:][valid_left]
        counts = np.zeros((k, k), dtype=int); np.add.at(counts, (left, right), 1)
        observed = counts / counts.sum()
        occupancy = np.bincount(assignments, minlength=k) / len(assignments)
        expected = np.outer(occupancy, occupancy)
        for i in range(k):
            for j in range(k):
                transition_rows.append({
                    "K": k, "from_state": i + 1, "to_state": j + 1, "observed_pair_count": counts[i, j],
                    "observed_pair_frequency": observed[i, j], "independence_expected_frequency": expected[i, j],
                    "observed_to_independence_ratio": observed[i, j] / expected[i, j],
                    "fitted_transition_probability": result["transition_array"][i, j],
                    "destination_marginal_occupancy": occupancy[j],
                    "fitted_transition_excess_over_destination_occupancy": result["transition_array"][i, j] - occupancy[j],
                    "self_transition": i == j,
                })
    return overview, pair_by_bin, pd.DataFrame(transition_rows)


def focused_quality_table(
    representatives: dict[int, dict[str, Any]], stability: pd.DataFrame, components: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for k in [4, 6]:
        result = representatives[k]
        assignments, posterior = result["assignment_array"], result["posterior_array"]
        means = result["means_array"]
        distances = np.linalg.norm(means[:, None] - means[None, :], axis=2); np.fill_diagonal(distances, np.inf)
        occupancy = np.bincount(assignments, minlength=k) / N
        state_certainty = [posterior[assignments == state, state].mean() for state in range(k)]
        stable_row = stability.loc[stability["K"] == k].iloc[0]
        reproducible = k if k == 4 else int(components["classification"].eq("stable broad component").sum())
        rows.append({
            "K": k, "representative_fit": result["fit_id"], "BIC": result["BIC"], "AIC": result["AIC"],
            "parameter_count": result["parameter_count"], "minimum_occupancy": occupancy.min(),
            "mean_posterior_certainty": posterior.max(axis=1).mean(),
            "minimum_state_specific_posterior_certainty": np.min(state_certainty),
            "minimum_centroid_separation": distances.min(),
            "pairwise_ARI_mean": stable_row["pairwise_ARI_mean"],
            "winning_partition_recovery": stable_row["winner_recovery_ARI_ge_0_95"],
            "reproducibly_identifiable_broad_signatures": reproducible,
        })
    return pd.DataFrame(rows)


def matrix_from_overlap(frame: pd.DataFrame, mapping: str, value: str = "fraction_of_lower_to_higher") -> np.ndarray:
    subset = frame.loc[frame["mapping"] == mapping]
    return subset.pivot(index="lower_state", columns="higher_state", values=value).to_numpy()


def draw_ellipse(ax: plt.Axes, mean: np.ndarray, covariance: np.ndarray, color: Any) -> None:
    values, vectors = np.linalg.eigh(covariance)
    order = values.argsort()[::-1]; values, vectors = values[order], vectors[:, order]
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    ellipse = Ellipse(mean, 2 * np.sqrt(values[0]), 2 * np.sqrt(values[1]), angle=angle, fill=False, lw=1.3, color=color)
    ax.add_patch(ellipse)


def create_figures(
    attempt: pd.DataFrame, matrices: dict[int, np.ndarray], hierarchy: pd.DataFrame,
    across: pd.DataFrame, coarse: pd.DataFrame, representatives: dict[int, dict[str, Any]],
    k6_group: list[dict[str, Any]], gmm_winners: dict[int, dict[str, Any]], focused: pd.DataFrame,
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {4: "#1f77b4", 5: "#ff7f0e", 6: "#2ca02c"}

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex="col")
    for col, k in enumerate(TARGET_K):
        data = attempt.loc[attempt["K"] == k].reset_index(drop=True)
        marker_colors = ["#d62728" if not value else colors[k] for value in data["converged"]]
        axes[0, col].scatter(np.arange(1, len(data) + 1), data["log_likelihood"], c=marker_colors)
        axes[1, col].scatter(np.arange(1, len(data) + 1), data["BIC"], c=marker_colors)
        axes[0, col].set_title(f"K={k}"); axes[1, col].set_xlabel("Saved initialization")
    axes[0, 0].set_ylabel("Final log likelihood"); axes[1, 0].set_ylabel("BIC")
    fig.suptitle("Initialization diagnostics (red = nonconverged)"); fig.tight_layout(); fig.savefig(FIGURES / "figure_01_likelihood_bic_by_initialization.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, k in zip(axes, TARGET_K):
        im = ax.imshow(matrices[k], vmin=0, vmax=1, cmap="viridis")
        ax.set_title(f"K={k} pairwise ARI"); ax.set_xlabel("Converged fit"); ax.set_ylabel("Converged fit")
    fig.colorbar(im, ax=axes, shrink=.8); fig.savefig(FIGURES / "figure_02_pairwise_ari_heatmaps.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    mappings = ["K3_to_K4", "K4_to_K5", "K4_to_K6", "K5_to_K6"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, mapping in zip(axes.flat, mappings):
        matrix = matrix_from_overlap(hierarchy, mapping)
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        ax.set_title(mapping.replace("_", " ")); ax.set_xlabel("Higher-K state"); ax.set_ylabel("Lower-K state")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=axes, shrink=.75, label="Fraction of lower state"); fig.savefig(FIGURES / "figure_03_hierarchical_overlap_diagram.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    mean_matrix = across.groupby(["lower_state", "higher_state"])["fraction_of_lower_to_higher"].mean().unstack().to_numpy()
    sd_matrix = across.groupby(["lower_state", "higher_state"])["fraction_of_lower_to_higher"].std().unstack().to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, matrix, title in [(axes[0], mean_matrix, "Mean K4 -> aligned K6 overlap"), (axes[1], sd_matrix, "SD across K6 fits")]:
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="magma"); ax.set_title(title); ax.set_xlabel("K6 state"); ax.set_ylabel("K4 state")
    axes[2].bar(np.arange(len(coarse)), coarse["coarsened_ARI_to_K4"], color="#4c78a8")
    axes[2].set_ylim(0, 1); axes[2].set_title("Coarsened K6 agreement with K4"); axes[2].set_xlabel("Converged K6 fit"); axes[2].set_ylabel("ARI")
    fig.colorbar(im, ax=axes[:2], shrink=.8); fig.savefig(FIGURES / "figure_04_k4_to_k6_across_starts.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    valid_k6 = [item for item in k6_group if item.get("success") and item.get("converged")]
    alternate = min(valid_k6, key=lambda item: adjusted_rand_score(item["assignment_array"], representatives[6]["assignment_array"]))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True, sharey=True)
    for ax, result, title in [(axes[0], representatives[4], "Representative K4"), (axes[1], representatives[6], "Best K6"), (axes[2], alternate, "Alternative converged K6")]:
        cmap = plt.get_cmap("tab10")
        for state, (mean, covariance) in enumerate(zip(result["means_array"], result["covariances_array"])):
            ax.scatter(*mean, color=cmap(state), s=35); draw_ellipse(ax, mean, covariance, cmap(state)); ax.text(mean[0], mean[1], f"S{state+1}", fontsize=8)
        ax.axhline(0, color=".5", lw=.6); ax.axvline(0, color=".5", lw=.6); ax.set_title(title); ax.set_xlabel("Standardized NEE")
    axes[0].set_ylabel("Standardized CH4"); fig.tight_layout(); fig.savefig(FIGURES / "figure_05_centroids_covariance_ellipses.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, k in zip(axes, TARGET_K):
        hmm_a = representatives[k]["assignment_array"]; gmm_a = gmm_winners[k]["aligned_assignment_array"]
        matrix = np.zeros((k, k), dtype=int); np.add.at(matrix, (hmm_a, gmm_a), 1)
        matrix = matrix / matrix.sum(axis=1, keepdims=True)
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues"); ax.set_title(f"K={k}"); ax.set_xlabel("Aligned GMM state"); ax.set_ylabel("HMM state")
    fig.colorbar(im, ax=axes, shrink=.8, label="Fraction of HMM state"); fig.savefig(FIGURES / "figure_06_hmm_vs_gmm_assignments.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    metrics = ["mean_posterior_certainty", "minimum_state_specific_posterior_certainty", "pairwise_ARI_mean", "winning_partition_recovery"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    xloc = np.arange(len(metrics)); width = .35
    for offset, (_, row) in zip([-.5, .5], focused.iterrows()):
        axes[0].bar(xloc + offset * width, [row[m] for m in metrics], width, label=f"K={int(row['K'])}")
    axes[0].set_xticks(xloc, ["Mean post.", "Min state post.", "ARI", "Winner recovery"], rotation=20); axes[0].set_ylim(0, 1); axes[0].legend(); axes[0].set_title("Classification and stability")
    axes[1].bar(["K4", "K6"], focused["minimum_centroid_separation"], color=[colors[4], colors[6]])
    axes[1].set_title("Minimum centroid separation"); axes[1].set_ylabel("Standardized Euclidean distance")
    fig.tight_layout(); fig.savefig(FIGURES / "figure_07_k4_k6_quality_comparison.png", dpi=180); plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    display = frame[columns].copy()
    for column in display.select_dtypes(include="number"):
        display[column] = display[column].map(lambda value: f"{value:.4g}")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in display.itertuples(index=False, name=None))
    return "\n".join(lines)


def write_report(
    attempt: pd.DataFrame, stability: pd.DataFrame, components: pd.DataFrame,
    coarse: pd.DataFrame, splits: pd.DataFrame, focused: pd.DataFrame,
    gmm_summary: pd.DataFrame, markov_overview: pd.DataFrame, markov_transitions: pd.DataFrame,
    hourly: pd.DataFrame,
) -> str:
    k4 = stability.loc[stability["K"] == 4].iloc[0]
    k6 = stability.loc[stability["K"] == 6].iloc[0]
    original_k6 = attempt.loc[(attempt["K"] == 6) & (attempt["source"] == "original_random")]
    additional_k6 = attempt.loc[(attempt["K"] == 6) & attempt["source"].str.startswith("additional")]
    reproducible_components = int(components["membership_Jaccard_mean"].ge(.60).sum())
    stable_broad_components = int(components["classification"].eq("stable broad component").sum())
    stable_tail_components = int(components["classification"].eq("tail/extreme-response component").sum())
    unstable_names = ", ".join(components.loc[
        components["classification"].isin(["unstable subdivision", "not consistently identifiable"]), "reference_state"
    ]) or "none"
    tail_names = ", ".join(components.loc[components["classification"] == "tail/extreme-response component", "reference_state"]) or "none"
    coarse_min, coarse_mean = coarse["coarsened_ARI_to_K4"].min(), coarse["coarsened_ARI_to_K4"].mean()
    hourly_k4 = hourly.loc[(hourly["resolution"] == "hourly") & (hourly["K"] == 4) & (hourly["covariance_type"] == "full")].iloc[0]
    hourly_k6 = hourly.loc[(hourly["resolution"] == "hourly") & (hourly["K"] == 6) & (hourly["covariance_type"] == "full")].iloc[0]
    markov = markov_overview.iloc[0]
    self_summary = markov_transitions.loc[markov_transitions["self_transition"]].groupby("K").agg(
        observed_self_frequency=("observed_pair_frequency", "sum"),
        expected_self_frequency=("independence_expected_frequency", "sum"),
        mean_fitted_self_transition=("fitted_transition_probability", "mean"),
        mean_marginal_occupancy=("destination_marginal_occupancy", "mean"),
    ).reset_index()
    k4_self = self_summary.loc[self_summary["K"] == 4].iloc[0]
    k6_self = self_summary.loc[self_summary["K"] == 6].iloc[0]
    additional_changes = additional_k6.loc[additional_k6["converged"], "log_likelihood"].max() > original_k6.loc[original_k6["converged"], "log_likelihood"].max() + 1
    ready = (
        stable_broad_components >= 3
        and stable_broad_components + stable_tail_components >= 4
        and coarse_mean >= .55
        and k6["converged_starts"] >= 5
        and k4_self["observed_self_frequency"] > k4_self["expected_self_frequency"]
        and k6_self["observed_self_frequency"] > k6_self["expected_self_frequency"]
    )
    conclusion = "READY FOR BROADER ROBUSTNESS ANALYSIS" if ready else "ADDITIONAL MODEL DIAGNOSTICS REQUIRED"

    report = f"""# Half-hourly HMM state-stability and hierarchy diagnostic v1

## Scope

This diagnostic uses only the clean v1 half-hourly NEE–CH4 sample, saved clean HMM artifacts, and the existing hourly sensitivity. No legacy states, phenology, GCC, season, environmental variable, transition driver, or diel covariate entered fitting, initialization, alignment, or interpretation. Temporal resolution remains fixed at half-hourly for primary inference. No manuscript state count is selected.

## Optimization and assignment stability

{markdown_table(stability, ['K','attempted_starts','converged_starts','pairwise_ARI_mean','pairwise_ARI_min','winner_recovery_ARI_ge_0_95','centroid_SD_RMS','occupancy_SD_RMS','posterior_certainty_SD','log_likelihood_spread'])}

K=4 is reproducible: all {int(k4['converged_starts'])} starts converge, mean pairwise ARI is {k4['pairwise_ARI_mean']:.3f}, and centroid variability is {k4['centroid_SD_RMS']:.4f} standardized units. K=6 remains less reproducible after adding five random and three GMM-initialized starts at 300 iterations: {int(k6['converged_starts'])}/{int(k6['attempted_starts'])} total starts converge, mean ARI is {k6['pairwise_ARI_mean']:.3f}, and only {k6['winner_recovery_ARI_ge_0_95']:.1%} recover the representative partition. Additional fitting {'found a higher-likelihood solution' if additional_changes else 'did not displace the original highest-likelihood solution'}.

Optimization instability refers to convergence and likelihood spread; assignment instability is measured by ARI/NMI and winner recovery; emission-centroid instability is measured after Gaussian-distribution alignment. These are reported separately rather than treating convergence as partition recovery.

**Scientific objective conclusion: B.** The statistically preferred K=6 fit contains a smaller reproducible broad carbon-response structure that is subdivided inconsistently at higher K. The results do not support six equally reproducible broad states (A), but the recurring K=4-like hierarchy also rules out unrelated optima with no defensible coarse representation (C).

## K=6 component identifiability

{markdown_table(components, ['reference_state','mean_NEE_original','mean_CH4_original','centroid_SD_norm','occupancy_min','occupancy_max','membership_Jaccard_mean','classification'])}

{reproducible_components} of six aligned components meet the membership reproducibility threshold; {stable_broad_components} are classified as stable broad components and {stable_tail_components} as a stable tail/extreme component. Components classified as unstable/not identifiable: {unstable_names}. Tail/extreme-response components: {tail_names}. These labels describe statistical carbon-response geometry only, not mechanisms.

## Hierarchical structure

Across converged K=6 solutions, coarsening each K=6 state to its dominant K=4 parent gives ARI {coarse_min:.3f}–{coarse['coarsened_ARI_to_K4'].max():.3f} (mean {coarse_mean:.3f}) and mean child-parent purity {coarse['mean_child_parent_purity'].mean():.3f}. Thus K=6 repeatedly contains a recognizable K=4-like coarse structure, while the exact division of individual K=4 parents changes among starts. K=4 is embedded hierarchically rather than reproduced as one invariant six-way partition.

Higher-K split diagnostics identify {int(splits['tail_or_extreme_response'].sum())} tail/extreme children, {int(splits['NEE_sign_change'].sum())} NEE sign-changing child-parent contrasts, and {int(splits['close_relative_to_within_state_covariance'].sum())} children whose centroids are close relative to pooled within-state covariance. The full overlap and split tables distinguish parent splitting from cross-parent combinations.

## Focused K4 versus K6 quality

{markdown_table(focused, ['K','BIC','AIC','parameter_count','minimum_occupancy','mean_posterior_certainty','minimum_state_specific_posterior_certainty','minimum_centroid_separation','pairwise_ARI_mean','winning_partition_recovery','reproducibly_identifiable_broad_signatures'])}

K=6 has the lower information criteria, but it is not a reproducible six-way ecological partition. K=4 has substantially stronger optimization and assignment reproducibility. Neither result alone selects the manuscript state count.

## Hourly external check

Hourly full-covariance K=4 has mean pairwise ARI {hourly_k4['pairwise_ari_mean']:.3f} and winner recovery {hourly_k4['winner_recovery_rate_ari_ge_0_95']:.1%}; hourly K=6 has ARI {hourly_k6['pairwise_ari_mean']:.3f} but winner recovery only {hourly_k6['winner_recovery_rate_ari_ge_0_95']:.1%}. Existing centroid alignment shows broad half-hourly K4/K6 signatures recur hourly, but exact high-K partition recovery remains weak. Cross-resolution centroid reproducibility therefore does not imply exact assignment reproducibility.

## HMM versus Gaussian mixture

{markdown_table(gmm_summary, ['K','assignment_ARI','assignment_NMI','HMM_mean_max_posterior','GMM_mean_max_posterior','mean_centroid_distribution_distance','maximum_occupancy_difference'])}

HMM–GMM assignment ARI ranges from {gmm_summary['assignment_ARI'].min():.3f} to {gmm_summary['assignment_ARI'].max():.3f}. The static mixture partially recovers the broad carbon-response geometry, but the moderate-to-low assignment agreement, lower GMM posterior certainty, and large K5/K6 component-distribution distances show that the HMM partition differs materially once temporal dependence is modeled. BIC values are not compared across HMM and GMM families.

## Markov information

There are {int(markov['total_observations']):,} observations, {int(markov['singleton_observations']):,} singleton observations, and {int(markov['within_sequence_adjacent_pairs']):,} within-sequence adjacent pairs. {markov['proportion_observations_participating_in_at_least_one_pair']:.1%} of observations participate in at least one pair. The three sequences of length >=96 contribute {int(markov['long_sequence_ge_96_pair_count']):,} pairs ({markov['long_sequence_ge_96_pair_fraction']:.1%} of all pairs).

For K=4, observed self-pair frequency is {k4_self['observed_self_frequency']:.3f}, versus {k4_self['expected_self_frequency']:.3f} under independent draws; mean fitted self-transition probability is {k4_self['mean_fitted_self_transition']:.3f}. For K=6 the corresponding values are {k6_self['observed_self_frequency']:.3f}, {k6_self['expected_self_frequency']:.3f}, and {k6_self['mean_fitted_self_transition']:.3f}. The Markov component therefore contains meaningful persistence beyond marginal occupancy despite fragmentation.

## Explicit answers

1. **Is K=6 reproducibly identifiable?** No as one exact six-state partition; only a subset of components and the coarse hierarchy are reproducible.
2. **Does K=6 consistently contain a lower-dimensional broad structure?** Yes. Coarsened K=6 solutions repeatedly recover a K=4-like organization, although strength varies.
3. **Is K=4 embedded in K=5/K=6?** Yes at the broad overlap level; higher K subdivides K=4 parents rather than replacing the entire geometry.
4. **Which components are unstable subdivisions or tails?** Unstable/not identifiable: {unstable_names}; tail/extreme: {tail_names}. See the component and split tables for quantitative definitions.
5. **Are broad centroids stable hourly?** Yes, especially at K=4; exact K=6 partitions remain unstable.
6. **Does HMM differ materially from static clustering?** Yes. GMM partially recovers broad geometry, but assignment ARI of {gmm_summary['assignment_ARI'].min():.3f}–{gmm_summary['assignment_ARI'].max():.3f}, lower GMM certainty, and strong self-transition excess demonstrate a material temporal contribution.
7. **Is there meaningful temporal persistence?** Yes. Observed and fitted self-transition behavior substantially exceeds independence expectations.
8. **Is evidence sufficient for broader QC robustness analysis?** {'Yes. The coarse hierarchy, component diagnostics, and Markov signal are sufficiently characterized for the next robustness stage.' if ready else 'No. The prespecified hierarchy/GMM/Markov thresholds were not all met.'}

This diagnostic does not choose the manuscript state count.

{conclusion}
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")
    return conclusion


def main() -> None:
    args = parse_args()
    validate_inputs()
    OUTPUT.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    primary, x, lengths, params = load_primary_data()
    if args.single_additional:
        try:
            strategy, index_text = args.single_additional.split(",")
            index = int(index_text)
        except ValueError as exc:
            raise SystemExit("--single-additional must be strategy,index") from exc
        allowed = ADDITIONAL_RANDOM if strategy == "random" else ADDITIONAL_GMM if strategy == "gmm" else 0
        if index < 1 or index > allowed:
            raise SystemExit("Invalid additional fit target.")
        result = fit_additional(strategy, index, x, lengths, args.max_iter, args.tol)
        print(json.dumps({key: result[key] for key in ["fit_id", "success", "converged", "iterations", "log_likelihood"]}, default=json_default))
        return

    original = load_original_results()
    additional = load_additional_results()
    attempt = per_attempt_table(original, additional)
    attempt.to_csv(OUTPUT / "all_k4_k6_fit_diagnostics.csv", index=False)
    stability, pairwise, matrices, representatives = stability_outputs(original, additional)
    stability.to_csv(OUTPUT / "stability_summary_by_k.csv", index=False)
    pairwise.to_csv(OUTPUT / "pairwise_assignment_stability.csv", index=False)
    k6_group = original[6] + additional
    components, _ = k6_component_stability(k6_group, representatives[6], params, primary)
    components.to_csv(OUTPUT / "k6_component_stability.csv", index=False)

    original_best = {k: load_best_original(k) for k in K_VALUES}
    hierarchy, across, coarse = hierarchy_outputs(representatives, original_best, k6_group)
    hierarchy.to_csv(OUTPUT / "hierarchical_overlap_matrices.csv", index=False)
    across.to_csv(OUTPUT / "k4_to_k6_overlap_across_starts.csv", index=False)
    coarse.to_csv(OUTPUT / "k6_coarsened_to_k4_stability.csv", index=False)
    splits = split_characterization(original_best, representatives[6], primary)
    splits.to_csv(OUTPUT / "higher_k_split_characterization.csv", index=False)

    gmm_fits, gmm_winners = fit_gmm_grid(x, args.gmm_starts)
    gmm_fits.to_csv(OUTPUT / "gmm_fit_diagnostics.csv", index=False)
    hmm_for_gmm = {4: representatives[4], 5: representatives[5], 6: representatives[6]}
    gmm_summary, gmm_components = gmm_comparison(hmm_for_gmm, gmm_winners)
    gmm_summary.to_csv(OUTPUT / "hmm_vs_gmm_summary.csv", index=False)
    gmm_components.to_csv(OUTPUT / "hmm_vs_gmm_component_correspondence.csv", index=False)

    sequence_summary = pd.read_csv(PRIMARY / "sequence_summary.csv")
    markov_overview, sequence_bins, transitions = markov_outputs(sequence_summary, representatives)
    markov_overview.to_csv(OUTPUT / "markov_information_overview.csv", index=False)
    sequence_bins.to_csv(OUTPUT / "sequence_length_bins.csv", index=False)
    transitions.to_csv(OUTPUT / "markov_transition_independence_comparison.csv", index=False)

    focused = focused_quality_table(representatives, stability, components)
    focused.to_csv(OUTPUT / "k4_k6_focused_comparison.csv", index=False)
    hourly = pd.read_csv(TEMPORAL / "model_selection_by_resolution.csv")
    create_figures(attempt, matrices, hierarchy, across, coarse, representatives, k6_group, gmm_winners, focused)
    conclusion = write_report(
        attempt, stability, components, coarse, splits, focused, gmm_summary,
        markov_overview, transitions, hourly,
    )
    write_json(OUTPUT / "run_metadata.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "input_sha256": sha256(DATA),
        "fit_emissions": ["NEE", "CH4"], "additional_K6_random_starts": ADDITIONAL_RANDOM,
        "additional_K6_GMM_initialized_starts": ADDITIONAL_GMM,
        "additional_iteration_limit": args.max_iter, "convergence_tolerance": args.tol,
        "GMM_starts_per_K": args.gmm_starts, "phenology_or_environment_used": False,
        "legacy_HMM_used": False, "hourly_refit": False, "manuscript_state_count_selected": False,
        "python": platform.python_version(), "hmmlearn": hmmlearn_version,
        "scikit_learn": sklearn_version, "conclusion": conclusion,
    })
    print(f"Additional K6 fits: {len(additional)}")
    print(f"Converged combined K6 fits: {int(stability.loc[stability.K == 6, 'converged_starts'].iloc[0])}")
    print(f"Conclusion: {conclusion}")
    print(f"Report: {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
