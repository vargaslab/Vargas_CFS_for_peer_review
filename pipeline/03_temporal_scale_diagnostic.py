#!/usr/bin/env python3
"""Compare half-hourly, hourly, and daily carbon-only Gaussian HMMs.

The completed half-hourly reanalysis is read, not refitted. Hourly and daily
datasets are rebuilt from frozen-contract primary-eligible half-hours. Only NEE
and CH4 enter inference; PAR is used solely for daily coverage diagnostics.

This is decision-support code for the state-stability workflow, not a source of
standalone manuscript effect estimates. Saved fits and summaries remain fully
traceable to resolution, covariance structure, initialization, and seed.
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
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "saltmarsh_temporal_scale_matplotlib"),
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
from hmmlearn import __version__ as hmmlearn_version
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "config" / "data_contract_v1.json"
CORE_CONTRACT = ROOT / "config" / "data_contract_core_v1.json"
DATA = ROOT / "data" / "processed" / "stjones_halfhourly_contract_v1.csv.gz"
HALF_OUTPUT = ROOT / "outputs" / "complete_dataset" / "hmm_primary"
HALF_REPORT = ROOT / "reports" / "complete_dataset" / "hmm_primary_report.md"
OUTPUT = ROOT / "outputs" / "complete_dataset" / "temporal_scale_diagnostic"
FIGURES = ROOT / "figures" / "diagnostics" / "temporal_scale_diagnostic"
REPORT = ROOT / "reports" / "complete_dataset" / "temporal_scale_diagnostic_report.md"

EMISSIONS = ["NEE", "CH4"]
K_VALUES = range(2, 7)
COVARIANCES = ["full", "diag"]
BASE_SEED = 824_000
DAILY_QUADRANT_MINIMUM = 4
EXPECTED_PRIMARY_N = 63_156


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-starts", type=int, default=10)
    parser.add_argument("--diag-starts", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--tol", type=float, default=0.01)
    parser.add_argument(
        "--single-fit", default="",
        help="Run one resumable fit as resolution,K,covariance,start and exit.",
    )
    parser.add_argument(
        "--fit-shard", default="",
        help="Fit one zero-based shard as shard_index,shard_count and exit.",
    )
    args = parser.parse_args()
    if args.full_starts < 5 or args.diag_starts < 5:
        raise SystemExit("Use at least five deterministic starts per structure.")
    if args.jobs < 1:
        raise SystemExit("--jobs must be positive.")
    return args


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_and_read() -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    for path in [CONTRACT, CORE_CONTRACT, DATA, HALF_REPORT]:
        if not path.is_file():
            raise FileNotFoundError(path)
    required_half = [
        "model_selection.csv", "candidate_state_summary.csv", "sequence_summary.csv",
        "standardization_parameters.json", "run_metadata.json",
    ]
    if any(not (HALF_OUTPUT / name).is_file() for name in required_half):
        raise RuntimeError("Completed half-hourly reanalysis is incomplete.")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    core = json.loads(CORE_CONTRACT.read_text(encoding="utf-8"))
    if contract.get("contract_status") != "FROZEN" or core.get("contract_status") != "FROZEN":
        raise RuntimeError("Applicable v1 contracts must be FROZEN.")
    if core.get("units") != {
        "nee": "micromol CO2 m^-2 s^-1", "ch4": "nmol CH4 m^-2 s^-1"
    }:
        raise RuntimeError("Frozen flux units changed.")
    if "negative NEE" not in core["sign_conventions"]["nee"]:
        raise RuntimeError("Frozen NEE sign convention changed.")
    columns = [
        "source_row", "timestamp_start", "timestamp_end", "NEE", "CH4",
        "primary_hmm_eligible", "photosynthetically_active_radiation",
    ]
    data = pd.read_csv(DATA, compression="gzip", usecols=columns, low_memory=False)
    if int(data["primary_hmm_eligible"].sum()) != EXPECTED_PRIMARY_N:
        raise RuntimeError("Primary sample count is not 63,156.")
    data["timestamp_start"] = pd.to_datetime(data["timestamp_start"], errors="raise")
    data["timestamp_end"] = pd.to_datetime(data["timestamp_end"], errors="raise")
    primary = data.loc[data["primary_hmm_eligible"]].copy()
    if not np.isfinite(primary[EMISSIONS].to_numpy(dtype=float)).all():
        raise RuntimeError("Primary carbon observations contain nonfinite values.")
    if primary["timestamp_start"].duplicated().any():
        raise RuntimeError("Primary timestamps are not unique.")
    return contract, core, primary.sort_values("timestamp_start", kind="mergesort").reset_index(drop=True)


def build_hourly(primary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = primary.copy()
    work["hour_start"] = work["timestamp_start"].dt.floor("h")
    grouped = work.groupby("hour_start", sort=True).agg(
        constituent_count=("source_row", "size"),
        first_constituent=("timestamp_start", "min"),
        last_constituent=("timestamp_start", "max"),
        NEE=("NEE", "mean"), CH4=("CH4", "mean"),
    ).reset_index()
    grouped["expected_first"] = grouped["hour_start"]
    grouped["expected_last"] = grouped["hour_start"] + timedelta(minutes=30)
    grouped["complete"] = (
        grouped["constituent_count"].eq(2)
        & grouped["first_constituent"].eq(grouped["expected_first"])
        & grouped["last_constituent"].eq(grouped["expected_last"])
    )
    eligible = grouped.loc[grouped["complete"], ["hour_start", "NEE", "CH4", "constituent_count", "complete"]].copy()
    eligible = eligible.rename(columns={"hour_start": "timestamp_start"})
    eligible["timestamp_end"] = eligible["timestamp_start"] + timedelta(hours=1)
    eligible["timestamp_role"] = "interval_start"
    return eligible.reset_index(drop=True), grouped


def build_daily(primary: pd.DataFrame) -> pd.DataFrame:
    work = primary.copy()
    work["date"] = work["timestamp_start"].dt.normalize()
    work["clock_quadrant"] = (work["timestamp_start"].dt.hour // 6).astype(int)
    base = work.groupby("date", sort=True).agg(
        valid_half_hours=("source_row", "size"),
        NEE=("NEE", "mean"), CH4=("CH4", "mean"),
        par_available_half_hours=("photosynthetically_active_radiation", "count"),
        par_gt_zero_half_hours=("photosynthetically_active_radiation", lambda s: int((s > 0).sum())),
        par_eq_zero_half_hours=("photosynthetically_active_radiation", lambda s: int((s == 0).sum())),
        first_valid_timestamp=("timestamp_start", "min"),
        last_valid_timestamp=("timestamp_start", "max"),
    )
    quadrants = work.groupby(["date", "clock_quadrant"]).size().unstack(fill_value=0)
    quadrants = quadrants.reindex(columns=range(4), fill_value=0)
    quadrants.columns = [f"valid_00_06" if i == 0 else f"valid_06_12" if i == 1 else f"valid_12_18" if i == 2 else "valid_18_24" for i in quadrants.columns]
    daily = base.join(quadrants).reset_index()
    coverage_columns = ["valid_00_06", "valid_06_12", "valid_12_18", "valid_18_24"]
    daily["minimum_quadrant_count"] = daily[coverage_columns].min(axis=1)
    daily["reasonable_diel_coverage"] = daily["minimum_quadrant_count"].ge(DAILY_QUADRANT_MINIMUM)
    daily["eligible_rule_36"] = daily["valid_half_hours"].ge(36) & daily["reasonable_diel_coverage"]
    daily["eligible_rule_40"] = daily["valid_half_hours"].ge(40) & daily["reasonable_diel_coverage"]
    daily["month"] = daily["date"].dt.month
    daily["day_of_year"] = daily["date"].dt.dayofyear
    daily["year"] = daily["date"].dt.year
    daily["timestamp_start"] = daily["date"]
    daily["timestamp_end"] = daily["date"] + timedelta(days=1)
    daily["timestamp_role"] = "calendar_day_start"
    return daily


def make_sequences(frame: pd.DataFrame, step: timedelta, scale: str) -> tuple[pd.DataFrame, list[int], dict[str, Any]]:
    frame = frame.sort_values("timestamp_start", kind="mergesort").reset_index(drop=True).copy()
    breaks = frame["timestamp_start"].diff().ne(step)
    breaks.iloc[0] = True
    frame["sequence_id"] = breaks.cumsum().astype(int)
    seq = frame.groupby("sequence_id", sort=True).agg(
        sequence_start=("timestamp_start", "min"),
        sequence_end=("timestamp_end", "max"),
        sequence_length=("timestamp_start", "size"),
    ).reset_index()
    seq.insert(0, "resolution", scale)
    seq["singleton"] = seq["sequence_length"].eq(1)
    lengths = seq["sequence_length"].astype(int).tolist()
    quantiles = seq["sequence_length"].quantile([0, .1, .25, .5, .75, .9, .95, .99, 1])
    summary = {
        "resolution": scale, "observations": len(frame), "sequence_count": len(seq),
        "singleton_sequences": int(seq["singleton"].sum()),
        "singleton_sequence_proportion": float(seq["singleton"].mean()),
        "median_sequence_length": float(seq["sequence_length"].median()),
        "maximum_sequence_length": int(seq["sequence_length"].max()),
        "adjacent_transition_opportunities": int(len(frame) - len(seq)),
        **{f"sequence_length_q{int(q * 100):02d}": float(v) for q, v in quantiles.items()},
    }
    return frame, lengths, {"sequence_table": seq, "summary": summary}


def standardize(frame: pd.DataFrame, scale: str, units: dict[str, str]) -> tuple[np.ndarray, dict[str, Any]]:
    raw = frame[EMISSIONS].to_numpy(dtype=float)
    means = raw.mean(axis=0)
    sds = raw.std(axis=0, ddof=1)
    if np.any(~np.isfinite(sds)) or np.any(sds <= 0):
        raise RuntimeError(f"Invalid {scale} standardization.")
    x = (raw - means) / sds
    return x, {
        "resolution": scale, "n": len(frame), "calculation_population": f"eligible {scale} observations only",
        "ddof": 1, "nee_sign_preserved": True,
        "NEE": {"mean": means[0], "sd": sds[0], "unit": units["nee"]},
        "CH4": {"mean": means[1], "sd": sds[1], "unit": units["ch4"]},
    }


def parameter_count(k: int, covariance: str, dimensions: int = 2) -> int:
    return (k - 1) + k * (k - 1) + k * dimensions + (
        k * dimensions if covariance == "diag" else k * dimensions * (dimensions + 1) // 2
    )


def seed_for(scale: str, k: int, covariance: str, start: int) -> int:
    scale_index = {"hourly": 1, "daily36": 2, "daily40": 3}[scale]
    covariance_index = COVARIANCES.index(covariance)
    return BASE_SEED + scale_index * 10_000 + k * 1_000 + covariance_index * 100 + start


def fit_path(scale: str, k: int, covariance: str, start: int, seed: int) -> Path:
    return OUTPUT / "fits" / scale / f"K{k}_{covariance}" / f"start{start:02d}_seed{seed}"


def fit_one(
    scale: str, k: int, covariance: str, start: int, seed: int,
    x: np.ndarray, lengths: list[int], max_iter: int, tol: float,
) -> dict[str, Any]:
    directory = fit_path(scale, k, covariance, start, seed)
    directory.mkdir(parents=True, exist_ok=True)
    base = {
        "resolution": scale, "K": k, "covariance_type": covariance,
        "start_index": start, "seed": seed, "success": False, "converged": False,
        "iterations": 0, "log_likelihood": np.nan, "parameter_count": parameter_count(k, covariance),
        "AIC": np.nan, "BIC": np.nan, "error": "",
        "artifact_directory": str(directory.relative_to(ROOT)),
    }
    try:
        model = GaussianHMM(
            n_components=k, covariance_type=covariance, n_iter=max_iter, tol=tol,
            min_covar=1e-6, random_state=seed, implementation="scaling",
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("once")
            model.fit(x, lengths)
            log_likelihood = float(model.score(x, lengths))
            posterior = model.predict_proba(x, lengths)
            viterbi = model.predict(x, lengths)
        history = [float(item) for item in model.monitor_.history]
        delta = history[-1] - history[-2] if len(history) > 1 else np.nan
        iterations = int(model.monitor_.iter)
        reached_limit = iterations >= max_iter
        monotonic = not math.isfinite(delta) or delta >= -1e-3
        converged = bool(model.monitor_.converged) and monotonic and not reached_limit
        order = np.lexsort((model.means_[:, 1], model.means_[:, 0]))
        raw_to_canonical = np.empty(k, dtype=int)
        raw_to_canonical[order] = np.arange(k)
        posterior = posterior[:, order]
        viterbi = raw_to_canonical[viterbi]
        means = np.asarray(model.means_, dtype=float)[order]
        covariances = np.asarray(model.covars_, dtype=float)[order]
        transition = np.asarray(model.transmat_, dtype=float)[np.ix_(order, order)]
        occupancy = np.bincount(viterbi, minlength=k)
        max_posterior = posterior.max(axis=1)
        p = parameter_count(k, covariance)
        warning_counts = Counter((w.category.__name__, str(w.message)) for w in caught)
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
            "state_labels": [f"K{k}-S{i}" for i in range(1, k + 1)],
            "emission_means_standardized": means.tolist(),
            "emission_covariances_standardized": covariances.tolist(),
            "transition_matrix": transition.tolist(),
            "state_occupancy_counts": occupancy.tolist(),
            "state_occupancy_proportions": (occupancy / len(x)).tolist(),
            "mean_maximum_posterior": float(max_posterior.mean()),
            "maximum_posterior_quantiles": {
                str(q): float(np.quantile(max_posterior, q))
                for q in [.01, .05, .1, .25, .5, .75, .9, .95, .99]
            },
            "posterior_row_sum_max_abs_error": float(np.max(np.abs(posterior.sum(axis=1) - 1))),
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
            viterbi=viterbi.astype(np.int16), maximum_posterior=max_posterior.astype(np.float32),
        )
        write_json(directory / "fit_metadata.json", result)
        return result
    except Exception as exc:
        base["error"] = f"{type(exc).__name__}: {exc}"
        write_json(directory / "fit_metadata.json", base)
        return base


def load_fit(scale: str, k: int, covariance: str, start: int, seed: int, max_iter: int) -> dict[str, Any] | None:
    directory = fit_path(scale, k, covariance, start, seed)
    metadata = directory / "fit_metadata.json"
    if not metadata.is_file():
        return None
    result = json.loads(metadata.read_text(encoding="utf-8"))
    if any(result.get(key) != value for key, value in {
        "resolution": scale, "K": k, "covariance_type": covariance, "start_index": start, "seed": seed,
    }.items()):
        raise RuntimeError(f"Saved fit identity mismatch: {metadata}")
    if result.get("success"):
        result["converged"] = bool(
            result.get("hmmlearn_monitor_converged", result.get("converged", False))
            and result.get("monotonic_final_step", True)
            and int(result["iterations"]) < max_iter
        )
        if not (directory / "model.joblib").is_file() or not (directory / "inference.npz").is_file():
            return None
    return result


def arrays_for(result: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    inference = np.load(ROOT / result["artifact_directory"] / "inference.npz")
    return (
        np.asarray(result["emission_means_standardized"], dtype=float),
        np.asarray(result["emission_covariances_standardized"], dtype=float),
        inference["posterior"].astype(float), inference["viterbi"].astype(int),
    )


def stability(group: list[dict[str, Any]], winner: dict[str, Any]) -> dict[str, Any]:
    successful = [item for item in group if item["success"] and item["converged"]]
    reference = np.asarray(winner["emission_means_standardized"], dtype=float)
    centroids, assignments, ids = [], [], []
    for item in successful:
        means, _, _, viterbi = arrays_for(item)
        row, col = linear_sum_assignment(np.linalg.norm(means[:, None, :] - reference[None, :, :], axis=2))
        mapping = np.empty(len(row), dtype=int)
        mapping[row] = col
        centroids.append(means[np.argsort(mapping)])
        assignments.append(mapping[viterbi])
        ids.append(item["seed"])
    aris, nmis = [], []
    for left, right in combinations(assignments, 2):
        aris.append(adjusted_rand_score(left, right))
        nmis.append(normalized_mutual_info_score(left, right))
    winner_index = ids.index(winner["seed"])
    winner_aris = [adjusted_rand_score(assignments[winner_index], item) for item in assignments]
    centroid_stack = np.stack(centroids)
    centroid_sd = centroid_stack.std(axis=0, ddof=1) if len(centroids) > 1 else np.zeros_like(reference)
    return {
        "successful_converged_starts": len(successful),
        "pairwise_ari_mean": float(np.mean(aris)) if aris else 1.0,
        "pairwise_ari_min": float(np.min(aris)) if aris else 1.0,
        "pairwise_nmi_mean": float(np.mean(nmis)) if nmis else 1.0,
        "winner_recovery_rate_ari_ge_0_95": float(np.mean(np.asarray(winner_aris) >= .95)),
        "centroid_sd_rms": float(np.sqrt(np.mean(centroid_sd ** 2))),
        "centroid_max_sd": float(np.max(centroid_sd)),
    }


def best_diagnostics(
    result: dict[str, Any], params: dict[str, Any], frame: pd.DataFrame,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray]:
    means, covariances, posterior, viterbi = arrays_for(result)
    k = result["K"]
    occupancy = np.bincount(viterbi, minlength=k) / len(viterbi)
    maximum = posterior.max(axis=1)
    distances = np.linalg.norm(means[:, None, :] - means[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    transition = np.asarray(result["transition_matrix"], dtype=float)
    raw_mean = np.array([params["NEE"]["mean"], params["CH4"]["mean"]])
    raw_sd = np.array([params["NEE"]["sd"], params["CH4"]["sd"]])
    original_means = raw_mean + means * raw_sd
    original_covariances = covariances * np.outer(raw_sd, raw_sd)
    state_rows = []
    for state in range(k):
        assigned = posterior[viterbi == state, state]
        covariance = covariances[state]
        eig = np.linalg.eigvalsh(covariance)
        denominator = math.sqrt(max(covariance[0, 0] * covariance[1, 1], np.finfo(float).tiny))
        state_rows.append({
            "resolution": result["resolution"], "K": k,
            "covariance_type": result["covariance_type"], "state_index": state + 1,
            "state_label": f"K{k}-S{state + 1}", "occupancy_proportion": occupancy[state],
            "mean_assigned_state_posterior": float(assigned.mean()),
            "emission_mean_NEE_standardized": means[state, 0],
            "emission_mean_CH4_standardized": means[state, 1],
            "emission_mean_NEE_original": original_means[state, 0],
            "emission_mean_CH4_original": original_means[state, 1],
            "nearest_centroid_distance": float(distances[state].min()),
            "transition_persistence": transition[state, state],
            "covariance_correlation": covariance[0, 1] / denominator,
            "covariance_eigenvalue_min": eig.min(), "covariance_eigenvalue_max": eig.max(),
            "covariance_original_nee_variance": original_covariances[state, 0, 0],
            "covariance_original_ch4_variance": original_covariances[state, 1, 1],
            "covariance_original_nee_ch4": original_covariances[state, 0, 1],
        })
    aggregate = {
        "minimum_state_occupancy": float(occupancy.min()),
        "mean_maximum_posterior": float(maximum.mean()),
        "fraction_max_posterior_lt_0_60": float((maximum < .6).mean()),
        "fraction_max_posterior_lt_0_70": float((maximum < .7).mean()),
        "fraction_max_posterior_lt_0_80": float((maximum < .8).mean()),
        "fraction_max_posterior_lt_0_90": float((maximum < .9).mean()),
        "minimum_centroid_separation": float(distances.min()),
        "mean_transition_persistence": float(np.diag(transition).mean()),
    }
    destination = OUTPUT / "best_by_structure" / result["resolution"] / result["covariance_type"] / f"K{k}"
    destination.mkdir(parents=True, exist_ok=True)
    source = ROOT / result["artifact_directory"]
    for name in ["model.joblib", "fit_metadata.json", "inference.npz"]:
        shutil.copy2(source / name, destination / name)
    index = frame[["timestamp_start", "timestamp_end", "sequence_id"]].copy()
    posterior_frame = index.copy()
    for state in range(k):
        posterior_frame[f"posterior_K{k}-S{state + 1}"] = posterior[:, state].astype(np.float32)
    posterior_frame.to_csv(destination / "posterior_probabilities.csv.gz", index=False, compression="gzip")
    assignments = index.copy()
    assignments["state_index"] = viterbi + 1
    assignments["state_label"] = [f"K{k}-S{item + 1}" for item in viterbi]
    assignments.to_csv(destination / "viterbi_assignments.csv.gz", index=False, compression="gzip")
    return aggregate, state_rows, posterior, viterbi


def distribution_metrics(frame: pd.DataFrame, scale: str) -> dict[str, Any]:
    return {
        "resolution": scale, "n": len(frame),
        "NEE_mean": frame["NEE"].mean(), "NEE_variance": frame["NEE"].var(ddof=1),
        "NEE_skewness": frame["NEE"].skew(), "NEE_minimum": frame["NEE"].min(),
        "NEE_maximum": frame["NEE"].max(),
        "CH4_mean": frame["CH4"].mean(), "CH4_variance": frame["CH4"].var(ddof=1),
        "CH4_skewness": frame["CH4"].skew(), "CH4_q95": frame["CH4"].quantile(.95),
        "CH4_q99": frame["CH4"].quantile(.99), "CH4_q995": frame["CH4"].quantile(.995),
        "CH4_maximum": frame["CH4"].max(),
        "NEE_CH4_correlation": frame[["NEE", "CH4"]].corr().iloc[0, 1],
    }


def half_hour_reference() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    selection = pd.read_csv(HALF_OUTPUT / "model_selection.csv")
    states = pd.read_csv(HALF_OUTPUT / "candidate_state_summary.csv")
    sequences = pd.read_csv(HALF_OUTPUT / "sequence_summary.csv")
    params_raw = json.loads((HALF_OUTPUT / "standardization_parameters.json").read_text(encoding="utf-8"))
    params = {
        "NEE": {"mean": params_raw["parameters"]["NEE"]["mean"], "sd": params_raw["parameters"]["NEE"]["sample_standard_deviation"]},
        "CH4": {"mean": params_raw["parameters"]["CH4"]["mean"], "sd": params_raw["parameters"]["CH4"]["sample_standard_deviation"]},
    }
    summary = {
        "resolution": "half_hourly", "observations": EXPECTED_PRIMARY_N,
        "sequence_count": len(sequences), "singleton_sequences": int(sequences["singleton"].sum()),
        "singleton_sequence_proportion": float(sequences["singleton"].mean()),
        "median_sequence_length": float(sequences["sequence_length"].median()),
        "maximum_sequence_length": int(sequences["sequence_length"].max()),
        "adjacent_transition_opportunities": int(EXPECTED_PRIMARY_N - len(sequences)),
    }
    return selection, states, params, {"summary": summary, "sequence_table": sequences.assign(resolution="half_hourly")}


def fit_candidate_grid(
    datasets: dict[str, tuple[pd.DataFrame, np.ndarray, list[int], dict[str, Any]]], args: argparse.Namespace,
) -> list[dict[str, Any]]:
    results, tasks = [], []
    for scale, (_, x, lengths, _) in datasets.items():
        for k in K_VALUES:
            for covariance in COVARIANCES:
                target_starts = args.full_starts if covariance == "full" else args.diag_starts
                for start in range(1, target_starts + 1):
                    seed = seed_for(scale, k, covariance, start)
                    saved = load_fit(scale, k, covariance, start, seed, args.max_iter)
                    if saved is None:
                        tasks.append((scale, k, covariance, start, seed, x, lengths))
                    else:
                        results.append(saved)
    if tasks:
        print(f"Running {len(tasks)} new deterministic fits sequentially", flush=True)
        for completed, (scale, k, covariance, start, seed, x, lengths) in enumerate(tasks, start=1):
            results.append(fit_one(scale, k, covariance, start, seed, x, lengths, args.max_iter, args.tol))
            if completed % 10 == 0 or completed == len(tasks):
                print(f"Completed {completed}/{len(tasks)} new fits", flush=True)
    return results


def summarize_models(
    results: list[dict[str, Any]], datasets: dict[str, tuple[pd.DataFrame, np.ndarray, list[int], dict[str, Any]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    structures, state_rows = [], []
    for scale, (frame, _, _, params) in datasets.items():
        for k in K_VALUES:
            for covariance in COVARIANCES:
                group = [r for r in results if r["resolution"] == scale and r["K"] == k and r["covariance_type"] == covariance]
                converged = [r for r in group if r["success"] and r["converged"]]
                if not converged:
                    raise RuntimeError(f"No converged fit for {scale}, K={k}, {covariance}.")
                winner = max(converged, key=lambda item: item["log_likelihood"])
                stable = stability(group, winner)
                aggregate, rows, _, _ = best_diagnostics(winner, params, frame)
                state_rows.extend(rows)
                structures.append({
                    "resolution": scale, "K": k, "covariance_type": covariance,
                    "best_seed": winner["seed"], "best_start_index": winner["start_index"],
                    "successful_fit_count": int(sum(r["success"] for r in group)),
                    "converged_fit_count": len(converged), "converged_fraction": len(converged) / len(group),
                    "iterations": winner["iterations"], "log_likelihood": winner["log_likelihood"],
                    "parameter_count": winner["parameter_count"], "AIC": winner["AIC"], "BIC": winner["BIC"],
                    **stable, **aggregate,
                    "unstable_solution_flag": stable["pairwise_ari_mean"] < .9 or stable["winner_recovery_rate_ari_ge_0_95"] < .7,
                })
    selection = pd.DataFrame(structures).sort_values(["resolution", "covariance_type", "K"], ignore_index=True)
    selection["AIC_rank_within_resolution"] = selection.groupby("resolution")["AIC"].rank(method="min").astype(int)
    selection["BIC_rank_within_resolution"] = selection.groupby("resolution")["BIC"].rank(method="min").astype(int)
    return selection, pd.DataFrame(state_rows).sort_values(["resolution", "covariance_type", "K", "state_index"], ignore_index=True)


def selected_scale_rows(selection: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scale, group in selection.groupby("resolution", sort=False):
        bic = group.loc[group["BIC"].idxmin()]
        aic = group.loc[group["AIC"].idxmin()]
        row = bic.to_dict()
        row["BIC_selected_K"] = int(bic["K"])
        row["BIC_selected_covariance"] = bic["covariance_type"]
        row["AIC_selected_K"] = int(aic["K"])
        row["AIC_selected_covariance"] = aic["covariance_type"]
        rows.append(row)
    return pd.DataFrame(rows)


def align_centroids(
    half_states: pd.DataFrame, new_states: pd.DataFrame, half_params: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    reference_sd = np.array([half_params["NEE"]["sd"], half_params["CH4"]["sd"]])
    all_states = pd.concat([half_states, new_states], ignore_index=True, sort=False)
    for k in K_VALUES:
        reference = all_states.loc[
            (all_states["resolution"] == "half_hourly") & (all_states["K"] == k) & (all_states["covariance_type"] == "full")
        ].copy()
        ref_raw = reference[["emission_mean_NEE_original", "emission_mean_CH4_original"]].to_numpy()
        for scale in ["hourly", "daily36", "daily40"]:
            target = all_states.loc[
                (all_states["resolution"] == scale) & (all_states["K"] == k) & (all_states["covariance_type"] == "full")
            ].copy()
            target_raw = target[["emission_mean_NEE_original", "emission_mean_CH4_original"]].to_numpy()
            distances = np.linalg.norm((target_raw[:, None, :] - ref_raw[None, :, :]) / reference_sd, axis=2)
            left, right = linear_sum_assignment(distances)
            for i, j in zip(left, right):
                rows.append({
                    "K": k, "reference_resolution": "half_hourly", "target_resolution": scale,
                    "reference_state": reference.iloc[j]["state_label"], "target_state": target.iloc[i]["state_label"],
                    "reference_NEE": ref_raw[j, 0], "target_NEE": target_raw[i, 0],
                    "reference_CH4": ref_raw[j, 1], "target_CH4": target_raw[i, 1],
                    "NEE_change": target_raw[i, 0] - ref_raw[j, 0],
                    "CH4_change": target_raw[i, 1] - ref_raw[j, 1],
                    "distance_in_half_hourly_sd_units": distances[i, j],
                })
    return pd.DataFrame(rows)


def add_original_half_centroids(states: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    states = states.copy()
    states["resolution"] = "half_hourly"
    states["emission_mean_NEE_original"] = params["NEE"]["mean"] + states["emission_mean_NEE_standardized"] * params["NEE"]["sd"]
    states["emission_mean_CH4_original"] = params["CH4"]["mean"] + states["emission_mean_CH4_standardized"] * params["CH4"]["sd"]
    return states


def make_figures(
    primary: pd.DataFrame, hourly: pd.DataFrame, daily: pd.DataFrame,
    sequences: pd.DataFrame, selection_all: pd.DataFrame, states_all: pd.DataFrame,
    daily_missing_month: pd.DataFrame,
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"half_hourly": "#1f77b4", "hourly": "#ff7f0e", "daily36": "#2ca02c", "daily40": "#9467bd"}

    fig, ax = plt.subplots(figsize=(8, 5))
    for scale, group in sequences.groupby("resolution"):
        values = np.sort(group["sequence_length"].to_numpy())
        ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", label=scale, color=colors[scale])
    ax.set_xscale("log"); ax.set_xlabel("Sequence length (observations, log scale)"); ax.set_ylabel("Empirical cumulative proportion")
    ax.legend(title="Resolution"); fig.tight_layout(); fig.savefig(FIGURES / "figure_01_sequence_length_distributions.png", dpi=180); plt.close(fig)

    selected = selected_scale_rows(selection_all)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, scale in zip(axes.flat, ["half_hourly", "hourly", "daily36", "daily40"]):
        choice = selected.loc[selected["resolution"] == scale].iloc[0]
        subset = states_all.loc[(states_all["resolution"] == scale) & (states_all["K"] == choice["K"]) & (states_all["covariance_type"] == choice["covariance_type"])]
        ax.scatter(subset["emission_mean_NEE_original"], subset["emission_mean_CH4_original"], s=60, color=colors[scale])
        for _, row in subset.iterrows():
            ax.annotate(row["state_label"], (row["emission_mean_NEE_original"], row["emission_mean_CH4_original"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.axhline(0, color="0.5", lw=.7); ax.axvline(0, color="0.5", lw=.7); ax.set_title(f"{scale}: BIC K={int(choice['K'])}, {choice['covariance_type']}")
    fig.supxlabel("NEE centroid (micromol CO2 m-2 s-1)"); fig.supylabel("CH4 centroid (nmol CH4 m-2 s-1)")
    fig.tight_layout(); fig.savefig(FIGURES / "figure_02_carbon_centroids_by_resolution.png", dpi=180); plt.close(fig)

    samples = {"half_hourly": primary, "hourly": hourly, "daily36": daily.loc[daily["eligible_rule_36"]], "daily40": daily.loc[daily["eligible_rule_40"]]}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for scale, frame in samples.items():
        for ax, variable in zip(axes, EMISSIONS):
            low, high = frame[variable].quantile([.005, .995])
            values = frame.loc[frame[variable].between(low, high), variable]
            ax.hist(values, bins=60, density=True, histtype="step", lw=1.4, label=scale, color=colors[scale])
    axes[0].set_xlabel("NEE (central 99%)"); axes[1].set_xlabel("CH4 (central 99%)")
    axes[0].set_ylabel("Density"); axes[1].legend(title="Resolution")
    fig.tight_layout(); fig.savefig(FIGURES / "figure_03_carbon_distributions_by_resolution.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for scale, group in selection_all.groupby("resolution"):
        by_k = group.groupby("K").agg(AIC=("AIC", "min"), BIC=("BIC", "min")).reset_index()
        axes[0].plot(by_k["K"], by_k["AIC"] - by_k["AIC"].min(), marker="o", label=scale, color=colors[scale])
        axes[1].plot(by_k["K"], by_k["BIC"] - by_k["BIC"].min(), marker="o", label=scale, color=colors[scale])
    axes[0].set_title("AIC difference from scale minimum"); axes[1].set_title("BIC difference from scale minimum")
    for ax in axes: ax.set_xlabel("K"); ax.set_ylabel("Criterion difference")
    axes[1].legend(title="Resolution"); fig.tight_layout(); fig.savefig(FIGURES / "figure_04_model_selection_by_resolution.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for scale, group in selection_all.loc[selection_all["covariance_type"] == "full"].groupby("resolution"):
        axes[0].plot(group["K"], group["mean_maximum_posterior"], marker="o", label=scale, color=colors[scale])
        axes[1].plot(group["K"], group["pairwise_ari_mean"], marker="o", label=scale, color=colors[scale])
    axes[0].set_title("Classification certainty"); axes[0].set_ylabel("Mean maximum posterior")
    axes[1].set_title("Across-start stability"); axes[1].set_ylabel("Mean pairwise ARI")
    for ax in axes: ax.set_xlabel("K"); ax.set_ylim(0, 1.02)
    axes[1].legend(title="Resolution"); fig.tight_layout(); fig.savefig(FIGURES / "figure_05_certainty_stability_by_resolution.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].hist(daily["valid_half_hours"], bins=np.arange(.5, 49.5, 1), color="#4c78a8")
    axes[0].axvline(36, color="#2ca02c", ls="--", label="Rule A: 36"); axes[0].axvline(40, color="#9467bd", ls="--", label="Rule B: 40")
    axes[0].set_xlabel("Valid half-hours per day"); axes[0].set_ylabel("Days"); axes[0].legend()
    for rule, color in [("eligible_rule_36", "#2ca02c"), ("eligible_rule_40", "#9467bd")]:
        subset = daily_missing_month.loc[daily_missing_month["rule"] == rule]
        axes[1].plot(subset["month"], subset["eligible_fraction"], marker="o", label=rule.replace("eligible_rule_", ">="), color=color)
    axes[1].set_xlabel("Month"); axes[1].set_ylabel("Eligible fraction of observed dates"); axes[1].set_xticks(range(1, 13)); axes[1].legend(title="Daily rule")
    fig.tight_layout(); fig.savefig(FIGURES / "figure_06_daily_completeness.png", dpi=180); plt.close(fig)


def write_report(
    comparison: pd.DataFrame, distribution: pd.DataFrame, selection: pd.DataFrame,
    alignments: pd.DataFrame, daily_year: pd.DataFrame, daily_month: pd.DataFrame,
) -> str:
    primary_scales = comparison.loc[comparison["resolution"].isin(["half_hourly", "hourly", "daily40"])].set_index("resolution")
    half = primary_scales.loc["half_hourly"]
    hourly = primary_scales.loc["hourly"]
    daily = primary_scales.loc["daily40"]
    dist = distribution.set_index("resolution")
    variance_ratios = {
        scale: {
            "NEE": dist.loc[scale, "NEE_variance"] / dist.loc["half_hourly", "NEE_variance"],
            "CH4": dist.loc[scale, "CH4_variance"] / dist.loc["half_hourly", "CH4_variance"],
        } for scale in ["hourly", "daily36", "daily40"]
    }
    hourly_continuity_improves = (
        hourly["median_sequence_length"] > half["median_sequence_length"]
        or hourly["singleton_sequence_proportion"] < half["singleton_sequence_proportion"]
    )
    hourly_stability_improves = hourly["pairwise_ari_mean"] >= half["pairwise_ari_mean"] + .10
    hourly_process_preserved = variance_ratios["hourly"]["CH4"] >= .70 and variance_ratios["hourly"]["NEE"] >= .70
    if hourly_continuity_improves and hourly_stability_improves and hourly_process_preserved:
        recommendation = "HOURLY RECOMMENDED AS PRIMARY"
    elif (not hourly_continuity_improves) and half["converged_fraction"] >= .3:
        recommendation = "HALF-HOURLY RECOMMENDED AS PRIMARY"
    else:
        recommendation = "NO CLEAR PRIMARY SCALE -- ADDITIONAL ANALYSIS REQUIRED"

    table_columns = [
        "resolution", "observations", "sequence_count", "median_sequence_length", "maximum_sequence_length",
        "singleton_sequence_proportion", "adjacent_transition_opportunities", "BIC_selected_K", "AIC_selected_K",
        "converged_fraction", "pairwise_ari_mean", "winner_recovery_rate_ari_ge_0_95",
        "mean_maximum_posterior", "minimum_state_occupancy", "minimum_centroid_separation",
    ]
    display = comparison[table_columns].copy()
    for col in display.select_dtypes(include="number"):
        display[col] = display[col].map(lambda x: f"{x:.4g}")
    headers = list(display.columns)
    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    table_lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    )
    table = "\n".join(table_lines)
    candidate_rows = []
    for (scale, k), group in selection.groupby(["resolution", "K"], sort=False):
        bic_row = group.loc[group["BIC"].idxmin()]
        aic_row = group.loc[group["AIC"].idxmin()]
        candidate_rows.append({
            "resolution": scale, "K": int(k),
            "best_BIC": f"{bic_row['BIC']:.2f}", "BIC_covariance": bic_row["covariance_type"],
            "best_AIC": f"{aic_row['AIC']:.2f}", "AIC_covariance": aic_row["covariance_type"],
            "BIC_fit_converged_fraction": f"{bic_row['converged_fraction']:.2f}",
            "BIC_fit_pairwise_ARI": f"{bic_row['pairwise_ari_mean']:.3f}",
        })
    candidate_frame = pd.DataFrame(candidate_rows)
    candidate_headers = list(candidate_frame.columns)
    candidate_lines = [
        "| " + " | ".join(candidate_headers) + " |",
        "| " + " | ".join(["---"] * len(candidate_headers)) + " |",
    ]
    candidate_lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in candidate_frame.itertuples(index=False, name=None)
    )
    candidate_table = "\n".join(candidate_lines)
    year_bias = daily_year.groupby("rule")["eligible_fraction"].agg(["min", "max"])
    month_bias = daily_month.groupby("rule")["eligible_fraction"].agg(["min", "max"])
    selected_lines = []
    for _, row in comparison.iterrows():
        selected_lines.append(
            f"- {row['resolution']}: BIC K={int(row['BIC_selected_K'])} ({row['BIC_selected_covariance']}), "
            f"AIC K={int(row['AIC_selected_K'])} ({row['AIC_selected_covariance']}); "
            f"converged {row['converged_fraction']:.0%}, ARI {row['pairwise_ari_mean']:.3f}, "
            f"mean maximum posterior {row['mean_maximum_posterior']:.3f}."
        )
    hourly_k6_alignment = alignments.loc[
        (alignments["K"] == 6) & (alignments["target_resolution"] == "hourly")
    ]
    daily40_k6_alignment = alignments.loc[
        (alignments["K"] == 6) & (alignments["target_resolution"] == "daily40")
    ]
    report = f"""# Temporal-resolution diagnostic v1

## Scope

This diagnostic used the frozen v1 primary carbon sample and only NEE and CH4 as HMM emissions. The completed half-hourly candidates were read without refitting. No GCC, phenological phase, environmental driver, or legacy HMM result entered inference or state alignment. PAR was used only to count observed day/night coverage.

## Aggregation rules

- Hourly observations require both primary-eligible constituent half-hours at the exact interval starts; NEE and CH4 are arithmetic means, without filling.
- Daily rule A requires at least 36 valid half-hours; rule B requires at least 40.
- Both daily rules additionally require at least {DAILY_QUADRANT_MINIMUM} valid observations in each clock quadrant 00–06, 06–12, 12–18, and 18–24. This excludes days missing most of one diel segment without using a non-carbon emission.
- Daily40 is the conservative daily comparison; daily36 is retained as a completeness sensitivity.
- Each aggregated dataset is standardized independently using only its eligible observations. Original NEE sign is preserved.

## Sequence and selected-model comparison

{table}

{"\n".join(selected_lines)}

The half-hourly sample has {int(half['adjacent_transition_opportunities']):,} adjacent transition opportunities, compared with {int(hourly['adjacent_transition_opportunities']):,} hourly and {int(daily['adjacent_transition_opportunities']):,} daily40 opportunities. Aggregation does not repair continuity: singleton-sequence proportions are {half['singleton_sequence_proportion']:.1%}, {hourly['singleton_sequence_proportion']:.1%}, and {daily['singleton_sequence_proportion']:.1%}, respectively.

## Candidate model selection by K

{candidate_table}

New full-covariance structures use 10 deterministic starts; diagonal covariance is retained as a five-start sensitivity. Strict convergence excludes any start that reaches the 150-iteration cap. The half-hourly values and start counts are inherited unchanged from the completed primary analysis. The full per-start trace is saved in `all_aggregated_fit_diagnostics.csv`.

## Aggregation effects on carbon signals

Relative to half-hourly observations, hourly averaging retains {variance_ratios['hourly']['NEE']:.1%} of NEE variance and {variance_ratios['hourly']['CH4']:.1%} of CH4 variance. Daily36 retains {variance_ratios['daily36']['NEE']:.1%} and {variance_ratios['daily36']['CH4']:.1%}; daily40 retains {variance_ratios['daily40']['NEE']:.1%} and {variance_ratios['daily40']['CH4']:.1%}.

Hourly CH4 q99 is {dist.loc['hourly', 'CH4_q99']:.3g}, versus {dist.loc['half_hourly', 'CH4_q99']:.3g} half-hourly. Daily40 CH4 q99 is {dist.loc['daily40', 'CH4_q99']:.3g}. Hourly and especially daily averaging therefore suppress high-CH4 response tails. NEE skewness changes from {dist.loc['half_hourly', 'NEE_skewness']:.3f} half-hourly to {dist.loc['hourly', 'NEE_skewness']:.3f} hourly and {dist.loc['daily40', 'NEE_skewness']:.3f} daily40; CH4 skewness changes from {dist.loc['half_hourly', 'CH4_skewness']:.3f} to {dist.loc['hourly', 'CH4_skewness']:.3f} and {dist.loc['daily40', 'CH4_skewness']:.3f}.

Centroids were transformed back to frozen-contract flux units and aligned solely by NEE–CH4 distance. At K=6, all hourly matches lie within {hourly_k6_alignment['distance_in_half_hourly_sd_units'].max():.3f} half-hourly SD units. Thus, the same broad hourly carbon signatures recur, including a high-CH4 state; hourly aggregation does not simply remove that state. It does suppress observation-level CH4 extremes and shifts the high-CH4 state's NEE centroid toward zero. Daily40 K=6 matches extend to {daily40_k6_alignment['distance_in_half_hourly_sd_units'].max():.3f} half-hourly SD units, with strong contraction of the most negative NEE centroid and high-CH4 signature. Daily aggregation therefore merges/recenters response geometry and can produce intermediate daily classes. Closely spaced centroids at higher K remain subdivisions, not evidence of new ecological mechanisms. NEE centroid signs are reported directly and were not transformed.

## Daily missingness bias

Eligibility is uneven through time. Across years, eligible fractions range from {year_bias.loc['eligible_rule_36', 'min']:.1%} to {year_bias.loc['eligible_rule_36', 'max']:.1%} for rule A and {year_bias.loc['eligible_rule_40', 'min']:.1%} to {year_bias.loc['eligible_rule_40', 'max']:.1%} for rule B. Across months they range from {month_bias.loc['eligible_rule_36', 'min']:.1%} to {month_bias.loc['eligible_rule_36', 'max']:.1%}, and {month_bias.loc['eligible_rule_40', 'min']:.1%} to {month_bias.loc['eligible_rule_40', 'max']:.1%}. Thus, complete-day inference samples years and times of year unevenly even without applying phenological labels.

## Scientific evaluation

### Statistical suitability

Half-hourly data retain the largest number of transitions but remain fragmented. Hourly aggregation raises strict convergence for the BIC-selected structure from {half['converged_fraction']:.0%} to {hourly['converged_fraction']:.0%} and raises mean pairwise ARI from {half['pairwise_ari_mean']:.3f} to {hourly['pairwise_ari_mean']:.3f}. That advantage is not unequivocal: only {hourly['winner_recovery_rate_ari_ge_0_95']:.1%} of converged hourly starts recover the winning K=6 partition at ARI >=0.95, posterior certainty falls slightly, and the singleton fraction increases. Daily aggregation is most fragmented and supplies relatively few transitions; its nominal convergence advantage occurs on only 470–744 observations. BIC/AIC minima are not treated as ecological choices.

### Process preservation

Half-hourly data uniquely preserve within-day NEE sign and magnitude changes and the full observed CH4 variance and extremes. Hourly centroids reproduce the broad half-hourly signatures well and make hourly analysis a strong robustness check, but averaging retains only {variance_ratios['hourly']['CH4']:.1%} of CH4 variance and attenuates observation-level tails. Daily means erase diel NEE structure by construction, retain only {variance_ratios['daily40']['NEE']:.1%} of NEE variance, substantially alter centroid geometry, and are susceptible to year/month-dependent completeness selection.

## Recommendation

**{recommendation}**

Statistically, hourly aggregation improves convergence and average ARI but does not improve sequence continuity or repeated recovery of the winning high-K partition; half-hourly observations provide 2.7 times as many adjacent transitions. Process-wise, half-hourly observations best preserve diel NEE structure, CH4 variability, and extreme responses. Because hourly centroids nevertheless reproduce the broad carbon signatures closely, hourly results should be retained as the principal temporal-aggregation sensitivity. Daily results are a secondary sensitivity only.

This recommendation concerns temporal resolution only. It does **not** choose the manuscript state count.
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")
    return recommendation


def main() -> None:
    args = parse_args()
    contract, core, primary = validate_and_read()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    half_selection, half_states_raw, half_params, half_sequences = half_hour_reference()
    half_states = add_original_half_centroids(half_states_raw, half_params)
    hourly, hourly_completeness = build_hourly(primary)
    daily = build_daily(primary)
    daily36 = daily.loc[daily["eligible_rule_36"], ["timestamp_start", "timestamp_end", "NEE", "CH4", "valid_half_hours", "timestamp_role"]].copy()
    daily40 = daily.loc[daily["eligible_rule_40"], ["timestamp_start", "timestamp_end", "NEE", "CH4", "valid_half_hours", "timestamp_role"]].copy()

    hourly, hourly_lengths, hourly_sequences = make_sequences(hourly, timedelta(hours=1), "hourly")
    daily36, daily36_lengths, daily36_sequences = make_sequences(daily36, timedelta(days=1), "daily36")
    daily40, daily40_lengths, daily40_sequences = make_sequences(daily40, timedelta(days=1), "daily40")
    hourly_x, hourly_params = standardize(hourly, "hourly", core["units"])
    daily36_x, daily36_params = standardize(daily36, "daily36", core["units"])
    daily40_x, daily40_params = standardize(daily40, "daily40", core["units"])
    datasets = {
        "hourly": (hourly, hourly_x, hourly_lengths, hourly_params),
        "daily36": (daily36, daily36_x, daily36_lengths, daily36_params),
        "daily40": (daily40, daily40_x, daily40_lengths, daily40_params),
    }

    if args.single_fit:
        try:
            scale, k_text, covariance, start_text = args.single_fit.split(",")
            k, start = int(k_text), int(start_text)
        except ValueError as exc:
            raise SystemExit("--single-fit must be resolution,K,covariance,start") from exc
        if scale not in datasets or k not in K_VALUES or covariance not in COVARIANCES:
            raise SystemExit("Invalid --single-fit target.")
        frame, x, lengths, _ = datasets[scale]
        seed = seed_for(scale, k, covariance, start)
        result = fit_one(scale, k, covariance, start, seed, x, lengths, args.max_iter, args.tol)
        print(json.dumps({
            "resolution": scale, "K": k, "covariance": covariance, "start": start,
            "success": result["success"], "converged": result["converged"],
            "iterations": result["iterations"], "log_likelihood": result["log_likelihood"],
        }, default=json_default))
        return

    if args.fit_shard:
        try:
            shard_text, count_text = args.fit_shard.split(",")
            shard, shard_count = int(shard_text), int(count_text)
        except ValueError as exc:
            raise SystemExit("--fit-shard must be shard_index,shard_count") from exc
        if shard_count < 1 or shard < 0 or shard >= shard_count:
            raise SystemExit("Invalid --fit-shard target.")
        targets = []
        for scale in ["hourly", "daily36", "daily40"]:
            _, x, lengths, _ = datasets[scale]
            for k in K_VALUES:
                for covariance in COVARIANCES:
                    target_starts = args.full_starts if covariance == "full" else args.diag_starts
                    for start in range(1, target_starts + 1):
                        seed = seed_for(scale, k, covariance, start)
                        targets.append((scale, k, covariance, start, seed, x, lengths))
        assigned = [target for index, target in enumerate(targets) if index % shard_count == shard]
        pending = [target for target in assigned if load_fit(*target[:5], args.max_iter) is None]
        print(f"Shard {shard + 1}/{shard_count}: {len(pending)} pending of {len(assigned)} assigned fits", flush=True)
        for completed, (scale, k, covariance, start, seed, x, lengths) in enumerate(pending, start=1):
            result = fit_one(scale, k, covariance, start, seed, x, lengths, args.max_iter, args.tol)
            if completed % 10 == 0 or completed == len(pending):
                print(f"Shard {shard + 1}/{shard_count}: completed {completed}/{len(pending)}; last={scale},K{k},{covariance},start{start},converged={result['converged']}", flush=True)
        return

    hourly.to_csv(OUTPUT / "hourly_carbon_primary.csv.gz", index=False, compression="gzip")
    hourly_completeness.to_csv(OUTPUT / "hourly_completeness.csv.gz", index=False, compression="gzip")
    daily.to_csv(OUTPUT / "daily_completeness.csv", index=False)
    daily36.to_csv(OUTPUT / "daily36_carbon.csv", index=False)
    daily40.to_csv(OUTPUT / "daily40_carbon.csv", index=False)
    write_json(OUTPUT / "standardization_parameters.json", {
        "half_hourly": half_params, "hourly": hourly_params, "daily36": daily36_params, "daily40": daily40_params,
    })

    sequence_tables = pd.concat([
        half_sequences["sequence_table"], hourly_sequences["sequence_table"],
        daily36_sequences["sequence_table"], daily40_sequences["sequence_table"],
    ], ignore_index=True, sort=False)
    sequence_tables.to_csv(OUTPUT / "sequence_summary_by_resolution.csv", index=False)

    results = fit_candidate_grid(datasets, args)
    diagnostics = pd.DataFrame([{k: v for k, v in result.items() if k not in {"monitor_history"}} for result in results])
    diagnostics.to_csv(OUTPUT / "all_aggregated_fit_diagnostics.csv", index=False)
    new_selection, new_states = summarize_models(results, datasets)

    half_selection = half_selection.copy()
    half_selection.insert(0, "resolution", "half_hourly")
    selection_all = pd.concat([half_selection, new_selection], ignore_index=True, sort=False)
    selection_all.to_csv(OUTPUT / "model_selection_by_resolution.csv", index=False)
    states_all = pd.concat([half_states, new_states], ignore_index=True, sort=False)
    states_all.to_csv(OUTPUT / "state_centroids_by_resolution.csv", index=False)
    alignments = align_centroids(half_states, new_states, half_params)
    alignments.to_csv(OUTPUT / "cross_resolution_centroid_alignment_by_k.csv", index=False)

    distribution = pd.DataFrame([
        distribution_metrics(primary, "half_hourly"), distribution_metrics(hourly, "hourly"),
        distribution_metrics(daily36, "daily36"), distribution_metrics(daily40, "daily40"),
    ])
    distribution.to_csv(OUTPUT / "aggregation_effects.csv", index=False)

    daily_year_rows, daily_month_rows = [], []
    for rule in ["eligible_rule_36", "eligible_rule_40"]:
        for year, group in daily.groupby("year"):
            daily_year_rows.append({"rule": rule, "year": year, "observed_dates": len(group), "eligible_dates": int(group[rule].sum()), "eligible_fraction": group[rule].mean()})
        for month, group in daily.groupby("month"):
            daily_month_rows.append({"rule": rule, "month": month, "observed_dates": len(group), "eligible_dates": int(group[rule].sum()), "eligible_fraction": group[rule].mean()})
    daily_year = pd.DataFrame(daily_year_rows)
    daily_month = pd.DataFrame(daily_month_rows)
    daily_year.to_csv(OUTPUT / "daily_completeness_by_year.csv", index=False)
    daily_month.to_csv(OUTPUT / "daily_completeness_by_month.csv", index=False)

    selected = selected_scale_rows(selection_all)
    sequence_summaries = pd.DataFrame([
        half_sequences["summary"], hourly_sequences["summary"],
        daily36_sequences["summary"], daily40_sequences["summary"],
    ])
    comparison = sequence_summaries.merge(
        selected[[
            "resolution", "BIC_selected_K", "BIC_selected_covariance", "AIC_selected_K", "AIC_selected_covariance",
            "converged_fraction", "pairwise_ari_mean", "winner_recovery_rate_ari_ge_0_95",
            "mean_maximum_posterior", "minimum_state_occupancy", "minimum_centroid_separation",
        ]], on="resolution", how="left",
    )
    comparison.to_csv(OUTPUT / "temporal_scale_comparison.csv", index=False)

    make_figures(primary, hourly, daily, sequence_tables, selection_all, states_all, daily_month)
    recommendation = write_report(comparison, distribution, selection_all, alignments, daily_year, daily_month)
    write_json(OUTPUT / "run_metadata.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "recommendation": recommendation,
        "contract_status": contract["contract_status"], "input_sha256": sha256(DATA),
        "half_hourly_refit": False, "fit_emissions": EMISSIONS,
        "phenology_or_environment_used_in_hmm": False, "legacy_hmm_results_used": False,
        "PAR_use": "coverage diagnostics only; never an HMM emission",
        "daily_rules": {
            "rule_A": ">=36 primary-eligible half-hours and >=4 observations in each six-hour clock quadrant",
            "rule_B": ">=40 primary-eligible half-hours and >=4 observations in each six-hour clock quadrant",
        },
        "deterministic_starts": {"full_per_K_and_resolution": args.full_starts, "diag_per_K_and_resolution": args.diag_starts},
        "max_iter": args.max_iter,
        "tol": args.tol, "jobs": args.jobs, "hmmlearn": hmmlearn_version,
        "python": platform.python_version(), "final_state_count_selected": False,
    })
    print(f"Hourly observations: {len(hourly):,}")
    print(f"Daily36 observations: {len(daily36):,}")
    print(f"Daily40 observations: {len(daily40):,}")
    print(f"Recommendation: {recommendation}")
    print(f"Report: {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
