#!/usr/bin/env python3
"""Audit and, only when fully supported, freeze the St Jones data contract.

This program is deliberately limited to metadata, schema, temporal, QC, and
source-to-canonical mapping checks. It never fits an HMM and never modifies the
raw input or legacy analysis artifacts.

Inputs are the registered raw table, author decisions, and versioned contract
templates. Outputs are atomic contract files, processed tables, audit reports,
and validation records. A failed integrity or regression check marks the
contract as failed and prevents partial processed tables from being retained.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "stjones_season.csv"
README_RTF = PROJECT_ROOT / "data" / "raw" / "Readme.rtf"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "data_contract"
PREVIEW_DIR = OUTPUT_DIR / "previews"
REPORT_PATH = PROJECT_ROOT / "reports" / "data_contract_audit_v1.md"
TARGETED_REPORT_PATH = PROJECT_ROOT / "reports" / "data_contract_targeted_resolution_v1.md"
FREEZE_REPORT_PATH = PROJECT_ROOT / "reports" / "data_contract_freeze_v1.md"
CONTRACT_PATH = PROJECT_ROOT / "config" / "data_contract_v1.json"
CORE_CONTRACT_PATH = PROJECT_ROOT / "config" / "data_contract_core_v1.json"
EXTENSION_CONTRACT_PATH = PROJECT_ROOT / "config" / "data_contract_phenology_environment_v1.json"
DECISIONS_PATH = PROJECT_ROOT / "config" / "data_contract_author_decisions.json"
TEST_PATH = PROJECT_ROOT / "tests" / "test_data_contract_v1.py"
HALFHOURLY_PATH = PROJECT_ROOT / "data" / "processed" / "stjones_halfhourly_contract_v1.csv.gz"
DAILY_PHENOLOGY_PATH = PROJECT_ROOT / "data" / "processed" / "stjones_daily_phenology_contract_v1.csv"
BAROMETRIC_PRESSURE_PATH = PROJECT_ROOT / "data" / "processed" / "stjones_barometric_pressure_v1.csv.gz"

EXPECTED_SHA256 = "49f16e69f4e26defac035cdd19bd4164f500a767102077e747755d906a49a73d"
EXPECTED_INTERVAL_MINUTES = 30
CONTRACT_VERSION = "1.0.0"
CREATION_DATE = "2026-08-04"
NA_TOKENS = ["NA", "", "-9999", "-9999.0", "NaN", "nan", "N/A"]
GAP_FILLED_RESPONSE_COLUMNS = {
    "NEE_f", "CH4_f", "CH4_f_RF", "FCH4_RF_filled", "FCH4_RF_model"
}

REQUIRED_SOURCE_COLUMNS = {
    "TIMESTAMP_START", "TIMESTAMP_END", "DATE_TIME", "DATE", "TIME",
    "NEE_orig", "CH4_orig", "qc_co2_flux", "qc_ch4_flux", "GCC", "Season",
}

NEE_CANDIDATES = [
    "co2_flux", "co2_strg", "NEE_orig", "NEE_f", "NEE_sum", "NEE_C_sum",
]
CH4_CANDIDATES = [
    "ch4_flux", "ch4_strg", "CH4_orig", "CH4_f", "CH4_f_RF",
    "FCH4_RF_filled", "FCH4_RF_model", "FCH4_RF_residual", "CH4_sum",
    "CH4_C_sum", "CH4_sum_rf", "CH4_C_sum_rf",
]

ENVIRONMENTAL_CANDIDATES = {
    "photosynthetically_active_radiation": ["PAR_MET", "PPFD_1_B1", "PPFD_1_B2"],
    "shortwave_radiation": ["Rg", "SWin_B1", "SWin_B2"],
    "vapor_pressure_deficit": ["VPD"],
    "air_temperature": ["Tair_MET", "Tair", "air_temperature", "sonic_temperature"],
    "soil_temperature": ["Tsoil_B1", "Tsoil_B2", "Tsoil"],
    "precipitation": ["Precip_MET"],
    "salinity": ["Sal_YSI", "Sal_YSI_f", "Sal_NOAA"],
    "recorded_water_level": ["Level_YSI", "Level_YSI_f", "Level_NOAA"],
    "wind_speed": ["WSpd_MET", "wind_speed", "max_wind_speed"],
    "turbulence": ["Ustar", "Tau", "TKE"],
}

ENVIRONMENTAL_RECOMMENDATIONS = {
    "photosynthetically_active_radiation": "PAR_MET",
    "shortwave_radiation": "Rg",
    "vapor_pressure_deficit": "VPD",
    "air_temperature": "Tair_MET",
    "soil_temperature": "Tsoil_B1",
    "precipitation": "Precip_MET",
    "salinity": "Sal_YSI",
    "recorded_water_level": "Level_YSI",
    "wind_speed": "WSpd_MET",
    "turbulence": "Ustar",
}

# These units are hypotheses inherited from legacy code, not authoritative
# documentation. They are retained to make the uncertainty explicit.
LEGACY_UNIT_HYPOTHESES = {
    "NEE_orig": "umol CO2 m^-2 s^-1",
    "NEE_f": "umol CO2 m^-2 s^-1",
    "co2_flux": "umol CO2 m^-2 s^-1",
    "co2_strg": "umol CO2 m^-2 s^-1",
    "CH4_orig": "nmol CH4 m^-2 s^-1",
    "CH4_f": "nmol CH4 m^-2 s^-1",
    "CH4_f_RF": "nmol CH4 m^-2 s^-1",
    "FCH4_RF_filled": "nmol CH4 m^-2 s^-1",
    "FCH4_RF_model": "nmol CH4 m^-2 s^-1",
    "FCH4_RF_residual": "nmol CH4 m^-2 s^-1",
    "ch4_flux": "umol CH4 m^-2 s^-1",
    "ch4_strg": "umol CH4 m^-2 s^-1",
    "PAR_MET": "umol photons m^-2 s^-1",
    "VPD": "hPa",
    "Tair_MET": "deg C",
    "Tair": "deg C",
    "air_temperature": "deg C",
    "sonic_temperature": "deg C",
    "Tsoil_B1": "deg C",
    "Tsoil_B2": "deg C",
    "Tsoil": "deg C",
    "Precip_MET": "mm per half-hour",
    "Sal_YSI": "PSU",
    "Sal_YSI_f": "PSU",
    "Sal_NOAA": "PSU",
    "Level_YSI": "m",
    "Level_YSI_f": "m",
    "Level_NOAA": "m",
    "WSpd_MET": "m s^-1",
    "wind_speed": "m s^-1",
    "Ustar": "m s^-1",
}

DOCUMENTED_DEFINITIONS = {
    "NEE_orig": "QA/QC'ed open-gap CO2 data",
    "NEE_f": "Gap-filled CO2 data using the MDS method",
    "CH4_orig": "QA/QC'ed open-gap CH4 data",
    "CH4_f": "Gap-filled CH4 data using the MDS method",
    "CH4_f_RF": "Only gaps filled with the random-forest method",
    "FCH4_RF_filled": "CH4_orig with gaps filled by the random-forest method; designated for budgets",
    "FCH4_RF_model": "Random-forest estimates for all half-hours",
    "FCH4_RF_residual": "Difference between measured and filled/model values",
    "Sal_YSI_f": "YSI salinity gap-filled using NOAA salinity and dissolved oxygen",
    "Level_YSI_f": "YSI water level gap-filled using NOAA water level",
}

PROHIBITED_HMM_COLUMNS = [
    "GCC", "Season", "PAR_MET", "VPD", "Tair_MET", "Tair", "Tsoil_B1",
    "Tsoil_B2", "Tsoil", "Precip_MET", "Sal_YSI", "Sal_YSI_f",
    "Level_YSI", "Level_YSI_f", "Sal_NOAA", "Level_NOAA", "Rg", "Ustar",
]

CORE_AUTHOR_DECISIONS = {
    "nee_unit",
    "ch4_unit",
    "ch4_sign_convention",
    "primary_qc_eligibility_rule",
    "sensitivity_qc_rule",
}

EXTENSION_AUTHOR_DECISIONS = {
    "gcc_definition",
    "gcc_source_and_daily_aggregation",
    "gcc_smoothing_method",
    "season_transition_method",
    "missing_gcc_season_assignment",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit", action="store_true", help="Run audit and write a draft or frozen contract")
    mode.add_argument("--freeze", action="store_true", help="Require a fully resolved contract and write official canonical tables")
    return parser.parse_args()


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
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if value is pd.NA:
        return None
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=False, default=json_default) + "\n")


def atomic_csv(frame: pd.DataFrame, path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".csv.gz" if path.name.endswith(".csv.gz") else ".csv"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=suffix, dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(tmp_name, index=False, **kwargs)
        os.replace(tmp_name, path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def decision_is_complete(entry: Any) -> bool:
    if not isinstance(entry, dict) or entry.get("approved") is not True:
        return False
    required = ["value", "evidence_source", "author", "decision_date", "confidence"]
    return all(entry.get(field) not in (None, "", {}) for field in required)


def load_author_decisions() -> tuple[dict[str, Any], list[str], list[str]]:
    if not DECISIONS_PATH.exists():
        return {}, sorted(CORE_AUTHOR_DECISIONS), sorted(EXTENSION_AUTHOR_DECISIONS)
    decisions = json.loads(DECISIONS_PATH.read_text(encoding="utf-8"))
    unresolved_core = [key for key in sorted(CORE_AUTHOR_DECISIONS) if not decision_is_complete(decisions.get(key))]
    unresolved_extension = [key for key in sorted(EXTENSION_AUTHOR_DECISIONS) if not decision_is_complete(decisions.get(key))]
    primary = decisions.get("primary_qc_eligibility_rule", {})
    sensitivity = decisions.get("sensitivity_qc_rule", {})
    if primary.get("approved") and primary.get("value") not in {"all_finite_observed", "qc_0_or_1", "qc_0_only"}:
        unresolved_core.append("primary_qc_eligibility_rule")
    allowed_sensitivity = {"qc_0_or_1", "qc_0_only"}
    sensitivity_value = sensitivity.get("value")
    valid_sensitivity = (
        isinstance(sensitivity_value, str) and sensitivity_value in allowed_sensitivity
    ) or (
        isinstance(sensitivity_value, list) and set(sensitivity_value) == allowed_sensitivity
    )
    if sensitivity.get("approved") and not valid_sensitivity:
        unresolved_core.append("sensitivity_qc_rule")
    return decisions, sorted(set(unresolved_core)), sorted(set(unresolved_extension))


def lexical_missing_tokens(path: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    targets = set(NA_TOKENS)
    with path.open("r", encoding="ascii", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            for value in row:
                if value in targets:
                    counts[value] += 1
    return {token: int(counts[token]) for token in NA_TOKENS if counts[token]}


def identify_role(column: str) -> str:
    lower = column.lower()
    if column in {"TIMESTAMP_START", "TIMESTAMP_END", "DATE_TIME", "DATE", "TIME"}:
        return "timestamp"
    if lower in {"x", "x.1", "x_all", "file_records", "used_records"}:
        return "identifier"
    if lower.startswith("qc_") or "test" in lower or "flag" in lower:
        return "QC flag"
    if column in {"NEE_orig", "CH4_orig", "co2_flux", "ch4_flux"}:
        return "observed flux"
    if column in GAP_FILLED_RESPONSE_COLUMNS or lower.endswith("_f") or "filled" in lower or "model" in lower:
        return "filled flux" if any(x in lower for x in ["nee", "ch4", "fch4"]) else "derived field"
    if any(x in lower for x in ["rand_err", "un_", "residual"]):
        return "uncertainty"
    if any(x in lower for x in ["sum", "_c_sum"]):
        return "cumulative quantity"
    if any(x in lower for x in ["level", "sal_", "ysi", "noaa", "swc"]):
        return "hydrology"
    if any(x in lower for x in ["ndvi", "pri", "gcc"]):
        return "remote sensing" if column != "GCC" else "phenology"
    if column == "Season":
        return "phenology"
    if any(x in lower for x in ["tair", "tsoil", "temperature", "vpd", "par", "precip", "wind", "ustar", "tau", "rg", "rn_"]):
        return "meteorology"
    if any(x in lower for x in ["strg", "scf", "mean", "var", "cov", "density", "ratio", "mole_fraction"]):
        return "derived field"
    return "unknown"


def parse_success(column: str, series: pd.Series) -> tuple[str, int, float]:
    nonmissing = int(series.notna().sum())
    if nonmissing == 0:
        return "all_missing", 0, 1.0
    if column in {"TIMESTAMP_START", "TIMESTAMP_END"}:
        parsed = pd.to_datetime(series.astype("Int64").astype("string"), format="%Y%m%d%H%M", errors="coerce")
        inferred = "timestamp_candidate"
    elif column in {"DATE_TIME", "DATE"}:
        parsed = pd.to_datetime(series, errors="coerce")
        inferred = "datetime" if column == "DATE_TIME" else "date"
    elif column == "TIME":
        parsed = pd.to_timedelta(series.astype("string"), errors="coerce")
        inferred = "time"
    elif pd.api.types.is_numeric_dtype(series):
        parsed = pd.to_numeric(series, errors="coerce")
        finite_values = parsed.dropna()
        inferred = "integer" if len(finite_values) and np.allclose(finite_values % 1, 0) else "floating_point"
    else:
        parsed = series.astype("string")
        inferred = "string"
    successes = int(parsed.notna().sum())
    return inferred, successes, successes / nonmissing


def series_examples(series: pd.Series, n: int = 3) -> str:
    values = series.dropna().drop_duplicates().head(n).tolist()
    return " | ".join(str(value) for value in values)


def semantic_digest(series: pd.Series) -> str:
    normalized = series.astype("string").fillna("<NA>")
    hashes = pd.util.hash_pandas_object(normalized, index=False).to_numpy(dtype="uint64")
    return hashlib.sha256(hashes.tobytes()).hexdigest()


def build_column_audit(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    signatures: defaultdict[str, list[str]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    for column in raw.columns:
        series = raw[column]
        inferred, successes, success_rate = parse_success(column, series)
        numeric = pd.to_numeric(series, errors="coerce") if pd.api.types.is_numeric_dtype(series) else None
        nonmissing = int(series.notna().sum())
        row: dict[str, Any] = {
            "original_column_name": column,
            "pandas_storage_type": str(series.dtype),
            "inferred_storage_type": inferred,
            "parse_success_count": successes,
            "parse_success_proportion_among_nonmissing": success_rate,
            "missing_count": int(series.isna().sum()),
            "missing_proportion": float(series.isna().mean()),
            "unique_nonmissing_values": int(series.nunique(dropna=True)),
            "numeric_minimum": float(numeric.min()) if numeric is not None and numeric.notna().any() else np.nan,
            "numeric_maximum": float(numeric.max()) if numeric is not None and numeric.notna().any() else np.nan,
            "example_values": series_examples(series),
            "is_constant_among_nonmissing": bool(series.nunique(dropna=True) <= 1),
            "is_all_missing": nonmissing == 0,
            "likely_role": identify_role(column),
            "role_evidence_level": "plausible inference from name; verify against metadata",
        }
        rows.append(row)
        signatures[semantic_digest(series)].append(column)

    duplicate_lookup: defaultdict[str, list[str]] = defaultdict(list)
    duplicate_rows: list[dict[str, Any]] = []
    for columns in signatures.values():
        if len(columns) < 2:
            continue
        for i, left in enumerate(columns):
            left_norm = raw[left].astype("string").fillna("<NA>")
            for right in columns[i + 1:]:
                right_norm = raw[right].astype("string").fillna("<NA>")
                if left_norm.equals(right_norm):
                    duplicate_lookup[left].append(right)
                    duplicate_lookup[right].append(left)
                    duplicate_rows.append({
                        "column_a": left,
                        "column_b": right,
                        "relationship_type": "exact_duplicate",
                        "n_rows_compared": len(raw),
                        "n_overlap_nonmissing": int((raw[left].notna() & raw[right].notna()).sum()),
                        "exact_equal_count": len(raw),
                        "approx_equal_count": len(raw),
                        "pearson_correlation": np.nan,
                        "maximum_absolute_difference": 0.0,
                        "notes": "Both columns are all missing" if raw[left].isna().all() else "Exact equality including missing-value positions",
                    })

    audited = pd.DataFrame(rows)
    audited["exact_duplicate_of"] = audited["original_column_name"].map(
        lambda value: " | ".join(sorted(duplicate_lookup.get(value, [])))
    )

    comparison_columns = sorted(set(
        NEE_CANDIDATES + CH4_CANDIDATES +
        [item for values in ENVIRONMENTAL_CANDIDATES.values() for item in values]
    ).intersection(raw.columns))
    for i, left in enumerate(comparison_columns):
        left_values = pd.to_numeric(raw[left], errors="coerce")
        for right in comparison_columns[i + 1:]:
            right_values = pd.to_numeric(raw[right], errors="coerce")
            valid = left_values.notna() & right_values.notna()
            n = int(valid.sum())
            if n < 100:
                continue
            corr = float(left_values[valid].corr(right_values[valid]))
            if math.isfinite(corr) and abs(corr) >= 0.995:
                diff = left_values[valid] - right_values[valid]
                duplicate_rows.append({
                    "column_a": left,
                    "column_b": right,
                    "relationship_type": "high_linear_agreement_candidate",
                    "n_rows_compared": len(raw),
                    "n_overlap_nonmissing": n,
                    "exact_equal_count": int(diff.eq(0).sum()),
                    "approx_equal_count": int(np.isclose(left_values[valid], right_values[valid], rtol=1e-7, atol=1e-8).sum()),
                    "pearson_correlation": corr,
                    "maximum_absolute_difference": float(diff.abs().max()),
                    "notes": "Candidate redundancy only; scale and provenance may differ",
                })
    duplicate_frame = pd.DataFrame(duplicate_rows).sort_values(
        ["relationship_type", "column_a", "column_b"], ignore_index=True
    )
    return audited, duplicate_frame, dict(duplicate_lookup)


EXPLICIT_TIMESTAMP_FORMATS = [
    "%Y%m%d%H%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%H:%M",
]


def strict_parse_count(series: pd.Series, format_string: str) -> int:
    count = 0
    for value in series.astype(str):
        try:
            datetime.strptime(value, format_string)
            count += 1
        except ValueError:
            pass
    return count


def normalize_scientific_timestamp(series: pd.Series) -> pd.Series:
    normalized: list[str | None] = []
    for value in series.astype(str):
        try:
            decimal = Decimal(value)
            normalized.append(str(int(decimal)) if decimal == decimal.to_integral_value() else None)
        except (InvalidOperation, ValueError, OverflowError):
            normalized.append(None)
    return pd.Series(normalized, index=series.index, dtype="string")


def parse_time(raw: pd.DataFrame, date_time_role: str) -> dict[str, pd.Series]:
    exported_start = pd.to_datetime(
        raw["TIMESTAMP_START"].astype("Int64").astype("string"),
        format="%Y%m%d%H%M", errors="coerce",
    )
    exported_end = pd.to_datetime(
        raw["TIMESTAMP_END"].astype("Int64").astype("string"),
        format="%Y%m%d%H%M", errors="coerce",
    )
    date_time = pd.to_datetime(
        raw["DATE_TIME"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    date_plus_time = pd.to_datetime(
        raw["DATE"].astype("string") + " " + raw["TIME"].astype("string"),
        format="%Y-%m-%d %H:%M", errors="coerce",
    )
    half = pd.Timedelta(minutes=EXPECTED_INTERVAL_MINUTES / 2)
    full = pd.Timedelta(minutes=EXPECTED_INTERVAL_MINUTES)
    if date_time_role == "interval_start":
        timestamp_start, timestamp_end = date_time, date_time + full
    elif date_time_role == "interval_midpoint":
        timestamp_start, timestamp_end = date_time - half, date_time + half
    else:
        timestamp_start, timestamp_end = date_time - full, date_time
    return {
        "exported_start": exported_start,
        "exported_end": exported_end,
        "date_time": date_time,
        "date_plus_time": date_plus_time,
        "timestamp_start": timestamp_start,
        "timestamp_end": timestamp_end,
    }


def timestamp_outputs(
    raw: pd.DataFrame, raw_text: pd.DataFrame, times: dict[str, pd.Series], date_time_role: str
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    date_time = times["date_time"]
    duplicate_mask = date_time.duplicated(keep=False) & date_time.notna()
    duplicate_rows: list[pd.DataFrame] = []
    for group_number, (key, group) in enumerate(raw.loc[duplicate_mask].groupby(date_time[duplicate_mask], sort=True), start=1):
        exact = all(group.iloc[0].equals(group.iloc[i]) for i in range(1, len(group)))
        conflicts = [column for column in raw.columns if group[column].nunique(dropna=False) > 1]
        piece = group.copy()
        piece.insert(0, "source_file_line", piece.index + 2)
        piece.insert(0, "source_row_number", piece.index + 1)
        piece.insert(0, "duplicate_group", group_number)
        piece.insert(1, "duplicate_timestamp", key.isoformat())
        piece.insert(2, "duplicate_occurrence", range(1, len(piece) + 1))
        piece.insert(3, "exact_duplicate_group", exact)
        piece.insert(4, "conflicting_nonkey_columns", " | ".join(conflicts))
        piece.insert(5, "recommended_action", "keep smallest source_row_number; retain all copies in this audit table")
        duplicate_rows.append(piece)
    duplicate_frame = pd.concat(duplicate_rows, ignore_index=True) if duplicate_rows else pd.DataFrame(
        columns=["duplicate_group", "duplicate_timestamp", "duplicate_occurrence", "exact_duplicate_group", "conflicting_nonkey_columns", "recommended_action", "source_row_number", "source_file_line"]
    )

    unique = pd.Series(date_time.dropna().drop_duplicates().sort_values().to_numpy())
    gaps: list[dict[str, Any]] = []
    diffs = unique.diff().dt.total_seconds().div(60)
    for index in diffs.index[diffs > EXPECTED_INTERVAL_MINUTES]:
        previous = unique.iloc[index - 1]
        following = unique.iloc[index]
        missing_count = int(diffs.iloc[index] / EXPECTED_INTERVAL_MINUTES) - 1
        missing_values = pd.date_range(previous + pd.Timedelta(minutes=30), periods=missing_count, freq="30min")
        gaps.append({
            "previous_date_time": previous.isoformat(),
            "next_date_time": following.isoformat(),
            "gap_minutes": float(diffs.iloc[index]),
            "missing_half_hours": missing_count,
            "missing_date_times": " | ".join(value.isoformat() for value in missing_values),
            "interpretation": "gap in DATE_TIME grid; sequence must split",
        })
    missing_frame = pd.DataFrame(gaps)

    daily = pd.DataFrame({"calendar_date": date_time.dt.date, "DATE_TIME": date_time})
    daily = daily.dropna(subset=["calendar_date"]).groupby("calendar_date", sort=True).agg(
        record_count=("DATE_TIME", "size"),
        unique_timestamp_count=("DATE_TIME", "nunique"),
    ).reset_index()
    daily["expected_record_count"] = 48
    daily["record_count_difference"] = daily["record_count"] - 48
    daily["possible_dst_artifact"] = daily["record_count"].isin([46, 50])
    daily["duplicate_day"] = daily["record_count"].gt(daily["unique_timestamp_count"])

    derived_lengths = (times["timestamp_end"] - times["timestamp_start"]).dt.total_seconds().div(60)
    date_matches = times["date_time"].eq(times["date_plus_time"])
    format_results = []
    field_profiles = []
    for field in ["TIMESTAMP_START", "TIMESTAMP_END", "DATE_TIME", "DATE", "TIME"]:
        values = raw_text[field]
        examples = values.drop_duplicates().head(8).tolist()
        lexical_type = (
            "scientific-notation numeric token"
            if values.str.fullmatch(r"[+-]?[0-9.]+[eE][+-]?[0-9]+").all()
            else "character date/time token"
        )
        field_profiles.append({
            "field": field,
            "pandas_inferred_storage_type": str(raw[field].dtype),
            "raw_lexical_storage_type": lexical_type,
            "nonmissing_count": int(values.ne("").sum()),
            "unique_raw_values": int(values.nunique()),
            "raw_examples": examples,
        })
        for format_string in EXPLICIT_TIMESTAMP_FORMATS:
            count = strict_parse_count(values, format_string)
            format_results.append({
                "field": field,
                "input_representation": "raw_lexeme",
                "explicit_format": format_string,
                "successful_rows": count,
                "parse_rate": count / len(values),
            })

    normalized_start = normalize_scientific_timestamp(raw_text["TIMESTAMP_START"])
    normalized_end = normalize_scientific_timestamp(raw_text["TIMESTAMP_END"])
    normalized_start_count = strict_parse_count(normalized_start.fillna(""), "%Y%m%d%H%M")
    normalized_end_count = strict_parse_count(normalized_end.fillna(""), "%Y%m%d%H%M")
    format_results.extend([
        {
            "field": "TIMESTAMP_START", "input_representation": "lossless_decimal_normalization",
            "explicit_format": "%Y%m%d%H%M", "successful_rows": normalized_start_count,
            "parse_rate": normalized_start_count / len(raw),
        },
        {
            "field": "TIMESTAMP_END", "input_representation": "lossless_decimal_normalization",
            "explicit_format": "%Y%m%d%H%M", "successful_rows": normalized_end_count,
            "parse_rate": normalized_end_count / len(raw),
        },
    ])

    def valid_month_prefix(value: Any) -> bool:
        if pd.isna(value) or len(str(value)) != 12:
            return False
        text = str(value)
        return text[:6].isdigit() and 1 <= int(text[4:6]) <= 12

    interpretable = normalized_start.map(valid_month_prefix) & normalized_end.map(valid_month_prefix)
    expected_start_month = times["date_time"].sub(pd.Timedelta(minutes=30)).dt.strftime("%Y%m")
    expected_end_month = times["date_time"].dt.strftime("%Y%m")
    start_month_match = normalized_start.str[:6].eq(expected_start_month)
    end_month_match = normalized_end.str[:6].eq(expected_end_month)
    raw_start_end_differ = raw_text["TIMESTAMP_START"].ne(raw_text["TIMESTAMP_END"])

    unique_times = date_time.dropna().drop_duplicates().sort_values()
    midnight_unique = unique_times.loc[unique_times.dt.hour.eq(0) & unique_times.dt.minute.eq(0)]
    unique_set = set(unique_times)
    midnight_with_previous = sum(
        value - pd.Timedelta(minutes=30) in unique_set for value in midnight_unique
    )
    audit = {
        "evidence_classification": {
            "documented_fact": "No supplied authoritative source defines timestamp role or timezone.",
            "empirically_verified": [
                f"DATE_TIME parses under %Y-%m-%d %H:%M:%S for {int(date_time.notna().sum())} of {len(raw)} rows.",
                f"DATE_TIME equals DATE plus TIME for {int(date_matches.sum())} rows.",
                "TIMESTAMP_START and TIMESTAMP_END parse for zero rows as raw or losslessly normalized %Y%m%d%H%M timestamps because day and time digits were destroyed.",
                f"For all {int(interpretable.sum())} rows retaining valid month prefixes, TIMESTAMP_START matches DATE_TIME minus 30 minutes at month grain and TIMESTAMP_END matches DATE_TIME at month grain.",
                f"All {int(raw_start_end_differ.sum())} surviving start/end month-boundary differences agree with a 30-minute interval ending at DATE_TIME.",
            ],
            "approved_structural_decision": "DATE_TIME is interval end; timestamp_start is DATE_TIME minus 30 minutes.",
            "unresolved_author_decision": "Timezone, local-standard-time, daylight-saving, or UTC convention remains undocumented.",
        },
        "raw_field_profiles": field_profiles,
        "explicit_format_parse_results": format_results,
        "date_time_role_used_for_audit": date_time_role,
        "date_time_role_status": "APPROVED_EMPIRICAL",
        "preferred_timestamp_source": {
            "timestamp_end": "DATE_TIME",
            "timestamp_start": "DATE_TIME - 30 minutes",
            "reason": "explicit TIMESTAMP_START/END are irreversibly precision-damaged; their surviving month prefixes confirm DATE_TIME is interval end",
        },
        "exported_timestamp_start_parse_success": int(times["exported_start"].notna().sum()),
        "exported_timestamp_end_parse_success": int(times["exported_end"].notna().sum()),
        "normalized_compact_timestamp_start_parse_success": normalized_start_count,
        "normalized_compact_timestamp_end_parse_success": normalized_end_count,
        "normalized_timestamp_start_examples": normalized_start.drop_duplicates().head(8).tolist(),
        "normalized_timestamp_end_examples": normalized_end.drop_duplicates().head(8).tolist(),
        "timestamp_start_end_raw_equal_rows": int((~raw_start_end_differ).sum()),
        "timestamp_start_end_raw_different_rows": int(raw_start_end_differ.sum()),
        "interpretable_month_prefix_rows": int(interpretable.sum()),
        "start_month_prefix_matches_date_time_minus_30_rows": int((interpretable & start_month_match).sum()),
        "end_month_prefix_matches_date_time_rows": int((interpretable & end_month_match).sum()),
        "month_boundary_difference_rows_consistent_with_interval_end": int((raw_start_end_differ & start_month_match & end_month_match).sum()),
        "date_time_parse_success": int(date_time.notna().sum()),
        "date_plus_time_parse_success": int(times["date_plus_time"].notna().sum()),
        "date_time_equals_date_plus_time_count": int(date_matches.sum()),
        "date_time_equals_valid_explicit_start_or_end": "not directly comparable because exported start/end timestamps are invalid",
        "explicit_start_end_interval_duration_rows": 0,
        "interval_duration_minutes": EXPECTED_INTERVAL_MINUTES,
        "interval_duration_evidence": "surviving month-boundary start/end prefixes plus modal unique DATE_TIME spacing",
        "derived_non_30_minute_interval_rows": int(derived_lengths.ne(30).sum()),
        "temporal_coverage_date_time": {
            "minimum": date_time.min(), "maximum": date_time.max(),
        },
        "timestamp_start_coverage": {
            "minimum": times["timestamp_start"].min(), "maximum": times["timestamp_start"].max(),
        },
        "timezone": None,
        "timezone_status": "UNRESOLVED",
        "original_order_non_monotonic_steps": int(date_time.diff().dt.total_seconds().lt(0).sum()),
        "duplicate_timestamp_affected_rows": int(duplicate_mask.sum()),
        "duplicate_timestamp_groups": int(date_time[duplicate_mask].nunique()),
        "all_duplicate_groups_exact_across_all_columns": bool(duplicate_frame["exact_duplicate_group"].all()) if len(duplicate_frame) else True,
        "missing_half_hours_within_observed_grid": int(missing_frame["missing_half_hours"].sum()) if len(missing_frame) else 0,
        "dates_with_less_than_48_rows": int(daily["record_count"].lt(48).sum()),
        "dates_with_more_than_48_rows": int(daily["record_count"].gt(48).sum()),
        "dates_with_46_or_50_rows": int(daily["possible_dst_artifact"].sum()),
        "intervals_crossing_midnight": int(times["timestamp_start"].dt.date.astype("string").ne(times["timestamp_end"].dt.date.astype("string")).sum()),
        "unique_midnight_date_time_rows": int(len(midnight_unique)),
        "unique_midnight_rows_with_preceding_23_30": int(midnight_with_previous),
        "midnight_date_matches_DATE_field_rows": int((date_time.dt.strftime("%Y-%m-%d").eq(raw_text["DATE"]) & date_time.dt.hour.eq(0) & date_time.dt.minute.eq(0)).sum()),
        "midnight_representation_conclusion": "Every observed unique 00:00 DATE_TIME has a preceding 23:30 record and DATE labels the interval end date; five year-boundary 00:00 records are absent and listed in missing_interval_summary.csv.",
        "dst_conclusion": "No recurring 46/50-record daily pattern is present, but timezone and DST treatment cannot be determined without documentation.",
        "timezone_documentation_search": {
            "result": "NO_AUTHORITATIVE_LOCAL_EVIDENCE_FOUND",
            "supporting_files": ["data/raw/Readme.rtf"],
            "note": "The supplied metadata are silent on timezone and DST treatment.",
        },
        "duplicate_handling_recommendation": "All 48 duplicate DATE_TIME groups are exact across all 284 columns. Keep the smallest source_row_number and preserve both copies in duplicate_timestamp_rows.csv.",
    }
    return audit, duplicate_frame, daily, missing_frame


def numeric_profile(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce")
    finite = values[np.isfinite(values)]
    result = {
        "nonmissing_count": int(series.notna().sum()),
        "finite_count": int(len(finite)),
        "missing_count": int(series.isna().sum()),
        "missing_proportion": float(series.isna().mean()),
        "minimum": np.nan, "q001": np.nan, "q01": np.nan, "median": np.nan,
        "q99": np.nan, "q999": np.nan, "maximum": np.nan,
        "tukey_outer_fence_count": 0,
    }
    if finite.empty:
        return result
    quantiles = finite.quantile([0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999])
    iqr = quantiles.loc[0.75] - quantiles.loc[0.25]
    lower, upper = quantiles.loc[0.25] - 3 * iqr, quantiles.loc[0.75] + 3 * iqr
    result.update({
        "minimum": float(finite.min()), "q001": float(quantiles.loc[0.001]),
        "q01": float(quantiles.loc[0.01]), "median": float(quantiles.loc[0.5]),
        "q99": float(quantiles.loc[0.99]), "q999": float(quantiles.loc[0.999]),
        "maximum": float(finite.max()),
        "tukey_outer_fence_count": int(((finite < lower) | (finite > upper)).sum()),
    })
    return result


def processing_class(column: str) -> str:
    if column in {"NEE_orig", "CH4_orig"}:
        return "documented QA/QC'ed open-gap observation"
    if column in {"co2_flux", "ch4_flux"}:
        return "turbulent flux component (legacy supporting interpretation)"
    if column in {"co2_strg", "ch4_strg"}:
        return "storage flux component (legacy supporting interpretation)"
    if column in GAP_FILLED_RESPONSE_COLUMNS or column in {"CH4_f"}:
        return "gap-filled or modeled"
    if "sum" in column.lower():
        return "cumulative/converted quantity"
    if "residual" in column.lower():
        return "derived residual"
    return "unresolved"


def candidate_comparison(raw: pd.DataFrame, times: dict[str, pd.Series], candidates: list[str], response: str, decisions: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for column in [value for value in candidates if value in raw.columns]:
        profile = numeric_profile(raw[column])
        finite_mask = np.isfinite(pd.to_numeric(raw[column], errors="coerce"))
        coverage = times["date_time"].where(finite_mask)
        unit_key = "nee_unit" if response == "NEE" else "ch4_unit"
        unit_decision = decisions.get(unit_key, {})
        if unit_decision.get("approved") is True and column in {"NEE_orig", "NEE_f", "CH4_orig", "CH4_f", "CH4_f_RF", "FCH4_RF_filled", "FCH4_RF_model", "FCH4_RF_residual"}:
            unit = unit_decision.get("value")
            unit_status = "author-approved"
        else:
            unit = LEGACY_UNIT_HYPOTHESES.get(column, "unresolved")
            unit_status = "legacy hypothesis; not authoritative"
        selected = column == ("NEE_orig" if response == "NEE" else "CH4_orig")
        source_status = "APPROVED source identity" if selected else "REJECTED as primary response"
        if selected and not unit_decision.get("approved"):
            suitability = "recommended observed/open-gap source; unit and/or sign metadata remain unresolved"
        elif selected:
            suitability = "approved observed/open-gap source"
        else:
            suitability = "context/sensitivity only; not the primary observed response"
        rows.append({
            "candidate": column,
            "documented_definition": DOCUMENTED_DEFINITIONS.get(column, "not documented in supplied authoritative metadata"),
            "unit": unit,
            "unit_evidence": unit_status,
            "sign_convention": "negative NEE = uptake; positive NEE = release (author instruction)" if response == "NEE" and column.startswith("NEE") else "unresolved",
            "processing_class": processing_class(column),
            "associated_qc_field": "qc_co2_flux" if response == "NEE" else "qc_ch4_flux",
            "qc_association_status": "plausible but not documented for the storage-adjusted open-gap product",
            "temporal_coverage_start": coverage.min(),
            "temporal_coverage_end": coverage.max(),
            **profile,
            "physically_extreme_value_interpretation": "Tukey outer-fence diagnostic only; no documented physical exclusion bound applied",
            "primary_response_source_status": source_status,
            "hmm_suitability": suitability,
        })
    return pd.DataFrame(rows)


def relationship_row(name: str, left: pd.Series, right: pd.Series, formula: str, support: str) -> dict[str, Any]:
    left_num = pd.to_numeric(left, errors="coerce")
    right_num = pd.to_numeric(right, errors="coerce")
    valid = np.isfinite(left_num) & np.isfinite(right_num)
    differences = left_num[valid] - right_num[valid]
    n = int(valid.sum())
    return {
        "check_name": name,
        "formula_tested": formula,
        "n_overlap": n,
        "exact_equal_count": int(differences.eq(0).sum()),
        "absolute_difference_le_1e_8_count": int(differences.abs().le(1e-8).sum()),
        "absolute_difference_le_1e_4_count": int(differences.abs().le(1e-4).sum()),
        "absolute_difference_le_1e_3_count": int(differences.abs().le(1e-3).sum()),
        "absolute_difference_gt_1e_3_count": int(differences.abs().gt(1e-3).sum()),
        "mean_absolute_difference": float(differences.abs().mean()) if n else np.nan,
        "median_absolute_difference": float(differences.abs().median()) if n else np.nan,
        "maximum_absolute_difference": float(differences.abs().max()) if n else np.nan,
        "pearson_correlation": float(left_num[valid].corr(right_num[valid])) if n > 1 else np.nan,
        "metadata_support": support,
        "interpretation": "Empirically verified numerical relationship; not authoritative metadata by itself",
    }


def relationship_checks(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    nee_rows = [relationship_row(
        "NEE_orig_vs_turbulent_plus_storage",
        raw["NEE_orig"], raw["co2_flux"] + raw["co2_strg"],
        "NEE_orig = co2_flux + co2_strg",
        "Readme documents storage calculations before post-processed fluxes but does not explicitly define NEE_orig as the sum",
    )]
    if "NEE_f" in raw:
        nee_rows.append(relationship_row(
            "NEE_f_equals_NEE_orig_where_observed", raw["NEE_f"], raw["NEE_orig"],
            "NEE_f = NEE_orig on observed overlap",
            "Readme defines NEE_f as the MDS gap-filled product",
        ))

    ch4_rows = [relationship_row(
        "CH4_orig_vs_1000_times_turbulent_plus_storage",
        raw["CH4_orig"], 1000.0 * (raw["ch4_flux"] + raw["ch4_strg"]),
        "CH4_orig = 1000 * (ch4_flux + ch4_strg)",
        "Readme documents storage calculations but does not document the 1000 scale factor or final CH4 unit",
    )]
    for candidate in ["CH4_f", "FCH4_RF_filled"]:
        if candidate in raw:
            relationship = relationship_row(
                f"{candidate}_equals_CH4_orig_where_observed", raw[candidate], raw["CH4_orig"],
                f"{candidate} = CH4_orig on observed overlap",
                DOCUMENTED_DEFINITIONS.get(candidate, "not documented"),
            )
            if candidate == "FCH4_RF_filled":
                left = pd.to_numeric(raw[candidate], errors="coerce")
                right = pd.to_numeric(raw["CH4_orig"], errors="coerce")
                differing = left.notna() & right.notna() & (left - right).abs().gt(1e-3)
                both_qc_missing = raw["qc_co2_flux"].isna() & raw["qc_ch4_flux"].isna()
                relationship["difference_qc_pattern"] = (
                    f"{int((differing & both_qc_missing).sum())} of {int(differing.sum())} differing rows have both qc_co2_flux and qc_ch4_flux missing"
                )
                relationship["interpretation"] = (
                    "Empirical conflict with the simplified Readme definition: the RF-filled product does not preserve every finite CH4_orig value"
                )
            ch4_rows.append(relationship)
    if {"FCH4_RF_model", "FCH4_RF_residual"}.issubset(raw.columns):
        ch4_rows.append(relationship_row(
            "CH4_orig_vs_RF_model_minus_residual", raw["CH4_orig"],
            raw["FCH4_RF_model"] - raw["FCH4_RF_residual"],
            "CH4_orig = FCH4_RF_model - FCH4_RF_residual",
            "Readme defines the model and residual but does not give a signed algebraic identity",
        ))
    return pd.DataFrame(nee_rows), pd.DataFrame(ch4_rows)


def qc_outputs(raw: pd.DataFrame, times: dict[str, pd.Series]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.Series]]:
    value_rows = []
    for column, response in [("qc_co2_flux", "NEE_orig"), ("qc_ch4_flux", "CH4_orig")]:
        counts = raw[column].value_counts(dropna=False)
        for value, count in counts.items():
            value_rows.append({
                "qc_field": column,
                "associated_response_candidate": response,
                "qc_value": "MISSING" if pd.isna(value) else int(value),
                "count": int(count),
                "proportion": float(count / len(raw)),
                "documented_class_meaning": "not documented in supplied authoritative metadata",
                "association_status": "candidate associated with turbulent flux; exact applicability to the storage-adjusted open-gap response is unresolved",
                "local_metadata_search_result": "No local authoritative definition found for values 0, 1, 2, or missing",
            })
    value_summary = pd.DataFrame(value_rows)

    timestamp_valid = times["date_time"].notna()
    later_duplicate = times["date_time"].duplicated(keep="first") & timestamp_valid
    base = timestamp_valid & ~later_duplicate
    nee_finite = np.isfinite(pd.to_numeric(raw["NEE_orig"], errors="coerce"))
    ch4_finite = np.isfinite(pd.to_numeric(raw["CH4_orig"], errors="coerce"))
    paired = base & nee_finite & ch4_finite
    masks = {
        "all_finite_observed": paired,
        "qc_0_only": paired & raw["qc_co2_flux"].eq(0) & raw["qc_ch4_flux"].eq(0),
        "qc_0_or_1": paired & raw["qc_co2_flux"].isin([0, 1]) & raw["qc_ch4_flux"].isin([0, 1]),
    }
    descriptions = {
        "all_finite_observed": "All finite documented QA/QC'ed open-gap NEE_orig and CH4_orig values after deterministic duplicate handling",
        "qc_0_only": "Both available turbulent-flux QC fields equal 0",
        "qc_0_or_1": "Both available turbulent-flux QC fields are 0 or 1 (legacy project rule)",
    }
    statuses = {
        "all_finite_observed": "COMPARISON_ONLY",
        "qc_0_only": "CONDITIONAL_SENSITIVITY_RECOMMENDATION_PENDING_QC_METADATA",
        "qc_0_or_1": "CONDITIONAL_PRIMARY_RECOMMENDATION_PENDING_QC_METADATA",
    }
    rule_rows = []
    for rule, mask in masks.items():
        retained_times = times["timestamp_start"].loc[mask].sort_values().reset_index(drop=True)
        if retained_times.empty:
            sequence_lengths = pd.Series(dtype="int64")
        else:
            new_sequence = retained_times.diff().ne(pd.Timedelta(minutes=EXPECTED_INTERVAL_MINUTES))
            new_sequence.iloc[0] = True
            sequence_lengths = retained_times.groupby(new_sequence.cumsum()).size()
        rule_rows.append({
            "rule_id": rule,
            "description": descriptions[rule],
            "eligible_rows": int(mask.sum()),
            "proportion_of_raw": float(mask.mean()),
            "proportion_of_unique_valid_timestamp_rows": float(mask.sum() / base.sum()),
            "rows_lost_vs_all_finite_observed": int(masks["all_finite_observed"].sum() - mask.sum()),
            "retained_duration_hours": float(mask.sum() / 2),
            "sequence_count": int(len(sequence_lengths)),
            "sequence_break_count": max(int(len(sequence_lengths)) - 1, 0),
            "singleton_sequence_count": int(sequence_lengths.eq(1).sum()),
            "median_sequence_length_half_hours": float(sequence_lengths.median()) if len(sequence_lengths) else np.nan,
            "mean_sequence_length_half_hours": float(sequence_lengths.mean()) if len(sequence_lengths) else np.nan,
            "maximum_sequence_length_half_hours": int(sequence_lengths.max()) if len(sequence_lengths) else 0,
            "p90_sequence_length_half_hours": float(sequence_lengths.quantile(0.9)) if len(sequence_lengths) else np.nan,
            "status": statuses[rule],
            "evidence": "Readme defines NEE_orig and CH4_orig as QA/QC'ed open-gap products; local metadata and legacy files do not define QC classes or prove qc_* applicability after storage correction",
            "qc_field_correspondence": "qc_co2_flux and qc_ch4_flux are the only plausible local candidates, but exact final-product correspondence is not documented",
        })
    rule_comparison = pd.DataFrame(rule_rows)

    waterfall_rows: list[dict[str, Any]] = []
    for rule, final_mask in masks.items():
        remaining = pd.Series(True, index=raw.index)
        ordered_steps = [
            ("invalid_timestamp", ~timestamp_valid),
            ("duplicate_timestamp_later_copy", later_duplicate),
            ("missing_NEE", raw["NEE_orig"].isna()),
            ("nonfinite_NEE", raw["NEE_orig"].notna() & ~nee_finite),
            ("missing_CH4", raw["CH4_orig"].isna()),
            ("nonfinite_CH4", raw["CH4_orig"].notna() & ~ch4_finite),
        ]
        if rule in {"qc_0_only", "qc_0_or_1"}:
            allowed = [0] if rule == "qc_0_only" else [0, 1]
            ordered_steps.extend([
                ("NEE_QC", ~raw["qc_co2_flux"].isin(allowed)),
                ("CH4_QC", ~raw["qc_ch4_flux"].isin(allowed)),
            ])
        ordered_steps.extend([
            ("non_30_minute_interval", pd.Series(False, index=raw.index)),
            ("documented_impossible_or_invalid_value", pd.Series(False, index=raw.index)),
            ("other_rule", pd.Series(False, index=raw.index)),
        ])
        for order, (reason, reason_mask) in enumerate(ordered_steps, start=1):
            excluded_now = remaining & reason_mask
            remaining &= ~reason_mask
            waterfall_rows.append({
                "rule_id": rule, "step_order": order, "exclusion_reason": reason,
                "excluded_at_step": int(excluded_now.sum()),
                "reason_count_nonexclusive": int(reason_mask.sum()),
                "remaining_after_step": int(remaining.sum()),
            })
        waterfall_rows.append({
            "rule_id": rule, "step_order": len(ordered_steps) + 1,
            "exclusion_reason": "ELIGIBLE_FINAL", "excluded_at_step": 0,
            "reason_count_nonexclusive": int(final_mask.sum()),
            "remaining_after_step": int(final_mask.sum()),
        })
    waterfall = pd.DataFrame(waterfall_rows)
    return value_summary, rule_comparison, waterfall, masks


def environment_outputs(raw: pd.DataFrame, times: dict[str, pd.Series]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for construct, candidates in ENVIRONMENTAL_CANDIDATES.items():
        present = [value for value in candidates if value in raw.columns]
        recommended = ENVIRONMENTAL_RECOMMENDATIONS[construct]
        recommended_values = pd.to_numeric(raw[recommended], errors="coerce") if recommended in raw else None
        for column in present:
            values = pd.to_numeric(raw[column], errors="coerce")
            finite = np.isfinite(values)
            coverage = times["date_time"].where(finite)
            if recommended_values is not None and column != recommended:
                valid = np.isfinite(values) & np.isfinite(recommended_values)
                n_overlap = int(valid.sum())
                correlation = float(values[valid].corr(recommended_values[valid])) if n_overlap > 1 else np.nan
                median_abs_diff = float((values[valid] - recommended_values[valid]).abs().median()) if n_overlap else np.nan
            else:
                n_overlap, correlation, median_abs_diff = int(finite.sum()), 1.0, 0.0
            if column.endswith("_f"):
                status = "filled"
            elif "NOAA" in column:
                status = "measured off-site/reference sensor"
            elif any(token in column for token in ["YSI", "MET", "B1", "B2"]):
                status = "measured site/sensor product"
            else:
                status = "derived or source status unresolved"
            source = "YSI at site" if "YSI" in column else "NOAA Scotton Landing" if "NOAA" in column else "site meteorology" if "MET" in column else "EC/biomet"
            mapping_status = "PROVISIONAL" if column == recommended else "ALTERNATIVE"
            if construct in {"soil_temperature", "recorded_water_level"} and column == recommended:
                mapping_status = "NEEDS_AUTHOR_DECISION"
            rows.append({
                "construct": construct,
                "candidate_column": column,
                "recommended_canonical_source": column == recommended,
                "mapping_status": mapping_status,
                "measured_or_filled_status": status,
                "sensor_source": source,
                "documented_definition_or_fill": DOCUMENTED_DEFINITIONS.get(column, "not documented; name-based interpretation only"),
                "unit": LEGACY_UNIT_HYPOTHESES.get(column, "unresolved"),
                "unit_evidence": "legacy code hypothesis; not authoritative",
                "nonmissing_count": int(raw[column].notna().sum()),
                "missing_proportion": float(raw[column].isna().mean()),
                "coverage_start": coverage.min(), "coverage_end": coverage.max(),
                "comparison_to_recommended_overlap": n_overlap,
                "comparison_to_recommended_pearson_r": correlation,
                "comparison_to_recommended_median_absolute_difference": median_abs_diff,
                "recommendation_rationale": "prefer the direct measured site series; keep filled/off-site alternatives for explicit downstream sensitivity" if column == recommended else "not selected as the single canonical source",
                "unresolved_metadata": (
                    "vertical datum, marsh-surface reference, inundation threshold, and units" if construct == "recorded_water_level" else
                    "sensor depth/location" if construct == "soil_temperature" else
                    "unit and sensor-processing details; salinity fill relationship documented but coefficients are not" if construct == "salinity" else
                    "unit and sensor-processing details"
                ),
            })
    return pd.DataFrame(rows)


def phenology_outputs(raw: pd.DataFrame, times: dict[str, pd.Series]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    phase_map = {
        "Spring": "Greenup", "Summer": "Maturity",
        "Autumn": "Senescence", "Winter": "Dormancy",
    }
    date_values = pd.to_datetime(raw["DATE"], errors="coerce").dt.normalize()
    work = pd.DataFrame({"date": date_values, "GCC": raw["GCC"], "Season": raw["Season"]})
    daily_rows = []
    for date, group in work.dropna(subset=["date"]).groupby("date", sort=True):
        gcc_values = group["GCC"].dropna().unique()
        season_values = group["Season"].dropna().astype(str).unique()
        daily_rows.append({
            "date": date,
            "GCC": float(gcc_values[0]) if len(gcc_values) == 1 else np.nan,
            "Season": season_values[0] if len(season_values) == 1 else pd.NA,
            "GCC_available": len(gcc_values) > 0,
            "source_record_count": len(group),
            "GCC_nonmissing_record_count": int(group["GCC"].notna().sum()),
            "Season_nonmissing_record_count": int(group["Season"].notna().sum()),
            "GCC_unique_nonmissing_values": len(gcc_values),
            "Season_unique_nonmissing_values": len(season_values),
            "GCC_within_date_consistent": len(gcc_values) <= 1,
            "Season_within_date_consistent": len(season_values) <= 1,
            "Season_populated_when_GCC_missing": len(gcc_values) == 0 and len(season_values) == 1,
            "interpolation_or_assignment_flag": "UNRESOLVED" if len(gcc_values) == 0 and len(season_values) == 1 else "not documented",
        })
    daily = pd.DataFrame(daily_rows).sort_values("date", ignore_index=True)
    unexpected = sorted(set(daily["Season"].dropna().astype(str)) - set(phase_map))
    if unexpected:
        raise RuntimeError(f"Unexpected raw Season labels; refusing silent recode: {unexpected}")
    daily["phenological_phase_source"] = daily["Season"].astype("string")
    daily["phenological_phase"] = daily["phenological_phase_source"].map(phase_map).astype("string")
    daily["phenological_phase_filled"] = (
        daily["Season_populated_when_GCC_missing"] & daily["phenological_phase"].notna()
    )

    boundaries: list[dict[str, Any]] = []
    daily_season = daily.dropna(subset=["Season"]).copy()
    daily_season["year"] = daily_season["date"].dt.year
    for year, year_group in daily_season.groupby("year", sort=True):
        year_group = year_group.sort_values("date").copy()
        new_segment = year_group["Season"].ne(year_group["Season"].shift()) | year_group["date"].diff().dt.days.ne(1)
        year_group["segment_id"] = new_segment.cumsum()
        for segment_order, (_, segment) in enumerate(year_group.groupby("segment_id", sort=True), start=1):
            boundaries.append({
                "year": int(year), "segment_order": segment_order,
                "Season": segment["Season"].iloc[0],
                "phenological_phase": phase_map[segment["Season"].iloc[0]],
                "start_date": segment["date"].min(), "end_date": segment["date"].max(),
                "duration_days_in_year_segment": int(len(segment)),
                "previous_season": year_group.loc[year_group["date"].lt(segment["date"].min()), "Season"].iloc[-1] if (year_group["date"] < segment["date"].min()).any() else pd.NA,
            })
    boundary_frame = pd.DataFrame(boundaries)
    daily["phase_interval_start"] = pd.NaT
    daily["phase_interval_end"] = pd.NaT
    daily["transition_dates_known"] = False
    for boundary in boundary_frame.itertuples(index=False):
        in_interval = (
            daily["date"].dt.year.eq(boundary.year)
            & daily["phenological_phase"].eq(boundary.phenological_phase)
            & daily["date"].between(boundary.start_date, boundary.end_date)
        )
        daily.loc[in_interval, "phase_interval_start"] = boundary.start_date
        daily.loc[in_interval, "phase_interval_end"] = boundary.end_date
        daily.loc[in_interval, "transition_dates_known"] = True
    daily["interpolation_or_assignment_flag"] = np.where(
        daily["phenological_phase_filled"],
        "approved_manual_categorical_phase_within_known_interval",
        "source_phase_label",
    )

    valid_categories = sorted(raw["Season"].dropna().astype(str).unique().tolist())
    gcc_by_date = daily.loc[daily["GCC_available"], "GCC"]
    transition_rows = boundary_frame.loc[boundary_frame["segment_order"].gt(1), ["year", "previous_season", "Season", "start_date"]]
    audit = {
        "documented_definition": "G / (R + G + B); author approved",
        "gcc_unit_or_dimensionless_definition": "dimensionless; author approved",
        "observed_gcc_range": {"minimum": float(gcc_by_date.min()), "maximum": float(gcc_by_date.max())},
        "finite_gcc_halfhour_rows": int(raw["GCC"].notna().sum()),
        "dates_in_source": int(len(daily)),
        "dates_with_gcc": int(daily["GCC_available"].sum()),
        "dates_missing_gcc": int((~daily["GCC_available"]).sum()),
        "dates_with_conflicting_gcc": int((~daily["GCC_within_date_consistent"]).sum()),
        "dates_with_conflicting_season": int((~daily["Season_within_date_consistent"]).sum()),
        "dates_with_season_but_no_gcc": int(daily["Season_populated_when_GCC_missing"].sum()),
        "gcc_temporal_resolution": "daily value repeated over half-hourly records (empirically verified)",
        "season_temporal_resolution": "daily category repeated over half-hourly records (empirically verified)",
        "valid_season_categories": valid_categories,
        "intended_order": ["Greenup", "Maturity", "Senescence", "Dormancy"],
        "order_matches_dataset": set(valid_categories) == {"Winter", "Spring", "Summer", "Autumn"},
        "gcc_coverage_start": daily.loc[daily["GCC_available"], "date"].min(),
        "gcc_coverage_end": daily.loc[daily["GCC_available"], "date"].max(),
        "season_coverage_start": daily.loc[daily["Season"].notna(), "date"].min(),
        "season_coverage_end": daily.loc[daily["Season"].notna(), "date"].max(),
        "season_boundary_transitions": transition_rows.to_dict(orient="records"),
        "season_derivation_method": "phenopix greenExplore annual curve fitting with gu transition dates; author approved",
        "gcc_smoothing_or_transition_detection_method": "phenopix autoFilter and spline filtering; author approved",
        "missing_gcc_interpolation_modeling_or_manual_assignment": "GCC not manually reconstructed; categorical phase filled only within known bounded annual intervals; author approved",
        "canonical_phase_mapping": phase_map,
        "days_with_manually_filled_phase": int(daily["phenological_phase_filled"].sum()),
        "future_analysis_scale": {
            "HMM": "half-hourly",
            "phenology_comparison": "daily state occupancy",
            "replication_rule": "Do not treat repeated daily GCC/Season as independent half-hourly phenology observations",
        },
    }
    return daily, boundary_frame, audit


def mapping_table(
    decisions: dict[str, Any], unresolved_core: list[str], unresolved_extension: list[str]
) -> pd.DataFrame:
    def decision_value(key: str, fallback: str) -> str:
        entry = decisions.get(key, {})
        return str(entry.get("value", fallback)) if entry.get("approved") else fallback

    date_role = decision_value("timestamp_interval_role", "interval_end")
    nee_unit = decision_value("nee_unit", LEGACY_UNIT_HYPOTHESES["NEE_orig"])
    ch4_unit = decision_value("ch4_unit", LEGACY_UNIT_HYPOTHESES["CH4_orig"])
    qc_rule = decision_value("primary_qc_eligibility_rule", "qc_0_or_1")
    rows = [
        ["timestamp_start", "DATE_TIME", "timezone unresolved", f"derive from DATE_TIME interpreted as {date_role}", "parseable; deterministic duplicate rule", "half-hourly", "source timestamp", "explicit compact fields are precision-damaged; all 87,691 interpretable month prefixes match DATE_TIME - 30 minutes", "APPROVED", "empirically verified interval start", "timezone/DST convention"],
        ["timestamp_end", "DATE_TIME", "timezone unresolved", f"identity because DATE_TIME is {date_role}", "parseable; deterministic duplicate rule", "half-hourly", "source timestamp", "all interpretable TIMESTAMP_END month prefixes and all 56 month-boundary differences match DATE_TIME", "APPROVED", "empirically verified interval end", "timezone/DST convention"],
        ["date", "DATE", "calendar date", "parse YYYY-MM-DD", "parseable", "daily label repeated half-hourly", "source field", "DATE exactly agrees with DATE_TIME calendar date", "APPROVED", "phenology grouping key", "whether midnight interval belongs to prior start-date remains tied to timestamp role"],
        ["nee", "NEE_orig", nee_unit, "identity; preserve sign", qc_rule, "half-hourly", "documented QA/QC'ed open-gap", "Readme definition plus empirical agreement with turbulent+storage; source column approved", "NEEDS_AUTHOR_DECISION" if "nee_unit" in unresolved_core else "APPROVED", "approved observed non-gap-filled NEE source", "authoritative unit; storage-correction wording"],
        ["nee_qc", "qc_co2_flux", "categorical flag", "identity", "sensitivity flags {0,1} and {0}", "half-hourly", "turbulent-flux QC candidate", "class values observed; exact meanings and final-product applicability unresolved", "SENSITIVITY_ONLY", "only plausible local CO2 QC field", "do not interpret as a quality ranking"],
        ["ch4", "CH4_orig", ch4_unit, "identity", qc_rule, "half-hourly", "documented QA/QC'ed open-gap", "Readme definition plus empirical agreement with 1000*(turbulent+storage); source column approved", "NEEDS_AUTHOR_DECISION" if {"ch4_unit", "ch4_sign_convention"}.intersection(unresolved_core) else "APPROVED", "approved observed non-gap-filled CH4 source", "authoritative unit, sign, and storage-correction wording"],
        ["ch4_qc", "qc_ch4_flux", "categorical flag", "identity", "sensitivity flags {0,1} and {0}", "half-hourly", "turbulent-flux QC candidate", "class values observed; exact meanings and final-product applicability unresolved", "SENSITIVITY_ONLY", "only plausible local CH4 QC field", "do not interpret as a quality ranking"],
        ["gcc", "GCC", "dimensionless definition unresolved", "daily unique value; never an HMM emission", "valid only when finite and within-date consistent", "daily repeated half-hourly", "derived phenology", "within-date constancy empirically verified", "PROVISIONAL", "independent phenology information", "definition and smoothing/interpolation provenance"],
        ["phenological_season", "Season", "categorical", "preserve dataset spelling/order; never an HMM emission", "unique within date", "daily repeated half-hourly", "GCC-derived status asserted by project request; method undocumented", "complete and within-date constant empirically", "NEEDS_AUTHOR_DECISION" if unresolved_extension else "APPROVED", "independent post-HMM comparison", "derivation, transition method, assignment where GCC missing"],
        ["par", "PAR_MET", LEGACY_UNIT_HYPOTHESES["PAR_MET"], "identity", "finite when used downstream", "half-hourly", "site meteorology", "legacy mapping; metadata incomplete", "PROVISIONAL", "direct PAR candidate", "authoritative unit and sensor details"],
        ["vpd", "VPD", LEGACY_UNIT_HYPOTHESES["VPD"], "identity", "finite when used downstream", "half-hourly", "derived meteorology", "legacy mapping; metadata incomplete", "PROVISIONAL", "only VPD candidate", "authoritative unit and derivation"],
        ["air_temperature", "Tair_MET", LEGACY_UNIT_HYPOTHESES["Tair_MET"], "identity", "finite when used downstream", "half-hourly", "site meteorology", "legacy mapping; metadata incomplete", "PROVISIONAL", "direct measured MET candidate", "authoritative unit/sensor details"],
        ["soil_temperature", "Tsoil_B1", LEGACY_UNIT_HYPOTHESES["Tsoil_B1"], "identity", "finite when used downstream", "half-hourly", "site biomet sensor", "multiple candidates with unresolved depth/location", "NEEDS_AUTHOR_DECISION", "measured candidate only", "select sensor and document depth/location"],
        ["precipitation", "Precip_MET", LEGACY_UNIT_HYPOTHESES["Precip_MET"], "identity", "finite when used downstream", "half-hourly", "site meteorology", "legacy mapping; metadata incomplete", "PROVISIONAL", "only precipitation candidate", "authoritative interval accumulation unit"],
        ["salinity", "Sal_YSI", LEGACY_UNIT_HYPOTHESES["Sal_YSI"], "identity; no implicit fallback", "finite measured values only", "half-hourly", "measured YSI", "Readme documents filled counterpart construction", "PROVISIONAL", "measured site series avoids implicit filling", "authoritative units and sensor details"],
        ["recorded_water_level", "Level_YSI", LEGACY_UNIT_HYPOTHESES["Level_YSI"], "identity; no implicit fallback", "finite measured values only", "half-hourly", "measured YSI", "Readme documents filled counterpart construction", "NEEDS_AUTHOR_DECISION", "measured site series; do not call inundation depth", "unit, datum, marsh elevation, sign, threshold"],
    ]
    frame = pd.DataFrame(rows, columns=[
        "canonical_name", "selected_source_column", "unit", "transformation",
        "eligibility_rule", "temporal_resolution", "measured_or_filled_status",
        "evidence", "status", "rationale", "unresolved_issue",
    ])
    frame["source_column_status"] = frame["canonical_name"].map({
        "timestamp_start": "APPROVED", "timestamp_end": "APPROVED", "date": "APPROVED",
        "nee": "APPROVED", "ch4": "APPROVED", "gcc": "PROVISIONAL",
        "phenological_season": "PROVISIONAL",
    }).fillna("PROVISIONAL")
    return frame


def provenance_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    provenance = pd.DataFrame([
        ["data/raw/stjones_season.csv", "new integrated raw dataset and empirical audit target", True, "schema, values, relationships, coverage, duplicates, missingness", "none; it contains GCC/Season not documented in Readme.rtf"],
        ["data/raw/Readme.rtf", "dataset processing notes and partial variable dictionary", True, "open-gap/gap-filled definitions, storage processing, YSI/NOAA fill relationships", "date range is inconsistent internally and GCC/Season, units, QC meanings, timestamps are absent"],
    ], columns=["file", "role", "authoritative", "information_extracted", "conflicts_with_other_sources"])

    inventory_rows = []
    for path in sorted(PROJECT_ROOT.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        rel = relative(path)
        generated_context = {
            "reports/data_contract_audit_v1.md",
            "reports/data_contract_targeted_resolution_v1.md",
            "config/data_contract_v1.json",
            "config/data_contract_core_v1.json",
            "config/data_contract_phenology_environment_v1.json",
        }
        if rel.startswith("outputs/data_contract/") or rel in generated_context:
            continue
        if rel.startswith("data/raw/"):
            role = "raw input or metadata"
        elif rel.startswith("data/processed/"):
            role = "processed analysis artifact; must remain untouched"
        elif rel.startswith("scripts/"):
            role = "analysis script" if path.name != Path(__file__).name else "new data-contract audit script"
        elif rel.startswith("reports/"):
            role = "analysis report; must remain untouched"
        elif rel.startswith("outputs/") or rel.startswith("figures/"):
            role = "analysis output; must remain untouched"
        else:
            role = "project context"
        stat = path.stat()
        inventory_rows.append({
            "file": rel, "role": role, "size_bytes": stat.st_size,
            "modification_time_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            "authoritative": rel in {"data/raw/stjones_season.csv", "data/raw/Readme.rtf"},
        })
    return provenance, pd.DataFrame(inventory_rows)


def legacy_snapshot() -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    protected_prefixes = ["data/processed/", "figures/", "outputs/", "reports/"]
    excluded = {
        "reports/data_contract_audit_v1.md", "reports/data_contract_targeted_resolution_v1.md",
        "reports/data_contract_freeze_v1.md", relative(HALFHOURLY_PATH), relative(DAILY_PHENOLOGY_PATH),
    }
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = relative(path)
        if rel.startswith("outputs/data_contract/") or rel in excluded or "__pycache__" in rel:
            continue
        if any(rel.startswith(prefix) for prefix in protected_prefixes):
            stat = path.stat()
            result[rel] = (stat.st_size, stat.st_mtime_ns)
    return result


def contract_dicts(
    manifest: dict[str, Any], mapping: pd.DataFrame, decisions: dict[str, Any],
    unresolved_core: list[str], unresolved_extension: list[str], qc_rules: pd.DataFrame,
    phenology: dict[str, Any], timestamp_audit: dict[str, Any], environment: pd.DataFrame,
    standardization: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    def approved_value(key: str, fallback: Any) -> Any:
        entry = decisions.get(key, {})
        return entry.get("value", fallback) if decision_is_complete(entry) else fallback

    core_status = "FROZEN" if not unresolved_core else "DRAFT_NEEDS_AUTHOR_DECISIONS"
    extension_status = "FROZEN" if not unresolved_extension else "DRAFT_NEEDS_AUTHOR_DECISIONS"
    overall_status = "FROZEN" if core_status == extension_status == "FROZEN" else "DRAFT_NEEDS_AUTHOR_DECISIONS"
    primary_rule = approved_value("primary_qc_eligibility_rule", "qc_0_or_1")
    sensitivity_rules = approved_value("sensitivity_qc_rule", ["qc_0_or_1", "qc_0_only"])
    mappings = mapping.replace({np.nan: None}).to_dict(orient="records")
    core_names = {"timestamp_start", "timestamp_end", "date", "nee", "nee_qc", "ch4", "ch4_qc"}
    core_mappings = [row for row in mappings if row["canonical_name"] in core_names]
    extension_mappings = [row for row in mappings if row["canonical_name"] not in core_names]
    env_mappings = environment.loc[environment["recommended_canonical_source"], [
        "construct", "candidate_column", "mapping_status", "measured_or_filled_status",
        "unit", "unresolved_metadata",
    ]].replace({np.nan: None}).to_dict(orient="records")
    raw_identity = {
        "path": "data/raw/stjones_season.csv", "sha256": manifest["sha256_before"],
        "size_bytes": manifest["size_bytes"], "rows": manifest["row_count"],
        "columns": manifest["column_count"],
    }
    timestamp_convention = {
        "source_timestamp_end": "DATE_TIME",
        "source_timestamp_start": "derived as DATE_TIME - 30 minutes",
        "date_time_role": "interval_end",
        "role_status": "APPROVED_EMPIRICAL",
        "interval_minutes": 30,
        "explicit_timestamp_start_end_usable": False,
        "timezone": approved_value("timezone_convention", None),
        "timezone_status": "UNDOCUMENTED_NO_CONVERSION_OR_DST_CORRECTION",
        "evidence": "all 87,691 interpretable surviving month prefixes and all 56 start/end month-boundary differences match a 30-minute interval ending at DATE_TIME",
    }
    missing_contract = {
        "status": "APPROVED",
        "observed_lexical_tokens": manifest["observed_missing_value_tokens"],
        "canonical_representation": "null/NA",
    }
    duplicate_contract = {
        "status": "APPROVED", "key": "DATE_TIME",
        "rule": "keep smallest source_row_number within exact duplicate group",
        "affected_rows": timestamp_audit["duplicate_timestamp_affected_rows"],
        "groups": timestamp_audit["duplicate_timestamp_groups"],
        "audit_trail": "outputs/data_contract/duplicate_timestamp_rows.csv",
    }
    qc_contract = {
        "status": "APPROVED_PRIMARY_WITH_UNDOCUMENTED_QC_SENSITIVITIES",
        "candidate_nee_qc_field": "qc_co2_flux",
        "candidate_ch4_qc_field": "qc_ch4_flux",
        "exact_final_product_correspondence": "UNRESOLVED",
        "class_definitions": approved_value("qc_class_definitions", "UNRESOLVED"),
        "selected_rule": primary_rule,
        "primary_rule": primary_rule,
        "sensitivity_rules": sensitivity_rules,
        "primary_rule_rationale": "Author-approved primary eligibility uses finite paired observed NEE_orig and CH4_orig after exact-duplicate resolution; undocumented QC classes do not exclude primary rows.",
        "sensitivity_rule_rationale": "qc_0_or_1 and qc_0_only flags are sensitivity datasets only. Their class meanings and final-product applicability remain undocumented and are not presented as quality rankings.",
        "rules_compared": qc_rules.replace({np.nan: None}).to_dict(orient="records"),
        "local_metadata_search": {
            "result": "NO_AUTHORITATIVE_CLASS_DEFINITIONS_FOUND",
            "searched_paths": [
                "data/raw/Readme.rtf",
                "scripts/**/*", "reports/**/*", "tests/**/*",
            ],
            "authoritative_readme": "states QA/QC filtering but gives no class meanings or final-product QC mapping",
            "supporting_files": [],
            "limitation": "No locally retained source defines the QC classes or their applicability to the final storage-corrected products.",
        },
    }
    core = {
        "contract_version": CONTRACT_VERSION,
        "contract_layer": "core_carbon_data",
        "creation_date": CREATION_DATE,
        "contract_status": core_status,
        "raw_input": raw_identity,
        "freeze_criteria": [
            "authoritative NEE column", "authoritative CH4 column", "NEE unit and sign",
            "CH4 unit and sign", "timestamp start/end convention", "duplicate handling",
            "missing-value handling", "QC eligibility rule",
        ],
        "approved_structural_decisions": {
            "nee_source": "NEE_orig",
            "ch4_source": "CH4_orig",
            "FCH4_RF_filled_primary_response": False,
            "duplicate_handling": duplicate_contract["rule"],
            "hmm_emissions": ["nee", "ch4"],
        },
        "canonical_source_column_mappings": core_mappings,
        "units": {
            "nee": approved_value("nee_unit", "UNRESOLVED"),
            "ch4": approved_value("ch4_unit", "UNRESOLVED"),
        },
        "sign_conventions": {
            "nee": "negative NEE = net ecosystem CO2 uptake; positive NEE = net ecosystem CO2 release",
            "nee_status": "APPROVED_FROM_AUTHOR_SCIENTIFIC_BOUNDARY",
            "ch4": approved_value("ch4_sign_convention", "UNRESOLVED"),
            "CO2_uptake": "display only; never an HMM response",
        },
        "storage_correction_wording": {
            "nee": approved_value("nee_storage_correction_wording", "UNRESOLVED_NONBLOCKING; empirical turbulent + storage identity is documented in audit"),
            "ch4": approved_value("ch4_storage_correction_wording", "UNRESOLVED_NONBLOCKING; empirical 1000*(turbulent + storage) identity is documented in audit"),
        },
        "timestamp_convention": timestamp_convention,
        "duplicate_handling": duplicate_contract,
        "missing_value_representation": missing_contract,
        "qc_rules": qc_contract,
        "hmm_response_variables": ["nee", "ch4"],
        "variables_prohibited_from_hmm_inference": PROHIBITED_HMM_COLUMNS + ["CO2_uptake"],
        "sequence_construction_rules": {
            "sort_by": "timestamp_start", "expected_spacing_minutes": 30,
            "split_when_retained_spacing_not_30_minutes": True,
            "split_across_excluded_or_missing_interval": True,
            "environment_and_phenology_do_not_determine_hmm_eligibility": True,
        },
        "standardization": {
            "rule": "calculate mean and sample SD only from observations eligible under the frozen core contract",
            "parameters": standardization,
        },
        "unresolved_critical_decisions": unresolved_core,
        "unresolved_noncritical_metadata": [
            "timezone and daylight-saving convention (no assignment, conversion, or correction applied)",
            "QC-class definitions and applicability to storage-adjusted response variables",
        ],
        "official_output": relative(HALFHOURLY_PATH),
        "author_decision_file": relative(DECISIONS_PATH),
    }
    extension = {
        "contract_version": CONTRACT_VERSION,
        "contract_layer": "phenology_environment_extension",
        "creation_date": CREATION_DATE,
        "contract_status": extension_status,
        "raw_input": raw_identity,
        "core_contract_reference": relative(CORE_CONTRACT_PATH),
        "core_contract_status_at_audit": core_status,
        "approved_structural_decisions": {
            "gcc_enters_hmm": False,
            "season_enters_hmm": False,
            "phenology_comparison_grain": "daily posterior state occupancy",
            "repeated_daily_values_are_independent_half_hours": False,
        },
        "canonical_source_column_mappings": extension_mappings,
        "daily_phenology_rules": {
            "gcc_source": "GCC", "season_source": "Season",
            "gcc_temporal_resolution": phenology["gcc_temporal_resolution"],
            "season_temporal_resolution": phenology["season_temporal_resolution"],
            "comparison_grain": "daily posterior state occupancy",
            "repeated_half_hours_are_not_independent": True,
            "canonical_phase_order": ["Greenup", "Maturity", "Senescence", "Dormancy"],
            "source_to_canonical_phase_mapping": {
                "Spring": "Greenup", "Summer": "Maturity",
                "Autumn": "Senescence", "Winter": "Dormancy",
            },
            "gcc_definition": approved_value("gcc_definition", "UNRESOLVED"),
            "gcc_source_and_daily_aggregation": approved_value("gcc_source_and_daily_aggregation", "UNRESOLVED"),
            "gcc_smoothing_method": approved_value("gcc_smoothing_method", "UNRESOLVED"),
            "season_transition_method": approved_value("season_transition_method", "UNRESOLVED"),
            "missing_gcc_season_assignment": approved_value("missing_gcc_season_assignment", "UNRESOLVED"),
        },
        "environmental_variable_mappings": env_mappings,
        "unresolved_critical_decisions": unresolved_extension,
        "unresolved_noncritical_metadata": [
            "some environmental-variable units", "soil-temperature sensor depth",
            "water-level datum, marsh reference, sign interpretation, and inundation threshold",
        ],
        "official_output": relative(DAILY_PHENOLOGY_PATH),
        "author_decision_file": relative(DECISIONS_PATH),
    }
    umbrella = {
        "contract_version": CONTRACT_VERSION,
        "creation_date": CREATION_DATE,
        "contract_status": overall_status,
        "audit_scope": "data contract only; no HMM fitting or downstream ecological analysis",
        "contract_architecture": "two_layer",
        "contract_layers": {
            "core_carbon_data": {"path": relative(CORE_CONTRACT_PATH), "status": core_status},
            "phenology_environment_extension": {"path": relative(EXTENSION_CONTRACT_PATH), "status": extension_status},
        },
        "analysis_branch": {
            "status": "UNAVAILABLE_NOT_A_GIT_REPOSITORY",
            "evidence": "No .git directory exists at the project root; audit outputs are isolated by path.",
        },
        "raw_input": raw_identity,
        "canonical_source_column_mappings": mappings,
        "units": core["units"],
        "sign_conventions": core["sign_conventions"],
        "timestamp_convention": timestamp_convention,
        "missing_value_representation": missing_contract,
        "duplicate_handling": duplicate_contract,
        "qc_rules": qc_contract,
        "exclusion_reasons": [
            "invalid_timestamp", "duplicate_timestamp_later_copy", "missing_NEE", "missing_CH4",
            "nonfinite_NEE", "nonfinite_CH4", "NEE_QC", "CH4_QC",
            "non_30_minute_interval", "documented_impossible_or_invalid_value", "other_rule",
        ],
        "measured_versus_filled_status": {
            "nee": "documented QA/QC'ed open-gap", "ch4": "documented QA/QC'ed open-gap",
            "gap_filled_responses_prohibited_by_default": sorted(GAP_FILLED_RESPONSE_COLUMNS),
        },
        "hmm_response_variables": core["hmm_response_variables"],
        "variables_prohibited_from_hmm_inference": core["variables_prohibited_from_hmm_inference"],
        "standardization": core["standardization"],
        "daily_phenology_rules": extension["daily_phenology_rules"],
        "environmental_variable_mappings": env_mappings,
        "sequence_construction_rules": core["sequence_construction_rules"],
        "approved_structural_decisions": {
            **core["approved_structural_decisions"],
            **extension["approved_structural_decisions"],
        },
        "unresolved_critical_decisions": {
            "core_carbon_data": unresolved_core,
            "phenology_environment_extension": unresolved_extension,
        },
        "author_decision_file": relative(DECISIONS_PATH),
    }
    return umbrella, core, extension


def eligibility_mask(raw: pd.DataFrame, times: dict[str, pd.Series], rule: str) -> pd.Series:
    timestamp_valid = times["date_time"].notna()
    duplicate_later = times["date_time"].duplicated(keep="first") & timestamp_valid
    finite = np.isfinite(pd.to_numeric(raw["NEE_orig"], errors="coerce")) & np.isfinite(pd.to_numeric(raw["CH4_orig"], errors="coerce"))
    mask = timestamp_valid & ~duplicate_later & finite
    if rule == "qc_0_only":
        mask &= raw["qc_co2_flux"].eq(0) & raw["qc_ch4_flux"].eq(0)
    elif rule == "qc_0_or_1":
        mask &= raw["qc_co2_flux"].isin([0, 1]) & raw["qc_ch4_flux"].isin([0, 1])
    return mask


def build_official_core_table(
    raw: pd.DataFrame, times: dict[str, pd.Series], core_contract: dict[str, Any],
    daily_phenology: pd.DataFrame,
) -> list[dict[str, Any]]:
    rule = core_contract["qc_rules"]["selected_rule"]
    eligible = eligibility_mask(raw, times, rule)
    qc_01 = eligibility_mask(raw, times, "qc_0_or_1")
    qc_0 = eligibility_mask(raw, times, "qc_0_only")
    duplicate_later = times["date_time"].duplicated(keep="first") & times["date_time"].notna()
    duplicate_any = times["date_time"].duplicated(keep=False) & times["date_time"].notna()
    duplicate_keys = sorted(times["date_time"].loc[duplicate_any].drop_duplicates())
    duplicate_group_map = {value: number for number, value in enumerate(duplicate_keys, start=1)}
    keep = ~duplicate_later
    reasons = []
    for index in raw.index:
        row_reasons = []
        if pd.isna(times["date_time"].iloc[index]): row_reasons.append("invalid_timestamp")
        if pd.isna(raw["NEE_orig"].iloc[index]): row_reasons.append("missing_NEE")
        elif not np.isfinite(raw["NEE_orig"].iloc[index]): row_reasons.append("nonfinite_NEE")
        if pd.isna(raw["CH4_orig"].iloc[index]): row_reasons.append("missing_CH4")
        elif not np.isfinite(raw["CH4_orig"].iloc[index]): row_reasons.append("nonfinite_CH4")
        reasons.append(" | ".join(row_reasons))
    canonical = pd.DataFrame({
        "source_row": raw.index + 1,
        "source_file_line": raw.index + 2,
        "timestamp_start": times["timestamp_start"], "timestamp_end": times["timestamp_end"],
        "date": pd.to_datetime(raw["DATE"], errors="coerce").dt.date,
        "NEE": raw["NEE_orig"], "CH4": raw["CH4_orig"],
        "nee_unit": core_contract["units"]["nee"], "ch4_unit": core_contract["units"]["ch4"],
        "primary_hmm_eligible": eligible,
        "qc_01_sensitivity_eligible": qc_01,
        "qc_0_sensitivity_eligible": qc_0,
        "exclusion_reasons": reasons,
        "duplicate_group": times["date_time"].map(duplicate_group_map).astype("Int64"),
        "duplicate_retained": duplicate_any & ~duplicate_later,
        "NEE_source_column": "NEE_orig", "CH4_source_column": "CH4_orig",
    })
    for canonical_name, source_column in ENVIRONMENTAL_RECOMMENDATIONS.items():
        if source_column in raw.columns:
            canonical[canonical_name] = raw[source_column]
            canonical[f"{canonical_name}_source_column"] = source_column
    daily_fields = daily_phenology[[
        "date", "GCC", "GCC_available", "phenological_phase", "phenological_phase_source",
        "phenological_phase_filled", "phase_interval_start", "phase_interval_end", "transition_dates_known",
    ]].rename(columns={"GCC_available": "gcc_available"}).copy()
    daily_fields["date"] = pd.to_datetime(daily_fields["date"]).dt.date
    canonical = canonical.merge(daily_fields, on="date", how="left", validate="many_to_one")
    canonical["GCC_source_column"] = "GCC"
    canonical["phenological_phase_source_column"] = "Season"
    canonical = canonical.loc[keep].sort_values(["timestamp_start", "source_row"], kind="mergesort").reset_index(drop=True)
    if "BP_MET" not in raw.columns:
        raise RuntimeError("BP_MET is required to create the frozen Figure 3 barometric-pressure input.")
    barometric_pressure = canonical.loc[:, ["source_row"]].copy()
    barometric_pressure["barometric_pressure"] = pd.to_numeric(
        raw.loc[barometric_pressure["source_row"].to_numpy() - 1, "BP_MET"], errors="coerce"
    ).to_numpy()
    atomic_csv(barometric_pressure, BAROMETRIC_PRESSURE_PATH, compression="gzip")
    params = []
    for column, canonical_name in [("NEE", "nee"), ("CH4", "ch4")]:
        mean = float(canonical.loc[canonical["primary_hmm_eligible"], column].mean())
        sd = float(canonical.loc[canonical["primary_hmm_eligible"], column].std(ddof=1))
        canonical[f"{canonical_name}_standardized"] = (canonical[column] - mean) / sd
        params.append({"source": canonical_name, "mean": mean, "sample_standard_deviation": sd, "n": int(canonical["primary_hmm_eligible"].sum())})
    atomic_csv(canonical, HALFHOURLY_PATH, compression="gzip", date_format="%Y-%m-%dT%H:%M:%S")

    return params


def build_official_extension_table(daily_audit: pd.DataFrame) -> None:
    daily = daily_audit[[
        "date", "GCC", "GCC_available", "phenological_phase", "phenological_phase_source",
        "phenological_phase_filled", "phase_interval_start", "phase_interval_end", "transition_dates_known",
        "source_record_count", "GCC_unique_nonmissing_values", "Season_unique_nonmissing_values",
        "GCC_within_date_consistent", "Season_within_date_consistent",
    ]].rename(columns={
        "GCC_available": "gcc_available",
        "GCC_unique_nonmissing_values": "number_unique_gcc_values_within_date",
        "Season_unique_nonmissing_values": "number_unique_phase_values_within_date",
        "GCC_within_date_consistent": "within_date_gcc_consistent",
        "Season_within_date_consistent": "within_date_phase_consistent",
    })
    if daily["date"].duplicated().any():
        raise RuntimeError("Daily phenology output is not unique by date")
    atomic_csv(daily, DAILY_PHENOLOGY_PATH, date_format="%Y-%m-%d")


def markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    display = frame if columns is None else frame.loc[:, columns]
    if display.empty:
        return "_No rows._"
    values = display.copy().replace({np.nan: ""})
    header = "| " + " | ".join(str(column) for column in values.columns) + " |"
    separator = "| " + " | ".join("---" for _ in values.columns) + " |"
    rows = []
    for row in values.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join([header, separator, *rows])


def write_report(
    contract: dict[str, Any], manifest: dict[str, Any], timestamp: dict[str, Any],
    nee: pd.DataFrame, ch4: pd.DataFrame, nee_rel: pd.DataFrame, ch4_rel: pd.DataFrame,
    qc_rules: pd.DataFrame, waterfall: pd.DataFrame, environment: pd.DataFrame,
    phenology: dict[str, Any], mappings: pd.DataFrame, provenance: pd.DataFrame,
    test_result: dict[str, Any], immutability: dict[str, Any],
) -> None:
    status = contract["contract_status"]
    selected_qc = contract["qc_rules"]["selected_rule"]
    eligible = int(qc_rules.set_index("rule_id").loc[selected_qc, "eligible_rows"])
    env_selected = environment.loc[environment["recommended_canonical_source"], [
        "construct", "candidate_column", "mapping_status", "missing_proportion", "unresolved_metadata"
    ]].copy()
    env_selected["missing_proportion"] = env_selected["missing_proportion"].map(lambda value: f"{100*value:.1f}%")
    report = f"""# Salt-marsh integrated data-contract audit v1

## 1. Executive decision — {status}

The contract now has two independently gated layers. The **core carbon-data contract is {contract['contract_layers']['core_carbon_data']['status']}** and the **phenology/environment extension is {contract['contract_layers']['phenology_environment_extension']['status']}**. The umbrella remains `{status}` until both layers are frozen.

No HMM was fit, no state count was selected, and no downstream ecological analysis or manuscript figure was run. Each official table is independently gated by its applicable layer: the half-hourly carbon table by the core contract and the daily phenology table by the extension. The requested Git branch could not be created because the project root is not a Git repository.

## 2. Raw source fingerprint

- Path: `data/raw/stjones_season.csv`
- SHA-256 before: `{manifest['sha256_before']}`
- SHA-256 after: `{manifest['sha256_after']}`
- Unchanged: `{manifest['raw_unchanged']}`
- Size: {manifest['size_bytes']:,} bytes
- Modification time: `{manifest['modification_time_utc']}`
- Format: `{manifest['encoding']}` encoding, comma delimiter, double-quote quoting, period decimal convention
- Observed missing tokens: `{json.dumps(manifest['observed_missing_value_tokens'], sort_keys=True)}`

## 3. Dataset dimensions and temporal coverage

The file contains **{manifest['row_count']:,} rows and {manifest['column_count']:,} ordered columns**. `DATE_TIME` spans `{timestamp['temporal_coverage_date_time']['minimum']}` through `{timestamp['temporal_coverage_date_time']['maximum']}`. The complete ordered schema is stored in `outputs/data_contract/raw_file_manifest.json`; per-column storage, parsing, missingness, cardinality, range, examples, constancy, duplicate status, and provisional roles are in `outputs/data_contract/column_audit.csv`.

## 4. Timestamp findings

`TIMESTAMP_START` and `TIMESTAMP_END` parse successfully for {timestamp['exported_timestamp_start_parse_success']:,} and {timestamp['exported_timestamp_end_parse_success']:,} rows as `%Y%m%d%H%M`; lossless normalization from scientific notation also yields zero successful rows. They are unusable because their day/time digits were replaced by zeros; all 2020 values collapsed to `2.02e+11`. `DATE_TIME` parses under the explicit `%Y-%m-%d %H:%M:%S` format for {timestamp['date_time_parse_success']:,} rows and equals explicitly parsed `DATE + TIME` for {timestamp['date_time_equals_date_plus_time_count']:,} rows.

The interval role is now **APPROVED from empirical structure**: for all {timestamp['interpretable_month_prefix_rows']:,} rows retaining valid start/end month prefixes, `TIMESTAMP_START` matches `DATE_TIME - 30 minutes` and `TIMESTAMP_END` matches `DATE_TIME`; all {timestamp['timestamp_start_end_raw_different_rows']} surviving month-boundary start/end differences agree. Canonical `timestamp_end = DATE_TIME` and `timestamp_start = DATE_TIME - 30 minutes`.

Timezone, local-standard-time/daylight-saving handling, and UTC status are unresolved. There are {timestamp['missing_half_hours_within_observed_grid']} missing grid positions, {timestamp['original_order_non_monotonic_steps']} non-monotonic step in source order, and no 46/50-record daily pattern that would independently establish DST behavior.

## 5. Duplicate-record findings

There are **{timestamp['duplicate_timestamp_groups']} duplicated timestamps affecting {timestamp['duplicate_timestamp_affected_rows']} rows**. All groups are exact across every source column. They are the 48 half-hours of 2018-12-31 repeated twice. The approved deterministic rule is to keep the smallest source-row number; every copy remains documented in `outputs/data_contract/duplicate_timestamp_rows.csv`.

Legacy artifact immutability check: {immutability['unchanged_file_count']:,} protected files were unchanged by size and nanosecond modification time; changed protected files: {immutability['changed_files']}.

## 6. NEE candidate comparison and recommendation

`NEE_orig` is the recommended observed response because the authoritative Readme defines it as QA/QC'ed open-gap CO2 data. It agrees with `co2_flux + co2_strg` to the CSV's numerical precision, which empirically supports—but does not itself prove—the storage-adjusted interpretation. Its unit remains insufficiently documented; the legacy `umol CO2 m^-2 s^-1` label is a hypothesis pending author confirmation. The original author-specified sign is preserved: negative NEE is uptake and positive NEE is release.

{markdown_table(nee[['candidate','documented_definition','missing_count','minimum','median','maximum','primary_response_source_status']])}

Key relationship check:

{markdown_table(nee_rel[['check_name','n_overlap','exact_equal_count','absolute_difference_le_1e_4_count','maximum_absolute_difference','metadata_support']])}

## 7. CH4 candidate comparison and recommendation

`CH4_orig` is the recommended observed response because the authoritative Readme defines it as QA/QC'ed open-gap CH4 data. It agrees closely with `1000 * (ch4_flux + ch4_strg)`, supporting a storage-adjusted interpretation and a scale conversion empirically. The final unit and CH4 sign convention are not documented and cannot be frozen from magnitude or column names.

The Readme describes `FCH4_RF_filled` as `CH4_orig` with gaps filled, but the two differ by more than 0.001 on **{int(ch4_rel.loc[ch4_rel['check_name'].eq('FCH4_RF_filled_equals_CH4_orig_where_observed'), 'absolute_difference_gt_1e_3_count'].iloc[0]):,} overlapping rows**. All of those rows have both `qc_co2_flux` and `qc_ch4_flux` missing. This conflict reinforces that `FCH4_RF_filled` is a budget/sensitivity product, not the primary observed HMM response.

{markdown_table(ch4[['candidate','documented_definition','missing_count','minimum','median','maximum','primary_response_source_status']])}

Key relationship checks:

{markdown_table(ch4_rel[['check_name','n_overlap','exact_equal_count','absolute_difference_le_1e_3_count','maximum_absolute_difference','metadata_support']])}

## 8. Units and sign conventions

- NEE source: `NEE_orig`; source identity supported, unit unresolved, sign explicitly supplied by the author.
- CH4 source: `CH4_orig`; source identity supported, unit and sign unresolved.
- Raw EddyPro-like CH4 component fields and post-processed CH4 are empirically separated by a factor of 1,000, but this is not a substitute for documented units.
- `CO2_uptake = -nee` is allowed only as a display variable and is prohibited as an HMM emission.

## 9. QC findings and candidate retention counts

The Readme says both `NEE_orig` and `CH4_orig` are already QA/QC'ed open-gap products. It does not define QC classes 0/1/2 or establish whether `qc_co2_flux` and `qc_ch4_flux` govern the final storage-adjusted fields. The author-approved primary rule is `{selected_qc}`—finite paired observed responses after duplicate handling—which retains **{eligible:,} observations**. The 0/1 and class-0 rules are retained only as sensitivity flags; neither is described as a higher-quality filter.

{markdown_table(qc_rules[['rule_id','eligible_rows','sequence_count','singleton_sequence_count','median_sequence_length_half_hours','maximum_sequence_length_half_hours','status']])}

## 10. Environmental-variable recommendations

These fields are post-HMM variables only. Mappings remain provisional where authoritative units, sensor details, or derivations are missing. Recorded water level is not labeled inundation depth.

{markdown_table(env_selected)}

## 11. GCC and Season findings

GCC is empirically daily: each date has at most one finite value repeated over its half-hours. It is available on {phenology['dates_with_gcc']:,} of {phenology['dates_in_source']:,} dates, with range {phenology['observed_gcc_range']['minimum']:.6f}–{phenology['observed_gcc_range']['maximum']:.6f}; {phenology['dates_with_conflicting_gcc']} dates conflict within date. Author-approved provenance defines GCC as `G / (R + G + B)` from the St. Jones PhenoCam canopy ROI, processed through 2021 using `phenopix` autoFilter, spline filtering, annual greenExplore curves, and gu transition dates. The canonical daily phases are Greenup, Maturity, Senescence, and Dormancy; source labels occur only in provenance fields. A phase remains permitted where GCC is missing only inside a known bounded annual interval.

Future comparison must aggregate HMM state occupancy daily; the repeated half-hours are not independent phenology observations.

## 12. Proposed canonical mapping

{markdown_table(mappings[['canonical_name','selected_source_column','unit','temporal_resolution','status','unresolved_issue']])}

## 13. Complete exclusion waterfall

The table below is the sequential waterfall for the provisional `{selected_qc}` rule. Counts for all alternative rules and nonexclusive reason counts are stored in `outputs/data_contract/exclusion_waterfall.csv`.

{markdown_table(waterfall.loc[waterfall['rule_id'].eq(selected_qc), ['step_order','exclusion_reason','excluded_at_step','remaining_after_step']])}

## 14. Tests run and results

- Command: `{test_result.get('command', 'not yet run')}`
- Result: **{test_result.get('status', 'NOT_RUN')}**
- Return code: `{test_result.get('return_code', 'not available')}`
- Summary: `{test_result.get('summary', 'not available')}`

## 15. Critical unresolved author decisions by contract layer

Core carbon-data contract:

{chr(10).join(f'- `{item}`' for item in contract['unresolved_critical_decisions']['core_carbon_data']) if contract['unresolved_critical_decisions']['core_carbon_data'] else '- None.'}

Phenology/environment extension:

{chr(10).join(f'- `{item}`' for item in contract['unresolved_critical_decisions']['phenology_environment_extension']) if contract['unresolved_critical_decisions']['phenology_environment_extension'] else '- None.'}

Additional noncritical metadata gaps are timezone/DST convention, QC-class interpretation, environmental units, soil-sensor depths, and water-level unit/datum/marsh reference/sign/inundation threshold.

## 16. Exact command to reproduce the audit

```bash
python scripts/01_audit_and_freeze_data_contract.py --audit
```

## 17. Exact freeze command after approval

Populate `config/data_contract_author_decisions.json` using the supplied template and cite the supporting documentation, then run:

```bash
python scripts/01_audit_and_freeze_data_contract.py --freeze
```

`--freeze` refuses when neither applicable layer is frozen. Each official table is gated by its own contract layer; a frozen core may create the carbon table while the extension remains draft.

## Metadata provenance

{markdown_table(provenance)}
"""
    atomic_text(REPORT_PATH, report)


def write_targeted_resolution_report(
    umbrella: dict[str, Any], core: dict[str, Any], extension: dict[str, Any],
    timestamp: dict[str, Any], qc_rules: pd.DataFrame, test_result: dict[str, Any],
) -> None:
    format_frame = pd.DataFrame(timestamp["explicit_format_parse_results"])
    required_formats = format_frame.loc[
        format_frame["explicit_format"].isin([
            "%Y%m%d%H%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"
        ])
    ].copy()
    profiles = pd.DataFrame(timestamp["raw_field_profiles"])
    profiles["raw_examples"] = profiles["raw_examples"].map(lambda values: " | ".join(values))
    qc_display = qc_rules[[
        "rule_id", "eligible_rows", "retained_duration_hours", "sequence_count",
        "singleton_sequence_count", "median_sequence_length_half_hours",
        "maximum_sequence_length_half_hours", "status",
    ]]
    if core["contract_status"] == "FROZEN" and extension["contract_status"] == "FROZEN":
        table_note = "Both applicable layers are frozen; both official tables are permitted."
    elif core["contract_status"] == "FROZEN":
        table_note = "Only the core layer is frozen; only the official half-hourly carbon table is permitted."
    elif extension["contract_status"] == "FROZEN":
        table_note = "Only the extension is frozen; only the official daily phenology table is permitted."
    else:
        table_note = "Neither applicable layer is frozen; no official contract table was created."
    report = f"""# Targeted data-contract resolution pass v1

## Decision

- Core carbon-data contract: **{core['contract_status']}**
- Phenology/environment extension: **{extension['contract_status']}**
- Umbrella contract: **{umbrella['contract_status']}**

No HMM was fit. {table_note}

## Timestamp resolution

The explicit `TIMESTAMP_START` and `TIMESTAMP_END` columns are not valid compact timestamps. Their raw values are scientific-notation strings such as `2.01601e+11`; lossless decimal normalization produces values such as `201601000000`, whose day is `00`. All 2020 rows collapse to `2.02e+11`. Parse success is zero under `%Y%m%d%H%M`, both before and after normalization.

`DATE_TIME` parses for all {timestamp['date_time_parse_success']:,} rows only under the explicit `%Y-%m-%d %H:%M:%S` representation among the required full timestamp formats. It equals explicitly parsed `DATE + TIME` for all rows.

The surviving month prefixes resolve the interval role: all **{timestamp['interpretable_month_prefix_rows']:,}** interpretable rows match `TIMESTAMP_START` to `DATE_TIME - 30 minutes` and `TIMESTAMP_END` to `DATE_TIME` at month grain. All **{timestamp['timestamp_start_end_raw_different_rows']}** surviving month-boundary differences have this structure. Therefore `DATE_TIME` is approved as interval end and canonical interval start is `DATE_TIME - 30 minutes`.

All {timestamp['unique_midnight_date_time_rows']:,} unique observed midnight timestamps have a preceding 23:30 record. Five year-boundary midnight timestamps are absent and remain explicit sequence breaks. No local source documents timezone, local-standard-time, DST, or UTC handling.

### Raw field profiles

{markdown_table(profiles[['field','pandas_inferred_storage_type','raw_lexical_storage_type','unique_raw_values','raw_examples']])}

### Required explicit-format results

{markdown_table(required_formats[['field','input_representation','explicit_format','successful_rows','parse_rate']])}

## QC resolution

The only plausible local fields are `qc_co2_flux` for the turbulent CO2 flux and `qc_ch4_flux` for turbulent CH4 flux. The search covered the raw Readme, project-level metadata, all local scripts/comments, tests, and reports. No local authoritative source defines values 0, 1, 2, or missing, and none explicitly states that these fields govern the final storage-adjusted `NEE_orig` and `CH4_orig` products. Legacy files repeatedly apply 0/1 as primary and 0-only as sensitivity, but also state that final QC-class interpretation is unresolved.

Empirically, neither `NEE_orig` nor `CH4_orig` is finite when its same-gas QC candidate equals 2; both remain finite on 4,727 rows where both QC candidates are missing. This is consistency evidence, not a class definition.

The author-approved primary rule uses finite paired observed responses with no QC-class exclusion. `qc_0_or_1` and `qc_0_only` are sensitivity flags only; neither is assigned a quality ranking:

{markdown_table(qc_display)}

QC-class definitions and final-product correspondence remain noncritical limitations of the frozen core contract.

## Approved structural decisions

- `NEE_orig` is the canonical observed NEE source.
- `CH4_orig` is the canonical observed CH4 source.
- `FCH4_RF_filled` is not a primary HMM response.
- Exact timestamp duplicates retain the smallest source-row number; all copies remain audited.
- HMM emissions are canonical NEE and CH4 only.
- GCC and Season never enter HMM inference.
- Phenology comparison uses daily posterior state occupancy.

## Two-layer gate

The core layer can freeze without the extension. Its unresolved critical decisions are:

{chr(10).join(f'- `{item}`' for item in core['unresolved_critical_decisions']) if core['unresolved_critical_decisions'] else '- None.'}

The phenology/environment extension remains provisional until these decisions are documented:

{chr(10).join(f'- `{item}`' for item in extension['unresolved_critical_decisions']) if extension['unresolved_critical_decisions'] else '- None.'}

Official carbon and daily phenology tables are independently gated by their applicable layer status.

## Validation

- Command: `{test_result.get('command', 'not yet run')}`
- Status: **{test_result.get('status', 'NOT_RUN')}**
- Summary: `{test_result.get('summary', 'not available')}`
"""
    atomic_text(TARGETED_REPORT_PATH, report)


def write_freeze_report(
    contract: dict[str, Any], core: dict[str, Any], extension: dict[str, Any],
    timestamp: dict[str, Any], qc_rules: pd.DataFrame, waterfall: pd.DataFrame,
    phenology_daily: pd.DataFrame, boundaries: pd.DataFrame, test_result: dict[str, Any],
    immutability: dict[str, Any],
) -> None:
    if contract["contract_status"] != "FROZEN":
        return
    halfhourly = pd.read_csv(HALFHOURLY_PATH, compression="gzip")
    daily = pd.read_csv(DAILY_PHENOLOGY_PATH)
    primary = qc_rules.set_index("rule_id").loc["all_finite_observed"]
    sequence_summary = qc_rules[[
        "rule_id", "eligible_rows", "sequence_count", "singleton_sequence_count",
        "median_sequence_length_half_hours", "maximum_sequence_length_half_hours",
    ]]
    boundary_display = boundaries[["year", "segment_order", "phenological_phase", "start_date", "end_date", "duration_days_in_year_segment"]]
    report = f"""# Frozen St. Jones data contract v1

## Final status

Both the core carbon-data contract and phenology/environment extension are **FROZEN**. No HMM was fit.

## Core carbon contract

- Canonical observed fluxes: `NEE_orig` → `NEE` and `CH4_orig` → `CH4`.
- NEE: `{core['units']['nee']}`; negative is ecosystem CO2 uptake and positive is ecosystem CO2 release.
- CH4: `{core['units']['ch4']}`; positive is ecosystem CH4 emission and negative is ecosystem CH4 uptake.
- Both are observed, QA/QC'ed, open-gap, storage-adjusted flux products; neither is described as gap-filled.
- `DATE_TIME` is interval end; `timestamp_start = DATE_TIME - 30 minutes`; interval duration is 30 minutes. Precision-damaged `TIMESTAMP_START`/`TIMESTAMP_END` are not authoritative.
- No timezone assignment, conversion, or DST correction was applied.
- Exact duplicates retain the smallest `source_row`; all 48 duplicate later copies are excluded from canonical timestamps and preserved in the audit trail.

Primary HMM eligibility is finite paired NEE and CH4 after duplicate resolution, with no undocumented QC-class exclusion: **{int(primary['eligible_rows']):,} rows**. QC sensitivity flags are present but are not quality rankings because class meanings and response-field applicability remain undocumented.

{markdown_table(sequence_summary)}

## Phenology/environment extension

- GCC is dimensionless `G / (R + G + B)` from St. Jones PhenoCam digital repeat photography, acquired approximately every 30 minutes over the salt-marsh vegetation-canopy ROI following Vazquez-Lule and Vargas (2021).
- Daily processing uses `phenopix`: `autoFilter`, spline filtering, annual year-specific `greenExplore` curves, and the `gu` transition-date method. The same procedure is approved for 2018–2021.
- Canonical phases are `Greenup`, `Maturity`, `Senescence`, and `Dormancy`; raw `Spring`, `Summer`, `Autumn`, and `Winter` occur only in `phenological_phase_source`.
- GCC was not manually reconstructed. A categorical phase may be present when GCC is missing only within a known bounded annual phase interval.
- GCC and phase are daily information repeated across half-hours, prohibited from HMM inference, and must be joined later to daily posterior state occupancy.

| Source provenance label | Canonical phenological phase |
| --- | --- |
| Spring | Greenup |
| Summer | Maturity |
| Autumn | Senescence |
| Winter | Dormancy |

## Canonical outputs

- Half-hourly table: **{len(halfhourly):,} rows × {halfhourly.shape[1]} columns**; primary eligible: **{int(halfhourly['primary_hmm_eligible'].sum()):,}**.
- Daily phenology table: **{len(daily):,} rows × {daily.shape[1]} columns**; dates with GCC: **{int(daily['gcc_available'].sum()):,}**; manually filled categorical phases: **{int(daily['phenological_phase_filled'].sum()):,}**.

### Annual phase-boundary summary

{markdown_table(boundary_display)}

### Primary exclusion waterfall

{markdown_table(waterfall.loc[waterfall['rule_id'].eq('all_finite_observed'), ['step_order','exclusion_reason','excluded_at_step','remaining_after_step']])}

## Remaining noncritical limitations

- Timezone and daylight-saving convention are undocumented; timestamps retain the observed local sequence.
- QC-class meanings and applicability to the storage-adjusted responses remain undocumented; QC flags are sensitivity-only.
- Some environmental units, soil-temperature sensor depth, and water-level datum/marsh reference/sign/inundation threshold remain unresolved and constrain later interpretation.

## Integrity and validation

- Raw SHA-256 unchanged: `{contract['raw_input']['sha256']}`.
- Protected legacy files changed: `{immutability['changed_files']}`.
- Tests: **{test_result['status']}** — `{test_result['summary']}`.
- Package versions are recorded in `outputs/data_contract/raw_file_manifest.json`.

## Reproduce

```bash
python scripts/01_audit_and_freeze_data_contract.py --freeze
```
"""
    atomic_text(FREEZE_REPORT_PATH, report)


def main() -> None:
    args = parse_args()
    if not RAW_PATH.is_file():
        raise FileNotFoundError(f"Raw source not found: {RAW_PATH}")
    before_hash = sha256(RAW_PATH)
    if before_hash != EXPECTED_SHA256:
        raise RuntimeError(f"Raw checksum mismatch: expected {EXPECTED_SHA256}, got {before_hash}")

    legacy_before = legacy_snapshot()
    decisions, unresolved_core, unresolved_extension = load_author_decisions()
    timestamp_decision = decisions.get("timestamp_interval_role", {})
    date_time_role = timestamp_decision.get("value", "interval_end") if decision_is_complete(timestamp_decision) else "interval_end"
    if date_time_role not in {"interval_start", "interval_midpoint", "interval_end"}:
        date_time_role = "interval_end"

    raw = pd.read_csv(
        RAW_PATH, low_memory=False, keep_default_na=True,
        na_values=["NA", -9999, -9999.0],
    )
    raw_text = pd.read_csv(
        RAW_PATH,
        usecols=["TIMESTAMP_START", "TIMESTAMP_END", "DATE_TIME", "DATE", "TIME"],
        dtype=str, keep_default_na=False,
    )
    missing_required = sorted(REQUIRED_SOURCE_COLUMNS - set(raw.columns))
    if missing_required:
        raise RuntimeError(f"Missing required source columns: {', '.join(missing_required)}")

    stat = RAW_PATH.stat()
    manifest: dict[str, Any] = {
        "raw_path": relative(RAW_PATH),
        "sha256_before": before_hash,
        "sha256_after": None,
        "raw_unchanged": None,
        "size_bytes": stat.st_size,
        "modification_time_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "row_count": int(len(raw)), "column_count": int(raw.shape[1]),
        "ordered_columns": raw.columns.tolist(),
        "delimiter": ",", "encoding": "US-ASCII", "quote_character": '"',
        "decimal_convention": "period",
        "observed_missing_value_tokens": lexical_missing_tokens(RAW_PATH),
        "package_versions": {
            "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
        },
    }

    times = parse_time(raw, date_time_role)
    timestamp_audit, duplicate_rows, daily_counts, missing_intervals = timestamp_outputs(
        raw, raw_text, times, date_time_role
    )
    column_audit, duplicate_columns, _ = build_column_audit(raw)
    missingness = column_audit[[
        "original_column_name", "missing_count", "missing_proportion", "unique_nonmissing_values",
        "is_all_missing", "likely_role",
    ]].sort_values(["missing_proportion", "original_column_name"], ascending=[False, True], ignore_index=True)

    nee_comparison = candidate_comparison(raw, times, NEE_CANDIDATES, "NEE", decisions)
    ch4_comparison = candidate_comparison(raw, times, CH4_CANDIDATES, "CH4", decisions)
    nee_relationships, ch4_relationships = relationship_checks(raw)
    qc_values, qc_rules, waterfall, qc_masks = qc_outputs(raw, times)
    environment = environment_outputs(raw, times)
    gcc_daily, season_boundaries, phenology = phenology_outputs(raw, times)
    mappings = mapping_table(decisions, unresolved_core, unresolved_extension)
    provenance, inventory = provenance_tables()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_csv(column_audit, OUTPUT_DIR / "column_audit.csv")
    atomic_csv(missingness, OUTPUT_DIR / "missingness_summary.csv")
    atomic_csv(duplicate_columns, OUTPUT_DIR / "duplicate_column_candidates.csv")
    atomic_json(OUTPUT_DIR / "timestamp_audit.json", timestamp_audit)
    atomic_csv(duplicate_rows, OUTPUT_DIR / "duplicate_timestamp_rows.csv")
    atomic_csv(daily_counts, OUTPUT_DIR / "daily_record_counts.csv", date_format="%Y-%m-%d")
    atomic_csv(missing_intervals, OUTPUT_DIR / "missing_interval_summary.csv")
    atomic_csv(nee_comparison, OUTPUT_DIR / "nee_candidate_comparison.csv")
    atomic_csv(nee_relationships, OUTPUT_DIR / "nee_relationship_checks.csv")
    atomic_csv(ch4_comparison, OUTPUT_DIR / "ch4_candidate_comparison.csv")
    atomic_csv(ch4_relationships, OUTPUT_DIR / "ch4_relationship_checks.csv")
    atomic_csv(qc_values, OUTPUT_DIR / "qc_value_summary.csv")
    atomic_csv(qc_rules, OUTPUT_DIR / "qc_rule_comparison.csv")
    atomic_csv(waterfall, OUTPUT_DIR / "exclusion_waterfall.csv")
    atomic_csv(environment, OUTPUT_DIR / "environmental_variable_comparison.csv")
    atomic_csv(gcc_daily, OUTPUT_DIR / "gcc_daily_audit.csv", date_format="%Y-%m-%d")
    atomic_csv(season_boundaries, OUTPUT_DIR / "season_boundaries_by_year.csv", date_format="%Y-%m-%d")
    atomic_json(OUTPUT_DIR / "phenology_audit.json", phenology)
    atomic_csv(mappings, OUTPUT_DIR / "canonical_variable_mapping.csv")
    atomic_csv(provenance, OUTPUT_DIR / "provenance_sources.csv")
    atomic_csv(inventory, OUTPUT_DIR / "repository_context_inventory.csv")

    standardization: list[dict[str, Any]] | None = None
    contract, core_contract, extension_contract = contract_dicts(
        manifest, mappings, decisions, unresolved_core, unresolved_extension,
        qc_rules, phenology, timestamp_audit, environment, standardization,
    )
    if core_contract["contract_status"] == "FROZEN":
        standardization = build_official_core_table(raw, times, core_contract, gcc_daily)
        core_contract["standardization"]["parameters"] = standardization
        contract["standardization"]["parameters"] = standardization
    if extension_contract["contract_status"] == "FROZEN":
        build_official_extension_table(gcc_daily)
    # Write a complete provisional manifest before the tests; it is refreshed
    # from a second checksum after the test process returns.
    manifest["sha256_after"] = before_hash
    manifest["raw_unchanged"] = True
    atomic_json(OUTPUT_DIR / "raw_file_manifest.json", manifest)
    atomic_json(CONTRACT_PATH, contract)
    atomic_json(CORE_CONTRACT_PATH, core_contract)
    atomic_json(EXTENSION_CONTRACT_PATH, extension_contract)

    use_pytest = importlib.util.find_spec("pytest") is not None
    validation_command_display = (
        "python -m pytest -q tests/test_data_contract_v1.py"
        if use_pytest else "python -m unittest tests/test_data_contract_v1.py"
    )
    preliminary_test = {"status": "NOT_RUN", "command": validation_command_display}
    preliminary_immutability = {"unchanged_file_count": 0, "changed_files": []}
    write_report(
        contract, manifest, timestamp_audit, nee_comparison, ch4_comparison,
        nee_relationships, ch4_relationships, qc_rules, waterfall, environment,
        phenology, mappings, provenance, preliminary_test, preliminary_immutability,
    )
    write_targeted_resolution_report(
        contract, core_contract, extension_contract, timestamp_audit, qc_rules,
        preliminary_test,
    )

    test_command = (
        [sys.executable, "-m", "pytest", "-q", "tests/test_data_contract_v1.py"]
        if use_pytest else [sys.executable, "-m", "unittest", "tests/test_data_contract_v1.py"]
    )
    test_process = subprocess.run(test_command, cwd=PROJECT_ROOT, text=True, capture_output=True)
    test_output = "\n".join(part.strip() for part in [test_process.stdout, test_process.stderr] if part.strip())
    test_result = {
        "command": validation_command_display,
        "status": "PASS" if test_process.returncode == 0 else "FAIL",
        "return_code": test_process.returncode,
        "summary": test_output.splitlines()[-1] if test_output else "no output",
        "output": test_output,
    }
    atomic_json(OUTPUT_DIR / "validation_test_result.json", test_result)

    after_hash = sha256(RAW_PATH)
    manifest["sha256_after"] = after_hash
    manifest["raw_unchanged"] = before_hash == after_hash
    if not manifest["raw_unchanged"]:
        for layer in [contract, core_contract, extension_contract]:
            layer["contract_status"] = "FAILED_VALIDATION"
            layer["validation_failure"] = "Raw checksum changed during audit"
    if test_process.returncode != 0:
        for layer in [contract, core_contract, extension_contract]:
            layer["contract_status"] = "FAILED_VALIDATION"
            layer["validation_failure"] = test_output
        if HALFHOURLY_PATH.exists(): HALFHOURLY_PATH.unlink()
        if DAILY_PHENOLOGY_PATH.exists(): DAILY_PHENOLOGY_PATH.unlink()

    legacy_after = legacy_snapshot()
    changed = sorted(path for path, fingerprint in legacy_before.items() if legacy_after.get(path) != fingerprint)
    immutability = {
        "protected_file_count_before": len(legacy_before),
        "unchanged_file_count": len(legacy_before) - len(changed),
        "changed_files": changed,
        "check": "size and nanosecond modification time before versus after audit",
    }
    if changed:
        for layer in [contract, core_contract, extension_contract]:
            layer["contract_status"] = "FAILED_VALIDATION"
            layer["validation_failure"] = f"Legacy protected files changed: {changed}"
    atomic_json(OUTPUT_DIR / "legacy_immutability_check.json", immutability)
    atomic_json(OUTPUT_DIR / "raw_file_manifest.json", manifest)
    atomic_json(CONTRACT_PATH, contract)
    atomic_json(CORE_CONTRACT_PATH, core_contract)
    atomic_json(EXTENSION_CONTRACT_PATH, extension_contract)
    write_report(
        contract, manifest, timestamp_audit, nee_comparison, ch4_comparison,
        nee_relationships, ch4_relationships, qc_rules, waterfall, environment,
        phenology, mappings, provenance, test_result, immutability,
    )
    write_targeted_resolution_report(
        contract, core_contract, extension_contract, timestamp_audit, qc_rules,
        test_result,
    )
    write_freeze_report(
        contract, core_contract, extension_contract, timestamp_audit, qc_rules,
        waterfall, gcc_daily, season_boundaries, test_result, immutability,
    )

    if contract["contract_status"] == "FAILED_VALIDATION":
        raise RuntimeError(contract.get("validation_failure", "Validation failed"))
    if args.freeze and core_contract["contract_status"] != "FROZEN" and extension_contract["contract_status"] != "FROZEN":
        raise SystemExit(
            "Freeze refused: unresolved core decisions: " +
            ", ".join(core_contract["unresolved_critical_decisions"]) +
            "; unresolved extension decisions: " +
            ", ".join(extension_contract["unresolved_critical_decisions"])
        )
    print(f"Contract status: {contract['contract_status']}")
    print(f"Core contract status: {core_contract['contract_status']}")
    print(f"Phenology/environment contract status: {extension_contract['contract_status']}")
    print(f"Raw SHA-256 unchanged: {manifest['raw_unchanged']}")
    print(f"Tests: {test_result['status']} ({test_result['summary']})")
    print(f"Report: {relative(REPORT_PATH)}")
    print(f"Contract: {relative(CONTRACT_PATH)}")


if __name__ == "__main__":
    main()
