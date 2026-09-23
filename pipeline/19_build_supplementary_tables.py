#!/usr/bin/env python3
"""Build the publication-ready workbook for Supplementary Tables S1-S4.

The numerical source is created by 18_make_supplementary_figures_and_sources.R.
This standard Python implementation deliberately uses only the public
``xlsxwriter`` package declared in environment.yml.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import xlsxwriter


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = ROOT / "outputs" / "complete_dataset" / "manuscript" / "canonical_supplementary_package"
SOURCE_PATH = PACKAGE_DIR / "supplementary_tables_source.json"
OUTPUT_PATH = PACKAGE_DIR / "Supplementary_Tables.xlsx"


def humanize(value: str) -> str:
    """Convert stable machine column names into publication-facing labels."""
    labels = {
        "converged_attempted_starts": "Converged/attempted starts",
        "representative_recovery_percent": "Representative recovery (%)",
        "centroid_variability_RMS_SD": "Centroid variability (RMS SD)",
        "carbon_flux_signature": "Carbon-flux signature",
        "mean_NEE": "Mean NEE (µmol CO2 m−2 s−1)",
        "mean_CH4_flux": "Mean CH4 flux (nmol CH4 m−2 s−1)",
        "nearest_centroid_distance": "Nearest-centroid distance",
        "training_observations": "Training observations",
        "held_out_observations": "Held-out observations",
        "converged_starts": "Converged starts (of 5)",
        "held_out_assignment_ARI": "Held-out assignment ARI",
        "held_out_assignment_NMI": "Held-out assignment NMI",
        "same_state_fraction": "Same-state fraction",
        "attribution_branch": "Attribution branch",
        "GWP_horizon": "GWP horizon",
        "full_sample_ratio": "Full-sample ratio",
        "leave_one_year_out_range": "Leave-one-year-out range",
    }
    if value in labels:
        return labels[value]
    replacements = {
        "ari": "ARI", "nmi": "NMI", "bic": "BIC", "aic": "AIC",
        "ch4": "CH4", "co2": "CO2", "gwp100": "GWP100",
        "gwp20": "GWP20", "wl": "WL", "nee": "NEE", "mds": "MDS",
        "rf": "RF", "k4": "K4", "k6": "K6",
    }
    return " ".join(replacements.get(word.lower(), word) for word in value.replace("_", " ").split())


def number_format(column: str) -> str:
    """Choose an Excel number format from the semantic role of a column."""
    if column.lower() in {"k", "year", "omitted_year"}:
        return "0"
    if column == "representative_recovery_percent":
        return "0.0"
    if column in {"mean_pairwise_ARI", "start_stability_ARI", "centroid_variability_RMS_SD"}:
        return "0.0000"
    if column == "full_sample_ratio":
        return "0.00"
    lower = column.lower()
    if any(token in lower for token in ("aic", "bic", "count", "observations", "halfhours", "days", "n_dates", "training", "test", "starts")):
        return "#,##0"
    if any(token in lower for token in ("fraction", "occupancy", "certainty", "recovery", "ari", "nmi", "probability", "ratio", "gain", "brier", "displacement", "distance", "spread")):
        return "0.000"
    return "0.00"


def write_table_sheet(
    workbook: xlsxwriter.Workbook,
    name: str,
    title: str,
    note: str,
    rows: list[dict[str, Any]],
    widths: dict[int, float] | None = None,
) -> None:
    """Write one consistently formatted supplementary-table worksheet."""
    sheet = workbook.add_worksheet(name)
    sheet.hide_gridlines(2)
    navy = "#17365D"
    blue = "#D9EAF7"
    pale_blue = "#EEF5FB"
    grey = "#D9E1F2"
    border = "#C8D0D9"
    title_format = workbook.add_format({"bg_color": navy, "font_color": "#FFFFFF", "bold": True, "font_name": "Arial", "font_size": 14, "valign": "vcenter"})
    note_format = workbook.add_format({"bg_color": pale_blue, "font_color": "#222222", "italic": True, "font_name": "Arial", "font_size": 9, "text_wrap": True, "valign": "vcenter", "border": 1, "border_color": border})
    section_format = workbook.add_format({"bg_color": grey, "font_color": "#222222", "bold": True, "font_name": "Arial", "font_size": 10, "valign": "vcenter"})
    header_format = workbook.add_format({"bg_color": blue, "font_color": "#222222", "bold": True, "font_name": "Arial", "font_size": 9, "align": "center", "valign": "vcenter", "text_wrap": True, "border": 1, "border_color": border})
    text_format = workbook.add_format({"font_color": "#222222", "font_name": "Arial", "font_size": 9, "valign": "vcenter", "border": 1, "border_color": "#E2E8EE"})
    numeric_formats: dict[str, xlsxwriter.format.Format] = {}

    columns = list(rows[0]) if rows else []
    end_column = max(len(columns) - 1, 0)
    sheet.merge_range(0, 0, 0, max(end_column, 7), title, title_format)
    sheet.set_row(0, 28)
    sheet.merge_range(1, 0, 2, max(end_column, 7), note, note_format)
    sheet.set_row(1, 18)
    sheet.set_row(2, 18)
    section_title = {
        "Table S1": "Candidate-model comparison",
        "Table S2": "Canonical four-state carbon-flux representation",
        "Table S3": "K4 leave-one-year-out results",
        "Table S4": "Greenhouse-gas accounting sensitivity",
    }[name]
    sheet.merge_range(4, 0, 4, end_column, section_title, section_format)
    sheet.set_row(4, 21)
    for column_index, column in enumerate(columns):
        sheet.write(5, column_index, humanize(column), header_format)
    sheet.set_row(5, 30)
    for row_index, row in enumerate(rows, start=6):
        for column_index, column in enumerate(columns):
            value = row.get(column)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                fmt = numeric_formats.setdefault(column, workbook.add_format({"font_color": "#222222", "font_name": "Arial", "font_size": 9, "align": "right", "valign": "vcenter", "border": 1, "border_color": "#E2E8EE", "num_format": number_format(column)}))
                sheet.write_number(row_index, column_index, value, fmt)
            elif value is None:
                sheet.write_blank(row_index, column_index, None, text_format)
            else:
                sheet.write(row_index, column_index, value, text_format)
        sheet.set_row(row_index, 18)
    for column_index, column in enumerate(columns):
        longest = max([len(humanize(column)), *[len(str(row.get(column, ""))) for row in rows]], default=10)
        sheet.set_column(column_index, column_index, min(max(longest + 2, 10), 30))
    for column_index, width in (widths or {}).items():
        sheet.set_column(column_index, column_index, width)
    sheet.freeze_panes(3, 0)


def write_readme(workbook: xlsxwriter.Workbook) -> None:
    """Write workbook scope and interpretation boundaries."""
    sheet = workbook.add_worksheet("Read me")
    sheet.hide_gridlines(2)
    title = workbook.add_format({"bg_color": "#17365D", "font_color": "#FFFFFF", "bold": True, "font_name": "Arial", "font_size": 14, "valign": "vcenter"})
    label = workbook.add_format({"bg_color": "#D9EAF7", "bold": True, "font_name": "Arial", "valign": "vcenter", "text_wrap": True, "border": 1, "border_color": "#C8D0D9"})
    text = workbook.add_format({"font_name": "Arial", "valign": "vcenter", "text_wrap": True, "border": 1, "border_color": "#C8D0D9"})
    sheet.merge_range("A1:H1", "Supplementary tables — Beyond canopy phenology", title)
    sheet.set_row(0, 30)
    rows = [
        ("Purpose", "Publication-ready numerical tables supporting the HMM selection and greenhouse-gas accounting."),
        ("Table S1", "Candidate-model selection and stability."),
        ("Table S2", "Definitions and diagnostic properties of the canonical carbon-flux states."),
        ("Table S3", "K4 leave-one-year-out generalizability."),
        ("Table S4", "GWP100 and GWP20 sensitivity of state reorganization relative to phenophase duration."),
        ("Interpretation", "All tables use frozen state assignments. They describe reproducibility, conditional prediction, or descriptive accounting; they do not identify causal mechanisms."),
        ("Source", "Generated from frozen publication inputs by scripts/pipeline/18_make_supplementary_figures_and_sources.R."),
    ]
    for row_index, (key, value) in enumerate(rows, start=2):
        sheet.write(row_index, 0, key, label)
        sheet.write(row_index, 1, value, text)
        sheet.set_row(row_index, 28)
    sheet.set_column("A:A", 16)
    sheet.set_column("B:B", 95)


def main() -> None:
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    expected_tables = {f"Table_S{i}" for i in range(1, 5)}
    if set(source) != expected_tables:
        raise RuntimeError(
            f"Expected {sorted(expected_tables)} in {SOURCE_PATH}, found {sorted(source)}"
        )
    if any(not source[table] for table in expected_tables):
        raise RuntimeError(f"At least one supplementary table is empty in {SOURCE_PATH}")

    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    # Build beside the destination and replace it only after XlsxWriter closes
    # successfully, so interruption cannot corrupt the last valid workbook.
    temporary_path = OUTPUT_PATH.with_name(f".{OUTPUT_PATH.stem}.tmp{OUTPUT_PATH.suffix}")
    temporary_path.unlink(missing_ok=True)
    workbook = xlsxwriter.Workbook(temporary_path)
    workbook.set_properties({"title": "Supplementary Tables S1-S4", "author": "Vargas et al."})
    write_readme(workbook)
    write_table_sheet(workbook, "Table S1", "Table S1. Candidate-model selection and stability", "Lower AIC and BIC indicate better fit; higher pairwise ARI and representative recovery and lower centroid variability indicate greater reproducibility. K6 had the best information criteria but a less reproducible detailed partition, whereas K4 was effectively identical across starts and was retained as the stable broad representation.", source["Table_S1"])
    write_table_sheet(workbook, "Table S2", "Table S2. Definitions and diagnostic properties of the canonical carbon-flux states", "Negative NEE denotes net CO2 uptake. Occupancy is the proportion of 63,156 eligible observations assigned to each CFS. Posterior certainty is the mean posterior probability of the assigned state; self-transition probability describes persistence between consecutive eligible 30-min observations. Distances are Euclidean in standardized NEE-CH4 space.", source["Table_S2"], {1: 42})
    write_table_sheet(workbook, "Table S3", "Table S3. Leave-one-year-out generalizability of the four-state representation", "For each fold, K4 was fitted to five years using five starts and applied to the omitted year after centroid alignment with the primary states. Held-out ARI and same-state fraction compare retrained-model assignments with primary-model assignments; values approaching one indicate greater agreement.", source["Table_S3"], {1: 22})
    write_table_sheet(workbook, "Table S4", "Table S4. Sensitivity of carbon-flux-state reorganization relative to phenophase duration", "Ratios compare median absolute CFS-reorganization contributions (occupancy plus within-state intensity) with duration contributions; values greater than one indicate a larger typical CFS-reorganization contribution. Results use the primary MDS-state/MDS-budget attribution. GWP100 and GWP20 use AR6 non-fossil CH4 factors of 27.0 and 79.7.", source["Table_S4"], {3: 24})
    workbook.close()
    temporary_path.replace(OUTPUT_PATH)
    print(f"Created {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
