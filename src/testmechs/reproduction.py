"""Unified paper reproduction comparison and E2E audit module.

This module combines empirical and Monte Carlo reproduction evidence into
a single structured report that compares Python ``testmechs`` results
against published paper statistics.  It provides:

- A comparison report with per-row pass/fail verdicts
- Data-provenance rows recording data sources and filtering rules
- Tolerance contracts explaining the acceptance criteria
- An end-to-end (E2E) audit layer with milestone and archive-boundary checks
- JSON persistence (write/load) for all report types
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
import json
import math
from pathlib import Path
import re
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from ._json_io import write_strict_json_atomic as _write_reproduction_json_atomic
from .empirical import paper_empirical_reproduction_report
from .monte_carlo import (
    load_paper_empirical_mixture_benchmark_data_sources,
    milestone_audit_status_for_version,
    milestone_audit_status_from_file,
    paper_monte_carlo_reproduction_report,
)
from .results import _is_scalar_missing, _reject_nonfinite_json_numbers


_PAPER_ACCEPTANCE_CONTRACT_MINIMA = {
    "paper_result_rows": 258,
    "scheduled_draws": 129000,
    "scheduled_bootstrap_draws": 36000000,
}
_PAPER_ACCEPTANCE_ABSOLUTE_TOLERANCE = 0.025
_PAPER_ACCEPTANCE_Z_TOLERANCE = 2.0
_PAPER_ACCEPTANCE_REPLICATIONS = 500
_PAPER_ACCEPTANCE_MAX_TARGET_MC_STANDARD_ERROR = math.sqrt(
    0.25 / _PAPER_ACCEPTANCE_REPLICATIONS
)
_REQUIRED_PAPER_REPRODUCTION_TOLERANCE_CONTRACTS = (
    "paper_rounded_percent_tolerance",
    "paper_full_suite_coverage_contract",
    "paper_default_schedule_contract",
)
_REQUIRED_PAPER_REPRODUCTION_COMPARISON_ROWS = 7
_REQUIRED_PAPER_REPRODUCTION_PROVENANCE_ROWS = 8
_REQUIRED_COMPARISON_TEXT_FIELDS = (
    "case_id",
    "report_family",
    "comparison_kind",
    "source_report",
    "row_type",
    "comparison_category",
    "metric",
    "paper_anchor",
    "reference_anchor",
    "truth_hierarchy",
)
_REQUIRED_COMPARISON_TARGET_VALUE_FIELDS = (
    "paper_value",
    "python_value",
    "tolerance_name",
)
_REQUIRED_COMPARISON_TARGET_NUMERIC_FIELDS = (
    "absolute_difference",
    "tolerance_value",
)
_REQUIRED_COMPARISON_ACCEPTED_CATEGORIES = (
    "exact_reproduction",
    "sampling_error_reproduction",
)
_COMPARISON_ACCEPTANCE_TOLERANCE_EPSILON = 1e-12
_REQUIRED_PROVENANCE_TEXT_FIELDS = (
    "provenance_key",
    "provenance_type",
    "provenance_category",
    "source_kind",
    "data_source",
    "filtering_rules",
    "paper_anchor",
    "reference_anchor",
)
_REQUIRED_PROVENANCE_NUMERIC_FIELDS = (
    "row_count",
    "complete_case_rows",
)
_REQUIRED_PROVENANCE_SUPPORT_FIELDS = (
    "treatment_levels",
    "mediator_levels",
)
_REQUIRED_TOLERANCE_CONTRACT_TEXT_FIELDS = {
    "paper_rounded_percent_tolerance": (
        "applies_to",
        "scale",
        "basis",
        "paper_anchor",
        "reference_anchor",
    ),
    "paper_full_suite_coverage_contract": (
        "applies_to",
        "scale",
        "basis",
        "paper_anchor",
        "reference_anchor",
    ),
    "paper_default_schedule_contract": (
        "applies_to",
        "scale",
        "basis",
        "paper_anchor",
        "reference_anchor",
    ),
}
_REQUIRED_TOLERANCE_CONTRACT_NUMERIC_FIELDS = {
    "paper_rounded_percent_tolerance": {
        "threshold",
    },
    "paper_full_suite_coverage_contract": {
        "threshold",
        "z_tolerance",
        "configured_rejection_rate_absolute_tolerance",
        "paper_nominal_alpha",
    },
    "paper_default_schedule_contract": {
        "paper_default_replications",
        "paper_default_bootstrap_replications",
    },
}
_REQUIRED_TOLERANCE_CONTRACT_NUMERIC_VALUES = {
    "paper_rounded_percent_tolerance": {"threshold": 0.005},
    "paper_full_suite_coverage_contract": {
        "threshold": 0.0,
        "z_tolerance": _PAPER_ACCEPTANCE_Z_TOLERANCE,
        "paper_nominal_alpha": 0.05,
    },
    "paper_default_schedule_contract": {
        "paper_default_replications": 500.0,
        "paper_default_bootstrap_replications": 500.0,
    },
}


@dataclass(frozen=True)
class PaperReproductionComparisonReport:
    """Unified paper reproduction comparison and provenance packet."""

    comparison_rows: tuple[dict[str, Any], ...]
    provenance_rows: tuple[dict[str, Any], ...]
    tolerance_rows: tuple[dict[str, Any], ...]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_export_payload(
            {
                "summary": dict(self.summary),
                "comparison_rows": [dict(row) for row in self.comparison_rows],
                "provenance_rows": [dict(row) for row in self.provenance_rows],
                "tolerance_rows": [dict(row) for row in self.tolerance_rows],
            }
        )

    @property
    def rows(self) -> tuple[dict[str, Any], ...]:
        return self.comparison_rows

    def to_frame(self) -> pd.DataFrame:
        return self.comparison_frame()

    def comparison_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            _json_safe_export_payload([dict(row) for row in self.comparison_rows]),
            dtype=object,
        )

    def provenance_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            _json_safe_export_payload([dict(row) for row in self.provenance_rows]),
            dtype=object,
        )

    def tolerance_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            _json_safe_export_payload([dict(row) for row in self.tolerance_rows]),
            dtype=object,
        )

    def display_frame(self) -> pd.DataFrame:
        """Return a compact human-readable paper comparison table."""

        return _paper_reproduction_comparison_display_frame(self.comparison_rows)


@dataclass(frozen=True)
class PaperReproductionE2EReport:
    """Public E2E reproduction packet with archive-boundary audit rows."""

    rows: tuple[dict[str, Any], ...]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_export_payload(
            {
                "summary": dict(self.summary),
                "rows": [dict(row) for row in self.rows],
            }
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            _json_safe_export_payload([dict(row) for row in self.rows]),
            dtype=object,
        )

    def display_frame(self) -> pd.DataFrame:
        """Return a compact human-readable E2E reproduction table."""

        return _paper_reproduction_e2e_display_frame(self.rows)


def paper_reproduction_comparison_report(
    evidence_dir: str | Path,
    *,
    fixtures_dir: str | Path | None = None,
    statistics_dir: str | Path | None = None,
    tables_dir: str | Path | None = None,
    **monte_carlo_kwargs: Any,
) -> PaperReproductionComparisonReport:
    """Build a unified paper-vs-Python comparison and provenance report.

    Combines empirical lower-bound reproduction evidence and Monte Carlo
    rejection-rate reproduction evidence into a single report with
    comparison rows, provenance rows, and tolerance contracts.

    Parameters
    ----------
    evidence_dir : str or Path
        Directory containing Monte Carlo evidence JSON files.
    fixtures_dir : str, Path, or None
        Directory containing CSV test fixtures.
    statistics_dir : str, Path, or None
        Directory containing LaTeX statistic files from the paper.
    tables_dir : str, Path, or None
        Directory containing LaTeX Monte Carlo tables.
    **monte_carlo_kwargs
        Additional keyword arguments forwarded to
        :func:`~testmechs.monte_carlo.paper_monte_carlo_reproduction_report`.

    Returns
    -------
    PaperReproductionComparisonReport
        Structured report with comparison rows, provenance, and tolerances.
    """

    started_at = perf_counter()
    evidence_path = Path(evidence_dir)
    fixtures = _default_fixture_inputs_dir() if fixtures_dir is None else Path(fixtures_dir)
    empirical_report = paper_empirical_reproduction_report(
        fixtures_dir=fixtures,
        statistics_dir=statistics_dir,
    )
    monte_carlo_report = paper_monte_carlo_reproduction_report(
        evidence_path,
        tables_dir=tables_dir,
        fixtures_dir=fixtures,
        **monte_carlo_kwargs,
    )
    comparison_rows = (
        *_empirical_comparison_rows(empirical_report.rows),
        *_monte_carlo_comparison_rows(
            monte_carlo_report.rows,
            summary=monte_carlo_report.summary,
        ),
    )
    provenance_rows = (
        *_empirical_provenance_rows(empirical_report.rows),
        *_monte_carlo_provenance_rows(fixtures),
    )
    tolerance_rows = _tolerance_rows(
        empirical_summary=empirical_report.summary,
        monte_carlo_summary=monte_carlo_report.summary,
        monte_carlo_kwargs=monte_carlo_kwargs,
    )
    summary = _comparison_summary(
        comparison_rows,
        provenance_rows=provenance_rows,
        tolerance_rows=tolerance_rows,
        empirical_summary=empirical_report.summary,
        monte_carlo_summary=monte_carlo_report.summary,
        runtime_seconds=float(perf_counter() - started_at),
    )
    portable_roots = _portable_report_roots(
        evidence_dir=evidence_path,
        fixtures_dir=fixtures,
        statistics_dir=statistics_dir,
        tables_dir=tables_dir,
    )
    comparison_rows = tuple(
        _portable_report_value(row, roots=portable_roots) for row in comparison_rows
    )
    provenance_rows = tuple(
        _portable_report_value(row, roots=portable_roots) for row in provenance_rows
    )
    tolerance_rows = tuple(
        _portable_report_value(row, roots=portable_roots) for row in tolerance_rows
    )
    summary = _portable_report_value(summary, roots=portable_roots)
    return PaperReproductionComparisonReport(
        comparison_rows=comparison_rows,
        provenance_rows=provenance_rows,
        tolerance_rows=tolerance_rows,
        summary=summary,
    )


def paper_reproduction_comparison_report_frame(
    evidence_dir: str | Path,
    **kwargs: Any,
) -> pd.DataFrame:
    """Return the unified paper reproduction comparison rows as a DataFrame.

    Parameters
    ----------
    evidence_dir : str or Path
        Directory containing Monte Carlo evidence JSON files.
    **kwargs
        Forwarded to :func:`paper_reproduction_comparison_report`.

    Returns
    -------
    pd.DataFrame
        Comparison rows with one row per paper-vs-Python metric.
    """

    return paper_reproduction_comparison_report(evidence_dir, **kwargs).comparison_frame()


def paper_reproduction_comparison_display_frame(
    evidence_dir: str | Path,
    **kwargs: Any,
) -> pd.DataFrame:
    """Return a compact human-readable paper comparison table.

    Parameters
    ----------
    evidence_dir : str or Path
        Directory containing Monte Carlo evidence JSON files.
    **kwargs
        Forwarded to :func:`paper_reproduction_comparison_report`.

    Returns
    -------
    pd.DataFrame
        Display-formatted table with status, source, category, metric, and
        tolerance columns.
    """

    return paper_reproduction_comparison_report(evidence_dir, **kwargs).display_frame()


_PAPER_REPRODUCTION_COMPARISON_DISPLAY_COLUMNS = [
    "status",
    "source",
    "category",
    "metric",
    "python_value",
    "paper_value",
    "gap",
    "tolerance",
    "evidence",
    "case_id",
]


def _paper_reproduction_comparison_display_frame(
    comparison_rows: tuple[dict[str, Any], ...],
) -> pd.DataFrame:
    display_rows = [
        _paper_reproduction_comparison_display_row(row)
        for row in comparison_rows
    ]
    return pd.DataFrame(
        display_rows,
        columns=_PAPER_REPRODUCTION_COMPARISON_DISPLAY_COLUMNS,
        dtype=object,
    )


def _paper_reproduction_comparison_display_row(row: dict[str, Any]) -> dict[str, Any]:
    missing_fields = [
        field
        for field in (
            "case_id",
            "source_report",
            "comparison_category",
            "metric",
        )
        if field not in row
    ]
    if missing_fields:
        raise ValueError(
            "Paper reproduction comparison display rows are missing required fields: "
            + ", ".join(missing_fields)
        )
    return {
        "status": _comparison_status_label(row["comparison_category"]),
        "source": _comparison_source_label(row["source_report"]),
        "category": _format_reproduction_display_text(
            str(row["comparison_category"]).replace("_", " "),
            field="category",
        ),
        "metric": _format_comparison_display_label(row["metric"], field="metric"),
        "python_value": _format_comparison_display_measure(row.get("python_value")),
        "paper_value": _format_comparison_display_measure(row.get("paper_value")),
        "gap": _format_comparison_display_measure(row.get("absolute_difference")),
        "tolerance": _format_comparison_display_measure(row.get("tolerance_value")),
        "evidence": _comparison_evidence_label(row),
        "case_id": _format_comparison_display_label(row["case_id"], field="case_id"),
    }


def _comparison_status_label(comparison_category: Any) -> str:
    category = _format_reproduction_display_text(
        comparison_category,
        field="comparison_category",
    )
    if category in {"exact_reproduction", "sampling_error_reproduction"}:
        return "PASS"
    if category == "documented_exception":
        return "REFERENCE"
    if category == "missing_data_or_source_limitation":
        return "BLOCKED"
    return category.upper().replace("_", " ")


def _comparison_source_label(source_report: Any) -> str:
    source = _format_reproduction_display_text(source_report, field="source_report")
    if source == "paper_empirical_reproduction_report":
        return "empirical"
    if source == "paper_monte_carlo_reproduction_report":
        return "monte_carlo"
    return _format_comparison_display_label(source, field="source_report")


_COMPARISON_DISPLAY_LABEL_MAX_CHARS = 96
_COMPARISON_DISPLAY_LABEL_TAIL_CHARS = 40
_COMPARISON_DISPLAY_VALUE_MAX_CHARS = 160
_COMPARISON_DISPLAY_VALUE_TAIL_CHARS = 60


def _format_comparison_display_label(value: Any, *, field: str) -> str:
    text = _format_reproduction_display_text(value, field=field)
    if len(text) <= _COMPARISON_DISPLAY_LABEL_MAX_CHARS:
        return text
    head_chars = (
        _COMPARISON_DISPLAY_LABEL_MAX_CHARS
        - _COMPARISON_DISPLAY_LABEL_TAIL_CHARS
        - 3
    )
    return f"{text[:head_chars]}...{text[-_COMPARISON_DISPLAY_LABEL_TAIL_CHARS:]}"


def _format_reproduction_display_text(value: Any, *, field: str) -> str:
    if _is_reproduction_missing_display_value(value):
        raise ValueError(f"Paper reproduction display field {field} is missing")
    text = str(value).strip()
    if not text:
        raise ValueError(f"Paper reproduction display field {field} is blank")
    return text


def _format_comparison_display_value(value: Any) -> str:
    if value is None:
        return ""
    if _is_reproduction_missing_display_value(value):
        return "NA"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric):
            if math.isnan(numeric):
                return "NA"
            return "+Inf" if numeric > 0 else "-Inf"
        if numeric.is_integer():
            return str(int(numeric))
        return f"{numeric:.6f}"
    if isinstance(value, Mapping):
        return _compact_comparison_display_value(
            json.dumps(
                _json_safe_export_payload(dict(value)),
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    if isinstance(value, (list, tuple, np.ndarray)):
        return _compact_comparison_display_value(
            json.dumps(
                _json_safe_export_payload(value),
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return str(value)


def _compact_comparison_display_value(text: str) -> str:
    if len(text) <= _COMPARISON_DISPLAY_VALUE_MAX_CHARS:
        return text
    head_chars = (
        _COMPARISON_DISPLAY_VALUE_MAX_CHARS
        - _COMPARISON_DISPLAY_VALUE_TAIL_CHARS
        - 3
    )
    return f"{text[:head_chars]}...{text[-_COMPARISON_DISPLAY_VALUE_TAIL_CHARS:]}"


def _format_comparison_display_measure(value: Any) -> str:
    if value is None:
        return ""
    if _is_reproduction_missing_display_value(value):
        return "NA"
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"Paper reproduction comparison display measure is boolean: {value!r}"
        )
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric):
            if math.isnan(numeric):
                return "NA"
            return "+Inf" if numeric > 0 else "-Inf"
        return f"{numeric:.6f}"
    return _format_comparison_display_value(value)


def _comparison_evidence_label(row: dict[str, Any]) -> str:
    if row.get("sample_size") is not None:
        sample = f"n={_format_optional_int(row['sample_size'])}"
        if row.get("n_treated") is not None and row.get("n_control") is not None:
            sample += (
                f" (T={_format_optional_int(row['n_treated'])}, "
                f"C={_format_optional_int(row['n_control'])})"
            )
        return sample
    covered = row.get("covered_result_rows")
    paper_rows = row.get("paper_result_rows")
    if covered is not None and paper_rows is not None:
        return (
            "covered_rows="
            f"{_format_optional_int(covered)}/{_format_optional_int(paper_rows)}"
        )
    scheduled_draws = row.get("scheduled_draws")
    if scheduled_draws is not None:
        return f"scheduled_draws={_format_optional_int(scheduled_draws)}"
    return "evidence unavailable"


def paper_reproduction_e2e_report(
    evidence_dir: str | Path,
    *,
    milestone_version: str = "v1.2",
    roadmap_analysis: dict[str, Any] | None = None,
    requirements_analysis: dict[str, Any] | None = None,
    requirements_path: str | Path | None = None,
    milestone_audit_status: str | None = None,
    milestone_audit_path: str | Path | None = None,
    planning_dir: str | Path = ".planning",
    **kwargs: Any,
) -> PaperReproductionE2EReport:
    """Return the public paper-reproduction E2E and archive-boundary report.

    Extends the comparison report with milestone-audit and archive-boundary
    checks to produce a complete submission-readiness assessment.

    Parameters
    ----------
    evidence_dir : str or Path
        Directory containing Monte Carlo evidence JSON files.
    milestone_version : str
        Target milestone version (e.g. ``"v1.2"``).
    roadmap_analysis : dict or None
        Pre-computed roadmap analysis; computed if ``None``.
    requirements_analysis : dict or None
        Pre-computed requirements analysis; loaded if ``None``.
    requirements_path : str, Path, or None
        Path to requirements YAML (used when *requirements_analysis* is None).
    milestone_audit_status : str or None
        Pre-computed audit status string; resolved if ``None``.
    milestone_audit_path : str, Path, or None
        Path to audit YAML (used when *milestone_audit_status* is None).
    planning_dir : str or Path
        Path to the planning directory for milestone resolution.
    **kwargs
        Forwarded to :func:`paper_reproduction_comparison_report`.

    Returns
    -------
    PaperReproductionE2EReport
        E2E report with comparison, audit, and archive-boundary rows.
    """

    started_at = perf_counter()
    comparison_report = paper_reproduction_comparison_report(
        evidence_dir,
        **kwargs,
    )
    rows = _public_e2e_rows(
        comparison_report,
        milestone_version=milestone_version,
        roadmap_analysis=roadmap_analysis,
        requirements_analysis=_resolve_public_e2e_requirements_analysis(
            requirements_analysis=requirements_analysis,
            requirements_path=requirements_path,
            milestone_version=milestone_version,
            planning_dir=planning_dir,
        ),
        milestone_audit_status=_resolve_public_e2e_milestone_audit_status(
            milestone_audit_status=milestone_audit_status,
            milestone_audit_path=milestone_audit_path,
            milestone_version=milestone_version,
            planning_dir=planning_dir,
        ),
    )
    summary = _public_e2e_summary(
        rows,
        comparison_summary=comparison_report.summary,
        milestone_version=milestone_version,
        runtime_seconds=float(perf_counter() - started_at),
    )
    return PaperReproductionE2EReport(rows=rows, summary=summary)


def paper_reproduction_e2e_report_frame(
    evidence_dir: str | Path,
    **kwargs: Any,
) -> pd.DataFrame:
    """Return the public paper-reproduction E2E audit rows as a DataFrame.

    Parameters
    ----------
    evidence_dir : str or Path
        Directory containing Monte Carlo evidence JSON files.
    **kwargs
        Forwarded to :func:`paper_reproduction_e2e_report`.

    Returns
    -------
    pd.DataFrame
        E2E audit rows as a DataFrame with object dtype.
    """

    return paper_reproduction_e2e_report(evidence_dir, **kwargs).to_frame()


def paper_reproduction_e2e_display_frame(
    evidence_dir: str | Path,
    **kwargs: Any,
) -> pd.DataFrame:
    """Return a compact human-readable public E2E reproduction table.

    Parameters
    ----------
    evidence_dir : str or Path
        Directory containing Monte Carlo evidence JSON files.
    **kwargs
        Forwarded to :func:`paper_reproduction_e2e_report`.

    Returns
    -------
    pd.DataFrame
        Display-formatted E2E table with status, check, kind, and summary
        columns.
    """

    return paper_reproduction_e2e_report(evidence_dir, **kwargs).display_frame()


_PAPER_REPRODUCTION_E2E_DISPLAY_COLUMNS = [
    "status",
    "check",
    "kind",
    "summary",
    "evidence",
    "next_action",
    "check_id",
]


def _paper_reproduction_e2e_display_frame(
    rows: tuple[dict[str, Any], ...],
) -> pd.DataFrame:
    display_rows = [_paper_reproduction_e2e_display_row(row) for row in rows]
    return pd.DataFrame(
        display_rows,
        columns=_PAPER_REPRODUCTION_E2E_DISPLAY_COLUMNS,
        dtype=object,
    )


def _paper_reproduction_e2e_display_row(row: dict[str, Any]) -> dict[str, Any]:
    missing_fields = [
        field
        for field in ("check_id", "check_kind", "status")
        if field not in row
    ]
    if missing_fields:
        raise ValueError(
            "Paper reproduction E2E display rows are missing required fields: "
            + ", ".join(missing_fields)
        )
    return {
        "status": _e2e_status_label(row["status"]),
        "check": _e2e_check_label(row),
        "kind": _e2e_kind_display_label(row["check_kind"]),
        "summary": _e2e_summary_label(row),
        "evidence": _e2e_evidence_label(row),
        "next_action": _e2e_next_action_label(row),
        "check_id": _format_e2e_display_label(row["check_id"], field="check_id"),
    }


def _e2e_status_label(status: Any) -> str:
    text = _format_reproduction_display_text(status, field="status")
    return _format_e2e_display_label(
        text.upper().replace("_", " "),
        field="status",
    )


def _e2e_kind_value(check_kind: Any) -> str:
    return _format_reproduction_display_text(check_kind, field="check_kind")


def _e2e_kind_display_label(check_kind: Any) -> str:
    return _format_e2e_display_label(
        _e2e_kind_value(check_kind).replace("_", " "),
        field="check_kind",
    )


def _e2e_check_label(row: dict[str, Any]) -> str:
    kind = _e2e_kind_value(row["check_kind"])
    if kind == "empirical_reproduction":
        return "Empirical targets"
    if kind == "monte_carlo_reproduction":
        return "Monte Carlo evidence"
    if kind == "comparison_provenance":
        return "Comparison packet"
    if kind == "milestone_archive_preflight":
        version = row.get("milestone_version")
        if version is not None and not _is_reproduction_missing_display_value(version):
            version_text = _format_reproduction_display_text(
                version, field="milestone_version"
            )
            return _format_e2e_display_label(
                f"{version_text} archive preflight",
                field="check",
            )
        return "Archive preflight"
    return _format_e2e_display_label(
        kind.replace("_", " ").title(),
        field="check",
    )


_E2E_DISPLAY_LABEL_MAX_CHARS = 96
_E2E_DISPLAY_LABEL_TAIL_CHARS = 44
_E2E_DISPLAY_VALUE_MAX_CHARS = 160
_E2E_DISPLAY_VALUE_TAIL_CHARS = 60


def _format_e2e_display_label(value: Any, *, field: str) -> str:
    text = _format_reproduction_display_text(value, field=field)
    if len(text) <= _E2E_DISPLAY_LABEL_MAX_CHARS:
        return text
    head_chars = _E2E_DISPLAY_LABEL_MAX_CHARS - _E2E_DISPLAY_LABEL_TAIL_CHARS - 3
    return f"{text[:head_chars]}...{text[-_E2E_DISPLAY_LABEL_TAIL_CHARS:]}"


def _format_optional_e2e_display_label(value: Any, *, field: str) -> str:
    if value is None:
        return "NA"
    if _is_reproduction_missing_display_value(value):
        return "NA"
    return _format_e2e_display_label(value, field=field)


def _compact_e2e_display_value(text: str) -> str:
    if len(text) <= _E2E_DISPLAY_VALUE_MAX_CHARS:
        return text
    head_chars = (
        _E2E_DISPLAY_VALUE_MAX_CHARS
        - _E2E_DISPLAY_VALUE_TAIL_CHARS
        - 3
    )
    return f"{text[:head_chars]}...{text[-_E2E_DISPLAY_VALUE_TAIL_CHARS:]}"


def _e2e_summary_label(row: dict[str, Any]) -> str:
    kind = _e2e_kind_value(row["check_kind"])
    if kind == "empirical_reproduction":
        targets = _count_pair(row.get("passed_target_rows"), row.get("paper_target_rows"))
        reference_only = row.get("reference_only_rows")
        summary = f"targets {targets} within tolerance"
        if reference_only is not None:
            summary += f"; reference_only={_format_optional_int(reference_only)}"
        return summary
    if kind == "monte_carlo_reproduction":
        coverage = _count_pair(row.get("covered_result_rows"), row.get("paper_result_rows"))
        gate_value = row.get("paper_acceptance_gate_passes")
        gate = (
            "paper acceptance passed"
            if gate_value is True
            else "paper acceptance blocked"
        )
        return f"{gate}; covered_rows={coverage}"
    if kind == "comparison_provenance":
        comparison = row.get("actual_comparison_row_count")
        provenance = row.get("actual_provenance_row_count")
        tolerance = row.get("actual_tolerance_row_count")
        condition_value = row.get("comparison_provenance_condition")
        if condition_value is None or _is_reproduction_missing_display_value(
            condition_value
        ):
            condition = "NA"
        else:
            condition = _format_e2e_display_label(
                str(condition_value).replace("_", " "),
                field="comparison_provenance_condition",
            )
        return (
            f"{condition}; "
            f"comparison={_format_optional_int(comparison)}, "
            f"provenance={_format_optional_int(provenance)}, "
            f"tolerance={_format_optional_int(tolerance)}"
        )
    if kind == "milestone_archive_preflight":
        version = _format_optional_e2e_display_label(
            row.get("milestone_version", "milestone"),
            field="milestone_version",
        )
        ready = row.get("milestone_archive_ready") is True
        return _format_e2e_display_label(
            f"{version} archive {'ready' if ready else 'blocked'}",
            field="summary",
        )
    return ""


def _e2e_evidence_label(row: dict[str, Any]) -> str:
    kind = _e2e_kind_value(row["check_kind"])
    if kind == "empirical_reproduction":
        return _compact_join(
            (
                _threshold_label(
                    "max_diff",
                    row.get("max_absolute_difference"),
                    row.get("tolerance"),
                ),
                f"min_n={_format_optional_int(row.get('min_sample_size'))}",
            )
        )
    if kind == "monte_carlo_reproduction":
        return _compact_join(
            (
                f"draws={_count_pair(row.get('executed_draws'), row.get('scheduled_draws'))}",
                (
                    "bootstrap="
                    f"{_count_pair(row.get('executed_bootstrap_draws'), row.get('scheduled_bootstrap_draws'))}"
                ),
                f"max_error={_format_comparison_display_measure(row.get('max_rejection_rate_absolute_error'))}",
            )
        )
    if kind == "comparison_provenance":
        return _compact_join(
            (
                (
                    "required="
                    f"{_format_optional_int(row.get('required_comparison_row_count'))}/"
                    f"{_format_optional_int(row.get('required_provenance_row_count'))}"
                ),
                (
                    "tolerances="
                    f"{_format_optional_int(row.get('actual_tolerance_row_count'))}"
                ),
                (
                    "duplicates="
                    f"{_format_e2e_sequence_count(row.get('duplicate_comparison_row_ids'))}/"
                    f"{_format_e2e_sequence_count(row.get('duplicate_provenance_row_ids'))}"
                ),
            )
        )
    if kind == "milestone_archive_preflight":
        return _compact_join(
            (
                f"roadmap_complete={row.get('roadmap_disk_complete')}",
                f"requirements_complete={row.get('requirements_complete')}",
                "audit="
                + _format_optional_e2e_display_label(
                    row.get("milestone_audit_status"),
                    field="milestone_audit_status",
                ),
                _blocking_conditions_label(row.get("active_blocking_conditions")),
            )
        )
    return ""


def _format_e2e_sequence_count(value: Any) -> str:
    if value is None:
        return "NA"
    if _is_reproduction_missing_display_value(value):
        return "NA"
    if isinstance(value, (str, bytes)):
        raise ValueError(
            "Paper reproduction E2E display duplicate row ids must be a sequence, "
            f"not a string: {value!r}"
        )
    if isinstance(value, Mapping):
        raise ValueError(
            "Paper reproduction E2E display duplicate row ids must be a sequence, "
            f"not a mapping: {value!r}"
        )
    if isinstance(value, (bool, np.bool_, int, float, np.integer, np.floating)):
        raise ValueError(
            "Paper reproduction E2E display duplicate row ids must be a sequence: "
            f"{value!r}"
        )
    try:
        count = len(value)
    except TypeError as exc:
        raise ValueError(
            "Paper reproduction E2E display duplicate row ids must be a sequence: "
            f"{value!r}"
        ) from exc
    return str(count)


def _e2e_next_action_label(row: dict[str, Any]) -> str:
    actions: list[str] = []
    seen_raw_actions: set[str] = set()
    for key in (
        "evidence_next_action",
        "milestone_audit_resolution_action",
        "next_chunk_rerun_call",
        "next_chunk_export_call",
        "rerun_hook",
    ):
        value = row.get(key)
        if value is not None and not _is_reproduction_missing_display_value(value):
            raw_text = _format_reproduction_display_text(value, field=key)
            if raw_text not in seen_raw_actions:
                text = _format_e2e_next_action_display_text(raw_text, field=key)
                actions.append(text)
                seen_raw_actions.add(raw_text)
    if actions:
        return "; ".join(actions[:2])
    blockers = row.get("active_blocking_conditions")
    if isinstance(blockers, Mapping) and blockers:
        active_keys = _positive_blocking_condition_keys(blockers)
        if active_keys:
            return "resolve " + ", ".join(
                _format_e2e_display_label(key, field="blocking_condition")
                for key in active_keys[:3]
            )
    return ""


_E2E_NEXT_ACTION_COMMAND_FIELDS = frozenset(
    {"next_chunk_rerun_call", "next_chunk_export_call", "rerun_hook"}
)
_E2E_NEXT_ACTION_MAX_COMMAND_CHARS = 120


def _format_e2e_next_action_display_text(value: Any, *, field: str) -> str:
    text = _format_reproduction_display_text(value, field=field)
    if field not in _E2E_NEXT_ACTION_COMMAND_FIELDS:
        return _format_e2e_display_label(text, field=field)
    if len(text) <= _E2E_NEXT_ACTION_MAX_COMMAND_CHARS:
        return f"{field}={text}"
    return f"{field}={text[:_E2E_NEXT_ACTION_MAX_COMMAND_CHARS]}..."


def _count_pair(numerator: Any, denominator: Any) -> str:
    return f"{_format_optional_int(numerator)}/{_format_optional_int(denominator)}"


def _format_optional_int(value: Any) -> str:
    if value is None:
        return "NA"
    if _is_reproduction_missing_display_value(value):
        return "NA"
    if isinstance(value, Mapping):
        return _compact_e2e_display_value(
            json.dumps(
                _json_safe_export_payload(dict(value)),
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    if isinstance(value, (list, tuple, np.ndarray)):
        return _compact_e2e_display_value(
            json.dumps(
                _json_safe_export_payload(value),
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"Paper reproduction display count is boolean: {value!r}")
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric):
            if math.isnan(numeric):
                return "NA"
            return "+Inf" if numeric > 0 else "-Inf"
        if numeric < 0:
            raise ValueError(f"Paper reproduction display count is negative: {value!r}")
        if not numeric.is_integer():
            raise ValueError(f"Paper reproduction display count is not an integer: {value!r}")
        return str(int(numeric))
    text = _format_reproduction_display_text(value, field="count")
    try:
        numeric = float(text)
    except ValueError as exc:
        raise ValueError(
            f"Paper reproduction display count is not numeric: {value!r}"
        ) from exc
    if not math.isfinite(numeric):
        if math.isnan(numeric):
            return "NA"
        return "+Inf" if numeric > 0 else "-Inf"
    if numeric < 0:
        raise ValueError(f"Paper reproduction display count is negative: {value!r}")
    if not numeric.is_integer():
        raise ValueError(f"Paper reproduction display count is not an integer: {value!r}")
    return str(int(numeric))


def _threshold_label(name: str, value: Any, threshold: Any) -> str:
    value_label = _format_comparison_display_measure(value)
    if threshold is None:
        return f"{name}={value_label}"
    threshold_label = _format_comparison_display_measure(threshold)
    return f"{name}={value_label} <= {threshold_label}"


def _blocking_conditions_label(value: Any) -> str:
    if not isinstance(value, Mapping) or not value:
        return "blockers=none"
    parts = [
        f"{_format_e2e_display_label(key, field='blocking_condition')}:"
        f"{_format_blocking_condition_count(value[key])}"
        for key in _positive_blocking_condition_keys(value)
    ]
    if not parts:
        return "blockers=none"
    return "blockers=" + ",".join(parts)


def _format_blocking_condition_count(value: Any) -> str:
    return _format_optional_int(value)


def _is_reproduction_missing_display_value(value: Any) -> bool:
    return _is_scalar_missing(value)


def _positive_blocking_condition_keys(value: Mapping[Any, Any]) -> list[str]:
    keys: list[str] = []
    for key in sorted(value, key=str):
        if _is_positive_blocking_condition_count(value[key]):
            keys.append(str(key))
    return keys


def _is_positive_blocking_condition_count(value: Any) -> bool:
    if _is_reproduction_missing_display_value(value):
        return True
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if math.isnan(numeric):
            return True
        if numeric < 0:
            raise ValueError(
                "Paper reproduction display blocker count is negative: "
                f"{value!r}"
            )
        return numeric > 0
    return bool(value)


def _compact_join(parts: tuple[str, ...]) -> str:
    return "; ".join(part for part in parts if part)


def write_paper_reproduction_e2e_report_json(
    output_path: str | Path,
    *,
    evidence_dir: str | Path,
    overwrite: bool = False,
    stable_runtime: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Write the public paper-reproduction E2E report as strict JSON.

    Parameters
    ----------
    output_path : str or Path
        Destination file path for the JSON artifact.
    evidence_dir : str or Path
        Directory containing Monte Carlo evidence JSON files.
    overwrite : bool
        If ``True``, overwrite an existing file; otherwise raise.
    stable_runtime : bool
        If ``True``, replace runtime_seconds fields with 0.0 for
        deterministic test comparisons.
    **kwargs
        Forwarded to :func:`paper_reproduction_e2e_report`.

    Returns
    -------
    dict[str, Any]
        The serialized report payload.

    Raises
    ------
    FileExistsError
        If *output_path* exists and *overwrite* is ``False``.
    """

    _require_reproduction_writer_overwrite(overwrite)
    _require_reproduction_writer_boolean(stable_runtime, name="stable_runtime")
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists; pass overwrite=True to replace it."
    )
    report = paper_reproduction_e2e_report(evidence_dir, **kwargs)
    payload = report.to_dict()
    if stable_runtime:
        payload = _stable_reproduction_runtime_payload(payload)
    _write_reproduction_json_atomic(path, payload)
    return payload


def load_paper_reproduction_e2e_report_json(
    path_like: str | Path,
) -> PaperReproductionE2EReport:
    """Load a saved strict-JSON public paper-reproduction E2E report.

    Parameters
    ----------
    path_like : str or Path
        Path to the JSON file to load.

    Returns
    -------
    PaperReproductionE2EReport
        Deserialized E2E report object.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is not valid strict JSON or has an unexpected schema.
    """

    path = Path(path_like)
    if not path.is_file():
        raise FileNotFoundError(
            f"paper reproduction E2E report JSON file does not exist: {path}"
        )
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_reproduction_report_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"paper reproduction E2E report must be valid JSON: {path}"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {"summary", "rows"}:
        raise ValueError(
            "paper reproduction E2E report JSON must contain only summary and rows."
        )
    summary = _require_json_object(payload["summary"], "summary")
    rows = _require_json_object_sequence(payload["rows"], "rows")
    _reject_nonfinite_json_numbers(
        summary,
        field_name="paper reproduction E2E report summary",
    )
    _reject_nonfinite_json_numbers(
        rows,
        field_name="paper reproduction E2E report rows",
    )
    return PaperReproductionE2EReport(
        rows=rows,
        summary=summary,
    )


def write_paper_reproduction_comparison_report_json(
    output_path: str | Path,
    *,
    evidence_dir: str | Path,
    overwrite: bool = False,
    stable_runtime: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Write the unified paper reproduction comparison report as strict JSON.

    Parameters
    ----------
    output_path : str or Path
        Destination file path for the JSON artifact.
    evidence_dir : str or Path
        Directory containing Monte Carlo evidence JSON files.
    overwrite : bool
        If ``True``, overwrite an existing file; otherwise raise.
    stable_runtime : bool
        If ``True``, replace runtime_seconds fields with 0.0 for
        deterministic test comparisons.
    **kwargs
        Forwarded to :func:`paper_reproduction_comparison_report`.

    Returns
    -------
    dict[str, Any]
        The serialized report payload.

    Raises
    ------
    FileExistsError
        If *output_path* exists and *overwrite* is ``False``.
    """

    _require_reproduction_writer_overwrite(overwrite)
    _require_reproduction_writer_boolean(stable_runtime, name="stable_runtime")
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists; pass overwrite=True to replace it."
    )
    report = paper_reproduction_comparison_report(evidence_dir, **kwargs)
    payload = report.to_dict()
    if stable_runtime:
        payload = _stable_reproduction_runtime_payload(payload)
    _write_reproduction_json_atomic(path, payload)
    return payload


def _require_reproduction_writer_overwrite(overwrite: Any) -> None:
    _require_reproduction_writer_boolean(overwrite, name="overwrite")


def _require_reproduction_writer_boolean(value: Any, *, name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"paper reproduction report {name} must be boolean.")


def _stable_reproduction_runtime_payload(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        return {
            key: (
                _stable_reproduction_runtime_value(value)
                if _is_reproduction_runtime_field(key)
                else _stable_reproduction_runtime_payload(value)
            )
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [_stable_reproduction_runtime_payload(value) for value in payload]
    if isinstance(payload, tuple):
        return tuple(_stable_reproduction_runtime_payload(value) for value in payload)
    return payload


def _is_reproduction_runtime_field(key: Any) -> bool:
    return isinstance(key, str) and (
        key == "runtime_seconds" or key.endswith("_runtime_seconds")
    )


def _stable_reproduction_runtime_value(value: Any) -> float | None:
    if value is None or _is_scalar_missing(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            "paper reproduction runtime fields must be numeric seconds, not boolean."
        )
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("paper reproduction runtime fields must be finite.")
        if numeric < 0:
            raise ValueError("paper reproduction runtime fields must be nonnegative.")
        return 0.0
    raise ValueError("paper reproduction runtime fields must be numeric seconds.")


def _reject_reproduction_report_json_constant(value: str) -> None:
    raise ValueError(
        f"paper reproduction report must be strict JSON; found {value}."
    )


def load_paper_reproduction_comparison_report_json(
    path_like: str | Path,
) -> PaperReproductionComparisonReport:
    """Load a saved strict-JSON paper reproduction comparison report.

    Parameters
    ----------
    path_like : str or Path
        Path to the JSON file to load.

    Returns
    -------
    PaperReproductionComparisonReport
        Deserialized comparison report object.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is not valid strict JSON or has an unexpected schema.
    """

    path = Path(path_like)
    if not path.is_file():
        raise FileNotFoundError(
            f"paper reproduction comparison report JSON file does not exist: {path}"
        )
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_reproduction_report_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"paper reproduction comparison report must be valid JSON: {path}"
        ) from exc
    required_keys = {"summary", "comparison_rows", "provenance_rows", "tolerance_rows"}
    if not isinstance(payload, dict) or set(payload) != required_keys:
        raise ValueError(
            "paper reproduction comparison report JSON must contain only summary, "
            "comparison_rows, provenance_rows, and tolerance_rows."
        )
    summary = _require_json_object(payload["summary"], "summary")
    comparison_rows = _require_json_object_sequence(
        payload["comparison_rows"],
        "comparison_rows",
    )
    provenance_rows = _require_json_object_sequence(
        payload["provenance_rows"],
        "provenance_rows",
    )
    tolerance_rows = _require_json_object_sequence(
        payload["tolerance_rows"],
        "tolerance_rows",
    )
    _reject_nonfinite_json_numbers(
        summary,
        field_name="paper reproduction comparison report summary",
    )
    _reject_nonfinite_json_numbers(
        comparison_rows,
        field_name="paper reproduction comparison report comparison_rows",
    )
    _reject_nonfinite_json_numbers(
        provenance_rows,
        field_name="paper reproduction comparison report provenance_rows",
    )
    _reject_nonfinite_json_numbers(
        tolerance_rows,
        field_name="paper reproduction comparison report tolerance_rows",
    )
    return PaperReproductionComparisonReport(
        comparison_rows=comparison_rows,
        provenance_rows=provenance_rows,
        tolerance_rows=tolerance_rows,
        summary=summary,
    )


def _empirical_comparison_rows(
    rows: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    comparison_rows: list[dict[str, Any]] = []
    for row in rows:
        comparison_category = _empirical_comparison_category(row)
        paper_target_claim = bool(row.get("paper_target_available"))
        comparison_rows.append(
            {
                "case_id": f"empirical_{row['case_id']}",
                "report_family": "paper_reproduction",
                "comparison_kind": "empirical_target",
                "source_report": "paper_empirical_reproduction_report",
                "source_case_id": row["case_id"],
                "row_type": "empirical_target_comparison",
                "comparison_category": comparison_category,
                "accepted_for_paper_reproduction": comparison_category
                == "sampling_error_reproduction",
                "paper_target_claim": paper_target_claim,
                "paper_target_available": paper_target_claim,
                "application": row.get("application"),
                "metric": row.get("metric"),
                "paper_value": row.get("target_value"),
                "python_value": row.get("python_value"),
                "absolute_difference": row.get("absolute_difference"),
                "tolerance_value": row.get("tolerance"),
                "tolerance_name": "paper_rounded_percent_tolerance"
                if paper_target_claim
                else "reference_only_no_paper_target",
                "sample_size": row.get("sample_size"),
                "analysis_frame_rows": row.get("analysis_frame_rows"),
                "n_treated": row.get("n_treated"),
                "n_control": row.get("n_control"),
                "cluster_count": row.get("cluster_count"),
                "applied_num_y_bins": row.get("applied_num_y_bins"),
                "size_risk": row.get("size_risk"),
                "allow_min_defiers": row.get("allow_min_defiers"),
                "requested_max_defiers_share": row.get("max_defiers_share"),
                "minimum_compatible_defiers_share": row.get(
                    "minimum_compatible_defiers_share"
                ),
                "actual_max_defiers_share": row.get("actual_max_defiers_share"),
                "defier_cap_actual_cap_source": row.get(
                    "defier_cap_actual_cap_source"
                ),
                "defier_cap_epsilon_relaxation": row.get(
                    "defier_cap_epsilon_relaxation"
                ),
                "defier_cap_reference_boundary": row.get(
                    "defier_cap_reference_boundary"
                ),
                "active_restriction": row.get("active_restriction"),
                "theta_kk_min": row.get("theta_kk_min"),
                "theta_kk_min_by_group": row.get("theta_kk_min_by_group"),
                "no_bite_flag": row.get("no_bite_flag"),
                "no_bite_reason": row.get("no_bite_reason"),
                "general_lfp_solution_basis": row.get("general_lfp_solution_basis"),
                "general_lfp_type_count": row.get("general_lfp_type_count"),
                "general_lfp_slack_count": row.get("general_lfp_slack_count"),
                "general_lfp_denominator_minimum": row.get(
                    "general_lfp_denominator_minimum"
                ),
                "general_lfp_objective_numerator": row.get(
                    "general_lfp_objective_numerator"
                ),
                "general_lfp_objective_denominator": row.get(
                    "general_lfp_objective_denominator"
                ),
                "general_lfp_primal_eq_max_abs_residual": row.get(
                    "general_lfp_primal_eq_max_abs_residual"
                ),
                "general_lfp_primal_ub_max_violation": row.get(
                    "general_lfp_primal_ub_max_violation"
                ),
                "general_lfp_marginal_fit_max_abs_difference": row.get(
                    "general_lfp_marginal_fit_max_abs_difference"
                ),
                "general_lfp_slack_constraint_max_violation": row.get(
                    "general_lfp_slack_constraint_max_violation"
                ),
                "general_lfp_defier_cap_max_violation": row.get(
                    "general_lfp_defier_cap_max_violation"
                ),
                "general_lfp_paper_inequality_max_violation": row.get(
                    "general_lfp_paper_inequality_max_violation"
                ),
                "general_lfp_objective_paper_inequality_max_violation": row.get(
                    "general_lfp_objective_paper_inequality_max_violation"
                ),
                "runtime_seconds": row.get("runtime_seconds"),
                "data_source": row.get("data_source"),
                "provenance_key": f"empirical::{row['case_id']}",
                "exception_status": row.get("exception_status"),
                "exception_message": row.get("exception_message"),
                "paper_anchor": row.get("paper_anchor"),
                "reference_anchor": row.get("reference_anchor"),
                "truth_hierarchy": row.get("truth_hierarchy"),
            }
        )
    return tuple(comparison_rows)


def _monte_carlo_comparison_rows(
    rows: tuple[dict[str, Any], ...],
    *,
    summary: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    paper_budget = {
        "replications": 500,
        "bootstrap_replications": 500,
    }
    observed_budget = {
        "replications": summary.get("paper_default_replications"),
        "bootstrap_replications": summary.get(
            "paper_default_bootstrap_replications"
        ),
    }
    active_blocking_conditions = dict(summary.get("active_blocking_conditions") or {})
    covered_result_rows = summary.get("covered_result_rows") or 0
    paper_result_rows = summary.get("paper_result_rows") or 0
    stale_evidence = {
        "stale_evidence_file_count": summary.get("stale_evidence_file_count"),
        "stale_evidence_files": summary.get("stale_evidence_files"),
        "stale_evidence_error": summary.get("stale_evidence_error"),
    }
    schedule_anchor = (
        "manuscript/sources/arxiv-2404.11739v3/draft.tex:391-456; "
        "manuscript/sources/arxiv-2404.11739v3/draft.tex:1094-1156"
    )
    reference_anchor = (
        "packages/r/TestMechs/R/simulate_data_binaryM.R:1-35; "
        "packages/r/TestMechs/R/test_sharp_null.R:90-315; "
        "packages/r/TestMechs/R/nonparametric_bootstrap.R:16-71"
    )
    return (
        {
            "case_id": "monte_carlo_paper_default_schedule",
            "report_family": "paper_reproduction",
            "comparison_kind": "monte_carlo_schedule",
            "source_report": "paper_monte_carlo_reproduction_report",
            "source_case_id": rows[0].get("case_id") if rows else None,
            "row_type": "monte_carlo_schedule_comparison",
            "comparison_category": (
                "exact_reproduction"
                if observed_budget == paper_budget
                else "missing_data_or_source_limitation"
            ),
            "accepted_for_paper_reproduction": observed_budget == paper_budget,
            "paper_target_claim": True,
            "paper_target_available": True,
            "application": "paper empirical-mixture Monte Carlo suite",
            "metric": "paper_default_schedule",
            "paper_value": paper_budget,
            "python_value": observed_budget,
            "absolute_difference": 0.0 if observed_budget == paper_budget else None,
            "tolerance_value": 0.0,
            "tolerance_name": "paper_default_schedule_contract",
            "scheduled_draws": summary.get("scheduled_draws"),
            "scheduled_bootstrap_draws": summary.get("scheduled_bootstrap_draws"),
            "runtime_seconds": summary.get("runtime_seconds"),
            "paper_acceptance_gate_passes": summary.get(
                "paper_acceptance_gate_passes"
            ),
            "active_blocking_conditions": active_blocking_conditions,
            **stale_evidence,
            "next_chunk_kwargs": summary.get("next_chunk_kwargs"),
            "next_chunk_rerun_call": summary.get("next_chunk_rerun_call"),
            "next_chunk_evidence_path": summary.get("next_chunk_evidence_path"),
            "next_chunk_export_call": summary.get("next_chunk_export_call"),
            "provenance_key": "monte_carlo::paper_empirical_mixture_sources",
            "paper_anchor": schedule_anchor,
            "reference_anchor": reference_anchor,
            "truth_hierarchy": (
                "paper_monte_carlo_protocol_then_python_strict_json_evidence_with_r_as_reference"
            ),
            "next_action": summary.get("next_action"),
        },
        {
            "case_id": "monte_carlo_paper_acceptance_evidence_state",
            "report_family": "paper_reproduction",
            "comparison_kind": "monte_carlo_acceptance_coverage",
            "source_report": "paper_monte_carlo_reproduction_report",
            "source_case_id": "paper_acceptance_gate",
            "row_type": "monte_carlo_acceptance_state_comparison",
            "comparison_category": (
                "sampling_error_reproduction"
                if summary.get("paper_acceptance_gate_passes") is True
                else "missing_data_or_source_limitation"
            ),
            "accepted_for_paper_reproduction": summary.get(
                "paper_acceptance_gate_passes"
            )
            is True,
            "paper_target_claim": True,
            "paper_target_available": True,
            "application": "paper empirical-mixture Monte Carlo suite",
            "metric": "paper_acceptance_covered_rows",
            "paper_value": paper_result_rows,
            "python_value": covered_result_rows,
            "absolute_difference": float(paper_result_rows - covered_result_rows),
            "tolerance_value": 0.0,
            "tolerance_name": "paper_full_suite_coverage_contract",
            "evidence_file_count": summary.get("evidence_file_count"),
            "scheduled_file_count": summary.get("scheduled_file_count"),
            "paper_result_rows": paper_result_rows,
            "covered_result_rows": covered_result_rows,
            "executed_result_rows": summary.get("executed_result_rows"),
            "executed_draws": summary.get("executed_draws"),
            "scheduled_draws": summary.get("scheduled_draws"),
            "executed_bootstrap_draws": summary.get("executed_bootstrap_draws"),
            "scheduled_bootstrap_draws": summary.get("scheduled_bootstrap_draws"),
            "max_rejection_rate_absolute_error": summary.get(
                "max_rejection_rate_absolute_error"
            ),
            "max_target_mc_standard_error": summary.get(
                "max_target_mc_standard_error"
            ),
            "runtime_seconds": summary.get("runtime_seconds"),
            "paper_acceptance_gate_passes": summary.get(
                "paper_acceptance_gate_passes"
            ),
            "active_blocking_conditions": active_blocking_conditions,
            **stale_evidence,
            "next_chunk_kwargs": summary.get("next_chunk_kwargs"),
            "next_chunk_rerun_call": summary.get("next_chunk_rerun_call"),
            "next_chunk_evidence_path": summary.get("next_chunk_evidence_path"),
            "next_chunk_export_call": summary.get("next_chunk_export_call"),
            "provenance_key": "monte_carlo::paper_empirical_mixture_sources",
            "paper_anchor": schedule_anchor,
            "reference_anchor": reference_anchor,
            "truth_hierarchy": (
                "paper_monte_carlo_protocol_then_python_strict_json_evidence_with_r_as_reference"
            ),
            "next_action": summary.get("next_action"),
        },
    )


def _empirical_comparison_category(row: dict[str, Any]) -> str:
    if not bool(row.get("paper_target_available")):
        return "documented_exception"
    exception_status = row.get("exception_status")
    if exception_status == "paper_target_within_tolerance":
        return "sampling_error_reproduction"
    if exception_status == "paper_target_read_error":
        return "missing_data_or_source_limitation"
    if exception_status == "paper_target_outside_tolerance":
        return "missing_data_or_source_limitation"
    if exception_status == "python_exception":
        return "missing_data_or_source_limitation"
    return "documented_exception"


def _empirical_provenance_rows(
    rows: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    provenance_rows: list[dict[str, Any]] = []
    for row in rows:
        source_path = Path(str(row["data_source"]))
        mediators = tuple(row.get("mediators") or ())
        required_columns = tuple(
            dict.fromkeys(
                (
                    row["d"],
                    *mediators,
                    row["y"],
                    *(row.get("analysis_frame_columns") or ()),
                )
            )
        )
        try:
            raw_frame = pd.read_csv(source_path)
            analysis_frame = raw_frame.dropna(subset=list(required_columns)).copy()
            support = _support_packet(
                analysis_frame,
                d=row["d"],
                mediators=mediators,
                y=row["y"],
                cluster=row.get("cluster"),
            )
            ready = True
            error = None
        except Exception as exc:
            raw_frame = pd.DataFrame()
            analysis_frame = pd.DataFrame()
            support = {}
            ready = False
            error = f"{type(exc).__name__}: {exc}"
        filtering_rule = f"drop missing values in columns={required_columns}"

        provenance_rows.append(
            {
                "provenance_key": f"empirical::{row['case_id']}",
                "provenance_type": "empirical_fixture",
                "provenance_category": "python_only_fixture_provenance",
                "source_kind": "empirical_fixture",
                "case_id": row["case_id"],
                "application": row.get("application"),
                "fixture_name": row.get("fixture_name"),
                "data_source": str(source_path),
                "d": row.get("d"),
                "mediators": tuple(mediators),
                "y": row.get("y"),
                "cluster": row.get("cluster"),
                "row_count": int(raw_frame.shape[0]),
                "complete_case_rows": int(analysis_frame.shape[0]),
                "raw_rows": int(raw_frame.shape[0]),
                "analysis_frame_rows": int(analysis_frame.shape[0]),
                "reported_sample_size": row.get("sample_size"),
                "n_treated": row.get("n_treated"),
                "n_control": row.get("n_control"),
                "applied_num_y_bins": row.get("applied_num_y_bins"),
                "cluster_count": support.get("cluster_count"),
                "treatment_levels": support.get("treatment_support"),
                "mediator_levels": support.get("mediator_support"),
                "filtering_rules": filtering_rule,
                "filtering_rule": filtering_rule,
                "ready": ready,
                "provenance_error": error,
                **support,
                "paper_anchor": row.get("paper_anchor"),
                "reference_anchor": row.get("reference_anchor"),
            }
        )
    return tuple(provenance_rows)


def _monte_carlo_provenance_rows(
    fixtures_dir: Path,
) -> tuple[dict[str, Any], ...]:
    data_sources = load_paper_empirical_mixture_benchmark_data_sources(fixtures_dir)
    fixture_paths = {
        "Bursztyn et al|binary|unclustered": fixtures_dir / "burstzyn_data.csv",
        "Baranov et al|binary|clustered": fixtures_dir / "baranov_mother_data.csv",
        "Baranov et al|nonbinary|clustered": fixtures_dir / "baranov_mother_data.csv",
    }
    provenance_rows: list[dict[str, Any]] = []
    for source_key, source in data_sources.items():
        raw_frame = source.df.copy()
        analysis_frame = source.analysis_frame()
        required_columns = tuple(
            dict.fromkeys((source.d, source.m, source.y, *source.analysis_frame_columns))
        )
        filtering_rule = f"drop missing values in columns={required_columns}"
        support = _support_packet(
            analysis_frame,
            d=source.d,
            mediators=(source.m,),
            y=source.y,
            cluster=source.cluster,
        )
        if source.d in analysis_frame:
            control_rows = int((analysis_frame[source.d] == 0).sum())
            treated_rows = int((analysis_frame[source.d] == 1).sum())
        else:
            control_rows = None
            treated_rows = None
        provenance_rows.append(
            {
                "provenance_key": f"monte_carlo::{source_key}",
                "provenance_type": "monte_carlo_empirical_mixture_source",
                "provenance_category": "python_source_binding",
                "source_kind": "monte_carlo_data_source",
                "data_source_key": source_key,
                "fixture_name": fixture_paths[source_key].name,
                "data_source": str(fixture_paths[source_key].resolve()),
                "d": source.d,
                "mediators": (source.m,),
                "y": source.y,
                "cluster": source.cluster,
                "row_count": int(raw_frame.shape[0]),
                "complete_case_rows": int(analysis_frame.shape[0]),
                "raw_rows": int(raw_frame.shape[0]),
                "analysis_frame_rows": int(analysis_frame.shape[0]),
                "control_rows": control_rows,
                "treated_rows": treated_rows,
                "cluster_count": support.get("cluster_count"),
                "treatment_levels": support.get("treatment_support"),
                "mediator_levels": support.get("mediator_support"),
                "expected_complete_case_rows": source.expected_complete_case_rows,
                "expected_control_rows": source.expected_control_rows,
                "expected_treated_rows": source.expected_treated_rows,
                "expected_source_clusters": source.expected_source_clusters,
                "expected_control_source_clusters": source.expected_control_source_clusters,
                "expected_treated_source_clusters": source.expected_treated_source_clusters,
                "filtering_rules": filtering_rule,
                "filtering_rule": filtering_rule,
                "ready": True,
                "provenance_error": None,
                **support,
                "paper_anchor": (
                    "manuscript/sources/arxiv-2404.11739v3/draft.tex:391-456; "
                    "manuscript/sources/arxiv-2404.11739v3/draft.tex:1094-1156"
                ),
                "reference_anchor": (
                    "packages/r/TestMechs/R/simulate_data_binaryM.R:1-35; "
                    "packages/r/TestMechs/R/test_sharp_null.R:90-315; "
                    "packages/r/TestMechs/R/nonparametric_bootstrap.R:16-71"
                ),
            }
        )
    return tuple(provenance_rows)


def _support_packet(
    frame: pd.DataFrame,
    *,
    d: str,
    mediators: tuple[str, ...],
    y: str,
    cluster: str | None,
) -> dict[str, Any]:
    treatment = frame[d] if d in frame else pd.Series(dtype=object)
    treatment_support = _ordered_unique(treatment)
    mediator_support = _mediator_support(frame, mediators)
    outcome_levels = _ordered_unique(frame[y]) if y in frame else ()
    cluster_counts = _cluster_counts(frame, d=d, cluster=cluster)
    return {
        "treatment_support": treatment_support,
        "treatment_levels": treatment_support,
        "treatment_level_count": int(len(treatment_support)),
        "mediator_support": mediator_support,
        "mediator_levels": mediator_support,
        "mediator_level_count": int(len(mediator_support)),
        "outcome_support": tuple(outcome_levels),
        "outcome_levels": tuple(outcome_levels),
        "outcome_level_count": int(len(outcome_levels)),
        "outcome_support_preview": tuple(outcome_levels[:10]),
        "outcome_support_truncated": len(outcome_levels) > 10,
        "cluster_count": cluster_counts["cluster_count"],
        "control_cluster_count": cluster_counts["control_cluster_count"],
        "treated_cluster_count": cluster_counts["treated_cluster_count"],
    }


def _mediator_support(
    frame: pd.DataFrame,
    mediators: tuple[str, ...],
) -> tuple[Any, ...]:
    if not mediators:
        return ()
    if len(mediators) == 1:
        return _ordered_unique(frame[mediators[0]])
    return tuple(
        tuple(_normalize_json_value(value) for value in record)
        for record in frame.loc[:, list(mediators)].drop_duplicates().itertuples(index=False, name=None)
    )


def _cluster_counts(
    frame: pd.DataFrame,
    *,
    d: str,
    cluster: str | None,
) -> dict[str, int | None]:
    if cluster is None or cluster not in frame or d not in frame:
        return {
            "cluster_count": None,
            "control_cluster_count": None,
            "treated_cluster_count": None,
        }
    return {
        "cluster_count": int(frame[cluster].nunique(dropna=True)),
        "control_cluster_count": int(frame.loc[frame[d] == 0, cluster].nunique(dropna=True)),
        "treated_cluster_count": int(frame.loc[frame[d] == 1, cluster].nunique(dropna=True)),
    }


def _tolerance_rows(
    *,
    empirical_summary: dict[str, Any],
    monte_carlo_summary: dict[str, Any],
    monte_carlo_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    absolute_tolerance = monte_carlo_kwargs.get("absolute_tolerance", 0.025)
    z_tolerance = monte_carlo_kwargs.get("z_tolerance", 2.0)
    alpha = monte_carlo_kwargs.get(
        "alpha",
        monte_carlo_summary.get("paper_nominal_alpha", 0.05),
    )
    return (
        {
            "tolerance_name": "paper_rounded_percent_tolerance",
            "applies_to": "empirical_lower_bound_targets",
            "threshold": empirical_summary.get("target_tolerance", 0.005),
            "scale": "proportion",
            "basis": (
                "Paper empirical targets are reported as whole percentages; "
                "the comparison uses a half-percentage-point rounding band."
            ),
            "paper_anchor": "manuscript/sources/arxiv-2404.11739v3/draft.tex:570-608",
            "reference_anchor": "packages/r/TestMechs/R/lb_frac_affected.R:318-347",
        },
        {
            "tolerance_name": "paper_full_suite_coverage_contract",
            "applies_to": "monte_carlo_acceptance_coverage",
            "threshold": 0,
            "z_tolerance": z_tolerance,
            "scale": "paper_result_rows",
            "basis": (
                "The comparison report must distinguish full paper coverage from "
                "low-budget smoke evidence; all paper rows must be represented "
                "before full-paper acceptance."
            ),
            "configured_rejection_rate_absolute_tolerance": absolute_tolerance,
            "paper_nominal_alpha": alpha,
            "paper_anchor": "manuscript/sources/arxiv-2404.11739v3/draft.tex:391-456",
            "reference_anchor": "packages/r/TestMechs/R/test_sharp_null.R:90-315",
        },
        {
            "tolerance_name": "paper_default_schedule_contract",
            "applies_to": "monte_carlo_acceptance_budget",
            "paper_default_replications": monte_carlo_summary.get(
                "paper_default_replications"
            ),
            "paper_default_bootstrap_replications": monte_carlo_summary.get(
                "paper_default_bootstrap_replications"
            ),
            "scale": "draw_budget",
            "basis": (
                "The paper tables report 500 simulation draws at 5% nominal size, "
                "with 500 bootstrap draws where the method protocol requires bootstrap."
            ),
            "paper_anchor": (
                "manuscript/sources/arxiv-2404.11739v3/draft.tex:391-456; "
                "manuscript/sources/arxiv-2404.11739v3/draft.tex:1094-1156"
            ),
            "reference_anchor": (
                "packages/r/TestMechs/R/nonparametric_bootstrap.R:16-71"
            ),
        },
    )


def _public_e2e_rows(
    comparison_report: PaperReproductionComparisonReport,
    *,
    milestone_version: str,
    roadmap_analysis: dict[str, Any] | None,
    requirements_analysis: dict[str, Any] | None,
    milestone_audit_status: str | None,
) -> tuple[dict[str, Any], ...]:
    comparison_rows = comparison_report.comparison_rows
    empirical_rows = tuple(
        row
        for row in comparison_rows
        if row.get("source_report") == "paper_empirical_reproduction_report"
    )
    monte_carlo_rows = tuple(
        row
        for row in comparison_rows
        if row.get("source_report") == "paper_monte_carlo_reproduction_report"
    )
    empirical_target_rows = tuple(
        row for row in empirical_rows if row.get("paper_target_available") is True
    )
    empirical_passed_rows = tuple(
        row
        for row in empirical_target_rows
        if row.get("accepted_for_paper_reproduction") is True
    )

    def claims_paper_target(row: Mapping[str, Any]) -> bool:
        if row.get("paper_target_claim") is False:
            return False
        if row.get("paper_target_available") is False:
            return False
        return (
            row.get("paper_target_claim") is True
            or row.get("paper_target_available") is True
            or row.get("case_id") == "monte_carlo_paper_acceptance_evidence_state"
        )

    def accepted_for_paper_reproduction(row: Mapping[str, Any]) -> bool:
        return (
            row.get("accepted_for_paper_reproduction") is True
            or (
                row.get("case_id") == "monte_carlo_paper_acceptance_evidence_state"
                and row.get("paper_acceptance_gate_passes") is True
            )
        )

    paper_target_rows = tuple(
        row for row in comparison_rows if claims_paper_target(row)
    )
    paper_accepted_rows = tuple(
        row for row in paper_target_rows if accepted_for_paper_reproduction(row)
    )
    monte_carlo_target_rows = tuple(
        row for row in monte_carlo_rows if claims_paper_target(row)
    )
    monte_carlo_accepted_rows = tuple(
        row
        for row in monte_carlo_target_rows
        if accepted_for_paper_reproduction(row)
    )
    empirical_differences = tuple(
        float(row["absolute_difference"])
        for row in empirical_target_rows
        if not _is_missing(row.get("absolute_difference"))
    )
    empirical_sample_sizes = tuple(
        int(row["sample_size"])
        for row in empirical_rows
        if not _is_missing(row.get("sample_size"))
    )
    evidence_state = _public_e2e_monte_carlo_evidence_row(monte_carlo_rows)
    comparison_summary = dict(comparison_report.summary)
    actual_comparison_row_count = len(comparison_report.comparison_rows)
    actual_provenance_row_count = len(comparison_report.provenance_rows)
    actual_provenance_ready_rows = int(
        sum(row.get("ready") is True for row in comparison_report.provenance_rows)
    )
    actual_tolerance_row_count = len(comparison_report.tolerance_rows)
    reported_comparison_row_count = _coerce_nonnegative_int(
        comparison_summary.get("comparison_row_count")
    )
    reported_provenance_row_count = _coerce_nonnegative_int(
        comparison_summary.get("provenance_row_count")
    )
    required_comparison_row_count = max(
        _REQUIRED_PAPER_REPRODUCTION_COMPARISON_ROWS,
        reported_comparison_row_count or 0,
    )
    required_provenance_row_count = max(
        _REQUIRED_PAPER_REPRODUCTION_PROVENANCE_ROWS,
        reported_provenance_row_count or 0,
    )
    comparison_row_count_shortfall = max(
        required_comparison_row_count - actual_comparison_row_count,
        0,
    )
    provenance_row_count_shortfall = max(
        required_provenance_row_count - actual_provenance_row_count,
        0,
    )
    comparison_row_state = _comparison_row_state(comparison_report.comparison_rows)
    provenance_row_state = _provenance_row_state(
        comparison_report.provenance_rows
    )
    tolerance_contract_state = _tolerance_contract_name_state(
        comparison_report.tolerance_rows
    )
    comparison_provenance_summary_claims_ready = (
        comparison_summary.get("phase17_contract_ready") is True
    )
    comparison_provenance_ready = (
        comparison_provenance_summary_claims_ready
        and comparison_row_count_shortfall == 0
        and provenance_row_count_shortfall == 0
        and not comparison_row_state["incomplete_row_ids"]
        and not comparison_row_state["invalid_row_ids"]
        and actual_provenance_ready_rows == actual_provenance_row_count
        and not provenance_row_state["incomplete_row_ids"]
        and not provenance_row_state["invalid_row_ids"]
        and not provenance_row_state["duplicate_row_ids"]
        and actual_tolerance_row_count >= 3
        and not tolerance_contract_state["missing_names"]
        and not tolerance_contract_state["duplicate_names"]
        and not tolerance_contract_state["incomplete_names"]
        and not tolerance_contract_state["invalid_names"]
    )
    comparison_provenance_condition = (
        "comparison_provenance_ready"
        if comparison_provenance_ready
        else "comparison_provenance_blocked"
    )
    comparison_summary["comparison_provenance_summary_claims_ready"] = (
        comparison_provenance_summary_claims_ready
    )
    comparison_summary["actual_comparison_row_count"] = actual_comparison_row_count
    comparison_summary["actual_provenance_row_count"] = actual_provenance_row_count
    comparison_summary["actual_provenance_ready_rows"] = actual_provenance_ready_rows
    comparison_summary["actual_tolerance_row_count"] = actual_tolerance_row_count
    comparison_summary["required_comparison_row_count"] = (
        required_comparison_row_count
    )
    comparison_summary["required_provenance_row_count"] = (
        required_provenance_row_count
    )
    comparison_summary["comparison_row_count_shortfall"] = (
        comparison_row_count_shortfall
    )
    comparison_summary["provenance_row_count_shortfall"] = (
        provenance_row_count_shortfall
    )
    comparison_summary["incomplete_comparison_row_ids"] = (
        comparison_row_state["incomplete_row_ids"]
    )
    comparison_summary["incomplete_comparison_fields"] = (
        comparison_row_state["incomplete_fields"]
    )
    comparison_summary["invalid_comparison_row_ids"] = (
        comparison_row_state["invalid_row_ids"]
    )
    comparison_summary["invalid_comparison_fields"] = (
        comparison_row_state["invalid_fields"]
    )
    comparison_summary["duplicate_comparison_row_ids"] = (
        comparison_row_state["duplicate_row_ids"]
    )
    comparison_summary["incomplete_provenance_row_ids"] = (
        provenance_row_state["incomplete_row_ids"]
    )
    comparison_summary["incomplete_provenance_fields"] = (
        provenance_row_state["incomplete_fields"]
    )
    comparison_summary["invalid_provenance_row_ids"] = (
        provenance_row_state["invalid_row_ids"]
    )
    comparison_summary["invalid_provenance_fields"] = (
        provenance_row_state["invalid_fields"]
    )
    comparison_summary["duplicate_provenance_row_ids"] = (
        provenance_row_state["duplicate_row_ids"]
    )
    comparison_summary["required_tolerance_contract_names"] = (
        tolerance_contract_state["required_names"]
    )
    comparison_summary["actual_tolerance_contract_names"] = (
        tolerance_contract_state["actual_names"]
    )
    comparison_summary["missing_tolerance_contract_names"] = (
        tolerance_contract_state["missing_names"]
    )
    comparison_summary["duplicate_tolerance_contract_names"] = (
        tolerance_contract_state["duplicate_names"]
    )
    comparison_summary["incomplete_tolerance_contract_names"] = (
        tolerance_contract_state["incomplete_names"]
    )
    comparison_summary["invalid_tolerance_contract_names"] = (
        tolerance_contract_state["invalid_names"]
    )
    comparison_summary["incomplete_tolerance_contract_fields"] = (
        tolerance_contract_state["incomplete_fields"]
    )
    comparison_summary["invalid_tolerance_contract_fields"] = (
        tolerance_contract_state["invalid_fields"]
    )
    empirical_runtime_seconds = comparison_summary.get("empirical_runtime_seconds")
    if empirical_runtime_seconds is None:
        empirical_runtime_values = tuple(
            float(row["runtime_seconds"])
            for row in empirical_rows
            if not _is_missing(row.get("runtime_seconds"))
        )
        empirical_runtime_seconds = (
            sum(empirical_runtime_values) if empirical_runtime_values else None
        )
    monte_carlo_runtime_seconds = comparison_summary.get("monte_carlo_runtime_seconds")
    comparison_runtime_seconds = comparison_summary.get("runtime_seconds")
    roadmap_state = _roadmap_analysis_archive_state(roadmap_analysis)
    roadmap_complete = roadmap_state["roadmap_disk_complete"]
    requirements_state = _requirements_analysis_archive_state(requirements_analysis)
    requirements_complete = requirements_state["requirements_complete"]
    audit_state = _public_e2e_milestone_audit_state(milestone_audit_status)
    audit_status = audit_state["milestone_audit_status"]
    audit_passed = audit_state["milestone_audit_passes"]
    paper_acceptance_passes = evidence_state.get("paper_acceptance_gate_passes") is True
    summary_paper_reproduction_gate_passes = (
        comparison_summary.get("paper_reproduction_gate_passes") is True
    )
    row_level_paper_reproduction_gate_passes = (
        len(paper_target_rows) > 0
        and len(paper_target_rows) == len(paper_accepted_rows)
        and len(empirical_target_rows) > 0
        and len(empirical_target_rows) == len(empirical_passed_rows)
        and len(monte_carlo_target_rows) > 0
        and len(monte_carlo_target_rows) == len(monte_carlo_accepted_rows)
    )
    paper_reproduction_gate_consistent = not (
        summary_paper_reproduction_gate_passes
        and not row_level_paper_reproduction_gate_passes
    )
    paper_reproduction_gate_passes = (
        summary_paper_reproduction_gate_passes
        and row_level_paper_reproduction_gate_passes
        and paper_reproduction_gate_consistent
    )
    comparison_summary["paper_reproduction_gate_passes"] = (
        paper_reproduction_gate_passes
    )
    comparison_summary["paper_reproduction_row_gate_passes"] = (
        row_level_paper_reproduction_gate_passes
    )
    comparison_summary["paper_reproduction_row_target_rows"] = len(
        paper_target_rows
    )
    comparison_summary["paper_reproduction_row_accepted_target_rows"] = len(
        paper_accepted_rows
    )
    comparison_summary["paper_reproduction_row_empirical_target_rows"] = len(
        empirical_target_rows
    )
    comparison_summary["paper_reproduction_row_accepted_empirical_target_rows"] = len(
        empirical_passed_rows
    )
    comparison_summary["paper_reproduction_row_monte_carlo_target_rows"] = len(
        monte_carlo_target_rows
    )
    comparison_summary[
        "paper_reproduction_row_accepted_monte_carlo_target_rows"
    ] = len(monte_carlo_accepted_rows)
    comparison_summary["paper_reproduction_gate_consistent"] = (
        paper_reproduction_gate_consistent
    )
    archive_blockers: dict[str, int] = {}
    if roadmap_state["condition_value"]:
        archive_blockers[roadmap_state["condition"]] = int(
            roadmap_state["condition_value"]
        )
    if requirements_state["condition_value"]:
        archive_blockers[requirements_state["condition"]] = int(
            requirements_state["condition_value"]
        )
    if not audit_passed:
        archive_blockers["milestone_audit_not_passed"] = 1
    if not paper_acceptance_passes:
        archive_blockers["paper_acceptance_gate_blocked"] = 1
    elif not paper_reproduction_gate_passes:
        archive_blockers["paper_reproduction_gate_blocked"] = 1
        if not paper_reproduction_gate_consistent:
            archive_blockers["paper_reproduction_gate_inconsistent"] = 1
    other_archive_gates_ready = (
        roadmap_complete is True
        and requirements_complete is True
        and audit_passed
        and paper_acceptance_passes
        and paper_reproduction_gate_passes
    )
    if not comparison_provenance_ready and other_archive_gates_ready:
        archive_blockers["comparison_provenance_blocked"] = 1
    archive_ready = (
        roadmap_complete is True
        and requirements_complete is True
        and audit_passed
        and paper_acceptance_passes
        and paper_reproduction_gate_passes
        and comparison_provenance_ready
    )

    return (
        {
            "check_id": "paper_empirical_reproduction_public_slice",
            "row_type": "public_reproduction_e2e_check",
            "check_kind": "empirical_reproduction",
            "status": (
                "pass"
                if len(empirical_target_rows) == len(empirical_passed_rows)
                and len(empirical_target_rows) > 0
                else "blocked"
            ),
            "paper_target_rows": len(empirical_target_rows),
            "passed_target_rows": len(empirical_passed_rows),
            "reference_only_rows": len(empirical_rows) - len(empirical_target_rows),
            "max_absolute_difference": (
                max(empirical_differences) if empirical_differences else None
            ),
            "tolerance": 0.005,
            "min_sample_size": min(empirical_sample_sizes)
            if empirical_sample_sizes
            else None,
            "sample_sizes": empirical_sample_sizes,
            "runtime_seconds": empirical_runtime_seconds,
            "paper_anchor": "manuscript/sources/arxiv-2404.11739v3/draft.tex:570-608",
            "reference_anchor": (
                "packages/r/TestMechs/R/lb_frac_affected.R:318-347; "
                "packages/r/TestMechs/R/partial_density_plot.R:24-155"
            ),
        },
        {
            "check_id": "paper_monte_carlo_reproduction_public_slice",
            "row_type": "public_reproduction_e2e_check",
            "check_kind": "monte_carlo_reproduction",
            "status": "pass" if paper_acceptance_passes else "blocked",
            "paper_result_rows": evidence_state.get("paper_result_rows"),
            "covered_result_rows": evidence_state.get("covered_result_rows"),
            "executed_result_rows": evidence_state.get("executed_result_rows"),
            "scheduled_draws": evidence_state.get("scheduled_draws"),
            "executed_draws": evidence_state.get("executed_draws"),
            "scheduled_bootstrap_draws": evidence_state.get(
                "scheduled_bootstrap_draws"
            ),
            "executed_bootstrap_draws": evidence_state.get("executed_bootstrap_draws"),
            "stale_evidence_file_count": evidence_state.get("stale_evidence_file_count"),
            "stale_evidence_files": evidence_state.get("stale_evidence_files"),
            "stale_evidence_error": evidence_state.get("stale_evidence_error"),
            "max_rejection_rate_absolute_error": evidence_state.get(
                "max_rejection_rate_absolute_error"
            ),
            "max_target_mc_standard_error": evidence_state.get(
                "max_target_mc_standard_error"
            ),
            "paper_acceptance_absolute_tolerance": (
                _PAPER_ACCEPTANCE_ABSOLUTE_TOLERANCE
            ),
            "paper_acceptance_z_tolerance": _PAPER_ACCEPTANCE_Z_TOLERANCE,
            "paper_acceptance_max_target_mc_standard_error": (
                _PAPER_ACCEPTANCE_MAX_TARGET_MC_STANDARD_ERROR
            ),
            "runtime_seconds": monte_carlo_runtime_seconds,
            "paper_acceptance_gate_passes": paper_acceptance_passes,
            "paper_acceptance_evidence_state_present": evidence_state.get(
                "paper_acceptance_evidence_state_present",
                False,
            ),
            "paper_acceptance_evidence_state_missing_fields": evidence_state.get(
                "paper_acceptance_evidence_state_missing_fields",
                (),
            ),
            "paper_acceptance_evidence_state_malformed_fields": evidence_state.get(
                "paper_acceptance_evidence_state_malformed_fields",
                (),
            ),
            "paper_acceptance_evidence_state_empty_fields": evidence_state.get(
                "paper_acceptance_evidence_state_empty_fields",
                (),
            ),
            "paper_acceptance_evidence_state_duplicate_rows": evidence_state.get(
                "paper_acceptance_evidence_state_duplicate_rows",
                0,
            ),
            "low_budget_probe_blocked": comparison_summary.get(
                "low_budget_blocked_rows",
                0,
            )
            > 0,
            "active_blocking_conditions": dict(
                evidence_state.get("active_blocking_conditions") or {}
            ),
            "next_chunk_kwargs": evidence_state.get("next_chunk_kwargs"),
            "next_chunk_rerun_call": evidence_state.get("next_chunk_rerun_call"),
            "next_chunk_evidence_path": evidence_state.get("next_chunk_evidence_path"),
            "next_chunk_export_call": evidence_state.get("next_chunk_export_call"),
            "paper_anchor": (
                "manuscript/sources/arxiv-2404.11739v3/draft.tex:391-456; "
                "manuscript/sources/arxiv-2404.11739v3/draft.tex:1094-1156"
            ),
            "reference_anchor": (
                "packages/r/TestMechs/R/simulate_data_binaryM.R:1-35; "
                "packages/r/TestMechs/R/test_sharp_null.R:90-315; "
                "packages/r/TestMechs/R/nonparametric_bootstrap.R:16-71"
            ),
        },
        {
            "check_id": "paper_comparison_provenance_public_packet",
            "row_type": "public_reproduction_e2e_check",
            "check_kind": "comparison_provenance",
            "status": "pass" if comparison_provenance_ready else "blocked",
            "comparison_row_count": comparison_summary.get("comparison_row_count"),
            "provenance_row_count": comparison_summary.get("provenance_row_count"),
            "provenance_ready_rows": comparison_summary.get("provenance_ready_rows"),
            "tolerance_row_count": comparison_summary.get("tolerance_row_count"),
            "comparison_provenance_ready": comparison_provenance_ready,
            "comparison_provenance_condition": comparison_provenance_condition,
            "comparison_provenance_summary_claims_ready": (
                comparison_provenance_summary_claims_ready
            ),
            "actual_comparison_row_count": actual_comparison_row_count,
            "actual_provenance_row_count": actual_provenance_row_count,
            "actual_provenance_ready_rows": actual_provenance_ready_rows,
            "actual_tolerance_row_count": actual_tolerance_row_count,
            "required_comparison_row_count": required_comparison_row_count,
            "required_provenance_row_count": required_provenance_row_count,
            "comparison_row_count_shortfall": comparison_row_count_shortfall,
            "provenance_row_count_shortfall": provenance_row_count_shortfall,
            "incomplete_comparison_row_ids": comparison_row_state[
                "incomplete_row_ids"
            ],
            "incomplete_comparison_fields": comparison_row_state[
                "incomplete_fields"
            ],
            "invalid_comparison_row_ids": comparison_row_state["invalid_row_ids"],
            "invalid_comparison_fields": comparison_row_state["invalid_fields"],
            "duplicate_comparison_row_ids": comparison_row_state[
                "duplicate_row_ids"
            ],
            "incomplete_provenance_row_ids": provenance_row_state[
                "incomplete_row_ids"
            ],
            "incomplete_provenance_fields": provenance_row_state[
                "incomplete_fields"
            ],
            "invalid_provenance_row_ids": provenance_row_state[
                "invalid_row_ids"
            ],
            "invalid_provenance_fields": provenance_row_state["invalid_fields"],
            "duplicate_provenance_row_ids": provenance_row_state[
                "duplicate_row_ids"
            ],
            "required_tolerance_contract_names": tolerance_contract_state[
                "required_names"
            ],
            "actual_tolerance_contract_names": tolerance_contract_state[
                "actual_names"
            ],
            "missing_tolerance_contract_names": tolerance_contract_state[
                "missing_names"
            ],
            "duplicate_tolerance_contract_names": tolerance_contract_state[
                "duplicate_names"
            ],
            "incomplete_tolerance_contract_names": tolerance_contract_state[
                "incomplete_names"
            ],
            "invalid_tolerance_contract_names": tolerance_contract_state[
                "invalid_names"
            ],
            "incomplete_tolerance_contract_fields": tolerance_contract_state[
                "incomplete_fields"
            ],
            "invalid_tolerance_contract_fields": tolerance_contract_state[
                "invalid_fields"
            ],
            "tolerance_contract_names": comparison_summary.get(
                "tolerance_contract_names"
            ),
            "runtime_seconds": comparison_runtime_seconds,
            "empirical_runtime_seconds": empirical_runtime_seconds,
            "monte_carlo_runtime_seconds": monte_carlo_runtime_seconds,
            "paper_reproduction_gate_passes": comparison_summary.get(
                "paper_reproduction_gate_passes"
            ),
            "paper_anchor": (
                "manuscript/sources/arxiv-2404.11739v3/draft.tex:570-608; "
                "manuscript/sources/arxiv-2404.11739v3/draft.tex:391-456"
            ),
            "reference_anchor": (
                "packages/r/TestMechs/R/lb_frac_affected.R:318-347; "
                "packages/r/TestMechs/R/test_sharp_null.R:90-315"
            ),
        },
        {
            "check_id": (
                f"{_milestone_version_check_prefix(milestone_version)}"
                "_milestone_archive_preflight"
            ),
            "row_type": "public_reproduction_e2e_check",
            "check_kind": "milestone_archive_preflight",
            "status": "pass" if archive_ready else "blocked",
            "milestone_version": milestone_version,
            "roadmap_disk_complete": roadmap_complete,
            "roadmap_analysis_available": roadmap_state["roadmap_analysis_available"],
            "roadmap_analysis_condition": roadmap_state["condition"],
            "roadmap_analysis_missing_fields": roadmap_state["missing_fields"],
            "roadmap_analysis_malformed_fields": roadmap_state["malformed_fields"],
            "roadmap_analysis_mismatch_fields": roadmap_state["mismatch_fields"],
            "roadmap_phase_count": None
            if roadmap_analysis is None
            else roadmap_analysis.get("phase_count"),
            "roadmap_completed_phases": None
            if roadmap_analysis is None
            else roadmap_analysis.get("completed_phases"),
            "roadmap_total_plans": roadmap_state["total_plans"],
            "roadmap_total_summaries": roadmap_state["total_summaries"],
            "roadmap_phase_plan_count_sum": roadmap_state["phase_plan_count_sum"],
            "roadmap_phase_summary_count_sum": roadmap_state[
                "phase_summary_count_sum"
            ],
            "requirements_complete": requirements_complete,
            "requirements_analysis_available": requirements_state[
                "requirements_analysis_available"
            ],
            "requirements_analysis_condition": requirements_state["condition"],
            "requirements_analysis_missing_fields": requirements_state[
                "missing_fields"
            ],
            "requirements_analysis_malformed_fields": requirements_state[
                "malformed_fields"
            ],
            "requirements_total": requirements_state["total_requirements"],
            "requirements_completed": requirements_state["completed_requirements"],
            "requirements_pending": requirements_state["pending_requirements"],
            "requirements_expected_ids": requirements_state["expected_ids"],
            "requirements_expected_id_count": requirements_state["expected_id_count"],
            "requirements_expected_unique_id_count": requirements_state[
                "expected_unique_id_count"
            ],
            "requirements_duplicate_expected_ids": requirements_state[
                "duplicate_expected_ids"
            ],
            "requirements_traceability_rows": requirements_state[
                "traceability_rows"
            ],
            "requirements_traceability_row_count": len(
                requirements_state["traceability_rows"]
            ),
            "requirements_traceability_unique_row_count": requirements_state[
                "traceability_unique_row_count"
            ],
            "requirements_duplicate_traceability_rows": requirements_state[
                "duplicate_traceability_rows"
            ],
            "requirements_missing_traceability_ids": requirements_state[
                "missing_traceability_ids"
            ],
            "requirements_unexpected_traceability_ids": requirements_state[
                "unexpected_traceability_ids"
            ],
            "requirements_non_complete_rows": requirements_state[
                "non_complete_rows"
            ],
            "milestone_audit_status": audit_status,
            "milestone_audit_condition": audit_state["condition"],
            "milestone_audit_resolution_action": audit_state["resolution_action"],
            "paper_acceptance_gate_passes": paper_acceptance_passes,
            "paper_acceptance_absolute_tolerance": (
                _PAPER_ACCEPTANCE_ABSOLUTE_TOLERANCE
            ),
            "paper_acceptance_z_tolerance": _PAPER_ACCEPTANCE_Z_TOLERANCE,
            "paper_acceptance_max_target_mc_standard_error": (
                _PAPER_ACCEPTANCE_MAX_TARGET_MC_STANDARD_ERROR
            ),
            "paper_acceptance_evidence_state_present": evidence_state.get(
                "paper_acceptance_evidence_state_present",
                False,
            ),
            "paper_acceptance_evidence_state_missing_fields": evidence_state.get(
                "paper_acceptance_evidence_state_missing_fields",
                (),
            ),
            "paper_acceptance_evidence_state_malformed_fields": evidence_state.get(
                "paper_acceptance_evidence_state_malformed_fields",
                (),
            ),
            "paper_acceptance_evidence_state_empty_fields": evidence_state.get(
                "paper_acceptance_evidence_state_empty_fields",
                (),
            ),
            "paper_acceptance_evidence_state_duplicate_rows": evidence_state.get(
                "paper_acceptance_evidence_state_duplicate_rows",
                0,
            ),
            "paper_reproduction_gate_passes": comparison_summary.get(
                "paper_reproduction_gate_passes"
            ),
            "paper_reproduction_row_gate_passes": comparison_summary.get(
                "paper_reproduction_row_gate_passes"
            ),
            "paper_reproduction_gate_consistent": comparison_summary.get(
                "paper_reproduction_gate_consistent"
            ),
            "paper_reproduction_target_rows": comparison_summary.get(
                "paper_reproduction_target_rows"
            ),
            "paper_reproduction_accepted_target_rows": comparison_summary.get(
                "paper_reproduction_accepted_target_rows"
            ),
            "paper_reproduction_row_target_rows": comparison_summary.get(
                "paper_reproduction_row_target_rows"
            ),
            "paper_reproduction_row_accepted_target_rows": comparison_summary.get(
                "paper_reproduction_row_accepted_target_rows"
            ),
            "paper_reproduction_empirical_target_rows": comparison_summary.get(
                "paper_reproduction_empirical_target_rows"
            ),
            "paper_reproduction_accepted_empirical_target_rows": comparison_summary.get(
                "paper_reproduction_accepted_empirical_target_rows"
            ),
            "paper_reproduction_row_empirical_target_rows": comparison_summary.get(
                "paper_reproduction_row_empirical_target_rows"
            ),
            "paper_reproduction_row_accepted_empirical_target_rows": (
                comparison_summary.get(
                    "paper_reproduction_row_accepted_empirical_target_rows"
                )
            ),
            "paper_reproduction_row_monte_carlo_target_rows": (
                comparison_summary.get(
                    "paper_reproduction_row_monte_carlo_target_rows"
                )
            ),
            "paper_reproduction_row_accepted_monte_carlo_target_rows": (
                comparison_summary.get(
                    "paper_reproduction_row_accepted_monte_carlo_target_rows"
                )
            ),
            "comparison_provenance_ready": comparison_provenance_ready,
            "comparison_provenance_condition": comparison_provenance_condition,
            "comparison_provenance_summary_claims_ready": (
                comparison_provenance_summary_claims_ready
            ),
            "actual_comparison_row_count": actual_comparison_row_count,
            "actual_provenance_row_count": actual_provenance_row_count,
            "actual_provenance_ready_rows": actual_provenance_ready_rows,
            "actual_tolerance_row_count": actual_tolerance_row_count,
            "required_comparison_row_count": required_comparison_row_count,
            "required_provenance_row_count": required_provenance_row_count,
            "comparison_row_count_shortfall": comparison_row_count_shortfall,
            "provenance_row_count_shortfall": provenance_row_count_shortfall,
            "incomplete_comparison_row_ids": comparison_row_state[
                "incomplete_row_ids"
            ],
            "incomplete_comparison_fields": comparison_row_state[
                "incomplete_fields"
            ],
            "invalid_comparison_row_ids": comparison_row_state["invalid_row_ids"],
            "invalid_comparison_fields": comparison_row_state["invalid_fields"],
            "duplicate_comparison_row_ids": comparison_row_state[
                "duplicate_row_ids"
            ],
            "incomplete_provenance_row_ids": provenance_row_state[
                "incomplete_row_ids"
            ],
            "incomplete_provenance_fields": provenance_row_state[
                "incomplete_fields"
            ],
            "invalid_provenance_row_ids": provenance_row_state[
                "invalid_row_ids"
            ],
            "invalid_provenance_fields": provenance_row_state["invalid_fields"],
            "duplicate_provenance_row_ids": provenance_row_state[
                "duplicate_row_ids"
            ],
            "required_tolerance_contract_names": tolerance_contract_state[
                "required_names"
            ],
            "actual_tolerance_contract_names": tolerance_contract_state[
                "actual_names"
            ],
            "missing_tolerance_contract_names": tolerance_contract_state[
                "missing_names"
            ],
            "duplicate_tolerance_contract_names": tolerance_contract_state[
                "duplicate_names"
            ],
            "incomplete_tolerance_contract_names": tolerance_contract_state[
                "incomplete_names"
            ],
            "invalid_tolerance_contract_names": tolerance_contract_state[
                "invalid_names"
            ],
            "incomplete_tolerance_contract_fields": tolerance_contract_state[
                "incomplete_fields"
            ],
            "invalid_tolerance_contract_fields": tolerance_contract_state[
                "invalid_fields"
            ],
            "milestone_archive_ready": archive_ready,
            "active_blocking_conditions": archive_blockers,
            "monte_carlo_active_blocking_conditions": dict(
                evidence_state.get("active_blocking_conditions") or {}
            ),
            "monte_carlo_stale_evidence_file_count": evidence_state.get(
                "stale_evidence_file_count"
            ),
            "monte_carlo_stale_evidence_files": evidence_state.get(
                "stale_evidence_files"
            ),
            "monte_carlo_stale_evidence_error": evidence_state.get(
                "stale_evidence_error"
            ),
            "evidence_next_action": evidence_state.get("next_action"),
            "evidence_next_chunk_kwargs": evidence_state.get("next_chunk_kwargs"),
            "evidence_next_chunk_rerun_call": evidence_state.get(
                "next_chunk_rerun_call"
            ),
            "evidence_next_chunk_evidence_path": evidence_state.get(
                "next_chunk_evidence_path"
            ),
            "evidence_next_chunk_export_call": evidence_state.get(
                "next_chunk_export_call"
            ),
            "runtime_seconds": comparison_runtime_seconds,
            "empirical_runtime_seconds": empirical_runtime_seconds,
            "monte_carlo_runtime_seconds": monte_carlo_runtime_seconds,
            "rerun_hook": _public_e2e_rerun_command(evidence_state),
            "rerun_hook_packet": _public_e2e_rerun_hook_packet(
                archive_ready=archive_ready,
                archive_blockers=archive_blockers,
                evidence_state=evidence_state,
                comparison_summary=comparison_summary,
                comparison_provenance_ready=comparison_provenance_ready,
                comparison_provenance_condition=comparison_provenance_condition,
                roadmap_state=roadmap_state,
                requirements_state=requirements_state,
                audit_state=audit_state,
            ),
            "paper_anchor": (
                "manuscript/sources/arxiv-2404.11739v3/draft.tex:391-456; "
                "manuscript/sources/arxiv-2404.11739v3/draft.tex:1094-1156"
            ),
            "reference_anchor": (
                "packages/r/TestMechs/R/test_sharp_null.R:90-315; "
                "packages/r/TestMechs/R/nonparametric_bootstrap.R:16-71"
            ),
        },
    )


def _resolve_public_e2e_milestone_audit_status(
    *,
    milestone_audit_status: str | None,
    milestone_audit_path: str | Path | None,
    milestone_version: str,
    planning_dir: str | Path,
) -> str | None:
    if milestone_audit_status is not None:
        return milestone_audit_status
    if milestone_audit_path is not None:
        return milestone_audit_status_from_file(
            milestone_audit_path,
            planning_dir=Path(milestone_audit_path).parent,
        )
    return milestone_audit_status_for_version(
        milestone_version,
        planning_dir=planning_dir,
    )


def _public_e2e_milestone_audit_state(
    milestone_audit_status: str | None,
) -> dict[str, Any]:
    status = "not_checked" if milestone_audit_status is None else str(
        milestone_audit_status
    ).strip().lower()
    if status in {"pass", "passed"}:
        return {
            "milestone_audit_status": "passed",
            "milestone_audit_passes": True,
            "condition": "milestone_audit_passed",
            "resolution_action": "archive_milestone",
        }
    condition = {
        "not_checked": "milestone_audit_missing",
        "missing": "milestone_audit_missing",
        "stale": "milestone_audit_stale",
        "gaps_found": "milestone_audit_gaps_found",
        "failed": "milestone_audit_failed",
    }.get(status, "milestone_audit_not_passed")
    resolution_action = {
        "not_checked": "run_milestone_audit",
        "missing": "run_milestone_audit",
        "stale": "run_milestone_audit",
        "gaps_found": "plan_milestone_gap_closure",
        "failed": "run_milestone_audit",
    }.get(status, "run_milestone_audit")
    return {
        "milestone_audit_status": status,
        "milestone_audit_passes": False,
        "condition": condition,
        "resolution_action": resolution_action,
    }


def _milestone_version_check_prefix(milestone_version: str) -> str:
    normalized = str(milestone_version).strip()
    if normalized and normalized[0].isdigit():
        normalized = f"v{normalized}"
    parts: list[str] = []
    current: list[str] = []
    for char in normalized.lower():
        if char.isalnum():
            current.append(char)
        elif current:
            parts.append("".join(current))
            current = []
    if current:
        parts.append("".join(current))
    return "_".join(parts) if parts else "milestone"


def _public_e2e_summary(
    rows: tuple[dict[str, Any], ...],
    *,
    comparison_summary: dict[str, Any],
    milestone_version: str,
    runtime_seconds: float,
) -> dict[str, Any]:
    status_counts = _count_by(rows, "status")
    archive_row = next(
        row for row in rows if row.get("check_kind") == "milestone_archive_preflight"
    )
    monte_carlo_row = next(
        row for row in rows if row.get("check_kind") == "monte_carlo_reproduction"
    )
    return {
        "milestone_version": milestone_version,
        "row_count": len(rows),
        "check_status_counts": status_counts,
        "public_e2e_contract_ready": all(
            row.get("status") == "pass"
            for row in rows
            if row.get("check_kind")
            in {"empirical_reproduction", "comparison_provenance"}
        ),
        "paper_acceptance_gate_passes": monte_carlo_row.get(
            "paper_acceptance_gate_passes"
        ),
        "paper_acceptance_absolute_tolerance": monte_carlo_row.get(
            "paper_acceptance_absolute_tolerance"
        ),
        "paper_acceptance_z_tolerance": monte_carlo_row.get(
            "paper_acceptance_z_tolerance"
        ),
        "paper_acceptance_max_target_mc_standard_error": monte_carlo_row.get(
            "paper_acceptance_max_target_mc_standard_error"
        ),
        "paper_acceptance_evidence_state_present": monte_carlo_row.get(
            "paper_acceptance_evidence_state_present"
        ),
        "paper_acceptance_evidence_state_missing_fields": monte_carlo_row.get(
            "paper_acceptance_evidence_state_missing_fields"
        ),
        "paper_acceptance_evidence_state_malformed_fields": monte_carlo_row.get(
            "paper_acceptance_evidence_state_malformed_fields"
        ),
        "paper_acceptance_evidence_state_empty_fields": monte_carlo_row.get(
            "paper_acceptance_evidence_state_empty_fields"
        ),
        "paper_acceptance_evidence_state_duplicate_rows": monte_carlo_row.get(
            "paper_acceptance_evidence_state_duplicate_rows"
        ),
        "paper_reproduction_gate_passes": archive_row.get(
            "paper_reproduction_gate_passes"
        ),
        "paper_reproduction_row_gate_passes": archive_row.get(
            "paper_reproduction_row_gate_passes"
        ),
        "paper_reproduction_gate_consistent": archive_row.get(
            "paper_reproduction_gate_consistent"
        ),
        "paper_reproduction_target_rows": archive_row.get(
            "paper_reproduction_target_rows"
        ),
        "paper_reproduction_accepted_target_rows": archive_row.get(
            "paper_reproduction_accepted_target_rows"
        ),
        "paper_reproduction_row_target_rows": archive_row.get(
            "paper_reproduction_row_target_rows"
        ),
        "paper_reproduction_row_accepted_target_rows": archive_row.get(
            "paper_reproduction_row_accepted_target_rows"
        ),
        "paper_reproduction_empirical_target_rows": archive_row.get(
            "paper_reproduction_empirical_target_rows"
        ),
        "paper_reproduction_accepted_empirical_target_rows": archive_row.get(
            "paper_reproduction_accepted_empirical_target_rows"
        ),
        "paper_reproduction_row_empirical_target_rows": archive_row.get(
            "paper_reproduction_row_empirical_target_rows"
        ),
        "paper_reproduction_row_accepted_empirical_target_rows": archive_row.get(
            "paper_reproduction_row_accepted_empirical_target_rows"
        ),
        "paper_reproduction_row_monte_carlo_target_rows": archive_row.get(
            "paper_reproduction_row_monte_carlo_target_rows"
        ),
        "paper_reproduction_row_accepted_monte_carlo_target_rows": archive_row.get(
            "paper_reproduction_row_accepted_monte_carlo_target_rows"
        ),
        "comparison_provenance_ready": archive_row.get(
            "comparison_provenance_ready"
        ),
        "comparison_provenance_condition": archive_row.get(
            "comparison_provenance_condition"
        ),
        "comparison_provenance_summary_claims_ready": archive_row.get(
            "comparison_provenance_summary_claims_ready"
        ),
        "actual_comparison_row_count": archive_row.get(
            "actual_comparison_row_count"
        ),
        "actual_provenance_row_count": archive_row.get(
            "actual_provenance_row_count"
        ),
        "actual_provenance_ready_rows": archive_row.get(
            "actual_provenance_ready_rows"
        ),
        "actual_tolerance_row_count": archive_row.get(
            "actual_tolerance_row_count"
        ),
        "required_comparison_row_count": archive_row.get(
            "required_comparison_row_count"
        ),
        "required_provenance_row_count": archive_row.get(
            "required_provenance_row_count"
        ),
        "comparison_row_count_shortfall": archive_row.get(
            "comparison_row_count_shortfall"
        ),
        "provenance_row_count_shortfall": archive_row.get(
            "provenance_row_count_shortfall"
        ),
        "incomplete_comparison_row_ids": archive_row.get(
            "incomplete_comparison_row_ids"
        ),
        "incomplete_comparison_fields": archive_row.get(
            "incomplete_comparison_fields"
        ),
        "invalid_comparison_row_ids": archive_row.get(
            "invalid_comparison_row_ids"
        ),
        "invalid_comparison_fields": archive_row.get(
            "invalid_comparison_fields"
        ),
        "duplicate_comparison_row_ids": archive_row.get(
            "duplicate_comparison_row_ids"
        ),
        "incomplete_provenance_row_ids": archive_row.get(
            "incomplete_provenance_row_ids"
        ),
        "incomplete_provenance_fields": archive_row.get(
            "incomplete_provenance_fields"
        ),
        "invalid_provenance_row_ids": archive_row.get(
            "invalid_provenance_row_ids"
        ),
        "invalid_provenance_fields": archive_row.get(
            "invalid_provenance_fields"
        ),
        "duplicate_provenance_row_ids": archive_row.get(
            "duplicate_provenance_row_ids"
        ),
        "required_tolerance_contract_names": archive_row.get(
            "required_tolerance_contract_names"
        ),
        "actual_tolerance_contract_names": archive_row.get(
            "actual_tolerance_contract_names"
        ),
        "missing_tolerance_contract_names": archive_row.get(
            "missing_tolerance_contract_names"
        ),
        "duplicate_tolerance_contract_names": archive_row.get(
            "duplicate_tolerance_contract_names"
        ),
        "incomplete_tolerance_contract_names": archive_row.get(
            "incomplete_tolerance_contract_names"
        ),
        "invalid_tolerance_contract_names": archive_row.get(
            "invalid_tolerance_contract_names"
        ),
        "incomplete_tolerance_contract_fields": archive_row.get(
            "incomplete_tolerance_contract_fields"
        ),
        "invalid_tolerance_contract_fields": archive_row.get(
            "invalid_tolerance_contract_fields"
        ),
        "low_budget_probe_blocked": monte_carlo_row.get("low_budget_probe_blocked"),
        "milestone_archive_ready": archive_row.get("milestone_archive_ready"),
        "archive_active_blocking_conditions": archive_row.get(
            "active_blocking_conditions"
        ),
        "roadmap_analysis_condition": archive_row.get("roadmap_analysis_condition"),
        "roadmap_analysis_missing_fields": archive_row.get(
            "roadmap_analysis_missing_fields"
        ),
        "roadmap_analysis_malformed_fields": archive_row.get(
            "roadmap_analysis_malformed_fields"
        ),
        "roadmap_analysis_mismatch_fields": archive_row.get(
            "roadmap_analysis_mismatch_fields"
        ),
        "roadmap_total_plans": archive_row.get("roadmap_total_plans"),
        "roadmap_total_summaries": archive_row.get("roadmap_total_summaries"),
        "roadmap_phase_plan_count_sum": archive_row.get(
            "roadmap_phase_plan_count_sum"
        ),
        "roadmap_phase_summary_count_sum": archive_row.get(
            "roadmap_phase_summary_count_sum"
        ),
        "requirements_complete": archive_row.get("requirements_complete"),
        "requirements_analysis_condition": archive_row.get(
            "requirements_analysis_condition"
        ),
        "requirements_analysis_missing_fields": archive_row.get(
            "requirements_analysis_missing_fields"
        ),
        "requirements_analysis_malformed_fields": archive_row.get(
            "requirements_analysis_malformed_fields"
        ),
        "requirements_total": archive_row.get("requirements_total"),
        "requirements_completed": archive_row.get("requirements_completed"),
        "requirements_pending": archive_row.get("requirements_pending"),
        "requirements_expected_ids": archive_row.get("requirements_expected_ids"),
        "requirements_expected_id_count": archive_row.get(
            "requirements_expected_id_count"
        ),
        "requirements_expected_unique_id_count": archive_row.get(
            "requirements_expected_unique_id_count"
        ),
        "requirements_duplicate_expected_ids": archive_row.get(
            "requirements_duplicate_expected_ids"
        ),
        "requirements_traceability_rows": archive_row.get(
            "requirements_traceability_rows"
        ),
        "requirements_traceability_row_count": archive_row.get(
            "requirements_traceability_row_count"
        ),
        "requirements_traceability_unique_row_count": archive_row.get(
            "requirements_traceability_unique_row_count"
        ),
        "requirements_duplicate_traceability_rows": archive_row.get(
            "requirements_duplicate_traceability_rows"
        ),
        "requirements_missing_traceability_ids": archive_row.get(
            "requirements_missing_traceability_ids"
        ),
        "requirements_unexpected_traceability_ids": archive_row.get(
            "requirements_unexpected_traceability_ids"
        ),
        "requirements_non_complete_rows": archive_row.get(
            "requirements_non_complete_rows"
        ),
        "milestone_audit_condition": archive_row.get("milestone_audit_condition"),
        "milestone_audit_resolution_action": archive_row.get(
            "milestone_audit_resolution_action"
        ),
        "monte_carlo_active_blocking_conditions": monte_carlo_row.get(
            "active_blocking_conditions"
        ),
        "monte_carlo_stale_evidence_file_count": monte_carlo_row.get(
            "stale_evidence_file_count"
        ),
        "monte_carlo_stale_evidence_files": monte_carlo_row.get(
            "stale_evidence_files"
        ),
        "monte_carlo_stale_evidence_error": monte_carlo_row.get(
            "stale_evidence_error"
        ),
        "evidence_next_action": archive_row.get("evidence_next_action"),
        "evidence_next_chunk_kwargs": archive_row.get("evidence_next_chunk_kwargs"),
        "evidence_next_chunk_rerun_call": archive_row.get(
            "evidence_next_chunk_rerun_call"
        ),
        "evidence_next_chunk_evidence_path": archive_row.get(
            "evidence_next_chunk_evidence_path"
        ),
        "evidence_next_chunk_export_call": archive_row.get(
            "evidence_next_chunk_export_call"
        ),
        "empirical_passed_target_rows": next(
            row for row in rows if row.get("check_kind") == "empirical_reproduction"
        ).get("passed_target_rows"),
        "empirical_max_absolute_difference": next(
            row for row in rows if row.get("check_kind") == "empirical_reproduction"
        ).get("max_absolute_difference"),
        "empirical_runtime_seconds": next(
            row for row in rows if row.get("check_kind") == "empirical_reproduction"
        ).get("runtime_seconds"),
        "monte_carlo_executed_result_rows": monte_carlo_row.get(
            "executed_result_rows"
        ),
        "monte_carlo_paper_result_rows": monte_carlo_row.get("paper_result_rows"),
        "monte_carlo_covered_result_rows": monte_carlo_row.get(
            "covered_result_rows"
        ),
        "monte_carlo_scheduled_draws": monte_carlo_row.get("scheduled_draws"),
        "monte_carlo_executed_draws": monte_carlo_row.get("executed_draws"),
        "monte_carlo_scheduled_bootstrap_draws": monte_carlo_row.get(
            "scheduled_bootstrap_draws"
        ),
        "monte_carlo_executed_bootstrap_draws": monte_carlo_row.get(
            "executed_bootstrap_draws"
        ),
        "monte_carlo_max_target_mc_standard_error": monte_carlo_row.get(
            "max_target_mc_standard_error"
        ),
        "monte_carlo_max_rejection_rate_absolute_error": monte_carlo_row.get(
            "max_rejection_rate_absolute_error"
        ),
        "monte_carlo_runtime_seconds": monte_carlo_row.get("runtime_seconds"),
        "comparison_row_count": comparison_summary.get("comparison_row_count"),
        "provenance_row_count": comparison_summary.get("provenance_row_count"),
        "tolerance_row_count": comparison_summary.get("tolerance_row_count"),
        "comparison_runtime_seconds": comparison_summary.get("runtime_seconds"),
        "rerun_hook": archive_row.get("rerun_hook"),
        "rerun_hook_packet": archive_row.get("rerun_hook_packet"),
        "next_action": (
            "archive_ready"
            if archive_row.get("milestone_archive_ready") is True
            else "preserve_archive_block_until_full_paper_evidence_and_audit_pass"
        ),
        "runtime_seconds": runtime_seconds,
        "paper_anchors": comparison_summary.get("paper_anchors"),
        "reference_anchors": comparison_summary.get("reference_anchors"),
    }


def _public_e2e_rerun_hook_packet(
    *,
    archive_ready: bool,
    archive_blockers: dict[str, int],
    evidence_state: dict[str, Any],
    comparison_summary: dict[str, Any],
    comparison_provenance_ready: bool,
    comparison_provenance_condition: str,
    roadmap_state: dict[str, Any],
    requirements_state: dict[str, Any],
    audit_state: dict[str, Any],
) -> dict[str, Any]:
    rerun_command = _public_e2e_rerun_command(evidence_state)
    return {
        "next_action": (
            "archive_ready"
            if archive_ready
            else "preserve_archive_block_until_full_paper_evidence_and_audit_pass"
        ),
        "rerun_command": rerun_command,
        "milestone_archive_ready": archive_ready,
        "archive_delete_tag_allowed": archive_ready,
        "active_blocking_conditions": dict(archive_blockers),
        "roadmap_disk_complete": roadmap_state.get("roadmap_disk_complete"),
        "roadmap_analysis_condition": roadmap_state.get("condition"),
        "roadmap_analysis_missing_fields": roadmap_state.get("missing_fields"),
        "roadmap_analysis_malformed_fields": roadmap_state.get("malformed_fields"),
        "roadmap_analysis_mismatch_fields": roadmap_state.get("mismatch_fields"),
        "roadmap_total_plans": roadmap_state.get("total_plans"),
        "roadmap_total_summaries": roadmap_state.get("total_summaries"),
        "roadmap_phase_plan_count_sum": roadmap_state.get("phase_plan_count_sum"),
        "roadmap_phase_summary_count_sum": roadmap_state.get(
            "phase_summary_count_sum"
        ),
        "requirements_analysis_condition": requirements_state.get("condition"),
        "requirements_analysis_missing_fields": requirements_state.get(
            "missing_fields"
        ),
        "requirements_analysis_malformed_fields": requirements_state.get(
            "malformed_fields"
        ),
        "requirements_complete": requirements_state.get("requirements_complete"),
        "requirements_total": requirements_state.get("total_requirements"),
        "requirements_completed": requirements_state.get("completed_requirements"),
        "requirements_pending": requirements_state.get("pending_requirements"),
        "requirements_expected_ids": requirements_state.get("expected_ids"),
        "requirements_expected_id_count": requirements_state.get("expected_id_count"),
        "requirements_expected_unique_id_count": requirements_state.get(
            "expected_unique_id_count"
        ),
        "requirements_duplicate_expected_ids": requirements_state.get(
            "duplicate_expected_ids"
        ),
        "requirements_traceability_rows": requirements_state.get(
            "traceability_rows"
        ),
        "requirements_traceability_row_count": len(
            requirements_state.get("traceability_rows") or ()
        ),
        "requirements_traceability_unique_row_count": requirements_state.get(
            "traceability_unique_row_count"
        ),
        "requirements_duplicate_traceability_rows": requirements_state.get(
            "duplicate_traceability_rows"
        ),
        "requirements_missing_traceability_ids": requirements_state.get(
            "missing_traceability_ids"
        ),
        "requirements_unexpected_traceability_ids": requirements_state.get(
            "unexpected_traceability_ids"
        ),
        "requirements_non_complete_rows": requirements_state.get(
            "non_complete_rows"
        ),
        "milestone_audit_status": audit_state.get("milestone_audit_status"),
        "milestone_audit_passes": audit_state.get("milestone_audit_passes"),
        "milestone_audit_condition": audit_state.get("condition"),
        "milestone_audit_resolution_action": audit_state.get("resolution_action"),
        "paper_result_rows": evidence_state.get("paper_result_rows"),
        "covered_result_rows": evidence_state.get("covered_result_rows"),
        "executed_result_rows": evidence_state.get("executed_result_rows"),
        "scheduled_draws": evidence_state.get("scheduled_draws"),
        "executed_draws": evidence_state.get("executed_draws"),
        "scheduled_bootstrap_draws": evidence_state.get("scheduled_bootstrap_draws"),
        "executed_bootstrap_draws": evidence_state.get("executed_bootstrap_draws"),
        "stale_evidence_file_count": evidence_state.get("stale_evidence_file_count"),
        "stale_evidence_files": evidence_state.get("stale_evidence_files"),
        "stale_evidence_error": evidence_state.get("stale_evidence_error"),
        "max_rejection_rate_absolute_error": evidence_state.get(
            "max_rejection_rate_absolute_error"
        ),
        "max_target_mc_standard_error": evidence_state.get(
            "max_target_mc_standard_error"
        ),
        "paper_acceptance_absolute_tolerance": (
            _PAPER_ACCEPTANCE_ABSOLUTE_TOLERANCE
        ),
        "paper_acceptance_z_tolerance": _PAPER_ACCEPTANCE_Z_TOLERANCE,
        "paper_acceptance_max_target_mc_standard_error": (
            _PAPER_ACCEPTANCE_MAX_TARGET_MC_STANDARD_ERROR
        ),
        "paper_acceptance_evidence_state_present": evidence_state.get(
            "paper_acceptance_evidence_state_present",
            False,
        ),
        "paper_acceptance_evidence_state_missing_fields": evidence_state.get(
            "paper_acceptance_evidence_state_missing_fields",
            (),
        ),
        "paper_acceptance_evidence_state_malformed_fields": evidence_state.get(
            "paper_acceptance_evidence_state_malformed_fields",
            (),
        ),
        "paper_acceptance_evidence_state_empty_fields": evidence_state.get(
            "paper_acceptance_evidence_state_empty_fields",
            (),
        ),
        "paper_acceptance_evidence_state_duplicate_rows": evidence_state.get(
            "paper_acceptance_evidence_state_duplicate_rows",
            0,
        ),
        "paper_acceptance_gate_passes": evidence_state.get(
            "paper_acceptance_gate_passes"
        )
        is True,
        "paper_reproduction_gate_passes": comparison_summary.get(
            "paper_reproduction_gate_passes"
        ),
        "paper_reproduction_row_gate_passes": comparison_summary.get(
            "paper_reproduction_row_gate_passes"
        ),
        "paper_reproduction_gate_consistent": comparison_summary.get(
            "paper_reproduction_gate_consistent"
        ),
        "paper_reproduction_target_rows": comparison_summary.get(
            "paper_reproduction_target_rows"
        ),
        "paper_reproduction_accepted_target_rows": comparison_summary.get(
            "paper_reproduction_accepted_target_rows"
        ),
        "paper_reproduction_row_target_rows": comparison_summary.get(
            "paper_reproduction_row_target_rows"
        ),
        "paper_reproduction_row_accepted_target_rows": comparison_summary.get(
            "paper_reproduction_row_accepted_target_rows"
        ),
        "paper_reproduction_empirical_target_rows": comparison_summary.get(
            "paper_reproduction_empirical_target_rows"
        ),
        "paper_reproduction_accepted_empirical_target_rows": comparison_summary.get(
            "paper_reproduction_accepted_empirical_target_rows"
        ),
        "paper_reproduction_row_empirical_target_rows": comparison_summary.get(
            "paper_reproduction_row_empirical_target_rows"
        ),
        "paper_reproduction_row_accepted_empirical_target_rows": (
            comparison_summary.get(
                "paper_reproduction_row_accepted_empirical_target_rows"
            )
        ),
        "paper_reproduction_row_monte_carlo_target_rows": comparison_summary.get(
            "paper_reproduction_row_monte_carlo_target_rows"
        ),
        "paper_reproduction_row_accepted_monte_carlo_target_rows": (
            comparison_summary.get(
                "paper_reproduction_row_accepted_monte_carlo_target_rows"
            )
        ),
        "comparison_provenance_ready": comparison_provenance_ready,
        "comparison_provenance_condition": comparison_provenance_condition,
        "comparison_provenance_summary_claims_ready": comparison_summary.get(
            "comparison_provenance_summary_claims_ready"
        ),
        "actual_comparison_row_count": comparison_summary.get(
            "actual_comparison_row_count"
        ),
        "actual_provenance_row_count": comparison_summary.get(
            "actual_provenance_row_count"
        ),
        "actual_provenance_ready_rows": comparison_summary.get(
            "actual_provenance_ready_rows"
        ),
        "actual_tolerance_row_count": comparison_summary.get(
            "actual_tolerance_row_count"
        ),
        "required_comparison_row_count": comparison_summary.get(
            "required_comparison_row_count"
        ),
        "required_provenance_row_count": comparison_summary.get(
            "required_provenance_row_count"
        ),
        "comparison_row_count_shortfall": comparison_summary.get(
            "comparison_row_count_shortfall"
        ),
        "provenance_row_count_shortfall": comparison_summary.get(
            "provenance_row_count_shortfall"
        ),
        "incomplete_comparison_row_ids": comparison_summary.get(
            "incomplete_comparison_row_ids"
        ),
        "incomplete_comparison_fields": comparison_summary.get(
            "incomplete_comparison_fields"
        ),
        "invalid_comparison_row_ids": comparison_summary.get(
            "invalid_comparison_row_ids"
        ),
        "invalid_comparison_fields": comparison_summary.get(
            "invalid_comparison_fields"
        ),
        "duplicate_comparison_row_ids": comparison_summary.get(
            "duplicate_comparison_row_ids"
        ),
        "incomplete_provenance_row_ids": comparison_summary.get(
            "incomplete_provenance_row_ids"
        ),
        "incomplete_provenance_fields": comparison_summary.get(
            "incomplete_provenance_fields"
        ),
        "invalid_provenance_row_ids": comparison_summary.get(
            "invalid_provenance_row_ids"
        ),
        "invalid_provenance_fields": comparison_summary.get(
            "invalid_provenance_fields"
        ),
        "duplicate_provenance_row_ids": comparison_summary.get(
            "duplicate_provenance_row_ids"
        ),
        "required_tolerance_contract_names": comparison_summary.get(
            "required_tolerance_contract_names"
        ),
        "actual_tolerance_contract_names": comparison_summary.get(
            "actual_tolerance_contract_names"
        ),
        "missing_tolerance_contract_names": comparison_summary.get(
            "missing_tolerance_contract_names"
        ),
        "duplicate_tolerance_contract_names": comparison_summary.get(
            "duplicate_tolerance_contract_names"
        ),
        "incomplete_tolerance_contract_names": comparison_summary.get(
            "incomplete_tolerance_contract_names"
        ),
        "invalid_tolerance_contract_names": comparison_summary.get(
            "invalid_tolerance_contract_names"
        ),
        "incomplete_tolerance_contract_fields": comparison_summary.get(
            "incomplete_tolerance_contract_fields"
        ),
        "invalid_tolerance_contract_fields": comparison_summary.get(
            "invalid_tolerance_contract_fields"
        ),
        "evidence_next_action": evidence_state.get("next_action"),
        "evidence_next_chunk_kwargs": evidence_state.get("next_chunk_kwargs"),
        "evidence_next_chunk_rerun_call": evidence_state.get("next_chunk_rerun_call"),
        "evidence_next_chunk_evidence_path": evidence_state.get(
            "next_chunk_evidence_path"
        ),
        "evidence_next_chunk_export_call": evidence_state.get("next_chunk_export_call"),
        "exit_criteria": (
            "roadmap_disk_complete == True and requirements_complete == True "
            "and roadmap_total_plans == roadmap_total_summaries "
            "and milestone_audit_status == 'passed' "
            "and paper_acceptance_gate_passes == True "
            "and paper_reproduction_gate_passes == True "
            "and paper_reproduction_gate_consistent == True "
            "and comparison_provenance_ready == True"
        ),
    }


def _public_e2e_rerun_command(evidence_state: dict[str, Any]) -> str | None:
    return (
        evidence_state.get("next_chunk_export_call")
        or evidence_state.get("next_chunk_rerun_call")
        or evidence_state.get("next_action")
    )


def _public_e2e_paper_acceptance_derived_blockers(
    evidence_row: dict[str, Any],
) -> tuple[dict[str, int], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    blockers: dict[str, int] = {}
    missing_fields: list[str] = []
    malformed_fields: list[str] = []
    empty_fields: list[str] = []

    paper_acceptance_gate_passes = evidence_row.get("paper_acceptance_gate_passes")
    if (
        "paper_acceptance_gate_passes" not in evidence_row
        or paper_acceptance_gate_passes is None
    ):
        missing_fields.append("paper_acceptance_gate_passes")
    elif not isinstance(paper_acceptance_gate_passes, bool):
        malformed_fields.append("paper_acceptance_gate_passes")
    if missing_fields or malformed_fields:
        blockers["paper_acceptance_evidence_state_inconsistent"] = 1
        return (
            blockers,
            tuple(dict.fromkeys(missing_fields)),
            tuple(dict.fromkeys(malformed_fields)),
            (),
        )
    if paper_acceptance_gate_passes is not True:
        return {}, (), (), ()

    def read_count(field: str) -> int | None:
        if field not in evidence_row or evidence_row.get(field) is None:
            missing_fields.append(field)
            return None
        value = _coerce_nonnegative_int(evidence_row.get(field))
        if value is None:
            malformed_fields.append(field)
        return value

    def read_metric(field: str) -> float | None:
        if field not in evidence_row or evidence_row.get(field) is None:
            missing_fields.append(field)
            return None
        value = _coerce_nonnegative_float(evidence_row.get(field))
        if value is None:
            malformed_fields.append(field)
        return value

    paper_rows = read_count("paper_result_rows")
    covered_rows = read_count("covered_result_rows")
    executed_rows = read_count("executed_result_rows")
    scheduled_draws = read_count("scheduled_draws")
    executed_draws = read_count("executed_draws")
    scheduled_bootstrap_draws = read_count("scheduled_bootstrap_draws")
    executed_bootstrap_draws = read_count("executed_bootstrap_draws")
    max_rejection_rate_absolute_error = read_metric(
        "max_rejection_rate_absolute_error"
    )
    max_target_mc_standard_error = read_metric("max_target_mc_standard_error")

    positive_evidence_counts = {
        "paper_result_rows": paper_rows,
        "covered_result_rows": covered_rows,
        "executed_result_rows": executed_rows,
        "scheduled_draws": scheduled_draws,
        "executed_draws": executed_draws,
        "scheduled_bootstrap_draws": scheduled_bootstrap_draws,
        "executed_bootstrap_draws": executed_bootstrap_draws,
    }
    empty_fields.extend(
        field
        for field, value in positive_evidence_counts.items()
        if value is not None and value <= 0
    )
    if empty_fields:
        blockers["paper_acceptance_evidence_state_empty"] = len(empty_fields)

    if paper_rows is not None:
        represented_rows = tuple(
            value for value in (covered_rows, executed_rows) if value is not None
        )
        if represented_rows:
            coverage_shortfall = max(paper_rows - min(represented_rows), 0)
            if coverage_shortfall:
                blockers["paper_coverage_shortfall_rows"] = coverage_shortfall
            coverage_overshoot = max(max(represented_rows) - paper_rows, 0)
            if coverage_overshoot:
                blockers[
                    "paper_acceptance_row_count_exceeds_paper_contract"
                ] = coverage_overshoot

    if (
        scheduled_draws is not None
        and executed_draws is not None
        and executed_draws < scheduled_draws
    ):
        blockers["target_replication_shortfall_rows"] = 1
    if (
        scheduled_draws is not None
        and executed_draws is not None
        and executed_draws > scheduled_draws
    ):
        blockers["paper_acceptance_executed_draws_exceed_schedule"] = (
            executed_draws - scheduled_draws
        )

    if (
        scheduled_bootstrap_draws is not None
        and executed_bootstrap_draws is not None
        and executed_bootstrap_draws < scheduled_bootstrap_draws
    ):
        blockers["bootstrap_replication_shortfall_rows"] = 1
    if (
        scheduled_bootstrap_draws is not None
        and executed_bootstrap_draws is not None
        and executed_bootstrap_draws > scheduled_bootstrap_draws
    ):
        blockers["paper_acceptance_executed_bootstrap_draws_exceed_schedule"] = (
            executed_bootstrap_draws - scheduled_bootstrap_draws
        )

    observed_counts = {
        "paper_result_rows": paper_rows,
        "covered_result_rows": covered_rows,
        "executed_result_rows": executed_rows,
        "scheduled_draws": scheduled_draws,
        "executed_draws": executed_draws,
        "scheduled_bootstrap_draws": scheduled_bootstrap_draws,
        "executed_bootstrap_draws": executed_bootstrap_draws,
    }
    for field, required_minimum in _PAPER_ACCEPTANCE_CONTRACT_MINIMA.items():
        observed = observed_counts[field]
        if observed is not None and 0 < observed < required_minimum:
            blockers[
                f"paper_acceptance_{field}_under_paper_contract"
            ] = required_minimum - observed

    if (
        max_target_mc_standard_error is not None
        and max_target_mc_standard_error
        > _PAPER_ACCEPTANCE_MAX_TARGET_MC_STANDARD_ERROR + 1e-12
    ):
        blockers[
            "paper_acceptance_target_mc_standard_error_exceeds_paper_contract"
        ] = 1

    if (
        max_rejection_rate_absolute_error is not None
        and max_target_mc_standard_error is not None
    ):
        sampling_error_band = (
            _PAPER_ACCEPTANCE_Z_TOLERANCE * max_target_mc_standard_error
        )
        if (
            max_rejection_rate_absolute_error
            > _PAPER_ACCEPTANCE_ABSOLUTE_TOLERANCE + 1e-12
            and max_rejection_rate_absolute_error > sampling_error_band + 1e-12
        ):
            blockers["paper_acceptance_rejection_rate_tolerance_exceeded"] = 1

    if blockers or missing_fields or malformed_fields:
        blockers["paper_acceptance_evidence_state_inconsistent"] = 1
    return (
        blockers,
        tuple(dict.fromkeys(missing_fields)),
        tuple(dict.fromkeys(malformed_fields)),
        tuple(dict.fromkeys(empty_fields)),
    )


def _public_e2e_active_blocking_conditions(
    value: Any,
) -> tuple[dict[str, int], int]:
    if value in (None, False):
        return {}, 0
    if not isinstance(value, Mapping):
        return {}, 1

    blockers: dict[str, int] = {}
    malformed_count = 0
    for raw_condition, raw_count in value.items():
        if not isinstance(raw_condition, str) or not raw_condition.strip():
            malformed_count += 1
            continue
        count = _coerce_nonnegative_int(raw_count)
        if count is None:
            malformed_count += 1
            continue
        if count > 0:
            blockers[raw_condition.strip()] = count
    return blockers, malformed_count


def _public_e2e_monte_carlo_evidence_row(
    monte_carlo_rows: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    evidence_rows = tuple(
        row
        for row in monte_carlo_rows
        if row.get("case_id") == "monte_carlo_paper_acceptance_evidence_state"
    )
    if evidence_rows:
        duplicate_count = len(evidence_rows) if len(evidence_rows) > 1 else 0
        for row in evidence_rows:
            evidence_row = dict(row)
            evidence_row["paper_acceptance_evidence_state_present"] = True
            evidence_row["paper_acceptance_evidence_state_duplicate_rows"] = (
                duplicate_count
            )
            active_blockers, malformed_active_blockers = (
                _public_e2e_active_blocking_conditions(
                    evidence_row.get("active_blocking_conditions")
                )
            )
            if malformed_active_blockers:
                active_blockers[
                    "paper_acceptance_active_blocking_conditions_malformed"
                ] = malformed_active_blockers
            if duplicate_count:
                active_blockers[
                    "paper_acceptance_evidence_state_duplicate_rows"
                ] = duplicate_count
            (
                derived_blockers,
                missing_fields,
                malformed_fields,
                empty_fields,
            ) = _public_e2e_paper_acceptance_derived_blockers(evidence_row)
            evidence_row[
                "paper_acceptance_evidence_state_missing_fields"
            ] = missing_fields
            evidence_row[
                "paper_acceptance_evidence_state_malformed_fields"
            ] = malformed_fields
            evidence_row["paper_acceptance_evidence_state_empty_fields"] = empty_fields
            if missing_fields:
                active_blockers[
                    "paper_acceptance_evidence_state_missing_fields"
                ] = len(missing_fields)
            if malformed_fields:
                active_blockers[
                    "paper_acceptance_evidence_state_malformed_fields"
                ] = len(malformed_fields)
            if empty_fields:
                active_blockers[
                    "paper_acceptance_evidence_state_empty_fields"
                ] = len(empty_fields)
            for condition, count in derived_blockers.items():
                active_blockers[condition] = max(
                    int(active_blockers.get(condition, 0)),
                    int(count),
                )
            evidence_row["active_blocking_conditions"] = active_blockers
            if (
                evidence_row.get("paper_acceptance_gate_passes") is True
                and active_blockers
            ):
                active_blockers["paper_acceptance_evidence_state_inconsistent"] = 1
                evidence_row["paper_acceptance_gate_passes"] = False
                evidence_row["active_blocking_conditions"] = active_blockers
                if evidence_row.get("next_action") in {None, "archive_ready"}:
                    evidence_row["next_action"] = "restore_paper_rerun_budget"
            elif active_blockers:
                evidence_row["active_blocking_conditions"] = active_blockers
                if evidence_row.get("next_action") in {None, "archive_ready"}:
                    evidence_row["next_action"] = "restore_paper_rerun_budget"
            return evidence_row
    if monte_carlo_rows:
        fallback = dict(monte_carlo_rows[-1])
        active_blockers, malformed_active_blockers = (
            _public_e2e_active_blocking_conditions(
                fallback.get("active_blocking_conditions")
            )
        )
        if malformed_active_blockers:
            active_blockers[
                "paper_acceptance_active_blocking_conditions_malformed"
            ] = malformed_active_blockers
        active_blockers["paper_acceptance_evidence_state_missing"] = 1
        fallback["case_id"] = "monte_carlo_paper_acceptance_evidence_state_missing"
        fallback["paper_acceptance_evidence_state_present"] = False
        fallback["paper_acceptance_gate_passes"] = False
        fallback["paper_acceptance_evidence_state_missing_fields"] = ()
        fallback["paper_acceptance_evidence_state_malformed_fields"] = ()
        fallback["paper_acceptance_evidence_state_empty_fields"] = ()
        fallback["paper_acceptance_evidence_state_duplicate_rows"] = 0
        fallback["active_blocking_conditions"] = active_blockers
        if fallback.get("next_action") in {None, "archive_ready"}:
            fallback["next_action"] = "restore_paper_rerun_budget"
        return fallback
    return {
        "case_id": "monte_carlo_paper_acceptance_evidence_state_missing",
        "paper_acceptance_evidence_state_present": False,
        "paper_acceptance_gate_passes": False,
        "paper_acceptance_evidence_state_missing_fields": (),
        "paper_acceptance_evidence_state_malformed_fields": (),
        "paper_acceptance_evidence_state_empty_fields": (),
        "paper_acceptance_evidence_state_duplicate_rows": 0,
        "active_blocking_conditions": {"paper_acceptance_evidence_state_missing": 1},
        "next_action": "restore_paper_rerun_budget",
    }


def _coerce_nonnegative_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer() and value >= 0:
            return int(value)
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if re.fullmatch(r"\d+", stripped):
            return int(stripped)
        return None
    return None


def _coerce_nonnegative_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) and numeric >= 0.0 else None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            numeric = float(stripped)
        except ValueError:
            return None
        return numeric if math.isfinite(numeric) and numeric >= 0.0 else None
    return None


def _coerce_requirements_non_complete_rows(
    value: Any,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if value is None:
        return (), ()
    if isinstance(value, tuple):
        rows = value
    elif isinstance(value, list):
        rows = tuple(value)
    else:
        return (), ("requirements_non_complete_rows",)
    if not all(isinstance(row, str) for row in rows):
        return (), ("requirements_non_complete_rows",)
    return rows, ()


def _coerce_requirements_traceability_rows(
    value: Any,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if value is None:
        return (), ()
    if isinstance(value, tuple):
        rows = value
    elif isinstance(value, list):
        rows = tuple(value)
    else:
        return (), ("requirements_traceability_rows",)
    normalized: list[str] = []
    for row in rows:
        if isinstance(row, str) and row.strip():
            normalized.append(row.strip())
        elif isinstance(row, Mapping):
            requirement = row.get("requirement") or row.get("requirement_id")
            status = row.get("status")
            if isinstance(requirement, str) and requirement.strip():
                if isinstance(status, str) and status.strip():
                    normalized.append(f"{requirement.strip()}:{status.strip()}")
                else:
                    normalized.append(requirement.strip())
            else:
                return (), ("requirements_traceability_rows",)
        else:
            return (), ("requirements_traceability_rows",)
    return tuple(normalized), ()


def _requirements_traceability_requirement_ids(
    traceability_rows: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(row.split(":", maxsplit=1)[0].strip() for row in traceability_rows)


def _coerce_requirements_expected_ids(
    value: Any,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    rows, malformed_fields = _coerce_requirements_traceability_rows(value)
    if malformed_fields:
        return (), ("requirements_expected_ids",)
    return _requirements_traceability_requirement_ids(rows), ()


def _duplicate_requirement_ids(requirement_ids: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for requirement_id in requirement_ids:
        if requirement_id in seen and requirement_id not in duplicates:
            duplicates.append(requirement_id)
        seen.add(requirement_id)
    return tuple(duplicates)


def _roadmap_analysis_archive_state(
    roadmap_analysis: dict[str, Any] | None,
) -> dict[str, Any]:
    if roadmap_analysis is None:
        return {
            "roadmap_analysis_available": False,
            "roadmap_disk_complete": None,
            "condition": "roadmap_not_checked",
            "condition_value": 1,
            "missing_fields": (),
            "malformed_fields": (),
            "mismatch_fields": (),
            "total_plans": None,
            "total_summaries": None,
            "phase_plan_count_sum": None,
            "phase_summary_count_sum": None,
        }

    missing_fields: list[str] = []
    malformed_fields: list[str] = []
    mismatch_fields: list[str] = []
    phases = roadmap_analysis.get("phases")
    completed_from_phases: int | None = None
    incomplete_from_phases: int | None = None
    phase_plan_count_sum: int | None = None
    phase_summary_count_sum: int | None = None
    if not isinstance(phases, list) or len(phases) == 0:
        missing_fields.append("phases")
    else:
        phase_records: list[tuple[dict[str, Any], int | None, int | None]] = []
        for phase in phases:
            if not isinstance(phase, Mapping):
                malformed_fields.append("phases")
                continue
            phase_mapping = dict(phase)
            phase_number = phase_mapping.get("number")
            if phase_number is None or (
                isinstance(phase_number, str) and not phase_number.strip()
            ):
                missing_fields.append("phases.number")
            elif isinstance(phase_number, bool) or not isinstance(
                phase_number,
                (str, int, float),
            ):
                malformed_fields.append("phases.number")
            elif isinstance(phase_number, float) and not math.isfinite(phase_number):
                malformed_fields.append("phases.number")
            plan_count = phase_mapping.get("plan_count")
            summary_count = phase_mapping.get("summary_count")
            if plan_count is None:
                missing_fields.append("phases.plan_count")
            if summary_count is None:
                missing_fields.append("phases.summary_count")
            plan_count_int = _coerce_nonnegative_int(plan_count)
            summary_count_int = _coerce_nonnegative_int(summary_count)
            if plan_count is not None and plan_count_int is None:
                malformed_fields.append("phases.plan_count")
            if summary_count is not None and summary_count_int is None:
                malformed_fields.append("phases.summary_count")
            phase_records.append((phase_mapping, plan_count_int, summary_count_int))
        if phase_records:
            if all(plan_count is not None for _, plan_count, _ in phase_records):
                phase_plan_count_sum = sum(
                    int(plan_count) for _, plan_count, _ in phase_records
                )
            if all(summary_count is not None for _, _, summary_count in phase_records):
                phase_summary_count_sum = sum(
                    int(summary_count) for _, _, summary_count in phase_records
                )
            completed_flags = tuple(
                phase.get("disk_status") == "complete"
                and phase.get("roadmap_complete") is True
                and plan_count is not None
                and summary_count is not None
                and plan_count > 0
                and summary_count == plan_count
                for phase, plan_count, summary_count in phase_records
            )
            completed_from_phases = int(sum(completed_flags))
            incomplete_from_phases = int(len(completed_flags) - completed_from_phases)
    phase_count = roadmap_analysis.get("phase_count")
    completed = roadmap_analysis.get("completed_phases")
    total_plans = roadmap_analysis.get("total_plans")
    total_summaries = roadmap_analysis.get("total_summaries")
    if phase_count is None and phases is not None:
        phase_count = len(phases)
    if completed is None and completed_from_phases is not None:
        completed = completed_from_phases
    if phase_count is None or completed is None:
        missing_fields.extend(
            field
            for field, value in (
                ("phase_count", phase_count),
                ("completed_phases", completed),
            )
            if value is None
        )
    if total_plans is None:
        missing_fields.append("total_plans")
    if total_summaries is None:
        missing_fields.append("total_summaries")
    progress_percent = roadmap_analysis.get("progress_percent")
    if progress_percent is None:
        missing_fields.append("progress_percent")
    if "next_phase" not in roadmap_analysis:
        missing_fields.append("next_phase")
    phase_count_int = _coerce_nonnegative_int(phase_count)
    completed_int = _coerce_nonnegative_int(completed)
    total_plans_int = _coerce_nonnegative_int(total_plans)
    total_summaries_int = _coerce_nonnegative_int(total_summaries)
    progress_percent_int = _coerce_nonnegative_int(progress_percent)
    malformed_fields.extend(
        field
        for field, raw_value, coerced_value in (
            ("phase_count", phase_count, phase_count_int),
            ("completed_phases", completed, completed_int),
            ("total_plans", total_plans, total_plans_int),
            ("total_summaries", total_summaries, total_summaries_int),
            ("progress_percent", progress_percent, progress_percent_int),
        )
        if raw_value is not None and coerced_value is None
    )
    if missing_fields or malformed_fields:
        problem_fields = tuple(dict.fromkeys((*missing_fields, *malformed_fields)))
        return {
            "roadmap_analysis_available": True,
            "roadmap_disk_complete": False,
            "condition": "roadmap_analysis_insufficient",
            "condition_value": len(problem_fields),
            "missing_fields": tuple(dict.fromkeys(missing_fields)),
            "malformed_fields": tuple(dict.fromkeys(malformed_fields)),
            "mismatch_fields": (),
            "total_plans": total_plans_int,
            "total_summaries": total_summaries_int,
            "phase_plan_count_sum": phase_plan_count_sum,
            "phase_summary_count_sum": phase_summary_count_sum,
        }

    if isinstance(phases, list):
        if phase_count_int != len(phases):
            mismatch_fields.append("phase_count_vs_phases")
        if (
            completed_from_phases is not None
            and completed_int != completed_from_phases
        ):
            mismatch_fields.append("completed_phases_vs_phase_rows")
        if (
            phase_plan_count_sum is not None
            and total_plans_int != phase_plan_count_sum
        ):
            mismatch_fields.append("total_plans_vs_phase_rows")
        if (
            phase_summary_count_sum is not None
            and total_summaries_int != phase_summary_count_sum
        ):
            mismatch_fields.append("total_summaries_vs_phase_rows")
    if total_plans_int != total_summaries_int:
        mismatch_fields.append("total_plans_vs_total_summaries")
    if mismatch_fields:
        return {
            "roadmap_analysis_available": True,
            "roadmap_disk_complete": False,
            "condition": "roadmap_analysis_inconsistent",
            "condition_value": len(tuple(dict.fromkeys(mismatch_fields))),
            "missing_fields": (),
            "malformed_fields": (),
            "mismatch_fields": tuple(dict.fromkeys(mismatch_fields)),
            "total_plans": total_plans_int,
            "total_summaries": total_summaries_int,
            "phase_plan_count_sum": phase_plan_count_sum,
            "phase_summary_count_sum": phase_summary_count_sum,
        }

    incomplete_count = (
        incomplete_from_phases
        if incomplete_from_phases is not None
        else max(phase_count_int - completed_int, 0)
    )
    incomplete_plan_count = max(total_plans_int - total_summaries_int, 0)
    progress_complete = progress_percent_int == 100
    next_phase_clear = roadmap_analysis.get("next_phase") is None
    disk_complete = (
        phase_count_int > 0
        and completed_int == phase_count_int
        and incomplete_count == 0
        and total_plans_int > 0
        and total_summaries_int == total_plans_int
        and progress_complete
        and next_phase_clear
    )
    return {
        "roadmap_analysis_available": True,
        "roadmap_disk_complete": disk_complete,
        "condition": "roadmap_disk_complete"
        if disk_complete
        else "roadmap_disk_incomplete",
        "condition_value": 0
        if disk_complete
        else max(incomplete_count, incomplete_plan_count, 1),
        "missing_fields": (),
        "malformed_fields": (),
        "mismatch_fields": (),
        "total_plans": total_plans_int,
        "total_summaries": total_summaries_int,
        "phase_plan_count_sum": phase_plan_count_sum,
        "phase_summary_count_sum": phase_summary_count_sum,
    }


def _resolve_public_e2e_requirements_analysis(
    *,
    requirements_analysis: dict[str, Any] | None,
    requirements_path: str | Path | None,
    milestone_version: str,
    planning_dir: str | Path,
) -> dict[str, Any] | None:
    if requirements_analysis is not None:
        return dict(requirements_analysis)
    if requirements_path is not None:
        path = Path(requirements_path)
        if not path.is_file():
            return None
        return _requirements_analysis_from_file(path)
    planning_path = Path(planning_dir)
    path = planning_path / "REQUIREMENTS.md"
    if not path.is_file():
        version = str(milestone_version).strip()
        if version and not version.startswith("v"):
            version = f"v{version}"
        archived_path = planning_path / "milestones" / f"{version}-REQUIREMENTS.md"
        if not archived_path.is_file():
            return None
        path = archived_path
    return _requirements_analysis_from_file(path)


def _requirements_analysis_from_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    checked_ids = re.findall(r"^\s*-\s*\[x\]\s+\*\*([^*]+)\*\*", text, re.MULTILINE)
    unchecked_ids = re.findall(
        r"^\s*-\s*\[\s\]\s+\*\*([^*]+)\*\*", text, re.MULTILINE
    )
    trace_rows: list[tuple[str, str]] = []
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) >= 3 and re.match(r"^[A-Z][A-Z0-9-]+$", cells[0]):
            trace_rows.append((cells[0], cells[2]))
    non_complete_trace_rows = tuple(
        req_id
        for req_id, status in trace_rows
        if status.strip().lower() not in {"complete", "completed", "pass", "passed"}
    )
    checkbox_ids = tuple(checked_ids) + tuple(unchecked_ids)
    trace_ids = {req_id for req_id, _status in trace_rows}
    if checkbox_ids and not trace_rows:
        missing_traceability_rows = ("requirements_traceability_missing",)
    else:
        missing_traceability_rows = tuple(
            f"requirements_traceability_missing:{req_id}"
            for req_id in checkbox_ids
            if req_id not in trace_ids
        )
    non_complete_rows = tuple(
        dict.fromkeys((*non_complete_trace_rows, *missing_traceability_rows))
    )
    total_requirements = len(checked_ids) + len(unchecked_ids)
    completed_requirements = len(checked_ids)
    if total_requirements == 0 and trace_rows:
        total_requirements = len(trace_rows)
        completed_requirements = len(trace_rows) - len(non_complete_trace_rows)
    pending_requirements = max(
        len(unchecked_ids),
        total_requirements - completed_requirements,
        len(non_complete_rows),
    )
    return {
        "requirements_total": total_requirements,
        "requirements_completed": completed_requirements,
        "requirements_pending": pending_requirements,
        "requirements_unchecked_ids": tuple(unchecked_ids),
        "requirements_expected_ids": checkbox_ids
        if checkbox_ids
        else tuple(req_id for req_id, _ in trace_rows),
        "requirements_traceability_rows": tuple(req_id for req_id, _ in trace_rows),
        "requirements_non_complete_rows": non_complete_rows,
        "requirements_complete": (
            total_requirements > 0
            and completed_requirements == total_requirements
            and pending_requirements == 0
            and len(non_complete_rows) == 0
        ),
    }


def _requirements_analysis_archive_state(
    requirements_analysis: dict[str, Any] | None,
) -> dict[str, Any]:
    if requirements_analysis is None:
        return {
            "requirements_analysis_available": False,
            "requirements_complete": None,
            "condition": "requirements_not_checked",
            "condition_value": 1,
            "missing_fields": (),
            "malformed_fields": (),
            "total_requirements": None,
            "completed_requirements": None,
            "pending_requirements": None,
            "expected_ids": (),
            "expected_id_count": 0,
            "expected_unique_id_count": 0,
            "duplicate_expected_ids": (),
            "traceability_rows": (),
            "traceability_unique_row_count": 0,
            "duplicate_traceability_rows": (),
            "missing_traceability_ids": (),
            "unexpected_traceability_ids": (),
            "non_complete_rows": (),
        }

    total = requirements_analysis.get("requirements_total")
    completed = requirements_analysis.get("requirements_completed")
    pending = requirements_analysis.get("requirements_pending")
    if total is None:
        total = requirements_analysis.get("total_requirements")
    if completed is None:
        completed = requirements_analysis.get("completed_requirements")
    if pending is None:
        pending = requirements_analysis.get("pending_requirements")
    has_non_complete_rows_field = "requirements_non_complete_rows" in requirements_analysis
    has_legacy_non_complete_rows_field = "non_complete_rows" in requirements_analysis
    raw_non_complete_rows = requirements_analysis.get("requirements_non_complete_rows")
    if raw_non_complete_rows is None and has_legacy_non_complete_rows_field:
        raw_non_complete_rows = requirements_analysis.get("non_complete_rows")
    has_any_non_complete_rows_field = (
        has_non_complete_rows_field or has_legacy_non_complete_rows_field
    )
    has_traceability_rows_field = "requirements_traceability_rows" in requirements_analysis
    has_legacy_traceability_rows_field = "traceability_rows" in requirements_analysis
    raw_traceability_rows = requirements_analysis.get("requirements_traceability_rows")
    if raw_traceability_rows is None and has_legacy_traceability_rows_field:
        raw_traceability_rows = requirements_analysis.get("traceability_rows")
    has_any_traceability_rows_field = (
        has_traceability_rows_field or has_legacy_traceability_rows_field
    )
    has_expected_ids_field = "requirements_expected_ids" in requirements_analysis
    has_legacy_expected_ids_field = "expected_requirement_ids" in requirements_analysis
    raw_expected_ids = requirements_analysis.get("requirements_expected_ids")
    if raw_expected_ids is None and has_legacy_expected_ids_field:
        raw_expected_ids = requirements_analysis.get("expected_requirement_ids")
    has_any_expected_ids_field = (
        has_expected_ids_field or has_legacy_expected_ids_field
    )
    non_complete_rows, malformed_row_fields = (
        _coerce_requirements_non_complete_rows(raw_non_complete_rows)
    )
    traceability_rows, malformed_traceability_fields = (
        _coerce_requirements_traceability_rows(raw_traceability_rows)
    )
    expected_ids, malformed_expected_fields = _coerce_requirements_expected_ids(
        raw_expected_ids
    )
    expected_unique_id_count = len(tuple(dict.fromkeys(expected_ids)))
    duplicate_expected_ids = _duplicate_requirement_ids(expected_ids)
    traceability_requirement_ids = _requirements_traceability_requirement_ids(
        traceability_rows
    )
    traceability_unique_row_count = len(
        tuple(dict.fromkeys(traceability_requirement_ids))
    )
    duplicate_traceability_rows = _duplicate_requirement_ids(
        traceability_requirement_ids
    )
    missing_fields = tuple(
        dict.fromkeys(
            (
                *(
                    field
                    for field, value in (
                        ("requirements_total", total),
                        ("requirements_completed", completed),
                        ("requirements_pending", pending),
                    )
                    if value is None
                ),
                *(
                    ("requirements_non_complete_rows",)
                    if not has_any_non_complete_rows_field
                    or raw_non_complete_rows is None
                    else ()
                ),
                *(
                    ("requirements_traceability_rows",)
                    if not has_any_traceability_rows_field
                    or raw_traceability_rows is None
                    else ()
                ),
                *(
                    ("requirements_expected_ids",)
                    if not has_any_expected_ids_field or raw_expected_ids is None
                    else ()
                ),
            )
        )
    )
    total_int = _coerce_nonnegative_int(total)
    completed_int = _coerce_nonnegative_int(completed)
    pending_int = _coerce_nonnegative_int(pending)
    malformed_fields = tuple(
        dict.fromkeys(
            (
                *(
                    field
                    for field, raw_value, coerced_value in (
                        ("requirements_total", total, total_int),
                        ("requirements_completed", completed, completed_int),
                        ("requirements_pending", pending, pending_int),
                    )
                    if raw_value is not None and coerced_value is None
                ),
                *malformed_row_fields,
                *malformed_traceability_fields,
                *malformed_expected_fields,
            )
        )
    )
    if missing_fields or malformed_fields:
        return {
            "requirements_analysis_available": True,
            "requirements_complete": False,
            "condition": "requirements_analysis_insufficient",
            "condition_value": len(
                tuple(dict.fromkeys((*missing_fields, *malformed_fields)))
            ),
            "missing_fields": missing_fields,
            "malformed_fields": malformed_fields,
            "total_requirements": total,
            "completed_requirements": completed,
            "pending_requirements": pending,
            "expected_ids": expected_ids,
            "expected_id_count": len(expected_ids),
            "expected_unique_id_count": expected_unique_id_count,
            "duplicate_expected_ids": duplicate_expected_ids,
            "traceability_rows": traceability_rows,
            "traceability_unique_row_count": traceability_unique_row_count,
            "duplicate_traceability_rows": duplicate_traceability_rows,
            "missing_traceability_ids": (),
            "unexpected_traceability_ids": (),
            "non_complete_rows": non_complete_rows,
        }

    explicit_complete = requirements_analysis.get("requirements_complete")
    counts_complete = (
        total_int > 0
        and completed_int == total_int
        and pending_int == 0
        and len(non_complete_rows) == 0
    )
    requirements_complete = counts_complete and explicit_complete is not False
    unique_expected_ids = tuple(dict.fromkeys(expected_ids))
    unique_traceability_ids = tuple(dict.fromkeys(traceability_requirement_ids))
    expected_id_set = set(unique_expected_ids)
    traceability_id_set = set(unique_traceability_ids)
    missing_traceability_ids = tuple(
        requirement_id
        for requirement_id in unique_expected_ids
        if requirement_id not in traceability_id_set
    )
    unexpected_traceability_ids = tuple(
        requirement_id
        for requirement_id in unique_traceability_ids
        if requirement_id not in expected_id_set
    )
    if (
        total_int is not None
        and total_int > 0
        and (
            len(expected_ids) != total_int
            or expected_unique_id_count != total_int
            or duplicate_expected_ids
        )
        and len(non_complete_rows) == 0
    ):
        non_complete_rows = tuple(
            dict.fromkeys(
                (
                    *non_complete_rows,
                    *(
                        ("requirements_expected_id_duplicate_rows",)
                        if duplicate_expected_ids
                        else ()
                    ),
                    "requirements_expected_id_count_mismatch",
                )
            )
        )
        requirements_complete = False
    if (
        total_int is not None
        and total_int > 0
        and (
            len(traceability_rows) != total_int
            or traceability_unique_row_count != total_int
            or missing_traceability_ids
            or unexpected_traceability_ids
        )
        and len(non_complete_rows) == 0
    ):
        non_complete_rows = tuple(
            dict.fromkeys(
                (
                    *non_complete_rows,
                    *(
                        ("requirements_traceability_duplicate_rows",)
                        if duplicate_traceability_rows
                        else ()
                    ),
                    *(
                        f"requirements_traceability_missing:{requirement_id}"
                        for requirement_id in missing_traceability_ids
                    ),
                    *(
                        ("requirements_traceability_unexpected_ids",)
                        if unexpected_traceability_ids
                        else ()
                    ),
                    "requirements_traceability_row_count_mismatch",
                )
            )
        )
        requirements_complete = False
    if explicit_complete is True and not counts_complete:
        non_complete_rows = tuple(
            dict.fromkeys((*non_complete_rows, "requirements_count_mismatch"))
        )
    if requirements_complete:
        return {
            "requirements_analysis_available": True,
            "requirements_complete": True,
            "condition": "requirements_complete",
            "condition_value": 0,
            "missing_fields": (),
            "malformed_fields": (),
            "total_requirements": total_int,
            "completed_requirements": completed_int,
            "pending_requirements": pending_int,
            "expected_ids": expected_ids,
            "expected_id_count": len(expected_ids),
            "expected_unique_id_count": expected_unique_id_count,
            "duplicate_expected_ids": duplicate_expected_ids,
            "traceability_rows": traceability_rows,
            "traceability_unique_row_count": traceability_unique_row_count,
            "duplicate_traceability_rows": duplicate_traceability_rows,
            "missing_traceability_ids": missing_traceability_ids,
            "unexpected_traceability_ids": unexpected_traceability_ids,
            "non_complete_rows": non_complete_rows,
        }

    blocker_count = max(
        pending_int,
        total_int - completed_int,
        len(non_complete_rows),
        1,
    )
    return {
        "requirements_analysis_available": True,
        "requirements_complete": False,
        "condition": "requirements_incomplete",
        "condition_value": blocker_count,
        "missing_fields": (),
        "malformed_fields": (),
        "total_requirements": total_int,
        "completed_requirements": completed_int,
        "pending_requirements": pending_int,
        "expected_ids": expected_ids,
        "expected_id_count": len(expected_ids),
        "expected_unique_id_count": expected_unique_id_count,
        "duplicate_expected_ids": duplicate_expected_ids,
        "traceability_rows": traceability_rows,
        "traceability_unique_row_count": traceability_unique_row_count,
        "duplicate_traceability_rows": duplicate_traceability_rows,
        "missing_traceability_ids": missing_traceability_ids,
        "unexpected_traceability_ids": unexpected_traceability_ids,
        "non_complete_rows": non_complete_rows,
    }


def _comparison_summary(
    comparison_rows: tuple[dict[str, Any], ...],
    *,
    provenance_rows: tuple[dict[str, Any], ...],
    tolerance_rows: tuple[dict[str, Any], ...],
    empirical_summary: dict[str, Any],
    monte_carlo_summary: dict[str, Any],
    runtime_seconds: float,
) -> dict[str, Any]:
    category_counts = _count_by(comparison_rows, "comparison_category")
    source_counts = _count_by(comparison_rows, "source_report")
    target_rows = [
        row for row in comparison_rows if row.get("paper_target_claim") is True
    ]
    accepted_target_rows = [
        row for row in target_rows if row["accepted_for_paper_reproduction"] is True
    ]
    empirical_target_rows = [
        row
        for row in target_rows
        if row.get("source_report") == "paper_empirical_reproduction_report"
    ]
    accepted_empirical_target_rows = [
        row
        for row in empirical_target_rows
        if row["accepted_for_paper_reproduction"] is True
    ]
    blocked_categories = {
        "missing_data_or_source_limitation",
    }
    blocked_rows = [
        row for row in comparison_rows if row["comparison_category"] in blocked_categories
    ]
    comparison_row_state = _comparison_row_state(comparison_rows)
    provenance_row_state = _provenance_row_state(provenance_rows)
    tolerance_contract_state = _tolerance_contract_name_state(tolerance_rows)
    next_action = (
        "inspect_missing_evidence_or_source_bindings"
        if blocked_rows
        else "advance_to_phase18_public_reproduction_e2e_and_milestone_audit"
    )
    return {
        "row_count": len(comparison_rows),
        "comparison_row_count": len(comparison_rows),
        "provenance_row_count": len(provenance_rows),
        "tolerance_row_count": len(tolerance_rows),
        "comparison_category_counts": category_counts,
        "comparison_status_counts": category_counts,
        "source_report_counts": source_counts,
        "paper_target_rows": len(target_rows),
        "accepted_paper_target_rows": len(accepted_target_rows),
        "paper_reproduction_target_rows": len(target_rows),
        "paper_reproduction_accepted_target_rows": len(accepted_target_rows),
        "paper_reproduction_empirical_target_rows": len(empirical_target_rows),
        "paper_reproduction_accepted_empirical_target_rows": len(
            accepted_empirical_target_rows
        ),
        "blocked_comparison_rows": len(blocked_rows),
        "exact_reproduction_rows": category_counts.get("exact_reproduction", 0),
        "sampling_error_reproduction_rows": category_counts.get(
            "sampling_error_reproduction", 0
        ),
        "missing_data_or_source_limitation_rows": category_counts.get(
            "missing_data_or_source_limitation", 0
        ),
        "missing_data_or_source_rows": category_counts.get(
            "missing_data_or_source_limitation", 0
        ),
        "documented_exception_rows": category_counts.get(
            "documented_exception", 0
        ),
        "low_budget_blocked_rows": int(
            (monte_carlo_summary.get("low_budget_probe_blocked") is True)
            and category_counts.get("missing_data_or_source_limitation", 0) > 0
        ),
        "provenance_ready_rows": int(sum(row.get("ready") is True for row in provenance_rows)),
        "incomplete_comparison_row_ids": comparison_row_state[
            "incomplete_row_ids"
        ],
        "incomplete_comparison_fields": comparison_row_state["incomplete_fields"],
        "invalid_comparison_row_ids": comparison_row_state["invalid_row_ids"],
        "invalid_comparison_fields": comparison_row_state["invalid_fields"],
        "duplicate_comparison_row_ids": comparison_row_state["duplicate_row_ids"],
        "incomplete_provenance_row_ids": provenance_row_state[
            "incomplete_row_ids"
        ],
        "incomplete_provenance_fields": provenance_row_state["incomplete_fields"],
        "invalid_provenance_row_ids": provenance_row_state["invalid_row_ids"],
        "invalid_provenance_fields": provenance_row_state["invalid_fields"],
        "duplicate_provenance_row_ids": provenance_row_state["duplicate_row_ids"],
        "provenance_types": _count_by(provenance_rows, "provenance_type"),
        "provenance_category_counts": _count_by(
            provenance_rows,
            "provenance_category",
        ),
        "tolerance_names": tolerance_contract_state["actual_names"],
        "tolerance_contract_names": tolerance_contract_state["actual_names"],
        "required_tolerance_contract_names": tolerance_contract_state[
            "required_names"
        ],
        "actual_tolerance_contract_names": tolerance_contract_state["actual_names"],
        "missing_tolerance_contract_names": tolerance_contract_state[
            "missing_names"
        ],
        "duplicate_tolerance_contract_names": tolerance_contract_state[
            "duplicate_names"
        ],
        "incomplete_tolerance_contract_names": tolerance_contract_state[
            "incomplete_names"
        ],
        "invalid_tolerance_contract_names": tolerance_contract_state[
            "invalid_names"
        ],
        "incomplete_tolerance_contract_fields": tolerance_contract_state[
            "incomplete_fields"
        ],
        "invalid_tolerance_contract_fields": tolerance_contract_state[
            "invalid_fields"
        ],
        "max_empirical_absolute_difference": empirical_summary.get(
            "max_absolute_difference"
        ),
        "empirical_runtime_seconds": empirical_summary.get("runtime_seconds"),
        "max_monte_carlo_rejection_rate_absolute_error": monte_carlo_summary.get(
            "max_rejection_rate_absolute_error"
        ),
        "max_monte_carlo_standard_error": monte_carlo_summary.get(
            "max_target_mc_standard_error"
        ),
        "monte_carlo_runtime_seconds": monte_carlo_summary.get("runtime_seconds"),
        "monte_carlo_paper_result_rows": monte_carlo_summary.get("paper_result_rows"),
        "monte_carlo_active_blocking_conditions": dict(
            monte_carlo_summary.get("active_blocking_conditions") or {}
        ),
        "monte_carlo_stale_evidence_file_count": monte_carlo_summary.get(
            "stale_evidence_file_count"
        ),
        "monte_carlo_stale_evidence_files": monte_carlo_summary.get(
            "stale_evidence_files"
        ),
        "monte_carlo_stale_evidence_error": monte_carlo_summary.get(
            "stale_evidence_error"
        ),
        "monte_carlo_executed_result_rows": monte_carlo_summary.get(
            "executed_result_rows"
        ),
        "monte_carlo_scheduled_draws": monte_carlo_summary.get("scheduled_draws"),
        "monte_carlo_executed_draws": monte_carlo_summary.get("executed_draws"),
        "paper_reproduction_gate_passes": (
            len(blocked_rows) == 0
            and len(target_rows) > 0
            and len(target_rows) == len(accepted_target_rows)
            and len(empirical_target_rows) > 0
            and len(empirical_target_rows) == len(accepted_empirical_target_rows)
        ),
        "phase17_contract_ready": bool(
            comparison_rows
            and provenance_rows
            and tolerance_rows
            and len(comparison_rows) >= _REQUIRED_PAPER_REPRODUCTION_COMPARISON_ROWS
            and len(provenance_rows) >= _REQUIRED_PAPER_REPRODUCTION_PROVENANCE_ROWS
            and not comparison_row_state["incomplete_row_ids"]
            and not comparison_row_state["invalid_row_ids"]
            and not provenance_row_state["incomplete_row_ids"]
            and not provenance_row_state["invalid_row_ids"]
            and len(tolerance_rows) >= 3
            and not tolerance_contract_state["missing_names"]
            and not tolerance_contract_state["duplicate_names"]
            and not tolerance_contract_state["incomplete_names"]
            and not tolerance_contract_state["invalid_names"]
        ),
        "next_action": next_action,
        "runtime_seconds": runtime_seconds,
        "paper_anchors": (
            "manuscript/sources/arxiv-2404.11739v3/draft.tex:570-608",
            "manuscript/sources/arxiv-2404.11739v3/draft.tex:391-456",
            "manuscript/sources/arxiv-2404.11739v3/draft.tex:1094-1156",
            "manuscript/sources/arxiv-2404.11739v3/draft.tex:1162-1175",
        ),
        "reference_anchors": (
            "packages/r/TestMechs/R/lb_frac_affected.R:318-347",
            "packages/r/TestMechs/R/partial_density_plot.R:24-155",
            "packages/r/TestMechs/R/simulate_data_binaryM.R:1-35",
            "packages/r/TestMechs/R/test_sharp_null.R:90-315",
            "packages/r/TestMechs/R/nonparametric_bootstrap.R:16-71",
        ),
    }


def _comparison_row_state(
    comparison_rows: tuple[dict[str, Any], ...],
) -> dict[str, tuple[str, ...]]:
    incomplete_row_ids: list[str] = []
    incomplete_fields: list[str] = []
    invalid_row_ids: list[str] = []
    invalid_fields: list[str] = []
    duplicate_row_ids: list[str] = []
    seen_row_ids: set[str] = set()

    for index, row in enumerate(comparison_rows):
        row_id = _comparison_row_id(row, index)
        row_incomplete = False
        row_invalid = False

        if not isinstance(row, Mapping):
            invalid_row_ids.append(row_id)
            invalid_fields.append(f"{row_id}.row")
            continue

        if row_id in seen_row_ids:
            duplicate_row_ids.append(row_id)
            invalid_fields.append(f"{row_id}.duplicate_comparison_row_id")
            row_invalid = True
        else:
            seen_row_ids.add(row_id)

        for field in _REQUIRED_COMPARISON_TEXT_FIELDS:
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                incomplete_fields.append(f"{row_id}.{field}")
                row_incomplete = True

        accepted = row.get("accepted_for_paper_reproduction")
        if not isinstance(accepted, bool):
            invalid_fields.append(f"{row_id}.accepted_for_paper_reproduction")
            row_invalid = True

        target_claim = row.get("paper_target_claim")
        target_available = row.get("paper_target_available")
        if target_claim is not None and not isinstance(target_claim, bool):
            invalid_fields.append(f"{row_id}.paper_target_claim")
            row_invalid = True
        if target_available is not None and not isinstance(target_available, bool):
            invalid_fields.append(f"{row_id}.paper_target_available")
            row_invalid = True
        if target_claim is True and target_available is False:
            invalid_fields.append(f"{row_id}.paper_target_available")
            row_invalid = True
        is_target_row = (
            target_claim is True
            or target_available is True
            or row.get("case_id") == "monte_carlo_paper_acceptance_evidence_state"
        )
        if is_target_row:
            if target_claim is not True:
                invalid_fields.append(f"{row_id}.paper_target_claim")
                row_invalid = True
            if (
                row.get("source_report") == "paper_empirical_reproduction_report"
                and target_available is not True
            ):
                invalid_fields.append(f"{row_id}.paper_target_available")
                row_invalid = True
            for field in _REQUIRED_COMPARISON_TARGET_VALUE_FIELDS:
                value = row.get(field)
                if _is_empty_comparison_value(value):
                    incomplete_fields.append(f"{row_id}.{field}")
                    row_incomplete = True
            for field in _REQUIRED_COMPARISON_TARGET_NUMERIC_FIELDS:
                if field not in row or row.get(field) is None:
                    incomplete_fields.append(f"{row_id}.{field}")
                    row_incomplete = True
                    continue
                if _coerce_nonnegative_float(row.get(field)) is None:
                    invalid_fields.append(f"{row_id}.{field}")
                    row_invalid = True
            if accepted is True and row.get("comparison_category") not in (
                _REQUIRED_COMPARISON_ACCEPTED_CATEGORIES
            ):
                invalid_fields.append(f"{row_id}.comparison_category")
                row_invalid = True
            absolute_difference = _coerce_nonnegative_float(
                row.get("absolute_difference")
            )
            tolerance_value = _coerce_nonnegative_float(row.get("tolerance_value"))
            if (
                accepted is True
                and absolute_difference is not None
                and tolerance_value is not None
                and absolute_difference
                > tolerance_value + _COMPARISON_ACCEPTANCE_TOLERANCE_EPSILON
            ):
                invalid_fields.append(f"{row_id}.absolute_difference")
                row_invalid = True
            if row.get("source_report") == "paper_empirical_reproduction_report":
                if _coerce_nonnegative_int(row.get("sample_size")) is None:
                    invalid_fields.append(f"{row_id}.sample_size")
                    row_invalid = True

        if row_incomplete:
            incomplete_row_ids.append(row_id)
        if row_invalid:
            invalid_row_ids.append(row_id)

    return {
        "incomplete_row_ids": tuple(dict.fromkeys(incomplete_row_ids)),
        "incomplete_fields": tuple(dict.fromkeys(incomplete_fields)),
        "invalid_row_ids": tuple(dict.fromkeys(invalid_row_ids)),
        "invalid_fields": tuple(dict.fromkeys(invalid_fields)),
        "duplicate_row_ids": tuple(dict.fromkeys(duplicate_row_ids)),
    }


def _comparison_row_id(row: Any, index: int) -> str:
    if isinstance(row, Mapping):
        for field in ("case_id", "source_case_id"):
            value = row.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return f"comparison_row_{index}"


def _is_empty_comparison_value(value: Any) -> bool:
    if _is_missing(value):
        return True
    if isinstance(value, bool):
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, Mapping):
        return len(value) == 0
    if isinstance(value, (list, tuple, set, frozenset)):
        return len(value) == 0
    return False


def _provenance_row_state(
    provenance_rows: tuple[dict[str, Any], ...],
) -> dict[str, tuple[str, ...]]:
    incomplete_row_ids: list[str] = []
    incomplete_fields: list[str] = []
    invalid_row_ids: list[str] = []
    invalid_fields: list[str] = []
    duplicate_row_ids: list[str] = []
    seen_row_ids: set[str] = set()

    for index, row in enumerate(provenance_rows):
        row_id = _provenance_row_id(row, index)
        row_incomplete = False
        row_invalid = False

        if not isinstance(row, Mapping):
            invalid_row_ids.append(row_id)
            invalid_fields.append(f"{row_id}.row")
            continue

        if row_id in seen_row_ids:
            duplicate_row_ids.append(row_id)
            invalid_fields.append(f"{row_id}.duplicate_provenance_row_id")
            row_invalid = True
        else:
            seen_row_ids.add(row_id)

        for field in _REQUIRED_PROVENANCE_TEXT_FIELDS:
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                incomplete_fields.append(f"{row_id}.{field}")
                row_incomplete = True

        for field in _REQUIRED_PROVENANCE_NUMERIC_FIELDS:
            if field not in row or row.get(field) is None:
                incomplete_fields.append(f"{row_id}.{field}")
                row_incomplete = True
                continue
            value = _coerce_nonnegative_int(row.get(field))
            if value is None or value <= 0:
                invalid_fields.append(f"{row_id}.{field}")
                row_invalid = True

        for field in _REQUIRED_PROVENANCE_SUPPORT_FIELDS:
            value = row.get(field)
            if _is_empty_provenance_support(value):
                incomplete_fields.append(f"{row_id}.{field}")
                row_incomplete = True

        if row.get("ready") is not True:
            invalid_fields.append(f"{row_id}.ready")
            row_invalid = True

        if row_incomplete:
            incomplete_row_ids.append(row_id)
        if row_invalid:
            invalid_row_ids.append(row_id)

    return {
        "incomplete_row_ids": tuple(dict.fromkeys(incomplete_row_ids)),
        "incomplete_fields": tuple(dict.fromkeys(incomplete_fields)),
        "invalid_row_ids": tuple(dict.fromkeys(invalid_row_ids)),
        "invalid_fields": tuple(dict.fromkeys(invalid_fields)),
        "duplicate_row_ids": tuple(dict.fromkeys(duplicate_row_ids)),
    }


def _provenance_row_id(row: Any, index: int) -> str:
    if isinstance(row, Mapping):
        for field in ("provenance_key", "case_id", "data_source_key"):
            value = row.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return f"provenance_row_{index}"


def _is_empty_provenance_support(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, Mapping):
        return len(value) == 0
    if isinstance(value, (list, tuple, set, frozenset)):
        return len(value) == 0
    return False


def _tolerance_contract_name_state(
    tolerance_rows: tuple[dict[str, Any], ...],
) -> dict[str, tuple[str, ...]]:
    actual_names_list: list[str] = []
    rows_by_name: dict[str, list[Mapping[str, Any]]] = {}
    for row in tolerance_rows:
        if not isinstance(row, Mapping):
            continue
        raw_name = row.get("tolerance_name")
        if raw_name is None or isinstance(raw_name, bool):
            continue
        name = str(raw_name).strip()
        if name:
            actual_names_list.append(name)
            rows_by_name.setdefault(name, []).append(row)
    actual_names = tuple(actual_names_list)
    actual_name_set = set(actual_names)
    duplicate_names = tuple(
        name
        for name in _REQUIRED_PAPER_REPRODUCTION_TOLERANCE_CONTRACTS
        if len(rows_by_name.get(name, ())) > 1
    )
    missing_names = tuple(
        name
        for name in _REQUIRED_PAPER_REPRODUCTION_TOLERANCE_CONTRACTS
        if name not in actual_name_set
    )
    incomplete_fields: list[str] = []
    invalid_fields: list[str] = []
    incomplete_names: list[str] = []
    invalid_names: list[str] = []
    for name in _REQUIRED_PAPER_REPRODUCTION_TOLERANCE_CONTRACTS:
        rows_for_name = rows_by_name.get(name)
        if not rows_for_name:
            continue
        if len(rows_for_name) > 1:
            invalid_fields.append(f"{name}.duplicate_tolerance_contract_name")
        row = rows_for_name[0]
        row_incomplete = False
        row_invalid = False
        for field in _REQUIRED_TOLERANCE_CONTRACT_TEXT_FIELDS[name]:
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                incomplete_fields.append(f"{name}.{field}")
                row_incomplete = True
        numeric_values: dict[str, float] = {}
        for field in _REQUIRED_TOLERANCE_CONTRACT_NUMERIC_FIELDS[name]:
            if field not in row or row.get(field) is None:
                incomplete_fields.append(f"{name}.{field}")
                row_incomplete = True
                continue
            value = _coerce_nonnegative_float(row.get(field))
            if value is None:
                invalid_fields.append(f"{name}.{field}")
                row_invalid = True
                continue
            numeric_values[field] = value
        for field, expected in _REQUIRED_TOLERANCE_CONTRACT_NUMERIC_VALUES[
            name
        ].items():
            value = numeric_values.get(field)
            if value is None:
                continue
            if not math.isclose(
                value,
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                invalid_fields.append(f"{name}.{field}")
                row_invalid = True
        if row_incomplete:
            incomplete_names.append(name)
        if row_invalid:
            invalid_names.append(name)
    return {
        "required_names": _REQUIRED_PAPER_REPRODUCTION_TOLERANCE_CONTRACTS,
        "actual_names": actual_names,
        "missing_names": missing_names,
        "duplicate_names": duplicate_names,
        "incomplete_names": tuple(dict.fromkeys(incomplete_names)),
        "invalid_names": tuple(dict.fromkeys(invalid_names)),
        "incomplete_fields": tuple(dict.fromkeys(incomplete_fields)),
        "invalid_fields": tuple(dict.fromkeys(invalid_fields)),
    }


def _count_by(rows: tuple[dict[str, Any], ...], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _first_present_numeric(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value is not None and not _is_missing(value):
            return float(value)
    return None


def _ordered_unique(values: Any) -> tuple[Any, ...]:
    unique: list[Any] = []
    seen: set[str] = set()
    for value in list(values):
        normalized = _normalize_json_value(value)
        key = repr(normalized)
        if key not in seen:
            seen.add(key)
            unique.append(normalized)
    return tuple(unique)


def _json_safe_export_payload(payload: Any) -> Any:
    return _normalize_json_value(payload)


def _portable_report_roots(
    *,
    evidence_dir: Path,
    fixtures_dir: Path,
    statistics_dir: str | Path | None,
    tables_dir: str | Path | None,
) -> tuple[tuple[str, str], ...]:
    roots = [
        evidence_dir,
        fixtures_dir,
        *(path for path in (statistics_dir, tables_dir) if path is not None),
        Path.cwd(),
    ]
    replacements: dict[str, str] = {}
    for root in roots:
        resolved = Path(root).resolve()
        source = str(resolved)
        replacement = _portable_path_replacement(resolved)
        replacements[source] = replacement
        if source.startswith("/private/var/"):
            replacements[source.removeprefix("/private")] = replacement
        elif source.startswith("/var/"):
            replacements[f"/private{source}"] = replacement
    return tuple(
        sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True)
    )


def _portable_path_replacement(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


def _portable_report_value(value: Any, *, roots: tuple[tuple[str, str], ...]) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _portable_report_value(item, roots=roots)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_portable_report_value(item, roots=roots) for item in value]
    if isinstance(value, Path):
        return _portable_report_string(str(value), roots=roots)
    if isinstance(value, str):
        return _portable_report_string(value, roots=roots)
    return value


def _portable_report_string(value: str, *, roots: tuple[tuple[str, str], ...]) -> str:
    portable = value
    for source, replacement in roots:
        portable = portable.replace(f"{source}/", f"{replacement}/")
        portable = portable.replace(f"'{source}'", f"'{replacement}'")
        portable = portable.replace(f'"{source}"', f'"{replacement}"')
        if portable == source:
            portable = replacement
    return portable


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, pd.Interval):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize_json_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        try:
            listed = value.tolist()
        except (TypeError, ValueError):
            pass
        else:
            if listed is not value:
                return _normalize_json_value(listed)
    if _is_missing(value):
        return None
    if hasattr(value, "item"):
        try:
            return _normalize_json_value(value.item())
        except ValueError:
            pass
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, float):
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
        return "positive_infinity" if numeric > 0 else "negative_infinity"
    if isinstance(value, int):
        return int(value)
    return value


def _require_json_object(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a JSON object.")
    return dict(value)


def _require_json_object_sequence(
    value: Any,
    field_name: str,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a JSON array of objects.")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"{field_name}[{index}] must be a JSON object.")
        rows.append(dict(item))
    return tuple(rows)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(result, bool):
        return result
    return False


def _default_fixture_inputs_dir() -> Path:
    checkout_dir = Path(__file__).resolve().parents[3] / "tests/python" / "fixtures" / "inputs"
    if checkout_dir.exists():
        return checkout_dir
    return Path(str(files("testmechs.resources.fixtures")))
