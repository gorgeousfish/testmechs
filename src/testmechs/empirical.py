"""Empirical reproduction report for the JSS manuscript.

This module computes and compares Python lower-bound and partial-density
results against paper-published statistics from the Testing Mechanisms
manuscript (arXiv:2404.11739v3).  The report verifies that the Python
``testmechs`` package reproduces the paper's empirical claims within
documented rounding tolerances.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
import json
from pathlib import Path
import math
import re
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .bounds import lb_frac_affected
from .partial_density import partial_density_data
from .preprocess import remove_missing_from_df
from .results import _is_scalar_missing, _json_safe_payload


@dataclass(frozen=True)
class PaperEmpiricalReproductionReport:
    """Empirical reproduction report comparing Python results to paper targets.

    Attributes
    ----------
    rows : tuple[dict[str, Any], ...]
        Per-case reproduction rows with computed values and comparison status.
    summary : dict[str, Any]
        Aggregated summary including pass/fail counts and runtime.
    """

    rows: tuple[dict[str, Any], ...]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report to a JSON-safe dictionary."""
        return _json_safe_payload(
            {
                "summary": dict(self.summary),
                "rows": [dict(row) for row in self.rows],
            }
        )

    def to_frame(self) -> pd.DataFrame:
        """Return the report rows as a DataFrame with object dtype."""
        return pd.DataFrame(
            _json_safe_payload([dict(row) for row in self.rows]),
            dtype=object,
        )

    def to_display_frame(self) -> pd.DataFrame:
        """Return a compact human-readable empirical reproduction table."""

        return _empirical_reproduction_display_frame(self.rows)


def paper_empirical_reproduction_report(
    *,
    fixtures_dir: str | Path | None = None,
    statistics_dir: str | Path | None = None,
) -> PaperEmpiricalReproductionReport:
    """Build the empirical reproduction report for the paper.

    Executes all configured empirical cases (lower-bound and partial-density)
    against test fixtures and compares Python values with published paper
    statistics.

    Parameters
    ----------
    fixtures_dir : str, Path, or None
        Directory containing CSV test fixtures.  Defaults to the repository
        checkout path ``tests/python/fixtures/inputs/``.
    statistics_dir : str, Path, or None
        Directory containing LaTeX statistic files from the paper source.
        Defaults to ``manuscript/sources/arxiv-2404.11739v3/Statistics/``.

    Returns
    -------
    PaperEmpiricalReproductionReport
        Report containing per-case rows and an aggregated summary.
    """

    start = perf_counter()
    fixtures = _default_fixture_inputs_dir() if fixtures_dir is None else Path(fixtures_dir)
    statistics = _default_statistics_dir() if statistics_dir is None else Path(statistics_dir)
    rows = tuple(
        _execute_case(case, fixtures_dir=fixtures, statistics_dir=statistics)
        for case in _case_definitions()
    )
    summary = _summarize_rows(rows)
    summary["runtime_seconds"] = float(perf_counter() - start)
    return PaperEmpiricalReproductionReport(rows=rows, summary=summary)


def paper_empirical_reproduction_report_frame(
    *,
    fixtures_dir: str | Path | None = None,
    statistics_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Return the empirical reproduction report as a row-level DataFrame.

    Parameters
    ----------
    fixtures_dir : str, Path, or None
        Directory containing CSV test fixtures.
    statistics_dir : str, Path, or None
        Directory containing LaTeX statistic files.

    Returns
    -------
    pd.DataFrame
        One row per empirical case with all comparison fields.
    """

    return paper_empirical_reproduction_report(
        fixtures_dir=fixtures_dir,
        statistics_dir=statistics_dir,
    ).to_frame()


def paper_empirical_reproduction_display_frame(
    *,
    fixtures_dir: str | Path | None = None,
    statistics_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Return a compact human-readable empirical reproduction report table.

    Parameters
    ----------
    fixtures_dir : str, Path, or None
        Directory containing CSV test fixtures.
    statistics_dir : str, Path, or None
        Directory containing LaTeX statistic files.

    Returns
    -------
    pd.DataFrame
        Display-formatted DataFrame with status, application, metric, and
        comparison columns.
    """

    return paper_empirical_reproduction_report(
        fixtures_dir=fixtures_dir,
        statistics_dir=statistics_dir,
    ).to_display_frame()


_EMPIRICAL_REPRODUCTION_DISPLAY_COLUMNS = [
    "status",
    "application",
    "metric",
    "python_value",
    "paper_target",
    "abs_diff",
    "tolerance",
    "sample",
    "source",
    "case_id",
]


def _empirical_reproduction_display_frame(
    rows: tuple[dict[str, Any], ...],
) -> pd.DataFrame:
    """Build display-frame from raw reproduction rows."""
    display_rows = [_empirical_reproduction_display_row(row) for row in rows]
    return pd.DataFrame(
        display_rows,
        columns=_EMPIRICAL_REPRODUCTION_DISPLAY_COLUMNS,
        dtype=object,
    )


def _empirical_reproduction_display_row(row: dict[str, Any]) -> dict[str, Any]:
    """Format a single reproduction row for human-readable display."""
    missing_fields = [
        field
        for field in (
            "exception_status",
            "application",
            "metric",
            "python_value",
            "target_value",
            "absolute_difference",
            "tolerance",
            "n_obs_used",
            "case_id",
        )
        if field not in row
    ]
    if missing_fields:
        raise ValueError(
            "Empirical reproduction display rows are missing required fields: "
            + ", ".join(missing_fields)
        )
    return {
        "status": _empirical_reproduction_status_label(row["exception_status"]),
        "application": _format_empirical_display_label(row["application"], field="application"),
        "metric": _format_empirical_metric_label(row["metric"]),
        "python_value": _format_report_number(row["python_value"], digits=6),
        "paper_target": _format_report_number(row["target_value"], digits=6),
        "abs_diff": _format_report_number(row["absolute_difference"], digits=6),
        "tolerance": _format_report_number(row["tolerance"], digits=6),
        "sample": _format_empirical_sample(row),
        "source": _format_empirical_source(row),
        "case_id": _format_empirical_display_label(row["case_id"], field="case_id"),
    }


def _empirical_reproduction_status_label(exception_status: Any) -> str:
    """Map exception_status to a short human-readable display label."""
    status = _format_report_text(exception_status, field="exception_status")
    if isinstance(exception_status, (bool, np.bool_)):
        raise ValueError(
            f"Empirical reproduction display field exception_status is boolean: {exception_status!r}"
        )
    if status == "paper_target_within_tolerance":
        return "PASS"
    if status == "reference_only_no_paper_target":
        return "REFERENCE"
    if status == "paper_target_read_error":
        return "TARGET ERROR"
    if status == "paper_target_outside_tolerance":
        return "FAIL"
    if status == "python_exception":
        return "SOURCE ERROR"
    return status.upper().replace("_", " ")


def _format_report_number(value: Any, *, digits: int) -> str:
    """Format a numeric value for report display with fixed decimal digits."""
    if value is None:
        return ""
    if _is_missing_display_value(value):
        return "NA"
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    if isinstance(value, dict):
        return _compact_empirical_json_display(
            json.dumps(
                _json_safe_display_payload(dict(value)),
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    if isinstance(value, (list, tuple, np.ndarray)):
        return _compact_empirical_json_display(
            json.dumps(
                _json_safe_display_payload(value),
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    numeric = float(value)
    if not math.isfinite(numeric):
        if math.isnan(numeric):
            return "NA"
        return "+Inf" if numeric > 0 else "-Inf"
    return f"{numeric:.{digits}f}"


_EMPIRICAL_JSON_DISPLAY_MAX_CHARS = 160
_EMPIRICAL_JSON_DISPLAY_TAIL_CHARS = 60


def _compact_empirical_json_display(text: str) -> str:
    """Truncate long JSON text with head...tail ellipsis for display."""
    if len(text) <= _EMPIRICAL_JSON_DISPLAY_MAX_CHARS:
        return text
    head_chars = (
        _EMPIRICAL_JSON_DISPLAY_MAX_CHARS
        - _EMPIRICAL_JSON_DISPLAY_TAIL_CHARS
        - 3
    )
    return f"{text[:head_chars]}...{text[-_EMPIRICAL_JSON_DISPLAY_TAIL_CHARS:]}"


def _format_report_text(value: Any, *, field: str) -> str:
    """Stringify a report text field, raising on missing or blank values."""
    if _is_missing_display_value(value):
        raise ValueError(f"Empirical reproduction display field {field} is missing")
    text = str(value).strip()
    if not text:
        raise ValueError(f"Empirical reproduction display field {field} is blank")
    return text


def _json_safe_display_payload(value: Any) -> Any:
    """Recursively convert a value to a JSON-safe display representation."""
    if isinstance(value, dict):
        return {str(key): _json_safe_display_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_display_payload(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe_display_payload(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe_display_payload(value.item())
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isfinite(numeric):
            return value
        if math.isnan(numeric):
            return None
        return "positive_infinity" if numeric > 0 else "negative_infinity"
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _format_empirical_sample(row: dict[str, Any]) -> str:
    """Format sample-size info string (n, T/C counts, clusters)."""
    sample = f"n={_format_report_count(row['n_obs_used'])}"
    if row.get("n_treated") is not None and row.get("n_control") is not None:
        sample += (
            f" (T={_format_report_count(row['n_treated'])}, "
            f"C={_format_report_count(row['n_control'])})"
        )
    if row.get("cluster_count") is not None:
        sample += f", clusters={_format_report_count(row['cluster_count'])}"
    return sample


def _format_report_count(value: Any) -> str:
    """Format an integer count value for display, rejecting non-integers."""
    if value is None:
        return "NA"
    if _is_missing_display_value(value):
        return "NA"
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"Empirical reproduction display count is boolean: {value!r}")
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric):
            if math.isnan(numeric):
                return "NA"
            return "+Inf" if numeric > 0 else "-Inf"
        if numeric < 0:
            raise ValueError(f"Empirical reproduction display count is negative: {value!r}")
        if not numeric.is_integer():
            raise ValueError(f"Empirical reproduction display count is not an integer: {value!r}")
        return str(int(numeric))
    text = str(value).strip()
    numeric = float(text)
    if not math.isfinite(numeric):
        if math.isnan(numeric):
            return "NA"
        return "+Inf" if numeric > 0 else "-Inf"
    if numeric < 0:
        raise ValueError(f"Empirical reproduction display count is negative: {value!r}")
    if not numeric.is_integer():
        raise ValueError(f"Empirical reproduction display count is not an integer: {value!r}")
    return str(int(numeric))


def _format_empirical_source(row: dict[str, Any]) -> str:
    """Extract and format the data source label from a reproduction row."""
    fixture_name = row.get("fixture_name")
    if not _is_missing_display_value(fixture_name):
        return _format_empirical_source_label(fixture_name)
    data_source = row.get("data_source")
    if _is_missing_display_value(data_source):
        return ""
    return _format_empirical_source_label(Path(str(data_source)).name)


_EMPIRICAL_TEXT_MAX_DISPLAY_CHARS = 96
_EMPIRICAL_TEXT_TAIL_CHARS = 36


def _format_empirical_display_label(value: Any, *, field: str) -> str:
    """Format a display label with truncation for long strings."""
    text = _format_report_text(value, field=field)
    if len(text) <= _EMPIRICAL_TEXT_MAX_DISPLAY_CHARS:
        return text
    head_chars = _EMPIRICAL_TEXT_MAX_DISPLAY_CHARS - _EMPIRICAL_TEXT_TAIL_CHARS - 3
    return f"{text[:head_chars]}...{text[-_EMPIRICAL_TEXT_TAIL_CHARS:]}"


_EMPIRICAL_METRIC_DISPLAY_LABELS = {
    "lower_bound_fraction_never_takers_affected": "NT lower bound",
    "pooled_lower_bound_fraction_always_takers_affected": "Pooled AT lower bound",
    "pooled_vector_lower_bound_fraction_always_takers_affected": "Pooled vector AT lower bound",
    "partial_density_discrete_row_count": "Partial-density row count",
}


def _format_empirical_metric_label(value: Any) -> str:
    """Map metric identifiers to short display labels."""
    text = _format_report_text(value, field="metric")
    label = _EMPIRICAL_METRIC_DISPLAY_LABELS.get(text, text)
    return _format_empirical_display_label(label, field="metric")


def _format_empirical_source_label(value: Any) -> str:
    """Format a source label with truncation."""
    return _format_empirical_display_label(value, field="source")


def _is_missing_display_value(value: Any) -> bool:
    """Check whether a value should be treated as missing for display."""
    return _is_scalar_missing(value)


def _execute_case(
    case: dict[str, Any],
    *,
    fixtures_dir: Path,
    statistics_dir: Path,
) -> dict[str, Any]:
    """Execute a single empirical case and return a comparison row dict."""
    start = perf_counter()
    row = _base_row(case, fixtures_dir=fixtures_dir, statistics_dir=statistics_dir)
    target_error: str | None = None
    try:
        target_value = _target_value(case, statistics_dir=statistics_dir)
    except Exception as exc:
        target_value = None
        target_error = f"{type(exc).__name__}: {exc}"
    try:
        if case["runner"] == "lb_frac_affected":
            result, analysis_frame, raw_rows = _run_lower_bound_case(
                case, fixtures_dir=fixtures_dir
            )
            python_value = float(result.lower_bound)
            diagnostics = result.diagnostics
            row.update(
                _sample_fields(
                    case,
                    analysis_frame,
                    raw_rows=raw_rows,
                    diagnostics=diagnostics,
                )
            )
            row.update(
                {
                    "python_value": python_value,
                    "applied_num_y_bins": diagnostics.get("applied_num_y_bins"),
                    "size_risk": diagnostics.get("size_risk"),
                    "minimum_compatible_defiers_share": diagnostics.get(
                        "minimum_compatible_defiers_share"
                    ),
                    "actual_max_defiers_share": diagnostics.get(
                        "actual_max_defiers_share"
                    ),
                    "defier_cap_actual_cap_source": _defier_cap_contract_field(
                        diagnostics, "actual_cap_source"
                    ),
                    "defier_cap_epsilon_relaxation": _defier_cap_contract_field(
                        diagnostics, "epsilon_relaxation"
                    ),
                    "defier_cap_reference_boundary": _defier_cap_contract_field(
                        diagnostics, "reference_boundary"
                    ),
                    **_lower_bound_diagnostic_fields(diagnostics),
                }
            )
        elif case["runner"] == "partial_density_data":
            result, analysis_frame, raw_rows = _run_partial_density_case(
                case,
                fixtures_dir=fixtures_dir,
            )
            row.update(
                _sample_fields(
                    case,
                    analysis_frame,
                    raw_rows=raw_rows,
                    diagnostics=result.diagnostics,
                )
            )
            row.update(
                {
                    "python_value": float(len(result.partial_mass_records)),
                    "applied_num_y_bins": result.diagnostics.get("applied_num_y_bins"),
                    "size_risk": None,
                    "minimum_compatible_defiers_share": None,
                    "actual_max_defiers_share": None,
                    "defier_cap_actual_cap_source": None,
                    "defier_cap_epsilon_relaxation": None,
                    "defier_cap_reference_boundary": None,
                    **_empty_lower_bound_diagnostic_fields(),
                }
            )
        else:
            raise ValueError(f"Unknown empirical report runner: {case['runner']!r}")
    except Exception as exc:
        exception_message = f"{type(exc).__name__}: {exc}"
        if target_error is not None:
            exception_message = (
                f"{exception_message}; paper target read failed: {target_error}"
            )
        row.update(
            {
                "python_value": None,
                "target_value": target_value,
                "paper_target": target_value,
                "absolute_difference": None,
                "abs_diff": None,
                "within_tolerance": False,
                "runtime_seconds": float(perf_counter() - start),
                "exception_status": "python_exception",
                "exception_message": exception_message,
            }
        )
        return row

    row["target_value"] = target_value
    row["paper_target"] = target_value
    if target_error is not None:
        row.update(
            {
                "absolute_difference": None,
                "abs_diff": None,
                "within_tolerance": False,
                "exception_status": "paper_target_read_error",
                "exception_message": target_error,
            }
        )
    elif target_value is None:
        row.update(
            {
                "absolute_difference": None,
                "abs_diff": None,
                "within_tolerance": None,
                "exception_status": "reference_only_no_paper_target",
                "exception_message": case["no_target_reason"],
            }
        )
    else:
        absolute_difference = abs(float(row["python_value"]) - target_value)
        within_tolerance = absolute_difference <= float(case["tolerance"])
        row.update(
            {
                "absolute_difference": absolute_difference,
                "abs_diff": absolute_difference,
                "within_tolerance": bool(within_tolerance),
                "exception_status": "paper_target_within_tolerance"
                if within_tolerance
                else "paper_target_outside_tolerance",
                "exception_message": None
                if within_tolerance
                else "Python value is outside the rounded paper target tolerance.",
            }
        )

    row["runtime_seconds"] = float(perf_counter() - start)
    return row


_GENERAL_LFP_EMPIRICAL_FIELDS = {
    "type_count": "general_lfp_type_count",
    "slack_count": "general_lfp_slack_count",
    "solution_basis": "general_lfp_solution_basis",
    "denominator_minimum": "general_lfp_denominator_minimum",
    "objective_numerator": "general_lfp_objective_numerator",
    "objective_denominator": "general_lfp_objective_denominator",
    "primal_eq_max_abs_residual": "general_lfp_primal_eq_max_abs_residual",
    "primal_ub_max_violation": "general_lfp_primal_ub_max_violation",
    "marginal_fit_max_abs_difference": "general_lfp_marginal_fit_max_abs_difference",
    "slack_constraint_max_violation": "general_lfp_slack_constraint_max_violation",
    "defier_cap_max_violation": "general_lfp_defier_cap_max_violation",
    "paper_inequality_max_violation": "general_lfp_paper_inequality_max_violation",
    "objective_paper_inequality_max_violation": (
        "general_lfp_objective_paper_inequality_max_violation"
    ),
}


def _lower_bound_diagnostic_fields(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Extract general-LFP and restriction diagnostic fields."""
    general_lfp = diagnostics.get("general_lfp")
    fields = _empty_lower_bound_diagnostic_fields()
    fields.update(
        {
            "active_restriction": diagnostics.get("active_restriction"),
            "theta_kk_min": diagnostics.get("theta_kk_min"),
            "theta_kk_min_by_group": diagnostics.get("theta_kk_min_by_group"),
            "no_bite_flag": _no_bite_field(diagnostics, "flag"),
            "no_bite_reason": _no_bite_field(diagnostics, "reason"),
        }
    )
    if isinstance(general_lfp, dict):
        for source_field, output_field in _GENERAL_LFP_EMPIRICAL_FIELDS.items():
            fields[output_field] = general_lfp.get(source_field)
    return fields


def _empty_lower_bound_diagnostic_fields() -> dict[str, Any]:
    """Return a dict of lower-bound diagnostic fields all set to None."""
    return {
        "active_restriction": None,
        "theta_kk_min": None,
        "theta_kk_min_by_group": None,
        "no_bite_flag": None,
        "no_bite_reason": None,
        **{
            output_field: None
            for output_field in _GENERAL_LFP_EMPIRICAL_FIELDS.values()
        },
    }


def _no_bite_field(diagnostics: dict[str, Any], field: str) -> Any:
    """Extract a nested no-bite diagnostic field safely."""
    no_bite = diagnostics.get("no_bite")
    if not isinstance(no_bite, dict):
        return None
    return no_bite.get(field)


def _defier_cap_contract_field(
    diagnostics: dict[str, Any],
    field: str,
) -> Any:
    """Extract a nested defier-cap contract diagnostic field safely."""
    contract = diagnostics.get("defier_cap_contract")
    if not isinstance(contract, dict):
        return None
    return contract.get(field)


def _run_lower_bound_case(
    case: dict[str, Any],
    *,
    fixtures_dir: Path,
) -> tuple[Any, pd.DataFrame, int]:
    """Execute a lower-bound case and return result, frame, and raw row count."""
    analysis_frame = _analysis_frame(case, fixtures_dir=fixtures_dir)
    raw_rows = _raw_row_count(case, fixtures_dir=fixtures_dir)
    mediators = tuple(case["mediators"])
    result = lb_frac_affected(
        df=analysis_frame,
        d=case["d"],
        m=mediators[0] if len(mediators) == 1 else list(mediators),
        y=case["y"],
        at_group=case["at_group"],
        num_y_bins=case["num_y_bins"],
        max_defiers_share=case["max_defiers_share"],
        allow_min_defiers=case["allow_min_defiers"],
    )
    return result, analysis_frame, raw_rows


def _run_partial_density_case(
    case: dict[str, Any],
    *,
    fixtures_dir: Path,
) -> tuple[Any, pd.DataFrame, int]:
    """Execute a partial-density case and return result, frame, and raw row count."""
    analysis_frame = _analysis_frame(case, fixtures_dir=fixtures_dir)
    raw_rows = _raw_row_count(case, fixtures_dir=fixtures_dir)
    result = partial_density_data(
        df=analysis_frame,
        d=case["d"],
        m=case["mediators"][0],
        y=case["y"],
        num_y_bins=case["num_y_bins"],
        plot_nts=case["plot_nts"],
    )
    return result, analysis_frame, raw_rows


def _analysis_frame(case: dict[str, Any], *, fixtures_dir: Path) -> pd.DataFrame:
    """Load and filter a fixture CSV to the required complete-case analysis frame."""
    path = fixtures_dir / case["fixture_name"]
    df = pd.read_csv(path)
    columns = [
        case["d"],
        *case["mediators"],
        case["y"],
        *case["analysis_frame_columns"],
    ]
    return df.dropna(subset=columns).copy()


def _raw_row_count(case: dict[str, Any], *, fixtures_dir: Path) -> int:
    """Return the total row count of the fixture before any filtering."""
    path = fixtures_dir / case["fixture_name"]
    return int(pd.read_csv(path, usecols=[case["d"]]).shape[0])


def _sample_fields(
    case: dict[str, Any],
    analysis_frame: pd.DataFrame,
    *,
    raw_rows: int,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Compute sample description fields for a reproduction row."""
    mediators = tuple(case["mediators"])
    data = remove_missing_from_df(
        df=analysis_frame,
        d=case["d"],
        m=mediators[0] if len(mediators) == 1 else list(mediators),
        y=case["y"],
    )
    treatment = data[case["d"]]
    cluster = case["cluster"]
    return {
        "n_obs_input": raw_rows,
        "analysis_frame_rows": int(len(analysis_frame)),
        "n_obs_used": int(diagnostics.get("n_obs_used", len(data))),
        "sample_size": int(diagnostics.get("n_obs_used", len(data))),
        "n_treated": int((treatment == 1).sum()),
        "n_control": int((treatment == 0).sum()),
        "cluster_count": None
        if cluster is None
        else int(data[cluster].nunique(dropna=True)),
        "mediator_level_count": int(
            data.loc[:, list(mediators)].drop_duplicates().shape[0]
        ),
    }


def _base_row(
    case: dict[str, Any],
    *,
    fixtures_dir: Path,
    statistics_dir: Path,
) -> dict[str, Any]:
    """Construct the base reproduction row dict before execution."""
    fixture_path = fixtures_dir / case["fixture_name"]
    return {
        "case_id": case["case_id"],
        "application": case["application"],
        "metric": case["metric"],
        "runner": case["runner"],
        "data_source": str(fixture_path.resolve()),
        "fixture_name": case["fixture_name"],
        "analysis_frame_columns": list(case["analysis_frame_columns"]),
        "paper_anchor": case["paper_anchor"],
        "reference_anchor": case["reference_anchor"],
        "paper_target_available": case["target_stat_file"] is not None,
        "paper_target_file": None
        if case["target_stat_file"] is None
        else str((statistics_dir / case["target_stat_file"]).resolve()),
        "target_scale": case["target_scale"],
        "target_rounding_contract": case["target_rounding_contract"],
        "target_value": None,
        "paper_target": None,
        "python_value": None,
        "absolute_difference": None,
        "abs_diff": None,
        "tolerance": case["tolerance"],
        "within_tolerance": None,
        "d": case["d"],
        "mediators": list(case["mediators"]),
        "y": case["y"],
        "cluster": case["cluster"],
        "num_y_bins": case["num_y_bins"],
        "at_group": case["at_group"],
        "allow_min_defiers": case["allow_min_defiers"],
        "max_defiers_share": case["max_defiers_share"],
        "truth_hierarchy": "paper_statistics_then_python_first_principles_with_r_as_reference",
        "negative_reference_evidence": _negative_reference_evidence(case),
    }


def _negative_reference_evidence(case: dict[str, Any]) -> str:
    """Return the negative-evidence annotation for a given case."""
    if case["case_id"] == "kerwin_partial_density_fixture_contract":
        return (
            "Documented reference-implementation anomalies are not target behavior; "
            "for partial_density_plot(plot_nts=True), R's default legend order is "
            "negative evidence because the plotted partial11 mass corresponds to "
            "original D=0,M=0 after the internal orientation flip."
        )
    return "Documented reference-implementation anomalies are not target behavior."


def _target_value(case: dict[str, Any], *, statistics_dir: Path) -> float | None:
    """Read and rescale the paper target value for a case (None if no target)."""
    stat_file = case["target_stat_file"]
    if stat_file is None:
        return None
    value = _read_latex_numeric(statistics_dir / stat_file)
    if case["target_scale"] == "percent":
        return value / 100.0
    return value


def _read_latex_numeric(path: Path) -> float:
    """Parse a single numeric value from a LaTeX statistics file."""
    text = path.read_text(encoding="utf-8")
    numeric_token = re.compile(
        r"(?<![A-Za-z0-9_.+-])"
        r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?"
        r"(?![A-Za-z0-9_.+-])"
    )
    matches = numeric_token.findall(text)
    if not matches:
        raise ValueError(f"No numeric value found in paper statistic file: {path}")
    if len(matches) > 1:
        raise ValueError(
            f"Expected exactly one numeric value in paper statistic file: {path}"
        )
    return float(matches[0])


def _summarize_rows(rows: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    """Aggregate pass/fail/exception counts across all reproduction rows."""
    target_rows = [row for row in rows if row["paper_target_available"]]
    passed_target_rows = [
        row for row in target_rows if row["within_tolerance"] is True
    ]
    python_exception_rows = [
        row for row in rows if row["exception_status"] == "python_exception"
    ]
    paper_target_read_error_rows = [
        row for row in rows if row["exception_status"] == "paper_target_read_error"
    ]
    paper_target_outside_tolerance_rows = [
        row
        for row in rows
        if row["exception_status"] == "paper_target_outside_tolerance"
    ]
    reference_only_rows = [
        row for row in rows if row["exception_status"] == "reference_only_no_paper_target"
    ]
    numeric_diffs = [
        float(row["absolute_difference"])
        for row in target_rows
        if row["absolute_difference"] is not None
    ]
    return {
        "row_count": len(rows),
        "target_row_count": len(target_rows),
        "passed_target_rows": len(passed_target_rows),
        "failed_target_rows": len(target_rows) - len(passed_target_rows),
        "python_exception_rows": len(python_exception_rows),
        "paper_target_read_error_rows": len(paper_target_read_error_rows),
        "paper_target_outside_tolerance_rows": len(paper_target_outside_tolerance_rows),
        "reference_only_rows": len(reference_only_rows),
        "exception_status_counts": _count_by(rows, "exception_status"),
        "python_exception_case_ids": _case_ids(python_exception_rows),
        "paper_target_read_error_case_ids": _case_ids(paper_target_read_error_rows),
        "paper_target_outside_tolerance_case_ids": _case_ids(
            paper_target_outside_tolerance_rows
        ),
        "reference_only_case_ids": _case_ids(reference_only_rows),
        "applications": sorted({str(row["application"]) for row in rows}),
        "max_absolute_difference": max(numeric_diffs) if numeric_diffs else None,
        "target_tolerance": 0.005,
        "next_action": _empirical_report_next_action(
            target_rows=target_rows,
            passed_target_rows=passed_target_rows,
            python_exception_rows=python_exception_rows,
        ),
        "paper_anchors": sorted({str(row["paper_anchor"]) for row in rows}),
        "reference_anchors": sorted({str(row["reference_anchor"]) for row in rows}),
        "negative_reference_evidence": (
            "Documented reference-implementation anomalies remain negative evidence: "
            "literal binary coding, no-bite NaN, silent defier-cap mutation, "
            "vector bounds_ade_ats entry crash, and plot_nts legend orientation "
            "drift are not Python targets."
        ),
    }


def _empirical_report_next_action(
    *,
    target_rows: list[dict[str, Any]],
    passed_target_rows: list[dict[str, Any]],
    python_exception_rows: list[dict[str, Any]],
) -> str:
    """Determine the recommended next action based on report results."""
    if python_exception_rows:
        return "inspect_empirical_report_source_failures"
    if len(passed_target_rows) == len(target_rows):
        return "advance_to_phase16_monte_carlo_reproduction_evidence_runner"
    return "inspect_empirical_report_target_failures"


def _count_by(rows: tuple[dict[str, Any], ...], key: str) -> dict[str, int]:
    """Count rows grouped by a specified key field."""
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        counts[value] = counts.get(value, 0) + 1
    return counts


def _case_ids(rows: list[dict[str, Any]]) -> list[str]:
    """Extract case_id strings from a list of rows."""
    return [str(row["case_id"]) for row in rows]


def _case_definitions() -> tuple[dict[str, Any], ...]:
    """Return the complete tuple of empirical case definitions for the paper."""
    return (
        _case(
            case_id="bursztyn_job_application_nt_lb",
            application="Bursztyn et al. (2020)",
            metric="lower_bound_fraction_never_takers_affected",
            fixture_name="burstzyn_data.csv",
            d="condition2",
            mediators=("signed_up_number",),
            y="applied_out_fl",
            num_y_bins=2,
            at_group=0,
            target_stat_file="burstzyn-percent-nts-affected-restricted-sample.tex",
            analysis_frame_columns=("index",),
            paper_anchor="manuscript/sources/arxiv-2404.11739v3/draft.tex:574-580",
            reference_anchor=(
                "packages/r/TestMechs/README.Rmd:184-203; "
                "tests/python/parity/r_readme_empirical_cases.R:323-354"
            ),
            target_rounding_contract=(
                "paper reports a whole-percent lower-bound statistic; compare as "
                "proportion with +/-0.005 rounding tolerance"
            ),
        ),
        _case(
            case_id="baranov_grandmother_nt_lb",
            application="Baranov et al. (2020)",
            metric="lower_bound_fraction_never_takers_affected",
            fixture_name="baranov_mother_data.csv",
            d="treat",
            mediators=("grandmother",),
            y="motherfinancial",
            cluster="uc",
            num_y_bins=5,
            at_group=0,
            target_stat_file="baranov-grandmother-percent-nts-affected.tex",
            paper_anchor="manuscript/sources/arxiv-2404.11739v3/draft.tex:591-593",
            reference_anchor=(
                "packages/r/TestMechs/README.Rmd:184-203; "
                "tests/python/parity/r_readme_empirical_cases.R:154-185"
            ),
            target_rounding_contract=(
                "paper reports a whole-percent lower-bound statistic; compare as "
                "proportion with +/-0.005 rounding tolerance"
            ),
        ),
        _case(
            case_id="baranov_relationship_pooled_lb_min_defiers",
            application="Baranov et al. (2020)",
            metric="pooled_lower_bound_fraction_always_takers_affected",
            fixture_name="baranov_mother_data.csv",
            d="treat",
            mediators=("relationship_husb",),
            y="motherfinancial",
            cluster="uc",
            num_y_bins=5,
            at_group=None,
            allow_min_defiers=True,
            target_stat_file="baranov-relationship-at-bounds-all-percent.tex",
            paper_anchor="manuscript/sources/arxiv-2404.11739v3/draft.tex:595",
            reference_anchor=(
                "packages/r/TestMechs/README.Rmd:249-280; "
                "tests/python/parity/r_readme_empirical_cases.R:288-321"
            ),
            target_rounding_contract=(
                "paper reports a whole-percent lower-bound statistic; compare as "
                "proportion with +/-0.005 rounding tolerance"
            ),
        ),
        _case(
            case_id="baranov_combined_pooled_lb_min_defiers",
            application="Baranov et al. (2020)",
            metric="pooled_vector_lower_bound_fraction_always_takers_affected",
            fixture_name="baranov_mother_data.csv",
            d="treat",
            mediators=("relationship_husb", "grandmother"),
            y="motherfinancial",
            cluster="uc",
            num_y_bins=5,
            at_group=None,
            allow_min_defiers=True,
            target_stat_file="baranov-both-tv-all-percent.tex",
            paper_anchor="manuscript/sources/arxiv-2404.11739v3/draft.tex:606",
            reference_anchor=(
                "packages/r/TestMechs/README.Rmd:284-317; "
                "packages/r/TestMechs/R/lb_frac_affected.R:318-347"
            ),
            target_rounding_contract=(
                "paper reports a whole-percent lower-bound statistic; compare as "
                "proportion with +/-0.005 rounding tolerance"
            ),
        ),
        _case(
            case_id="kerwin_partial_density_fixture_contract",
            application="Kerwin fixture",
            metric="partial_density_discrete_row_count",
            fixture_name="kerwin_data.csv",
            d="treated",
            mediators=("primarily_leblango",),
            y="EL_EGRA_PCA_Index",
            num_y_bins=5,
            runner="partial_density_data",
            target_stat_file=None,
            paper_anchor="No paper empirical target; fixture-level R package example only",
            reference_anchor=(
                "packages/r/TestMechs/R/partial_density_plot.R:24-155; "
                "tests/python/parity/r_readme_empirical_cases.R:449-485"
            ),
            target_rounding_contract="not_applicable",
            no_target_reason=(
                "Kerwin is a bundled R-package fixture without a paper statistic target "
                "in manuscript/sources/arxiv-2404.11739v3."
            ),
            tolerance=None,
            target_scale="not_applicable",
        ),
    )


def _case(
    *,
    case_id: str,
    application: str,
    metric: str,
    fixture_name: str,
    d: str,
    mediators: tuple[str, ...],
    y: str,
    paper_anchor: str,
    reference_anchor: str,
    target_stat_file: str | None,
    num_y_bins: int | None,
    runner: str = "lb_frac_affected",
    cluster: str | None = None,
    at_group: Any = None,
    allow_min_defiers: bool = False,
    max_defiers_share: float = 0.0,
    analysis_frame_columns: tuple[str, ...] = (),
    target_rounding_contract: str,
    no_target_reason: str | None = None,
    tolerance: float | None = 0.005,
    target_scale: str = "percent",
    plot_nts: bool = False,
) -> dict[str, Any]:
    """Construct a single empirical case definition dict."""
    return {
        "case_id": case_id,
        "application": application,
        "metric": metric,
        "fixture_name": fixture_name,
        "d": d,
        "mediators": mediators,
        "y": y,
        "runner": runner,
        "cluster": cluster,
        "num_y_bins": num_y_bins,
        "at_group": at_group,
        "allow_min_defiers": allow_min_defiers,
        "max_defiers_share": max_defiers_share,
        "analysis_frame_columns": analysis_frame_columns,
        "target_stat_file": target_stat_file,
        "target_scale": target_scale,
        "target_rounding_contract": target_rounding_contract,
        "paper_anchor": paper_anchor,
        "reference_anchor": reference_anchor,
        "no_target_reason": no_target_reason,
        "tolerance": tolerance,
        "plot_nts": plot_nts,
    }


def _default_fixture_inputs_dir() -> Path:
    """Resolve the default test fixture inputs directory."""
    checkout_dir = Path(__file__).resolve().parents[3] / "tests/python" / "fixtures" / "inputs"
    if checkout_dir.exists():
        return checkout_dir
    return Path(str(files("testmechs.resources.fixtures")))


def _default_statistics_dir() -> Path:
    """Resolve the default LaTeX statistics directory from the paper source."""
    checkout_dir = Path(__file__).resolve().parents[3] / "manuscript/sources/arxiv-2404.11739v3" / "Statistics"
    if checkout_dir.exists():
        return checkout_dir
    return Path(str(files("testmechs.resources.statistics")))
