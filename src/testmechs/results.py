"""Result dataclass objects for Testing Mechanisms computations.

This module defines the immutable result containers returned by the public
estimation and inference APIs in ``testmechs``.  Each result class is a
frozen dataclass that carries:

- Point estimates, test statistics, or interval endpoints.
- Diagnostics dictionaries preserving solver/internal audit metadata.
- Serialization helpers (``to_dict``, ``to_frame``) for JSON export and
  tabular display.
- Rich ``__str__`` and ``_repr_html_`` renderers for notebooks and CLI.

The result layer is intentionally decoupled from estimation logic so that
consumers can inspect, compare, and export result payloads without importing
any estimator machinery.

Result Classes
--------------
LowerBoundResult
    Always-taker affected-fraction lower-bound point estimate.
ADEBoundsResult
    Lee-style ADE trimming bounds (lower and upper).
SharpNullResult
    Sharp-null hypothesis test decision with test statistic details.
TVConfidenceIntervalResult
    Grid-inversion confidence interval for total-variation targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
import html
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_DIAGNOSTIC_KEY_PREVIEW_LIMIT = 5

_SEPARATOR = "-" * 39


def _vertical_card(title: str, rows: list[tuple[str, str]], footer: str = "") -> str:
    """Format a vertical key-value card for terminal display."""
    lines = [_SEPARATOR, f" {title}", _SEPARATOR]
    if rows:
        max_key = max(len(k) for k, _ in rows)
        for key, value in rows:
            lines.append(f" {key + ':':<{max_key + 1}}  {value}")
    lines.append(_SEPARATOR)
    if footer:
        lines.append(f" {footer}")
    return "\n".join(lines)


@dataclass(frozen=True)
class LowerBoundResult:
    """Point-estimate result for an always-taker affected-fraction lower bound.

    Encapsulates the scalar lower-bound value together with the estimand
    label, target always-taker group, active restriction description, and
    full diagnostics dictionary produced by the underlying optimizer.

    Attributes
    ----------
    lower_bound : float
        Estimated lower bound on the affected fraction. May be ``inf`` when
        the denominator has no identifying bite.
    estimand : str
        Human-readable label for the estimand, e.g.
        ``"fraction of always-takers affected"``.
    at_group : object or None
        Target always-taker mediator group, or ``None`` for pooled targets.
    restriction : str
        Active monotonicity restriction label
        (e.g. ``"ordered"`` or ``"elementwise"``).
    diagnostics : dict of str to Any
        Solver metadata including cell-level contributions, theta minimums,
        no-bite flags, and paper-inequality audit fields.

    See Also
    --------
    LowerBoundRequest : Request descriptor that produces this result.
    ADEBoundsResult : Related bounds result for ADE targets.
    """

    lower_bound: float
    estimand: str
    at_group: object | None
    restriction: str
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a strict-JSON-safe dictionary.

        Non-finite floats are replaced with ``None`` and annotated with
        ``_is_finite`` / ``_nonfinite`` companion keys.

        Returns
        -------
        dict of str to Any
            Flat dictionary suitable for ``json.dumps`` without custom encoders.
        """
        lower_bound, lower_bound_finite, lower_bound_nonfinite = _json_safe_float(
            self.lower_bound
        )
        return {
            "lower_bound": lower_bound,
            "lower_bound_is_finite": lower_bound_finite,
            "lower_bound_nonfinite": lower_bound_nonfinite,
            "estimand": self.estimand,
            "at_group": _json_safe_payload(self.at_group),
            "restriction": self.restriction,
            "diagnostics": _json_safe_payload(self.diagnostics),
        }

    def to_frame(self) -> pd.DataFrame:
        """Convert to a single-row summary DataFrame.

        Returns
        -------
        pandas.DataFrame
            One-row frame with columns for result kind, estimand, group,
            restriction, lower bound value, finite status, and diagnostic
            summary counts.
        """
        lower_bound, lower_bound_finite, lower_bound_nonfinite = _json_safe_float(
            self.lower_bound
        )
        return pd.DataFrame(
            [
                {
                    "result": "lower_bound",
                    "estimand": self.estimand,
                    "at_group": _display_label(self.at_group, none_label="pooled"),
                    "restriction": self.restriction,
                    "lower_bound": lower_bound,
                    "lower_bound_status": _finite_status(
                        lower_bound_finite,
                        lower_bound_nonfinite,
                    ),
                    **_diagnostic_summary_fields(self.diagnostics),
                }
            ]
        )

    def __repr__(self) -> str:
        return (
            f"LowerBoundResult(lower_bound={self.lower_bound}, "
            f"at_group={self.at_group!r}, restriction={self.restriction!r})"
        )

    def __str__(self) -> str:
        """Return a vertical key-value summary for terminal display."""
        lb = self.lower_bound
        lb_str = f"{lb:.4f}" if math.isfinite(lb) else str(lb)
        group = _display_label(self.at_group, none_label="pooled")
        rows = [
            ("Lower Bound", lb_str),
            ("Estimand", self.estimand),
            ("AT Group", str(group)),
            ("Restriction", self.restriction),
        ]
        n_diag = len(self.diagnostics)
        footer = f"[{n_diag} diagnostics via .diagnostics]"
        return _vertical_card("Lower Bound on Fraction Affected", rows, footer)

    def summary(self, *, include_diagnostics: bool = False) -> str:
        """Return a formatted text summary of the result.

        Parameters
        ----------
        include_diagnostics : bool, default False
            If True, include full diagnostics dictionary in the output.

        Returns
        -------
        str
            Formatted multi-line summary string.
        """
        parts = [str(self)]
        if include_diagnostics:
            parts.append("")
            parts.append(" Diagnostics:")
            for key in sorted(self.diagnostics):
                parts.append(f"   {key}: {self.diagnostics[key]!r}")
            parts.append(_SEPARATOR)
        return "\n".join(parts)

    def _repr_html_(self) -> str:
        """Return styled HTML card for Jupyter notebook rendering."""
        return _summary_frame_html(self.to_frame())


@dataclass(frozen=True)
class ADEBoundsResult:
    """Bounds result for average direct effect on always-taker subgroups.

    Reports Lee-style trimming lower and upper bounds for the ADE
    estimand at a specified always-taker mediator group.  Either endpoint
    may be ``None`` when the group has no identifying bite (theta_kk_min
    equals zero or treatment-arm target mass is zero).

    Attributes
    ----------
    lower_bound : float or None
        Lower bound on the ADE.  ``None`` if not identified.
    upper_bound : float or None
        Upper bound on the ADE.  ``None`` if not identified.
    at_group : object
        Target always-taker mediator group value.
    restriction : str
        Active monotonicity restriction label.
    diagnostics : dict of str to Any
        Solver metadata including theta_kk values, treatment-arm masses,
        trimming quantiles, and paper-inequality checks.

    See Also
    --------
    ADEBoundsRequest : Request descriptor that produces this result.
    LowerBoundResult : Related affected-fraction lower-bound result.
    """

    lower_bound: float | None
    upper_bound: float | None
    at_group: object
    restriction: str
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a strict-JSON-safe dictionary.

        Non-finite floats are replaced with ``None`` and annotated with
        ``_is_finite`` / ``_nonfinite`` companion keys.  ``None`` endpoints
        produce ``null`` finite-status markers.

        Returns
        -------
        dict of str to Any
            Flat dictionary suitable for ``json.dumps``.
        """
        lower_bound, lower_bound_finite, lower_bound_nonfinite = _json_safe_optional_float(
            self.lower_bound
        )
        upper_bound, upper_bound_finite, upper_bound_nonfinite = _json_safe_optional_float(
            self.upper_bound
        )
        return {
            "lower_bound": lower_bound,
            "lower_bound_is_finite": lower_bound_finite,
            "lower_bound_nonfinite": lower_bound_nonfinite,
            "upper_bound": upper_bound,
            "upper_bound_is_finite": upper_bound_finite,
            "upper_bound_nonfinite": upper_bound_nonfinite,
            "at_group": _json_safe_payload(self.at_group),
            "restriction": self.restriction,
            "diagnostics": _json_safe_payload(self.diagnostics),
        }

    def to_frame(self) -> pd.DataFrame:
        """Convert to a single-row summary DataFrame.

        Returns
        -------
        pandas.DataFrame
            One-row frame with lower/upper bounds, finite-status columns,
            group label, restriction, and diagnostic summary.
        """
        lower_bound, lower_bound_finite, lower_bound_nonfinite = _json_safe_optional_float(
            self.lower_bound
        )
        upper_bound, upper_bound_finite, upper_bound_nonfinite = _json_safe_optional_float(
            self.upper_bound
        )
        return pd.DataFrame(
            [
                {
                    "result": "ade_bounds",
                    "at_group": _display_label(self.at_group, none_label="pooled"),
                    "restriction": self.restriction,
                    "lower_bound": lower_bound,
                    "lower_bound_status": _finite_status(
                        lower_bound_finite,
                        lower_bound_nonfinite,
                    ),
                    "upper_bound": upper_bound,
                    "upper_bound_status": _finite_status(
                        upper_bound_finite,
                        upper_bound_nonfinite,
                    ),
                    **_diagnostic_summary_fields(self.diagnostics),
                }
            ]
        )

    def __repr__(self) -> str:
        return (
            f"ADEBoundsResult(lower_bound={self.lower_bound}, "
            f"upper_bound={self.upper_bound}, "
            f"at_group={self.at_group!r}, restriction={self.restriction!r})"
        )

    def __str__(self) -> str:
        """Return a vertical key-value summary for terminal display."""
        def _fmt(v: "float | None") -> str:
            if v is None:
                return "not identified"
            if math.isfinite(v):
                return f"{v:.4f}"
            return str(v)

        group = _display_label(self.at_group, none_label="pooled")
        rows = [
            ("Lower Bound", _fmt(self.lower_bound)),
            ("Upper Bound", _fmt(self.upper_bound)),
            ("AT Group", str(group)),
            ("Restriction", self.restriction),
        ]
        n_diag = len(self.diagnostics)
        footer = f"[{n_diag} diagnostics via .diagnostics]"
        return _vertical_card("ADE Bounds", rows, footer)

    def summary(self, *, include_diagnostics: bool = False) -> str:
        """Return a formatted text summary of the result.

        Parameters
        ----------
        include_diagnostics : bool, default False
            If True, include full diagnostics dictionary in the output.

        Returns
        -------
        str
            Formatted multi-line summary string.
        """
        parts = [str(self)]
        if include_diagnostics:
            parts.append("")
            parts.append(" Diagnostics:")
            for key in sorted(self.diagnostics):
                parts.append(f"   {key}: {self.diagnostics[key]!r}")
            parts.append(_SEPARATOR)
        return "\n".join(parts)

    def _repr_html_(self) -> str:
        """Return styled HTML card for Jupyter notebook rendering."""
        return _summary_frame_html(self.to_frame())


@dataclass(frozen=True)
class SharpNullResult:
    """Decision result for a sharp-null hypothesis test.

    Records the method used, the null hypothesis label, the binary
    rejection decision, test statistic, critical value, p-value, the
    observed nuisance parameter vector beta, and the approximation method
    label.  Diagnostics carry method-specific solver metadata.

    Attributes
    ----------
    method : str
        Statistical method identifier (e.g. ``"CS"``, ``"ARP"``,
        ``"FSSTdd"``, ``"FSSTndd"``, ``"K"``).
    null_hypothesis : str
        Human-readable null-hypothesis statement.
    reject : bool
        ``True`` if the null is rejected at the requested significance level.
    test_stat : float
        Observed test statistic value.
    critical_value : float
        Critical value for the test at the requested alpha.
    p_value : float
        p-value for the test.
    beta_observed : list of float
        Observed nuisance parameter vector (moment-inequality betas).
    approximation : str
        Approximation method label (e.g. ``"chi2"``, ``"normal"``).
    diagnostics : dict of str to Any
        Method-specific metadata including constraint rows, shape matrices,
        and solver convergence information.

    See Also
    --------
    SharpNullRequest : Request descriptor that produces this result.
    TVConfidenceIntervalResult : Confidence interval via grid inversion.
    """

    method: str
    null_hypothesis: str
    reject: bool
    test_stat: float
    critical_value: float
    p_value: float
    beta_observed: list[float]
    approximation: str
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a strict-JSON-safe dictionary.

        Non-finite test statistics, critical values, and p-values are
        replaced with ``None`` and annotated with companion keys.
        The ``beta_observed`` vector is element-wise safe-encoded.

        Returns
        -------
        dict of str to Any
            Flat dictionary suitable for ``json.dumps``.
        """
        test_stat_value, test_stat_finite, test_stat_nonfinite = _json_safe_float(
            self.test_stat
        )
        critical_value, critical_value_finite, critical_value_nonfinite = _json_safe_float(
            self.critical_value
        )
        p_value, p_value_finite, p_value_nonfinite = _json_safe_float(self.p_value)
        beta_observed, beta_observed_is_finite, beta_observed_nonfinite = (
            _json_safe_float_sequence(self.beta_observed)
        )
        return {
            "method": self.method,
            "null_hypothesis": self.null_hypothesis,
            "reject": self.reject,
            "test_stat": test_stat_value,
            "test_stat_is_finite": test_stat_finite,
            "test_stat_nonfinite": test_stat_nonfinite,
            "critical_value": critical_value,
            "critical_value_is_finite": critical_value_finite,
            "critical_value_nonfinite": critical_value_nonfinite,
            "p_value": p_value,
            "p_value_is_finite": p_value_finite,
            "p_value_nonfinite": p_value_nonfinite,
            "beta_observed": beta_observed,
            "beta_observed_is_finite": beta_observed_is_finite,
            "beta_observed_nonfinite": beta_observed_nonfinite,
            "approximation": self.approximation,
            "diagnostics": _json_safe_payload(self.diagnostics),
        }

    def to_frame(self) -> pd.DataFrame:
        """Convert to a single-row summary DataFrame.

        Returns
        -------
        pandas.DataFrame
            One-row frame with method, hypothesis, rejection decision,
            test statistic, critical value, p-value, beta dimension,
            approximation, and diagnostic summary.
        """
        test_stat_value, test_stat_finite, test_stat_nonfinite = _json_safe_float(
            self.test_stat
        )
        critical_value, critical_value_finite, critical_value_nonfinite = _json_safe_float(
            self.critical_value
        )
        p_value, p_value_finite, p_value_nonfinite = _json_safe_float(self.p_value)
        return pd.DataFrame(
            [
                {
                    "result": "sharp_null",
                    "method": self.method,
                    "null_hypothesis": self.null_hypothesis,
                    "reject": bool(self.reject),
                    "test_stat": test_stat_value,
                    "test_stat_status": _finite_status(
                        test_stat_finite,
                        test_stat_nonfinite,
                    ),
                    "critical_value": critical_value,
                    "critical_value_status": _finite_status(
                        critical_value_finite,
                        critical_value_nonfinite,
                    ),
                    "p_value": p_value,
                    "p_value_status": _finite_status(p_value_finite, p_value_nonfinite),
                    "beta_dimension": len(self.beta_observed),
                    "approximation": self.approximation,
                    **_diagnostic_summary_fields(self.diagnostics),
                }
            ]
        )

    def __repr__(self) -> str:
        return (
            f"SharpNullResult(method={self.method!r}, reject={self.reject}, "
            f"p_value={self.p_value}, test_stat={self.test_stat}, "
            f"critical_value={self.critical_value})"
        )

    def __str__(self) -> str:
        """Return a vertical key-value summary for terminal display."""
        reject_str = "Yes" if self.reject else "No"
        p_str = f"{self.p_value:.4f}" if math.isfinite(self.p_value) else str(self.p_value)
        ts_str = f"{self.test_stat:.4f}" if math.isfinite(self.test_stat) else str(self.test_stat)
        cv_str = (
            f"{self.critical_value:.4f}"
            if math.isfinite(self.critical_value)
            else str(self.critical_value)
        )
        rows = [
            ("Reject H\u2080", reject_str),
            ("P-value", p_str),
            ("Test Statistic", ts_str),
            ("Critical Value", cv_str),
            ("Null Hypothesis", self.null_hypothesis),
            ("Approximation", self.approximation),
            ("Beta Dimension", str(len(self.beta_observed))),
        ]
        n_diag = len(self.diagnostics)
        footer = f"[{n_diag} diagnostics via .diagnostics]"
        return _vertical_card(f"Sharp Null Test \u2500 {self.method}", rows, footer)

    def summary(self, *, include_diagnostics: bool = False) -> str:
        """Return a formatted text summary of the result.

        Parameters
        ----------
        include_diagnostics : bool, default False
            If True, include full diagnostics dictionary in the output.

        Returns
        -------
        str
            Formatted multi-line summary string.
        """
        parts = [str(self)]
        if include_diagnostics:
            parts.append("")
            parts.append(" Diagnostics:")
            for key in sorted(self.diagnostics):
                parts.append(f"   {key}: {self.diagnostics[key]!r}")
            parts.append(_SEPARATOR)
        return "\n".join(parts)

    def _repr_html_(self) -> str:
        """Return styled HTML card for Jupyter notebook rendering."""
        return _summary_frame_html(self.to_frame())


@dataclass(frozen=True)
class TVConfidenceIntervalResult:
    """Grid-inversion confidence interval for a total-variation target.

    Constructed by testing a grid of candidate parameter values and
    collecting those not rejected to form a confidence set.  The interval
    endpoints are the minimum and maximum of the accepted grid points.

    Attributes
    ----------
    at_group : object
        Target always-taker mediator group.
    alpha : float
        Significance level used for the confidence interval.
    method : str
        Statistical method used for each grid-point test.
    accepted_grid : list of float
        Grid points not rejected at the given alpha.
    lower : float or None
        Lower endpoint of the confidence interval, or ``None`` if the
        accepted set is empty.
    upper : float or None
        Upper endpoint of the confidence interval, or ``None`` if the
        accepted set is empty.
    test_grid : list of dict
        Per-grid-point test results including rejection decisions.
    approximation : str
        Approximation method label.
    diagnostics : dict of str to Any
        Grid-level summary metadata.

    See Also
    --------
    SharpNullResult : Individual test result at a single grid point.
    """

    at_group: object
    alpha: float
    method: str
    accepted_grid: list[float]
    lower: float | None
    upper: float | None
    test_grid: list[dict[str, Any]]
    approximation: str
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a strict-JSON-safe dictionary.

        Returns
        -------
        dict of str to Any
            Dictionary with interval endpoints, accepted grid, full
            test-grid records, and diagnostics.
        """
        lower, lower_finite, lower_nonfinite = _json_safe_optional_float(self.lower)
        upper, upper_finite, upper_nonfinite = _json_safe_optional_float(self.upper)
        return {
            "at_group": _json_safe_payload(self.at_group),
            "alpha": float(self.alpha),
            "method": self.method,
            "accepted_grid": _json_safe_payload(self.accepted_grid),
            "lower": lower,
            "lower_is_finite": lower_finite,
            "lower_nonfinite": lower_nonfinite,
            "upper": upper,
            "upper_is_finite": upper_finite,
            "upper_nonfinite": upper_nonfinite,
            "test_grid": _json_safe_payload(self.test_grid),
            "approximation": self.approximation,
            "diagnostics": _json_safe_payload(self.diagnostics),
        }

    def to_frame(self) -> pd.DataFrame:
        """Convert to a single-row summary DataFrame.

        Returns
        -------
        pandas.DataFrame
            One-row frame with method, group, alpha, endpoints,
            grid-point counts, approximation, and diagnostic summary.
        """
        lower, lower_finite, lower_nonfinite = _json_safe_optional_float(self.lower)
        upper, upper_finite, upper_nonfinite = _json_safe_optional_float(self.upper)
        return pd.DataFrame(
            [
                {
                    "result": "tv_confidence_interval",
                    "method": self.method,
                    "at_group": _display_label(self.at_group, none_label="pooled"),
                    "alpha": float(self.alpha),
                    "lower": lower,
                    "lower_status": _finite_status(lower_finite, lower_nonfinite),
                    "upper": upper,
                    "upper_status": _finite_status(upper_finite, upper_nonfinite),
                    "grid_points": len(self.test_grid),
                    "accepted_grid_points": len(self.accepted_grid),
                    "approximation": self.approximation,
                    **_diagnostic_summary_fields(self.diagnostics),
                }
            ]
        )

    def test_grid_frame(self) -> pd.DataFrame:
        """Return per-grid-point test results as a DataFrame.

        Returns
        -------
        pandas.DataFrame
            One row per grid point with columns from the test-grid
            dictionaries (test_stat, critical_value, reject, etc.).
        """
        return pd.DataFrame(self.test_grid)

    def __repr__(self) -> str:
        return (
            f"TVConfidenceIntervalResult(lower={self.lower}, upper={self.upper}, "
            f"method={self.method!r}, grid_points={len(self.test_grid)})"
        )

    def __str__(self) -> str:
        """Return a vertical key-value summary for terminal display."""
        def _fmt(v: "float | None") -> str:
            if v is None:
                return "empty set"
            if math.isfinite(v):
                return f"{v:.4f}"
            return str(v)

        group = _display_label(self.at_group, none_label="pooled")
        rows = [
            ("CI Lower", _fmt(self.lower)),
            ("CI Upper", _fmt(self.upper)),
            ("Alpha", f"{self.alpha:.4f}"),
            ("Method", self.method),
            ("AT Group", str(group)),
            ("Grid Points", str(len(self.test_grid))),
            ("Accepted Points", str(len(self.accepted_grid))),
            ("Approximation", self.approximation),
        ]
        n_diag = len(self.diagnostics)
        footer = f"[{n_diag} diagnostics via .diagnostics]"
        return _vertical_card("TV Confidence Interval", rows, footer)

    def summary(self, *, include_diagnostics: bool = False) -> str:
        """Return a formatted text summary of the result.

        Parameters
        ----------
        include_diagnostics : bool, default False
            If True, include full diagnostics dictionary in the output.

        Returns
        -------
        str
            Formatted multi-line summary string.
        """
        parts = [str(self)]
        if include_diagnostics:
            parts.append("")
            parts.append(" Diagnostics:")
            for key in sorted(self.diagnostics):
                parts.append(f"   {key}: {self.diagnostics[key]!r}")
            parts.append(_SEPARATOR)
        return "\n".join(parts)

    def _repr_html_(self) -> str:
        """Return styled HTML card for Jupyter notebook rendering."""
        return _summary_frame_html(self.to_frame())


def _json_safe_float(value: float) -> tuple[float | None, bool, str | None]:
    """Convert a float to a JSON-safe triple (value, is_finite, nonfinite_marker).

    Parameters
    ----------
    value : float
        Numeric value to convert.

    Returns
    -------
    tuple of (float or None, bool, str or None)
        ``(numeric, True, None)`` for finite values; ``(None, False, marker)``
        for NaN or infinity where marker is ``"nan"``,
        ``"positive_infinity"``, or ``"negative_infinity"``.
    """
    numeric = float(value)
    if math.isfinite(numeric):
        return numeric, True, None
    if math.isnan(numeric):
        return None, False, "nan"
    return None, False, "positive_infinity" if numeric > 0 else "negative_infinity"


def _json_safe_optional_float(value: float | None) -> tuple[float | None, bool | None, str | None]:
    """Convert an optional float to a JSON-safe triple.

    Parameters
    ----------
    value : float or None
        Numeric value or ``None``.

    Returns
    -------
    tuple of (float or None, bool or None, str or None)
        ``(None, None, None)`` when input is ``None``; otherwise delegates
        to :func:`_json_safe_float`.
    """
    if value is None:
        return None, None, None
    return _json_safe_float(value)


def _json_safe_float_sequence(
    values: list[float],
) -> tuple[list[float | None], list[bool], list[str | None]]:
    """Apply JSON-safe float encoding element-wise to a list.

    Parameters
    ----------
    values : list of float
        Numeric sequence to encode.

    Returns
    -------
    tuple of (list, list, list)
        Parallel lists of safe values, finite flags, and nonfinite markers.
    """
    safe_values: list[float | None] = []
    finite_flags: list[bool] = []
    nonfinite_markers: list[str | None] = []
    for value in values:
        safe_value, is_finite, nonfinite = _json_safe_float(value)
        safe_values.append(safe_value)
        finite_flags.append(is_finite)
        nonfinite_markers.append(nonfinite)
    return safe_values, finite_flags, nonfinite_markers


def _finite_status(is_finite: bool | None, nonfinite: str | None) -> str:
    """Map finite/nonfinite flags to a human-readable status label.

    Parameters
    ----------
    is_finite : bool or None
        Finite flag from JSON-safe encoding.
    nonfinite : str or None
        Nonfinite marker string.

    Returns
    -------
    str
        One of ``"finite"``, ``"not_applicable"``, ``"nan"``,
        ``"positive_infinity"``, or ``"negative_infinity"``.
    """
    if is_finite is None:
        return "not_applicable"
    if is_finite:
        return "finite"
    if nonfinite is None:
        raise ValueError("non-finite status requires a marker.")
    return nonfinite


def _display_label(value: Any, *, none_label: str) -> Any:
    """Format a group value for display, substituting a label for None.

    Parameters
    ----------
    value : Any
        Raw group value.
    none_label : str
        Label to display when the value is ``None``.

    Returns
    -------
    str
        Formatted display label.
    """
    safe = _json_safe_payload(value)
    if safe is None:
        return none_label
    return _display_label_atom(safe)


def _display_label_atom(value: Any) -> str:
    """Recursively format an atomic or nested value into a display string."""
    if isinstance(value, list):
        return "(" + ", ".join(_display_label_atom(item) for item in value) + ")"
    if isinstance(value, bool):
        return "1" if value else "0"
    if value is None:
        return "NA"
    return str(value)


def _diagnostic_summary_fields(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Extract compact summary fields from a diagnostics dictionary.

    Returns a two-key dict with ``diagnostic_count`` and a truncated
    ``diagnostic_keys`` preview string.
    """
    keys = sorted(str(key) for key in diagnostics)
    return {
        "diagnostic_count": len(keys),
        "diagnostic_keys": _diagnostic_key_summary(keys),
    }


def _diagnostic_key_summary(keys: list[str]) -> str:
    """Return a comma-separated preview of diagnostic keys, truncating if needed."""
    if len(keys) <= _DIAGNOSTIC_KEY_PREVIEW_LIMIT:
        return ", ".join(keys)
    preview = ", ".join(keys[:_DIAGNOSTIC_KEY_PREVIEW_LIMIT])
    omitted = len(keys) - _DIAGNOSTIC_KEY_PREVIEW_LIMIT
    return f"{preview}, ... (+{omitted} more)"


def _summary_frame_html(frame: pd.DataFrame) -> str:
    """Render a result summary DataFrame as a styled HTML card."""
    result_kind = _result_kind_from_frame(frame)
    status = _summary_status_from_frame(frame)
    caption = html.escape(f"TestMechs {result_kind.replace('_', ' ')} summary")
    table = _summary_frame_table_html(frame, caption=caption)
    kind_class = html.escape(f"testmechs-result-{result_kind}")
    status_class = html.escape(f"testmechs-status-{status}")
    return (
        _SUMMARY_HTML_STYLE
        + f'<div class="testmechs-result-card {kind_class} {status_class}" '
        f'role="region" aria-label="{caption}" '
        f'data-result-kind="{html.escape(result_kind)}" '
        f'data-status="{html.escape(status)}">'
        f'<div class="testmechs-result-caption">{caption}</div>'
        f"{table}</div>"
    )


def _summary_frame_text(frame: pd.DataFrame) -> str:
    """Render a result summary DataFrame as a compact text table."""
    display_frame = pd.DataFrame(
        [
            {
                _summary_column_label(column): _summary_text_value(
                    column=column,
                    value=row[column],
                )
                for column in frame.columns
            }
            for _, row in frame.iterrows()
        ]
    )
    return display_frame.to_string(index=False)


def _summary_frame_table_html(frame: pd.DataFrame, *, caption: str) -> str:
    """Render the inner HTML table for a result summary card."""
    header = "".join(
        f'<th scope="col" data-column="{html.escape(str(column))}" '
        f'data-column-label="{html.escape(_summary_column_label(column))}">'
        f"{html.escape(_summary_column_label(column))}</th>"
        for column in frame.columns
    )
    body_rows = []
    for _, row in frame.iterrows():
        cells = []
        for column in frame.columns:
            value = row[column]
            cells.append(_summary_frame_cell_html(column=column, value=value))
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    body = "".join(body_rows)
    return (
        '<table class="testmechs-result testmechs-result-summary">'
        f'<caption class="testmechs-result-table-caption">{caption}</caption>'
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def _summary_column_label(column: object) -> str:
    """Convert an underscore-separated column name to a space-separated label."""
    return str(column).replace("_", " ")


def _summary_frame_cell_html(*, column: object, value: Any) -> str:
    """Render a single table cell as HTML with accessibility attributes."""
    raw_column = str(column)
    raw_value = str(value)
    column_label = _summary_column_label(column)
    css_classes = ["testmechs-cell", _summary_cell_value_class(value)]
    displayed_value = _summary_display_value(value)
    aria_value = raw_value
    if raw_column.endswith("_status"):
        displayed_value = raw_value.replace("_", " ")
        aria_value = raw_value
        css_classes.extend(
            [
                "testmechs-cell-status",
                f"testmechs-cell-status-{_summary_status_cell_class(raw_value)}",
            ]
        )
    return (
        f'<td class="{" ".join(css_classes)}" '
        f'data-column="{html.escape(raw_column)}" '
        f'data-column-label="{html.escape(column_label)}" '
        f'data-value="{html.escape(raw_value)}" '
        f'data-display-value="{html.escape(displayed_value)}" '
        f'title="{html.escape(raw_value)}" '
        f'aria-label="{html.escape(f"{column_label}: {aria_value}")}">'
        f"{html.escape(displayed_value)}</td>"
    )


def _summary_display_value(value: Any) -> str:
    """Format a cell value for HTML display."""
    if _is_scalar_missing(value):
        return "not available"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if math.isfinite(numeric):
            return f"{numeric:.6g}"
    return str(value)


def _summary_cell_value_class(value: Any) -> str:
    """Return the CSS class name for a cell value's data type."""
    if isinstance(value, (bool, np.bool_)):
        return "testmechs-cell-boolean"
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value,
        (bool, np.bool_),
    ):
        return "testmechs-cell-numeric"
    if value is None:
        return "testmechs-cell-missing"
    return "testmechs-cell-text"


def _summary_text_value(*, column: object, value: Any) -> str:
    """Format a cell value for plain-text table display."""
    raw_column = str(column)
    if raw_column == "result":
        return str(value).replace("_", " ")
    if raw_column.endswith("_status"):
        return str(value).replace("_", " ")
    if value is None:
        return "not available"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if math.isfinite(numeric):
            return repr(numeric)
    return str(value)


def _summary_status_cell_class(status: str) -> str:
    """Map a status string to a CSS modifier class."""
    if status == "finite":
        return "finite"
    if status == "not_applicable":
        return "not-applicable"
    return "nonfinite"


_SUMMARY_HTML_STYLE = (
    "<style>"
    ".testmechs-result-card{font-family:system-ui,-apple-system,BlinkMacSystemFont,"
    "'Segoe UI',sans-serif;font-size:13px;line-height:1.35;color:#1f2933;"
    "border:1px solid #d9e2ec;border-radius:6px;background:#fff;"
    "padding:10px 12px;margin:0.5em 0;max-width:100%;overflow-x:auto;}"
    ".testmechs-result-caption{font-weight:600;margin-bottom:6px;color:#102a43;}"
    ".testmechs-result-table-caption{position:absolute;width:1px;height:1px;"
    "padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);"
    "white-space:nowrap;border:0;}"
    ".testmechs-result-summary{border-collapse:collapse;width:100%;}"
    ".testmechs-result-summary th{background:#f0f4f8;color:#243b53;"
    "font-weight:600;text-align:left;white-space:nowrap;}"
    ".testmechs-result-summary td,.testmechs-result-summary th{"
    "border-bottom:1px solid #d9e2ec;padding:5px 7px;vertical-align:top;}"
    ".testmechs-result-summary tr:last-child td{border-bottom:0;}"
    ".testmechs-cell{overflow-wrap:anywhere;word-break:normal;}"
    ".testmechs-cell-numeric{text-align:right;font-variant-numeric:tabular-nums;}"
    ".testmechs-cell-boolean,.testmechs-cell-missing{text-align:center;}"
    ".testmechs-cell-status{font-weight:600;white-space:nowrap;}"
    ".testmechs-cell-status-finite{color:#0b6b3a;background:#e7f6ec;}"
    ".testmechs-cell-status-not-applicable{color:#52606d;background:#f0f4f8;}"
    ".testmechs-cell-status-nonfinite{color:#8a4b0f;background:#fff4d6;}"
    ".testmechs-status-nonfinite .testmechs-result-caption{color:#8a4b0f;}"
    "</style>"
)


def _result_kind_from_frame(frame: pd.DataFrame) -> str:
    """Extract the result-kind identifier from a summary frame's first row."""
    if "result" not in frame or frame.empty:
        return "summary"
    result = str(frame.iloc[0]["result"])
    return "".join(character if character.isalnum() else "_" for character in result).strip("_")


def _summary_status_from_frame(frame: pd.DataFrame) -> str:
    """Determine overall finite-status for a summary frame."""
    status_columns = [column for column in frame.columns if str(column).endswith("_status")]
    statuses = {
        str(value)
        for column in status_columns
        for value in frame[column].tolist()
        if value is not None
    }
    if any(status not in {"finite", "not_applicable"} for status in statuses):
        return "nonfinite"
    if "finite" in statuses:
        return "finite"
    return "not_applicable"


def _json_safe_payload(value: Any) -> Any:
    """Recursively convert an arbitrary value to a strict-JSON-safe payload.

    Handles numpy arrays/scalars, pandas objects, pathlib paths, datetimes,
    tuples, and nested dicts/lists.  Non-finite floats are replaced with
    their string markers.
    """
    if isinstance(value, dict):
        return {str(key): _json_safe_payload(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe_payload(item) for item in value]
    if isinstance(value, list):
        return [_json_safe_payload(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe_payload(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe_payload(value.item())
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        safe_value, _, nonfinite = _json_safe_float(value)
        return safe_value if nonfinite is None else nonfinite
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Interval):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    return value


def _is_scalar_missing(value: Any) -> bool:
    """Return True for scalar missing sentinels without collapsing arrays."""

    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    return False


def _reject_nonfinite_json_numbers(value: Any, *, field_name: str) -> None:
    """Reject JSON payloads that contain parsed non-finite numeric values."""

    if isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite_json_numbers(item, field_name=f"{field_name}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_nonfinite_json_numbers(item, field_name=f"{field_name}[{index}]")
        return
    if isinstance(value, np.ndarray):
        for index, item in enumerate(value.tolist()):
            _reject_nonfinite_json_numbers(item, field_name=f"{field_name}[{index}]")
        return
    if isinstance(value, np.generic):
        _reject_nonfinite_json_numbers(value.item(), field_name=field_name)
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(
            f"{field_name} must not contain non-finite numeric values."
        )
