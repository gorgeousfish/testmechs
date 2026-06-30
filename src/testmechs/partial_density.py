"""Partial-density data computation and visualization for Testing Mechanisms.

This module implements the partial-density (and partial-PMF) diagnostic used
to visualize the identifying variation in the Testing Mechanisms framework.
For a binary mediator with levels {0, 1}, the partial density
``f(Y, M=m | D=d)`` decomposes the outcome distribution into components
attributable to always-takers and compliers.

Key public functions:

- ``partial_density_data`` -- compute plot-ready partial-PMF or
  partial-density records (discrete or continuous Y).
- ``partial_density_plot`` -- render the partial-density evidence as a
  Matplotlib figure.

The module supports both unadjusted (sample-frequency) and regression-adjusted
(formula-based) estimation.  All public functions validate inputs eagerly
and raise ``ValueError`` on violation.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
from numbers import Integral
from pathlib import Path
import textwrap
from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid as scipy_trapezoid

from .preprocess import discretize_y, ordered_binary_support_levels, remove_missing_from_df
from .regression import compute_adjusted_mediator_masses, compute_adjusted_probabilities, parse_reg_formula
from .results import (
    _finite_status,
    _json_safe_float,
    _json_safe_payload,
    _summary_frame_html,
    _summary_frame_text,
)

_PARTIAL_DENSITY_COLORS = ("#0072B2", "#D55E00")
_KDE_MAX_CHUNK_CELLS = 1_000_000
_DISCRETE_BAR_LABEL_MAX_LEVELS = 6
_POSITIVE_PART_BUSY_EDGE_RATIO = 0.7


@dataclass(frozen=True)
class PartialDensityDataResult:
    """Immutable result container for partial-density or partial-PMF records.

    Attributes
    ----------
    frac_compliers : float
        Estimated fraction of compliers (P(M=1|D=1) - P(M=1|D=0)).
    frac_ats : float
        Estimated fraction of always-takers (P(M=1|D=0)).
    theta_ats : float
        Ratio frac_ats / (frac_compliers + frac_ats).
    y_levels : list of str
        Ordered outcome support labels (discrete case).
    partial_mass_records : list of dict
        Discrete partial-PMF records with keys ``'y'``, ``'partial11'``,
        ``'partial01'``.
    diagnostics : dict
        Rich diagnostic payload including positive-part, target role, and
        estimation metadata.
    partial_density_records : list of dict or None
        Continuous partial-density grid records (only when ``continuous_y=True``).

    See Also
    --------
    partial_density_data : Factory that produces this dataclass.
    partial_density_plot : Renders this result as a figure.
    """

    frac_compliers: float
    frac_ats: float
    theta_ats: float
    y_levels: list[str]
    partial_mass_records: list[dict[str, float | str]]
    diagnostics: dict[str, Any]
    partial_density_records: list[dict[str, float | str]] | None = None

    @property
    def partial_density_row_records(self) -> list[dict[str, Any]]:
        source_records = (
            self.partial_density_records
            if self.partial_density_records is not None
            else self.partial_mass_records
        )
        row_records: list[dict[str, Any]] = []
        for record in source_records:
            for value_column in ("partial11", "partial01"):
                density, density_is_finite, density_nonfinite = _partial_density_value_payload(
                    record[value_column]
                )
                row_records.append(
                    {
                        "y": record["y"],
                        "partial_density_role": value_column,
                        "partial_density": density,
                        "partial_density_is_finite": density_is_finite,
                        "partial_density_nonfinite": density_nonfinite,
                        "target_original_mediator_level": self.diagnostics.get(
                            "target_original_mediator_level"
                        ),
                        "target_normalized_mediator_level": self.diagnostics.get(
                            "target_normalized_mediator_level"
                        ),
                        "original_treatment_level": self.diagnostics.get(
                            f"{value_column}_original_treatment_level"
                        ),
                        "partial_density_target_role": self.diagnostics.get(
                            "partial_density_target_role"
                        ),
                    }
                )
        return _json_safe_payload(row_records)  # type: ignore[no-any-return]

    def to_dict(self) -> dict[str, Any]:
        frac_compliers, frac_compliers_is_finite, frac_compliers_nonfinite = _json_safe_float(
            self.frac_compliers
        )
        frac_ats, frac_ats_is_finite, frac_ats_nonfinite = _json_safe_float(self.frac_ats)
        theta_ats, theta_ats_is_finite, theta_ats_nonfinite = _json_safe_float(self.theta_ats)
        return {
            "frac_compliers": frac_compliers,
            "frac_compliers_is_finite": frac_compliers_is_finite,
            "frac_compliers_nonfinite": frac_compliers_nonfinite,
            "frac_ats": frac_ats,
            "frac_ats_is_finite": frac_ats_is_finite,
            "frac_ats_nonfinite": frac_ats_nonfinite,
            "theta_ats": theta_ats,
            "theta_ats_is_finite": theta_ats_is_finite,
            "theta_ats_nonfinite": theta_ats_nonfinite,
            "y_levels": _json_safe_payload(self.y_levels),
            "partial_mass_records": _json_safe_payload(
                [dict(record) for record in self.partial_mass_records]
            ),
            "partial_density_records": None
            if self.partial_density_records is None
            else _json_safe_payload([dict(record) for record in self.partial_density_records]),
            "partial_density_row_records": [dict(record) for record in self.partial_density_row_records],
            "diagnostics": _json_safe_payload(self.diagnostics),
        }

    def to_frame(self) -> pd.DataFrame:
        frac_compliers, frac_compliers_is_finite, frac_compliers_nonfinite = _json_safe_float(
            self.frac_compliers
        )
        frac_ats, frac_ats_is_finite, frac_ats_nonfinite = _json_safe_float(self.frac_ats)
        theta_ats, theta_ats_is_finite, theta_ats_nonfinite = _json_safe_float(self.theta_ats)
        positive_part, positive_part_is_finite, positive_part_nonfinite = _json_safe_float(
            _partial_density_positive_part_value(self.diagnostics)
        )
        return pd.DataFrame(
            [
                {
                    "result": "partial_density_data",
                    "partial_density_target_role": self.diagnostics.get(
                        "partial_density_target_role"
                    ),
                    "continuous_y": bool(self.diagnostics.get("continuous_y", False)),
                    "adjusted": bool(self.diagnostics.get("adjusted", False)),
                    "record_count": _partial_density_record_count(self),
                    "support_count": _partial_density_support_count(self),
                    "frac_compliers": frac_compliers,
                    "frac_compliers_status": _finite_status(
                        frac_compliers_is_finite,
                        frac_compliers_nonfinite,
                    ),
                    "frac_ats": frac_ats,
                    "frac_ats_status": _finite_status(frac_ats_is_finite, frac_ats_nonfinite),
                    "theta_ats": theta_ats,
                    "theta_ats_status": _finite_status(theta_ats_is_finite, theta_ats_nonfinite),
                    "positive_part": positive_part,
                    "positive_part_status": _finite_status(
                        positive_part_is_finite,
                        positive_part_nonfinite,
                    ),
                    "positive_part_rule": self.diagnostics.get("positive_part_integral_rule"),
                    "diagnostic_count": len(self.diagnostics),
                }
            ],
            dtype=object,
        )

    def __str__(self) -> str:
        return _summary_frame_text(self.to_frame())

    def _repr_html_(self) -> str:
        return _summary_frame_html(self.to_frame())


def _partial_density_record_count(result: PartialDensityDataResult) -> int:
    """Count the active partial-density records."""
    if result.partial_density_records is not None:
        return len(result.partial_density_records)
    return len(result.partial_mass_records)


def _partial_density_support_count(result: PartialDensityDataResult) -> int:
    """Count the outcome support or grid points."""
    if result.partial_density_records is not None:
        return int(result.diagnostics.get("output_grid_points", len(result.partial_density_records)))
    return len(result.y_levels)


def _partial_density_positive_part_value(diagnostics: dict[str, Any]) -> float:
    """Extract the positive-part scalar from diagnostics."""
    if diagnostics.get("continuous_y"):
        value = diagnostics.get("positive_part_partial_density_integral")
    else:
        value = diagnostics.get("positive_part_partial_pmf_diff")
    if value is None:
        return math.nan
    return float(value)


def _partial_density_value_payload(value: Any) -> tuple[Any, bool | None, str | None]:
    """Coerce a partial-density value to a JSON-safe triple."""
    if isinstance(value, bool):
        return _json_safe_payload(value), None, None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return _json_safe_payload(value), None, None
    safe_value, is_finite, nonfinite = _json_safe_float(numeric)
    return safe_value, is_finite, nonfinite


def partial_density_data(
    *,
    df: pd.DataFrame | None = None,
    data_path: str | Path | None = None,
    d: str,
    m: str,
    y: str,
    num_y_bins: int | None = None,
    plot_nts: bool = False,
    continuous_y: bool = False,
    num_grid_points: int = 10000,
    reg_formula: str | None = None,
) -> PartialDensityDataResult:
    """Return plot-ready partial-density or partial-PMF records.

    The mediator must be a scalar binary column.  With ``continuous_y=False``,
    the function returns finite-support partial-PMF records, optionally after
    outcome binning.  With ``continuous_y=True``, it evaluates a Gaussian
    kernel-density grid with ``num_grid_points`` points.

    The result keeps row-level records, positive-part diagnostics, support
    metadata, and strict-JSON export payloads.  When ``reg_formula`` is
    supplied, the probabilities are regression-adjusted rather than
    sample-frequency based.

    Parameters
    ----------
    df : pd.DataFrame or None
        Analysis dataframe.  Exactly one of *df* or *data_path* must be given.
    data_path : str, Path, or None
        Path to a CSV file (loaded via ``pd.read_csv``).
    d : str
        Binary treatment column name.
    m : str
        Binary mediator column name.
    y : str
        Outcome column name.
    num_y_bins : int or None, optional
        If given, discretize the outcome into this many quantile bins.
        Only valid when ``continuous_y=False``.
    plot_nts : bool, default False
        If ``True``, flip treatment and mediator internally to target
        never-takers (M=0) instead of always-takers (M=1).
    continuous_y : bool, default False
        Treat Y as continuous and estimate kernel densities.
    num_grid_points : int, default 10000
        Number of grid points for the continuous density evaluation.
    reg_formula : str or None, optional
        Regression formula for observational adjustment.

    Returns
    -------
    PartialDensityDataResult
        Frozen dataclass with partial-density records, complier/AT fractions,
        positive-part diagnostics, and rich metadata.

    Raises
    ------
    ValueError
        If input validation fails (e.g., non-binary mediator, both df and
        data_path provided, non-finite Y values, empty arms).

    Examples
    --------
    >>> import pandas as pd
    >>> from testmechs.partial_density import partial_density_data
    >>> df = pd.DataFrame({
    ...     "D": [0,0,0,1,1,1], "M": [0,1,0,1,1,1], "Y": [1,2,3,1,2,3]
    ... })
    >>> result = partial_density_data(df=df, d="D", m="M", y="Y")
    >>> result.frac_compliers  # doctest: +SKIP
    0.666...

    Notes
    -----
    This is the Python equivalent of the data layer behind the R function
    ``partial_density_plot``.  The Python implementation separates data
    computation (this function) from rendering (``partial_density_plot``).

    See Also
    --------
    partial_density_plot : Render partial-density records as a figure.
    discretize_y : The binning function used when *num_y_bins* is set.
    compute_adjusted_probabilities : The regression layer used when
        *reg_formula* is supplied (discrete case).
    """

    d = _validate_scalar_column_name(d, name="d")
    m = _validate_scalar_mediator_column(m)
    y = _validate_scalar_column_name(y, name="y")
    plot_nts = _validate_bool_flag(plot_nts, name="plot_nts")
    continuous_y = _validate_bool_flag(continuous_y, name="continuous_y")
    num_grid_points = _validate_num_grid_points(num_grid_points)
    if continuous_y and num_y_bins is not None:
        raise ValueError("num_y_bins is only supported when continuous_y is False.")
    if reg_formula is not None:
        _reject_partial_density_formula_role_reuse(reg_formula=reg_formula, d=d, m=m, y=y)
    if df is not None and data_path is not None:
        raise ValueError("Exactly one of df or data_path must be provided.")
    if df is None:
        if data_path is None:
            raise ValueError("Exactly one of df or data_path must be provided.")
        df = pd.read_csv(data_path)

    source_n_obs = int(len(df))
    data = remove_missing_from_df(df=df, d=d, m=m, y=y, reg_formula=reg_formula)
    treatment_levels = _binary_levels(data[d], column=d)
    mediator_levels = _binary_levels(data[m], column=m)
    _reject_nonfinite_partial_density_y(data[y], column=y)

    normalized = data.copy()
    normalized[d] = _normalize_binary_series(normalized[d], treatment_levels)
    normalized[m] = _normalize_binary_series(normalized[m], mediator_levels)

    if plot_nts:
        normalized[d] = 1 - normalized[d]
        normalized[m] = 1 - normalized[m]

    if continuous_y:
        _reject_nonfinite_continuous_partial_density_y(normalized[y], column=y)
        if reg_formula is not None:
            return _adjusted_continuous_partial_density_data(
                normalized=normalized,
                d=d,
                m=m,
                y=y,
                num_grid_points=num_grid_points,
                plot_nts=plot_nts,
                treatment_levels=treatment_levels,
                mediator_levels=mediator_levels,
                reg_formula=reg_formula,
                source_n_obs=source_n_obs,
            )
        return _continuous_partial_density_data(
            normalized=normalized,
            d=d,
            m=m,
            y=y,
            num_grid_points=num_grid_points,
            plot_nts=plot_nts,
            treatment_levels=treatment_levels,
            mediator_levels=mediator_levels,
        )

    requested_num_y_bins = num_y_bins
    if num_y_bins is not None:
        normalized[y] = discretize_y(normalized[y], num_bins=num_y_bins)
    elif (
        _discrete_partial_density_y_support_is_numeric(normalized[y])
        and len(normalized) / normalized[y].nunique(dropna=False) <= 30
    ):
        normalized[y] = discretize_y(normalized[y], num_bins=5)
        num_y_bins = 5

    if reg_formula is not None:
        return _adjusted_discrete_partial_density_data(
            normalized=normalized,
            d=d,
            m=m,
            y=y,
            requested_num_y_bins=requested_num_y_bins,
            applied_num_y_bins=num_y_bins,
            plot_nts=plot_nts,
            treatment_levels=treatment_levels,
            mediator_levels=mediator_levels,
            reg_formula=reg_formula,
            source_n_obs=source_n_obs,
        )

    dvec = normalized[d]
    mvec = normalized[m]
    yvec = normalized[y]
    frac_compliers = float(mvec[dvec == 1].mean() - mvec[dvec == 0].mean())
    frac_ats = float(mvec[dvec == 0].mean())
    theta_denominator = frac_compliers + frac_ats
    theta_ats = float(frac_ats / theta_denominator) if theta_denominator != 0 else math.nan
    treated_mask = (dvec == 1) & (mvec == 1)
    untreated_mask = (dvec == 0) & (mvec == 1)
    treated_m1_count = int(treated_mask.sum())
    untreated_m1_count = int(untreated_mask.sum())

    records: list[dict[str, float | str]] = []
    ordered_y_values = _ordered_support_values(yvec)
    y_levels = [str(level) for level in ordered_y_values]
    for y_value in ordered_y_values:
        pmf_treated = _conditional_indicator_mean(yvec[treated_mask] == y_value)
        pmf_untreated = _conditional_indicator_mean(yvec[untreated_mask] == y_value)
        records.append(
            {
                "y": str(y_value),
                "partial11": (frac_ats + frac_compliers) * pmf_treated,
                "partial01": frac_ats * pmf_untreated,
            }
        )
    positive_part_diagnostics = _discrete_positive_part_diagnostics(records)

    return PartialDensityDataResult(
        frac_compliers=frac_compliers,
        frac_ats=frac_ats,
        theta_ats=theta_ats,
        y_levels=y_levels,
        partial_mass_records=records,
        diagnostics={
            "outcome_column": y,
            "requested_num_y_bins": requested_num_y_bins,
            "applied_num_y_bins": int(yvec.nunique(dropna=False)),
            "auto_num_y_bins": num_y_bins if requested_num_y_bins is None else None,
            "plot_nts": bool(plot_nts),
            "continuous_y": bool(continuous_y),
            "original_treatment_levels": [_normalize_scalar(value) for value in treatment_levels],
            "original_mediator_levels": [_normalize_scalar(value) for value in mediator_levels],
            "normalized_treatment_levels": [0, 1],
            "normalized_mediator_levels": [0, 1],
            **_partial_density_target_diagnostics(
                plot_nts=plot_nts,
                treatment_levels=treatment_levels,
                mediator_levels=mediator_levels,
            ),
            "treatment_levels": [_normalize_scalar(value) for value in treatment_levels],
            "mediator_levels": [_normalize_scalar(value) for value in mediator_levels],
            "reg_formula": None,
            "adjusted": False,
            "treated_m1_count": treated_m1_count,
            "untreated_m1_count": untreated_m1_count,
            "target_partial11_mass": float(theta_denominator),
            "target_partial01_mass": float(frac_ats),
            **positive_part_diagnostics,
            "data_contract_only": True,
        },
    )


def partial_density_plot(
    *,
    df: pd.DataFrame | None = None,
    data_path: str | Path | None = None,
    d: str,
    m: str,
    y: str,
    num_grid_points: int = 10000,
    plot_nts: bool = False,
    density_1_label: str = "f(Y,M=1|D=1)",
    density_0_label: str = "f(Y,M=1|D=0)",
    num_y_bins: int | None = None,
    reg_formula: str | None = None,
    continuous_y: bool = False,
    caption: str | None = None,
) -> Any:
    """Render partial-density records as a Matplotlib figure.

    This is the Python equivalent of the R function ``partial_density_plot``.
    It validates display labels and captions, computes partial-density data
    via ``partial_density_data()``, renders discrete bar charts or continuous
    line plots, and attaches strict-JSON copies of the consumed data contract
    and render metadata to the returned ``Figure``.

    Parameters
    ----------
    df : pd.DataFrame or None
        Analysis dataframe.  Exactly one of *df* or *data_path* must be given.
    data_path : str, Path, or None
        Path to a CSV file.
    d : str
        Binary treatment column name.
    m : str
        Binary mediator column name.
    y : str
        Outcome column name.
    num_grid_points : int, default 10000
        Grid points for continuous-Y kernel density.
    plot_nts : bool, default False
        Target never-takers instead of always-takers.
    density_1_label : str, default 'f(Y,M=1|D=1)'
        Legend label for the D=1 partial density.
    density_0_label : str, default 'f(Y,M=1|D=0)'
        Legend label for the D=0 partial density.
    num_y_bins : int or None, optional
        Discretize Y into this many bins (discrete mode only).
    reg_formula : str or None, optional
        Regression formula for observational adjustment.
    continuous_y : bool, default False
        Treat Y as continuous.
    caption : str or None, optional
        Figure caption text.

    Returns
    -------
    matplotlib.figure.Figure
        A Matplotlib figure with the partial-density visualization.  The
        figure carries ``testmechs_partial_density_contract`` and
        ``testmechs_partial_density_render_metadata`` attributes.

    Raises
    ------
    ValueError
        If labels are blank or identical, or if underlying data validation
        fails.
    ModuleNotFoundError
        If Matplotlib is not installed.

    Examples
    --------
    >>> import pandas as pd
    >>> from testmechs.partial_density import partial_density_plot
    >>> df = pd.DataFrame({
    ...     "D": [0,0,0,1,1,1], "M": [0,1,0,1,1,1], "Y": [1,2,3,1,2,3]
    ... })
    >>> fig = partial_density_plot(df=df, d="D", m="M", y="Y")  # doctest: +SKIP

    Notes
    -----
    Positive-part regions (where partial11 > partial01) are emphasized:
    in discrete mode via thicker bar edges, in continuous mode via shaded
    fill-between regions.

    See Also
    --------
    partial_density_data : The data-computation layer.
    """

    density_1_label = _validate_partial_density_plot_label(
        density_1_label, name="density_1_label"
    )
    density_0_label = _validate_partial_density_plot_label(
        density_0_label, name="density_0_label"
    )
    _validate_partial_density_plot_label_pair(
        density_1_label=density_1_label,
        density_0_label=density_0_label,
    )
    caption = _validate_partial_density_plot_caption(caption)
    _import_matplotlib_pyplot()
    contract = partial_density_data(
        df=df,
        data_path=data_path,
        d=d,
        m=m,
        y=y,
        num_y_bins=num_y_bins,
        plot_nts=plot_nts,
        continuous_y=continuous_y,
        num_grid_points=num_grid_points,
        reg_formula=reg_formula,
    )
    density_1_label, density_0_label = _partial_density_labels(
        plot_nts=plot_nts,
        density_1_label=density_1_label,
        density_0_label=density_0_label,
        diagnostics=contract.diagnostics,
    )
    if continuous_y:
        return _plot_continuous_partial_density(
            contract=contract,
            density_1_label=density_1_label,
            density_0_label=density_0_label,
            caption=caption,
        )
    return _plot_discrete_partial_density(
        contract=contract,
        density_1_label=density_1_label,
        density_0_label=density_0_label,
        caption=caption,
    )


def _adjusted_discrete_partial_density_data(
    *,
    normalized: pd.DataFrame,
    d: str,
    m: str,
    y: str,
    requested_num_y_bins: int | None,
    applied_num_y_bins: int | None,
    plot_nts: bool,
    treatment_levels: list[object],
    mediator_levels: list[object],
    reg_formula: str,
    source_n_obs: int,
) -> PartialDensityDataResult:
    """Regression-adjusted discrete partial-PMF data."""
    adjusted = compute_adjusted_probabilities(
        df=normalized,
        d=d,
        m=m,
        y=y,
        reg_formula=reg_formula,
    )
    _require_valid_adjusted_probability_grid_for_partial_density(adjusted.diagnostics)
    m_one_value = max(adjusted.m_values)  # type: ignore[type-var]
    p_m_0 = float(adjusted.p_m_d0.get(m_one_value, 0.0))
    p_m_1 = float(adjusted.p_m_d1.get(m_one_value, 0.0))
    frac_compliers = p_m_1 - p_m_0
    frac_ats = p_m_0
    theta_ats = float(frac_ats / p_m_1) if p_m_1 != 0 else math.nan

    records: list[dict[str, float | str]] = []
    y_values = list(adjusted.y_values)
    for y_value in y_values:
        records.append(
            {
                "y": str(y_value),
                "partial11": float(adjusted.p_ym_d1.get((y_value, m_one_value), 0.0)),
                "partial01": float(adjusted.p_ym_d0.get((y_value, m_one_value), 0.0)),
            }
        )
    positive_part_diagnostics = _discrete_positive_part_diagnostics(records)

    return PartialDensityDataResult(
        frac_compliers=frac_compliers,
        frac_ats=frac_ats,
        theta_ats=theta_ats,
        y_levels=[str(level) for level in y_values],
        partial_mass_records=records,
        diagnostics={
            "outcome_column": y,
            "requested_num_y_bins": requested_num_y_bins,
            "applied_num_y_bins": int(len(y_values)),
            "auto_num_y_bins": applied_num_y_bins if requested_num_y_bins is None else None,
            "plot_nts": bool(plot_nts),
            "continuous_y": False,
            "original_treatment_levels": [_normalize_scalar(value) for value in treatment_levels],
            "original_mediator_levels": [_normalize_scalar(value) for value in mediator_levels],
            "normalized_treatment_levels": [0, 1],
            "normalized_mediator_levels": [0, 1],
            **_partial_density_target_diagnostics(
                plot_nts=plot_nts,
                treatment_levels=treatment_levels,
                mediator_levels=mediator_levels,
            ),
            "treatment_levels": [_normalize_scalar(value) for value in treatment_levels],
            "mediator_levels": [_normalize_scalar(value) for value in mediator_levels],
            "reg_formula": reg_formula,
            "adjusted": True,
            "regression": _adjusted_regression_diagnostics(
                adjusted.diagnostics,
                treatment_levels=treatment_levels,
                mediator_levels=mediator_levels,
            ),
            "adjusted_complete_case_source_n_obs": int(source_n_obs),
            "adjusted_complete_case_dropped_rows": int(source_n_obs - len(normalized)),
            "target_partial11_mass": float(p_m_1),
            "target_partial01_mass": float(p_m_0),
            "adjusted_probability_grid_points": int(len(y_values)),
            **positive_part_diagnostics,
            "data_contract_only": True,
        },
    )


def _validate_partial_density_plot_label(value: object, *, name: str) -> str:
    """Validate a density label as a non-blank string."""
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(
            f"{name} must be a non-blank string for partial_density_plot()."
        )
    return value.strip()


def _validate_partial_density_plot_label_pair(
    *,
    density_1_label: str,
    density_0_label: str,
) -> None:
    """Raise if the two density labels are identical after trimming."""
    if density_1_label.strip() == density_0_label.strip():
        raise ValueError(
            "density_1_label and density_0_label must be distinct after trimming "
            "whitespace for partial_density_plot()."
        )


def _validate_partial_density_plot_caption(value: object) -> str | None:
    """Validate and normalize an optional figure caption."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("caption must be None or a non-blank string for partial_density_plot().")
    caption = " ".join(value.split())
    if caption == "":
        raise ValueError("caption must be None or a non-blank string for partial_density_plot().")
    return caption


def _partial_density_labels(
    *,
    plot_nts: bool,
    density_1_label: str,
    density_0_label: str,
    diagnostics: dict[str, Any],
) -> tuple[str, str]:
    """Resolve final legend labels accounting for plot_nts orientation."""
    if not plot_nts:
        if density_1_label == "f(Y,M=1|D=1)" and density_0_label == "f(Y,M=1|D=0)":
            return _partial_density_default_labels(diagnostics)
        return density_1_label, density_0_label
    if density_1_label == "f(Y,M=1|D=1)" and density_0_label == "f(Y,M=1|D=0)":
        return _partial_density_default_labels(diagnostics)
    return density_0_label, density_1_label


def _partial_density_default_labels(diagnostics: dict[str, Any]) -> tuple[str, str]:
    """Generate default legend labels from diagnostics metadata."""
    mediator_level = _format_label_value(diagnostics["target_original_mediator_level"])
    partial11_d = _format_label_value(diagnostics["partial11_original_treatment_level"])
    partial01_d = _format_label_value(diagnostics["partial01_original_treatment_level"])
    return f"f(Y,M={mediator_level}|D={partial11_d})", f"f(Y,M={mediator_level}|D={partial01_d})"


def _discrete_positive_part_diagnostics(
    records: list[dict[str, float | str]],
) -> dict[str, Any]:
    """Compute positive-part diagnostics for a discrete partial-PMF."""
    rows: list[dict[str, Any]] = []
    positive_part_sum = 0.0
    partial11_mass = 0.0
    for record in records:
        partial11 = float(record["partial11"])
        partial01 = float(record["partial01"])
        delta = partial11 - partial01
        contribution = max(delta, 0.0)
        positive_part_sum += contribution
        partial11_mass += partial11
        rows.append(
            {
                "y": record["y"],
                "partial11": partial11,
                "partial01": partial01,
                "delta": delta,
                "positive_part_contribution": contribution,
            }
        )
    return {
        "positive_part_partial_pmf_diff": float(positive_part_sum),
        "positive_part_partial11_mass_gap": float(partial11_mass - positive_part_sum),
        "positive_part_cell_rows": rows,
        "positive_part_integral_rule": "discrete_sum_over_y_levels",
        "positive_part_support_rule": "displayed_y_levels_with_zero_contribution_rows_allowed",
    }


def _format_label_value(value: object) -> str:
    """Format a support-level value for display in labels."""
    if isinstance(value, (bool, np.bool_)):
        return "1" if bool(value) else "0"
    return str(value)


def _format_column_display_label(value: object) -> str:
    """Format a column name for axis display."""
    return str(value).replace("_", " ")


def _format_title_label_value(value: object, *, max_chars: int = 28) -> str:
    """Format a value for plot title, truncating if needed."""
    text = _format_label_value(value)
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


def _format_legend_label(
    value: str,
    *,
    width: int = 34,
    max_lines: int | None = None,
) -> str:
    """Wrap a legend label to fit within a character width."""
    if len(value) <= width:
        return value
    wrapped = textwrap.wrap(
        value,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if not wrapped or max(len(line) for line in wrapped) > width:
        wrapped = textwrap.wrap(
            value,
            width=width,
            break_long_words=True,
            break_on_hyphens=False,
        )
    if max_lines is None or len(wrapped) <= max_lines:
        return "\n".join(wrapped)
    if max_lines < 2:
        raise ValueError("max_lines must be at least 2 when provided.")
    head_count = max_lines - 1
    tail_budget = max(4, width - 3)
    return "\n".join(
        [
            *wrapped[:head_count],
            f"...{wrapped[-1][-tail_budget:]}",
        ]
    )


def _format_axis_tick_label(
    value: object,
    *,
    width: int = 14,
    max_lines: int | None = None,
) -> str:
    """Format a tick label, wrapping or truncating long text."""
    text = _format_label_value(value)
    if len(text) <= width:
        return text
    wrapped = textwrap.wrap(
        text,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if not wrapped or max(len(line) for line in wrapped) > width:
        wrapped = textwrap.wrap(
            text,
            width=width,
            break_long_words=True,
            break_on_hyphens=False,
        )
    if max_lines is None or len(wrapped) <= max_lines:
        return "\n".join(wrapped)
    if max_lines < 2:
        raise ValueError("max_lines must be at least 2 when provided.")
    head_count = max_lines - 1
    tail_budget = max(4, width - 3)
    return "\n".join(
        [
            *wrapped[:head_count],
            f"...{wrapped[-1][-tail_budget:]}",
        ]
    )


def _partial_density_outcome_axis_label(contract: PartialDensityDataResult) -> str:
    """Build the x-axis label for the partial-density figure."""
    outcome_column = str(contract.diagnostics.get("outcome_column", "Y"))
    if outcome_column == "Y":
        return "Outcome (Y)"
    label = _format_axis_tick_label(
        _format_column_display_label(outcome_column),
        width=42,
        max_lines=2,
    )
    return f"Outcome: {label}"


def _plot_discrete_partial_density(
    *,
    contract: PartialDensityDataResult,
    density_1_label: str,
    density_0_label: str,
    caption: str | None = None,
) -> Any:
    """Render a discrete partial-PMF bar chart."""
    plt = _import_matplotlib_pyplot()

    y_levels = list(contract.y_levels)
    partial11_by_y = {
        record["y"]: _validate_plotted_partial_density_value(
            record["partial11"], role="partial11", y=record["y"]
        )
        for record in contract.partial_mass_records
    }
    partial01_by_y = {
        record["y"]: _validate_plotted_partial_density_value(
            record["partial01"], role="partial01", y=record["y"]
        )
        for record in contract.partial_mass_records
    }

    positions = list(range(len(y_levels)))
    width = 0.35
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    partial11_bars = axis.bar(
        [position - width / 2 for position in positions],
        [partial11_by_y[level] for level in y_levels],
        width=width,
        label=_format_legend_label(density_1_label, max_lines=4),
        color=_PARTIAL_DENSITY_COLORS[0],
        edgecolor="white",
        linewidth=0.8,
    )
    partial01_bars = axis.bar(
        [position + width / 2 for position in positions],
        [partial01_by_y[level] for level in y_levels],
        width=width,
        label=_format_legend_label(density_0_label, max_lines=4),
        color=_PARTIAL_DENSITY_COLORS[1],
        edgecolor="white",
        linewidth=0.8,
    )
    _emphasize_discrete_positive_part_bars(
        bars=partial11_bars,
        y_levels=y_levels,
        partial11_by_y=partial11_by_y,  # type: ignore[arg-type]
        partial01_by_y=partial01_by_y,  # type: ignore[arg-type]
    )
    if len(y_levels) <= _DISCRETE_BAR_LABEL_MAX_LEVELS:
        _label_discrete_partial_density_bars(axis, partial11_bars)
        _label_discrete_partial_density_bars(axis, partial01_bars)
    max_height = max(
        [partial11_by_y[level] for level in y_levels]
        + [partial01_by_y[level] for level in y_levels],
        default=0.0,
    )
    label_headroom = 0.0
    if len(y_levels) <= _DISCRETE_BAR_LABEL_MAX_LEVELS and max_height > 0.0:
        label_headroom = max(0.01, max_height * 0.02) + max(0.01, max_height * 0.08)
    axis.set_ylim(top=max(0.05, max_height * 1.18, max_height + label_headroom))
    axis.set_xlabel(_partial_density_outcome_axis_label(contract))
    axis.set_ylabel("Partial PMF")
    axis.set_xticks(positions)
    dense_tick_max_lines = 4 if len(y_levels) > 6 else None
    displayed_y_levels = [
        _format_axis_tick_label(level, max_lines=dense_tick_max_lines)
        for level in y_levels
    ]
    axis.set_xticklabels(displayed_y_levels)
    has_wrapped_tick_labels = any("\n" in label for label in displayed_y_levels)
    if len(y_levels) > 6 and not has_wrapped_tick_labels:
        axis.tick_params(axis="x", labelrotation=30)
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")
    has_caption = caption is not None
    _set_partial_density_plot_title(axis, contract)
    legend_y_anchor = (
        -0.82
        if len(y_levels) > 6 and has_wrapped_tick_labels and has_caption
        else -0.76
        if len(y_levels) > 6 and has_wrapped_tick_labels
        else -0.31
        if len(y_levels) > 6
        else -0.18
    )
    _style_partial_density_axis(
        axis,
        legend_title="Partial density",
        legend_y_anchor=legend_y_anchor,
    )
    annotation_metadata = _annotate_partial_density_positive_part(axis, contract)
    rendered_caption = _add_partial_density_caption(figure, caption)
    has_caption = rendered_caption is not None
    if len(y_levels) > 6 and has_wrapped_tick_labels:
        layout_branch = "dense_wrapped_ticks_caption" if has_caption else "dense_wrapped_ticks"
        figure.subplots_adjust(bottom=0.62 if has_caption else 0.54)
    elif has_caption:
        layout_branch = "caption_margin"
        figure.subplots_adjust(bottom=0.26)
    else:
        layout_branch = "tight_layout"
        figure.tight_layout()
    _attach_partial_density_plot_metadata(
        figure,
        contract,
        axis=axis,
        density_1_label=density_1_label,
        density_0_label=density_0_label,
        caption=rendered_caption,
        legend_y_anchor=legend_y_anchor,
        has_caption=has_caption,
        layout_branch=layout_branch,
        positive_part_annotation=annotation_metadata,
    )
    return figure


def _emphasize_discrete_positive_part_bars(
    *,
    bars: Any,
    y_levels: list[str],
    partial11_by_y: dict[str, float],
    partial01_by_y: dict[str, float],
) -> None:
    """Thicken bar edges where partial11 exceeds partial01."""
    for level, bar in zip(y_levels, bars, strict=True):
        if partial11_by_y[level] > partial01_by_y[level]:
            bar.set_edgecolor("#333333")
            bar.set_linewidth(1.4)


def _label_discrete_partial_density_bars(axis: Any, bars: Any) -> None:
    """Add numeric labels above each non-zero bar."""
    for bar in bars:
        height = float(bar.get_height())
        if height == 0.0:
            continue
        label_y = height + max(0.01, height * 0.02)
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f"{height:.3g}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )


def _continuous_partial_density_data(
    *,
    normalized: pd.DataFrame,
    d: str,
    m: str,
    y: str,
    num_grid_points: int,
    plot_nts: bool,
    treatment_levels: list[object],
    mediator_levels: list[object],
) -> PartialDensityDataResult:
    """Compute unadjusted continuous partial-density records."""
    yvec = pd.to_numeric(normalized[y], errors="raise").astype(float)
    dvec = normalized[d]
    mvec = normalized[m]
    frac_compliers = float(mvec[dvec == 1].mean() - mvec[dvec == 0].mean())
    frac_ats = float(mvec[dvec == 0].mean())
    theta_denominator = frac_compliers + frac_ats
    theta_ats = float(frac_ats / theta_denominator) if theta_denominator != 0 else math.nan

    treated_y = yvec[(dvec == 1) & (mvec == 1)].to_numpy(dtype=float)
    untreated_y = yvec[(dvec == 0) & (mvec == 1)].to_numpy(dtype=float)
    target_partial11_mass = frac_ats + frac_compliers
    target_partial01_mass = frac_ats
    if (target_partial11_mass > 0 and treated_y.size < 2) or (target_partial01_mass > 0 and untreated_y.size < 2):
        raise ValueError(
            "Continuous partial-density estimation requires at least two M=1 observations in each positive-mass arm."
        )

    y_grid = _continuous_partial_density_grid(yvec.to_numpy(dtype=float), num_grid_points)
    output_grid_points = int(y_grid.size)

    treated_density, treated_bandwidth = _density_or_zero_for_partial_mass(
        values=treated_y,
        grid=y_grid,
        target_mass=target_partial11_mass,
    )
    untreated_density, untreated_bandwidth = _density_or_zero_for_partial_mass(
        values=untreated_y,
        grid=y_grid,
        target_mass=target_partial01_mass,
    )
    partial11 = target_partial11_mass * treated_density
    partial01 = target_partial01_mass * untreated_density
    partial11_integral = _trapezoid_integral(partial11, y_grid)
    partial01_integral = _trapezoid_integral(partial01, y_grid)
    positive_part_integral = _trapezoid_integral(
        np.maximum(partial11 - partial01, 0.0),
        y_grid,
    )
    positive_part_partial11_integral_gap = partial11_integral - positive_part_integral

    density_records = [
        {
            "y": float(y_value),
            "partial11": float(partial11_value),
            "partial01": float(partial01_value),
        }
        for y_value, partial11_value, partial01_value in zip(y_grid, partial11, partial01, strict=True)
    ]
    return PartialDensityDataResult(
        frac_compliers=frac_compliers,
        frac_ats=frac_ats,
        theta_ats=theta_ats,
        y_levels=[],
        partial_mass_records=[],
        partial_density_records=density_records,  # type: ignore[arg-type]
        diagnostics={
            "outcome_column": y,
            "requested_num_y_bins": None,
            "applied_num_y_bins": None,
            "auto_num_y_bins": None,
            "plot_nts": bool(plot_nts),
            "continuous_y": True,
            "num_grid_points": int(num_grid_points),
            "output_grid_points": output_grid_points,
            "grid_min": float(y_grid[0]),
            "grid_max": float(y_grid[-1]),
            "treated_kernel_bandwidth": treated_bandwidth,
            "untreated_kernel_bandwidth": untreated_bandwidth,
            "treated_m1_count": int(treated_y.size),
            "untreated_m1_count": int(untreated_y.size),
            "target_partial11_mass": float(target_partial11_mass),
            "target_partial01_mass": float(target_partial01_mass),
            "partial11_integral": partial11_integral,
            "partial01_integral": partial01_integral,
            "partial11_integral_absolute_error": abs(partial11_integral - target_partial11_mass),
            "partial01_integral_absolute_error": abs(partial01_integral - target_partial01_mass),
            "positive_part_partial_density_integral": positive_part_integral,
            "positive_part_partial11_integral_gap": positive_part_partial11_integral_gap,
            "positive_part_integral_rule": "trapezoid_on_output_grid",
            "partial_density_integral_rule": "trapezoid_on_output_grid",
            "original_treatment_levels": [_normalize_scalar(value) for value in treatment_levels],
            "original_mediator_levels": [_normalize_scalar(value) for value in mediator_levels],
            "normalized_treatment_levels": [0, 1],
            "normalized_mediator_levels": [0, 1],
            **_partial_density_target_diagnostics(
                plot_nts=plot_nts,
                treatment_levels=treatment_levels,
                mediator_levels=mediator_levels,
            ),
            "treatment_levels": [_normalize_scalar(value) for value in treatment_levels],
            "mediator_levels": [_normalize_scalar(value) for value in mediator_levels],
            "reg_formula": None,
            "adjusted": False,
            "data_contract_only": True,
        },
    )


def _adjusted_continuous_partial_density_data(
    *,
    normalized: pd.DataFrame,
    d: str,
    m: str,
    y: str,
    num_grid_points: int,
    plot_nts: bool,
    treatment_levels: list[object],
    mediator_levels: list[object],
    reg_formula: str,
    source_n_obs: int,
) -> PartialDensityDataResult:
    """Regression-adjusted continuous partial-density records."""
    adjusted = compute_adjusted_mediator_masses(
        df=normalized,
        d=d,
        m=m,
        reg_formula=reg_formula,
    )
    _require_valid_adjusted_mediator_mass_grid_for_partial_density(adjusted.diagnostics)
    m_one_value = max(adjusted.m_values)  # type: ignore[type-var]
    target_partial01_mass = float(adjusted.p_m_d0.get(m_one_value, 0.0))
    target_partial11_mass = float(adjusted.p_m_d1.get(m_one_value, 0.0))
    frac_compliers = target_partial11_mass - target_partial01_mass
    frac_ats = target_partial01_mass
    theta_ats = float(frac_ats / target_partial11_mass) if target_partial11_mass != 0 else math.nan

    regression_spec = parse_reg_formula(reg_formula, d=d)
    shape_columns = list(dict.fromkeys([d, m, y, *regression_spec.variables]))
    shape_data = normalized.dropna(subset=shape_columns).copy()
    yvec = pd.to_numeric(shape_data[y], errors="raise").astype(float)
    dvec = shape_data[d]
    mvec = shape_data[m]
    treated_y = yvec[(dvec == 1) & (mvec == 1)].to_numpy(dtype=float)
    untreated_y = yvec[(dvec == 0) & (mvec == 1)].to_numpy(dtype=float)
    if (target_partial11_mass > 0 and treated_y.size < 2) or (
        target_partial01_mass > 0 and untreated_y.size < 2
    ):
        raise ValueError(
            "Continuous partial-density estimation requires at least two M=1 observations in each positive-mass arm."
        )

    y_grid = _continuous_partial_density_grid(yvec.to_numpy(dtype=float), num_grid_points)
    output_grid_points = int(y_grid.size)

    treated_density, treated_bandwidth = _density_or_zero_for_partial_mass(
        values=treated_y,
        grid=y_grid,
        target_mass=target_partial11_mass,
    )
    untreated_density, untreated_bandwidth = _density_or_zero_for_partial_mass(
        values=untreated_y,
        grid=y_grid,
        target_mass=target_partial01_mass,
    )
    partial11 = target_partial11_mass * treated_density
    partial01 = target_partial01_mass * untreated_density
    partial11_integral = _trapezoid_integral(partial11, y_grid)
    partial01_integral = _trapezoid_integral(partial01, y_grid)
    positive_part_integral = _trapezoid_integral(
        np.maximum(partial11 - partial01, 0.0),
        y_grid,
    )
    positive_part_partial11_integral_gap = partial11_integral - positive_part_integral
    density_records = [
        {
            "y": float(y_value),
            "partial11": float(partial11_value),
            "partial01": float(partial01_value),
        }
        for y_value, partial11_value, partial01_value in zip(y_grid, partial11, partial01, strict=True)
    ]

    return PartialDensityDataResult(
        frac_compliers=frac_compliers,
        frac_ats=frac_ats,
        theta_ats=theta_ats,
        y_levels=[],
        partial_mass_records=[],
        partial_density_records=density_records,  # type: ignore[arg-type]
        diagnostics={
            "outcome_column": y,
            "requested_num_y_bins": None,
            "applied_num_y_bins": None,
            "auto_num_y_bins": None,
            "plot_nts": bool(plot_nts),
            "continuous_y": True,
            "num_grid_points": int(num_grid_points),
            "output_grid_points": output_grid_points,
            "grid_min": float(y_grid[0]),
            "grid_max": float(y_grid[-1]),
            "treated_kernel_bandwidth": treated_bandwidth,
            "untreated_kernel_bandwidth": untreated_bandwidth,
            "treated_m1_count": int(treated_y.size),
            "untreated_m1_count": int(untreated_y.size),
            "target_partial11_mass": float(target_partial11_mass),
            "target_partial01_mass": float(target_partial01_mass),
            "partial11_integral": partial11_integral,
            "partial01_integral": partial01_integral,
            "partial11_integral_absolute_error": abs(partial11_integral - target_partial11_mass),
            "partial01_integral_absolute_error": abs(partial01_integral - target_partial01_mass),
            "positive_part_partial_density_integral": positive_part_integral,
            "positive_part_partial11_integral_gap": positive_part_partial11_integral_gap,
            "positive_part_integral_rule": "trapezoid_on_output_grid",
            "partial_density_integral_rule": "trapezoid_on_output_grid",
            "original_treatment_levels": [_normalize_scalar(value) for value in treatment_levels],
            "original_mediator_levels": [_normalize_scalar(value) for value in mediator_levels],
            "normalized_treatment_levels": [0, 1],
            "normalized_mediator_levels": [0, 1],
            **_partial_density_target_diagnostics(
                plot_nts=plot_nts,
                treatment_levels=treatment_levels,
                mediator_levels=mediator_levels,
            ),
            "treatment_levels": [_normalize_scalar(value) for value in treatment_levels],
            "mediator_levels": [_normalize_scalar(value) for value in mediator_levels],
            "reg_formula": reg_formula,
            "adjusted": True,
            "regression": _adjusted_regression_diagnostics(
                adjusted.diagnostics,
                treatment_levels=treatment_levels,
                mediator_levels=mediator_levels,
            ),
            "adjusted_complete_case_source_n_obs": int(source_n_obs),
            "adjusted_complete_case_dropped_rows": int(source_n_obs - len(normalized)),
            "continuous_density_shape_contract": (
                "regression_complete_case_observed_y_given_d_m1_kernel_scaled_to_adjusted_mediator_mass"
            ),
            "continuous_density_shape_n_obs": int(len(shape_data)),
            "continuous_density_shape_columns": list(shape_columns),
            "continuous_density_shape_dropped_rows": int(source_n_obs - len(shape_data)),
            "data_contract_only": True,
        },
    )


def _adjusted_regression_diagnostics(
    diagnostics: dict[str, Any],
    *,
    treatment_levels: list[object],
    mediator_levels: list[object],
) -> dict[str, Any]:
    """Augment regression diagnostics with treatment/mediator level info."""
    regression_diagnostics = dict(diagnostics)
    regression_diagnostics.update(
        {
            "original_treatment_levels": [_normalize_scalar(value) for value in treatment_levels],
            "normalized_treatment_levels": [0, 1],
            "original_mediator_levels": [_normalize_scalar(value) for value in mediator_levels],
            "normalized_mediator_levels": [0, 1],
        }
    )
    return regression_diagnostics


def _require_valid_adjusted_probability_grid_for_partial_density(
    diagnostics: dict[str, Any],
) -> None:
    """Raise if the adjusted grid is invalid for partial-density use."""
    grid_contract = diagnostics.get("probability_grid_contract")
    if not isinstance(grid_contract, dict) or not grid_contract.get("valid_for_bounds", False):
        raise ValueError(
            "Adjusted partial-density data requires a finite nonnegative joint "
            "probability grid with probabilities no larger than 1 and valid "
            "mediator-mass totals."
        )


def _require_valid_adjusted_mediator_mass_grid_for_partial_density(
    diagnostics: dict[str, Any],
) -> None:
    """Raise if adjusted mediator masses are invalid for partial-density."""
    mass_contract = diagnostics.get("mediator_mass_grid_contract")
    if not isinstance(mass_contract, dict) or not mass_contract.get(
        "valid_for_partial_density",
        False,
    ):
        raise ValueError(
            "Adjusted continuous partial-density data requires finite nonnegative "
            "mediator masses no larger than 1 that sum to 1 in each treatment arm."
        )


def _reject_partial_density_formula_role_reuse(
    *,
    reg_formula: str,
    d: str,
    m: str,
    y: str,
) -> None:
    """Raise if formula variables reuse mediator or outcome columns."""
    spec = parse_reg_formula(reg_formula, d=d)
    reserved_columns = {m, y}
    if any(variable in reserved_columns for variable in spec.variables if variable != spec.treatment):
        raise ValueError("reg_formula variables must not reuse outcome or mediator columns.")


def _conditional_indicator_mean(values: pd.Series) -> float:
    """Mean of a boolean series, returning 0.0 for empty input."""
    if len(values) == 0:
        return 0.0
    return float(values.mean())


def _validate_bool_flag(value: object, *, name: str) -> bool:
    """Validate *value* as a strict boolean."""
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean.")
    return value


def _validate_num_grid_points(value: object) -> int:
    """Validate *value* as an integer greater than 1."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("num_grid_points must be an integer greater than 1.")
    value = int(value)
    if value <= 1:
        raise ValueError("num_grid_points must be an integer greater than 1.")
    return value


def _reject_nonfinite_continuous_partial_density_y(
    series: pd.Series,
    *,
    column: str,
) -> None:
    """Raise if continuous Y contains non-finite values."""
    values = pd.to_numeric(series, errors="raise")
    if not bool(np.isfinite(values.to_numpy(dtype=float)).all()):
        raise ValueError(
            f"{column} must contain only finite numeric values for continuous partial-density estimation."
        )


def _reject_nonfinite_partial_density_y(
    series: pd.Series,
    *,
    column: str,
) -> None:
    """Raise if Y support contains non-finite numeric values."""
    for value in pd.unique(series.dropna()):
        normalized = _normalize_scalar(value)
        if isinstance(normalized, bool):
            continue
        if isinstance(normalized, (int, float, np.integer, np.floating)) and not math.isfinite(float(normalized)):
            raise ValueError(f"{column} must contain only finite numeric values for partial-density data.")


def _discrete_partial_density_y_support_is_numeric(series: pd.Series) -> bool:
    """Check whether the Y support is numeric (non-boolean)."""
    for value in pd.unique(series.dropna()):
        normalized = _normalize_scalar(value)
        if isinstance(normalized, (bool, np.bool_)):
            return False
    try:
        pd.to_numeric(series.dropna(), errors="raise")
    except (TypeError, ValueError):
        return False
    return True


def _validate_scalar_column_name(value: object, *, name: str) -> str:
    """Validate *value* as a non-empty column name string."""
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{name} must name one scalar DataFrame column.")
    return value


def _validate_scalar_mediator_column(m: object) -> str:
    """Validate *m* as a scalar binary mediator column name."""
    if not isinstance(m, str):
        raise ValueError(
            "partial-density data and plotting require m to name one scalar binary mediator column."
        )
    if m == "":
        raise ValueError(
            "partial-density data and plotting require m to name one scalar binary mediator column."
        )
    return m


def _partial_density_target_diagnostics(
    *,
    plot_nts: bool,
    treatment_levels: list[object],
    mediator_levels: list[object],
) -> dict[str, object]:
    """Build target-role diagnostics for partial-density metadata."""
    target_mediator_index = 0 if plot_nts else 1
    partial11_treatment_index = 0 if plot_nts else 1
    partial01_treatment_index = 1 if plot_nts else 0
    diagnostics = {
        "target_original_mediator_level": _normalize_scalar(
            mediator_levels[target_mediator_index]
        ),
        "target_normalized_mediator_level": target_mediator_index,
        "partial11_original_treatment_level": _normalize_scalar(
            treatment_levels[partial11_treatment_index]
        ),
        "partial01_original_treatment_level": _normalize_scalar(
            treatment_levels[partial01_treatment_index]
        ),
        "partial_density_target_role": "never_takers" if plot_nts else "always_takers",
        "partial_density_orientation": (
            "plot_nts flips treatment and mediator internally; partial11 is original D=0,M=0 "
            "and partial01 is original D=1,M=0"
            if plot_nts
            else "partial11 is original D=1,M=1 and partial01 is original D=0,M=1"
        ),
    }
    partial11_label, partial01_label = _partial_density_default_labels(diagnostics)
    return {
        **diagnostics,
        "partial11_default_label": partial11_label,
        "partial01_default_label": partial01_label,
        "partial_density_default_labels": [partial11_label, partial01_label],
    }


def _density_or_zero_for_partial_mass(
    *,
    values: np.ndarray,
    grid: np.ndarray,
    target_mass: float,
) -> tuple[np.ndarray, float | None]:
    """Return kernel density or zeros if target mass is zero."""
    if target_mass == 0:
        return np.zeros_like(grid, dtype=float), None
    return _gaussian_kernel_density_on_grid(values, grid)


def _continuous_partial_density_grid(values: np.ndarray, num_grid_points: int) -> np.ndarray:
    """Construct a finite, strictly-increasing evaluation grid."""
    values = np.asarray(values, dtype=float)
    spread = _finite_scaled_sample_spread(values)
    lower = _finite_grid_endpoint(float(np.min(values)), -spread)
    upper = _finite_grid_endpoint(float(np.max(values)), spread)
    if lower >= upper:
        center = float(np.mean(values / _finite_value_scale(values))) * _finite_value_scale(values)
        lower = _finite_grid_endpoint(center, -spread)
        upper = _finite_grid_endpoint(center, spread)
    scale = _finite_value_scale(np.array([lower, upper], dtype=float))
    grid = np.linspace(lower / scale, upper / scale, num_grid_points) * scale
    return _strictly_increasing_finite_grid(grid)


def _strictly_increasing_finite_grid(grid: np.ndarray) -> np.ndarray:
    """Filter to unique finite grid points; require at least two."""
    finite_grid = np.asarray(grid, dtype=float)
    finite_grid = finite_grid[np.isfinite(finite_grid)]
    unique_grid = np.unique(finite_grid)
    if unique_grid.size < 2:
        raise ValueError(
            "Continuous partial-density output grid must contain at least two finite "
            "representable points."
        )
    return unique_grid


def _finite_grid_endpoint(value: float, offset: float) -> float:
    """Compute a finite grid endpoint with fallback."""
    endpoint = value + offset
    if math.isfinite(endpoint) and endpoint != value:
        return float(endpoint)
    direction = -math.inf if offset < 0.0 else math.inf
    stepped = _finite_nextafter(value, direction)
    if math.isfinite(stepped):
        return stepped
    return float(value)


def _finite_nextafter(value: float, direction: float) -> float:
    """Safe nextafter that respects float limits."""
    float_limits = np.finfo(float)
    if direction > 0.0 and value >= float_limits.max:
        return float(value)
    if direction < 0.0 and value <= -float_limits.max:
        return float(value)
    return float(np.nextafter(value, direction))


def _finite_value_scale(values: np.ndarray) -> float:
    """Return max absolute value for numerical scaling."""
    max_abs = float(np.max(np.abs(values)))
    if not math.isfinite(max_abs) or max_abs <= 0.0:
        return 1.0
    return max_abs


def _finite_scaled_sample_spread(values: np.ndarray) -> float:
    """Robust spread estimator using scaled std or range."""
    scale = _finite_value_scale(values)
    scaled = values / scale
    spread = float(np.std(scaled, ddof=1)) * scale
    if math.isfinite(spread) and spread > 0.0:
        return spread
    span = float(np.max(scaled) - np.min(scaled)) * scale
    if math.isfinite(span) and span > 0.0:
        return span
    return _finite_representable_spacing(values)


def _finite_representable_spacing(values: np.ndarray) -> float:
    """Maximum ULP-based spacing across *values*."""
    max_spacing = 0.0
    for value in values:
        numeric = float(value)
        lower = _finite_nextafter(numeric, -math.inf)
        upper = _finite_nextafter(numeric, math.inf)
        if math.isfinite(lower):
            max_spacing = max(max_spacing, abs(numeric - lower))
        if math.isfinite(upper):
            max_spacing = max(max_spacing, abs(upper - numeric))
    if math.isfinite(max_spacing) and max_spacing > 0.0:
        return max(1.0, max_spacing)
    return 1.0


def _gaussian_kernel_density_on_grid(values: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, float]:
    """Evaluate a Gaussian KDE on *grid* using Silverman bandwidth."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("Kernel density estimation requires at least two observations.")
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1:
        raise ValueError("Kernel density estimation grid must be one-dimensional.")
    spread = _finite_scaled_sample_spread(values)
    bandwidth = 1.06 * spread * (values.size ** (-1.0 / 5.0))
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        bandwidth = 1.0
    bandwidth = max(bandwidth, _finite_grid_resolution(grid))
    density = np.empty(grid.shape, dtype=float)
    normalizer = math.sqrt(2.0 * math.pi)
    scale = _finite_value_scale(np.concatenate([values, grid, np.array([bandwidth], dtype=float)]))
    scaled_values = values / scale
    scaled_grid = grid / scale
    scaled_bandwidth = bandwidth / scale
    rows_per_chunk = max(1, _KDE_MAX_CHUNK_CELLS // values.size)
    for start in range(0, grid.size, rows_per_chunk):
        stop = min(start + rows_per_chunk, grid.size)
        scaled = (scaled_grid[start:stop, None] - scaled_values[None, :]) / scaled_bandwidth
        scaled = np.clip(scaled, -40.0, 40.0)
        density[start:stop] = np.exp(-0.5 * scaled * scaled).mean(axis=1) / bandwidth / normalizer
    return density, float(bandwidth)


def _finite_grid_resolution(grid: np.ndarray) -> float:
    """Minimum positive step in the grid."""
    if grid.size < 2:
        return 0.0
    scale = _finite_value_scale(grid)
    scaled_grid = grid / scale
    diffs = np.diff(scaled_grid) * scale
    positive_diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if positive_diffs.size == 0:
        return 0.0
    resolution = float(np.min(positive_diffs))
    if math.isfinite(resolution) and resolution > 0.0:
        return resolution
    return 0.0


def _trapezoid_integral(values: np.ndarray, grid: np.ndarray) -> float:
    """Trapezoidal numerical integration."""
    return float(scipy_trapezoid(values, grid))


def _plot_continuous_partial_density(
    *,
    contract: PartialDensityDataResult,
    density_1_label: str,
    density_0_label: str,
    caption: str | None = None,
) -> Any:
    """Render a continuous partial-density line plot with positive-part shading."""
    plt = _import_matplotlib_pyplot()

    if contract.partial_density_records is None:
        raise ValueError("Continuous partial-density records are missing.")
    y_values = [
        _validate_plotted_partial_density_axis_value(record["y"])
        for record in contract.partial_density_records
    ]
    partial11 = [
        _validate_plotted_partial_density_value(
            record["partial11"], role="partial11", y=record["y"]
        )
        for record in contract.partial_density_records
    ]
    partial01 = [
        _validate_plotted_partial_density_value(
            record["partial01"], role="partial01", y=record["y"]
        )
        for record in contract.partial_density_records
    ]

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    y_array = np.asarray(y_values, dtype=float)
    partial11_array = np.asarray(partial11, dtype=float)
    partial01_array = np.asarray(partial01, dtype=float)
    positive_part_mask = partial11_array > partial01_array
    if bool(np.any(positive_part_mask)):
        axis.fill_between(
            y_array,
            partial01_array,
            partial11_array,
            where=positive_part_mask,
            interpolate=True,
            color=_PARTIAL_DENSITY_COLORS[0],
            alpha=0.14,
            linewidth=0,
        )
    axis.plot(
        y_values,
        partial11,
        label=_format_legend_label(density_1_label, max_lines=4),
        color=_PARTIAL_DENSITY_COLORS[0],
        linewidth=2.0,
    )
    axis.plot(
        y_values,
        partial01,
        label=_format_legend_label(density_0_label, max_lines=4),
        color=_PARTIAL_DENSITY_COLORS[1],
        linewidth=2.0,
    )
    axis.set_xlabel(_partial_density_outcome_axis_label(contract))
    axis.set_ylabel("Partial Density")
    _set_partial_density_plot_title(axis, contract)
    legend_y_anchor = _continuous_legend_y_anchor(density_1_label, density_0_label)
    _style_partial_density_axis(
        axis,
        legend_title="Partial density",
        legend_y_anchor=legend_y_anchor,
    )
    annotation_metadata = _annotate_partial_density_positive_part(axis, contract)
    rendered_caption = _add_partial_density_caption(figure, caption)
    has_caption = rendered_caption is not None
    if legend_y_anchor < -0.18:
        layout_branch = "long_legend_caption" if has_caption else "long_legend"
        figure.subplots_adjust(bottom=0.48 if has_caption else 0.40)
    elif has_caption:
        layout_branch = "caption_margin"
        figure.subplots_adjust(bottom=0.30)
    else:
        layout_branch = "tight_layout"
        figure.tight_layout()
    _attach_partial_density_plot_metadata(
        figure,
        contract,
        axis=axis,
        density_1_label=density_1_label,
        density_0_label=density_0_label,
        caption=rendered_caption,
        legend_y_anchor=legend_y_anchor,
        has_caption=has_caption,
        layout_branch=layout_branch,
        positive_part_annotation=annotation_metadata,
    )
    return figure


def _attach_partial_density_plot_metadata(
    figure: Any,
    contract: PartialDensityDataResult,
    *,
    axis: Any,
    density_1_label: str,
    density_0_label: str,
    caption: str | None,
    legend_y_anchor: float,
    has_caption: bool,
    layout_branch: str,
    positive_part_annotation: dict[str, Any] | None,
) -> None:
    """Attach contract and render metadata to the figure object."""
    figure.testmechs_partial_density_contract = contract.to_dict()
    legend = axis.get_legend()
    legend_labels = [] if legend is None else [text.get_text() for text in legend.get_texts()]
    legend_title = None if legend is None else legend.get_title().get_text()
    legend_label_line_counts = [_rendered_text_line_count(label) for label in legend_labels]
    legend_labels_truncated = [
        _rendered_text_has_ellipsis_tail(label) for label in legend_labels
    ]
    positive_part_shading = _positive_part_shading_metadata(axis, contract)
    positive_part_bar_emphasis = _positive_part_bar_emphasis_metadata(axis, contract)
    figure.testmechs_partial_density_render_metadata = _json_safe_payload(
        {
            "figure_kind": "partial_density_plot",
            "density_1_label": density_1_label,
            "density_0_label": density_0_label,
            "legend_labels": legend_labels,
            "legend_label_line_counts": legend_label_line_counts,
            "legend_labels_truncated": legend_labels_truncated,
            "legend_title": legend_title,
            "legend_y_anchor": float(legend_y_anchor),
            "caption": caption,
            "caption_line_count": None
            if caption is None
            else _rendered_text_line_count(caption),
            "caption_truncated": None
            if caption is None
            else _rendered_text_has_ellipsis_tail(caption),
            "has_caption": bool(has_caption),
            "layout_branch": layout_branch,
            "positive_part_annotation": positive_part_annotation,
            "positive_part_shading": positive_part_shading,
            "positive_part_bar_emphasis": positive_part_bar_emphasis,
            "layout_clearance": _partial_density_render_layout_clearance(
                figure,
                axis,
            ),
            "subplot_margins": {
                "left": float(figure.subplotpars.left),
                "right": float(figure.subplotpars.right),
                "bottom": float(figure.subplotpars.bottom),
                "top": float(figure.subplotpars.top),
            },
            "figure_size_inches": [float(value) for value in figure.get_size_inches()],
            "xlabel": axis.get_xlabel(),
            "ylabel": axis.get_ylabel(),
            "title": axis.get_title(loc="left"),
            "continuous_y": bool(contract.diagnostics.get("continuous_y")),
            "adjusted": bool(contract.diagnostics.get("adjusted")),
            "plot_nts": bool(contract.diagnostics.get("plot_nts")),
            "partial_density_target_role": contract.diagnostics.get(
                "partial_density_target_role"
            ),
            "data_contract_attribute": "testmechs_partial_density_contract",
        }
    )


def _partial_density_render_layout_clearance(figure: Any, axis: Any) -> dict[str, Any]:
    """Compute layout-clearance checks for rendered elements."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    legend = axis.get_legend()
    legend_bbox = None if legend is None else legend.get_window_extent(renderer)
    axis_bbox = axis.get_window_extent(renderer)
    xlabel_bbox = axis.xaxis.label.get_window_extent(renderer)
    title_bbox = _partial_density_axis_title_artist(axis).get_window_extent(renderer)
    caption_bbox = figure.texts[0].get_window_extent(renderer) if figure.texts else None
    positive_part_annotations = [
        text for text in axis.texts if text.get_text().startswith("Positive part")
    ]
    annotation_bbox = (
        positive_part_annotations[0].get_window_extent(renderer)
        if len(positive_part_annotations) == 1
        else None
    )
    return {
        "legend_inside_figure": None
        if legend_bbox is None
        else _bbox_inside_figure(figure, legend_bbox),
        "legend_axis_overlap": None
        if legend_bbox is None
        else bool(legend_bbox.overlaps(axis_bbox)),
        "legend_xlabel_overlap": None
        if legend_bbox is None
        else bool(legend_bbox.overlaps(xlabel_bbox)),
        "legend_caption_overlap": None
        if legend_bbox is None or caption_bbox is None
        else bool(legend_bbox.overlaps(caption_bbox)),
        "xlabel_inside_figure": _bbox_inside_figure(figure, xlabel_bbox),
        "xlabel_caption_overlap": None
        if caption_bbox is None
        else bool(xlabel_bbox.overlaps(caption_bbox)),
        "caption_inside_figure": None
        if caption_bbox is None
        else _bbox_inside_figure(figure, caption_bbox),
        "title_inside_figure": _bbox_inside_figure(figure, title_bbox),
        "title_axis_overlap": bool(title_bbox.overlaps(axis_bbox)),
        "positive_part_annotation_count": len(positive_part_annotations),
        "positive_part_annotation_inside_figure": None
        if annotation_bbox is None
        else _bbox_inside_figure(figure, annotation_bbox),
        "positive_part_annotation_title_overlap": None
        if annotation_bbox is None
        else bool(annotation_bbox.overlaps(title_bbox)),
        "positive_part_annotation_legend_overlap": None
        if annotation_bbox is None or legend_bbox is None
        else bool(annotation_bbox.overlaps(legend_bbox)),
        "positive_part_annotation_caption_overlap": None
        if annotation_bbox is None or caption_bbox is None
        else bool(annotation_bbox.overlaps(caption_bbox)),
    }


def _bbox_inside_figure(figure: Any, bbox: Any) -> bool:
    """Check if all corners of *bbox* are inside the figure."""
    return all(
        figure.bbox.contains(x, y)
        for x, y in (
            (bbox.x0, bbox.y0),
            (bbox.x1, bbox.y1),
        )
    )


def _partial_density_axis_title_artist(axis: Any) -> Any:
    """Return the effective title artist (left title if set)."""
    left_title = getattr(axis, "_left_title", None)
    if left_title is not None and left_title.get_text():
        return left_title
    return axis.title


def _positive_part_shading_metadata(
    axis: Any,
    contract: PartialDensityDataResult,
) -> dict[str, Any] | None:
    """Validate and report positive-part shading for continuous plots."""
    if not contract.diagnostics.get("continuous_y"):
        return None
    collections = list(axis.collections)
    positive_part = float(contract.diagnostics["positive_part_partial_density_integral"])
    expected = positive_part > 0.0
    if expected and not collections:
        raise ValueError("continuous partial-density plot must shade a positive positive-part region.")
    if not expected and collections:
        raise ValueError("continuous partial-density plot must not shade an empty positive-part region.")
    if len(collections) > 1:
        raise ValueError("continuous partial-density plot must use one positive-part shading collection.")
    alpha = None if not collections else collections[0].get_alpha()
    return {
        "expected_from_positive_part": expected,
        "present": bool(collections),
        "collection_count": len(collections),
        "alpha": None if alpha is None else float(alpha),
        "path_count": 0 if not collections else int(len(collections[0].get_paths())),
    }


def _positive_part_bar_emphasis_metadata(
    axis: Any,
    contract: PartialDensityDataResult,
) -> dict[str, Any] | None:
    """Validate and report positive-part bar emphasis for discrete plots."""
    if contract.diagnostics.get("continuous_y"):
        return None
    records = list(contract.partial_mass_records)
    partial11_bars = list(axis.patches)[: len(records)]
    if len(partial11_bars) != len(records):
        raise ValueError("discrete partial-density plot must expose one partial11 bar per outcome level.")
    rows: list[dict[str, Any]] = []
    expected_y_levels: list[str] = []
    emphasized_y_levels: list[str] = []
    for index, (record, bar) in enumerate(zip(records, partial11_bars, strict=True)):
        y_level = str(record["y"])
        expected = float(record["partial11"]) > float(record["partial01"])
        rendered = float(bar.get_linewidth()) > 1.0
        if expected:
            expected_y_levels.append(y_level)
        if rendered:
            emphasized_y_levels.append(y_level)
        rows.append(
            {
                "bar_index": index,
                "y": y_level,
                "expected_positive_part": expected,
                "rendered_emphasized": rendered,
                "edgecolor_rgba": [float(value) for value in bar.get_edgecolor()],
                "linewidth": float(bar.get_linewidth()),
            }
        )
    if expected_y_levels != emphasized_y_levels:
        raise ValueError("discrete partial-density positive-part bar emphasis must match positive cells.")
    return {
        "expected_from_positive_part": bool(expected_y_levels),
        "expected_emphasized_bar_count": len(expected_y_levels),
        "emphasized_bar_count": len(emphasized_y_levels),
        "expected_y_levels": expected_y_levels,
        "emphasized_y_levels": emphasized_y_levels,
        "bar_rows": rows,
    }


def _rendered_text_line_count(value: str) -> int:
    """Count rendered newline-separated lines."""
    return len(value.splitlines()) if value else 0


def _rendered_text_has_ellipsis_tail(value: str) -> bool:
    """Check if the last line starts with '...' (truncated)."""
    lines = value.splitlines()
    return bool(lines and lines[-1].startswith("..."))


def _continuous_legend_y_anchor(density_1_label: str, density_0_label: str) -> float:
    """Choose legend y-anchor based on label wrapping."""
    formatted_labels = [
        _format_legend_label(density_1_label, max_lines=4),
        _format_legend_label(density_0_label, max_lines=4),
    ]
    max_label_lines = max(label.count("\n") + 1 for label in formatted_labels)
    return -0.31 if max_label_lines > 2 else -0.18


def _set_partial_density_plot_title(
    axis: Any,
    contract: PartialDensityDataResult,
) -> None:
    """Set a descriptive multi-line title on the plot axis."""
    diagnostics = contract.diagnostics
    target_role = str(diagnostics.get("partial_density_target_role", "target")).replace(
        "_", " "
    )
    mediator_level = _format_title_label_value(diagnostics.get("target_original_mediator_level"))
    density_kind = "Continuous partial density" if diagnostics.get("continuous_y") else "Discrete partial PMF"
    adjustment = "adjusted" if diagnostics.get("adjusted") else "unadjusted"
    axis.set_title(
        f"Partial density for {target_role} (M={mediator_level})\n"
        f"{density_kind}, {adjustment}",
        loc="left",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )


def _annotate_partial_density_positive_part(
    axis: Any,
    contract: PartialDensityDataResult,
) -> dict[str, Any] | None:
    """Add a text annotation showing the positive-part value."""
    diagnostics = contract.diagnostics
    key = (
        "positive_part_partial_density_integral"
        if diagnostics.get("continuous_y")
        else "positive_part_partial_pmf_diff"
    )
    value = diagnostics.get(key)
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    x, y, horizontal_alignment, vertical_alignment = _positive_part_annotation_anchor(contract)
    text = f"Positive part = {numeric:.3g}"
    axis.text(
        x,
        y,
        text,
        transform=axis.transAxes,
        ha=horizontal_alignment,
        va=vertical_alignment,
        fontsize=8,
        color="#333333",
        bbox={
            "facecolor": "white",
            "edgecolor": "#BDBDBD",
            "boxstyle": "round,pad=0.25",
            "alpha": 0.86,
            "linewidth": 0.6,
        },
    )
    return {
        "text": text,
        "x": float(x),
        "y": float(y),
        "horizontal_alignment": horizontal_alignment,
        "vertical_alignment": vertical_alignment,
    }


def _positive_part_annotation_anchor(
    contract: PartialDensityDataResult,
) -> tuple[float, float, str, str]:
    """Choose optimal anchor position for the positive-part annotation."""
    occupancy = _partial_density_plot_occupancy(contract)
    if occupancy is None:
        return 0.98, 0.04, "right", "bottom"
    left_height, center_height, right_height, max_height = occupancy
    if max_height <= 0.0:
        return 0.98, 0.04, "right", "bottom"
    if (
        left_height / max_height >= _POSITIVE_PART_BUSY_EDGE_RATIO
        and right_height / max_height >= _POSITIVE_PART_BUSY_EDGE_RATIO
    ):
        if center_height / max_height >= _POSITIVE_PART_BUSY_EDGE_RATIO:
            if not contract.diagnostics.get("continuous_y"):
                return 0.98, 1.02, "right", "bottom"
            return 0.5, 0.04, "center", "bottom"
        return 0.5, 0.96, "center", "top"
    use_left = left_height < right_height
    side_height = left_height if use_left else right_height
    x = 0.02 if use_left else 0.98
    horizontal_alignment = "left" if use_left else "right"
    if side_height <= 0.0:
        return x, 0.04, horizontal_alignment, "bottom"
    return x, 0.96, horizontal_alignment, "top"


def _partial_density_plot_occupancy(
    contract: PartialDensityDataResult,
) -> tuple[float, float, float, float] | None:
    """Measure left/center/right plot occupancy for annotation placement."""
    if contract.diagnostics.get("continuous_y"):
        records = contract.partial_density_records
    else:
        records = contract.partial_mass_records
    if not records:
        return None
    edge_count = max(1, int(math.ceil(len(records) * 0.15)))
    center_start = max(edge_count, int(math.floor(len(records) * 0.35)))
    center_stop = min(len(records) - edge_count, int(math.ceil(len(records) * 0.65)))

    def record_height(record: dict[str, Any]) -> float:
        return max(float(record["partial11"]), float(record["partial01"]))

    left_height = max(record_height(record) for record in records[:edge_count])
    right_height = max(record_height(record) for record in records[-edge_count:])
    center_records = records[center_start:center_stop]
    center_height = (
        0.0
        if not center_records
        else max(record_height(record) for record in center_records)
    )
    max_height = max(record_height(record) for record in records)
    return left_height, center_height, right_height, max_height


def _style_partial_density_axis(
    axis: Any,
    *,
    legend_title: str,
    legend_y_anchor: float = -0.18,
) -> None:
    """Apply standard axis styling: grid, spines, and legend."""
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_ylim(bottom=0.0)
    axis.margins(x=0.02)
    axis.legend(
        title=legend_title,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y_anchor),
        ncol=2,
        borderaxespad=0.0,
    )


def _add_partial_density_caption(figure: Any, caption: str | None) -> str | None:
    """Render a wrapped caption at the bottom of the figure."""
    if caption is None:
        return None
    wrapped = _format_legend_label(caption, width=94, max_lines=3)
    figure.text(
        0.01,
        0.015,
        wrapped,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#4D4D4D",
    )
    return wrapped


def _validate_plotted_partial_density_axis_value(value: Any) -> float:
    """Validate that an outcome grid value is finite numeric."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            "partial_density_plot requires numeric finite plotted continuous outcome grid "
            f"values; y is boolean {value!r}."
        )
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "partial_density_plot requires numeric finite plotted continuous outcome grid "
            f"values; y is {value!r}."
        ) from exc
    if not math.isfinite(numeric):
        raise ValueError(
            "partial_density_plot requires numeric finite plotted continuous outcome grid "
            f"values; y is {numeric!r}."
        )
    return numeric


def _validate_plotted_partial_density_value(value: Any, *, role: str, y: Any) -> float:
    """Validate that a partial-density value is finite nonnegative."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            "partial_density_plot requires numeric finite nonnegative plotted partial "
            f"density values; {role} at y={y!r} is boolean {value!r}."
        )
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "partial_density_plot requires numeric finite nonnegative plotted partial "
            f"density values; {role} at y={y!r} is {value!r}."
        ) from exc
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(
            "partial_density_plot requires numeric finite nonnegative plotted partial "
            f"density values; {role} at y={y!r} is {numeric!r}."
        )
    return numeric


def _import_matplotlib_pyplot() -> Any:
    """Lazy-import matplotlib.pyplot with a helpful error message."""
    try:
        return importlib.import_module("matplotlib.pyplot")
    except ModuleNotFoundError as exc:
        if exc.name and not exc.name.startswith("matplotlib"):
            raise
        raise ModuleNotFoundError(
            "partial_density_plot() requires Matplotlib. Install the plotting extra with "
            "`pip install -e .[plot]` or install `testmechs[plot]`."
        ) from exc


def _binary_levels(series: pd.Series, *, column: str) -> list[object]:
    """Extract ordered binary levels from *series*."""
    return list(ordered_binary_support_levels(series, column=column))


def _normalize_binary_series(series: pd.Series, levels: list[object]) -> pd.Series:
    """Map a binary series to {0, 1} given its ordered levels."""
    return series.map({levels[0]: 0, levels[1]: 1}).astype(int)


def _ordered_support_values(series: pd.Series) -> tuple[Any, ...]:
    """Return observed support of *series* in deterministic order."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        observed_values = list(pd.unique(series.dropna()))
        return tuple(
            _normalize_scalar(category)
            for category in series.cat.categories
            if any(category == observed_value for observed_value in observed_values)
        )
    return tuple(sorted((_normalize_scalar(value) for value in pd.unique(series)), key=_support_sort_key))


def _support_sort_key(value: Any) -> tuple[Any, ...]:
    """Deterministic sort key for support values."""
    normalized = _normalize_scalar(value)
    if isinstance(normalized, bool):
        return ("bool", int(normalized))
    if isinstance(normalized, (int, float)):
        return ("number", float(normalized))
    return (type(normalized).__name__, repr(normalized))


def _normalize_scalar(value: object) -> object:
    """Convert NumPy scalars to plain Python types."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except (AttributeError, ValueError):
            return value
    return value
