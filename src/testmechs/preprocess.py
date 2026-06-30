"""Data preprocessing utilities for the Testing Mechanisms framework.

This module provides functions for preparing analysis data prior to bounds
estimation or partial-density computation.  The primary responsibilities are:

- Removing rows with missing values in treatment, mediator, outcome, or
  regression-formula columns (``remove_missing_from_df``).
- Discretizing continuous outcome vectors into quantile-based bins
  (``discretize_y``).
- Detecting and normalizing binary support levels to {0, 1}
  (``normalize_binary_support``, ``ordered_binary_support_levels``).
- Building cell-count diagnostic payloads that identify small or empty
  treatment-mediator-outcome cells (``build_cell_count_diagnostics``).

All public functions validate inputs eagerly and raise ``ValueError`` on
violation.  The module parallels the R package helpers ``remove_missing_from_df``
and ``discretize_y`` documented in the TestMechs R manual.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import math
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BinarySupportNormalization:
    """Immutable record of a binary-support normalization mapping.

    Stores the original two-level support of a binary variable and provides
    a ``transform`` method that maps any conforming series to {0, 1}.

    Parameters
    ----------
    original_levels : tuple of two elements
        The detected (low, high) support levels in their natural order.
    normalized_levels : tuple of int, default (0, 1)
        The integer encoding used after normalization.

    See Also
    --------
    normalize_binary_support : Factory that constructs this dataclass.
    ordered_binary_support_levels : Extracts and orders binary levels.
    """

    original_levels: tuple[Any, Any]
    normalized_levels: tuple[int, int] = (0, 1)

    def transform(self, series: pd.Series) -> pd.Series:
        """Map *series* values to normalized integer levels.

        Parameters
        ----------
        series : pd.Series
            A series whose values must belong to ``original_levels``.

        Returns
        -------
        pd.Series
            Integer-typed series with values in ``normalized_levels``.

        Raises
        ------
        ValueError
            If *series* contains values outside the fitted binary support.
        """
        level_mapping = dict(
            zip(self.original_levels, self.normalized_levels, strict=True)
        )
        mapped = series.map(lambda value: level_mapping.get(_normalize_scalar(value)))
        if mapped.isna().any():
            bad_values = [
                _normalize_scalar(value)
                for value in pd.unique(series.loc[mapped.isna()])
            ]
            formatted = ", ".join(repr(value) for value in bad_values)
            raise ValueError(
                "series contains values outside the fitted binary support: "
                f"{formatted}."
            )
        return mapped.astype(int)

    def diagnostics(self, *, original_key: str, normalized_key: str) -> dict[str, object]:
        """Return a diagnostic dict recording the normalization mapping."""
        return {
            original_key: [_normalize_scalar(value) for value in self.original_levels],
            normalized_key: list(self.normalized_levels),
        }


def remove_missing_from_df(
    *,
    df: pd.DataFrame,
    d: str,
    m: str | Sequence[str],
    y: str,
    w: str | None = None,
    reg_formula: str | None = None,
) -> pd.DataFrame:
    """Drop rows with missing values in analysis-relevant columns.

    This is the Python equivalent of the R function ``remove_missing_from_df``.
    It identifies the minimal set of required columns from the treatment,
    mediator, outcome, optional weight, and optional regression-formula
    specification, validates their presence in *df*, then returns a complete-case
    copy of the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The input analysis dataframe.
    d : str
        Column name of the binary treatment variable.
    m : str or sequence of str
        Column name(s) of the mediator variable(s).
    y : str
        Column name of the outcome variable.
    w : str or None, optional
        Column name of an observation-weight variable.  If ``None`` (default),
        equal weights are implied.
    reg_formula : str or None, optional
        A regression-formula string in the supported subset (see
        ``parse_reg_formula``).  When provided, additional covariate columns
        are included in the complete-case requirement.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with rows containing any ``NaN`` in the required columns
        removed.

    Raises
    ------
    ValueError
        If *df* is missing required columns, if role-overlap constraints are
        violated, or if no complete observations remain.

    Examples
    --------
    >>> import pandas as pd
    >>> from testmechs.preprocess import remove_missing_from_df
    >>> df = pd.DataFrame({"D": [1, 0, 1], "M": [1, None, 0], "Y": [3, 2, 1]})
    >>> clean = remove_missing_from_df(df=df, d="D", m="M", y="Y")
    >>> len(clean)
    2

    Notes
    -----
    Column-role validation ensures that treatment and outcome do not overlap,
    mediators do not coincide with treatment or outcome, and regression-formula
    variables do not reuse mediator or outcome roles.

    See Also
    --------
    discretize_y : Discretize continuous outcomes before bounds estimation.
    parse_reg_formula : Parse the regression formula whose variables extend
        the required-column set.
    """
    _validate_scalar_column_name(d, argument="d")
    _validate_scalar_column_name(y, argument="y")
    _reject_treatment_outcome_role_overlap(d=d, y=y)
    if w is not None:
        _validate_scalar_column_name(w, argument="w")
    mediator_columns = list(_mediator_columns(m))
    _reject_mediator_role_overlap(tuple(mediator_columns), d=d, y=y)
    required_columns = [d, y, *mediator_columns]
    if w is not None:
        required_columns.append(w)
    if reg_formula is not None:
        from .regression import parse_reg_formula

        spec = parse_reg_formula(reg_formula, d=d)
        _reject_reg_formula_role_reuse(
            spec_variables=spec.variables,
            treatment=spec.treatment,
            mediator_columns=tuple(mediator_columns),
            y=y,
        )
        required_columns.extend(spec.variables)
    required_columns = list(dict.fromkeys(required_columns))
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        formatted = ", ".join(repr(column) for column in missing_columns)
        raise ValueError(f"df is missing required columns: {formatted}.")
    cleaned = df.dropna(subset=required_columns).copy()
    if cleaned.empty:
        if reg_formula is not None:
            raise ValueError("No complete observations remain after applying reg_formula.")
        raise ValueError("No complete observations remain after dropping required columns.")
    return cleaned


def normalize_binary_support(series: pd.Series, *, column: str) -> BinarySupportNormalization:
    """Detect binary support levels and build a normalization mapping.

    Inspects *series* for exactly two unique non-missing values, orders them,
    and returns a frozen ``BinarySupportNormalization`` that can later
    ``transform`` conforming series to {0, 1}.

    Parameters
    ----------
    series : pd.Series
        A series expected to contain exactly two distinct non-missing values.
    column : str
        A descriptive name for the series, used in error messages.

    Returns
    -------
    BinarySupportNormalization
        A frozen dataclass recording the original levels and providing a
        ``transform`` method.

    Raises
    ------
    ValueError
        If *series* does not contain exactly two finite support levels.

    Examples
    --------
    >>> import pandas as pd
    >>> from testmechs.preprocess import normalize_binary_support
    >>> norm = normalize_binary_support(pd.Series(["ctrl", "treat", "ctrl"]), column="D")
    >>> norm.original_levels
    ('ctrl', 'treat')

    See Also
    --------
    ordered_binary_support_levels : Lower-level extractor.
    BinarySupportNormalization : The returned dataclass.
    """
    levels = ordered_binary_support_levels(series, column=column)
    return BinarySupportNormalization(original_levels=levels)


def ordered_binary_support_levels(series: pd.Series, *, column: str) -> tuple[Any, Any]:
    """Extract and sort the two support levels of a binary series.

    Parameters
    ----------
    series : pd.Series
        A series with exactly two distinct non-missing values.
    column : str
        Descriptive name used in error messages.

    Returns
    -------
    tuple of two elements
        The (low, high) support levels in deterministic sort order.

    Raises
    ------
    ValueError
        If the series does not contain exactly two levels or if levels are
        non-finite numeric values.

    See Also
    --------
    normalize_binary_support : Higher-level convenience wrapper.
    """
    levels = tuple(_normalize_scalar(value) for value in pd.unique(series.dropna()))
    if len(levels) != 2:
        raise ValueError(f"{column} must contain exactly two support levels.")
    _reject_nonfinite_numeric_support_levels(levels, column=column)
    return tuple(sorted(levels, key=_binary_support_sort_key))  # type: ignore[return-value]


def discretize_y(yvec: pd.Series, num_bins: int) -> pd.Series:
    """Discretize an outcome vector into quantile-based bins.

    If *yvec* already has ``num_bins`` or fewer unique values the series is
    returned as a categorical without modification.  Otherwise quantile
    cutpoints at 1/num_bins, 2/num_bins, ... are computed (using R type-7
    interpolation) and the outcome is cut into interval bins.  When duplicate
    cutpoints arise from point masses, adjacent bins are merged so the actual
    number of bins may be less than *num_bins*.

    This is the Python equivalent of the R function ``discretize_y``.

    Parameters
    ----------
    yvec : pd.Series
        Outcome values.  Must be non-empty and contain no missing values.
        Non-numeric outcomes are allowed only when the observed support already
        has ``num_bins`` or fewer levels.
    num_bins : int
        Target number of bins (must be a positive integer).

    Returns
    -------
    pd.Series
        A categorical series with at most *num_bins* levels.

    Raises
    ------
    ValueError
        If *num_bins* is not a positive integer, if *yvec* is empty, contains
        missing or non-finite numeric values, or if quantile discretization
        is needed but the outcome is non-numeric.

    Examples
    --------
    >>> import pandas as pd
    >>> from testmechs.preprocess import discretize_y
    >>> y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    >>> binned = discretize_y(y, num_bins=3)
    >>> binned.nunique()
    3

    Notes
    -----
    The quantile computation uses R type-7 interpolation (linear
    interpolation of the order statistics) for exact numerical parity with
    the R implementation.

    See Also
    --------
    remove_missing_from_df : Should be called before discretization to ensure
        no missing values remain.
    """
    if isinstance(num_bins, bool) or not isinstance(num_bins, Integral):
        raise ValueError("num_bins must be a positive integer")
    if num_bins <= 0:
        raise ValueError("num_bins must be a positive integer")
    num_bins = int(num_bins)

    y_series = pd.Series(yvec).copy()
    if y_series.empty:
        raise ValueError("Cannot discretize an empty outcome vector.")
    if y_series.isna().any():
        raise ValueError("Cannot discretize an outcome vector with missing values.")
    if pd.api.types.is_numeric_dtype(y_series):
        _reject_nonfinite_numeric_y(pd.to_numeric(y_series, errors="raise"))
    if y_series.nunique(dropna=False) <= num_bins:
        return _observed_category_series(y_series)

    numeric_y = _numeric_y_for_quantile_discretization(y_series)
    quantiles = [_r_type7_quantile(numeric_y, probability=index / num_bins) for index in range(1, num_bins)]
    cutpoints = [-float("inf"), *pd.Index(quantiles).unique().tolist(), float("inf")]
    return pd.cut(numeric_y, bins=cutpoints, include_lowest=True, duplicates="drop")


def build_cell_count_diagnostics(
    *,
    df: pd.DataFrame,
    d: str,
    m: str,
    y: str,
    cluster: str | None = None,
    requested_num_y_bins: int | None,
    applied_num_y_bins: int | None,
    size_risk_threshold: int = 15,
    no_bite_flag: bool | None = None,
    theta_kk_min: float | None = None,
    no_bite_reason: str | None = None,
    support_diagnostics: dict[str, object] | None = None,
) -> dict[str, object]:
    """Compute cell-count diagnostics for the (D, M, Y) cross-tabulation.

    Constructs the full factorial grid of treatment x mediator x outcome
    levels, counts observations (and optionally clusters) in each cell, and
    returns a JSON-safe diagnostic dictionary reporting empty cells, small
    cells, and aggregated risk indicators.

    Parameters
    ----------
    df : pd.DataFrame
        The analysis dataframe (must be non-empty with no missing group labels).
    d : str
        Treatment column name.
    m : str
        Mediator column name.
    y : str
        Outcome column name.
    cluster : str or None, optional
        Cluster-identifier column.  When provided, per-cell cluster counts
        are computed alongside observation counts.
    requested_num_y_bins : int or None
        The user-requested number of outcome bins (recorded for audit).
    applied_num_y_bins : int or None
        The actual number of outcome bins applied after discretization.
    size_risk_threshold : int, default 15
        Cells with fewer than this many observations (or clusters) are
        flagged as "small".
    no_bite_flag : bool or None, optional
        Pre-computed no-bite indicator from bounds estimation.
    theta_kk_min : float or None, optional
        Minimum diagonal element of Theta used in no-bite assessment.
    no_bite_reason : str or None, optional
        Human-readable reason for the no-bite flag.
    support_diagnostics : dict or None, optional
        Additional support-level diagnostics to merge into the result.

    Returns
    -------
    dict
        A JSON-safe nested dictionary containing cell counts, cluster counts,
        empty-cell lists, small-cell lists, risk flags, and no-bite metadata.

    Raises
    ------
    ValueError
        If required columns are missing, *df* is empty, group labels contain
        missing values, or parameter constraints are violated.

    Examples
    --------
    >>> import pandas as pd
    >>> from testmechs.preprocess import build_cell_count_diagnostics
    >>> df = pd.DataFrame({"D": [0,0,1,1], "M": [0,1,0,1], "Y": ["a","b","a","b"]})
    >>> diag = build_cell_count_diagnostics(
    ...     df=df, d="D", m="M", y="Y",
    ...     requested_num_y_bins=None, applied_num_y_bins=None,
    ... )
    >>> diag["min_cell_count"]
    1

    Notes
    -----
    The full factorial grid is constructed from the observed support of each
    grouping column.  Cells absent from the data receive a count of zero.

    See Also
    --------
    remove_missing_from_df : Prepare analysis data before diagnostics.
    discretize_y : Reduce outcome cardinality before diagnostics.
    """
    _validate_scalar_column_name(d, argument="d")
    _validate_scalar_column_name(m, argument="m")
    _validate_scalar_column_name(y, argument="y")
    _reject_cell_count_role_overlap(d=d, m=m, y=y)
    if cluster is not None:
        _validate_scalar_column_name(cluster, argument="cluster")
    _validate_optional_positive_integer(requested_num_y_bins, argument="requested_num_y_bins")
    _validate_optional_positive_integer(applied_num_y_bins, argument="applied_num_y_bins")
    if (
        requested_num_y_bins is not None
        and applied_num_y_bins is not None
        and applied_num_y_bins > requested_num_y_bins
    ):
        raise ValueError("applied_num_y_bins cannot exceed requested_num_y_bins.")
    _validate_positive_integer(size_risk_threshold, argument="size_risk_threshold")
    group_columns = [d, m, y]
    required_columns = list(group_columns)
    if cluster is not None:
        required_columns.append(cluster)
    missing_columns = [column for column in dict.fromkeys(required_columns) if column not in df.columns]
    if missing_columns:
        formatted = ", ".join(repr(column) for column in missing_columns)
        raise ValueError(f"df is missing required columns: {formatted}.")
    if df.empty:
        raise ValueError("Cell-count diagnostics require a nonempty analysis sample.")
    missing_group_columns = [column for column in group_columns if df[column].isna().any()]
    if missing_group_columns:
        formatted = ", ".join(repr(column) for column in missing_group_columns)
        raise ValueError(f"Cell-count diagnostics require non-missing group labels: {formatted}.")
    if cluster is not None and df[cluster].isna().any():
        raise ValueError("cluster must not contain missing values for cell-count diagnostics.")

    full_index = pd.MultiIndex.from_product(
        [_series_support_values(df[column]) for column in group_columns],
        names=group_columns,
    )
    cell_counts = (
        df.groupby(group_columns, dropna=False, observed=False, sort=False)
        .size()
        .reindex(full_index, fill_value=0)
        .astype(int)
        .reset_index(name="count")
    )
    if cluster is None:
        cluster_counts = cell_counts.rename(columns={"count": "cluster_count"})
    else:
        cluster_counts = (
            df.groupby(group_columns, dropna=False, observed=False, sort=False)[cluster]
            .nunique()
            .reindex(full_index, fill_value=0)
            .astype(int)
            .reset_index(name="cluster_count")
        )

    min_cell_count = int(cell_counts["count"].min()) if not cell_counts.empty else 0
    min_cluster_count = (
        int(cluster_counts["cluster_count"].min()) if not cluster_counts.empty else 0
    )
    empty_cell_rows = cell_counts.loc[cell_counts["count"] == 0]
    empty_cluster_cell_rows = cluster_counts.loc[cluster_counts["cluster_count"] == 0]
    small_cell_rows = cell_counts.loc[cell_counts["count"] < size_risk_threshold]
    small_cluster_cell_rows = cluster_counts.loc[
        cluster_counts["cluster_count"] < size_risk_threshold
    ]

    diagnostics = {
        "requested_num_y_bins": requested_num_y_bins,
        "applied_num_y_bins": applied_num_y_bins,
        "n_obs_used": int(len(df)),
        "size_risk_threshold": int(size_risk_threshold),
        "binary_mediator": bool(df[m].nunique(dropna=False) == 2),
        "y_levels": [_normalize_scalar(value) for value in _series_support_values(df[y])],
        "cell_counts": [
            {
                d: _normalize_scalar(row[d]),
                m: _normalize_scalar(row[m]),
                y: _normalize_scalar(row[y]),
                "count": int(row["count"]),
            }
            for row in cell_counts.to_dict(orient="records")
        ],
        "cluster_counts": [
            {
                d: _normalize_scalar(row[d]),
                m: _normalize_scalar(row[m]),
                y: _normalize_scalar(row[y]),
                "cluster_count": int(row["cluster_count"]),
            }
            for row in cluster_counts.to_dict(orient="records")
        ],
        "min_cell_count": min_cell_count,
        "min_cluster_count": min_cluster_count,
        "support_cell_count": int(len(cell_counts)),
        "empty_cell_count": int(len(empty_cell_rows)),
        "empty_cluster_cell_count": int(len(empty_cluster_cell_rows)),
        "small_cell_count": int(len(small_cell_rows)),
        "small_cluster_cell_count": int(len(small_cluster_cell_rows)),
        "empty_cells": _cell_count_records(empty_cell_rows, d=d, m=m, y=y, count_column="count"),
        "empty_cluster_cells": _cell_count_records(
            empty_cluster_cell_rows,
            d=d,
            m=m,
            y=y,
            count_column="cluster_count",
        ),
        "small_cells": _cell_count_records(small_cell_rows, d=d, m=m, y=y, count_column="count"),
        "small_cluster_cells": _cell_count_records(
            small_cluster_cell_rows,
            d=d,
            m=m,
            y=y,
            count_column="cluster_count",
        ),
        "size_risk": bool(
            min_cell_count < size_risk_threshold or min_cluster_count < size_risk_threshold
        ),
        "no_bite": {
            "assessed": no_bite_flag is not None,
            "flag": no_bite_flag,
            "theta_kk_min": theta_kk_min,
            "reason": no_bite_reason,
        },
    }
    if support_diagnostics is not None:
        diagnostics.update(support_diagnostics)  # type: ignore[arg-type]
    return _json_safe_diagnostic_payload(diagnostics)  # type: ignore[return-value]


def _cell_count_records(
    frame: pd.DataFrame,
    *,
    d: str,
    m: str,
    y: str,
    count_column: str,
) -> list[dict[str, object]]:
    """Convert a cell-count frame to a list of normalized record dicts."""
    return [
        {
            d: _normalize_scalar(row[d]),
            m: _normalize_scalar(row[m]),
            y: _normalize_scalar(row[y]),
            count_column: int(row[count_column]),
        }
        for row in frame.to_dict(orient="records")
    ]


def _normalize_scalar(value: object) -> object:
    """Convert NumPy scalars and Interval objects to plain Python types."""
    if isinstance(value, pd.Interval):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (AttributeError, ValueError):
            return value
    return value


def _json_safe_diagnostic_payload(value: object) -> object:
    """Recursively coerce a diagnostic payload to JSON-serializable types."""
    if isinstance(value, dict):
        return {str(key): _json_safe_diagnostic_payload(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe_diagnostic_payload(item) for item in value]
    if isinstance(value, list):
        return [_json_safe_diagnostic_payload(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe_diagnostic_payload(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe_diagnostic_payload(value.item())
    if isinstance(value, pd.Interval):
        return str(value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return None
        return "positive_infinity" if value > 0 else "negative_infinity"
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _validate_scalar_column_name(value: object, *, argument: str) -> None:
    """Raise ValueError if *value* is not a non-empty string."""
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{argument} must be a non-empty string column name.")


def _validate_positive_integer(value: object, *, argument: str) -> None:
    """Raise ValueError if *value* is not a positive integer."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{argument} must be a positive integer.")


def _validate_optional_positive_integer(value: object, *, argument: str) -> None:
    """Validate *value* as a positive integer or None."""
    if value is None:
        return
    _validate_positive_integer(value, argument=argument)


def _mediator_columns(m: str | Sequence[str]) -> tuple[str, ...]:
    """Normalize *m* to a validated tuple of mediator column names."""
    if isinstance(m, str):
        if m == "":
            raise ValueError("m must be a non-empty mediator column name.")
        return (m,)
    if not isinstance(m, Sequence):
        raise ValueError("m must be a mediator column name or a sequence of mediator column names.")
    columns = tuple(m)
    if not columns:
        raise ValueError("m must contain at least one mediator column.")
    if not all(isinstance(column, str) and column != "" for column in columns):
        raise ValueError("All mediator columns must be non-empty strings.")
    if len(set(columns)) != len(columns):
        raise ValueError("m must not contain duplicate mediator columns.")
    return columns


def _reject_mediator_role_overlap(
    mediator_columns: tuple[str, ...],
    *,
    d: str,
    y: str,
) -> None:
    """Raise if any mediator column duplicates treatment or outcome."""
    if any(column in {d, y} for column in mediator_columns):
        raise ValueError("m must not include treatment or outcome columns.")


def _reject_treatment_outcome_role_overlap(*, d: str, y: str) -> None:
    """Raise if treatment and outcome reference the same column."""
    if d == y:
        raise ValueError("d and y must name distinct treatment and outcome columns.")


def _reject_reg_formula_role_reuse(
    *,
    spec_variables: Sequence[str],
    treatment: str,
    mediator_columns: tuple[str, ...],
    y: str,
) -> None:
    """Raise if formula variables reuse mediator or outcome columns."""
    reserved_columns = {y, *mediator_columns}
    if any(variable in reserved_columns for variable in spec_variables if variable != treatment):
        raise ValueError("reg_formula variables must not reuse outcome or mediator columns.")


def _reject_cell_count_role_overlap(*, d: str, m: str, y: str) -> None:
    """Raise if d, m, y do not name three distinct columns."""
    if len({d, m, y}) != 3:
        raise ValueError("d, m, and y must name distinct cell-count role columns.")


def _series_support_values(series: pd.Series) -> tuple[object, ...]:
    """Return the observed support of *series* in deterministic order."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        observed_values = list(pd.unique(series.dropna()))
        return tuple(
            category
            for category in series.cat.categories
            if any(category == observed_value for observed_value in observed_values)
        )
    return tuple(sorted(pd.unique(series), key=_support_sort_key))


def _observed_category_series(series: pd.Series) -> pd.Series:
    """Convert *series* to a categorical with only observed levels."""
    category_series = series.astype("category")
    if isinstance(category_series.dtype, pd.CategoricalDtype):
        return category_series.cat.remove_unused_categories()
    return category_series


def _binary_support_sort_key(value: object) -> tuple[str, str]:
    """Sort key for binary support levels."""
    return _support_sort_key(value)  # type: ignore[return-value]


def _support_sort_key(value: object) -> tuple[object, ...]:
    """Deterministic sort key: booleans, then numbers, then by repr."""
    normalized = _normalize_scalar(value)
    if isinstance(normalized, bool):
        return ("bool", int(normalized))
    if isinstance(normalized, (int, float)):
        return ("number", float(normalized))
    return (type(normalized).__name__, repr(normalized))


def _numeric_y_for_quantile_discretization(yvec: pd.Series) -> pd.Series:
    """Coerce *yvec* to numeric, raising if non-numeric and bins needed."""
    try:
        numeric_y = pd.to_numeric(yvec, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "discretize_y requires numeric y values when quantile cutpoints are needed; "
            "non-numeric outcomes are allowed only when the observed support already has "
            "num_bins or fewer levels."
        ) from exc
    _reject_nonfinite_numeric_y(numeric_y)
    return numeric_y


def _reject_nonfinite_numeric_y(yvec: pd.Series) -> None:
    """Raise if *yvec* contains non-finite numeric values."""
    if not all(math.isfinite(float(value)) for value in yvec.tolist()):
        raise ValueError("discretize_y requires finite numeric y values.")


def _reject_nonfinite_numeric_support_levels(
    levels: Sequence[object],
    *,
    column: str,
) -> None:
    """Raise if any numeric support level is non-finite."""
    for value in levels:
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            raise ValueError(f"{column} must contain only finite numeric support levels.")


def _r_type7_quantile(yvec: pd.Series, *, probability: float) -> float:
    """Compute a single quantile using R type-7 linear interpolation."""
    values = sorted(float(value) for value in yvec.tolist())
    if not values:
        raise ValueError("Cannot discretize an empty outcome vector.")
    if probability <= 0:
        return values[0]
    if probability >= 1:
        return values[-1]

    h = (len(values) - 1) * probability + 1
    lower_rank = math.floor(h)
    gamma = h - lower_rank
    lower_value = values[lower_rank - 1]
    if gamma == 0:
        return lower_value
    upper_value = values[lower_rank]
    return (1 - gamma) * lower_value + gamma * upper_value
