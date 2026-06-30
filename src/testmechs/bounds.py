"""Finite-support Testing Mechanisms bound estimators.

This module implements the core bound computations for the ``testmechs``
package: lower bounds on the fraction of always-takers affected by treatment
outside the mediator channel, ADE/ATS partial-identification bounds, and
breakdown-point analysis for defier-share relaxations.

All public estimators accept a pandas DataFrame (or CSV path), column names
for treatment, mediator(s), and outcome, and return structured result
dataclasses (``LowerBoundResult`` or ``ADEBoundsResult``) that expose numeric
estimates, rich diagnostics, and JSON/table export views.

The estimators support:

- Scalar or vector (multivariate) finite-support mediators.
- Ordered-monotone and elementwise-monotone restrictions.
- Bounded defier-share relaxations via linear/fractional programming.
- Optional regression-adjusted probability grids.
- Outcome discretization for continuous outcomes.

References
----------
Kwon, S. and Roth, J. (2026). "Testing Mechanisms."
    *The Review of Economic Studies*, rdag028.
    doi:10.1093/restud/rdag028.
"""

from __future__ import annotations

import numbers
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from .preprocess import (
    build_cell_count_diagnostics,
    discretize_y,
    normalize_binary_support,
    remove_missing_from_df,
)
from .regression import AdjustedProbabilityResult, compute_adjusted_probabilities, parse_reg_formula
from .results import ADEBoundsResult, LowerBoundResult


def theta_kk_min_ordered_monotone(
    *,
    p_m_given_d0: Mapping[object, float] | Sequence[float],
    p_m_given_d1: Mapping[object, float] | Sequence[float],
    mediator_order: Sequence[object] | None = None,
) -> dict[object, float]:
    """Compute minimum always-taker shares under ordered monotonicity.

    For each mediator level *k*, computes the minimum probability
    theta_{kk} = P(M(0) = k, M(1) = k) that is compatible with the
    observed mediator marginals P(M | D=0) and P(M | D=1) under an
    ordered-monotone restriction M(1) >= M(0).

    Parameters
    ----------
    p_m_given_d0 : Mapping[object, float] or Sequence[float]
        Probability distribution of the mediator given D=0.  May be a
        dict mapping mediator levels to probabilities, or a sequence of
        probabilities indexed by integer position.
    p_m_given_d1 : Mapping[object, float] or Sequence[float]
        Probability distribution of the mediator given D=1.  Same
        structure as ``p_m_given_d0``.
    mediator_order : sequence of objects or None, optional
        Explicit ordering of mediator support levels.  If ``None``,
        levels must be numeric or boolean so that a natural order exists.

    Returns
    -------
    dict[object, float]
        Dictionary mapping each mediator level to its minimum
        always-taker share theta_{kk}^{min}.

    Raises
    ------
    ValueError
        If distributions are invalid, support is non-orderable without
        an explicit ``mediator_order``, or marginals violate the
        ordered-monotone stochastic dominance condition.

    Notes
    -----
    Implements the closed-form theta_{kk}^{min} from Kwon and Roth (2026),
    which equals max(P(M=k|D=1) - survival_gap_k, 0) where survival_gap_k
    is the treated-minus-control survival difference at level k.

    See Also
    --------
    lb_frac_affected : Lower bound using these always-taker shares.
    """
    p0 = _coerce_distribution(p_m_given_d0, name="p_m_given_d0")
    p1 = _coerce_distribution(p_m_given_d1, name="p_m_given_d1")
    levels = _sort_mediator_levels(set(p0) | set(p1), order=mediator_order)
    _validate_public_theta_ordered_support(levels=levels, mediator_order=mediator_order)
    _validate_ordered_monotone_marginals(
        p0=p0,
        p1=p1,
        levels=levels,
        mediator_order=mediator_order,
    )
    theta: dict[object, float] = {}
    for level in levels:
        survival_gap = sum(
            p1.get(candidate, 0.0)
            for candidate in levels
            if _mediator_leq(level, candidate, order=mediator_order)
        )
        survival_gap -= sum(
            p0.get(candidate, 0.0)
            for candidate in levels
            if _mediator_leq(level, candidate, order=mediator_order)
        )
        theta[level] = max(p1.get(level, 0.0) - survival_gap, 0.0)
    return theta


def _validate_public_theta_ordered_support(
    *,
    levels: Sequence[object],
    mediator_order: Sequence[object] | None,
) -> None:
    """Validate that mediator support is orderable for theta computation."""
    if mediator_order is not None:
        order_levels = list(mediator_order)
        if len(set(order_levels)) != len(order_levels) or set(order_levels) != set(levels):
            raise ValueError(
                "theta_kk_min_ordered_monotone mediator_order must contain each "
                "mediator support value exactly once."
            )
        return
    for level in levels:
        normalized = _normalize_scalar(level)
        if not isinstance(normalized, (bool, int, float)):
            raise ValueError(
                "theta_kk_min_ordered_monotone requires mediator support "
                "to be numeric, boolean, or accompanied by an explicit mediator_order."
            )


def _validate_ordered_monotone_marginals(
    *,
    p0: Mapping[object, float],
    p1: Mapping[object, float],
    levels: Sequence[object],
    mediator_order: Sequence[object] | None,
) -> None:
    """Verify treated survival dominates control at every support level."""
    for level in levels:
        treated_survival = sum(
            p1.get(candidate, 0.0)
            for candidate in levels
            if _mediator_leq(level, candidate, order=mediator_order)
        )
        control_survival = sum(
            p0.get(candidate, 0.0)
            for candidate in levels
            if _mediator_leq(level, candidate, order=mediator_order)
        )
        if treated_survival + 1e-10 < control_survival:
            raise ValueError(
                "theta_kk_min_ordered_monotone requires mediator marginals "
                "compatible with M(1) >= M(0); treated survival mass is smaller "
                "than control survival mass for at least one ordered support level."
            )


def _prepare_bounds_dataframe(
    *,
    df: pd.DataFrame | None,
    data_path: str | Path | None,
    d: str,
    m: str | Sequence[str],
    y: str,
    num_y_bins: int | None,
    reg_formula: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load, validate, and normalise data for bound estimation.

    Handles CSV loading, missing-value removal, outcome discretization,
    treatment normalization to {0, 1}, and vector-mediator tuple encoding.
    Returns the prepared DataFrame and a mediator metadata dictionary.
    """
    d = _validate_scalar_column_name(d, name="d")
    y = _validate_scalar_column_name(y, name="y")
    if df is not None and data_path is not None:
        raise ValueError("Exactly one of df or data_path must be provided.")
    if df is None:
        if data_path is None:
            raise ValueError("Exactly one of df or data_path must be provided.")
        df = pd.read_csv(data_path)
    mediator_columns = _mediator_columns(m)
    vector_mediator = len(mediator_columns) > 1
    data = remove_missing_from_df(
        df=df,
        d=d,
        m=mediator_columns if vector_mediator else mediator_columns[0],
        y=y,
        reg_formula=reg_formula,
    )
    if num_y_bins is not None:
        data = data.copy()
        data[y] = discretize_y(data[y], num_bins=num_y_bins)
    else:
        _reject_nonfinite_numeric_bounds_outcome(data[y], column=y)
    treatment_support = normalize_binary_support(data[d], column=d)
    data = data.copy()
    data[d] = treatment_support.transform(data[d])
    _validate_binary_treatment(data[d], d)

    mediator_order = None
    if vector_mediator:
        _validate_vector_mediator_elementwise_support(data, mediator_columns)
        mediator_column = _internal_mediator_column_name(data)
        data = data.copy()
        data[mediator_column] = [
            _normalize_mediator_level(tuple(row))
            for row in data.loc[:, list(mediator_columns)].itertuples(index=False, name=None)
        ]
    else:
        mediator_column = mediator_columns[0]
        if isinstance(data[mediator_column].dtype, pd.CategoricalDtype) and data[
            mediator_column
        ].dtype.ordered:
            observed_levels = set(pd.unique(data[mediator_column].dropna()))
            mediator_order = tuple(
                _normalize_scalar(level)
                for level in data[mediator_column].dtype.categories
                if level in observed_levels
            )
    _reject_nonfinite_numeric_mediator_support(data[mediator_column], column=mediator_column)

    return data, {
        "column": mediator_column,
        "columns": mediator_columns,
        "vector": vector_mediator,
        "level_order": mediator_order,
        "treatment_support": treatment_support,
    }


def _reject_nonfinite_numeric_bounds_outcome(series: pd.Series, *, column: str) -> None:
    """Raise ValueError if a numeric outcome column contains non-finite values."""
    if not pd.api.types.is_numeric_dtype(series):
        return
    values = pd.to_numeric(series, errors="raise")
    if not bool(np.isfinite(values.to_numpy(dtype=float)).all()):
        raise ValueError(f"{column} must contain only finite numeric outcome values.")


def _validate_bounds_ade_numeric_outcome(series: pd.Series, *, column: str) -> None:
    """Ensure outcome column is fully numeric and finite for ADE bounds."""
    try:
        values = pd.to_numeric(series, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{column} must contain finite numeric outcome values for bounds_ade_ats."
        ) from exc
    if not bool(np.isfinite(values.to_numpy(dtype=float)).all()):
        raise ValueError(
            f"{column} must contain finite numeric outcome values for bounds_ade_ats."
        )


def _reject_nonfinite_numeric_mediator_support(series: pd.Series, *, column: str) -> None:
    """Raise ValueError if mediator support contains non-finite numeric values."""
    for value in pd.unique(series.dropna()):
        if _contains_nonfinite_numeric_support(value):
            raise ValueError(f"{column} must contain only finite numeric mediator support values.")


def _contains_nonfinite_numeric_support(value: object) -> bool:
    """Return True if *value* (possibly a tuple) has non-finite numeric parts."""
    normalized = _normalize_scalar(value)
    if isinstance(normalized, tuple):
        return any(_contains_nonfinite_numeric_support(component) for component in normalized)
    if isinstance(normalized, bool):
        return False
    if isinstance(normalized, (int, float)):
        return not np.isfinite(float(normalized))
    return False


def compute_positive_part_partial_pmf_diff(
    *,
    df: pd.DataFrame,
    d: str,
    m: str | Sequence[str],
    y: str,
    at_group: object,
    num_y_bins: int | None = None,
) -> float:
    """Compute the positive-part sum of partial PMF differences for an AT group.

    Calculates sum_y max(P(Y=y, M=k | D=1) - P(Y=y, M=k | D=0), 0) for
    the specified always-taker group *k*.  This quantity is the numerator
    ingredient in the lower-bound formula.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis data frame.
    d : str
        Binary treatment column name.
    m : str or sequence of str
        Mediator column name(s).
    y : str
        Outcome column name (finite support).
    at_group : object
        Mediator level identifying the always-taker group.
    num_y_bins : int or None, optional
        If provided, discretize the outcome into this many bins.

    Returns
    -------
    float
        The positive-part partial PMF difference for the group.

    Raises
    ------
    ValueError
        If ``at_group`` is absent from mediator support or inputs are invalid.
    """
    data, mediator = _prepare_bounds_dataframe(
        df=df,
        data_path=None,
        d=d,
        m=m,
        y=y,
        num_y_bins=num_y_bins,
    )
    at_group = _coerce_at_group(at_group, mediator_columns=mediator["columns"])
    _validate_binary_treatment(data[d], d)
    rows = _positive_part_partial_pmf_diff_rows_from_prepared(
        data=data,
        d=d,
        m=mediator["column"],
        y=y,
        at_group=at_group,
    )
    return float(sum(row["positive_part_contribution"] for row in rows))


def _positive_part_partial_pmf_diff_rows_from_prepared(
    *,
    data: pd.DataFrame,
    d: str,
    m: str,
    y: str,
    at_group: object,
) -> list[dict[str, Any]]:
    """Compute per-outcome positive-part partial PMF diff rows from prepared data."""
    y_values = _sort_support_levels(
        pd.unique(data.loc[data[m] == at_group, y]).tolist()
    )
    if not y_values:
        raise ValueError("at_group is not present in the mediator support.")

    arm_sizes = _arm_sizes(data, d=d)
    rows = []
    for y_value in y_values:
        p1 = _partial_mass(
            data,
            d=d,
            m=m,
            y=y,
            at_group=at_group,
            y_value=y_value,
            arm=1,
            arm_sizes=arm_sizes,
        )
        p0 = _partial_mass(
            data,
            d=d,
            m=m,
            y=y,
            at_group=at_group,
            y_value=y_value,
            arm=0,
            arm_sizes=arm_sizes,
        )
        delta = p1 - p0
        rows.append({
            "at_group": _normalize_mediator_level(at_group),
            "y_value": _normalize_scalar(y_value),
            "p_y_m_given_d1": float(p1),
            "p_y_m_given_d0": float(p0),
            "delta": float(delta),
            "positive_part_contribution": float(max(delta, 0.0)),
        })
    return rows


def lb_frac_affected(
    *,
    df: pd.DataFrame | None = None,
    data_path: str | Path | None = None,
    d: str,
    m: str | Sequence[str],
    y: str,
    at_group: object | None = None,
    num_y_bins: int | None = None,
    max_defiers_share: float = 0.0,
    allow_min_defiers: bool = False,
    return_min_defiers: bool = False,
    reg_formula: str | None = None,
) -> LowerBoundResult:
    """Compute a lower bound on the fraction of always-takers affected by treatment.

    Estimates a finite-support lower bound on the total variation distance between
    Y(1, m) and Y(0, m) for always-takers with M(1) = M(0) = m.  This
    equals a lower bound on the fraction of those always-takers for whom
    treatment has a direct effect on the outcome not mediated through M.

    When ``at_group`` is ``None``, the function returns a lower bound on
    the population-weighted average across all always-taker groups.

    Parameters
    ----------
    df : pd.DataFrame or None, optional
        Analysis data frame.  Exactly one of ``df`` or ``data_path`` must
        be provided.
    data_path : str, Path, or None, optional
        Path to a CSV file to load as the analysis data frame.
    d : str
        Name of the binary treatment column.
    m : str or sequence of str
        Name(s) of the mediator column(s).  For a scalar mediator pass a
        string; for a vector mediator pass a sequence of column names.
    y : str
        Name of the outcome column.  Must have finite-support values (or be
        discretized via ``num_y_bins``).
    at_group : object or None, optional
        Mediator level identifying the always-taker group of interest.
        If ``None`` (default), computes the pooled bound across all
        always-taker groups weighted by their population shares.
    num_y_bins : int or None, optional
        If provided, discretize the outcome into this many bins before
        computing the bound.
    max_defiers_share : float, default 0.0
        Maximum allowed share of defiers in the population.  Zero
        imposes strict ordered monotonicity.
    allow_min_defiers : bool, default False
        If ``True`` and the data require more defiers than
        ``max_defiers_share``, use the minimum compatible cap instead
        of raising an error.
    return_min_defiers : bool, default False
        If ``True``, return the minimum defier share compatible with
        the data rather than the lower bound itself.
    reg_formula : str or None, optional
        Regression formula for covariate-adjusted probability grids.
        If ``None``, empirical distributions are used.

    Returns
    -------
    LowerBoundResult
        Structured result containing:

        - ``lower_bound``: the numeric lower bound (or NaN if
          ``return_min_defiers=True``).
        - ``estimand``: ``"nu_k"`` (single group) or ``"bar_nu"``
          (pooled).
        - ``at_group``: the queried always-taker group.
        - ``restriction``: label for the active monotonicity assumption.
        - ``diagnostics``: rich dict with cell counts, theta values,
          defier-cap contract, and paper-inequality checks.

    Raises
    ------
    ValueError
        If columns are missing, treatment is not binary, mediator
        support is non-orderable, or the defier cap is incompatible
        and ``allow_min_defiers`` is ``False``.

    Examples
    --------
    >>> import pandas as pd
    >>> import testmechs
    >>> df = pd.DataFrame({
    ...     "d": [0, 0, 0, 1, 1, 1],
    ...     "m": [0, 1, 1, 0, 1, 1],
    ...     "y": [0, 1, 0, 1, 1, 0],
    ... })
    >>> result = testmechs.lb_frac_affected(df=df, d="d", m="m", y="y")
    >>> result.lower_bound  # numeric lower bound
    0.0

    Notes
    -----
    Implements the lower-bound estimator from Kwon and Roth (2026,
    Section 3) [1]_.  Uses linear programming for vector mediators or
    nonzero defier caps; uses the closed-form ordered-monotone formula
    for binary scalar mediators with zero defier share.

    References
    ----------
    .. [1] Kwon, S. and Roth, J. (2026). "Testing Mechanisms."
       The Review of Economic Studies, rdag028. doi:10.1093/restud/rdag028.

    See Also
    --------
    breakdown_defier_share : Defier share at which the bound loses bite.
    bounds_ade_ats : ADE bounds for always-taker groups.
    theta_kk_min_ordered_monotone : Underlying always-taker share computation.
    """

    max_defiers_share = _validate_defier_share(max_defiers_share)
    allow_min_defiers = _validate_bool_flag(allow_min_defiers, name="allow_min_defiers")
    return_min_defiers = _validate_bool_flag(return_min_defiers, name="return_min_defiers")
    regression_diagnostics = _regression_diagnostics(
        reg_formula=reg_formula,
        d=d,
        supported_scope=(
            "adjusted discrete probability grid for scalar ordered-monotone "
            "and vector elementwise-monotone lower bounds"
        ),
    )
    data, mediator = _prepare_bounds_dataframe(
        df=df,
        data_path=data_path,
        d=d,
        m=m,
        y=y,
        num_y_bins=num_y_bins,
        reg_formula=reg_formula,
    )
    m_key = mediator["column"]
    at_group = _coerce_at_group(at_group, mediator_columns=mediator["columns"])
    adjusted_m_argument: str | Sequence[str] = mediator["columns"] if mediator["vector"] else m_key
    adjusted_probabilities = (
        compute_adjusted_probabilities(
            df=data,
            d=d,
            m=adjusted_m_argument,
            y=y,
            reg_formula=reg_formula,
        )
        if reg_formula is not None
        else None
    )
    mediator_order = mediator["level_order"]
    if adjusted_probabilities is None:
        p0, p1 = _mediator_distributions(data, d=d, m=m_key, mediator_order=mediator_order)
    else:
        _validate_adjusted_probability_grid(adjusted_probabilities)
        p0, p1 = adjusted_probabilities.p_m_d0, adjusted_probabilities.p_m_d1
        regression_diagnostics = {
            **(regression_diagnostics or {}),
            **adjusted_probabilities.diagnostics,
            **mediator["treatment_support"].diagnostics(
                original_key="original_treatment_levels",
                normalized_key="normalized_treatment_levels",
            ),
        }
    levels = _sort_mediator_levels(set(p0) | set(p1), order=mediator_order)
    _validate_scalar_ordered_support(levels=levels, mediator=mediator)
    min_defiers = _minimum_compatible_defiers_share(
        p_m_given_d0=p0,
        p_m_given_d1=p1,
        mediator_order=mediator_order,
    )

    if return_min_defiers:
        query_cap = max(max_defiers_share, min_defiers)
        theta = _theta_kk_min_for_diagnostics(
            p0=p0,
            p1=p1,
            vector_mediator=mediator["vector"],
            max_defiers_share=query_cap,
            mediator_order=mediator_order,
        )
        base_diagnostics = _base_diagnostics(
            data=data,
            d=d,
            m=m_key,
            y=y,
            mediator=mediator,
            requested_num_y_bins=num_y_bins,
            theta=theta,
            requested_max_defiers_share=max_defiers_share,
            minimum_compatible_defiers_share=min_defiers,
            actual_max_defiers_share=query_cap,
            active_restriction="minimum-compatible-defiers-query",
        )
        if regression_diagnostics is not None:
            base_diagnostics["regression"] = regression_diagnostics
        return LowerBoundResult(
            lower_bound=float("nan"),
            estimand="minimum compatible defiers share",
            at_group=at_group,
            restriction="ordered mediator compatibility query",
            diagnostics=base_diagnostics,
        )

    actual_max_defiers_share = _resolve_defier_cap(
        requested_max_defiers_share=max_defiers_share,
        minimum_compatible_defiers_share=min_defiers,
        allow_min_defiers=allow_min_defiers,
    )
    theta = _theta_kk_min_for_diagnostics(
        p0=p0,
        p1=p1,
        vector_mediator=mediator["vector"],
        max_defiers_share=actual_max_defiers_share,
        mediator_order=mediator_order,
    )

    base_diagnostics = _base_diagnostics(
        data=data,
        d=d,
        m=m_key,
        y=y,
        mediator=mediator,
        requested_num_y_bins=num_y_bins,
        theta=theta,
        requested_max_defiers_share=max_defiers_share,
        minimum_compatible_defiers_share=min_defiers,
        actual_max_defiers_share=actual_max_defiers_share,
        active_restriction="ordered-monotone",
    )
    if regression_diagnostics is not None:
        base_diagnostics["regression"] = regression_diagnostics

    if mediator["vector"] or max_defiers_share > 0 or min_defiers > max_defiers_share + 1e-12:
        base_diagnostics["active_restriction"] = (
            "general-lp-elementwise-monotone-vector"
            if mediator["vector"] and actual_max_defiers_share <= 1e-12
            else "general-lp-minimum-compatible-defiers"
            if min_defiers > max_defiers_share + 1e-12
            else "general-lp-defier-cap"
        )
        return _general_lfp_lb_frac_affected(
            data=data,
            d=d,
            m=m_key,
            y=y,
            at_group=at_group,
            p0=p0,
            p1=p1,
            diagnostics=base_diagnostics,
            max_defiers_share=actual_max_defiers_share,
            adjusted_probabilities=adjusted_probabilities,
            mediator_order=mediator_order,
        )

    if at_group is None:
        return _pooled_lb_frac_affected(
            data=data,
            d=d,
            m=m_key,
            y=y,
            theta=theta,
            diagnostics=base_diagnostics,
            adjusted_probabilities=adjusted_probabilities,
            mediator_order=mediator_order,
        )

    if at_group not in levels:
        raise ValueError("at_group is not present in the mediator support.")

    lower_bound, group_diagnostics = _single_group_lower_bound(
        data=data,
        d=d,
        m=m_key,
        y=y,
        at_group=at_group,
        theta_kk_min=theta[at_group],
        p_m_given_d1=p1.get(at_group, 0.0),
        adjusted_probabilities=adjusted_probabilities,
    )
    target_group_result = {
        "at_group": _normalize_mediator_level(at_group),
        "lower_bound": lower_bound,
        **group_diagnostics,
        "in_objective": True,
        "objective_role": "target",
    }
    diagnostics = {
        **base_diagnostics,
        **group_diagnostics,
        "objective_levels": [_normalize_scalar(at_group)],
        "target_group_result": target_group_result,
    }
    return LowerBoundResult(
        lower_bound=lower_bound,
        estimand="nu_k",
        at_group=at_group,
        restriction="ordered monotonicity",
        diagnostics=diagnostics,
    )


def breakdown_defier_share(
    *,
    df: pd.DataFrame | None = None,
    data_path: str | Path | None = None,
    d: str,
    m: str | Sequence[str],
    y: str,
    at_group: object | None = None,
    num_y_bins: int | None = None,
    reg_formula: str | None = None,
    tol: float = 1e-4,
    max_iterations: int = 80,
) -> LowerBoundResult:
    """Find the defier-share breakdown point for the affected-fraction bound.

    Computes the minimum value of the defier cap ``max_defiers_share`` at which
    ``lb_frac_affected`` returns a lower bound of zero (within
    tolerance ``tol``).  This is the breakdown point: the smallest
    relaxation of monotonicity that eliminates evidence against the
    sharp null of full mediation.

    Uses binary search between the minimum compatible defier share
    and 1.0 to locate the threshold.

    Parameters
    ----------
    df : pd.DataFrame or None, optional
        Analysis data frame.  Exactly one of ``df`` or ``data_path``
        must be provided.
    data_path : str, Path, or None, optional
        Path to a CSV file to load.
    d : str
        Name of the binary treatment column.
    m : str or sequence of str
        Name(s) of the mediator column(s).
    y : str
        Name of the outcome column (finite support).
    at_group : object or None, optional
        Mediator level for the always-taker group.  If ``None``,
        uses the pooled bound.
    num_y_bins : int or None, optional
        Discretize outcome into this many bins.
    reg_formula : str or None, optional
        Regression formula for adjusted probability grids.
    tol : float, default 1e-4
        Convergence tolerance for the binary search.  The search
        terminates when the bracket width is below ``tol``.
    max_iterations : int, default 80
        Maximum binary-search iterations before raising.

    Returns
    -------
    LowerBoundResult
        Result with:

        - ``lower_bound``: the breakdown defier share (may be ``inf``
          if the bound never reaches zero).
        - ``estimand``: ``"breakdown defier share"``.
        - ``diagnostics``: bracket rows, iteration count, convergence
          status, and inherited lower-bound diagnostics.

    Raises
    ------
    ValueError
        If inputs are invalid or columns are missing.
    RuntimeError
        If the binary search does not converge within
        ``max_iterations``.

    Examples
    --------
    >>> import pandas as pd
    >>> import testmechs
    >>> df = pd.DataFrame({
    ...     "d": [0, 0, 0, 0, 1, 1, 1, 1],
    ...     "m": [0, 0, 1, 1, 0, 1, 1, 1],
    ...     "y": [0, 1, 0, 1, 0, 0, 1, 1],
    ... })
    >>> result = testmechs.breakdown_defier_share(
    ...     df=df, d="d", m="m", y="y"
    ... )
    >>> result.lower_bound  # doctest: +SKIP
    0.25

    Notes
    -----
    Corresponds to the R function ``breakdown_defier_share()`` in the
    TestMechs package.  The Python implementation computes the minimum
    compatible cap explicitly and uses exact binary search without
    epsilon relaxation.

    References
    ----------
    .. [1] Kwon, S. and Roth, J. (2026). "Testing Mechanisms."
       The Review of Economic Studies, rdag028. doi:10.1093/restud/rdag028.

    See Also
    --------
    lb_frac_affected : The lower bound that is driven to zero.
    """

    tol = _validate_positive_tolerance(tol, name="tol")
    max_iterations = _validate_positive_integer(max_iterations, name="max_iterations")

    min_defiers_result = lb_frac_affected(
        df=df,
        data_path=data_path,
        d=d,
        m=m,
        y=y,
        at_group=at_group,
        num_y_bins=num_y_bins,
        reg_formula=reg_formula,
        return_min_defiers=True,
    )
    minimum_compatible_defiers = float(
        min_defiers_result.diagnostics["minimum_compatible_defiers_share"]
    )

    def evaluate(cap: float) -> LowerBoundResult:
        """Evaluate the lower bound at the given defier cap."""
        return lb_frac_affected(
            df=df,
            data_path=data_path,
            d=d,
            m=m,
            y=y,
            at_group=at_group,
            num_y_bins=num_y_bins,
            reg_formula=reg_formula,
            max_defiers_share=cap,
            allow_min_defiers=False,
        )

    lower_cap = minimum_compatible_defiers
    lower_result = evaluate(lower_cap)
    lower_value = float(lower_result.lower_bound)
    upper_result = evaluate(1.0)
    upper_value = float(upper_result.lower_bound)
    iterations = 0
    bracket_rows: list[dict[str, float | str]] = [
        {"role": "minimum_compatible", "cap": lower_cap, "lower_bound": lower_value},
        {"role": "unit_cap", "cap": 1.0, "lower_bound": upper_value},
    ]

    if lower_value <= tol:
        breakdown = lower_cap
        final_value = lower_value
        status = "minimum_compatible_cap_already_breaks_down"
        final_bracket_lower_cap = lower_cap
        final_bracket_upper_cap = lower_cap
        final_bracket_width = 0.0
    elif upper_value > tol:
        breakdown = float("inf")
        final_value = upper_value
        status = "lower_bound_positive_even_at_unit_defier_cap"
        final_bracket_lower_cap = 1.0
        final_bracket_upper_cap = None
        final_bracket_width = None
    else:
        upper_cap = 1.0
        while upper_cap - lower_cap > tol and iterations < max_iterations:
            iterations += 1
            mid_cap = 0.5 * (lower_cap + upper_cap)
            mid_result = evaluate(mid_cap)
            mid_value = float(mid_result.lower_bound)
            bracket_rows.append(
                {"role": "binary_search", "cap": mid_cap, "lower_bound": mid_value}
            )
            if mid_value > tol:
                lower_cap = mid_cap
                lower_value = mid_value
            else:
                upper_cap = mid_cap
                upper_value = mid_value
        if upper_cap - lower_cap > tol:
            raise RuntimeError(
                "breakdown_defier_share did not converge within max_iterations; "
                f"bracket_width={upper_cap - lower_cap:.12g}, tol={tol:.12g}, "
                f"lower_cap={lower_cap:.12g}, upper_cap={upper_cap:.12g}, "
                f"max_iterations={max_iterations}."
            )
        breakdown = upper_cap
        final_value = upper_value
        status = "bounded_by_binary_search"
        final_bracket_lower_cap = lower_cap
        final_bracket_upper_cap = upper_cap
        final_bracket_width = upper_cap - lower_cap

    diagnostics = {
        **min_defiers_result.diagnostics,
        "active_restriction": "breakdown-defier-share",
        "breakdown_defier_share": None if np.isinf(breakdown) else float(breakdown),
        "breakdown_defier_share_is_finite": bool(np.isfinite(breakdown)),
        "breakdown_defier_share_nonfinite": "positive_infinity"
        if np.isposinf(breakdown)
        else None,
        "breakdown_lower_bound_at_cap": float(final_value),
        "breakdown_tolerance": float(tol),
        "breakdown_status": status,
        "breakdown_bracket_lower_cap": float(final_bracket_lower_cap),
        "breakdown_bracket_upper_cap": (
            None if final_bracket_upper_cap is None else float(final_bracket_upper_cap)
        ),
        "breakdown_bracket_width": (
            None if final_bracket_width is None else float(final_bracket_width)
        ),
        "minimum_compatible_defiers_share": minimum_compatible_defiers,
        "iterations": int(iterations),
        "max_iterations": int(max_iterations),
        "bracket_rows": bracket_rows,
        "reference_boundary": (
            "packages/r/TestMechs/R/lb_frac_affected.R:631-695 uses a binary search over "
            "max_defiers_share; Python computes the minimum-compatible cap explicitly "
            "and never passes dangling defier-cap promises."
        ),
        "paper_reference": (
            "manuscript/sources/arxiv-2404.11739v3/draft.tex:58,183-185 describes relaxations of "
            "monotonicity by allowing a bounded defier share."
        ),
    }
    return LowerBoundResult(
        lower_bound=breakdown,
        estimand="breakdown defier share",
        at_group=at_group,
        restriction="bounded defier-share relaxation",
        diagnostics=diagnostics,
    )


def bounds_ade_ats(
    *,
    df: pd.DataFrame | None = None,
    data_path: str | Path | None = None,
    d: str,
    m: str | Sequence[str],
    y: str,
    at_group: object = 1,
    max_defiers_share: float = 0.0,
    allow_min_defiers: bool = False,
    reg_formula: str | None = None,
) -> ADEBoundsResult:
    """Compute bounds on the average direct effect for an always-taker group.

    Estimates sharp bounds on E[Y(1,k) - Y(0,k) | G = kk], the average
    direct effect of treatment on the outcome for k-always-takers (units
    with M(1) = M(0) = k).  Uses Lee-style trimming on the conditional
    outcome distribution within the identified always-taker subpopulation.

    Parameters
    ----------
    df : pd.DataFrame or None, optional
        Analysis data frame.  Exactly one of ``df`` or ``data_path``
        must be provided.
    data_path : str, Path, or None, optional
        Path to a CSV file to load.
    d : str
        Name of the binary treatment column.
    m : str or sequence of str
        Name(s) of the mediator column(s).
    y : str
        Name of the outcome column.  Must be numeric with finite values
        (no discretization via ``num_y_bins``; ADE bounds require
        raw numeric outcomes).
    at_group : object, default 1
        Mediator level identifying the always-taker group for which
        the ADE bounds are computed.
    max_defiers_share : float, default 0.0
        Maximum allowed proportion of defiers.  Zero imposes strict
        ordered monotonicity.
    allow_min_defiers : bool, default False
        If ``True``, automatically use the minimum compatible defier
        cap when the data are incompatible with ``max_defiers_share``.
    reg_formula : str or None, optional
        Regression formula for covariate-adjusted probability grids.

    Returns
    -------
    ADEBoundsResult
        Structured result containing:

        - ``lower_bound``: lower endpoint of the ADE bound interval
          (``None`` if no bite).
        - ``upper_bound``: upper endpoint (``None`` if no bite).
        - ``at_group``: the queried always-taker group.
        - ``restriction``: the active monotonicity label.
        - ``diagnostics``: theta values, trimming fractions,
          no-bite flags, and paper-inequality checks.

    Raises
    ------
    ValueError
        If columns are missing, outcome is non-numeric, treatment is
        not binary, or the defier cap is incompatible.

    Examples
    --------
    >>> import pandas as pd
    >>> import testmechs
    >>> df = pd.DataFrame({
    ...     "d": [0, 0, 0, 0, 1, 1, 1, 1],
    ...     "m": [0, 0, 1, 1, 0, 1, 1, 1],
    ...     "y": [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0],
    ... })
    >>> result = testmechs.bounds_ade_ats(
    ...     df=df, d="d", m="m", y="y", at_group=1
    ... )
    >>> result.lower_bound  # doctest: +SKIP
    -0.5
    >>> result.upper_bound  # doctest: +SKIP
    0.5

    Notes
    -----
    Implements the ADE bounds from Kwon and Roth (2026, Section 4) [1]_.
    The trimming fraction equals theta_{kk}^{min} / P(M=k | D=d).
    When theta_{kk}^{min} is zero the bounds have no bite and
    ``None`` is returned for both endpoints.

    References
    ----------
    .. [1] Kwon, S. and Roth, J. (2026). "Testing Mechanisms."
       The Review of Economic Studies, rdag028. doi:10.1093/restud/rdag028.

    See Also
    --------
    lb_frac_affected : Lower bound on the affected fraction.
    breakdown_defier_share : Breakdown analysis for defier caps.
    """

    max_defiers_share = _validate_defier_share(max_defiers_share)
    allow_min_defiers = _validate_bool_flag(allow_min_defiers, name="allow_min_defiers")
    regression_diagnostics = _validate_bounds_ade_regression_scope(reg_formula=reg_formula, d=d)
    data, mediator = _prepare_bounds_dataframe(
        df=df,
        data_path=data_path,
        d=d,
        m=m,
        y=y,
        num_y_bins=None,
        reg_formula=reg_formula,
    )
    _validate_bounds_ade_numeric_outcome(data[y], column=y)
    m_key = mediator["column"]
    at_group = _coerce_at_group(at_group, mediator_columns=mediator["columns"])
    adjusted_m_argument: str | Sequence[str] = mediator["columns"] if mediator["vector"] else m_key
    adjusted_probabilities = (
        compute_adjusted_probabilities(
            df=data,
            d=d,
            m=adjusted_m_argument,
            y=y,
            reg_formula=reg_formula,
        )
        if reg_formula is not None
        else None
    )
    mediator_order = mediator["level_order"]
    if adjusted_probabilities is None:
        p0, p1 = _mediator_distributions(data, d=d, m=m_key, mediator_order=mediator_order)
    else:
        _validate_adjusted_probability_grid(adjusted_probabilities)
        p0, p1 = adjusted_probabilities.p_m_d0, adjusted_probabilities.p_m_d1
        regression_diagnostics = {
            **(regression_diagnostics or {}),
            **adjusted_probabilities.diagnostics,
            **mediator["treatment_support"].diagnostics(
                original_key="original_treatment_levels",
                normalized_key="normalized_treatment_levels",
            ),
            "adjusted_probability_contract": (
                "joint adjusted P(Y=y,M=m|D=d) grid, normalized within "
                "the requested mediator group for Lee-style ADE trimming"
            ),
        }
    levels = _sort_mediator_levels(set(p0) | set(p1), order=mediator_order)
    _validate_scalar_ordered_support(levels=levels, mediator=mediator)
    if at_group not in levels:
        raise ValueError("at_group is not present in the mediator support.")

    min_defiers = _minimum_compatible_defiers_share(
        p_m_given_d0=p0,
        p_m_given_d1=p1,
        mediator_order=mediator_order,
    )
    actual_max_defiers_share = _resolve_defier_cap(
        requested_max_defiers_share=max_defiers_share,
        minimum_compatible_defiers_share=min_defiers,
        allow_min_defiers=allow_min_defiers,
    )
    uses_general_theta_lp = mediator["vector"] or actual_max_defiers_share > 1e-12
    theta_lp_diagnostics: dict[str, Any] | None = None
    if uses_general_theta_lp:
        theta_lp = _general_theta_kk_min_for_ade(
            data=data,
            d=d,
            m=m_key,
            y=y,
            at_group=at_group,
            p0=p0,
            p1=p1,
            max_defiers_share=actual_max_defiers_share,
            adjusted_probabilities=adjusted_probabilities,
            mediator_order=mediator_order,
        )
        theta = theta_lp["theta_by_group"]
        theta_lp_diagnostics = theta_lp["diagnostics"]
    else:
        theta = _theta_kk_min_for_diagnostics(
            p0=p0,
            p1=p1,
            vector_mediator=mediator["vector"],
            max_defiers_share=actual_max_defiers_share,
            mediator_order=mediator_order,
        )
    theta_kk_min = theta[at_group]
    diagnostics = _base_diagnostics(
        data=data,
        d=d,
        m=m_key,
        y=y,
        mediator=mediator,
        requested_num_y_bins=None,
        theta=theta,
        requested_max_defiers_share=max_defiers_share,
        minimum_compatible_defiers_share=min_defiers,
        actual_max_defiers_share=actual_max_defiers_share,
        active_restriction=(
            "elementwise-monotone-vector"
            if mediator["vector"] and actual_max_defiers_share <= 1e-12
            else "general-lp-minimum-compatible-defiers"
            if min_defiers > max_defiers_share + 1e-12
            else "general-lp-defier-cap"
            if actual_max_defiers_share > 1e-12
            else "ordered-monotone"
        ),
    )
    if regression_diagnostics is not None:
        diagnostics["regression"] = regression_diagnostics
    if theta_lp_diagnostics is not None:
        diagnostics["general_theta_lp"] = theta_lp_diagnostics
    diagnostics["theta_kk_min"] = theta_kk_min
    diagnostics["objective_levels"] = [_normalize_scalar(at_group)]
    diagnostics["outcome_support_contract"] = (
        "raw finite numeric outcome support for Lee-style ADE trimming; "
        "num_y_bins is not an ADE argument"
    )

    if theta_kk_min <= 1e-12:
        diagnostics["no_bite"] = {
            "flag": True,
            "theta_kk_min": 0.0,
            "reason": "theta_kk_min is zero",
        }
        diagnostics["target_group_result"] = _ade_target_group_result(
            at_group=at_group,
            lower_bound=None,
            upper_bound=None,
            theta_kk_min=0.0,
            p_mk_d1=p1.get(at_group, 0.0),
            p_mk_d0=p0.get(at_group, 0.0),
            check_theta=None,
            no_bite=diagnostics["no_bite"],
            unsupported=None,
        )
        return ADEBoundsResult(
            lower_bound=None,
            upper_bound=None,
            at_group=at_group,
            restriction=_ade_restriction_label(
                vector_mediator=mediator["vector"],
                max_defiers_share=actual_max_defiers_share,
                minimum_compatible_defiers_share=min_defiers,
                requested_max_defiers_share=max_defiers_share,
            ),
            diagnostics=diagnostics,
        )

    p_mk_d1 = p1.get(at_group, 0.0)
    p_mk_d0 = p0.get(at_group, 0.0)
    if p_mk_d1 <= 0.0 or p_mk_d0 <= 0.0:
        diagnostics["no_bite"] = {
            "flag": False,
            "theta_kk_min": theta_kk_min,
            "reason": None,
        }
        diagnostics["unsupported"] = {
            "flag": True,
            "reason": "P(M=at_group | D=d) is zero for some treatment arm",
        }
        diagnostics["target_group_result"] = _ade_target_group_result(
            at_group=at_group,
            lower_bound=None,
            upper_bound=None,
            theta_kk_min=theta_kk_min,
            p_mk_d1=p_mk_d1,
            p_mk_d0=p_mk_d0,
            check_theta=None,
            no_bite=diagnostics.get("no_bite"),
            unsupported=diagnostics["unsupported"],
        )
        return ADEBoundsResult(
            lower_bound=None,
            upper_bound=None,
            at_group=at_group,
            restriction=_ade_restriction_label(
                vector_mediator=mediator["vector"],
                max_defiers_share=actual_max_defiers_share,
                minimum_compatible_defiers_share=min_defiers,
                requested_max_defiers_share=max_defiers_share,
            ),
            diagnostics=diagnostics,
        )

    check_theta_d1 = theta_kk_min / p_mk_d1
    check_theta_d0 = theta_kk_min / p_mk_d0
    if adjusted_probabilities is None:
        y_d1 = data.loc[(data[d] == 1) & (data[m_key] == at_group), y]
        y_d0 = data.loc[(data[d] == 0) & (data[m_key] == at_group), y]

        y1_lower = _trimmed_expectation(y_d1, frac=check_theta_d1, upper=False)
        y1_upper = _trimmed_expectation(y_d1, frac=check_theta_d1, upper=True)
        y0_lower = _trimmed_expectation(y_d0, frac=check_theta_d0, upper=False)
        y0_upper = _trimmed_expectation(y_d0, frac=check_theta_d0, upper=True)
    else:
        adjusted_trimming = _adjusted_ade_trimmed_expectations(
            adjusted_probabilities=adjusted_probabilities,
            at_group=at_group,
            check_theta_d1=check_theta_d1,
            check_theta_d0=check_theta_d0,
        )
        y1_lower = adjusted_trimming["y1_lower"]
        y1_upper = adjusted_trimming["y1_upper"]
        y0_lower = adjusted_trimming["y0_lower"]
        y0_upper = adjusted_trimming["y0_upper"]
        diagnostics["adjusted_trimming"] = adjusted_trimming["diagnostics"]
    diagnostics["check_theta"] = {"d1": check_theta_d1, "d0": check_theta_d0}
    diagnostics["no_bite"] = {
        "flag": False,
        "theta_kk_min": theta_kk_min,
        "reason": None,
    }
    lower_bound = y1_lower - y0_upper
    upper_bound = y1_upper - y0_lower
    diagnostics["target_group_result"] = _ade_target_group_result(
        at_group=at_group,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        theta_kk_min=theta_kk_min,
        p_mk_d1=p_mk_d1,
        p_mk_d0=p_mk_d0,
        check_theta=diagnostics["check_theta"],
        no_bite=diagnostics["no_bite"],
        unsupported=None,
    )

    return ADEBoundsResult(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        at_group=at_group,
        restriction=_ade_restriction_label(
            vector_mediator=mediator["vector"],
            max_defiers_share=actual_max_defiers_share,
            minimum_compatible_defiers_share=min_defiers,
            requested_max_defiers_share=max_defiers_share,
        ),
        diagnostics=diagnostics,
    )


def _ade_target_group_result(
    *,
    at_group: object,
    lower_bound: float | None,
    upper_bound: float | None,
    theta_kk_min: float,
    p_mk_d1: float,
    p_mk_d0: float,
    check_theta: Mapping[str, float] | None,
    no_bite: Mapping[str, Any] | None,
    unsupported: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Assemble the target-group diagnostic record for ADE bounds."""
    result: dict[str, Any] = {
        "at_group": _normalize_scalar(at_group),
        "lower_bound": None if lower_bound is None else float(lower_bound),
        "upper_bound": None if upper_bound is None else float(upper_bound),
        "theta_kk_min": float(theta_kk_min),
        "p_mk_given_d1": float(p_mk_d1),
        "p_mk_given_d0": float(p_mk_d0),
        "in_objective": True,
        "objective_role": "target",
        "no_bite": no_bite,
    }
    if check_theta is not None:
        result["check_theta"] = dict(check_theta)
    if unsupported is not None:
        result["unsupported"] = unsupported
    return result


def _regression_diagnostics(
    *,
    reg_formula: str | None,
    d: str,
    supported_scope: str,
) -> dict[str, Any] | None:
    """Parse regression formula and return diagnostics dict, or None if no formula."""
    if reg_formula is None:
        return None
    spec = parse_reg_formula(reg_formula, d=d)
    return {
        "formula": reg_formula,
        "formula_kind": spec.formula_kind,
        "supported_scope": supported_scope,
    }


def _validate_bounds_ade_regression_scope(*, reg_formula: str | None, d: str) -> dict[str, Any] | None:
    """Build regression diagnostics scoped to ADE bounds."""
    diagnostics = _regression_diagnostics(
        reg_formula=reg_formula,
        d=d,
        supported_scope=(
            "adjusted discrete probability grid for scalar ordered-monotone "
            "and vector elementwise-monotone ADE bounds"
        ),
    )
    return diagnostics


def _validate_regression_scope(*, reg_formula: str | None, d: str) -> dict[str, Any] | None:
    """Validate regression scope; currently only trivial formula is supported."""
    if reg_formula is None:
        return None
    spec = parse_reg_formula(reg_formula, d=d)
    diagnostics = {
        "formula": reg_formula,
        "formula_kind": spec.formula_kind,
        "supported_scope": "trivial equivalence for ordered-monotone bounds",
    }
    if spec.formula_kind != "trivial":
        raise NotImplementedError(
            "Bounds regression adjustment is currently implemented only for reg_formula='~ treatment'."
        )
    return diagnostics


def _coerce_distribution(
    distribution: Mapping[object, float] | Sequence[float],
    *,
    name: str,
) -> dict[object, float]:
    """Coerce a mapping or sequence into a validated probability distribution dict."""
    if isinstance(distribution, Mapping):
        items = list(distribution.items())
    else:
        items = list(enumerate(distribution))
    if not items:
        raise ValueError(
            f"{name} must be a probability distribution with finite nonnegative masses that sum to 1."
        )
    coerced: dict[object, float] = {}
    for key, value in items:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise ValueError(
                f"{name} must be a probability distribution with finite nonnegative masses that sum to 1."
            )
        mass = float(value)
        if not np.isfinite(mass) or mass < 0.0:
            raise ValueError(
                f"{name} must be a probability distribution with finite nonnegative masses that sum to 1."
            )
        coerced[key] = mass
    total = float(sum(coerced.values()))
    if not np.isclose(total, 1.0, rtol=1e-10, atol=1e-10):
        raise ValueError(
            f"{name} must be a probability distribution with finite nonnegative masses that sum to 1."
        )
    return coerced


def _mediator_columns(m: str | Sequence[str]) -> tuple[str, ...]:
    """Normalise the mediator argument to a tuple of column name strings."""
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


def _validate_scalar_column_name(value: object, *, name: str) -> str:
    """Ensure *value* is a non-empty string suitable as a column name."""
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{name} must name one scalar DataFrame column.")
    return value


def _coerce_at_group(at_group: object | None, *, mediator_columns: Sequence[str]) -> object | None:
    """Normalise and validate the at_group argument for scalar or vector mediators."""
    if at_group is None:
        return None
    if len(mediator_columns) == 1:
        return _normalize_mediator_level(at_group)
    if isinstance(at_group, tuple):
        values = at_group
    elif isinstance(at_group, Sequence) and not isinstance(at_group, (str, bytes)):
        values = tuple(at_group)
    else:
        raise TypeError(
            "Vector mediator lower bounds require at_group to be a sequence "
            "with one value per mediator column."
        )
    if len(values) != len(mediator_columns):
        raise ValueError("at_group must contain one value per mediator column.")
    return _normalize_mediator_level(tuple(values))


def _validate_binary_treatment(series: pd.Series, column: str) -> None:
    """Raise ValueError if treatment column does not have exactly {0, 1} levels."""
    levels = set(series.dropna().unique().tolist())
    if levels != {0, 1}:
        raise ValueError(f"{column} must contain exactly the binary treatment levels 0 and 1.")


def _arm_sizes(data: pd.DataFrame, *, d: str) -> dict[int, int]:
    """Return sample sizes per treatment arm; raises if either arm is empty."""
    sizes = {arm: int((data[d] == arm).sum()) for arm in (0, 1)}
    if sizes[0] == 0 or sizes[1] == 0:
        raise ValueError("Both treatment arms must contain observations.")
    return sizes


def _partial_mass(
    data: pd.DataFrame,
    *,
    d: str,
    m: str,
    y: str,
    at_group: object,
    y_value: object,
    arm: int,
    arm_sizes: dict[int, int],
) -> float:
    """Compute P(Y=y_value, M=at_group | D=arm) from prepared data."""
    mask = (data[d] == arm) & (data[m] == at_group) & (data[y] == y_value)
    return float(mask.sum() / arm_sizes[arm])


def _mediator_distributions(
    data: pd.DataFrame,
    *,
    d: str,
    m: str,
    mediator_order: Sequence[object] | None = None,
) -> tuple[dict[object, float], dict[object, float]]:
    """Compute empirical P(M=level | D=0) and P(M=level | D=1) distributions."""
    sizes = _arm_sizes(data, d=d)
    levels = _sort_mediator_levels(pd.unique(data[m]).tolist(), order=mediator_order)
    p0 = {level: float(((data[d] == 0) & (data[m] == level)).sum() / sizes[0]) for level in levels}
    p1 = {level: float(((data[d] == 1) & (data[m] == level)).sum() / sizes[1]) for level in levels}
    return p0, p1


def _minimum_compatible_defiers_share(
    *,
    p_m_given_d0: Mapping[object, float],
    p_m_given_d1: Mapping[object, float],
    mediator_order: Sequence[object] | None = None,
) -> float:
    """Find the minimum defier share compatible with observed mediator marginals via LP."""
    levels = _sort_mediator_levels(set(p_m_given_d0) | set(p_m_given_d1), order=mediator_order)
    types = _build_type_pairs(levels)
    type_count = len(types)
    a_eq = []
    b_eq = []
    for level in levels:
        row = [1.0 if m1_value == level else 0.0 for _, m1_value in types]
        a_eq.append(row)
        b_eq.append(float(p_m_given_d1.get(level, 0.0)))
    for level in levels:
        row = [1.0 if m0_value == level else 0.0 for m0_value, _ in types]
        a_eq.append(row)
        b_eq.append(float(p_m_given_d0.get(level, 0.0)))
    c = np.asarray(
        [
            1.0 if _is_defier_type(m0_value, m1_value, order=mediator_order) else 0.0
            for m0_value, m1_value in types
        ],
        dtype=float,
    )
    solution = linprog(
        c=c,
        A_eq=np.asarray(a_eq, dtype=float),
        b_eq=np.asarray(b_eq, dtype=float),
        bounds=(0.0, None),
        method="highs",
    )
    if not solution.success:
        raise ValueError(f"Unable to compute the minimum compatible defier share: {solution.message}")
    return float(max(solution.fun, 0.0))


def _theta_kk_min_for_diagnostics(
    *,
    p0: Mapping[object, float],
    p1: Mapping[object, float],
    vector_mediator: bool,
    max_defiers_share: float,
    mediator_order: Sequence[object] | None = None,
) -> dict[object, float]:
    """Compute theta_{kk}^{min} for each level; dispatches to closed-form or LP."""
    if not vector_mediator and max_defiers_share <= 1e-12:
        theta_order = (
            tuple(_sort_mediator_levels(set(p0) | set(p1), order=mediator_order))
            if mediator_order is None
            else mediator_order
        )
        return theta_kk_min_ordered_monotone(
            p_m_given_d0=p0,
            p_m_given_d1=p1,
            mediator_order=theta_order,
        )
    levels = _sort_mediator_levels(set(p0) | set(p1), order=mediator_order)
    types = _build_type_pairs(levels)
    type_count = len(types)
    a_eq = []
    b_eq = []
    for level in levels:
        a_eq.append([1.0 if m1_value == level else 0.0 for _, m1_value in types])
        b_eq.append(float(p1.get(level, 0.0)))
    for level in levels:
        a_eq.append([1.0 if m0_value == level else 0.0 for m0_value, _ in types])
        b_eq.append(float(p0.get(level, 0.0)))
    a_ub = [
        [
            1.0 if _is_defier_type(m0_value, m1_value, order=mediator_order) else 0.0
            for m0_value, m1_value in types
        ]
    ]
    b_ub = [float(max_defiers_share)]
    theta: dict[object, float] = {}
    for level in levels:
        c = np.asarray([1.0 if (m0_value == level and m1_value == level) else 0.0 for m0_value, m1_value in types], dtype=float)
        solution = linprog(
            c=c,
            A_eq=np.asarray(a_eq, dtype=float),
            b_eq=np.asarray(b_eq, dtype=float),
            A_ub=np.asarray(a_ub, dtype=float),
            b_ub=np.asarray(b_ub, dtype=float),
            bounds=(0.0, None),
            method="highs",
        )
        if not solution.success:
            raise ValueError(f"Unable to compute theta_kk minimum diagnostics: {solution.message}")
        theta[level] = float(max(solution.fun, 0.0))
    return theta


def _base_diagnostics(
    *,
    data: pd.DataFrame,
    d: str,
    m: str,
    y: str,
    mediator: Mapping[str, Any],
    requested_num_y_bins: int | None,
    theta: Mapping[object, float],
    requested_max_defiers_share: float,
    minimum_compatible_defiers_share: float,
    active_restriction: str,
    actual_max_defiers_share: float | None = None,
) -> dict[str, Any]:
    """Assemble the common diagnostics dict shared by all bound estimators."""
    finite_sample = build_cell_count_diagnostics(
        df=data,
        d=d,
        m=m,
        y=y,
        cluster=None,
        requested_num_y_bins=requested_num_y_bins,
        applied_num_y_bins=int(data[y].nunique(dropna=False)),
        support_diagnostics=mediator["treatment_support"].diagnostics(
            original_key="original_treatment_levels",
            normalized_key="normalized_treatment_levels",
        ),
    )
    if mediator["vector"] and finite_sample["cell_counts"] and m in finite_sample["cell_counts"][0]:  # type: ignore[index]
        finite_sample["cell_counts"] = [
            {**{key: value for key, value in record.items() if key != m}, "m": record[m]}
            for record in finite_sample["cell_counts"]  # type: ignore[attr-defined]
        ]
        finite_sample["cluster_counts"] = [
            {**{key: value for key, value in record.items() if key != m}, "m": record[m]}
            for record in finite_sample["cluster_counts"]  # type: ignore[attr-defined]
        ]
    mediator_order = mediator["level_order"]
    levels = _sort_mediator_levels(pd.unique(data[m]).tolist(), order=mediator_order)
    types = _build_type_pairs(levels)
    actual_cap = float(
        requested_max_defiers_share if actual_max_defiers_share is None else actual_max_defiers_share
    )
    return {
        **finite_sample,
        "mediator_levels": [_normalize_mediator_level(value) for value in levels],
        "mediator_columns": list(mediator["columns"]),
        "mediator_dimension": len(mediator["columns"]),
        "vector_mediator": bool(mediator["vector"]),
        "support_normalization": (
            "vector mediator support is mapped to deterministic tuple values"
            if mediator["vector"]
            else "ordered categorical mediator support is preserved using declared category order"
            if mediator_order is not None
            else "scalar mediator support labels are preserved and ordered deterministically"
        ),
        "ordered_mediator_levels": (
            None
            if mediator["vector"]
            else (
                [_normalize_mediator_level(value) for value in mediator_order]
                if mediator_order is not None
                else None
            )
        ),
        "mediator_support_ordering": (
            "tuple support is ordered componentwise using natural comparisons where possible "
            "and deterministic label keys otherwise"
            if mediator["vector"]
            else "scalar ordered categorical support uses the declared category order"
            if mediator_order is not None
            else "scalar support is ordered using natural comparisons where possible and "
            "deterministic label keys otherwise"
        ),
        "ordered_categorical_mediator": bool(
            (not mediator["vector"]) and mediator_order is not None
        ),
        "treatment_support_normalization": (
            "binary treatment support is mapped to internal {0, 1} in deterministic support order"
        ),
        "allowed_mediator_type_pairs": [
            {"from": _normalize_mediator_level(low), "to": _normalize_mediator_level(high)}
            for low, high in types
            if _mediator_leq(low, high, order=mediator_order)
        ],
        "allowed_mediator_type_count": sum(
            1 for low, high in types if _mediator_leq(low, high, order=mediator_order)
        ),
        "forbidden_mediator_type_count": sum(
            1 for low, high in types if not _mediator_leq(low, high, order=mediator_order)
        ),
        "theta_kk_min_by_group": {
            _normalize_mediator_level(key): float(value) for key, value in theta.items()
        },
        "theta_kk_min_rows": [
            {
                "at_group": _normalize_mediator_level(key),
                "theta_kk_min": float(value),
            }
            for key, value in sorted(
                theta.items(),
                key=lambda item: _mediator_sort_key(item[0], order=mediator_order),
            )
        ],
        "requested_max_defiers_share": float(requested_max_defiers_share),
        "minimum_compatible_defiers_share": float(minimum_compatible_defiers_share),
        "actual_max_defiers_share": actual_cap,
        "defier_cap_contract": _defier_cap_contract(
            requested_max_defiers_share=requested_max_defiers_share,
            minimum_compatible_defiers_share=minimum_compatible_defiers_share,
            actual_max_defiers_share=actual_cap,
        ),
        "active_restriction": active_restriction,
    }


def _defier_cap_contract(
    *,
    requested_max_defiers_share: float,
    minimum_compatible_defiers_share: float,
    actual_max_defiers_share: float,
) -> dict[str, Any]:
    """Build the defier-cap contract diagnostic recording source and relaxation."""
    exact_minimum_source = (
        minimum_compatible_defiers_share > requested_max_defiers_share + 1e-12
        and abs(actual_max_defiers_share - minimum_compatible_defiers_share) <= 1e-12
    )
    return {
        "requested_max_defiers_share": float(requested_max_defiers_share),
        "minimum_compatible_defiers_share": float(minimum_compatible_defiers_share),
        "actual_max_defiers_share": float(actual_max_defiers_share),
        "actual_cap_source": (
            "minimum_compatible_defiers_share"
            if exact_minimum_source
            else "requested_max_defiers_share"
        ),
        "epsilon_relaxation": 0.0,
        "reference_boundary": (
            "R TestMechs adds 1e-6 when allow_min_defiers=True; Python uses "
            "the exact minimum compatible cap and reports it explicitly."
        ),
    }


def _single_group_lower_bound(
    *,
    data: pd.DataFrame,
    d: str,
    m: str,
    y: str,
    at_group: object,
    theta_kk_min: float,
    p_m_given_d1: float,
    adjusted_probabilities: AdjustedProbabilityResult | None = None,
) -> tuple[float, dict[str, Any]]:
    """Compute the lower bound for a single always-taker group and return diagnostics."""
    positive_part_cell_rows = (
        _adjusted_positive_part_partial_pmf_diff_rows(
            adjusted_probabilities=adjusted_probabilities,
            at_group=at_group,
        )
        if adjusted_probabilities is not None
        else _positive_part_partial_pmf_diff_rows_from_prepared(
            data=data,
            d=d,
            m=m,
            y=y,
            at_group=at_group,
        )
    )
    positive_diff = float(
        sum(row["positive_part_contribution"] for row in positive_part_cell_rows)
    )
    max_complier_share_to_group = max(p_m_given_d1 - theta_kk_min, 0.0)
    if theta_kk_min <= 1e-12:
        paper_inequality = _paper_inequality_diagnostics(
            theta_kk=0.0,
            lower_bound=0.0,
            positive_part_partial_pmf_diff=positive_diff,
            max_complier_share_to_group=max_complier_share_to_group,
        )
        return 0.0, {
            "theta_kk_min": 0.0,
            "positive_part_partial_pmf_diff": float(positive_diff),
            "positive_part_cell_rows": positive_part_cell_rows,
            "max_complier_share_to_group": float(max_complier_share_to_group),
            **paper_inequality,
            "no_bite": {
                "flag": True,
                "theta_kk_min": 0.0,
                "reason": "theta_kk_min is zero",
            },
        }

    lower_bound = max((positive_diff - max_complier_share_to_group) / theta_kk_min, 0.0)
    lower_bound = min(lower_bound, 1.0)
    paper_inequality = _paper_inequality_diagnostics(
        theta_kk=theta_kk_min,
        lower_bound=lower_bound,
        positive_part_partial_pmf_diff=positive_diff,
        max_complier_share_to_group=max_complier_share_to_group,
    )
    return float(lower_bound), {
        "theta_kk_min": float(theta_kk_min),
        "positive_part_partial_pmf_diff": float(positive_diff),
        "positive_part_cell_rows": positive_part_cell_rows,
        "max_complier_share_to_group": float(max_complier_share_to_group),
        **paper_inequality,
        "no_bite": {
            "flag": False,
            "theta_kk_min": float(theta_kk_min),
            "reason": None,
        },
    }


def _paper_inequality_diagnostics(
    *,
    theta_kk: float,
    lower_bound: float,
    positive_part_partial_pmf_diff: float,
    max_complier_share_to_group: float,
) -> dict[str, float]:
    """Compute LHS, RHS, gap, and violation of the paper's identifying inequality."""
    lhs = float(theta_kk * lower_bound)
    rhs = float(max(positive_part_partial_pmf_diff - max_complier_share_to_group, 0.0))
    return {
        "paper_inequality_lhs": lhs,
        "paper_inequality_rhs": rhs,
        "paper_inequality_gap": float(lhs - rhs),
        "paper_inequality_violation": float(max(rhs - lhs, 0.0)),
    }


def _pooled_lb_frac_affected(
    *,
    data: pd.DataFrame,
    d: str,
    m: str,
    y: str,
    theta: Mapping[object, float],
    diagnostics: dict[str, Any],
    adjusted_probabilities: AdjustedProbabilityResult | None = None,
    mediator_order: Sequence[object] | None = None,
) -> LowerBoundResult:
    """Compute the pooled (weighted-average) lower bound across all AT groups."""
    if len(theta) != 2:
        if adjusted_probabilities is None:
            p0, p1 = _mediator_distributions(data, d=d, m=m, mediator_order=mediator_order)
        else:
            p0, p1 = adjusted_probabilities.p_m_d0, adjusted_probabilities.p_m_d1
        general_diagnostics = {
            **diagnostics,
            "active_restriction": "general-lp-shared-feasible-set",
        }
        return _general_lfp_lb_frac_affected(
            data=data,
            d=d,
            m=m,
            y=y,
            at_group=None,
            p0=p0,
            p1=p1,
            diagnostics=general_diagnostics,
            max_defiers_share=diagnostics["actual_max_defiers_share"],
            adjusted_probabilities=adjusted_probabilities,
            mediator_order=mediator_order,
        )
    if adjusted_probabilities is None:
        _, p1 = _mediator_distributions(data, d=d, m=m)
    else:
        p1 = adjusted_probabilities.p_m_d1
    numerator = 0.0
    denominator = 0.0
    group_results = {}
    for level in _sort_mediator_levels(theta, order=mediator_order):
        lower_bound, group_diagnostics = _single_group_lower_bound(
            data=data,
            d=d,
            m=m,
            y=y,
            at_group=level,
            theta_kk_min=theta[level],
            p_m_given_d1=p1.get(level, 0.0),
            adjusted_probabilities=adjusted_probabilities,
        )
        group_results[_normalize_scalar(level)] = {
            "at_group": _normalize_mediator_level(level),
            "lower_bound": lower_bound,
            "theta_kk_min": group_diagnostics["theta_kk_min"],
            "positive_part_partial_pmf_diff": group_diagnostics[
                "positive_part_partial_pmf_diff"
            ],
            "positive_part_cell_rows": group_diagnostics["positive_part_cell_rows"],
            "max_complier_share_to_group": group_diagnostics[
                "max_complier_share_to_group"
            ],
            "paper_inequality_lhs": group_diagnostics["paper_inequality_lhs"],
            "paper_inequality_rhs": group_diagnostics["paper_inequality_rhs"],
            "paper_inequality_gap": group_diagnostics["paper_inequality_gap"],
            "paper_inequality_violation": group_diagnostics[
                "paper_inequality_violation"
            ],
            "in_objective": True,
            "objective_role": "pooled_component",
            "no_bite": group_diagnostics["no_bite"],
        }
        numerator += theta[level] * lower_bound
        denominator += theta[level]

    no_bite = denominator <= 1e-12
    pooled_diagnostics = {
        **diagnostics,
        "pooled": {
            "shared_feasible_set": True,
            "implementation_scope": "ordered-monotone closed-form baseline",
            "post_hoc_weighted_average": True,
            "equivalence_basis": (
                "binary ordered-monotone closed form; matches the shared fractional-LP "
                "pooled objective"
            ),
            "group_results": group_results,
            "group_result_rows": list(group_results.values()),
            "denominator": denominator,
            "paper_inequality_max_violation": float(
                max(
                    (
                        row["paper_inequality_violation"]
                        for row in group_results.values()
                    ),
                    default=0.0,
                )
            ),
        },
        "no_bite": {
            "flag": no_bite,
            "theta_kk_min": denominator,
            "reason": "sum theta_kk_min is zero" if no_bite else None,
        },
    }
    return LowerBoundResult(
        lower_bound=0.0 if no_bite else float(numerator / denominator),
        estimand="bar_nu",
        at_group=None,
        restriction="ordered monotonicity",
        diagnostics=pooled_diagnostics,
    )


def _resolve_defier_cap(
    *,
    requested_max_defiers_share: float,
    minimum_compatible_defiers_share: float,
    allow_min_defiers: bool,
) -> float:
    """Resolve actual defier cap: use requested, minimum compatible, or raise."""
    if minimum_compatible_defiers_share <= requested_max_defiers_share + 1e-12:
        return float(requested_max_defiers_share)
    if allow_min_defiers:
        return float(minimum_compatible_defiers_share)
    raise ValueError(
        "Data are incompatible with max_defiers_share; set allow_min_defiers=True "
        "to use the minimum compatible defier share explicitly."
    )


def _validate_positive_tolerance(value: object, *, name: str) -> float:
    """Validate that *value* is a finite positive numeric tolerance."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite positive numeric tolerance.")
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be a finite positive numeric tolerance.")
    return numeric


def _validate_positive_integer(value: object, *, name: str) -> int:
    """Validate that *value* is a positive integer."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _validate_defier_share(value: object, *, name: str = "max_defiers_share") -> float:
    """Validate that *value* is a numeric share in [0, 1]."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a numeric share between 0 and 1.")
    share = float(value)
    if not np.isfinite(share) or share < 0.0 or share > 1.0:
        raise ValueError(f"{name} must be a numeric share between 0 and 1.")
    return share


def _validate_bool_flag(value: object, *, name: str) -> bool:
    """Validate that *value* is a boolean."""
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean.")
    return value


def _ade_restriction_label(
    *,
    vector_mediator: bool,
    max_defiers_share: float,
    minimum_compatible_defiers_share: float,
    requested_max_defiers_share: float,
) -> str:
    """Return a human-readable restriction label for ADE bound results."""
    if max_defiers_share > 1e-12 or minimum_compatible_defiers_share > requested_max_defiers_share + 1e-12:
        return "general defier cap"
    return "elementwise monotonicity" if vector_mediator else "ordered monotonicity"


def _general_theta_kk_min_for_ade(
    *,
    data: pd.DataFrame,
    d: str,
    m: str,
    y: str,
    at_group: object,
    p0: Mapping[object, float],
    p1: Mapping[object, float],
    max_defiers_share: float,
    adjusted_probabilities: AdjustedProbabilityResult | None = None,
    mediator_order: Sequence[object] | None = None,
) -> dict[str, Any]:
    """Compute theta_{kk}^{min} via general LP for ADE bounds with defier cap."""
    levels = _sort_mediator_levels(set(p0) | set(p1), order=mediator_order)
    if at_group not in levels:
        raise ValueError("at_group is not present in the mediator support.")

    max_p_diff_cell_rows: list[dict[str, Any]] = []
    max_p_diffs: dict[object, float] = {}
    for level in levels:
        cell_rows = (
            _adjusted_positive_part_partial_pmf_diff_rows(
                adjusted_probabilities=adjusted_probabilities,
                at_group=level,
            )
            if adjusted_probabilities is not None
            else _positive_part_partial_pmf_diff_rows_from_prepared(
                data=data,
                d=d,
                m=m,
                y=y,
                at_group=level,
            )
        )
        max_p_diff_cell_rows.extend(cell_rows)
        max_p_diffs[level] = float(
            sum(row["positive_part_contribution"] for row in cell_rows)
        )

    problem = _build_general_lfp_problem(
        levels=levels,
        p0=p0,
        p1=p1,
        max_p_diffs=max_p_diffs,
        max_defiers_share=max_defiers_share,
        mediator_order=mediator_order,
    )
    variable_count = len(problem["types"]) + len(levels)
    theta_by_group: dict[object, float] = {}
    target_solution: Any | None = None

    for level in levels:
        objective = np.zeros(variable_count, dtype=float)
        objective[problem["types"].index((level, level))] = 1.0
        solution = linprog(
            c=objective,
            A_ub=problem["a_ub"],
            b_ub=problem["b_ub"],
            A_eq=problem["a_eq"],
            b_eq=problem["b_eq"],
            bounds=(0.0, None),
            method="highs",
        )
        if not solution.success:
            raise ValueError(f"Unable to compute ADE theta_kk minimum: {solution.message}")
        theta_by_group[level] = float(max(solution.fun, 0.0))
        if level == at_group:
            target_solution = solution

    if target_solution is None:
        raise ValueError("at_group is not present in the mediator support.")

    eq_residual, ub_violation = _linear_constraint_residuals(
        solution=target_solution.x,
        a_eq=problem["a_eq"],
        b_eq=problem["b_eq"],
        a_ub=problem["a_ub"],
        b_ub=problem["b_ub"],
    )
    solution_rows = _general_theta_lp_solution_rows(
        problem=problem,
        p0=p0,
        p1=p1,
        max_p_diffs=max_p_diffs,
        max_defiers_share=max_defiers_share,
        solution=target_solution.x,
        target_level=at_group,
        mediator_order=mediator_order,
    )
    target_theta_from_rows = float(solution_rows["theta_kk_from_rows"])
    theta_kk_min = float(theta_by_group[at_group])
    solution_rows["theta_kk_from_rows_gap"] = float(abs(target_theta_from_rows - theta_kk_min))

    return {
        "theta_by_group": theta_by_group,
        "diagnostics": {
            "type_count": len(problem["types"]),
            "slack_count": len(levels),
            "solver_status": str(target_solution.message),
            "target_at_group": _normalize_mediator_level(at_group),
            "theta_kk_min": theta_kk_min,
            "solver_objective": float(target_solution.fun),
            "primal_eq_max_abs_residual": eq_residual,
            "primal_ub_max_violation": ub_violation,
            "max_p_diffs": {
                _normalize_scalar(level): float(value) for level, value in max_p_diffs.items()
            },
            "max_p_diff_rows": [
                {
                    "at_group": _normalize_mediator_level(level),
                    "positive_part_partial_pmf_diff": float(value),
                }
                for level, value in sorted(
                    max_p_diffs.items(),
                    key=lambda item: _mediator_sort_key(item[0], order=mediator_order),
                )
            ],
            "max_p_diff_cell_rows": max_p_diff_cell_rows,
            "theta_kk_min_rows": [
                {
                    "at_group": _normalize_mediator_level(level),
                    "theta_kk_min": float(theta_by_group[level]),
                    "in_objective": level == at_group,
                }
                for level in levels
            ],
            "paper_anchor": (
                "manuscript/sources/arxiv-2404.11739v3/draft.tex:214-228,883-890"
            ),
            "reference_anchor": "packages/r/TestMechs/R/bounds_ade_ats.R:219-347",
            **solution_rows,
        },
    }


def _general_theta_lp_solution_rows(
    *,
    problem: Mapping[str, Any],
    p0: Mapping[object, float],
    p1: Mapping[object, float],
    max_p_diffs: Mapping[object, float],
    max_defiers_share: float,
    solution: np.ndarray,
    target_level: object,
    mediator_order: Sequence[object] | None = None,
) -> dict[str, Any]:
    """Build detailed solution diagnostic rows for the general theta LP."""
    levels = list(problem["levels"])
    types = list(problem["types"])
    type_count = len(types)
    target_key = _normalize_scalar(target_level)

    type_share_rows = []
    for type_index, (m0_value, m1_value) in enumerate(types):
        in_theta_objective = m0_value == target_level and m1_value == target_level
        type_share = float(solution[type_index])
        type_share_rows.append({
            "m0": _normalize_mediator_level(m0_value),
            "m1": _normalize_mediator_level(m1_value),
            "type_share": type_share,
            "is_defier": _is_defier_type(m0_value, m1_value, order=mediator_order),
            "is_always_taker": m0_value == m1_value,
            "in_theta_objective": in_theta_objective,
            "theta_objective_contribution": type_share if in_theta_objective else 0.0,
        })

    actual_defiers_share = float(
        sum(
            solution[type_index]
            for type_index, (m0_value, m1_value) in enumerate(types)
            if _is_defier_type(m0_value, m1_value, order=mediator_order)
        )
    )
    defier_cap_residual = actual_defiers_share - float(max_defiers_share)
    defier_cap_rows = [
        {
            "requested_max_defiers_share": float(max_defiers_share),
            "actual_defiers_share": actual_defiers_share,
            "defier_cap_residual": float(defier_cap_residual),
            "defier_cap_violation": float(max(defier_cap_residual, 0.0)),
            "binding": bool(abs(defier_cap_residual) <= 1e-10),
        }
    ]

    slack_rows = []
    for slack_index, level in enumerate(levels):
        slack = float(solution[type_count + slack_index])
        positive_part = float(max_p_diffs[level])
        max_complier_share_to_group = float(
            sum(
                solution[type_index]
                for type_index, (m0_value, m1_value) in enumerate(types)
                if m1_value == level and m0_value != m1_value
            )
        )
        constraint_residual = positive_part - max_complier_share_to_group - slack
        level_key = _normalize_scalar(level)
        slack_rows.append({
            "at_group": _normalize_mediator_level(level),
            "slack": slack,
            "positive_part_partial_pmf_diff": positive_part,
            "max_complier_share_to_group": max_complier_share_to_group,
            "slack_constraint_residual": float(constraint_residual),
            "slack_constraint_violation": float(max(constraint_residual, 0.0)),
            "in_theta_objective": level_key == target_key,
        })

    marginal_fit_rows = []
    for level in levels:
        observed_d0 = float(p0.get(level, 0.0))
        observed_d1 = float(p1.get(level, 0.0))
        reconstructed_d0 = float(
            sum(
                solution[type_index]
                for type_index, (m0_value, _) in enumerate(types)
                if m0_value == level
            )
        )
        reconstructed_d1 = float(
            sum(
                solution[type_index]
                for type_index, (_, m1_value) in enumerate(types)
                if m1_value == level
            )
        )
        marginal_fit_rows.append({
            "at_group": _normalize_mediator_level(level),
            "observed_p_m_given_d0": observed_d0,
            "reconstructed_p_m_given_d0": reconstructed_d0,
            "d0_abs_difference": float(abs(reconstructed_d0 - observed_d0)),
            "observed_p_m_given_d1": observed_d1,
            "reconstructed_p_m_given_d1": reconstructed_d1,
            "d1_abs_difference": float(abs(reconstructed_d1 - observed_d1)),
        })

    theta_kk_from_rows = float(
        sum(row["theta_objective_contribution"] for row in type_share_rows)
    )
    return {
        "type_share_rows": type_share_rows,
        "slack_rows": slack_rows,
        "marginal_fit_rows": marginal_fit_rows,
        "defier_cap_rows": defier_cap_rows,
        "type_share_sum": float(sum(row["type_share"] for row in type_share_rows)),
        "theta_kk_from_rows": theta_kk_from_rows,
        "marginal_fit_max_abs_difference": float(
            max(
                (
                    max(row["d0_abs_difference"], row["d1_abs_difference"])  # type: ignore[call-overload]
                    for row in marginal_fit_rows
                ),
                default=0.0,
            )
        ),
        "slack_constraint_max_violation": float(
            max((row["slack_constraint_violation"] for row in slack_rows), default=0.0)  # type: ignore[arg-type, type-var]
        ),
        "defier_cap_max_violation": float(
            max((row["defier_cap_violation"] for row in defier_cap_rows), default=0.0)
        ),
    }


def _general_lfp_lb_frac_affected(
    *,
    data: pd.DataFrame,
    d: str,
    m: str,
    y: str,
    at_group: object | None,
    p0: Mapping[object, float],
    p1: Mapping[object, float],
    diagnostics: dict[str, Any],
    max_defiers_share: float,
    adjusted_probabilities: AdjustedProbabilityResult | None = None,
    mediator_order: Sequence[object] | None = None,
) -> LowerBoundResult:
    """Compute the lower bound using the general linear-fractional program."""
    levels = _sort_mediator_levels(set(p0) | set(p1), order=mediator_order)
    if at_group is not None and at_group not in levels:
        raise ValueError("at_group is not present in the mediator support.")

    max_p_diff_cell_rows: list[dict[str, Any]] = []
    max_p_diffs = {}
    for level in levels:
        cell_rows = (
            _adjusted_positive_part_partial_pmf_diff_rows(
                adjusted_probabilities=adjusted_probabilities,
                at_group=level,
            )
            if adjusted_probabilities is not None
            else _positive_part_partial_pmf_diff_rows_from_prepared(
                data=data,
                d=d,
                m=m,
                y=y,
                at_group=level,
            )
        )
        max_p_diff_cell_rows.extend(cell_rows)
        max_p_diffs[level] = float(
            sum(row["positive_part_contribution"] for row in cell_rows)
        )
    problem = _build_general_lfp_problem(
        levels=levels,
        p0=p0,
        p1=p1,
        max_p_diffs=max_p_diffs,
        max_defiers_share=max_defiers_share,
        mediator_order=mediator_order,
    )
    objective = _general_lfp_objective(problem=problem, at_group=at_group)
    lfp = _solve_linear_fractional_program(
        numerator=objective["numerator"],
        denominator=objective["denominator"],
        a_eq=problem["a_eq"],
        b_eq=problem["b_eq"],
        a_ub=problem["a_ub"],
        b_ub=problem["b_ub"],
    )

    no_bite = lfp["denominator_minimum"] <= 1e-12
    value = 0.0 if no_bite else lfp["optimum"]
    solution_basis = "denominator-minimum certificate" if no_bite else "fractional-lp optimum"
    objective_levels = levels if at_group is None else [at_group]
    group_results = _general_lfp_group_results(
        problem=problem,
        max_p_diffs=max_p_diffs,
        solution=lfp["solution"],
        solution_basis=solution_basis,
        objective_levels=objective_levels,
        pooled_objective=at_group is None,
    )
    target_group_result = None if at_group is None else group_results[_normalize_scalar(at_group)]
    solution_rows = _general_lfp_solution_rows(
        problem=problem,
        p0=p0,
        p1=p1,
        max_p_diffs=max_p_diffs,
        max_defiers_share=max_defiers_share,
        solution=lfp["solution"],
        objective_levels=objective_levels,
        mediator_order=mediator_order,
    )
    rows_denominator = solution_rows["objective_denominator_from_rows"]
    rows_ratio = (
        None
        if rows_denominator <= 1e-12
        else solution_rows["objective_numerator_from_rows"] / rows_denominator
    )
    solution_rows["objective_ratio_from_rows"] = rows_ratio
    solution_rows["objective_ratio_from_rows_gap"] = (
        None if rows_ratio is None else float(abs(rows_ratio - value))
    )
    pooled_diagnostics = {}
    if at_group is None:
        pooled_diagnostics["pooled"] = {
            "shared_feasible_set": True,
            "implementation_scope": "general shared feasible-set LFP",
            "post_hoc_weighted_average": False,
            "denominator": float(np.dot(objective["denominator"], lfp["solution"])),
            "group_results": group_results,
            "group_result_rows": list(group_results.values()),
        }
    paper_inequality_max_violation = float(
        max(
            (row["paper_inequality_violation"] for row in group_results.values()),
            default=0.0,
        )
    )
    objective_paper_inequality_max_violation = float(
        max(
            (
                row["paper_inequality_violation"]
                for row in group_results.values()
                if row["in_objective"]
            ),
            default=0.0,
        )
    )

    return LowerBoundResult(
        lower_bound=float(min(max(value, 0.0), 1.0)),
        estimand="bar_nu" if at_group is None else "nu_k",
        at_group=at_group,
        restriction="general defier cap",
        diagnostics={
            **diagnostics,
            **pooled_diagnostics,
            "general_lfp": {
                "type_count": len(problem["types"]),
                "slack_count": len(levels),
                "solver_status": lfp["status"],
                "solution_basis": solution_basis,
                "denominator_minimum": lfp["denominator_minimum"],
                "transformed_objective": lfp["transformed_objective"],
                "scale_variable": lfp["scale_variable"],
                "denominator_normalization": lfp["denominator_normalization"],
                "primal_eq_max_abs_residual": lfp["primal_eq_max_abs_residual"],
                "primal_ub_max_violation": lfp["primal_ub_max_violation"],
                "objective_ratio_gap": lfp["objective_ratio_gap"],
                "objective_levels": [_normalize_scalar(level) for level in objective_levels],
                "objective_numerator": float(np.dot(objective["numerator"], lfp["solution"])),
                "objective_denominator": float(np.dot(objective["denominator"], lfp["solution"])),
                "max_p_diffs": {
                    _normalize_scalar(level): float(value) for level, value in max_p_diffs.items()
                },
                "max_p_diff_rows": [
                    {
                        "at_group": _normalize_mediator_level(level),
                        "positive_part_partial_pmf_diff": float(value),
                    }
                    for level, value in sorted(
                        max_p_diffs.items(),
                        key=lambda item: _mediator_sort_key(item[0], order=mediator_order),
                    )
                ],
                "max_p_diff_cell_rows": max_p_diff_cell_rows,
                "group_results": group_results,
                "group_result_rows": list(group_results.values()),
                "paper_inequality_max_violation": paper_inequality_max_violation,
                "objective_paper_inequality_max_violation": (
                    objective_paper_inequality_max_violation
                ),
                "target_group_result": target_group_result,
                **solution_rows,
            },
            "no_bite": {
                "flag": no_bite,
                "theta_kk_min": float(np.dot(objective["denominator"], lfp["solution"])),
                "reason": "objective denominator can be zero" if no_bite else None,
            },
        },
    )


def _general_lfp_group_results(
    *,
    problem: Mapping[str, Any],
    max_p_diffs: Mapping[object, float],
    solution: np.ndarray,
    solution_basis: str,
    objective_levels: Sequence[object],
    pooled_objective: bool,
) -> dict[object, dict[str, Any]]:
    """Extract per-group lower-bound diagnostics from the LFP solution."""
    levels = list(problem["levels"])
    types = list(problem["types"])
    type_count = len(types)
    objective_level_keys = {_normalize_scalar(level) for level in objective_levels}
    group_results: dict[object, dict[str, Any]] = {}
    for slack_index, level in enumerate(levels):
        level_key = _normalize_scalar(level)
        theta_kk = float(solution[types.index((level, level))])
        numerator_contribution = float(solution[type_count + slack_index])
        max_complier_share_to_group = float(
            sum(
                solution[type_index]
                for type_index, (m0_value, m1_value) in enumerate(types)
                if m1_value == level and m0_value != m1_value
            )
        )
        no_bite = theta_kk <= 1e-12
        lower_bound = 0.0 if no_bite else numerator_contribution / theta_kk
        in_objective = level_key in objective_level_keys
        paper_inequality_rhs = float(
            max(float(max_p_diffs[level]) - max_complier_share_to_group, 0.0)
        )
        paper_inequality_lhs = float(theta_kk * lower_bound)
        group_results[level_key] = {
            "at_group": _normalize_mediator_level(level),
            "lower_bound": float(min(max(lower_bound, 0.0), 1.0)),
            "theta_kk": theta_kk,
            "numerator_contribution": numerator_contribution,
            "positive_part_partial_pmf_diff": float(max_p_diffs[level]),
            "max_complier_share_to_group": max_complier_share_to_group,
            "paper_inequality_lhs": paper_inequality_lhs,
            "paper_inequality_rhs": paper_inequality_rhs,
            "paper_inequality_gap": float(paper_inequality_lhs - paper_inequality_rhs),
            "paper_inequality_violation": float(
                max(paper_inequality_rhs - paper_inequality_lhs, 0.0)
            ),
            "solution_basis": solution_basis,
            "in_objective": in_objective,
            "objective_role": (
                "pooled_component"
                if pooled_objective and in_objective
                else "target"
                if in_objective
                else "non_target"
            ),
            "no_bite": {
                "flag": no_bite,
                "theta_kk": 0.0 if no_bite else theta_kk,
                "reason": "theta_kk is zero at the reported LFP solution" if no_bite else None,
            },
        }
    return group_results


def _general_lfp_solution_rows(
    *,
    problem: Mapping[str, Any],
    p0: Mapping[object, float],
    p1: Mapping[object, float],
    max_p_diffs: Mapping[object, float],
    max_defiers_share: float,
    solution: np.ndarray,
    objective_levels: Sequence[object],
    mediator_order: Sequence[object] | None = None,
) -> dict[str, Any]:
    """Build detailed solution diagnostic rows for the general LFP."""
    levels = list(problem["levels"])
    types = list(problem["types"])
    type_count = len(types)
    objective_level_keys = {_normalize_scalar(level) for level in objective_levels}

    type_share_rows = [
        {
            "m0": _normalize_mediator_level(m0_value),
            "m1": _normalize_mediator_level(m1_value),
            "type_share": float(solution[type_index]),
            "is_defier": _is_defier_type(m0_value, m1_value, order=mediator_order),
            "is_always_taker": m0_value == m1_value,
            "in_objective_denominator": (
                m0_value == m1_value and _normalize_scalar(m1_value) in objective_level_keys
            ),
            "objective_denominator_contribution": (
                float(solution[type_index])
                if m0_value == m1_value and _normalize_scalar(m1_value) in objective_level_keys
                else 0.0
            ),
        }
        for type_index, (m0_value, m1_value) in enumerate(types)
    ]
    actual_defiers_share = float(
        sum(
            solution[type_index]
            for type_index, (m0_value, m1_value) in enumerate(types)
            if _is_defier_type(m0_value, m1_value, order=mediator_order)
        )
    )
    defier_cap_residual = actual_defiers_share - float(max_defiers_share)
    defier_cap_rows = [
        {
            "requested_max_defiers_share": float(max_defiers_share),
            "actual_defiers_share": actual_defiers_share,
            "defier_cap_residual": float(defier_cap_residual),
            "defier_cap_violation": float(max(defier_cap_residual, 0.0)),
            "binding": bool(abs(defier_cap_residual) <= 1e-10),
        }
    ]
    slack_rows = []
    for slack_index, level in enumerate(levels):
        slack = float(solution[type_count + slack_index])
        positive_part = float(max_p_diffs[level])
        max_complier_share_to_group = float(
            sum(
                solution[type_index]
                for type_index, (m0_value, m1_value) in enumerate(types)
                if m1_value == level and m0_value != m1_value
            )
        )
        constraint_residual = positive_part - max_complier_share_to_group - slack
        slack_rows.append({
            "at_group": _normalize_mediator_level(level),
            "slack": slack,
            "positive_part_partial_pmf_diff": positive_part,
            "max_complier_share_to_group": max_complier_share_to_group,
            "slack_constraint_residual": float(constraint_residual),
            "slack_constraint_violation": float(max(constraint_residual, 0.0)),
            "in_objective_numerator": _normalize_scalar(level) in objective_level_keys,
            "objective_numerator_contribution": (
                slack if _normalize_scalar(level) in objective_level_keys else 0.0
            ),
        })
    marginal_fit_rows = []
    for level in levels:
        observed_d0 = float(p0.get(level, 0.0))
        observed_d1 = float(p1.get(level, 0.0))
        reconstructed_d0 = float(
            sum(
                solution[type_index]
                for type_index, (m0_value, _) in enumerate(types)
                if m0_value == level
            )
        )
        reconstructed_d1 = float(
            sum(
                solution[type_index]
                for type_index, (_, m1_value) in enumerate(types)
                if m1_value == level
            )
        )
        marginal_fit_rows.append({
            "at_group": _normalize_mediator_level(level),
            "observed_p_m_given_d0": observed_d0,
            "reconstructed_p_m_given_d0": reconstructed_d0,
            "d0_abs_difference": float(abs(reconstructed_d0 - observed_d0)),
            "observed_p_m_given_d1": observed_d1,
            "reconstructed_p_m_given_d1": reconstructed_d1,
            "d1_abs_difference": float(abs(reconstructed_d1 - observed_d1)),
        })
    return {
        "type_share_rows": type_share_rows,
        "slack_rows": slack_rows,
        "marginal_fit_rows": marginal_fit_rows,
        "defier_cap_rows": defier_cap_rows,
        "type_share_sum": float(sum(row["type_share"] for row in type_share_rows)),
        "objective_numerator_from_rows": float(
            sum(row["objective_numerator_contribution"] for row in slack_rows)  # type: ignore[misc]
        ),
        "objective_denominator_from_rows": float(
            sum(row["objective_denominator_contribution"] for row in type_share_rows)
        ),
        "marginal_fit_max_abs_difference": float(
            max(
                (
                    max(row["d0_abs_difference"], row["d1_abs_difference"])  # type: ignore[call-overload]
                    for row in marginal_fit_rows
                ),
                default=0.0,
            )
        ),
        "slack_constraint_max_violation": float(
            max((row["slack_constraint_violation"] for row in slack_rows), default=0.0)  # type: ignore[arg-type, type-var]
        ),
        "defier_cap_max_violation": float(
            max((row["defier_cap_violation"] for row in defier_cap_rows), default=0.0)
        ),
    }


def _build_general_lfp_problem(
    *,
    levels: Sequence[object],
    p0: Mapping[object, float],
    p1: Mapping[object, float],
    max_p_diffs: Mapping[object, float],
    max_defiers_share: float,
    mediator_order: Sequence[object] | None = None,
) -> dict[str, Any]:
    """Construct the equality/inequality constraint matrices for the general LFP."""
    types = _build_type_pairs(levels)
    type_count = len(types)
    slack_count = len(levels)
    variable_count = type_count + slack_count

    a_eq: list[list[float]] = []
    b_eq: list[float] = []
    for level in levels:
        row = [0.0] * variable_count
        for index, (_, m1_value) in enumerate(types):
            if m1_value == level:
                row[index] = 1.0
        a_eq.append(row)
        b_eq.append(float(p1.get(level, 0.0)))
    for level in levels:
        row = [0.0] * variable_count
        for index, (m0_value, _) in enumerate(types):
            if m0_value == level:
                row[index] = 1.0
        a_eq.append(row)
        b_eq.append(float(p0.get(level, 0.0)))

    a_ub: list[list[float]] = []
    b_ub: list[float] = []
    for slack_index, level in enumerate(levels):
        row = [0.0] * variable_count
        for index, (m0_value, m1_value) in enumerate(types):
            if m1_value == level and m0_value != m1_value:
                row[index] = -1.0
        row[type_count + slack_index] = -1.0
        a_ub.append(row)
        b_ub.append(-float(max_p_diffs[level]))

    row = [0.0] * variable_count
    for index, (m0_value, m1_value) in enumerate(types):
        if _is_defier_type(m0_value, m1_value, order=mediator_order):
            row[index] = 1.0
    a_ub.append(row)
    b_ub.append(float(max_defiers_share))

    return {
        "levels": list(levels),
        "types": types,
        "a_eq": np.asarray(a_eq, dtype=float),
        "b_eq": np.asarray(b_eq, dtype=float),
        "a_ub": np.asarray(a_ub, dtype=float),
        "b_ub": np.asarray(b_ub, dtype=float),
    }


def _general_lfp_objective(*, problem: Mapping[str, Any], at_group: object | None) -> dict[str, np.ndarray]:
    """Build numerator and denominator coefficient vectors for the LFP objective."""
    levels = list(problem["levels"])
    types = list(problem["types"])
    type_count = len(types)
    variable_count = type_count + len(levels)
    numerator = np.zeros(variable_count, dtype=float)
    denominator = np.zeros(variable_count, dtype=float)

    objective_levels = levels if at_group is None else [at_group]
    for level in objective_levels:
        slack_index = levels.index(level)
        numerator[type_count + slack_index] = 1.0
        type_index = types.index((level, level))
        denominator[type_index] = 1.0
    return {"numerator": numerator, "denominator": denominator}


def _solve_linear_fractional_program(
    *,
    numerator: np.ndarray,
    denominator: np.ndarray,
    a_eq: np.ndarray,
    b_eq: np.ndarray,
    a_ub: np.ndarray,
    b_ub: np.ndarray,
) -> dict[str, Any]:
    """Solve a linear-fractional program via Charnes-Cooper transformation."""
    denominator_lp = linprog(
        c=denominator,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=(0.0, None),
        method="highs",
    )
    if not denominator_lp.success:
        raise ValueError(f"The general lower-bound LP is infeasible: {denominator_lp.message}")
    if denominator_lp.fun <= 1e-12:
        eq_residual, ub_violation = _linear_constraint_residuals(
            solution=denominator_lp.x,
            a_eq=a_eq,
            b_eq=b_eq,
            a_ub=a_ub,
            b_ub=b_ub,
        )
        return {
            "optimum": 0.0,
            "solution": denominator_lp.x,
            "status": denominator_lp.message,
            "denominator_minimum": float(denominator_lp.fun),
            "transformed_objective": None,
            "scale_variable": None,
            "denominator_normalization": None,
            "primal_eq_max_abs_residual": eq_residual,
            "primal_ub_max_violation": ub_violation,
            "objective_ratio_gap": None,
        }

    variable_count = len(numerator)
    c = np.concatenate([numerator, np.array([0.0])])
    a_eq_flp = np.vstack(
        [
            np.column_stack([a_eq, -b_eq]),
            np.concatenate([denominator, np.array([0.0])]),
        ]
    )
    b_eq_flp = np.concatenate([np.zeros(len(b_eq)), np.array([1.0])])
    a_ub_flp = np.column_stack([a_ub, -b_ub])
    b_ub_flp = np.zeros(len(b_ub))

    solution = linprog(
        c=c,
        A_ub=a_ub_flp,
        b_ub=b_ub_flp,
        A_eq=a_eq_flp,
        b_eq=b_eq_flp,
        bounds=(0.0, None),
        method="highs",
    )
    if not solution.success:
        raise ValueError(f"The transformed general lower-bound LP is infeasible: {solution.message}")

    t_value = solution.x[-1]
    if t_value <= 1e-12:
        raise ValueError("The transformed general lower-bound LP returned a zero scale variable.")
    original_solution = solution.x[:variable_count] / t_value
    eq_residual, ub_violation = _linear_constraint_residuals(
        solution=original_solution,
        a_eq=a_eq,
        b_eq=b_eq,
        a_ub=a_ub,
        b_ub=b_ub,
    )
    original_denominator = float(np.dot(denominator, original_solution))
    original_ratio = float(np.dot(numerator, original_solution) / original_denominator)
    return {
        "optimum": float(solution.fun),
        "solution": original_solution,
        "status": solution.message,
        "denominator_minimum": float(denominator_lp.fun),
        "transformed_objective": float(solution.fun),
        "scale_variable": float(t_value),
        "denominator_normalization": float(np.dot(denominator, solution.x[:variable_count])),
        "primal_eq_max_abs_residual": eq_residual,
        "primal_ub_max_violation": ub_violation,
        "objective_ratio_gap": float(abs(original_ratio - solution.fun)),
    }


def _linear_constraint_residuals(
    *,
    solution: np.ndarray,
    a_eq: np.ndarray,
    b_eq: np.ndarray,
    a_ub: np.ndarray,
    b_ub: np.ndarray,
) -> tuple[float, float]:
    """Compute max absolute equality residual and max inequality violation."""
    eq_residual = np.abs(a_eq @ solution - b_eq)
    ub_violation = a_ub @ solution - b_ub
    return (
        float(eq_residual.max(initial=0.0)),
        float(np.maximum(ub_violation, 0.0).max(initial=0.0)),
    )


def _adjusted_positive_part_partial_pmf_diff(
    *,
    adjusted_probabilities: AdjustedProbabilityResult,
    at_group: object,
) -> float:
    """Compute positive-part partial PMF diff from the adjusted probability grid."""
    rows = _adjusted_positive_part_partial_pmf_diff_rows(
        adjusted_probabilities=adjusted_probabilities,
        at_group=at_group,
    )
    return float(sum(row["positive_part_contribution"] for row in rows))


def _adjusted_positive_part_partial_pmf_diff_rows(
    *,
    adjusted_probabilities: AdjustedProbabilityResult,
    at_group: object,
) -> list[dict[str, Any]]:
    """Compute per-outcome positive-part rows from adjusted probability grid."""
    if at_group not in adjusted_probabilities.m_values:
        raise ValueError("at_group is not present in the mediator support.")
    _adjusted_joint_masses_for_group(
        adjusted_probabilities=adjusted_probabilities,
        at_group=at_group,
        probability_map=adjusted_probabilities.p_ym_d1,
        expected_joint_mass=adjusted_probabilities.p_m_d1.get(at_group, 0.0),
        arm_label="d1",
    )
    _adjusted_joint_masses_for_group(
        adjusted_probabilities=adjusted_probabilities,
        at_group=at_group,
        probability_map=adjusted_probabilities.p_ym_d0,
        expected_joint_mass=adjusted_probabilities.p_m_d0.get(at_group, 0.0),
        arm_label="d0",
    )
    rows = []
    for y_value in adjusted_probabilities.y_values:
        p1 = adjusted_probabilities.p_ym_d1.get((y_value, at_group), 0.0)
        p0 = adjusted_probabilities.p_ym_d0.get((y_value, at_group), 0.0)
        delta = p1 - p0
        rows.append({
            "at_group": _normalize_mediator_level(at_group),
            "y_value": _normalize_scalar(y_value),
            "p_y_m_given_d1": float(p1),
            "p_y_m_given_d0": float(p0),
            "delta": float(delta),
            "positive_part_contribution": float(max(delta, 0.0)),
        })
    return rows


def _validate_adjusted_probability_grid(
    adjusted_probabilities: AdjustedProbabilityResult,
) -> None:
    """Validate that the adjusted probability grid is internally consistent."""
    for treatment_value, probabilities, mediator_masses in (
        (0, adjusted_probabilities.p_ym_d0, adjusted_probabilities.p_m_d0),
        (1, adjusted_probabilities.p_ym_d1, adjusted_probabilities.p_m_d1),
    ):
        arm_label = f"d{treatment_value}"
        for m_value in adjusted_probabilities.m_values:
            mass = float(mediator_masses.get(m_value, 0.0))
            if not np.isfinite(mass) or mass < 0.0 or mass > 1.0:
                raise ValueError(
                    "Adjusted mediator mass grid must contain finite nonnegative "
                    f"probabilities no larger than 1 for {arm_label}."
                )
            _adjusted_joint_masses_for_group(
                adjusted_probabilities=adjusted_probabilities,
                at_group=m_value,
                probability_map=probabilities,
                expected_joint_mass=mass,
                arm_label=arm_label,
            )
        total_mass = float(
            sum(
                float(mediator_masses.get(m_value, 0.0))
                for m_value in adjusted_probabilities.m_values
            )
        )
        if not np.isclose(total_mass, 1.0, rtol=1e-10, atol=1e-10):
            raise ValueError(
                f"Adjusted mediator mass grid for {arm_label} must sum to 1."
            )


def _adjusted_ade_trimmed_expectations(
    *,
    adjusted_probabilities: AdjustedProbabilityResult,
    at_group: object,
    check_theta_d1: float,
    check_theta_d0: float,
) -> dict[str, Any]:
    """Compute trimmed expectations for ADE bounds using adjusted probabilities."""
    d1_distribution = _adjusted_conditional_outcome_distribution(
        adjusted_probabilities=adjusted_probabilities,
        at_group=at_group,
        probability_map=adjusted_probabilities.p_ym_d1,
        expected_joint_mass=adjusted_probabilities.p_m_d1.get(at_group, 0.0),
        arm_label="d1",
    )
    d0_distribution = _adjusted_conditional_outcome_distribution(
        adjusted_probabilities=adjusted_probabilities,
        at_group=at_group,
        probability_map=adjusted_probabilities.p_ym_d0,
        expected_joint_mass=adjusted_probabilities.p_m_d0.get(at_group, 0.0),
        arm_label="d0",
    )
    y1_lower = _weighted_trimmed_expectation(
        y_values=d1_distribution["y_values"],
        pmf=d1_distribution["pmf"],
        frac=check_theta_d1,
        upper=False,
    )
    y1_upper = _weighted_trimmed_expectation(
        y_values=d1_distribution["y_values"],
        pmf=d1_distribution["pmf"],
        frac=check_theta_d1,
        upper=True,
    )
    y0_lower = _weighted_trimmed_expectation(
        y_values=d0_distribution["y_values"],
        pmf=d0_distribution["pmf"],
        frac=check_theta_d0,
        upper=False,
    )
    y0_upper = _weighted_trimmed_expectation(
        y_values=d0_distribution["y_values"],
        pmf=d0_distribution["pmf"],
        frac=check_theta_d0,
        upper=True,
    )
    return {
        "y1_lower": y1_lower,
        "y1_upper": y1_upper,
        "y0_lower": y0_lower,
        "y0_upper": y0_upper,
        "diagnostics": {
            "distribution_contract": "adjusted joint probability grid conditioned on at_group",
            "outcome_grid_points": len(adjusted_probabilities.y_values),
            "check_theta": {"d1": float(check_theta_d1), "d0": float(check_theta_d0)},
            "d1": d1_distribution["diagnostics"],
            "d0": d0_distribution["diagnostics"],
        },
    }


def _adjusted_conditional_outcome_distribution(
    *,
    adjusted_probabilities: AdjustedProbabilityResult,
    at_group: object,
    probability_map: Mapping[tuple[object, object], float],
    expected_joint_mass: float,
    arm_label: str,
) -> dict[str, Any]:
    """Extract and normalise the conditional outcome PMF from adjusted joint grid."""
    if at_group not in adjusted_probabilities.m_values:
        raise ValueError("at_group is not present in the mediator support.")

    y_values, raw_joint, joint_mass = _adjusted_joint_masses_for_group(
        adjusted_probabilities=adjusted_probabilities,
        at_group=at_group,
        probability_map=probability_map,
        expected_joint_mass=expected_joint_mass,
        arm_label=arm_label,
    )
    if joint_mass <= 1e-12:
        raise ValueError(
            f"Adjusted conditional outcome mass is zero for {arm_label} and at_group."
        )
    pmf = [float(value / joint_mass) for value in raw_joint]
    return {
        "y_values": y_values,
        "pmf": pmf,
        "diagnostics": {
            "raw_joint_mass": joint_mass,
            "normalized_joint_mass": joint_mass,
            "pmf_sum": float(sum(pmf)),
            "pmf": {
                _normalize_scalar(y_value): float(probability)
                for y_value, probability in zip(y_values, pmf, strict=True)
            },
        },
    }


def _adjusted_joint_masses_for_group(
    *,
    adjusted_probabilities: AdjustedProbabilityResult,
    at_group: object,
    probability_map: Mapping[tuple[object, object], float],
    expected_joint_mass: float,
    arm_label: str,
) -> tuple[list[object], list[float], float]:
    """Extract and validate joint masses for a single mediator group."""
    y_values = list(adjusted_probabilities.y_values)
    raw_joint = [
        float(probability_map.get((y_value, at_group), 0.0))
        for y_value in y_values
    ]
    invalid_cells = [
        _normalize_scalar(y_value)
        for y_value, value in zip(y_values, raw_joint, strict=True)
        if not np.isfinite(value) or value < 0.0 or value > 1.0
    ]
    if invalid_cells:
        raise ValueError(
            "Adjusted joint probability grid must contain finite nonnegative "
            f"probabilities no larger than 1 for {arm_label} and at_group."
        )
    joint_mass = float(sum(raw_joint))
    if not np.isclose(joint_mass, float(expected_joint_mass), rtol=1e-10, atol=1e-10):
        raise ValueError(
            "Adjusted joint probability grid mass for at_group must equal "
            f"P(M=at_group | {arm_label})."
        )
    return y_values, raw_joint, joint_mass


def _weighted_trimmed_expectation(
    *,
    y_values: Sequence[object],
    pmf: Sequence[float],
    frac: float,
    upper: bool,
) -> float:
    """Compute a trimmed expectation from a discrete PMF."""
    frac = min(max(float(frac), 0.0), 1.0)
    if frac <= 1e-12:
        raise ValueError("Trim fraction must be positive.")
    if len(y_values) != len(pmf):
        raise ValueError("y_values and pmf must have the same length.")

    pairs = sorted(
        ((float(y_value), float(probability)) for y_value, probability in zip(y_values, pmf, strict=True)),  # type: ignore[arg-type]
        key=lambda item: item[0],
        reverse=upper,
    )
    total_mass = sum(probability for _, probability in pairs)
    if total_mass <= 1e-12:
        raise ValueError("Conditional outcome distribution has zero mass.")

    remaining = frac
    weighted_sum = 0.0
    for y_value, probability in pairs:
        if remaining <= 1e-12:
            break
        take = min(probability / total_mass, remaining)
        weighted_sum += take * y_value
        remaining -= take
    return float(weighted_sum / frac)


def _build_type_pairs(levels: Sequence[object]) -> list[tuple[object, object]]:
    """Return all (M(0), M(1)) type pairs as the Cartesian product of levels."""
    return [(m0_value, m1_value) for m0_value in levels for m1_value in levels]


def _is_defier_type(
    m0_value: object,
    m1_value: object,
    *,
    order: Sequence[object] | None = None,
) -> bool:
    """Return True if the (m0, m1) pair represents a defier type."""
    return not _mediator_leq(m0_value, m1_value, order=order)


def _validate_scalar_ordered_support(*, levels: Sequence[object], mediator: Mapping[str, Any]) -> None:
    """Ensure nonbinary scalar mediator support is orderable for monotone bounds."""
    if mediator["vector"] or len(levels) <= 2:
        return
    if mediator["level_order"] is not None:
        return
    for value in levels:
        normalized = _normalize_scalar(value)
        if not isinstance(normalized, (bool, int, float)):
            raise ValueError(
                "Nonbinary scalar mediator support must be numeric, boolean, "
                "or an ordered pandas Categorical before using ordered-monotone bounds."
            )
    for index, left in enumerate(levels):
        for right in levels[index + 1 :]:
            try:
                left <= right  # type: ignore[operator]
                right <= left  # type: ignore[operator]
            except TypeError as exc:
                raise ValueError(
                    "Nonbinary scalar mediator support must be numeric, boolean, "
                    "or an ordered pandas Categorical before using ordered-monotone bounds."
                ) from exc


def _validate_vector_mediator_elementwise_support(
    data: pd.DataFrame,
    mediator_columns: Sequence[str],
) -> None:
    """Ensure each vector mediator component has numeric or boolean support."""
    for column in mediator_columns:
        for value in pd.unique(data[column].dropna()):
            normalized = _normalize_scalar(value)
            if isinstance(normalized, bool):
                continue
            if isinstance(normalized, (int, float)):
                continue
            raise ValueError(
                "Vector mediator elementwise-monotone bounds require each mediator "
                "component to have numeric or boolean ordered support; recode "
                f"{column} to an ordered numeric indicator before using vector bounds."
            )


def _mediator_leq(
    left: object,
    right: object,
    *,
    order: Sequence[object] | None = None,
) -> bool:
    """Return True if *left* <= *right* under the mediator ordering."""
    if order is not None and not isinstance(left, tuple) and not isinstance(right, tuple):
        order_index = {value: index for index, value in enumerate(order)}
        if left in order_index and right in order_index:
            return order_index[left] <= order_index[right]
    if isinstance(left, tuple) and isinstance(right, tuple):
        if len(left) != len(right):
            raise ValueError("Vector mediator support points must have equal dimension.")
        return all(
            _mediator_leq(component_left, component_right)
            for component_left, component_right in zip(left, right, strict=True)
        )
    try:
        return bool(left <= right)  # type: ignore[operator]
    except TypeError:
        return _mediator_sort_key(left, order=order) <= _mediator_sort_key(right, order=order)


def _sort_mediator_levels(
    values: object,
    *,
    order: Sequence[object] | None = None,
) -> list[object]:
    """Sort mediator support levels using the deterministic sort-key logic."""
    return _sort_support_levels(values, order=order)


def _sort_support_levels(
    values: object,
    *,
    order: Sequence[object] | None = None,
) -> list[object]:
    """Sort arbitrary support levels using the deterministic sort-key logic."""
    return sorted(list(values), key=lambda value: _mediator_sort_key(value, order=order))  # type: ignore[call-overload]


def _mediator_sort_key(
    value: object,
    *,
    order: Sequence[object] | None = None,
) -> tuple[object, ...]:
    """Return a deterministic sort key for a mediator level."""
    if order is not None:
        order_index = {item: index for index, item in enumerate(order)}
        if value in order_index:
            return ("ordered_category", order_index[value])
    if isinstance(value, tuple):
        return ("tuple", *(_mediator_sort_key(item) for item in value))
    normalized = _normalize_scalar(value)
    if isinstance(normalized, bool):
        return ("bool", int(normalized))
    if isinstance(normalized, (int, float)):
        return ("number", float(normalized))
    return (type(normalized).__name__, repr(normalized))


def _normalize_mediator_level(value: object) -> object:
    """Normalise a mediator level (scalar or tuple) for serialisation."""
    if isinstance(value, tuple):
        return tuple(_normalize_scalar(item) for item in value)
    return _normalize_scalar(value)


def _internal_mediator_column_name(data: pd.DataFrame) -> str:
    """Generate a unique internal column name for processed vector mediators."""
    column = "_tm_m_processed"
    while column in data.columns:
        column = f"{column}_v"
    return column


def _trimmed_expectation(values: pd.Series, *, frac: float, upper: bool) -> float:
    """Compute a trimmed expectation from a Series of numeric outcome values."""
    if values.empty:
        raise ValueError("Cannot trim an empty conditional outcome distribution.")
    frac = min(max(float(frac), 0.0), 1.0)
    if frac <= 1e-12:
        raise ValueError("Trim fraction must be positive.")

    ordered = sorted(float(value) for value in values.tolist())
    if upper:
        ordered = list(reversed(ordered))

    target_mass = frac * len(ordered)
    remaining = target_mass
    weighted_sum = 0.0
    for value in ordered:
        if remaining <= 1e-12:
            break
        take = min(1.0, remaining)
        weighted_sum += take * value
        remaining -= take
    return float(weighted_sum / target_mass)


def _normalize_scalar(value: object) -> object:
    """Normalise a scalar value for JSON-safe serialisation."""
    if isinstance(value, pd.Interval):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (AttributeError, ValueError):
            return value
    return value
