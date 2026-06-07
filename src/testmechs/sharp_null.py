from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
import math
import numbers
from pathlib import Path

import numpy as np
import osqp
import pandas as pd
from scipy.linalg import qr
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize
from scipy import sparse
from scipy.stats import chi2, norm

from .preprocess import (
    build_cell_count_diagnostics,
    discretize_y,
    normalize_binary_support,
    remove_missing_from_df,
)
from .regression import compute_adjusted_probability_influences, parse_reg_formula
from .results import SharpNullResult, TVConfidenceIntervalResult


def test_sharp_null(
    *,
    data_path: str | Path | None = None,
    df: pd.DataFrame | None = None,
    d: str,
    m: str | Sequence[str],
    y: str,
    method: str = "CS",
    num_y_bins: int | None = None,
    alpha: float = 0.05,
    cluster: str | None = None,
    reg_formula: str | None = None,
    bootstrap_replications: int = 500,
    random_state: int | None = None,
    kitagawa_xi: float = 0.07,
    frac_ats_affected: float | None = None,
    max_defiers_share: float = 0.0,
) -> SharpNullResult:
    """Test the sharp-null hypothesis of full mediation.

    Tests whether the treatment effect on outcome Y operates entirely
    through the mediator M, i.e., H0: Y(1,m) = Y(0,m) for all m.
    The outcome and mediator are both assumed to take finitely many
    values. Several inference procedures are available.

    Parameters
    ----------
    data_path : str or Path, optional
        Path to a CSV file containing the analysis data. Exactly one
        of ``data_path`` or ``df`` must be provided.
    df : pd.DataFrame, optional
        Analysis data frame. Exactly one of ``data_path`` or ``df``
        must be provided.
    d : str
        Name of the binary treatment column in the data.
    m : str or sequence of str
        Name(s) of the mediator column(s). Pass a single string for
        scalar mediator; pass a sequence for vector mediator with
        elementwise monotonicity ordering.
    y : str
        Name of the discrete outcome column.
    method : str, default "CS"
        Inference method. One of:

        - ``"CS"`` : Cox and Shi (2023) conditional test (recommended).
        - ``"ARP"`` : Andrews, Roth, and Pakes (2023) hybrid test.
        - ``"FSSTdd"`` : FSST with data-driven lambda selection.
        - ``"FSSTndd"`` : FSST with non-data-driven lambda.
        - ``"K"`` : Kitagawa (2015) combined Z-test (binary mediator only).
    num_y_bins : int, optional
        If specified, discretize Y into this many quantile bins before
        testing. Ignored if Y already has fewer unique values.
    alpha : float, default 0.05
        Significance level for the hypothesis test.
    cluster : str, optional
        Name of the cluster variable for cluster-robust inference.
    reg_formula : str, optional
        Regression formula for non-experimental settings. Currently
        release-scoped for binary-mediator CS with controls or
        one-way fixed effects. Format: ``"~ x1 + x2"`` or
        ``"~ x1 + x2 | fe1"``.
    bootstrap_replications : int, default 500
        Number of bootstrap replications for methods requiring
        simulation-based critical values (FSST, K).
    random_state : int, optional
        Random seed for reproducibility of bootstrap draws.
    kitagawa_xi : float, default 0.07
        Tuning parameter for the Kitagawa test statistic scaling.
    frac_ats_affected : float, optional
        Relaxes the sharp null to test that the fraction of
        always-takers whose outcome is affected is at most this value.
        Default ``None`` corresponds to the sharp null (zero affected).
    max_defiers_share : float, default 0.0
        Upper bound on the proportion of defiers in the population.
        Default 0.0 imposes strict monotonicity.

    Returns
    -------
    SharpNullResult
        Result object containing:

        - ``reject`` : bool - whether the null is rejected at level alpha.
        - ``p_value`` : float - p-value of the test.
        - ``method`` : str - method label used.
        - ``diagnostics`` : dict - solver and support diagnostics.
        - ``to_frame()`` : summary as a one-row DataFrame.
        - ``to_dict()`` : strict-JSON-safe payload.

    Raises
    ------
    ValueError
        If both or neither of ``data_path``/``df`` are provided,
        if required columns are missing or contain only missing values,
        if treatment is not binary, or if parameters are out of range.
    NotImplementedError
        If ``method`` is not one of the supported methods, or if
        an unsupported ``reg_formula`` scope is requested.

    Examples
    --------
    >>> import pandas as pd
    >>> import testmechs
    >>> df = pd.DataFrame({
    ...     "treat": [0, 0, 0, 0, 1, 1, 1, 1],
    ...     "mediator": [0, 0, 1, 1, 1, 1, 1, 1],
    ...     "outcome": [0, 1, 0, 1, 0, 1, 1, 1],
    ... })
    >>> result = testmechs.test_sharp_null(
    ...     df=df, d="treat", m="mediator", y="outcome", method="CS"
    ... )
    >>> result.reject
    False
    >>> result.p_value  # doctest: +SKIP
    0.317...

    Notes
    -----
    Implements the testing framework from Kwon and Roth (2024) [1]_.
    The sharp null H0: Y(1,m) = Y(0,m) for all m implies that
    treatment effects operate entirely through the mediator.

    The test constructs moment inequalities from observable joint
    distributions P(Y=y, M=m | D=d) and tests them using the
    selected inference procedure.

    References
    ----------
    .. [1] Kwon, S. and Roth, J. (2024). "Testing Mechanisms."
       arXiv:2404.11739v3.

    See Also
    --------
    test_sharp_null_cr : CR confidence-set approach for scalar mediators.
    lb_frac_affected : Lower bound on the fraction of always-takers affected.
    bounds_ade_ats : Average direct effect bounds.
    """

    d = _validate_scalar_column_name(d, name="d")
    y = _validate_scalar_column_name(y, name="y")
    if method == "FSST":
        method = "FSSTdd"
    if method not in {"ARP", "CS", "FSSTdd", "FSSTndd", "K"}:
        raise NotImplementedError(
            "Python currently implements the CS, ARP, FSSTdd, FSSTndd, and K sharp-null runners."
        )
    alpha = _validate_probability_share(alpha, name="alpha", strict=True)
    num_y_bins = _validate_optional_positive_integer(num_y_bins, name="num_y_bins")
    if method == "K":
        bootstrap_replications = _validate_positive_integer(
            bootstrap_replications,
            name="bootstrap_replications",
            minimum=1,
            context="method='K'",
        )
        kitagawa_xi = _validate_positive_real(kitagawa_xi, name="kitagawa_xi")
        random_state = _validate_optional_nonnegative_integer(random_state, name="random_state")
    elif method in {"FSSTdd", "FSSTndd"}:
        bootstrap_replications = _validate_positive_integer(
            bootstrap_replications,
            name="bootstrap_replications",
            minimum=2,
            context="FSST runners",
        )
        random_state = _validate_optional_nonnegative_integer(random_state, name="random_state")
    if frac_ats_affected is not None:
        frac_ats_affected = _validate_probability_share(
            frac_ats_affected,
            name="frac_ats_affected",
        )
        if method != "CS":
            raise NotImplementedError(
                "frac_ats_affected is currently implemented for method='CS' only."
            )
    max_defiers_share = _validate_probability_share(
        max_defiers_share,
        name="max_defiers_share",
    )

    regression_spec = None
    regression_diagnostics = None
    if reg_formula is not None:
        regression_spec = parse_reg_formula(reg_formula, d=d)
        regression_diagnostics = {
            "formula": reg_formula,
            "formula_kind": regression_spec.formula_kind,
            "supported_scope": _sharp_null_regression_supported_scope(regression_spec),
        }

    working_df = _load_dataframe(data_path=data_path, df=df)
    mediator_columns = _mediator_columns(m)
    vector_mediator = len(mediator_columns) > 1
    cleaned_df = remove_missing_from_df(
        df=working_df,
        d=d,
        m=mediator_columns if vector_mediator else mediator_columns[0],
        y=y,
        reg_formula=reg_formula,
    )
    cluster = _validate_cluster_column(cleaned_df, cluster=cluster, treatment=d)

    treatment_support = normalize_binary_support(cleaned_df[d], column=d)
    if vector_mediator:
        mediator_levels = _ordered_vector_support_values(cleaned_df, mediator_columns)
    else:
        mediator_levels = _ordered_support_values(cleaned_df[mediator_columns[0]])
        _validate_scalar_ordered_mediator_support(
            series=cleaned_df[mediator_columns[0]],
            levels=mediator_levels,
        )
    _validate_sharp_null_regression_scope(
        method=method,
        regression_spec=regression_spec,
        vector_mediator=vector_mediator,
        mediator_level_count=len(mediator_levels),
    )

    requested_num_y_bins = num_y_bins
    y_processed = cleaned_df[y]
    if method == "K":
        y_processed = cleaned_df[y]
    elif num_y_bins is not None:
        y_processed = discretize_y(cleaned_df[y], num_bins=num_y_bins)
    elif len(cleaned_df) / cleaned_df[y].nunique(dropna=False) <= 30:
        y_processed = discretize_y(cleaned_df[y], num_bins=5)

    analysis_df = cleaned_df.copy()
    analysis_df["_tm_y_processed"] = y_processed
    analysis_df["_tm_d_processed"] = treatment_support.transform(analysis_df[d])
    mediator_level_map = {level: index for index, level in enumerate(mediator_levels)}
    if vector_mediator:
        analysis_df["_tm_m_processed"] = [
            mediator_level_map[level]
            for level in _vector_mediator_keys(analysis_df, mediator_columns)
        ]
    else:
        analysis_df["_tm_m_processed"] = (
            analysis_df[mediator_columns[0]].map(mediator_level_map).astype(int)
        )


    if method == "K":
        if frac_ats_affected is not None:
            raise NotImplementedError(
                "frac_ats_affected uses the ordered-nuisance CS formulation, not method='K'."
            )
        if max_defiers_share > 0.0:
            raise NotImplementedError(
                "max_defiers_share uses the ordered-nuisance sharp-null formulation, not method='K'."
            )
        if vector_mediator or len(mediator_levels) != 2:
            raise NotImplementedError(
                "The K runner is release-scoped to scalar binary mediators, as in the paper comparator rows."
            )
        return _test_binary_kitagawa(
            analysis_df=analysis_df,
            d="_tm_d_processed",
            m="_tm_m_processed",
            y="_tm_y_processed",
            diagnostics_d=d,
            diagnostics_m=mediator_columns[0],
            original_mediator_levels=mediator_levels,
            requested_num_y_bins=requested_num_y_bins,
            alpha=alpha,
            cluster=cluster,
            treatment_support=treatment_support,
            mediator_columns=mediator_columns,
            bootstrap_replications=bootstrap_replications,
            random_state=random_state,
            xi=kitagawa_xi,
        )

    if method in {"FSSTdd", "FSSTndd"}:
        fsst_lambda_mode = "dd" if method == "FSSTdd" else "ndd"
        if vector_mediator:
            return _test_ordered_nonbinary_fsst(
                analysis_df=analysis_df,
                d="_tm_d_processed",
                m="_tm_m_processed",
                y="_tm_y_processed",
                original_mediator_levels=mediator_levels,
                requested_num_y_bins=requested_num_y_bins,
                alpha=alpha,
                cluster=cluster,
                treatment_support=treatment_support,
                mediator_ordering=_elementwise_mediator_ordering(mediator_levels),
                mediator_columns=mediator_columns,
                support_normalization=(
                    "vector mediator support is mapped to consecutive integer "
                    "levels after deterministic tuple ordering"
                ),
                fsst_scope="vector mediator with elementwise monotonicity",
                method=method,
                lambda_mode=fsst_lambda_mode,
                bootstrap_replications=bootstrap_replications,
                random_state=random_state,
                frac_ats_affected=frac_ats_affected,
                max_defiers_share=max_defiers_share,
            )
        if len(mediator_levels) > 2 or max_defiers_share > 0.0:
            return _test_ordered_nonbinary_fsst(
                analysis_df=analysis_df,
                d="_tm_d_processed",
                m="_tm_m_processed",
                y="_tm_y_processed",
                original_mediator_levels=mediator_levels,
                requested_num_y_bins=requested_num_y_bins,
                alpha=alpha,
                cluster=cluster,
                treatment_support=treatment_support,
                mediator_ordering=_total_order_mediator_ordering(len(mediator_levels)),
                mediator_columns=mediator_columns,
                support_normalization=(
                    "ordered nonbinary mediator support is mapped to consecutive "
                    "integer levels in deterministic support order"
                    if len(mediator_levels) > 2
                    else "binary support is mapped to internal {0, 1} in deterministic support order"
                ),
                fsst_scope="ordered scalar nonbinary mediator with monotonicity",
                method=method,
                lambda_mode=fsst_lambda_mode,
                bootstrap_replications=bootstrap_replications,
                random_state=random_state,
                frac_ats_affected=frac_ats_affected,
                max_defiers_share=max_defiers_share,
            )
        if len(mediator_levels) != 2:
            raise ValueError(f"{m} must contain at least two support levels.")
        beta_observed = _compute_binary_beta(
            dvec=analysis_df["_tm_d_processed"],
            mvec=analysis_df["_tm_m_processed"],
            yvec=analysis_df["_tm_y_processed"],
            mediator_values=[0, 1],
        )
        bootstrap_betas = _bootstrap_beta_draws(
            analysis_df=analysis_df,
            d="_tm_d_processed",
            cluster=cluster,
            bootstrap_replications=bootstrap_replications,
            random_state=random_state,
            statistic=lambda frame: _compute_binary_beta(
                dvec=frame["_tm_d_processed"],
                mvec=frame["_tm_m_processed"],
                yvec=frame["_tm_y_processed"],
                mediator_values=[0, 1],
            ),
        )
        fsst_result = _fsst_nonuisance_test(
            beta=beta_observed,
            bootstrap_betas=bootstrap_betas,
            alpha=alpha,
            lambda_mode=fsst_lambda_mode,
            effective_sample_size=_effective_bootstrap_sample_size(
                analysis_df=analysis_df,
                cluster=cluster,
            ),
        )
        diagnostics = build_cell_count_diagnostics(
            df=analysis_df,
            d=d,
            m=m,
            y="_tm_y_processed",
            cluster=cluster,
            requested_num_y_bins=requested_num_y_bins,
            applied_num_y_bins=int(analysis_df["_tm_y_processed"].nunique(dropna=False)),
            no_bite_reason="FSST uses the paper's bootstrap moment-selection runner",
            support_diagnostics={
                **treatment_support.diagnostics(
                    original_key="original_treatment_levels",
                    normalized_key="normalized_treatment_levels",
                ),
                "original_mediator_levels": [_normalize_scalar(value) for value in mediator_levels],
                "normalized_mediator_levels": [0, 1],
                "mediator_columns": list(mediator_columns),
                "mediator_dimension": 1,
                "vector_mediator": False,
                "support_normalization": "binary support is mapped to internal {0, 1} in deterministic support order",
                "fsst_scope": "binary mediator without nuisance parameters",
                "fsst_reference": (
                    "manuscript/sources/arxiv-2404.11739v3/draft.tex:412-421; "
                    "packages/r/TestMechs/R/test_sharp_null_binary_m.R:193-224"
                ),
            },
        )
        diagnostics["fsst"] = fsst_result["diagnostics"]
        return SharpNullResult(
            method=method,
            null_hypothesis="Y(0,m) = Y(1,m) for all mediator support points",
            reject=bool(fsst_result["reject"]),
            test_stat=float(fsst_result["test_stat"]),
            critical_value=float(fsst_result["critical_value"]),
            p_value=float(fsst_result["p_value"]),
            beta_observed=beta_observed.tolist(),
            approximation=(
                "FSST bootstrap moment-selection runner using the paper's "
                f"{fsst_lambda_mode} lambda protocol."
            ),
            diagnostics=diagnostics,
        )

    if method == "ARP":
        if vector_mediator:
            result = _test_ordered_nonbinary_arp(
                analysis_df=analysis_df,
                d="_tm_d_processed",
                m="_tm_m_processed",
                y="_tm_y_processed",
                original_mediator_levels=mediator_levels,
                requested_num_y_bins=requested_num_y_bins,
                alpha=alpha,
                cluster=cluster,
                treatment_support=treatment_support,
                mediator_ordering=_elementwise_mediator_ordering(mediator_levels),
                mediator_columns=mediator_columns,
                support_normalization=(
                    "vector mediator support is mapped to consecutive integer "
                    "levels after deterministic tuple ordering"
                ),
                arp_scope="vector mediator with elementwise monotonicity",
                frac_ats_affected=frac_ats_affected,
                max_defiers_share=max_defiers_share,
            )
            return result

        if len(mediator_levels) > 2 or max_defiers_share > 0.0:
            result = _test_ordered_nonbinary_arp(
                analysis_df=analysis_df,
                d="_tm_d_processed",
                m="_tm_m_processed",
                y="_tm_y_processed",
                original_mediator_levels=mediator_levels,
                requested_num_y_bins=requested_num_y_bins,
                alpha=alpha,
                cluster=cluster,
                treatment_support=treatment_support,
                mediator_ordering=_total_order_mediator_ordering(len(mediator_levels)),
                mediator_columns=mediator_columns,
                support_normalization=(
                    "ordered nonbinary mediator support is mapped to consecutive "
                    "integer levels in deterministic support order"
                    if len(mediator_levels) > 2
                    else "binary support is mapped to internal {0, 1} in deterministic support order"
                ),
                arp_scope="ordered scalar nonbinary mediator with monotonicity",
                frac_ats_affected=frac_ats_affected,
                max_defiers_share=max_defiers_share,
            )
            return result

        if len(mediator_levels) != 2:
            raise ValueError(f"{m} must contain at least two support levels.")

        beta_observed = _compute_binary_beta(
            dvec=analysis_df["_tm_d_processed"],
            mvec=analysis_df["_tm_m_processed"],
            yvec=analysis_df["_tm_y_processed"],
            mediator_values=[0, 1],
        )
        sigma = _analytic_variance_binary(
            dvec=analysis_df["_tm_d_processed"],
            mvec=analysis_df["_tm_m_processed"],
            yvec=analysis_df["_tm_y_processed"],
            clustervec=analysis_df[cluster] if cluster is not None else None,
            mediator_values=[0, 1],
        )
        arp_result = _arp_honest_test(
            y_t=beta_observed,
            x_t=np.zeros((len(beta_observed), 1), dtype=float),
            sigma=sigma,
            alpha=alpha,
            hybrid_kappa=alpha / 10.0,
        )

        diagnostics = build_cell_count_diagnostics(
            df=analysis_df,
            d=d,
            m=m,
            y="_tm_y_processed",
            cluster=cluster,
            requested_num_y_bins=requested_num_y_bins,
            applied_num_y_bins=int(analysis_df["_tm_y_processed"].nunique(dropna=False)),
            no_bite_reason="ARP uses the paper's hybrid moment-inequality runner",
            support_diagnostics={
                **treatment_support.diagnostics(
                    original_key="original_treatment_levels",
                    normalized_key="normalized_treatment_levels",
                ),
                "original_mediator_levels": [_normalize_scalar(value) for value in mediator_levels],
                "normalized_mediator_levels": [0, 1],
                "mediator_columns": list(mediator_columns),
                "mediator_dimension": 1,
                "vector_mediator": False,
                "support_normalization": "binary support is mapped to internal {0, 1} in deterministic support order",
                "arp_scope": "binary mediator with monotonicity",
                "arp_reference": "manuscript/sources/arxiv-2404.11739v3/draft.tex:412-421; packages/r/TestMechs/R/test_sharp_null_binary_m.R:225-251",
            },
        )
        diagnostics["arp"] = arp_result["diagnostics"]
        if regression_diagnostics is not None:
            diagnostics["regression"] = regression_diagnostics

        return SharpNullResult(
            method="ARP",
            null_hypothesis="Y(0,m) = Y(1,m) for all mediator support points",
            reject=bool(arp_result["reject"]),
            test_stat=float(arp_result["standardized_stat"]),
            critical_value=float(arp_result["critical_value"]),
            p_value=float("nan"),
            beta_observed=beta_observed.tolist(),
            approximation="Paper hybrid moment-inequality ARP runner with analytic variance; p-value not reported by the R reference.",
            diagnostics=diagnostics,
        )

    if vector_mediator:
        result = _test_ordered_nonbinary_cs(
            analysis_df=analysis_df,
            d="_tm_d_processed",
            m="_tm_m_processed",
            y="_tm_y_processed",
            original_mediator_levels=mediator_levels,
            requested_num_y_bins=requested_num_y_bins,
            alpha=alpha,
            cluster=cluster,
            treatment_support=treatment_support,
            regression_diagnostics=regression_diagnostics,
            mediator_ordering=_elementwise_mediator_ordering(mediator_levels),
            mediator_columns=mediator_columns,
            support_normalization=(
                "vector mediator support is mapped to consecutive integer "
                "levels after deterministic tuple ordering"
            ),
            cs_scope="vector mediator with elementwise monotonicity",
            frac_ats_affected=frac_ats_affected,
            max_defiers_share=max_defiers_share,
        )
        return result

    if len(mediator_levels) > 2 or frac_ats_affected is not None or max_defiers_share > 0.0:
        beta_observed_override = None
        sigma_obs_override = None
        diagnostics_df = None
        if (
            frac_ats_affected is not None
            and regression_spec is not None
            and regression_spec.formula_kind in {"controls", "fixed_effects"}
            and not vector_mediator
            and len(mediator_levels) == 2
        ):
            (
                beta_observed_override,
                sigma_obs_override,
                adjusted_regression_diagnostics,
                row_index,
            ) = _adjusted_ordered_binary_beta_and_variance(
                analysis_df=analysis_df,
                d="_tm_d_processed",
                m="_tm_m_processed",
                y="_tm_y_processed",
                clustervec=analysis_df[cluster] if cluster is not None else None,
                mediator_values=[0, 1],
                regression_spec=regression_spec,
            )
            diagnostics_df = analysis_df.loc[row_index]
            if regression_diagnostics is not None:
                regression_diagnostics = {
                    **regression_diagnostics,
                    **adjusted_regression_diagnostics,
                    **treatment_support.diagnostics(
                        original_key="original_treatment_levels",
                        normalized_key="normalized_treatment_levels",
                    ),
                }
        result = _test_ordered_nonbinary_cs(
            analysis_df=analysis_df,
            d="_tm_d_processed",
            m="_tm_m_processed",
            y="_tm_y_processed",
            original_mediator_levels=mediator_levels,
            requested_num_y_bins=requested_num_y_bins,
            alpha=alpha,
            cluster=cluster,
            treatment_support=treatment_support,
            regression_diagnostics=regression_diagnostics,
            mediator_ordering=_total_order_mediator_ordering(len(mediator_levels)),
            mediator_columns=mediator_columns,
            support_normalization=(
                "ordered nonbinary mediator support is mapped to consecutive "
                "integer levels in deterministic support order"
                if len(mediator_levels) > 2
                else "binary support is mapped to internal {0, 1} in deterministic support order"
            ),
            cs_scope=(
                "ordered scalar nonbinary mediator with monotonicity"
                if len(mediator_levels) > 2
                else "binary mediator with monotonicity"
            ),
            frac_ats_affected=frac_ats_affected,
            max_defiers_share=max_defiers_share,
            beta_observed_override=beta_observed_override,
            sigma_obs_override=sigma_obs_override,
            diagnostics_df=diagnostics_df,
        )
        return result

    if len(mediator_levels) != 2:
        raise ValueError(f"{m} must contain at least two support levels.")

    diagnostics_df = analysis_df
    if regression_spec is not None and regression_spec.formula_kind in {"controls", "fixed_effects"}:
        beta_observed, sigma, adjusted_regression_diagnostics, row_index = _adjusted_binary_beta_and_variance(
            analysis_df=analysis_df,
            d="_tm_d_processed",
            m="_tm_m_processed",
            y="_tm_y_processed",
            clustervec=analysis_df[cluster] if cluster is not None else None,
            mediator_values=[0, 1],
            regression_spec=regression_spec,
        )
        diagnostics_df = analysis_df.loc[row_index]
        if regression_diagnostics is not None:
            regression_diagnostics = {
                **regression_diagnostics,
                **adjusted_regression_diagnostics,
                **treatment_support.diagnostics(
                    original_key="original_treatment_levels",
                    normalized_key="normalized_treatment_levels",
                ),
            }
    else:
        beta_observed = _compute_binary_beta(
            dvec=analysis_df["_tm_d_processed"],
            mvec=analysis_df["_tm_m_processed"],
            yvec=analysis_df["_tm_y_processed"],
            mediator_values=[0, 1],
        )
        sigma = _analytic_variance_binary(
            dvec=analysis_df["_tm_d_processed"],
            mvec=analysis_df["_tm_m_processed"],
            yvec=analysis_df["_tm_y_processed"],
            clustervec=analysis_df[cluster] if cluster is not None else None,
            mediator_values=[0, 1],
        )
    cs_result = _cox_shi_nonuisance(y=-beta_observed, sigma=sigma, alpha=alpha, refinement=True)

    diagnostics = build_cell_count_diagnostics(
        df=diagnostics_df,
        d=d,
        m=m,
        y="_tm_y_processed",
        cluster=cluster,
        requested_num_y_bins=requested_num_y_bins,
        applied_num_y_bins=int(diagnostics_df["_tm_y_processed"].nunique(dropna=False)),
        no_bite_reason="theta_kk_min is not identified in the CS baseline",
        support_diagnostics={
            **treatment_support.diagnostics(
                original_key="original_treatment_levels",
                normalized_key="normalized_treatment_levels",
            ),
            "original_mediator_levels": [_normalize_scalar(value) for value in mediator_levels],
            "normalized_mediator_levels": [0, 1],
            "mediator_columns": list(mediator_columns),
            "mediator_dimension": 1,
            "vector_mediator": False,
            "support_normalization": "binary support is mapped to internal {0, 1} in deterministic support order",
        },
    )
    if regression_diagnostics is not None:
        diagnostics["regression"] = regression_diagnostics

    return SharpNullResult(
        method=method,
        null_hypothesis="Y(0,m) = Y(1,m) for all mediator support points",
        reject=bool(cs_result["reject"]),
        test_stat=float(cs_result["test_stat"]),
        critical_value=float(cs_result["critical_value"]),
        p_value=float(cs_result["p_value"]),
        beta_observed=beta_observed.tolist(),
        approximation="Discretized Y with explicit num_y_bins; valid but potentially non-sharp.",
        diagnostics=diagnostics,
    )


def _test_ordered_nonbinary_arp(
    *,
    analysis_df: pd.DataFrame,
    d: str,
    m: str,
    y: str,
    original_mediator_levels: tuple[object, ...],
    requested_num_y_bins: int | None,
    alpha: float,
    cluster: str | None,
    treatment_support: object,
    mediator_ordering: tuple[tuple[int, int], ...],
    mediator_columns: tuple[str, ...],
    support_normalization: str,
    arp_scope: str,
    frac_ats_affected: float | None,
    max_defiers_share: float,
) -> SharpNullResult:
    mediator_values = list(range(len(original_mediator_levels)))
    y_values = _series_support_values(analysis_df[y])
    beta_observed = _compute_ordered_nonbinary_beta(
        dvec=analysis_df[d],
        mvec=analysis_df[m],
        yvec=analysis_df[y],
        mediator_values=mediator_values,
        y_values=y_values,
    )
    matrices = _construct_ordered_nonbinary_moment_matrices(
        mediator_count=len(mediator_values),
        outcome_count=len(y_values),
        allowed_theta_pairs=mediator_ordering,
        frac_ats_affected=frac_ats_affected,
        max_defiers_share=max_defiers_share,
    )
    sigma_obs = _analytic_variance_ordered_nonbinary(
        dvec=analysis_df[d],
        mvec=analysis_df[m],
        yvec=analysis_df[y],
        clustervec=analysis_df[cluster] if cluster is not None else None,
        mediator_values=mediator_values,
        y_values=y_values,
    )
    a_matrix = np.vstack([matrices["a_observed"], matrices["a_shape"]])
    arp_result = _arp_honest_test(
        y_t=beta_observed,
        x_t=np.zeros((len(beta_observed), 1), dtype=float),
        sigma=sigma_obs,
        alpha=alpha,
        hybrid_kappa=alpha / 10.0,
    )

    if max_defiers_share > 0.0 and "relaxed monotonicity defier cap" not in arp_scope:
        arp_scope = f"{arp_scope} and relaxed monotonicity defier cap"

    diagnostics = build_cell_count_diagnostics(
        df=analysis_df,
        d=d,
        m=m,
        y=y,
        cluster=cluster,
        requested_num_y_bins=requested_num_y_bins,
        applied_num_y_bins=len(y_values),
        no_bite_reason="ARP uses the paper's hybrid moment-inequality runner",
        support_diagnostics={
            **treatment_support.diagnostics(
                original_key="original_treatment_levels",
                normalized_key="normalized_treatment_levels",
            ),
            "original_mediator_levels": [
                _normalize_mediator_level(value) for value in original_mediator_levels
            ],
            "normalized_mediator_levels": mediator_values,
            "mediator_columns": list(mediator_columns),
            "mediator_dimension": len(mediator_columns),
            "vector_mediator": len(mediator_columns) > 1,
            "support_normalization": support_normalization,
            "arp_scope": arp_scope,
            "arp_reference": "manuscript/sources/arxiv-2404.11739v3/draft.tex:412-421; packages/r/TestMechs/R/test_sharp_null.R:326-360,707-977",
            "allowed_mediator_type_count": len(mediator_ordering),
            "forbidden_mediator_type_count": len(mediator_values) ** 2 - len(mediator_ordering),
            "allowed_mediator_type_pairs": [
                {"from": low, "to": high} for low, high in mediator_ordering
            ],
            "nuisance_parameter_count": int(a_matrix.shape[1]),
            "moment_inequality_count": int(a_matrix.shape[0]),
            "observed_moment_count": int(beta_observed.shape[0]),
            "shape_moment_count": int(matrices["beta_shape"].shape[0]),
            "arp_eta": float(arp_result["eta"]),
            "arp_standardized_stat": float(arp_result["standardized_stat"]),
            "arp_critical_value": float(arp_result["critical_value"]),
            "arp_sigma_B": float(arp_result["sigma_B"]),
            "arp_lambda": arp_result["lambda"].tolist(),
        },
    )

    if max_defiers_share > 0.0:
        diagnostics["relaxed_monotonicity"] = _relaxed_monotonicity_diagnostics(
            mediator_levels=original_mediator_levels,
            monotone_pairs=mediator_ordering,
            max_defiers_share=max_defiers_share,
            matrices=matrices,
        )
    diagnostics["arp"] = arp_result["diagnostics"]
    return SharpNullResult(
        method="ARP",
        null_hypothesis="Y(0,m) = Y(1,m) for all mediator support points",
        reject=bool(arp_result["reject"]),
        test_stat=float(arp_result["standardized_stat"]),
        critical_value=float(arp_result["critical_value"]),
        p_value=float("nan"),
        beta_observed=beta_observed.tolist(),
        approximation=(
            "Paper hybrid moment-inequality ARP runner with analytic variance; "
            "p-value not reported by the R reference."
        ),
        diagnostics=diagnostics,
    )


def ci_TV(
    *,
    data_path: str | Path | None = None,
    df: pd.DataFrame | None = None,
    d: str,
    m: str,
    y: str,
    at_group: object,
    ordering: Mapping[object, Sequence[object]] | None = None,
    alpha: float = 0.05,
    bootstrap_replications: int = 500,
    grid_step: float = 0.02,
    bisec: bool = False,
    eps: float | None = None,
    max_bisec_iterations: int = 25,
    weight_matrix: str = "diag",
    method: str = "FSSTdd",
    random_state: int | None = None,
) -> TVConfidenceIntervalResult:
    """Confidence interval for the total-variation causal effect.

    Inverts the FSST moment-inequality test to construct a confidence
    interval for the total-variation distance
    ``TV_k = 0.5 * sum_y abs(P(Y(1,k)=y) - P(Y(0,k)=y))`` for a
    specified always-taker mediator group k.
    The interval quantifies how much the outcome distribution shifts
    when treatment changes for individuals who always take mediator
    level k.

    The current release supports scalar ordered mediators and discrete
    outcomes, using grid search (and optional bisection refinement)
    over the unit interval [0, 1].

    Parameters
    ----------
    data_path : str or Path, optional
        Path to a CSV file containing the analysis data. Exactly one
        of ``data_path`` or ``df`` must be provided.
    df : pd.DataFrame, optional
        Analysis data frame. Exactly one of ``data_path`` or ``df``
        must be provided.
    d : str
        Name of the binary treatment column in the data.
    m : str
        Name of the scalar mediator column. Must be an ordered
        discrete variable with finite support.
    y : str
        Name of the discrete outcome column.
    at_group : object
        The always-taker mediator level for which the TV confidence
        interval is computed. Must be a support point of M.
    ordering : mapping, optional
        Explicit mediator ordering as a dictionary mapping each
        mediator level to the sequence of levels it dominates.
        If ``None``, natural sort order of mediator values is used.
    alpha : float, default 0.05
        Significance level for the confidence interval (produces a
        1 - alpha confidence interval).
    bootstrap_replications : int, default 500
        Number of bootstrap replications for FSST critical-value
        simulation.
    grid_step : float, default 0.02
        Step size for the grid search over [0, 1]. Smaller values
        give finer resolution at increased computational cost.
    bisec : bool, default False
        Whether to refine endpoints with bisection after the initial
        grid search. Improves precision of the confidence bounds.
    eps : float, optional
        Convergence tolerance for bisection. Defaults to
        ``alpha * 0.1`` if not specified.
    max_bisec_iterations : int, default 25
        Maximum number of bisection iterations per endpoint.
    weight_matrix : str, default "diag"
        Weight matrix for the FSST test statistic. Currently only
        ``"diag"`` (diagonal inverse-variance) is supported.
    method : str, default "FSSTdd"
        FSST variant to use. One of:

        - ``"FSSTdd"`` : data-driven lambda selection.
        - ``"FSSTndd"`` : non-data-driven lambda.
    random_state : int, optional
        Random seed for reproducibility of bootstrap draws.

    Returns
    -------
    TVConfidenceIntervalResult
        Result object containing:

        - ``lower`` : float - lower bound of the CI.
        - ``upper`` : float - upper bound of the CI.
        - ``test_grid`` : list - grid-point test decisions.
        - ``diagnostics`` : dict - solver and convergence info.
        - ``to_frame()`` : summary as a one-row DataFrame.
        - ``to_dict()`` : strict-JSON-safe payload.

    Raises
    ------
    ValueError
        If both or neither of ``data_path``/``df`` are provided,
        if ``at_group`` is not a support point of M, if the outcome
        has fewer than two support points, or if parameters are out
        of range.
    NotImplementedError
        If ``method`` is not ``"FSSTdd"`` or ``"FSSTndd"``.

    Examples
    --------
    >>> import pandas as pd
    >>> import testmechs
    >>> df = pd.DataFrame({
    ...     "treat": [0, 0, 0, 0, 1, 1, 1, 1],
    ...     "mediator": [0, 0, 1, 1, 0, 1, 1, 1],
    ...     "outcome": [0, 1, 0, 1, 0, 0, 1, 1],
    ... })
    >>> ci = testmechs.ci_TV(
    ...     df=df, d="treat", m="mediator", y="outcome",
    ...     at_group=1, grid_step=0.1
    ... )
    >>> ci.lower  # doctest: +SKIP
    0.0
    >>> ci.upper  # doctest: +SKIP
    0.8

    Notes
    -----
    Implements the confidence-interval construction from Kwon and
    Roth (2024) [1]_. The total-variation target is defined as
    eta_k = theta_kk * TV_kk, where theta_kk is the always-taker
    proportion for group k and TV_kk is the TV distance for that
    group. The interval is obtained by inverting the FSST test at
    each grid point and collecting values not rejected.

    References
    ----------
    .. [1] Kwon, S. and Roth, J. (2024). "Testing Mechanisms."
       arXiv:2404.11739v3.

    See Also
    --------
    test_sharp_null : Main sharp-null test with multiple method options.
    test_sharp_null_cr : CR confidence-set approach.
    lb_frac_affected : Lower bound on the fraction of always-takers affected.
    """

    d = _validate_scalar_column_name(d, name="d")
    m = _validate_scalar_column_name(m, name="m")
    y = _validate_scalar_column_name(y, name="y")
    alpha = _validate_probability_share(alpha, name="alpha", strict=True)
    bootstrap_replications = _validate_positive_integer(
        bootstrap_replications,
        name="bootstrap_replications",
        minimum=2,
        context="ci_TV FSST inversion",
    )
    grid_step = _validate_grid_step(grid_step)
    bisec = _validate_boolean(bisec, name="bisec")
    eps = _validate_positive_real(alpha * 0.1 if eps is None else eps, name="eps")
    max_bisec_iterations = _validate_positive_integer(
        max_bisec_iterations,
        name="max_bisec_iterations",
        minimum=1,
        context="ci_TV bisection inversion",
    )
    weight_matrix = _validate_fsst_weight_matrix(weight_matrix)
    random_state = _validate_optional_nonnegative_integer(random_state, name="random_state")
    if method == "FSST":
        method = "FSSTdd"
    if method not in {"FSSTdd", "FSSTndd"}:
        raise NotImplementedError("ci_TV currently supports method='FSSTdd' or method='FSSTndd'.")
    lambda_mode = "dd" if method == "FSSTdd" else "ndd"

    working_df = _load_dataframe(data_path=data_path, df=df)
    cleaned_df = remove_missing_from_df(df=working_df, d=d, m=m, y=y)
    treatment_support = normalize_binary_support(cleaned_df[d], column=d)
    mediator_levels = _ordered_support_values(cleaned_df[m])
    _validate_scalar_ordered_mediator_support(series=cleaned_df[m], levels=mediator_levels)
    if at_group not in mediator_levels:
        raise ValueError("at_group must be a support point of M.")
    y_levels = _ordered_support_values(cleaned_df[y])
    if len(y_levels) < 2:
        raise ValueError("ci_TV requires a discrete outcome with at least two observed support points.")

    analysis_df = cleaned_df.copy()
    mediator_level_map = {level: index for index, level in enumerate(mediator_levels)}
    analysis_df["_tm_d_processed"] = treatment_support.transform(analysis_df[d])
    analysis_df["_tm_m_processed"] = analysis_df[m].map(mediator_level_map).astype(int)
    analysis_df["_tm_y_processed"] = pd.Categorical(analysis_df[y], categories=y_levels, ordered=True)

    mediator_values = list(range(len(mediator_levels)))
    at_index = mediator_level_map[at_group]
    beta_observed = _compute_ordered_nonbinary_beta(
        dvec=analysis_df["_tm_d_processed"],
        mvec=analysis_df["_tm_m_processed"],
        yvec=analysis_df["_tm_y_processed"],
        mediator_values=mediator_values,
        y_values=y_levels,
    )
    bootstrap_betas = _bootstrap_beta_draws(
        analysis_df=analysis_df,
        d="_tm_d_processed",
        cluster=None,
        bootstrap_replications=bootstrap_replications,
        random_state=random_state,
        statistic=lambda frame: _compute_ordered_nonbinary_beta(
            dvec=frame["_tm_d_processed"],
            mvec=frame["_tm_m_processed"],
            yvec=frame["_tm_y_processed"],
            mediator_values=mediator_values,
            y_values=y_levels,
        ),
    )
    mediator_ordering = _parse_scalar_mediator_ordering(
        ordering=ordering,
        mediator_levels=mediator_levels,
        parameter_name="ordering",
    )
    tv_matrices = _construct_ordered_nonbinary_tv_moment_matrices(
        mediator_count=len(mediator_values),
        outcome_count=len(y_levels),
        at_index=at_index,
        allowed_theta_pairs=mediator_ordering,
    )
    test_rows: list[dict[str, object]] = []
    accepted: list[float] = []
    effective_n = _effective_bootstrap_sample_size(analysis_df=analysis_df, cluster=None)

    def test_tv_null(null_value: float) -> dict[str, object]:
        a_shape = _tv_a_shape_for_null(tv_matrices=tv_matrices, null_value=float(null_value))
        fsst_result = _fsst_nuisance_test(
            beta_observed=beta_observed,
            bootstrap_betas=bootstrap_betas,
            a_observed=tv_matrices["a_observed"],
            a_shape=a_shape,
            beta_shape=tv_matrices["beta_shape"],
            alpha=alpha,
            lambda_mode=lambda_mode,
            effective_sample_size=effective_n,
            weight_matrix=weight_matrix,
        )
        p_value = float(fsst_result["p_value"])
        reject = bool(fsst_result["reject"])
        return {
            "null_value": float(null_value),
            "p_value": p_value,
            "reject": reject,
            "test_stat": float(fsst_result["test_stat"]),
            "critical_value": float(fsst_result["critical_value"]),
        }

    if bisec:
        lower_row = test_tv_null(0.0)
        upper_row = test_tv_null(1.0)
        test_rows.extend([lower_row, upper_row])
        lower_accepted = not bool(lower_row["reject"])
        upper_accepted = not bool(upper_row["reject"])
        if lower_accepted:
            accepted.append(0.0)
        if upper_accepted:
            accepted.append(1.0)

        bisection_iterations = 0
        if lower_accepted != upper_accepted:
            rejected_endpoint = 1.0 if lower_accepted else 0.0
            accepted_endpoint = 0.0 if lower_accepted else 1.0
            while (
                bisection_iterations < max_bisec_iterations
                and abs(rejected_endpoint - accepted_endpoint) > eps
            ):
                midpoint = (rejected_endpoint + accepted_endpoint) / 2.0
                row = test_tv_null(midpoint)
                test_rows.append(row)
                bisection_iterations += 1
                if bool(row["reject"]):
                    rejected_endpoint = midpoint
                else:
                    accepted_endpoint = midpoint
                    accepted.append(float(midpoint))
    else:
        grid = _tv_grid(grid_step)
        bisection_iterations = None
        for null_value in grid:
            row = test_tv_null(float(null_value))
            if not bool(row["reject"]):
                accepted.append(float(null_value))
            test_rows.append(row)

    lower = min(accepted) if accepted else None
    upper = max(accepted) if accepted else None
    diagnostics = {
        "paper_reference": "manuscript/sources/arxiv-2404.11739v3/draft.tex:657-688",
        "fsst_reference": "manuscript/sources/arxiv-2404.11739v3/draft.tex:416-421",
        "reference_implementation": "packages/r/TestMechs/R/ci_TV.R",
        "reference_boundary": (
            "R ci_TV grid branch contains an orphan `j`; Python implements the "
            "paper/R matrix target with explicit grid inversion instead of "
            "copying that failure."
        ),
        "inversion": "bisection" if bisec else "grid",
        "grid_step": float(grid_step),
        "grid_points": int(len(test_rows)),
        "accepted_grid_points": int(len(accepted)),
        "bisection_tolerance": float(eps),
        "bisection_iterations": bisection_iterations,
        "max_bisection_iterations": int(max_bisec_iterations),
        "bootstrap_replications": int(bootstrap_replications),
        "lambda_mode": lambda_mode,
        "weight_matrix": weight_matrix,
        "mediator_levels": [_normalize_mediator_level(level) for level in mediator_levels],
        "ordering": _ordering_diagnostics(mediator_levels, mediator_ordering),
        "at_group_index": int(at_index),
        "outcome_levels": [_normalize_mediator_level(level) for level in y_levels],
        "moment_count": int(beta_observed.size),
        "nuisance_parameter_count": int(tv_matrices["a_observed"].shape[1]),
    }
    return TVConfidenceIntervalResult(
        at_group=at_group,
        alpha=alpha,
        method=method,
        accepted_grid=accepted,
        lower=lower,
        upper=upper,
        test_grid=test_rows,
        approximation=(
            "FSST grid/bisection inversion for the total-variation target eta_k = "
            "theta_kk * TV_kk, release-scoped to scalar ordered mediators and "
            "discrete outcomes."
        ),
        diagnostics=diagnostics,
    )


def test_sharp_null_cr(
    *,
    data_path: str | Path | None = None,
    df: pd.DataFrame | None = None,
    d: str,
    m: str,
    y: str,
    ordering: Mapping[object, Sequence[object]] | None = None,
    B: int = 500,
    eps_bar: float = 1e-3,
    alpha: float = 0.05,
    num_Ybins: int | None = None,
    random_state: int | None = None,
) -> SharpNullResult:
    """Test the sharp null via CR confidence-set inversion.

    Constructs a confidence set for the joint distribution of potential
    outcomes under the sharp null H0: Y(1,m) = Y(0,m) for all m, using
    a linear-programming feasibility approach. The test rejects when
    the confidence set is empty at level ``alpha``.

    This implementation covers scalar ordered mediators with explicit
    mediator orderings and discrete outcomes. It uses SciPy's HiGHS
    linear programming solver in place of the R package's Gurobi
    dependency, making the test freely reproducible without a
    commercial license.

    Parameters
    ----------
    data_path : str or Path, optional
        Path to a CSV file containing the analysis data. Exactly one
        of ``data_path`` or ``df`` must be provided.
    df : pd.DataFrame, optional
        Analysis data frame. Exactly one of ``data_path`` or ``df``
        must be provided.
    d : str
        Name of the binary treatment column in the data.
    m : str
        Name of the scalar mediator column. Must be an ordered
        discrete variable with finite support.
    y : str
        Name of the discrete outcome column.
    ordering : mapping, optional
        Explicit mediator ordering as a dictionary mapping each
        mediator level to the sequence of levels it dominates.
        If ``None``, natural sort order of mediator values is used.
    B : int, default 500
        Number of bootstrap replications for constructing the
        confidence set.
    eps_bar : float, default 1e-3
        Tolerance for the linear programming feasibility check.
        Values below this threshold are treated as zero.
    alpha : float, default 0.05
        Significance level for the hypothesis test.
    num_Ybins : int, optional
        If specified, discretize Y into this many quantile bins
        before testing.
    random_state : int, optional
        Random seed for reproducibility of bootstrap draws.

    Returns
    -------
    SharpNullResult
        Result object containing:

        - ``reject`` : bool - whether the null is rejected at level alpha.
        - ``p_value`` : float - p-value from the bootstrap inversion.
        - ``method`` : str - ``"CR"``.
        - ``diagnostics`` : dict - LP solver status and support info.
        - ``to_frame()`` : summary as a one-row DataFrame.
        - ``to_dict()`` : strict-JSON-safe payload.

    Raises
    ------
    ValueError
        If both or neither of ``data_path``/``df`` are provided,
        if required columns are missing, if treatment is not binary,
        or if the mediator has fewer than two support points.
    NotImplementedError
        If a vector mediator is passed (only scalar mediators are
        supported in this function).

    Examples
    --------
    >>> import pandas as pd
    >>> import testmechs
    >>> df = pd.DataFrame({
    ...     "treat": [0, 0, 0, 0, 1, 1, 1, 1],
    ...     "mediator": [0, 0, 1, 1, 0, 1, 1, 1],
    ...     "outcome": [0, 1, 0, 1, 0, 0, 1, 1],
    ... })
    >>> result = testmechs.test_sharp_null_cr(
    ...     df=df, d="treat", m="mediator", y="outcome"
    ... )
    >>> result.reject
    False

    Notes
    -----
    Implements the confidence-set construction from Kwon and Roth
    (2024) [1]_. Under the sharp null, the observable joint
    distributions P(Y=y, M=m | D=d) must be consistent with a
    feasible set of conditional distributions. The LP checks whether
    the bootstrap-perturbed moments remain in this feasible region.

    The SciPy HiGHS solver replaces the R implementation's Gurobi
    dependency while producing numerically equivalent results within
    floating-point tolerance.

    References
    ----------
    .. [1] Kwon, S. and Roth, J. (2024). "Testing Mechanisms."
       arXiv:2404.11739v3.

    See Also
    --------
    test_sharp_null : Main sharp-null test with multiple method options.
    ci_TV : Confidence interval for total-variation distance.
    """

    d = _validate_scalar_column_name(d, name="d")
    m = _validate_scalar_column_name(m, name="m")
    y = _validate_scalar_column_name(y, name="y")
    alpha = _validate_probability_share(alpha, name="alpha", strict=True)
    bootstrap_replications = _validate_positive_integer(
        B,
        name="B",
        minimum=1,
        context="test_sharp_null_cr bootstrap confidence set",
    )
    eps_bar = _validate_positive_real(eps_bar, name="eps_bar")
    num_Ybins = _validate_optional_positive_integer(num_Ybins, name="num_Ybins")
    random_state = _validate_optional_nonnegative_integer(random_state, name="random_state")

    working_df = _load_dataframe(data_path=data_path, df=df)
    cleaned_df = remove_missing_from_df(df=working_df, d=d, m=m, y=y)
    y_processed = cleaned_df[y]
    if num_Ybins is not None:
        y_processed = discretize_y(cleaned_df[y], num_bins=num_Ybins)

    treatment_support = normalize_binary_support(cleaned_df[d], column=d)
    mediator_levels = _ordered_support_values(cleaned_df[m])
    _validate_scalar_ordered_mediator_support(series=cleaned_df[m], levels=mediator_levels)
    y_levels = _ordered_support_values(pd.Series(y_processed, index=cleaned_df.index))
    if len(y_levels) < 2:
        raise ValueError("test_sharp_null_cr requires a discrete outcome with at least two observed support points.")

    analysis_df = cleaned_df.copy()
    mediator_level_map = {level: index for index, level in enumerate(mediator_levels)}
    analysis_df["_tm_d_processed"] = treatment_support.transform(analysis_df[d])
    analysis_df["_tm_m_processed"] = analysis_df[m].map(mediator_level_map).astype(int)
    analysis_df["_tm_y_processed"] = pd.Categorical(y_processed, categories=y_levels, ordered=True)

    mediator_values = list(range(len(mediator_levels)))
    mediator_ordering = _parse_scalar_mediator_ordering(
        ordering=ordering,
        mediator_levels=mediator_levels,
        parameter_name="ordering",
    )
    beta_observed = _compute_ordered_nonbinary_beta(
        dvec=analysis_df["_tm_d_processed"],
        mvec=analysis_df["_tm_m_processed"],
        yvec=analysis_df["_tm_y_processed"],
        mediator_values=mediator_values,
        y_values=y_levels,
    )
    matrices = _construct_ordered_nonbinary_cr_matrices(
        mediator_count=len(mediator_values),
        outcome_count=len(y_levels),
        allowed_theta_pairs=mediator_ordering,
    )
    rng = np.random.default_rng(random_state)
    cr_result = _cr_confidence_set(
        beta_observed=beta_observed,
        bootstrap_betas=_bootstrap_beta_draws(
            analysis_df=analysis_df,
            d="_tm_d_processed",
            cluster=None,
            bootstrap_replications=max(2, bootstrap_replications),
            random_state=random_state,
            statistic=lambda frame: _compute_ordered_nonbinary_beta(
                dvec=frame["_tm_d_processed"],
                mvec=frame["_tm_m_processed"],
                yvec=frame["_tm_y_processed"],
                mediator_values=mediator_values,
                y_values=y_levels,
            ),
        )[:bootstrap_replications],
        a_observed=matrices["a_observed"],
        a_shape=matrices["a_shape"],
        beta_shape=matrices["beta_shape"],
        target=matrices["target"],
        eps_bar=eps_bar,
        alpha=alpha,
        sample_size=int(analysis_df.shape[0]),
        rng=rng,
    )
    diagnostics = build_cell_count_diagnostics(
        df=analysis_df,
        d=d,
        m=m,
        y="_tm_y_processed",
        cluster=None,
        requested_num_y_bins=num_Ybins,
        applied_num_y_bins=len(y_levels),
        no_bite_reason="CR confidence-set runner reports the target interval directly",
        support_diagnostics={
            **treatment_support.diagnostics(
                original_key="original_treatment_levels",
                normalized_key="normalized_treatment_levels",
            ),
            "original_mediator_levels": [_normalize_mediator_level(level) for level in mediator_levels],
            "normalized_mediator_levels": mediator_values,
            "ordering": _ordering_diagnostics(mediator_levels, mediator_ordering),
            "mediator_columns": [m],
            "mediator_dimension": 1,
            "vector_mediator": False,
            "support_normalization": "scalar mediator support is mapped to consecutive integer levels in deterministic support order",
            "cr_reference": "packages/r/TestMechs/R/test_sharp_null_cr.R",
            "cox_shi_reference": "packages/r/TestMechs/R/cox_shi_nonuisance.R",
            "paper_reference": "manuscript/sources/arxiv-2404.11739v3/draft.tex:416-421,657-688",
            "cr_scope": "scalar mediator CR confidence set with explicit mediator ordering and SciPy LP backend",
            "gurobi_dependency_replaced": True,
        },
    )
    diagnostics["cr_confidence_set"] = cr_result
    return SharpNullResult(
        method="CR",
        null_hypothesis="Y(0,m) = Y(1,m) for all mediator support points",
        reject=bool(cr_result["reject"]),
        test_stat=float(cr_result["confidence_interval_lower"]),
        critical_value=0.0,
        p_value=float("nan"),
        beta_observed=beta_observed.tolist(),
        approximation=(
            "Cho-Russell style confidence-set inversion for the sharp-null "
            "target using SciPy linear programming instead of the R/Gurobi backend."
        ),
        diagnostics=diagnostics,
    )


def _test_ordered_nonbinary_fsst(
    *,
    analysis_df: pd.DataFrame,
    d: str,
    m: str,
    y: str,
    original_mediator_levels: tuple[object, ...],
    requested_num_y_bins: int | None,
    alpha: float,
    cluster: str | None,
    treatment_support: object,
    mediator_ordering: tuple[tuple[int, int], ...],
    mediator_columns: tuple[str, ...],
    support_normalization: str,
    fsst_scope: str,
    method: str,
    lambda_mode: str,
    bootstrap_replications: int,
    random_state: int | None,
    frac_ats_affected: float | None,
    max_defiers_share: float,
) -> SharpNullResult:
    mediator_values = list(range(len(original_mediator_levels)))
    y_values = _series_support_values(analysis_df[y])
    beta_observed = _compute_ordered_nonbinary_beta(
        dvec=analysis_df[d],
        mvec=analysis_df[m],
        yvec=analysis_df[y],
        mediator_values=mediator_values,
        y_values=y_values,
    )
    matrices = _construct_ordered_nonbinary_moment_matrices(
        mediator_count=len(mediator_values),
        outcome_count=len(y_values),
        allowed_theta_pairs=mediator_ordering,
        frac_ats_affected=frac_ats_affected,
        max_defiers_share=max_defiers_share,
    )
    bootstrap_betas = _bootstrap_beta_draws(
        analysis_df=analysis_df,
        d=d,
        cluster=cluster,
        bootstrap_replications=bootstrap_replications,
        random_state=random_state,
        statistic=lambda frame: _compute_ordered_nonbinary_beta(
            dvec=frame[d],
            mvec=frame[m],
            yvec=frame[y],
            mediator_values=mediator_values,
            y_values=y_values,
        ),
    )
    fsst_result = _fsst_nuisance_test(
        beta_observed=beta_observed,
        bootstrap_betas=bootstrap_betas,
        a_observed=matrices["a_observed"],
        a_shape=matrices["a_shape"],
        beta_shape=matrices["beta_shape"],
        alpha=alpha,
        lambda_mode=lambda_mode,
        effective_sample_size=_effective_bootstrap_sample_size(
            analysis_df=analysis_df,
            cluster=cluster,
        ),
    )
    a_matrix = np.vstack([matrices["a_observed"], matrices["a_shape"]])
    if max_defiers_share > 0.0 and "relaxed monotonicity defier cap" not in fsst_scope:
        fsst_scope = f"{fsst_scope} and relaxed monotonicity defier cap"

    diagnostics = build_cell_count_diagnostics(
        df=analysis_df,
        d=d,
        m=m,
        y=y,
        cluster=cluster,
        requested_num_y_bins=requested_num_y_bins,
        applied_num_y_bins=len(y_values),
        no_bite_reason="FSST uses the paper's bootstrap moment-selection runner",
        support_diagnostics={
            **treatment_support.diagnostics(
                original_key="original_treatment_levels",
                normalized_key="normalized_treatment_levels",
            ),
            "original_mediator_levels": [
                _normalize_mediator_level(value) for value in original_mediator_levels
            ],
            "normalized_mediator_levels": mediator_values,
            "mediator_columns": list(mediator_columns),
            "mediator_dimension": len(mediator_columns),
            "vector_mediator": len(mediator_columns) > 1,
            "support_normalization": support_normalization,
            "fsst_scope": fsst_scope,
            "fsst_reference": (
                "manuscript/sources/arxiv-2404.11739v3/draft.tex:412-421; "
                "packages/r/TestMechs/R/test_sharp_null.R:269-303,728-977"
            ),
            "allowed_mediator_type_count": len(mediator_ordering),
            "forbidden_mediator_type_count": len(mediator_values) ** 2 - len(mediator_ordering),
            "allowed_mediator_type_pairs": [
                {"from": low, "to": high} for low, high in mediator_ordering
            ],
            "nuisance_parameter_count": int(a_matrix.shape[1]),
            "moment_inequality_count": int(a_matrix.shape[0]),
            "observed_moment_count": int(beta_observed.shape[0]),
            "shape_moment_count": int(matrices["beta_shape"].shape[0]),
        },
    )
    if max_defiers_share > 0.0:
        diagnostics["relaxed_monotonicity"] = _relaxed_monotonicity_diagnostics(
            mediator_levels=original_mediator_levels,
            monotone_pairs=mediator_ordering,
            max_defiers_share=max_defiers_share,
            matrices=matrices,
        )
    diagnostics["fsst"] = fsst_result["diagnostics"]
    return SharpNullResult(
        method=method,
        null_hypothesis="Y(0,m) = Y(1,m) for all mediator support points",
        reject=bool(fsst_result["reject"]),
        test_stat=float(fsst_result["test_stat"]),
        critical_value=float(fsst_result["critical_value"]),
        p_value=float(fsst_result["p_value"]),
        beta_observed=beta_observed.tolist(),
        approximation=(
            "FSST bootstrap moment-selection runner with ordered nonbinary "
            "mediator nuisance constraints."
        ),
        diagnostics=diagnostics,
    )


def _test_ordered_nonbinary_cs(
    *,
    analysis_df: pd.DataFrame,
    d: str,
    m: str,
    y: str,
    original_mediator_levels: tuple[object, ...],
    requested_num_y_bins: int | None,
    alpha: float,
    cluster: str | None,
    treatment_support: object,
    regression_diagnostics: dict[str, object] | None,
    mediator_ordering: tuple[tuple[int, int], ...],
    mediator_columns: tuple[str, ...],
    support_normalization: str,
    cs_scope: str,
    frac_ats_affected: float | None,
    max_defiers_share: float,
    beta_observed_override: np.ndarray | None = None,
    sigma_obs_override: np.ndarray | None = None,
    diagnostics_df: pd.DataFrame | None = None,
) -> SharpNullResult:
    if len(original_mediator_levels) < 2:
        raise ValueError("CS requires at least two mediator support levels.")

    mediator_values = list(range(len(original_mediator_levels)))
    y_values = _series_support_values(analysis_df[y])
    if beta_observed_override is None:
        beta_observed = _compute_ordered_nonbinary_beta(
            dvec=analysis_df[d],
            mvec=analysis_df[m],
            yvec=analysis_df[y],
            mediator_values=mediator_values,
            y_values=y_values,
        )
    else:
        beta_observed = np.asarray(beta_observed_override, dtype=float)
    matrices = _construct_ordered_nonbinary_moment_matrices(
        mediator_count=len(mediator_values),
        outcome_count=len(y_values),
        allowed_theta_pairs=mediator_ordering,
        frac_ats_affected=frac_ats_affected,
        max_defiers_share=max_defiers_share,
    )
    if beta_observed.shape[0] != matrices["a_observed"].shape[0]:
        raise ValueError("Adjusted ordered CS beta length must match the observed moment matrix.")
    if sigma_obs_override is None:
        sigma_obs = _analytic_variance_ordered_nonbinary(
            dvec=analysis_df[d],
            mvec=analysis_df[m],
            yvec=analysis_df[y],
            clustervec=analysis_df[cluster] if cluster is not None else None,
            mediator_values=mediator_values,
            y_values=y_values,
        )
    else:
        sigma_obs = np.asarray(sigma_obs_override, dtype=float)
    if sigma_obs.shape != (beta_observed.shape[0], beta_observed.shape[0]):
        raise ValueError("Adjusted ordered CS covariance must be square with one row per observed moment.")
    sigma = _augment_covariance_with_shape_constraints(
        sigma_obs,
        shape_rows=matrices["a_shape"].shape[0],
    )
    beta = np.concatenate([beta_observed, matrices["beta_shape"]])
    a_matrix = np.vstack([matrices["a_observed"], matrices["a_shape"]])
    cs_result = _cox_shi_nuisance(
        beta=beta,
        sigma=sigma,
        constraint_matrix=a_matrix,
        alpha=alpha,
    )
    if frac_ats_affected is not None and "relaxed always-taker affected-fraction null" not in cs_scope:
        cs_scope = f"{cs_scope} and relaxed always-taker affected-fraction null"
    if max_defiers_share > 0.0 and "relaxed monotonicity defier cap" not in cs_scope:
        cs_scope = f"{cs_scope} and relaxed monotonicity defier cap"

    diagnostics = build_cell_count_diagnostics(
        df=analysis_df if diagnostics_df is None else diagnostics_df,
        d=d,
        m=m,
        y=y,
        cluster=cluster,
        requested_num_y_bins=requested_num_y_bins,
        applied_num_y_bins=len(y_values),
        no_bite_reason=(
            "theta_kk feasibility is handled through ordered nonbinary CS "
            "nuisance constraints"
        ),
        support_diagnostics={
            **treatment_support.diagnostics(
                original_key="original_treatment_levels",
                normalized_key="normalized_treatment_levels",
            ),
            "original_mediator_levels": [
                _normalize_mediator_level(value) for value in original_mediator_levels
            ],
            "normalized_mediator_levels": mediator_values,
            "mediator_columns": list(mediator_columns),
            "mediator_dimension": len(mediator_columns),
            "vector_mediator": len(mediator_columns) > 1,
            "support_normalization": support_normalization,
            "cs_scope": cs_scope,
            "cs_reference": (
                "manuscript/sources/arxiv-2404.11739v3/draft.tex:164-275,606-610; "
                "packages/r/TestMechs/R/test_sharp_null.R:707-1269"
            ),
            "allowed_mediator_type_count": len(mediator_ordering),
            "forbidden_mediator_type_count": len(mediator_values) ** 2 - len(mediator_ordering),
            "allowed_mediator_type_pairs": [
                {"from": low, "to": high} for low, high in mediator_ordering
            ],
            "nuisance_parameter_count": int(a_matrix.shape[1]),
            "moment_inequality_count": int(a_matrix.shape[0]),
            "observed_moment_count": int(beta_observed.shape[0]),
            "shape_moment_count": int(matrices["beta_shape"].shape[0]),
            "cox_shi_degrees_of_freedom": int(cs_result["degrees_of_freedom"]),
            "cox_shi_solver": cs_result["solver"],
        },
    )
    if frac_ats_affected is not None:
        diagnostics["relaxed_null"] = {
            "frac_ats_affected": float(frac_ats_affected),
            "estimand": "pooled_fraction_of_always_takers_affected",
            "constraint": "sum_iota <= frac_ats_affected * sum_theta_kk",
            "iota_parameter_count": len(mediator_values),
            "constraint_rows": _relaxed_null_constraint_rows(
                mediator_levels=original_mediator_levels,
                frac_ats_affected=float(frac_ats_affected),
                matrix_rows=matrices["relaxed_null_constraint_rows"],
            ),
            "paper_reference": (
                "manuscript/sources/arxiv-2404.11739v3/draft.tex:164-275 and lower-bound "
                "fraction-of-always-takers logic in draft.tex:633-835"
            ),
            "reference_implementation": (
                "packages/r/TestMechs/R/test_sharp_null.R:755-832 "
                "frac_ATs_affected nuisance constraints"
            ),
        }
    if max_defiers_share > 0.0:
        diagnostics["relaxed_monotonicity"] = _relaxed_monotonicity_diagnostics(
            mediator_levels=original_mediator_levels,
            monotone_pairs=mediator_ordering,
            max_defiers_share=max_defiers_share,
            matrices=matrices,
        )
    if regression_diagnostics is not None:
        diagnostics["regression"] = regression_diagnostics

    return SharpNullResult(
        method="CS",
        null_hypothesis="Y(0,m) = Y(1,m) for all mediator support points",
        reject=bool(cs_result["reject"]),
        test_stat=float(cs_result["test_stat"]),
        critical_value=float(cs_result["critical_value"]),
        p_value=float(cs_result["p_value"]),
        beta_observed=beta.tolist(),
        approximation=(
            "Discretized Y with ordered nonbinary mediator nuisance constraints; "
            "valid but potentially non-sharp when Y is discretized."
        ),
        diagnostics=diagnostics,
    )


def _test_binary_kitagawa(
    *,
    analysis_df: pd.DataFrame,
    d: str,
    m: str,
    y: str,
    diagnostics_d: str,
    diagnostics_m: str,
    original_mediator_levels: tuple[object, ...],
    requested_num_y_bins: int | None,
    alpha: float,
    cluster: str | None,
    treatment_support: object,
    mediator_columns: tuple[str, ...],
    bootstrap_replications: int,
    random_state: int | None,
    xi: float,
) -> SharpNullResult:
    bootstrap_replications = _validate_positive_integer(
        bootstrap_replications,
        name="bootstrap_replications",
        minimum=1,
        context="method='K'",
    )
    xi = _validate_positive_real(xi, name="kitagawa_xi")

    observed_stat, observed_components = _kitagawa_binary_statistic(
        analysis_df=analysis_df,
        d=d,
        m=m,
        y=y,
        xi=xi,
    )
    rng = np.random.default_rng(random_state)
    bootstrap_stats = np.asarray(
        [
            _kitagawa_binary_statistic(
                analysis_df=_kitagawa_bootstrap_draw(
                    analysis_df=analysis_df,
                    d=d,
                    cluster=cluster,
                    rng=rng,
                ),
                d=d,
                m=m,
                y=y,
                xi=xi,
            )[0]
            for _ in range(bootstrap_replications)
        ],
        dtype=float,
    )
    critical_value = float(np.quantile(bootstrap_stats, 1.0 - alpha))
    p_value = float((1 + np.sum(bootstrap_stats >= observed_stat - 1e-12)) / (bootstrap_replications + 1))

    diagnostics = build_cell_count_diagnostics(
        df=analysis_df,
        d=diagnostics_d,
        m=diagnostics_m,
        y=y,
        cluster=cluster,
        requested_num_y_bins=requested_num_y_bins,
        applied_num_y_bins=None,
        no_bite_reason="K uses the original outcome and does not identify theta_kk_min.",
        support_diagnostics={
            **treatment_support.diagnostics(
                original_key="original_treatment_levels",
                normalized_key="normalized_treatment_levels",
            ),
            "original_mediator_levels": [_normalize_scalar(value) for value in original_mediator_levels],
            "normalized_mediator_levels": [0, 1],
            "mediator_columns": list(mediator_columns),
            "mediator_dimension": 1,
            "vector_mediator": False,
            "support_normalization": "binary support is mapped to internal {0, 1} in deterministic support order",
            "outcome_contract": "original_outcome_no_discretization",
            "num_y_bins_policy": "ignored_by_k_original_outcome_contract",
        },
    )
    diagnostics["kitagawa"] = {
        "paper_reference": "manuscript/sources/arxiv-2404.11739v3/draft.tex:412-438",
        "r_reference": "packages/r/TestMechs/R/test_sharp_null_toru.R:12-36,54-88,178-316",
        "r_negative_evidence": (
            "Python does not inherit the R toru wrapper's ignored B/num_Ybins "
            "or literal {0,1} mediator support behavior."
        ),
        "bootstrap_replications": int(bootstrap_replications),
        "bootstrap_unit": "cluster" if cluster is not None else "individual",
        "bootstrap_resampling": "stratified_nonparametric_by_treatment_arm",
        "xi": float(xi),
        "test_stat": float(observed_stat),
        "critical_value": critical_value,
        "p_value": p_value,
        "bootstrap_min": float(np.min(bootstrap_stats)),
        "bootstrap_max": float(np.max(bootstrap_stats)),
        "bootstrap_mean": float(np.mean(bootstrap_stats)),
        "component_stats": observed_components,
    }
    return SharpNullResult(
        method="K",
        null_hypothesis="Y(0,m) = Y(1,m) for both binary mediator support points",
        reject=bool(observed_stat > critical_value),
        test_stat=float(observed_stat),
        critical_value=critical_value,
        p_value=p_value,
        beta_observed=[float(observed_stat)],
        approximation=(
            "Kitagawa binary-mediator comparator using the original outcome and "
            "a treatment-arm-stratified nonparametric bootstrap."
        ),
        diagnostics=diagnostics,
    )


def _bootstrap_beta_draws(
    *,
    analysis_df: pd.DataFrame,
    d: str,
    cluster: str | None,
    bootstrap_replications: int,
    random_state: int | None,
    statistic: Callable[[pd.DataFrame], np.ndarray],
) -> np.ndarray:
    bootstrap_replications = _validate_positive_integer(
        bootstrap_replications,
        name="bootstrap_replications",
        minimum=2,
        context="FSST runners",
    )
    rng = np.random.default_rng(random_state)
    draws = [
        np.asarray(
            statistic(
                _kitagawa_bootstrap_draw(
                    analysis_df=analysis_df,
                    d=d,
                    cluster=cluster,
                    rng=rng,
                )
            ),
            dtype=float,
        )
        for _ in range(bootstrap_replications)
    ]
    return np.vstack(draws)


def _effective_bootstrap_sample_size(
    *,
    analysis_df: pd.DataFrame,
    cluster: str | None,
) -> int:
    if cluster is None:
        return int(analysis_df.shape[0])
    return int(analysis_df[cluster].nunique(dropna=True))


def _fsst_nonuisance_test(
    *,
    beta: np.ndarray,
    bootstrap_betas: np.ndarray,
    alpha: float,
    lambda_mode: str,
    effective_sample_size: int,
) -> dict[str, object]:
    beta = np.asarray(beta, dtype=float)
    bootstrap_betas = np.asarray(bootstrap_betas, dtype=float)
    scale = _fsst_bootstrap_scale(bootstrap_betas)
    standardized_beta = beta / scale
    lambda_value = _fsst_lambda_value(
        lambda_mode=lambda_mode,
        moment_count=int(beta.size),
        effective_sample_size=effective_sample_size,
        bootstrap_errors=(bootstrap_betas - beta) / scale,
    )
    selected = standardized_beta <= lambda_value
    if not bool(np.any(selected)):
        selected[int(np.argmin(standardized_beta))] = True

    observed_stat = float(max(0.0, np.max(-standardized_beta)))
    centered_errors = (bootstrap_betas - beta) / scale
    bootstrap_stats = np.maximum(
        0.0,
        np.max(-centered_errors[:, selected], axis=1),
    )
    critical_value = float(np.quantile(bootstrap_stats, 1.0 - alpha))
    p_value = float((1 + np.sum(bootstrap_stats >= observed_stat - 1e-12)) / (len(bootstrap_stats) + 1))
    return {
        "reject": bool(observed_stat > critical_value),
        "test_stat": observed_stat,
        "critical_value": critical_value,
        "p_value": p_value,
        "diagnostics": {
            "variant": "fsst_bootstrap_moment_selection_nonuisance",
            "paper_reference": "manuscript/sources/arxiv-2404.11739v3/draft.tex:412-421",
            "reference_implementation": (
                "packages/r/TestMechs/R/test_sharp_null_binary_m.R:193-224; "
                "lpinfer::fsst via lpmodel with A.obs = I"
            ),
            "lambda_mode": lambda_mode,
            "lambda_value": lambda_value,
            "bootstrap_replications": int(bootstrap_betas.shape[0]),
            "bootstrap_resampling": "treatment-arm-stratified nonparametric bootstrap, clustered when cluster is supplied",
            "effective_sample_size": int(effective_sample_size),
            "moment_count": int(beta.size),
            "selected_moment_count": int(selected.sum()),
            "selected_moment_indices": np.flatnonzero(selected).astype(int).tolist(),
            "scale_min": float(np.min(scale)),
            "scale_max": float(np.max(scale)),
            "test_stat": observed_stat,
            "critical_value": critical_value,
            "p_value": p_value,
        },
    }


def _fsst_nuisance_test(
    *,
    beta_observed: np.ndarray,
    bootstrap_betas: np.ndarray,
    a_observed: np.ndarray,
    a_shape: np.ndarray,
    beta_shape: np.ndarray,
    alpha: float,
    lambda_mode: str,
    effective_sample_size: int,
    weight_matrix: str = "diag",
) -> dict[str, object]:
    beta_observed = np.asarray(beta_observed, dtype=float)
    bootstrap_betas = np.asarray(bootstrap_betas, dtype=float)
    a_full = np.vstack([a_observed, a_shape])
    beta_full = np.concatenate([beta_observed, beta_shape])
    observed_scale = _fsst_weight_scale(
        bootstrap_betas=bootstrap_betas,
        weight_matrix=weight_matrix,
    )
    shape_scale = np.zeros(a_shape.shape[0], dtype=float)
    scale_full = np.concatenate([observed_scale, shape_scale])
    observed_stat, nuisance_solution = _fsst_nuisance_sup_violation(
        beta=beta_full,
        constraint_matrix=a_full,
        scale=scale_full,
    )
    observed_slack = a_observed @ nuisance_solution - beta_observed
    lambda_value = _fsst_lambda_value(
        lambda_mode=lambda_mode,
        moment_count=int(beta_observed.size),
        effective_sample_size=effective_sample_size,
        bootstrap_errors=(bootstrap_betas - beta_observed) / observed_scale,
    )
    selected_observed = observed_slack / observed_scale <= lambda_value
    if not bool(np.any(selected_observed)):
        selected_observed[int(np.argmin(observed_slack / observed_scale))] = True

    selected_a = np.vstack([a_observed[selected_observed], a_shape])
    selected_scale = np.concatenate([observed_scale[selected_observed], shape_scale])
    bootstrap_stats = []
    for bootstrap_beta in bootstrap_betas:
        centered = bootstrap_beta - beta_observed
        selected_beta = np.concatenate([centered[selected_observed], beta_shape])
        stat, _ = _fsst_nuisance_sup_violation(
            beta=selected_beta,
            constraint_matrix=selected_a,
            scale=selected_scale,
        )
        bootstrap_stats.append(stat)
    bootstrap_stats_array = np.asarray(bootstrap_stats, dtype=float)
    critical_value = float(np.quantile(bootstrap_stats_array, 1.0 - alpha))
    p_value = float(
        (1 + np.sum(bootstrap_stats_array >= observed_stat - 1e-12))
        / (len(bootstrap_stats_array) + 1)
    )
    return {
        "reject": bool(observed_stat > critical_value),
        "test_stat": float(observed_stat),
        "critical_value": critical_value,
        "p_value": p_value,
        "diagnostics": {
            "variant": "fsst_bootstrap_moment_selection_with_nuisance",
            "paper_reference": "manuscript/sources/arxiv-2404.11739v3/draft.tex:372-376,412-421",
            "reference_implementation": (
                "packages/r/TestMechs/R/test_sharp_null.R:269-303 and "
                "construct_Aobs_Ashp_betashp nuisance formulation"
            ),
            "lambda_mode": lambda_mode,
            "weight_matrix": weight_matrix,
            "lambda_value": lambda_value,
            "bootstrap_replications": int(bootstrap_betas.shape[0]),
            "effective_sample_size": int(effective_sample_size),
            "observed_moment_count": int(beta_observed.size),
            "shape_moment_count": int(beta_shape.size),
            "selected_observed_moment_count": int(selected_observed.sum()),
            "selected_observed_moment_indices": np.flatnonzero(selected_observed).astype(int).tolist(),
            "min_observed_slack": float(np.min(observed_slack)),
            "max_observed_slack": float(np.max(observed_slack)),
            "scale_min": float(np.min(observed_scale)),
            "scale_max": float(np.max(observed_scale)),
            "test_stat": float(observed_stat),
            "critical_value": critical_value,
            "p_value": p_value,
        },
    }


def _construct_ordered_nonbinary_tv_moment_matrices(
    *,
    mediator_count: int,
    outcome_count: int,
    at_index: int,
    allowed_theta_pairs: tuple[tuple[int, int], ...] | None = None,
) -> dict[str, np.ndarray]:
    theta_types = list(
        allowed_theta_pairs
        if allowed_theta_pairs is not None
        else _total_order_mediator_ordering(mediator_count)
    )
    theta_index = {theta_type: index for index, theta_type in enumerate(theta_types)}
    delta_offset = len(theta_types)
    eta_offset = delta_offset + mediator_count * outcome_count
    variable_count = eta_offset + mediator_count

    def delta_index(mediator: int, outcome: int) -> int:
        return delta_offset + mediator * outcome_count + outcome

    def eta_index(mediator: int) -> int:
        return eta_offset + mediator

    match_control = np.zeros((mediator_count, variable_count), dtype=float)
    match_treated = np.zeros((mediator_count, variable_count), dtype=float)
    for mediator in range(mediator_count):
        for high in range(mediator_count):
            column = theta_index.get((mediator, high))
            if column is not None:
                match_control[mediator, column] = 1.0
        for low in range(mediator_count):
            column = theta_index.get((low, mediator))
            if column is not None:
                match_treated[mediator, column] = 1.0

    partial_delta = np.zeros((mediator_count * outcome_count, variable_count), dtype=float)
    for mediator in range(mediator_count):
        for outcome in range(outcome_count):
            partial_delta[mediator * outcome_count + outcome, delta_index(mediator, outcome)] = 1.0

    a_observed = np.vstack(
        [
            match_control,
            match_treated,
            -match_control,
            -match_treated,
            partial_delta,
        ]
    )

    tv_shape = np.zeros((mediator_count, variable_count), dtype=float)
    for mediator in range(mediator_count):
        for low in range(mediator_count):
            if low != mediator:
                column = theta_index.get((low, mediator))
                if column is not None:
                    tv_shape[mediator, column] = 1.0
        for outcome in range(outcome_count):
            tv_shape[mediator, delta_index(mediator, outcome)] = -1.0
        tv_shape[mediator, eta_index(mediator)] = 1.0

    target = np.zeros(variable_count, dtype=float)
    target[eta_index(at_index)] = 1.0
    theta_diag = theta_index.get((at_index, at_index))
    if theta_diag is None:
        raise RuntimeError("TV target requires the at_group diagonal theta parameter.")
    target[theta_diag] = -1.0

    return {
        "a_observed": a_observed,
        "base_shape": np.vstack([tv_shape, np.eye(variable_count, dtype=float)]),
        "target": target,
        "beta_shape": np.zeros(tv_shape.shape[0] + variable_count + 2, dtype=float),
    }


def _tv_a_shape_for_null(
    *,
    tv_matrices: dict[str, np.ndarray],
    null_value: float,
) -> np.ndarray:
    target = np.asarray(tv_matrices["target"], dtype=float).copy()
    target = target.copy()
    theta_columns = np.flatnonzero(target < 0.0)
    if theta_columns.size != 1:
        raise RuntimeError("TV target must contain exactly one theta diagonal column.")
    target[theta_columns[0]] = -float(null_value)
    return np.vstack([tv_matrices["base_shape"], target, -target])


def _construct_ordered_nonbinary_cr_matrices(
    *,
    mediator_count: int,
    outcome_count: int,
    allowed_theta_pairs: tuple[tuple[int, int], ...] | None = None,
) -> dict[str, np.ndarray]:
    tv_matrices = _construct_ordered_nonbinary_tv_moment_matrices(
        mediator_count=mediator_count,
        outcome_count=outcome_count,
        at_index=0,
        allowed_theta_pairs=allowed_theta_pairs,
    )
    target = np.zeros(tv_matrices["a_observed"].shape[1], dtype=float)
    eta_offset = target.size - mediator_count
    target[eta_offset:] = 1.0
    return {
        "a_observed": tv_matrices["a_observed"],
        "a_shape": tv_matrices["base_shape"],
        "beta_shape": np.zeros(tv_matrices["base_shape"].shape[0], dtype=float),
        "target": target,
    }


def _cr_confidence_set(
    *,
    beta_observed: np.ndarray,
    bootstrap_betas: np.ndarray,
    a_observed: np.ndarray,
    a_shape: np.ndarray,
    beta_shape: np.ndarray,
    target: np.ndarray,
    eps_bar: float,
    alpha: float,
    sample_size: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    a_full = np.vstack([a_shape, a_observed])
    rhs = np.concatenate([beta_shape, beta_observed])
    variable_count = int(target.size)
    lower_bounds = np.zeros(variable_count, dtype=float)
    upper_bounds = np.ones(variable_count, dtype=float)

    xi_obj = rng.uniform(0.0, eps_bar, size=variable_count)
    xi_rhs = rng.uniform(0.0, eps_bar, size=rhs.size)
    xi_lb = rng.uniform(0.0, eps_bar, size=variable_count)
    xi_ub = rng.uniform(0.0, eps_bar, size=variable_count)
    perturbed_bounds = [
        (float(lb), float(ub))
        for lb, ub in zip(lower_bounds - xi_lb, upper_bounds + xi_ub, strict=True)
    ]
    perturbed_rhs = rhs - xi_rhs

    lbminus = _solve_ge_lp(
        objective=target - xi_obj,
        a_matrix=a_full,
        rhs=perturbed_rhs,
        bounds=perturbed_bounds,
        maximize=False,
    )
    ubminus = _solve_ge_lp(
        objective=target - xi_obj,
        a_matrix=a_full,
        rhs=perturbed_rhs,
        bounds=perturbed_bounds,
        maximize=True,
    )
    lbplus = _solve_ge_lp(
        objective=target + xi_obj,
        a_matrix=a_full,
        rhs=perturbed_rhs,
        bounds=perturbed_bounds,
        maximize=False,
    )
    ubplus = _solve_ge_lp(
        objective=target + xi_obj,
        a_matrix=a_full,
        rhs=perturbed_rhs,
        bounds=perturbed_bounds,
        maximize=True,
    )

    boot_lbminus: list[float] = []
    boot_lbplus: list[float] = []
    boot_ubminus: list[float] = []
    boot_ubplus: list[float] = []
    for bootstrap_beta in np.asarray(bootstrap_betas, dtype=float):
        bootstrap_rhs = np.concatenate([beta_shape, bootstrap_beta]) - xi_rhs
        boot_lbminus.append(
            _solve_ge_lp(
                objective=target - xi_obj,
                a_matrix=a_full,
                rhs=bootstrap_rhs,
                bounds=perturbed_bounds,
                maximize=False,
            )
        )
        boot_ubminus.append(
            _solve_ge_lp(
                objective=target - xi_obj,
                a_matrix=a_full,
                rhs=bootstrap_rhs,
                bounds=perturbed_bounds,
                maximize=True,
            )
        )
        boot_lbplus.append(
            _solve_ge_lp(
                objective=target + xi_obj,
                a_matrix=a_full,
                rhs=bootstrap_rhs,
                bounds=perturbed_bounds,
                maximize=False,
            )
        )
        boot_ubplus.append(
            _solve_ge_lp(
                objective=target + xi_obj,
                a_matrix=a_full,
                rhs=bootstrap_rhs,
                bounds=perturbed_bounds,
                maximize=True,
            )
        )

    n_sqrt = math.sqrt(float(sample_size))
    bn = 1.0 / math.sqrt(math.log(max(float(sample_size), math.e)))
    delta = max(ubplus, ubminus) - min(lbminus, lbplus)
    dn = 1.0 if delta > bn else 0.0
    kappa = (1.0 - alpha) * dn + (1.0 - alpha / 2.0) * (1.0 - dn)
    lbminus_q = n_sqrt * (np.asarray(boot_lbminus) - lbminus)
    lbplus_q = n_sqrt * (np.asarray(boot_lbplus) - lbplus)
    ubminus_q = -n_sqrt * (np.asarray(boot_ubminus) - ubminus)
    ubplus_q = -n_sqrt * (np.asarray(boot_ubplus) - ubplus)
    lower = min(lbminus, lbplus) - max(
        float(np.quantile(lbminus_q, kappa)),
        float(np.quantile(lbplus_q, kappa)),
    ) / n_sqrt
    upper = max(ubminus, ubplus) + max(
        float(np.quantile(ubminus_q, kappa)),
        float(np.quantile(ubplus_q, kappa)),
    ) / n_sqrt
    return {
        "confidence_interval_lower": float(lower),
        "confidence_interval_upper": float(upper),
        "reject": bool(0.0 < lower),
        "alpha": float(alpha),
        "bootstrap_replications": int(len(bootstrap_betas)),
        "eps_bar": float(eps_bar),
        "kappa": float(kappa),
        "delta": float(delta),
        "bn": float(bn),
        "sample_size": int(sample_size),
        "initial_lower_minus": float(lbminus),
        "initial_lower_plus": float(lbplus),
        "initial_upper_minus": float(ubminus),
        "initial_upper_plus": float(ubplus),
        "lp_backend": "scipy.optimize.linprog(highs)",
    }


def _solve_ge_lp(
    *,
    objective: np.ndarray,
    a_matrix: np.ndarray,
    rhs: np.ndarray,
    bounds: list[tuple[float, float]],
    maximize: bool,
) -> float:
    objective = np.asarray(objective, dtype=float)
    c = -objective if maximize else objective
    result = linprog(
        c=c,
        A_ub=-np.asarray(a_matrix, dtype=float),
        b_ub=-np.asarray(rhs, dtype=float),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"CR confidence-set linear program failed: {result.message}")
    value = float(objective @ result.x)
    return value


def _fsst_nuisance_sup_violation(
    *,
    beta: np.ndarray,
    constraint_matrix: np.ndarray,
    scale: np.ndarray,
) -> tuple[float, np.ndarray]:
    beta = np.asarray(beta, dtype=float)
    constraint_matrix = np.asarray(constraint_matrix, dtype=float)
    scale = np.asarray(scale, dtype=float)
    if constraint_matrix.shape[0] != beta.size:
        raise ValueError("constraint_matrix must have one row per beta moment.")
    if scale.shape[0] != beta.size:
        raise ValueError("scale must have one value per beta moment.")

    nuisance_dim = constraint_matrix.shape[1]
    objective = np.zeros(nuisance_dim + 1, dtype=float)
    objective[-1] = 1.0
    a_ub = np.column_stack([-constraint_matrix, -np.maximum(scale, 1e-12)])
    result = linprog(
        c=objective,
        A_ub=a_ub,
        b_ub=-beta,
        bounds=[(0.0, 1.0)] * nuisance_dim + [(0.0, None)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"FSST nuisance violation LP failed: {result.message}")
    return float(result.x[-1]), np.asarray(result.x[:nuisance_dim], dtype=float)


def _fsst_bootstrap_scale(bootstrap_betas: np.ndarray) -> np.ndarray:
    if bootstrap_betas.ndim != 2 or bootstrap_betas.shape[0] < 2:
        raise ValueError("FSST bootstrap beta draws must be a two-dimensional array with at least two draws.")
    scale = np.std(bootstrap_betas, axis=0, ddof=1)
    positive = scale[scale > 1e-12]
    fallback = float(np.median(positive)) if positive.size else 1.0
    return np.where(scale > 1e-12, scale, fallback)


def _fsst_weight_scale(
    *,
    bootstrap_betas: np.ndarray,
    weight_matrix: str,
) -> np.ndarray:
    if weight_matrix == "diag":
        return _fsst_bootstrap_scale(bootstrap_betas)
    if weight_matrix == "identity":
        if bootstrap_betas.ndim != 2 or bootstrap_betas.shape[1] == 0:
            raise ValueError("FSST bootstrap beta draws must be a non-empty two-dimensional array.")
        return np.ones(bootstrap_betas.shape[1], dtype=float)
    raise NotImplementedError("Unsupported FSST weight matrix.")


def _fsst_lambda_value(
    *,
    lambda_mode: str,
    moment_count: int,
    effective_sample_size: int,
    bootstrap_errors: np.ndarray,
) -> float:
    if lambda_mode not in {"dd", "ndd"}:
        raise ValueError("lambda_mode must be either 'dd' or 'ndd'.")
    p = max(float(moment_count), float(np.e))
    n = max(float(effective_sample_size), float(np.e))
    ndd = 1.0 / np.sqrt(np.log(p) * np.log(max(float(np.e), np.log(max(float(np.e), n)))))
    if lambda_mode == "ndd":
        return float(ndd)

    if bootstrap_errors.size == 0:
        return float(ndd)
    max_abs_errors = np.max(np.abs(bootstrap_errors), axis=1)
    data_driven = float(np.quantile(max_abs_errors, 0.25) / np.sqrt(max(n, 1.0)))
    return float(min(ndd, max(data_driven, ndd / 4.0)))


def _kitagawa_bootstrap_draw(
    *,
    analysis_df: pd.DataFrame,
    d: str,
    cluster: str | None,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if cluster is None:
        pieces = []
        for treatment_value in (0, 1):
            arm = analysis_df.loc[analysis_df[d] == treatment_value]
            if arm.empty:
                raise ValueError("Both treatment arms must be non-empty for K bootstrap.")
            positions = rng.integers(0, len(arm), size=len(arm))
            pieces.append(arm.iloc[positions].copy())
        return pd.concat(pieces, ignore_index=True)

    pieces = []
    for treatment_value in (0, 1):
        arm = analysis_df.loc[analysis_df[d] == treatment_value]
        clusters = list(dict.fromkeys(arm[cluster].dropna().tolist()))
        if not clusters:
            raise ValueError("Both treatment arms must contain non-empty clusters for K bootstrap.")
        sampled_clusters = [clusters[int(index)] for index in rng.integers(0, len(clusters), size=len(clusters))]
        for bootstrap_cluster, source_cluster in enumerate(sampled_clusters):
            cluster_rows = arm.loc[arm[cluster] == source_cluster].copy()
            cluster_rows[cluster] = f"{treatment_value}:{bootstrap_cluster}"
            pieces.append(cluster_rows)
    return pd.concat(pieces, ignore_index=True)


def _kitagawa_binary_statistic(
    *,
    analysis_df: pd.DataFrame,
    d: str,
    m: str,
    y: str,
    xi: float,
) -> tuple[float, dict[str, float]]:
    zvec = analysis_df[d]
    mvec = analysis_df[m]
    yvec = analysis_df[y]
    n0 = int((zvec == 0).sum())
    n1 = int((zvec == 1).sum())
    if n0 == 0 or n1 == 0:
        raise ValueError("Both treatment arms must be non-empty for the K statistic.")

    lambda_weight = n1 / (n0 + n1)
    partial = _kitagawa_partial_cdfs(zvec=zvec, mvec=mvec, yvec=yvec)
    scale = float(np.sqrt((n0 * n1) / (n0 + n1)))
    mediator_one = _kitagawa_component_stat(
        low_cdf=partial[(1, 0)],
        high_cdf=partial[(1, 1)],
        xi=xi,
        lambda_weight=lambda_weight,
        direction="low_minus_high",
    )
    mediator_zero = _kitagawa_component_stat(
        low_cdf=partial[(0, 0)],
        high_cdf=partial[(0, 1)],
        xi=xi,
        lambda_weight=lambda_weight,
        direction="high_minus_low",
    )
    component_stats = {
        "mediator_1_treated_arm_violation": float(scale * mediator_one),
        "mediator_0_control_arm_violation": float(scale * mediator_zero),
    }
    return max(0.0, *component_stats.values()), component_stats


def _kitagawa_partial_cdfs(
    *,
    zvec: pd.Series,
    mvec: pd.Series,
    yvec: pd.Series,
) -> dict[tuple[int, int], np.ndarray]:
    output: dict[tuple[int, int], np.ndarray] = {}
    for mediator_value in (0, 1):
        grid = np.asarray(
            sorted(yvec.loc[mvec == mediator_value].astype(float).unique().tolist()),
            dtype=float,
        )
        for treatment_value in (0, 1):
            arm_size = int((zvec == treatment_value).sum())
            if arm_size == 0:
                raise ValueError("Both treatment arms must be non-empty for K partial CDFs.")
            values = np.asarray(
                sorted(
                    yvec.loc[(zvec == treatment_value) & (mvec == mediator_value)]
                    .astype(float)
                    .tolist()
                ),
                dtype=float,
            )
            output[(mediator_value, treatment_value)] = (
                np.searchsorted(values, grid, side="right").astype(float) / arm_size
            )
    return output


def _kitagawa_component_stat(
    *,
    low_cdf: np.ndarray,
    high_cdf: np.ndarray,
    xi: float,
    lambda_weight: float,
    direction: str,
) -> float:
    if low_cdf.size == 0 or high_cdf.size == 0:
        return 0.0
    if low_cdf.shape != high_cdf.shape:
        raise ValueError("K partial CDF arrays must share a common outcome grid.")

    best = 0.0
    for start in range(low_cdf.size):
        low_start = 0.0 if start == 0 else float(low_cdf[start - 1])
        high_start = 0.0 if start == 0 else float(high_cdf[start - 1])
        for end in range(start, low_cdf.size):
            low_mass = float(low_cdf[end] - low_start)
            high_mass = float(high_cdf[end] - high_start)
            moment = low_mass - high_mass if direction == "low_minus_high" else high_mass - low_mass
            variance_weight = np.sqrt(
                lambda_weight * low_mass * (1.0 - low_mass)
                + (1.0 - lambda_weight) * high_mass * (1.0 - high_mass)
            )
            best = max(best, moment / max(float(xi), float(variance_weight)))
    return float(best)


def _load_dataframe(
    *,
    data_path: str | Path | None,
    df: pd.DataFrame | None,
) -> pd.DataFrame:
    if df is not None and data_path is not None:
        raise ValueError("Exactly one of df or data_path must be provided.")
    if df is not None:
        return df.copy()
    if data_path is None:
        raise ValueError("Exactly one of df or data_path must be provided.")
    return pd.read_csv(Path(data_path))


def _validate_probability_share(value: object, *, name: str, strict: bool = False) -> float:
    boundary = "between 0 and 1" if strict else "between 0 and 1 inclusive"
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a numeric probability {boundary}.")
    share = float(value)
    valid = 0.0 < share < 1.0 if strict else 0.0 <= share <= 1.0
    if not np.isfinite(share) or not valid:
        raise ValueError(f"{name} must be a numeric probability {boundary}.")
    return share


def _validate_positive_integer(
    value: object,
    *,
    name: str,
    minimum: int,
    context: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer at least {minimum} for {context}.")
    count = int(value)
    if count < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum} for {context}.")
    return count


def _validate_positive_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite positive numeric value.")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive numeric value.")
    return result


def _validate_boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean value.")
    return bool(value)


def _validate_fsst_weight_matrix(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("weight_matrix must be one of 'diag', 'identity', or 'avar'.")
    normalized = value.strip().lower()
    if normalized not in {"diag", "identity", "avar"}:
        raise ValueError("weight_matrix must be one of 'diag', 'identity', or 'avar'.")
    if normalized == "avar":
        raise NotImplementedError(
            "ci_TV weight_matrix='avar' is source-scoped out: the R all-configuration "
            "FSST check filters out weight.matrix='avar' and records a singular-matrix "
            "failure, so Python does not treat it as a parity target. Use 'diag' or "
            "'identity'."
        )
    return normalized


def _validate_optional_nonnegative_integer(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be None or a non-negative integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be None or a non-negative integer.")
    return result


def _validate_optional_positive_integer(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be None or a positive integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be None or a positive integer.")
    return result


def _validate_grid_step(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError("grid_step must be a finite numeric value in (0, 1].")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0 or result > 1.0:
        raise ValueError("grid_step must be a finite numeric value in (0, 1].")
    return result


def _tv_grid(step: float) -> np.ndarray:
    grid = np.arange(0.0, 1.0 + step / 2.0, step, dtype=float)
    grid = np.clip(grid, 0.0, 1.0)
    if not np.isclose(grid[-1], 1.0):
        grid = np.append(grid, 1.0)
    return np.unique(np.round(grid, 12))


def _parse_scalar_mediator_ordering(
    *,
    ordering: Mapping[object, Sequence[object]] | None,
    mediator_levels: tuple[object, ...],
    parameter_name: str,
) -> tuple[tuple[int, int], ...]:
    if ordering is None:
        return _total_order_mediator_ordering(len(mediator_levels))
    if not isinstance(ordering, Mapping):
        raise ValueError(f"{parameter_name} must be a mapping from each mediator level to its lower set.")

    level_index = {level: index for index, level in enumerate(mediator_levels)}
    string_index: dict[str, int] = {}
    for index, level in enumerate(mediator_levels):
        label = str(level)
        if label in string_index:
            string_index = {}
            break
        string_index[label] = index

    def resolve_level(value: object) -> int:
        if value in level_index:
            return level_index[value]
        if isinstance(value, str) and value in string_index:
            return string_index[value]
        raise ValueError(f"{parameter_name} contains mediator level {value!r}, which is not in the observed support.")

    pairs: list[tuple[int, int]] = []
    for high_level in mediator_levels:
        if high_level in ordering:
            raw_lower = ordering[high_level]
        elif str(high_level) in ordering:
            raw_lower = ordering[str(high_level)]
        else:
            raise ValueError(f"{parameter_name} must include every observed mediator support point.")
        if isinstance(raw_lower, (str, bytes)) or not isinstance(raw_lower, Sequence):
            raw_values = (raw_lower,)
        else:
            raw_values = tuple(raw_lower)
        high_index = level_index[high_level]
        lower_indices = tuple(dict.fromkeys(resolve_level(value) for value in raw_values))
        if high_index not in lower_indices:
            raise ValueError(f"{parameter_name} lower set for {high_level!r} must include the level itself.")
        pairs.extend((low_index, high_index) for low_index in lower_indices)
    return tuple(dict.fromkeys(pairs))


def _ordering_diagnostics(
    mediator_levels: tuple[object, ...],
    ordering: tuple[tuple[int, int], ...],
) -> dict[str, object]:
    normalized_levels = [_normalize_mediator_level(level) for level in mediator_levels]
    return {
        "allowed_pair_count": len(ordering),
        "allowed_pairs": [
            {
                "from_index": low,
                "to_index": high,
                "from_level": normalized_levels[low],
                "to_level": normalized_levels[high],
            }
            for low, high in ordering
        ],
    }


def _validate_cluster_column(df: pd.DataFrame, *, cluster: object | None, treatment: str) -> str | None:
    if cluster is None:
        return None
    if not isinstance(cluster, str) or cluster == "":
        raise ValueError("cluster must be None or the name of an existing cluster column.")
    if cluster not in df.columns:
        raise ValueError(f"cluster column {cluster!r} is not present in df.")
    if df[cluster].isna().any():
        raise ValueError("cluster labels must be non-missing on the sharp-null analysis sample.")
    treatment_counts = df.groupby(cluster, sort=False, dropna=False)[treatment].nunique(dropna=False)
    if bool((treatment_counts > 1).any()):
        raise ValueError("cluster must identify units with a fixed treatment level.")
    return cluster


def _validate_scalar_column_name(value: object, *, name: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{name} must name one scalar DataFrame column.")
    return value


def _mediator_columns(m: str | Sequence[str]) -> tuple[str, ...]:
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


def _ordered_support_values(series: pd.Series) -> tuple[object, ...]:
    if isinstance(series.dtype, pd.CategoricalDtype):
        observed_values = list(pd.unique(series.dropna()))
        return tuple(
            category
            for category in series.cat.categories
            if any(category == observed_value for observed_value in observed_values)
        )
    values = list(pd.unique(series.dropna()))
    return tuple(sorted(values, key=_support_sort_key))


def _ordered_vector_support_values(
    df: pd.DataFrame,
    columns: tuple[str, ...],
) -> tuple[tuple[object, ...], ...]:
    values = list(dict.fromkeys(_vector_mediator_keys(df, columns)))
    return tuple(
        sorted(
            values,
            key=lambda level: tuple(_support_sort_key(value) for value in level),
        )
    )


def _vector_mediator_keys(df: pd.DataFrame, columns: tuple[str, ...]) -> list[tuple[object, ...]]:
    return [
        tuple(_normalize_scalar(value) for value in row)
        for row in df.loc[:, list(columns)].itertuples(index=False, name=None)
    ]


def _total_order_mediator_ordering(mediator_count: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (low, high)
        for high in range(mediator_count)
        for low in range(mediator_count)
        if low <= high
    )


def _elementwise_mediator_ordering(
    mediator_levels: tuple[tuple[object, ...], ...],
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (low_index, high_index)
        for high_index, high_level in enumerate(mediator_levels)
        for low_index, low_level in enumerate(mediator_levels)
        if _elementwise_leq(low_level, high_level)
    )


def _all_mediator_type_pairs(mediator_count: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (low, high)
        for high in range(mediator_count)
        for low in range(mediator_count)
    )


def _elementwise_leq(low_level: tuple[object, ...], high_level: tuple[object, ...]) -> bool:
    if len(low_level) != len(high_level):
        raise ValueError("Vector mediator support points must have equal dimension.")
    comparisons: list[bool] = []
    for low_value, high_value in zip(low_level, high_level, strict=True):
        try:
            comparisons.append(bool(low_value <= high_value))
        except TypeError as exc:
            raise TypeError(
                "Vector mediator elementwise monotonicity requires comparable "
                f"component values; got {low_value!r} and {high_value!r}."
            ) from exc
    return all(comparisons)


def _validate_scalar_ordered_mediator_support(
    *,
    series: pd.Series,
    levels: tuple[object, ...],
) -> None:
    _reject_nonfinite_numeric_mediator_levels(levels, column=str(series.name))
    if len(levels) <= 2:
        return
    if isinstance(series.dtype, pd.CategoricalDtype) and series.dtype.ordered:
        return
    for value in levels:
        normalized = _normalize_scalar(value)
        if not isinstance(normalized, (bool, int, float)):
            raise ValueError(
                "Nonbinary scalar mediator support must be naturally comparable "
                "or an ordered pandas Categorical before using ordered-monotone "
                "sharp-null runners."
            )
    for index, left in enumerate(levels):
        for right in levels[index + 1 :]:
            try:
                left <= right
                right <= left
            except TypeError as exc:
                raise ValueError(
                    "Nonbinary scalar mediator support must be naturally comparable "
                    "or an ordered pandas Categorical before using ordered-monotone "
                    "sharp-null runners."
                ) from exc


def _reject_nonfinite_numeric_mediator_levels(
    levels: tuple[object, ...],
    *,
    column: str,
) -> None:
    for value in levels:
        normalized = _normalize_scalar(value)
        if isinstance(normalized, bool):
            continue
        if isinstance(normalized, (int, float)) and not np.isfinite(float(normalized)):
            raise ValueError(f"{column} must contain only finite numeric support levels.")


def _series_support_values(series: pd.Series) -> tuple[object, ...]:
    return _ordered_support_values(series)


def _support_sort_key(value: object) -> tuple[object, ...]:
    normalized = _normalize_scalar(value)
    if isinstance(normalized, bool):
        return ("bool", int(normalized))
    if isinstance(normalized, (int, float)):
        return ("number", float(normalized))
    return (type(normalized).__name__, repr(normalized))


def _normalize_mediator_level(value: object) -> object:
    if isinstance(value, tuple):
        return tuple(_normalize_scalar(item) for item in value)
    return _normalize_scalar(value)


def _normalize_scalar(value: object) -> object:
    if isinstance(value, pd.Interval):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (AttributeError, ValueError):
            return value
    return value


def _sharp_null_regression_supported_scope(regression_spec) -> str:
    if regression_spec.formula_kind == "trivial":
        return "trivial equivalence for CS analytic variance"
    if regression_spec.formula_kind in {"controls", "fixed_effects"}:
        return (
            "binary-mediator CS analytic variance using adjusted joint "
            "probability influence functions"
        )
    return "unsupported IV adjusted sharp-null inference"


def _validate_sharp_null_regression_scope(
    *,
    method: str,
    regression_spec,
    vector_mediator: bool,
    mediator_level_count: int,
) -> None:
    if regression_spec is None:
        return
    if method != "CS":
        raise NotImplementedError(
            "Adjusted sharp-null inference is currently release-scoped only for method='CS'."
        )
    if regression_spec.formula_kind in {"iv", "iv_fixed_effects"}:
        raise NotImplementedError(
            "IV adjusted sharp-null analytic variance is not yet release-scoped."
        )
    if regression_spec.formula_kind in {"controls", "fixed_effects"} and (
        vector_mediator or mediator_level_count != 2
    ):
        raise NotImplementedError(
            "Non-trivial adjusted CS analytic variance is currently release-scoped "
            "for binary mediators only."
        )


def _regression_formula_with_treatment(regression_spec, *, treatment: str) -> str:
    rhs_terms = [treatment, *regression_spec.controls]
    rhs = " + ".join(rhs_terms)
    if regression_spec.formula_kind == "fixed_effects":
        fixed_effects = " + ".join(regression_spec.fixed_effects)
        return f"~ {rhs} | {fixed_effects}"
    return f"~ {rhs}"


def _adjusted_binary_beta_and_variance(
    *,
    analysis_df: pd.DataFrame,
    d: str,
    m: str,
    y: str,
    clustervec: pd.Series | None,
    mediator_values: list[object],
    regression_spec,
) -> tuple[np.ndarray, np.ndarray, dict[str, object], pd.Index]:
    internal_formula = _regression_formula_with_treatment(regression_spec, treatment=d)
    adjusted = compute_adjusted_probability_influences(
        df=analysis_df,
        d=d,
        m=m,
        y=y,
        reg_formula=internal_formula,
    )
    probabilities = adjusted.probabilities
    low_mediator, high_mediator = mediator_values
    if low_mediator not in probabilities.m_values or high_mediator not in probabilities.m_values:
        raise ValueError("Adjusted reg_formula complete cases must retain both binary mediator levels.")

    beta: list[float] = []
    influence_columns: list[np.ndarray] = []
    for y_value in probabilities.y_values:
        key = (y_value, low_mediator)
        beta.append(probabilities.p_ym_d0[key] - probabilities.p_ym_d1[key])
        influence_columns.append(
            adjusted.p_ym_d0_influence[key] - adjusted.p_ym_d1_influence[key]
        )
    for y_value in probabilities.y_values:
        key = (y_value, high_mediator)
        beta.append(probabilities.p_ym_d1[key] - probabilities.p_ym_d0[key])
        influence_columns.append(
            adjusted.p_ym_d1_influence[key] - adjusted.p_ym_d0_influence[key]
        )

    influence_matrix = np.column_stack(influence_columns)
    cluster_used = clustervec.loc[adjusted.row_index] if clustervec is not None else None
    sigma = _covariance_from_influence_matrix(influence_matrix, clustervec=cluster_used)
    diagnostics = {
        **adjusted.diagnostics,
        "adjusted_probability_contract": (
            "joint adjusted P(Y=y,M=m|D=d) grid consumed by the binary "
            "sharp-null moment vector"
        ),
        "variance_contract": (
            "cluster-robust covariance of adjusted joint-probability treatment "
            "coefficient influence functions"
        ),
        "internal_reg_formula": internal_formula,
        "reference_anchor": (
            "packages/r/TestMechs/R/test_sharp_null_binary_m.R:87-183; "
            "packages/r/TestMechs/R/test_sharp_null.R:1004-1270"
        ),
    }
    return np.asarray(beta, dtype=float), sigma, diagnostics, adjusted.row_index


def _adjusted_ordered_binary_beta_and_variance(
    *,
    analysis_df: pd.DataFrame,
    d: str,
    m: str,
    y: str,
    clustervec: pd.Series | None,
    mediator_values: list[object],
    regression_spec,
) -> tuple[np.ndarray, np.ndarray, dict[str, object], pd.Index]:
    internal_formula = _regression_formula_with_treatment(regression_spec, treatment=d)
    adjusted = compute_adjusted_probability_influences(
        df=analysis_df,
        d=d,
        m=m,
        y=y,
        reg_formula=internal_formula,
    )
    probabilities = adjusted.probabilities
    missing_mediators = [mediator for mediator in mediator_values if mediator not in probabilities.m_values]
    if missing_mediators:
        raise ValueError("Adjusted reg_formula complete cases must retain both binary mediator levels.")

    beta: list[float] = []
    influence_columns: list[np.ndarray] = []

    for mediator in mediator_values:
        beta.append(probabilities.p_m_d0[mediator])
        influence_columns.append(
            sum(
                adjusted.p_ym_d0_influence[(y_value, mediator)]
                for y_value in probabilities.y_values
            )
        )
    for mediator in mediator_values:
        beta.append(probabilities.p_m_d1[mediator])
        influence_columns.append(
            sum(
                adjusted.p_ym_d1_influence[(y_value, mediator)]
                for y_value in probabilities.y_values
            )
        )
    for mediator in mediator_values:
        beta.append(-probabilities.p_m_d0[mediator])
        influence_columns.append(
            -sum(
                adjusted.p_ym_d0_influence[(y_value, mediator)]
                for y_value in probabilities.y_values
            )
        )
    for mediator in mediator_values:
        beta.append(-probabilities.p_m_d1[mediator])
        influence_columns.append(
            -sum(
                adjusted.p_ym_d1_influence[(y_value, mediator)]
                for y_value in probabilities.y_values
            )
        )

    for mediator in mediator_values:
        for y_value in probabilities.y_values:
            key = (y_value, mediator)
            beta.append(probabilities.p_ym_d1[key] - probabilities.p_ym_d0[key])
            influence_columns.append(
                adjusted.p_ym_d1_influence[key] - adjusted.p_ym_d0_influence[key]
            )

    influence_matrix = np.column_stack(influence_columns)
    cluster_used = clustervec.loc[adjusted.row_index] if clustervec is not None else None
    sigma = _covariance_from_influence_matrix(influence_matrix, clustervec=cluster_used)
    diagnostics = {
        **adjusted.diagnostics,
        "adjusted_probability_contract": (
            "joint adjusted mediator-mass and P(Y=y,M=m|D=d) grids consumed by "
            "the ordered CS nuisance moment vector"
        ),
        "variance_contract": (
            "cluster-robust covariance of adjusted mediator-mass and joint-probability "
            "treatment coefficient influence functions"
        ),
        "internal_reg_formula": internal_formula,
        "reference_anchor": (
            "packages/r/TestMechs/R/test_sharp_null.R:892-918 observed ordered-nuisance "
            "moments; packages/r/TestMechs/R/test_sharp_null.R:1004-1270 adjusted "
            "probability construction"
        ),
    }
    return np.asarray(beta, dtype=float), sigma, diagnostics, adjusted.row_index


def _covariance_from_influence_matrix(
    influence_matrix: np.ndarray,
    *,
    clustervec: pd.Series | None,
) -> np.ndarray:
    if clustervec is None:
        grouped = influence_matrix
    else:
        group_labels = pd.Series(clustervec.to_numpy(), index=range(len(clustervec)))
        grouped = pd.DataFrame(influence_matrix).groupby(group_labels, sort=False).sum().to_numpy()

    n = influence_matrix.shape[0]
    n_clusters = grouped.shape[0]
    covariance = np.cov(grouped * (n_clusters / n), rowvar=False, bias=False) / n_clusters
    return np.atleast_2d(covariance)


def _compute_binary_beta(
    *,
    dvec: pd.Series,
    mvec: pd.Series,
    yvec: pd.Series,
    mediator_values: list[object],
) -> np.ndarray:
    yvalues = _ordered_support_values(yvec)
    low_mediator, high_mediator = mediator_values
    beta = []
    for y_value in yvalues:
        beta.append(
            _conditional_probability(dvec, mvec, yvec, treatment=0, mediator=low_mediator, y_value=y_value)
            - _conditional_probability(
                dvec, mvec, yvec, treatment=1, mediator=low_mediator, y_value=y_value
            )
        )
    for y_value in yvalues:
        beta.append(
            _conditional_probability(dvec, mvec, yvec, treatment=1, mediator=high_mediator, y_value=y_value)
            - _conditional_probability(
                dvec, mvec, yvec, treatment=0, mediator=high_mediator, y_value=y_value
            )
        )
    return np.asarray(beta, dtype=float)


def _conditional_probability(
    dvec: pd.Series,
    mvec: pd.Series,
    yvec: pd.Series,
    *,
    treatment: int,
    mediator: object,
    y_value: object,
) -> float:
    treated = dvec == treatment
    denominator = treated.sum()
    if denominator == 0:
        raise ValueError("Treatment arm is empty.")
    numerator = ((yvec == y_value) & (mvec == mediator) & treated).sum()
    return float(numerator / denominator)


def _analytic_variance_binary(
    *,
    dvec: pd.Series,
    mvec: pd.Series,
    yvec: pd.Series,
    clustervec: pd.Series | None,
    mediator_values: list[object],
) -> np.ndarray:
    yvalues = _ordered_support_values(yvec)
    low_mediator, high_mediator = mediator_values
    n = len(yvec)
    n0 = int((dvec == 0).sum())
    n1 = int((dvec == 1).sum())

    if n0 == 0 or n1 == 0:
        raise ValueError("Both treatment arms must contain observations.")

    influence_columns = []
    for mediator in mediator_values:
        for y_value in yvalues:
            indicator_0 = ((yvec == y_value) & (mvec == mediator) & (dvec == 0)).astype(float)
            indicator_1 = ((yvec == y_value) & (mvec == mediator) & (dvec == 1)).astype(float)

            centered_0 = (dvec == 0).astype(float) * (indicator_0 - indicator_0.mean() / (n0 / n)) / (n0 / n)
            centered_1 = (dvec == 1).astype(float) * (indicator_1 - indicator_1.mean() / (n1 / n)) / (n1 / n)
            influence_columns.append(centered_0 - centered_1 if mediator == low_mediator else centered_1 - centered_0)

    influence_matrix = np.column_stack(influence_columns)
    if clustervec is None:
        grouped = influence_matrix
    else:
        group_labels = pd.Series(clustervec.to_numpy(), index=range(len(clustervec)))
        grouped = pd.DataFrame(influence_matrix).groupby(group_labels, sort=False).sum().to_numpy()

    n_clusters = grouped.shape[0]
    covariance = np.cov(grouped * (n_clusters / n), rowvar=False, bias=False) / n_clusters
    return np.atleast_2d(covariance)


def _compute_ordered_nonbinary_beta(
    *,
    dvec: pd.Series,
    mvec: pd.Series,
    yvec: pd.Series,
    mediator_values: list[int],
    y_values: tuple[object, ...],
) -> np.ndarray:
    p_m_d0 = [
        _conditional_mediator_probability(dvec, mvec, treatment=0, mediator=mediator)
        for mediator in mediator_values
    ]
    p_m_d1 = [
        _conditional_mediator_probability(dvec, mvec, treatment=1, mediator=mediator)
        for mediator in mediator_values
    ]
    partial_mass_diffs = []
    for mediator in mediator_values:
        for y_value in y_values:
            partial_mass_diffs.append(
                _conditional_probability(
                    dvec,
                    mvec,
                    yvec,
                    treatment=1,
                    mediator=mediator,
                    y_value=y_value,
                )
                - _conditional_probability(
                    dvec,
                    mvec,
                    yvec,
                    treatment=0,
                    mediator=mediator,
                    y_value=y_value,
                )
            )
    return np.asarray(
        [*p_m_d0, *p_m_d1, *[-value for value in p_m_d0], *[-value for value in p_m_d1], *partial_mass_diffs],
        dtype=float,
    )


def _conditional_mediator_probability(
    dvec: pd.Series,
    mvec: pd.Series,
    *,
    treatment: int,
    mediator: object,
) -> float:
    treated = dvec == treatment
    denominator = treated.sum()
    if denominator == 0:
        raise ValueError("Treatment arm is empty.")
    return float(((mvec == mediator) & treated).sum() / denominator)


def _relaxed_null_constraint_rows(
    *,
    mediator_levels: tuple[object, ...],
    frac_ats_affected: float,
    matrix_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    normalized_levels = [_normalize_mediator_level(level) for level in mediator_levels]
    rows: list[dict[str, object]] = []
    for index, level in enumerate(normalized_levels):
        matrix_row = matrix_rows[index]
        rows.append(
            {
                "row_type": "iota_upper_bound",
                "constraint": "iota_k <= theta_kk",
                "mediator_index": index,
                "mediator_level": level,
                "mediator_indices": None,
                "mediator_levels": None,
                "theta_pair": {"from": level, "to": level},
                "theta_diagonal_pair_count": None,
                "theta_kk_coefficient": 1.0,
                "iota_coefficient": -1.0,
                "rhs": 0.0,
                "shape_row_index": matrix_row["shape_row_index"],
                "moment_inequality_row_index": matrix_row["moment_inequality_row_index"],
                "sharp_shape_row_index": matrix_row["sharp_shape_row_index"],
                "theta_column_index": matrix_row["theta_column_index"],
                "iota_column_index": matrix_row["iota_column_index"],
                "theta_column_indices": None,
                "iota_column_indices": None,
            }
        )
    pooled_matrix_row = matrix_rows[-1]
    rows.append(
        {
            "row_type": "pooled_upper_bound",
            "constraint": "sum_iota <= frac_ats_affected * sum_theta_kk",
            "mediator_index": None,
            "mediator_level": None,
            "mediator_indices": list(range(len(normalized_levels))),
            "mediator_levels": normalized_levels,
            "theta_pair": None,
            "theta_diagonal_pair_count": len(normalized_levels),
            "theta_kk_coefficient": float(frac_ats_affected),
            "iota_coefficient": -1.0,
            "rhs": 0.0,
            "shape_row_index": pooled_matrix_row["shape_row_index"],
            "moment_inequality_row_index": pooled_matrix_row[
                "moment_inequality_row_index"
            ],
            "sharp_shape_row_index": None,
            "theta_column_index": None,
            "iota_column_index": None,
            "theta_column_indices": pooled_matrix_row["theta_column_indices"],
            "iota_column_indices": pooled_matrix_row["iota_column_indices"],
        }
    )
    return rows


def _relaxed_monotonicity_diagnostics(
    *,
    mediator_levels: tuple[object, ...],
    monotone_pairs: tuple[tuple[int, int], ...],
    max_defiers_share: float,
    matrices: dict[str, object],
) -> dict[str, object]:
    normalized_levels = [_normalize_mediator_level(level) for level in mediator_levels]
    defier_pairs = list(matrices["defier_theta_pairs"])
    theta_pairs = list(matrices["theta_type_pairs"])
    constraint_rows = []
    for matrix_row in matrices["defier_cap_constraint_rows"]:
        constraint_rows.append(
            {
                "row_type": matrix_row["row_type"],
                "constraint": "sum_defier_theta <= max_defiers_share",
                "requested_max_defiers_share": float(max_defiers_share),
                "rhs": float(max_defiers_share),
                "shape_rhs": matrix_row["rhs"],
                "shape_row_index": matrix_row["shape_row_index"],
                "moment_inequality_row_index": matrix_row["moment_inequality_row_index"],
                "theta_column_indices": matrix_row["theta_column_indices"],
                "defier_theta_pairs": _theta_pair_diagnostic_rows(
                    pairs=matrix_row["defier_theta_pairs"],
                    normalized_levels=normalized_levels,
                ),
            }
        )
    return {
        "requested_max_defiers_share": float(max_defiers_share),
        "constraint": "sum_defier_theta <= max_defiers_share",
        "theta_parameter_count": len(theta_pairs),
        "monotone_theta_parameter_count": len(monotone_pairs),
        "defier_theta_parameter_count": len(defier_pairs),
        "defier_theta_pairs": _theta_pair_diagnostic_rows(
            pairs=defier_pairs,
            normalized_levels=normalized_levels,
        ),
        "constraint_rows": constraint_rows,
        "paper_reference": (
            "manuscript/sources/arxiv-2404.11739v3/draft.tex:360-376 "
            "linear moment-inequality representation"
        ),
        "reference_implementation": (
            "packages/r/TestMechs/R/test_sharp_null.R:787-889 "
            "max_defiers_share nuisance constraints"
        ),
    }


def _theta_pair_diagnostic_rows(
    *,
    pairs: Sequence[tuple[int, int]],
    normalized_levels: Sequence[object],
) -> list[dict[str, object]]:
    return [
        {
            "from_index": low,
            "to_index": high,
            "from_level": normalized_levels[low],
            "to_level": normalized_levels[high],
        }
        for low, high in pairs
    ]


def _construct_ordered_nonbinary_moment_matrices(
    *,
    mediator_count: int,
    outcome_count: int,
    allowed_theta_pairs: tuple[tuple[int, int], ...] | None = None,
    frac_ats_affected: float | None = None,
    max_defiers_share: float = 0.0,
) -> dict[str, np.ndarray]:
    monotone_pairs = list(
        allowed_theta_pairs
        if allowed_theta_pairs is not None
        else _total_order_mediator_ordering(mediator_count)
    )
    monotone_pair_set = set(monotone_pairs)
    has_defier_cap = max_defiers_share > 0.0
    report_relaxed_null = frac_ats_affected is not None
    has_relaxed_null = report_relaxed_null and float(frac_ats_affected) > 0.0
    theta_types = list(
        _all_mediator_type_pairs(mediator_count)
        if has_defier_cap
        else monotone_pairs
    )
    defier_pairs = [pair for pair in theta_types if pair not in monotone_pair_set]
    theta_index = {theta_type: index for index, theta_type in enumerate(theta_types)}
    delta_offset = len(theta_types)
    iota_offset = delta_offset + mediator_count * outcome_count
    variable_count = iota_offset + (mediator_count if has_relaxed_null else 0)

    def delta_index(mediator: int, outcome: int) -> int:
        return delta_offset + mediator * outcome_count + outcome

    def iota_index(mediator: int) -> int:
        if not has_relaxed_null:
            raise RuntimeError("iota_index is only defined for relaxed-null matrices.")
        return iota_offset + mediator

    match_control = np.zeros((mediator_count, variable_count), dtype=float)
    match_treated = np.zeros((mediator_count, variable_count), dtype=float)
    for mediator in range(mediator_count):
        for high in range(mediator_count):
            if (mediator, high) in theta_index:
                match_control[mediator, theta_index[(mediator, high)]] = 1.0
        for low in range(mediator_count):
            if (low, mediator) in theta_index:
                match_treated[mediator, theta_index[(low, mediator)]] = 1.0

    partial_delta = np.zeros((mediator_count * outcome_count, variable_count), dtype=float)
    for mediator in range(mediator_count):
        for outcome in range(outcome_count):
            row = mediator * outcome_count + outcome
            partial_delta[row, delta_index(mediator, outcome)] = 1.0

    a_observed = np.vstack(
        [
            match_control,
            match_treated,
            -match_control,
            -match_treated,
            partial_delta,
        ]
    )

    sharp_shape = np.zeros((mediator_count, variable_count), dtype=float)
    for mediator in range(mediator_count):
        for low in range(mediator_count):
            if low != mediator and (low, mediator) in theta_index:
                sharp_shape[mediator, theta_index[(low, mediator)]] = 1.0
        for outcome in range(outcome_count):
            sharp_shape[mediator, delta_index(mediator, outcome)] = -1.0
        if has_relaxed_null:
            sharp_shape[mediator, iota_index(mediator)] = 1.0

    shape_blocks = [sharp_shape]
    shape_rhs_blocks = [np.zeros(sharp_shape.shape[0], dtype=float)]
    defier_cap_constraint_rows: list[dict[str, object]] = []
    if has_defier_cap:
        defier_cap = np.zeros((1, variable_count), dtype=float)
        defier_columns = [theta_index[pair] for pair in defier_pairs]
        for column in defier_columns:
            defier_cap[0, column] = -1.0
        shape_row_index = mediator_count
        defier_cap_constraint_rows.append(
            {
                "row_type": "defier_cap",
                "shape_row_index": shape_row_index,
                "moment_inequality_row_index": int(a_observed.shape[0] + shape_row_index),
                "theta_column_indices": defier_columns,
                "defier_theta_pairs": defier_pairs,
                "rhs": -float(max_defiers_share),
            }
        )
        shape_blocks.append(defier_cap)
        shape_rhs_blocks.append(np.asarray([-float(max_defiers_share)], dtype=float))

    if report_relaxed_null:
        iota_leq_theta = np.zeros((mediator_count, variable_count), dtype=float)
        relaxed_null_constraint_rows: list[dict[str, object]] = []
        observed_row_count = int(a_observed.shape[0])
        relaxed_start_row = mediator_count + (1 if has_defier_cap else 0)
        for mediator in range(mediator_count):
            theta_diag = theta_index.get((mediator, mediator))
            if has_relaxed_null and theta_diag is not None:
                iota_leq_theta[mediator, theta_diag] = 1.0
            if has_relaxed_null:
                iota_leq_theta[mediator, iota_index(mediator)] = -1.0
            shape_row_index = relaxed_start_row + mediator if has_relaxed_null else None
            relaxed_null_constraint_rows.append(
                {
                    "row_type": "iota_upper_bound",
                    "mediator_index": mediator,
                    "shape_row_index": shape_row_index,
                    "moment_inequality_row_index": (
                        observed_row_count + shape_row_index
                        if shape_row_index is not None
                        else None
                    ),
                    "sharp_shape_row_index": mediator,
                    "theta_column_index": theta_diag,
                    "iota_column_index": (
                        iota_index(mediator) if has_relaxed_null else None
                    ),
                }
            )

        pooled_relaxation = np.zeros((1, variable_count), dtype=float)
        pooled_theta_columns: list[int] = []
        pooled_iota_columns: list[int] = []
        for mediator in range(mediator_count):
            theta_diag = theta_index.get((mediator, mediator))
            if has_relaxed_null and theta_diag is not None:
                pooled_relaxation[0, theta_diag] = float(frac_ats_affected)
                pooled_theta_columns.append(theta_diag)
            if has_relaxed_null:
                pooled_relaxation[0, iota_index(mediator)] = -1.0
                pooled_iota_columns.append(iota_index(mediator))
        pooled_shape_row_index = (
            relaxed_start_row + mediator_count if has_relaxed_null else None
        )
        relaxed_null_constraint_rows.append(
            {
                "row_type": "pooled_upper_bound",
                "shape_row_index": pooled_shape_row_index,
                "moment_inequality_row_index": (
                    observed_row_count + pooled_shape_row_index
                    if pooled_shape_row_index is not None
                    else None
                ),
                "theta_column_indices": pooled_theta_columns,
                "iota_column_indices": pooled_iota_columns,
            }
        )
        if has_relaxed_null:
            shape_blocks.extend([iota_leq_theta, pooled_relaxation])
            shape_rhs_blocks.extend(
                [
                    np.zeros(iota_leq_theta.shape[0], dtype=float),
                    np.zeros(pooled_relaxation.shape[0], dtype=float),
                ]
            )
    else:
        relaxed_null_constraint_rows = []

    nonnegative_shape = np.eye(variable_count, dtype=float)
    a_shape = np.vstack([*shape_blocks, nonnegative_shape])
    beta_shape = np.concatenate(
        [*shape_rhs_blocks, np.zeros(nonnegative_shape.shape[0], dtype=float)]
    )
    return {
        "a_observed": a_observed,
        "a_shape": a_shape,
        "beta_shape": beta_shape,
        "relaxed_null_constraint_rows": relaxed_null_constraint_rows,
        "defier_cap_constraint_rows": defier_cap_constraint_rows,
        "theta_type_pairs": theta_types,
        "defier_theta_pairs": defier_pairs,
    }


def _analytic_variance_ordered_nonbinary(
    *,
    dvec: pd.Series,
    mvec: pd.Series,
    yvec: pd.Series,
    clustervec: pd.Series | None,
    mediator_values: list[int],
    y_values: tuple[object, ...],
) -> np.ndarray:
    n = len(yvec)
    n0 = int((dvec == 0).sum())
    n1 = int((dvec == 1).sum())
    if n0 == 0 or n1 == 0:
        raise ValueError("Both treatment arms must contain observations.")

    p0 = n0 / n
    p1 = n1 / n
    columns: list[np.ndarray] = []

    p_m0_columns: list[np.ndarray] = []
    p_m1_columns: list[np.ndarray] = []
    for mediator in mediator_values:
        prob0 = float(((dvec == 0) & (mvec == mediator)).sum() / n0)
        prob1 = float(((dvec == 1) & (mvec == mediator)).sum() / n1)
        p_m0_columns.append(((dvec == 0) * ((mvec == mediator).astype(float) - prob0) / p0).to_numpy())
        p_m1_columns.append(((dvec == 1) * ((mvec == mediator).astype(float) - prob1) / p1).to_numpy())

    columns.extend(p_m0_columns)
    columns.extend(p_m1_columns)
    columns.extend([-column for column in p_m0_columns])
    columns.extend([-column for column in p_m1_columns])

    for mediator in mediator_values:
        for y_value in y_values:
            prob0 = float(((dvec == 0) & (mvec == mediator) & (yvec == y_value)).sum() / n0)
            prob1 = float(((dvec == 1) & (mvec == mediator) & (yvec == y_value)).sum() / n1)
            if0 = ((dvec == 0) * (((mvec == mediator) & (yvec == y_value)).astype(float) - prob0) / p0).to_numpy()
            if1 = ((dvec == 1) * (((mvec == mediator) & (yvec == y_value)).astype(float) - prob1) / p1).to_numpy()
            columns.append(if1 - if0)

    influence_matrix = np.column_stack(columns)
    if clustervec is None:
        grouped = influence_matrix
    else:
        group_labels = pd.Series(clustervec.to_numpy(), index=range(len(clustervec)))
        grouped = pd.DataFrame(influence_matrix).groupby(group_labels, sort=False).sum().to_numpy()

    n_clusters = grouped.shape[0]
    covariance = np.cov(grouped * (n_clusters / n), rowvar=False, bias=False) / n_clusters
    return np.atleast_2d(covariance)


def _augment_covariance_with_shape_constraints(
    sigma_obs: np.ndarray,
    *,
    shape_rows: int,
) -> np.ndarray:
    return np.block(
        [
            [sigma_obs, np.zeros((sigma_obs.shape[0], shape_rows), dtype=float)],
            [np.zeros((shape_rows, sigma_obs.shape[1]), dtype=float), np.zeros((shape_rows, shape_rows), dtype=float)],
        ]
    )


def _cox_shi_nuisance(
    *,
    beta: np.ndarray,
    sigma: np.ndarray,
    constraint_matrix: np.ndarray,
    alpha: float,
    tol: float = 1e-8,
) -> dict[str, float | bool | int | str]:
    sigma = _positive_semidefinite_covariance(sigma, tol=tol)
    eigenvalues, eigenvectors = np.linalg.eigh(sigma)
    positive = eigenvalues > tol
    if not bool(np.any(positive)):
        return {
            "reject": False,
            "test_stat": 0.0,
            "critical_value": 0.0,
            "p_value": 1.0,
            "degrees_of_freedom": 0,
            "solver": "cox-shi-nuisance-no-positive-variance",
        }

    b_matrix = eigenvectors[:, positive]
    beta_red = b_matrix.T @ beta
    sigma_inv = np.diag(1.0 / eigenvalues[positive])
    d_vector = b_matrix @ beta_red - beta
    if not _nuisance_projection_support_is_feasible(
        beta_red=beta_red,
        b_matrix=b_matrix,
        constraint_matrix=constraint_matrix,
        d_vector=d_vector,
    ):
        return {
            "reject": True,
            "test_stat": float("inf"),
            "critical_value": 0.0,
            "p_value": 0.0,
            "degrees_of_freedom": 0,
            "solver": "scipy-linprog-infeasible-reduced-support",
        }
    beta_red_star, nuisance_star, test_stat, solver_status = _solve_nuisance_projection_qp(
        beta_red=beta_red,
        sigma_inv=sigma_inv,
        b_matrix=b_matrix,
        constraint_matrix=constraint_matrix,
        d_vector=d_vector,
    )
    dof = _cox_shi_nuisance_degrees_of_freedom(
        b_matrix=b_matrix,
        constraint_matrix=constraint_matrix,
        d_vector=d_vector,
        beta_red_star=beta_red_star,
        nuisance_star=nuisance_star,
    )
    if dof <= 0:
        critical_value = 0.0
        p_value = 0.0 if test_stat > 1e-10 else 1.0
        return {
            "reject": bool(test_stat > critical_value + 1e-10),
            "test_stat": float(test_stat),
            "critical_value": critical_value,
            "p_value": p_value,
            "degrees_of_freedom": 0,
            "solver": solver_status,
        }

    critical_value = float(chi2.ppf(1 - alpha, df=dof))
    p_value = float(chi2.sf(test_stat, df=dof))
    return {
        "reject": bool(test_stat > critical_value),
        "test_stat": float(test_stat),
        "critical_value": critical_value,
        "p_value": p_value,
        "degrees_of_freedom": int(dof),
        "solver": solver_status,
    }


def _solve_nuisance_projection_qp(
    *,
    beta_red: np.ndarray,
    sigma_inv: np.ndarray,
    b_matrix: np.ndarray,
    constraint_matrix: np.ndarray,
    d_vector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    reduced_dim = len(beta_red)
    nuisance_dim = constraint_matrix.shape[1]
    beta = b_matrix @ beta_red - d_vector

    # If the observed reduced point is already feasible, the Cox-Shi objective
    # is exactly zero and we can skip the nonlinear solve entirely.
    feasible_reduced = linprog(
        c=np.zeros(nuisance_dim, dtype=float),
        A_ub=-constraint_matrix,
        b_ub=-beta,
        bounds=[(0.0, 1.0)] * nuisance_dim,
        method="highs",
    )
    if feasible_reduced.success:
        nuisance_star = np.asarray(feasible_reduced.x, dtype=float)
        return beta_red, nuisance_star, 0.0, "scipy-linprog-feasible-reduced"

    osqp_result = _solve_nuisance_projection_osqp(
        beta_red=beta_red,
        sigma_inv=sigma_inv,
        b_matrix=b_matrix,
        constraint_matrix=constraint_matrix,
        d_vector=d_vector,
    )
    if osqp_result is not None:
        return osqp_result

    start = _nuisance_projection_feasible_start(
        beta_red=beta_red,
        b_matrix=b_matrix,
        constraint_matrix=constraint_matrix,
        d_vector=d_vector,
    )
    if start is None:
        raise RuntimeError("Cox-Shi nuisance projection has no feasible point in the reduced covariance support.")
    constraint_jacobian = np.vstack(
        [
            np.column_stack([-b_matrix, constraint_matrix]),
            np.column_stack([np.zeros((nuisance_dim, reduced_dim)), np.eye(nuisance_dim)]),
            np.column_stack([np.zeros((nuisance_dim, reduced_dim)), -np.eye(nuisance_dim)]),
        ]
    )

    def objective(values: np.ndarray) -> float:
        diff = values[:reduced_dim] - beta_red
        return float(diff.T @ sigma_inv @ diff)

    def gradient(values: np.ndarray) -> np.ndarray:
        diff = values[:reduced_dim] - beta_red
        return np.concatenate([2.0 * sigma_inv @ diff, np.zeros(nuisance_dim, dtype=float)])

    def constraints(values: np.ndarray) -> np.ndarray:
        reduced = values[:reduced_dim]
        nuisance = values[reduced_dim:]
        return np.concatenate(
            [
                d_vector - b_matrix @ reduced + constraint_matrix @ nuisance,
                nuisance,
                1.0 - nuisance,
            ]
        )

    result = minimize(
        objective,
        start,
        jac=gradient,
        constraints=({"type": "ineq", "fun": constraints, "jac": lambda _: constraint_jacobian},),
        method="SLSQP",
        options={"ftol": 1e-8, "maxiter": 2000, "disp": False},
    )
    def trust_region_numerically_converged(candidate: object) -> bool:
        return (
            math.isfinite(float(getattr(candidate, "fun", float("nan"))))
            and float(getattr(candidate, "constr_violation", float("inf"))) <= 1e-8
            and float(getattr(candidate, "optimality", float("inf"))) <= 2e-2
        )

    if not result.success:
        result = _solve_nuisance_projection_trust_region(
            start=start,
            beta_red=beta_red,
            sigma_inv=sigma_inv,
            b_matrix=b_matrix,
            constraint_matrix=constraint_matrix,
            d_vector=d_vector,
            maxiter=50,
        )
        if not result.success and not trust_region_numerically_converged(result):
            result = _solve_nuisance_projection_trust_region(
                start=np.asarray(result.x, dtype=float),
                beta_red=beta_red,
                sigma_inv=sigma_inv,
                b_matrix=b_matrix,
                constraint_matrix=constraint_matrix,
                d_vector=d_vector,
                maxiter=500,
            )
    if not result.success:
        if trust_region_numerically_converged(result):
            beta_red_star = np.asarray(result.x[:reduced_dim], dtype=float)
            nuisance_star = np.asarray(result.x[reduced_dim:], dtype=float)
            test_stat = objective(result.x)
            solver = f"scipy-trust-constr-numerically-converged:{getattr(result, 'message', 'maxiter')}"
            return beta_red_star, nuisance_star, test_stat, solver
        raise RuntimeError(f"Cox-Shi nuisance projection QP failed: {result.message}")

    beta_red_star = np.asarray(result.x[:reduced_dim], dtype=float)
    nuisance_star = np.asarray(result.x[reduced_dim:], dtype=float)
    test_stat = objective(result.x)
    solver = "scipy-trust-constr" if getattr(result, "method", None) == "trust-constr" else "scipy-slsqp"
    return beta_red_star, nuisance_star, test_stat, f"{solver}:{result.message}"


def _solve_nuisance_projection_osqp(
    *,
    beta_red: np.ndarray,
    sigma_inv: np.ndarray,
    b_matrix: np.ndarray,
    constraint_matrix: np.ndarray,
    d_vector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, str] | None:
    reduced_dim = len(beta_red)
    nuisance_dim = constraint_matrix.shape[1]
    dim = reduced_dim + nuisance_dim
    p_matrix = sparse.csc_matrix(
        np.block(
            [
                [2.0 * sigma_inv, np.zeros((reduced_dim, nuisance_dim), dtype=float)],
                [np.zeros((nuisance_dim, reduced_dim), dtype=float), np.zeros((nuisance_dim, nuisance_dim), dtype=float)],
            ]
        )
    )
    q_vector = np.concatenate(
        [-2.0 * sigma_inv @ beta_red, np.zeros(nuisance_dim, dtype=float)]
    )
    constraint_operator = sparse.vstack(
        [
            sparse.csc_matrix(np.column_stack([b_matrix, -constraint_matrix])),
            sparse.hstack(
                [
                    sparse.csc_matrix((nuisance_dim, reduced_dim)),
                    sparse.eye(nuisance_dim, format="csc"),
                ],
                format="csc",
            ),
        ],
        format="csc",
    )
    lower = np.concatenate(
        [np.full(constraint_matrix.shape[0], -np.inf), np.zeros(nuisance_dim)]
    )
    upper = np.concatenate([d_vector, np.ones(nuisance_dim)])
    solver = osqp.OSQP()
    solver.setup(
        P=p_matrix,
        q=q_vector,
        A=constraint_operator,
        l=lower,
        u=upper,
        verbose=False,
        eps_abs=1e-8,
        eps_rel=1e-8,
    )
    result = solver.solve(raise_error=False)
    status = str(result.info.status).lower()
    if status not in {"solved", "solved inaccurate"} or result.x is None:
        return None
    solution = np.asarray(result.x[:dim], dtype=float)
    beta_red_star_raw = solution[:reduced_dim]
    nuisance_star = solution[reduced_dim:]
    diff = beta_red - beta_red_star_raw
    test_stat = float(diff.T @ sigma_inv @ diff)
    beta_red_star = beta_red_star_raw.copy()
    beta_red_star[np.abs(beta_red_star) < 1e-8] = 0.0
    return beta_red_star, nuisance_star, test_stat, f"osqp:{result.info.status}"


def _solve_nuisance_projection_trust_region(
    *,
    start: np.ndarray,
    beta_red: np.ndarray,
    sigma_inv: np.ndarray,
    b_matrix: np.ndarray,
    constraint_matrix: np.ndarray,
    d_vector: np.ndarray,
    maxiter: int,
) -> Any:
    reduced_dim = len(beta_red)
    nuisance_dim = constraint_matrix.shape[1]
    linear_constraint = LinearConstraint(
        np.column_stack([b_matrix, -constraint_matrix]),
        -np.inf * np.ones(constraint_matrix.shape[0], dtype=float),
        d_vector,
    )
    bounds = Bounds(
        np.concatenate([np.full(reduced_dim, -np.inf), np.zeros(nuisance_dim)]),
        np.concatenate([np.full(reduced_dim, np.inf), np.ones(nuisance_dim)]),
    )

    def objective(values: np.ndarray) -> float:
        diff = values[:reduced_dim] - beta_red
        return float(diff.T @ sigma_inv @ diff)

    def gradient(values: np.ndarray) -> np.ndarray:
        diff = values[:reduced_dim] - beta_red
        return np.concatenate([2.0 * sigma_inv @ diff, np.zeros(nuisance_dim, dtype=float)])

    def hessian(_: np.ndarray) -> np.ndarray:
        matrix = np.zeros((reduced_dim + nuisance_dim, reduced_dim + nuisance_dim), dtype=float)
        matrix[:reduced_dim, :reduced_dim] = 2.0 * sigma_inv
        return matrix

    result = minimize(
        objective,
        start,
        jac=gradient,
        hess=hessian,
        constraints=(linear_constraint,),
        bounds=bounds,
        method="trust-constr",
        options={
            "gtol": 1e-6,
            "xtol": 1e-6,
            "barrier_tol": 1e-7,
            "maxiter": maxiter,
            "verbose": 0,
        },
    )
    result.method = "trust-constr"
    return result


def _nuisance_projection_support_is_feasible(
    *,
    beta_red: np.ndarray,
    b_matrix: np.ndarray,
    constraint_matrix: np.ndarray,
    d_vector: np.ndarray,
) -> bool:
    return _nuisance_projection_feasible_start(
        beta_red=beta_red,
        b_matrix=b_matrix,
        constraint_matrix=constraint_matrix,
        d_vector=d_vector,
    ) is not None


def _nuisance_projection_feasible_start(
    *,
    beta_red: np.ndarray,
    b_matrix: np.ndarray,
    constraint_matrix: np.ndarray,
    d_vector: np.ndarray,
) -> np.ndarray | None:
    reduced_dim = len(beta_red)
    nuisance_dim = constraint_matrix.shape[1]
    a_ub = np.vstack(
        [
            np.column_stack([b_matrix, -constraint_matrix]),
            np.column_stack([np.zeros((nuisance_dim, reduced_dim)), -np.eye(nuisance_dim)]),
            np.column_stack([np.zeros((nuisance_dim, reduced_dim)), np.eye(nuisance_dim)]),
        ]
    )
    b_ub = np.concatenate([d_vector, np.zeros(nuisance_dim), np.ones(nuisance_dim)])
    result = linprog(
        c=np.zeros(reduced_dim + nuisance_dim, dtype=float),
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=[(None, None)] * reduced_dim + [(0.0, 1.0)] * nuisance_dim,
        method="highs",
    )
    if result.success:
        return np.asarray(result.x, dtype=float)
    return None


def _cox_shi_nuisance_degrees_of_freedom(
    *,
    b_matrix: np.ndarray,
    constraint_matrix: np.ndarray,
    d_vector: np.ndarray,
    beta_red_star: np.ndarray,
    nuisance_star: np.ndarray,
    tol: float = 1e-6,
) -> int:
    del nuisance_star
    d_ineq = constraint_matrix.shape[0]
    d_nuis = constraint_matrix.shape[1]
    mstar = d_vector - b_matrix @ beta_red_star
    a_eq = np.vstack([constraint_matrix.T, np.ones((1, d_ineq), dtype=float)])
    b_eq = np.concatenate([np.zeros(d_nuis, dtype=float), np.array([1.0])])
    try:
        result = linprog(
            c=mstar,
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=(0.0, None),
            method="highs",
        )
        if (not result.success) or float(result.fun) >= 5e-5:
            return 0
        v_min = float(result.fun)
        psis = np.empty(d_ineq, dtype=float)
        a_eq_psi = np.vstack([constraint_matrix.T, mstar.reshape(1, -1), np.ones((1, d_ineq), dtype=float)])
        b_eq_psi = np.concatenate([np.zeros(d_nuis, dtype=float), np.array([v_min, 1.0])])
        for index in range(d_ineq):
            objective = np.zeros(d_ineq, dtype=float)
            objective[index] = -1.0
            psi_result = linprog(
                c=objective,
                A_eq=a_eq_psi,
                b_eq=b_eq_psi,
                bounds=(0.0, None),
                method="highs",
            )
            psis[index] = float(psi_result.fun) if psi_result.success else -1.0

        implicit_equalities = np.eye(d_ineq, dtype=float)[(-psis) < tol]
        a_full_eq = np.vstack(
            [
                implicit_equalities,
                -constraint_matrix.T,
                (b_matrix @ beta_red_star - d_vector).reshape(1, -1),
            ]
        )
        rank_a = int(np.linalg.matrix_rank(a_full_eq, tol=1e-5))
        rank_b = int(np.linalg.matrix_rank(b_matrix, tol=1e-5))
        if rank_b == d_ineq:
            return max(0, d_ineq - rank_a)

        g_matrix = a_full_eq.T
        rank_g = rank_a
        if rank_g == g_matrix.shape[0]:
            return 0

        _, _, pivots = qr(g_matrix.T, pivoting=True, mode="economic")
        g1_indices = list(pivots[:rank_g])
        g2_indices = [index for index in range(g_matrix.shape[0]) if index not in set(g1_indices)]
        if not g2_indices:
            return 0
        g1 = g_matrix[g1_indices, :]
        g2 = g_matrix[g2_indices, :]
        b1 = b_matrix[g1_indices, :]
        b2 = b_matrix[g2_indices, :]
        gamma = -np.linalg.lstsq(g1.T, g2.T, rcond=None)[0]
        return int(np.linalg.matrix_rank(gamma.T @ b1 + b2, tol=1e-5))
    except (ValueError, np.linalg.LinAlgError):
        return _binding_rank_degrees_of_freedom(
            b_matrix=b_matrix,
            constraint_matrix=constraint_matrix,
            d_vector=d_vector,
            beta_red_star=beta_red_star,
            nuisance_star=nuisance_star,
        )


def _binding_rank_degrees_of_freedom(
    *,
    b_matrix: np.ndarray,
    constraint_matrix: np.ndarray,
    d_vector: np.ndarray,
    beta_red_star: np.ndarray,
    nuisance_star: np.ndarray,
) -> int:
    slack = d_vector - b_matrix @ beta_red_star + constraint_matrix @ nuisance_star
    active = np.abs(slack) < 1e-5
    if not bool(np.any(active)):
        return 0
    return int(np.linalg.matrix_rank(b_matrix[active, :], tol=1e-5))


def _arp_honest_test(
    *,
    y_t: np.ndarray,
    x_t: np.ndarray,
    sigma: np.ndarray,
    alpha: float,
    hybrid_kappa: float,
) -> dict[str, object]:
    y_t = np.asarray(y_t, dtype=float)
    x_t = np.asarray(x_t, dtype=float)
    if x_t.ndim == 1:
        x_t = x_t.reshape(-1, 1)
    if x_t.shape[0] != y_t.shape[0]:
        raise ValueError("x_t must have one row per ARP moment.")

    sigma = _positive_semidefinite_covariance(np.asarray(sigma, dtype=float), tol=1e-10)
    original_moment_count = int(y_t.size)
    positive_variance = np.diag(sigma) > 1e-12
    positive_variance_selector = positive_variance.tolist()
    if not bool(np.all(positive_variance)):
        y_t = y_t[positive_variance]
        x_t = x_t[positive_variance, :]
        sigma = sigma[np.ix_(positive_variance, positive_variance)]
    if y_t.size == 0:
        return {
            "reject": False,
            "eta": 0.0,
            "standardized_stat": 0.0,
            "critical_value": 0.0,
            "sigma_B": 1.0,
            "lambda": np.asarray([], dtype=float),
            "diagnostics": {
                "variant": "least_favorable_arp",
                "reason": "no positive-variance ARP moments",
            },
        }

    y_honest = -y_t
    lin_sol = _arp_test_delta_lp(y_t=y_honest, x_t=x_t, sigma=sigma)
    lf_cv = _arp_least_favorable_cv(
        x_t=x_t,
        sigma=sigma,
        hybrid_kappa=hybrid_kappa,
    )
    eta = float(lin_sol["eta"])
    reject = bool(eta > lf_cv)
    lambda_vec = np.asarray(lin_sol["lambda"], dtype=float)
    lambda_full = np.zeros(original_moment_count, dtype=float)
    lambda_full[positive_variance] = lambda_vec
    sigma_b = float(lambda_vec.T @ sigma @ lambda_vec)
    sigma_b = float(np.sqrt(max(sigma_b, 0.0))) if sigma_b > 1e-12 else 1.0
    standardized_stat = eta / sigma_b
    critical_value = lf_cv / sigma_b
    return {
        "reject": reject,
        "eta": eta,
        "standardized_stat": standardized_stat,
        "critical_value": critical_value,
        "sigma_B": sigma_b,
        "lambda": lambda_vec,
            "diagnostics": {
                "variant": "least_favorable_arp",
                "paper_reference": "manuscript/sources/arxiv-2404.11739v3/draft.tex:412-421",
                "reference_implementation": (
                    "HonestDiD:::.compute_least_favorable_cv and "
                    "HonestDiD:::.test_delta_lp_fn via TestMechs ARP"
                ),
                "eta": eta,
                "least_favorable_cv": lf_cv,
                "standardized_stat": standardized_stat,
                "critical_value": critical_value,
                "sigma_B": sigma_b,
                "lambda": lambda_full.tolist(),
                "hybrid_kappa": hybrid_kappa,
                "alpha": alpha,
                "simulation_draws": 1000,
                "simulation_seed": 0,
                "original_moment_count": original_moment_count,
                "positive_variance_moments": int(y_t.size),
                "removed_zero_variance_moments": int((~positive_variance).sum()),
                "positive_variance_selector": positive_variance_selector,
                "p_value_note": "The R HonestDiD/TestMechs ARP reference returns reject/eta, not a p-value.",
            },
        }


def _arp_test_delta_lp(
    *,
    y_t: np.ndarray,
    x_t: np.ndarray,
    sigma: np.ndarray,
) -> dict[str, object]:
    sd_vec = np.sqrt(np.diag(sigma))
    design = np.column_stack([sd_vec, x_t])
    result = linprog(
        c=np.r_[1.0, np.zeros(x_t.shape[1], dtype=float)],
        A_ub=-design,
        b_ub=-y_t,
        bounds=[(None, None)] * design.shape[1],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"ARP LP failed: {result.message}")
    marginals = np.asarray(result.ineqlin.marginals, dtype=float)
    lambda_vec = -marginals
    lambda_vec[np.abs(lambda_vec) < 1e-10] = 0.0
    return {
        "eta": float(result.fun),
        "delta": np.asarray(result.x[1:], dtype=float),
        "lambda": lambda_vec,
    }


def _arp_least_favorable_cv(
    *,
    x_t: np.ndarray,
    sigma: np.ndarray,
    hybrid_kappa: float,
    sims: int = 1000,
    seed: int = 0,
) -> float:
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(
        mean=np.zeros(sigma.shape[0], dtype=float),
        cov=sigma,
        size=sims,
        method="svd",
    )
    eta = np.asarray(
        [
            _arp_compute_eta(y_t=-draw, x_t=x_t, sigma=sigma)
            for draw in draws
        ],
        dtype=float,
    )
    eta = eta[np.isfinite(eta)]
    if eta.size == 0:
        return 0.0
    return float(np.quantile(eta, 1.0 - hybrid_kappa))


def _arp_compute_eta(
    *,
    y_t: np.ndarray,
    x_t: np.ndarray,
    sigma: np.ndarray,
) -> float:
    return float(_arp_test_delta_lp(y_t=-np.asarray(y_t, dtype=float), x_t=x_t, sigma=sigma)["eta"])


def _positive_semidefinite_covariance(sigma: np.ndarray, *, tol: float) -> np.ndarray:
    working = (np.asarray(sigma, dtype=float) + np.asarray(sigma, dtype=float).T) / 2.0
    eigenvalues = np.linalg.eigvalsh(working)
    if float(eigenvalues.min()) < -tol:
        working = working + np.eye(working.shape[0]) * (abs(float(eigenvalues.min())) + tol)
    return working


def _cox_shi_nonuisance(
    *,
    y: np.ndarray,
    sigma: np.ndarray,
    alpha: float,
    refinement: bool,
    tol: float = 1e-8,
) -> dict[str, float | bool]:
    y = np.asarray(y, dtype=float).reshape(-1)
    sigma = np.asarray(sigma, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(sigma)
    if float(eigenvalues.min()) < tol:
        positive = eigenvalues > tol
        selector = np.eye(len(eigenvalues))[positive]
        if selector.ndim == 1:
            selector = selector.reshape(1, -1)
        x_star = (selector @ eigenvectors.T @ y).reshape(-1)
        sigma_working = np.diag(eigenvalues[positive])
        a_matrix = eigenvectors @ selector.T
        b_vector = (a_matrix @ x_star - y).reshape(-1)
        y_working = x_star
    else:
        sigma_working = sigma
        y_working = y
        a_matrix = np.eye(sigma.shape[0])
        b_vector = np.zeros(sigma.shape[0], dtype=float)

    if y_working.size == 0:
        if bool(np.all(b_vector >= -tol)):
            return {
                "reject": False,
                "test_stat": 0.0,
                "critical_value": 0.0,
                "p_value": 1.0,
            }
        return {
            "reject": True,
            "test_stat": float("inf"),
            "critical_value": 0.0,
            "p_value": 0.0,
        }

    support_feasibility = linprog(
        c=np.zeros(len(y_working), dtype=float),
        A_ub=a_matrix,
        b_ub=b_vector,
        bounds=[(None, None)] * len(y_working),
        method="highs",
    )
    if not support_feasibility.success:
        return {
            "reject": True,
            "test_stat": float("inf"),
            "critical_value": 0.0,
            "p_value": 0.0,
        }

    solution, test_stat = _solve_qp_active_set(
        y=y_working,
        sigma=sigma_working,
        a_matrix=a_matrix,
        b_vector=b_vector,
    )

    binding = np.abs(a_matrix @ solution - b_vector) < 1e-5
    chisquared_df = int(binding.sum())
    if chisquared_df == 0:
        return {
            "reject": False,
            "test_stat": float(test_stat),
            "critical_value": 0.0,
            "p_value": 1.0,
        }

    if (not refinement) or chisquared_df != 1:
        critical_value = float(chi2.ppf(1 - alpha, df=chisquared_df))
        p_value = float(chi2.sf(test_stat, df=chisquared_df))
        return {
            "reject": bool(test_stat > critical_value),
            "test_stat": float(test_stat),
            "critical_value": critical_value,
            "p_value": p_value,
        }

    projection_cov = a_matrix @ sigma_working @ a_matrix.T
    binding_vector = binding.astype(float)
    binding_norm = float(np.sqrt(binding_vector.T @ projection_cov @ binding_vector))
    numerator = -binding_norm * (a_matrix @ solution - b_vector)
    denominator = binding_norm * np.sqrt(np.diag(projection_cov)) - projection_cov @ binding_vector
    candidate_mask = (~binding) & (denominator > 0)

    if np.any(candidate_mask):
        tau_hat = float(np.min(numerator[candidate_mask] / denominator[candidate_mask]))
        beta_hat = float(2 * alpha * norm.cdf(tau_hat))
    else:
        tau_hat = 0.0
        beta_hat = float(alpha)

    critical_value = float(chi2.ppf(1 - beta_hat, df=chisquared_df))
    p_value = float(chi2.sf(test_stat, df=chisquared_df) / (2 * norm.cdf(tau_hat)))
    return {
        "reject": bool(test_stat > critical_value),
        "test_stat": float(test_stat),
        "critical_value": critical_value,
        "p_value": p_value,
    }


def _solve_qp_active_set(
    *,
    y: np.ndarray,
    sigma: np.ndarray,
    a_matrix: np.ndarray,
    b_vector: np.ndarray,
) -> tuple[np.ndarray, float]:
    sigma_inv = np.linalg.inv(sigma)
    dimension = len(y)
    num_constraints = a_matrix.shape[0]
    best_solution: np.ndarray | None = None
    best_value = float("inf")

    for active_size in range(num_constraints + 1):
        for active in combinations(range(num_constraints), active_size):
            active_indices = list(active)
            if not active_indices:
                candidate = y.copy()
                multipliers = np.zeros(0)
            else:
                active_matrix = a_matrix[active_indices, :]
                kkt_matrix = np.block(
                    [
                        [2 * sigma_inv, active_matrix.T],
                        [active_matrix, np.zeros((active_size, active_size))],
                    ]
                )
                rhs = np.concatenate([2 * sigma_inv @ y, b_vector[active_indices]])
                solution = np.linalg.lstsq(kkt_matrix, rhs, rcond=None)[0]
                candidate = solution[:dimension]
                multipliers = solution[dimension:]

            slack = a_matrix @ candidate - b_vector
            if np.any(slack > 1e-7):
                continue
            if active_indices and np.any(multipliers < -1e-7):
                continue

            value = float((y - candidate).T @ sigma_inv @ (y - candidate))
            if value < best_value:
                best_solution = candidate
                best_value = value

    if best_solution is None:
        raise RuntimeError("Failed to solve the Cox-Shi quadratic program.")

    return best_solution, best_value
