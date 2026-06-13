"""Request descriptors and support-scope contracts for Testing Mechanisms.

This module defines the immutable request dataclasses and support-scope
contract objects that form the validation and declaration layer of
``testmechs``.  Each request class:

- Declares the inputs, method choice, and tuning parameters for one
  public estimation or inference endpoint.
- Validates fields on construction via ``__post_init__`` (frozen dataclass).
- Exposes a ``comparison_view()`` method returning a strict-JSON-safe
  dictionary suitable for reproducibility logs and request-deduplication.

Support-scope contracts (``RegressionAdjustmentSupport``,
``SharpNullDiagnosticsSupport``, ``PartialDensitySupport``,
``CellCountDiagnosticsSupport``, ``BoundsSupport``) declare the scope,
diagnostic schema, and release-status metadata for each public surface,
consumed by manuscript validation, release audits, and API documentation.

Request And Support Classes
---------------------------
SharedCSVInput
    Common data-source specification shared across estimation requests.
SharpNullRequest
    Request for a sharp-null hypothesis test.
LowerBoundRequest
    Request for an always-taker affected-fraction lower bound.
BreakdownDefierShareRequest
    Request for breakdown-defier-share diagnostic computation.
ADEBoundsRequest
    Request for ADE Lee-style trimming bounds.
PartialDensityRequest
    Request for partial-density data computation.
PaperEmpiricalReproductionRequest
    Request for the paper empirical reproduction report.
PaperMonteCarloReproductionRequest
    Request for the paper Monte Carlo reproduction report.
PaperReproductionComparisonRequest
    Request for the combined empirical + Monte Carlo comparison report.
PaperReproductionE2ERequest
    Request for the end-to-end reproduction pipeline.
PaperReproductionResourceManifestRequest
    Request for the reproduction resource manifest.
RegressionAdjustmentSupport
    Release-scope contract for adjusted regression paths.
SharpNullDiagnosticsSupport
    Release-scope contract for sharp-null diagnostic schemas.
PartialDensitySupport
    Release-scope contract for partial-density paths.
CellCountDiagnosticsSupport
    Release-scope contract for cell-count diagnostics.
BoundsSupport
    Release-scope contract for bounds paths.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd

from .results import _json_safe_payload

REGRESSION_FORMULA_KINDS = (
    "trivial",
    "controls",
    "fixed_effects",
    "iv",
    "iv_fixed_effects",
)


def _request_text(value: object, *, field: str) -> str:
    """Validate and strip a required text field; raise on blank."""
    text = str(value).strip()
    if not text:
        raise ValueError(f"Request comparison field {field} is blank")
    return text


def _optional_request_text(value: object | None, *, field: str) -> str | None:
    """Validate an optional text field; return None for missing/NA values."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return _request_text(value, field=field)


def _optional_reproduction_filter(value: object | None, *, field: str) -> str | None:
    """Validate an optional reproduction filter string."""
    return _optional_request_text(value, field=field)


def _optional_reproduction_sequence(value: object | None, *, field: str) -> object | None:
    """Validate an optional typed sequence for reproduction filters."""
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(
            f"Request comparison field {field} must be a non-string sequence or None"
        )
    if field in {"clusters", "bins"}:
        for item in value:
            if item is not None and (isinstance(item, bool) or not isinstance(item, Integral)):
                raise ValueError(
                    f"Request comparison field {field} entries must be integers or None"
                )
    elif field == "t_values":
        for item in value:
            if isinstance(item, bool) or not isinstance(item, Real):
                raise ValueError(
                    "Request comparison field t_values entries must be numeric"
                )
    return value


def _request_nonnegative_integer(value: object, *, field: str) -> int:
    """Validate and return a nonnegative integer field."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"Request comparison field {field} must be an integer")
    numeric_value = int(value)
    if numeric_value < 0:
        raise ValueError(f"Request comparison field {field} must be nonnegative")
    return numeric_value


def _request_positive_integer(value: object, *, field: str) -> int:
    """Validate and return a strictly positive integer field."""
    numeric_value = _request_nonnegative_integer(value, field=field)
    if numeric_value <= 0:
        raise ValueError(f"Request comparison field {field} must be positive")
    return numeric_value


def _optional_positive_integer(value: object | None, *, field: str) -> int | None:
    """Validate an optional positive integer; return None when absent."""
    if value is None:
        return None
    return _request_positive_integer(value, field=field)


def _request_finite_real(value: object, *, field: str) -> float:
    """Validate and return a finite real-valued numeric field."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"Request comparison field {field} must be numeric")
    numeric_value = float(value)
    if not isfinite(numeric_value):
        raise ValueError(f"Request comparison field {field} must be finite")
    return numeric_value


def _request_probability(value: object, *, field: str) -> float:
    """Validate a strict probability in (0, 1) exclusive."""
    numeric_value = _request_finite_real(value, field=field)
    if not 0.0 < numeric_value < 1.0:
        raise ValueError(f"Request comparison field {field} must be between 0 and 1")
    return numeric_value


def _request_inclusive_probability(value: object, *, field: str) -> float:
    """Validate a probability in [0, 1] inclusive."""
    numeric_value = _request_finite_real(value, field=field)
    if not 0.0 <= numeric_value <= 1.0:
        raise ValueError(
            f"Request comparison field {field} must be between 0 and 1 inclusive"
        )
    return numeric_value


def _optional_inclusive_probability(value: object | None, *, field: str) -> float | None:
    """Validate an optional inclusive probability; return None when absent."""
    if value is None:
        return None
    return _request_inclusive_probability(value, field=field)


def _request_unit_share(value: object, *, field: str) -> float:
    """Validate a nonneg share in [0, 1]."""
    numeric_value = _request_nonnegative_real(value, field=field)
    if numeric_value > 1.0:
        raise ValueError(f"Request comparison field {field} must be no larger than 1")
    return numeric_value


def _request_nonnegative_real(value: object, *, field: str) -> float:
    """Validate and return a nonnegative finite real."""
    numeric_value = _request_finite_real(value, field=field)
    if numeric_value < 0.0:
        raise ValueError(f"Request comparison field {field} must be nonnegative")
    return numeric_value


def _optional_nonnegative_real(value: object | None, *, field: str) -> float | None:
    """Validate an optional nonnegative real; return None when absent."""
    if value is None:
        return None
    return _request_nonnegative_real(value, field=field)


def _request_boolean(value: object, *, field: str) -> bool:
    """Validate and return a boolean field."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"Request comparison field {field} must be boolean")


def _request_positive_real(value: object, *, field: str) -> float:
    """Validate and return a strictly positive finite real."""
    numeric_value = _request_finite_real(value, field=field)
    if numeric_value <= 0.0:
        raise ValueError(f"Request comparison field {field} must be positive")
    return numeric_value


@dataclass(frozen=True)
class SharedCSVInput:
    """Common data-source specification shared across estimation requests.

    Declares the CSV file path and column-role assignments (treatment,
    mediators, outcome) consumed by all public Testing Mechanisms
    estimators.  Validates that mediators is a non-empty sequence and
    resolves the data path to an absolute path on construction.

    Attributes
    ----------
    data_path : Path
        Absolute resolved path to the input CSV file.
    treatment : str
        Column name for the binary treatment indicator.
    mediators : tuple of str
        One or more column names for mediator variables.
    outcome : str
        Column name for the outcome variable.
    cluster : str or None
        Optional column name for cluster identifiers.
    id_column : str or None
        Optional column name for observation identifiers.

    Examples
    --------
    >>> data = SharedCSVInput(
    ...     data_path="data.csv",
    ...     treatment="D",
    ...     mediators=("M",),
    ...     outcome="Y",
    ... )
    >>> data.primary_mediator
    'M'
    """
    data_path: Path
    treatment: str
    mediators: tuple[str, ...]
    outcome: str
    cluster: str | None = None
    id_column: str | None = None

    def __post_init__(self) -> None:
        """Validate and normalize all fields on construction."""
        object.__setattr__(self, "data_path", Path(self.data_path).resolve())
        if isinstance(self.mediators, (str, bytes)) or not isinstance(
            self.mediators, Sequence
        ):
            raise ValueError("mediators must be a non-string sequence of column labels")
        if not self.mediators:
            raise ValueError("mediators must contain at least one column")
        object.__setattr__(self, "treatment", _request_text(self.treatment, field="treatment"))
        object.__setattr__(self, "mediators", tuple(
            _request_text(mediator, field="mediators")
            for mediator in self.mediators
        ))
        object.__setattr__(self, "outcome", _request_text(self.outcome, field="outcome"))
        object.__setattr__(self, "cluster", _optional_request_text(self.cluster, field="cluster"))
        object.__setattr__(self, "id_column", _optional_request_text(self.id_column, field="id_column"))

    def base_comparison_fields(self) -> dict[str, object]:
        """Return a JSON-safe dictionary of the shared data-source fields.

        Used by request ``comparison_view()`` methods to embed the data
        specification into reproducibility logs.

        Returns
        -------
        dict of str to object
            Strict-JSON-safe dictionary of path, columns, and roles.
        """
        return _json_safe_payload({
            "data_path": str(self.data_path),
            "treatment": self.treatment,
            "mediators": list(self.mediators),
            "outcome": self.outcome,
            "cluster": self.cluster,
            "id_column": self.id_column,
        })

    @property
    def primary_mediator(self) -> str:
        """Return the first mediator column name."""
        return self.mediators[0]


@dataclass(frozen=True)
class SharpNullRequest:
    """Request descriptor for a sharp-null hypothesis test.

    Records the input data source, column roles, method choice, and tuning
    parameters before the test statistic is computed.

    Attributes
    ----------
    dataset : SharedCSVInput
        Data source specification with file path and column roles.
    method : str
        Statistical method. One of ``"CS"``, ``"ARP"``, ``"FSSTdd"``,
        ``"FSSTndd"``, ``"K"``.
    num_y_bins : int or None
        Optional outcome discretization bin count.
    alpha : float
        Significance level (default 0.05).
    reg_formula : str or None
        Optional regression formula for adjustment.
    frac_ats_affected : float or None
        Optional fraction of always-takers affected for relaxed null.
    max_defiers_share : float
        Maximum defier share cap (default 0.0 for strict monotonicity).

    See Also
    --------
    SharpNullResult : Result object returned by this test.
    """
    dataset: SharedCSVInput
    method: str = "CS"
    num_y_bins: int | None = None
    alpha: float = 0.05
    reg_formula: str | None = None
    frac_ats_affected: float | None = None
    max_defiers_share: float = 0.0

    def comparison_view(self) -> dict[str, object]:
        """Return a strict-JSON-safe dictionary for reproducibility logging.

        Returns
        -------
        dict of str to object
            Validated request parameters suitable for deduplication and audit.
        """
        return _json_safe_payload({
            "function": "test_sharp_null",
            **self.dataset.base_comparison_fields(),
            "method": _request_text(self.method, field="method"),
            "num_y_bins": _optional_positive_integer(
                self.num_y_bins,
                field="num_y_bins",
            ),
            "alpha": _request_probability(self.alpha, field="alpha"),
            "reg_formula": self.reg_formula,
            "frac_ats_affected": _optional_inclusive_probability(
                self.frac_ats_affected,
                field="frac_ats_affected",
            ),
            "max_defiers_share": _request_unit_share(
                self.max_defiers_share,
                field="max_defiers_share",
            ),
        })


@dataclass(frozen=True)
class LowerBoundRequest:
    """Request descriptor for an always-taker affected-fraction lower bound.

    Attributes
    ----------
    dataset : SharedCSVInput
        Data source specification.
    at_group : object or None
        Target always-taker group, or ``None`` for pooled.
    num_y_bins : int or None
        Optional outcome discretization bin count.
    reg_formula : str or None
        Optional regression formula for adjustment.
    max_defiers_share : float
        Maximum defier share cap (default 0.0).
    allow_min_defiers : bool
        Use exact minimum compatible defier cap (default False).
    return_min_defiers : bool
        Include minimum compatible defier share in diagnostics (default False).

    See Also
    --------
    LowerBoundResult : Result object returned by this computation.
    """
    dataset: SharedCSVInput
    at_group: object | None = None
    num_y_bins: int | None = None
    reg_formula: str | None = None
    max_defiers_share: float = 0.0
    allow_min_defiers: bool = False
    return_min_defiers: bool = False

    def comparison_view(self) -> dict[str, object]:
        """Return a strict-JSON-safe dictionary for reproducibility logging."""
        return _json_safe_payload({
            "function": "lb_frac_affected",
            **self.dataset.base_comparison_fields(),
            "at_group": self.at_group,
            "num_y_bins": _optional_positive_integer(
                self.num_y_bins,
                field="num_y_bins",
            ),
            "reg_formula": self.reg_formula,
            "max_defiers_share": _request_unit_share(
                self.max_defiers_share,
                field="max_defiers_share",
            ),
            "allow_min_defiers": _request_boolean(
                self.allow_min_defiers,
                field="allow_min_defiers",
            ),
            "return_min_defiers": _request_boolean(
                self.return_min_defiers,
                field="return_min_defiers",
            ),
        })


@dataclass(frozen=True)
class BreakdownDefierShareRequest:
    """Request descriptor for breakdown-defier-share diagnostic.

    Computes the smallest defier-share cap at which the lower bound
    loses identifying bite.

    Attributes
    ----------
    dataset : SharedCSVInput
        Data source specification.
    at_group : object or None
        Target always-taker group, or ``None`` for pooled.
    num_y_bins : int or None
        Optional outcome discretization bin count.
    reg_formula : str or None
        Optional regression formula for adjustment.
    tol : float
        Bisection convergence tolerance (default 1e-4).
    max_iterations : int
        Maximum bisection iterations (default 80).
    """
    dataset: SharedCSVInput
    at_group: object | None = None
    num_y_bins: int | None = None
    reg_formula: str | None = None
    tol: float = 1e-4
    max_iterations: int = 80

    def comparison_view(self) -> dict[str, object]:
        """Return a strict-JSON-safe dictionary for reproducibility logging."""
        return _json_safe_payload({
            "function": "breakdown_defier_share",
            **self.dataset.base_comparison_fields(),
            "at_group": self.at_group,
            "num_y_bins": _optional_positive_integer(
                self.num_y_bins,
                field="num_y_bins",
            ),
            "reg_formula": self.reg_formula,
            "tol": _request_positive_real(self.tol, field="tol"),
            "max_iterations": _request_positive_integer(
                self.max_iterations,
                field="max_iterations",
            ),
        })


@dataclass(frozen=True)
class ADEBoundsRequest:
    """Request descriptor for ADE Lee-style trimming bounds.

    Attributes
    ----------
    dataset : SharedCSVInput
        Data source specification.
    at_group : object
        Target always-taker group (default 1).
    reg_formula : str or None
        Optional regression formula for adjustment.
    max_defiers_share : float
        Maximum defier share cap (default 0.0).
    allow_min_defiers : bool
        Use exact minimum compatible defier cap (default False).

    See Also
    --------
    ADEBoundsResult : Result object returned by this computation.
    """
    dataset: SharedCSVInput
    at_group: object = 1
    reg_formula: str | None = None
    max_defiers_share: float = 0.0
    allow_min_defiers: bool = False

    def comparison_view(self) -> dict[str, object]:
        """Return a strict-JSON-safe dictionary for reproducibility logging."""
        return _json_safe_payload({
            "function": "bounds_ade_ats",
            **self.dataset.base_comparison_fields(),
            "at_group": self.at_group,
            "reg_formula": self.reg_formula,
            "max_defiers_share": _request_unit_share(
                self.max_defiers_share,
                field="max_defiers_share",
            ),
            "allow_min_defiers": _request_boolean(
                self.allow_min_defiers,
                field="allow_min_defiers",
            ),
        })


@dataclass(frozen=True)
class PartialDensityRequest:
    """Request descriptor for partial-density data computation.

    Attributes
    ----------
    dataset : SharedCSVInput
        Data source specification.
    num_y_bins : int or None
        Optional outcome discretization bin count.
    plot_nts : bool
        Plot never-taker orientation (default False).
    continuous_y : bool
        Use continuous kernel-density estimation (default False).
    num_grid_points : int
        Number of grid points for continuous density (default 10000).
    reg_formula : str or None
        Optional regression formula for adjustment.
    """
    dataset: SharedCSVInput
    num_y_bins: int | None = None
    plot_nts: bool = False
    continuous_y: bool = False
    num_grid_points: int = 10000
    reg_formula: str | None = None

    def comparison_view(self) -> dict[str, object]:
        """Return a strict-JSON-safe dictionary for reproducibility logging."""
        return _json_safe_payload({
            "function": "partial_density_data",
            **self.dataset.base_comparison_fields(),
            "num_y_bins": _optional_positive_integer(
                self.num_y_bins,
                field="num_y_bins",
            ),
            "plot_nts": _request_boolean(self.plot_nts, field="plot_nts"),
            "continuous_y": _request_boolean(
                self.continuous_y,
                field="continuous_y",
            ),
            "num_grid_points": _request_positive_integer(
                self.num_grid_points,
                field="num_grid_points",
            ),
            "reg_formula": self.reg_formula,
        })


@dataclass(frozen=True)
class PaperEmpiricalReproductionRequest:
    """Request descriptor for the paper empirical reproduction report.

    Attributes
    ----------
    fixtures_dir : str, Path, or None
        Optional directory for fixture data files.
    statistics_dir : str, Path, or None
        Optional directory for pre-computed statistic files.
    """
    fixtures_dir: str | Path | None = None
    statistics_dir: str | Path | None = None

    def comparison_view(self) -> dict[str, object]:
        """Return a strict-JSON-safe dictionary for reproducibility logging."""
        return _json_safe_payload({
            "function": "paper_empirical_reproduction_report",
            "fixtures_dir": None
            if self.fixtures_dir is None
            else str(Path(self.fixtures_dir).resolve()),
            "statistics_dir": None
            if self.statistics_dir is None
            else str(Path(self.statistics_dir).resolve()),
        })


@dataclass(frozen=True)
class PaperMonteCarloReproductionRequest:
    """Request descriptor for the paper Monte Carlo reproduction report.

    Controls the simulation configuration, tolerance thresholds, and
    optional slice-level filters for the Monte Carlo comparison.

    Attributes
    ----------
    evidence_dir : str or Path
        Directory containing Monte Carlo evidence artifacts.
    tables_dir : str, Path, or None
        Optional directory for output table files.
    fixtures_dir : str, Path, or None
        Optional directory for fixture data files.
    seed : int
        Random seed for reproducibility (default 20260509).
    cell_chunk_size : int
        Number of cells per processing chunk (default 5).
    paper_replications : int
        Number of replications matching the paper (default 500).
    slice_replications : int or None
        Optional override for per-slice replication count.
    bootstrap_replications : int
        Bootstrap replication count (default 500).
    mediator : str or None
        Optional mediator filter.
    design : str or None
        Optional design filter.
    table : str or None
        Optional table filter.
    clusters : tuple of (int or None) or None
        Optional cluster configuration filter.
    bins : tuple of (int or None) or None
        Optional bin configuration filter.
    t_values : tuple of float or None
        Optional treatment-value filter.
    alpha : float
        Significance level (default 0.05).
    absolute_tolerance : float
        Absolute tolerance for comparison (default 0.025).
    z_tolerance : float
        Z-score tolerance threshold (default 2.0).
    cell_count_absolute_tolerance : float or None
        Optional per-cell count tolerance.
    source_mixture_absolute_tolerance : float or None
        Optional source-mixture tolerance.
    owner : str
        Report owner label.
    rerun_command : str or None
        Optional shell command to reproduce this report.
    """
    evidence_dir: str | Path
    tables_dir: str | Path | None = None
    fixtures_dir: str | Path | None = None
    seed: int = 20260509
    cell_chunk_size: int = 5
    paper_replications: int = 500
    slice_replications: int | None = None
    bootstrap_replications: int = 500
    mediator: str | None = None
    design: str | None = None
    table: str | None = None
    clusters: tuple[int | None, ...] | None = None
    bins: tuple[int | None, ...] | None = None
    t_values: tuple[float, ...] | None = None
    alpha: float = 0.05
    absolute_tolerance: float = 0.025
    z_tolerance: float = 2.0
    cell_count_absolute_tolerance: float | None = None
    source_mixture_absolute_tolerance: float | None = None
    owner: str = "Phase 16 paper Monte Carlo reproduction report"
    rerun_command: str | None = None

    def comparison_view(self) -> dict[str, object]:
        """Return a strict-JSON-safe dictionary for reproducibility logging."""
        return _json_safe_payload({
            "function": "paper_monte_carlo_reproduction_report",
            "evidence_dir": str(Path(self.evidence_dir).resolve()),
            "tables_dir": None
            if self.tables_dir is None
            else str(Path(self.tables_dir).resolve()),
            "fixtures_dir": None
            if self.fixtures_dir is None
            else str(Path(self.fixtures_dir).resolve()),
            "seed": _request_nonnegative_integer(self.seed, field="seed"),
            "cell_chunk_size": _request_positive_integer(
                self.cell_chunk_size,
                field="cell_chunk_size",
            ),
            "paper_replications": _request_positive_integer(
                self.paper_replications,
                field="paper_replications",
            ),
            "slice_replications": _optional_positive_integer(
                self.slice_replications,
                field="slice_replications",
            ),
            "bootstrap_replications": _request_positive_integer(
                self.bootstrap_replications,
                field="bootstrap_replications",
            ),
            "mediator": _optional_reproduction_filter(self.mediator, field="mediator"),
            "design": _optional_reproduction_filter(self.design, field="design"),
            "table": _optional_reproduction_filter(self.table, field="table"),
            "clusters": _optional_reproduction_sequence(self.clusters, field="clusters"),
            "bins": _optional_reproduction_sequence(self.bins, field="bins"),
            "t_values": _optional_reproduction_sequence(self.t_values, field="t_values"),
            "alpha": _request_probability(self.alpha, field="alpha"),
            "absolute_tolerance": _request_nonnegative_real(
                self.absolute_tolerance,
                field="absolute_tolerance",
            ),
            "z_tolerance": _request_nonnegative_real(
                self.z_tolerance,
                field="z_tolerance",
            ),
            "cell_count_absolute_tolerance": _optional_nonnegative_real(
                self.cell_count_absolute_tolerance,
                field="cell_count_absolute_tolerance",
            ),
            "source_mixture_absolute_tolerance": _optional_nonnegative_real(
                self.source_mixture_absolute_tolerance,
                field="source_mixture_absolute_tolerance",
            ),
            "owner": self.owner,
            "rerun_command": self.rerun_command,
        })


@dataclass(frozen=True)
class PaperReproductionComparisonRequest:
    """Request descriptor for combined empirical + Monte Carlo comparison.

    Merges the empirical and Monte Carlo reproduction pipelines into a
    single comparison report with shared tolerance parameters.

    Attributes
    ----------
    evidence_dir : str or Path
        Directory containing evidence artifacts.
    fixtures_dir : str, Path, or None
        Optional fixture directory.
    statistics_dir : str, Path, or None
        Optional statistics directory.
    tables_dir : str, Path, or None
        Optional table output directory.
    seed : int
        Random seed (default 20260509).
    cell_chunk_size : int
        Cells per chunk (default 5).
    paper_replications : int
        Paper replication count (default 500).
    slice_replications : int or None
        Optional per-slice override.
    bootstrap_replications : int
        Bootstrap count (default 500).
    mediator : str or None
        Optional mediator filter.
    design : str or None
        Optional design filter.
    table : str or None
        Optional table filter.
    clusters : tuple of (int or None) or None
        Optional cluster filter.
    bins : tuple of (int or None) or None
        Optional bin filter.
    t_values : tuple of float or None
        Optional treatment-value filter.
    alpha : float
        Significance level (default 0.05).
    absolute_tolerance : float
        Absolute tolerance (default 0.025).
    z_tolerance : float
        Z-score tolerance (default 2.0).
    cell_count_absolute_tolerance : float or None
        Optional per-cell count tolerance.
    source_mixture_absolute_tolerance : float or None
        Optional source-mixture tolerance.
    """
    evidence_dir: str | Path
    fixtures_dir: str | Path | None = None
    statistics_dir: str | Path | None = None
    tables_dir: str | Path | None = None
    seed: int = 20260509
    cell_chunk_size: int = 5
    paper_replications: int = 500
    slice_replications: int | None = None
    bootstrap_replications: int = 500
    mediator: str | None = None
    design: str | None = None
    table: str | None = None
    clusters: tuple[int | None, ...] | None = None
    bins: tuple[int | None, ...] | None = None
    t_values: tuple[float, ...] | None = None
    alpha: float = 0.05
    absolute_tolerance: float = 0.025
    z_tolerance: float = 2.0
    cell_count_absolute_tolerance: float | None = None
    source_mixture_absolute_tolerance: float | None = None

    def comparison_view(self) -> dict[str, object]:
        """Return a strict-JSON-safe dictionary for reproducibility logging."""
        return _json_safe_payload({
            "function": "paper_reproduction_comparison_report",
            "evidence_dir": str(Path(self.evidence_dir).resolve()),
            "fixtures_dir": None
            if self.fixtures_dir is None
            else str(Path(self.fixtures_dir).resolve()),
            "statistics_dir": None
            if self.statistics_dir is None
            else str(Path(self.statistics_dir).resolve()),
            "tables_dir": None
            if self.tables_dir is None
            else str(Path(self.tables_dir).resolve()),
            "seed": _request_nonnegative_integer(self.seed, field="seed"),
            "cell_chunk_size": _request_positive_integer(
                self.cell_chunk_size,
                field="cell_chunk_size",
            ),
            "paper_replications": _request_positive_integer(
                self.paper_replications,
                field="paper_replications",
            ),
            "slice_replications": _optional_positive_integer(
                self.slice_replications,
                field="slice_replications",
            ),
            "bootstrap_replications": _request_positive_integer(
                self.bootstrap_replications,
                field="bootstrap_replications",
            ),
            "mediator": _optional_reproduction_filter(self.mediator, field="mediator"),
            "design": _optional_reproduction_filter(self.design, field="design"),
            "table": _optional_reproduction_filter(self.table, field="table"),
            "clusters": _optional_reproduction_sequence(self.clusters, field="clusters"),
            "bins": _optional_reproduction_sequence(self.bins, field="bins"),
            "t_values": _optional_reproduction_sequence(self.t_values, field="t_values"),
            "alpha": _request_probability(self.alpha, field="alpha"),
            "absolute_tolerance": _request_nonnegative_real(
                self.absolute_tolerance,
                field="absolute_tolerance",
            ),
            "z_tolerance": _request_nonnegative_real(
                self.z_tolerance,
                field="z_tolerance",
            ),
            "cell_count_absolute_tolerance": _optional_nonnegative_real(
                self.cell_count_absolute_tolerance,
                field="cell_count_absolute_tolerance",
            ),
            "source_mixture_absolute_tolerance": _optional_nonnegative_real(
                self.source_mixture_absolute_tolerance,
                field="source_mixture_absolute_tolerance",
            ),
        })


@dataclass(frozen=True)
class PaperReproductionE2ERequest:
    """Request descriptor for the end-to-end reproduction pipeline.

    Orchestrates the full reproduction workflow including empirical,
    Monte Carlo, milestone auditing, and roadmap analysis components.

    Attributes
    ----------
    evidence_dir : str or Path
        Directory containing evidence artifacts.
    milestone_version : str
        Target milestone version label (default ``"v1.2"``).
    roadmap_analysis : dict or None
        Optional pre-computed roadmap analysis payload.
    requirements_analysis : dict or None
        Optional pre-computed requirements analysis payload.
    requirements_path : str, Path, or None
        Optional path to requirements specification.
    milestone_audit_status : str or None
        Optional milestone audit status label.
    milestone_audit_path : str, Path, or None
        Optional path to milestone audit file.
    planning_dir : str or Path
        Planning directory path (default ``".planning"``).
    fixtures_dir : str, Path, or None
        Optional fixture directory.
    statistics_dir : str, Path, or None
        Optional statistics directory.
    tables_dir : str, Path, or None
        Optional table output directory.
    seed : int
        Random seed (default 20260509).
    cell_chunk_size : int
        Cells per chunk (default 5).
    paper_replications : int
        Paper replication count (default 500).
    slice_replications : int or None
        Optional per-slice override.
    bootstrap_replications : int
        Bootstrap count (default 500).
    mediator : str or None
        Optional mediator filter.
    design : str or None
        Optional design filter.
    table : str or None
        Optional table filter.
    clusters : tuple of (int or None) or None
        Optional cluster filter.
    bins : tuple of (int or None) or None
        Optional bin filter.
    t_values : tuple of float or None
        Optional treatment-value filter.
    alpha : float
        Significance level (default 0.05).
    absolute_tolerance : float
        Absolute tolerance (default 0.025).
    z_tolerance : float
        Z-score tolerance (default 2.0).
    cell_count_absolute_tolerance : float or None
        Optional per-cell count tolerance.
    source_mixture_absolute_tolerance : float or None
        Optional source-mixture tolerance.
    """
    evidence_dir: str | Path
    milestone_version: str = "v1.2"
    roadmap_analysis: dict[str, object] | None = None
    requirements_analysis: dict[str, object] | None = None
    requirements_path: str | Path | None = None
    milestone_audit_status: str | None = None
    milestone_audit_path: str | Path | None = None
    planning_dir: str | Path = ".planning"
    fixtures_dir: str | Path | None = None
    statistics_dir: str | Path | None = None
    tables_dir: str | Path | None = None
    seed: int = 20260509
    cell_chunk_size: int = 5
    paper_replications: int = 500
    slice_replications: int | None = None
    bootstrap_replications: int = 500
    mediator: str | None = None
    design: str | None = None
    table: str | None = None
    clusters: tuple[int | None, ...] | None = None
    bins: tuple[int | None, ...] | None = None
    t_values: tuple[float, ...] | None = None
    alpha: float = 0.05
    absolute_tolerance: float = 0.025
    z_tolerance: float = 2.0
    cell_count_absolute_tolerance: float | None = None
    source_mixture_absolute_tolerance: float | None = None

    def comparison_view(self) -> dict[str, object]:
        """Return a strict-JSON-safe dictionary for reproducibility logging."""
        return _json_safe_payload({
            "function": "paper_reproduction_e2e_report",
            "evidence_dir": str(Path(self.evidence_dir).resolve()),
            "milestone_version": self.milestone_version,
            "roadmap_analysis": self.roadmap_analysis,
            "requirements_analysis": self.requirements_analysis,
            "requirements_path": None
            if self.requirements_path is None
            else str(Path(self.requirements_path).resolve()),
            "milestone_audit_status": self.milestone_audit_status,
            "milestone_audit_path": None
            if self.milestone_audit_path is None
            else str(Path(self.milestone_audit_path).resolve()),
            "planning_dir": str(Path(self.planning_dir).resolve()),
            "fixtures_dir": None
            if self.fixtures_dir is None
            else str(Path(self.fixtures_dir).resolve()),
            "statistics_dir": None
            if self.statistics_dir is None
            else str(Path(self.statistics_dir).resolve()),
            "tables_dir": None
            if self.tables_dir is None
            else str(Path(self.tables_dir).resolve()),
            "seed": _request_nonnegative_integer(self.seed, field="seed"),
            "cell_chunk_size": _request_positive_integer(
                self.cell_chunk_size,
                field="cell_chunk_size",
            ),
            "paper_replications": _request_positive_integer(
                self.paper_replications,
                field="paper_replications",
            ),
            "slice_replications": _optional_positive_integer(
                self.slice_replications,
                field="slice_replications",
            ),
            "bootstrap_replications": _request_positive_integer(
                self.bootstrap_replications,
                field="bootstrap_replications",
            ),
            "mediator": _optional_reproduction_filter(self.mediator, field="mediator"),
            "design": _optional_reproduction_filter(self.design, field="design"),
            "table": _optional_reproduction_filter(self.table, field="table"),
            "clusters": _optional_reproduction_sequence(self.clusters, field="clusters"),
            "bins": _optional_reproduction_sequence(self.bins, field="bins"),
            "t_values": _optional_reproduction_sequence(self.t_values, field="t_values"),
            "alpha": _request_probability(self.alpha, field="alpha"),
            "absolute_tolerance": _request_nonnegative_real(
                self.absolute_tolerance,
                field="absolute_tolerance",
            ),
            "z_tolerance": _request_nonnegative_real(
                self.z_tolerance,
                field="z_tolerance",
            ),
            "cell_count_absolute_tolerance": _optional_nonnegative_real(
                self.cell_count_absolute_tolerance,
                field="cell_count_absolute_tolerance",
            ),
            "source_mixture_absolute_tolerance": _optional_nonnegative_real(
                self.source_mixture_absolute_tolerance,
                field="source_mixture_absolute_tolerance",
            ),
        })


@dataclass(frozen=True)
class PaperReproductionResourceManifestRequest:
    """Request descriptor for reproduction resource manifest generation.

    Declares expected resource counts and output path for the
    packaged-resource manifest that audits the fixture, statistic, and
    table inventories.

    Attributes
    ----------
    output_path : str, Path, or None
        Optional output file path for the manifest JSON.
    overwrite : bool
        Whether to overwrite an existing manifest (default False).
    expected_resource_count : int
        Expected total number of resources (default 40).
    expected_fixture_count : int
        Expected number of fixture files (default 3).
    expected_empirical_statistic_count : int
        Expected number of empirical statistic files (default 30).
    expected_monte_carlo_table_count : int
        Expected number of Monte Carlo table files (default 7).
    """
    output_path: str | Path | None = None
    overwrite: bool = False
    expected_resource_count: int = 40
    expected_fixture_count: int = 3
    expected_empirical_statistic_count: int = 30
    expected_monte_carlo_table_count: int = 7

    def comparison_view(self) -> dict[str, object]:
        """Return a strict-JSON-safe dictionary for reproducibility logging."""
        if not isinstance(self.overwrite, bool):
            raise ValueError("Request comparison field overwrite must be boolean")
        return _json_safe_payload({
            "function": "paper_reproduction_resource_manifest_packet",
            "writer_function": "write_paper_reproduction_resource_manifest_json",
            "loader_functions": [
                "load_paper_reproduction_resource_manifest_json",
                "load_paper_reproduction_resource_manifest_packet_json",
            ],
            "output_path": None
            if self.output_path is None
            else str(Path(self.output_path).resolve()),
            "overwrite": self.overwrite,
            "resource_categories": [
                "fixture",
                "empirical_statistic",
                "monte_carlo_table",
            ],
            "expected_resource_count": _request_positive_integer(
                self.expected_resource_count,
                field="expected_resource_count",
            ),
            "expected_fixture_count": _request_positive_integer(
                self.expected_fixture_count,
                field="expected_fixture_count",
            ),
            "expected_empirical_statistic_count": _request_positive_integer(
                self.expected_empirical_statistic_count,
                field="expected_empirical_statistic_count",
            ),
            "expected_monte_carlo_table_count": _request_positive_integer(
                self.expected_monte_carlo_table_count,
                field="expected_monte_carlo_table_count",
            ),
        })


@dataclass(frozen=True)
class RegressionAdjustmentSupport:
    """Release-scope contract for adjusted regression paths.

    Declares the supported formula kinds, scope narrative, release-blocker
    status, and traceability anchors for a regression-adjusted estimation
    surface.

    Attributes
    ----------
    surface : str
        Public function name this contract governs.
    status : str
        Release-readiness status label.
    supported_formula_kinds : tuple of str
        Formula kinds supported (e.g. ``"trivial"``, ``"controls"``).
    supported_scope : str
        Narrative describing what is supported.
    release_blocker : bool
        Whether unsupported scope blocks release.
    requirement_ids : tuple of str
        Traceability requirement identifiers.
    unsupported_scope : tuple of str
        Narrative items explicitly not supported.
    paper_anchor : str
        Source-manuscript location reference.
    reference_anchor : str or None
        R-package reference implementation location.
    """
    surface: str
    status: str
    supported_formula_kinds: tuple[str, ...]
    supported_scope: str
    release_blocker: bool
    requirement_ids: tuple[str, ...]
    unsupported_scope: tuple[str, ...]
    paper_anchor: str
    reference_anchor: str | None

    def to_dict(self) -> dict[str, object]:
        """Serialize to a strict-JSON-safe dictionary for release audits."""
        return _json_safe_payload({
            "surface": self.surface,
            "status": self.status,
            "supported_formula_kinds": list(self.supported_formula_kinds),
            "supported_scope": self.supported_scope,
            "release_blocker": self.release_blocker,
            "requirement_ids": list(self.requirement_ids),
            "unsupported_scope": list(self.unsupported_scope),
            "paper_anchor": self.paper_anchor,
            "reference_anchor": self.reference_anchor,
        })


@dataclass(frozen=True)
class SharpNullDiagnosticsSupport:
    """Release-scope contract for sharp-null diagnostic schemas.

    Declares the diagnostic fields, row-level field schemas, scope,
    and traceability anchors for sharp-null test diagnostics.

    Attributes
    ----------
    surface : str
        Public function name this contract governs.
    status : str
        Release-readiness status label.
    supported_scope : str
        Narrative describing supported diagnostic scope.
    diagnostics_contract : str
        Narrative of what diagnostics are emitted.
    diagnostic_fields : tuple of str
        Top-level diagnostic dictionary keys.
    diagnostic_row_fields : tuple of (str, tuple of str)
        Row-level field schemas as (path, field_names) pairs.
    release_blocker : bool
        Whether unsupported scope blocks release.
    requirement_ids : tuple of str
        Traceability requirement identifiers.
    unsupported_scope : tuple of str
        Items explicitly not supported.
    paper_anchor : str
        Source-manuscript location reference.
    reference_anchor : str or None
        R-package reference implementation location.
    """
    surface: str
    status: str
    supported_scope: str
    diagnostics_contract: str
    diagnostic_fields: tuple[str, ...]
    diagnostic_row_fields: tuple[tuple[str, tuple[str, ...]], ...]
    release_blocker: bool
    requirement_ids: tuple[str, ...]
    unsupported_scope: tuple[str, ...]
    paper_anchor: str
    reference_anchor: str | None

    def to_dict(self) -> dict[str, object]:
        """Serialize to a strict-JSON-safe dictionary for release audits."""
        return _json_safe_payload({
            "surface": self.surface,
            "status": self.status,
            "supported_scope": self.supported_scope,
            "diagnostics_contract": self.diagnostics_contract,
            "diagnostic_fields": list(self.diagnostic_fields),
            "diagnostic_row_fields": {
                row_path: list(fields)
                for row_path, fields in self.diagnostic_row_fields
            },
            "release_blocker": self.release_blocker,
            "requirement_ids": list(self.requirement_ids),
            "unsupported_scope": list(self.unsupported_scope),
            "paper_anchor": self.paper_anchor,
            "reference_anchor": self.reference_anchor,
        })


@dataclass(frozen=True)
class PartialDensitySupport:
    """Release-scope contract for partial-density paths.

    Declares diagnostic fields, row schemas, scope, and traceability
    for the partial-density data and plotting surfaces.

    Attributes
    ----------
    surface : str
        Public function name this contract governs.
    status : str
        Release-readiness status label.
    supported_scope : str
        Narrative describing supported scope.
    diagnostics_contract : str
        Narrative of what diagnostics are emitted.
    diagnostic_fields : tuple of str
        Top-level diagnostic dictionary keys.
    diagnostic_row_fields : tuple of (str, tuple of str)
        Row-level field schemas.
    release_blocker : bool
        Whether unsupported scope blocks release.
    requirement_ids : tuple of str
        Traceability requirement identifiers.
    unsupported_scope : tuple of str
        Items explicitly not supported.
    paper_anchor : str
        Source-manuscript location reference.
    reference_anchor : str or None
        R-package reference implementation location.
    """
    surface: str
    status: str
    supported_scope: str
    diagnostics_contract: str
    diagnostic_fields: tuple[str, ...]
    diagnostic_row_fields: tuple[tuple[str, tuple[str, ...]], ...]
    release_blocker: bool
    requirement_ids: tuple[str, ...]
    unsupported_scope: tuple[str, ...]
    paper_anchor: str
    reference_anchor: str | None

    def to_dict(self) -> dict[str, object]:
        """Serialize to a strict-JSON-safe dictionary for release audits."""
        return _json_safe_payload({
            "surface": self.surface,
            "status": self.status,
            "supported_scope": self.supported_scope,
            "diagnostics_contract": self.diagnostics_contract,
            "diagnostic_fields": list(self.diagnostic_fields),
            "diagnostic_row_fields": {
                row_path: list(fields)
                for row_path, fields in self.diagnostic_row_fields
            },
            "release_blocker": self.release_blocker,
            "requirement_ids": list(self.requirement_ids),
            "unsupported_scope": list(self.unsupported_scope),
            "paper_anchor": self.paper_anchor,
            "reference_anchor": self.reference_anchor,
        })


@dataclass(frozen=True)
class CellCountDiagnosticsSupport:
    """Release-scope contract for cell-count diagnostics.

    Declares the support-grid and count contracts, scope, and
    traceability for cell-count diagnostic reporting.

    Attributes
    ----------
    surface : str
        Public function name this contract governs.
    status : str
        Release-readiness status label.
    support_grid_contract : str
        Narrative of the support-grid preconditions.
    count_contract : str
        Narrative of how counts are computed and reported.
    release_blocker : bool
        Whether unsupported scope blocks release.
    requirement_ids : tuple of str
        Traceability requirement identifiers.
    paper_anchor : str
        Source-manuscript location reference.
    reference_anchor : str or None
        R-package reference implementation location.
    """
    surface: str
    status: str
    support_grid_contract: str
    count_contract: str
    release_blocker: bool
    requirement_ids: tuple[str, ...]
    paper_anchor: str
    reference_anchor: str | None

    def to_dict(self) -> dict[str, object]:
        """Serialize to a strict-JSON-safe dictionary for release audits."""
        return _json_safe_payload({
            "surface": self.surface,
            "status": self.status,
            "support_grid_contract": self.support_grid_contract,
            "count_contract": self.count_contract,
            "release_blocker": self.release_blocker,
            "requirement_ids": list(self.requirement_ids),
            "paper_anchor": self.paper_anchor,
            "reference_anchor": self.reference_anchor,
        })


@dataclass(frozen=True)
class BoundsSupport:
    """Release-scope contract for bounds estimation paths.

    Declares the diagnostic fields, row schemas, scope, and
    traceability anchors for lower-bound, breakdown, and ADE bounds
    surfaces.

    Attributes
    ----------
    surface : str
        Public function name this contract governs.
    status : str
        Release-readiness status label.
    supported_scope : str
        Narrative describing supported scope.
    diagnostics_contract : str
        Narrative of what diagnostics are emitted.
    diagnostic_fields : tuple of str
        Top-level diagnostic dictionary keys.
    diagnostic_row_fields : tuple of (str, tuple of str)
        Row-level field schemas.
    release_blocker : bool
        Whether unsupported scope blocks release.
    requirement_ids : tuple of str
        Traceability requirement identifiers.
    unsupported_scope : tuple of str
        Items explicitly not supported.
    paper_anchor : str
        Source-manuscript location reference.
    reference_anchor : str or None
        R-package reference implementation location.
    """
    surface: str
    status: str
    supported_scope: str
    diagnostics_contract: str
    diagnostic_fields: tuple[str, ...]
    diagnostic_row_fields: tuple[tuple[str, tuple[str, ...]], ...]
    release_blocker: bool
    requirement_ids: tuple[str, ...]
    unsupported_scope: tuple[str, ...]
    paper_anchor: str
    reference_anchor: str | None

    def to_dict(self) -> dict[str, object]:
        """Serialize to a strict-JSON-safe dictionary for release audits."""
        return _json_safe_payload({
            "surface": self.surface,
            "status": self.status,
            "supported_scope": self.supported_scope,
            "diagnostics_contract": self.diagnostics_contract,
            "diagnostic_fields": list(self.diagnostic_fields),
            "diagnostic_row_fields": {
                row_path: list(fields)
                for row_path, fields in self.diagnostic_row_fields
            },
            "release_blocker": self.release_blocker,
            "requirement_ids": list(self.requirement_ids),
            "unsupported_scope": list(self.unsupported_scope),
            "paper_anchor": self.paper_anchor,
            "reference_anchor": self.reference_anchor,
        })


def bounds_support_contract() -> tuple[BoundsSupport, ...]:
    """Return the release-facing support scope for bounds paths."""

    paper_anchor = (
        "manuscript/sources/arxiv-2404.11739v3/draft.tex:230-259,837-904,906-933; "
        "always-taker affected-fraction lower bounds, ADE Lee-style trimming, "
        "and theta-min no-bite boundaries"
    )
    return (
        BoundsSupport(
            surface="lb_frac_affected",
            status="supported_scalar_vector_bounds",
            supported_scope=(
                "Scalar, nonbinary, and vector mediator lower bounds for single-group "
                "and pooled always-taker affected-fraction targets under ordered or "
                "elementwise monotonicity; binary supports use the ordered-monotone "
                "closed form, nonbinary/vector supports use the shared feasible-set "
                "linear-fractional program, ordered pandas Categorical scalar "
                "mediators use declared category order, unordered nonnumeric scalar supports "
                "are rejected before monotonicity constraints are built, vector mediator components must carry "
                "numeric or boolean elementwise order, and allow_min_defiers=True uses the exact "
                "minimum compatible defier cap rather than the R package's +1e-6 cap."
            ),
            diagnostics_contract=(
                "Diagnostics expose theta_kk_min_rows, objective_levels, "
                "top-level no_bite status, "
                "treatment normalization, mediator support, sample-size, cell-risk, "
                "defier-cap, theta-minimum, and active-restriction metadata, "
                "target_group_result for single-group targets, pooled.group_result_rows "
                "for pooled targets, general_lfp.group_result_rows and max_p_diff_rows "
                "plus max_p_diff_cell_rows for shared-LFP paths, "
                "positive_part_cell_rows for ordered-monotone closed-form group paths, "
                "paper inequality lhs/rhs/gap/violation diagnostics for both "
                "closed-form and shared-LFP group paths, "
                "type_share_rows and slack_rows for the returned "
                "optimizer solution, marginal_fit_rows for reconstructed D=0/D=1 "
                "mediator margins, slack-row inequality residuals for the paper "
                "positive-part constraints, defier_cap_rows for the relaxed-monotonicity "
                "cap constraint, objective contribution rows for numerator/denominator "
                "reconstruction, row-level objective-ratio reconstruction and gap checks, "
                "per-group paper inequality lhs/rhs/gap/violation diagnostics with "
                "all-group and objective-group maximum violation summaries, "
                "row-audit maximum residual summaries, "
                "per-row in_objective/objective_role markers, "
                "shared-LFP type_count and slack_count dimension diagnostics, "
                "Charnes-Cooper transformed_objective / scale_variable / "
                "denominator_normalization fields for finite-ratio solves, "
                "primal feasibility residuals for the returned original-space "
                "solution, and denominator-minimum no-bite certificates; result payloads are "
                "strict-JSON-safe without parsing tuple-keyed dictionaries; "
                "LowerBoundResult.to_frame() renders nested/vector boolean group display labels "
                "as 0/1 and HTML missing cells as not available for compact, reader-facing "
                "summaries while to_dict() preserves strict-JSON-safe payload labels."
            ),
            diagnostic_fields=(
                "theta_kk_min_rows",
                "objective_levels",
                "target_group_result",
                "positive_part_cell_rows",
                "paper_inequality_lhs",
                "paper_inequality_rhs",
                "paper_inequality_gap",
                "paper_inequality_violation",
                "active_restriction",
                "requested_num_y_bins",
                "applied_num_y_bins",
                "requested_max_defiers_share",
                "minimum_compatible_defiers_share",
                "actual_max_defiers_share",
                "original_treatment_levels",
                "normalized_treatment_levels",
                "treatment_support_normalization",
                "mediator_columns",
                "mediator_dimension",
                "mediator_levels",
                "ordered_mediator_levels",
                "ordered_categorical_mediator",
                "mediator_support_ordering",
                "binary_mediator",
                "vector_mediator",
                "y_levels",
                "n_obs_used",
                "min_cell_count",
                "min_cluster_count",
                "size_risk",
                "size_risk_threshold",
                "theta_kk_min",
                "theta_kk_min_by_group",
                "positive_part_partial_pmf_diff",
                "max_complier_share_to_group",
                "no_bite",
                "no_bite.flag",
                "no_bite.theta_kk_min",
                "no_bite.reason",
                "pooled",
                "pooled.group_results",
                "pooled.group_result_rows",
                "pooled.shared_feasible_set",
                "pooled.implementation_scope",
                "pooled.post_hoc_weighted_average",
                "pooled.denominator",
                "pooled.equivalence_basis",
                "pooled.paper_inequality_max_violation",
                "general_lfp",
                "general_lfp.type_count",
                "general_lfp.slack_count",
                "general_lfp.solver_status",
                "general_lfp.solution_basis",
                "general_lfp.denominator_minimum",
                "general_lfp.objective_levels",
                "general_lfp.target_group_result",
                "general_lfp.group_results",
                "general_lfp.group_result_rows",
                "general_lfp.max_p_diffs",
                "general_lfp.max_p_diff_rows",
                "general_lfp.max_p_diff_cell_rows",
                "general_lfp.type_share_rows",
                "general_lfp.slack_rows",
                "general_lfp.marginal_fit_rows",
                "general_lfp.defier_cap_rows",
                "general_lfp.type_share_sum",
                "general_lfp.objective_numerator",
                "general_lfp.objective_denominator",
                "general_lfp.objective_numerator_from_rows",
                "general_lfp.objective_denominator_from_rows",
                "general_lfp.objective_ratio_from_rows",
                "general_lfp.objective_ratio_from_rows_gap",
                "general_lfp.paper_inequality_max_violation",
                "general_lfp.objective_paper_inequality_max_violation",
                "general_lfp.marginal_fit_max_abs_difference",
                "general_lfp.slack_constraint_max_violation",
                "general_lfp.defier_cap_max_violation",
                "general_lfp.transformed_objective",
                "general_lfp.scale_variable",
                "general_lfp.denominator_normalization",
                "general_lfp.primal_eq_max_abs_residual",
                "general_lfp.primal_ub_max_violation",
                "general_lfp.objective_ratio_gap",
            ),
            diagnostic_row_fields=(
                (
                    "theta_kk_min_rows",
                    ("at_group", "theta_kk_min"),
                ),
                (
                    "target_group_result",
                    (
                        "at_group",
                        "lower_bound",
                        "theta_kk_min",
                        "positive_part_partial_pmf_diff",
                        "positive_part_cell_rows",
                        "max_complier_share_to_group",
                        "paper_inequality_lhs",
                        "paper_inequality_rhs",
                        "paper_inequality_gap",
                        "paper_inequality_violation",
                        "in_objective",
                        "objective_role",
                        "no_bite",
                    ),
                ),
                (
                    "positive_part_cell_rows",
                    (
                        "at_group",
                        "y_value",
                        "p_y_m_given_d1",
                        "p_y_m_given_d0",
                        "delta",
                        "positive_part_contribution",
                    ),
                ),
                (
                    "pooled.group_result_rows",
                    (
                        "at_group",
                        "lower_bound",
                        "theta_kk_min",
                        "positive_part_partial_pmf_diff",
                        "positive_part_cell_rows",
                        "max_complier_share_to_group",
                        "paper_inequality_lhs",
                        "paper_inequality_rhs",
                        "paper_inequality_gap",
                        "paper_inequality_violation",
                        "in_objective",
                        "objective_role",
                        "no_bite",
                    ),
                ),
                (
                    "general_lfp.target_group_result",
                    (
                        "at_group",
                        "lower_bound",
                        "theta_kk",
                        "numerator_contribution",
                        "positive_part_partial_pmf_diff",
                        "max_complier_share_to_group",
                        "paper_inequality_lhs",
                        "paper_inequality_rhs",
                        "paper_inequality_gap",
                        "paper_inequality_violation",
                        "solution_basis",
                        "in_objective",
                        "objective_role",
                        "no_bite",
                    ),
                ),
                (
                    "general_lfp.group_result_rows",
                    (
                        "at_group",
                        "lower_bound",
                        "theta_kk",
                        "numerator_contribution",
                        "positive_part_partial_pmf_diff",
                        "max_complier_share_to_group",
                        "paper_inequality_lhs",
                        "paper_inequality_rhs",
                        "paper_inequality_gap",
                        "paper_inequality_violation",
                        "solution_basis",
                        "in_objective",
                        "objective_role",
                        "no_bite",
                    ),
                ),
                (
                    "general_lfp.max_p_diff_rows",
                    (
                        "at_group",
                        "positive_part_partial_pmf_diff",
                    ),
                ),
                (
                    "general_lfp.max_p_diff_cell_rows",
                    (
                        "at_group",
                        "y_value",
                        "p_y_m_given_d1",
                        "p_y_m_given_d0",
                        "delta",
                        "positive_part_contribution",
                    ),
                ),
                (
                    "general_lfp.type_share_rows",
                    (
                        "m0",
                        "m1",
                        "type_share",
                        "is_defier",
                        "is_always_taker",
                        "in_objective_denominator",
                        "objective_denominator_contribution",
                    ),
                ),
                (
                    "general_lfp.slack_rows",
                    (
                        "at_group",
                        "slack",
                        "positive_part_partial_pmf_diff",
                        "max_complier_share_to_group",
                        "slack_constraint_residual",
                        "slack_constraint_violation",
                        "in_objective_numerator",
                        "objective_numerator_contribution",
                    ),
                ),
                (
                    "general_lfp.marginal_fit_rows",
                    (
                        "at_group",
                        "observed_p_m_given_d0",
                        "reconstructed_p_m_given_d0",
                        "d0_abs_difference",
                        "observed_p_m_given_d1",
                        "reconstructed_p_m_given_d1",
                        "d1_abs_difference",
                    ),
                ),
                (
                    "general_lfp.defier_cap_rows",
                    (
                        "requested_max_defiers_share",
                        "actual_defiers_share",
                        "defier_cap_residual",
                        "defier_cap_violation",
                        "binding",
                    ),
                ),
            ),
            release_blocker=False,
            requirement_ids=("SURF-01", "DIAG-01"),
            unsupported_scope=(
                "Relaxed-defier ADE bounds",
                "Inference for lower-bound point estimates",
            ),
            paper_anchor=paper_anchor,
            reference_anchor=(
                "packages/r/TestMechs/R/lb_frac_affected.R:318-347,366-405; "
                "packages/r/TestMechs/R/bounds_ade_ats.R:219-347"
            ),
        ),
        BoundsSupport(
            surface="breakdown_defier_share",
            status="supported_relaxed_monotonicity_diagnostic",
            supported_scope=(
                "Binary, nonbinary, and vector mediator diagnostic for the smallest "
                "defier-share cap at which the always-taker affected-fraction lower "
                "bound loses identifying bite; the search starts at the exact "
                "minimum-compatible defier share and returns positive infinity when "
                "the lower bound remains positive even under a unit defier cap; "
                "when reg_formula is supplied, the derived search consumes the same "
                "adjusted joint probability grid as lb_frac_affected and rejects "
                "invalid adjusted probabilities before minimum-compatible cap setup."
            ),
            diagnostics_contract=(
                "Diagnostics expose breakdown_status, breakdown_defier_share, "
                "breakdown_lower_bound_at_cap, breakdown_tolerance, iterations, "
                "bracket_rows, final breakdown_bracket_* precision fields, "
                "theta_kk_min_rows, minimum_compatible_defiers_share, and the same "
                "strict defier-cap contract used by lb_frac_affected; strict JSON "
                "payloads encode infinite breakdown shares with finite markers."
            ),
            diagnostic_fields=(
                "breakdown_status",
                "breakdown_defier_share",
                "breakdown_defier_share_is_finite",
                "breakdown_defier_share_nonfinite",
                "breakdown_lower_bound_at_cap",
                "breakdown_tolerance",
                "iterations",
                "max_iterations",
                "bracket_rows",
                "theta_kk_min_rows",
                "breakdown_bracket_lower_cap",
                "breakdown_bracket_upper_cap",
                "breakdown_bracket_width",
                "minimum_compatible_defiers_share",
                "requested_max_defiers_share",
                "actual_max_defiers_share",
                "defier_cap_contract",
            ),
            diagnostic_row_fields=(
                (
                    "bracket_rows",
                    ("role", "cap", "lower_bound"),
                ),
                (
                    "theta_kk_min_rows",
                    ("at_group", "theta_kk_min"),
                ),
            ),
            release_blocker=False,
            requirement_ids=("SURF-01", "DIAG-01"),
            unsupported_scope=(
                "Inference for the breakdown threshold",
                "ADE breakdown thresholds",
            ),
            paper_anchor=(
                "manuscript/sources/arxiv-2404.11739v3/draft.tex:183-185,906-933; "
                "bounded defier-share relaxations and theta-min lower-bound logic"
            ),
            reference_anchor=(
                "packages/r/TestMechs/R/lb_frac_affected.R:631-695; "
                "packages/r/TestMechs/man/breakdown_defier_share.Rd:1-45"
            ),
        ),
        BoundsSupport(
            surface="bounds_ade_ats",
            status="supported_scalar_vector_defier_cap_trimming",
            supported_scope=(
                "Scalar ordered-monotone and vector elementwise-monotone "
                "ADE bounds for requested always-taker groups using theta_kk_min "
                "and Lee-style lower/upper trimmed expectations on finite numeric "
                "outcomes on the raw outcome support, not the num_y_bins discretized "
                "partial-PMF support used by affected-fraction lower bounds; "
                "positive max_defiers_share values use the general theta_kk LP under "
                "the requested defier cap, and allow_min_defiers=True uses the exact "
                "minimum compatible cap rather than the R package's +1e-6 relaxation; "
                "vector mediator components must carry numeric or boolean "
                "elementwise order, ordered pandas Categorical scalar mediators use "
                "declared category order, and scalar/vector reg_formula paths consume "
                "the adjusted joint probability grid for trimming."
            ),
            diagnostics_contract=(
                "Diagnostics expose objective_levels, target_group_result, "
                "theta_kk_min, treatment-arm mediator masses, check_theta shares "
                "and the raw numeric outcome-support contract "
                "when identified, unsupported/no-bite states, and strict-JSON-safe "
                "null endpoints when theta_kk_min or treatment-arm target mass has "
                "no identifying bite; relaxed-defier paths additionally expose "
                "general_theta_lp type/slack dimensions, primal residuals, marginal-fit rows, "
                "defier-cap rows, slack rows, type-share rows, and theta row reconstruction; "
                "ADEBoundsResult.to_frame() renders nested/vector boolean "
                "group display labels as 0/1 and HTML missing cells as not available "
                "for compact, reader-facing summaries while to_dict() preserves "
                "strict-JSON-safe payload labels."
            ),
            diagnostic_fields=(
                "objective_levels",
                "target_group_result",
                "theta_kk_min",
                "theta_kk_min_rows",
                "theta_kk_min_by_group",
                "check_theta",
                "no_bite",
                "mediator_levels",
                "ordered_mediator_levels",
                "ordered_categorical_mediator",
                "mediator_support_ordering",
                "requested_max_defiers_share",
                "minimum_compatible_defiers_share",
                "actual_max_defiers_share",
                "defier_cap_contract",
                "outcome_support_contract",
                "general_theta_lp",
                "general_theta_lp.type_count",
                "general_theta_lp.slack_count",
                "general_theta_lp.theta_kk_min_rows",
                "general_theta_lp.type_share_rows",
                "general_theta_lp.defier_cap_rows",
                "general_theta_lp.marginal_fit_rows",
                "general_theta_lp.theta_kk_min",
                "general_theta_lp.primal_eq_max_abs_residual",
                "general_theta_lp.primal_ub_max_violation",
                "general_theta_lp.theta_kk_from_rows",
                "general_theta_lp.theta_kk_from_rows_gap",
                "general_theta_lp.marginal_fit_max_abs_difference",
                "general_theta_lp.slack_constraint_max_violation",
                "general_theta_lp.defier_cap_max_violation",
            ),
            diagnostic_row_fields=(
                (
                    "theta_kk_min_rows",
                    ("at_group", "theta_kk_min"),
                ),
                (
                    "target_group_result",
                    (
                        "at_group",
                        "lower_bound",
                        "upper_bound",
                        "theta_kk_min",
                        "p_mk_given_d1",
                        "p_mk_given_d0",
                        "in_objective",
                        "objective_role",
                        "no_bite",
                    ),
                ),
                (
                    "general_theta_lp.theta_kk_min_rows",
                    ("at_group", "theta_kk_min", "in_objective"),
                ),
                (
                    "general_theta_lp.type_share_rows",
                    (
                        "m0",
                        "m1",
                        "type_share",
                        "is_defier",
                        "is_always_taker",
                        "in_theta_objective",
                        "theta_objective_contribution",
                    ),
                ),
                (
                    "general_theta_lp.defier_cap_rows",
                    (
                        "requested_max_defiers_share",
                        "actual_defiers_share",
                        "defier_cap_residual",
                        "defier_cap_violation",
                        "binding",
                    ),
                ),
                (
                    "general_theta_lp.marginal_fit_rows",
                    (
                        "at_group",
                        "observed_p_m_given_d0",
                        "reconstructed_p_m_given_d0",
                        "d0_abs_difference",
                        "observed_p_m_given_d1",
                        "reconstructed_p_m_given_d1",
                        "d1_abs_difference",
                    ),
                ),
            ),
            release_blocker=False,
            requirement_ids=("ADJ-02", "ADJ-03", "SURF-01"),
            unsupported_scope=(),
            paper_anchor=paper_anchor,
            reference_anchor="packages/r/TestMechs/R/bounds_ade_ats.R:100-160,219-347,374-459",
        ),
    )


def bounds_support_frame() -> list[dict[str, object]]:
    """Return a serializable release-scope table for bounds paths."""

    return [contract.to_dict() for contract in bounds_support_contract()]


def bounds_diagnostic_schema_frame() -> list[dict[str, object]]:
    """Return one row per release-facing bounds diagnostic field."""

    rows: list[dict[str, object]] = []
    schema_order = 0
    for contract in bounds_support_contract():
        for field_order, path in enumerate(contract.diagnostic_fields):
            rows.append({
                "schema_key": f"{contract.surface}:{path}:diagnostic_path",
                "schema_order": schema_order,
                "surface": contract.surface,
                "diagnostic_path": path,
                "field": path.rsplit(".", maxsplit=1)[-1],
                "field_order": field_order,
                "record_kind": "diagnostic_path",
                "schema_kind": "diagnostic_path",
                "row_path": None,
                "row_field_order": None,
                "paper_anchor": contract.paper_anchor,
                "reference_anchor": contract.reference_anchor,
            })
            schema_order += 1
        for row_order, (row_path, fields) in enumerate(contract.diagnostic_row_fields):
            for row_field_order, field in enumerate(fields):
                diagnostic_path = f"{row_path}.{field}"
                rows.append({
                    "schema_key": (
                        f"{contract.surface}:{diagnostic_path}:diagnostic_row_field"
                    ),
                    "schema_order": schema_order,
                    "surface": contract.surface,
                    "diagnostic_path": diagnostic_path,
                    "field": field,
                    "field_order": row_order,
                    "record_kind": "diagnostic_row_field",
                    "schema_kind": "diagnostic_row_field",
                    "row_path": row_path,
                    "row_field_order": row_field_order,
                    "paper_anchor": contract.paper_anchor,
                    "reference_anchor": contract.reference_anchor,
                })
                schema_order += 1
    return _json_safe_payload(rows)


def regression_adjustment_support_contract() -> tuple[RegressionAdjustmentSupport, ...]:
    """Return the release-facing support scope for adjusted regression paths."""

    paper_anchor = (
        "manuscript/sources/arxiv-2404.11739v3/draft.tex:391-438,480-568; "
        "discrete probability moment-inequality contract with analytic variance "
        "for ARP/CS, plus non-experimental replacement of the probability "
        "vector by adjusted estimates"
    )
    return (
        RegressionAdjustmentSupport(
            surface="compute_adjusted_probabilities",
            status="supported_engine",
            supported_formula_kinds=REGRESSION_FORMULA_KINDS,
            supported_scope=(
                "Adjusted joint P(Y=y,M=m|D=d) grids for scalar and vector "
                "mediator probability contracts; D and Y columns must be distinct "
                "where an outcome grid is required, vector mediators are represented "
                "as deterministic tuple support values, mediator/outcome support "
                "is ordered deterministically while preserving labels, exactly "
                "two treatment support levels are normalized to the paper's 0/1 "
                "treatment scale with original levels exposed in diagnostics, and "
                "strict-JSON exports include probability_row_records with an explicit "
                "treatment field for each treatment-arm cell, while "
                "probability_grid_contract diagnostics expose treatment-arm mass "
                "totals, invalid probability counts, bounds-validity flags, and "
                "joint-to-mediator reconstruction rows before consuming bounds "
                "helpers validate the grid; "
                "formula RHS variables cannot reuse target outcome or mediator "
                "columns; unobserved categorical design levels are excluded from "
                "controls and fixed effects."
            ),
            release_blocker=False,
            requirement_ids=("ADJ-03",),
            unsupported_scope=(
                "Direct statistical inference claims without a consuming public API",
                "Adjusted probability influence functions for vector mediator sharp-null inference",
            ),
            paper_anchor=paper_anchor,
            reference_anchor="packages/r/TestMechs/R/test_sharp_null_binary_m.R:54-126",
        ),
        RegressionAdjustmentSupport(
            surface="compute_adjusted_mediator_masses",
            status="supported_engine",
            supported_formula_kinds=REGRESSION_FORMULA_KINDS,
            supported_scope=(
                "Adjusted P(M=m|D=d) mediator-mass grids for scalar and vector "
                "mediator probability contracts; vector mediators are represented "
                "as deterministic tuple support values, exactly two treatment "
                "support levels are normalized to the paper's 0/1 treatment scale "
                "with original levels exposed in diagnostics, strict-JSON exports "
                "include mediator_mass_row_records with an explicit treatment field, "
                "formula RHS variables "
                "cannot reuse target mediator columns, and the helper does not "
                "require an outcome grid."
            ),
            release_blocker=False,
            requirement_ids=("ADJ-03", "SURF-01"),
            unsupported_scope=(
                "Direct statistical inference claims without a consuming public API",
                "Outcome-grid or influence-function claims",
            ),
            paper_anchor=paper_anchor,
            reference_anchor=(
                "packages/r/TestMechs/R/lb_frac_affected.R:318-347,470-530; "
                "packages/r/TestMechs/R/bounds_ade_ats.R:127-155"
            ),
        ),
        RegressionAdjustmentSupport(
            surface="test_sharp_null",
            status="supported_public_runner",
            supported_formula_kinds=("trivial", "controls", "fixed_effects"),
            supported_scope=(
                "Binary-mediator CS analytic variance for reg_formula='~ treatment', "
                "controls, and one-way fixed effects using adjusted joint-probability "
                "treatment-coefficient influence functions with distinct D and Y columns; "
                "the relaxed pooled always-taker affected-fraction null uses the adjusted "
                "mediator-mass and joint-probability influence functions in the ordered "
                "nuisance moment vector; "
                "strict-JSON influence exports include influence_row_records with an "
                "explicit treatment field; cell-count and outcome-grid "
                "diagnostics use the regression complete-case sample."
            ),
            release_blocker=False,
            requirement_ids=("ADJ-01", "PMETH-01"),
            unsupported_scope=(
                "Adjusted nonbinary/vector sharp-null inference",
                "IV / IV+FE adjusted sharp-null analytic variance",
                "ARP with nontrivial regression adjustment",
            ),
            paper_anchor=paper_anchor,
            reference_anchor=(
                "packages/r/TestMechs/R/test_sharp_null_binary_m.R:87-183; "
                "packages/r/TestMechs/R/test_sharp_null.R:1004-1270"
            ),
        ),
        RegressionAdjustmentSupport(
            surface="lb_frac_affected",
            status="supported_scalar_vector_grid",
            supported_formula_kinds=REGRESSION_FORMULA_KINDS,
            supported_scope=(
                "Scalar and vector ordered-monotone lower bounds using the "
                "adjusted discrete joint probability grid, including the "
                "Baranov vector-regression-equivalence and combined-mediator "
                "paper-target contracts; adjusted pooled at_group=None targets "
                "use the binary ordered-monotone closed-form equivalence when "
                "there are two mediator levels and the shared feasible-set LFP "
                "for nonbinary/vector support; invalid adjusted joint probabilities "
                "fail before positive-part PMF differences are computed; when allow_min_defiers=True, Python "
                "uses the exact minimum compatible defier cap rather than the "
                "R package's +1e-6 relaxation."
            ),
            release_blocker=False,
            requirement_ids=("SURF-01",),
            unsupported_scope=(),
            paper_anchor=paper_anchor,
            reference_anchor=(
                "packages/r/TestMechs/R/lb_frac_affected.R:231-236,318-347,520-610; "
                "packages/r/TestMechs/tests/testthat/test-regression-equivalence.R:295-313"
            ),
        ),
        RegressionAdjustmentSupport(
            surface="breakdown_defier_share",
            status="supported_derived_lower_bound_diagnostic",
            supported_formula_kinds=REGRESSION_FORMULA_KINDS,
            supported_scope=(
                "Derived breakdown-defier diagnostic using the same adjusted "
                "joint probability grid and exact minimum-compatible defier cap "
                "setup as lb_frac_affected; invalid adjusted joint probabilities "
                "fail before the minimum-compatible cap query or binary-search "
                "lower-bound evaluations are computed."
            ),
            release_blocker=False,
            requirement_ids=("SURF-01", "DIAG-01"),
            unsupported_scope=(
                "Inference for the breakdown threshold",
                "ADE breakdown thresholds",
            ),
            paper_anchor=paper_anchor,
            reference_anchor=(
                "packages/r/TestMechs/R/lb_frac_affected.R:631-695; "
                "packages/r/TestMechs/man/breakdown_defier_share.Rd:1-45"
            ),
        ),
        RegressionAdjustmentSupport(
            surface="bounds_ade_ats",
            status="supported_scalar_vector_defier_cap_trimming",
            supported_formula_kinds=REGRESSION_FORMULA_KINDS,
            supported_scope=(
                "Scalar ordered-monotone and vector elementwise-monotone Lee-style "
                "ADE trimming from the adjusted joint probability grid, including "
                "positive defier caps via the general theta_kk LP; invalid "
                "adjusted joint probabilities fail before trimming instead of "
                "being clipped into a conditional outcome distribution."
            ),
            release_blocker=False,
            requirement_ids=("ADJ-02", "ADJ-03"),
            unsupported_scope=(),
            paper_anchor=paper_anchor,
            reference_anchor="packages/r/TestMechs/R/bounds_ade_ats.R:114-347",
        ),
    )


def regression_adjustment_support_frame() -> list[dict[str, object]]:
    """Return a serializable release-scope table for adjusted regression paths."""

    return [contract.to_dict() for contract in regression_adjustment_support_contract()]


def sharp_null_diagnostics_support_contract() -> tuple[SharpNullDiagnosticsSupport, ...]:
    """Return the release-facing diagnostic scope for sharp-null paths."""

    paper_anchor = (
        "manuscript/sources/arxiv-2404.11739v3/draft.tex:230-337; "
        "nu_k, sharp-null testable implications, and pooled always-taker "
        "affected-fraction relaxation"
    )
    return (
        SharpNullDiagnosticsSupport(
            surface="test_sharp_null",
            status="supported_public_runner_diagnostics",
            supported_scope=(
                "CS relaxed pooled always-taker affected-fraction diagnostics for "
                "unadjusted binary, ordered nonbinary, vector mediator paths, "
                "and adjusted binary controls/fixed-effects paths; constraint_rows "
                "bind each public diagnostic row to the ordered CS shape matrix "
                "and the full moment-inequality row consumed by the Cox-Shi "
                "nuisance solver."
            ),
            diagnostics_contract=(
                "diagnostics['relaxed_null'] reports the requested share, "
                "estimand name, pooled constraint label, iota parameter count, "
                "and strict-JSON constraint_rows for iota_k <= theta_kk plus "
                "sum_iota <= frac_ats_affected * sum_theta_kk with mediator "
                "labels, theta-pair metadata, shape-row, moment-row, "
                "theta-column, and iota-column indices."
            ),
            diagnostic_fields=(
                "relaxed_null",
                "relaxed_null.frac_ats_affected",
                "relaxed_null.estimand",
                "relaxed_null.constraint",
                "relaxed_null.iota_parameter_count",
                "relaxed_null.constraint_rows",
                "relaxed_null.paper_reference",
                "relaxed_null.reference_implementation",
            ),
            diagnostic_row_fields=(
                (
                    "relaxed_null.constraint_rows",
                    (
                        "row_type",
                        "constraint",
                        "mediator_index",
                        "mediator_level",
                        "mediator_indices",
                        "mediator_levels",
                        "theta_pair",
                        "theta_diagonal_pair_count",
                        "shape_row_index",
                        "moment_inequality_row_index",
                        "sharp_shape_row_index",
                        "theta_column_index",
                        "iota_column_index",
                        "theta_column_indices",
                        "iota_column_indices",
                        "theta_kk_coefficient",
                        "iota_coefficient",
                        "rhs",
                    ),
                ),
            ),
            release_blocker=False,
            requirement_ids=("PMETH-01",),
            unsupported_scope=(
                "Relaxed pooled always-taker affected-fraction diagnostics for ARP, FSST, or K",
                "Adjusted nonbinary/vector sharp-null inference",
            ),
            paper_anchor=paper_anchor,
            reference_anchor="packages/r/TestMechs/R/test_sharp_null.R:755-832",
        ),
        SharpNullDiagnosticsSupport(
            surface="test_sharp_null",
            status="supported_public_runner_defier_cap_diagnostics",
            supported_scope=(
                "Positive max_defiers_share diagnostics for unadjusted CS, ARP, "
                "and FSST ordered-nuisance sharp-null runners with binary, "
                "ordered nonbinary, or vector mediators; the diagnostic binds the "
                "public cap to the shape-row constraint consumed by the nuisance "
                "solver."
            ),
            diagnostics_contract=(
                "diagnostics['relaxed_monotonicity'] reports the requested cap, "
                "the defier-theta parameter count, the defier theta pairs, and a "
                "strict-JSON constraint row for sum_defier_theta <= "
                "max_defiers_share with theta-column and moment-row indices."
            ),
            diagnostic_fields=(
                "relaxed_monotonicity",
                "relaxed_monotonicity.requested_max_defiers_share",
                "relaxed_monotonicity.constraint",
                "relaxed_monotonicity.theta_parameter_count",
                "relaxed_monotonicity.monotone_theta_parameter_count",
                "relaxed_monotonicity.defier_theta_parameter_count",
                "relaxed_monotonicity.defier_theta_pairs",
                "relaxed_monotonicity.constraint_rows",
                "relaxed_monotonicity.paper_reference",
                "relaxed_monotonicity.reference_implementation",
            ),
            diagnostic_row_fields=(
                (
                    "relaxed_monotonicity.constraint_rows",
                    (
                        "row_type",
                        "constraint",
                        "requested_max_defiers_share",
                        "rhs",
                        "shape_rhs",
                        "shape_row_index",
                        "moment_inequality_row_index",
                        "theta_column_indices",
                        "defier_theta_pairs",
                    ),
                ),
            ),
            release_blocker=False,
            requirement_ids=("PMETH-01",),
            unsupported_scope=(
                "Defier-cap diagnostics for the K comparator",
                "Adjusted nonbinary/vector sharp-null inference",
            ),
            paper_anchor=(
                "manuscript/sources/arxiv-2404.11739v3/draft.tex:360-376; "
                "linear nuisance moment-inequality representation"
            ),
            reference_anchor="packages/r/TestMechs/R/test_sharp_null.R:787-889",
        ),
    )


def sharp_null_diagnostics_support_frame() -> list[dict[str, object]]:
    """Return a serializable release-scope table for sharp-null diagnostics."""

    return [
        contract.to_dict()
        for contract in sharp_null_diagnostics_support_contract()
    ]


def sharp_null_diagnostic_schema_frame() -> list[dict[str, object]]:
    """Return one row per release-facing sharp-null diagnostic field."""

    rows: list[dict[str, object]] = []
    schema_order = 0
    for contract in sharp_null_diagnostics_support_contract():
        for field_order, path in enumerate(contract.diagnostic_fields):
            rows.append({
                "schema_key": f"{contract.surface}:{path}:diagnostic_path",
                "schema_order": schema_order,
                "surface": contract.surface,
                "diagnostic_path": path,
                "field": path.rsplit(".", maxsplit=1)[-1],
                "field_order": field_order,
                "record_kind": "diagnostic_path",
                "schema_kind": "diagnostic_path",
                "row_path": None,
                "row_field_order": None,
                "paper_anchor": contract.paper_anchor,
                "reference_anchor": contract.reference_anchor,
            })
            schema_order += 1
        for row_order, (row_path, fields) in enumerate(contract.diagnostic_row_fields):
            for row_field_order, field in enumerate(fields):
                diagnostic_path = f"{row_path}.{field}"
                rows.append({
                    "schema_key": (
                        f"{contract.surface}:{diagnostic_path}:diagnostic_row_field"
                    ),
                    "schema_order": schema_order,
                    "surface": contract.surface,
                    "diagnostic_path": diagnostic_path,
                    "field": field,
                    "field_order": row_order,
                    "record_kind": "diagnostic_row_field",
                    "schema_kind": "diagnostic_row_field",
                    "row_path": row_path,
                    "row_field_order": row_field_order,
                    "paper_anchor": contract.paper_anchor,
                    "reference_anchor": contract.reference_anchor,
                })
                schema_order += 1
    return _json_safe_payload(rows)


def partial_density_support_contract() -> tuple[PartialDensitySupport, ...]:
    """Return the release-facing support scope for partial-density paths."""

    paper_anchor = (
        "manuscript/sources/arxiv-2404.11739v3/draft.tex:657-673; "
        "positive-part partial-density object for binary mediator applications"
    )
    return (
        PartialDensitySupport(
            surface="partial_density_data",
            status="supported_data_contract",
            supported_scope=(
                "Discrete partial-PMF records and continuous kernel-density records "
                "for binary treatment and binary mediator supports; d and y must name "
                "distinct scalar DataFrame columns, m must name one scalar binary "
                "mediator column that does not reuse the treatment or outcome column, invalid "
                "column specifications fail before pandas indexing, numeric binary support "
                "levels must be finite before normalization, adjusted reg_formula "
                "RHS variables cannot reuse the target outcome or mediator columns, and original "
                "support labels are preserved in diagnostics while "
                "exposing always-taker / never-taker orientation; adjusted discrete "
                "and continuous plot_nts paths use the same internal support "
                "normalization and original-label orientation as unadjusted paths; "
                "adjusted partial-density consumers reject non-finite, negative, "
                "above-one, or mass-total-inconsistent adjusted probability and "
                "mediator-mass grids before constructing partial PMFs or kernel densities."
            ),
            diagnostics_contract=(
                "Unadjusted discrete paths report deterministic Y support and partial-mass "
                "records that are stable to input row order; discrete adjusted reg_formula "
                "paths report outcome-grid diagnostics from the adjusted regression "
                "complete-case probability grid; unadjusted and adjusted continuous paths "
                "report explicit adjusted/reg_formula branch identity plus target masses, "
                "grid bounds, output_grid_points, bandwidths, and KDE distance calculations "
                "computed from a scaled finite spread for extreme finite outcome values, "
                "duplicate representable grid points removed when finite precision cannot "
                "honor every requested num_grid_points location, bandwidths floored at the "
                "finite output-grid resolution, clipped kernel tail distances "
                "for overflow-free density evaluation, and trapezoid integral errors; continuous adjusted "
                "reg_formula paths additionally report regression complete-case shape "
                "columns, dropped rows, and shape n; "
                "discrete paths report the paper positive-part object as "
                "positive_part_partial_pmf_diff plus positive_part_cell_rows, "
                "summing over displayed outcome levels and allowing zero-contribution "
                "rows for outcome levels outside the target-mediator support, "
                "non-numeric discrete outcome supports are displayed directly rather "
                "than routed through numeric auto-binning when num_y_bins is omitted, "
                "while continuous paths report positive_part_partial_density_integral "
                "using the output-grid trapezoid rule; positive_part_partial11_*_gap "
                "diagnostics report the residual between the first partial-density "
                "mass/integral and the positive-part object; "
                "to_dict payloads expose partial_density_row_records as a long-form "
                "strict-JSON view of partial11 and partial01 with target mediator and "
                "original treatment-arm identity plus partial-density finite/nonfinite "
                "markers; "
                "PartialDensityDataResult.to_frame(), str(result), and notebook HTML "
                "provide compact one-row summaries of branch identity, target role, "
                "record/support counts, finite-status markers, and the paper positive-part "
                "diagnostic to avoid dumping dense partial-density records in interactive use; "
                "nested regression diagnostics preserve the source treatment and mediator "
                "labels alongside normalized 0/1 support levels."
            ),
            diagnostic_fields=(
                "requested_num_y_bins",
                "applied_num_y_bins",
                "auto_num_y_bins",
                "plot_nts",
                "continuous_y",
                "adjusted",
                "reg_formula",
                "original_treatment_levels",
                "original_mediator_levels",
                "normalized_treatment_levels",
                "normalized_mediator_levels",
                "target_original_mediator_level",
                "target_normalized_mediator_level",
                "partial11_original_treatment_level",
                "partial01_original_treatment_level",
                "partial_density_target_role",
                "partial_density_orientation",
                "partial_density_default_labels",
                "treated_m1_count",
                "untreated_m1_count",
                "target_partial11_mass",
                "target_partial01_mass",
                "positive_part_partial_pmf_diff",
                "positive_part_partial11_mass_gap",
                "positive_part_support_rule",
                "positive_part_cell_rows",
                "positive_part_partial_density_integral",
                "positive_part_partial11_integral_gap",
                "positive_part_integral_rule",
                "partial_density_integral_rule",
                "num_grid_points",
                "output_grid_points",
                "grid_min",
                "grid_max",
                "treated_kernel_bandwidth",
                "untreated_kernel_bandwidth",
                "partial11_integral",
                "partial01_integral",
                "partial11_integral_absolute_error",
                "partial01_integral_absolute_error",
                "regression",
                "adjusted_complete_case_source_n_obs",
                "adjusted_complete_case_dropped_rows",
                "adjusted_probability_grid_points",
                "continuous_density_shape_contract",
                "continuous_density_shape_n_obs",
                "continuous_density_shape_columns",
                "continuous_density_shape_dropped_rows",
                "data_contract_only",
            ),
            diagnostic_row_fields=(
                (
                    "positive_part_cell_rows",
                    (
                        "y",
                        "partial11",
                        "partial01",
                        "delta",
                        "positive_part_contribution",
                    ),
                ),
                (
                    "partial_density_row_records",
                    (
                        "y",
                        "partial_density_role",
                        "partial_density",
                        "partial_density_is_finite",
                        "partial_density_nonfinite",
                        "target_original_mediator_level",
                        "target_normalized_mediator_level",
                        "original_treatment_level",
                        "partial_density_target_role",
                    ),
                ),
            ),
            release_blocker=False,
            requirement_ids=("PDENS-01", "ADJ-03"),
            unsupported_scope=(
                "Nonbinary mediator partial-density plotting",
                "Vector mediator partial-density plotting",
            ),
            paper_anchor=paper_anchor,
            reference_anchor="packages/r/TestMechs/R/partial_density_plot.R:24-284",
        ),
        PartialDensitySupport(
            surface="partial_density_plot",
            status="supported_plot_wrapper",
            supported_scope=(
                "Matplotlib rendering wrapper over partial_density_data() discrete and "
                "continuous records; d and y must name distinct scalar DataFrame columns, "
                "m must name one scalar binary mediator column that does not reuse "
                "the treatment or outcome column, invalid column specifications fail "
                "before pandas indexing, numeric binary support levels must be finite "
                "before normalization, adjusted reg_formula RHS variables cannot reuse "
                "the target outcome or mediator columns, and plot_nts label orientation is based on "
                "original treatment and mediator labels for unadjusted and adjusted "
                "discrete/continuous paths."
            ),
            diagnostics_contract=(
                "Plotting consumes the same data-contract diagnostics as "
                "partial_density_data(), including partial_density_row_records for "
                "the two rendered partial-density arms with finite/nonfinite markers; "
                "the returned Figure carries a strict-JSON copy of the consumed data contract "
                "under testmechs_partial_density_contract so saved or inspected plots can be "
                "traced back to the rendered values and diagnostics; a second strict-JSON render metadata "
                "payload under testmechs_partial_density_render_metadata records legend labels, "
                "legend_label_line_counts, legend_labels_truncated, caption, "
                "caption_line_count, caption_truncated, axis labels, title, figure size, "
                "legend anchor, positive_part_annotation anchor metadata, "
                "positive_part_shading collection count/alpha/path metadata for continuous plots, "
                "positive_part_bar_emphasis row metadata for discrete plots, "
                "layout_clearance renderer-measured legend, caption, axis-label, title, and positive-part annotation clearance, "
                "subplot margins, and layout branch; "
                "the publication-oriented default style uses fixed figure dimensions, "
                "colorblind-safe two-series colors, a y-axis reference grid, a "
                "nonnegative y-axis baseline, de-emphasized top/right spines, a titled "
                "legend placed below the plotting area with long labels wrapped, unbroken "
                "labels split, discrete legend labels capped to four wrapped lines with an ellipsis tail, "
                "extreme continuous legend labels capped to four wrapped lines with an ellipsis tail, "
                "long continuous legends given extra bottom margin so they do not overlap "
                "the x-axis label, snake_case outcome column names converted to readable words, "
                "readable binary boolean support labels rendered as 0/1, "
                "dense wrapped discrete outcome ticks capped to "
                "four lines with an ellipsis tail, and clearance from wrapped long outcome ticks "
                "or rotated dense-support outcome ticks to avoid covering or overflowing plotted evidence, "
                "compact context titles derived from partial_density_data() "
                "diagnostics naming the always-taker or never-taker target, a truncated "
                "mediator level for long category labels, discrete/continuous density type, and adjusted/unadjusted "
                "status, positive bar-height labels on compact discrete partial-PMF plots "
                "with six or fewer outcome levels while zero-height bars stay unlabeled, "
                "y-axis headroom that keeps compact bar labels inside the plotting area, "
                "positive-part edge emphasis for "
                "discrete partial11 bars where partial11 exceeds partial01, positive-part "
                "shading for continuous plots where partial11 exceeds partial01, a compact "
                "adaptive boxed positive-part value annotation with compact boxed positive-part value annotation styling that moves away from occupied edge data, centers when both plot edges are busy, drops continuous plots to bottom-center when the middle peak is also busy, and moves dense discrete plots above the right edge when all bar regions are occupied, "
                "wrapped discrete outcome tick labels with dense wrapped labels capped to four lines, optional publication captions wrapped below the plotting area with reserved bottom margin, additional reserved bottom margin when continuous long legends and captions are combined, and automatic rotation for dense "
                "discrete outcome support when tick labels do not already wrap; "
                "custom density labels are trimmed before rendering and must be non-blank and distinct; caption text is optional, whitespace-normalized, and must be non-blank when supplied; "
                "partial_density_plot() checks that Matplotlib is available before reading "
                "CSV inputs or constructing the data contract, so missing plot extras cannot "
                "be hidden by downstream data-path or regression errors; "
                "plotting fails before rendering if any plotted partial-density value "
                "is boolean, nonnumeric, negative, or nonfinite, and continuous outcome grid "
                "values must be numeric finite non-boolean values, so the nonnegative baseline "
                "cannot hide invalid density evidence; Matplotlib remains an optional plot extra."
            ),
            diagnostic_fields=(
                "plot_nts",
                "continuous_y",
                "adjusted",
                "reg_formula",
                "partial_density_default_labels",
                "partial_density_target_role",
                "target_partial11_mass",
                "target_partial01_mass",
                "positive_part_partial_pmf_diff",
                "positive_part_partial_density_integral",
                "partial_density_integral_rule",
            ),
            diagnostic_row_fields=(
                (
                    "partial_density_row_records",
                    (
                        "y",
                        "partial_density_role",
                        "partial_density",
                        "partial_density_is_finite",
                        "partial_density_nonfinite",
                    ),
                ),
            ),
            release_blocker=False,
            requirement_ids=("PDENS-01", "PKG-01"),
            unsupported_scope=(
                "Plot rendering without Matplotlib installed",
                "Nonbinary mediator partial-density plotting",
                "Vector mediator partial-density plotting",
            ),
            paper_anchor=paper_anchor,
            reference_anchor="packages/r/TestMechs/R/partial_density_plot.R:24-284",
        ),
    )


def partial_density_support_frame() -> list[dict[str, object]]:
    """Return a serializable release-scope table for partial-density paths."""

    return [contract.to_dict() for contract in partial_density_support_contract()]


def partial_density_diagnostic_schema_frame() -> list[dict[str, object]]:
    """Return a flattened diagnostic schema table for partial-density payloads."""

    rows: list[dict[str, object]] = []
    schema_order = 0
    for contract in partial_density_support_contract():
        for field_order, path in enumerate(contract.diagnostic_fields):
            rows.append({
                "schema_key": f"{contract.surface}:diagnostic:{path}",
                "schema_order": schema_order,
                "surface": contract.surface,
                "schema_kind": "diagnostic_field",
                "record_kind": "diagnostic_field",
                "diagnostic_path": path,
                "row_path": None,
                "field": None,
                "field_order": field_order,
                "status": contract.status,
                "release_blocker": contract.release_blocker,
                "paper_anchor": contract.paper_anchor,
                "reference_anchor": contract.reference_anchor,
            })
            schema_order += 1
        for row_order, (row_path, fields) in enumerate(contract.diagnostic_row_fields):
            for field_order, field in enumerate(fields):
                rows.append({
                    "schema_key": f"{contract.surface}:row:{row_path}:{field}",
                    "schema_order": schema_order,
                    "surface": contract.surface,
                    "schema_kind": "diagnostic_row_field",
                    "record_kind": "diagnostic_row_field",
                    "diagnostic_path": row_path,
                    "row_path": row_path,
                    "field": field,
                    "field_order": field_order,
                    "row_order": row_order,
                    "status": contract.status,
                    "release_blocker": contract.release_blocker,
                    "paper_anchor": contract.paper_anchor,
                    "reference_anchor": contract.reference_anchor,
                })
                schema_order += 1
    return _json_safe_payload(rows)


def cell_count_diagnostics_support_contract() -> tuple[CellCountDiagnosticsSupport, ...]:
    """Return the release-facing support scope for cell-count diagnostics."""

    paper_anchor = (
        "manuscript/sources/arxiv-2404.11739v3/draft.tex:1094-1156; cells are support points "
        "of (D,M,Ydisc), with independent-cluster counts summarized by the "
        "median across cells and compared to the paper's heuristic threshold"
    )
    return (
        CellCountDiagnosticsSupport(
            surface="build_cell_count_diagnostics",
            status="supported_public_diagnostic",
            support_grid_contract=(
                "Requires distinct D, M, and Y_disc role columns and a nonempty "
                "analysis sample with non-missing D, M, and Y_disc group labels before forming the full observed "
                "(D,M,Y_disc) support grid; categorical columns use observed "
                "categories only, mixed labels use deterministic type-aware "
                "ordering, and y_levels follows the same support order as the "
                "cell-count rows."
            ),
            count_contract=(
                "Realized-empty support cells are retained with zero observation "
                "and cluster counts and exposed as row-level empty/small cell "
                "records; size_risk compares the minimum observation or "
                "independent-cluster count to the requested threshold."
            ),
            release_blocker=False,
            requirement_ids=("DIAG-01", "PMETH-01"),
            paper_anchor=paper_anchor,
            reference_anchor="packages/r/TestMechs/R/test_sharp_null.R:1-140",
        ),
        CellCountDiagnosticsSupport(
            surface="paper_monte_carlo_reproduction_report",
            status="supported_monte_carlo_diagnostic",
            support_grid_contract=(
                "Monte Carlo draw diagnostics use the same nonempty, non-missing "
                "group-label precondition and full observed (D,M,Y_disc) "
                "support-grid semantics, including observed-only categorical "
                "levels and deterministic mixed-label ordering."
            ),
            count_contract=(
                "Median observation and independent-cluster counts are computed "
                "over the full grid with realized-empty cells filled as zero, "
                "matching the paper's independent-cell-count reporting target."
            ),
            release_blocker=False,
            requirement_ids=("MC-01", "DIAG-01"),
            paper_anchor=paper_anchor,
            reference_anchor=None,
        ),
    )


def cell_count_diagnostics_support_frame() -> list[dict[str, object]]:
    """Return a serializable release-scope table for cell-count diagnostics."""

    return [
        contract.to_dict() for contract in cell_count_diagnostics_support_contract()
    ]
