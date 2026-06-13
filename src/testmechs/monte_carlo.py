"""Monte Carlo simulation helpers for Testing Mechanisms.

This module contains optional simulation runners plus developer-facing
infrastructure for future manuscript-budget Monte Carlo evidence. The current
JSS article does not use this module to support size, power, performance,
method-paper Supplementary Table S1, or full method-paper table-reproduction
claims.

The user-facing surface is the small family of simulation runners such as
:func:`run_binary_cs_monte_carlo`. Suite scheduling, chunk persistence, archive
continuation, and evidence summarization helpers are retained for future
fixed-seed evidence builds and should not be cited as current article evidence
until the manuscript replication outputs record complete accepted coverage.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from importlib.resources import files
import glob
import hashlib
import json
import math
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ._json_io import write_strict_json_atomic as _write_monte_carlo_json_atomic
from .preprocess import (
    discretize_y,
    normalize_binary_support,
    ordered_binary_support_levels,
    remove_missing_from_df,
)
from .results import _is_scalar_missing, _reject_nonfinite_json_numbers
from .sharp_null import test_sharp_null


_PAPER_ACCEPTANCE_ABSOLUTE_TOLERANCE = 0.025
_PAPER_ACCEPTANCE_Z_TOLERANCE = 2.0
_PAPER_ACCEPTANCE_REPLICATIONS = 500
_PAPER_BOOTSTRAP_REPLICATIONS = 500


def _meets_paper_replication_budget(replications: int) -> bool:
    return int(replications) >= _PAPER_ACCEPTANCE_REPLICATIONS


def _meets_paper_bootstrap_budget(replications: int | None) -> bool:
    return (
        replications is not None
        and int(replications) >= _PAPER_BOOTSTRAP_REPLICATIONS
    )


@dataclass(frozen=True)
class MonteCarloResultRow:
    """Paper Monte Carlo rejection-rate row parsed from the LaTeX tables."""

    table: str
    panel: str
    design: str
    mediator: str
    clusters: int | None
    bins: int | None
    t: float
    bar_nu_lb: float
    rejection_rates: dict[str, float]

    @property
    def is_null_size_row(self) -> bool:
        return abs(self.t) <= 1e-12


@dataclass(frozen=True)
class ClusterCellCount:
    """Median independent-cluster count for one paper Monte Carlo cell design."""

    table: str
    panel: str
    design: str
    mediator: str
    clusters: int
    bins: int
    t: float
    median_independent_clusters_per_cell: float
    size_risk_threshold: float = 15.0

    @property
    def size_risk(self) -> bool:
        return self.median_independent_clusters_per_cell < self.size_risk_threshold


@dataclass(frozen=True)
class MonteCarloCellCountHeuristic:
    """Paper bin-size heuristic summarized from clustered Monte Carlo cell counts."""

    design: str
    mediator: str
    clusters: int
    method: str
    nominal_alpha: float
    tolerance: float
    size_risk_threshold: float
    min_median_independent_clusters_per_cell_by_bins: dict[int, float]
    null_rejection_rate_by_bins: dict[int, float]

    @property
    def safe_bins(self) -> tuple[int, ...]:
        return tuple(
            bins
            for bins, value in self.min_median_independent_clusters_per_cell_by_bins.items()
            if value >= self.size_risk_threshold
        )

    @property
    def risky_bins(self) -> tuple[int, ...]:
        return tuple(
            bins
            for bins, value in self.min_median_independent_clusters_per_cell_by_bins.items()
            if value < self.size_risk_threshold
        )

    @property
    def recommended_max_bins(self) -> int | None:
        if not self.safe_bins:
            return None
        return max(self.safe_bins)

    @property
    def size_distortion_bins(self) -> tuple[int, ...]:
        return tuple(
            bins
            for bins, rate in self.null_rejection_rate_by_bins.items()
            if rate - self.nominal_alpha > self.tolerance
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "design": self.design,
            "mediator": self.mediator,
            "clusters": self.clusters,
            "method": self.method,
            "nominal_alpha": self.nominal_alpha,
            "tolerance": self.tolerance,
            "size_risk_threshold": self.size_risk_threshold,
            "min_median_independent_clusters_per_cell_by_bins": dict(
                self.min_median_independent_clusters_per_cell_by_bins
            ),
            "null_rejection_rate_by_bins": dict(self.null_rejection_rate_by_bins),
            "safe_bins": self.safe_bins,
            "risky_bins": self.risky_bins,
            "recommended_max_bins": self.recommended_max_bins,
            "size_distortion_bins": self.size_distortion_bins,
        }

    def to_frame(self) -> pd.DataFrame:
        """Return row-level bin guidance for this paper cell-count heuristic."""

        rows: list[dict[str, Any]] = []
        for bins in sorted(self.min_median_independent_clusters_per_cell_by_bins):
            min_cell_count = self.min_median_independent_clusters_per_cell_by_bins[bins]
            null_rejection_rate = self.null_rejection_rate_by_bins.get(bins)
            cell_count_size_risk = min_cell_count < self.size_risk_threshold
            size_distortion = (
                None
                if null_rejection_rate is None
                else null_rejection_rate - self.nominal_alpha > self.tolerance
            )
            rows.append(
                {
                    "design": self.design,
                    "mediator": self.mediator,
                    "clusters": self.clusters,
                    "method": self.method,
                    "bins": bins,
                    "min_median_independent_clusters_per_cell": min_cell_count,
                    "size_risk_threshold": self.size_risk_threshold,
                    "cell_count_size_risk": cell_count_size_risk,
                    "recommended_by_cell_count": not cell_count_size_risk,
                    "recommended_max_bins": self.recommended_max_bins,
                    "null_rejection_rate": null_rejection_rate,
                    "nominal_alpha": self.nominal_alpha,
                    "tolerance": self.tolerance,
                    "size_distortion": size_distortion,
                    "bin_policy": (
                        "within_cell_count_heuristic"
                        if not cell_count_size_risk
                        else "below_cell_count_heuristic"
                    ),
                    "paper_rule": "at least 15 independent observations per cell",
                }
            )
        return _json_safe_export_frame(rows)


@dataclass(frozen=True)
class MonteCarloSizeDiagnostic:
    """Size-control diagnostic for one paper Monte Carlo null row and method."""

    row: MonteCarloResultRow
    method: str
    nominal_alpha: float
    tolerance: float
    cluster_cell_count: ClusterCellCount | None

    @property
    def table(self) -> str:
        return self.row.table

    @property
    def panel(self) -> str:
        return self.row.panel

    @property
    def design(self) -> str:
        return self.row.design

    @property
    def mediator(self) -> str:
        return self.row.mediator

    @property
    def clusters(self) -> int | None:
        return self.row.clusters

    @property
    def bins(self) -> int | None:
        return self.row.bins

    @property
    def rejection_rate(self) -> float:
        return self.row.rejection_rates[self.method]

    @property
    def excess_rejection_rate(self) -> float:
        return self.rejection_rate - self.nominal_alpha

    @property
    def size_distortion(self) -> bool:
        return self.excess_rejection_rate > self.tolerance

    @property
    def cell_count_size_risk(self) -> bool:
        return (
            _method_uses_discretized_outcome(self.method)
            and self.cluster_cell_count is not None
            and self.cluster_cell_count.size_risk
        )

    @property
    def needs_attention(self) -> bool:
        return self.size_distortion or self.cell_count_size_risk


@dataclass(frozen=True)
class MonteCarloMethodGuidance:
    """Paper-level method-choice guidance joined to the Monte Carlo contract."""

    method: str
    paper_order_index: int
    paper_role: str
    paper_recommendation: str
    python_execution_status: str
    paper_default: bool
    small_cluster_size_control_alternative: bool
    large_independent_sample_power_candidate: bool
    binary_mediator_comparator: bool
    uses_discretized_outcome: bool
    method_summary: dict[str, Any]

    @property
    def python_executable(self) -> bool:
        return self.python_execution_status != "paper_contract_only"

    @property
    def needs_size_caution(self) -> bool:
        return int(self.method_summary["attention_size_rows"]) > 0

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_export_payload({
            "method": self.method,
            "paper_order_index": self.paper_order_index,
            "paper_role": self.paper_role,
            "paper_recommendation": self.paper_recommendation,
            "python_execution_status": self.python_execution_status,
            "python_executable": self.python_executable,
            "paper_default": self.paper_default,
            "small_cluster_size_control_alternative": self.small_cluster_size_control_alternative,
            "large_independent_sample_power_candidate": self.large_independent_sample_power_candidate,
            "binary_mediator_comparator": self.binary_mediator_comparator,
            "uses_discretized_outcome": self.uses_discretized_outcome,
            "needs_size_caution": self.needs_size_caution,
            **self.method_summary,
        })


@dataclass(frozen=True)
class MonteCarloMethodExecutionContract:
    """Paper execution protocol for one Monte Carlo inference method."""

    method: str
    paper_order_index: int
    paper_role: str
    paper_recommendation: str
    paper_default: bool
    uses_discretized_outcome: bool
    supports_binary_mediator: bool
    supports_nonbinary_mediator: bool
    outcome_contract: str
    variance_estimator: str
    variance_contract: str
    bootstrap_required: bool
    bootstrap_unit_modes: tuple[str, ...]
    bootstrap_unit_contract: str
    tuning_contract: str
    nominal_alpha: float
    paper_replications: int
    paper_reported_rows: int
    python_executable_reported_rows: int
    python_blocked_reported_rows: int
    python_execution_status: str
    python_executable: bool
    paper_contract_only: bool
    next_action: str
    method_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_export_payload({
            "method": self.method,
            "paper_order_index": self.paper_order_index,
            "paper_role": self.paper_role,
            "paper_recommendation": self.paper_recommendation,
            "paper_default": self.paper_default,
            "uses_discretized_outcome": self.uses_discretized_outcome,
            "supports_binary_mediator": self.supports_binary_mediator,
            "supports_nonbinary_mediator": self.supports_nonbinary_mediator,
            "outcome_contract": self.outcome_contract,
            "variance_estimator": self.variance_estimator,
            "variance_contract": self.variance_contract,
            "bootstrap_required": self.bootstrap_required,
            "bootstrap_unit_modes": self.bootstrap_unit_modes,
            "bootstrap_unit_contract": self.bootstrap_unit_contract,
            "tuning_contract": self.tuning_contract,
            "nominal_alpha": self.nominal_alpha,
            "paper_replications": self.paper_replications,
            "paper_reported_rows": self.paper_reported_rows,
            "python_executable_reported_rows": self.python_executable_reported_rows,
            "python_blocked_reported_rows": self.python_blocked_reported_rows,
            "python_execution_status": self.python_execution_status,
            "python_executable": self.python_executable,
            "paper_contract_only": self.paper_contract_only,
            "next_action": self.next_action,
            "method_summary": dict(self.method_summary),
        })


@dataclass(frozen=True)
class MonteCarloBenchmarkCell:
    """Executable benchmark cell drawn from one paper Monte Carlo row."""

    table: str
    panel: str
    design: str
    mediator: str
    clusters: int | None
    bins: int | None
    t: float
    method: str
    bar_nu_lb: float
    target_rejection_rate: float
    is_null_size_row: bool
    cluster_cell_count: ClusterCellCount | None = None
    paper_row_index: int | None = None
    benchmark_row_index: int | None = None

    @classmethod
    def from_result_row(
        cls,
        row: MonteCarloResultRow,
        *,
        method: str,
        cluster_cell_count: ClusterCellCount | None,
        paper_row_index: int | None = None,
        benchmark_row_index: int | None = None,
    ) -> "MonteCarloBenchmarkCell":
        if method not in row.rejection_rates:
            raise KeyError(
                f"Monte Carlo row table={row.table!r}, design={row.design!r}, "
                f"mediator={row.mediator!r}, clusters={row.clusters!r}, bins={row.bins!r}, "
                f"t={row.t!r} does not expose method {method!r}."
            )
        return cls(
            table=row.table,
            panel=row.panel,
            design=row.design,
            mediator=row.mediator,
            clusters=row.clusters,
            bins=row.bins,
            t=row.t,
            method=method,
            bar_nu_lb=row.bar_nu_lb,
            target_rejection_rate=row.rejection_rates[method],
            is_null_size_row=row.is_null_size_row,
            cluster_cell_count=cluster_cell_count,
            paper_row_index=paper_row_index,
            benchmark_row_index=benchmark_row_index,
        )

    @property
    def requires_cluster_resampling(self) -> bool:
        return self.clusters is not None

    @property
    def cell_count_size_risk(self) -> bool:
        return self.cluster_cell_count is not None and self.cluster_cell_count.size_risk

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "panel": self.panel,
            "design": self.design,
            "mediator": self.mediator,
            "clusters": self.clusters,
            "bins": self.bins,
            "t": self.t,
            "method": self.method,
            "bar_nu_lb": self.bar_nu_lb,
            "target_rejection_rate": self.target_rejection_rate,
            "size_row": self.is_null_size_row,
            "requires_cluster_resampling": self.requires_cluster_resampling,
            "cell_count_size_risk": self.cell_count_size_risk,
            "median_independent_clusters_per_cell": (
                None
                if self.cluster_cell_count is None
                else self.cluster_cell_count.median_independent_clusters_per_cell
            ),
        }


@dataclass(frozen=True)
class MonteCarloBlockedBenchmarkRow:
    """Paper Monte Carlo row that is part of a plan but not executable yet."""

    row: MonteCarloResultRow
    method: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        row = self.row
        return {
            "table": row.table,
            "panel": row.panel,
            "design": row.design,
            "mediator": row.mediator,
            "clusters": row.clusters,
            "bins": row.bins,
            "t": row.t,
            "method": self.method,
            "bar_nu_lb": row.bar_nu_lb,
            "target_rejection_rate": row.rejection_rates.get(self.method),
            "size_row": row.is_null_size_row,
            "blocked_reason": self.reason,
        }


@dataclass(frozen=True)
class MonteCarloBenchmarkDataSourceDiagnostic:
    """Preflight status for binding one paper design to observed data."""

    design: str
    executable_rows: int
    requires_cluster_resampling: bool
    analysis_frame_columns: tuple[str, ...]
    d: str | None
    m: str | None
    y: str | None
    rows: int | None
    complete_case_rows: int | None
    treatment_levels: tuple[Any, ...]
    mediator_levels: tuple[Any, ...]
    control_rows: int | None
    treated_rows: int | None
    cluster: str | None
    source_clusters: int | None
    control_source_clusters: int | None
    treated_source_clusters: int | None
    arm_fixed_source_clusters: bool | None
    expected_complete_case_rows: int | None = None
    expected_control_rows: int | None = None
    expected_treated_rows: int | None = None
    expected_source_clusters: int | None = None
    expected_control_source_clusters: int | None = None
    expected_treated_source_clusters: int | None = None
    outcome_level_count: int | None = None
    outcome_binary: bool | None = None
    unbinned_outcome_binary_required: bool = False
    blocking_reasons: tuple[str, ...] = ()
    data_source_key: str | None = None

    @property
    def ready(self) -> bool:
        return not self.blocking_reasons

    def support_summary(self) -> dict[str, Any]:
        """Return compact support-shape evidence for empirical-mixture preflight."""

        return _json_safe_export_payload({
            "design": self.design,
            "treatment_level_count": len(self.treatment_levels),
            "mediator_level_count": len(self.mediator_levels),
            "outcome_level_count": self.outcome_level_count,
            "treatment_binary": len(self.treatment_levels) == 2,
            "mediator_binary": len(self.mediator_levels) == 2,
            "outcome_binary": self.outcome_binary,
            "unbinned_outcome_binary_required": self.unbinned_outcome_binary_required,
            "ready": self.ready,
            "blocking_reasons": self.blocking_reasons,
            "data_source_key": self.data_source_key,
        })

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_export_payload({
            "design": self.design,
            "executable_rows": self.executable_rows,
            "requires_cluster_resampling": self.requires_cluster_resampling,
            "analysis_frame_columns": self.analysis_frame_columns,
            "d": self.d,
            "m": self.m,
            "y": self.y,
            "rows": self.rows,
            "complete_case_rows": self.complete_case_rows,
            "treatment_levels": tuple(_json_safe_diagnostic_value(value) for value in self.treatment_levels),
            "mediator_levels": tuple(_json_safe_diagnostic_value(value) for value in self.mediator_levels),
            "control_rows": self.control_rows,
            "treated_rows": self.treated_rows,
            "cluster": self.cluster,
            "source_clusters": self.source_clusters,
            "control_source_clusters": self.control_source_clusters,
            "treated_source_clusters": self.treated_source_clusters,
            "arm_fixed_source_clusters": self.arm_fixed_source_clusters,
            "expected_complete_case_rows": self.expected_complete_case_rows,
            "expected_control_rows": self.expected_control_rows,
            "expected_treated_rows": self.expected_treated_rows,
            "expected_source_clusters": self.expected_source_clusters,
            "expected_control_source_clusters": self.expected_control_source_clusters,
            "expected_treated_source_clusters": self.expected_treated_source_clusters,
            "outcome_level_count": self.outcome_level_count,
            "outcome_binary": self.outcome_binary,
            "unbinned_outcome_binary_required": self.unbinned_outcome_binary_required,
            "ready": self.ready,
            "blocking_reasons": self.blocking_reasons,
            "data_source_key": self.data_source_key,
        })


@dataclass(frozen=True)
class MonteCarloBenchmarkPlan:
    """Machine-checkable plan for executing a paper Monte Carlo benchmark slice."""

    method: str
    replications: int
    executable_cells: tuple[MonteCarloBenchmarkCell, ...]
    blocked_rows: tuple[MonteCarloBlockedBenchmarkRow, ...]
    paper_result_rows: int | None = None

    def __post_init__(self) -> None:
        if self.replications <= 0:
            raise ValueError("replications must be positive.")

    @property
    def covered_result_rows(self) -> int:
        return len(self.executable_cells) + len(self.blocked_rows)

    @property
    def paper_coverage_known(self) -> bool:
        return self.paper_result_rows is not None

    @property
    def paper_coverage_shortfall_rows(self) -> int | None:
        if self.paper_result_rows is None:
            return None
        return max(0, int(self.paper_result_rows) - self.covered_result_rows)

    @property
    def paper_coverage_complete(self) -> bool:
        return self.paper_result_rows is not None and self.paper_coverage_shortfall_rows == 0

    def paper_coverage_summary(self) -> dict[str, Any]:
        return {
            "paper_coverage_known": self.paper_coverage_known,
            "paper_result_rows": self.paper_result_rows,
            "covered_result_rows": self.covered_result_rows,
            "paper_coverage_shortfall_rows": self.paper_coverage_shortfall_rows,
            "paper_coverage_complete": self.paper_coverage_complete,
        }

    @property
    def planned_draws(self) -> int:
        return len(self.executable_cells) * self.replications

    @property
    def executable_designs(self) -> tuple[str, ...]:
        """Return paper designs that require empirical source data."""

        return tuple(dict.fromkeys(cell.design for cell in self.executable_cells))

    @property
    def executable_data_source_keys(self) -> tuple[str, ...]:
        """Return empirical source keys required by this plan."""

        return tuple(dict.fromkeys(_benchmark_data_source_key(cell) for cell in self.executable_cells))

    def executable_cell_specs(self) -> list[dict[str, Any]]:
        return [cell.to_dict() for cell in self.executable_cells]

    def summary(self) -> dict[str, Any]:
        blocked_reasons: dict[str, int] = {}
        for blocked in self.blocked_rows:
            blocked_reasons[blocked.reason] = blocked_reasons.get(blocked.reason, 0) + 1

        summary = {
            "method": self.method,
            "total_result_rows": len(self.executable_cells) + len(self.blocked_rows),
            "executable_rows": len(self.executable_cells),
            "blocked_rows": len(self.blocked_rows),
            "executable_designs": self.executable_designs,
            "default_replications": self.replications,
            "planned_draws": self.planned_draws,
            "size_rows": int(sum(cell.is_null_size_row for cell in self.executable_cells)),
            "power_rows": int(sum(not cell.is_null_size_row for cell in self.executable_cells)),
            "clustered_rows": int(sum(cell.requires_cluster_resampling for cell in self.executable_cells)),
            "cell_count_size_risk_rows": int(sum(cell.cell_count_size_risk for cell in self.executable_cells)),
            "blocked_reasons": blocked_reasons,
        }
        summary.update(_benchmark_target_precision_summary(self.executable_cells, replications=self.replications))
        return summary

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_export_payload({
            "summary": self.summary(),
            "paper_coverage": self.paper_coverage_summary(),
            "executable_cells": self.executable_cell_specs(),
            "blocked_rows": [blocked.to_dict() for blocked in self.blocked_rows],
        })

    def to_frame(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for cell in self.executable_cells:
            rows.append(
                {
                    **cell.to_dict(),
                    "status": "executable",
                    "blocked_reason": None,
                    "planned_replications": self.replications,
                    "planned_draws": self.replications,
                }
            )
        for blocked in self.blocked_rows:
            rows.append(
                {
                    **blocked.to_dict(),
                    "status": "blocked",
                    "planned_replications": 0,
                    "planned_draws": 0,
                }
            )
        return _json_safe_export_frame(rows)

    def target_precision_frame(
        self,
        *,
        absolute_tolerance: float | None = 0.025,
        z_tolerance: float | None = 2.0,
    ) -> pd.DataFrame:
        """Return row-level target rejection-rate precision for the plan budget."""

        _validate_target_precision_tolerances(
            absolute_tolerance=absolute_tolerance,
            z_tolerance=z_tolerance,
        )
        return _json_safe_export_frame(
            _target_precision_rows_from_plan_cells(
                self.executable_cells,
                self.blocked_rows,
                replications=self.replications,
                absolute_tolerance=absolute_tolerance,
                z_tolerance=z_tolerance,
            )
        )

    def cell_count_policy_frame(self) -> pd.DataFrame:
        """Return row-level paper cell-count bin policy for the plan."""

        return _json_safe_export_frame(
            _cell_count_policy_rows_from_plan_cells(
                self.executable_cells,
                self.blocked_rows,
                replications=self.replications,
            )
        )

    def cell_count_policy_summary(self) -> dict[str, Any]:
        """Return compact paper cell-count bin-policy counts for the plan."""

        return _cell_count_policy_summary_from_frame(self.cell_count_policy_frame())

    def data_source_diagnostics(
        self,
        data_sources: dict[str, "BinaryEmpiricalMixtureBenchmarkDataSource"],
    ) -> tuple[MonteCarloBenchmarkDataSourceDiagnostic, ...]:
        """Preflight observed-data bindings before running expensive simulations."""

        _raise_for_ambiguous_design_level_data_sources(
            cells=self.executable_cells,
            data_sources=data_sources,
        )
        cells_by_design: dict[str, list[MonteCarloBenchmarkCell]] = {}
        for cell in self.executable_cells:
            cells_by_design.setdefault(_benchmark_data_source_key(cell), []).append(cell)

        return tuple(
            replace(
                _empirical_mixture_data_source_diagnostic(
                    design=cells[0].design,
                    cells=tuple(cells),
                    source=data_sources.get(data_source_key) or data_sources.get(cells[0].design),
                ),
                data_source_key=data_source_key,
            )
            for data_source_key, cells in cells_by_design.items()
        )

    def validate_data_sources(
        self,
        data_sources: dict[str, "BinaryEmpiricalMixtureBenchmarkDataSource"],
    ) -> tuple[MonteCarloBenchmarkDataSourceDiagnostic, ...]:
        """Return diagnostics or raise if a required paper design cannot run."""

        diagnostics = self.data_source_diagnostics(data_sources)
        failures = tuple(diagnostic for diagnostic in diagnostics if not diagnostic.ready)
        if failures:
            preview = [diagnostic.to_dict() for diagnostic in failures]
            raise ValueError(f"Monte Carlo benchmark data source preflight failed: {preview}")
        return diagnostics

    def rerun_manifest(
        self,
        data_sources: dict[str, "BinaryEmpiricalMixtureBenchmarkDataSource"],
        *,
        seed: int,
        alpha: float = 0.05,
        bootstrap_replications: int = 500,
        absolute_tolerance: float = 0.025,
        z_tolerance: float = 2.0,
        cell_count_absolute_tolerance: float | None = None,
        source_mixture_absolute_tolerance: float | None = None,
    ) -> "MonteCarloBenchmarkPlanRerunManifest":
        """Build the full benchmark rerun manifest without consuming simulation budget."""

        _validate_benchmark_tolerances(
            absolute_tolerance=absolute_tolerance,
            z_tolerance=z_tolerance,
            cell_count_absolute_tolerance=cell_count_absolute_tolerance,
            source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
        )
        if not 0 < alpha < 1:
            raise ValueError("alpha must be strictly between 0 and 1.")
        if bootstrap_replications <= 0:
            raise ValueError("bootstrap_replications must be positive.")
        _raise_for_ambiguous_design_level_data_sources(
            cells=self.executable_cells,
            data_sources=data_sources,
        )
        missing_data_sources = [
            key
            for key in self.executable_data_source_keys
            if key not in data_sources
            and not any(
                cell.design in data_sources
                for cell in self.executable_cells
                if _benchmark_data_source_key(cell) == key
            )
        ]
        if missing_data_sources:
            raise KeyError(f"Missing empirical-mixture data sources for keys: {missing_data_sources}.")

        diagnostics = self.validate_data_sources(data_sources)
        seed_rng = np.random.default_rng(seed)
        cell_seeds = seed_rng.integers(
            low=0,
            high=np.iinfo(np.uint32).max,
            size=len(self.executable_cells),
            dtype=np.uint32,
        )
        cells = tuple(
            MonteCarloBenchmarkPlanRerunCell(
                cell=cell,
                seed=int(cell_seed),
                replications=self.replications,
                bootstrap_replications=bootstrap_replications,
                alpha=alpha,
                source=_resolve_benchmark_data_source(data_sources, cell),
            )
            for cell, cell_seed in zip(self.executable_cells, cell_seeds.tolist(), strict=True)
        )
        return MonteCarloBenchmarkPlanRerunManifest(
            plan=self,
            cells=cells,
            data_source_diagnostics=diagnostics,
            seed=seed,
            alpha=alpha,
            absolute_tolerance=absolute_tolerance,
            z_tolerance=z_tolerance,
            cell_count_absolute_tolerance=cell_count_absolute_tolerance,
            source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
            data_source_fingerprints={
                key: _benchmark_data_source_fingerprint(
                    _resolve_benchmark_data_source(data_sources, cell)
                )
                for key, cell in {
                    _benchmark_data_source_key(cell): cell for cell in self.executable_cells
                }.items()
            },
        )

    def rerun_readiness_report(
        self,
        data_sources: dict[str, "BinaryEmpiricalMixtureBenchmarkDataSource"],
        *,
        seed: int,
        alpha: float = 0.05,
        absolute_tolerance: float = 0.025,
        z_tolerance: float = 2.0,
        cell_count_absolute_tolerance: float | None = None,
        source_mixture_absolute_tolerance: float | None = None,
        owner: str = "Phase 7 Monte Carlo verification hardening",
        rerun_command: str | None = None,
    ) -> "MonteCarloBenchmarkPlanReadinessReport":
        """Summarize rerun readiness without consuming Monte Carlo budget."""

        _validate_benchmark_tolerances(
            absolute_tolerance=absolute_tolerance,
            z_tolerance=z_tolerance,
            cell_count_absolute_tolerance=cell_count_absolute_tolerance,
            source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
        )
        if not 0 < alpha < 1:
            raise ValueError("alpha must be strictly between 0 and 1.")
        return MonteCarloBenchmarkPlanReadinessReport(
            plan=self,
            data_source_diagnostics=self.data_source_diagnostics(data_sources),
            seed=seed,
            alpha=alpha,
            absolute_tolerance=absolute_tolerance,
            z_tolerance=z_tolerance,
            cell_count_absolute_tolerance=cell_count_absolute_tolerance,
            source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
            owner=owner,
            rerun_command=rerun_command,
        )


@dataclass(frozen=True)
class MonteCarloBenchmarkPlanRerunCell:
    """One paper cell scheduled for a benchmark rerun."""

    cell: MonteCarloBenchmarkCell
    seed: int
    replications: int
    bootstrap_replications: int
    alpha: float
    source: "BinaryEmpiricalMixtureBenchmarkDataSource"
    replication_start: int = 0
    seed_replications: int | None = None

    def __post_init__(self) -> None:
        if self.replication_start < 0:
            raise ValueError("replication_start must be non-negative.")
        if self.seed_replications is not None:
            if self.seed_replications <= 0:
                raise ValueError("seed_replications must be positive when provided.")
            if self.replication_start + self.replications > self.seed_replications:
                raise ValueError(
                    "replication_start + replications must not exceed seed_replications."
                )

    @property
    def planned_draws(self) -> int:
        return self.replications

    def to_design_kwargs(self, *, name: str | None = None) -> dict[str, Any]:
        cell = self.cell
        return {
            "name": name
            or _benchmark_matrix_design_name(
                table=cell.table,
                design=cell.design,
                mediator=cell.mediator,
                clusters=cell.clusters,
                bins=cell.bins,
                t=cell.t,
            ),
            "df": self.source.analysis_frame(),
            "d": self.source.d,
            "m": self.source.m,
            "y": self.source.y,
            "table": cell.table,
            "design": cell.design,
            "mediator": cell.mediator,
            "clusters": cell.clusters,
            "bins": cell.bins,
            "t": cell.t,
            "seed": self.seed,
            "cluster": self.source.cluster,
            "replications": self.replications,
            "replication_start": self.replication_start,
            "seed_replications": self.seed_replications,
            "bootstrap_replications": self.bootstrap_replications,
            "alpha": self.alpha,
        }

    def to_dict(self) -> dict[str, Any]:
        source_payload = self.source.to_dict()
        return {
            **self.cell.to_dict(),
            "seed": self.seed,
            "replications": self.replications,
            "replication_start": self.replication_start,
            "replication_stop": self.replication_start + self.replications,
            "seed_replications": self.seed_replications,
            "bootstrap_replications": self.bootstrap_replications,
            "planned_draws": self.planned_draws,
            "alpha": self.alpha,
            "data_source": {
                "rows": source_payload["rows"],
                "analysis_frame_columns": source_payload["analysis_frame_columns"],
                "d": source_payload["d"],
                "m": source_payload["m"],
                "y": source_payload["y"],
                "cluster": source_payload["cluster"],
            },
        }


@dataclass(frozen=True)
class MonteCarloBenchmarkPlanRerunManifest:
    """Precomputed execution manifest for a paper benchmark plan rerun."""

    plan: MonteCarloBenchmarkPlan
    cells: tuple[MonteCarloBenchmarkPlanRerunCell, ...]
    data_source_diagnostics: tuple[MonteCarloBenchmarkDataSourceDiagnostic, ...]
    seed: int
    alpha: float
    absolute_tolerance: float
    z_tolerance: float
    cell_count_absolute_tolerance: float | None = None
    source_mixture_absolute_tolerance: float | None = None
    data_source_fingerprints: dict[str, str] | None = None

    def __post_init__(self) -> None:
        _validate_benchmark_tolerances(
            absolute_tolerance=self.absolute_tolerance,
            z_tolerance=self.z_tolerance,
            cell_count_absolute_tolerance=self.cell_count_absolute_tolerance,
            source_mixture_absolute_tolerance=self.source_mixture_absolute_tolerance,
        )
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be strictly between 0 and 1.")

        for scheduled_cell in self.cells:
            if scheduled_cell.replications <= 0:
                raise ValueError("Monte Carlo benchmark rerun manifest scheduled replications must be positive.")
            if scheduled_cell.bootstrap_replications <= 0:
                raise ValueError(
                    "Monte Carlo benchmark rerun manifest scheduled bootstrap_replications must be positive."
                )
            if scheduled_cell.seed < 0:
                raise ValueError("Monte Carlo benchmark rerun manifest scheduled seeds must be non-negative.")

        scheduled_cells = tuple(scheduled_cell.cell for scheduled_cell in self.cells)
        if scheduled_cells != self.plan.executable_cells:
            raise ValueError(
                "Monte Carlo benchmark rerun manifest scheduled cells must "
                "match plan executable cells. Use focused_slice() or cell_index_slice() "
                "for budgeted subsets."
            )

        mismatched_alpha_cells = [
            scheduled_cell.cell.to_dict()
            for scheduled_cell in self.cells
            if not math.isclose(scheduled_cell.alpha, self.alpha, rel_tol=0.0, abs_tol=1e-12)
        ]
        if mismatched_alpha_cells:
            raise ValueError(
                "Monte Carlo benchmark rerun manifest has scheduled cells with "
                f"alpha different from manifest alpha: {mismatched_alpha_cells}"
            )

        scheduled_data_source_keys = {
            _benchmark_data_source_key(scheduled_cell.cell)
            for scheduled_cell in self.cells
        }
        diagnostics_by_key = _diagnostics_by_data_source_key(
            self.data_source_diagnostics
        )
        diagnostic_keys = set(diagnostics_by_key)
        if diagnostic_keys != scheduled_data_source_keys:
            raise ValueError(
                "Monte Carlo benchmark rerun manifest data-source diagnostics "
                f"must match scheduled data-source keys: expected {sorted(scheduled_data_source_keys)}, "
                f"got {sorted(diagnostic_keys)}."
            )
        scheduled_by_data_source_key = _scheduled_cells_by_data_source_key(self.cells)
        diagnostic_drift = []
        for diagnostic in self.data_source_diagnostics:
            diagnostic_key = diagnostic.data_source_key or diagnostic.design
            scheduled_for_key = scheduled_by_data_source_key[diagnostic_key]
            expected_rows = len(scheduled_for_key)
            expected_cluster_resampling = any(
                scheduled_cell.cell.requires_cluster_resampling
                for scheduled_cell in scheduled_for_key
            )
            if (
                diagnostic.executable_rows != expected_rows
                or diagnostic.requires_cluster_resampling != expected_cluster_resampling
            ):
                diagnostic_drift.append(
                    {
                        "data_source_key": diagnostic_key,
                        "design": diagnostic.design,
                        "diagnostic_executable_rows": diagnostic.executable_rows,
                        "scheduled_executable_rows": expected_rows,
                        "diagnostic_requires_cluster_resampling": (
                            diagnostic.requires_cluster_resampling
                        ),
                        "scheduled_requires_cluster_resampling": expected_cluster_resampling,
                    }
                )
        if diagnostic_drift:
            raise ValueError(
                "Monte Carlo benchmark rerun manifest data-source diagnostics "
                "must match scheduled cell counts and cluster requirements: "
                f"{diagnostic_drift}"
            )

        if self.data_source_fingerprints is not None:
            fingerprint_keys = set(self.data_source_fingerprints)
            if fingerprint_keys != scheduled_data_source_keys:
                raise ValueError(
                    "Monte Carlo benchmark rerun manifest data-source fingerprints "
                    f"must match scheduled data-source keys: expected {sorted(scheduled_data_source_keys)}, "
                    f"got {sorted(fingerprint_keys)}."
                )

    @property
    def ready(self) -> bool:
        return all(diagnostic.ready for diagnostic in self.data_source_diagnostics)

    @property
    def planned_draws(self) -> int:
        return int(sum(cell.planned_draws for cell in self.cells))

    def focused_slice(
        self,
        *,
        table: str | None = None,
        design: str | None = None,
        clusters: tuple[int | None, ...] | None = None,
        bins: tuple[int | None, ...] | None = None,
        t_values: tuple[float, ...] | None = None,
        max_cells: int | None = None,
        replications: int | None = None,
    ) -> "MonteCarloBenchmarkPlanRerunManifest":
        """Return a deterministic low-budget slice of a preflighted rerun manifest."""

        if max_cells is not None and max_cells <= 0:
            raise ValueError("max_cells must be positive when provided.")
        if replications is not None and replications <= 0:
            raise ValueError("replications must be positive when provided.")

        selected_cells: list[MonteCarloBenchmarkPlanRerunCell] = []
        for scheduled_cell in self.cells:
            cell = scheduled_cell.cell
            if table is not None and cell.table != table:
                continue
            if design is not None and cell.design != design:
                continue
            if clusters is not None and cell.clusters not in clusters:
                continue
            if bins is not None and cell.bins not in bins:
                continue
            if t_values is not None and not any(abs(cell.t - float(t_value)) <= 1e-12 for t_value in t_values):
                continue
            selected_cells.append(scheduled_cell)
            if max_cells is not None and len(selected_cells) >= max_cells:
                break

        if not selected_cells:
            raise ValueError("focused_slice selected no scheduled Monte Carlo cells.")

        if replications is not None:
            selected_cells = [
                replace(scheduled_cell, replications=replications)
                for scheduled_cell in selected_cells
            ]

        return self._slice_from_selected_cells(tuple(selected_cells))

    def cell_index_slice(
        self,
        start: int,
        stop: int | None = None,
        *,
        replications: int | None = None,
    ) -> "MonteCarloBenchmarkPlanRerunManifest":
        """Return a contiguous scheduled-cell slice for resumable full-manifest runs."""

        if start < 0:
            raise ValueError("start must be non-negative.")
        resolved_stop = len(self.cells) if stop is None else stop
        if resolved_stop < start:
            raise ValueError("stop must be greater than or equal to start.")
        if replications is not None and replications <= 0:
            raise ValueError("replications must be positive when provided.")

        selected_cells = list(self.cells[start:resolved_stop])
        if not selected_cells:
            raise ValueError("cell_index_slice selected no scheduled Monte Carlo cells.")

        if replications is not None:
            selected_cells = [
                replace(scheduled_cell, replications=replications)
                for scheduled_cell in selected_cells
            ]

        return self._slice_from_selected_cells(tuple(selected_cells))

    def replication_index_slice(
        self,
        start: int,
        stop: int,
    ) -> "MonteCarloBenchmarkPlanRerunManifest":
        """Return a one-cell shard over the full deterministic replication schedule."""

        if len(self.cells) != 1:
            raise ValueError("replication_index_slice requires exactly one scheduled cell.")
        if start < 0:
            raise ValueError("replication slice start must be non-negative.")
        if stop <= start:
            raise ValueError("replication slice stop must be greater than start.")

        scheduled_cell = self.cells[0]
        if stop > scheduled_cell.replications:
            raise ValueError(
                "replication slice stop must not exceed the scheduled cell replications."
            )
        shard = replace(
            scheduled_cell,
            replications=stop - start,
            replication_start=start,
            seed_replications=scheduled_cell.replications,
        )
        return self._slice_from_selected_cells((shard,))

    def _slice_from_selected_cells(
        self,
        selected_cells: tuple[MonteCarloBenchmarkPlanRerunCell, ...],
    ) -> "MonteCarloBenchmarkPlanRerunManifest":
        selected_data_source_keys = {
            _benchmark_data_source_key(scheduled_cell.cell) for scheduled_cell in selected_cells
        }
        selected_by_data_source_key = _scheduled_cells_by_data_source_key(tuple(selected_cells))
        diagnostics = tuple(
            _diagnostic_for_scheduled_cells(
                diagnostic,
                scheduled_cells=selected_by_data_source_key[diagnostic.data_source_key or diagnostic.design],
            )
            for diagnostic in self.data_source_diagnostics
            if (diagnostic.data_source_key or diagnostic.design) in selected_data_source_keys
        )
        slice_replications = selected_cells[0].replications
        return MonteCarloBenchmarkPlanRerunManifest(
            plan=MonteCarloBenchmarkPlan(
                method=self.plan.method,
                replications=slice_replications,
                executable_cells=tuple(scheduled_cell.cell for scheduled_cell in selected_cells),
                blocked_rows=self.plan.blocked_rows,
                paper_result_rows=self.plan.paper_result_rows,
            ),
            cells=tuple(selected_cells),
            data_source_diagnostics=diagnostics,
            seed=self.seed,
            alpha=self.alpha,
            absolute_tolerance=self.absolute_tolerance,
            z_tolerance=self.z_tolerance,
            cell_count_absolute_tolerance=self.cell_count_absolute_tolerance,
            source_mixture_absolute_tolerance=self.source_mixture_absolute_tolerance,
            data_source_fingerprints=None
            if self.data_source_fingerprints is None
            else {
                key: fingerprint
                for key, fingerprint in self.data_source_fingerprints.items()
                if key in selected_data_source_keys
            },
        )

    def summary(self) -> dict[str, Any]:
        plan_summary = self.plan.summary()
        summary = {
            "method": self.plan.method,
            "executable_rows": len(self.cells),
            "blocked_rows": plan_summary["blocked_rows"],
            "executable_designs": plan_summary["executable_designs"],
            "default_replications": self.plan.replications,
            "planned_draws": self.planned_draws,
            "ready": self.ready,
            "seed": self.seed,
            "alpha": self.alpha,
            "absolute_tolerance": self.absolute_tolerance,
            "z_tolerance": self.z_tolerance,
            "cell_count_gate_active": self.cell_count_absolute_tolerance is not None,
            "source_mixture_gate_active": self.source_mixture_absolute_tolerance is not None,
            "data_source_ready_rows": int(sum(diagnostic.ready for diagnostic in self.data_source_diagnostics)),
            "data_source_blocked_rows": int(
                sum(not diagnostic.ready for diagnostic in self.data_source_diagnostics)
            ),
        }
        summary.update(_benchmark_target_precision_summary_from_scheduled_cells(self.cells))
        summary.update(
            _source_mixture_precision_summary_from_scheduled_cells(
                self.cells,
                self.data_source_diagnostics,
                z_tolerance=self.z_tolerance,
                source_mixture_absolute_tolerance=self.source_mixture_absolute_tolerance,
            )
        )
        summary.update(_data_source_diagnostic_summary(self.data_source_diagnostics))
        return summary

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_export_payload({
            "summary": self.summary(),
            "data_source_diagnostics": [
                diagnostic.to_dict() for diagnostic in self.data_source_diagnostics
            ],
            "data_source_fingerprints": dict(self.data_source_fingerprints or {}),
            "cells": [cell.to_dict() for cell in self.cells],
            "blocked_rows": [blocked.to_dict() for blocked in self.plan.blocked_rows],
        })

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {
                **cell.to_dict(),
                "status": "scheduled",
                "blocked_reason": None,
            }
            for cell in self.cells
        ]
        blocked_frame = self.plan.to_frame()
        if not blocked_frame.empty:
            for row in blocked_frame.loc[blocked_frame["status"] == "blocked"].to_dict("records"):
                rows.append(row)
        return _json_safe_export_frame(rows)

    def target_precision_frame(self) -> pd.DataFrame:
        """Return row-level target rejection-rate precision for scheduled cells."""

        return _json_safe_export_frame(
            _target_precision_rows_from_scheduled_cells(
                self.cells,
                self.plan.blocked_rows,
                absolute_tolerance=self.absolute_tolerance,
                z_tolerance=self.z_tolerance,
            )
        )

    def source_mixture_precision_frame(self) -> pd.DataFrame:
        """Return row-level precision for the paper empirical-mixture t mechanism."""

        return _json_safe_export_frame(
            _source_mixture_precision_rows_from_scheduled_cells(
                self.cells,
                self.plan.blocked_rows,
                self.data_source_diagnostics,
                z_tolerance=self.z_tolerance,
                source_mixture_absolute_tolerance=self.source_mixture_absolute_tolerance,
            )
        )

    def replication_budget_frame(self) -> pd.DataFrame:
        """Return row-level replication budgets implied by the active tolerance gates."""

        return _json_safe_export_frame(
            _replication_budget_rows_from_scheduled_cells(
                self.cells,
                self.plan.blocked_rows,
                self.data_source_diagnostics,
                absolute_tolerance=self.absolute_tolerance,
                z_tolerance=self.z_tolerance,
                source_mixture_absolute_tolerance=self.source_mixture_absolute_tolerance,
            )
        )

    def cell_count_policy_frame(self) -> pd.DataFrame:
        """Return row-level paper cell-count bin policy for scheduled cells."""

        return _json_safe_export_frame(
            _cell_count_policy_rows_from_scheduled_cells(
                self.cells,
                self.plan.blocked_rows,
            )
        )

    def cell_count_policy_summary(self) -> dict[str, Any]:
        """Return compact paper cell-count bin-policy counts for the manifest."""

        return _cell_count_policy_summary_from_frame(self.cell_count_policy_frame())


@dataclass(frozen=True)
class MonteCarloBenchmarkPlanReadinessReport:
    """Non-executing audit packet for a planned paper benchmark rerun."""

    plan: MonteCarloBenchmarkPlan
    data_source_diagnostics: tuple[MonteCarloBenchmarkDataSourceDiagnostic, ...]
    seed: int
    alpha: float
    absolute_tolerance: float
    z_tolerance: float
    cell_count_absolute_tolerance: float | None = None
    source_mixture_absolute_tolerance: float | None = None
    owner: str = "Phase 7 Monte Carlo verification hardening"
    rerun_command: str | None = None

    @property
    def data_sources_ready(self) -> bool:
        return all(diagnostic.ready for diagnostic in self.data_source_diagnostics)

    @property
    def executable_slice_ready(self) -> bool:
        return bool(self.plan.executable_cells) and self.data_sources_ready

    @property
    def full_paper_matrix_ready(self) -> bool:
        return (
            self.executable_slice_ready
            and self.plan.paper_coverage_complete
            and not self.plan.blocked_rows
        )

    @property
    def blocked_reason_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for blocked in self.plan.blocked_rows:
            counts[blocked.reason] = counts.get(blocked.reason, 0) + 1
        for diagnostic in self.data_source_diagnostics:
            for reason in diagnostic.blocking_reasons:
                counts[reason] = counts.get(reason, 0) + 1
        return counts

    def summary(self) -> dict[str, Any]:
        plan_summary = self.plan.summary()
        summary = {
            "method": self.plan.method,
            "executable_rows": plan_summary["executable_rows"],
            "blocked_rows": plan_summary["blocked_rows"],
            "executable_designs": plan_summary["executable_designs"],
            "default_replications": plan_summary["default_replications"],
            "planned_draws": plan_summary["planned_draws"],
            "data_sources_ready": self.data_sources_ready,
            "executable_slice_ready": self.executable_slice_ready,
            "full_paper_matrix_ready": self.full_paper_matrix_ready,
            "seed": self.seed,
            "alpha": self.alpha,
            "absolute_tolerance": self.absolute_tolerance,
            "z_tolerance": self.z_tolerance,
            "cell_count_gate_active": self.cell_count_absolute_tolerance is not None,
            "source_mixture_gate_active": self.source_mixture_absolute_tolerance is not None,
            "blocked_reason_counts": self.blocked_reason_counts,
        }
        summary.update(_benchmark_target_precision_summary(self.plan.executable_cells, replications=self.plan.replications))
        summary.update(
            _source_mixture_precision_summary_from_plan_cells(
                self.plan.executable_cells,
                self.data_source_diagnostics,
                replications=self.plan.replications,
                z_tolerance=self.z_tolerance,
                source_mixture_absolute_tolerance=self.source_mixture_absolute_tolerance,
            )
        )
        summary.update(_data_source_diagnostic_summary(self.data_source_diagnostics))
        return summary

    def blocker_packet(
        self,
        *,
        owner: str | None = None,
        rerun_command: str | None = None,
    ) -> dict[str, Any]:
        summary = self.summary()
        precision_budget = _precision_budget_summary_from_frames(
            target_frame=self.plan.target_precision_frame(
                absolute_tolerance=self.absolute_tolerance,
                z_tolerance=self.z_tolerance,
            ),
            source_mixture_frame=self.source_mixture_precision_frame(),
        )
        replication_budget = _replication_budget_summary_from_frame(
            self.replication_budget_frame()
        )
        cell_count_policy = _cell_count_policy_summary_from_frame(
            self.plan.cell_count_policy_frame()
        )
        resolved_owner = self.owner if owner is None else owner
        resolved_rerun_command = self.rerun_command if rerun_command is None else rerun_command
        return {
            "owner": resolved_owner,
            "cleared_same_line_items": {
                "data_source_ready_designs": summary.get("data_source_ready_designs", 0),
                "executable_rows": summary["executable_rows"],
                "planned_draws": summary["planned_draws"],
                "cell_count_gate_active": summary["cell_count_gate_active"],
                "source_mixture_gate_active": summary["source_mixture_gate_active"],
            },
            "blocked_same_line_items": {
                "blocked_rows": summary["blocked_rows"],
                "blocked_reason_counts": summary["blocked_reason_counts"],
                "full_paper_matrix_ready": summary["full_paper_matrix_ready"],
            },
            "verified_boundary": {
                "method": summary["method"],
                "executable_designs": summary["executable_designs"],
                "default_replications": summary["default_replications"],
                "min_target_rejection_rate": summary["min_target_rejection_rate"],
                "max_target_rejection_rate": summary["max_target_rejection_rate"],
                "max_target_mc_standard_error": summary["max_target_mc_standard_error"],
            },
            "evidence_checked": {
                "data_source_complete_case_rows": summary.get("data_source_complete_case_rows", {}),
                "data_source_source_clusters": summary.get("data_source_source_clusters", {}),
                "data_source_blocking_reasons": summary.get("data_source_blocking_reasons", {}),
                "data_source_complete_case_rows_by_source_key": summary.get(
                    "data_source_complete_case_rows_by_source_key",
                    {},
                ),
                "data_source_source_clusters_by_source_key": summary.get(
                    "data_source_source_clusters_by_source_key",
                    {},
                ),
                "data_source_blocking_reasons_by_source_key": summary.get(
                    "data_source_blocking_reasons_by_source_key",
                    {},
                ),
            },
            "precision_budget": precision_budget,
            "replication_budget": replication_budget,
            "cell_count_policy": cell_count_policy,
            "paper_acceptance_gate": self.paper_acceptance_gate(),
            "rerun_command": resolved_rerun_command,
            "exit_criteria": "paper_acceptance_gate.gate_passes == True",
        }

    def paper_acceptance_gate(self) -> dict[str, Any]:
        """Return a machine-readable full-paper readiness verdict."""

        summary = self.summary()
        replication_budget = _replication_budget_summary_from_frame(
            self.replication_budget_frame()
        )
        cell_count_policy = _cell_count_policy_summary_from_frame(
            self.plan.cell_count_policy_frame()
        )
        blocking_conditions = {
            "paper_coverage_unknown": int(not self.plan.paper_coverage_known),
            "paper_coverage_shortfall_rows": self.plan.paper_coverage_shortfall_rows or 0,
            "blocked_rows": summary["blocked_rows"],
            "data_source_blocked_designs": summary.get("data_source_blocked_designs", 0),
            "target_replication_shortfall_rows": replication_budget["target_shortfall_rows"],
            "source_mixture_replication_shortfall_rows": replication_budget[
                "source_mixture_shortfall_rows"
            ],
            "cell_count_policy_size_risk_rows": cell_count_policy[
                "cell_count_policy_size_risk_rows"
            ],
            "benchmark_run_not_executed": 1,
        }
        active_blocking_conditions = _active_paper_acceptance_blocking_conditions(
            blocking_conditions
        )
        gate_passes = summary["full_paper_matrix_ready"] and not active_blocking_conditions
        blocking_condition_rows = _paper_acceptance_blocker_rows_from_conditions(
            blocking_conditions
        )
        return {
            "stage": "readiness",
            "gate_passes": gate_passes,
            "verdict": "pass" if gate_passes else "blocked",
            "method": summary["method"],
            **self.plan.paper_coverage_summary(),
            "executable_rows": summary["executable_rows"],
            "blocked_rows": summary["blocked_rows"],
            "planned_draws": summary["planned_draws"],
            "data_sources_ready": summary["data_sources_ready"],
            "executable_slice_ready": summary["executable_slice_ready"],
            "full_paper_matrix_ready": summary["full_paper_matrix_ready"],
            "blocking_conditions": blocking_conditions,
            "active_blocking_conditions": active_blocking_conditions,
            "active_blocking_condition_count": len(active_blocking_conditions),
            "blocking_condition_rows": list(blocking_condition_rows),
            "blocked_reason_counts": summary["blocked_reason_counts"],
            "next_action": _paper_acceptance_next_action(
                blocking_conditions,
                data_sources_ready=summary["data_sources_ready"],
                executed=False,
            ),
        }

    def paper_acceptance_blocker_frame(self) -> pd.DataFrame:
        """Return row-level release blockers from the paper acceptance gate."""

        return _json_safe_export_frame(
            _paper_acceptance_gate_blocker_rows(
                self.paper_acceptance_gate(),
                evidence_by_condition=self._paper_acceptance_blocker_evidence(),
            )
        )

    def _paper_acceptance_blocker_evidence(self) -> dict[str, dict[str, Any]]:
        precision_budget = _precision_budget_summary_from_frames(
            target_frame=self.plan.target_precision_frame(
                absolute_tolerance=self.absolute_tolerance,
                z_tolerance=self.z_tolerance,
            ),
            source_mixture_frame=self.source_mixture_precision_frame(),
        )
        replication_budget = _replication_budget_summary_from_frame(
            self.replication_budget_frame()
        )
        cell_count_policy = self.cell_count_policy_summary()
        summary = self.summary()
        return _paper_acceptance_blocker_evidence_from_summaries(
            paper_coverage=self.plan.paper_coverage_summary(),
            blocked_reason_counts=self.blocked_reason_counts,
            data_source_summary=_data_source_diagnostic_summary(self.data_source_diagnostics),
            precision_budget=precision_budget,
            replication_budget=replication_budget,
            bootstrap_budget={},
            cell_count_policy=cell_count_policy,
            execution_summary=summary,
        )

    def unresolved_paper_row_frame(self) -> pd.DataFrame:
        """Return unresolved paper rows that remain after readiness preflight."""

        frame = self.plan.to_frame().reset_index(drop=True)
        _append_row_aligned_columns(
            frame,
            self.replication_budget_frame().reset_index(drop=True),
            columns=(
                "target_replication_shortfall",
                "source_mixture_replication_shortfall",
                "data_source_ready",
                "data_source_blocking_reasons",
            ),
            source_name="readiness replication budget",
        )
        _append_row_aligned_columns(
            frame,
            self.cell_count_policy_frame().reset_index(drop=True),
            columns=("cell_count_policy_size_risk",),
            source_name="readiness cell-count policy",
        )
        return _paper_acceptance_unresolved_row_frame(
            frame,
            source="readiness",
            include_failed_executed_rows=False,
        )

    def milestone_completion_frame(
        self,
        *,
        owner: str | None = None,
        rerun_command: str | None = None,
    ) -> pd.DataFrame:
        """Return a row-level closeout frame for the current readiness verdict."""

        return _paper_acceptance_gate_completion_frame(
            self.milestone_completion_summary(
                owner=owner,
                rerun_command=rerun_command,
            )
        )

    def milestone_completion_summary(
        self,
        *,
        owner: str | None = None,
        rerun_command: str | None = None,
    ) -> dict[str, Any]:
        """Return the release/milestone completion blocker summary for this preflight."""

        return _paper_acceptance_gate_completion_summary(
            self.paper_acceptance_gate(),
            owner=self.owner if owner is None else owner,
            rerun_command=self.rerun_command if rerun_command is None else rerun_command,
        )

    def raise_for_milestone_completion_blockers(
        self,
        *,
        owner: str | None = None,
        rerun_command: str | None = None,
    ) -> None:
        """Raise if this preflight does not clear the full-paper closeout gate."""

        _raise_for_milestone_completion_blockers(
            self.milestone_completion_summary(
                owner=owner,
                rerun_command=rerun_command,
            )
        )

    def to_dict(
        self,
        *,
        owner: str | None = None,
        rerun_command: str | None = None,
    ) -> dict[str, Any]:
        """Serialize the run result with the same closeout hook used by blockers."""

        return _json_safe_export_payload({
            "summary": self.summary(),
            "blocker_packet": self.blocker_packet(
                owner=owner,
                rerun_command=rerun_command,
            ),
            "paper_acceptance_gate": self.paper_acceptance_gate(),
            "milestone_completion_summary": self.milestone_completion_summary(
                owner=owner,
                rerun_command=rerun_command,
            ),
            "data_source_diagnostics": [
                diagnostic.to_dict() for diagnostic in self.data_source_diagnostics
            ],
            "blocked_rows": [blocked.to_dict() for blocked in self.plan.blocked_rows],
        })

    def source_mixture_precision_frame(self) -> pd.DataFrame:
        """Return row-level precision for the planned empirical-mixture t mechanism."""

        return _json_safe_export_frame(
            _source_mixture_precision_rows_from_plan_cells(
                self.plan.executable_cells,
                self.plan.blocked_rows,
                self.data_source_diagnostics,
                replications=self.plan.replications,
                z_tolerance=self.z_tolerance,
                source_mixture_absolute_tolerance=self.source_mixture_absolute_tolerance,
            )
        )

    def replication_budget_frame(self) -> pd.DataFrame:
        """Return row-level replication budgets implied by the active tolerance gates."""

        return _json_safe_export_frame(
            _replication_budget_rows_from_plan_cells(
                self.plan.executable_cells,
                self.plan.blocked_rows,
                self.data_source_diagnostics,
                replications=self.plan.replications,
                absolute_tolerance=self.absolute_tolerance,
                z_tolerance=self.z_tolerance,
                source_mixture_absolute_tolerance=self.source_mixture_absolute_tolerance,
            )
        )

    def cell_count_policy_frame(self) -> pd.DataFrame:
        """Return row-level paper cell-count bin policy for the planned rerun."""

        return self.plan.cell_count_policy_frame()

    def cell_count_policy_summary(self) -> dict[str, Any]:
        """Return compact paper cell-count bin-policy counts for the readiness report."""

        return self.plan.cell_count_policy_summary()


@dataclass(frozen=True)
class BinaryCSMonteCarloDesign:
    """Simulation design for repeated binary-mediator CS sharp-null draws."""

    name: str
    n_obs: int
    replications: int
    seed: int
    cluster_count: int | None = None
    treatment_probability: float = 0.5
    mediator_control_probability: float = 0.2
    mediator_treatment_shift: float = 0.6
    mediator_effect: float = 2.0
    direct_effect: float = 0.0
    outcome_noise_sd: float = 0.5
    num_y_bins: int | None = None
    alpha: float = 0.05

    def __post_init__(self) -> None:
        if self.n_obs < 4:
            raise ValueError("n_obs must be at least 4.")
        if self.replications <= 0:
            raise ValueError("replications must be positive.")
        if self.cluster_count is not None:
            if self.cluster_count < 2:
                raise ValueError("cluster_count must be at least 2 when provided.")
            if self.n_obs % self.cluster_count != 0:
                raise ValueError("n_obs must be divisible by cluster_count for clustered Monte Carlo designs.")
        if not 0 < self.treatment_probability < 1:
            raise ValueError("treatment_probability must be strictly between 0 and 1.")
        if self.cluster_count is not None:
            treated_clusters = _treated_cluster_count(
                cluster_count=self.cluster_count,
                treatment_probability=self.treatment_probability,
            )
            if treated_clusters == 0 or treated_clusters == self.cluster_count:
                raise ValueError(
                    "cluster_count and treatment_probability must imply at least one treated "
                    "and one control cluster."
                )
        control_prob = self.mediator_control_probability
        treated_prob = self.mediator_control_probability + self.mediator_treatment_shift
        if not 0 <= control_prob <= 1 or not 0 <= treated_prob <= 1:
            raise ValueError("mediator probabilities must stay inside [0, 1] in both treatment arms.")
        if self.outcome_noise_sd < 0:
            raise ValueError("outcome_noise_sd must be non-negative.")
        if self.num_y_bins is not None and self.num_y_bins <= 0:
            raise ValueError("num_y_bins must be positive when provided.")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be strictly between 0 and 1.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_obs": self.n_obs,
            "replications": self.replications,
            "seed": self.seed,
            "cluster_count": self.cluster_count,
            "treatment_probability": self.treatment_probability,
            "mediator_control_probability": self.mediator_control_probability,
            "mediator_treatment_shift": self.mediator_treatment_shift,
            "mediator_effect": self.mediator_effect,
            "direct_effect": self.direct_effect,
            "outcome_noise_sd": self.outcome_noise_sd,
            "num_y_bins": self.num_y_bins,
            "alpha": self.alpha,
        }


@dataclass(frozen=True)
class BinaryPartialDensityMonteCarloDesign:
    """Binary-mediator DGP matching the R package's partial-density simulator."""

    name: str
    n_obs: int
    replications: int
    seed: int
    p_y_m1d1: tuple[float, ...]
    p_y_m1d0: tuple[float, ...]
    p_y_m0d1: tuple[float, ...]
    p_y_m0d0: tuple[float, ...]
    p_m_1: float
    p_m_0: float
    p_d: float = 0.5
    y_values: tuple[Any, ...] | None = None
    num_y_bins: int | None = None
    alpha: float = 0.05
    dgp_source: str = "binary-partial-density"
    paper_reference: str = "packages/r/TestMechs/R/simulate_data_binaryM.R"

    def __post_init__(self) -> None:
        if self.n_obs < 4:
            raise ValueError("n_obs must be at least 4.")
        if self.replications <= 0:
            raise ValueError("replications must be positive.")
        if not 0 < self.p_d < 1:
            raise ValueError("p_d must be strictly between 0 and 1.")
        if not 0 <= self.p_m_1 <= 1 or not 0 <= self.p_m_0 <= 1:
            raise ValueError("p_m_1 and p_m_0 must be probabilities in [0, 1].")

        probability_vectors = {
            "p_y_m1d1": self.p_y_m1d1,
            "p_y_m1d0": self.p_y_m1d0,
            "p_y_m0d1": self.p_y_m0d1,
            "p_y_m0d0": self.p_y_m0d0,
        }
        coerced_vectors = {
            name: _coerce_probability_vector(name=name, values=values)
            for name, values in probability_vectors.items()
        }
        lengths = {len(values) for values in coerced_vectors.values()}
        if len(lengths) != 1:
            raise ValueError("All partial-density outcome probability vectors must have the same length.")
        for name, values in coerced_vectors.items():
            object.__setattr__(self, name, values)

        vector_length = next(iter(lengths))
        if self.y_values is None:
            y_values = tuple(range(1, vector_length + 1))
        else:
            y_values = tuple(self.y_values)
        if len(y_values) != vector_length:
            raise ValueError("y_values must have the same length as the outcome probability vectors.")
        object.__setattr__(self, "y_values", y_values)

        if self.num_y_bins is not None and self.num_y_bins <= 0:
            raise ValueError("num_y_bins must be positive when provided.")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be strictly between 0 and 1.")

    @property
    def cluster_count(self) -> None:
        return None

    @classmethod
    def from_observed_data(
        cls,
        *,
        name: str,
        df: pd.DataFrame,
        d: str,
        m: str,
        y: str,
        n_obs: int,
        replications: int,
        seed: int,
        num_y_bins: int | None = None,
        alpha: float = 0.05,
    ) -> "BinaryPartialDensityMonteCarloDesign":
        cleaned_df = remove_missing_from_df(df=df, d=d, m=m, y=y)
        d0, d1 = _binary_levels(cleaned_df[d], name=d)
        m0, m1 = _binary_levels(cleaned_df[m], name=m)
        y_processed = discretize_y(cleaned_df[y], num_bins=num_y_bins) if num_y_bins is not None else cleaned_df[y]
        analysis_df = cleaned_df.copy()
        analysis_df["_tm_y_processed"] = y_processed
        y_values = _observed_outcome_values(analysis_df["_tm_y_processed"])

        treated_df = analysis_df[analysis_df[d] == d1]
        control_df = analysis_df[analysis_df[d] == d0]
        if treated_df.empty or control_df.empty:
            raise ValueError("Observed binary partial-density calibration requires observations in both treatment arms.")

        return cls(
            name=name,
            n_obs=n_obs,
            replications=replications,
            seed=seed,
            p_y_m1d1=_conditional_y_probabilities(
                analysis_df,
                d=d,
                m=m,
                y="_tm_y_processed",
                d_value=d1,
                m_value=m1,
                y_values=y_values,
            ),
            p_y_m1d0=_conditional_y_probabilities(
                analysis_df,
                d=d,
                m=m,
                y="_tm_y_processed",
                d_value=d0,
                m_value=m1,
                y_values=y_values,
            ),
            p_y_m0d1=_conditional_y_probabilities(
                analysis_df,
                d=d,
                m=m,
                y="_tm_y_processed",
                d_value=d1,
                m_value=m0,
                y_values=y_values,
            ),
            p_y_m0d0=_conditional_y_probabilities(
                analysis_df,
                d=d,
                m=m,
                y="_tm_y_processed",
                d_value=d0,
                m_value=m0,
                y_values=y_values,
            ),
            p_m_1=float((treated_df[m] == m1).mean()),
            p_m_0=float((control_df[m] == m1).mean()),
            p_d=float((analysis_df[d] == d1).mean()),
            y_values=y_values,
            num_y_bins=num_y_bins,
            alpha=alpha,
            dgp_source="observed-binary-partial-density",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_obs": self.n_obs,
            "replications": self.replications,
            "seed": self.seed,
            "p_y_m1d1": self.p_y_m1d1,
            "p_y_m1d0": self.p_y_m1d0,
            "p_y_m0d1": self.p_y_m0d1,
            "p_y_m0d0": self.p_y_m0d0,
            "p_m_1": self.p_m_1,
            "p_m_0": self.p_m_0,
            "p_d": self.p_d,
            "y_values": self.y_values,
            "num_y_bins": self.num_y_bins,
            "alpha": self.alpha,
            "dgp_source": self.dgp_source,
            "paper_reference": self.paper_reference,
        }

    def expected_cell_probabilities(self) -> pd.DataFrame:
        """Return the exact DGP probabilities for each (D, M, Y) cell."""

        return _binary_partial_density_expected_cell_probabilities(self)


@dataclass(frozen=True)
class BinaryEmpiricalMixtureMonteCarloDesign:
    """Paper-calibrated binary-mediator empirical-mixture Monte Carlo design."""

    name: str
    replications: int
    seed: int
    t: float
    control_pool: pd.DataFrame
    treated_pool: pd.DataFrame
    n_control_per_draw: int
    n_treated_per_draw: int
    arm_assignment: str = "fixed_observed_arms"
    treatment_probability: float | None = None
    cluster_count: int | None = None
    clusters_per_arm: int | None = None
    num_y_bins: int | None = None
    bootstrap_replications: int = 500
    alpha: float = 0.05
    replication_start: int = 0
    seed_replications: int | None = None
    dgp_source: str = "paper-empirical-mixture"
    paper_reference: str = "manuscript/sources/arxiv-2404.11739v3/draft.tex:409"
    paper_contract_dict: dict[str, Any] | None = None
    source_treatment_levels: tuple[Any, Any] | None = None
    source_mediator_levels: tuple[Any, ...] | None = None

    def __post_init__(self) -> None:
        if self.replications <= 0:
            raise ValueError("replications must be positive.")
        seed_replications = (
            self.replications
            if self.seed_replications is None
            else int(self.seed_replications)
        )
        if seed_replications <= 0:
            raise ValueError("seed_replications must be positive.")
        if self.replication_start < 0:
            raise ValueError("replication_start must be non-negative.")
        if self.replication_start + self.replications > seed_replications:
            raise ValueError(
                "replication_start + replications must not exceed seed_replications."
            )
        if not 0 <= self.t <= 1:
            raise ValueError("t must be in [0, 1].")
        if self.n_control_per_draw <= 0 or self.n_treated_per_draw <= 0:
            raise ValueError("Both treatment arms must have positive draw sizes.")
        if self.arm_assignment not in {"fixed_observed_arms", "iid_bernoulli"}:
            raise ValueError("arm_assignment must be 'fixed_observed_arms' or 'iid_bernoulli'.")
        treatment_probability = (
            self.treatment_probability
            if self.treatment_probability is not None
            else self.n_treated_per_draw / self.n_obs_per_draw
        )
        if not 0 < treatment_probability < 1:
            raise ValueError("treatment_probability must be strictly between 0 and 1.")
        object.__setattr__(self, "treatment_probability", float(treatment_probability))
        if self.cluster_count is not None:
            if self.arm_assignment != "fixed_observed_arms":
                raise ValueError("Clustered empirical-mixture designs require fixed_observed_arms.")
            if self.cluster_count < 2 or self.cluster_count % 2 != 0:
                raise ValueError("cluster_count must be an even integer of at least 2.")
            expected_clusters_per_arm = self.cluster_count // 2
            if self.clusters_per_arm is None:
                object.__setattr__(self, "clusters_per_arm", expected_clusters_per_arm)
            elif self.clusters_per_arm != expected_clusters_per_arm:
                raise ValueError("clusters_per_arm must equal cluster_count / 2.")
            for pool_name, pool in {"control_pool": self.control_pool, "treated_pool": self.treated_pool}.items():
                if "_source_cluster" not in pool:
                    raise ValueError(f"{pool_name} must contain _source_cluster for clustered designs.")
        elif self.clusters_per_arm is not None:
            raise ValueError("clusters_per_arm requires cluster_count.")
        if self.num_y_bins is not None and self.num_y_bins <= 0:
            raise ValueError("num_y_bins must be positive when provided.")
        if self.bootstrap_replications <= 0:
            raise ValueError("bootstrap_replications must be positive.")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be strictly between 0 and 1.")

        object.__setattr__(self, "seed_replications", seed_replications)
        object.__setattr__(self, "control_pool", self.control_pool.reset_index(drop=True).copy())
        object.__setattr__(self, "treated_pool", self.treated_pool.reset_index(drop=True).copy())
        if self.paper_contract_dict is None:
            object.__setattr__(self, "paper_contract_dict", {})
        if self.source_treatment_levels is not None:
            object.__setattr__(self, "source_treatment_levels", tuple(self.source_treatment_levels))
        if self.source_mediator_levels is not None:
            object.__setattr__(self, "source_mediator_levels", tuple(self.source_mediator_levels))

    @property
    def n_obs_per_draw(self) -> int:
        return int(self.n_control_per_draw + self.n_treated_per_draw)

    @classmethod
    def from_observed_data(
        cls,
        *,
        name: str,
        df: pd.DataFrame,
        d: str,
        m: str,
        y: str,
        replications: int,
        seed: int,
        t: float,
        cluster: str | None = None,
        cluster_count: int | None = None,
        num_y_bins: int | None = None,
        bootstrap_replications: int = 500,
        alpha: float = 0.05,
        replication_start: int = 0,
        seed_replications: int | None = None,
        arm_assignment: str = "fixed_observed_arms",
        paper_contract_dict: dict[str, Any] | None = None,
        mediator_kind: str = "binary",
    ) -> "BinaryEmpiricalMixtureMonteCarloDesign":
        cleaned_df = remove_missing_from_df(df=df, d=d, m=m, y=y)
        d0, d1 = _binary_levels(cleaned_df[d], name=d)
        _raise_for_nonfinite_numeric_values(cleaned_df[y], column=y)
        if mediator_kind == "binary":
            mediator_levels = normalize_binary_support(cleaned_df[m], column=m).original_levels
        elif mediator_kind == "nonbinary":
            mediator_levels = _ordered_nonbinary_mediator_levels(cleaned_df[m], column=m)
            if len(mediator_levels) <= 2:
                raise ValueError(
                    "Observed nonbinary empirical-mixture calibration requires more than two mediator levels."
                )
            _raise_for_nonfinite_numeric_support_levels(mediator_levels, column=m)
        else:
            raise ValueError("mediator_kind must be either 'binary' or 'nonbinary'.")

        analysis_df = cleaned_df.copy()
        mediator_map = {level: index for index, level in enumerate(mediator_levels)}
        analysis_df["_tm_mediator"] = analysis_df[m].map(mediator_map).astype(int)
        analysis_df["_tm_outcome"] = analysis_df[y]
        pool_columns = ["_tm_mediator", "_tm_outcome"]
        if cluster is not None:
            if cluster not in analysis_df:
                raise ValueError(f"cluster column {cluster!r} is not present in df.")
            if bool(analysis_df[cluster].isna().any()):
                raise ValueError("Observed clustered empirical-mixture calibration requires non-missing cluster labels.")
            analysis_df["_source_cluster"] = analysis_df[cluster]
            treatment_levels_by_cluster = analysis_df.groupby("_source_cluster", sort=False)[d].nunique()
            if bool((treatment_levels_by_cluster > 1).any()):
                raise ValueError(
                    "Observed clustered empirical-mixture calibration requires treatment to be fixed "
                    "within each source cluster."
                )
            pool_columns.append("_source_cluster")

        control_pool = analysis_df.loc[analysis_df[d] == d0, pool_columns].rename(
            columns={"_tm_mediator": "mediator", "_tm_outcome": "outcome"}
        )
        treated_pool = analysis_df.loc[analysis_df[d] == d1, pool_columns].rename(
            columns={"_tm_mediator": "mediator", "_tm_outcome": "outcome"}
        )
        if control_pool.empty or treated_pool.empty:
            raise ValueError("Observed empirical-mixture calibration requires both treatment arms.")

        requested_cluster_count = cluster_count
        if requested_cluster_count is None and cluster is not None:
            requested_cluster_count = int(analysis_df["_source_cluster"].nunique())
        control_count = int((analysis_df[d] == d0).sum())
        treated_count = int((analysis_df[d] == d1).sum())

        return cls(
            name=name,
            replications=replications,
            seed=seed,
            t=t,
            control_pool=control_pool,
            treated_pool=treated_pool,
            n_control_per_draw=control_count,
            n_treated_per_draw=treated_count,
            arm_assignment=arm_assignment,
            treatment_probability=treated_count / (control_count + treated_count),
            cluster_count=requested_cluster_count,
            clusters_per_arm=requested_cluster_count // 2 if requested_cluster_count is not None else None,
            num_y_bins=num_y_bins,
            bootstrap_replications=bootstrap_replications,
            alpha=alpha,
            replication_start=replication_start,
            seed_replications=seed_replications,
            paper_contract_dict=paper_contract_dict,
            source_treatment_levels=(d0, d1),
            source_mediator_levels=mediator_levels,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "replications": self.replications,
            "seed": self.seed,
            "t": self.t,
            "n_obs_per_draw": self.n_obs_per_draw,
            "n_control_per_draw": self.n_control_per_draw,
            "n_treated_per_draw": self.n_treated_per_draw,
            "arm_assignment": self.arm_assignment,
            "treatment_probability": self.treatment_probability,
            "cluster_count": self.cluster_count,
            "clusters_per_arm": self.clusters_per_arm,
            "num_y_bins": self.num_y_bins,
            "bootstrap_replications": self.bootstrap_replications,
            "alpha": self.alpha,
            "replication_start": self.replication_start,
            "replication_stop": self.replication_start + self.replications,
            "seed_replications": self.seed_replications,
            "dgp_source": self.dgp_source,
            "paper_reference": self.paper_reference,
            "paper_contract": dict(self.paper_contract_dict or {}),
            "source_treatment_levels": None
            if self.source_treatment_levels is None
            else tuple(_json_safe_diagnostic_value(value) for value in self.source_treatment_levels),
            "source_mediator_levels": None
            if self.source_mediator_levels is None
            else tuple(_json_safe_diagnostic_value(value) for value in self.source_mediator_levels),
            "internal_mediator_levels": tuple(
                _normalize_diagnostic_value(value)
                for value in _ordered_unique_values(
                    pd.concat(
                        [self.control_pool["mediator"], self.treated_pool["mediator"]],
                        ignore_index=True,
                    )
                )
            ),
        }

    def source_mixture_contract(self) -> pd.DataFrame:
        """Return the exact paper source-pool mixture probabilities by simulated arm."""

        return _binary_empirical_mixture_source_mixture_contract(self)


def _binary_empirical_mixture_source_mixture_contract(
    design: BinaryEmpiricalMixtureMonteCarloDesign,
) -> pd.DataFrame:
    sampling_unit = "cluster" if design.cluster_count is not None else "unit"
    rows: list[dict[str, Any]] = []
    for simulated_treated, source_pool, probability in (
        (0, "control", 1.0),
        (0, "treated", 0.0),
        (1, "control", 1.0 - float(design.t)),
        (1, "treated", float(design.t)),
    ):
        rows.append(
            {
                "sampling_unit": sampling_unit,
                "simulated_treated": simulated_treated,
                "source_pool": source_pool,
                "source_mixture_probability": float(probability),
                "t": float(design.t),
            }
        )
    return _json_safe_export_frame(rows)


@dataclass(frozen=True)
class BinaryEmpiricalMixtureBenchmarkDataSource:
    """Observed data binding for one paper empirical-mixture design."""

    df: pd.DataFrame
    d: str
    m: str
    y: str
    cluster: str | None = None
    analysis_frame_columns: tuple[str, ...] = ()
    expected_complete_case_rows: int | None = None
    expected_control_rows: int | None = None
    expected_treated_rows: int | None = None
    expected_source_clusters: int | None = None
    expected_control_source_clusters: int | None = None
    expected_treated_source_clusters: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.analysis_frame_columns, str):
            analysis_frame_columns = (self.analysis_frame_columns,)
        else:
            analysis_frame_columns = tuple(self.analysis_frame_columns)
        object.__setattr__(self, "analysis_frame_columns", analysis_frame_columns)

        missing_columns = [
            column
            for column in (self.d, self.m, self.y, *analysis_frame_columns)
            if column not in self.df
        ]
        if missing_columns:
            raise KeyError(f"Benchmark data source is missing columns: {missing_columns}.")
        if self.cluster is not None and self.cluster not in self.df:
            raise KeyError(f"Benchmark data source is missing cluster column {self.cluster!r}.")
        for field_name in (
            "expected_complete_case_rows",
            "expected_control_rows",
            "expected_treated_rows",
            "expected_source_clusters",
            "expected_control_source_clusters",
            "expected_treated_source_clusters",
        ):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must be non-negative when provided.")

    def analysis_frame(self) -> pd.DataFrame:
        required_columns = tuple(dict.fromkeys((self.d, self.m, self.y, *self.analysis_frame_columns)))
        return self.df.dropna(subset=list(required_columns)).copy()

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": int(self.df.shape[0]),
            "columns": tuple(self.df.columns),
            "analysis_frame_columns": self.analysis_frame_columns,
            "d": self.d,
            "m": self.m,
            "y": self.y,
            "cluster": self.cluster,
            "expected_complete_case_rows": self.expected_complete_case_rows,
            "expected_control_rows": self.expected_control_rows,
            "expected_treated_rows": self.expected_treated_rows,
            "expected_source_clusters": self.expected_source_clusters,
            "expected_control_source_clusters": self.expected_control_source_clusters,
            "expected_treated_source_clusters": self.expected_treated_source_clusters,
        }


@dataclass(frozen=True)
class MonteCarloDrawResult:
    """Draw-level result from one Monte Carlo replication."""

    replication: int
    seed: int
    reject: bool
    test_stat: float
    critical_value: float
    p_value: float
    applied_num_y_bins: int | None
    n_obs_used: int
    control_observations: int
    treated_observations: int
    min_cell_count: int
    min_cluster_count: int
    median_cell_count: float
    median_independent_count_per_cell: float
    size_risk: bool
    empty_cell_count: int = 0
    empty_cluster_cell_count: int = 0
    small_cell_count: int = 0
    small_cluster_cell_count: int = 0
    size_risk_threshold: int = 15
    empty_cells: tuple[dict[str, Any], ...] = ()
    small_cells: tuple[dict[str, Any], ...] = ()
    empty_cluster_cells: tuple[dict[str, Any], ...] = ()
    small_cluster_cells: tuple[dict[str, Any], ...] = ()
    n_clusters_used: int | None = None
    treated_clusters: int | None = None
    control_clusters: int | None = None
    cluster_size: int | None = None
    treated_source_treated_draws: int | None = None
    treated_source_control_draws: int | None = None
    treated_source_treated_clusters: int | None = None
    treated_source_control_clusters: int | None = None

    def to_dict(self) -> dict[str, Any]:
        test_stat, test_stat_finite, test_stat_nonfinite = _json_safe_float_with_marker(
            self.test_stat
        )
        critical_value, critical_value_finite, critical_value_nonfinite = _json_safe_float_with_marker(
            self.critical_value
        )
        p_value, p_value_finite, p_value_nonfinite = _json_safe_float_with_marker(
            self.p_value
        )
        return _json_safe_export_payload({
            "replication": self.replication,
            "seed": self.seed,
            "reject": self.reject,
            "test_stat": test_stat,
            "test_stat_is_finite": test_stat_finite,
            "test_stat_nonfinite": test_stat_nonfinite,
            "critical_value": critical_value,
            "critical_value_is_finite": critical_value_finite,
            "critical_value_nonfinite": critical_value_nonfinite,
            "p_value": p_value,
            "p_value_is_finite": p_value_finite,
            "p_value_nonfinite": p_value_nonfinite,
            "applied_num_y_bins": self.applied_num_y_bins,
            "n_obs_used": self.n_obs_used,
            "control_observations": self.control_observations,
            "treated_observations": self.treated_observations,
            "min_cell_count": self.min_cell_count,
            "min_cluster_count": self.min_cluster_count,
            "median_cell_count": self.median_cell_count,
            "median_independent_count_per_cell": self.median_independent_count_per_cell,
            "size_risk": self.size_risk,
            "empty_cell_count": self.empty_cell_count,
            "empty_cluster_cell_count": self.empty_cluster_cell_count,
            "small_cell_count": self.small_cell_count,
            "small_cluster_cell_count": self.small_cluster_cell_count,
            "size_risk_threshold": self.size_risk_threshold,
            "empty_cells": self.empty_cells,
            "small_cells": self.small_cells,
            "empty_cluster_cells": self.empty_cluster_cells,
            "small_cluster_cells": self.small_cluster_cells,
            "n_clusters_used": self.n_clusters_used,
            "treated_clusters": self.treated_clusters,
            "control_clusters": self.control_clusters,
            "cluster_size": self.cluster_size,
            "treated_source_treated_draws": self.treated_source_treated_draws,
            "treated_source_control_draws": self.treated_source_control_draws,
            "treated_source_treated_clusters": self.treated_source_treated_clusters,
            "treated_source_control_clusters": self.treated_source_control_clusters,
        })


@dataclass(frozen=True)
class MonteCarloSimulationResult:
    """Repeated-simulation output with draw-level CS sharp-null diagnostics."""

    design: BinaryCSMonteCarloDesign | BinaryPartialDensityMonteCarloDesign | BinaryEmpiricalMixtureMonteCarloDesign
    draws: tuple[MonteCarloDrawResult, ...]

    @property
    def method(self) -> str:
        if isinstance(self.design, BinaryEmpiricalMixtureMonteCarloDesign):
            paper_contract = self.design.paper_contract_dict or {}
            return str(paper_contract.get("target_method", "CS"))
        return "CS"

    @property
    def rejection_rate(self) -> float:
        return float(np.mean([draw.reject for draw in self.draws]))

    @property
    def mean_p_value(self) -> float:
        return float(np.mean([draw.p_value for draw in self.draws]))

    @property
    def mean_median_cell_count(self) -> float:
        return float(np.mean([draw.median_cell_count for draw in self.draws]))

    @property
    def mean_median_independent_count_per_cell(self) -> float:
        return float(np.mean([draw.median_independent_count_per_cell for draw in self.draws]))

    @property
    def mean_treated_source_treated_draw_share(self) -> float | None:
        return _mean_binary_source_share(
            self.draws,
            numerator_attr="treated_source_treated_draws",
            complement_attr="treated_source_control_draws",
        )

    @property
    def mean_treated_source_treated_cluster_share(self) -> float | None:
        return _mean_binary_source_share(
            self.draws,
            numerator_attr="treated_source_treated_clusters",
            complement_attr="treated_source_control_clusters",
        )

    def summary(self) -> dict[str, Any]:
        summary = {
            "design": self.design.name,
            "method": self.method,
            "replications": len(self.draws),
            "rejection_rate": self.rejection_rate,
            "mean_p_value": self.mean_p_value,
            "mean_median_cell_count": self.mean_median_cell_count,
            "mean_median_independent_count_per_cell": self.mean_median_independent_count_per_cell,
            "nonfinite_test_stat_draws": int(
                sum(not math.isfinite(float(draw.test_stat)) for draw in self.draws)
            ),
            "nonfinite_critical_value_draws": int(
                sum(not math.isfinite(float(draw.critical_value)) for draw in self.draws)
            ),
            "nonfinite_p_value_draws": int(
                sum(not math.isfinite(float(draw.p_value)) for draw in self.draws)
            ),
            "size_risk_draws": int(sum(draw.size_risk for draw in self.draws)),
            "empty_cell_rows": int(sum(draw.empty_cell_count for draw in self.draws)),
            "empty_cluster_cell_rows": int(
                sum(draw.empty_cluster_cell_count for draw in self.draws)
            ),
            "small_cell_rows": int(sum(draw.small_cell_count for draw in self.draws)),
            "small_cluster_cell_rows": int(
                sum(draw.small_cluster_cell_count for draw in self.draws)
            ),
            "empty_cell_preview": _monte_carlo_draw_cell_preview(
                self.draws,
                cell_attr="empty_cells",
            ),
            "small_cell_preview": _monte_carlo_draw_cell_preview(
                self.draws,
                cell_attr="small_cells",
            ),
            "empty_cluster_cell_preview": _monte_carlo_draw_cell_preview(
                self.draws,
                cell_attr="empty_cluster_cells",
            ),
            "small_cluster_cell_preview": _monte_carlo_draw_cell_preview(
                self.draws,
                cell_attr="small_cluster_cells",
            ),
            "clustered": self.design.cluster_count is not None,
            "cluster_count": self.design.cluster_count,
        }
        if isinstance(self.design, BinaryEmpiricalMixtureMonteCarloDesign):
            summary.update(
                {
                    "target_t": self.design.t,
                    "mean_treated_source_treated_draw_share": (
                        self.mean_treated_source_treated_draw_share
                    ),
                    "mean_treated_source_treated_cluster_share": (
                        self.mean_treated_source_treated_cluster_share
                    ),
                }
            )
        return _json_safe_export_payload(summary)

    def sample_shape_summary(self) -> dict[str, Any]:
        """Summarize draw-level sample sizes and treatment-arm shape."""

        if not self.draws:
            return _json_safe_export_payload({
                "design": self.design.name,
                "replications": 0,
                "clustered": self.design.cluster_count is not None,
                "cluster_count": self.design.cluster_count,
            })

        n_obs_used = [draw.n_obs_used for draw in self.draws]
        control_observations = [draw.control_observations for draw in self.draws]
        treated_observations = [draw.treated_observations for draw in self.draws]
        summary = {
            "design": self.design.name,
            "replications": len(self.draws),
            "clustered": self.design.cluster_count is not None,
            "cluster_count": self.design.cluster_count,
            "min_n_obs_used": int(min(n_obs_used)),
            "max_n_obs_used": int(max(n_obs_used)),
            "mean_n_obs_used": float(np.mean(n_obs_used)),
            "min_control_observations": int(min(control_observations)),
            "max_control_observations": int(max(control_observations)),
            "mean_control_observations": float(np.mean(control_observations)),
            "min_treated_observations": int(min(treated_observations)),
            "max_treated_observations": int(max(treated_observations)),
            "mean_treated_observations": float(np.mean(treated_observations)),
        }

        cluster_counts = [draw.n_clusters_used for draw in self.draws if draw.n_clusters_used is not None]
        if cluster_counts:
            control_clusters = [
                draw.control_clusters for draw in self.draws if draw.control_clusters is not None
            ]
            treated_clusters = [
                draw.treated_clusters for draw in self.draws if draw.treated_clusters is not None
            ]
            summary.update(
                {
                    "min_n_clusters_used": int(min(cluster_counts)),
                    "max_n_clusters_used": int(max(cluster_counts)),
                    "mean_n_clusters_used": float(np.mean(cluster_counts)),
                    "min_control_clusters": int(min(control_clusters)),
                    "max_control_clusters": int(max(control_clusters)),
                    "mean_control_clusters": float(np.mean(control_clusters)),
                    "min_treated_clusters": int(min(treated_clusters)),
                    "max_treated_clusters": int(max(treated_clusters)),
                    "mean_treated_clusters": float(np.mean(treated_clusters)),
                }
            )

        return _json_safe_export_payload(summary)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_export_payload({
            "design": self.design.to_dict(),
            "summary": self.summary(),
            "draws": [draw.to_dict() for draw in self.draws],
        })

    def to_frame(self) -> pd.DataFrame:
        return _json_safe_export_frame([draw.to_dict() for draw in self.draws])


@dataclass(frozen=True)
class MonteCarloBenchmarkDiagnostic:
    """Sampling-error-aware comparison between a simulation and a paper row."""

    simulation: MonteCarloSimulationResult
    row: MonteCarloResultRow
    method: str
    absolute_tolerance: float
    z_tolerance: float
    cell_count_absolute_tolerance: float | None = None
    source_mixture_absolute_tolerance: float | None = None
    cluster_cell_count: ClusterCellCount | None = None

    @property
    def target_rejection_rate(self) -> float:
        return self.row.rejection_rates[self.method]

    @property
    def observed_rejection_rate(self) -> float:
        return self.simulation.rejection_rate

    @property
    def absolute_error(self) -> float:
        return abs(self.observed_rejection_rate - self.target_rejection_rate)

    @property
    def rejection_rate_absolute_error(self) -> float:
        return self.absolute_error

    @property
    def monte_carlo_standard_error(self) -> float:
        target = self.target_rejection_rate
        replications = len(self.simulation.draws)
        return float(math.sqrt(target * (1.0 - target) / replications))

    @property
    def z_score(self) -> float:
        standard_error = self.monte_carlo_standard_error
        if standard_error == 0:
            return 0.0 if self.absolute_error == 0 else math.inf
        return self.absolute_error / standard_error

    @property
    def within_absolute_tolerance(self) -> bool:
        return self.absolute_error <= self.absolute_tolerance

    @property
    def within_sampling_error(self) -> bool:
        return self.z_score <= self.z_tolerance

    @property
    def rejection_rate_passes(self) -> bool:
        return self.within_absolute_tolerance or self.within_sampling_error

    @property
    def passes(self) -> bool:
        return (
            self.rejection_rate_passes
            and self.within_cell_count_tolerance
            and self.within_source_mixture_tolerance
        )

    @property
    def target_median_independent_count_per_cell(self) -> float | None:
        if self.cluster_cell_count is None:
            return None
        return self.cluster_cell_count.median_independent_clusters_per_cell

    @property
    def observed_mean_median_independent_count_per_cell(self) -> float:
        return self.simulation.mean_median_independent_count_per_cell

    @property
    def cell_count_absolute_error(self) -> float | None:
        target = self.target_median_independent_count_per_cell
        if target is None:
            return None
        return abs(self.observed_mean_median_independent_count_per_cell - target)

    @property
    def cell_count_tolerance_active(self) -> bool:
        return self.cell_count_absolute_tolerance is not None and self.cluster_cell_count is not None

    @property
    def within_cell_count_tolerance(self) -> bool:
        if not self.cell_count_tolerance_active:
            return True
        error = self.cell_count_absolute_error
        return error is not None and error <= float(self.cell_count_absolute_tolerance)

    @property
    def observed_source_mixture_share(self) -> float | None:
        if not isinstance(self.simulation.design, BinaryEmpiricalMixtureMonteCarloDesign):
            return None
        if self.simulation.design.cluster_count is not None:
            return self.simulation.mean_treated_source_treated_cluster_share
        return self.simulation.mean_treated_source_treated_draw_share

    @property
    def source_mixture_absolute_error(self) -> float | None:
        observed = self.observed_source_mixture_share
        if observed is None:
            return None
        return abs(observed - self.row.t)

    @property
    def source_mixture_effective_trials(self) -> int | None:
        if not isinstance(self.simulation.design, BinaryEmpiricalMixtureMonteCarloDesign):
            return None
        if self.observed_source_mixture_share is None:
            return None
        if self.simulation.design.cluster_count is not None:
            denominator = _source_count_total(
                self.simulation.draws,
                numerator_attr="treated_source_treated_clusters",
                complement_attr="treated_source_control_clusters",
            )
        else:
            denominator = _source_count_total(
                self.simulation.draws,
                numerator_attr="treated_source_treated_draws",
                complement_attr="treated_source_control_draws",
            )
        if denominator is None or denominator <= 0:
            return None
        return int(denominator)

    @property
    def source_mixture_standard_error(self) -> float | None:
        effective_trials = self.source_mixture_effective_trials
        if effective_trials is None:
            return None
        target = self.row.t
        return float(math.sqrt(target * (1.0 - target) / effective_trials))

    @property
    def source_mixture_z_score(self) -> float | None:
        error = self.source_mixture_absolute_error
        standard_error = self.source_mixture_standard_error
        if error is None or standard_error is None:
            return None
        if standard_error == 0:
            return 0.0 if error == 0 else math.inf
        return error / standard_error

    @property
    def source_mixture_tolerance_active(self) -> bool:
        return (
            self.source_mixture_absolute_tolerance is not None
            and self.source_mixture_absolute_error is not None
        )

    @property
    def within_source_mixture_tolerance(self) -> bool:
        if not self.source_mixture_tolerance_active:
            return True
        error = self.source_mixture_absolute_error
        return error is not None and error <= float(self.source_mixture_absolute_tolerance)

    @property
    def pass_reason(self) -> str | None:
        if not self.within_cell_count_tolerance or not self.within_source_mixture_tolerance:
            return None
        if self.within_absolute_tolerance:
            return "absolute_tolerance"
        if self.within_sampling_error:
            return "sampling_error"
        return None

    @property
    def failure_reasons(self) -> tuple[str, ...]:
        if self.passes:
            return ()
        reasons: list[str] = []
        if not self.rejection_rate_passes:
            if not self.within_absolute_tolerance:
                reasons.append("absolute_tolerance")
            if not self.within_sampling_error:
                reasons.append("sampling_error")
        if not self.within_cell_count_tolerance:
            reasons.append("cell_count_tolerance")
        if not self.within_source_mixture_tolerance:
            reasons.append("source_mixture_tolerance")
        return tuple(reasons)

    def summary(self) -> dict[str, Any]:
        z_score, z_score_is_finite, z_score_nonfinite = _json_safe_float_with_marker(
            self.z_score
        )
        (
            source_mixture_z_score,
            source_mixture_z_score_is_finite,
            source_mixture_z_score_nonfinite,
        ) = _json_safe_optional_float_with_marker(self.source_mixture_z_score)
        return {
            "simulation": self.simulation.design.name,
            "method": self.method,
            "target_rejection_rate": self.target_rejection_rate,
            "observed_rejection_rate": self.observed_rejection_rate,
            "absolute_error": self.absolute_error,
            "rejection_rate_absolute_error": self.rejection_rate_absolute_error,
            "monte_carlo_standard_error": self.monte_carlo_standard_error,
            "z_score": z_score,
            "z_score_is_finite": z_score_is_finite,
            "z_score_nonfinite": z_score_nonfinite,
            "passes": self.passes,
            "target_t": self.row.t,
            "observed_source_mixture_share": self.observed_source_mixture_share,
            "source_mixture_absolute_error": self.source_mixture_absolute_error,
            "source_mixture_effective_trials": self.source_mixture_effective_trials,
            "source_mixture_standard_error": self.source_mixture_standard_error,
            "source_mixture_z_score": source_mixture_z_score,
            "source_mixture_z_score_is_finite": source_mixture_z_score_is_finite,
            "source_mixture_z_score_nonfinite": source_mixture_z_score_nonfinite,
            "empty_cell_rows": self.simulation.summary()["empty_cell_rows"],
            "empty_cluster_cell_rows": self.simulation.summary()["empty_cluster_cell_rows"],
            "small_cell_rows": self.simulation.summary()["small_cell_rows"],
            "small_cluster_cell_rows": self.simulation.summary()["small_cluster_cell_rows"],
            "empty_cell_preview": self.simulation.summary()["empty_cell_preview"],
            "empty_cluster_cell_preview": self.simulation.summary()["empty_cluster_cell_preview"],
            "small_cell_preview": self.simulation.summary()["small_cell_preview"],
            "small_cluster_cell_preview": self.simulation.summary()["small_cluster_cell_preview"],
        }

    def to_dict(self) -> dict[str, Any]:
        row = self.row
        return {
            **self.summary(),
            "pass_reason": self.pass_reason,
            "failure_reasons": self.failure_reasons,
            "within_absolute_tolerance": self.within_absolute_tolerance,
            "within_sampling_error": self.within_sampling_error,
            "absolute_tolerance": self.absolute_tolerance,
            "z_tolerance": self.z_tolerance,
            "table": row.table,
            "panel": row.panel,
            "design": row.design,
            "mediator": row.mediator,
            "clusters": row.clusters,
            "bins": row.bins,
            "t": row.t,
            "bar_nu_lb": row.bar_nu_lb,
            "size_row": row.is_null_size_row,
            "target_median_independent_count_per_cell": self.target_median_independent_count_per_cell,
            "observed_mean_median_independent_count_per_cell": (
                self.observed_mean_median_independent_count_per_cell
            ),
            "cell_count_absolute_error": self.cell_count_absolute_error,
            "cell_count_absolute_tolerance": self.cell_count_absolute_tolerance,
            "cell_count_tolerance_active": self.cell_count_tolerance_active,
            "within_cell_count_tolerance": self.within_cell_count_tolerance,
            "cell_count_size_risk": self.cluster_cell_count.size_risk if self.cluster_cell_count is not None else None,
            "source_mixture_absolute_tolerance": self.source_mixture_absolute_tolerance,
            "source_mixture_tolerance_active": self.source_mixture_tolerance_active,
            "within_source_mixture_tolerance": self.within_source_mixture_tolerance,
        }


@dataclass(frozen=True)
class MonteCarloBenchmarkMatrixResult:
    """Matrix-level benchmark results for a batch of paper Monte Carlo cells."""

    simulations: tuple[MonteCarloSimulationResult, ...]
    diagnostics: tuple[MonteCarloBenchmarkDiagnostic, ...]

    @property
    def method(self) -> str:
        methods = {diagnostic.method for diagnostic in self.diagnostics}
        if len(methods) != 1:
            return "mixed"
        return next(iter(methods))

    @property
    def passes(self) -> bool:
        return all(diagnostic.passes for diagnostic in self.diagnostics)

    @property
    def failed_diagnostics(self) -> tuple[MonteCarloBenchmarkDiagnostic, ...]:
        return tuple(diagnostic for diagnostic in self.diagnostics if not diagnostic.passes)

    @property
    def size_diagnostics(self) -> tuple[MonteCarloBenchmarkDiagnostic, ...]:
        return tuple(diagnostic for diagnostic in self.diagnostics if diagnostic.row.is_null_size_row)

    @property
    def power_diagnostics(self) -> tuple[MonteCarloBenchmarkDiagnostic, ...]:
        return tuple(diagnostic for diagnostic in self.diagnostics if not diagnostic.row.is_null_size_row)

    def summary(self) -> dict[str, Any]:
        absolute_errors = [diagnostic.absolute_error for diagnostic in self.diagnostics]
        z_scores = [diagnostic.z_score for diagnostic in self.diagnostics]
        finite_z_scores = [score for score in z_scores if math.isfinite(float(score))]
        cell_count_errors = [
            diagnostic.cell_count_absolute_error
            for diagnostic in self.diagnostics
            if diagnostic.cell_count_absolute_error is not None
        ]
        source_mixture_errors = [
            diagnostic.source_mixture_absolute_error
            for diagnostic in self.diagnostics
            if diagnostic.source_mixture_absolute_error is not None
        ]
        source_mixture_standard_errors = [
            diagnostic.source_mixture_standard_error
            for diagnostic in self.diagnostics
            if diagnostic.source_mixture_standard_error is not None
        ]
        source_mixture_z_scores = [
            diagnostic.source_mixture_z_score
            for diagnostic in self.diagnostics
            if diagnostic.source_mixture_z_score is not None
        ]
        finite_source_mixture_z_scores = [
            score
            for score in source_mixture_z_scores
            if math.isfinite(float(score))
        ]
        return {
            "method": self.method,
            "benchmark_rows": len(self.diagnostics),
            "simulation_count": len(self.simulations),
            "total_draws": int(sum(len(simulation.draws) for simulation in self.simulations)),
            "passed_rows": int(sum(diagnostic.passes for diagnostic in self.diagnostics)),
            "failed_rows": int(sum(not diagnostic.passes for diagnostic in self.diagnostics)),
            "size_rows": int(sum(diagnostic.row.is_null_size_row for diagnostic in self.diagnostics)),
            "power_rows": int(sum(not diagnostic.row.is_null_size_row for diagnostic in self.diagnostics)),
            "max_absolute_error": (
                float(max(absolute_errors)) if absolute_errors else None
            ),
            "mean_absolute_error": (
                float(np.mean(absolute_errors)) if absolute_errors else None
            ),
            "max_z_score": float(max(finite_z_scores)) if finite_z_scores else None,
            "nonfinite_z_score_rows": int(
                sum(not math.isfinite(float(score)) for score in z_scores)
            ),
            "cell_count_checked_rows": int(sum(diagnostic.cell_count_tolerance_active for diagnostic in self.diagnostics)),
            "cell_count_failed_rows": int(
                sum(
                    diagnostic.cell_count_tolerance_active and not diagnostic.within_cell_count_tolerance
                    for diagnostic in self.diagnostics
                )
            ),
            "max_cell_count_absolute_error": (
                float(max(cell_count_errors)) if cell_count_errors else None
            ),
            "source_mixture_checked_rows": int(
                sum(diagnostic.source_mixture_tolerance_active for diagnostic in self.diagnostics)
            ),
            "source_mixture_failed_rows": int(
                sum(
                    diagnostic.source_mixture_tolerance_active
                    and not diagnostic.within_source_mixture_tolerance
                    for diagnostic in self.diagnostics
                )
            ),
            "max_source_mixture_absolute_error": (
                float(max(source_mixture_errors)) if source_mixture_errors else None
            ),
            "source_mixture_sampling_error_rows": int(len(source_mixture_standard_errors)),
            "max_source_mixture_standard_error": (
                float(max(source_mixture_standard_errors))
                if source_mixture_standard_errors
                else None
            ),
            "max_source_mixture_z_score": (
                float(max(finite_source_mixture_z_scores))
                if finite_source_mixture_z_scores
                else None
            ),
            "nonfinite_source_mixture_z_score_rows": int(
                sum(
                    not math.isfinite(float(score))
                    for score in source_mixture_z_scores
                )
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_export_payload({
            "summary": self.summary(),
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
            "simulations": [simulation.to_dict() for simulation in self.simulations],
        })

    def to_frame(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for diagnostic in self.diagnostics:
            row = diagnostic.row
            z_score, z_score_is_finite, z_score_nonfinite = _json_safe_float_with_marker(
                diagnostic.z_score
            )
            (
                source_mixture_z_score,
                source_mixture_z_score_is_finite,
                source_mixture_z_score_nonfinite,
            ) = _json_safe_optional_float_with_marker(diagnostic.source_mixture_z_score)
            rows.append(
                {
                    "simulation": diagnostic.simulation.design.name,
                    "method": diagnostic.method,
                    "table": row.table,
                    "panel": row.panel,
                    "design": row.design,
                    "mediator": row.mediator,
                    "clusters": row.clusters,
                    "bins": row.bins,
                    "t": row.t,
                    "bar_nu_lb": row.bar_nu_lb,
                    "target_rejection_rate": diagnostic.target_rejection_rate,
                    "observed_rejection_rate": diagnostic.observed_rejection_rate,
                    "absolute_error": diagnostic.absolute_error,
                    "rejection_rate_absolute_error": diagnostic.rejection_rate_absolute_error,
                    "monte_carlo_standard_error": diagnostic.monte_carlo_standard_error,
                    "z_score": z_score,
                    "z_score_is_finite": z_score_is_finite,
                    "z_score_nonfinite": z_score_nonfinite,
                    "passed": diagnostic.passes,
                    "pass_reason": diagnostic.pass_reason,
                    "failure_reasons": diagnostic.failure_reasons,
                    "within_absolute_tolerance": diagnostic.within_absolute_tolerance,
                    "within_sampling_error": diagnostic.within_sampling_error,
                    "absolute_tolerance": diagnostic.absolute_tolerance,
                    "z_tolerance": diagnostic.z_tolerance,
                    "target_median_independent_count_per_cell": (
                        diagnostic.target_median_independent_count_per_cell
                    ),
                    "observed_mean_median_independent_count_per_cell": (
                        diagnostic.observed_mean_median_independent_count_per_cell
                    ),
                    "cell_count_absolute_error": diagnostic.cell_count_absolute_error,
                    "cell_count_absolute_tolerance": diagnostic.cell_count_absolute_tolerance,
                    "cell_count_tolerance_active": diagnostic.cell_count_tolerance_active,
                    "within_cell_count_tolerance": diagnostic.within_cell_count_tolerance,
                    "cell_count_size_risk": (
                        diagnostic.cluster_cell_count.size_risk
                        if diagnostic.cluster_cell_count is not None
                        else None
                    ),
                    "target_t": row.t,
                    "observed_source_mixture_share": diagnostic.observed_source_mixture_share,
                    "source_mixture_absolute_error": diagnostic.source_mixture_absolute_error,
                    "source_mixture_effective_trials": diagnostic.source_mixture_effective_trials,
                    "source_mixture_standard_error": diagnostic.source_mixture_standard_error,
                    "source_mixture_z_score": source_mixture_z_score,
                    "source_mixture_z_score_is_finite": source_mixture_z_score_is_finite,
                    "source_mixture_z_score_nonfinite": source_mixture_z_score_nonfinite,
                    "source_mixture_absolute_tolerance": diagnostic.source_mixture_absolute_tolerance,
                    "source_mixture_tolerance_active": diagnostic.source_mixture_tolerance_active,
                    "within_source_mixture_tolerance": diagnostic.within_source_mixture_tolerance,
                    "replications": len(diagnostic.simulation.draws),
                    "seed": diagnostic.simulation.design.seed,
                    "size_row": row.is_null_size_row,
                }
            )
        return _json_safe_export_frame(rows)

    def failure_frame(self) -> pd.DataFrame:
        frame = self.to_frame()
        if frame.empty:
            return frame
        failed = ~frame["passed"].astype(bool)
        return frame.loc[failed].reset_index(drop=True)

    def raise_for_failures(self) -> None:
        failures = self.failure_frame()
        if failures.empty:
            return
        preview_columns = [
            "simulation",
            "target_rejection_rate",
            "observed_rejection_rate",
            "absolute_error",
            "z_score",
            "failure_reasons",
        ]
        preview = failures.loc[:, preview_columns].head(5).to_dict("records")
        raise AssertionError(
            f"Monte Carlo benchmark matrix failed {failures.shape[0]} of "
            f"{len(self.diagnostics)} rows: {preview}"
        )


@dataclass(frozen=True)
class MonteCarloBenchmarkPlanRunResult:
    """Execution result for a planned paper Monte Carlo benchmark slice."""

    plan: MonteCarloBenchmarkPlan
    matrix: MonteCarloBenchmarkMatrixResult
    data_source_diagnostics: tuple[MonteCarloBenchmarkDataSourceDiagnostic, ...] = ()
    scheduled_cells: tuple[MonteCarloBenchmarkPlanRerunCell, ...] = ()
    manifest_seed: int | None = None

    @property
    def executable_slice_passes(self) -> bool:
        return self.matrix.passes

    @property
    def full_plan_passes(self) -> bool:
        return self.executable_slice_passes and not self.blocked_rows

    @property
    def passes(self) -> bool:
        return self.full_plan_passes

    @property
    def blocked_rows(self) -> tuple[MonteCarloBlockedBenchmarkRow, ...]:
        return self.plan.blocked_rows

    @property
    def failed_diagnostics(self) -> tuple[MonteCarloBenchmarkDiagnostic, ...]:
        return self.matrix.failed_diagnostics

    def summary(self) -> dict[str, Any]:
        plan_summary = self.plan.summary()
        matrix_summary = self.matrix.summary()
        planned_draws = (
            int(sum(cell.planned_draws for cell in self.scheduled_cells))
            if self.scheduled_cells
            else plan_summary["planned_draws"]
        )
        summary = {
            "method": self.plan.method,
            "planned_rows": plan_summary["total_result_rows"],
            "executable_rows": plan_summary["executable_rows"],
            "blocked_rows": plan_summary["blocked_rows"],
            "executable_designs": plan_summary["executable_designs"],
            "default_replications": plan_summary["default_replications"],
            "planned_draws": planned_draws,
            "simulation_count": matrix_summary["simulation_count"],
            "total_draws": matrix_summary["total_draws"],
            "passed_rows": matrix_summary["passed_rows"],
            "failed_rows": matrix_summary["failed_rows"],
            "executable_slice_passes": self.executable_slice_passes,
            "full_plan_passes": self.full_plan_passes,
            "cell_count_checked_rows": matrix_summary["cell_count_checked_rows"],
            "cell_count_failed_rows": matrix_summary["cell_count_failed_rows"],
            "max_cell_count_absolute_error": matrix_summary["max_cell_count_absolute_error"],
            "source_mixture_checked_rows": matrix_summary["source_mixture_checked_rows"],
            "source_mixture_failed_rows": matrix_summary["source_mixture_failed_rows"],
            "max_source_mixture_absolute_error": matrix_summary["max_source_mixture_absolute_error"],
            "source_mixture_sampling_error_rows": matrix_summary["source_mixture_sampling_error_rows"],
            "max_source_mixture_standard_error": matrix_summary["max_source_mixture_standard_error"],
            "max_source_mixture_z_score": matrix_summary["max_source_mixture_z_score"],
            "blocked_reasons": plan_summary["blocked_reasons"],
        }
        if self.scheduled_cells:
            self._validate_scheduled_simulation_alignment()
            summary.update(
                _benchmark_target_precision_summary_from_scheduled_cells(
                    self.scheduled_cells
                )
            )
            scheduled_z_tolerance = _single_diagnostic_field_or_none(
                self.matrix.diagnostics,
                field_name="z_tolerance",
            )
            scheduled_source_mixture_tolerance = _single_diagnostic_field_or_none(
                self.matrix.diagnostics,
                field_name="source_mixture_absolute_tolerance",
            )
            summary.update(
                _source_mixture_precision_summary_from_scheduled_cells(
                    self.scheduled_cells,
                    self.data_source_diagnostics,
                    z_tolerance=scheduled_z_tolerance,
                    source_mixture_absolute_tolerance=scheduled_source_mixture_tolerance,
                )
            )
        else:
            summary.update(
                _benchmark_target_precision_summary(
                    self.plan.executable_cells,
                    replications=self.plan.replications,
                )
            )
        summary.update(_data_source_diagnostic_summary(self.data_source_diagnostics))
        return summary

    def to_dict(
        self,
        *,
        owner: str = "Phase 7 Monte Carlo verification hardening",
        rerun_command: str | None = None,
    ) -> dict[str, Any]:
        """Serialize the run result with optional owner-facing closeout context."""

        return _json_safe_export_payload({
            "summary": self.summary(),
            "blocker_packet": self.blocker_packet(
                owner=owner,
                rerun_command=rerun_command,
            ),
            "paper_acceptance_gate": self.paper_acceptance_gate(),
            "milestone_completion_summary": self.milestone_completion_summary(
                owner=owner,
                rerun_command=rerun_command,
            ),
            "data_source_diagnostics": [
                diagnostic.to_dict() for diagnostic in self.data_source_diagnostics
            ],
            "scheduled_cells": [cell.to_dict() for cell in self.scheduled_cells],
            "manifest_seed": self.manifest_seed,
            "plan": self.plan.to_dict(),
            "matrix": self.matrix.to_dict(),
        })

    def to_frame(self) -> pd.DataFrame:
        matrix_frame = self.matrix.to_frame()
        if not matrix_frame.empty:
            matrix_frame = matrix_frame.copy()
            matrix_frame["status"] = "executed"
            matrix_frame["blocked_reason"] = None
            if self.scheduled_cells:
                if len(self.scheduled_cells) != matrix_frame.shape[0]:
                    raise ValueError(
                        "to_frame requires one benchmark diagnostic per scheduled cell; "
                        f"got {matrix_frame.shape[0]} result rows for "
                        f"{len(self.scheduled_cells)} scheduled cells."
                    )
                self._validate_scheduled_simulation_alignment()
                matrix_frame["planned_replications"] = [
                    cell.replications for cell in self.scheduled_cells
                ]
                matrix_frame["planned_draws"] = [
                    cell.planned_draws for cell in self.scheduled_cells
                ]
                matrix_frame["planned_bootstrap_replications"] = [
                    cell.bootstrap_replications for cell in self.scheduled_cells
                ]
            else:
                matrix_frame["planned_replications"] = self.plan.replications
                matrix_frame["planned_draws"] = self.plan.replications
                matrix_frame["planned_bootstrap_replications"] = 0
            matrix_frame["bootstrap_required"] = matrix_frame["method"].map(
                _method_requires_bootstrap
            )
            matrix_frame["expected_bootstrap_replications"] = matrix_frame["method"].map(
                lambda method: _PAPER_BOOTSTRAP_REPLICATIONS if _method_requires_bootstrap(method) else 0
            )
            matrix_frame["bootstrap_replication_shortfall"] = (
                matrix_frame["expected_bootstrap_replications"]
                - matrix_frame["planned_bootstrap_replications"]
            ).clip(lower=0)
        blocked_frame = self.blocked_frame()
        if blocked_frame.empty:
            return _json_safe_export_frame(matrix_frame.to_dict("records"))
        if matrix_frame.empty:
            return _json_safe_export_frame(blocked_frame.to_dict("records"))
        columns = list(dict.fromkeys([*matrix_frame.columns, *blocked_frame.columns]))
        records = [
            *matrix_frame.reindex(columns=columns).to_dict("records"),
            *blocked_frame.reindex(columns=columns).to_dict("records"),
        ]
        return _json_safe_export_frame(records, columns=columns)

    def sample_shape_frame(self) -> pd.DataFrame:
        """Return per-executed-cell sample-size and arm-shape diagnostics."""

        if self.scheduled_cells:
            if len(self.matrix.simulations) != len(self.scheduled_cells):
                raise ValueError(
                    "sample_shape_frame requires one simulation result per scheduled cell; "
                    f"got {len(self.matrix.simulations)} simulations for "
                    f"{len(self.scheduled_cells)} scheduled cells."
                )
            if len(self.matrix.diagnostics) != len(self.scheduled_cells):
                raise ValueError(
                    "sample_shape_frame requires one benchmark diagnostic per scheduled cell; "
                    f"got {len(self.matrix.diagnostics)} diagnostics for "
                    f"{len(self.scheduled_cells)} scheduled cells."
                )
            self._validate_scheduled_simulation_alignment()

        rows: list[dict[str, Any]] = []
        for index, simulation in enumerate(self.matrix.simulations):
            shape = dict(simulation.sample_shape_summary())
            simulation_name = shape.pop("design")
            row: dict[str, Any] = {
                "simulation": simulation_name,
                "status": "executed",
                "blocked_reason": None,
                **shape,
            }
            if self.scheduled_cells and index < len(self.scheduled_cells):
                scheduled_cell = self.scheduled_cells[index]
                cell = scheduled_cell.cell
                row.update(
                    {
                        "table": cell.table,
                        "panel": cell.panel,
                        "paper_design": cell.design,
                        "mediator": cell.mediator,
                        "clusters": cell.clusters,
                        "bins": cell.bins,
                        "t": cell.t,
                        "target_rejection_rate": cell.target_rejection_rate,
                        "planned_replications": scheduled_cell.replications,
                        "planned_draws": scheduled_cell.planned_draws,
                        "seed": scheduled_cell.seed,
                    }
                )
            elif index < len(self.matrix.diagnostics):
                diagnostic = self.matrix.diagnostics[index]
                row_identity = diagnostic.row
                row.update(
                    {
                        "table": row_identity.table,
                        "panel": row_identity.panel,
                        "paper_design": row_identity.design,
                        "mediator": row_identity.mediator,
                        "clusters": row_identity.clusters,
                        "bins": row_identity.bins,
                        "t": row_identity.t,
                        "target_rejection_rate": row_identity.rejection_rates.get(
                            diagnostic.method
                        ),
                    }
                )
            rows.append(row)
        frame = _json_safe_export_frame(rows)
        numeric_columns = (
            "clusters",
            "bins",
            "t",
            "target_rejection_rate",
            "planned_replications",
            "planned_draws",
            "seed",
            "replications",
            "min_n_obs_used",
            "max_n_obs_used",
            "min_control_observations",
            "max_control_observations",
            "min_treated_observations",
            "max_treated_observations",
            "min_n_clusters_used",
            "max_n_clusters_used",
            "min_control_clusters",
            "max_control_clusters",
            "min_treated_clusters",
            "max_treated_clusters",
        )
        for column in numeric_columns:
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame

    def sample_shape_summary(self) -> dict[str, Any]:
        """Return plan-level sample-size and arm-shape diagnostics."""

        frame = self.sample_shape_frame()
        summary: dict[str, Any] = {
            "sample_shape_rows": int(frame.shape[0]),
            "sample_shape_blocked_rows": len(self.blocked_rows),
        }
        if frame.empty:
            return summary

        if "clustered" in frame:
            clustered = frame["clustered"].map(
                lambda value: False if value is None or pd.isna(value) else bool(value)
            )
            summary["clustered_sample_shape_rows"] = int(clustered.sum())
            summary["unclustered_sample_shape_rows"] = int((~clustered).sum())

        if "replications" in frame:
            replications = pd.to_numeric(frame["replications"], errors="coerce").dropna()
            if not replications.empty:
                summary["min_sample_shape_replications"] = int(replications.min())
                summary["max_sample_shape_replications"] = int(replications.max())

        for column in (
            "n_obs_used",
            "control_observations",
            "treated_observations",
            "n_clusters_used",
            "control_clusters",
            "treated_clusters",
        ):
            min_column = f"min_{column}"
            max_column = f"max_{column}"
            if min_column in frame:
                values = pd.to_numeric(frame[min_column], errors="coerce").dropna()
                if not values.empty:
                    summary[min_column] = int(values.min())
            if max_column in frame:
                values = pd.to_numeric(frame[max_column], errors="coerce").dropna()
                if not values.empty:
                    summary[max_column] = int(values.max())

        return summary

    def acceptance_frame(self) -> pd.DataFrame:
        """Return one row-level audit frame for paper-cell acceptance checks."""

        frame = self.to_frame().reset_index(drop=True)
        if frame.empty:
            return frame

        frame = frame.copy()
        self._validate_scheduled_simulation_alignment()
        self._append_acceptance_columns(
            frame,
            self.target_precision_frame().reset_index(drop=True),
            columns=(
                "target_mc_standard_error",
                "target_mc_error_band",
                "target_tolerance_below_error_band",
            ),
            source_name="target precision",
        )
        self._append_acceptance_columns(
            frame,
            self.source_mixture_precision_frame().reset_index(drop=True),
            columns=(
                "source_mixture_trials_per_draw",
                "source_mixture_effective_trials",
                "source_mixture_mc_standard_error",
                "source_mixture_mc_error_band",
                "source_mixture_tolerance_below_error_band",
                "data_source_ready",
                "data_source_blocking_reasons",
            ),
            aliases={
                "source_mixture_effective_trials": "planned_source_mixture_effective_trials",
            },
            source_name="source-mixture precision",
        )
        self._append_acceptance_columns(
            frame,
            self.replication_budget_frame().reset_index(drop=True),
            columns=(
                "target_required_replications_for_tolerance",
                "target_replication_shortfall",
                "source_mixture_required_replications_for_tolerance",
                "source_mixture_replication_shortfall",
            ),
            source_name="replication budget",
        )
        self._append_acceptance_columns(
            frame,
            self.cell_count_policy_frame().reset_index(drop=True),
            columns=(
                "cell_count_policy_available",
                "target_median_independent_clusters_per_cell",
                "cell_count_size_risk_threshold",
                "cell_count_policy_size_risk",
                "recommended_by_cell_count",
                "bin_policy",
                "paper_rule",
            ),
            source_name="cell-count policy",
        )
        self._append_sample_shape_acceptance_columns(frame, self.sample_shape_frame())
        return frame

    def acceptance_summary(self) -> dict[str, Any]:
        """Return a compact summary of the row-level paper acceptance audit."""

        frame = self.acceptance_frame()
        summary: dict[str, Any] = {
            "acceptance_rows": int(frame.shape[0]),
            "executed_rows": 0,
            "blocked_rows": 0,
            "passed_executed_rows": 0,
            "failed_executed_rows": 0,
            "executable_slice_passes": self.executable_slice_passes,
            "full_plan_passes": self.full_plan_passes,
        }
        if frame.empty:
            return summary

        status = frame["status"] if "status" in frame else pd.Series("", index=frame.index)
        executed = status.eq("executed")
        blocked = status.eq("blocked")
        passed = frame["passed"].map(_truthy_value) if "passed" in frame else pd.Series(False, index=frame.index)

        summary.update(
            {
                "executed_rows": int(executed.sum()),
                "blocked_rows": int(blocked.sum()),
                "passed_executed_rows": int((executed & passed).sum()),
                "failed_executed_rows": int((executed & ~passed).sum()),
            }
        )

        executed_frame = frame.loc[executed]
        for column, output_column in (
            ("target_mc_standard_error", "target_precision_rows"),
            ("source_mixture_mc_standard_error", "source_mixture_precision_rows"),
            ("min_n_obs_used", "sample_shape_rows"),
        ):
            summary[output_column] = (
                int(executed_frame[column].notna().sum())
                if column in executed_frame
                else 0
            )

        for column, output_column in (
            ("target_tolerance_below_error_band", "target_tolerance_below_error_band_rows"),
            ("source_mixture_tolerance_below_error_band", "source_mixture_tolerance_below_error_band_rows"),
            ("data_source_ready", "data_source_ready_rows"),
            ("cell_count_policy_available", "cell_count_policy_rows"),
            ("cell_count_policy_size_risk", "cell_count_policy_size_risk_rows"),
            ("recommended_by_cell_count", "cell_count_recommended_rows"),
        ):
            summary[output_column] = _count_truthy_values(executed_frame[column]) if column in executed_frame else 0

        if "cell_count_policy_available" in frame:
            summary["cell_count_policy_unavailable_rows"] = int(
                (~frame["cell_count_policy_available"].map(_truthy_value)).sum()
            )

        for column, output_column in (
            ("target_required_replications_for_tolerance", "target_budget_rows"),
            ("source_mixture_required_replications_for_tolerance", "source_mixture_budget_rows"),
            ("target_replication_shortfall", "target_shortfall_rows"),
            ("source_mixture_replication_shortfall", "source_mixture_shortfall_rows"),
        ):
            if output_column.endswith("_shortfall_rows"):
                summary[output_column] = _positive_numeric_column_count(executed_frame, column)
            else:
                summary[output_column] = _numeric_column_count(executed_frame, column)

        for column, output_column in (
            ("target_mc_standard_error", "max_target_mc_standard_error"),
            ("rejection_rate_absolute_error", "max_rejection_rate_absolute_error"),
            ("source_mixture_mc_standard_error", "max_source_mixture_mc_standard_error"),
            ("target_required_replications_for_tolerance", "max_target_required_replications"),
            ("source_mixture_required_replications_for_tolerance", "max_source_mixture_required_replications"),
            ("target_replication_shortfall", "max_target_replication_shortfall"),
            ("source_mixture_replication_shortfall", "max_source_mixture_replication_shortfall"),
            ("target_median_independent_clusters_per_cell", "min_target_median_independent_clusters_per_cell"),
            ("source_mixture_effective_trials", "min_source_mixture_effective_trials"),
            ("source_mixture_effective_trials", "max_source_mixture_effective_trials"),
            ("planned_source_mixture_effective_trials", "min_planned_source_mixture_effective_trials"),
            ("planned_source_mixture_effective_trials", "max_planned_source_mixture_effective_trials"),
            ("min_n_obs_used", "min_n_obs_used"),
            ("max_n_obs_used", "max_n_obs_used"),
            ("min_n_clusters_used", "min_n_clusters_used"),
            ("max_n_clusters_used", "max_n_clusters_used"),
        ):
            if column in executed_frame:
                summary[output_column] = _numeric_min_or_max(
                    executed_frame[column],
                    choose="max" if output_column.startswith("max_") else "min",
                )
            else:
                summary[output_column] = None
        return summary

    def blocker_packet(
        self,
        *,
        owner: str = "Phase 7 Monte Carlo verification hardening",
        rerun_command: str | None = None,
        runtime_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Return an owner-facing packet for post-run paper-matrix blockers."""

        acceptance_summary = self.acceptance_summary()
        run_summary = self.summary()
        failure_frame = self.failure_frame()
        acceptance_frame = self.acceptance_frame()
        precision_budget = _precision_budget_summary_from_frames(
            target_frame=acceptance_frame,
            source_mixture_frame=acceptance_frame,
        )
        replication_budget = _replication_budget_summary_from_frame(
            self.replication_budget_frame()
        )
        cell_count_policy = _cell_count_policy_summary_from_frame(
            acceptance_frame
        )
        preview_columns = (
            "status",
            "simulation",
            "table",
            "design",
            "mediator",
            "clusters",
            "bins",
            "t",
            "target_rejection_rate",
            "observed_rejection_rate",
            "absolute_error",
            "z_score",
            "failure_reasons",
            "blocked_reason",
        )
        failure_preview = failure_frame.loc[
            :,
            [column for column in preview_columns if column in failure_frame],
        ].head(5).to_dict("records")

        return {
            "owner": owner,
            "cleared_same_line_items": {
                "data_source_ready_designs": run_summary.get("data_source_ready_designs", 0),
                "executed_rows": acceptance_summary["executed_rows"],
                "passed_executed_rows": acceptance_summary["passed_executed_rows"],
                "total_draws": run_summary["total_draws"],
                "executable_slice_passes": acceptance_summary["executable_slice_passes"],
            },
            "blocked_same_line_items": {
                "failed_executed_rows": acceptance_summary["failed_executed_rows"],
                "blocked_rows": acceptance_summary["blocked_rows"],
                "blocked_reason_counts": run_summary["blocked_reasons"],
                "full_plan_passes": acceptance_summary["full_plan_passes"],
            },
            "verified_boundary": {
                "method": run_summary["method"],
                "executable_designs": run_summary["executable_designs"],
                "default_replications": run_summary["default_replications"],
                "planned_draws": run_summary["planned_draws"],
                "total_draws": run_summary["total_draws"],
                "runtime_seconds": runtime_seconds,
                "max_target_mc_standard_error": acceptance_summary.get(
                    "max_target_mc_standard_error"
                ),
                "max_source_mixture_mc_standard_error": acceptance_summary.get(
                    "max_source_mixture_mc_standard_error"
                ),
                "min_n_obs_used": acceptance_summary.get("min_n_obs_used"),
                "max_n_obs_used": acceptance_summary.get("max_n_obs_used"),
                "min_n_clusters_used": acceptance_summary.get("min_n_clusters_used"),
                "max_n_clusters_used": acceptance_summary.get("max_n_clusters_used"),
            },
            "evidence_checked": {
                "data_source_complete_case_rows": run_summary.get(
                    "data_source_complete_case_rows",
                    {},
                ),
                "data_source_source_clusters": run_summary.get(
                    "data_source_source_clusters",
                    {},
                ),
                "data_source_blocking_reasons": run_summary.get(
                    "data_source_blocking_reasons",
                    {},
                ),
                "data_source_complete_case_rows_by_source_key": run_summary.get(
                    "data_source_complete_case_rows_by_source_key",
                    {},
                ),
                "data_source_source_clusters_by_source_key": run_summary.get(
                    "data_source_source_clusters_by_source_key",
                    {},
                ),
                "data_source_blocking_reasons_by_source_key": run_summary.get(
                    "data_source_blocking_reasons_by_source_key",
                    {},
                ),
                "max_target_mc_standard_error": acceptance_summary.get(
                    "max_target_mc_standard_error"
                ),
                "max_source_mixture_mc_standard_error": acceptance_summary.get(
                    "max_source_mixture_mc_standard_error"
                ),
                "max_rejection_rate_absolute_error": acceptance_summary.get(
                    "max_rejection_rate_absolute_error"
                ),
                "cell_count_checked_rows": acceptance_summary.get(
                    "cell_count_checked_rows"
                ),
                "cell_count_failed_rows": acceptance_summary.get(
                    "cell_count_failed_rows"
                ),
                "source_mixture_checked_rows": acceptance_summary.get(
                    "source_mixture_checked_rows"
                ),
                "source_mixture_failed_rows": acceptance_summary.get(
                    "source_mixture_failed_rows"
                ),
                "runtime_seconds": runtime_seconds,
            },
            "precision_budget": precision_budget,
            "replication_budget": replication_budget,
            "cell_count_policy": cell_count_policy,
            "paper_acceptance_gate": self.paper_acceptance_gate(),
            "failure_preview": failure_preview,
            "rerun_command": rerun_command,
            "exit_criteria": "paper_acceptance_gate.gate_passes == True",
        }

    def paper_acceptance_gate(self) -> dict[str, Any]:
        """Return a machine-readable post-run full-paper acceptance verdict."""

        acceptance_frame = self.acceptance_frame()
        acceptance_summary = self.acceptance_summary()
        run_summary = self.summary()
        replication_budget = _replication_budget_summary_from_frame(
            self.replication_budget_frame()
        )
        bootstrap_budget = _bootstrap_budget_summary_from_frame(acceptance_frame)
        cell_count_policy = _cell_count_policy_summary_from_frame(acceptance_frame)
        tolerance_contract = _paper_acceptance_tolerance_contract_summary(
            acceptance_frame
        )
        blocking_conditions = {
            "paper_coverage_unknown": int(not self.plan.paper_coverage_known),
            "paper_coverage_shortfall_rows": self.plan.paper_coverage_shortfall_rows or 0,
            "blocked_rows": acceptance_summary["blocked_rows"],
            "failed_executed_rows": acceptance_summary["failed_executed_rows"],
            "data_source_blocked_designs": run_summary.get("data_source_blocked_designs", 0),
            "target_replication_shortfall_rows": replication_budget["target_shortfall_rows"],
            "source_mixture_replication_shortfall_rows": replication_budget[
                "source_mixture_shortfall_rows"
            ],
            "bootstrap_replication_shortfall_rows": bootstrap_budget[
                "bootstrap_shortfall_rows"
            ],
            "documented_tolerance_contract_missing": tolerance_contract[
                "documented_tolerance_contract_missing"
            ],
            "cell_count_policy_size_risk_rows": cell_count_policy[
                "cell_count_policy_size_risk_rows"
            ],
        }
        active_blocking_conditions = _active_paper_acceptance_blocking_conditions(
            blocking_conditions
        )
        gate_passes = acceptance_summary["full_plan_passes"] and not active_blocking_conditions
        blocking_condition_rows = _paper_acceptance_blocker_rows_from_conditions(
            blocking_conditions
        )
        return {
            "stage": "post_run",
            "gate_passes": gate_passes,
            "verdict": "pass" if gate_passes else "blocked",
            "method": run_summary["method"],
            **self.plan.paper_coverage_summary(),
            "planned_rows": run_summary["planned_rows"],
            "executed_rows": acceptance_summary["executed_rows"],
            "blocked_rows": acceptance_summary["blocked_rows"],
            "passed_executed_rows": acceptance_summary["passed_executed_rows"],
            "failed_executed_rows": acceptance_summary["failed_executed_rows"],
            "planned_draws": run_summary["planned_draws"],
            "total_draws": run_summary["total_draws"],
            "executable_slice_passes": acceptance_summary["executable_slice_passes"],
            "full_plan_passes": acceptance_summary["full_plan_passes"],
            "tolerance_contract_status": tolerance_contract[
                "tolerance_contract_status"
            ],
            "bootstrap_budget": bootstrap_budget,
            "bootstrap_replication_shortfall_rows": bootstrap_budget[
                "bootstrap_shortfall_rows"
            ],
            "documented_tolerance_contract_missing": tolerance_contract[
                "documented_tolerance_contract_missing"
            ],
            "tolerance_contract": tolerance_contract,
            "blocking_conditions": blocking_conditions,
            "active_blocking_conditions": active_blocking_conditions,
            "active_blocking_condition_count": len(active_blocking_conditions),
            "blocking_condition_rows": list(blocking_condition_rows),
            "blocked_reason_counts": run_summary["blocked_reasons"],
            "next_action": _paper_acceptance_next_action(
                blocking_conditions,
                data_sources_ready=run_summary.get("data_source_blocked_designs", 0) == 0,
                executed=True,
            ),
        }

    def paper_acceptance_blocker_frame(self) -> pd.DataFrame:
        """Return row-level release blockers from the post-run acceptance gate."""

        return _json_safe_export_frame(
            _paper_acceptance_gate_blocker_rows(
                self.paper_acceptance_gate(),
                evidence_by_condition=self._paper_acceptance_blocker_evidence(),
            )
        )

    def _paper_acceptance_blocker_evidence(self) -> dict[str, dict[str, Any]]:
        acceptance_summary = self.acceptance_summary()
        run_summary = self.summary()
        acceptance_frame = self.acceptance_frame()
        precision_budget = _precision_budget_summary_from_frames(
            target_frame=acceptance_frame,
            source_mixture_frame=acceptance_frame,
        )
        replication_budget = _replication_budget_summary_from_frame(
            self.replication_budget_frame()
        )
        bootstrap_budget = _bootstrap_budget_summary_from_frame(acceptance_frame)
        cell_count_policy = _cell_count_policy_summary_from_frame(acceptance_frame)
        tolerance_contract = _paper_acceptance_tolerance_contract_summary(
            acceptance_frame
        )
        return _paper_acceptance_blocker_evidence_from_summaries(
            paper_coverage=self.plan.paper_coverage_summary(),
            blocked_reason_counts=run_summary["blocked_reasons"],
            data_source_summary={
                "data_source_blocked_designs": run_summary.get("data_source_blocked_designs", 0),
                "data_source_blocking_reasons": run_summary.get("data_source_blocking_reasons", {}),
            },
            precision_budget=precision_budget,
            replication_budget=replication_budget,
            bootstrap_budget=bootstrap_budget,
            cell_count_policy=cell_count_policy,
            tolerance_contract=tolerance_contract,
            execution_summary={
                **run_summary,
                **acceptance_summary,
            },
        )

    def unresolved_paper_row_frame(self) -> pd.DataFrame:
        """Return unresolved paper rows that remain after execution."""

        return _paper_acceptance_unresolved_row_frame(
            self.acceptance_frame(),
            source="post_run",
            include_failed_executed_rows=True,
        )

    def milestone_completion_frame(
        self,
        *,
        owner: str = "Phase 7 Monte Carlo verification hardening",
        rerun_command: str | None = None,
    ) -> pd.DataFrame:
        """Return a row-level closeout frame for the executed rerun verdict."""

        return _paper_acceptance_gate_completion_frame(
            self.milestone_completion_summary(
                owner=owner,
                rerun_command=rerun_command,
            )
        )

    def milestone_completion_summary(
        self,
        *,
        owner: str = "Phase 7 Monte Carlo verification hardening",
        rerun_command: str | None = None,
    ) -> dict[str, Any]:
        """Return the release/milestone completion blocker summary after execution."""

        return _paper_acceptance_gate_completion_summary(
            self.paper_acceptance_gate(),
            owner=owner,
            rerun_command=rerun_command,
        )

    def raise_for_milestone_completion_blockers(
        self,
        *,
        owner: str = "Phase 7 Monte Carlo verification hardening",
        rerun_command: str | None = None,
    ) -> None:
        """Raise if this executed rerun does not clear the full-paper closeout gate."""

        _raise_for_milestone_completion_blockers(
            self.milestone_completion_summary(
                owner=owner,
                rerun_command=rerun_command,
            )
        )

    def _validate_scheduled_simulation_alignment(self) -> None:
        if not self.scheduled_cells:
            return

        expected_names = [
            _benchmark_matrix_design_name(
                table=scheduled_cell.cell.table,
                design=scheduled_cell.cell.design,
                mediator=scheduled_cell.cell.mediator,
                clusters=scheduled_cell.cell.clusters,
                bins=scheduled_cell.cell.bins,
                t=scheduled_cell.cell.t,
            )
            for scheduled_cell in self.scheduled_cells
        ]
        simulation_names = [simulation.design.name for simulation in self.matrix.simulations]
        diagnostic_names = [diagnostic.simulation.design.name for diagnostic in self.matrix.diagnostics]
        expected_diagnostic_rows = [
            _benchmark_cell_paper_identity(scheduled_cell.cell)
            for scheduled_cell in self.scheduled_cells
        ]
        diagnostic_rows = [
            _benchmark_diagnostic_paper_identity(diagnostic)
            for diagnostic in self.matrix.diagnostics
        ]

        if simulation_names != expected_names:
            raise ValueError(
                "Monte Carlo benchmark run result simulation order drifted from scheduled cells: "
                f"expected {expected_names}, got {simulation_names}."
            )
        if diagnostic_names != expected_names:
            raise ValueError(
                "Monte Carlo benchmark run result diagnostic order drifted from scheduled cells: "
                f"expected {expected_names}, got {diagnostic_names}."
            )
        if diagnostic_rows != expected_diagnostic_rows:
            raise ValueError(
                "Monte Carlo benchmark run result diagnostic paper row drifted from scheduled cells: "
                f"expected {expected_diagnostic_rows}, got {diagnostic_rows}."
            )

    @staticmethod
    def _append_acceptance_columns(
        frame: pd.DataFrame,
        supplement: pd.DataFrame,
        *,
        columns: tuple[str, ...],
        aliases: dict[str, str] | None = None,
        source_name: str,
    ) -> None:
        if supplement.empty:
            return
        if supplement.shape[0] != frame.shape[0]:
            raise ValueError(
                "acceptance_frame requires row-aligned "
                f"{source_name} rows; got {supplement.shape[0]} rows for "
                f"{frame.shape[0]} result rows."
            )
        for column in columns:
            if column not in supplement.columns:
                continue
            target_column = None if aliases is None else aliases.get(column)
            if target_column is None:
                target_column = column
            if target_column in frame.columns:
                continue
            frame[target_column] = supplement[column]

    @staticmethod
    def _append_sample_shape_acceptance_columns(
        frame: pd.DataFrame,
        sample_shape_frame: pd.DataFrame,
    ) -> None:
        if sample_shape_frame.empty:
            return
        if "simulation" not in frame.columns:
            raise ValueError("acceptance_frame requires simulation names to align sample-shape rows.")

        identity_columns = {
            "simulation",
            "status",
            "blocked_reason",
            "table",
            "panel",
            "paper_design",
            "mediator",
            "clusters",
            "bins",
            "t",
            "target_rejection_rate",
            "planned_replications",
            "planned_draws",
            "seed",
            "replications",
        }
        shape_columns = [
            column
            for column in sample_shape_frame.columns
            if column not in identity_columns and column not in frame.columns
        ]
        for column in shape_columns:
            frame[column] = None

        for _, shape_row in sample_shape_frame.iterrows():
            matches = frame.index[frame["simulation"].eq(shape_row["simulation"])].tolist()
            if len(matches) != 1:
                raise ValueError(
                    "acceptance_frame requires exactly one result row for each "
                    f"sample-shape simulation; got {len(matches)} matches for "
                    f"{shape_row['simulation']!r}."
                )
            frame_index = matches[0]
            for column in shape_columns:
                frame.at[frame_index, column] = shape_row[column]

    def blocked_frame(self) -> pd.DataFrame:
        """Return blocked paper rows preserved by the executed benchmark plan."""

        plan_frame = self.plan.to_frame()
        if plan_frame.empty:
            return plan_frame
        return plan_frame.loc[plan_frame["status"] == "blocked"].reset_index(drop=True)

    def failure_frame(self) -> pd.DataFrame:
        frame = self.to_frame()
        if frame.empty:
            return frame

        failed = pd.Series(False, index=frame.index)
        if "passed" in frame:
            failed = failed | frame["passed"].eq(False)
        if "status" in frame:
            failed = failed | frame["status"].eq("blocked")
        return frame.loc[failed].reset_index(drop=True)

    def target_precision_frame(
        self,
        *,
        absolute_tolerance: float | None = None,
        z_tolerance: float | None = None,
    ) -> pd.DataFrame:
        """Return row-level target rejection-rate precision for the executed budget."""

        resolved_absolute_tolerance = (
            absolute_tolerance
            if absolute_tolerance is not None
            else _single_diagnostic_field_or_none(
                self.matrix.diagnostics,
                field_name="absolute_tolerance",
            )
        )
        resolved_z_tolerance = (
            z_tolerance
            if z_tolerance is not None
            else _single_diagnostic_field_or_none(
                self.matrix.diagnostics,
                field_name="z_tolerance",
            )
        )
        if self.scheduled_cells:
            self._validate_scheduled_simulation_alignment()
            _validate_target_precision_tolerances(
                absolute_tolerance=resolved_absolute_tolerance,
                z_tolerance=resolved_z_tolerance,
            )
            return _json_safe_export_frame(
                _target_precision_rows_from_scheduled_cells(
                    self.scheduled_cells,
                    self.blocked_rows,
                    absolute_tolerance=resolved_absolute_tolerance,
                    z_tolerance=resolved_z_tolerance,
                )
            )
        return self.plan.target_precision_frame(
            absolute_tolerance=resolved_absolute_tolerance,
            z_tolerance=resolved_z_tolerance,
        )

    def source_mixture_precision_frame(
        self,
        *,
        z_tolerance: float | None = None,
        source_mixture_absolute_tolerance: float | None = None,
    ) -> pd.DataFrame:
        """Return row-level precision for the executed empirical-mixture t mechanism."""

        resolved_z_tolerance = (
            z_tolerance
            if z_tolerance is not None
            else _single_diagnostic_field_or_none(
                self.matrix.diagnostics,
                field_name="z_tolerance",
            )
        )
        resolved_source_mixture_tolerance = (
            source_mixture_absolute_tolerance
            if source_mixture_absolute_tolerance is not None
            else _single_diagnostic_field_or_none(
                self.matrix.diagnostics,
                field_name="source_mixture_absolute_tolerance",
            )
        )
        if resolved_z_tolerance is not None and resolved_z_tolerance <= 0:
            raise ValueError("z_tolerance must be positive when provided.")
        if resolved_source_mixture_tolerance is not None and resolved_source_mixture_tolerance < 0:
            raise ValueError("source_mixture_absolute_tolerance must be non-negative when provided.")

        if self.scheduled_cells:
            self._validate_scheduled_simulation_alignment()
            return _json_safe_export_frame(
                _source_mixture_precision_rows_from_scheduled_cells(
                    self.scheduled_cells,
                    self.blocked_rows,
                    self.data_source_diagnostics,
                    z_tolerance=resolved_z_tolerance,
                    source_mixture_absolute_tolerance=resolved_source_mixture_tolerance,
                )
            )
        return _json_safe_export_frame(
            _source_mixture_precision_rows_from_plan_cells(
                self.plan.executable_cells,
                self.blocked_rows,
                self.data_source_diagnostics,
                replications=self.plan.replications,
                z_tolerance=resolved_z_tolerance,
                source_mixture_absolute_tolerance=resolved_source_mixture_tolerance,
            )
        )

    def replication_budget_frame(
        self,
        *,
        absolute_tolerance: float | None = None,
        z_tolerance: float | None = None,
        source_mixture_absolute_tolerance: float | None = None,
    ) -> pd.DataFrame:
        """Return row-level replication budgets for the executed rerun."""

        resolved_absolute_tolerance = (
            absolute_tolerance
            if absolute_tolerance is not None
            else _single_diagnostic_field_or_none(
                self.matrix.diagnostics,
                field_name="absolute_tolerance",
            )
        )
        resolved_z_tolerance = (
            z_tolerance
            if z_tolerance is not None
            else _single_diagnostic_field_or_none(
                self.matrix.diagnostics,
                field_name="z_tolerance",
            )
        )
        resolved_source_mixture_tolerance = (
            source_mixture_absolute_tolerance
            if source_mixture_absolute_tolerance is not None
            else _single_diagnostic_field_or_none(
                self.matrix.diagnostics,
                field_name="source_mixture_absolute_tolerance",
            )
        )

        if self.scheduled_cells:
            self._validate_scheduled_simulation_alignment()
            _validate_target_precision_tolerances(
                absolute_tolerance=resolved_absolute_tolerance,
                z_tolerance=resolved_z_tolerance,
            )
            if resolved_source_mixture_tolerance is not None and resolved_source_mixture_tolerance < 0:
                raise ValueError("source_mixture_absolute_tolerance must be non-negative when provided.")
            return _json_safe_export_frame(
                _replication_budget_rows_from_scheduled_cells(
                    self.scheduled_cells,
                    self.blocked_rows,
                    self.data_source_diagnostics,
                    absolute_tolerance=resolved_absolute_tolerance,
                    z_tolerance=resolved_z_tolerance,
                    source_mixture_absolute_tolerance=resolved_source_mixture_tolerance,
                )
            )
        return _json_safe_export_frame(
            _replication_budget_rows_from_plan_cells(
                self.plan.executable_cells,
                self.blocked_rows,
                self.data_source_diagnostics,
                replications=self.plan.replications,
                absolute_tolerance=resolved_absolute_tolerance,
                z_tolerance=resolved_z_tolerance,
                source_mixture_absolute_tolerance=resolved_source_mixture_tolerance,
            )
        )

    def cell_count_policy_frame(self) -> pd.DataFrame:
        """Return row-level paper cell-count bin policy for executed scheduled cells."""

        if self.scheduled_cells:
            self._validate_scheduled_simulation_alignment()
            return _json_safe_export_frame(
                _cell_count_policy_rows_from_scheduled_cells(
                    self.scheduled_cells,
                    self.blocked_rows,
                )
            )
        return self.plan.cell_count_policy_frame()

    def cell_count_policy_summary(self) -> dict[str, Any]:
        """Return compact paper cell-count bin-policy counts for the run result."""

        return _cell_count_policy_summary_from_frame(self.cell_count_policy_frame())

    def raise_for_failures(self) -> None:
        failures = self.failure_frame()
        if failures.empty:
            return
        preview_columns = [
            "status",
            "simulation",
            "table",
            "design",
            "mediator",
            "clusters",
            "bins",
            "t",
            "target_rejection_rate",
            "observed_rejection_rate",
            "absolute_error",
            "z_score",
            "failure_reasons",
            "blocked_reason",
        ]
        preview = failures.loc[:, [column for column in preview_columns if column in failures]].head(5).to_dict(
            "records"
        )
        summary = self.summary()
        raise AssertionError(
            f"Monte Carlo benchmark plan failed {failures.shape[0]} of "
            f"{summary['planned_rows']} planned rows "
            f"({summary['failed_rows']} executed failures, {summary['blocked_rows']} blocked rows): "
            f"{preview}"
        )


@dataclass(frozen=True)
class MonteCarloBenchmarkSuiteReadinessReport:
    """Aggregate readiness packet for all reported paper benchmark methods."""

    reports: tuple[MonteCarloBenchmarkPlanReadinessReport, ...]

    @property
    def method_names(self) -> tuple[str, ...]:
        return tuple(report.plan.method for report in self.reports)

    def summary(self) -> dict[str, Any]:
        return _paper_acceptance_suite_summary(
            gates=tuple(report.paper_acceptance_gate() for report in self.reports),
            methods=self.method_names,
            executed=False,
        )

    def paper_acceptance_gate(self) -> dict[str, Any]:
        return _paper_acceptance_suite_gate(
            gates=tuple(report.paper_acceptance_gate() for report in self.reports),
            methods=self.method_names,
            executed=False,
        )

    def paper_acceptance_blocker_frame(self) -> pd.DataFrame:
        return _paper_acceptance_suite_blocker_frame(
            methods=self.method_names,
            gates=tuple(report.paper_acceptance_gate() for report in self.reports),
            blocker_frames=tuple(report.paper_acceptance_blocker_frame() for report in self.reports),
        )

    def unresolved_paper_row_frame(self) -> pd.DataFrame:
        return _paper_acceptance_suite_unresolved_frame(
            methods=self.method_names,
            frames=tuple(report.unresolved_paper_row_frame() for report in self.reports),
        )


@dataclass(frozen=True)
class MonteCarloBenchmarkSuiteRunResult:
    """Aggregate post-run acceptance packet for all reported paper methods."""

    results: tuple[MonteCarloBenchmarkPlanRunResult, ...]

    @classmethod
    def from_chunk_results(
        cls,
        chunks: tuple["MonteCarloBenchmarkSuiteRunResult", ...],
    ) -> "MonteCarloBenchmarkSuiteRunResult":
        """Combine resumable suite chunks into one post-run acceptance packet."""

        return _combine_benchmark_suite_chunk_results(chunks)

    @property
    def method_names(self) -> tuple[str, ...]:
        return tuple(result.plan.method for result in self.results)

    def summary(self) -> dict[str, Any]:
        return _paper_acceptance_suite_summary(
            gates=tuple(result.paper_acceptance_gate() for result in self.results),
            methods=self.method_names,
            executed=True,
        )

    def paper_acceptance_gate(self) -> dict[str, Any]:
        return _paper_acceptance_suite_gate(
            gates=tuple(result.paper_acceptance_gate() for result in self.results),
            methods=self.method_names,
            executed=True,
        )

    def paper_acceptance_blocker_frame(self) -> pd.DataFrame:
        return _paper_acceptance_suite_blocker_frame(
            methods=self.method_names,
            gates=tuple(result.paper_acceptance_gate() for result in self.results),
            blocker_frames=tuple(result.paper_acceptance_blocker_frame() for result in self.results),
        )

    def unresolved_paper_row_frame(self) -> pd.DataFrame:
        return _paper_acceptance_suite_unresolved_frame(
            methods=self.method_names,
            frames=tuple(result.unresolved_paper_row_frame() for result in self.results),
        )

    def blocker_packet(
        self,
        *,
        owner: str = "Phase 11 full paper Monte Carlo acceptance run",
        rerun_command: str | None = None,
        runtime_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Return an owner-facing packet for suite-level post-run acceptance."""

        summary = self.summary()
        gate = self.paper_acceptance_gate()
        blocker_frame = self.paper_acceptance_blocker_frame()
        unresolved_frame = self.unresolved_paper_row_frame()
        method_gates = {
            method: result.paper_acceptance_gate()
            for method, result in zip(self.method_names, self.results, strict=True)
        }
        method_summaries = {
            method: result.summary()
            for method, result in zip(self.method_names, self.results, strict=True)
        }
        active_blocker_rows = (
            blocker_frame.loc[blocker_frame["blocked"].map(_truthy_value)]
            if "blocked" in blocker_frame
            else pd.DataFrame()
        )
        return {
            "owner": owner,
            "cleared_same_line_items": {
                "stage": summary["stage"],
                "method_count": summary["method_count"],
                "methods": summary["methods"],
                "covered_result_rows": summary["covered_result_rows"],
                "executed_rows": summary["executed_rows"],
                "passed_executed_rows": summary["passed_executed_rows"],
                "total_draws": summary["total_draws"],
                "benchmark_run_not_executed": int(
                    summary["blocking_conditions"].get(
                        "benchmark_run_not_executed",
                        0,
                    )
                ),
            },
            "blocked_same_line_items": {
                "gate_passes": gate["gate_passes"],
                "active_blocking_conditions": dict(
                    summary["active_blocking_conditions"]
                ),
                "active_blocking_condition_count": summary[
                    "active_blocking_condition_count"
                ],
                "paper_coverage_shortfall_rows": summary[
                    "paper_coverage_shortfall_rows"
                ],
                "blocked_rows": summary["blocked_rows"],
                "failed_executed_rows": summary["failed_executed_rows"],
                "target_replication_shortfall_rows": int(
                    summary["blocking_conditions"].get(
                        "target_replication_shortfall_rows",
                        0,
                    )
                ),
                "source_mixture_replication_shortfall_rows": int(
                    summary["blocking_conditions"].get(
                        "source_mixture_replication_shortfall_rows",
                        0,
                    )
                ),
                "bootstrap_replication_shortfall_rows": int(
                    summary["blocking_conditions"].get(
                        "bootstrap_replication_shortfall_rows",
                        0,
                    )
                ),
                "unresolved_rows": int(unresolved_frame.shape[0]),
            },
            "verified_boundary": {
                "paper_result_rows": summary["paper_result_rows"],
                "covered_result_rows": summary["covered_result_rows"],
                "paper_coverage_complete": summary["paper_coverage_complete"],
                "planned_rows": summary["planned_rows"],
                "executed_rows": summary["executed_rows"],
                "planned_draws": summary["planned_draws"],
                "total_draws": summary["total_draws"],
                "runtime_seconds": runtime_seconds,
                "full_plan_passes": summary["full_plan_passes"],
                "executable_slice_passes": summary["executable_slice_passes"],
            },
            "evidence_checked": {
                "method_gate_passes": {
                    method: bool(method_gate["gate_passes"])
                    for method, method_gate in method_gates.items()
                },
                "method_active_blocking_conditions": {
                    method: dict(method_gate["active_blocking_conditions"])
                    for method, method_gate in method_gates.items()
                },
                "method_planned_draws": {
                    method: int(method_summary["planned_draws"])
                    for method, method_summary in method_summaries.items()
                },
                "method_total_draws": {
                    method: int(method_summary["total_draws"])
                    for method, method_summary in method_summaries.items()
                },
                "blocked_reason_counts": dict(summary["blocked_reason_counts"]),
                "blocking_condition_rows": tuple(
                    active_blocker_rows.to_dict("records")
                ),
                "unresolved_row_count": int(unresolved_frame.shape[0]),
                "max_target_mc_standard_error": summary.get(
                    "max_target_mc_standard_error"
                ),
                "max_source_mixture_mc_standard_error": summary.get(
                    "max_source_mixture_mc_standard_error"
                ),
                "bootstrap_budget": summary.get("bootstrap_budget"),
                "bootstrap_replication_shortfall_rows": summary.get(
                    "bootstrap_replication_shortfall_rows"
                ),
                "max_rejection_rate_absolute_error": summary.get(
                    "max_rejection_rate_absolute_error"
                ),
                "cell_count_checked_rows": summary.get("cell_count_checked_rows"),
                "cell_count_failed_rows": summary.get("cell_count_failed_rows"),
                "source_mixture_checked_rows": summary.get(
                    "source_mixture_checked_rows"
                ),
                "source_mixture_failed_rows": summary.get("source_mixture_failed_rows"),
                "runtime_seconds": runtime_seconds,
            },
            "paper_acceptance_gate": gate,
            "rerun_command": rerun_command,
            "exit_criteria": "paper_acceptance_gate.gate_passes == True",
        }

    def to_frame(self) -> pd.DataFrame:
        """Return a row-level frame for all suite methods in method order."""

        frames = [result.acceptance_frame() for result in self.results]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def to_dict(
        self,
        *,
        owner: str = "Phase 11 full paper Monte Carlo acceptance run",
        rerun_command: str | None = None,
        runtime_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Serialize the suite-level acceptance packet for archive and JSON export."""

        return _json_safe_export_payload({
            "summary": self.summary(),
            "paper_acceptance_gate": self.paper_acceptance_gate(),
            "milestone_completion_summary": self.milestone_completion_summary(
                owner=owner,
                rerun_command=rerun_command,
            ),
            "blocker_packet": self.blocker_packet(
                owner=owner,
                rerun_command=rerun_command,
                runtime_seconds=runtime_seconds,
            ),
            "frame": self.to_frame().to_dict("records"),
            "results": [
                result.to_dict(
                    owner=owner,
                    rerun_command=rerun_command,
                )
                for result in self.results
            ],
        })

    def milestone_completion_summary(
        self,
        *,
        owner: str = "Phase 11 full paper Monte Carlo acceptance run",
        rerun_command: str | None = None,
    ) -> dict[str, Any]:
        """Return the release/milestone completion summary for the suite run."""

        return _paper_acceptance_gate_completion_summary(
            self.paper_acceptance_gate(),
            owner=owner,
            rerun_command=rerun_command,
        )

    def milestone_completion_frame(
        self,
        *,
        owner: str = "Phase 11 full paper Monte Carlo acceptance run",
        rerun_command: str | None = None,
    ) -> pd.DataFrame:
        """Return row-level suite closeout conditions for the post-run verdict."""

        return _paper_acceptance_gate_completion_frame(
            self.milestone_completion_summary(
                owner=owner,
                rerun_command=rerun_command,
            )
        )

    def raise_for_milestone_completion_blockers(
        self,
        *,
        owner: str = "Phase 11 full paper Monte Carlo acceptance run",
        rerun_command: str | None = None,
    ) -> None:
        """Raise if the suite run has not cleared paper acceptance closeout."""

        _raise_for_milestone_completion_blockers(
            self.milestone_completion_summary(
                owner=owner,
                rerun_command=rerun_command,
            )
        )


def _combine_benchmark_plan_run_result_chunks(
    chunks: tuple[MonteCarloBenchmarkPlanRunResult, ...],
) -> MonteCarloBenchmarkPlanRunResult:
    if not chunks:
        raise ValueError("combine requires at least one benchmark run result chunk.")

    first = chunks[0]
    method = first.plan.method
    paper_result_rows = first.plan.paper_result_rows
    blocked_rows = tuple(blocked.to_dict() for blocked in first.blocked_rows)
    entries_by_cell_key: dict[
        tuple[Any, ...],
        list[
            tuple[
                MonteCarloBenchmarkPlanRerunCell,
                MonteCarloSimulationResult,
                MonteCarloBenchmarkDiagnostic,
            ]
        ],
    ] = {}
    combined_entries: list[
        tuple[
            MonteCarloBenchmarkPlanRerunCell,
            MonteCarloSimulationResult,
            MonteCarloBenchmarkDiagnostic,
        ]
    ] = []
    data_source_diagnostic_templates: dict[str, MonteCarloBenchmarkDataSourceDiagnostic] = {}
    representative_bootstrap_replications = (
        first.scheduled_cells[0].bootstrap_replications if first.scheduled_cells else None
    )
    representative_alpha = first.scheduled_cells[0].alpha if first.scheduled_cells else None
    representative_manifest_seed = first.manifest_seed

    for chunk in chunks:
        if chunk.plan.method != method:
            raise ValueError(
                "benchmark run result chunks must share the same method to be combined."
            )
        if chunk.manifest_seed != representative_manifest_seed:
            raise ValueError(
                "benchmark run result chunks must share the same manifest_seed to be combined."
            )
        if chunk.plan.paper_result_rows != paper_result_rows:
            raise ValueError(
                "benchmark run result chunks must share the same paper_result_rows to be combined."
            )
        if tuple(blocked.to_dict() for blocked in chunk.blocked_rows) != blocked_rows:
            raise ValueError(
                "benchmark run result chunks must share the same blocked_rows to be combined."
            )
        if not chunk.scheduled_cells:
            raise ValueError(
                "benchmark run result chunks must expose scheduled_cells for combination."
            )

        chunk_replications = chunk.scheduled_cells[0].replications
        if any(cell.replications != chunk_replications for cell in chunk.scheduled_cells):
            raise ValueError("benchmark run result chunks must use uniform replications within a chunk.")
        chunk_bootstrap_replications = chunk.scheduled_cells[0].bootstrap_replications
        if any(
            cell.bootstrap_replications != chunk_bootstrap_replications
            for cell in chunk.scheduled_cells
        ):
            raise ValueError(
                "benchmark run result chunks must use uniform bootstrap_replications within a chunk."
            )
        if (
            representative_bootstrap_replications is not None
            and representative_bootstrap_replications != chunk_bootstrap_replications
        ):
            raise ValueError(
                "benchmark run result chunks must share the same bootstrap_replications to be combined."
            )

        for scheduled_cell, simulation, diagnostic in zip(
            chunk.scheduled_cells,
            chunk.matrix.simulations,
            chunk.matrix.diagnostics,
            strict=True,
        ):
            if representative_alpha is not None and scheduled_cell.alpha != representative_alpha:
                raise ValueError("benchmark run result chunks must share the same alpha to be combined.")
            cell_key = _paper_result_cell_key(scheduled_cell.cell)
            entries_by_cell_key.setdefault(cell_key, []).append(
                (scheduled_cell, simulation, diagnostic)
            )

        chunk_diagnostics_by_key = _diagnostics_by_data_source_key(
            chunk.data_source_diagnostics
        )
        for key, diagnostic in chunk_diagnostics_by_key.items():
            existing = data_source_diagnostic_templates.get(key)
            if existing is None:
                data_source_diagnostic_templates[key] = diagnostic
                continue
            if _data_source_diagnostic_template(existing) != _data_source_diagnostic_template(diagnostic):
                raise ValueError(
                    "benchmark run result chunks must expose consistent data source diagnostics."
                )

    for entries in entries_by_cell_key.values():
        combined_entries.append(_combine_benchmark_plan_run_result_cell_entries(entries))

    combined_entries = sorted(
        combined_entries,
        key=lambda entry: _scheduled_benchmark_cell_sort_key(entry[0]),
    )
    combined_cells = [entry[0] for entry in combined_entries]
    if not combined_cells:
        raise ValueError("benchmark run result chunks selected no scheduled paper cells.")
    representative_replications = combined_cells[0].replications
    if any(cell.replications != representative_replications for cell in combined_cells):
        raise ValueError(
            "combined benchmark run result chunks must share the same replications "
            "(effective replications after chunk selection)."
        )
    combined_simulations = [entry[1] for entry in combined_entries]
    combined_diagnostics = [entry[2] for entry in combined_entries]
    combined_plan = MonteCarloBenchmarkPlan(
        method=method,
        replications=representative_replications,
        executable_cells=tuple(scheduled_cell.cell for scheduled_cell in combined_cells),
        blocked_rows=first.blocked_rows,
        paper_result_rows=paper_result_rows,
    )
    combined_source_keys = {
        key: tuple(
            scheduled_cell
            for scheduled_cell in combined_cells
            if _benchmark_data_source_key(scheduled_cell.cell) == key
        )
        for key in data_source_diagnostic_templates
    }
    combined_data_source_diagnostics_ordered = tuple(
        replace(
            data_source_diagnostic_templates[key],
            executable_rows=len(combined_source_keys[key]),
            requires_cluster_resampling=any(
                scheduled_cell.cell.requires_cluster_resampling
                for scheduled_cell in combined_source_keys[key]
            ),
        )
        for key in sorted(data_source_diagnostic_templates)
    )
    _diagnostics_by_data_source_key(combined_data_source_diagnostics_ordered)
    return MonteCarloBenchmarkPlanRunResult(
        plan=combined_plan,
        matrix=MonteCarloBenchmarkMatrixResult(
            simulations=tuple(combined_simulations),
            diagnostics=tuple(combined_diagnostics),
        ),
        data_source_diagnostics=combined_data_source_diagnostics_ordered,
        scheduled_cells=tuple(combined_cells),
        manifest_seed=representative_manifest_seed,
    )


def _data_source_diagnostic_template(
    diagnostic: MonteCarloBenchmarkDataSourceDiagnostic,
) -> dict[str, Any]:
    payload = dict(diagnostic.to_dict())
    payload.pop("executable_rows", None)
    payload.pop("requires_cluster_resampling", None)
    return payload


def _combine_benchmark_plan_run_result_cell_entries(
    entries: list[
        tuple[
            MonteCarloBenchmarkPlanRerunCell,
            MonteCarloSimulationResult,
            MonteCarloBenchmarkDiagnostic,
        ]
    ],
) -> tuple[
    MonteCarloBenchmarkPlanRerunCell,
    MonteCarloSimulationResult,
    MonteCarloBenchmarkDiagnostic,
]:
    if not entries:
        raise ValueError("cell entry combination requires at least one entry.")
    if len(entries) == 1:
        return entries[0]

    sorted_entries = sorted(entries, key=lambda entry: entry[0].replication_start)
    first_cell, first_simulation, first_diagnostic = sorted_entries[0]
    expected_identity = _paper_result_cell_key(first_cell.cell)
    expected_seed = first_cell.seed
    expected_bootstrap = first_cell.bootstrap_replications
    expected_alpha = first_cell.alpha
    expected_seed_replications = (
        first_cell.seed_replications
        if first_cell.seed_replications is not None
        else first_cell.replications
    )

    ranges: list[tuple[int, int]] = []
    draws: list[MonteCarloDrawResult] = []
    for scheduled_cell, simulation, diagnostic in sorted_entries:
        if _paper_result_cell_key(scheduled_cell.cell) != expected_identity:
            raise ValueError("replication shards must share the same paper cell.")
        if scheduled_cell.seed != expected_seed:
            raise ValueError("replication shards must share the same cell seed.")
        if scheduled_cell.bootstrap_replications != expected_bootstrap:
            raise ValueError("replication shards must share bootstrap_replications.")
        if not math.isclose(scheduled_cell.alpha, expected_alpha, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("replication shards must share alpha.")
        seed_replications = (
            scheduled_cell.seed_replications
            if scheduled_cell.seed_replications is not None
            else scheduled_cell.replications
        )
        if seed_replications != expected_seed_replications:
            raise ValueError("replication shards must share seed_replications.")
        if diagnostic.method != first_diagnostic.method:
            raise ValueError("replication shards must share the same method.")
        if _benchmark_diagnostic_paper_identity(diagnostic) != _benchmark_diagnostic_paper_identity(first_diagnostic):
            raise ValueError("replication shards must share diagnostic paper identity.")

        start = scheduled_cell.replication_start
        stop = start + scheduled_cell.replications
        observed_replications = tuple(draw.replication for draw in simulation.draws)
        expected_replications = tuple(range(start + 1, stop + 1))
        if observed_replications != expected_replications:
            raise ValueError(
                "replication shard draw indices must match the scheduled replication range."
            )
        ranges.append((start, stop))
        draws.extend(simulation.draws)

    for previous, current in zip(ranges, ranges[1:], strict=False):
        if previous[1] != current[0]:
            raise ValueError("replication shards must be contiguous and non-overlapping.")

    start = ranges[0][0]
    stop = ranges[-1][1]
    combined_replications = stop - start
    combined_cell = replace(
        first_cell,
        replications=combined_replications,
        replication_start=start,
        seed_replications=expected_seed_replications,
    )
    combined_simulation = replace(
        first_simulation,
        design=replace(
            first_simulation.design,
            replications=combined_replications,
            replication_start=start,
            seed_replications=expected_seed_replications,
        ),
        draws=tuple(draws),
    )
    combined_diagnostic = replace(first_diagnostic, simulation=combined_simulation)
    return combined_cell, combined_simulation, combined_diagnostic


def _combine_benchmark_suite_chunk_results(
    chunks: tuple[MonteCarloBenchmarkSuiteRunResult, ...],
) -> MonteCarloBenchmarkSuiteRunResult:
    if not chunks:
        raise ValueError("combine requires at least one suite run result chunk.")

    grouped_results: dict[str, list[MonteCarloBenchmarkPlanRunResult]] = {}
    for chunk in chunks:
        for result in chunk.results:
            method = result.plan.method
            if method not in grouped_results:
                grouped_results[method] = []
            grouped_results[method].append(result)

    method_order = tuple(_ordered_monte_carlo_methods(set(grouped_results)))
    combined_results = tuple(
        _combine_benchmark_plan_run_result_chunks(tuple(grouped_results[method]))
        for method in method_order
    )
    return MonteCarloBenchmarkSuiteRunResult(results=combined_results)


def write_monte_carlo_suite_run_result_json(
    run_result: MonteCarloBenchmarkSuiteRunResult,
    path_like: str | Path,
    *,
    owner: str = "Phase 11 full paper Monte Carlo acceptance run",
    rerun_command: str | None = None,
    runtime_seconds: float | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a suite acceptance payload to a strict JSON file.

    Parameters
    ----------
    run_result : MonteCarloBenchmarkSuiteRunResult
        Complete suite run result to serialize.
    path_like : str or Path
        Destination file path.
    owner : str
        Label identifying the owner/phase of this run.
    rerun_command : str or None
        Shell command to reproduce this run.
    runtime_seconds : float or None
        Total runtime of the run in seconds.
    overwrite : bool
        If ``True``, overwrite an existing file; otherwise raise.

    Returns
    -------
    Path
        The written file path.

    Raises
    ------
    FileExistsError
        If file exists and *overwrite* is ``False``.
    """

    _require_monte_carlo_writer_overwrite(overwrite)
    path = Path(path_like)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists; pass overwrite=True to replace it."
        )
    payload = run_result.to_dict(
        owner=owner,
        rerun_command=rerun_command,
        runtime_seconds=runtime_seconds,
    )
    _write_monte_carlo_json_atomic(path, payload)
    return path


def load_monte_carlo_suite_run_result_json(
    path_like: str | Path,
) -> MonteCarloBenchmarkSuiteRunResult:
    """Load a strict JSON suite run result.

    Parameters
    ----------
    path_like : str or Path
        Path to the JSON file written by
        :func:`write_monte_carlo_suite_run_result_json`.

    Returns
    -------
    MonteCarloBenchmarkSuiteRunResult
        Deserialized suite run result.

    Raises
    ------
    ValueError
        If the file is not valid strict JSON or has unexpected schema.
    """

    path = Path(path_like)
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_monte_carlo_json_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError("Monte Carlo suite run result JSON must contain an object.")
    return _monte_carlo_suite_run_result_from_payload(payload)


def write_merged_monte_carlo_suite_run_result_json(
    input_paths: tuple[str | Path, ...],
    output_path: str | Path,
    *,
    owner: str = "JSS manuscript Monte Carlo shard merge",
    rerun_command: str | None = None,
    runtime_seconds: float | None = None,
    overwrite: bool = False,
) -> Path:
    """Merge compatible persisted suite shard JSON files and write a result.

    Loads multiple chunk JSON files produced by separate runs, combines
    them into a single suite result, and writes the merged result as strict
    JSON.

    Parameters
    ----------
    input_paths : tuple[str | Path, ...]
        Paths to the JSON shard files to merge.
    output_path : str or Path
        Destination for the merged JSON file.
    owner : str
        Label for the merged output.
    rerun_command : str or None
        Shell command to reproduce the merge.
    runtime_seconds : float or None
        Total runtime.
    overwrite : bool
        If ``True``, overwrite an existing output file.

    Returns
    -------
    Path
        The written merged file path.

    Raises
    ------
    ValueError
        If *input_paths* is empty.
    FileExistsError
        If *output_path* exists and *overwrite* is ``False``.
    """

    if not input_paths:
        raise ValueError("input_paths must contain at least one Monte Carlo JSON file.")
    chunks = tuple(load_monte_carlo_suite_run_result_json(path) for path in input_paths)
    combined = MonteCarloBenchmarkSuiteRunResult.from_chunk_results(chunks)
    return write_monte_carlo_suite_run_result_json(
        combined,
        output_path,
        owner=owner,
        rerun_command=rerun_command,
        runtime_seconds=runtime_seconds,
        overwrite=overwrite,
    )


def _monte_carlo_suite_run_result_from_payload(
    payload: dict[str, Any],
) -> MonteCarloBenchmarkSuiteRunResult:
    results_payload = payload.get("results")
    if not isinstance(results_payload, list) or not results_payload:
        raise ValueError("Monte Carlo suite run result JSON must contain non-empty results.")
    return MonteCarloBenchmarkSuiteRunResult(
        results=tuple(
            _monte_carlo_plan_run_result_from_payload(result_payload)
            for result_payload in results_payload
        )
    )


def _monte_carlo_plan_run_result_from_payload(
    payload: dict[str, Any],
) -> MonteCarloBenchmarkPlanRunResult:
    scheduled_cells = tuple(
        _scheduled_cell_from_payload(cell_payload)
        for cell_payload in payload.get("scheduled_cells", ())
    )
    plan_payload = payload.get("plan")
    if not isinstance(plan_payload, dict):
        raise ValueError("Monte Carlo run result JSON is missing plan.")
    plan_summary = plan_payload.get("summary")
    if not isinstance(plan_summary, dict):
        raise ValueError("Monte Carlo run result JSON plan is missing summary.")
    blocked_rows = tuple(
        _blocked_row_from_payload(row_payload)
        for row_payload in plan_payload.get("blocked_rows", ())
    )
    plan = MonteCarloBenchmarkPlan(
        method=str(plan_summary["method"]),
        replications=int(plan_summary["default_replications"]),
        executable_cells=tuple(cell.cell for cell in scheduled_cells),
        blocked_rows=blocked_rows,
        paper_result_rows=plan_payload.get("paper_coverage", {}).get("paper_result_rows"),
    )
    matrix_payload = payload.get("matrix")
    if not isinstance(matrix_payload, dict):
        raise ValueError("Monte Carlo run result JSON is missing matrix.")
    simulations = tuple(
        _simulation_from_payload(simulation_payload)
        for simulation_payload in matrix_payload.get("simulations", ())
    )
    diagnostics = tuple(
        _diagnostic_from_payload(diagnostic_payload, simulation=simulation)
        for diagnostic_payload, simulation in zip(
            matrix_payload.get("diagnostics", ()),
            simulations,
            strict=True,
        )
    )
    return MonteCarloBenchmarkPlanRunResult(
        plan=plan,
        matrix=MonteCarloBenchmarkMatrixResult(
            simulations=simulations,
            diagnostics=diagnostics,
        ),
        data_source_diagnostics=tuple(
            _data_source_diagnostic_from_payload(diagnostic_payload)
            for diagnostic_payload in payload.get("data_source_diagnostics", ())
        ),
        scheduled_cells=scheduled_cells,
        manifest_seed=payload.get("manifest_seed"),
    )


def _scheduled_cell_from_payload(payload: dict[str, Any]) -> MonteCarloBenchmarkPlanRerunCell:
    cell = _benchmark_cell_from_payload(payload)
    source = _placeholder_data_source_from_scheduled_cell_payload(payload)
    return MonteCarloBenchmarkPlanRerunCell(
        cell=cell,
        seed=int(payload["seed"]),
        replications=int(payload["replications"]),
        bootstrap_replications=int(payload["bootstrap_replications"]),
        alpha=float(payload["alpha"]),
        source=source,
        replication_start=int(payload.get("replication_start") or 0),
        seed_replications=payload.get("seed_replications"),
    )


def _benchmark_cell_from_payload(payload: dict[str, Any]) -> MonteCarloBenchmarkCell:
    clusters = payload.get("clusters")
    bins = payload.get("bins")
    median_clusters = payload.get("median_independent_clusters_per_cell")
    cluster_cell_count = None
    if clusters is not None and bins is not None and median_clusters is not None:
        cluster_cell_count = ClusterCellCount(
            table=str(payload["table"]),
            panel=str(payload["panel"]),
            design=str(payload["design"]),
            mediator=str(payload["mediator"]),
            clusters=int(clusters),
            bins=int(bins),
            t=float(payload["t"]),
            median_independent_clusters_per_cell=float(median_clusters),
        )
    return MonteCarloBenchmarkCell(
        table=str(payload["table"]),
        panel=str(payload["panel"]),
        design=str(payload["design"]),
        mediator=str(payload["mediator"]),
        clusters=None if clusters is None else int(clusters),
        bins=None if bins is None else int(bins),
        t=float(payload["t"]),
        method=str(payload["method"]),
        bar_nu_lb=float(payload["bar_nu_lb"]),
        target_rejection_rate=float(payload["target_rejection_rate"]),
        is_null_size_row=bool(payload["size_row"]),
        cluster_cell_count=cluster_cell_count,
    )


def _blocked_row_from_payload(payload: dict[str, Any]) -> MonteCarloBlockedBenchmarkRow:
    method = str(payload["method"])
    row = MonteCarloResultRow(
        table=str(payload["table"]),
        panel=str(payload["panel"]),
        design=str(payload["design"]),
        mediator=str(payload["mediator"]),
        clusters=payload.get("clusters"),
        bins=payload.get("bins"),
        t=float(payload["t"]),
        bar_nu_lb=float(payload["bar_nu_lb"]),
        rejection_rates={method: float(payload["target_rejection_rate"])},
    )
    return MonteCarloBlockedBenchmarkRow(
        row=row,
        method=method,
        reason=str(payload["blocked_reason"]),
    )


def _placeholder_data_source_from_scheduled_cell_payload(
    payload: dict[str, Any],
) -> BinaryEmpiricalMixtureBenchmarkDataSource:
    data_source = payload.get("data_source") or {}
    d = str(data_source.get("d") or "d")
    m = str(data_source.get("m") or "m")
    y = str(data_source.get("y") or "y")
    cluster = data_source.get("cluster")
    columns = [d, m, y]
    columns.extend(str(column) for column in data_source.get("analysis_frame_columns") or ())
    if cluster is not None:
        columns.append(str(cluster))
    frame = pd.DataFrame({column: [0, 1] for column in dict.fromkeys(columns)})
    return BinaryEmpiricalMixtureBenchmarkDataSource(
        df=frame,
        d=d,
        m=m,
        y=y,
        cluster=None if cluster is None else str(cluster),
        analysis_frame_columns=tuple(data_source.get("analysis_frame_columns") or ()),
    )


def _simulation_from_payload(payload: dict[str, Any]) -> MonteCarloSimulationResult:
    design = _empirical_mixture_design_from_payload(payload["design"])
    draws = tuple(_draw_result_from_payload(draw_payload) for draw_payload in payload.get("draws", ()))
    return MonteCarloSimulationResult(design=design, draws=draws)


def _empirical_mixture_design_from_payload(
    payload: dict[str, Any],
) -> BinaryEmpiricalMixtureMonteCarloDesign:
    clustered = payload.get("cluster_count") is not None
    pool = pd.DataFrame(
        {
            "mediator": [0, 1],
            "outcome": [0, 1],
            **({"_source_cluster": [0, 1]} if clustered else {}),
        }
    )
    return BinaryEmpiricalMixtureMonteCarloDesign(
        name=str(payload["name"]),
        replications=int(payload["replications"]),
        seed=int(payload["seed"]),
        t=float(payload["t"]),
        control_pool=pool,
        treated_pool=pool,
        n_control_per_draw=int(payload["n_control_per_draw"]),
        n_treated_per_draw=int(payload["n_treated_per_draw"]),
        arm_assignment=str(payload["arm_assignment"]),
        treatment_probability=float(payload["treatment_probability"]),
        cluster_count=payload.get("cluster_count"),
        clusters_per_arm=payload.get("clusters_per_arm"),
        num_y_bins=payload.get("num_y_bins"),
        bootstrap_replications=int(payload["bootstrap_replications"]),
        alpha=float(payload["alpha"]),
        replication_start=int(payload.get("replication_start") or 0),
        seed_replications=payload.get("seed_replications"),
        dgp_source=str(payload.get("dgp_source") or "paper-empirical-mixture"),
        paper_reference=str(payload.get("paper_reference") or ""),
        paper_contract_dict=dict(payload.get("paper_contract") or {}),
        source_treatment_levels=tuple(payload.get("source_treatment_levels") or ()),
        source_mediator_levels=tuple(payload.get("source_mediator_levels") or ()),
    )


def _draw_result_from_payload(payload: dict[str, Any]) -> MonteCarloDrawResult:
    return MonteCarloDrawResult(
        replication=int(payload["replication"]),
        seed=int(payload["seed"]),
        reject=bool(payload["reject"]),
        test_stat=_marked_float_from_payload(payload.get("test_stat"), payload.get("test_stat_nonfinite")),
        critical_value=_marked_float_from_payload(payload.get("critical_value"), payload.get("critical_value_nonfinite")),
        p_value=_marked_float_from_payload(payload.get("p_value"), payload.get("p_value_nonfinite")),
        applied_num_y_bins=payload.get("applied_num_y_bins"),
        n_obs_used=int(payload["n_obs_used"]),
        control_observations=int(payload["control_observations"]),
        treated_observations=int(payload["treated_observations"]),
        min_cell_count=int(payload["min_cell_count"]),
        min_cluster_count=int(payload["min_cluster_count"]),
        median_cell_count=float(payload["median_cell_count"]),
        median_independent_count_per_cell=float(payload["median_independent_count_per_cell"]),
        size_risk=bool(payload["size_risk"]),
        empty_cell_count=int(payload.get("empty_cell_count") or 0),
        empty_cluster_cell_count=int(payload.get("empty_cluster_cell_count") or 0),
        small_cell_count=int(payload.get("small_cell_count") or 0),
        small_cluster_cell_count=int(payload.get("small_cluster_cell_count") or 0),
        size_risk_threshold=int(payload.get("size_risk_threshold") or 15),
        empty_cells=tuple(payload.get("empty_cells") or ()),
        small_cells=tuple(payload.get("small_cells") or ()),
        empty_cluster_cells=tuple(payload.get("empty_cluster_cells") or ()),
        small_cluster_cells=tuple(payload.get("small_cluster_cells") or ()),
        n_clusters_used=payload.get("n_clusters_used"),
        treated_clusters=payload.get("treated_clusters"),
        control_clusters=payload.get("control_clusters"),
        cluster_size=payload.get("cluster_size"),
        treated_source_treated_draws=payload.get("treated_source_treated_draws"),
        treated_source_control_draws=payload.get("treated_source_control_draws"),
        treated_source_treated_clusters=payload.get("treated_source_treated_clusters"),
        treated_source_control_clusters=payload.get("treated_source_control_clusters"),
    )


def _marked_float_from_payload(value: Any, marker: Any) -> float:
    if marker == "positive_infinity":
        return math.inf
    if marker == "negative_infinity":
        return -math.inf
    if value is None:
        return math.nan
    return float(value)


def _diagnostic_from_payload(
    payload: dict[str, Any],
    *,
    simulation: MonteCarloSimulationResult,
) -> MonteCarloBenchmarkDiagnostic:
    method = str(payload["method"])
    row = MonteCarloResultRow(
        table=str(payload["table"]),
        panel=str(payload["panel"]),
        design=str(payload["design"]),
        mediator=str(payload["mediator"]),
        clusters=payload.get("clusters"),
        bins=payload.get("bins"),
        t=float(payload["t"]),
        bar_nu_lb=float(payload["bar_nu_lb"]),
        rejection_rates={method: float(payload["target_rejection_rate"])},
    )
    target_cell_count = payload.get("target_median_independent_count_per_cell")
    cluster_cell_count = None
    if payload.get("clusters") is not None and payload.get("bins") is not None and target_cell_count is not None:
        cluster_cell_count = ClusterCellCount(
            table=row.table,
            panel=row.panel,
            design=row.design,
            mediator=row.mediator,
            clusters=int(payload["clusters"]),
            bins=int(payload["bins"]),
            t=row.t,
            median_independent_clusters_per_cell=float(target_cell_count),
        )
    return MonteCarloBenchmarkDiagnostic(
        simulation=simulation,
        row=row,
        method=method,
        absolute_tolerance=float(payload["absolute_tolerance"]),
        z_tolerance=float(payload["z_tolerance"]),
        cell_count_absolute_tolerance=payload.get("cell_count_absolute_tolerance"),
        source_mixture_absolute_tolerance=payload.get("source_mixture_absolute_tolerance"),
        cluster_cell_count=cluster_cell_count,
    )


def _data_source_diagnostic_from_payload(
    payload: dict[str, Any],
) -> MonteCarloBenchmarkDataSourceDiagnostic:
    return MonteCarloBenchmarkDataSourceDiagnostic(
        design=str(payload["design"]),
        executable_rows=int(payload["executable_rows"]),
        requires_cluster_resampling=bool(payload["requires_cluster_resampling"]),
        analysis_frame_columns=tuple(payload.get("analysis_frame_columns") or ()),
        d=payload.get("d"),
        m=payload.get("m"),
        y=payload.get("y"),
        rows=payload.get("rows"),
        complete_case_rows=payload.get("complete_case_rows"),
        treatment_levels=tuple(payload.get("treatment_levels") or ()),
        mediator_levels=tuple(payload.get("mediator_levels") or ()),
        control_rows=payload.get("control_rows"),
        treated_rows=payload.get("treated_rows"),
        cluster=payload.get("cluster"),
        source_clusters=payload.get("source_clusters"),
        control_source_clusters=payload.get("control_source_clusters"),
        treated_source_clusters=payload.get("treated_source_clusters"),
        arm_fixed_source_clusters=payload.get("arm_fixed_source_clusters"),
        expected_complete_case_rows=payload.get("expected_complete_case_rows"),
        expected_control_rows=payload.get("expected_control_rows"),
        expected_treated_rows=payload.get("expected_treated_rows"),
        expected_source_clusters=payload.get("expected_source_clusters"),
        expected_control_source_clusters=payload.get("expected_control_source_clusters"),
        expected_treated_source_clusters=payload.get("expected_treated_source_clusters"),
        outcome_level_count=payload.get("outcome_level_count"),
        outcome_binary=payload.get("outcome_binary"),
        unbinned_outcome_binary_required=bool(payload.get("unbinned_outcome_binary_required")),
        blocking_reasons=tuple(payload.get("blocking_reasons") or ()),
        data_source_key=payload.get("data_source_key"),
    )


@dataclass(frozen=True)
class PaperMonteCarloReproductionReport:
    """Checkout-local paper Monte Carlo reproduction status packet."""

    rows: tuple[dict[str, Any], ...]
    summary: dict[str, Any]
    evidence_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_export_payload(
            {
                "summary": dict(self.summary),
                "rows": [dict(row) for row in self.rows],
                "evidence_summary": dict(self.evidence_summary),
            }
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            _json_safe_export_payload([dict(row) for row in self.rows]),
            dtype=object,
        )

    def display_frame(self) -> pd.DataFrame:
        """Return a compact human-readable Monte Carlo reproduction table."""

        return _paper_monte_carlo_reproduction_display_frame(self.rows)


def paper_monte_carlo_reproduction_report(
    evidence_dir: str | Path,
    *,
    tables_dir: str | Path | None = None,
    fixtures_dir: str | Path | None = None,
    seed: int = 20260509,
    cell_chunk_size: int = 5,
    paper_replications: int = 500,
    slice_replications: int | None = None,
    bootstrap_replications: int = 500,
    mediator: str | None = None,
    design: str | None = None,
    table: str | None = None,
    clusters: tuple[int | None, ...] | None = None,
    bins: tuple[int | None, ...] | None = None,
    t_values: tuple[float, ...] | None = None,
    alpha: float = 0.05,
    absolute_tolerance: float = 0.025,
    z_tolerance: float = 2.0,
    cell_count_absolute_tolerance: float | None = None,
    source_mixture_absolute_tolerance: float | None = None,
    owner: str = "Phase 16 paper Monte Carlo reproduction report",
    rerun_command: str | None = None,
) -> PaperMonteCarloReproductionReport:
    """Summarize paper Monte Carlo rerun evidence without executing new draws.

    Loads persisted chunk evidence from *evidence_dir*, matches against the
    paper's LaTeX table targets, and produces a structured report with
    per-cell pass/fail verdicts.

    Parameters
    ----------
    evidence_dir : str or Path
        Directory containing chunk-evidence JSON files.
    tables_dir : str, Path, or None
        Directory with LaTeX Monte Carlo tables (defaults to paper source).
    fixtures_dir : str, Path, or None
        Directory with CSV empirical fixtures.
    seed : int
        Master seed for the paper schedule.
    cell_chunk_size : int
        Number of cells per chunk in the schedule.
    paper_replications : int
        Target replication count per cell.
    slice_replications : int or None
        If set, restrict each cell to a shard of this size.
    bootstrap_replications : int
        Bootstrap replications for CR method cells.
    mediator, design, table, clusters, bins, t_values
        Optional filters to restrict which cells are included.
    alpha : float
        Nominal significance level.
    absolute_tolerance : float
        Maximum acceptable absolute gap between paper and Python rates.
    z_tolerance : float
        Maximum acceptable z-score gap for sampling-error rows.
    cell_count_absolute_tolerance : float or None
        Tolerance for cell-count comparisons.
    source_mixture_absolute_tolerance : float or None
        Tolerance for source-mixture comparisons.
    owner : str
        Owner label for the report.
    rerun_command : str or None
        Shell command to reproduce the report.

    Returns
    -------
    PaperMonteCarloReproductionReport
        Structured report with per-cell rows and aggregated summary.
    """

    started_at = time.perf_counter()
    contracts = load_paper_monte_carlo_contracts(tables_dir)
    data_sources = load_paper_empirical_mixture_benchmark_data_sources(fixtures_dir)
    evidence_summary = summarize_paper_empirical_mixture_benchmark_suite_evidence(
        evidence_dir,
        tables_dir=tables_dir,
        fixtures_dir=fixtures_dir,
        seed=seed,
        cell_chunk_size=cell_chunk_size,
        paper_replications=paper_replications,
        slice_replications=slice_replications,
        bootstrap_replications=bootstrap_replications,
        mediator=mediator,
        design=design,
        table=table,
        clusters=clusters,
        bins=bins,
        t_values=t_values,
        alpha=alpha,
        absolute_tolerance=absolute_tolerance,
        z_tolerance=z_tolerance,
        cell_count_absolute_tolerance=cell_count_absolute_tolerance,
        source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
        owner=owner,
        rerun_command=rerun_command,
    )
    evidence_files = tuple(evidence_summary.get("evidence_files") or ())
    acceptance_frame = contracts.empirical_mixture_benchmark_suite_export_acceptance_frame(
        evidence_files
    )
    rows = _paper_monte_carlo_reproduction_rows(
        acceptance_frame,
        evidence_summary=evidence_summary,
        alpha=alpha,
    )
    summary = _paper_monte_carlo_reproduction_summary(
        rows,
        acceptance_frame=acceptance_frame,
        evidence_summary=evidence_summary,
        alpha=alpha,
        runtime_seconds=float(time.perf_counter() - started_at),
    )
    return PaperMonteCarloReproductionReport(
        rows=rows,
        summary=summary,
        evidence_summary=evidence_summary,
    )


def paper_monte_carlo_reproduction_report_frame(
    evidence_dir: str | Path,
    **kwargs: Any,
) -> pd.DataFrame:
    """Return the paper Monte Carlo reproduction report as a row-level frame.

    Parameters
    ----------
    evidence_dir : str or Path
        Directory containing chunk-evidence JSON files.
    **kwargs
        Forwarded to :func:`paper_monte_carlo_reproduction_report`.

    Returns
    -------
    pd.DataFrame
        One row per Monte Carlo cell with comparison fields.
    """

    return paper_monte_carlo_reproduction_report(
        evidence_dir,
        **kwargs,
    ).to_frame()


def paper_monte_carlo_reproduction_display_frame(
    evidence_dir: str | Path,
    **kwargs: Any,
) -> pd.DataFrame:
    """Return a compact human-readable paper Monte Carlo reproduction table.

    Parameters
    ----------
    evidence_dir : str or Path
        Directory containing chunk-evidence JSON files.
    **kwargs
        Forwarded to :func:`paper_monte_carlo_reproduction_report`.

    Returns
    -------
    pd.DataFrame
        Display-formatted table with status, method, case, target/python
        rates, and precision columns.
    """

    return paper_monte_carlo_reproduction_report(
        evidence_dir,
        **kwargs,
    ).display_frame()


_PAPER_MONTE_CARLO_REPRODUCTION_DISPLAY_COLUMNS = [
    "status",
    "method",
    "paper_case",
    "target_rate",
    "python_rate",
    "gap",
    "precision",
    "evidence",
    "next_action",
    "case_id",
]


def _paper_monte_carlo_reproduction_display_frame(
    rows: tuple[dict[str, Any], ...],
) -> pd.DataFrame:
    display_rows = [_paper_monte_carlo_reproduction_display_row(row) for row in rows]
    return pd.DataFrame(
        display_rows,
        columns=_PAPER_MONTE_CARLO_REPRODUCTION_DISPLAY_COLUMNS,
        dtype=object,
    )


def _paper_monte_carlo_reproduction_display_row(row: dict[str, Any]) -> dict[str, Any]:
    missing_fields = [field for field in ("case_id", "row_type") if field not in row]
    if missing_fields:
        raise ValueError(
            "Paper Monte Carlo reproduction display rows are missing required fields: "
            + ", ".join(missing_fields)
        )
    return {
        "status": _paper_monte_carlo_display_status(row),
        "method": _paper_monte_carlo_display_label(row.get("method", "schedule"), field="method"),
        "paper_case": _paper_monte_carlo_display_case(row),
        "target_rate": _paper_monte_carlo_format_rate(row.get("target_rejection_rate")),
        "python_rate": _paper_monte_carlo_format_rate(row.get("observed_rejection_rate")),
        "gap": _paper_monte_carlo_format_rate(
            row.get("rejection_rate_absolute_error", row.get("absolute_error"))
        ),
        "precision": _paper_monte_carlo_precision_label(row),
        "evidence": _paper_monte_carlo_evidence_label(row),
        "next_action": _paper_monte_carlo_next_action_label(row),
        "case_id": _paper_monte_carlo_display_label(row["case_id"], field="case_id"),
    }


def _paper_monte_carlo_display_status(row: dict[str, Any]) -> str:
    if row.get("row_type") == "monte_carlo_schedule_row":
        return "SCHEDULED" if row.get("paper_acceptance_gate_passes") else "BLOCKED"
    if row.get("paper_acceptance_gate_passes") is True:
        if row.get("passed") is True:
            return "PASS"
        return "FAIL"
    if row.get("passed") is True:
        return "LOW-BUDGET PASS"
    if row.get("passed") is False:
        return "BLOCKED"
    status = _paper_monte_carlo_display_text(
        row.get("status", row.get("exception_status", "unknown")),
        field="status",
    )
    return status.upper().replace("_", " ")


def _paper_monte_carlo_display_case(row: dict[str, Any]) -> str:
    if row.get("row_type") == "monte_carlo_schedule_row":
        return "paper default schedule"
    case_fields = ("table", "design", "mediator", "clusters", "bins", "t")
    if all(row.get(field) is None for field in case_fields):
        return "case details unavailable"
    parts = (
        _paper_monte_carlo_display_text(row.get("table"), field="table"),
        _paper_monte_carlo_display_text(row.get("design"), field="design"),
        _paper_monte_carlo_display_text(row.get("mediator"), field="mediator"),
        f"clusters={_paper_monte_carlo_cluster_label(row.get('clusters'))}",
        f"outcome_bins={_paper_monte_carlo_bin_label(row.get('bins'))}",
        f"t={_paper_monte_carlo_t_label(row.get('t'))}",
    )
    return _paper_monte_carlo_compact_display_label(
        "; ".join(str(part) for part in parts if part is not None)
    )


def _paper_monte_carlo_cluster_label(value: Any) -> str:
    if value is None:
        return "unclustered"
    return _paper_monte_carlo_format_value(value)


def _paper_monte_carlo_bin_label(value: Any) -> str:
    if value is None:
        return "as observed"
    return _paper_monte_carlo_format_value(value)


def _paper_monte_carlo_t_label(value: Any) -> str:
    if value is None:
        return "NA"
    return _paper_monte_carlo_format_value(value)


def _paper_monte_carlo_precision_label(row: dict[str, Any]) -> str:
    return _paper_monte_carlo_compact_join(
        (
            _paper_monte_carlo_threshold_label(
                "se",
                row.get("target_mc_standard_error"),
                row.get("paper_acceptance_max_target_mc_standard_error"),
            ),
            _paper_monte_carlo_threshold_label("z", row.get("z_score"), row.get("z_tolerance")),
            _paper_monte_carlo_threshold_label(
                "cell_gap",
                row.get("cell_count_absolute_error"),
                row.get("cell_count_absolute_tolerance"),
            ),
        )
    )


def _paper_monte_carlo_evidence_label(row: dict[str, Any]) -> str:
    executed_rows = row.get("executed_result_rows", row.get("covered_result_rows"))
    planned_rows = row.get("paper_result_rows", row.get("scheduled_result_rows"))
    if executed_rows is None and row.get("replications") is not None:
        executed_rows = 1
        planned_rows = 1
    executed_draws = row.get("executed_draws", row.get("replications"))
    scheduled_draws = row.get("scheduled_draws", row.get("planned_draws"))
    executed_bootstrap = row.get(
        "executed_bootstrap_draws",
        row.get("bootstrap_replications", row.get("expected_bootstrap_replications")),
    )
    scheduled_bootstrap = row.get(
        "scheduled_bootstrap_draws",
        row.get("planned_bootstrap_replications"),
    )
    return _paper_monte_carlo_compact_join(
        (
            (
                "rows="
                + _paper_monte_carlo_count_pair(executed_rows, planned_rows)
            ),
            (
                "draws="
                + _paper_monte_carlo_count_pair(executed_draws, scheduled_draws)
            ),
            (
                "bootstrap="
                + _paper_monte_carlo_count_pair(executed_bootstrap, scheduled_bootstrap)
            ),
            (
                "files="
                + _paper_monte_carlo_count_pair(
                    row.get("evidence_file_count"),
                    row.get("scheduled_file_count"),
                )
            ),
            _paper_monte_carlo_blockers_label(row.get("active_blocking_conditions")),
        )
    )


def _paper_monte_carlo_next_action_label(row: dict[str, Any]) -> str:
    actions = []
    seen_raw_actions: set[str] = set()
    for key in ("next_action", "progress_next_action", "next_chunk_rerun_call", "next_chunk_export_call"):
        value = row.get(key)
        if value is not None:
            raw_text = _paper_monte_carlo_display_text(value, field=key)
            if raw_text not in seen_raw_actions:
                text = _paper_monte_carlo_next_action_display_text(raw_text, field=key)
                actions.append(text)
                seen_raw_actions.add(raw_text)
    return "; ".join(actions[:2])


_PAPER_MONTE_CARLO_NEXT_ACTION_COMMAND_FIELDS = frozenset(
    {"next_chunk_rerun_call", "next_chunk_export_call"}
)
_PAPER_MONTE_CARLO_NEXT_ACTION_MAX_COMMAND_CHARS = 120
_PAPER_MONTE_CARLO_JSON_DISPLAY_MAX_CHARS = 160
_PAPER_MONTE_CARLO_JSON_DISPLAY_TAIL_CHARS = 60


def _paper_monte_carlo_next_action_display_text(value: Any, *, field: str) -> str:
    text = _paper_monte_carlo_display_text(value, field=field)
    if field not in _PAPER_MONTE_CARLO_NEXT_ACTION_COMMAND_FIELDS:
        return text
    if len(text) <= _PAPER_MONTE_CARLO_NEXT_ACTION_MAX_COMMAND_CHARS:
        return f"{field}={text}"
    return f"{field}={text[:_PAPER_MONTE_CARLO_NEXT_ACTION_MAX_COMMAND_CHARS]}..."


def _paper_monte_carlo_threshold_label(name: str, value: Any, threshold: Any) -> str:
    if value is None:
        return ""
    value_label = _paper_monte_carlo_format_measure(value, field=name)
    if threshold is None:
        return f"{name}={value_label}"
    threshold_label = _paper_monte_carlo_format_measure(threshold, field=f"{name} threshold")
    return f"{name}={value_label} <= {threshold_label}"


def _paper_monte_carlo_count_pair(numerator: Any, denominator: Any) -> str:
    return f"{_paper_monte_carlo_format_count(numerator)}/{_paper_monte_carlo_format_count(denominator)}"


def _paper_monte_carlo_format_count(value: Any) -> str:
    if value is None:
        return "NA"
    if _paper_monte_carlo_is_missing_display_value(value):
        return "NA"
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"Paper Monte Carlo display count is boolean: {value!r}")
    if isinstance(value, Mapping):
        return _paper_monte_carlo_json_display(dict(value))
    if isinstance(value, (list, tuple, np.ndarray)):
        return _paper_monte_carlo_json_display(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric):
            if math.isnan(numeric):
                return "NA"
            return "+Inf" if numeric > 0 else "-Inf"
        if numeric < 0:
            raise ValueError(f"Paper Monte Carlo display count is negative: {value!r}")
        if not numeric.is_integer():
            raise ValueError(
                f"Paper Monte Carlo display count is not an integer: {value!r}"
            )
        return str(int(numeric))
    text = _paper_monte_carlo_display_text(value, field="count")
    try:
        numeric = float(text)
    except ValueError as exc:
        raise ValueError(
            f"Paper Monte Carlo display count is not numeric: {value!r}"
        ) from exc
    if not math.isfinite(numeric):
        if math.isnan(numeric):
            return "NA"
        return "+Inf" if numeric > 0 else "-Inf"
    if numeric < 0:
        raise ValueError(f"Paper Monte Carlo display count is negative: {value!r}")
    if not numeric.is_integer():
        raise ValueError(f"Paper Monte Carlo display count is not an integer: {value!r}")
    return str(int(numeric))


def _paper_monte_carlo_blockers_label(value: Any) -> str:
    if not isinstance(value, Mapping) or not value:
        return "blockers=none"
    active_keys = [
        key for key in sorted(value) if _paper_monte_carlo_blocker_count_is_active(value[key])
    ]
    if not active_keys:
        return "blockers=none"
    return "blockers=" + ",".join(
        f"{key}:{_paper_monte_carlo_format_count(value[key])}" for key in active_keys
    )


def _paper_monte_carlo_blocker_count_is_active(value: Any) -> bool:
    if _paper_monte_carlo_is_missing_display_value(value):
        return True
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"Paper Monte Carlo display count is boolean: {value!r}")
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if math.isnan(numeric):
            return True
        if numeric < 0:
            raise ValueError(f"Paper Monte Carlo display count is negative: {value!r}")
        return numeric > 0
    return True


def _paper_monte_carlo_compact_join(parts: tuple[str, ...]) -> str:
    return "; ".join(part for part in parts if part)


def _paper_monte_carlo_format_value(value: Any) -> str:
    if value is None:
        return ""
    if _paper_monte_carlo_is_missing_display_value(value):
        return "NA"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, Mapping):
        return _paper_monte_carlo_json_display(dict(value))
    if isinstance(value, (list, tuple, np.ndarray)):
        return _paper_monte_carlo_json_display(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric):
            if math.isnan(numeric):
                return "NA"
            return "+Inf" if numeric > 0 else "-Inf"
        if numeric.is_integer():
            return str(int(numeric))
        return f"{numeric:.6f}"
    return str(value)


def _paper_monte_carlo_format_measure(value: Any, *, field: str) -> str:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"Paper Monte Carlo display measure {field} is boolean: {value!r}"
        )
    return _paper_monte_carlo_format_value(value)


def _paper_monte_carlo_json_display(value: Any) -> str:
    text = json.dumps(
        _json_safe_export_payload(value),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(text) <= _PAPER_MONTE_CARLO_JSON_DISPLAY_MAX_CHARS:
        return text
    head_chars = (
        _PAPER_MONTE_CARLO_JSON_DISPLAY_MAX_CHARS
        - _PAPER_MONTE_CARLO_JSON_DISPLAY_TAIL_CHARS
        - 3
    )
    return f"{text[:head_chars]}...{text[-_PAPER_MONTE_CARLO_JSON_DISPLAY_TAIL_CHARS:]}"


def _paper_monte_carlo_format_rate(value: Any) -> str:
    if value is None:
        return ""
    if _paper_monte_carlo_is_missing_display_value(value):
        return "NA"
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"Paper Monte Carlo display rate is boolean: {value!r}")
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        numeric = float(value)
        if not math.isfinite(numeric):
            if math.isnan(numeric):
                return "NA"
            return "+Inf" if numeric > 0 else "-Inf"
        if numeric == 0:
            return "0"
        return f"{numeric:.6f}"
    if isinstance(value, str):
        text = value.strip()
        try:
            numeric = float(text)
        except ValueError:
            return _paper_monte_carlo_format_value(value)
        if not math.isfinite(numeric):
            if math.isnan(numeric):
                return "NA"
            return "+Inf" if numeric > 0 else "-Inf"
        if numeric == 0:
            return "0"
        return f"{numeric:.6f}"
    return _paper_monte_carlo_format_value(value)


def _paper_monte_carlo_display_text(value: Any, *, field: str) -> str:
    if _paper_monte_carlo_is_missing_display_value(value):
        raise ValueError(f"Paper Monte Carlo display field {field} is missing")
    text = str(value).strip()
    if not text:
        raise ValueError(f"Paper Monte Carlo display field {field} is blank")
    return text


_PAPER_MONTE_CARLO_LABEL_MAX_DISPLAY_CHARS = 96
_PAPER_MONTE_CARLO_LABEL_TAIL_CHARS = 36


def _paper_monte_carlo_display_label(value: Any, *, field: str) -> str:
    return _paper_monte_carlo_compact_display_label(
        _paper_monte_carlo_display_text(value, field=field)
    )


def _paper_monte_carlo_compact_display_label(text: str) -> str:
    if len(text) <= _PAPER_MONTE_CARLO_LABEL_MAX_DISPLAY_CHARS:
        return text
    head_chars = (
        _PAPER_MONTE_CARLO_LABEL_MAX_DISPLAY_CHARS
        - _PAPER_MONTE_CARLO_LABEL_TAIL_CHARS
        - 3
    )
    return f"{text[:head_chars]}...{text[-_PAPER_MONTE_CARLO_LABEL_TAIL_CHARS:]}"


def _paper_monte_carlo_is_missing_display_value(value: Any) -> bool:
    return _is_scalar_missing(value)


def write_paper_monte_carlo_reproduction_report_json(
    output_path: str | Path,
    *,
    evidence_dir: str | Path,
    overwrite: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Write the paper Monte Carlo reproduction report as strict JSON.

    Parameters
    ----------
    output_path : str or Path
        Destination file path.
    evidence_dir : str or Path
        Directory containing chunk-evidence JSON files.
    overwrite : bool
        If ``True``, overwrite an existing file.
    **kwargs
        Forwarded to :func:`paper_monte_carlo_reproduction_report`.

    Returns
    -------
    dict[str, Any]
        The serialized report payload.

    Raises
    ------
    FileExistsError
        If *output_path* exists and *overwrite* is ``False``.
    """

    _require_monte_carlo_writer_overwrite(overwrite)
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists; pass overwrite=True to replace it."
    )
    report = paper_monte_carlo_reproduction_report(evidence_dir, **kwargs)
    payload = report.to_dict()
    _write_monte_carlo_json_atomic(path, payload)
    return payload


def load_paper_monte_carlo_reproduction_report_json(
    path_like: str | Path,
) -> PaperMonteCarloReproductionReport:
    """Load a saved strict-JSON paper Monte Carlo reproduction report.

    Parameters
    ----------
    path_like : str or Path
        Path to the JSON file.

    Returns
    -------
    PaperMonteCarloReproductionReport
        Deserialized report object.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is not valid strict JSON or has unexpected schema.
    """

    path = Path(path_like)
    if not path.is_file():
        raise FileNotFoundError(
            f"paper Monte Carlo reproduction report JSON file does not exist: {path}"
        )
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_monte_carlo_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"paper Monte Carlo reproduction report must be valid JSON: {path}"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {
        "summary",
        "rows",
        "evidence_summary",
    }:
        raise ValueError(
            "paper Monte Carlo reproduction report JSON must contain only summary, "
            "rows, and evidence_summary."
        )
    summary = _require_json_object(payload["summary"], "summary")
    rows = _require_json_object_sequence(payload["rows"], "rows")
    evidence_summary = _require_json_object(payload["evidence_summary"], "evidence_summary")
    _reject_nonfinite_json_numbers(
        summary,
        field_name="paper Monte Carlo reproduction report summary",
    )
    _reject_nonfinite_json_numbers(
        rows,
        field_name="paper Monte Carlo reproduction report rows",
    )
    _reject_nonfinite_json_numbers(
        evidence_summary,
        field_name="paper Monte Carlo reproduction report evidence_summary",
    )
    return PaperMonteCarloReproductionReport(
        rows=rows,
        summary=summary,
        evidence_summary=evidence_summary,
    )


def _paper_monte_carlo_reproduction_rows(
    acceptance_frame: pd.DataFrame,
    *,
    evidence_summary: dict[str, Any],
    alpha: float,
) -> tuple[dict[str, Any], ...]:
    if acceptance_frame.empty:
        return (
            _paper_monte_carlo_reproduction_schedule_row(
                evidence_summary,
                alpha=alpha,
            ),
        )

    gate = dict(evidence_summary["paper_acceptance_gate"])
    progress = dict(evidence_summary["progress_summary"])
    rows: list[dict[str, Any]] = []
    for row in acceptance_frame.to_dict("records"):
        evidence_row = {
            **row,
            "case_id": _paper_monte_carlo_reproduction_case_id(row),
            "row_type": "monte_carlo_evidence_row",
            "paper_anchor": (
                "manuscript/sources/arxiv-2404.11739v3/draft.tex:391-456; "
                "manuscript/sources/arxiv-2404.11739v3/draft.tex:1094-1156"
            ),
            "reference_anchor": (
                "packages/r/TestMechs/R/simulate_data_binaryM.R:1-35; "
                "packages/r/TestMechs/R/test_sharp_null.R:90-315; "
                "packages/r/TestMechs/R/nonparametric_bootstrap.R:16-71"
            ),
            "truth_hierarchy": "paper_monte_carlo_protocol_then_python_strict_json_evidence_with_r_as_reference",
            "paper_nominal_alpha": alpha,
            "evidence_dir": evidence_summary["evidence_dir"],
            "evidence_file_count": evidence_summary["evidence_file_count"],
            "scheduled_file_count": evidence_summary["scheduled_file_count"],
            "paper_acceptance_stage": gate["stage"],
            "paper_acceptance_gate_passes": gate["gate_passes"],
            "active_blocking_conditions": dict(gate["active_blocking_conditions"]),
            "progress_next_action": progress["next_action"],
            "next_action": evidence_summary["next_action"],
            "exception_status": (
                "paper_acceptance_gate_passes"
                if gate["gate_passes"]
                else "paper_acceptance_gate_blocked"
            ),
        }
        rows.append(evidence_row)
    return tuple(rows)


def _paper_monte_carlo_reproduction_schedule_row(
    evidence_summary: dict[str, Any],
    *,
    alpha: float,
) -> dict[str, Any]:
    schedule = dict(evidence_summary["schedule_summary"])
    progress = dict(evidence_summary["progress_summary"])
    gate = dict(evidence_summary["paper_acceptance_gate"])
    return {
        "case_id": "paper_monte_carlo_default_schedule",
        "row_type": "monte_carlo_schedule_row",
        "stage": schedule["stage"],
        "runner": schedule["runner"],
        "status": "scheduled",
        "paper_anchor": (
            "manuscript/sources/arxiv-2404.11739v3/draft.tex:391-456; "
            "manuscript/sources/arxiv-2404.11739v3/draft.tex:1094-1156"
        ),
        "reference_anchor": (
            "packages/r/TestMechs/R/simulate_data_binaryM.R:1-35; "
            "packages/r/TestMechs/R/test_sharp_null.R:90-315; "
            "packages/r/TestMechs/R/nonparametric_bootstrap.R:16-71"
        ),
        "truth_hierarchy": "paper_monte_carlo_protocol_then_python_strict_json_evidence_with_r_as_reference",
        "paper_nominal_alpha": alpha,
        "paper_replications": schedule.get("paper_replications"),
        "bootstrap_replications": schedule.get("bootstrap_replications"),
        "paper_default_replications": schedule.get("paper_default_replications"),
        "paper_default_bootstrap_replications": schedule.get(
            "paper_default_bootstrap_replications"
        ),
        "paper_replication_budget_ready": schedule.get(
            "paper_replication_budget_ready"
        ),
        "chunk_replication_budget_ready": schedule.get(
            "chunk_replication_budget_ready"
        ),
        "bootstrap_budget_ready": schedule.get("bootstrap_budget_ready"),
        "full_paper_budget_ready": schedule.get("full_paper_budget_ready"),
        "tolerance_contract_status": schedule.get("tolerance_contract_status"),
        "paper_result_rows": schedule.get("paper_result_rows"),
        "scheduled_result_rows": progress.get("scheduled_result_rows"),
        "executed_result_rows": progress.get("executed_result_rows"),
        "scheduled_draws": progress.get("scheduled_draws"),
        "executed_draws": progress.get("executed_draws"),
        "scheduled_bootstrap_draws": progress.get("scheduled_bootstrap_draws"),
        "executed_bootstrap_draws": progress.get("executed_bootstrap_draws"),
        "evidence_dir": evidence_summary["evidence_dir"],
        "evidence_file_count": evidence_summary["evidence_file_count"],
        "scheduled_file_count": evidence_summary["scheduled_file_count"],
        "stale_evidence_file_count": evidence_summary.get("stale_evidence_file_count"),
        "stale_evidence_files": evidence_summary.get("stale_evidence_files"),
        "stale_evidence_error": evidence_summary.get("stale_evidence_error"),
        "paper_acceptance_stage": gate["stage"],
        "paper_acceptance_gate_passes": gate["gate_passes"],
        "active_blocking_conditions": dict(gate["active_blocking_conditions"]),
        "next_chunk_kwargs": evidence_summary.get("next_chunk_kwargs"),
        "next_chunk_rerun_call": evidence_summary.get("next_chunk_rerun_call"),
        "next_chunk_evidence_path": evidence_summary.get("next_chunk_evidence_path"),
        "next_chunk_export_call": evidence_summary.get("next_chunk_export_call"),
        "next_action": evidence_summary["next_action"],
        "exception_status": "paper_acceptance_gate_blocked",
    }


def _paper_monte_carlo_reproduction_summary(
    rows: tuple[dict[str, Any], ...],
    *,
    acceptance_frame: pd.DataFrame,
    evidence_summary: dict[str, Any],
    alpha: float,
    runtime_seconds: float,
) -> dict[str, Any]:
    schedule = dict(evidence_summary["schedule_summary"])
    progress = dict(evidence_summary["progress_summary"])
    gate = dict(evidence_summary["paper_acceptance_gate"])
    active_blocking_conditions = dict(gate.get("active_blocking_conditions", {}))
    rejection_error = _max_report_numeric(
        acceptance_frame,
        "rejection_rate_absolute_error",
        "absolute_error",
    )
    return {
        "row_count": len(rows),
        "paper_nominal_alpha": alpha,
        "paper_result_rows": gate.get("paper_result_rows"),
        "covered_result_rows": gate.get("covered_result_rows"),
        "scheduled_result_rows": progress.get("scheduled_result_rows"),
        "executed_result_rows": progress.get("executed_result_rows"),
        "scheduled_draws": progress.get("scheduled_draws"),
        "executed_draws": progress.get("executed_draws"),
        "scheduled_bootstrap_draws": progress.get("scheduled_bootstrap_draws"),
        "executed_bootstrap_draws": progress.get("executed_bootstrap_draws"),
        "paper_default_replications": schedule.get("paper_default_replications"),
        "paper_default_bootstrap_replications": schedule.get(
            "paper_default_bootstrap_replications"
        ),
        "paper_replication_budget_ready": schedule.get(
            "paper_replication_budget_ready"
        ),
        "chunk_replication_budget_ready": schedule.get(
            "chunk_replication_budget_ready"
        ),
        "bootstrap_budget_ready": schedule.get("bootstrap_budget_ready"),
        "full_paper_budget_ready": schedule.get("full_paper_budget_ready"),
        "tolerance_contract_status": gate.get(
            "tolerance_contract_status",
            schedule.get("tolerance_contract_status"),
        ),
        "evidence_file_count": evidence_summary["evidence_file_count"],
        "scheduled_file_count": evidence_summary["scheduled_file_count"],
        "stale_evidence_file_count": evidence_summary.get("stale_evidence_file_count"),
        "stale_evidence_files": evidence_summary.get("stale_evidence_files"),
        "stale_evidence_error": evidence_summary.get("stale_evidence_error"),
        "paper_acceptance_stage": gate["stage"],
        "paper_acceptance_gate_passes": gate["gate_passes"],
        "active_blocking_conditions": active_blocking_conditions,
        "low_budget_probe_blocked": bool(
            evidence_summary["evidence_file_count"] > 0
            and not gate["gate_passes"]
            and (
                active_blocking_conditions.get("target_replication_shortfall_rows", 0)
                > 0
                or active_blocking_conditions.get(
                    "bootstrap_replication_shortfall_rows",
                    0,
                )
                > 0
            )
        ),
        "max_rejection_rate_absolute_error": rejection_error,
        "target_precision_rows": _numeric_column_count(
            acceptance_frame,
            "target_mc_standard_error",
        ),
        "max_target_mc_standard_error": _max_numeric_column(
            acceptance_frame,
            "target_mc_standard_error",
        ),
        "source_mixture_precision_rows": _numeric_column_count(
            acceptance_frame,
            "source_mixture_mc_standard_error",
        ),
        "max_source_mixture_mc_standard_error": _max_numeric_column(
            acceptance_frame,
            "source_mixture_mc_standard_error",
        ),
        "cell_count_policy_rows": _numeric_column_count(
            acceptance_frame,
            "target_median_independent_clusters_per_cell",
        ),
        "cell_count_policy_size_risk_rows": _truthy_column_count(
            acceptance_frame,
            "cell_count_policy_size_risk",
        ),
        "method_blocker_rows": len(gate.get("blocking_condition_rows", ())),
        "unresolved_row_count": evidence_summary["unresolved_row_count"],
        "unresolved_root_cause_counts": dict(
            evidence_summary.get("unresolved_root_cause_counts", {})
        ),
        "next_chunk_kwargs": evidence_summary.get("next_chunk_kwargs"),
        "next_chunk_rerun_call": evidence_summary.get("next_chunk_rerun_call"),
        "next_chunk_evidence_path": evidence_summary.get("next_chunk_evidence_path"),
        "next_chunk_export_call": evidence_summary.get("next_chunk_export_call"),
        "next_action": evidence_summary["next_action"],
        "runtime_seconds": runtime_seconds,
        "paper_anchors": (
            "manuscript/sources/arxiv-2404.11739v3/draft.tex:391-456",
            "manuscript/sources/arxiv-2404.11739v3/draft.tex:1094-1156",
        ),
        "reference_anchors": (
            "packages/r/TestMechs/R/simulate_data_binaryM.R:1-35",
            "packages/r/TestMechs/R/test_sharp_null.R:90-315",
            "packages/r/TestMechs/R/nonparametric_bootstrap.R:16-71",
        ),
    }


def _paper_monte_carlo_reproduction_case_id(row: dict[str, Any]) -> str:
    raw = "|".join(
        str(row.get(key))
        for key in ("method", "table", "design", "mediator", "clusters", "bins", "t")
    )
    return re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_").lower()



def _max_report_numeric(frame: pd.DataFrame, *columns: str) -> float | int | None:
    for column in columns:
        value = _max_numeric_column(frame, column)
        if value is not None:
            return value
    return None


@dataclass(frozen=True)
class MonteCarloContracts:
    """Executable contract surface for the paper Monte Carlo tables."""

    result_rows: tuple[MonteCarloResultRow, ...]
    cell_counts: tuple[ClusterCellCount, ...]

    def raise_for_method_support_blockers(
        self,
        *,
        nominal_alpha: float = 0.05,
        tolerance: float = 0.025,
    ) -> None:
        """Raise if Python support does not cover every paper-reported method row."""

        summary = self.method_support_blocker_summary(
            nominal_alpha=nominal_alpha,
            tolerance=tolerance,
        )
        if bool(summary["method_support_blocker_gate_passes"]):
            return
        raise AssertionError(
            "Monte Carlo paper method support gate is blocked: "
            f"{summary['python_blocked_method_rows']} of "
            f"{summary['paper_reported_method_rows']} paper method rows are not "
            "Python-executable; "
            f"blocked_methods={summary['blocked_method_names']}; "
            f"blocking_reason_counts={summary['blocking_reason_counts']}; "
            f"next_action={summary['next_action']}; "
            f"exit_criteria={summary['exit_criteria']}"
        )

    def result_row(
        self,
        *,
        table: str,
        design: str,
        mediator: str,
        clusters: int | None,
        bins: int | None,
        t: float,
    ) -> MonteCarloResultRow:
        for paper_row_index, row in enumerate(self.result_rows):
            if (
                row.table == table
                and row.design == design
                and row.mediator == mediator
                and row.clusters == clusters
                and row.bins == bins
                and abs(row.t - float(t)) <= 1e-12
            ):
                return row
        raise KeyError(
            "No Monte Carlo result row for "
            f"table={table!r}, design={design!r}, mediator={mediator!r}, "
            f"clusters={clusters!r}, bins={bins!r}, t={t!r}."
        )

    def cell_count(
        self,
        *,
        design: str,
        mediator: str,
        clusters: int,
        bins: int,
        t: float,
    ) -> ClusterCellCount:
        for row in self.cell_counts:
            if (
                row.design == design
                and row.mediator == mediator
                and row.clusters == clusters
                and row.bins == bins
                and abs(row.t - float(t)) <= 1e-12
            ):
                return row
        raise KeyError(
            "No clustered cell-count row for "
            f"design={design!r}, mediator={mediator!r}, clusters={clusters!r}, "
            f"bins={bins!r}, t={t!r}."
        )

    def size_risk_cell_counts(self) -> tuple[ClusterCellCount, ...]:
        return tuple(row for row in self.cell_counts if row.size_risk)

    def cell_count_heuristics(
        self,
        *,
        method: str = "CS",
        nominal_alpha: float = 0.05,
        tolerance: float = 0.025,
        size_risk_threshold: float = 15.0,
    ) -> tuple[MonteCarloCellCountHeuristic, ...]:
        """Translate the paper's 15-independent-unit bin heuristic into contracts."""

        if not 0 < nominal_alpha < 1:
            raise ValueError("nominal_alpha must be strictly between 0 and 1.")
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative.")
        if size_risk_threshold < 0:
            raise ValueError("size_risk_threshold must be non-negative.")
        if not any(method in row.rejection_rates for row in self.result_rows):
            raise KeyError(f"No Monte Carlo result rows expose method {method!r}.")

        grouped_counts: dict[tuple[str, str, int], list[ClusterCellCount]] = {}
        for cell_count in self.cell_counts:
            key = (cell_count.design, cell_count.mediator, cell_count.clusters)
            grouped_counts.setdefault(key, []).append(cell_count)

        heuristics: list[MonteCarloCellCountHeuristic] = []
        for (design, mediator, clusters), counts in grouped_counts.items():
            bin_values = sorted({count.bins for count in counts})
            min_counts_by_bins = {
                bins: float(
                    min(
                        count.median_independent_clusters_per_cell
                        for count in counts
                        if count.bins == bins
                    )
                )
                for bins in bin_values
            }
            null_rejection_rate_by_bins: dict[int, float] = {}
            for bins in bin_values:
                null_row = self.result_row(
                    table=_result_table_for_mediator_and_bins(mediator=mediator, bins=bins),
                    design=design,
                    mediator=mediator,
                    clusters=clusters,
                    bins=bins,
                    t=0.0,
                )
                if method in null_row.rejection_rates:
                    null_rejection_rate_by_bins[bins] = null_row.rejection_rates[method]

            heuristics.append(
                MonteCarloCellCountHeuristic(
                    design=design,
                    mediator=mediator,
                    clusters=clusters,
                    method=method,
                    nominal_alpha=nominal_alpha,
                    tolerance=tolerance,
                    size_risk_threshold=size_risk_threshold,
                    min_median_independent_clusters_per_cell_by_bins=min_counts_by_bins,
                    null_rejection_rate_by_bins=null_rejection_rate_by_bins,
                )
            )
        return tuple(heuristics)

    def cell_count_heuristics_frame(
        self,
        *,
        method: str = "CS",
        nominal_alpha: float = 0.05,
        tolerance: float = 0.025,
        size_risk_threshold: float = 15.0,
    ) -> pd.DataFrame:
        """Return row-level bin guidance from the paper cell-count heuristic."""

        frames = [
            heuristic.to_frame()
            for heuristic in self.cell_count_heuristics(
                method=method,
                nominal_alpha=nominal_alpha,
                tolerance=tolerance,
                size_risk_threshold=size_risk_threshold,
            )
        ]
        if not frames:
            return pd.DataFrame(
                columns=[
                    "design",
                    "mediator",
                    "clusters",
                    "method",
                    "bins",
                    "min_median_independent_clusters_per_cell",
                    "size_risk_threshold",
                    "cell_count_size_risk",
                    "recommended_by_cell_count",
                    "recommended_max_bins",
                    "null_rejection_rate",
                    "nominal_alpha",
                    "tolerance",
                    "size_distortion",
                    "bin_policy",
                    "paper_rule",
                ]
            )
        return pd.concat(frames, ignore_index=True)

    def benchmark_cells(
        self,
        *,
        method: str = "CS",
        mediator: str = "binary",
        design: str | None = None,
        table: str | None = None,
        clusters: tuple[int | None, ...] | None = None,
        bins: tuple[int | None, ...] | None = None,
        t_values: tuple[float, ...] | None = None,
        supported_only: bool = True,
    ) -> tuple[MonteCarloBenchmarkCell, ...]:
        """Return structured paper cells for executable simulation benchmarks."""

        cells: list[MonteCarloBenchmarkCell] = []
        for paper_row_index, row in enumerate(self.result_rows):
            if row.mediator != mediator:
                continue
            if design is not None and row.design != design:
                continue
            if table is not None and row.table != table:
                continue
            if clusters is not None and row.clusters not in clusters:
                continue
            if bins is not None and row.bins not in bins:
                continue
            if t_values is not None and not any(abs(row.t - float(t_value)) <= 1e-12 for t_value in t_values):
                continue
            if method not in row.rejection_rates:
                if supported_only:
                    continue
                raise KeyError(
                    f"Monte Carlo row table={row.table!r}, design={row.design!r}, "
                    f"mediator={row.mediator!r}, clusters={row.clusters!r}, bins={row.bins!r}, "
                    f"t={row.t!r} does not expose method {method!r}."
                )
            cells.append(
                MonteCarloBenchmarkCell.from_result_row(
                    row,
                    method=method,
                    cluster_cell_count=self._cluster_cell_count_for_result_row(row),
                    paper_row_index=paper_row_index,
                    benchmark_row_index=len(cells),
                )
            )
        return tuple(cells)

    def size_diagnostics(
        self,
        *,
        method: str,
        nominal_alpha: float = 0.05,
        tolerance: float = 0.025,
    ) -> tuple[MonteCarloSizeDiagnostic, ...]:
        """Return null-row size diagnostics joined to paper cell-count contracts."""

        if not 0 < nominal_alpha < 1:
            raise ValueError("nominal_alpha must be strictly between 0 and 1.")
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative.")
        if not any(method in row.rejection_rates for row in self.result_rows):
            raise KeyError(f"No Monte Carlo result rows expose method {method!r}.")

        diagnostics: list[MonteCarloSizeDiagnostic] = []
        for row in self.result_rows:
            if not row.is_null_size_row or method not in row.rejection_rates:
                continue
            diagnostics.append(
                MonteCarloSizeDiagnostic(
                    row=row,
                    method=method,
                    nominal_alpha=nominal_alpha,
                    tolerance=tolerance,
                    cluster_cell_count=self._cluster_cell_count_for_result_row(row),
                )
            )
        return tuple(diagnostics)

    def method_summary(
        self,
        *,
        method: str,
        nominal_alpha: float = 0.05,
        tolerance: float = 0.025,
    ) -> dict[str, Any]:
        """Summarize paper-table support and size/power roles for one inference method."""

        diagnostics = self.size_diagnostics(
            method=method,
            nominal_alpha=nominal_alpha,
            tolerance=tolerance,
        )
        supported_rows = tuple(row for row in self.result_rows if method in row.rejection_rates)
        size_rows = tuple(row for row in supported_rows if row.is_null_size_row)
        power_rows = tuple(row for row in supported_rows if not row.is_null_size_row)

        return {
            "method": method,
            "total_result_rows": len(self.result_rows),
            "supported_result_rows": len(supported_rows),
            "unsupported_result_rows": len(self.result_rows) - len(supported_rows),
            "size_rows": len(size_rows),
            "power_rows": len(power_rows),
            "size_distortion_rows": int(sum(diag.size_distortion for diag in diagnostics)),
            "cell_count_size_risk_rows": int(sum(diag.cell_count_size_risk for diag in diagnostics)),
            "attention_size_rows": int(sum(diag.needs_attention for diag in diagnostics)),
            "max_size_rejection_rate": _max_rejection_rate(size_rows, method=method),
            "min_power_rejection_rate": _min_rejection_rate(power_rows, method=method),
            "max_power_rejection_rate": _max_rejection_rate(power_rows, method=method),
        }

    def method_guidance(
        self,
        *,
        method: str,
        nominal_alpha: float = 0.05,
        tolerance: float = 0.025,
    ) -> MonteCarloMethodGuidance:
        """Return paper method-choice guidance joined to current Python support."""

        _require_monte_carlo_method_guidance(method)
        return _monte_carlo_method_guidance(
            method=method,
            method_summary=self.method_summary(
                method=method,
                nominal_alpha=nominal_alpha,
                tolerance=tolerance,
            ),
        )

    def method_guidance_frame(
        self,
        *,
        nominal_alpha: float = 0.05,
        tolerance: float = 0.025,
    ) -> pd.DataFrame:
        """Return one row per paper method with role, risk, and Python status."""

        methods = _ordered_monte_carlo_methods(
            {method for row in self.result_rows for method in row.rejection_rates}
        )
        return pd.DataFrame(
            [
                {
                    **self.method_guidance(
                        method=method,
                        nominal_alpha=nominal_alpha,
                        tolerance=tolerance,
                    ).to_dict(),
                    "paper_order_index": index,
                }
                for index, method in enumerate(methods)
            ]
        )

    def method_guidance_summary(
        self,
        *,
        nominal_alpha: float = 0.05,
        tolerance: float = 0.025,
    ) -> dict[str, Any]:
        """Return compact method-choice counts for the paper Monte Carlo surface."""

        frame = self.method_guidance_frame(
            nominal_alpha=nominal_alpha,
            tolerance=tolerance,
        )
        executable = frame["python_executable"].map(_truthy_value)
        paper_default = frame["paper_default"].map(_truthy_value)
        needs_size_caution = frame["needs_size_caution"].map(_truthy_value)
        return {
            "paper_methods": int(frame.shape[0]),
            "python_executable_methods": int(executable.sum()),
            "paper_contract_only_methods": int((~executable).sum()),
            "paper_default_methods": int(paper_default.sum()),
            "methods_needing_size_caution": int(needs_size_caution.sum()),
            "paper_default_executable": int((paper_default & executable).sum()),
            "small_cluster_alternative_methods": int(
                frame["small_cluster_size_control_alternative"].map(_truthy_value).sum()
            ),
            "large_sample_power_candidate_methods": int(
                frame["large_independent_sample_power_candidate"].map(_truthy_value).sum()
            ),
            "binary_mediator_comparator_methods": int(
                frame["binary_mediator_comparator"].map(_truthy_value).sum()
            ),
            "executable_method_names": tuple(
                frame.loc[executable, "method"].astype(str).tolist()
            ),
            "paper_contract_only_method_names": tuple(
                frame.loc[~executable, "method"].astype(str).tolist()
            ),
        }

    def method_execution_contract(
        self,
        *,
        method: str,
        nominal_alpha: float = 0.05,
        tolerance: float = 0.025,
        paper_replications: int = 500,
    ) -> MonteCarloMethodExecutionContract:
        """Return the paper's method-specific Monte Carlo execution protocol."""

        if int(paper_replications) <= 0:
            raise ValueError("paper_replications must be positive.")
        guidance = self.method_guidance(
            method=method,
            nominal_alpha=nominal_alpha,
            tolerance=tolerance,
        )
        acceptance_summary = self.method_acceptance_summary(
            method=method,
            nominal_alpha=nominal_alpha,
            tolerance=tolerance,
        )
        support_gate = _python_method_support_gate(
            method=method,
            result_rows=self.result_rows,
            acceptance_summary=acceptance_summary,
        )
        return _monte_carlo_method_execution_contract(
            guidance=guidance,
            support_gate=support_gate,
            nominal_alpha=nominal_alpha,
            paper_replications=int(paper_replications),
        )

    def method_execution_contract_frame(
        self,
        *,
        nominal_alpha: float = 0.05,
        tolerance: float = 0.025,
        paper_replications: int = 500,
    ) -> pd.DataFrame:
        """Return one row per paper method with its execution protocol."""

        methods = _ordered_monte_carlo_methods(
            {method for row in self.result_rows for method in row.rejection_rates}
        )
        return _json_safe_export_frame(
            [
                self.method_execution_contract(
                    method=method,
                    nominal_alpha=nominal_alpha,
                    tolerance=tolerance,
                    paper_replications=paper_replications,
                ).to_dict()
                for method in methods
            ]
        )

    def method_execution_contract_summary(
        self,
        *,
        nominal_alpha: float = 0.05,
        tolerance: float = 0.025,
        paper_replications: int = 500,
    ) -> dict[str, Any]:
        """Return compact counts for the paper method execution protocol."""

        frame = self.method_execution_contract_frame(
            nominal_alpha=nominal_alpha,
            tolerance=tolerance,
            paper_replications=paper_replications,
        )
        if frame.empty:
            return {
                "paper_methods": 0,
                "paper_nominal_alpha": nominal_alpha,
                "paper_replications": int(paper_replications),
                "methods_using_discretized_outcome": 0,
                "methods_using_original_outcome": 0,
                "bootstrap_required_methods": 0,
                "bootstrap_individual_unit_methods": 0,
                "bootstrap_cluster_unit_methods": 0,
                "analytic_variance_methods": 0,
                "bootstrap_resampling_methods": 0,
                "binary_mediator_methods": 0,
                "nonbinary_mediator_methods": 0,
                "paper_reported_method_rows": 0,
                "python_executable_reported_rows": 0,
                "python_blocked_reported_rows": 0,
                "python_executable_methods": 0,
                "paper_contract_only_methods": 0,
                "paper_reported_rows_by_method": {},
                "python_executable_reported_rows_by_method": {},
                "python_blocked_reported_rows_by_method": {},
                "discretized_outcome_method_names": (),
                "original_outcome_method_names": (),
                "bootstrap_required_method_names": (),
                "bootstrap_individual_unit_method_names": (),
                "bootstrap_cluster_unit_method_names": (),
                "analytic_variance_method_names": (),
                "python_executable_method_names": (),
                "paper_contract_only_method_names": (),
                "exit_criteria": "method_execution_contract_frame covers every reported paper method",
            }

        uses_discretized = frame["uses_discretized_outcome"].map(_truthy_value)
        bootstrap_required = frame["bootstrap_required"].map(_truthy_value)
        bootstrap_individual = frame["bootstrap_unit_modes"].map(
            lambda modes: "individual" in tuple(modes)
        )
        bootstrap_cluster = frame["bootstrap_unit_modes"].map(
            lambda modes: "cluster" in tuple(modes)
        )
        analytic_variance = frame["variance_estimator"].eq("analytic")
        bootstrap_resampling = frame["variance_estimator"].eq("bootstrap")
        supports_binary = frame["supports_binary_mediator"].map(_truthy_value)
        supports_nonbinary = frame["supports_nonbinary_mediator"].map(_truthy_value)
        python_executable = frame["python_executable"].map(_truthy_value)
        paper_reported_rows_by_method = {
            str(row["method"]): int(row["paper_reported_rows"])
            for row in frame.to_dict("records")
        }
        python_executable_reported_rows_by_method = {
            str(row["method"]): int(row["python_executable_reported_rows"])
            for row in frame.to_dict("records")
        }
        python_blocked_reported_rows_by_method = {
            str(row["method"]): int(row["python_blocked_reported_rows"])
            for row in frame.to_dict("records")
        }
        return {
            "paper_methods": int(frame.shape[0]),
            "paper_nominal_alpha": nominal_alpha,
            "paper_replications": int(paper_replications),
            "methods_using_discretized_outcome": int(uses_discretized.sum()),
            "methods_using_original_outcome": int((~uses_discretized).sum()),
            "bootstrap_required_methods": int(bootstrap_required.sum()),
            "bootstrap_individual_unit_methods": int(bootstrap_individual.sum()),
            "bootstrap_cluster_unit_methods": int(bootstrap_cluster.sum()),
            "analytic_variance_methods": int(analytic_variance.sum()),
            "bootstrap_resampling_methods": int(bootstrap_resampling.sum()),
            "binary_mediator_methods": int(supports_binary.sum()),
            "nonbinary_mediator_methods": int(supports_nonbinary.sum()),
            "paper_reported_method_rows": int(frame["paper_reported_rows"].sum()),
            "python_executable_reported_rows": int(
                frame["python_executable_reported_rows"].sum()
            ),
            "python_blocked_reported_rows": int(
                frame["python_blocked_reported_rows"].sum()
            ),
            "python_executable_methods": int(python_executable.sum()),
            "paper_contract_only_methods": int((~python_executable).sum()),
            "paper_reported_rows_by_method": paper_reported_rows_by_method,
            "python_executable_reported_rows_by_method": python_executable_reported_rows_by_method,
            "python_blocked_reported_rows_by_method": python_blocked_reported_rows_by_method,
            "discretized_outcome_method_names": tuple(
                frame.loc[uses_discretized, "method"].astype(str).tolist()
            ),
            "original_outcome_method_names": tuple(
                frame.loc[~uses_discretized, "method"].astype(str).tolist()
            ),
            "bootstrap_required_method_names": tuple(
                frame.loc[bootstrap_required, "method"].astype(str).tolist()
            ),
            "bootstrap_individual_unit_method_names": tuple(
                frame.loc[bootstrap_individual, "method"].astype(str).tolist()
            ),
            "bootstrap_cluster_unit_method_names": tuple(
                frame.loc[bootstrap_cluster, "method"].astype(str).tolist()
            ),
            "analytic_variance_method_names": tuple(
                frame.loc[analytic_variance, "method"].astype(str).tolist()
            ),
            "python_executable_method_names": tuple(
                frame.loc[python_executable, "method"].astype(str).tolist()
            ),
            "paper_contract_only_method_names": tuple(
                frame.loc[~python_executable, "method"].astype(str).tolist()
            ),
            "exit_criteria": "method_execution_contract_frame covers every reported paper method",
        }

    def method_runner_readiness_frame(
        self,
        *,
        nominal_alpha: float = 0.05,
        tolerance: float = 0.025,
        paper_replications: int = 500,
    ) -> pd.DataFrame:
        """Return method-level implementation readiness for paper rerun runners."""

        execution = self.method_execution_contract_frame(
            nominal_alpha=nominal_alpha,
            tolerance=tolerance,
            paper_replications=paper_replications,
        )
        support = (
            self.method_support_gate_frame(
                nominal_alpha=nominal_alpha,
                tolerance=tolerance,
            )
            .set_index("method")
            .to_dict("index")
        )
        rows: list[dict[str, Any]] = []
        for contract in execution.to_dict("records"):
            method = str(contract["method"])
            spec = _monte_carlo_method_runner_readiness_specs()[method]
            support_gate = support[method]
            runner_ready = bool(contract["python_executable"])
            plan_runner_ready = runner_ready
            blocking_reason = None if runner_ready else f"{method.lower()}_public_runner_not_implemented"
            rows.append(
                {
                    "method": method,
                    "paper_order_index": int(contract["paper_order_index"]),
                    "paper_role": contract["paper_role"],
                    "paper_reported_rows": int(contract["paper_reported_rows"]),
                    "python_executable_reported_rows": int(
                        contract["python_executable_reported_rows"]
                    ),
                    "python_blocked_reported_rows": (
                        0 if runner_ready else int(contract["paper_reported_rows"])
                    ),
                    "runner_family": spec["runner_family"],
                    "runner_entrypoint": spec["runner_entrypoint"],
                    "sharp_null_method": spec["sharp_null_method"],
                    "current_public_entrypoint": spec["current_public_entrypoint"],
                    "current_plan_runner": spec["current_plan_runner"],
                    "current_entrypoint_accepts_method": runner_ready,
                    "current_plan_runner_accepts_method": plan_runner_ready,
                    "requires_discretized_outcome_runner": bool(
                        contract["uses_discretized_outcome"]
                    ),
                    "requires_original_outcome_runner": not bool(
                        contract["uses_discretized_outcome"]
                    ),
                    "requires_bootstrap_runner": bool(contract["bootstrap_required"]),
                    "requires_individual_bootstrap_runner": bool(
                        "individual" in tuple(contract["bootstrap_unit_modes"])
                    ),
                    "requires_cluster_bootstrap_runner": bool(
                        "cluster" in tuple(contract["bootstrap_unit_modes"])
                    ),
                    "requires_nonbinary_mediator_runner": bool(
                        contract["supports_nonbinary_mediator"]
                    ),
                    "requires_binary_mediator_only": not bool(
                        contract["supports_nonbinary_mediator"]
                    ),
                    "requires_fsst_lambda_runner": method in {"FSSTdd", "FSSTndd"},
                    "runner_ready": runner_ready,
                    "blocking_reason": blocking_reason,
                    "method_support_gate_verdict": support_gate["method_gate_verdict"],
                    "next_action": (
                        "run_full_method_matrix"
                        if runner_ready
                        else spec["next_action"]
                    ),
                    "exit_criteria": (
                        "method_runner_readiness_summary.runner_gate_passes == True"
                    ),
                }
            )
        return _json_safe_export_frame(rows)

    def method_runner_readiness_summary(
        self,
        *,
        nominal_alpha: float = 0.05,
        tolerance: float = 0.025,
        paper_replications: int = 500,
    ) -> dict[str, Any]:
        """Return compact paper-runner readiness counts and plan-bridge diagnostics."""

        frame = self.method_runner_readiness_frame(
            nominal_alpha=nominal_alpha,
            tolerance=tolerance,
            paper_replications=paper_replications,
        )
        ready = frame["runner_ready"].map(_truthy_value)
        blocked = ~ready
        plan_runner_ready = frame["current_plan_runner_accepts_method"].map(_truthy_value)
        plan_runner_blocked = ~plan_runner_ready
        reason_counts: dict[str, int] = {}
        for row in frame.loc[blocked].to_dict("records"):
            reason = str(row["blocking_reason"])
            reason_counts[reason] = reason_counts.get(reason, 0) + int(
                row["paper_reported_rows"]
            )
        gate_passes = not bool(blocked.any())
        return {
            "paper_methods": int(frame.shape[0]),
            "runner_gate_passes": gate_passes,
            "runner_gate_verdict": "pass" if gate_passes else "blocked",
            "runner_ready_methods": int(ready.sum()),
            "runner_blocked_methods": int(blocked.sum()),
            "paper_reported_method_rows": int(frame["paper_reported_rows"].sum()),
            "runner_ready_reported_rows": int(
                frame.loc[ready, "paper_reported_rows"].sum()
            ),
            "runner_blocked_reported_rows": int(
                frame.loc[blocked, "paper_reported_rows"].sum()
            ),
            "current_plan_runner_ready_methods": int(plan_runner_ready.sum()),
            "current_plan_runner_blocked_methods": int(plan_runner_blocked.sum()),
            "current_plan_runner_ready_reported_rows": int(
                frame.loc[plan_runner_ready, "paper_reported_rows"].sum()
            ),
            "current_plan_runner_blocked_reported_rows": int(
                frame.loc[plan_runner_blocked, "paper_reported_rows"].sum()
            ),
            "bootstrap_runner_methods": int(
                frame["requires_bootstrap_runner"].map(_truthy_value).sum()
            ),
            "individual_bootstrap_runner_methods": int(
                frame["requires_individual_bootstrap_runner"].map(_truthy_value).sum()
            ),
            "cluster_bootstrap_runner_methods": int(
                frame["requires_cluster_bootstrap_runner"].map(_truthy_value).sum()
            ),
            "nonbinary_mediator_runner_methods": int(
                frame["requires_nonbinary_mediator_runner"].map(_truthy_value).sum()
            ),
            "original_outcome_runner_methods": int(
                frame["requires_original_outcome_runner"].map(_truthy_value).sum()
            ),
            "fsst_lambda_runner_methods": int(
                frame["requires_fsst_lambda_runner"].map(_truthy_value).sum()
            ),
            "runner_ready_method_names": tuple(
                frame.loc[ready, "method"].astype(str).tolist()
            ),
            "runner_blocked_method_names": tuple(
                frame.loc[blocked, "method"].astype(str).tolist()
            ),
            "current_plan_runner_ready_method_names": tuple(
                frame.loc[plan_runner_ready, "method"].astype(str).tolist()
            ),
            "current_plan_runner_blocked_method_names": tuple(
                frame.loc[plan_runner_blocked, "method"].astype(str).tolist()
            ),
            "blocking_reason_counts": reason_counts,
            "next_action": (
                "run_full_method_matrix"
                if gate_passes
                else "implement_paper_method_public_runners"
            ),
            "exit_criteria": "method_runner_readiness_summary.runner_gate_passes == True",
        }

    def method_support_gate_frame(
        self,
        *,
        nominal_alpha: float = 0.05,
        tolerance: float = 0.025,
    ) -> pd.DataFrame:
        """Return method-level Python support blockers for the paper Monte Carlo surface."""

        guidance_frame = self.method_guidance_frame(
            nominal_alpha=nominal_alpha,
            tolerance=tolerance,
        )
        rows: list[dict[str, Any]] = []
        for guidance in guidance_frame.to_dict("records"):
            method = str(guidance["method"])
            acceptance_summary = self.method_acceptance_summary(
                method=method,
                nominal_alpha=nominal_alpha,
                tolerance=tolerance,
            )
            support_gate = _python_method_support_gate(
                method=method,
                result_rows=self.result_rows,
                acceptance_summary=acceptance_summary,
            )
            rows.append(
                {
                    "method": method,
                    "paper_order_index": guidance["paper_order_index"],
                    "paper_role": guidance["paper_role"],
                    "paper_recommendation": guidance["paper_recommendation"],
                    "python_execution_status": guidance["python_execution_status"],
                    "paper_default": guidance["paper_default"],
                    "paper_contract_only": not bool(guidance["python_executable"]),
                    "uses_discretized_outcome": guidance["uses_discretized_outcome"],
                    "needs_size_caution": guidance["needs_size_caution"],
                    "paper_reported_rows": acceptance_summary["supported_result_rows"],
                    "paper_unreported_rows": acceptance_summary["unsupported_result_rows"],
                    "size_rows": acceptance_summary["size_rows"],
                    "power_rows": acceptance_summary["power_rows"],
                    "size_distortion_rows": acceptance_summary["size_distortion_rows"],
                    "cell_count_policy_rows": acceptance_summary["cell_count_policy_rows"],
                    "cell_count_size_risk_rows": acceptance_summary["cell_count_size_risk_rows"],
                    "attention_size_rows": acceptance_summary["attention_size_rows"],
                    **support_gate,
                }
            )
        return _json_safe_export_frame(rows)

    def method_support_gate_summary(
        self,
        *,
        nominal_alpha: float = 0.05,
        tolerance: float = 0.025,
    ) -> dict[str, Any]:
        """Return compact method-support blockers for release and milestone gates."""

        frame = self.method_support_gate_frame(
            nominal_alpha=nominal_alpha,
            tolerance=tolerance,
        )
        blocked = frame["method_gate_passes"].map(lambda value: not _truthy_value(value))
        partial = blocked & frame["python_execution_scope"].str.contains(
            "empirical_mixture_cs_rows",
            regex=False,
        )
        paper_contract_only = frame["paper_contract_only"].map(_truthy_value)
        reason_counts: dict[str, int] = {}
        for counts in frame["python_blocking_reason_counts"]:
            for reason, count in dict(counts).items():
                reason_counts[reason] = reason_counts.get(reason, 0) + int(count)
        gate_passes = not bool(blocked.any())
        return {
            "paper_methods": int(frame.shape[0]),
            "method_gate_passes": gate_passes,
            "method_gate_verdict": "pass" if gate_passes else "blocked",
            "methods_with_full_python_support": int((~blocked).sum()),
            "methods_with_partial_python_support": int(partial.sum()),
            "paper_contract_only_methods": int(paper_contract_only.sum()),
            "blocked_methods": int(blocked.sum()),
            "paper_reported_rows": int(frame["paper_reported_rows"].sum()),
            "python_executable_reported_rows": int(frame["python_executable_reported_rows"].sum()),
            "python_blocked_reported_rows": int(frame["python_blocked_reported_rows"].sum()),
            "size_distortion_rows": int(frame["size_distortion_rows"].sum()),
            "attention_size_rows": int(frame["attention_size_rows"].sum()),
            "blocking_reason_counts": reason_counts,
            "blocked_method_names": tuple(frame.loc[blocked, "method"].astype(str).tolist()),
            "partial_python_support_method_names": tuple(
                frame.loc[partial, "method"].astype(str).tolist()
            ),
            "paper_contract_only_method_names": tuple(
                frame.loc[paper_contract_only, "method"].astype(str).tolist()
            ),
            "next_action": _method_support_gate_next_action(reason_counts),
            "exit_criteria": "method_support_gate_summary.method_gate_passes == True",
        }

    def method_support_blocker_frame(
        self,
        *,
        nominal_alpha: float = 0.05,
        tolerance: float = 0.025,
        paper_replications: int = 500,
    ) -> pd.DataFrame:
        """Return row-level Python support blockers for every reported paper method."""

        methods = _ordered_monte_carlo_methods(
            {method for row in self.result_rows for method in row.rejection_rates}
        )
        execution_contract_by_method = (
            self.method_execution_contract_frame(
                nominal_alpha=nominal_alpha,
                tolerance=tolerance,
                paper_replications=paper_replications,
            )
            .set_index("method")
            .to_dict("index")
        )
        method_gate_by_method = (
            self.method_support_gate_frame(
                nominal_alpha=nominal_alpha,
                tolerance=tolerance,
            )
            .set_index("method")
            .to_dict("index")
        )
        rows: list[dict[str, Any]] = []
        for method_index, method in enumerate(methods):
            method_gate = method_gate_by_method[method]
            execution_contract = execution_contract_by_method[method]
            support_status = (
                "executable"
                if method_gate["method_gate_verdict"] == "pass"
                else "rescoped"
                if method_gate["method_gate_verdict"] == "rescoped"
                else "blocked"
            )
            for paper_row_index, result_row in enumerate(self.result_rows):
                if method not in result_row.rejection_rates:
                    continue
                reason = (
                    _python_method_support_blocking_reason(method=method, row=result_row)
                    if support_status == "blocked"
                    else None
                )
                rows.append(
                    {
                        **_paper_result_row_identity(result_row),
                        "method": method,
                        "method_order_index": method_index,
                        "paper_row_index": paper_row_index,
                        "paper_role": method_gate["paper_role"],
                        "python_execution_scope": method_gate["python_execution_scope"],
                        "paper_contract_only": bool(method_gate["paper_contract_only"]),
                        "paper_replications": execution_contract["paper_replications"],
                        "nominal_alpha": execution_contract["nominal_alpha"],
                        "uses_discretized_outcome": execution_contract[
                            "uses_discretized_outcome"
                        ],
                        "supports_binary_mediator": execution_contract[
                            "supports_binary_mediator"
                        ],
                        "supports_nonbinary_mediator": execution_contract[
                            "supports_nonbinary_mediator"
                        ],
                        "outcome_contract": execution_contract["outcome_contract"],
                        "variance_estimator": execution_contract["variance_estimator"],
                        "variance_contract": execution_contract["variance_contract"],
                        "bootstrap_required": execution_contract["bootstrap_required"],
                        "bootstrap_unit_modes": execution_contract["bootstrap_unit_modes"],
                        "bootstrap_unit_contract": execution_contract[
                            "bootstrap_unit_contract"
                        ],
                        "tuning_contract": execution_contract["tuning_contract"],
                        "python_executable": support_status == "executable",
                        "support_status": support_status,
                        "blocked_reason": reason,
                        "resolution_action": (
                            _method_support_resolution_action(reason)
                            if support_status == "blocked"
                            else "run_full_method_matrix"
                            if support_status == "executable"
                            else "maintain_non_release_scope"
                        ),
                        "target_rejection_rate": result_row.rejection_rates[method],
                        "acceptance_role": (
                            "size_control" if result_row.is_null_size_row else "power"
                        ),
                        "source_contract": "paper_monte_carlo_rejection_rate_row",
                        "exit_criteria": "method_support_gate_summary.method_gate_passes == True",
                    }
                )
        return _json_safe_export_frame(rows)

    def method_support_blocker_summary(
        self,
        *,
        nominal_alpha: float = 0.05,
        tolerance: float = 0.025,
        paper_replications: int = 500,
    ) -> dict[str, Any]:
        """Return compact row-level Python support blocker counts."""

        frame = self.method_support_blocker_frame(
            nominal_alpha=nominal_alpha,
            tolerance=tolerance,
            paper_replications=paper_replications,
        )
        if frame.empty:
            return {
                "paper_reported_method_rows": 0,
                "python_executable_method_rows": 0,
                "python_blocked_method_rows": 0,
                "method_support_blocker_gate_passes": True,
                "method_support_blocker_gate_verdict": "pass",
                "blocking_reason_counts": {},
                "blocked_method_names": (),
                "next_action": "run_full_method_matrix",
                "exit_criteria": "method_support_gate_summary.method_gate_passes == True",
            }

        executable = frame["support_status"].eq("executable")
        blocked_frame = frame.loc[frame["support_status"].eq("blocked")]
        reason_counts = _value_counts_tuple_or_str(blocked_frame["blocked_reason"])
        gate_passes = blocked_frame.empty
        return {
            "paper_reported_method_rows": int(frame.shape[0]),
            "python_executable_method_rows": int(executable.sum()),
            "python_blocked_method_rows": int(frame["support_status"].eq("blocked").sum()),
            "method_support_blocker_gate_passes": gate_passes,
            "method_support_blocker_gate_verdict": "pass" if gate_passes else "blocked",
            "blocking_reason_counts": reason_counts,
            "blocked_method_names": tuple(
                blocked_frame["method"].drop_duplicates().astype(str).tolist()
            ),
            "next_action": _method_support_gate_next_action(reason_counts),
            "exit_criteria": "method_support_gate_summary.method_gate_passes == True",
        }

    def empirical_mixture_benchmark_suite_plans(
        self,
        *,
        replications: int = 500,
        method: str | None = None,
        mediator: str | None = None,
        design: str | None = None,
        table: str | None = None,
        clusters: tuple[int | None, ...] | None = None,
        bins: tuple[int | None, ...] | None = None,
        t_values: tuple[float, ...] | None = None,
    ) -> tuple[MonteCarloBenchmarkPlan, ...]:
        """Return one empirical-mixture benchmark plan per reported paper method."""

        methods = {name for row in self.result_rows for name in row.rejection_rates}
        if method is not None:
            if method not in methods:
                raise ValueError(f"Unknown paper Monte Carlo method: {method!r}.")
            methods = {method}
        ordered_methods = _ordered_monte_carlo_methods(methods)
        return tuple(
            self.empirical_mixture_benchmark_plan(
                method=method_name,
                replications=replications,
                mediator=mediator,
                design=design,
                table=table,
                clusters=clusters,
                bins=bins,
                t_values=t_values,
            )
            for method_name in ordered_methods
        )

    def empirical_mixture_benchmark_suite_readiness_report(
        self,
        data_sources: dict[str, "BinaryEmpiricalMixtureBenchmarkDataSource"],
        *,
        seed: int,
        replications: int = 500,
        mediator: str | None = None,
        design: str | None = None,
        table: str | None = None,
        clusters: tuple[int | None, ...] | None = None,
        bins: tuple[int | None, ...] | None = None,
        t_values: tuple[float, ...] | None = None,
        alpha: float = 0.05,
        absolute_tolerance: float = 0.025,
        z_tolerance: float = 2.0,
        cell_count_absolute_tolerance: float | None = None,
        source_mixture_absolute_tolerance: float | None = None,
    ) -> MonteCarloBenchmarkSuiteReadinessReport:
        """Return one readiness report per paper method."""

        seed_rng = np.random.default_rng(seed)
        reports: list[MonteCarloBenchmarkPlanReadinessReport] = []
        for plan in self.empirical_mixture_benchmark_suite_plans(
            replications=replications,
            mediator=mediator,
            design=design,
            table=table,
            clusters=clusters,
            bins=bins,
            t_values=t_values,
        ):
            method_seed = int(seed_rng.integers(low=0, high=np.iinfo(np.uint32).max, dtype=np.uint32))
            reports.append(
                plan.rerun_readiness_report(
                    data_sources,
                    seed=method_seed,
                    alpha=alpha,
                    absolute_tolerance=absolute_tolerance,
                    z_tolerance=z_tolerance,
                    cell_count_absolute_tolerance=cell_count_absolute_tolerance,
                    source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
                )
            )
        return MonteCarloBenchmarkSuiteReadinessReport(reports=tuple(reports))

    def empirical_mixture_benchmark_suite_run_result(
        self,
        data_sources: dict[str, "BinaryEmpiricalMixtureBenchmarkDataSource"],
        *,
        seed: int,
        replications: int = 500,
        bootstrap_replications: int = 500,
        mediator: str | None = None,
        design: str | None = None,
        table: str | None = None,
        clusters: tuple[int | None, ...] | None = None,
        bins: tuple[int | None, ...] | None = None,
        t_values: tuple[float, ...] | None = None,
        alpha: float = 0.05,
        absolute_tolerance: float = 0.025,
        z_tolerance: float = 2.0,
        cell_count_absolute_tolerance: float | None = None,
        source_mixture_absolute_tolerance: float | None = None,
        ) -> MonteCarloBenchmarkSuiteRunResult:
        """Execute one preflighted rerun result per paper method."""

        seed_rng = np.random.default_rng(seed)
        results: list[MonteCarloBenchmarkPlanRunResult] = []
        for plan in self.empirical_mixture_benchmark_suite_plans(
            replications=replications,
            mediator=mediator,
            design=design,
            table=table,
            clusters=clusters,
            bins=bins,
            t_values=t_values,
        ):
            method_seed = int(seed_rng.integers(low=0, high=np.iinfo(np.uint32).max, dtype=np.uint32))
            manifest = plan.rerun_manifest(
                data_sources,
                seed=method_seed,
                alpha=alpha,
                bootstrap_replications=bootstrap_replications,
                absolute_tolerance=absolute_tolerance,
                z_tolerance=z_tolerance,
                cell_count_absolute_tolerance=cell_count_absolute_tolerance,
                source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
            )
            results.append(run_empirical_mixture_benchmark_manifest(self, manifest=manifest))
        return MonteCarloBenchmarkSuiteRunResult(results=tuple(results))

    def empirical_mixture_benchmark_suite_focused_run_result(
        self,
        data_sources: dict[str, "BinaryEmpiricalMixtureBenchmarkDataSource"],
        *,
        seed: int,
        paper_replications: int = 500,
        focused_replications: int = 1,
        bootstrap_replications: int = 500,
        mediator: str | None = None,
        table: str | None = None,
        design: str | None = None,
        clusters: tuple[int | None, ...] | None = None,
        bins: tuple[int | None, ...] | None = None,
        t_values: tuple[float, ...] | None = None,
        max_cells: int | None = 1,
        alpha: float = 0.05,
        absolute_tolerance: float = 0.025,
        z_tolerance: float = 2.0,
        cell_count_absolute_tolerance: float | None = None,
        source_mixture_absolute_tolerance: float | None = None,
    ) -> MonteCarloBenchmarkSuiteRunResult:
        """Execute deterministic focused slices while preserving full-plan coverage blockers."""

        seed_rng = np.random.default_rng(seed)
        results: list[MonteCarloBenchmarkPlanRunResult] = []
        for plan in self.empirical_mixture_benchmark_suite_plans(
            replications=paper_replications,
            mediator=mediator,
            design=design,
            table=table,
            clusters=clusters,
            bins=bins,
            t_values=t_values,
        ):
            if not plan.executable_cells:
                continue
            method_seed = int(seed_rng.integers(low=0, high=np.iinfo(np.uint32).max, dtype=np.uint32))
            manifest = plan.rerun_manifest(
                data_sources,
                seed=method_seed,
                alpha=alpha,
                bootstrap_replications=bootstrap_replications,
                absolute_tolerance=absolute_tolerance,
                z_tolerance=z_tolerance,
                cell_count_absolute_tolerance=cell_count_absolute_tolerance,
                source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
            )
            focused_manifest = manifest.focused_slice(
                table=table,
                design=design,
                clusters=clusters,
                bins=bins,
                t_values=t_values,
                max_cells=max_cells,
                replications=focused_replications,
            )
            results.append(run_empirical_mixture_benchmark_manifest(self, manifest=focused_manifest))
        if not results:
            raise ValueError("suite focused run selected no scheduled Monte Carlo cells.")
        return MonteCarloBenchmarkSuiteRunResult(results=tuple(results))

    def empirical_mixture_benchmark_suite_cell_index_run_result(
        self,
        data_sources: dict[str, "BinaryEmpiricalMixtureBenchmarkDataSource"],
        *,
        seed: int,
        cell_start: int = 0,
        cell_stop: int | None = None,
        paper_replications: int = 500,
        slice_replications: int | None = None,
        bootstrap_replications: int = 500,
        method: str | None = None,
        mediator: str | None = None,
        design: str | None = None,
        table: str | None = None,
        clusters: tuple[int | None, ...] | None = None,
        bins: tuple[int | None, ...] | None = None,
        t_values: tuple[float, ...] | None = None,
        alpha: float = 0.05,
        absolute_tolerance: float = 0.025,
        z_tolerance: float = 2.0,
        cell_count_absolute_tolerance: float | None = None,
        source_mixture_absolute_tolerance: float | None = None,
    ) -> MonteCarloBenchmarkSuiteRunResult:
        """Execute contiguous per-method row slices for resumable suite reruns."""

        if cell_start < 0:
            raise ValueError("cell_start must be non-negative.")
        if cell_stop is not None and cell_stop <= cell_start:
            raise ValueError("cell_stop must be greater than cell_start when provided.")
        if slice_replications is not None and slice_replications <= 0:
            raise ValueError("slice_replications must be positive when provided.")

        seed_rng = np.random.default_rng(seed)
        results: list[MonteCarloBenchmarkPlanRunResult] = []
        for plan in self.empirical_mixture_benchmark_suite_plans(
            replications=paper_replications,
            mediator=mediator,
            design=design,
            table=table,
            clusters=clusters,
            bins=bins,
            t_values=t_values,
        ):
            method_seed = int(seed_rng.integers(low=0, high=np.iinfo(np.uint32).max, dtype=np.uint32))
            if method is not None and plan.method != method:
                continue
            if cell_start >= len(plan.executable_cells):
                continue
            indexed_manifest = _cell_index_slice_manifest_from_plan(
                plan,
                data_sources,
                seed=method_seed,
                cell_start=cell_start,
                cell_stop=cell_stop,
                slice_replications=slice_replications,
                alpha=alpha,
                bootstrap_replications=bootstrap_replications,
                absolute_tolerance=absolute_tolerance,
                z_tolerance=z_tolerance,
                cell_count_absolute_tolerance=cell_count_absolute_tolerance,
                source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
            )
            results.append(run_empirical_mixture_benchmark_manifest(self, manifest=indexed_manifest))

        if not results:
            raise ValueError("suite cell-index slice selected no scheduled Monte Carlo cells.")
        return MonteCarloBenchmarkSuiteRunResult(results=tuple(results))

    def empirical_mixture_benchmark_suite_cell_index_progress_frame(
        self,
        run_result: MonteCarloBenchmarkSuiteRunResult,
        data_sources: dict[str, "BinaryEmpiricalMixtureBenchmarkDataSource"],
        *,
        seed: int,
        cell_chunk_size: int = 1,
        paper_replications: int = 500,
        slice_replications: int | None = None,
        bootstrap_replications: int = 500,
        mediator: str | None = None,
        design: str | None = None,
        table: str | None = None,
        clusters: tuple[int | None, ...] | None = None,
        bins: tuple[int | None, ...] | None = None,
        t_values: tuple[float, ...] | None = None,
        alpha: float = 0.05,
        absolute_tolerance: float = 0.025,
        z_tolerance: float = 2.0,
        cell_count_absolute_tolerance: float | None = None,
        source_mixture_absolute_tolerance: float | None = None,
    ) -> pd.DataFrame:
        """Compare executed suite chunks to the resumable full-paper schedule."""

        if cell_chunk_size <= 0:
            raise ValueError("cell_chunk_size must be positive.")
        if slice_replications is not None and slice_replications <= 0:
            raise ValueError("slice_replications must be positive when provided.")
        if bootstrap_replications <= 0:
            raise ValueError("bootstrap_replications must be positive.")

        seed_rng = np.random.default_rng(seed)
        method_manifests: list[tuple[str, MonteCarloBenchmarkPlanRerunManifest]] = []
        for plan in self.empirical_mixture_benchmark_suite_plans(
            replications=paper_replications,
            mediator=mediator,
            design=design,
            table=table,
            clusters=clusters,
            bins=bins,
            t_values=t_values,
        ):
            method_seed = int(seed_rng.integers(low=0, high=np.iinfo(np.uint32).max, dtype=np.uint32))
            manifest = plan.rerun_manifest(
                data_sources,
                seed=method_seed,
                alpha=alpha,
                bootstrap_replications=bootstrap_replications,
                absolute_tolerance=absolute_tolerance,
                z_tolerance=z_tolerance,
                cell_count_absolute_tolerance=cell_count_absolute_tolerance,
                source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
            )
            method_manifests.append((plan.method, manifest))

        if not method_manifests:
            return pd.DataFrame()

        expected_methods = tuple(method for method, _ in method_manifests)
        expected_keys_by_method = _suite_schedule_keys_by_method(method_manifests)
        observed_methods = tuple(run_result.method_names)
        if len(set(observed_methods)) != len(observed_methods):
            raise ValueError(
                "suite cell-index progress requires each run_result method to appear at most once."
            )
        unexpected_methods = tuple(
            method for method in observed_methods if method not in expected_methods
        )
        if unexpected_methods:
            raise ValueError(
                "suite cell-index progress requires run_result methods to belong "
                f"to the paper suite order: unexpected {unexpected_methods}."
            )
        ordered_observed_methods = tuple(
            method for method in expected_methods if method in observed_methods
        )
        if observed_methods != ordered_observed_methods:
            raise ValueError(
                "suite cell-index progress requires run_result methods to follow "
                f"the paper suite order: expected ordered subset {ordered_observed_methods}, "
                f"got {observed_methods}."
            )

        executed_keys_by_method: dict[str, set[tuple[Any, ...]]] = {
            method: set() for method in expected_methods
        }
        executed_draws_by_method_key: dict[str, dict[tuple[Any, ...], int]] = {
            method: {} for method in expected_methods
        }
        executed_bootstrap_draws_by_method_key: dict[str, dict[tuple[Any, ...], int]] = {
            method: {} for method in expected_methods
        }
        for result in run_result.results:
            method = result.plan.method
            if not result.scheduled_cells:
                raise ValueError(
                    "suite cell-index progress requires run_result chunks to expose scheduled_cells."
                )
            for scheduled_cell in result.scheduled_cells:
                key = _paper_result_cell_key(scheduled_cell.cell)
                if key not in expected_keys_by_method[method]:
                    raise ValueError(
                        "suite cell-index progress requires run_result cells to belong "
                        "to the current suite schedule."
                    )
                if key in executed_keys_by_method[method]:
                    raise ValueError(
                        "suite cell-index progress requires unique scheduled cells across chunks."
                    )
                executed_keys_by_method[method].add(key)
                executed_draws_by_method_key[method][key] = int(scheduled_cell.planned_draws)
                executed_bootstrap_draws_by_method_key[method][key] = _scheduled_cell_bootstrap_draws(
                    scheduled_cell
                )

        schedule = self.empirical_mixture_benchmark_suite_cell_index_schedule_frame(
            data_sources,
            seed=seed,
            cell_chunk_size=cell_chunk_size,
            paper_replications=paper_replications,
            slice_replications=slice_replications,
            bootstrap_replications=bootstrap_replications,
            mediator=mediator,
            design=design,
            table=table,
            clusters=clusters,
            bins=bins,
            t_values=t_values,
            alpha=alpha,
            absolute_tolerance=absolute_tolerance,
            z_tolerance=z_tolerance,
            cell_count_absolute_tolerance=cell_count_absolute_tolerance,
            source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
        )
        if schedule.empty:
            return schedule

        rows: list[dict[str, Any]] = []
        cumulative_executed_rows = 0
        cumulative_executed_draws = 0
        suite_result_rows = int(schedule["paper_result_rows"].iloc[0])
        suite_scheduled_draws = int(schedule["suite_planned_draws"].iloc[0])
        for schedule_row in schedule.to_dict("records"):
            cell_start = int(schedule_row["cell_start"])
            cell_stop = int(schedule_row["cell_stop"])
            scheduled_result_rows = int(schedule_row["covered_result_rows"])
            scheduled_draws = int(schedule_row["planned_draws"])
            executed_result_rows = 0
            executed_draws = 0
            scheduled_bootstrap_draws = 0
            executed_bootstrap_draws = 0
            executed_methods: list[str] = []

            for method, manifest in method_manifests:
                chunk_cells = tuple(manifest.cells[cell_start:cell_stop])
                chunk_keys = tuple(_paper_result_cell_key(scheduled_cell.cell) for scheduled_cell in chunk_cells)
                if not chunk_keys:
                    continue
                scheduled_bootstrap_draws += sum(
                    _scheduled_cell_bootstrap_draws(scheduled_cell)
                    for scheduled_cell in chunk_cells
                )
                method_executed_rows = sum(
                    1 for key in chunk_keys if key in executed_keys_by_method[method]
                )
                method_executed_draws = sum(
                    executed_draws_by_method_key[method].get(key, 0)
                    for key in chunk_keys
                )
                method_executed_bootstrap_draws = sum(
                    executed_bootstrap_draws_by_method_key[method].get(key, 0)
                    for key in chunk_keys
                )
                if method_executed_rows:
                    executed_methods.append(method)
                executed_result_rows += method_executed_rows
                executed_draws += method_executed_draws
                executed_bootstrap_draws += method_executed_bootstrap_draws

            row_complete = executed_result_rows == scheduled_result_rows
            draw_complete = executed_draws == scheduled_draws
            bootstrap_complete = executed_bootstrap_draws == scheduled_bootstrap_draws
            chunk_complete = row_complete and draw_complete and bootstrap_complete
            if executed_result_rows == 0:
                chunk_status = "pending"
            elif chunk_complete:
                chunk_status = "complete"
            elif row_complete and draw_complete:
                chunk_status = "bootstrap_incomplete"
            elif row_complete:
                chunk_status = "row_complete_only"
            else:
                chunk_status = "partial"

            cumulative_executed_rows += executed_result_rows
            cumulative_executed_draws += executed_draws
            rows.append(
                {
                    **schedule_row,
                    "scheduled_result_rows": scheduled_result_rows,
                    "scheduled_draws": scheduled_draws,
                    "executed_result_rows": executed_result_rows,
                    "executed_draws": executed_draws,
                    "scheduled_bootstrap_draws": scheduled_bootstrap_draws,
                    "executed_bootstrap_draws": executed_bootstrap_draws,
                    "executed_method_count": len(executed_methods),
                    "executed_methods": tuple(executed_methods),
                    "row_completion_fraction": (
                        1.0
                        if scheduled_result_rows == 0
                        else executed_result_rows / scheduled_result_rows
                    ),
                    "draw_completion_fraction": (
                        1.0
                        if scheduled_draws == 0
                        else executed_draws / scheduled_draws
                    ),
                    "bootstrap_completion_fraction": (
                        1.0
                        if scheduled_bootstrap_draws == 0
                        else executed_bootstrap_draws / scheduled_bootstrap_draws
                    ),
                    "row_complete": row_complete,
                    "draw_complete": draw_complete,
                    "bootstrap_complete": bootstrap_complete,
                    "chunk_complete": chunk_complete,
                    "chunk_status": chunk_status,
                    "chunk_shortfall_rows": max(scheduled_result_rows - executed_result_rows, 0),
                    "chunk_shortfall_draws": max(scheduled_draws - executed_draws, 0),
                    "chunk_shortfall_bootstrap_draws": max(
                        scheduled_bootstrap_draws - executed_bootstrap_draws,
                        0,
                    ),
                    "cumulative_executed_result_rows": cumulative_executed_rows,
                    "cumulative_executed_draws": cumulative_executed_draws,
                    "remaining_result_rows_after_progress": max(
                        suite_result_rows - cumulative_executed_rows,
                        0,
                    ),
                    "remaining_draws_after_progress": max(
                        suite_scheduled_draws - cumulative_executed_draws,
                        0,
                    ),
                }
            )
        return _json_safe_export_frame(rows)

    def empirical_mixture_benchmark_suite_cell_index_progress_summary(
        self,
        run_result: MonteCarloBenchmarkSuiteRunResult,
        data_sources: dict[str, "BinaryEmpiricalMixtureBenchmarkDataSource"],
        *,
        seed: int,
        cell_chunk_size: int = 1,
        paper_replications: int = 500,
        slice_replications: int | None = None,
        bootstrap_replications: int = 500,
        mediator: str | None = None,
        design: str | None = None,
        table: str | None = None,
        clusters: tuple[int | None, ...] | None = None,
        bins: tuple[int | None, ...] | None = None,
        t_values: tuple[float, ...] | None = None,
        alpha: float = 0.05,
        absolute_tolerance: float = 0.025,
        z_tolerance: float = 2.0,
        cell_count_absolute_tolerance: float | None = None,
        source_mixture_absolute_tolerance: float | None = None,
    ) -> dict[str, Any]:
        """Summarize executed suite chunk progress against the cell-index schedule."""

        frame = self.empirical_mixture_benchmark_suite_cell_index_progress_frame(
            run_result,
            data_sources,
            seed=seed,
            cell_chunk_size=cell_chunk_size,
            paper_replications=paper_replications,
            slice_replications=slice_replications,
            bootstrap_replications=bootstrap_replications,
            mediator=mediator,
            design=design,
            table=table,
            clusters=clusters,
            bins=bins,
            t_values=t_values,
            alpha=alpha,
            absolute_tolerance=absolute_tolerance,
            z_tolerance=z_tolerance,
            cell_count_absolute_tolerance=cell_count_absolute_tolerance,
            source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
        )
        return _suite_cell_index_progress_summary_from_frame(
            frame,
            seed=seed,
            cell_chunk_size=cell_chunk_size,
            paper_replications=paper_replications,
            slice_replications=slice_replications,
        )

    def empirical_mixture_benchmark_suite_chunk_progress_frame(
        self,
        run_result_chunks: tuple[MonteCarloBenchmarkSuiteRunResult, ...],
        data_sources: dict[str, "BinaryEmpiricalMixtureBenchmarkDataSource"],
        *,
        seed: int,
        cell_chunk_size: int = 1,
        paper_replications: int = 500,
        slice_replications: int | None = None,
        bootstrap_replications: int = 500,
        mediator: str | None = None,
        design: str | None = None,
        table: str | None = None,
        clusters: tuple[int | None, ...] | None = None,
        bins: tuple[int | None, ...] | None = None,
        t_values: tuple[float, ...] | None = None,
        alpha: float = 0.05,
        absolute_tolerance: float = 0.025,
        z_tolerance: float = 2.0,
        cell_count_absolute_tolerance: float | None = None,
        source_mixture_absolute_tolerance: float | None = None,
    ) -> pd.DataFrame:
        """Compare a tuple of executed suite chunks to the full-paper schedule."""

        combined_run_result = (
            MonteCarloBenchmarkSuiteRunResult(results=())
            if not run_result_chunks
            else MonteCarloBenchmarkSuiteRunResult.from_chunk_results(run_result_chunks)
        )
        return self.empirical_mixture_benchmark_suite_cell_index_progress_frame(
            combined_run_result,
            data_sources,
            seed=seed,
            cell_chunk_size=cell_chunk_size,
            paper_replications=paper_replications,
            slice_replications=slice_replications,
            bootstrap_replications=bootstrap_replications,
            mediator=mediator,
            design=design,
            table=table,
            clusters=clusters,
            bins=bins,
            t_values=t_values,
            alpha=alpha,
            absolute_tolerance=absolute_tolerance,
            z_tolerance=z_tolerance,
            cell_count_absolute_tolerance=cell_count_absolute_tolerance,
            source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
        )

    def empirical_mixture_benchmark_suite_chunk_progress_summary(
        self,
        run_result_chunks: tuple[MonteCarloBenchmarkSuiteRunResult, ...],
        data_sources: dict[str, "BinaryEmpiricalMixtureBenchmarkDataSource"],
        *,
        seed: int,
        cell_chunk_size: int = 1,
        paper_replications: int = 500,
        slice_replications: int | None = None,
        bootstrap_replications: int = 500,
        mediator: str | None = None,
        design: str | None = None,
        table: str | None = None,
        clusters: tuple[int | None, ...] | None = None,
        bins: tuple[int | None, ...] | None = None,
        t_values: tuple[float, ...] | None = None,
        alpha: float = 0.05,
        absolute_tolerance: float = 0.025,
        z_tolerance: float = 2.0,
        cell_count_absolute_tolerance: float | None = None,
        source_mixture_absolute_tolerance: float | None = None,
    ) -> dict[str, Any]:
        """Summarize a tuple of executed suite chunks against the full-paper schedule."""

        combined_run_result = (
            MonteCarloBenchmarkSuiteRunResult(results=())
            if not run_result_chunks
            else MonteCarloBenchmarkSuiteRunResult.from_chunk_results(run_result_chunks)
        )
        return self.empirical_mixture_benchmark_suite_cell_index_progress_summary(
            combined_run_result,
            data_sources,
            seed=seed,
            cell_chunk_size=cell_chunk_size,
            paper_replications=paper_replications,
            slice_replications=slice_replications,
            bootstrap_replications=bootstrap_replications,
            mediator=mediator,
            design=design,
            table=table,
            clusters=clusters,
            bins=bins,
            t_values=t_values,
            alpha=alpha,
            absolute_tolerance=absolute_tolerance,
            z_tolerance=z_tolerance,
            cell_count_absolute_tolerance=cell_count_absolute_tolerance,
            source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
        )

    def empirical_mixture_benchmark_suite_export_progress_frame(
        self,
        exported_run_results: Any,
        data_sources: dict[str, "BinaryEmpiricalMixtureBenchmarkDataSource"],
        *,
        seed: int,
        cell_chunk_size: int = 1,
        paper_replications: int = 500,
        slice_replications: int | None = None,
        bootstrap_replications: int = 500,
        mediator: str | None = None,
        design: str | None = None,
        table: str | None = None,
        clusters: tuple[int | None, ...] | None = None,
        bins: tuple[int | None, ...] | None = None,
        t_values: tuple[float, ...] | None = None,
        alpha: float = 0.05,
        absolute_tolerance: float = 0.025,
        z_tolerance: float = 2.0,
        cell_count_absolute_tolerance: float | None = None,
        source_mixture_absolute_tolerance: float | None = None,
    ) -> pd.DataFrame:
        """Compare exported suite-run rows to the full-paper chunk schedule."""

        exported_frame = _coerce_suite_run_result_export_frame(exported_run_results)
        seed_rng = np.random.default_rng(seed)
        method_manifests: list[tuple[str, MonteCarloBenchmarkPlanRerunManifest]] = []
        for plan in self.empirical_mixture_benchmark_suite_plans(
            replications=paper_replications,
            mediator=mediator,
            design=design,
            table=table,
            clusters=clusters,
            bins=bins,
            t_values=t_values,
        ):
            method_seed = int(seed_rng.integers(low=0, high=np.iinfo(np.uint32).max, dtype=np.uint32))
            manifest = plan.rerun_manifest(
                data_sources,
                seed=method_seed,
                alpha=alpha,
                bootstrap_replications=bootstrap_replications,
                absolute_tolerance=absolute_tolerance,
                z_tolerance=z_tolerance,
                cell_count_absolute_tolerance=cell_count_absolute_tolerance,
                source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
            )
            method_manifests.append((plan.method, manifest))

        if not method_manifests:
            return pd.DataFrame()

        expected_methods = tuple(method for method, _ in method_manifests)
        expected_keys_by_method = _suite_schedule_keys_by_method(method_manifests)
        expected_seeds_by_method_key = _suite_schedule_seeds_by_method_key(method_manifests)
        executed_keys_by_method, executed_draws_by_method_key, executed_bootstrap_draws_by_method_key = (
            _suite_executed_maps_from_export_frame(
                exported_frame,
                expected_methods,
                expected_keys_by_method=expected_keys_by_method,
                expected_seeds_by_method_key=expected_seeds_by_method_key,
            )
        )
        schedule = self.empirical_mixture_benchmark_suite_cell_index_schedule_frame(
            data_sources,
            seed=seed,
            cell_chunk_size=cell_chunk_size,
            paper_replications=paper_replications,
            slice_replications=slice_replications,
            bootstrap_replications=bootstrap_replications,
            mediator=mediator,
            design=design,
            table=table,
            clusters=clusters,
            bins=bins,
            t_values=t_values,
            alpha=alpha,
            absolute_tolerance=absolute_tolerance,
            z_tolerance=z_tolerance,
            cell_count_absolute_tolerance=cell_count_absolute_tolerance,
            source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
        )
        return _suite_cell_index_progress_frame_from_executed_maps(
            schedule,
            method_manifests=method_manifests,
            executed_keys_by_method=executed_keys_by_method,
            executed_draws_by_method_key=executed_draws_by_method_key,
            executed_bootstrap_draws_by_method_key=executed_bootstrap_draws_by_method_key,
        )

    def empirical_mixture_benchmark_suite_export_progress_summary(
        self,
        exported_run_results: Any,
        data_sources: dict[str, "BinaryEmpiricalMixtureBenchmarkDataSource"],
        *,
        seed: int,
        cell_chunk_size: int = 1,
        paper_replications: int = 500,
        slice_replications: int | None = None,
        bootstrap_replications: int = 500,
        mediator: str | None = None,
        design: str | None = None,
        table: str | None = None,
        clusters: tuple[int | None, ...] | None = None,
        bins: tuple[int | None, ...] | None = None,
        t_values: tuple[float, ...] | None = None,
        alpha: float = 0.05,
        absolute_tolerance: float = 0.025,
        z_tolerance: float = 2.0,
        cell_count_absolute_tolerance: float | None = None,
        source_mixture_absolute_tolerance: float | None = None,
    ) -> dict[str, Any]:
        """Summarize exported suite-run rows against the full-paper schedule."""

        frame = self.empirical_mixture_benchmark_suite_export_progress_frame(
            exported_run_results,
            data_sources,
            seed=seed,
            cell_chunk_size=cell_chunk_size,
            paper_replications=paper_replications,
            slice_replications=slice_replications,
            bootstrap_replications=bootstrap_replications,
            mediator=mediator,
            design=design,
            table=table,
            clusters=clusters,
            bins=bins,
            t_values=t_values,
            alpha=alpha,
            absolute_tolerance=absolute_tolerance,
            z_tolerance=z_tolerance,
            cell_count_absolute_tolerance=cell_count_absolute_tolerance,
            source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
        )
        return _suite_cell_index_progress_summary_from_frame(
            frame,
            seed=seed,
            cell_chunk_size=cell_chunk_size,
            paper_replications=paper_replications,
            slice_replications=slice_replications,
        )

    def empirical_mixture_benchmark_suite_export_acceptance_frame(
        self,
        exported_run_results: Any,
    ) -> pd.DataFrame:
        """Return saved suite evidence as a row-level paper acceptance frame."""

        exported_frame = _coerce_suite_run_result_export_frame(exported_run_results)
        return _suite_export_acceptance_frame(exported_frame)

    def empirical_mixture_benchmark_suite_export_unresolved_frame(
        self,
        exported_run_results: Any,
    ) -> pd.DataFrame:
        """Return unresolved paper rows recovered from saved suite evidence."""

        acceptance_frame = self.empirical_mixture_benchmark_suite_export_acceptance_frame(
            exported_run_results
        )
        return _paper_acceptance_unresolved_row_frame(
            acceptance_frame,
            source="exported_evidence",
            include_failed_executed_rows=True,
        )

    def empirical_mixture_benchmark_suite_export_acceptance_gate(
        self,
        exported_run_results: Any,
        data_sources: dict[str, "BinaryEmpiricalMixtureBenchmarkDataSource"],
        *,
        seed: int,
        cell_chunk_size: int = 1,
        paper_replications: int = 500,
        slice_replications: int | None = None,
        bootstrap_replications: int = 500,
        mediator: str | None = None,
        design: str | None = None,
        table: str | None = None,
        clusters: tuple[int | None, ...] | None = None,
        bins: tuple[int | None, ...] | None = None,
        t_values: tuple[float, ...] | None = None,
        alpha: float = 0.05,
        absolute_tolerance: float = 0.025,
        z_tolerance: float = 2.0,
        cell_count_absolute_tolerance: float | None = None,
        source_mixture_absolute_tolerance: float | None = None,
    ) -> dict[str, Any]:
        """Return a post-run release gate from saved suite evidence payloads."""

        acceptance_frame = self.empirical_mixture_benchmark_suite_export_acceptance_frame(
            exported_run_results
        )
        progress_summary = self.empirical_mixture_benchmark_suite_export_progress_summary(
            exported_run_results,
            data_sources,
            seed=seed,
            cell_chunk_size=cell_chunk_size,
            paper_replications=paper_replications,
            slice_replications=slice_replications,
            bootstrap_replications=bootstrap_replications,
            mediator=mediator,
            design=design,
            table=table,
            clusters=clusters,
            bins=bins,
            t_values=t_values,
            alpha=alpha,
            absolute_tolerance=absolute_tolerance,
            z_tolerance=z_tolerance,
            cell_count_absolute_tolerance=cell_count_absolute_tolerance,
            source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
        )
        return _paper_acceptance_export_gate_from_frame(
            acceptance_frame,
            progress_summary=progress_summary,
        )

    def empirical_mixture_benchmark_suite_export_completion_summary(
        self,
        exported_run_results: Any,
        data_sources: dict[str, "BinaryEmpiricalMixtureBenchmarkDataSource"],
        *,
        seed: int,
        owner: str = "Phase 11 full paper Monte Carlo acceptance run",
        rerun_command: str | None = None,
        cell_chunk_size: int = 1,
        paper_replications: int = 500,
        slice_replications: int | None = None,
        bootstrap_replications: int = 500,
        mediator: str | None = None,
        design: str | None = None,
        table: str | None = None,
        clusters: tuple[int | None, ...] | None = None,
        bins: tuple[int | None, ...] | None = None,
        t_values: tuple[float, ...] | None = None,
        alpha: float = 0.05,
        absolute_tolerance: float = 0.025,
        z_tolerance: float = 2.0,
        cell_count_absolute_tolerance: float | None = None,
        source_mixture_absolute_tolerance: float | None = None,
    ) -> dict[str, Any]:
        """Return milestone closeout status from saved suite evidence payloads."""

        gate = self.empirical_mixture_benchmark_suite_export_acceptance_gate(
            exported_run_results,
            data_sources,
            seed=seed,
            cell_chunk_size=cell_chunk_size,
            paper_replications=paper_replications,
            slice_replications=slice_replications,
            bootstrap_replications=bootstrap_replications,
            mediator=mediator,
            design=design,
            table=table,
            clusters=clusters,
            bins=bins,
            t_values=t_values,
            alpha=alpha,
            absolute_tolerance=absolute_tolerance,
            z_tolerance=z_tolerance,
            cell_count_absolute_tolerance=cell_count_absolute_tolerance,
            source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
        )
        return _paper_acceptance_gate_completion_summary(
            gate,
            owner=owner,
            rerun_command=rerun_command,
        )

    def empirical_mixture_benchmark_suite_cell_index_schedule_frame(
        self,
        data_sources: dict[str, "BinaryEmpiricalMixtureBenchmarkDataSource"],
        *,
        seed: int,
        cell_chunk_size: int = 1,
        paper_replications: int = 500,
        slice_replications: int | None = None,
        bootstrap_replications: int = 500,
        method: str | None = None,
        mediator: str | None = None,
        design: str | None = None,
        table: str | None = None,
        clusters: tuple[int | None, ...] | None = None,
        bins: tuple[int | None, ...] | None = None,
        t_values: tuple[float, ...] | None = None,
        alpha: float = 0.05,
        absolute_tolerance: float = 0.025,
        z_tolerance: float = 2.0,
        cell_count_absolute_tolerance: float | None = None,
        source_mixture_absolute_tolerance: float | None = None,
        evidence_dir: str | Path | None = None,
    ) -> pd.DataFrame:
        """Return resumable cell-index chunks for the full paper method suite."""

        if cell_chunk_size <= 0:
            raise ValueError("cell_chunk_size must be positive.")
        if slice_replications is not None and slice_replications <= 0:
            raise ValueError("slice_replications must be positive when provided.")
        if bootstrap_replications <= 0:
            raise ValueError("bootstrap_replications must be positive.")

        seed_rng = np.random.default_rng(seed)
        method_manifests: list[tuple[str, MonteCarloBenchmarkPlanRerunManifest]] = []
        for plan in self.empirical_mixture_benchmark_suite_plans(
            replications=paper_replications,
            method=method,
            mediator=mediator,
            design=design,
            table=table,
            clusters=clusters,
            bins=bins,
            t_values=t_values,
        ):
            method_seed = int(seed_rng.integers(low=0, high=np.iinfo(np.uint32).max, dtype=np.uint32))
            manifest = plan.rerun_manifest(
                data_sources,
                seed=method_seed,
                alpha=alpha,
                bootstrap_replications=bootstrap_replications,
                absolute_tolerance=absolute_tolerance,
                z_tolerance=z_tolerance,
                cell_count_absolute_tolerance=cell_count_absolute_tolerance,
                source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
            )
            method_manifests.append((plan.method, manifest))

        if not method_manifests:
            return pd.DataFrame()

        paper_result_rows = int(
            sum(manifest.plan.paper_result_rows for _, manifest in method_manifests)
        )
        max_cells = max(len(manifest.cells) for _, manifest in method_manifests)
        chunk_replications = paper_replications if slice_replications is None else slice_replications
        suite_chunk_count = int(math.ceil(max_cells / cell_chunk_size))
        suite_planned_draws = paper_result_rows * chunk_replications
        suite_planned_bootstrap_draws = sum(
            len(manifest.cells) * chunk_replications * bootstrap_replications
            for method, manifest in method_manifests
            if _method_requires_bootstrap(method)
        )
        cumulative_covered_result_rows = 0
        cumulative_planned_draws = 0
        cumulative_planned_bootstrap_draws = 0
        rows: list[dict[str, Any]] = []
        for chunk_index, cell_start in enumerate(range(0, max_cells, cell_chunk_size), start=1):
            cell_stop = min(cell_start + cell_chunk_size, max_cells)
            selected_methods: list[str] = []
            selected_cell_count = 0
            selected_by_method: dict[str, int] = {}
            selected_bootstrap_draws = 0
            selected_bootstrap_draws_by_method: dict[str, int] = {}
            for method_name, manifest in method_manifests:
                method_cell_count = max(0, min(cell_stop, len(manifest.cells)) - cell_start)
                if method_cell_count:
                    selected_methods.append(method_name)
                    selected_by_method[method_name] = method_cell_count
                    selected_cell_count += method_cell_count
                    method_bootstrap_draws = (
                        method_cell_count * chunk_replications * bootstrap_replications
                        if _method_requires_bootstrap(method_name)
                        else 0
                    )
                    selected_bootstrap_draws_by_method[method_name] = method_bootstrap_draws
                    selected_bootstrap_draws += method_bootstrap_draws

            planned_draws = selected_cell_count * chunk_replications
            cumulative_covered_result_rows += selected_cell_count
            cumulative_planned_draws += planned_draws
            cumulative_planned_bootstrap_draws += selected_bootstrap_draws
            chunk_rerun_kwargs = _suite_cell_index_chunk_kwargs(
                seed=seed,
                cell_start=cell_start,
                cell_stop=cell_stop,
                paper_replications=paper_replications,
                slice_replications=slice_replications,
                bootstrap_replications=bootstrap_replications,
                method=method,
                mediator=mediator,
                design=design,
                table=table,
                clusters=clusters,
                bins=bins,
                t_values=t_values,
                alpha=alpha,
                absolute_tolerance=absolute_tolerance,
                z_tolerance=z_tolerance,
                cell_count_absolute_tolerance=cell_count_absolute_tolerance,
                source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
            )
            chunk_rerun_call = _suite_cell_index_chunk_call(chunk_rerun_kwargs)
            chunk_evidence_path = _suite_cell_index_chunk_evidence_path(
                evidence_dir,
                chunk_index=chunk_index,
                cell_start=cell_start,
                cell_stop=cell_stop,
            )
            rows.append(
                {
                    "chunk_index": chunk_index,
                    "cell_start": cell_start,
                    "cell_stop": cell_stop,
                    "cell_chunk_size": cell_chunk_size,
                    "seed": seed,
                    "paper_replications": paper_replications,
                    "slice_replications": slice_replications,
                    "chunk_replications": chunk_replications,
                    "bootstrap_replications": bootstrap_replications,
                    "method_count": len(selected_methods),
                    "methods": tuple(selected_methods),
                    "selected_cells_by_method": selected_by_method,
                    "paper_result_rows": paper_result_rows,
                    "covered_result_rows": selected_cell_count,
                    "paper_coverage_shortfall_rows": paper_result_rows - selected_cell_count,
                    "planned_draws": planned_draws,
                    "planned_bootstrap_draws": selected_bootstrap_draws,
                    "planned_bootstrap_draws_by_method": selected_bootstrap_draws_by_method,
                    "suite_chunk_count": suite_chunk_count,
                    "suite_planned_draws": suite_planned_draws,
                    "suite_planned_bootstrap_draws": suite_planned_bootstrap_draws,
                    "cumulative_covered_result_rows": cumulative_covered_result_rows,
                    "cumulative_planned_draws": cumulative_planned_draws,
                    "cumulative_planned_bootstrap_draws": cumulative_planned_bootstrap_draws,
                    "remaining_result_rows_after_chunk": max(
                        paper_result_rows - cumulative_covered_result_rows,
                        0,
                    ),
                    "remaining_draws_after_chunk": max(
                        suite_planned_draws - cumulative_planned_draws,
                        0,
                    ),
                    "remaining_bootstrap_draws_after_chunk": max(
                        suite_planned_bootstrap_draws - cumulative_planned_bootstrap_draws,
                        0,
                    ),
                    "suite_coverage_complete_after_chunk": (
                        cumulative_covered_result_rows >= paper_result_rows
                    ),
                    "next_action": "run_suite_cell_index_chunk",
                    "runner": "empirical_mixture_benchmark_suite_cell_index_run_result",
                    "chunk_rerun_kwargs": chunk_rerun_kwargs,
                    "chunk_rerun_call": chunk_rerun_call,
                    "chunk_evidence_path": chunk_evidence_path,
                    "chunk_export_call": _suite_cell_index_chunk_export_call(
                        chunk_rerun_call,
                        evidence_path=chunk_evidence_path,
                        chunk_index=chunk_index,
                    ),
                }
            )
        return _json_safe_export_frame(rows)

    def empirical_mixture_benchmark_suite_cell_index_schedule_summary(
        self,
        data_sources: dict[str, "BinaryEmpiricalMixtureBenchmarkDataSource"],
        *,
        seed: int,
        cell_chunk_size: int = 1,
        paper_replications: int = 500,
        slice_replications: int | None = None,
        bootstrap_replications: int = 500,
        method: str | None = None,
        mediator: str | None = None,
        design: str | None = None,
        table: str | None = None,
        clusters: tuple[int | None, ...] | None = None,
        bins: tuple[int | None, ...] | None = None,
        t_values: tuple[float, ...] | None = None,
        alpha: float = 0.05,
        absolute_tolerance: float = 0.025,
        z_tolerance: float = 2.0,
        cell_count_absolute_tolerance: float | None = None,
        source_mixture_absolute_tolerance: float | None = None,
        evidence_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Summarize the chunked full-paper rerun hook without executing draws."""

        tolerance_contract = _paper_acceptance_schedule_tolerance_contract_summary(
            absolute_tolerance=absolute_tolerance,
            z_tolerance=z_tolerance,
        )
        requested_chunk_replications = (
            paper_replications if slice_replications is None else slice_replications
        )
        paper_replication_budget_ready = _meets_paper_replication_budget(
            paper_replications
        )
        chunk_replication_budget_ready = (
            int(requested_chunk_replications) == int(paper_replications)
        )
        bootstrap_budget_ready = _meets_paper_bootstrap_budget(
            bootstrap_replications
        )
        full_paper_budget_ready = (
            paper_replication_budget_ready
            and chunk_replication_budget_ready
            and bootstrap_budget_ready
        )

        frame = self.empirical_mixture_benchmark_suite_cell_index_schedule_frame(
            data_sources,
            seed=seed,
            cell_chunk_size=cell_chunk_size,
            paper_replications=paper_replications,
            slice_replications=slice_replications,
            bootstrap_replications=bootstrap_replications,
            method=method,
            mediator=mediator,
            design=design,
            table=table,
            clusters=clusters,
            bins=bins,
            t_values=t_values,
            alpha=alpha,
            absolute_tolerance=absolute_tolerance,
            z_tolerance=z_tolerance,
            cell_count_absolute_tolerance=cell_count_absolute_tolerance,
            source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
            evidence_dir=evidence_dir,
        )
        if frame.empty:
            return {
                "stage": "rerun_schedule",
                "runner": "empirical_mixture_benchmark_suite_cell_index_run_result",
                "hook_ready": False,
                "chunk_count": 0,
                "bootstrap_replications": bootstrap_replications,
                "paper_default_replications": _PAPER_ACCEPTANCE_REPLICATIONS,
                "paper_default_bootstrap_replications": _PAPER_BOOTSTRAP_REPLICATIONS,
                "paper_replication_budget_ready": paper_replication_budget_ready,
                "chunk_replication_budget_ready": chunk_replication_budget_ready,
                "bootstrap_budget_ready": bootstrap_budget_ready,
                "full_paper_budget_ready": full_paper_budget_ready,
                **tolerance_contract,
                "method_count": 0,
                "methods": (),
                "paper_result_rows": 0,
                "scheduled_result_rows": 0,
                "paper_coverage_shortfall_rows": 0,
                "scheduled_draws": 0,
                "scheduled_bootstrap_draws": 0,
                "scheduled_bootstrap_draws_by_method": {},
                "evidence_dir": None if evidence_dir is None else str(evidence_dir),
                "first_chunk_evidence_path": None,
                "first_chunk_export_call": None,
                "next_action": "repair_schedule_coverage",
                "exit_criteria": "paper_acceptance_gate.gate_passes == True",
            }

        selected_by_method: dict[str, int] = {}
        selected_bootstrap_draws_by_method: dict[str, int] = {}
        methods: list[str] = []
        for row in frame.to_dict("records"):
            for method in row["methods"]:
                if method not in methods:
                    methods.append(method)
            for method, count in dict(row["selected_cells_by_method"]).items():
                selected_by_method[method] = selected_by_method.get(method, 0) + int(count)
            for method, draws in dict(row["planned_bootstrap_draws_by_method"]).items():
                selected_bootstrap_draws_by_method[method] = (
                    selected_bootstrap_draws_by_method.get(method, 0) + int(draws)
                )

        paper_result_rows = int(frame["paper_result_rows"].iloc[0])
        scheduled_result_rows = int(frame["covered_result_rows"].sum())
        scheduled_draws = int(frame["planned_draws"].sum())
        scheduled_bootstrap_draws = int(frame["planned_bootstrap_draws"].sum())
        chunk_replications = int(frame["chunk_replications"].iloc[0])
        chunk_replication_budget_ready = (
            int(chunk_replications) == int(paper_replications)
        )
        full_paper_budget_ready = (
            paper_replication_budget_ready
            and chunk_replication_budget_ready
            and bootstrap_budget_ready
        )
        paper_coverage_shortfall_rows = max(paper_result_rows - scheduled_result_rows, 0)
        hook_ready = (
            paper_coverage_shortfall_rows == 0
            and paper_replication_budget_ready
            and chunk_replication_budget_ready
            and bootstrap_budget_ready
            and tolerance_contract["tolerance_contract_ready"]
        )
        next_action = (
            "run_suite_cell_index_chunks"
            if hook_ready
            else "repair_schedule_coverage"
            if paper_coverage_shortfall_rows > 0
            else "restore_paper_rerun_budget"
            if not full_paper_budget_ready
            else "restore_paper_tolerance_contract"
        )
        return {
            "stage": "rerun_schedule",
            "runner": "empirical_mixture_benchmark_suite_cell_index_run_result",
            "hook_ready": hook_ready,
            "seed": seed,
            "cell_chunk_size": cell_chunk_size,
            "chunk_count": int(frame.shape[0]),
            "paper_replications": paper_replications,
            "slice_replications": slice_replications,
            "chunk_replications": chunk_replications,
            "bootstrap_replications": bootstrap_replications,
            "paper_default_replications": _PAPER_ACCEPTANCE_REPLICATIONS,
            "paper_default_bootstrap_replications": _PAPER_BOOTSTRAP_REPLICATIONS,
            "paper_replication_budget_ready": paper_replication_budget_ready,
            "chunk_replication_budget_ready": chunk_replication_budget_ready,
            "bootstrap_budget_ready": bootstrap_budget_ready,
            "full_paper_budget_ready": full_paper_budget_ready,
            **tolerance_contract,
            "method_count": len(methods),
            "methods": tuple(methods),
            "selected_cells_by_method": selected_by_method,
            "scheduled_bootstrap_draws_by_method": selected_bootstrap_draws_by_method,
            "paper_result_rows": paper_result_rows,
            "scheduled_result_rows": scheduled_result_rows,
            "paper_coverage_shortfall_rows": paper_coverage_shortfall_rows,
            "scheduled_draws": scheduled_draws,
            "scheduled_bootstrap_draws": scheduled_bootstrap_draws,
            "evidence_dir": None if evidence_dir is None else str(evidence_dir),
            "first_chunk_evidence_path": frame["chunk_evidence_path"].iloc[0],
            "first_chunk_export_call": frame["chunk_export_call"].iloc[0],
            "last_chunk_remaining_result_rows": int(
                frame["remaining_result_rows_after_chunk"].iloc[-1]
            ),
            "last_chunk_remaining_draws": int(frame["remaining_draws_after_chunk"].iloc[-1]),
            "last_chunk_remaining_bootstrap_draws": int(
                frame["remaining_bootstrap_draws_after_chunk"].iloc[-1]
            ),
            "next_action": next_action,
            "exit_criteria": "paper_acceptance_gate.gate_passes == True",
        }

    def milestone_closeout_gate_summary(
        self,
        paper_acceptance_gate: dict[str, Any],
        *,
        owner: str = "Phase 7 Monte Carlo verification hardening",
        nominal_alpha: float = 0.05,
        tolerance: float = 0.025,
        rerun_command: str | None = None,
    ) -> dict[str, Any]:
        """Combine paper-run acceptance with paper-method support for milestone closeout."""

        method_support = self.method_support_gate_summary(
            nominal_alpha=nominal_alpha,
            tolerance=tolerance,
        )
        paper_gate_passes = bool(paper_acceptance_gate.get("gate_passes"))
        method_gate_passes = bool(method_support["method_gate_passes"])
        paper_scope_conditions = _paper_acceptance_full_suite_scope_conditions(
            paper_acceptance_gate=paper_acceptance_gate,
            method_support=method_support,
        )
        active_paper_scope_conditions = _active_paper_acceptance_blocking_conditions(
            paper_scope_conditions
        )
        paper_active_conditions = {
            **dict(paper_acceptance_gate.get("active_blocking_conditions", {})),
            **active_paper_scope_conditions,
        }
        method_conditions = {
            "method_support_blocked_methods": method_support["blocked_methods"],
            "method_support_blocked_reported_rows": method_support[
                "python_blocked_reported_rows"
            ],
        }
        active_method_conditions = _active_paper_acceptance_blocking_conditions(
            method_conditions
        )
        active_blocking_conditions = {
            **paper_active_conditions,
            **active_method_conditions,
        }
        closeout_workstreams = _milestone_closeout_workstream_rows(
            paper_acceptance_gate=paper_acceptance_gate,
            method_support=method_support,
            paper_active_conditions=paper_active_conditions,
            method_active_conditions=active_method_conditions,
        )
        active_closeout_workstreams = tuple(
            row for row in closeout_workstreams if bool(row.get("blocked"))
        )
        closeout_ready = paper_gate_passes and method_gate_passes and not active_blocking_conditions
        paper_next_action = (
            "run_full_paper_method_suite"
            if active_paper_scope_conditions
            else paper_acceptance_gate.get("next_action")
        )
        next_action = (
            paper_next_action
            if (not paper_gate_passes or paper_active_conditions)
            else method_support["next_action"]
            if not method_gate_passes
            else "ready_for_milestone_archive"
        )
        return {
            "owner": owner,
            "milestone_closeout_ready": closeout_ready,
            "verdict": "pass" if closeout_ready else "blocked",
            "paper_acceptance_gate_passes": paper_gate_passes,
            "method_support_gate_passes": method_gate_passes,
            "paper_acceptance_stage": paper_acceptance_gate.get("stage"),
            "paper_acceptance_next_action": paper_next_action,
            "method_support_next_action": method_support["next_action"],
            "paper_acceptance_active_blocking_conditions": paper_active_conditions,
            "method_support_active_blocking_conditions": active_method_conditions,
            "active_blocking_conditions": active_blocking_conditions,
            "active_blocking_condition_count": len(active_blocking_conditions),
            "closeout_workstreams": list(closeout_workstreams),
            "active_closeout_workstreams": list(active_closeout_workstreams),
            "active_closeout_workstream_count": len(active_closeout_workstreams),
            "blocking_condition_rows": list(
                _milestone_closeout_blocker_rows(
                    paper_acceptance_gate=paper_acceptance_gate,
                    method_support=method_support,
                    paper_scope_conditions=paper_scope_conditions,
                    paper_active_conditions=paper_active_conditions,
                    method_active_conditions=active_method_conditions,
                )
            ),
            "paper_acceptance_gate": paper_acceptance_gate,
            "method_support_gate_summary": method_support,
            "next_action": next_action,
            "resolution_action": next_action,
            "rerun_command": rerun_command,
            "exit_criteria": (
                "paper_acceptance_gate.gate_passes == True and "
                "method_support_gate_summary.method_gate_passes == True"
            ),
        }

    def milestone_closeout_protocol_summary(
        self,
        paper_acceptance_gate: dict[str, Any],
        *,
        owner: str = "Phase 7 Monte Carlo verification hardening",
        nominal_alpha: float = 0.05,
        tolerance: float = 0.025,
        paper_replications: int = 500,
        rerun_command: str | None = None,
    ) -> dict[str, Any]:
        """Return a compact release closeout summary with the paper execution protocol."""

        closeout = self.milestone_closeout_gate_summary(
            paper_acceptance_gate,
            owner=owner,
            nominal_alpha=nominal_alpha,
            tolerance=tolerance,
            rerun_command=rerun_command,
        )
        execution = self.method_execution_contract_summary(
            nominal_alpha=nominal_alpha,
            tolerance=tolerance,
            paper_replications=paper_replications,
        )
        method_support = closeout["method_support_gate_summary"]
        return {
            "owner": closeout["owner"],
            "milestone_closeout_ready": closeout["milestone_closeout_ready"],
            "verdict": closeout["verdict"],
            "paper_acceptance_gate_passes": closeout["paper_acceptance_gate_passes"],
            "method_support_gate_passes": closeout["method_support_gate_passes"],
            "paper_nominal_alpha": execution["paper_nominal_alpha"],
            "paper_replications": execution["paper_replications"],
            "paper_methods": execution["paper_methods"],
            "methods_using_discretized_outcome": execution[
                "methods_using_discretized_outcome"
            ],
            "methods_using_original_outcome": execution[
                "methods_using_original_outcome"
            ],
            "bootstrap_required_methods": execution["bootstrap_required_methods"],
            "analytic_variance_methods": execution["analytic_variance_methods"],
            "python_executable_methods": execution["python_executable_methods"],
            "paper_contract_only_methods": execution["paper_contract_only_methods"],
            "paper_reported_method_rows": method_support["paper_reported_rows"],
            "python_executable_reported_rows": method_support[
                "python_executable_reported_rows"
            ],
            "python_blocked_reported_rows": method_support[
                "python_blocked_reported_rows"
            ],
            "active_blocking_conditions": closeout["active_blocking_conditions"],
            "active_closeout_workstream_count": closeout[
                "active_closeout_workstream_count"
            ],
            "active_closeout_workstreams": closeout["active_closeout_workstreams"],
            "blocking_reason_counts": method_support["blocking_reason_counts"],
            "blocked_method_names": method_support["blocked_method_names"],
            "python_executable_method_names": execution["python_executable_method_names"],
            "paper_contract_only_method_names": execution[
                "paper_contract_only_method_names"
            ],
            "next_action": closeout["next_action"],
            "rerun_command": closeout["rerun_command"],
            "exit_criteria": closeout["exit_criteria"],
            "method_execution_contract_summary": execution,
            "milestone_closeout_gate_summary": closeout,
        }

    def milestone_closeout_blocker_packet(
        self,
        paper_acceptance_gate: dict[str, Any],
        *,
        owner: str = "Phase 7 Monte Carlo verification hardening",
        nominal_alpha: float = 0.05,
        tolerance: float = 0.025,
        paper_replications: int = 500,
        rerun_command: str | None = None,
    ) -> dict[str, Any]:
        """Return an owner-facing packet for combined milestone closeout blockers."""

        protocol = self.milestone_closeout_protocol_summary(
            paper_acceptance_gate,
            owner=owner,
            nominal_alpha=nominal_alpha,
            tolerance=tolerance,
            paper_replications=paper_replications,
            rerun_command=rerun_command,
        )
        closeout = protocol["milestone_closeout_gate_summary"]
        method_support = closeout["method_support_gate_summary"]
        blocked_rows = tuple(
            dict(row)
            for row in closeout.get("blocking_condition_rows", ())
            if bool(row.get("blocked"))
        )
        return {
            "owner": owner,
            "cleared_same_line_items": {
                "paper_methods": protocol["paper_methods"],
                "method_execution_protocol_rows": protocol["paper_methods"],
                "paper_reported_method_rows": protocol["paper_reported_method_rows"],
                "python_executable_reported_rows": protocol[
                    "python_executable_reported_rows"
                ],
                "paper_acceptance_stage": closeout["paper_acceptance_stage"],
            },
            "blocked_same_line_items": {
                "milestone_closeout_ready": protocol["milestone_closeout_ready"],
                "paper_acceptance_gate_passes": protocol[
                    "paper_acceptance_gate_passes"
                ],
                "method_support_gate_passes": protocol["method_support_gate_passes"],
                "active_blocking_conditions": dict(
                    protocol["active_blocking_conditions"]
                ),
                "active_closeout_workstream_count": protocol[
                    "active_closeout_workstream_count"
                ],
                "blocked_method_names": protocol["blocked_method_names"],
                "python_blocked_reported_rows": protocol[
                    "python_blocked_reported_rows"
                ],
            },
            "verified_boundary": {
                "paper_nominal_alpha": protocol["paper_nominal_alpha"],
                "paper_replications": protocol["paper_replications"],
                "methods_using_discretized_outcome": protocol[
                    "methods_using_discretized_outcome"
                ],
                "methods_using_original_outcome": protocol[
                    "methods_using_original_outcome"
                ],
                "bootstrap_required_methods": protocol["bootstrap_required_methods"],
                "analytic_variance_methods": protocol["analytic_variance_methods"],
                "paper_acceptance_next_action": closeout[
                    "paper_acceptance_next_action"
                ],
                "method_support_next_action": closeout["method_support_next_action"],
            },
            "evidence_checked": {
                "paper_acceptance_active_blocking_conditions": dict(
                    closeout["paper_acceptance_active_blocking_conditions"]
                ),
                "method_support_active_blocking_conditions": dict(
                    closeout["method_support_active_blocking_conditions"]
                ),
                "active_closeout_workstreams": tuple(
                    dict(row) for row in protocol["active_closeout_workstreams"]
                ),
                "blocking_reason_counts": dict(protocol["blocking_reason_counts"]),
                "blocked_condition_rows": blocked_rows,
                "python_executable_method_names": protocol[
                    "python_executable_method_names"
                ],
                "paper_contract_only_method_names": protocol[
                    "paper_contract_only_method_names"
                ],
            },
            "paper_acceptance_gate": closeout["paper_acceptance_gate"],
            "method_support_gate_summary": method_support,
            "milestone_closeout_gate_summary": closeout,
            "method_execution_contract_summary": protocol[
                "method_execution_contract_summary"
            ],
            "rerun_command": closeout["rerun_command"],
            "exit_criteria": closeout["exit_criteria"],
        }

    def milestone_closeout_blocker_frame(
        self,
        paper_acceptance_gate: dict[str, Any],
        *,
        owner: str = "Phase 7 Monte Carlo verification hardening",
        nominal_alpha: float = 0.05,
        tolerance: float = 0.025,
        rerun_command: str | None = None,
    ) -> pd.DataFrame:
        """Return row-level blockers for the combined milestone closeout gate."""

        summary = self.milestone_closeout_gate_summary(
            paper_acceptance_gate,
            owner=owner,
            nominal_alpha=nominal_alpha,
            tolerance=tolerance,
            rerun_command=rerun_command,
        )
        rows: list[dict[str, Any]] = []
        blocking_rows = tuple(summary.get("blocking_condition_rows", ()))
        for index, blocker_row in enumerate(blocking_rows, start=1):
            rows.append(
                {
                    "owner": summary["owner"],
                    "milestone_closeout_ready": summary["milestone_closeout_ready"],
                    "verdict": summary["verdict"],
                    "paper_acceptance_gate_passes": summary["paper_acceptance_gate_passes"],
                    "method_support_gate_passes": summary["method_support_gate_passes"],
                    "paper_acceptance_stage": summary["paper_acceptance_stage"],
                    "paper_acceptance_next_action": summary["paper_acceptance_next_action"],
                    "method_support_next_action": summary["method_support_next_action"],
                    "active_blocking_condition_count": summary["active_blocking_condition_count"],
                    "active_closeout_workstream_count": summary[
                        "active_closeout_workstream_count"
                    ],
                    "blocking_condition_index": index,
                    "blocking_condition_count": len(blocking_rows),
                    "next_action": summary["next_action"],
                    "exit_criteria": summary["exit_criteria"],
                    "rerun_command": summary["rerun_command"],
                    **blocker_row,
                }
            )
        return _json_safe_export_frame(rows)

    def raise_for_milestone_closeout_blockers(
        self,
        paper_acceptance_gate: dict[str, Any],
        *,
        owner: str = "Phase 7 Monte Carlo verification hardening",
        nominal_alpha: float = 0.05,
        tolerance: float = 0.025,
        rerun_command: str | None = None,
    ) -> None:
        """Raise if paper-run acceptance or paper-method support blocks closeout."""

        summary = self.milestone_closeout_gate_summary(
            paper_acceptance_gate,
            owner=owner,
            nominal_alpha=nominal_alpha,
            tolerance=tolerance,
            rerun_command=rerun_command,
        )
        if bool(summary["milestone_closeout_ready"]):
            return
        blocked_rows = [
            {
                "gate": row.get("gate"),
                "condition": row.get("condition"),
                "value": row.get("value"),
                "category": row.get("category"),
                "resolution_action": row.get("resolution_action"),
            }
            for row in summary.get("blocking_condition_rows", ())
            if bool(row.get("blocked"))
        ]
        raise AssertionError(
            "Milestone closeout blocked by Monte Carlo release gates: "
            f"paper_acceptance_gate_passes={summary['paper_acceptance_gate_passes']!r}, "
            f"method_support_gate_passes={summary['method_support_gate_passes']!r}, "
            f"next_action={summary['next_action']!r}, "
            f"active_blocking_conditions={summary['active_blocking_conditions']!r}, "
            f"blocking_condition_preview={blocked_rows[:5]!r}, "
            f"exit_criteria={summary['exit_criteria']!r}"
        )

    def milestone_archive_gate_summary(
        self,
        paper_acceptance_gate: dict[str, Any],
        *,
        roadmap_analysis: dict[str, Any] | None,
        milestone_audit_status: str | None = None,
        milestone_audit_path: str | Path | None = None,
        milestone_version: str | None = None,
        planning_dir: str | Path = ".planning",
        owner: str = "Phase 7 Monte Carlo verification hardening",
        nominal_alpha: float = 0.05,
        tolerance: float = 0.025,
        rerun_command: str | None = None,
    ) -> dict[str, Any]:
        """Combine GSD completion preflight with Monte Carlo closeout blockers."""

        closeout = self.milestone_closeout_gate_summary(
            paper_acceptance_gate,
            owner=owner,
            nominal_alpha=nominal_alpha,
            tolerance=tolerance,
            rerun_command=rerun_command,
        )
        roadmap = _milestone_archive_roadmap_summary(roadmap_analysis)
        resolved_audit_status = _resolve_milestone_archive_audit_status(
            milestone_audit_status=milestone_audit_status,
            milestone_audit_path=milestone_audit_path,
            milestone_version=milestone_version,
            planning_dir=planning_dir,
        )
        if (
            not bool(roadmap["roadmap_disk_complete"])
            and milestone_audit_status is None
            and milestone_audit_path is None
            and milestone_version is not None
        ):
            resolved_audit_status = None
        audit = _milestone_archive_audit_summary(resolved_audit_status)
        blocker_rows = _milestone_archive_blocker_rows(
            roadmap=roadmap,
            audit=audit,
            closeout=closeout,
        )
        active_blocking_conditions = _active_blocking_conditions_from_rows(blocker_rows)
        archive_ready = (
            bool(roadmap["roadmap_disk_complete"])
            and bool(audit["milestone_audit_passes"])
            and bool(closeout["milestone_closeout_ready"])
            and not active_blocking_conditions
        )
        next_action = _milestone_archive_next_action(
            roadmap=roadmap,
            audit=audit,
            closeout=closeout,
        )
        complete_milestone_preflight = _complete_milestone_preflight_decision(
            archive_ready
        )
        return {
            "owner": owner,
            "milestone_archive_ready": archive_ready,
            "verdict": "pass" if archive_ready else "blocked",
            "roadmap_analysis_available": roadmap["roadmap_analysis_available"],
            "roadmap_disk_complete": roadmap["roadmap_disk_complete"],
            "roadmap_phase_count": roadmap["phase_count"],
            "roadmap_completed_phases": roadmap["completed_phases"],
            "roadmap_total_plans": roadmap["total_plans"],
            "roadmap_total_summaries": roadmap["total_summaries"],
            "roadmap_progress_percent": roadmap["progress_percent"],
            "roadmap_incomplete_phases": roadmap["incomplete_phases"],
            "milestone_audit_status": audit["milestone_audit_status"],
            "milestone_audit_passes": audit["milestone_audit_passes"],
            "milestone_closeout_ready": closeout["milestone_closeout_ready"],
            "paper_acceptance_gate_passes": closeout["paper_acceptance_gate_passes"],
            "method_support_gate_passes": closeout["method_support_gate_passes"],
            "active_blocking_conditions": active_blocking_conditions,
            "active_blocking_condition_count": len(active_blocking_conditions),
            "blocking_condition_rows": list(blocker_rows),
            "milestone_closeout_gate_summary": closeout,
            "complete_milestone_preflight": complete_milestone_preflight,
            "next_action": next_action,
            "resolution_action": next_action,
            "rerun_command": rerun_command,
            "exit_criteria": (
                "roadmap_disk_complete == True and "
                "milestone_audit_status == 'passed' and "
                "milestone_closeout_gate_summary.milestone_closeout_ready == True"
            ),
        }

    def milestone_archive_live_gate_summary(
        self,
        paper_acceptance_gate: dict[str, Any],
        *,
        milestone_version: str,
        repo_root: str | Path = ".",
        planning_dir: str | Path = ".planning",
        gsd_tools_path: str | Path | None = None,
        owner: str = "Phase 12 release re-audit and archive readiness",
        nominal_alpha: float = 0.05,
        tolerance: float = 0.025,
        rerun_command: str | None = None,
    ) -> dict[str, Any]:
        """Resolve live roadmap state, then apply the existing archive gate."""

        resolved_repo_root = Path(repo_root).resolve()
        resolved_planning_dir = Path(planning_dir)
        if not resolved_planning_dir.is_absolute():
            resolved_planning_dir = (resolved_repo_root / resolved_planning_dir).resolve()
        roadmap = _load_roadmap_analysis_from_gsd_tools(
            repo_root=resolved_repo_root,
            gsd_tools_path=gsd_tools_path,
        )
        return self.milestone_archive_gate_summary(
            paper_acceptance_gate,
            roadmap_analysis=roadmap,
            milestone_version=milestone_version,
            planning_dir=resolved_planning_dir,
            owner=owner,
            nominal_alpha=nominal_alpha,
            tolerance=tolerance,
            rerun_command=rerun_command,
        )

    def milestone_archive_export_live_gate_summary(
        self,
        exported_run_results: Any,
        data_sources: dict[str, "BinaryEmpiricalMixtureBenchmarkDataSource"],
        *,
        seed: int,
        milestone_version: str,
        repo_root: str | Path = ".",
        planning_dir: str | Path = ".planning",
        gsd_tools_path: str | Path | None = None,
        owner: str = "Phase 12 release re-audit and archive readiness",
        nominal_alpha: float = 0.05,
        tolerance: float = 0.025,
        cell_chunk_size: int = 1,
        paper_replications: int = 500,
        slice_replications: int | None = None,
        bootstrap_replications: int = 500,
        mediator: str | None = None,
        design: str | None = None,
        table: str | None = None,
        clusters: tuple[int | None, ...] | None = None,
        bins: tuple[int | None, ...] | None = None,
        t_values: tuple[float, ...] | None = None,
        alpha: float = 0.05,
        absolute_tolerance: float = 0.025,
        z_tolerance: float = 2.0,
        cell_count_absolute_tolerance: float | None = None,
        source_mixture_absolute_tolerance: float | None = None,
        rerun_command: str | None = None,
    ) -> dict[str, Any]:
        """Resolve saved suite evidence, then apply the live archive gate."""

        paper_acceptance_gate = self.empirical_mixture_benchmark_suite_export_acceptance_gate(
            exported_run_results,
            data_sources,
            seed=seed,
            cell_chunk_size=cell_chunk_size,
            paper_replications=paper_replications,
            slice_replications=slice_replications,
            bootstrap_replications=bootstrap_replications,
            mediator=mediator,
            design=design,
            table=table,
            clusters=clusters,
            bins=bins,
            t_values=t_values,
            alpha=alpha,
            absolute_tolerance=absolute_tolerance,
            z_tolerance=z_tolerance,
            cell_count_absolute_tolerance=cell_count_absolute_tolerance,
            source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
        )
        return self.milestone_archive_live_gate_summary(
            paper_acceptance_gate,
            milestone_version=milestone_version,
            repo_root=repo_root,
            planning_dir=planning_dir,
            gsd_tools_path=gsd_tools_path,
            owner=owner,
            nominal_alpha=nominal_alpha,
            tolerance=tolerance,
            rerun_command=rerun_command,
        )

    def milestone_archive_live_blocker_frame(
        self,
        paper_acceptance_gate: dict[str, Any],
        *,
        milestone_version: str,
        repo_root: str | Path = ".",
        planning_dir: str | Path = ".planning",
        gsd_tools_path: str | Path | None = None,
        owner: str = "Phase 12 release re-audit and archive readiness",
        nominal_alpha: float = 0.05,
        tolerance: float = 0.025,
        rerun_command: str | None = None,
    ) -> pd.DataFrame:
        """Return row-level live archive blockers from checkout state."""

        summary = self.milestone_archive_live_gate_summary(
            paper_acceptance_gate,
            milestone_version=milestone_version,
            repo_root=repo_root,
            planning_dir=planning_dir,
            gsd_tools_path=gsd_tools_path,
            owner=owner,
            nominal_alpha=nominal_alpha,
            tolerance=tolerance,
            rerun_command=rerun_command,
        )
        rows: list[dict[str, Any]] = []
        blocking_rows = tuple(summary.get("blocking_condition_rows", ()))
        for index, blocker_row in enumerate(blocking_rows, start=1):
            rows.append(
                {
                    "owner": summary["owner"],
                    "milestone_archive_ready": summary["milestone_archive_ready"],
                    "verdict": summary["verdict"],
                    "roadmap_disk_complete": summary["roadmap_disk_complete"],
                    "milestone_audit_status": summary["milestone_audit_status"],
                    "milestone_audit_passes": summary["milestone_audit_passes"],
                    "milestone_closeout_ready": summary["milestone_closeout_ready"],
                    "active_blocking_condition_count": summary[
                        "active_blocking_condition_count"
                    ],
                    "blocking_condition_index": index,
                    "blocking_condition_count": len(blocking_rows),
                    "next_action": summary["next_action"],
                    "exit_criteria": summary["exit_criteria"],
                    "rerun_command": summary["rerun_command"],
                    **blocker_row,
                }
            )
        return _json_safe_export_frame(rows)

    def milestone_archive_live_blocker_packet(
        self,
        paper_acceptance_gate: dict[str, Any],
        *,
        milestone_version: str,
        repo_root: str | Path = ".",
        planning_dir: str | Path = ".planning",
        gsd_tools_path: str | Path | None = None,
        owner: str = "Phase 12 release re-audit and archive readiness",
        nominal_alpha: float = 0.05,
        tolerance: float = 0.025,
        rerun_command: str | None = None,
    ) -> dict[str, Any]:
        """Return an owner-facing packet for live archive blockers."""

        archive = self.milestone_archive_live_gate_summary(
            paper_acceptance_gate,
            milestone_version=milestone_version,
            repo_root=repo_root,
            planning_dir=planning_dir,
            gsd_tools_path=gsd_tools_path,
            owner=owner,
            nominal_alpha=nominal_alpha,
            tolerance=tolerance,
            rerun_command=rerun_command,
        )
        closeout = archive["milestone_closeout_gate_summary"]
        blocked_rows = tuple(
            dict(row)
            for row in archive.get("blocking_condition_rows", ())
            if bool(row.get("blocked"))
        )
        return {
            "owner": owner,
            "cleared_same_line_items": {
                "roadmap_analysis_available": archive["roadmap_analysis_available"],
                "roadmap_disk_complete": archive["roadmap_disk_complete"],
                "roadmap_phase_count": archive["roadmap_phase_count"],
                "roadmap_completed_phases": archive["roadmap_completed_phases"],
                "roadmap_total_plans": archive["roadmap_total_plans"],
                "roadmap_total_summaries": archive["roadmap_total_summaries"],
            },
            "blocked_same_line_items": {
                "milestone_archive_ready": archive["milestone_archive_ready"],
                "milestone_audit_status": archive["milestone_audit_status"],
                "milestone_audit_passes": archive["milestone_audit_passes"],
                "milestone_closeout_ready": archive["milestone_closeout_ready"],
                "paper_acceptance_gate_passes": archive[
                    "paper_acceptance_gate_passes"
                ],
                "method_support_gate_passes": archive["method_support_gate_passes"],
                "active_blocking_conditions": dict(
                    archive["active_blocking_conditions"]
                ),
            },
            "verified_boundary": {
                "roadmap_progress_percent": archive["roadmap_progress_percent"],
                "roadmap_incomplete_phases": tuple(
                    archive["roadmap_incomplete_phases"]
                ),
                "milestone_audit_status": archive["milestone_audit_status"],
                "paper_acceptance_stage": closeout["paper_acceptance_stage"],
                "paper_acceptance_next_action": closeout[
                    "paper_acceptance_next_action"
                ],
                "method_support_next_action": closeout["method_support_next_action"],
                "archive_next_action": archive["next_action"],
            },
            "evidence_checked": {
                "blocked_condition_rows": blocked_rows,
                "blocking_condition_count": len(
                    archive.get("blocking_condition_rows", ())
                ),
                "active_closeout_workstreams": tuple(
                    dict(row) for row in closeout["active_closeout_workstreams"]
                ),
                "closeout_active_blocking_conditions": dict(
                    closeout["active_blocking_conditions"]
                ),
            },
            "paper_acceptance_gate": closeout["paper_acceptance_gate"],
            "method_support_gate_summary": closeout["method_support_gate_summary"],
            "milestone_closeout_gate_summary": closeout,
            "milestone_archive_gate_summary": archive,
            "complete_milestone_preflight": archive["complete_milestone_preflight"],
            "rerun_hook": _archive_gate_rerun_hook(archive),
            "rerun_command": archive["rerun_command"],
            "exit_criteria": archive["exit_criteria"],
        }

    def milestone_archive_blocker_packet(
        self,
        paper_acceptance_gate: dict[str, Any],
        *,
        roadmap_analysis: dict[str, Any] | None,
        milestone_audit_status: str | None = None,
        milestone_audit_path: str | Path | None = None,
        milestone_version: str | None = None,
        planning_dir: str | Path = ".planning",
        owner: str = "Phase 7 Monte Carlo verification hardening",
        nominal_alpha: float = 0.05,
        tolerance: float = 0.025,
        rerun_command: str | None = None,
    ) -> dict[str, Any]:
        """Return an owner-facing packet for archive/tag blockers."""

        archive = self.milestone_archive_gate_summary(
            paper_acceptance_gate,
            roadmap_analysis=roadmap_analysis,
            milestone_audit_status=milestone_audit_status,
            milestone_audit_path=milestone_audit_path,
            milestone_version=milestone_version,
            planning_dir=planning_dir,
            owner=owner,
            nominal_alpha=nominal_alpha,
            tolerance=tolerance,
            rerun_command=rerun_command,
        )
        closeout = archive["milestone_closeout_gate_summary"]
        blocked_rows = tuple(
            dict(row)
            for row in archive.get("blocking_condition_rows", ())
            if bool(row.get("blocked"))
        )
        return {
            "owner": owner,
            "cleared_same_line_items": {
                "roadmap_analysis_available": archive["roadmap_analysis_available"],
                "roadmap_disk_complete": archive["roadmap_disk_complete"],
                "roadmap_phase_count": archive["roadmap_phase_count"],
                "roadmap_completed_phases": archive["roadmap_completed_phases"],
                "roadmap_total_plans": archive["roadmap_total_plans"],
                "roadmap_total_summaries": archive["roadmap_total_summaries"],
            },
            "blocked_same_line_items": {
                "milestone_archive_ready": archive["milestone_archive_ready"],
                "milestone_audit_status": archive["milestone_audit_status"],
                "milestone_audit_passes": archive["milestone_audit_passes"],
                "milestone_closeout_ready": archive["milestone_closeout_ready"],
                "paper_acceptance_gate_passes": archive[
                    "paper_acceptance_gate_passes"
                ],
                "method_support_gate_passes": archive["method_support_gate_passes"],
                "active_blocking_conditions": dict(
                    archive["active_blocking_conditions"]
                ),
            },
            "verified_boundary": {
                "roadmap_progress_percent": archive["roadmap_progress_percent"],
                "roadmap_incomplete_phases": tuple(
                    archive["roadmap_incomplete_phases"]
                ),
                "milestone_audit_status": archive["milestone_audit_status"],
                "paper_acceptance_stage": closeout["paper_acceptance_stage"],
                "paper_acceptance_next_action": closeout[
                    "paper_acceptance_next_action"
                ],
                "method_support_next_action": closeout["method_support_next_action"],
                "archive_next_action": archive["next_action"],
            },
            "evidence_checked": {
                "blocked_condition_rows": blocked_rows,
                "blocking_condition_count": len(
                    archive.get("blocking_condition_rows", ())
                ),
                "active_closeout_workstreams": tuple(
                    dict(row) for row in closeout["active_closeout_workstreams"]
                ),
                "closeout_active_blocking_conditions": dict(
                    closeout["active_blocking_conditions"]
                ),
            },
            "paper_acceptance_gate": closeout["paper_acceptance_gate"],
            "method_support_gate_summary": closeout["method_support_gate_summary"],
            "milestone_closeout_gate_summary": closeout,
            "milestone_archive_gate_summary": archive,
            "complete_milestone_preflight": archive["complete_milestone_preflight"],
            "rerun_hook": _archive_gate_rerun_hook(archive),
            "rerun_command": archive["rerun_command"],
            "exit_criteria": archive["exit_criteria"],
        }

    def milestone_archive_blocker_frame(
        self,
        paper_acceptance_gate: dict[str, Any],
        *,
        roadmap_analysis: dict[str, Any] | None,
        milestone_audit_status: str | None = None,
        milestone_audit_path: str | Path | None = None,
        milestone_version: str | None = None,
        planning_dir: str | Path = ".planning",
        owner: str = "Phase 7 Monte Carlo verification hardening",
        nominal_alpha: float = 0.05,
        tolerance: float = 0.025,
        rerun_command: str | None = None,
    ) -> pd.DataFrame:
        """Return row-level GSD archive blockers joined to release closeout blockers."""

        summary = self.milestone_archive_gate_summary(
            paper_acceptance_gate,
            roadmap_analysis=roadmap_analysis,
            milestone_audit_status=milestone_audit_status,
            milestone_audit_path=milestone_audit_path,
            milestone_version=milestone_version,
            planning_dir=planning_dir,
            owner=owner,
            nominal_alpha=nominal_alpha,
            tolerance=tolerance,
            rerun_command=rerun_command,
        )
        rows: list[dict[str, Any]] = []
        blocking_rows = tuple(summary.get("blocking_condition_rows", ()))
        for index, blocker_row in enumerate(blocking_rows, start=1):
            rows.append(
                {
                    "owner": summary["owner"],
                    "milestone_archive_ready": summary["milestone_archive_ready"],
                    "verdict": summary["verdict"],
                    "roadmap_disk_complete": summary["roadmap_disk_complete"],
                    "milestone_audit_status": summary["milestone_audit_status"],
                    "milestone_audit_passes": summary["milestone_audit_passes"],
                    "milestone_closeout_ready": summary["milestone_closeout_ready"],
                    "active_blocking_condition_count": summary[
                        "active_blocking_condition_count"
                    ],
                    "blocking_condition_index": index,
                    "blocking_condition_count": len(blocking_rows),
                    "next_action": summary["next_action"],
                    "exit_criteria": summary["exit_criteria"],
                    "rerun_command": summary["rerun_command"],
                    **blocker_row,
                }
            )
        return _json_safe_export_frame(rows)

    def raise_for_milestone_archive_blockers(
        self,
        paper_acceptance_gate: dict[str, Any],
        *,
        roadmap_analysis: dict[str, Any] | None,
        milestone_audit_status: str | None = None,
        milestone_audit_path: str | Path | None = None,
        milestone_version: str | None = None,
        planning_dir: str | Path = ".planning",
        owner: str = "Phase 7 Monte Carlo verification hardening",
        nominal_alpha: float = 0.05,
        tolerance: float = 0.025,
        rerun_command: str | None = None,
    ) -> None:
        """Raise if GSD preflight or Monte Carlo release gates block archive/tag."""

        summary = self.milestone_archive_gate_summary(
            paper_acceptance_gate,
            roadmap_analysis=roadmap_analysis,
            milestone_audit_status=milestone_audit_status,
            milestone_audit_path=milestone_audit_path,
            milestone_version=milestone_version,
            planning_dir=planning_dir,
            owner=owner,
            nominal_alpha=nominal_alpha,
            tolerance=tolerance,
            rerun_command=rerun_command,
        )
        if bool(summary["milestone_archive_ready"]):
            return
        blocked_rows = [
            {
                "gate": row.get("gate"),
                "condition": row.get("condition"),
                "value": row.get("value"),
                "category": row.get("category"),
                "resolution_action": row.get("resolution_action"),
            }
            for row in summary.get("blocking_condition_rows", ())
            if bool(row.get("blocked"))
        ]
        raise AssertionError(
            "Milestone archive blocked by GSD preflight or release gates: "
            f"roadmap_disk_complete={summary['roadmap_disk_complete']!r}, "
            f"milestone_audit_status={summary['milestone_audit_status']!r}, "
            f"milestone_closeout_ready={summary['milestone_closeout_ready']!r}, "
            f"next_action={summary['next_action']!r}, "
            f"active_blocking_conditions={summary['active_blocking_conditions']!r}, "
            f"blocking_condition_preview={blocked_rows[:5]!r}, "
            f"exit_criteria={summary['exit_criteria']!r}"
        )

    def nonbinary_empirical_mixture_cs_readiness_frame(
        self,
        data_sources: dict[str, "BinaryEmpiricalMixtureBenchmarkDataSource"] | None = None,
        *,
        replications: int = 500,
        absolute_tolerance: float | None = 0.025,
        z_tolerance: float | None = 2.0,
        design: str | None = None,
        table: str | None = None,
        clusters: tuple[int | None, ...] | None = None,
        bins: tuple[int | None, ...] | None = None,
        t_values: tuple[float, ...] | None = None,
    ) -> pd.DataFrame:
        """Return row-level preflight evidence for the nonbinary CS paper rows."""

        if replications <= 0:
            raise ValueError("replications must be positive.")
        _validate_target_precision_tolerances(
            absolute_tolerance=absolute_tolerance,
            z_tolerance=z_tolerance,
        )

        sources = data_sources or {}
        cells = self.benchmark_cells(
            method="CS",
            mediator="nonbinary",
            design=design,
            table=table,
            clusters=clusters,
            bins=bins,
            t_values=t_values,
        )
        cells_by_design: dict[str, list[MonteCarloBenchmarkCell]] = {}
        for cell in cells:
            cells_by_design.setdefault(cell.design, []).append(cell)

        diagnostics_by_design = {}
        for paper_design, design_cells in cells_by_design.items():
            data_source_key, source = _data_source_binding_for_benchmark_cells(
                sources,
                design=paper_design,
                cells=tuple(design_cells),
            )
            diagnostic = _nonbinary_empirical_mixture_data_source_diagnostic(
                design=paper_design,
                cells=tuple(design_cells),
                source=source,
            )
            diagnostics_by_design[paper_design] = {
                **diagnostic,
                "data_source_key": data_source_key,
            }

        rows: list[dict[str, Any]] = []
        for cell in cells:
            diagnostic = diagnostics_by_design[cell.design]
            cell_count = cell.cluster_cell_count
            size_risk = None if cell_count is None else cell_count.size_risk
            target_standard_error = _target_rejection_rate_standard_error(
                cell.target_rejection_rate,
                replications=replications,
            )
            target_error_band = (
                None
                if z_tolerance is None
                else float(z_tolerance) * target_standard_error
            )
            target_required_replications = _required_replications_for_binomial_precision(
                probability=cell.target_rejection_rate,
                absolute_tolerance=absolute_tolerance,
                z_tolerance=z_tolerance,
                trials_per_replication=1,
            )
            data_source_ready = bool(diagnostic["ready"])
            implementation_ready = True
            implementation_blocking_reasons: tuple[str, ...] = ()
            release_blocking_reasons: tuple[str, ...] = (
                "nonbinary_cs_paper_matrix_not_run",
            )
            caution_reasons: tuple[str, ...] = (
                ("cell_count_policy_size_risk",) if bool(size_risk) else ()
            )
            row_preflight_ready = (
                data_source_ready
                and implementation_ready
            )
            row_release_ready = row_preflight_ready and not bool(release_blocking_reasons)
            rows.append(
                {
                    **cell.to_dict(),
                    "status": "ready" if row_preflight_ready else "blocked",
                    "preflight_status": "ready" if row_preflight_ready else "blocked",
                    "release_status": (
                        "ready"
                        if row_release_ready and not bool(release_blocking_reasons)
                        else "blocked"
                    ),
                    "planned_replications": 0,
                    "planned_draws": 0,
                    "paper_default_replications": replications,
                    "target_precision_replications": replications,
                    "target_mc_standard_error": target_standard_error,
                    "z_tolerance": z_tolerance,
                    "target_mc_error_band": target_error_band,
                    "absolute_tolerance": absolute_tolerance,
                    "target_tolerance_below_error_band": bool(
                        absolute_tolerance is not None
                        and target_error_band is not None
                        and target_error_band > float(absolute_tolerance)
                    ),
                    "target_required_replications_for_tolerance": target_required_replications,
                    "target_replication_shortfall": _replication_shortfall(
                        required_replications=target_required_replications,
                        planned_replications=replications,
                    ),
                    "implementation_ready": implementation_ready,
                    "implementation_blocking_reasons": implementation_blocking_reasons,
                    "required_runner": "nonbinary_empirical_mixture_cs_runner",
                    "data_source_ready": diagnostic["ready"],
                    "data_source_key": diagnostic["data_source_key"],
                    "data_source_blocking_reasons": diagnostic["blocking_reasons"],
                    "data_source_complete_case_rows": diagnostic["complete_case_rows"],
                    "data_source_control_rows": diagnostic["control_rows"],
                    "data_source_treated_rows": diagnostic["treated_rows"],
                    "data_source_source_clusters": diagnostic["source_clusters"],
                    "data_source_control_source_clusters": diagnostic[
                        "control_source_clusters"
                    ],
                    "data_source_treated_source_clusters": diagnostic[
                        "treated_source_clusters"
                    ],
                    "source_mediator_level_count": diagnostic[
                        "mediator_level_count"
                    ],
                    "source_mediator_levels": diagnostic["mediator_levels"],
                    "expected_mediator_level_count": diagnostic[
                        "expected_mediator_level_count"
                    ],
                    "source_outcome_level_count": diagnostic["outcome_level_count"],
                    "arm_fixed_source_clusters": diagnostic[
                        "arm_fixed_source_clusters"
                    ],
                    "cell_count_policy_available": cell_count is not None,
                    "target_median_independent_clusters_per_cell": (
                        None
                        if cell_count is None
                        else cell_count.median_independent_clusters_per_cell
                    ),
                    "cell_count_size_risk_threshold": (
                        None if cell_count is None else cell_count.size_risk_threshold
                    ),
                    "cell_count_policy_size_risk": size_risk,
                    "recommended_by_cell_count": (
                        None if size_risk is None else not size_risk
                    ),
                    "bin_policy": (
                        None
                        if size_risk is None
                        else (
                            "below_cell_count_heuristic"
                            if size_risk
                            else "within_cell_count_heuristic"
                        )
                    ),
                    "paper_rule": (
                        None
                        if cell_count is None
                        else "at least 15 independent observations per cell"
                    ),
                    "paper_source_contract": (
                        "Baranov relationship-quality empirical mixture"
                    ),
                    "paper_reference": (
                        "manuscript/sources/arxiv-2404.11739v3/draft.tex:391-438; "
                        "packages/r/TestMechs/README.md:299-344"
                    ),
                    "caution_reasons": caution_reasons,
                    "row_preflight_ready": row_preflight_ready,
                    "row_release_ready": row_release_ready,
                    "release_blocking_reasons": release_blocking_reasons,
                }
            )
        return _json_safe_export_frame(rows)

    def nonbinary_empirical_mixture_cs_readiness_summary(
        self,
        data_sources: dict[str, "BinaryEmpiricalMixtureBenchmarkDataSource"] | None = None,
        *,
        replications: int = 500,
        absolute_tolerance: float | None = 0.025,
        z_tolerance: float | None = 2.0,
        design: str | None = None,
        table: str | None = None,
        clusters: tuple[int | None, ...] | None = None,
        bins: tuple[int | None, ...] | None = None,
        t_values: tuple[float, ...] | None = None,
    ) -> dict[str, Any]:
        """Return compact readiness counts for the nonbinary CS paper surface."""

        frame = self.nonbinary_empirical_mixture_cs_readiness_frame(
            data_sources,
            replications=replications,
            absolute_tolerance=absolute_tolerance,
            z_tolerance=z_tolerance,
            design=design,
            table=table,
            clusters=clusters,
            bins=bins,
            t_values=t_values,
        )
        if frame.empty:
            return {
                "method": "CS",
                "mediator": "nonbinary",
                "paper_rows": 0,
                "paper_default_replications": replications,
                "planned_draws": 0,
                "data_source_checked_designs": 0,
                "data_source_ready_designs": 0,
                "data_source_blocked_designs": 0,
                "data_source_ready_rows": 0,
                "data_source_blocked_rows": 0,
                "implementation_blocked_rows": 0,
                "release_blocked_rows": 0,
                "row_preflight_ready_rows": 0,
                "row_release_ready_rows": 0,
                "cell_count_policy_rows": 0,
                "cell_count_policy_size_risk_rows": 0,
                "target_precision_rows": 0,
                "target_tolerance_below_error_band_rows": 0,
                "target_budget_rows": 0,
                "target_shortfall_rows": 0,
                "max_target_mc_standard_error": None,
                "max_target_mc_error_band": None,
                "max_target_required_replications": None,
                "max_target_replication_shortfall": None,
                "min_target_median_independent_clusters_per_cell": None,
                "source_complete_case_rows": {},
                "source_complete_case_rows_by_source_key": {},
                "source_mediator_level_counts": {},
                "source_mediator_level_counts_by_source_key": {},
                "source_clusters": {},
                "source_clusters_by_source_key": {},
                "data_source_blocking_reason_counts": {},
                "implementation_blocking_reason_counts": {},
                "release_blocking_reason_counts": {},
                "caution_reason_counts": {},
                "blocking_reason_counts": {},
                "next_action": "no_nonbinary_cs_rows_selected",
                "exit_criteria": (
                    "nonbinary_empirical_mixture_cs_readiness_summary."
                    "row_release_ready_rows == paper_rows"
                ),
            }

        data_source_ready = frame["data_source_ready"].map(_truthy_value)
        implementation_ready = frame["implementation_ready"].map(_truthy_value)
        row_preflight_ready = frame["row_preflight_ready"].map(_truthy_value)
        row_release_ready = frame["row_release_ready"].map(_truthy_value)
        cell_count_available = frame["cell_count_policy_available"].map(_truthy_value)
        cell_count_size_risk = frame["cell_count_policy_size_risk"].map(_truthy_value)

        data_source_reason_counts: dict[str, int] = {}
        implementation_reason_counts: dict[str, int] = {}
        release_reason_counts: dict[str, int] = {}
        caution_reason_counts: dict[str, int] = {}
        reason_counts: dict[str, int] = {}
        for reasons in frame["data_source_blocking_reasons"]:
            for reason in tuple(reasons):
                data_source_reason_counts[reason] = data_source_reason_counts.get(reason, 0) + 1
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        for reasons in frame["implementation_blocking_reasons"]:
            for reason in tuple(reasons):
                implementation_reason_counts[reason] = implementation_reason_counts.get(reason, 0) + 1
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        for reasons in frame["release_blocking_reasons"]:
            for reason in tuple(reasons):
                release_reason_counts[reason] = release_reason_counts.get(reason, 0) + 1
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        for reasons in frame["caution_reasons"]:
            for reason in tuple(reasons):
                caution_reason_counts[reason] = caution_reason_counts.get(reason, 0) + 1
        release_blocked_rows = int(
            sum(bool(tuple(reasons)) for reasons in frame["release_blocking_reasons"])
        )
        design_frame = frame.drop_duplicates(subset=["design"]).copy()
        design_ready = design_frame["data_source_ready"].map(_truthy_value)
        return {
            "method": "CS",
            "mediator": "nonbinary",
            "paper_rows": int(frame.shape[0]),
            "paper_default_replications": replications,
            "planned_draws": 0,
            "data_source_checked_designs": int(design_frame.shape[0]),
            "data_source_ready_designs": int(design_ready.sum()),
            "data_source_blocked_designs": int((~design_ready).sum()),
            "data_source_ready_rows": int(data_source_ready.sum()),
            "data_source_blocked_rows": int((~data_source_ready).sum()),
            "implementation_blocked_rows": int((~implementation_ready).sum()),
            "release_blocked_rows": release_blocked_rows,
            "row_preflight_ready_rows": int(row_preflight_ready.sum()),
            "row_release_ready_rows": int(row_release_ready.sum()),
            "cell_count_policy_rows": int(cell_count_available.sum()),
            "cell_count_policy_size_risk_rows": int(cell_count_size_risk.sum()),
            "target_precision_rows": _numeric_column_count(frame, "target_mc_standard_error"),
            "target_tolerance_below_error_band_rows": _truthy_column_count(
                frame,
                "target_tolerance_below_error_band",
            ),
            "target_budget_rows": _numeric_column_count(
                frame,
                "target_required_replications_for_tolerance",
            ),
            "target_shortfall_rows": _positive_numeric_column_count(
                frame,
                "target_replication_shortfall",
            ),
            "max_target_mc_standard_error": _max_numeric_column(
                frame,
                "target_mc_standard_error",
            ),
            "max_target_mc_error_band": _max_numeric_column(
                frame,
                "target_mc_error_band",
            ),
            "max_target_required_replications": _max_numeric_column(
                frame,
                "target_required_replications_for_tolerance",
            ),
            "max_target_replication_shortfall": _max_numeric_column(
                frame,
                "target_replication_shortfall",
            ),
            "min_target_median_independent_clusters_per_cell": _numeric_min_or_max(
                frame.loc[
                    cell_count_available,
                    "target_median_independent_clusters_per_cell",
                ],
                choose="min",
            ),
            "source_complete_case_rows": _unique_design_values(frame, value_column="data_source_complete_case_rows"),
            "source_complete_case_rows_by_source_key": _unique_source_key_values(
                frame,
                key_column="data_source_key",
                value_column="data_source_complete_case_rows",
            ),
            "source_mediator_level_counts": _unique_design_values(
                frame,
                value_column="source_mediator_level_count",
            ),
            "source_mediator_level_counts_by_source_key": _unique_source_key_values(
                frame,
                key_column="data_source_key",
                value_column="source_mediator_level_count",
            ),
            "source_clusters": _unique_design_values(
                frame,
                value_column="data_source_source_clusters",
            ),
            "source_clusters_by_source_key": _unique_source_key_values(
                frame,
                key_column="data_source_key",
                value_column="data_source_source_clusters",
            ),
            "data_source_blocking_reason_counts": data_source_reason_counts,
            "implementation_blocking_reason_counts": implementation_reason_counts,
            "release_blocking_reason_counts": release_reason_counts,
            "caution_reason_counts": caution_reason_counts,
            "blocking_reason_counts": reason_counts,
            "next_action": _nonbinary_readiness_next_action(reason_counts),
            "exit_criteria": (
                "nonbinary_empirical_mixture_cs_readiness_summary."
                "row_release_ready_rows == paper_rows"
            ),
        }

    def nonbinary_empirical_mixture_cs_blocker_frame(
        self,
        data_sources: dict[str, "BinaryEmpiricalMixtureBenchmarkDataSource"] | None = None,
        *,
        replications: int = 500,
        absolute_tolerance: float | None = 0.025,
        z_tolerance: float | None = 2.0,
        design: str | None = None,
        table: str | None = None,
        clusters: tuple[int | None, ...] | None = None,
        bins: tuple[int | None, ...] | None = None,
        t_values: tuple[float, ...] | None = None,
    ) -> pd.DataFrame:
        """Return row-level pre-run release blockers for the nonbinary CS paper rows."""

        readiness = self.nonbinary_empirical_mixture_cs_readiness_frame(
            data_sources,
            replications=replications,
            absolute_tolerance=absolute_tolerance,
            z_tolerance=z_tolerance,
            design=design,
            table=table,
            clusters=clusters,
            bins=bins,
            t_values=t_values,
        )
        if readiness.empty:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for row in readiness.to_dict("records"):
            implementation_reasons = tuple(row["implementation_blocking_reasons"] or ())
            data_source_reasons = tuple(row["data_source_blocking_reasons"] or ())
            release_reasons = tuple(row["release_blocking_reasons"] or ())
            caution_reasons = tuple(row.get("caution_reasons") or ())
            blocking_reasons = [
                *implementation_reasons,
                *data_source_reasons,
                *release_reasons,
            ]
            blocking_reasons = list(dict.fromkeys(blocking_reasons))
            if "nonbinary_cs_paper_matrix_not_run" not in blocking_reasons:
                blocking_reasons.append("nonbinary_cs_paper_matrix_not_run")
            blocked = bool(blocking_reasons)

            rows.append(
                {
                    **row,
                    "blocker_status": "blocked" if blocked else "ready",
                    "blocked": blocked,
                    "blocking_reasons": tuple(blocking_reasons),
                    "blocking_reason_count": len(blocking_reasons),
                    "caution_reasons": caution_reasons,
                    "caution_reason_count": len(caution_reasons),
                    "resolution_actions": tuple(
                        _nonbinary_readiness_resolution_action(reason)
                        for reason in blocking_reasons
                    ),
                    "caution_resolution_actions": tuple(
                        _nonbinary_readiness_resolution_action(reason)
                        for reason in caution_reasons
                    ),
                    "source_contract": (
                        "nonbinary_empirical_mixture_cs_readiness_surface"
                    ),
                    "exit_criteria": (
                        "nonbinary_empirical_mixture_cs_blocker_summary."
                        "blocked_rows == 0"
                    ),
                }
            )
        return _json_safe_export_frame(rows)

    def nonbinary_empirical_mixture_cs_blocker_summary(
        self,
        data_sources: dict[str, "BinaryEmpiricalMixtureBenchmarkDataSource"] | None = None,
        *,
        replications: int = 500,
        absolute_tolerance: float | None = 0.025,
        z_tolerance: float | None = 2.0,
        design: str | None = None,
        table: str | None = None,
        clusters: tuple[int | None, ...] | None = None,
        bins: tuple[int | None, ...] | None = None,
        t_values: tuple[float, ...] | None = None,
    ) -> dict[str, Any]:
        """Return compact release blocker counts for the nonbinary CS surface."""

        frame = self.nonbinary_empirical_mixture_cs_blocker_frame(
            data_sources,
            replications=replications,
            absolute_tolerance=absolute_tolerance,
            z_tolerance=z_tolerance,
            design=design,
            table=table,
            clusters=clusters,
            bins=bins,
            t_values=t_values,
        )
        if frame.empty:
            return {
                "method": "CS",
                "mediator": "nonbinary",
                "paper_rows": 0,
                "paper_default_replications": replications,
                "blocked_rows": 0,
                "ready_rows": 0,
                "row_preflight_ready_rows": 0,
                "row_release_ready_rows": 0,
                "data_source_blocked_rows": 0,
                "implementation_blocked_rows": 0,
                "cell_count_policy_size_risk_rows": 0,
                "target_shortfall_rows": 0,
                "max_target_required_replications": None,
                "max_target_replication_shortfall": None,
                "source_complete_case_rows_by_source_key": {},
                "source_mediator_level_counts_by_source_key": {},
                "source_clusters_by_source_key": {},
                "caution_reason_counts": {},
                "blocking_reason_counts": {},
                "next_action": "no_nonbinary_cs_rows_selected",
                "exit_criteria": (
                    "nonbinary_empirical_mixture_cs_blocker_summary."
                    "blocked_rows == 0"
                ),
            }

        blocked = frame["blocked"].map(_truthy_value)
        data_source_ready = frame["data_source_ready"].map(_truthy_value)
        implementation_ready = frame["implementation_ready"].map(_truthy_value)
        row_preflight_ready = frame["row_preflight_ready"].map(_truthy_value)
        row_release_ready = frame["row_release_ready"].map(_truthy_value)
        reason_counts: dict[str, int] = {}
        for reasons in frame["blocking_reasons"]:
            for reason in tuple(reasons):
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        caution_reason_counts: dict[str, int] = {}
        for reasons in frame["caution_reasons"]:
            for reason in tuple(reasons):
                caution_reason_counts[reason] = caution_reason_counts.get(reason, 0) + 1

        return {
            "method": "CS",
            "mediator": "nonbinary",
            "paper_rows": int(frame.shape[0]),
            "paper_default_replications": replications,
            "blocked_rows": int(blocked.sum()),
            "ready_rows": int((~blocked).sum()),
            "row_preflight_ready_rows": int(row_preflight_ready.sum()),
            "row_release_ready_rows": int(row_release_ready.sum()),
            "data_source_blocked_rows": int((~data_source_ready).sum()),
            "implementation_blocked_rows": int((~implementation_ready).sum()),
            "cell_count_policy_size_risk_rows": _truthy_column_count(
                frame,
                "cell_count_policy_size_risk",
            ),
            "target_shortfall_rows": _positive_numeric_column_count(
                frame,
                "target_replication_shortfall",
            ),
            "max_target_required_replications": _max_numeric_column(
                frame,
                "target_required_replications_for_tolerance",
            ),
            "max_target_replication_shortfall": _max_numeric_column(
                frame,
                "target_replication_shortfall",
            ),
            "source_complete_case_rows_by_source_key": _unique_source_key_values(
                frame,
                key_column="data_source_key",
                value_column="data_source_complete_case_rows",
            ),
            "source_mediator_level_counts_by_source_key": _unique_source_key_values(
                frame,
                key_column="data_source_key",
                value_column="source_mediator_level_count",
            ),
            "source_clusters_by_source_key": _unique_source_key_values(
                frame,
                key_column="data_source_key",
                value_column="data_source_source_clusters",
            ),
            "caution_reason_counts": caution_reason_counts,
            "blocking_reason_counts": reason_counts,
            "next_action": _nonbinary_readiness_next_action(reason_counts),
            "exit_criteria": (
                "nonbinary_empirical_mixture_cs_blocker_summary."
                "blocked_rows == 0"
            ),
        }

    def raise_for_nonbinary_empirical_mixture_cs_blockers(
        self,
        data_sources: dict[str, "BinaryEmpiricalMixtureBenchmarkDataSource"] | None = None,
        *,
        replications: int = 500,
        absolute_tolerance: float | None = 0.025,
        z_tolerance: float | None = 2.0,
        design: str | None = None,
        table: str | None = None,
        clusters: tuple[int | None, ...] | None = None,
        bins: tuple[int | None, ...] | None = None,
        t_values: tuple[float, ...] | None = None,
    ) -> None:
        """Raise if any nonbinary CS paper row is not release-ready."""

        summary = self.nonbinary_empirical_mixture_cs_blocker_summary(
            data_sources,
            replications=replications,
            absolute_tolerance=absolute_tolerance,
            z_tolerance=z_tolerance,
            design=design,
            table=table,
            clusters=clusters,
            bins=bins,
            t_values=t_values,
        )
        if summary["blocked_rows"] == 0:
            return
        raise AssertionError(
            "Nonbinary empirical-mixture CS readiness blocked "
            f"{summary['blocked_rows']} of {summary['paper_rows']} paper rows: "
            f"blocking_reason_counts={summary['blocking_reason_counts']}; "
            "source_complete_case_rows_by_source_key="
            f"{summary['source_complete_case_rows_by_source_key']}; "
            "source_mediator_level_counts_by_source_key="
            f"{summary['source_mediator_level_counts_by_source_key']}; "
            "source_clusters_by_source_key="
            f"{summary['source_clusters_by_source_key']}; "
            f"next_action={summary['next_action']}; "
            f"exit_criteria={summary['exit_criteria']}"
        )

    def nonbinary_empirical_mixture_cs_run_frame(
        self,
        run_result: "MonteCarloBenchmarkPlanRunResult",
        *,
        paper_default_replications: int = 500,
        absolute_tolerance: float | None = 0.025,
        z_tolerance: float | None = 2.0,
        design: str | None = None,
        table: str | None = None,
        clusters: tuple[int | None, ...] | None = None,
        bins: tuple[int | None, ...] | None = None,
        t_values: tuple[float, ...] | None = None,
    ) -> pd.DataFrame:
        """Return post-run coverage and release blockers for nonbinary CS paper rows."""

        if paper_default_replications <= 0:
            raise ValueError("paper_default_replications must be positive.")
        _validate_target_precision_tolerances(
            absolute_tolerance=absolute_tolerance,
            z_tolerance=z_tolerance,
        )

        expected_cells = self.benchmark_cells(
            method="CS",
            mediator="nonbinary",
            design=design,
            table=table,
            clusters=clusters,
            bins=bins,
            t_values=t_values,
        )
        if not expected_cells:
            return pd.DataFrame()

        acceptance_frame = run_result.acceptance_frame()
        diagnostics_by_key = _diagnostics_by_data_source_key(
            run_result.data_source_diagnostics
        )
        executed_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
        if not acceptance_frame.empty:
            for executed_row in acceptance_frame.to_dict("records"):
                if (
                    executed_row.get("status") == "executed"
                    and executed_row.get("mediator") == "nonbinary"
                    and executed_row.get("method") == "CS"
                ):
                    executed_by_key[
                        _paper_result_mapping_key(executed_row, method="CS")
                    ] = executed_row

        rows: list[dict[str, Any]] = []
        for cell in expected_cells:
            data_source_diagnostic = _diagnostic_for_benchmark_cell(
                diagnostics_by_key,
                cell,
            )
            executed_row = executed_by_key.get(_paper_result_cell_key(cell))
            status = "executed" if executed_row is not None else "not_run"
            executed = executed_row is not None
            planned_replications = (
                _coerce_optional_int(executed_row.get("planned_replications"))
                if executed_row is not None
                else None
            )
            if planned_replications is None and executed_row is not None:
                planned_replications = _coerce_optional_int(executed_row.get("replications"))
            if planned_replications is None:
                planned_replications = 0
            precision_replications = (
                planned_replications if executed else paper_default_replications
            )
            target_standard_error = _target_rejection_rate_standard_error(
                cell.target_rejection_rate,
                replications=precision_replications,
            )
            target_error_band = (
                None
                if z_tolerance is None
                else float(z_tolerance) * target_standard_error
            )
            target_required_replications = _required_replications_for_binomial_precision(
                probability=cell.target_rejection_rate,
                absolute_tolerance=absolute_tolerance,
                z_tolerance=z_tolerance,
                trials_per_replication=1,
            )
            target_replication_shortfall = _replication_shortfall(
                required_replications=target_required_replications,
                planned_replications=precision_replications,
            )
            cell_count = cell.cluster_cell_count
            cell_count_size_risk = (
                None if cell_count is None else cell_count.size_risk
            )
            passed = (
                bool(executed_row.get("passed"))
                if executed_row is not None
                else False
            )
            data_source_blocking_reasons = (
                ()
                if data_source_diagnostic is None
                else _nonbinary_run_data_source_blocking_reasons(
                    cell=cell,
                    diagnostic=data_source_diagnostic,
                )
            )

            blocking_reasons: list[str] = []
            if data_source_diagnostic is None:
                blocking_reasons.append("nonbinary_cs_data_source_diagnostic_missing")
            blocking_reasons.extend(data_source_blocking_reasons)
            if not executed:
                blocking_reasons.append("nonbinary_cs_paper_row_not_run")
            elif not passed:
                blocking_reasons.append("nonbinary_cs_executed_row_failed")
            if _is_positive_numeric_value(target_replication_shortfall):
                blocking_reasons.append("target_replication_shortfall")
            blocking_reasons = list(dict.fromkeys(blocking_reasons))
            caution_reasons = (
                ("cell_count_policy_size_risk",)
                if _truthy_value(cell_count_size_risk)
                else ()
            )

            row: dict[str, Any] = {
                **cell.to_dict(),
                "status": status,
                "executed": executed,
                "passed": passed if executed else None,
                "blocked": bool(blocking_reasons),
                "blocker_status": "blocked" if blocking_reasons else "ready",
                "blocking_reasons": tuple(blocking_reasons),
                "blocking_reason_count": len(blocking_reasons),
                "caution_reasons": caution_reasons,
                "caution_reason_count": len(caution_reasons),
                "resolution_actions": tuple(
                    _nonbinary_run_resolution_action(reason)
                    for reason in blocking_reasons
                ),
                "caution_resolution_actions": tuple(
                    _nonbinary_run_resolution_action(reason)
                    for reason in caution_reasons
                ),
                "planned_replications": planned_replications,
                "planned_draws": planned_replications,
                "paper_default_replications": paper_default_replications,
                "target_precision_replications": precision_replications,
                "target_mc_standard_error": target_standard_error,
                "z_tolerance": z_tolerance,
                "target_mc_error_band": target_error_band,
                "absolute_tolerance": absolute_tolerance,
                "target_tolerance_below_error_band": bool(
                    absolute_tolerance is not None
                    and target_error_band is not None
                    and target_error_band > float(absolute_tolerance)
                ),
                "target_required_replications_for_tolerance": target_required_replications,
                "target_replication_shortfall": target_replication_shortfall,
                "cell_count_policy_available": cell_count is not None,
                "target_median_independent_clusters_per_cell": (
                    None
                    if cell_count is None
                    else cell_count.median_independent_clusters_per_cell
                ),
                "cell_count_size_risk_threshold": (
                    None if cell_count is None else cell_count.size_risk_threshold
                ),
                "cell_count_policy_size_risk": cell_count_size_risk,
                "recommended_by_cell_count": (
                    None
                    if cell_count_size_risk is None
                    else not cell_count_size_risk
                ),
                "bin_policy": (
                    None
                    if cell_count_size_risk is None
                    else (
                        "below_cell_count_heuristic"
                        if cell_count_size_risk
                        else "within_cell_count_heuristic"
                    )
                ),
                "paper_rule": (
                    None
                    if cell_count is None
                    else "at least 15 independent observations per cell"
                ),
                "source_contract": "nonbinary_empirical_mixture_cs_post_run_surface",
                "exit_criteria": (
                    "nonbinary_empirical_mixture_cs_run_summary.blocked_rows == 0"
                ),
            }
            if data_source_diagnostic is not None:
                row.update(
                    {
                        "data_source_key": (
                            data_source_diagnostic.data_source_key
                            or data_source_diagnostic.design
                        ),
                        "data_source_complete_case_rows": (
                            data_source_diagnostic.complete_case_rows
                        ),
                        "source_mediator_level_count": len(
                            data_source_diagnostic.mediator_levels
                        ),
                        "source_mediator_levels": tuple(
                            _normalize_diagnostic_value(value)
                            for value in data_source_diagnostic.mediator_levels
                        ),
                        "data_source_source_clusters": (
                            data_source_diagnostic.source_clusters
                        ),
                        "data_source_blocking_reasons": (
                            data_source_blocking_reasons
                        ),
                    }
                )
            if executed_row is not None:
                for column in (
                    "observed_rejection_rate",
                    "rejection_rate_absolute_error",
                    "z_score",
                    "failure_reasons",
                    "observed_source_mixture_share",
                    "source_mixture_absolute_error",
                    "source_mixture_effective_trials",
                    "source_mixture_standard_error",
                    "source_mixture_z_score",
                    "n_obs_used",
                    "min_n_obs_used",
                    "max_n_obs_used",
                    "min_n_clusters_used",
                    "max_n_clusters_used",
                ):
                    if column in executed_row:
                        row[column] = executed_row[column]
            rows.append(row)

        return _json_safe_export_frame(rows)

    def nonbinary_empirical_mixture_cs_run_summary(
        self,
        run_result: "MonteCarloBenchmarkPlanRunResult",
        *,
        paper_default_replications: int = 500,
        absolute_tolerance: float | None = 0.025,
        z_tolerance: float | None = 2.0,
        design: str | None = None,
        table: str | None = None,
        clusters: tuple[int | None, ...] | None = None,
        bins: tuple[int | None, ...] | None = None,
        t_values: tuple[float, ...] | None = None,
    ) -> dict[str, Any]:
        """Return compact post-run release blockers for nonbinary CS paper rows."""

        frame = self.nonbinary_empirical_mixture_cs_run_frame(
            run_result,
            paper_default_replications=paper_default_replications,
            absolute_tolerance=absolute_tolerance,
            z_tolerance=z_tolerance,
            design=design,
            table=table,
            clusters=clusters,
            bins=bins,
            t_values=t_values,
        )
        if frame.empty:
            return {
                "method": "CS",
                "mediator": "nonbinary",
                "paper_rows": 0,
                "paper_default_replications": paper_default_replications,
                "executed_rows": 0,
                "not_run_rows": 0,
                "passed_executed_rows": 0,
                "failed_executed_rows": 0,
                "blocked_rows": 0,
                "ready_rows": 0,
                "paper_coverage_complete": True,
                "paper_coverage_shortfall_rows": 0,
                "cell_count_policy_size_risk_rows": 0,
                "target_shortfall_rows": 0,
                "max_target_required_replications": None,
                "max_target_replication_shortfall": None,
                "source_complete_case_rows_by_source_key": {},
                "source_mediator_level_counts_by_source_key": {},
                "source_clusters_by_source_key": {},
                "caution_reason_counts": {},
                "blocking_reason_counts": {},
                "next_action": "no_nonbinary_cs_rows_selected",
                "exit_criteria": (
                    "nonbinary_empirical_mixture_cs_run_summary.blocked_rows == 0"
                ),
            }

        executed = frame["executed"].map(_truthy_value)
        passed = frame["passed"].map(_truthy_value)
        blocked = frame["blocked"].map(_truthy_value)
        reason_counts: dict[str, int] = {}
        for reasons in frame["blocking_reasons"]:
            for reason in tuple(reasons):
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        caution_reason_counts: dict[str, int] = {}
        for reasons in frame["caution_reasons"]:
            for reason in tuple(reasons):
                caution_reason_counts[reason] = caution_reason_counts.get(reason, 0) + 1

        not_run_rows = int((~executed).sum())
        return {
            "method": "CS",
            "mediator": "nonbinary",
            "paper_rows": int(frame.shape[0]),
            "paper_default_replications": paper_default_replications,
            "executed_rows": int(executed.sum()),
            "not_run_rows": not_run_rows,
            "passed_executed_rows": int((executed & passed).sum()),
            "failed_executed_rows": int((executed & ~passed).sum()),
            "blocked_rows": int(blocked.sum()),
            "ready_rows": int((~blocked).sum()),
            "paper_coverage_complete": not_run_rows == 0,
            "paper_coverage_shortfall_rows": not_run_rows,
            "cell_count_policy_size_risk_rows": _truthy_column_count(
                frame,
                "cell_count_policy_size_risk",
            ),
            "target_shortfall_rows": _positive_numeric_column_count(
                frame,
                "target_replication_shortfall",
            ),
            "max_target_required_replications": _max_numeric_column(
                frame,
                "target_required_replications_for_tolerance",
            ),
            "max_target_replication_shortfall": _max_numeric_column(
                frame,
                "target_replication_shortfall",
            ),
            "source_complete_case_rows_by_source_key": _unique_source_key_values(
                frame,
                key_column="data_source_key",
                value_column="data_source_complete_case_rows",
            ),
            "source_mediator_level_counts_by_source_key": _unique_source_key_values(
                frame,
                key_column="data_source_key",
                value_column="source_mediator_level_count",
            ),
            "source_clusters_by_source_key": _unique_source_key_values(
                frame,
                key_column="data_source_key",
                value_column="data_source_source_clusters",
            ),
            "caution_reason_counts": caution_reason_counts,
            "blocking_reason_counts": reason_counts,
            "next_action": _nonbinary_run_next_action(reason_counts),
            "exit_criteria": (
                "nonbinary_empirical_mixture_cs_run_summary.blocked_rows == 0"
            ),
        }

    def raise_for_nonbinary_empirical_mixture_cs_run_blockers(
        self,
        run_result: "MonteCarloBenchmarkPlanRunResult",
        *,
        paper_default_replications: int = 500,
        absolute_tolerance: float | None = 0.025,
        z_tolerance: float | None = 2.0,
        design: str | None = None,
        table: str | None = None,
        clusters: tuple[int | None, ...] | None = None,
        bins: tuple[int | None, ...] | None = None,
        t_values: tuple[float, ...] | None = None,
    ) -> None:
        """Raise if an executed rerun has not cleared nonbinary CS paper rows."""

        summary = self.nonbinary_empirical_mixture_cs_run_summary(
            run_result,
            paper_default_replications=paper_default_replications,
            absolute_tolerance=absolute_tolerance,
            z_tolerance=z_tolerance,
            design=design,
            table=table,
            clusters=clusters,
            bins=bins,
            t_values=t_values,
        )
        if summary["blocked_rows"] == 0:
            return
        raise AssertionError(
            "Nonbinary empirical-mixture CS post-run gate is blocked: "
            f"{summary['blocked_rows']} of {summary['paper_rows']} paper rows "
            "are not release-ready; "
            f"executed_rows={summary['executed_rows']}; "
            f"not_run_rows={summary['not_run_rows']}; "
            f"blocking_reason_counts={summary['blocking_reason_counts']}; "
            "source_complete_case_rows_by_source_key="
            f"{summary['source_complete_case_rows_by_source_key']}; "
            "source_mediator_level_counts_by_source_key="
            f"{summary['source_mediator_level_counts_by_source_key']}; "
            "source_clusters_by_source_key="
            f"{summary['source_clusters_by_source_key']}; "
            f"next_action={summary['next_action']}; "
            f"exit_criteria={summary['exit_criteria']}"
        )

    def method_acceptance_frame(
        self,
        *,
        method: str,
        nominal_alpha: float = 0.05,
        tolerance: float = 0.025,
    ) -> pd.DataFrame:
        """Return row-level paper acceptance semantics for one reported method."""

        if not 0 < nominal_alpha < 1:
            raise ValueError("nominal_alpha must be strictly between 0 and 1.")
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative.")
        if not any(method in row.rejection_rates for row in self.result_rows):
            raise KeyError(f"No Monte Carlo result rows expose method {method!r}.")

        cell_count_policy_applies = _method_uses_discretized_outcome(method)
        rows: list[dict[str, Any]] = []
        for row in self.result_rows:
            method_supported = method in row.rejection_rates
            rejection_rate = row.rejection_rates.get(method)
            excess_rejection_rate = (
                None
                if rejection_rate is None or not row.is_null_size_row
                else rejection_rate - nominal_alpha
            )
            size_distortion = bool(
                excess_rejection_rate is not None
                and excess_rejection_rate > tolerance
            )
            cell_count = (
                self._cluster_cell_count_for_result_row(row)
                if cell_count_policy_applies
                else None
            )
            cell_count_size_risk = (
                None if cell_count is None else cell_count.size_risk
            )
            needs_attention = bool(
                method_supported
                and row.is_null_size_row
                and (
                    size_distortion
                    or (cell_count_size_risk is not None and cell_count_size_risk)
                )
            )
            rows.append(
                {
                    "table": row.table,
                    "panel": row.panel,
                    "design": row.design,
                    "mediator": row.mediator,
                    "clusters": row.clusters,
                    "bins": row.bins,
                    "t": row.t,
                    "method": method,
                    "bar_nu_lb": row.bar_nu_lb,
                    "acceptance_role": (
                        "size_control" if row.is_null_size_row else "power"
                    ),
                    "size_row": row.is_null_size_row,
                    "method_supported": method_supported,
                    "paper_method_status": (
                        "reported" if method_supported else "not_reported"
                    ),
                    "target_rejection_rate": rejection_rate,
                    "nominal_alpha": nominal_alpha,
                    "tolerance": tolerance,
                    "excess_rejection_rate": excess_rejection_rate,
                    "size_distortion": size_distortion,
                    "cell_count_policy_available": cell_count is not None,
                    "target_median_independent_clusters_per_cell": (
                        None
                        if cell_count is None
                        else cell_count.median_independent_clusters_per_cell
                    ),
                    "cell_count_size_risk_threshold": (
                        None if cell_count is None else cell_count.size_risk_threshold
                    ),
                    "cell_count_size_risk": cell_count_size_risk,
                    "recommended_by_cell_count": (
                        None
                        if cell_count_size_risk is None
                        else not cell_count_size_risk
                    ),
                    "bin_policy": (
                        None
                        if cell_count_size_risk is None
                        else (
                            "below_cell_count_heuristic"
                            if cell_count_size_risk
                            else "within_cell_count_heuristic"
                        )
                    ),
                    "paper_rule": (
                        None
                        if cell_count is None
                        else "at least 15 independent observations per cell"
                    ),
                    "needs_attention": needs_attention,
                }
            )
        return _json_safe_export_frame(rows)

    def method_acceptance_summary(
        self,
        *,
        method: str,
        nominal_alpha: float = 0.05,
        tolerance: float = 0.025,
    ) -> dict[str, Any]:
        """Return compact counts from the row-level paper acceptance frame."""

        frame = self.method_acceptance_frame(
            method=method,
            nominal_alpha=nominal_alpha,
            tolerance=tolerance,
        )
        supported = frame["method_supported"].map(_truthy_value)
        size_rows = frame["size_row"].map(_truthy_value)
        power_rows = ~size_rows
        cell_count_available = frame["cell_count_policy_available"].map(_truthy_value)
        cell_count_size_risk = frame["cell_count_size_risk"].map(_truthy_value)

        supported_frame = frame.loc[supported]
        supported_size_frame = frame.loc[supported & size_rows]
        supported_power_frame = frame.loc[supported & power_rows]
        supported_cell_count_size_risk = supported & cell_count_size_risk
        return {
            "method": method,
            "total_result_rows": int(frame.shape[0]),
            "supported_result_rows": int(supported.sum()),
            "unsupported_result_rows": int((~supported).sum()),
            "size_rows": int((supported & size_rows).sum()),
            "power_rows": int((supported & power_rows).sum()),
            "unsupported_size_rows": int((~supported & size_rows).sum()),
            "unsupported_power_rows": int((~supported & power_rows).sum()),
            "size_distortion_rows": _truthy_column_count(
                supported_size_frame,
                "size_distortion",
            ),
            "cell_count_policy_rows": int((supported & cell_count_available).sum()),
            "cell_count_size_risk_rows": int(
                supported_cell_count_size_risk.sum()
            ),
            "cell_count_size_risk_size_rows": int(
                (supported_cell_count_size_risk & size_rows).sum()
            ),
            "cell_count_size_risk_power_rows": int(
                (supported_cell_count_size_risk & power_rows).sum()
            ),
            "attention_size_rows": _truthy_column_count(
                supported_size_frame,
                "needs_attention",
            ),
            "max_size_rejection_rate": _max_numeric_column(
                supported_size_frame,
                "target_rejection_rate",
            ),
            "min_power_rejection_rate": _numeric_min_or_max(
                supported_power_frame["target_rejection_rate"],
                choose="min",
            )
            if "target_rejection_rate" in supported_power_frame
            else None,
            "max_power_rejection_rate": _max_numeric_column(
                supported_power_frame,
                "target_rejection_rate",
            ),
            "min_target_median_independent_clusters_per_cell": _numeric_min_or_max(
                supported_frame.loc[
                    supported_frame["cell_count_policy_available"].map(_truthy_value),
                    "target_median_independent_clusters_per_cell",
                ],
                choose="min",
            )
            if "target_median_independent_clusters_per_cell" in supported_frame
            else None,
        }

    def _cluster_cell_count_for_result_row(self, row: MonteCarloResultRow) -> ClusterCellCount | None:
        if row.clusters is None or row.bins is None:
            return None
        try:
            return self.cell_count(
                design=row.design,
                mediator=row.mediator,
                clusters=row.clusters,
                bins=row.bins,
                t=row.t,
            )
        except KeyError:
            return None

    def benchmark_simulation(
        self,
        simulation: MonteCarloSimulationResult,
        *,
        table: str,
        design: str,
        mediator: str,
        clusters: int | None,
        bins: int | None,
        t: float,
        method: str | None = None,
        absolute_tolerance: float = 0.025,
        z_tolerance: float = 2.0,
        cell_count_absolute_tolerance: float | None = None,
        source_mixture_absolute_tolerance: float | None = None,
    ) -> MonteCarloBenchmarkDiagnostic:
        """Compare a simulation result with one paper rejection-rate contract row."""

        _validate_benchmark_tolerances(
            absolute_tolerance=absolute_tolerance,
            z_tolerance=z_tolerance,
            cell_count_absolute_tolerance=cell_count_absolute_tolerance,
            source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
        )

        benchmark_method = simulation.method if method is None else method
        row = self.result_row(
            table=table,
            design=design,
            mediator=mediator,
            clusters=clusters,
            bins=bins,
            t=t,
        )
        if benchmark_method not in row.rejection_rates:
            raise KeyError(
                f"Monte Carlo row table={table!r}, design={design!r}, mediator={mediator!r}, "
                f"clusters={clusters!r}, bins={bins!r}, t={t!r} does not expose method "
                f"{benchmark_method!r}."
            )
        return MonteCarloBenchmarkDiagnostic(
            simulation=simulation,
            row=row,
            method=benchmark_method,
            absolute_tolerance=absolute_tolerance,
            z_tolerance=z_tolerance,
            cell_count_absolute_tolerance=cell_count_absolute_tolerance,
            source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
            cluster_cell_count=self._cluster_cell_count_for_result_row(row),
        )

    def binary_empirical_mixture_design(
        self,
        *,
        name: str,
        df: pd.DataFrame,
        d: str,
        m: str,
        y: str,
        table: str,
        design: str,
        mediator: str,
        clusters: int | None,
        bins: int | None,
        t: float,
        seed: int,
        cluster: str | None = None,
        method: str = "CS",
        replications: int = 500,
        bootstrap_replications: int = 500,
        alpha: float = 0.05,
        replication_start: int = 0,
        seed_replications: int | None = None,
    ) -> BinaryEmpiricalMixtureMonteCarloDesign:
        """Build the paper's default empirical-mixture simulation design for one row."""

        row = self.result_row(
            table=table,
            design=design,
            mediator=mediator,
            clusters=clusters,
            bins=bins,
            t=t,
        )
        if mediator != "binary":
            raise NotImplementedError("Binary empirical-mixture simulations require mediator='binary'.")
        if method not in row.rejection_rates:
            raise KeyError(
                f"Monte Carlo row table={table!r}, design={design!r}, mediator={mediator!r}, "
                f"clusters={clusters!r}, bins={bins!r}, t={t!r} does not expose method {method!r}."
            )
        paper_contract = {
            "table": table,
            "design": design,
            "mediator": mediator,
            "clusters": clusters,
            "bins": bins,
            "t": float(t),
            "bar_nu_lb": row.bar_nu_lb,
            "target_method": method,
            "target_rejection_rate": row.rejection_rates[method],
        }
        arm_assignment = "iid_bernoulli" if clusters is None else "fixed_observed_arms"
        paper_contract["arm_assignment"] = arm_assignment
        return BinaryEmpiricalMixtureMonteCarloDesign.from_observed_data(
            name=name,
            df=df,
            d=d,
            m=m,
            y=y,
            replications=replications,
            seed=seed,
            t=t,
            cluster=cluster if clusters is not None else None,
            cluster_count=clusters,
            num_y_bins=bins,
            bootstrap_replications=bootstrap_replications,
            alpha=alpha,
            replication_start=replication_start,
            seed_replications=seed_replications,
            arm_assignment=arm_assignment,
            paper_contract_dict=paper_contract,
        )

    def nonbinary_empirical_mixture_design(
        self,
        *,
        name: str,
        df: pd.DataFrame,
        d: str,
        m: str,
        y: str,
        table: str,
        design: str,
        mediator: str,
        clusters: int | None,
        bins: int | None,
        t: float,
        seed: int,
        cluster: str | None = None,
        replications: int = 500,
        bootstrap_replications: int = 500,
        alpha: float = 0.05,
        method: str = "CS",
        replication_start: int = 0,
        seed_replications: int | None = None,
    ) -> BinaryEmpiricalMixtureMonteCarloDesign:
        """Create an ordered nonbinary empirical-mixture design for paper rows."""

        if method not in {"CS", "ARP", "FSSTdd", "FSSTndd"}:
            raise NotImplementedError(
                "Nonbinary empirical-mixture designs support CS, ARP, FSSTdd, and FSSTndd."
            )
        row = self.result_row(
            table=table,
            design=design,
            mediator=mediator,
            clusters=clusters,
            bins=bins,
            t=t,
        )
        if mediator != "nonbinary":
            raise NotImplementedError("Nonbinary empirical-mixture simulations require mediator='nonbinary'.")
        if method not in row.rejection_rates:
            raise KeyError(
                f"Monte Carlo row table={table!r}, design={design!r}, mediator={mediator!r}, "
                f"clusters={clusters!r}, bins={bins!r}, t={t!r} does not expose method {method!r}."
            )
        paper_contract = {
            "table": table,
            "design": design,
            "mediator": mediator,
            "clusters": clusters,
            "bins": bins,
            "t": float(t),
            "bar_nu_lb": row.bar_nu_lb,
            "target_method": method,
            "target_rejection_rate": row.rejection_rates[method],
            "arm_assignment": "fixed_observed_arms" if clusters is not None else "iid_bernoulli",
        }
        return BinaryEmpiricalMixtureMonteCarloDesign.from_observed_data(
            name=name,
            df=df,
            d=d,
            m=m,
            y=y,
            replications=replications,
            seed=seed,
            t=t,
            cluster=cluster if clusters is not None else None,
            cluster_count=clusters,
            num_y_bins=bins,
            bootstrap_replications=bootstrap_replications,
            alpha=alpha,
            replication_start=replication_start,
            seed_replications=seed_replications,
            arm_assignment=paper_contract["arm_assignment"],
            paper_contract_dict=paper_contract,
            mediator_kind="nonbinary",
        )

    def nonbinary_empirical_mixture_cs_design(
        self,
        *,
        name: str,
        df: pd.DataFrame,
        d: str,
        m: str,
        y: str,
        table: str,
        design: str,
        mediator: str,
        clusters: int | None,
        bins: int | None,
        t: float,
        seed: int,
        cluster: str | None = None,
        replications: int = 500,
        alpha: float = 0.05,
        method: str = "CS",
    ) -> BinaryEmpiricalMixtureMonteCarloDesign:
        """Create an ordered nonbinary empirical-mixture CS design for paper rows."""

        if method != "CS":
            raise NotImplementedError("Nonbinary empirical-mixture CS designs require method='CS'.")
        return self.nonbinary_empirical_mixture_design(
            name=name,
            df=df,
            d=d,
            m=m,
            y=y,
            table=table,
            design=design,
            mediator=mediator,
            clusters=clusters,
            bins=bins,
            t=t,
            seed=seed,
            cluster=cluster,
            replications=replications,
            alpha=alpha,
            method=method,
        )

    def binary_empirical_mixture_benchmark_cells(
        self,
        *,
        method: str = "CS",
        design: str | None = None,
        table: str | None = None,
        clusters: tuple[int | None, ...] | None = None,
        bins: tuple[int | None, ...] | None = None,
        t_values: tuple[float, ...] | None = None,
    ) -> tuple[MonteCarloBenchmarkCell, ...]:
        """Return the paper rows currently executable by the binary empirical-mixture CS runner."""

        if method != "CS":
            raise NotImplementedError("Binary empirical-mixture benchmark cells currently support method='CS'.")
        return self.benchmark_cells(
            method=method,
            mediator="binary",
            design=design,
            table=table,
            clusters=clusters,
            bins=bins,
            t_values=t_values,
        )

    def empirical_mixture_benchmark_plan(
        self,
        *,
        method: str = "CS",
        replications: int = 500,
        mediator: str | None = None,
        design: str | None = None,
        table: str | None = None,
        clusters: tuple[int | None, ...] | None = None,
        bins: tuple[int | None, ...] | None = None,
        t_values: tuple[float, ...] | None = None,
    ) -> MonteCarloBenchmarkPlan:
        """Plan the executable empirical-mixture paper rows for one reported method."""

        if method not in {"CS", "ARP", "FSSTdd", "FSSTndd", "K"}:
            raise KeyError(f"No paper empirical-mixture benchmark runner for method {method!r}.")
        if replications <= 0:
            raise ValueError("replications must be positive.")

        executable_cells: list[MonteCarloBenchmarkCell] = []
        for paper_row_index, row in enumerate(self.result_rows):
            if mediator is not None and row.mediator != mediator:
                continue
            if not _row_matches_benchmark_filters(
                row,
                design=design,
                table=table,
                clusters=clusters,
                bins=bins,
                t_values=t_values,
            ):
                continue
            if method not in row.rejection_rates:
                continue
            executable_cells.append(
                MonteCarloBenchmarkCell.from_result_row(
                    row,
                    method=method,
                    cluster_cell_count=self._cluster_cell_count_for_result_row(row),
                    paper_row_index=paper_row_index,
                    benchmark_row_index=len(executable_cells),
                )
            )

        return MonteCarloBenchmarkPlan(
            method=method,
            replications=replications,
            executable_cells=tuple(executable_cells),
            blocked_rows=(),
            paper_result_rows=len(executable_cells),
        )

    def empirical_mixture_cs_benchmark_plan(
        self,
        *,
        method: str = "CS",
        replications: int = 500,
        mediator: str | None = None,
        design: str | None = None,
        table: str | None = None,
        clusters: tuple[int | None, ...] | None = None,
        bins: tuple[int | None, ...] | None = None,
        t_values: tuple[float, ...] | None = None,
    ) -> MonteCarloBenchmarkPlan:
        """Plan the executable CS empirical-mixture slice of the paper matrix."""

        if method != "CS":
            raise NotImplementedError("Empirical-mixture benchmark plans currently support method='CS'.")
        if replications <= 0:
            raise ValueError("replications must be positive.")

        executable_cells: list[MonteCarloBenchmarkCell] = []
        blocked_rows: list[MonteCarloBlockedBenchmarkRow] = []
        for paper_row_index, row in enumerate(self.result_rows):
            if mediator is not None and row.mediator != mediator:
                continue
            if not _row_matches_benchmark_filters(
                row,
                design=design,
                table=table,
                clusters=clusters,
                bins=bins,
                t_values=t_values,
            ):
                continue
            if method not in row.rejection_rates:
                blocked_rows.append(
                    MonteCarloBlockedBenchmarkRow(
                        row=row,
                        method=method,
                        reason="method_not_reported",
                    )
                )
                continue
            executable_cells.append(
                MonteCarloBenchmarkCell.from_result_row(
                    row,
                    method=method,
                    cluster_cell_count=self._cluster_cell_count_for_result_row(row),
                    paper_row_index=paper_row_index,
                    benchmark_row_index=len(executable_cells),
                )
            )

        return MonteCarloBenchmarkPlan(
            method=method,
            replications=replications,
            executable_cells=tuple(executable_cells),
            blocked_rows=tuple(blocked_rows),
            paper_result_rows=len(self.result_rows),
        )

    def binary_empirical_mixture_benchmark_plan(
        self,
        *,
        method: str = "CS",
        replications: int = 500,
        design: str | None = None,
        table: str | None = None,
        clusters: tuple[int | None, ...] | None = None,
        bins: tuple[int | None, ...] | None = None,
        t_values: tuple[float, ...] | None = None,
    ) -> MonteCarloBenchmarkPlan:
        """Plan the executable binary empirical-mixture slice of the paper matrix."""

        if method != "CS":
            raise NotImplementedError("Binary empirical-mixture benchmark plans currently support method='CS'.")
        if replications <= 0:
            raise ValueError("replications must be positive.")

        executable_cells: list[MonteCarloBenchmarkCell] = []
        blocked_rows: list[MonteCarloBlockedBenchmarkRow] = []
        for paper_row_index, row in enumerate(self.result_rows):
            if not _row_matches_benchmark_filters(
                row,
                design=design,
                table=table,
                clusters=clusters,
                bins=bins,
                t_values=t_values,
            ):
                continue
            if row.mediator != "binary":
                blocked_rows.append(
                    MonteCarloBlockedBenchmarkRow(
                        row=row,
                        method=method,
                        reason="nonbinary_cs_benchmark_matrix_not_integrated",
                    )
                )
                continue
            if method not in row.rejection_rates:
                blocked_rows.append(
                    MonteCarloBlockedBenchmarkRow(
                        row=row,
                        method=method,
                        reason="method_not_reported",
                    )
                )
                continue
            executable_cells.append(
                MonteCarloBenchmarkCell.from_result_row(
                    row,
                    method=method,
                    cluster_cell_count=self._cluster_cell_count_for_result_row(row),
                    paper_row_index=paper_row_index,
                    benchmark_row_index=len(executable_cells),
                )
            )

        return MonteCarloBenchmarkPlan(
            method=method,
            replications=replications,
            executable_cells=tuple(executable_cells),
            blocked_rows=tuple(blocked_rows),
            paper_result_rows=len(self.result_rows),
        )


def run_binary_cs_monte_carlo(design: BinaryCSMonteCarloDesign) -> MonteCarloSimulationResult:
    """Run repeated binary-mediator CS sharp-null simulations for one design.

    Generates *design.replications* synthetic datasets from a binary-mediator
    DGP, applies the CS sharp-null test to each, and collects rejection
    indicators and diagnostics.

    Parameters
    ----------
    design : BinaryCSMonteCarloDesign
        Design specification including sample size, effect, bins, clusters,
        replications, alpha, and seed.

    Returns
    -------
    MonteCarloSimulationResult
        Result containing per-draw rejection indicators and diagnostics.
    """

    seed_rng = np.random.default_rng(design.seed)
    draw_seeds = seed_rng.integers(
        low=0,
        high=np.iinfo(np.uint32).max,
        size=design.replications,
        dtype=np.uint32,
    )

    draws: list[MonteCarloDrawResult] = []
    for replication, draw_seed in enumerate(draw_seeds.tolist(), start=1):
        draw_df = _simulate_binary_cs_draw(design=design, seed=int(draw_seed))
        cluster_column = "cluster" if design.cluster_count is not None else None
        sharp_result = test_sharp_null(
            df=draw_df,
            d="treated",
            m="mediator",
            y="outcome",
            method="CS",
            num_y_bins=design.num_y_bins,
            alpha=design.alpha,
            cluster=cluster_column,
        )
        diagnostics = sharp_result.diagnostics
        count_summary = _monte_carlo_full_support_count_summary(
            draw_df=draw_df,
            num_y_bins=design.num_y_bins,
            cluster_column=cluster_column,
        )
        cluster_diagnostics = _draw_cluster_diagnostics(draw_df)
        arm_diagnostics = _draw_arm_diagnostics(draw_df)
        draws.append(
            MonteCarloDrawResult(
                replication=replication,
                seed=int(draw_seed),
                reject=sharp_result.reject,
                test_stat=sharp_result.test_stat,
                critical_value=sharp_result.critical_value,
                p_value=sharp_result.p_value,
                applied_num_y_bins=_optional_int(diagnostics["applied_num_y_bins"]),
                n_obs_used=int(diagnostics["n_obs_used"]),
                control_observations=arm_diagnostics["control_observations"],
                treated_observations=arm_diagnostics["treated_observations"],
                min_cell_count=count_summary["min_cell_count"],
                min_cluster_count=count_summary["min_cluster_count"],
                median_cell_count=count_summary["median_cell_count"],
                median_independent_count_per_cell=count_summary["median_independent_count_per_cell"],
                size_risk=count_summary["size_risk"],
                empty_cell_count=count_summary["empty_cell_count"],
                empty_cluster_cell_count=count_summary["empty_cluster_cell_count"],
                small_cell_count=count_summary["small_cell_count"],
                small_cluster_cell_count=count_summary["small_cluster_cell_count"],
                size_risk_threshold=count_summary["size_risk_threshold"],
                empty_cells=count_summary["empty_cells"],
                small_cells=count_summary["small_cells"],
                empty_cluster_cells=count_summary["empty_cluster_cells"],
                small_cluster_cells=count_summary["small_cluster_cells"],
                n_clusters_used=cluster_diagnostics["n_clusters_used"],
                treated_clusters=cluster_diagnostics["treated_clusters"],
                control_clusters=cluster_diagnostics["control_clusters"],
                cluster_size=cluster_diagnostics["cluster_size"],
            )
        )

    return MonteCarloSimulationResult(design=design, draws=tuple(draws))


def run_binary_partial_density_cs_monte_carlo(
    design: BinaryPartialDensityMonteCarloDesign,
) -> MonteCarloSimulationResult:
    """Run CS sharp-null draws from a binary partial-density DGP.

    Parameters
    ----------
    design : BinaryPartialDensityMonteCarloDesign
        Design specification for the partial-density DGP.

    Returns
    -------
    MonteCarloSimulationResult
        Result containing per-draw rejection indicators and diagnostics.
    """

    seed_rng = np.random.default_rng(design.seed)
    draw_seeds = seed_rng.integers(
        low=0,
        high=np.iinfo(np.uint32).max,
        size=design.replications,
        dtype=np.uint32,
    )

    draws: list[MonteCarloDrawResult] = []
    for replication, draw_seed in enumerate(draw_seeds.tolist(), start=1):
        draw_df = _simulate_binary_partial_density_draw(design=design, seed=int(draw_seed))
        sharp_result = test_sharp_null(
            df=draw_df,
            d="treated",
            m="mediator",
            y="outcome",
            method="CS",
            num_y_bins=design.num_y_bins,
            alpha=design.alpha,
        )
        diagnostics = sharp_result.diagnostics
        count_summary = _monte_carlo_full_support_count_summary(
            draw_df=draw_df,
            num_y_bins=design.num_y_bins,
            cluster_column=None,
        )
        arm_diagnostics = _draw_arm_diagnostics(draw_df)
        draws.append(
            MonteCarloDrawResult(
                replication=replication,
                seed=int(draw_seed),
                reject=sharp_result.reject,
                test_stat=sharp_result.test_stat,
                critical_value=sharp_result.critical_value,
                p_value=sharp_result.p_value,
                applied_num_y_bins=_optional_int(diagnostics["applied_num_y_bins"]),
                n_obs_used=int(diagnostics["n_obs_used"]),
                control_observations=arm_diagnostics["control_observations"],
                treated_observations=arm_diagnostics["treated_observations"],
                min_cell_count=count_summary["min_cell_count"],
                min_cluster_count=count_summary["min_cluster_count"],
                median_cell_count=count_summary["median_cell_count"],
                median_independent_count_per_cell=count_summary["median_independent_count_per_cell"],
                size_risk=count_summary["size_risk"],
                empty_cell_count=count_summary["empty_cell_count"],
                empty_cluster_cell_count=count_summary["empty_cluster_cell_count"],
                small_cell_count=count_summary["small_cell_count"],
                small_cluster_cell_count=count_summary["small_cluster_cell_count"],
                size_risk_threshold=count_summary["size_risk_threshold"],
                empty_cells=count_summary["empty_cells"],
                small_cells=count_summary["small_cells"],
                empty_cluster_cells=count_summary["empty_cluster_cells"],
                small_cluster_cells=count_summary["small_cluster_cells"],
            )
        )

    return MonteCarloSimulationResult(design=design, draws=tuple(draws))


def run_binary_empirical_mixture_monte_carlo(
    design: BinaryEmpiricalMixtureMonteCarloDesign,
    *,
    method: str | None = None,
) -> MonteCarloSimulationResult:
    """Run sharp-null draws from the paper's binary empirical-mixture DGP.

    Parameters
    ----------
    design : BinaryEmpiricalMixtureMonteCarloDesign
        Empirical-mixture design specification including source data,
        effect size, bins, cluster structure, and replication parameters.
    method : str or None
        Sharp-null testing method (``"CS"`` or ``"K"``).  Defaults to the
        method in the design's paper contract.

    Returns
    -------
    MonteCarloSimulationResult
        Result containing per-draw rejection indicators and diagnostics.
    """

    paper_contract = design.paper_contract_dict or {}
    resolved_method = str(method or paper_contract.get("target_method", "CS"))
    seed_rng = np.random.default_rng(design.seed)
    draw_seeds = seed_rng.integers(
        low=0,
        high=np.iinfo(np.uint32).max,
        size=design.seed_replications,
        dtype=np.uint32,
    )
    shard_draw_seeds = draw_seeds[
        design.replication_start: design.replication_start + design.replications
    ]

    draws: list[MonteCarloDrawResult] = []
    for replication, draw_seed in enumerate(
        shard_draw_seeds.tolist(),
        start=design.replication_start + 1,
    ):
        draw_df, source_counts = _simulate_binary_empirical_mixture_draw(
            design=design,
            seed=int(draw_seed),
        )
        cluster_column = "cluster" if design.cluster_count is not None else None
        sharp_result = test_sharp_null(
            df=draw_df,
            d="treated",
            m="mediator",
            y="outcome",
            method=resolved_method,
            num_y_bins=design.num_y_bins,
            alpha=design.alpha,
            cluster=cluster_column,
            bootstrap_replications=design.bootstrap_replications,
            random_state=int(draw_seed),
        )
        diagnostics = sharp_result.diagnostics
        count_summary = _monte_carlo_full_support_count_summary(
            draw_df=draw_df,
            num_y_bins=design.num_y_bins,
            cluster_column=cluster_column,
        )
        cluster_diagnostics = _draw_cluster_diagnostics(draw_df)
        arm_diagnostics = _draw_arm_diagnostics(draw_df)
        draws.append(
            MonteCarloDrawResult(
                replication=replication,
                seed=int(draw_seed),
                reject=sharp_result.reject,
                test_stat=sharp_result.test_stat,
                critical_value=sharp_result.critical_value,
                p_value=sharp_result.p_value,
                applied_num_y_bins=_optional_int(diagnostics["applied_num_y_bins"]),
                n_obs_used=int(diagnostics["n_obs_used"]),
                control_observations=arm_diagnostics["control_observations"],
                treated_observations=arm_diagnostics["treated_observations"],
                min_cell_count=count_summary["min_cell_count"],
                min_cluster_count=count_summary["min_cluster_count"],
                median_cell_count=count_summary["median_cell_count"],
                median_independent_count_per_cell=count_summary["median_independent_count_per_cell"],
                size_risk=count_summary["size_risk"],
                empty_cell_count=count_summary["empty_cell_count"],
                empty_cluster_cell_count=count_summary["empty_cluster_cell_count"],
                small_cell_count=count_summary["small_cell_count"],
                small_cluster_cell_count=count_summary["small_cluster_cell_count"],
                size_risk_threshold=count_summary["size_risk_threshold"],
                empty_cells=count_summary["empty_cells"],
                small_cells=count_summary["small_cells"],
                empty_cluster_cells=count_summary["empty_cluster_cells"],
                small_cluster_cells=count_summary["small_cluster_cells"],
                n_clusters_used=cluster_diagnostics["n_clusters_used"],
                treated_clusters=cluster_diagnostics["treated_clusters"],
                control_clusters=cluster_diagnostics["control_clusters"],
                cluster_size=cluster_diagnostics["cluster_size"],
                treated_source_treated_draws=source_counts["treated_source_treated_draws"],
                treated_source_control_draws=source_counts["treated_source_control_draws"],
                treated_source_treated_clusters=source_counts["treated_source_treated_clusters"],
                treated_source_control_clusters=source_counts["treated_source_control_clusters"],
            )
        )

    return MonteCarloSimulationResult(design=design, draws=tuple(draws))


def run_binary_empirical_mixture_cs_monte_carlo(
    design: BinaryEmpiricalMixtureMonteCarloDesign,
) -> MonteCarloSimulationResult:
    """Run CS sharp-null draws from the paper's empirical-mixture DGP.

    Convenience wrapper that calls :func:`run_binary_empirical_mixture_monte_carlo`
    with ``method="CS"``.

    Parameters
    ----------
    design : BinaryEmpiricalMixtureMonteCarloDesign
        Empirical-mixture design specification.

    Returns
    -------
    MonteCarloSimulationResult
        CS-method simulation result.
    """

    return run_binary_empirical_mixture_monte_carlo(design, method="CS")


def run_nonbinary_empirical_mixture_monte_carlo(
    design: BinaryEmpiricalMixtureMonteCarloDesign,
    *,
    method: str | None = None,
) -> MonteCarloSimulationResult:
    """Run ordered nonbinary sharp-null draws from the paper empirical-mixture DGP.

    Parameters
    ----------
    design : BinaryEmpiricalMixtureMonteCarloDesign
        Design with a nonbinary paper contract.
    method : str or None
        Testing method.  Defaults to the contract's ``target_method``.

    Returns
    -------
    MonteCarloSimulationResult
        Result containing per-draw rejection indicators.

    Raises
    ------
    ValueError
        If the paper contract's mediator type is not ``"nonbinary"``.
    NotImplementedError
        If the K method is requested for nonbinary mediators.
    """

    paper_contract = design.paper_contract_dict or {}
    if paper_contract.get("mediator") != "nonbinary":
        raise ValueError("Nonbinary empirical-mixture runner requires a nonbinary paper contract.")
    resolved_method = str(method or paper_contract.get("target_method", "CS"))
    if resolved_method == "K":
        raise NotImplementedError("K empirical-mixture runner is release-scoped to binary mediators.")

    seed_rng = np.random.default_rng(design.seed)
    draw_seeds = seed_rng.integers(
        low=0,
        high=np.iinfo(np.uint32).max,
        size=design.seed_replications,
        dtype=np.uint32,
    )
    shard_draw_seeds = draw_seeds[
        design.replication_start: design.replication_start + design.replications
    ]

    draws: list[MonteCarloDrawResult] = []
    for replication, draw_seed in enumerate(
        shard_draw_seeds.tolist(),
        start=design.replication_start + 1,
    ):
        draw_df, source_counts = _simulate_binary_empirical_mixture_draw(
            design=design,
            seed=int(draw_seed),
        )
        cluster_column = "cluster" if design.cluster_count is not None else None
        sharp_result = test_sharp_null(
            df=draw_df,
            d="treated",
            m="mediator",
            y="outcome",
            method=resolved_method,
            num_y_bins=design.num_y_bins,
            alpha=design.alpha,
            cluster=cluster_column,
            bootstrap_replications=design.bootstrap_replications,
            random_state=int(draw_seed),
        )
        diagnostics = sharp_result.diagnostics
        count_summary = _monte_carlo_full_support_count_summary(
            draw_df=draw_df,
            num_y_bins=design.num_y_bins,
            cluster_column=cluster_column,
        )
        cluster_diagnostics = _draw_cluster_diagnostics(draw_df)
        arm_diagnostics = _draw_arm_diagnostics(draw_df)
        draws.append(
            MonteCarloDrawResult(
                replication=replication,
                seed=int(draw_seed),
                reject=sharp_result.reject,
                test_stat=sharp_result.test_stat,
                critical_value=sharp_result.critical_value,
                p_value=sharp_result.p_value,
                applied_num_y_bins=_optional_int(diagnostics["applied_num_y_bins"]),
                n_obs_used=int(diagnostics["n_obs_used"]),
                control_observations=arm_diagnostics["control_observations"],
                treated_observations=arm_diagnostics["treated_observations"],
                min_cell_count=count_summary["min_cell_count"],
                min_cluster_count=count_summary["min_cluster_count"],
                median_cell_count=count_summary["median_cell_count"],
                median_independent_count_per_cell=count_summary["median_independent_count_per_cell"],
                size_risk=count_summary["size_risk"],
                empty_cell_count=count_summary["empty_cell_count"],
                empty_cluster_cell_count=count_summary["empty_cluster_cell_count"],
                small_cell_count=count_summary["small_cell_count"],
                small_cluster_cell_count=count_summary["small_cluster_cell_count"],
                size_risk_threshold=count_summary["size_risk_threshold"],
                empty_cells=count_summary["empty_cells"],
                small_cells=count_summary["small_cells"],
                empty_cluster_cells=count_summary["empty_cluster_cells"],
                small_cluster_cells=count_summary["small_cluster_cells"],
                n_clusters_used=cluster_diagnostics["n_clusters_used"],
                treated_clusters=cluster_diagnostics["treated_clusters"],
                control_clusters=cluster_diagnostics["control_clusters"],
                cluster_size=cluster_diagnostics["cluster_size"],
                treated_source_treated_draws=source_counts["treated_source_treated_draws"],
                treated_source_control_draws=source_counts["treated_source_control_draws"],
                treated_source_treated_clusters=source_counts["treated_source_treated_clusters"],
                treated_source_control_clusters=source_counts["treated_source_control_clusters"],
            )
        )

    return MonteCarloSimulationResult(design=design, draws=tuple(draws))


def run_nonbinary_empirical_mixture_cs_monte_carlo(
    design: BinaryEmpiricalMixtureMonteCarloDesign,
) -> MonteCarloSimulationResult:
    """Run ordered nonbinary CS sharp-null draws from the paper empirical-mixture DGP.

    Convenience wrapper that calls :func:`run_nonbinary_empirical_mixture_monte_carlo`
    with ``method="CS"``.

    Parameters
    ----------
    design : BinaryEmpiricalMixtureMonteCarloDesign
        Design with a nonbinary paper contract.

    Returns
    -------
    MonteCarloSimulationResult
        CS-method nonbinary simulation result.
    """

    return run_nonbinary_empirical_mixture_monte_carlo(design, method="CS")


def run_binary_empirical_mixture_cs_benchmark_matrix(
    contracts: MonteCarloContracts,
    *,
    df: pd.DataFrame,
    d: str,
    m: str,
    y: str,
    cells: list[dict[str, Any] | MonteCarloBenchmarkCell] | tuple[dict[str, Any] | MonteCarloBenchmarkCell, ...],
    seed: int,
    cluster: str | None = None,
    replications: int = 500,
    alpha: float = 0.05,
    absolute_tolerance: float = 0.025,
    z_tolerance: float = 2.0,
    cell_count_absolute_tolerance: float | None = None,
    source_mixture_absolute_tolerance: float | None = None,
) -> MonteCarloBenchmarkMatrixResult:
    """Run and benchmark binary empirical-mixture CS simulations for multiple paper cells.

    Executes the full simulation matrix for a single paper design/data-source
    combination, compares rejection rates against paper targets, and produces
    per-cell pass/fail diagnostics.

    Parameters
    ----------
    contracts : MonteCarloContracts
        Parsed paper Monte Carlo table contracts.
    df : pd.DataFrame
        Empirical source data.
    d, m, y : str
        Treatment, mediator, and outcome column names.
    cells : list or tuple
        Paper Monte Carlo cell specifications to simulate.
    seed : int
        Master seed for the benchmark.
    cluster : str or None
        Cluster column name, or ``None`` for unclustered designs.
    replications : int
        Number of simulation draws per cell.
    alpha : float
        Nominal significance level.
    absolute_tolerance : float
        Acceptance tolerance for rate differences.
    z_tolerance : float
        Z-score tolerance for sampling-error acceptance.
    cell_count_absolute_tolerance : float or None
        Tolerance for cell-count comparisons.
    source_mixture_absolute_tolerance : float or None
        Tolerance for source-mixture proportion comparisons.

    Returns
    -------
    MonteCarloBenchmarkMatrixResult
        Result with per-cell diagnostics and overall acceptance status.

    Raises
    ------
    ValueError
        If *cells* is empty or contains mixed designs.
    """

    if not cells:
        raise ValueError("cells must contain at least one paper Monte Carlo row specification.")
    cell_specs = tuple(_benchmark_cell_to_spec(cell) for cell in cells)
    cell_designs = tuple(dict.fromkeys(str(cell_spec["design"]) for cell_spec in cell_specs))
    if len(cell_designs) > 1:
        raise ValueError(
            "run_binary_empirical_mixture_cs_benchmark_matrix received mixed paper designs "
            f"{cell_designs}; use run_binary_empirical_mixture_cs_benchmark_plan_by_source "
            "with one data source per paper design."
        )
    seed_rng = np.random.default_rng(seed)
    cell_seeds = seed_rng.integers(
        low=0,
        high=np.iinfo(np.uint32).max,
        size=len(cell_specs),
        dtype=np.uint32,
    )

    simulations: list[MonteCarloSimulationResult] = []
    diagnostics: list[MonteCarloBenchmarkDiagnostic] = []
    for cell_spec, cell_seed in zip(cell_specs, cell_seeds.tolist(), strict=True):
        table = cell_spec["table"]
        paper_design = cell_spec["design"]
        mediator = cell_spec["mediator"]
        clusters = cell_spec["clusters"]
        bins = cell_spec["bins"]
        t_value = cell_spec["t"]
        design = _build_empirical_mixture_cs_design(
            contracts,
            name=_benchmark_matrix_design_name(
                table=table,
                design=paper_design,
                mediator=mediator,
                clusters=clusters,
                bins=bins,
                t=t_value,
            ),
            df=df,
            d=d,
            m=m,
            y=y,
            table=table,
            design=paper_design,
            mediator=mediator,
            clusters=clusters,
            bins=bins,
            t=t_value,
            seed=int(cell_seed),
            cluster=cluster,
            replications=replications,
            alpha=alpha,
        )
        simulation = _run_empirical_mixture_cs_monte_carlo(design)
        diagnostic = contracts.benchmark_simulation(
            simulation,
            table=table,
            design=paper_design,
            mediator=mediator,
            clusters=clusters,
            bins=bins,
            t=t_value,
            absolute_tolerance=absolute_tolerance,
            z_tolerance=z_tolerance,
            cell_count_absolute_tolerance=cell_count_absolute_tolerance,
            source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
        )
        simulations.append(simulation)
        diagnostics.append(diagnostic)

    return MonteCarloBenchmarkMatrixResult(
        simulations=tuple(simulations),
        diagnostics=tuple(diagnostics),
    )


def run_empirical_mixture_cs_benchmark_matrix(
    contracts: MonteCarloContracts,
    *,
    df: pd.DataFrame,
    d: str,
    m: str,
    y: str,
    cells: list[dict[str, Any] | MonteCarloBenchmarkCell] | tuple[dict[str, Any] | MonteCarloBenchmarkCell, ...],
    seed: int,
    cluster: str | None = None,
    replications: int = 500,
    alpha: float = 0.05,
    absolute_tolerance: float = 0.025,
    z_tolerance: float = 2.0,
    cell_count_absolute_tolerance: float | None = None,
    source_mixture_absolute_tolerance: float | None = None,
) -> MonteCarloBenchmarkMatrixResult:
    """Run and benchmark empirical-mixture CS simulations for one paper design."""

    return run_binary_empirical_mixture_cs_benchmark_matrix(
        contracts,
        df=df,
        d=d,
        m=m,
        y=y,
        cells=cells,
        seed=seed,
        cluster=cluster,
        replications=replications,
        alpha=alpha,
        absolute_tolerance=absolute_tolerance,
        z_tolerance=z_tolerance,
        cell_count_absolute_tolerance=cell_count_absolute_tolerance,
        source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
    )


def run_empirical_mixture_benchmark_plan_by_source(
    contracts: MonteCarloContracts,
    *,
    plan: MonteCarloBenchmarkPlan,
    data_sources: dict[str, BinaryEmpiricalMixtureBenchmarkDataSource],
    seed: int,
    alpha: float = 0.05,
    absolute_tolerance: float = 0.025,
    z_tolerance: float = 2.0,
    cell_count_absolute_tolerance: float | None = None,
    source_mixture_absolute_tolerance: float | None = None,
) -> MonteCarloBenchmarkPlanRunResult:
    """Execute an empirical-mixture benchmark plan for any paper-reported method."""

    manifest = plan.rerun_manifest(
        data_sources,
        seed=seed,
        alpha=alpha,
        absolute_tolerance=absolute_tolerance,
        z_tolerance=z_tolerance,
        cell_count_absolute_tolerance=cell_count_absolute_tolerance,
        source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
    )
    return run_empirical_mixture_benchmark_manifest(contracts, manifest=manifest)


def run_empirical_mixture_benchmark_manifest(
    contracts: MonteCarloContracts,
    *,
    manifest: MonteCarloBenchmarkPlanRerunManifest,
) -> MonteCarloBenchmarkPlanRunResult:
    """Execute a preflighted empirical-mixture rerun manifest for any reported method."""

    if manifest.plan.method not in {"CS", "ARP", "FSSTdd", "FSSTndd", "K"}:
        raise KeyError(f"No empirical-mixture runner for method {manifest.plan.method!r}.")
    if not manifest.ready:
        failures = [
            diagnostic.to_dict()
            for diagnostic in manifest.data_source_diagnostics
            if not diagnostic.ready
        ]
        raise ValueError(f"Monte Carlo benchmark rerun manifest is not ready: {failures}")
    _validate_rerun_manifest_source_integrity(manifest)

    simulations: list[MonteCarloSimulationResult] = []
    diagnostics: list[MonteCarloBenchmarkDiagnostic] = []
    for scheduled_cell in manifest.cells:
        cell = scheduled_cell.cell
        design = _build_empirical_mixture_design(
            contracts,
            **scheduled_cell.to_design_kwargs(),
            method=manifest.plan.method,
        )
        simulation = _run_empirical_mixture_monte_carlo(design)
        diagnostic = contracts.benchmark_simulation(
            simulation,
            table=cell.table,
            design=cell.design,
            mediator=cell.mediator,
            clusters=cell.clusters,
            bins=cell.bins,
            t=cell.t,
            method=manifest.plan.method,
            absolute_tolerance=manifest.absolute_tolerance,
            z_tolerance=manifest.z_tolerance,
            cell_count_absolute_tolerance=manifest.cell_count_absolute_tolerance,
            source_mixture_absolute_tolerance=manifest.source_mixture_absolute_tolerance,
        )
        simulations.append(simulation)
        diagnostics.append(diagnostic)

    return MonteCarloBenchmarkPlanRunResult(
        plan=manifest.plan,
        data_source_diagnostics=manifest.data_source_diagnostics,
        scheduled_cells=manifest.cells,
        manifest_seed=manifest.seed,
        matrix=MonteCarloBenchmarkMatrixResult(
            simulations=tuple(simulations),
            diagnostics=tuple(diagnostics),
        ),
    )


def run_empirical_mixture_benchmark_focused_probe(
    contracts: MonteCarloContracts,
    *,
    manifest: MonteCarloBenchmarkPlanRerunManifest,
    table: str | None = None,
    design: str | None = None,
    clusters: tuple[int | None, ...] | None = None,
    bins: tuple[int | None, ...] | None = None,
    t_values: tuple[float, ...] | None = None,
    max_cells: int | None = None,
    replications: int | None = None,
) -> MonteCarloBenchmarkPlanRunResult:
    """Execute a deterministic low-budget empirical-mixture slice for any reported method."""

    return run_empirical_mixture_benchmark_manifest(
        contracts,
        manifest=manifest.focused_slice(
            table=table,
            design=design,
            clusters=clusters,
            bins=bins,
            t_values=t_values,
            max_cells=max_cells,
            replications=replications,
        ),
    )


def _run_method_specific_empirical_mixture_manifest(
    contracts: MonteCarloContracts,
    *,
    manifest: MonteCarloBenchmarkPlanRerunManifest,
    method: str,
) -> MonteCarloBenchmarkPlanRunResult:
    if manifest.plan.method != method:
        raise ValueError(
            f"{method} empirical-mixture runner received method={manifest.plan.method!r}."
        )
    return run_empirical_mixture_benchmark_manifest(contracts, manifest=manifest)


def run_empirical_mixture_arp_benchmark_manifest(
    contracts: MonteCarloContracts,
    *,
    manifest: MonteCarloBenchmarkPlanRerunManifest,
) -> MonteCarloBenchmarkPlanRunResult:
    """Execute a preflighted ARP empirical-mixture benchmark manifest."""

    return _run_method_specific_empirical_mixture_manifest(
        contracts,
        manifest=manifest,
        method="ARP",
    )


def run_empirical_mixture_fsstdd_benchmark_manifest(
    contracts: MonteCarloContracts,
    *,
    manifest: MonteCarloBenchmarkPlanRerunManifest,
) -> MonteCarloBenchmarkPlanRunResult:
    """Execute a preflighted FSSTdd empirical-mixture benchmark manifest."""

    return _run_method_specific_empirical_mixture_manifest(
        contracts,
        manifest=manifest,
        method="FSSTdd",
    )


def run_empirical_mixture_fsstndd_benchmark_manifest(
    contracts: MonteCarloContracts,
    *,
    manifest: MonteCarloBenchmarkPlanRerunManifest,
) -> MonteCarloBenchmarkPlanRunResult:
    """Execute a preflighted FSSTndd empirical-mixture benchmark manifest."""

    return _run_method_specific_empirical_mixture_manifest(
        contracts,
        manifest=manifest,
        method="FSSTndd",
    )


def run_empirical_mixture_k_benchmark_manifest(
    contracts: MonteCarloContracts,
    *,
    manifest: MonteCarloBenchmarkPlanRerunManifest,
) -> MonteCarloBenchmarkPlanRunResult:
    """Execute a preflighted K empirical-mixture benchmark manifest."""

    return _run_method_specific_empirical_mixture_manifest(
        contracts,
        manifest=manifest,
        method="K",
    )


def run_binary_empirical_mixture_cs_benchmark_plan(
    contracts: MonteCarloContracts,
    *,
    df: pd.DataFrame,
    d: str,
    m: str,
    y: str,
    plan: MonteCarloBenchmarkPlan,
    seed: int,
    cluster: str | None = None,
    alpha: float = 0.05,
    absolute_tolerance: float = 0.025,
    z_tolerance: float = 2.0,
    cell_count_absolute_tolerance: float | None = None,
    source_mixture_absolute_tolerance: float | None = None,
) -> MonteCarloBenchmarkPlanRunResult:
    """Execute all currently supported rows in a binary empirical-mixture benchmark plan."""

    if plan.method != "CS":
        raise NotImplementedError("Binary empirical-mixture benchmark plan runs currently support method='CS'.")
    if len(plan.executable_designs) > 1:
        raise ValueError(
            "run_binary_empirical_mixture_cs_benchmark_plan received a mixed-design plan "
            f"{plan.executable_designs}; use run_binary_empirical_mixture_cs_benchmark_plan_by_source "
            "with one data source per paper design."
        )
    if plan.executable_cells:
        matrix = run_binary_empirical_mixture_cs_benchmark_matrix(
            contracts,
            df=df,
            d=d,
            m=m,
            y=y,
            cells=plan.executable_cells,
            seed=seed,
            cluster=cluster,
            replications=plan.replications,
            alpha=alpha,
            absolute_tolerance=absolute_tolerance,
            z_tolerance=z_tolerance,
            cell_count_absolute_tolerance=cell_count_absolute_tolerance,
            source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
        )
    else:
        matrix = MonteCarloBenchmarkMatrixResult(simulations=(), diagnostics=())

    return MonteCarloBenchmarkPlanRunResult(
        plan=plan,
        matrix=matrix,
        manifest_seed=seed,
    )


def run_empirical_mixture_cs_benchmark_plan(
    contracts: MonteCarloContracts,
    *,
    df: pd.DataFrame,
    d: str,
    m: str,
    y: str,
    plan: MonteCarloBenchmarkPlan,
    seed: int,
    cluster: str | None = None,
    alpha: float = 0.05,
    absolute_tolerance: float = 0.025,
    z_tolerance: float = 2.0,
    cell_count_absolute_tolerance: float | None = None,
    source_mixture_absolute_tolerance: float | None = None,
) -> MonteCarloBenchmarkPlanRunResult:
    """Execute all rows in a single-source empirical-mixture CS benchmark plan."""

    return run_binary_empirical_mixture_cs_benchmark_plan(
        contracts,
        df=df,
        d=d,
        m=m,
        y=y,
        plan=plan,
        seed=seed,
        cluster=cluster,
        alpha=alpha,
        absolute_tolerance=absolute_tolerance,
        z_tolerance=z_tolerance,
        cell_count_absolute_tolerance=cell_count_absolute_tolerance,
        source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
    )


def run_binary_empirical_mixture_cs_benchmark_plan_by_source(
    contracts: MonteCarloContracts,
    *,
    plan: MonteCarloBenchmarkPlan,
    data_sources: dict[str, BinaryEmpiricalMixtureBenchmarkDataSource],
    seed: int,
    alpha: float = 0.05,
    absolute_tolerance: float = 0.025,
    z_tolerance: float = 2.0,
    cell_count_absolute_tolerance: float | None = None,
    source_mixture_absolute_tolerance: float | None = None,
) -> MonteCarloBenchmarkPlanRunResult:
    """Execute a benchmark plan with paper-design-specific empirical data sources."""

    if plan.method != "CS":
        raise NotImplementedError("Binary empirical-mixture benchmark plan runs currently support method='CS'.")

    manifest = plan.rerun_manifest(
        data_sources,
        seed=seed,
        alpha=alpha,
        absolute_tolerance=absolute_tolerance,
        z_tolerance=z_tolerance,
        cell_count_absolute_tolerance=cell_count_absolute_tolerance,
        source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
    )

    return run_binary_empirical_mixture_cs_benchmark_manifest(
        contracts,
        manifest=manifest,
    )


def run_empirical_mixture_cs_benchmark_plan_by_source(
    contracts: MonteCarloContracts,
    *,
    plan: MonteCarloBenchmarkPlan,
    data_sources: dict[str, BinaryEmpiricalMixtureBenchmarkDataSource],
    seed: int,
    alpha: float = 0.05,
    absolute_tolerance: float = 0.025,
    z_tolerance: float = 2.0,
    cell_count_absolute_tolerance: float | None = None,
    source_mixture_absolute_tolerance: float | None = None,
) -> MonteCarloBenchmarkPlanRunResult:
    """Execute an empirical-mixture CS benchmark plan with one source per design."""

    return run_binary_empirical_mixture_cs_benchmark_plan_by_source(
        contracts,
        plan=plan,
        data_sources=data_sources,
        seed=seed,
        alpha=alpha,
        absolute_tolerance=absolute_tolerance,
        z_tolerance=z_tolerance,
        cell_count_absolute_tolerance=cell_count_absolute_tolerance,
        source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
    )


def run_binary_empirical_mixture_cs_benchmark_manifest(
    contracts: MonteCarloContracts,
    *,
    manifest: MonteCarloBenchmarkPlanRerunManifest,
) -> MonteCarloBenchmarkPlanRunResult:
    """Execute a preflighted paper benchmark rerun manifest."""

    if manifest.plan.method != "CS":
        raise NotImplementedError("Binary empirical-mixture benchmark manifest runs currently support method='CS'.")
    if not manifest.ready:
        failures = [
            diagnostic.to_dict()
            for diagnostic in manifest.data_source_diagnostics
            if not diagnostic.ready
        ]
        raise ValueError(f"Monte Carlo benchmark rerun manifest is not ready: {failures}")
    _validate_rerun_manifest_source_integrity(manifest)

    simulations: list[MonteCarloSimulationResult] = []
    diagnostics: list[MonteCarloBenchmarkDiagnostic] = []
    for scheduled_cell in manifest.cells:
        cell = scheduled_cell.cell
        design = _build_empirical_mixture_cs_design(
            contracts,
            **scheduled_cell.to_design_kwargs(),
        )
        simulation = _run_empirical_mixture_cs_monte_carlo(design)
        diagnostic = contracts.benchmark_simulation(
            simulation,
            table=cell.table,
            design=cell.design,
            mediator=cell.mediator,
            clusters=cell.clusters,
            bins=cell.bins,
            t=cell.t,
            absolute_tolerance=manifest.absolute_tolerance,
            z_tolerance=manifest.z_tolerance,
            cell_count_absolute_tolerance=manifest.cell_count_absolute_tolerance,
            source_mixture_absolute_tolerance=manifest.source_mixture_absolute_tolerance,
        )
        simulations.append(simulation)
        diagnostics.append(diagnostic)

    return MonteCarloBenchmarkPlanRunResult(
        plan=manifest.plan,
        data_source_diagnostics=manifest.data_source_diagnostics,
        scheduled_cells=manifest.cells,
        manifest_seed=manifest.seed,
        matrix=MonteCarloBenchmarkMatrixResult(
            simulations=tuple(simulations),
            diagnostics=tuple(diagnostics),
        ),
    )


def run_empirical_mixture_cs_benchmark_manifest(
    contracts: MonteCarloContracts,
    *,
    manifest: MonteCarloBenchmarkPlanRerunManifest,
) -> MonteCarloBenchmarkPlanRunResult:
    """Execute a preflighted empirical-mixture CS benchmark rerun manifest."""

    return run_binary_empirical_mixture_cs_benchmark_manifest(
        contracts,
        manifest=manifest,
    )


def run_binary_empirical_mixture_cs_benchmark_focused_probe(
    contracts: MonteCarloContracts,
    *,
    manifest: MonteCarloBenchmarkPlanRerunManifest,
    table: str | None = None,
    design: str | None = None,
    clusters: tuple[int | None, ...] | None = None,
    bins: tuple[int | None, ...] | None = None,
    t_values: tuple[float, ...] | None = None,
    max_cells: int | None = None,
    replications: int | None = None,
) -> MonteCarloBenchmarkPlanRunResult:
    """Execute a deterministic low-budget probe from a preflighted paper rerun manifest."""

    focused_manifest = manifest.focused_slice(
        table=table,
        design=design,
        clusters=clusters,
        bins=bins,
        t_values=t_values,
        max_cells=max_cells,
        replications=replications,
    )
    return run_binary_empirical_mixture_cs_benchmark_manifest(
        contracts,
        manifest=focused_manifest,
    )


def run_empirical_mixture_cs_benchmark_focused_probe(
    contracts: MonteCarloContracts,
    *,
    manifest: MonteCarloBenchmarkPlanRerunManifest,
    table: str | None = None,
    design: str | None = None,
    clusters: tuple[int | None, ...] | None = None,
    bins: tuple[int | None, ...] | None = None,
    t_values: tuple[float, ...] | None = None,
    max_cells: int | None = None,
    replications: int | None = None,
) -> MonteCarloBenchmarkPlanRunResult:
    """Execute a deterministic low-budget empirical-mixture CS probe."""

    return run_binary_empirical_mixture_cs_benchmark_focused_probe(
        contracts,
        manifest=manifest,
        table=table,
        design=design,
        clusters=clusters,
        bins=bins,
        t_values=t_values,
        max_cells=max_cells,
        replications=replications,
    )


def _validate_benchmark_tolerances(
    *,
    absolute_tolerance: float,
    z_tolerance: float,
    cell_count_absolute_tolerance: float | None,
    source_mixture_absolute_tolerance: float | None,
) -> None:
    if absolute_tolerance < 0:
        raise ValueError("absolute_tolerance must be non-negative.")
    if z_tolerance <= 0:
        raise ValueError("z_tolerance must be positive.")
    if cell_count_absolute_tolerance is not None and cell_count_absolute_tolerance < 0:
        raise ValueError("cell_count_absolute_tolerance must be non-negative when provided.")
    if source_mixture_absolute_tolerance is not None and source_mixture_absolute_tolerance < 0:
        raise ValueError("source_mixture_absolute_tolerance must be non-negative when provided.")


def _json_safe_export_payload(value: Any) -> Any:
    """Return a payload that can be serialized with strict JSON encoders."""

    if isinstance(value, pd.Interval):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe_export_payload(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_json_safe_export_payload(item) for item in value)
    if isinstance(value, list):
        return [_json_safe_export_payload(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe_export_payload(item) for item in value.tolist()]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        float_value = float(value)
        if math.isfinite(float_value):
            return float_value
        if math.isnan(float_value):
            return None
        return "positive_infinity" if float_value > 0 else "negative_infinity"
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


def _json_safe_export_frame(
    records: list[dict[str, Any]],
    *,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Return a strict-JSON-safe DataFrame without pandas NaN coercion."""

    safe_records = _json_safe_export_payload(records)
    if columns is None:
        resolved_columns = list(
            dict.fromkeys(
                column
                for row in safe_records
                for column in row
            )
        )
    else:
        resolved_columns = columns
    frame = pd.DataFrame(
        {
            column: pd.Series(
                [row.get(column) for row in safe_records],
                dtype=object,
            )
            for column in resolved_columns
        },
        columns=resolved_columns,
    )
    if frame.empty:
        return frame
    object_frame = frame.astype(object)
    return object_frame.where(pd.notna(object_frame), None)


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


def _json_safe_float_with_marker(value: float) -> tuple[float | None, bool, str | None]:
    numeric = float(value)
    if math.isfinite(numeric):
        return numeric, True, None
    if math.isnan(numeric):
        return None, False, "nan"
    return None, False, "positive_infinity" if numeric > 0 else "negative_infinity"


def _json_safe_optional_float_with_marker(
    value: float | None,
) -> tuple[float | None, bool | None, str | None]:
    if value is None:
        return None, None, None
    return _json_safe_float_with_marker(value)


def _validate_target_precision_tolerances(
    *,
    absolute_tolerance: float | None,
    z_tolerance: float | None,
) -> None:
    if absolute_tolerance is not None and absolute_tolerance < 0:
        raise ValueError("absolute_tolerance must be non-negative when provided.")
    if z_tolerance is not None and z_tolerance <= 0:
        raise ValueError("z_tolerance must be positive when provided.")


def _scheduled_cell_bootstrap_draws(scheduled_cell: MonteCarloBenchmarkPlanRerunCell) -> int:
    """Return bootstrap work only for paper methods whose protocol requires it."""

    method_spec = _monte_carlo_method_execution_specs().get(scheduled_cell.cell.method, {})
    if not bool(method_spec.get("bootstrap_required")):
        return 0
    return int(scheduled_cell.replications) * int(scheduled_cell.bootstrap_replications)


def _cell_index_slice_manifest_from_plan(
    plan: MonteCarloBenchmarkPlan,
    data_sources: dict[str, "BinaryEmpiricalMixtureBenchmarkDataSource"],
    *,
    seed: int,
    cell_start: int,
    cell_stop: int | None,
    slice_replications: int | None,
    alpha: float,
    bootstrap_replications: int,
    absolute_tolerance: float,
    z_tolerance: float,
    cell_count_absolute_tolerance: float | None,
    source_mixture_absolute_tolerance: float | None,
) -> MonteCarloBenchmarkPlanRerunManifest:
    """Build a cell-index manifest while validating only the scheduled slice."""

    resolved_stop = len(plan.executable_cells) if cell_stop is None else cell_stop
    selected_cells = tuple(plan.executable_cells[cell_start:resolved_stop])
    if not selected_cells:
        raise ValueError("cell_index_slice selected no scheduled Monte Carlo cells.")

    selected_plan = replace(
        plan,
        replications=slice_replications or plan.replications,
        executable_cells=selected_cells,
        blocked_rows=(),
    )
    diagnostics = selected_plan.validate_data_sources(data_sources)
    seed_rng = np.random.default_rng(seed)
    full_cell_seeds = seed_rng.integers(
        low=0,
        high=np.iinfo(np.uint32).max,
        size=len(plan.executable_cells),
        dtype=np.uint32,
    ).tolist()
    scheduled_cells = tuple(
        MonteCarloBenchmarkPlanRerunCell(
            cell=cell,
            seed=int(full_cell_seeds[cell_index]),
            replications=slice_replications or plan.replications,
            bootstrap_replications=bootstrap_replications,
            alpha=alpha,
            source=_resolve_benchmark_data_source(data_sources, cell),
        )
        for cell_index, cell in enumerate(
            plan.executable_cells[cell_start:resolved_stop],
            start=cell_start,
        )
    )
    selected_keys = {_benchmark_data_source_key(cell) for cell in selected_cells}
    return MonteCarloBenchmarkPlanRerunManifest(
        plan=selected_plan,
        cells=scheduled_cells,
        data_source_diagnostics=diagnostics,
        seed=seed,
        alpha=alpha,
        absolute_tolerance=absolute_tolerance,
        z_tolerance=z_tolerance,
        cell_count_absolute_tolerance=cell_count_absolute_tolerance,
        source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
        data_source_fingerprints={
            key: _benchmark_data_source_fingerprint(
                _resolve_benchmark_data_source(data_sources, cell)
            )
            for key, cell in {
                _benchmark_data_source_key(cell): cell
                for cell in selected_cells
            }.items()
            if key in selected_keys
        },
    )


def _coerce_suite_run_result_export_frame(exported_run_results: Any) -> pd.DataFrame:
    """Return a DataFrame from saved suite evidence payloads, rows, files, or frames."""

    if exported_run_results is None:
        return pd.DataFrame()
    if isinstance(exported_run_results, pd.DataFrame):
        return exported_run_results.copy()
    if isinstance(exported_run_results, (str, Path)):
        return _coerce_suite_run_result_export_path(exported_run_results)
    if isinstance(exported_run_results, dict):
        if "frame" in exported_run_results:
            return pd.DataFrame(
                _require_json_object_sequence(
                    exported_run_results["frame"],
                    "frame",
                )
            )
        return pd.DataFrame([exported_run_results])

    frames: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    try:
        iterator = iter(exported_run_results)
    except TypeError as exc:
        raise TypeError(
            "exported_run_results must be a DataFrame, a suite to_dict payload, "
            "row records, a JSON file path, or an iterable of those objects."
        ) from exc

    for item in iterator:
        if isinstance(item, pd.DataFrame):
            frame = item.copy()
        elif isinstance(item, (str, Path)):
            frame = _coerce_suite_run_result_export_path(item)
        elif isinstance(item, dict) and "frame" in item:
            frame = pd.DataFrame(
                _require_json_object_sequence(
                    item["frame"],
                    "frame",
                )
            )
        elif isinstance(item, dict):
            records.append(item)
            continue
        else:
            raise TypeError(
                "exported_run_results iterable items must be DataFrames, "
                "suite to_dict payloads, JSON file paths, or row records."
            )
        frames.append(frame.dropna(axis=1, how="all"))
    if records:
        frames.append(pd.DataFrame(records).dropna(axis=1, how="all"))
    if not frames:
        return pd.DataFrame()
    nonempty_frames = [frame for frame in frames if not frame.empty and frame.notna().values.any()]
    if not nonempty_frames:
        return pd.DataFrame()
    frame = pd.concat(nonempty_frames, ignore_index=True)
    _validate_suite_export_execution_budget_integrity(frame)
    return frame


def _validate_suite_export_execution_budget_integrity(frame: pd.DataFrame) -> None:
    """Reject saved evidence whose executed budgets drift from its plan fields."""

    if frame.empty:
        return
    _validate_suite_export_matching_int_columns(
        frame,
        actual_column="replications",
        planned_columns=("planned_replications",),
        label="replications",
    )
    _validate_suite_export_matching_int_columns(
        frame,
        actual_column="bootstrap_replications",
        planned_columns=("planned_bootstrap_replications",),
        label="bootstrap_replications",
    )


def _validate_suite_export_matching_int_columns(
    frame: pd.DataFrame,
    *,
    actual_column: str,
    planned_columns: tuple[str, ...],
    label: str,
) -> None:
    if actual_column not in frame:
        return
    actual = pd.to_numeric(frame[actual_column], errors="coerce")
    for planned_column in planned_columns:
        if planned_column not in frame:
            continue
        planned = pd.to_numeric(frame[planned_column], errors="coerce")
        comparable = actual.notna() & planned.notna()
        mismatched = comparable & actual.ne(planned)
        if bool(mismatched.any()):
            preview_columns = [
                column
                for column in (
                    "method",
                    "table",
                    "design",
                    "mediator",
                    "clusters",
                    "bins",
                    "t",
                    actual_column,
                    planned_column,
                )
                if column in frame
            ]
            preview = frame.loc[mismatched, preview_columns].head(5).to_dict("records")
            raise ValueError(
                "suite export progress requires saved evidence actual "
                f"{label} to match {planned_column}: {preview}"
            )


def _coerce_suite_run_result_export_path(path_like: str | Path) -> pd.DataFrame:
    """Return saved evidence rows from a JSON file, directory, or glob pattern."""

    path_text = str(path_like)
    if glob.has_magic(path_text):
        matches = tuple(Path(path) for path in sorted(glob.glob(path_text)))
        if not matches:
            raise FileNotFoundError(
                f"suite export evidence glob matched no JSON files: {path_text}"
            )
        return _coerce_suite_run_result_export_frame(matches)

    path = Path(path_like)
    if path.is_dir():
        matches = tuple(sorted(path.glob("*.json")))
        if not matches:
            raise FileNotFoundError(
                f"suite export evidence directory contains no JSON files: {path}"
            )
        return _coerce_suite_run_result_export_frame(matches)

    return _coerce_suite_run_result_export_frame(
        _load_suite_run_result_export_payload(path)
    )


def _load_suite_run_result_export_payload(path_like: str | Path) -> Any:
    """Load a saved suite export payload from a JSON file."""

    path = Path(path_like)
    if not path.is_file():
        raise FileNotFoundError(
            f"suite export evidence JSON file does not exist: {path}"
        )
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_monte_carlo_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"suite export evidence file must contain JSON saved from to_dict(): {path}"
        ) from exc
    _validate_suite_run_result_export_payload(payload, path=path)
    return payload


def _validate_suite_run_result_export_payload(payload: Any, *, path: Path) -> None:
    if isinstance(payload, dict):
        if "frame" in payload:
            _require_json_object_sequence(payload["frame"], "frame")
        return
    if isinstance(payload, list):
        _require_json_object_sequence(payload, "suite export evidence rows")
        return
    raise ValueError(
        "suite export evidence JSON must be an object payload or an array of "
        f"row objects: {path}"
    )


_SUITE_RUN_RESULT_ACCEPTANCE_COLUMNS = (
    "simulation",
    "method",
    "table",
    "panel",
    "design",
    "mediator",
    "clusters",
    "bins",
    "t",
    "bar_nu_lb",
    "target_rejection_rate",
    "observed_rejection_rate",
    "absolute_error",
    "rejection_rate_absolute_error",
    "monte_carlo_standard_error",
    "z_score",
    "z_score_is_finite",
    "z_score_nonfinite",
    "passed",
    "pass_reason",
    "failure_reasons",
    "within_absolute_tolerance",
    "within_sampling_error",
    "absolute_tolerance",
    "z_tolerance",
    "target_median_independent_count_per_cell",
    "observed_mean_median_independent_count_per_cell",
    "cell_count_absolute_error",
    "cell_count_absolute_tolerance",
    "cell_count_tolerance_active",
    "within_cell_count_tolerance",
    "cell_count_size_risk",
    "target_t",
    "observed_source_mixture_share",
    "source_mixture_absolute_error",
    "source_mixture_effective_trials",
    "source_mixture_standard_error",
    "source_mixture_z_score",
    "source_mixture_z_score_is_finite",
    "source_mixture_z_score_nonfinite",
    "source_mixture_absolute_tolerance",
    "source_mixture_tolerance_active",
    "within_source_mixture_tolerance",
    "replications",
    "seed",
    "size_row",
    "status",
    "blocked_reason",
    "planned_replications",
    "planned_draws",
    "planned_bootstrap_replications",
    "bootstrap_required",
    "expected_bootstrap_replications",
    "bootstrap_replication_shortfall",
    "target_mc_standard_error",
    "target_mc_error_band",
    "target_tolerance_below_error_band",
    "source_mixture_trials_per_draw",
    "planned_source_mixture_effective_trials",
    "source_mixture_mc_standard_error",
    "source_mixture_mc_error_band",
    "source_mixture_tolerance_below_error_band",
    "data_source_ready",
    "data_source_blocking_reasons",
    "target_required_replications_for_tolerance",
    "target_replication_shortfall",
    "source_mixture_required_replications_for_tolerance",
    "source_mixture_replication_shortfall",
    "cell_count_policy_available",
    "target_median_independent_clusters_per_cell",
    "cell_count_size_risk_threshold",
    "cell_count_policy_size_risk",
    "recommended_by_cell_count",
    "bin_policy",
    "paper_rule",
    "clustered",
    "cluster_count",
    "min_n_obs_used",
    "max_n_obs_used",
    "mean_n_obs_used",
    "min_control_observations",
    "max_control_observations",
    "mean_control_observations",
    "min_treated_observations",
    "max_treated_observations",
    "mean_treated_observations",
)


def _suite_export_acceptance_frame(exported_frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize saved suite rows into the acceptance columns used by release gates."""

    if exported_frame.empty:
        return exported_frame.copy()

    frame = exported_frame.copy()
    original_columns = set(exported_frame.columns)

    def column_needs_values(column: str) -> bool:
        if column not in original_columns:
            return True
        return not bool(frame[column].map(_evidence_value_present).any())

    if column_needs_values("status"):
        frame["status"] = "executed"
    if column_needs_values("planned_replications") and "replications" in frame:
        frame["planned_replications"] = frame["replications"]
    if column_needs_values("planned_draws") and "planned_replications" in frame:
        frame["planned_draws"] = frame["planned_replications"]
    if column_needs_values("target_mc_standard_error") and {
        "target_rejection_rate",
        "planned_replications",
    }.issubset(frame.columns):
        frame["target_mc_standard_error"] = frame.apply(
            lambda row: _target_rejection_rate_standard_error(
                float(row["target_rejection_rate"]),
                replications=int(row["planned_replications"]),
            )
            if _evidence_value_present(row.get("target_rejection_rate"))
            and _evidence_value_present(row.get("planned_replications"))
            else None,
            axis=1,
        )
    if column_needs_values("target_mc_error_band") and {
        "target_mc_standard_error",
        "z_tolerance",
    }.issubset(frame.columns):
        frame["target_mc_error_band"] = frame.apply(
            lambda row: (
                float(row["target_mc_standard_error"]) * float(row["z_tolerance"])
                if _evidence_value_present(row.get("target_mc_standard_error"))
                and _evidence_value_present(row.get("z_tolerance"))
                else None
            ),
            axis=1,
        )
    if column_needs_values("target_tolerance_below_error_band") and {
        "absolute_tolerance",
        "target_mc_error_band",
    }.issubset(frame.columns):
        frame["target_tolerance_below_error_band"] = frame.apply(
            lambda row: bool(
                _evidence_value_present(row.get("absolute_tolerance"))
                and _evidence_value_present(row.get("target_mc_error_band"))
                and float(row["target_mc_error_band"]) > float(row["absolute_tolerance"])
            ),
            axis=1,
        )
    if column_needs_values("target_required_replications_for_tolerance") and {
        "target_rejection_rate",
        "absolute_tolerance",
        "z_tolerance",
    }.issubset(frame.columns):
        frame["target_required_replications_for_tolerance"] = frame.apply(
            lambda row: _required_replications_for_binomial_precision(
                probability=float(row["target_rejection_rate"]),
                absolute_tolerance=(
                    None
                    if not _evidence_value_present(row.get("absolute_tolerance"))
                    else float(row["absolute_tolerance"])
                ),
                z_tolerance=(
                    None
                    if not _evidence_value_present(row.get("z_tolerance"))
                    else float(row["z_tolerance"])
                ),
                trials_per_replication=1,
            )
            if _evidence_value_present(row.get("target_rejection_rate"))
            else None,
            axis=1,
        )
    if column_needs_values("target_replication_shortfall") and {
        "target_required_replications_for_tolerance",
        "planned_replications",
    }.issubset(frame.columns):
        frame["target_replication_shortfall"] = frame.apply(
            lambda row: _replication_shortfall(
                required_replications=row.get("target_required_replications_for_tolerance"),
                planned_replications=int(row["planned_replications"]),
            )
            if _evidence_value_present(row.get("planned_replications"))
            else None,
            axis=1,
        )
    if column_needs_values("source_mixture_required_replications_for_tolerance") and {
        "t",
        "source_mixture_absolute_tolerance",
        "z_tolerance",
        "source_mixture_trials_per_draw",
    }.issubset(frame.columns):
        frame["source_mixture_required_replications_for_tolerance"] = frame.apply(
            lambda row: _required_replications_for_binomial_precision(
                probability=float(row["t"]),
                absolute_tolerance=(
                    None
                    if not _evidence_value_present(row.get("source_mixture_absolute_tolerance"))
                    else float(row["source_mixture_absolute_tolerance"])
                ),
                z_tolerance=(
                    None
                    if not _evidence_value_present(row.get("z_tolerance"))
                    else float(row["z_tolerance"])
                ),
                trials_per_replication=(
                    None
                    if not _evidence_value_present(row.get("source_mixture_trials_per_draw"))
                    else int(row["source_mixture_trials_per_draw"])
                ),
            )
            if _evidence_value_present(row.get("t"))
            else None,
            axis=1,
        )
    if column_needs_values("source_mixture_replication_shortfall") and {
        "source_mixture_required_replications_for_tolerance",
        "planned_replications",
    }.issubset(frame.columns):
        frame["source_mixture_replication_shortfall"] = frame.apply(
            lambda row: _replication_shortfall(
                required_replications=row.get(
                    "source_mixture_required_replications_for_tolerance"
                ),
                planned_replications=int(row["planned_replications"]),
            )
            if _evidence_value_present(row.get("planned_replications"))
            else None,
            axis=1,
        )
    if column_needs_values("cell_count_policy_size_risk") and "cell_count_size_risk" in frame:
        frame["cell_count_policy_size_risk"] = frame["cell_count_size_risk"]
    if (
        column_needs_values("cell_count_policy_available")
        and "target_median_independent_clusters_per_cell" in frame
    ):
        frame["cell_count_policy_available"] = frame[
            "target_median_independent_clusters_per_cell"
        ].map(_evidence_value_present)
    if column_needs_values("recommended_by_cell_count") and "cell_count_policy_size_risk" in frame:
        frame["recommended_by_cell_count"] = ~frame[
            "cell_count_policy_size_risk"
        ].map(_truthy_value)
    if column_needs_values("bin_policy") and "cell_count_policy_size_risk" in frame:
        frame["bin_policy"] = frame["cell_count_policy_size_risk"].map(
            lambda value: (
                "below_cell_count_heuristic"
                if _truthy_value(value)
                else "within_cell_count_heuristic"
            )
        )
    for column in _SUITE_RUN_RESULT_ACCEPTANCE_COLUMNS:
        if column not in frame:
            frame[column] = None
    columns = list(
        dict.fromkeys((*_SUITE_RUN_RESULT_ACCEPTANCE_COLUMNS, *frame.columns.tolist()))
    )
    return _json_safe_export_frame(frame.to_dict("records"), columns=columns)


def _suite_executed_maps_from_export_frame(
    exported_frame: pd.DataFrame,
    expected_methods: tuple[str, ...],
    *,
    expected_keys_by_method: dict[str, set[tuple[Any, ...]]] | None = None,
    expected_seeds_by_method_key: dict[str, dict[tuple[Any, ...], int]] | None = None,
) -> tuple[
    dict[str, set[tuple[Any, ...]]],
    dict[str, dict[tuple[Any, ...], int]],
    dict[str, dict[tuple[Any, ...], int]],
]:
    executed_keys_by_method: dict[str, set[tuple[Any, ...]]] = {
        method: set() for method in expected_methods
    }
    executed_draws_by_method_key: dict[str, dict[tuple[Any, ...], int]] = {
        method: {} for method in expected_methods
    }
    executed_bootstrap_draws_by_method_key: dict[str, dict[tuple[Any, ...], int]] = {
        method: {} for method in expected_methods
    }
    if exported_frame.empty:
        return (
            executed_keys_by_method,
            executed_draws_by_method_key,
            executed_bootstrap_draws_by_method_key,
        )
    if "method" not in exported_frame:
        raise ValueError("suite export progress requires a method column.")

    expected_method_set = set(expected_methods)
    method_specs = _monte_carlo_method_execution_specs()
    for row in exported_frame.to_dict("records"):
        if row.get("status", "executed") != "executed":
            continue
        method = str(row.get("method"))
        if method not in expected_method_set:
            raise ValueError(
                "suite export progress requires exported rows to belong to the "
                f"paper suite methods: unexpected {method!r}."
            )
        key = _paper_result_mapping_key(row, method=method)
        if expected_keys_by_method is not None and key not in expected_keys_by_method.get(method, set()):
            raise ValueError(
                "suite export progress requires exported rows to belong to the "
                "current suite schedule."
            )
        if expected_seeds_by_method_key is not None:
            expected_seed = expected_seeds_by_method_key.get(method, {}).get(key)
            if expected_seed is None:
                raise ValueError(
                    "suite export progress requires exported rows to belong to the "
                    "current suite schedule."
                )
            observed_seed = _exported_progress_row_int(row, "seed")
            if observed_seed != expected_seed:
                raise ValueError(
                    "suite export progress requires exported row seeds to match "
                    "the current suite schedule."
                )
        if key in executed_keys_by_method[method]:
            raise ValueError(
                "suite export progress requires unique paper result rows across "
                "exported payloads."
            )
        planned_draws = _exported_progress_row_int(
            row,
            "planned_draws",
            "planned_replications",
            "replications",
        )
        planned_replications = _exported_progress_row_int(
            row,
            "planned_replications",
            "replications",
        )
        planned_bootstrap_replications = _exported_progress_row_int(
            row,
            "planned_bootstrap_replications",
            "bootstrap_replications",
            default=0,
        )
        bootstrap_required = bool(method_specs.get(method, {}).get("bootstrap_required"))
        executed_keys_by_method[method].add(key)
        executed_draws_by_method_key[method][key] = planned_draws
        executed_bootstrap_draws_by_method_key[method][key] = (
            planned_replications * planned_bootstrap_replications
            if bootstrap_required
            else 0
        )
    return (
        executed_keys_by_method,
        executed_draws_by_method_key,
        executed_bootstrap_draws_by_method_key,
    )


def _suite_schedule_keys_by_method(
    method_manifests: list[tuple[str, MonteCarloBenchmarkPlanRerunManifest]],
) -> dict[str, set[tuple[Any, ...]]]:
    return {
        method: {
            _paper_result_cell_key(scheduled_cell.cell)
            for scheduled_cell in manifest.cells
        }
        for method, manifest in method_manifests
    }


def _suite_schedule_seeds_by_method_key(
    method_manifests: list[tuple[str, MonteCarloBenchmarkPlanRerunManifest]],
) -> dict[str, dict[tuple[Any, ...], int]]:
    return {
        method: {
            _paper_result_cell_key(scheduled_cell.cell): int(scheduled_cell.seed)
            for scheduled_cell in manifest.cells
        }
        for method, manifest in method_manifests
    }


def _exported_progress_row_int(
    row: dict[str, Any],
    *field_names: str,
    default: int | None = None,
) -> int:
    for field_name in field_names:
        value = row.get(field_name)
        if value is not None and not pd.isna(value):
            return int(value)
    if default is not None:
        return default
    raise ValueError(
        "suite export progress requires one of these integer fields: "
        f"{field_names}."
    )


def _suite_cell_index_progress_frame_from_executed_maps(
    schedule: pd.DataFrame,
    *,
    method_manifests: list[tuple[str, MonteCarloBenchmarkPlanRerunManifest]],
    executed_keys_by_method: dict[str, set[tuple[Any, ...]]],
    executed_draws_by_method_key: dict[str, dict[tuple[Any, ...], int]],
    executed_bootstrap_draws_by_method_key: dict[str, dict[tuple[Any, ...], int]],
) -> pd.DataFrame:
    if schedule.empty:
        return schedule

    rows: list[dict[str, Any]] = []
    cumulative_executed_rows = 0
    cumulative_executed_draws = 0
    suite_result_rows = int(schedule["paper_result_rows"].iloc[0])
    suite_scheduled_draws = int(schedule["suite_planned_draws"].iloc[0])
    for schedule_row in schedule.to_dict("records"):
        cell_start = int(schedule_row["cell_start"])
        cell_stop = int(schedule_row["cell_stop"])
        scheduled_result_rows = int(schedule_row["covered_result_rows"])
        scheduled_draws = int(schedule_row["planned_draws"])
        executed_result_rows = 0
        executed_draws = 0
        scheduled_bootstrap_draws = 0
        executed_bootstrap_draws = 0
        executed_methods: list[str] = []

        for method, manifest in method_manifests:
            chunk_cells = tuple(manifest.cells[cell_start:cell_stop])
            chunk_keys = tuple(_paper_result_cell_key(scheduled_cell.cell) for scheduled_cell in chunk_cells)
            if not chunk_keys:
                continue
            scheduled_bootstrap_draws += sum(
                _scheduled_cell_bootstrap_draws(scheduled_cell)
                for scheduled_cell in chunk_cells
            )
            method_executed_rows = sum(
                1 for key in chunk_keys if key in executed_keys_by_method[method]
            )
            method_executed_draws = sum(
                executed_draws_by_method_key[method].get(key, 0)
                for key in chunk_keys
            )
            method_executed_bootstrap_draws = sum(
                executed_bootstrap_draws_by_method_key[method].get(key, 0)
                for key in chunk_keys
            )
            if method_executed_rows:
                executed_methods.append(method)
            executed_result_rows += method_executed_rows
            executed_draws += method_executed_draws
            executed_bootstrap_draws += method_executed_bootstrap_draws

        row_complete = executed_result_rows == scheduled_result_rows
        draw_complete = executed_draws == scheduled_draws
        bootstrap_complete = executed_bootstrap_draws == scheduled_bootstrap_draws
        chunk_complete = row_complete and draw_complete and bootstrap_complete
        if executed_result_rows == 0:
            chunk_status = "pending"
        elif chunk_complete:
            chunk_status = "complete"
        elif row_complete and draw_complete:
            chunk_status = "bootstrap_incomplete"
        elif row_complete:
            chunk_status = "row_complete_only"
        else:
            chunk_status = "partial"

        cumulative_executed_rows += executed_result_rows
        cumulative_executed_draws += executed_draws
        rows.append(
            {
                **schedule_row,
                "scheduled_result_rows": scheduled_result_rows,
                "scheduled_draws": scheduled_draws,
                "executed_result_rows": executed_result_rows,
                "executed_draws": executed_draws,
                "scheduled_bootstrap_draws": scheduled_bootstrap_draws,
                "executed_bootstrap_draws": executed_bootstrap_draws,
                "executed_method_count": len(executed_methods),
                "executed_methods": tuple(executed_methods),
                "row_completion_fraction": (
                    1.0
                    if scheduled_result_rows == 0
                    else executed_result_rows / scheduled_result_rows
                ),
                "draw_completion_fraction": (
                    1.0
                    if scheduled_draws == 0
                    else executed_draws / scheduled_draws
                ),
                "bootstrap_completion_fraction": (
                    1.0
                    if scheduled_bootstrap_draws == 0
                    else executed_bootstrap_draws / scheduled_bootstrap_draws
                ),
                "row_complete": row_complete,
                "draw_complete": draw_complete,
                "bootstrap_complete": bootstrap_complete,
                "chunk_complete": chunk_complete,
                "chunk_status": chunk_status,
                "chunk_shortfall_rows": max(scheduled_result_rows - executed_result_rows, 0),
                "chunk_shortfall_draws": max(scheduled_draws - executed_draws, 0),
                "chunk_shortfall_bootstrap_draws": max(
                    scheduled_bootstrap_draws - executed_bootstrap_draws,
                    0,
                ),
                "cumulative_executed_result_rows": cumulative_executed_rows,
                "cumulative_executed_draws": cumulative_executed_draws,
                "remaining_result_rows_after_progress": max(
                    suite_result_rows - cumulative_executed_rows,
                    0,
                ),
                "remaining_draws_after_progress": max(
                    suite_scheduled_draws - cumulative_executed_draws,
                    0,
                ),
            }
        )
    return _json_safe_export_frame(rows)


def _suite_cell_index_progress_summary_from_frame(
    frame: pd.DataFrame,
    *,
    seed: int,
    cell_chunk_size: int,
    paper_replications: int,
    slice_replications: int | None,
) -> dict[str, Any]:
    if frame.empty:
        return {
            "stage": "chunk_progress",
            "runner": "empirical_mixture_benchmark_suite_cell_index_run_result",
            "schedule_ready": False,
            "paper_replications": paper_replications,
            "slice_replications": slice_replications,
            "chunk_replications": (
                paper_replications if slice_replications is None else slice_replications
            ),
            "bootstrap_replications": None,
            "paper_default_replications": _PAPER_ACCEPTANCE_REPLICATIONS,
            "paper_default_bootstrap_replications": _PAPER_BOOTSTRAP_REPLICATIONS,
            "paper_replication_budget_ready": _meets_paper_replication_budget(
                paper_replications
            ),
            "chunk_replication_budget_ready": (
                int(paper_replications if slice_replications is None else slice_replications)
                == int(paper_replications)
            ),
            "bootstrap_budget_ready": False,
            "full_paper_budget_ready": False,
            "chunk_count": 0,
            "next_chunk_kwargs": None,
            "next_chunk_rerun_call": None,
            "next_chunk_evidence_path": None,
            "next_chunk_export_call": None,
            "next_action": "repair_schedule_coverage",
            "exit_criteria": "all_chunks_complete and paper_acceptance_gate.gate_passes == True",
        }

    chunk_complete = frame["chunk_complete"].map(_truthy_value)
    row_complete = frame["row_complete"].map(_truthy_value)
    draw_complete = frame["draw_complete"].map(_truthy_value)
    bootstrap_complete = frame["bootstrap_complete"].map(_truthy_value)
    row_complete_only = row_complete & ~draw_complete
    bootstrap_incomplete = row_complete & draw_complete & ~bootstrap_complete
    started = pd.to_numeric(frame["executed_result_rows"], errors="coerce").fillna(0) > 0
    partial = started & ~row_complete
    pending = ~started
    incomplete = frame.loc[~chunk_complete]
    first_incomplete = None if incomplete.empty else incomplete.iloc[0]
    next_chunk_kwargs = (
        None if first_incomplete is None else dict(first_incomplete["chunk_rerun_kwargs"])
    )
    next_chunk_rerun_call = (
        None
        if next_chunk_kwargs is None
        else str(first_incomplete.get("chunk_rerun_call") or _suite_cell_index_chunk_call(next_chunk_kwargs))
    )
    next_chunk_evidence_path = None
    next_chunk_export_call = None
    if first_incomplete is not None:
        evidence_path = first_incomplete.get("chunk_evidence_path")
        export_call = first_incomplete.get("chunk_export_call")
        next_chunk_evidence_path = (
            str(evidence_path) if _evidence_value_present(evidence_path) else None
        )
        next_chunk_export_call = (
            str(export_call) if _evidence_value_present(export_call) else None
        )
    scheduled_result_rows = int(frame["scheduled_result_rows"].sum())
    scheduled_draws = int(frame["scheduled_draws"].sum())
    scheduled_bootstrap_draws = int(frame["scheduled_bootstrap_draws"].sum())
    executed_result_rows = int(frame["executed_result_rows"].sum())
    executed_draws = int(frame["executed_draws"].sum())
    executed_bootstrap_draws = int(frame["executed_bootstrap_draws"].sum())
    chunk_replications = (
        int(frame["chunk_replications"].iloc[0])
        if "chunk_replications" in frame
        else int(paper_replications if slice_replications is None else slice_replications)
    )
    bootstrap_replications = (
        int(frame["bootstrap_replications"].iloc[0])
        if "bootstrap_replications" in frame
        else None
    )
    paper_replication_budget_ready = _meets_paper_replication_budget(
        paper_replications
    )
    chunk_replication_budget_ready = chunk_replications == int(paper_replications)
    bootstrap_budget_ready = _meets_paper_bootstrap_budget(bootstrap_replications)
    full_paper_budget_ready = (
        paper_replication_budget_ready
        and chunk_replication_budget_ready
        and bootstrap_budget_ready
    )

    return {
        "stage": "chunk_progress",
        "runner": "empirical_mixture_benchmark_suite_cell_index_run_result",
        "schedule_ready": True,
        "seed": seed,
        "cell_chunk_size": cell_chunk_size,
        "paper_replications": paper_replications,
        "slice_replications": slice_replications,
        "chunk_replications": chunk_replications,
        "bootstrap_replications": bootstrap_replications,
        "paper_default_replications": _PAPER_ACCEPTANCE_REPLICATIONS,
        "paper_default_bootstrap_replications": _PAPER_BOOTSTRAP_REPLICATIONS,
        "paper_replication_budget_ready": paper_replication_budget_ready,
        "chunk_replication_budget_ready": chunk_replication_budget_ready,
        "bootstrap_budget_ready": bootstrap_budget_ready,
        "full_paper_budget_ready": full_paper_budget_ready,
        "method_count": int(frame["method_count"].iloc[0]),
        "methods": tuple(frame["methods"].iloc[0]),
        "chunk_count": int(frame.shape[0]),
        "completed_chunk_count": int(chunk_complete.sum()),
        "row_complete_only_chunk_count": int(row_complete_only.sum()),
        "bootstrap_incomplete_chunk_count": int(bootstrap_incomplete.sum()),
        "partial_chunk_count": int(partial.sum()),
        "pending_chunk_count": int(pending.sum()),
        "scheduled_result_rows": scheduled_result_rows,
        "executed_result_rows": executed_result_rows,
        "paper_coverage_shortfall_rows": max(scheduled_result_rows - executed_result_rows, 0),
        "scheduled_draws": scheduled_draws,
        "executed_draws": executed_draws,
        "draw_shortfall": max(scheduled_draws - executed_draws, 0),
        "scheduled_bootstrap_draws": scheduled_bootstrap_draws,
        "executed_bootstrap_draws": executed_bootstrap_draws,
        "bootstrap_draw_shortfall": max(
            scheduled_bootstrap_draws - executed_bootstrap_draws,
            0,
        ),
        "row_completion_fraction": (
            1.0
            if scheduled_result_rows == 0
            else executed_result_rows / scheduled_result_rows
        ),
        "draw_completion_fraction": (
            1.0 if scheduled_draws == 0 else executed_draws / scheduled_draws
        ),
        "bootstrap_completion_fraction": (
            1.0
            if scheduled_bootstrap_draws == 0
            else executed_bootstrap_draws / scheduled_bootstrap_draws
        ),
        "all_chunks_complete": bool(chunk_complete.all()),
        "first_incomplete_chunk_index": (
            None if first_incomplete is None else int(first_incomplete["chunk_index"])
        ),
        "first_incomplete_chunk_status": (
            None if first_incomplete is None else first_incomplete["chunk_status"]
        ),
        "first_incomplete_cell_start": (
            None if first_incomplete is None else int(first_incomplete["cell_start"])
        ),
        "first_incomplete_cell_stop": (
            None if first_incomplete is None else int(first_incomplete["cell_stop"])
        ),
        "next_chunk_kwargs": next_chunk_kwargs,
        "next_chunk_rerun_call": next_chunk_rerun_call,
        "next_chunk_evidence_path": next_chunk_evidence_path,
        "next_chunk_export_call": next_chunk_export_call,
        "next_action": (
            "ready_for_release_gate"
            if bool(chunk_complete.all()) and full_paper_budget_ready
            else "restore_paper_rerun_budget"
            if bool(chunk_complete.all())
            else "run_suite_cell_index_chunk"
        ),
        "exit_criteria": "all_chunks_complete and paper_acceptance_gate.gate_passes == True",
    }


def _suite_cell_index_chunk_kwargs(
    *,
    seed: int,
    cell_start: int,
    cell_stop: int,
    paper_replications: int,
    slice_replications: int | None,
    bootstrap_replications: int,
    method: str | None,
    mediator: str | None,
    design: str | None,
    table: str | None,
    clusters: tuple[int | None, ...] | None,
    bins: tuple[int | None, ...] | None,
    t_values: tuple[float, ...] | None,
    alpha: float,
    absolute_tolerance: float,
    z_tolerance: float,
    cell_count_absolute_tolerance: float | None,
    source_mixture_absolute_tolerance: float | None,
) -> dict[str, Any]:
    kwargs = {
        "seed": seed,
        "cell_start": cell_start,
        "cell_stop": cell_stop,
        "paper_replications": paper_replications,
        "slice_replications": slice_replications,
        "bootstrap_replications": bootstrap_replications,
        "mediator": mediator,
        "design": design,
        "table": table,
        "clusters": clusters,
        "bins": bins,
        "t_values": t_values,
        "alpha": alpha,
        "absolute_tolerance": absolute_tolerance,
        "z_tolerance": z_tolerance,
        "cell_count_absolute_tolerance": cell_count_absolute_tolerance,
        "source_mixture_absolute_tolerance": source_mixture_absolute_tolerance,
    }
    if method is not None:
        kwargs["method"] = method
    return kwargs


def _suite_cell_index_chunk_call(kwargs: dict[str, Any]) -> str:
    arguments = ", ".join(f"{key}={value!r}" for key, value in kwargs.items())
    return f"contracts.empirical_mixture_benchmark_suite_cell_index_run_result(data_sources, {arguments})"


def _suite_cell_index_chunk_evidence_path(
    evidence_dir: str | Path | None,
    *,
    chunk_index: int,
    cell_start: int,
    cell_stop: int,
) -> str | None:
    if evidence_dir is None:
        return None
    path = Path(evidence_dir)
    filename = (
        f"phase11-suite-chunk-{chunk_index:03d}-cells-{cell_start:03d}-{cell_stop:03d}.json"
    )
    return str(path / filename)


def _suite_cell_index_chunk_export_call(
    chunk_rerun_call: str,
    *,
    evidence_path: str | None,
    chunk_index: int,
) -> str | None:
    if evidence_path is None:
        return None
    return (
        "testmechs.write_monte_carlo_suite_run_result_json("
        f"{chunk_rerun_call}, Path({evidence_path!r}), "
        f"owner='Phase 11 chunk {chunk_index:03d}')"
    )


def _scheduled_cells_by_design(
    cells: tuple[MonteCarloBenchmarkPlanRerunCell, ...],
) -> dict[str, tuple[MonteCarloBenchmarkPlanRerunCell, ...]]:
    scheduled_by_design: dict[str, list[MonteCarloBenchmarkPlanRerunCell]] = {}
    for scheduled_cell in cells:
        scheduled_by_design.setdefault(scheduled_cell.cell.design, []).append(scheduled_cell)
    return {
        design: tuple(scheduled_cells)
        for design, scheduled_cells in scheduled_by_design.items()
    }


def _scheduled_cells_by_data_source_key(
    cells: tuple[MonteCarloBenchmarkPlanRerunCell, ...],
) -> dict[str, tuple[MonteCarloBenchmarkPlanRerunCell, ...]]:
    scheduled_by_data_source_key: dict[str, list[MonteCarloBenchmarkPlanRerunCell]] = {}
    for scheduled_cell in cells:
        scheduled_by_data_source_key.setdefault(
            _benchmark_data_source_key(scheduled_cell.cell),
            [],
        ).append(scheduled_cell)
    return {
        key: tuple(scheduled_cells)
        for key, scheduled_cells in scheduled_by_data_source_key.items()
    }


def _diagnostic_for_scheduled_cells(
    diagnostic: MonteCarloBenchmarkDataSourceDiagnostic,
    *,
    scheduled_cells: tuple[MonteCarloBenchmarkPlanRerunCell, ...],
) -> MonteCarloBenchmarkDataSourceDiagnostic:
    return replace(
        diagnostic,
        executable_rows=len(scheduled_cells),
        requires_cluster_resampling=any(
            scheduled_cell.cell.requires_cluster_resampling
            for scheduled_cell in scheduled_cells
        ),
    )


def _validate_rerun_manifest_source_integrity(
    manifest: MonteCarloBenchmarkPlanRerunManifest,
) -> None:
    """Recheck scheduled empirical sources before consuming simulation budget."""

    diagnostics_by_data_source_key = _diagnostics_by_data_source_key(
        manifest.data_source_diagnostics
    )
    scheduled_by_data_source_key = _scheduled_cells_by_data_source_key(manifest.cells)

    expected_fingerprints = manifest.data_source_fingerprints or {}
    for data_source_key, scheduled_cells in scheduled_by_data_source_key.items():
        if data_source_key not in diagnostics_by_data_source_key:
            raise ValueError(
                f"Monte Carlo benchmark rerun manifest is missing data-source diagnostics for {data_source_key!r}."
            )

        first_fingerprint = _benchmark_data_source_fingerprint(scheduled_cells[0].source)
        expected_fingerprint = expected_fingerprints.get(data_source_key)
        if expected_fingerprint is not None and first_fingerprint != expected_fingerprint:
            raise ValueError(
                "Monte Carlo benchmark rerun manifest data source fingerprint drift "
                f"for {data_source_key!r}."
            )

        for scheduled_cell in scheduled_cells[1:]:
            fingerprint = _benchmark_data_source_fingerprint(scheduled_cell.source)
            if fingerprint != first_fingerprint:
                raise ValueError(
                    "Monte Carlo benchmark rerun manifest has inconsistent scheduled "
                    f"data sources for {data_source_key!r}."
                )

        recomputed = replace(
            _empirical_mixture_data_source_diagnostic(
                design=scheduled_cells[0].cell.design,
                cells=tuple(scheduled_cell.cell for scheduled_cell in scheduled_cells),
                source=scheduled_cells[0].source,
            ),
            data_source_key=data_source_key,
        )
        if not recomputed.ready:
            raise ValueError(
                "Monte Carlo benchmark rerun manifest scheduled data-source preflight "
                f"failed for {data_source_key!r}: {recomputed.to_dict()}"
            )
        stored_diagnostic = diagnostics_by_data_source_key[data_source_key]
        if recomputed.to_dict() != stored_diagnostic.to_dict():
            raise ValueError(
                "Monte Carlo benchmark rerun manifest data-source diagnostic drift "
                f"for {data_source_key!r}."
            )


def _benchmark_data_source_fingerprint(source: BinaryEmpiricalMixtureBenchmarkDataSource) -> str:
    complete_cases = source.analysis_frame()
    columns = list(dict.fromkeys((source.d, source.m, source.y, *source.analysis_frame_columns)))
    if source.cluster is not None and source.cluster not in columns:
        columns.append(source.cluster)
    fingerprint_frame = complete_cases.loc[:, columns].reset_index(drop=True)

    digest = hashlib.sha256()
    digest.update(repr(source.to_dict()).encode("utf-8"))
    row_hashes = pd.util.hash_pandas_object(fingerprint_frame, index=True)
    digest.update(row_hashes.to_numpy(dtype=np.uint64).tobytes())
    return digest.hexdigest()


def _benchmark_target_precision_summary(
    cells: tuple[MonteCarloBenchmarkCell, ...],
    *,
    replications: int,
) -> dict[str, Any]:
    target_rates = [cell.target_rejection_rate for cell in cells]
    target_standard_errors = [
        _target_rejection_rate_standard_error(rate, replications=replications)
        for rate in target_rates
    ]
    positive_standard_errors = [
        standard_error for standard_error in target_standard_errors if standard_error > 0
    ]
    return {
        "min_target_rejection_rate": float(min(target_rates)) if target_rates else None,
        "max_target_rejection_rate": float(max(target_rates)) if target_rates else None,
        "zero_target_mc_se_rows": int(sum(standard_error == 0 for standard_error in target_standard_errors)),
        "min_target_mc_standard_error": (
            float(min(positive_standard_errors)) if positive_standard_errors else None
        ),
        "max_target_mc_standard_error": (
            float(max(target_standard_errors)) if target_standard_errors else None
        ),
    }


def _target_precision_rows_from_plan_cells(
    cells: tuple[MonteCarloBenchmarkCell, ...],
    blocked_rows: tuple[MonteCarloBlockedBenchmarkRow, ...],
    *,
    replications: int,
    absolute_tolerance: float | None,
    z_tolerance: float | None,
) -> list[dict[str, Any]]:
    return [
        *(
            _target_precision_row_from_cell(
                cell,
                status="executable",
                blocked_reason=None,
                replications=replications,
                absolute_tolerance=absolute_tolerance,
                z_tolerance=z_tolerance,
            )
            for cell in cells
        ),
        *(
            _target_precision_row_from_blocked(
                blocked,
                absolute_tolerance=absolute_tolerance,
                z_tolerance=z_tolerance,
            )
            for blocked in blocked_rows
        ),
    ]


def _target_precision_rows_from_scheduled_cells(
    cells: tuple[MonteCarloBenchmarkPlanRerunCell, ...],
    blocked_rows: tuple[MonteCarloBlockedBenchmarkRow, ...],
    *,
    absolute_tolerance: float | None,
    z_tolerance: float | None,
) -> list[dict[str, Any]]:
    return [
        *(
            _target_precision_row_from_cell(
                scheduled_cell.cell,
                status="scheduled",
                blocked_reason=None,
                replications=scheduled_cell.replications,
                absolute_tolerance=absolute_tolerance,
                z_tolerance=z_tolerance,
            )
            for scheduled_cell in cells
        ),
        *(
            _target_precision_row_from_blocked(
                blocked,
                absolute_tolerance=absolute_tolerance,
                z_tolerance=z_tolerance,
            )
            for blocked in blocked_rows
        ),
    ]


def _target_precision_row_from_cell(
    cell: MonteCarloBenchmarkCell,
    *,
    status: str,
    blocked_reason: str | None,
    replications: int,
    absolute_tolerance: float | None,
    z_tolerance: float | None,
) -> dict[str, Any]:
    target_standard_error = _target_rejection_rate_standard_error(
        cell.target_rejection_rate,
        replications=replications,
    )
    target_error_band = (
        None if z_tolerance is None else float(z_tolerance) * target_standard_error
    )
    return {
        **cell.to_dict(),
        "status": status,
        "blocked_reason": blocked_reason,
        "planned_replications": replications,
        "planned_draws": replications,
        "target_mc_standard_error": target_standard_error,
        "z_tolerance": z_tolerance,
        "target_mc_error_band": target_error_band,
        "absolute_tolerance": absolute_tolerance,
        "target_tolerance_below_error_band": bool(
            absolute_tolerance is not None
            and target_error_band is not None
            and target_error_band > float(absolute_tolerance)
        ),
    }


def _target_precision_row_from_blocked(
    blocked: MonteCarloBlockedBenchmarkRow,
    *,
    absolute_tolerance: float | None,
    z_tolerance: float | None,
) -> dict[str, Any]:
    return {
        **blocked.to_dict(),
        "status": "blocked",
        "planned_replications": 0,
        "planned_draws": 0,
        "target_mc_standard_error": None,
        "z_tolerance": z_tolerance,
        "target_mc_error_band": None,
        "absolute_tolerance": absolute_tolerance,
        "target_tolerance_below_error_band": False,
    }


def _source_mixture_precision_rows_from_plan_cells(
    cells: tuple[MonteCarloBenchmarkCell, ...],
    blocked_rows: tuple[MonteCarloBlockedBenchmarkRow, ...],
    diagnostics: tuple[MonteCarloBenchmarkDataSourceDiagnostic, ...],
    *,
    replications: int,
    z_tolerance: float | None,
    source_mixture_absolute_tolerance: float | None,
) -> list[dict[str, Any]]:
    diagnostics_by_key = _diagnostics_by_data_source_key(diagnostics)
    return [
        *(
            _source_mixture_precision_row_from_cell(
                cell,
                status="executable",
                blocked_reason=None,
                replications=replications,
                diagnostic=_diagnostic_for_benchmark_cell(diagnostics_by_key, cell),
                z_tolerance=z_tolerance,
                source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
            )
            for cell in cells
        ),
        *(
            _source_mixture_precision_row_from_blocked(
                blocked,
                z_tolerance=z_tolerance,
                source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
            )
            for blocked in blocked_rows
        ),
    ]


def _source_mixture_precision_rows_from_scheduled_cells(
    cells: tuple[MonteCarloBenchmarkPlanRerunCell, ...],
    blocked_rows: tuple[MonteCarloBlockedBenchmarkRow, ...],
    diagnostics: tuple[MonteCarloBenchmarkDataSourceDiagnostic, ...],
    *,
    z_tolerance: float | None,
    source_mixture_absolute_tolerance: float | None,
) -> list[dict[str, Any]]:
    diagnostics_by_key = _diagnostics_by_data_source_key(diagnostics)
    return [
        *(
            _source_mixture_precision_row_from_cell(
                scheduled_cell.cell,
                status="scheduled",
                blocked_reason=None,
                replications=scheduled_cell.replications,
                diagnostic=_diagnostic_for_benchmark_cell(
                    diagnostics_by_key,
                    scheduled_cell.cell,
                ),
                z_tolerance=z_tolerance,
                source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
            )
            for scheduled_cell in cells
        ),
        *(
            _source_mixture_precision_row_from_blocked(
                blocked,
                z_tolerance=z_tolerance,
                source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
            )
            for blocked in blocked_rows
        ),
    ]


def _source_mixture_precision_row_from_cell(
    cell: MonteCarloBenchmarkCell,
    *,
    status: str,
    blocked_reason: str | None,
    replications: int,
    diagnostic: MonteCarloBenchmarkDataSourceDiagnostic | None,
    z_tolerance: float | None,
    source_mixture_absolute_tolerance: float | None,
) -> dict[str, Any]:
    standard_error = _source_mixture_standard_error(
        cell=cell,
        replications=replications,
        diagnostic=diagnostic,
    )
    trials_per_draw = _source_mixture_trials_per_draw(cell=cell, diagnostic=diagnostic)
    effective_trials = None if trials_per_draw is None else int(trials_per_draw) * int(replications)
    error_band = None if standard_error is None or z_tolerance is None else float(z_tolerance) * standard_error
    return {
        **cell.to_dict(),
        "status": status,
        "blocked_reason": blocked_reason,
        "planned_replications": replications,
        "planned_draws": replications,
        "source_mixture_trials_per_draw": trials_per_draw,
        "source_mixture_effective_trials": effective_trials,
        "source_mixture_mc_standard_error": standard_error,
        "z_tolerance": z_tolerance,
        "source_mixture_mc_error_band": error_band,
        "source_mixture_absolute_tolerance": source_mixture_absolute_tolerance,
        "source_mixture_tolerance_below_error_band": bool(
            source_mixture_absolute_tolerance is not None
            and error_band is not None
            and error_band > float(source_mixture_absolute_tolerance)
        ),
        "data_source_ready": None if diagnostic is None else diagnostic.ready,
        "data_source_blocking_reasons": None if diagnostic is None else diagnostic.blocking_reasons,
    }


def _source_mixture_precision_row_from_blocked(
    blocked: MonteCarloBlockedBenchmarkRow,
    *,
    z_tolerance: float | None,
    source_mixture_absolute_tolerance: float | None,
) -> dict[str, Any]:
    return {
        **blocked.to_dict(),
        "status": "blocked",
        "planned_replications": 0,
        "planned_draws": 0,
        "source_mixture_trials_per_draw": None,
        "source_mixture_effective_trials": None,
        "source_mixture_mc_standard_error": None,
        "z_tolerance": z_tolerance,
        "source_mixture_mc_error_band": None,
        "source_mixture_absolute_tolerance": source_mixture_absolute_tolerance,
        "source_mixture_tolerance_below_error_band": False,
        "data_source_ready": None,
        "data_source_blocking_reasons": None,
    }


def _replication_budget_rows_from_plan_cells(
    cells: tuple[MonteCarloBenchmarkCell, ...],
    blocked_rows: tuple[MonteCarloBlockedBenchmarkRow, ...],
    diagnostics: tuple[MonteCarloBenchmarkDataSourceDiagnostic, ...],
    *,
    replications: int,
    absolute_tolerance: float | None,
    z_tolerance: float | None,
    source_mixture_absolute_tolerance: float | None,
) -> list[dict[str, Any]]:
    diagnostics_by_key = _diagnostics_by_data_source_key(diagnostics)
    return [
        *(
            _replication_budget_row_from_cell(
                cell,
                status="executable",
                blocked_reason=None,
                replications=replications,
                diagnostic=_diagnostic_for_benchmark_cell(diagnostics_by_key, cell),
                absolute_tolerance=absolute_tolerance,
                z_tolerance=z_tolerance,
                source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
            )
            for cell in cells
        ),
        *(_replication_budget_row_from_blocked(blocked) for blocked in blocked_rows),
    ]


def _replication_budget_rows_from_scheduled_cells(
    cells: tuple[MonteCarloBenchmarkPlanRerunCell, ...],
    blocked_rows: tuple[MonteCarloBlockedBenchmarkRow, ...],
    diagnostics: tuple[MonteCarloBenchmarkDataSourceDiagnostic, ...],
    *,
    absolute_tolerance: float | None,
    z_tolerance: float | None,
    source_mixture_absolute_tolerance: float | None,
) -> list[dict[str, Any]]:
    diagnostics_by_key = _diagnostics_by_data_source_key(diagnostics)
    return [
        *(
            _replication_budget_row_from_cell(
                scheduled_cell.cell,
                status="scheduled",
                blocked_reason=None,
                replications=scheduled_cell.replications,
                diagnostic=_diagnostic_for_benchmark_cell(
                    diagnostics_by_key,
                    scheduled_cell.cell,
                ),
                absolute_tolerance=absolute_tolerance,
                z_tolerance=z_tolerance,
                source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
            )
            for scheduled_cell in cells
        ),
        *(_replication_budget_row_from_blocked(blocked) for blocked in blocked_rows),
    ]


def _replication_budget_row_from_cell(
    cell: MonteCarloBenchmarkCell,
    *,
    status: str,
    blocked_reason: str | None,
    replications: int,
    diagnostic: MonteCarloBenchmarkDataSourceDiagnostic | None,
    absolute_tolerance: float | None,
    z_tolerance: float | None,
    source_mixture_absolute_tolerance: float | None,
) -> dict[str, Any]:
    source_trials_per_draw = _source_mixture_trials_per_draw(cell=cell, diagnostic=diagnostic)
    target_required = _required_replications_for_binomial_precision(
        probability=cell.target_rejection_rate,
        absolute_tolerance=absolute_tolerance,
        z_tolerance=z_tolerance,
        trials_per_replication=1,
    )
    source_required = _required_replications_for_binomial_precision(
        probability=cell.t,
        absolute_tolerance=source_mixture_absolute_tolerance,
        z_tolerance=z_tolerance,
        trials_per_replication=source_trials_per_draw,
    )
    return {
        **cell.to_dict(),
        "status": status,
        "blocked_reason": blocked_reason,
        "planned_replications": replications,
        "target_absolute_tolerance": absolute_tolerance,
        "source_mixture_absolute_tolerance": source_mixture_absolute_tolerance,
        "z_tolerance": z_tolerance,
        "target_required_replications_for_tolerance": target_required,
        "target_replication_shortfall": _replication_shortfall(
            required_replications=target_required,
            planned_replications=replications,
        ),
        "source_mixture_trials_per_draw": source_trials_per_draw,
        "source_mixture_required_replications_for_tolerance": source_required,
        "source_mixture_replication_shortfall": _replication_shortfall(
            required_replications=source_required,
            planned_replications=replications,
        ),
        "data_source_ready": None if diagnostic is None else diagnostic.ready,
        "data_source_blocking_reasons": None if diagnostic is None else diagnostic.blocking_reasons,
    }


def _replication_budget_row_from_blocked(
    blocked: MonteCarloBlockedBenchmarkRow,
) -> dict[str, Any]:
    return {
        **blocked.to_dict(),
        "status": "blocked",
        "planned_replications": 0,
        "target_absolute_tolerance": None,
        "source_mixture_absolute_tolerance": None,
        "z_tolerance": None,
        "target_required_replications_for_tolerance": None,
        "target_replication_shortfall": None,
        "source_mixture_trials_per_draw": None,
        "source_mixture_required_replications_for_tolerance": None,
        "source_mixture_replication_shortfall": None,
        "data_source_ready": None,
        "data_source_blocking_reasons": None,
    }


def _cell_count_policy_rows_from_plan_cells(
    cells: tuple[MonteCarloBenchmarkCell, ...],
    blocked_rows: tuple[MonteCarloBlockedBenchmarkRow, ...],
    *,
    replications: int,
) -> list[dict[str, Any]]:
    return [
        *(
            _cell_count_policy_row_from_cell(
                cell,
                status="executable",
                blocked_reason=None,
                replications=replications,
            )
            for cell in cells
        ),
        *(_cell_count_policy_row_from_blocked(blocked) for blocked in blocked_rows),
    ]


def _cell_count_policy_rows_from_scheduled_cells(
    cells: tuple[MonteCarloBenchmarkPlanRerunCell, ...],
    blocked_rows: tuple[MonteCarloBlockedBenchmarkRow, ...],
) -> list[dict[str, Any]]:
    return [
        *(
            _cell_count_policy_row_from_cell(
                scheduled_cell.cell,
                status="scheduled",
                blocked_reason=None,
                replications=scheduled_cell.replications,
            )
            for scheduled_cell in cells
        ),
        *(_cell_count_policy_row_from_blocked(blocked) for blocked in blocked_rows),
    ]


def _cell_count_policy_row_from_cell(
    cell: MonteCarloBenchmarkCell,
    *,
    status: str,
    blocked_reason: str | None,
    replications: int,
) -> dict[str, Any]:
    cell_count = cell.cluster_cell_count
    target_count = None if cell_count is None else cell_count.median_independent_clusters_per_cell
    threshold = None if cell_count is None else cell_count.size_risk_threshold
    size_risk = None if cell_count is None else cell_count.size_risk
    return {
        **cell.to_dict(),
        "status": status,
        "blocked_reason": blocked_reason,
        "planned_replications": replications,
        "planned_draws": replications,
        "cell_count_policy_available": cell_count is not None,
        "target_median_independent_clusters_per_cell": target_count,
        "cell_count_size_risk_threshold": threshold,
        "cell_count_policy_size_risk": size_risk,
        "recommended_by_cell_count": None if size_risk is None else not size_risk,
        "bin_policy": (
            None
            if size_risk is None
            else (
                "below_cell_count_heuristic"
                if size_risk
                else "within_cell_count_heuristic"
            )
        ),
        "paper_rule": (
            None
            if cell_count is None
            else "at least 15 independent observations per cell"
        ),
    }


def _cell_count_policy_row_from_blocked(
    blocked: MonteCarloBlockedBenchmarkRow,
) -> dict[str, Any]:
    return {
        **blocked.to_dict(),
        "status": "blocked",
        "planned_replications": 0,
        "planned_draws": 0,
        "cell_count_policy_available": False,
        "target_median_independent_clusters_per_cell": None,
        "cell_count_size_risk_threshold": None,
        "cell_count_policy_size_risk": None,
        "recommended_by_cell_count": None,
        "bin_policy": None,
        "paper_rule": None,
    }


def _benchmark_target_precision_summary_from_scheduled_cells(
    cells: tuple[MonteCarloBenchmarkPlanRerunCell, ...],
) -> dict[str, Any]:
    target_rates = [cell.cell.target_rejection_rate for cell in cells]
    target_standard_errors = [
        _target_rejection_rate_standard_error(
            cell.cell.target_rejection_rate,
            replications=cell.replications,
        )
        for cell in cells
    ]
    positive_standard_errors = [
        standard_error
        for standard_error in target_standard_errors
        if standard_error > 0
    ]
    return {
        "min_target_rejection_rate": float(min(target_rates)) if target_rates else None,
        "max_target_rejection_rate": float(max(target_rates)) if target_rates else None,
        "zero_target_mc_se_rows": int(sum(standard_error == 0 for standard_error in target_standard_errors)),
        "min_target_mc_standard_error": (
            float(min(positive_standard_errors)) if positive_standard_errors else None
        ),
        "max_target_mc_standard_error": (
            float(max(target_standard_errors)) if target_standard_errors else None
        ),
    }


def _single_diagnostic_field_or_none(
    diagnostics: tuple[MonteCarloBenchmarkDiagnostic, ...],
    *,
    field_name: str,
) -> Any | None:
    if not diagnostics:
        return None
    values = [getattr(diagnostic, field_name) for diagnostic in diagnostics]
    first = values[0]
    if all(value == first for value in values):
        return first
    return None


def _diagnostics_by_data_source_key(
    diagnostics: tuple[MonteCarloBenchmarkDataSourceDiagnostic, ...],
) -> dict[str, MonteCarloBenchmarkDataSourceDiagnostic]:
    diagnostics_by_key: dict[str, MonteCarloBenchmarkDataSourceDiagnostic] = {}
    for diagnostic in diagnostics:
        key = diagnostic.data_source_key or diagnostic.design
        if key in diagnostics_by_key:
            raise ValueError(
                "data-source diagnostics must be unique by data-source key; "
                f"duplicate key {key!r}"
            )
        if key == diagnostic.design:
            for existing_key, existing_diagnostic in diagnostics_by_key.items():
                if (
                    existing_key != existing_diagnostic.design
                    and existing_diagnostic.design == diagnostic.design
                ):
                    raise ValueError(
                        "data-source diagnostics must not mix design-level and "
                        "full data-source-key diagnostics for the same design; "
                        f"design {diagnostic.design!r}"
                    )
        else:
            design_level_diagnostic = diagnostics_by_key.get(diagnostic.design)
            if design_level_diagnostic is not None:
                raise ValueError(
                    "data-source diagnostics must not mix design-level and "
                    "full data-source-key diagnostics for the same design; "
                    f"design {diagnostic.design!r}"
                )
        diagnostics_by_key[key] = diagnostic
    return diagnostics_by_key


def _diagnostic_for_benchmark_cell(
    diagnostics_by_key: dict[str, MonteCarloBenchmarkDataSourceDiagnostic],
    cell: MonteCarloBenchmarkCell,
) -> MonteCarloBenchmarkDataSourceDiagnostic | None:
    return diagnostics_by_key.get(_benchmark_data_source_key(cell)) or diagnostics_by_key.get(
        cell.design
    )


def _nonbinary_run_data_source_blocking_reasons(
    *,
    cell: MonteCarloBenchmarkCell,
    diagnostic: MonteCarloBenchmarkDataSourceDiagnostic,
) -> tuple[str, ...]:
    reasons = list(diagnostic.blocking_reasons)
    if diagnostic.complete_case_rows is None or diagnostic.complete_case_rows <= 0:
        reasons.append("empty_complete_cases")
    if len(diagnostic.treatment_levels) != 2:
        reasons.append("treatment_not_binary")
    if (
        diagnostic.control_rows is None
        or diagnostic.treated_rows is None
        or diagnostic.control_rows <= 0
        or diagnostic.treated_rows <= 0
    ):
        reasons.append("treatment_arm_empty")

    mediator_level_count = len(diagnostic.mediator_levels)
    if mediator_level_count <= 2:
        reasons.append("mediator_not_nonbinary")
    else:
        expected_mediator_level_count = _nonbinary_empirical_mixture_expected_mediator_level_count(
            cell.design
        )
        if (
            expected_mediator_level_count is not None
            and mediator_level_count != expected_mediator_level_count
        ):
            reasons.append("unexpected_mediator_level_count")

    if (
        diagnostic.expected_complete_case_rows is not None
        and diagnostic.complete_case_rows != diagnostic.expected_complete_case_rows
    ):
        reasons.append("unexpected_complete_case_rows")
    if (
        diagnostic.expected_control_rows is not None
        and diagnostic.control_rows != diagnostic.expected_control_rows
    ):
        reasons.append("unexpected_control_rows")
    if (
        diagnostic.expected_treated_rows is not None
        and diagnostic.treated_rows != diagnostic.expected_treated_rows
    ):
        reasons.append("unexpected_treated_rows")

    if cell.requires_cluster_resampling:
        if diagnostic.cluster is None:
            reasons.append("cluster_column_required")
        if diagnostic.source_clusters is None or diagnostic.source_clusters <= 0:
            reasons.append("unexpected_source_clusters")
        if diagnostic.arm_fixed_source_clusters is not True:
            reasons.append("treatment_not_fixed_within_source_cluster")
        if (
            diagnostic.control_source_clusters is None
            or diagnostic.treated_source_clusters is None
            or diagnostic.control_source_clusters <= 0
            or diagnostic.treated_source_clusters <= 0
        ):
            reasons.append("cluster_arm_empty")
        if (
            diagnostic.expected_source_clusters is not None
            and diagnostic.source_clusters != diagnostic.expected_source_clusters
        ):
            reasons.append("unexpected_source_clusters")
        if (
            diagnostic.expected_control_source_clusters is not None
            and diagnostic.control_source_clusters != diagnostic.expected_control_source_clusters
        ):
            reasons.append("unexpected_control_source_clusters")
        if (
            diagnostic.expected_treated_source_clusters is not None
            and diagnostic.treated_source_clusters != diagnostic.expected_treated_source_clusters
        ):
            reasons.append("unexpected_treated_source_clusters")
    return tuple(dict.fromkeys(reasons))


def _count_truthy_values(values: pd.Series) -> int:
    return int(values.map(_truthy_value).sum())


def _numeric_min_or_max(values: pd.Series, *, choose: str) -> float | int | None:
    numeric_values = pd.to_numeric(values, errors="coerce").dropna()
    if numeric_values.empty:
        return None
    value = numeric_values.max() if choose == "max" else numeric_values.min()
    if float(value).is_integer():
        return int(value)
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _numeric_column_count(frame: pd.DataFrame, column: str) -> int:
    if column not in frame:
        return 0
    return int(pd.to_numeric(frame[column], errors="coerce").notna().sum())


def _positive_numeric_column_count(frame: pd.DataFrame, column: str) -> int:
    if column not in frame:
        return 0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return int((values > 0).sum())


def _max_numeric_column(frame: pd.DataFrame, column: str) -> float | int | None:
    if column not in frame:
        return None
    return _numeric_min_or_max(frame[column], choose="max")


def _truthy_column_count(frame: pd.DataFrame, column: str) -> int:
    if column not in frame:
        return 0
    return _count_truthy_values(frame[column])


def _paper_acceptance_next_action(
    blocking_conditions: dict[str, int],
    *,
    data_sources_ready: bool,
    executed: bool,
) -> str:
    if not data_sources_ready or blocking_conditions.get("data_source_blocked_designs", 0) > 0:
        return "fix_data_sources"
    if (
        blocking_conditions.get("paper_coverage_unknown", 0) > 0
        or blocking_conditions.get("paper_coverage_shortfall_rows", 0) > 0
    ):
        return "run_full_paper_plan"
    if blocking_conditions.get("blocked_rows", 0) > 0:
        return "implement_blocked_paper_rows"
    if blocking_conditions.get("failed_executed_rows", 0) > 0:
        return "inspect_failed_executed_rows"
    if blocking_conditions.get("documented_tolerance_contract_missing", 0) > 0:
        return "restore_paper_tolerance_or_document_equivalent_contract"
    if blocking_conditions.get("bootstrap_replication_shortfall_rows", 0) > 0:
        return "increase_bootstrap_replications"
    if (
        not executed
        and blocking_conditions.get("benchmark_run_not_executed", 0) > 0
    ):
        return "run_manifest"
    if (
        blocking_conditions.get("target_replication_shortfall_rows", 0) > 0
        or blocking_conditions.get("source_mixture_replication_shortfall_rows", 0) > 0
    ):
        return "increase_replications_or_relax_documented_tolerance"
    return "ready_for_release_gate" if executed else "run_manifest"


def _paper_acceptance_tolerance_contract_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize whether target rejection-rate tolerances still match the paper gate."""

    if frame.empty:
        return {
            "tolerance_contract_status": "paper_default",
            "paper_target_absolute_tolerance": _PAPER_ACCEPTANCE_ABSOLUTE_TOLERANCE,
            "paper_z_tolerance": _PAPER_ACCEPTANCE_Z_TOLERANCE,
            "max_target_absolute_tolerance": None,
            "max_target_z_tolerance": None,
            "loose_target_absolute_tolerance_rows": 0,
            "loose_target_z_tolerance_rows": 0,
            "loose_target_tolerance_rows": 0,
            "documented_tolerance_contract_missing": 0,
        }

    loose_rows = pd.Series(False, index=frame.index)
    loose_absolute_rows = pd.Series(False, index=frame.index)
    loose_z_rows = pd.Series(False, index=frame.index)
    missing_rows = pd.Series(False, index=frame.index)
    missing_absolute_rows = pd.Series(False, index=frame.index)
    missing_z_rows = pd.Series(False, index=frame.index)
    max_absolute_tolerance = None
    max_z_tolerance = None

    if "absolute_tolerance" in frame:
        absolute_values = pd.to_numeric(frame["absolute_tolerance"], errors="coerce")
        valid_absolute_values = absolute_values.dropna()
        if not valid_absolute_values.empty:
            max_absolute_tolerance = float(valid_absolute_values.max())
        missing_absolute_rows = absolute_values.isna()
        loose_absolute_rows = absolute_values.gt(
            _PAPER_ACCEPTANCE_ABSOLUTE_TOLERANCE + 1e-12
        ).fillna(False)
        loose_rows = loose_rows | loose_absolute_rows
    else:
        missing_absolute_rows = pd.Series(True, index=frame.index)
    missing_rows = missing_rows | missing_absolute_rows

    if "z_tolerance" in frame:
        z_values = pd.to_numeric(frame["z_tolerance"], errors="coerce")
        valid_z_values = z_values.dropna()
        if not valid_z_values.empty:
            max_z_tolerance = float(valid_z_values.max())
        missing_z_rows = z_values.isna()
        loose_z_rows = z_values.gt(_PAPER_ACCEPTANCE_Z_TOLERANCE + 1e-12).fillna(
            False
        )
        loose_rows = loose_rows | loose_z_rows
    else:
        missing_z_rows = pd.Series(True, index=frame.index)
    missing_rows = missing_rows | missing_z_rows

    loose_target_tolerance_rows = int(loose_rows.sum())
    missing_target_tolerance_rows = int(missing_rows.sum())
    documented_tolerance_contract_missing = int((loose_rows | missing_rows).sum())
    return {
        "tolerance_contract_status": (
            "paper_default"
            if documented_tolerance_contract_missing == 0
            else "documented_tolerance_contract_missing"
        ),
        "paper_target_absolute_tolerance": _PAPER_ACCEPTANCE_ABSOLUTE_TOLERANCE,
        "paper_z_tolerance": _PAPER_ACCEPTANCE_Z_TOLERANCE,
        "max_target_absolute_tolerance": max_absolute_tolerance,
        "max_target_z_tolerance": max_z_tolerance,
        "missing_target_absolute_tolerance_rows": int(missing_absolute_rows.sum()),
        "missing_target_z_tolerance_rows": int(missing_z_rows.sum()),
        "missing_target_tolerance_rows": missing_target_tolerance_rows,
        "loose_target_absolute_tolerance_rows": int(loose_absolute_rows.sum()),
        "loose_target_z_tolerance_rows": int(loose_z_rows.sum()),
        "loose_target_tolerance_rows": loose_target_tolerance_rows,
        "documented_tolerance_contract_missing": documented_tolerance_contract_missing,
    }


def _paper_acceptance_schedule_tolerance_contract_summary(
    *,
    absolute_tolerance: float,
    z_tolerance: float,
) -> dict[str, Any]:
    """Summarize whether the requested rerun hook still matches the paper tolerance."""

    try:
        absolute_value = float(absolute_tolerance)
    except (TypeError, ValueError):
        absolute_value = math.inf
    try:
        z_value = float(z_tolerance)
    except (TypeError, ValueError):
        z_value = math.inf

    tolerance_contract_ready = (
        absolute_value <= _PAPER_ACCEPTANCE_ABSOLUTE_TOLERANCE + 1e-12
        and z_value <= _PAPER_ACCEPTANCE_Z_TOLERANCE + 1e-12
    )
    return {
        "absolute_tolerance": absolute_tolerance,
        "z_tolerance": z_tolerance,
        "paper_target_absolute_tolerance": _PAPER_ACCEPTANCE_ABSOLUTE_TOLERANCE,
        "paper_z_tolerance": _PAPER_ACCEPTANCE_Z_TOLERANCE,
        "tolerance_contract_ready": tolerance_contract_ready,
        "tolerance_contract_status": (
            "paper_default"
            if tolerance_contract_ready
            else "documented_tolerance_contract_missing"
        ),
    }


def _paper_acceptance_gate_completion_summary(
    gate: dict[str, Any],
    *,
    owner: str | None,
    rerun_command: str | None,
) -> dict[str, Any]:
    blocking_conditions = dict(gate.get("blocking_conditions", {}))
    active_blocking_conditions = dict(
        gate.get("active_blocking_conditions")
        or _active_paper_acceptance_blocking_conditions(blocking_conditions)
    )
    blocking_condition_rows = list(
        gate.get("blocking_condition_rows") or _paper_acceptance_gate_blocker_rows(gate)
    )
    summary: dict[str, Any] = {
        "owner": owner,
        "stage": gate["stage"],
        "milestone_completion_ready": bool(gate["gate_passes"]),
        "verdict": gate["verdict"],
        "method": gate.get("method", gate.get("methods")),
        "paper_coverage_known": gate.get("paper_coverage_known"),
        "paper_coverage_complete": gate.get("paper_coverage_complete"),
        "paper_result_rows": gate.get("paper_result_rows"),
        "covered_result_rows": gate.get("covered_result_rows"),
        "paper_coverage_shortfall_rows": gate.get("paper_coverage_shortfall_rows"),
        "blocking_conditions": blocking_conditions,
        "active_blocking_conditions": active_blocking_conditions,
        "active_blocking_condition_count": len(active_blocking_conditions),
        "blocked_reason_counts": dict(gate.get("blocked_reason_counts", {})),
        "blocking_condition_rows": blocking_condition_rows,
        "next_action": gate["next_action"],
        "resolution_action": gate["next_action"],
        "rerun_command": rerun_command,
        "exit_criteria": "paper_acceptance_gate.gate_passes == True",
    }
    for key in (
        "executable_rows",
        "planned_rows",
        "executed_rows",
        "blocked_rows",
        "passed_executed_rows",
        "failed_executed_rows",
        "planned_draws",
        "total_draws",
        "method_count",
        "methods",
        "evidence_method_count",
        "evidence_methods",
        "data_sources_ready",
        "executable_slice_ready",
        "full_paper_matrix_ready",
        "executable_slice_passes",
        "full_plan_passes",
        "tolerance_contract_status",
        "bootstrap_replication_shortfall_rows",
        "documented_tolerance_contract_missing",
    ):
        if key in gate:
            summary[key] = gate[key]
    return summary


def _paper_acceptance_gate_completion_frame(summary: dict[str, Any]) -> pd.DataFrame:
    blocker_rows = tuple(summary.get("blocking_condition_rows", ()))
    rows: list[dict[str, Any]] = []
    for index, blocker_row in enumerate(blocker_rows, start=1):
        rows.append(
            {
                "owner": summary.get("owner"),
                "stage": summary.get("stage"),
                "verdict": summary.get("verdict"),
                "method": summary.get("method"),
                "milestone_completion_ready": summary.get("milestone_completion_ready"),
                "active_blocking_condition_count": summary.get("active_blocking_condition_count"),
                "blocking_condition_index": index,
                "blocking_condition_count": len(blocker_rows),
                "next_action": summary.get("next_action"),
                "exit_criteria": summary.get("exit_criteria"),
                "rerun_command": summary.get("rerun_command"),
                "paper_result_rows": summary.get("paper_result_rows"),
                "covered_result_rows": summary.get("covered_result_rows"),
                "paper_coverage_shortfall_rows": summary.get("paper_coverage_shortfall_rows"),
                "planned_rows": summary.get("planned_rows"),
                "executed_rows": summary.get("executed_rows"),
                "blocked_rows": summary.get("blocked_rows"),
                "passed_executed_rows": summary.get("passed_executed_rows"),
                "failed_executed_rows": summary.get("failed_executed_rows"),
                "planned_draws": summary.get("planned_draws"),
                "total_draws": summary.get("total_draws"),
                "data_sources_ready": summary.get("data_sources_ready"),
                "executable_slice_ready": summary.get("executable_slice_ready"),
                "full_paper_matrix_ready": summary.get("full_paper_matrix_ready"),
                "paper_coverage_known": summary.get("paper_coverage_known"),
                "paper_coverage_complete": summary.get("paper_coverage_complete"),
                "blocking_conditions": summary.get("blocking_conditions"),
                "blocked_reason_counts": summary.get("blocked_reason_counts"),
                **blocker_row,
            }
        )
    return _json_safe_export_frame(rows)


def _paper_acceptance_suite_summary(
    *,
    gates: tuple[dict[str, Any], ...],
    methods: tuple[str, ...],
    executed: bool,
) -> dict[str, Any]:
    gates = tuple(gates)
    method_names = tuple(methods)
    has_method_evidence = bool(gates) and bool(method_names)
    blocking_conditions = _paper_acceptance_suite_blocking_conditions(gates, executed=executed)
    active_blocking_conditions = _active_paper_acceptance_blocking_conditions(blocking_conditions)
    method_gates_pass = has_method_evidence and all(bool(g.get("gate_passes")) for g in gates)
    summary = {
        "stage": "post_run" if executed else "readiness",
        "verdict": "pass" if not active_blocking_conditions and method_gates_pass else "blocked",
        "gate_passes": not active_blocking_conditions and method_gates_pass,
        "method_count": len(method_names),
        "methods": method_names,
        "paper_result_rows": sum(int(g.get("paper_result_rows") or 0) for g in gates),
        "covered_result_rows": sum(int(g.get("covered_result_rows") or 0) for g in gates),
        "paper_coverage_shortfall_rows": sum(int(g.get("paper_coverage_shortfall_rows") or 0) for g in gates),
        "planned_rows": sum(int(g.get("planned_rows") or 0) for g in gates),
        "executed_rows": sum(int(g.get("executed_rows") or 0) for g in gates),
        "blocked_rows": sum(int(g.get("blocked_rows") or 0) for g in gates),
        "passed_executed_rows": sum(int(g.get("passed_executed_rows") or 0) for g in gates),
        "failed_executed_rows": sum(int(g.get("failed_executed_rows") or 0) for g in gates),
        "planned_draws": sum(int(g.get("planned_draws") or 0) for g in gates),
        "total_draws": sum(int(g.get("total_draws") or 0) for g in gates),
        "full_plan_passes": has_method_evidence and all(bool(g.get("full_plan_passes")) for g in gates),
        "executable_slice_passes": has_method_evidence and all(bool(g.get("executable_slice_passes")) for g in gates),
        "paper_coverage_known": has_method_evidence and all(bool(g.get("paper_coverage_known")) for g in gates),
        "paper_coverage_complete": has_method_evidence and all(bool(g.get("paper_coverage_complete")) for g in gates),
        "tolerance_contract_status": (
            "paper_default"
            if blocking_conditions["documented_tolerance_contract_missing"] == 0
            else "documented_tolerance_contract_missing"
        ),
        "bootstrap_replication_shortfall_rows": blocking_conditions[
            "bootstrap_replication_shortfall_rows"
        ],
        "documented_tolerance_contract_missing": blocking_conditions[
            "documented_tolerance_contract_missing"
        ],
        "blocking_conditions": blocking_conditions,
        "active_blocking_conditions": active_blocking_conditions,
        "active_blocking_condition_count": len(active_blocking_conditions),
        "blocking_condition_rows": list(
            _paper_acceptance_blocker_rows_from_conditions(blocking_conditions)
        ),
        "blocked_reason_counts": _paper_acceptance_suite_blocked_reason_counts(gates),
        "next_action": _paper_acceptance_suite_next_action(
            blocking_conditions,
            executed=executed,
        ),
    }
    return summary


def _paper_acceptance_suite_gate(
    *,
    gates: tuple[dict[str, Any], ...],
    methods: tuple[str, ...],
    executed: bool,
) -> dict[str, Any]:
    summary = _paper_acceptance_suite_summary(
        gates=gates,
        methods=methods,
        executed=executed,
    )
    return {
        "stage": summary["stage"],
        "gate_passes": summary["gate_passes"],
        "verdict": summary["verdict"],
        "methods": summary["methods"],
        "method_count": summary["method_count"],
        "paper_result_rows": summary["paper_result_rows"],
        "covered_result_rows": summary["covered_result_rows"],
        "paper_coverage_shortfall_rows": summary["paper_coverage_shortfall_rows"],
        "paper_coverage_known": summary["paper_coverage_known"],
        "paper_coverage_complete": summary["paper_coverage_complete"],
        "planned_rows": summary["planned_rows"],
        "executed_rows": summary["executed_rows"],
        "blocked_rows": summary["blocked_rows"],
        "passed_executed_rows": summary["passed_executed_rows"],
        "failed_executed_rows": summary["failed_executed_rows"],
        "planned_draws": summary["planned_draws"],
        "total_draws": summary["total_draws"],
        "full_plan_passes": summary["full_plan_passes"],
        "executable_slice_passes": summary["executable_slice_passes"],
        "tolerance_contract_status": summary["tolerance_contract_status"],
        "bootstrap_replication_shortfall_rows": summary[
            "bootstrap_replication_shortfall_rows"
        ],
        "documented_tolerance_contract_missing": summary[
            "documented_tolerance_contract_missing"
        ],
        "blocking_conditions": summary["blocking_conditions"],
        "active_blocking_conditions": summary["active_blocking_conditions"],
        "active_blocking_condition_count": summary["active_blocking_condition_count"],
        "blocking_condition_rows": summary["blocking_condition_rows"],
        "blocked_reason_counts": summary["blocked_reason_counts"],
        "next_action": summary["next_action"],
    }


def _paper_acceptance_export_gate_from_frame(
    frame: pd.DataFrame,
    *,
    progress_summary: dict[str, Any],
) -> dict[str, Any]:
    """Return a suite-level acceptance gate from persisted row evidence."""

    if frame.empty:
        status = pd.Series(dtype=object)
        executed = pd.Series(dtype=bool)
        blocked = pd.Series(dtype=bool)
        passed = pd.Series(dtype=bool)
    else:
        status = frame["status"] if "status" in frame else pd.Series("executed", index=frame.index)
        executed = status.eq("executed")
        blocked = status.eq("blocked")
        passed = (
            frame["passed"].map(_truthy_value)
            if "passed" in frame
            else pd.Series(False, index=frame.index)
        )

    expected_methods = tuple(progress_summary.get("methods") or ())
    evidence_methods = (
        tuple(_ordered_monte_carlo_methods(set(frame["method"].dropna())))
        if not frame.empty and "method" in frame
        else ()
    )
    method_names = expected_methods or evidence_methods
    paper_result_rows = int(progress_summary.get("scheduled_result_rows") or 0)
    covered_result_rows = int(progress_summary.get("executed_result_rows") or 0)
    paper_coverage_shortfall_rows = int(
        progress_summary.get("paper_coverage_shortfall_rows") or 0
    )
    planned_draws = int(progress_summary.get("executed_draws") or 0)
    total_draws = planned_draws
    executed_row_count = int(executed.sum()) if not frame.empty else 0
    failed_executed_rows = int((executed & ~passed).sum()) if not frame.empty else 0
    blocked_rows = int(blocked.sum()) if not frame.empty else 0
    has_post_run_evidence = executed_row_count > 0
    not_executed_count = (
        0
        if has_post_run_evidence
        else int(progress_summary.get("method_count") or len(method_names) or 1)
    )
    data_source_blocked_designs = (
        int(frame.loc[frame["data_source_ready"].map(lambda value: value is False), "design"].nunique())
        if not frame.empty and {"data_source_ready", "design"}.issubset(frame.columns)
        else 0
    )
    replication_budget = _replication_budget_summary_from_frame(frame)
    bootstrap_budget = _bootstrap_budget_summary_from_frame(frame)
    cell_count_policy = _cell_count_policy_summary_from_frame(frame)
    tolerance_contract = _paper_acceptance_tolerance_contract_summary(frame)
    blocking_conditions = {
        "paper_coverage_unknown": int(not bool(progress_summary.get("schedule_ready"))),
        "paper_coverage_shortfall_rows": paper_coverage_shortfall_rows,
        "blocked_rows": blocked_rows,
        "failed_executed_rows": failed_executed_rows,
        "data_source_blocked_designs": data_source_blocked_designs,
        "target_replication_shortfall_rows": replication_budget["target_shortfall_rows"],
        "source_mixture_replication_shortfall_rows": replication_budget[
            "source_mixture_shortfall_rows"
        ],
        "bootstrap_replication_shortfall_rows": bootstrap_budget[
            "bootstrap_shortfall_rows"
        ],
        "documented_tolerance_contract_missing": tolerance_contract[
            "documented_tolerance_contract_missing"
        ],
        "cell_count_policy_size_risk_rows": cell_count_policy[
            "cell_count_policy_size_risk_rows"
        ],
        "benchmark_run_not_executed": not_executed_count,
    }
    active_blocking_conditions = _active_paper_acceptance_blocking_conditions(
        blocking_conditions
    )
    paper_coverage_known = bool(progress_summary.get("schedule_ready"))
    paper_coverage_complete = (
        paper_coverage_known and paper_coverage_shortfall_rows == 0
    )
    executable_slice_passes = failed_executed_rows == 0 and blocked_rows == 0
    full_plan_passes = executable_slice_passes and paper_coverage_complete
    gate_passes = full_plan_passes and not active_blocking_conditions
    return {
        "stage": "post_run" if has_post_run_evidence else "readiness",
        "gate_passes": gate_passes,
        "verdict": "pass" if gate_passes else "blocked",
        "methods": method_names,
        "method_count": len(method_names),
        "evidence_methods": evidence_methods,
        "evidence_method_count": len(evidence_methods),
        "paper_result_rows": paper_result_rows,
        "covered_result_rows": covered_result_rows,
        "paper_coverage_shortfall_rows": paper_coverage_shortfall_rows,
        "paper_coverage_known": paper_coverage_known,
        "paper_coverage_complete": paper_coverage_complete,
        "planned_rows": paper_result_rows,
        "executed_rows": executed_row_count,
        "blocked_rows": blocked_rows,
        "passed_executed_rows": int((executed & passed).sum()) if not frame.empty else 0,
        "failed_executed_rows": failed_executed_rows,
        "planned_draws": planned_draws,
        "total_draws": total_draws,
        "full_plan_passes": full_plan_passes,
        "executable_slice_passes": executable_slice_passes,
        "tolerance_contract_status": tolerance_contract["tolerance_contract_status"],
        "bootstrap_budget": bootstrap_budget,
        "bootstrap_replication_shortfall_rows": bootstrap_budget[
            "bootstrap_shortfall_rows"
        ],
        "documented_tolerance_contract_missing": tolerance_contract[
            "documented_tolerance_contract_missing"
        ],
        "tolerance_contract": tolerance_contract,
        "blocking_conditions": blocking_conditions,
        "active_blocking_conditions": active_blocking_conditions,
        "active_blocking_condition_count": len(active_blocking_conditions),
        "blocking_condition_rows": list(
            _paper_acceptance_blocker_rows_from_conditions(
                blocking_conditions,
                evidence_by_condition=_paper_acceptance_blocker_evidence_from_summaries(
                    paper_coverage={
                        "paper_coverage_known": paper_coverage_known,
                        "paper_result_rows": paper_result_rows,
                        "covered_result_rows": covered_result_rows,
                        "paper_coverage_complete": paper_coverage_complete,
                    },
                    blocked_reason_counts={},
                    data_source_summary={
                        "data_source_blocked_designs": data_source_blocked_designs,
                        "data_source_blocking_reasons": {},
                    },
                    precision_budget=_precision_budget_summary_from_frames(
                        target_frame=frame,
                        source_mixture_frame=frame,
                    ),
                    replication_budget=replication_budget,
                    bootstrap_budget=bootstrap_budget,
                    cell_count_policy=cell_count_policy,
                    tolerance_contract=tolerance_contract,
                    execution_summary={
                        "executed_rows": int(executed.sum()) if not frame.empty else 0,
                        "passed_executed_rows": int((executed & passed).sum())
                        if not frame.empty
                        else 0,
                        "failed_executed_rows": failed_executed_rows,
                    },
                ),
            )
        ),
        "blocked_reason_counts": {},
        "next_action": _paper_acceptance_suite_next_action(
            blocking_conditions,
            executed=has_post_run_evidence,
        ),
        "progress_summary": dict(progress_summary),
    }


def _paper_acceptance_suite_blocker_frame(
    *,
    methods: tuple[str, ...],
    gates: tuple[dict[str, Any], ...],
    blocker_frames: tuple[pd.DataFrame, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, gate, blocker_frame in zip(methods, gates, blocker_frames, strict=True):
        if blocker_frame.empty:
            rows.extend(
                {
                    "method": method,
                    **row,
                }
                for row in _paper_acceptance_gate_blocker_rows(gate)
            )
            continue
        for row in blocker_frame.to_dict("records"):
            rows.append({"method": method, **row})
    return _json_safe_export_frame(rows)


def _paper_acceptance_suite_unresolved_frame(
    *,
    methods: tuple[str, ...],
    frames: tuple[pd.DataFrame, ...],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for method, frame in zip(methods, frames, strict=True):
        if frame.empty:
            continue
        labeled = frame.copy()
        labeled["method"] = method
        rows.append(labeled)
    if not rows:
        return pd.DataFrame()
    return _json_safe_export_frame(
        pd.concat(rows, ignore_index=True).to_dict("records")
    )


def _paper_acceptance_suite_blocked_reason_counts(gates: tuple[dict[str, Any], ...]) -> dict[str, int]:
    reason_counts: dict[str, int] = {}
    for gate in gates:
        for reason, count in dict(gate.get("blocked_reason_counts", {})).items():
            reason_counts[reason] = reason_counts.get(reason, 0) + int(count)
    return reason_counts


def _paper_acceptance_suite_blocking_conditions(
    gates: tuple[dict[str, Any], ...],
    *,
    executed: bool,
) -> dict[str, int]:
    blocking_conditions: dict[str, int] = {
        "paper_coverage_unknown": 0,
        "paper_coverage_shortfall_rows": 0,
        "blocked_rows": 0,
        "failed_executed_rows": 0,
        "data_source_blocked_designs": 0,
        "target_replication_shortfall_rows": 0,
        "source_mixture_replication_shortfall_rows": 0,
        "bootstrap_replication_shortfall_rows": 0,
        "documented_tolerance_contract_missing": 0,
        "cell_count_policy_size_risk_rows": 0,
        "benchmark_run_not_executed": 0,
    }
    if not gates:
        blocking_conditions["paper_coverage_unknown"] = 1
        blocking_conditions["benchmark_run_not_executed"] = 1
        return blocking_conditions
    for gate in gates:
        for key in (
            "paper_coverage_unknown",
            "paper_coverage_shortfall_rows",
            "blocked_rows",
            "failed_executed_rows",
            "data_source_blocked_designs",
            "target_replication_shortfall_rows",
            "source_mixture_replication_shortfall_rows",
            "bootstrap_replication_shortfall_rows",
            "documented_tolerance_contract_missing",
            "cell_count_policy_size_risk_rows",
        ):
            blocking_conditions[key] += int(gate.get("blocking_conditions", {}).get(key, 0))
        if not executed and not bool(gate.get("gate_passes")):
            blocking_conditions["benchmark_run_not_executed"] += 1
    return blocking_conditions


def _paper_acceptance_suite_next_action(
    blocking_conditions: dict[str, int],
    *,
    executed: bool,
) -> str:
    if blocking_conditions.get("data_source_blocked_designs", 0) > 0:
        return "fix_data_sources"
    if (
        blocking_conditions.get("paper_coverage_unknown", 0) > 0
        or blocking_conditions.get("paper_coverage_shortfall_rows", 0) > 0
    ):
        return "run_full_paper_plan"
    if blocking_conditions.get("blocked_rows", 0) > 0:
        return "implement_blocked_paper_rows"
    if blocking_conditions.get("failed_executed_rows", 0) > 0:
        return "inspect_failed_executed_rows"
    if blocking_conditions.get("documented_tolerance_contract_missing", 0) > 0:
        return "restore_paper_tolerance_or_document_equivalent_contract"
    if blocking_conditions.get("bootstrap_replication_shortfall_rows", 0) > 0:
        return "increase_bootstrap_replications"
    if (
        not executed
        and blocking_conditions.get("benchmark_run_not_executed", 0) > 0
    ):
        return "run_manifest"
    if (
        blocking_conditions.get("target_replication_shortfall_rows", 0) > 0
        or blocking_conditions.get("source_mixture_replication_shortfall_rows", 0) > 0
    ):
        return "increase_replications_or_relax_documented_tolerance"
    return "ready_for_release_gate" if executed else "run_manifest"


def _milestone_closeout_blocker_rows(
    *,
    paper_acceptance_gate: dict[str, Any],
    method_support: dict[str, Any],
    paper_scope_conditions: dict[str, int],
    paper_active_conditions: dict[str, int],
    method_active_conditions: dict[str, int],
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for paper_row in _paper_acceptance_gate_blocker_rows(paper_acceptance_gate):
        row = dict(paper_row)
        condition = str(row.get("condition"))
        rows.append(
            {
                **row,
                "gate": "paper_acceptance",
                "blocked": condition in paper_active_conditions,
            }
        )

    paper_scope_evidence = _paper_acceptance_full_suite_scope_evidence(
        paper_acceptance_gate=paper_acceptance_gate,
        method_support=method_support,
    )
    for condition, value in paper_scope_conditions.items():
        rows.append(
            {
                "gate": "paper_acceptance",
                "condition": condition,
                "value": int(value),
                "blocked": condition in paper_active_conditions,
                "category": "paper_scope",
                "resolution_action": "run_full_paper_method_suite",
                "evidence": dict(paper_scope_evidence),
            }
        )

    method_evidence = {
        "paper_reported_rows": method_support["paper_reported_rows"],
        "python_executable_reported_rows": method_support[
            "python_executable_reported_rows"
        ],
        "python_blocked_reported_rows": method_support[
            "python_blocked_reported_rows"
        ],
        "blocked_method_names": method_support["blocked_method_names"],
        "paper_contract_only_method_names": method_support[
            "paper_contract_only_method_names"
        ],
        "blocking_reason_counts": method_support["blocking_reason_counts"],
    }
    for condition, value in (
        ("method_support_blocked_methods", method_support["blocked_methods"]),
        (
            "method_support_blocked_reported_rows",
            method_support["python_blocked_reported_rows"],
        ),
    ):
        rows.append(
            {
                "gate": "method_support",
                "condition": condition,
                "value": int(value),
                "blocked": condition in method_active_conditions,
                "category": "method_support",
                "resolution_action": method_support["next_action"],
                "evidence": dict(method_evidence),
            }
        )
    return tuple(rows)


def _paper_acceptance_observed_method_count(
    paper_acceptance_gate: dict[str, Any],
) -> int:
    if paper_acceptance_gate.get("method_count") is not None:
        return int(paper_acceptance_gate["method_count"])
    methods = paper_acceptance_gate.get("methods")
    if methods is not None:
        return len(tuple(methods))
    if paper_acceptance_gate.get("method") is not None:
        return 1
    return 0


def _paper_acceptance_full_suite_scope_conditions(
    *,
    paper_acceptance_gate: dict[str, Any],
    method_support: dict[str, Any],
) -> dict[str, int]:
    expected_methods = int(method_support.get("paper_methods") or 0)
    expected_rows = int(method_support.get("paper_reported_rows") or 0)
    observed_methods = _paper_acceptance_observed_method_count(paper_acceptance_gate)
    observed_rows = int(paper_acceptance_gate.get("paper_result_rows") or 0)
    return {
        "paper_acceptance_method_scope_incomplete": max(
            expected_methods - observed_methods,
            0,
        ),
        "paper_acceptance_reported_rows_incomplete": max(
            expected_rows - observed_rows,
            0,
        ),
    }


def _paper_acceptance_full_suite_scope_evidence(
    *,
    paper_acceptance_gate: dict[str, Any],
    method_support: dict[str, Any],
) -> dict[str, Any]:
    return {
        "expected_paper_methods": int(method_support.get("paper_methods") or 0),
        "observed_paper_methods": _paper_acceptance_observed_method_count(
            paper_acceptance_gate
        ),
        "expected_paper_reported_rows": int(
            method_support.get("paper_reported_rows") or 0
        ),
        "observed_paper_result_rows": int(
            paper_acceptance_gate.get("paper_result_rows") or 0
        ),
        "observed_methods": tuple(paper_acceptance_gate.get("methods") or ())
        or (() if paper_acceptance_gate.get("method") is None else (paper_acceptance_gate["method"],)),
        "required_scope": "full all-method paper Monte Carlo suite",
    }


def _milestone_closeout_workstream_rows(
    *,
    paper_acceptance_gate: dict[str, Any],
    method_support: dict[str, Any],
    paper_active_conditions: dict[str, int],
    method_active_conditions: dict[str, int],
) -> tuple[dict[str, Any], ...]:
    return (
        {
            "workstream": "paper_acceptance",
            "gate": "paper_acceptance",
            "blocked": bool(paper_active_conditions),
            "gate_passes": bool(paper_acceptance_gate.get("gate_passes")),
            "stage": paper_acceptance_gate.get("stage"),
            "next_action": paper_acceptance_gate.get("next_action"),
            "exit_criteria": "paper_acceptance_gate.gate_passes == True",
            "active_blocking_conditions": dict(paper_active_conditions),
            "active_blocking_condition_count": len(paper_active_conditions),
            "paper_result_rows": paper_acceptance_gate.get("paper_result_rows"),
            "covered_result_rows": paper_acceptance_gate.get("covered_result_rows"),
            "paper_coverage_shortfall_rows": paper_acceptance_gate.get(
                "paper_coverage_shortfall_rows"
            ),
            "blocked_rows": paper_acceptance_gate.get("blocked_rows"),
            "failed_executed_rows": paper_acceptance_gate.get("failed_executed_rows"),
        },
        {
            "workstream": "method_support",
            "gate": "method_support",
            "blocked": bool(method_active_conditions),
            "gate_passes": bool(method_support.get("method_gate_passes")),
            "stage": "method_support",
            "next_action": method_support.get("next_action"),
            "exit_criteria": method_support.get("exit_criteria"),
            "active_blocking_conditions": dict(method_active_conditions),
            "active_blocking_condition_count": len(method_active_conditions),
            "paper_methods": method_support.get("paper_methods"),
            "blocked_methods": method_support.get("blocked_methods"),
            "blocked_method_names": method_support.get("blocked_method_names"),
            "paper_reported_rows": method_support.get("paper_reported_rows"),
            "python_blocked_reported_rows": method_support.get(
                "python_blocked_reported_rows"
            ),
            "blocking_reason_counts": dict(
                method_support.get("blocking_reason_counts", {})
            ),
        },
    )


def _milestone_archive_roadmap_summary(
    roadmap_analysis: dict[str, Any] | None,
) -> dict[str, Any]:
    if roadmap_analysis is None:
        return {
            "roadmap_analysis_available": False,
            "roadmap_disk_complete": False,
            "phase_count": None,
            "completed_phases": None,
            "total_plans": None,
            "total_summaries": None,
            "progress_percent": None,
            "incomplete_phases": (),
            "condition": "roadmap_analysis_missing",
            "condition_value": 1,
            "resolution_action": "run_roadmap_analyze",
            "evidence": {},
        }

    phases = tuple(roadmap_analysis.get("phases", ()) or ())
    phase_count = int(roadmap_analysis.get("phase_count") or len(phases))
    completed_phases = int(
        roadmap_analysis.get("completed_phases")
        or sum(
            bool(phase.get("disk_status") == "complete")
            and bool(phase.get("roadmap_complete"))
            for phase in phases
        )
    )
    total_plans = int(roadmap_analysis.get("total_plans") or 0)
    total_summaries = int(roadmap_analysis.get("total_summaries") or 0)
    progress_percent = int(roadmap_analysis.get("progress_percent") or 0)
    incomplete_phases = tuple(
        str(phase.get("number"))
        for phase in phases
        if not (
            bool(phase.get("disk_status") == "complete")
            and bool(phase.get("roadmap_complete"))
        )
    )
    roadmap_disk_complete = (
        phase_count > 0
        and completed_phases == phase_count
        and progress_percent == 100
        and not incomplete_phases
        and roadmap_analysis.get("next_phase") is None
    )
    condition_value = max(phase_count - completed_phases, len(incomplete_phases), 1)
    return {
        "roadmap_analysis_available": True,
        "roadmap_disk_complete": roadmap_disk_complete,
        "phase_count": phase_count,
        "completed_phases": completed_phases,
        "total_plans": total_plans,
        "total_summaries": total_summaries,
        "progress_percent": progress_percent,
        "incomplete_phases": incomplete_phases,
        "condition": "roadmap_disk_incomplete",
        "condition_value": condition_value,
        "resolution_action": "finish_or_sync_phase_summaries",
        "evidence": {
            "phase_count": phase_count,
            "completed_phases": completed_phases,
            "total_plans": total_plans,
            "total_summaries": total_summaries,
            "progress_percent": progress_percent,
            "incomplete_phases": incomplete_phases,
            "next_phase": roadmap_analysis.get("next_phase"),
            "missing_phase_details": roadmap_analysis.get("missing_phase_details"),
        },
    }


def _milestone_archive_audit_summary(
    milestone_audit_status: str | None,
) -> dict[str, Any]:
    status = (milestone_audit_status or "missing").strip().lower()
    passes = status in {"pass", "passed"}
    condition = {
        "missing": "milestone_audit_missing",
        "stale": "milestone_audit_stale",
        "gaps_found": "milestone_audit_gaps_found",
        "failed": "milestone_audit_failed",
    }.get(status, "milestone_audit_not_passed")
    resolution_action = {
        "missing": "run_milestone_audit",
        "stale": "run_milestone_audit",
        "gaps_found": "plan_milestone_gap_closure",
        "failed": "run_milestone_audit",
    }.get(status, "run_milestone_audit")
    return {
        "milestone_audit_status": "passed" if passes else status,
        "milestone_audit_passes": passes,
        "condition": condition,
        "condition_value": 0 if passes else 1,
        "resolution_action": "archive_milestone" if passes else resolution_action,
        "evidence": {"milestone_audit_status": "passed" if passes else status},
    }


def _milestone_audit_passes(status: str | None) -> bool:
    return (status or "").strip().lower() in {"pass", "passed"}


def _planning_control_anchor_paths(planning_dir: str | Path) -> tuple[Path, ...]:
    resolved = Path(planning_dir)
    return tuple(
        resolved / name
        for name in (
            "STATE.md",
            "ROADMAP.md",
            "REQUIREMENTS.md",
        )
    )


def _passed_milestone_audit_is_older_than_planning_controls(
    audit_path: Path,
    *,
    planning_dir: str | Path,
) -> bool:
    try:
        audit_mtime = audit_path.stat().st_mtime
    except OSError:
        return False
    for anchor_path in _planning_control_anchor_paths(planning_dir):
        try:
            anchor_mtime = anchor_path.stat().st_mtime
        except OSError:
            continue
        if audit_mtime + 1.0 < anchor_mtime:
            return True
    return False


def milestone_audit_status_from_file(
    path: str | Path,
    *,
    planning_dir: str | Path | None = None,
) -> str | None:
    """Return the frontmatter status from a GSD milestone audit file.

    Parses the YAML frontmatter of a Markdown audit file and extracts the
    ``status`` field.  When *planning_dir* is provided, a passed audit file
    that predates the current planning control files is surfaced as ``stale``.

    Parameters
    ----------
    path : str or Path
        Path to the milestone audit Markdown file.
    planning_dir : str, Path, or None
        Planning directory for staleness detection.  If ``None``, staleness
        is not checked.

    Returns
    -------
    str or None
        The audit status string (e.g. ``"passed"``, ``"stale"``), or
        ``None`` if the file does not exist.
    """

    audit_path = Path(path)
    if not audit_path.exists():
        return None
    in_frontmatter = False
    status: str | None = None
    for raw_line in audit_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "---":
            if not in_frontmatter:
                in_frontmatter = True
                continue
            break
        if not in_frontmatter or not line or line.startswith("#"):
            continue
        match = re.match(r"status\s*:\s*(.+)$", line)
        if match:
            status = match.group(1).strip().strip("\"'")
            break
    if (
        status is not None
        and planning_dir is not None
        and _milestone_audit_passes(status)
        and _passed_milestone_audit_is_older_than_planning_controls(
            audit_path,
            planning_dir=planning_dir,
        )
    ):
        return "stale"
    return status


def milestone_audit_status_for_version(
    version: str,
    *,
    planning_dir: str | Path = ".planning",
) -> str | None:
    """Return the audit status for a versioned milestone audit.

    Searches for the audit file across standard locations (live planning,
    milestones subdirectory, manuscript planning directory) and returns
    the parsed status.

    Parameters
    ----------
    version : str
        Milestone version (e.g. ``"v1.2"`` or ``"1.2"``).
    planning_dir : str or Path
        Root planning directory to search.

    Returns
    -------
    str or None
        The audit status, or ``None`` if no audit file is found.
    """

    normalized = str(version).strip()
    if not normalized:
        return None
    if not normalized.startswith("v"):
        normalized = f"v{normalized}"
    resolved_planning_dir = Path(planning_dir)
    audit_name = f"{normalized}-MILESTONE-AUDIT.md"
    candidate_paths = (
        resolved_planning_dir / audit_name,
        resolved_planning_dir / "milestones" / audit_name,
        resolved_planning_dir.parent
        / "docs"
        / "internal"
        / "manuscript-planning"
        / "milestones"
        / audit_name,
    )
    for audit_path in candidate_paths:
        status = milestone_audit_status_from_file(
            audit_path,
            planning_dir=(
                planning_dir
                if audit_path.is_relative_to(resolved_planning_dir)
                else None
            ),
        )
        if status is not None:
            return status
    return None


def _load_roadmap_analysis_from_gsd_tools(
    *,
    repo_root: str | Path,
    gsd_tools_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return the live `roadmap analyze` JSON emitted by the GSD tools CLI."""

    resolved_repo_root = Path(repo_root).resolve()
    resolved_tools_path = (
        Path.home() / ".codex" / "get-shit-done" / "bin" / "gsd-tools.cjs"
        if gsd_tools_path is None
        else Path(gsd_tools_path)
    )
    if not resolved_tools_path.exists():
        raise FileNotFoundError(
            f"GSD roadmap analyzer not found at {resolved_tools_path}"
        )
    try:
        output = subprocess.check_output(
            ["node", str(resolved_tools_path), "roadmap", "analyze"],
            cwd=resolved_repo_root,
            text=True,
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            f"Failed to run roadmap analyze via {resolved_tools_path}"
        ) from exc
    payload = json.loads(
        output,
        parse_constant=_reject_roadmap_analysis_json_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError("roadmap analyze output must be a JSON object.")
    _reject_nonfinite_json_numbers(
        payload,
        field_name="roadmap analysis from gsd-tools payload",
    )
    return payload


def _resolve_milestone_archive_audit_status(
    *,
    milestone_audit_status: str | None,
    milestone_audit_path: str | Path | None,
    milestone_version: str | None,
    planning_dir: str | Path,
) -> str | None:
    if milestone_audit_status is not None:
        return milestone_audit_status
    if milestone_audit_path is not None:
        return milestone_audit_status_from_file(
            milestone_audit_path,
            planning_dir=Path(milestone_audit_path).parent,
        )
    if milestone_version is not None:
        return milestone_audit_status_for_version(
            milestone_version,
            planning_dir=planning_dir,
        )
    return None


def _milestone_archive_blocker_rows(
    *,
    roadmap: dict[str, Any],
    audit: dict[str, Any],
    closeout: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = [
        {
            "gate": "roadmap",
            "condition": roadmap["condition"],
            "value": 0 if roadmap["roadmap_disk_complete"] else roadmap["condition_value"],
            "blocked": not bool(roadmap["roadmap_disk_complete"]),
            "category": "roadmap_completion",
            "resolution_action": roadmap["resolution_action"],
            "evidence": dict(roadmap["evidence"]),
        },
        {
            "gate": "milestone_audit",
            "condition": audit["condition"],
            "value": audit["condition_value"],
            "blocked": not bool(audit["milestone_audit_passes"]),
            "category": "milestone_audit",
            "resolution_action": audit["resolution_action"],
            "evidence": dict(audit["evidence"]),
        },
    ]
    rows.extend(dict(row) for row in closeout.get("blocking_condition_rows", ()))
    return tuple(rows)


def _active_blocking_conditions_from_rows(
    rows: tuple[dict[str, Any], ...],
) -> dict[str, int]:
    active: dict[str, int] = {}
    for row in rows:
        if not bool(row.get("blocked")):
            continue
        condition = str(row.get("condition"))
        value = row.get("value")
        if isinstance(value, (int, np.integer)):
            active[condition] = int(value)
        elif isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
            active[condition] = int(value) if float(value).is_integer() else 1
        else:
            active[condition] = 1
    return active


def _milestone_archive_next_action(
    *,
    roadmap: dict[str, Any],
    audit: dict[str, Any],
    closeout: dict[str, Any],
) -> str:
    if not bool(roadmap["roadmap_disk_complete"]):
        return str(roadmap["resolution_action"])
    if not bool(audit["milestone_audit_passes"]):
        return str(audit["resolution_action"])
    if not bool(closeout["milestone_closeout_ready"]):
        return str(closeout["next_action"])
    return "archive_milestone"


def _suite_archive_summary_next_action(
    *,
    evidence_summary: dict[str, Any],
    archive_gate: dict[str, Any],
) -> str:
    if not bool(archive_gate["paper_acceptance_gate_passes"]):
        return str(evidence_summary["next_action"])
    return str(archive_gate["next_action"])


def _complete_milestone_preflight_decision(archive_ready: bool) -> dict[str, Any]:
    """Return the archive/delete/tag decision for `$gsd-complete-milestone`."""

    return {
        "decision": (
            "proceed_to_gsd_complete_milestone"
            if archive_ready
            else "blocked_before_gsd_complete_milestone"
        ),
        "archive_delete_tag_allowed": archive_ready,
        "blocked_steps": ()
        if archive_ready
        else (
            "collapse_roadmap",
            "archive_requirements",
            "delete_active_requirements",
            "commit_milestone_archive",
            "tag_release",
        ),
        "required_gate": (
            "roadmap_disk_complete == True and "
            "milestone_audit_status == 'passed' and "
            "milestone_closeout_gate_summary.milestone_closeout_ready == True"
        ),
    }


def _archive_gate_rerun_hook(summary: dict[str, Any]) -> dict[str, Any]:
    """Return a stable rerun hook for archive blocker packets."""

    preflight = dict(summary.get("complete_milestone_preflight", {}))
    rerun_command = summary.get("rerun_command")
    if rerun_command is None:
        rerun_command = (
            summary.get("evidence_next_chunk_export_call")
            or summary.get("evidence_next_chunk_rerun_call")
        )
    return {
        "next_action": summary.get("next_action"),
        "rerun_command": rerun_command,
        "archive_delete_tag_allowed": preflight.get("archive_delete_tag_allowed"),
        "blocked_archive_steps": tuple(preflight.get("blocked_steps", ())),
        "exit_criteria": summary.get("exit_criteria"),
    }


def _raise_for_milestone_completion_blockers(summary: dict[str, Any]) -> None:
    if bool(summary.get("milestone_completion_ready")):
        return

    blocked_rows = [
        dict(row)
        for row in summary.get("blocking_condition_rows", ())
        if bool(row.get("blocked"))
    ]
    preview = [
        {
            "condition": row.get("condition"),
            "value": row.get("value"),
            "category": row.get("category"),
            "resolution_action": row.get("resolution_action"),
        }
        for row in blocked_rows[:5]
    ]
    raise AssertionError(
        "Milestone completion blocked by paper acceptance gate: "
        f"stage={summary.get('stage')!r}, "
        f"method={summary.get('method')!r}, "
        f"next_action={summary.get('next_action')!r}, "
        f"active_blocking_conditions={summary.get('active_blocking_conditions')!r}, "
        f"blocked_reason_counts={summary.get('blocked_reason_counts')!r}, "
        f"paper_coverage_shortfall_rows={summary.get('paper_coverage_shortfall_rows')!r}, "
        f"blocked_rows={summary.get('blocked_rows')!r}, "
        f"failed_executed_rows={summary.get('failed_executed_rows')!r}, "
        f"blocking_condition_preview={preview!r}, "
        f"exit_criteria={summary.get('exit_criteria')!r}"
    )


def _paper_acceptance_gate_blocker_rows(
    gate: dict[str, Any],
    *,
    evidence_by_condition: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], ...]:
    if "blocking_condition_rows" in gate:
        rows = tuple(dict(row) for row in gate["blocking_condition_rows"])
        if evidence_by_condition is None:
            return rows
        return tuple(_with_blocker_evidence(row, evidence_by_condition) for row in rows)
    return _paper_acceptance_blocker_rows_from_conditions(
        dict(gate.get("blocking_conditions", {})),
        evidence_by_condition=evidence_by_condition,
    )


def _append_row_aligned_columns(
    frame: pd.DataFrame,
    supplement: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    source_name: str,
) -> None:
    if supplement.empty:
        return
    if supplement.shape[0] != frame.shape[0]:
        raise ValueError(
            f"{source_name} rows must align with paper rows; got "
            f"{supplement.shape[0]} rows for {frame.shape[0]} paper rows."
        )
    for column in columns:
        if column not in supplement.columns or column in frame.columns:
            continue
        frame[column] = supplement[column]


def _paper_acceptance_unresolved_row_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    include_failed_executed_rows: bool,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    unresolved = frame["status"].eq("blocked") if "status" in frame else pd.Series(False, index=frame.index)
    if include_failed_executed_rows and "passed" in frame:
        unresolved = unresolved | ~frame["passed"].map(_truthy_value)
    if "target_replication_shortfall" in frame:
        unresolved = unresolved | frame["target_replication_shortfall"].map(_is_positive_numeric_value)
    if "source_mixture_replication_shortfall" in frame:
        unresolved = unresolved | frame["source_mixture_replication_shortfall"].map(
            _is_positive_numeric_value
        )
    if "bootstrap_replication_shortfall" in frame:
        unresolved = unresolved | frame["bootstrap_replication_shortfall"].map(
            _is_positive_numeric_value
        )
    unresolved = unresolved | frame.apply(
        _paper_acceptance_row_has_missing_tolerance_contract,
        axis=1,
    )
    unresolved = unresolved | frame.apply(
        _paper_acceptance_row_has_loose_tolerance_contract,
        axis=1,
    )
    if "data_source_ready" in frame:
        unresolved = unresolved | frame["data_source_ready"].map(lambda value: value is False)

    unresolved_frame = frame.loc[unresolved].copy()
    unresolved_frame["source"] = source
    if unresolved_frame.empty:
        unresolved_frame["unresolved_reason"] = pd.Series(dtype=object)
        unresolved_frame["unresolved_reason_count"] = pd.Series(dtype=int)
        unresolved_frame["unresolved_root_cause"] = pd.Series(dtype=object)
        unresolved_frame["unresolved_resolution_action"] = pd.Series(dtype=object)
        unresolved_frame["unresolved_rerun_scope"] = pd.Series(dtype=object)
        unresolved_frame["unresolved_fresh_evidence"] = pd.Series(dtype=object)
        return unresolved_frame.reset_index(drop=True)
    unresolved_frame["unresolved_reason"] = unresolved_frame.apply(
        _paper_acceptance_unresolved_reason,
        axis=1,
        include_failed_executed_rows=include_failed_executed_rows,
    )
    unresolved_frame["unresolved_reason_count"] = unresolved_frame["unresolved_reason"].map(len)
    unresolved_frame["unresolved_root_cause"] = unresolved_frame["unresolved_reason"].map(
        _paper_acceptance_unresolved_root_cause
    )
    unresolved_frame["unresolved_resolution_action"] = unresolved_frame["unresolved_reason"].map(
        _paper_acceptance_unresolved_resolution_action
    )
    unresolved_frame["unresolved_rerun_scope"] = unresolved_frame.apply(
        _paper_acceptance_unresolved_rerun_scope,
        axis=1,
    )
    unresolved_frame["unresolved_fresh_evidence"] = unresolved_frame.apply(
        _paper_acceptance_unresolved_fresh_evidence,
        axis=1,
    )
    return unresolved_frame.reset_index(drop=True)


def _paper_acceptance_unresolved_reason(
    row: pd.Series,
    *,
    include_failed_executed_rows: bool,
) -> tuple[str, ...]:
    reasons: list[str] = []
    status = row.get("status")
    blocked_reason = row.get("blocked_reason")
    if status == "blocked":
        reasons.append(f"blocked:{blocked_reason}" if blocked_reason else "blocked")
    elif include_failed_executed_rows and not _truthy_value(row.get("passed")):
        failure_reasons = row.get("failure_reasons")
        if isinstance(failure_reasons, (list, tuple)) and failure_reasons:
            reasons.extend(f"failed:{reason}" for reason in failure_reasons)
        else:
            reasons.append("failed_executed_row")

    for column, label in (
        ("target_replication_shortfall", "target_replication_shortfall"),
        ("source_mixture_replication_shortfall", "source_mixture_replication_shortfall"),
        ("bootstrap_replication_shortfall", "bootstrap_replication_shortfall"),
    ):
        if _is_positive_numeric_value(row.get(column)):
            reasons.append(label)

    if row.get("data_source_ready") is False:
        reasons.append("data_source_preflight")

    if _paper_acceptance_row_has_missing_tolerance_contract(row):
        reasons.append("documented_tolerance_contract_missing")
    if _paper_acceptance_row_has_loose_tolerance_contract(row):
        reasons.append("documented_tolerance_contract_missing")

    return tuple(dict.fromkeys(reasons))


def _paper_acceptance_unresolved_root_cause(reasons: tuple[str, ...]) -> str:
    reason_set = set(reasons)
    if any(reason.startswith("blocked:") or reason == "blocked" for reason in reasons):
        return "implementation_scope"
    if any(reason.startswith("failed:") or reason == "failed_executed_row" for reason in reasons):
        return "executed_result_drift"
    if "data_source_preflight" in reason_set:
        return "data_source_preflight"
    if "documented_tolerance_contract_missing" in reason_set:
        return "tolerance_contract"
    if "bootstrap_replication_shortfall" in reason_set:
        return "bootstrap_budget"
    if (
        "target_replication_shortfall" in reason_set
        or "source_mixture_replication_shortfall" in reason_set
    ):
        return "replication_budget"
    return "unknown"


def _paper_acceptance_unresolved_resolution_action(reasons: tuple[str, ...]) -> str:
    root_cause = _paper_acceptance_unresolved_root_cause(reasons)
    return {
        "implementation_scope": "implement_blocked_paper_rows",
        "executed_result_drift": "inspect_failed_executed_rows",
        "data_source_preflight": "fix_data_sources",
        "tolerance_contract": "restore_paper_tolerance_or_document_equivalent_contract",
        "bootstrap_budget": "increase_bootstrap_replications",
        "replication_budget": "increase_replications_or_document_tolerance",
    }.get(root_cause, "inspect_unresolved_paper_row")


def _paper_acceptance_unresolved_rerun_scope(row: pd.Series) -> str:
    reason_set = set(row.get("unresolved_reason") or ())
    if row.get("status") == "blocked":
        return "blocked_until_implementation_or_rescope"
    if "documented_tolerance_contract_missing" in reason_set:
        return "same_paper_cell_with_paper_default_or_documented_tolerance"
    if "bootstrap_replication_shortfall" in reason_set:
        return "same_paper_cell_with_paper_bootstrap_budget"
    if (
        "target_replication_shortfall" in reason_set
        or "source_mixture_replication_shortfall" in reason_set
    ):
        return "same_paper_cell_with_larger_replication_budget"
    if row.get("status") == "executed":
        return "same_paper_cell_after_fix"
    return "inspect_row"


def _paper_acceptance_unresolved_fresh_evidence(row: pd.Series) -> dict[str, Any]:
    evidence_columns = (
        "source",
        "status",
        "table",
        "design",
        "mediator",
        "clusters",
        "bins",
        "t",
        "method",
        "target_rejection_rate",
        "observed_rejection_rate",
        "rejection_rate_absolute_error",
        "z_score",
        "failure_reasons",
        "blocked_reason",
        "target_replication_shortfall",
        "source_mixture_replication_shortfall",
        "bootstrap_required",
        "planned_bootstrap_replications",
        "expected_bootstrap_replications",
        "bootstrap_replication_shortfall",
        "absolute_tolerance",
        "z_tolerance",
        "target_mc_error_band",
        "target_tolerance_below_error_band",
        "data_source_ready",
        "data_source_blocking_reasons",
        "planned_replications",
        "planned_draws",
        "replications",
    )
    return {
        column: _normalize_diagnostic_value(row.get(column))
        for column in evidence_columns
        if column in row.index and _evidence_value_present(row.get(column))
    }


def _paper_acceptance_row_has_loose_tolerance_contract(row: pd.Series) -> bool:
    return _paper_acceptance_tolerance_value_is_loose(
        row.get("absolute_tolerance"),
        threshold=_PAPER_ACCEPTANCE_ABSOLUTE_TOLERANCE,
    ) or _paper_acceptance_tolerance_value_is_loose(
        row.get("z_tolerance"),
        threshold=_PAPER_ACCEPTANCE_Z_TOLERANCE,
    )


def _paper_acceptance_row_has_missing_tolerance_contract(row: pd.Series) -> bool:
    absolute_tolerance = row.get("absolute_tolerance")
    z_tolerance = row.get("z_tolerance")
    if absolute_tolerance is None or z_tolerance is None:
        return True
    try:
        if bool(pd.isna(absolute_tolerance)) or bool(pd.isna(z_tolerance)):
            return True
    except (TypeError, ValueError):
        pass
    return False


def _paper_acceptance_tolerance_value_is_loose(
    value: Any,
    *,
    threshold: float,
) -> bool:
    if value is None:
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        pass
    try:
        return float(value) > threshold + 1e-12
    except (TypeError, ValueError):
        return False


def _evidence_value_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    try:
        return not bool(pd.isna(value))
    except (TypeError, ValueError):
        return True


def _is_positive_numeric_value(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _active_paper_acceptance_blocking_conditions(
    blocking_conditions: dict[str, int],
) -> dict[str, int]:
    return {
        key: value
        for key, value in blocking_conditions.items()
        if int(value) != 0 and _paper_acceptance_condition_is_blocking(key)
    }


def _paper_acceptance_condition_is_blocking(condition: str) -> bool:
    return condition != "cell_count_policy_size_risk_rows"


def _paper_acceptance_blocker_rows_from_conditions(
    blocking_conditions: dict[str, int],
    *,
    evidence_by_condition: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        _with_blocker_evidence(
            {
            "condition": condition,
            "value": value,
            "blocked": int(value) != 0
            and _paper_acceptance_condition_is_blocking(condition),
            "category": _paper_acceptance_blocker_category(condition),
            "resolution_action": _paper_acceptance_blocker_resolution_action(condition),
            },
            evidence_by_condition,
        )
        for condition, value in blocking_conditions.items()
    )


def _with_blocker_evidence(
    row: dict[str, Any],
    evidence_by_condition: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    if evidence_by_condition is None:
        return row
    evidence = evidence_by_condition.get(str(row.get("condition")), {})
    return _json_safe_export_payload({**row, "evidence": dict(evidence)})


def _paper_acceptance_blocker_evidence_from_summaries(
    *,
    paper_coverage: dict[str, Any],
    blocked_reason_counts: dict[str, int],
    data_source_summary: dict[str, Any],
    precision_budget: dict[str, Any],
    replication_budget: dict[str, Any],
    bootstrap_budget: dict[str, Any],
    cell_count_policy: dict[str, Any],
    execution_summary: dict[str, Any],
    tolerance_contract: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    tolerance_contract = {} if tolerance_contract is None else dict(tolerance_contract)
    return {
        "paper_coverage_unknown": {
            "paper_coverage_known": paper_coverage.get("paper_coverage_known"),
            "paper_result_rows": paper_coverage.get("paper_result_rows"),
        },
        "paper_coverage_shortfall_rows": {
            "paper_result_rows": paper_coverage.get("paper_result_rows"),
            "covered_result_rows": paper_coverage.get("covered_result_rows"),
            "paper_coverage_complete": paper_coverage.get("paper_coverage_complete"),
        },
        "blocked_rows": {
            "blocked_reason_counts": dict(blocked_reason_counts),
        },
        "failed_executed_rows": {
            "executed_rows": execution_summary.get("executed_rows"),
            "passed_executed_rows": execution_summary.get("passed_executed_rows"),
            "failed_executed_rows": execution_summary.get("failed_executed_rows"),
        },
        "data_source_blocked_designs": {
            "data_source_blocked_designs": data_source_summary.get("data_source_blocked_designs"),
            "data_source_blocking_reasons": data_source_summary.get(
                "data_source_blocking_reasons",
                {},
            ),
        },
        "target_replication_shortfall_rows": {
            "target_precision_rows": precision_budget.get("target_precision_rows"),
            "target_tolerance_below_error_band_rows": precision_budget.get(
                "target_tolerance_below_error_band_rows"
            ),
            "max_target_mc_error_band": precision_budget.get("max_target_mc_error_band"),
            "max_target_required_replications": replication_budget.get(
                "max_target_required_replications"
            ),
            "max_target_replication_shortfall": replication_budget.get(
                "max_target_replication_shortfall"
            ),
        },
        "source_mixture_replication_shortfall_rows": {
            "source_mixture_precision_rows": precision_budget.get(
                "source_mixture_precision_rows"
            ),
            "source_mixture_tolerance_below_error_band_rows": precision_budget.get(
                "source_mixture_tolerance_below_error_band_rows"
            ),
            "max_source_mixture_mc_error_band": precision_budget.get(
                "max_source_mixture_mc_error_band"
            ),
            "max_source_mixture_required_replications": replication_budget.get(
                "max_source_mixture_required_replications"
            ),
            "max_source_mixture_replication_shortfall": replication_budget.get(
                "max_source_mixture_replication_shortfall"
            ),
        },
        "bootstrap_replication_shortfall_rows": {
            "bootstrap_required_rows": bootstrap_budget.get("bootstrap_required_rows"),
            "bootstrap_shortfall_rows": bootstrap_budget.get("bootstrap_shortfall_rows"),
            "paper_bootstrap_replications": bootstrap_budget.get(
                "paper_bootstrap_replications"
            ),
            "min_planned_bootstrap_replications": bootstrap_budget.get(
                "min_planned_bootstrap_replications"
            ),
            "max_bootstrap_replication_shortfall": bootstrap_budget.get(
                "max_bootstrap_replication_shortfall"
            ),
        },
        "documented_tolerance_contract_missing": {
            "tolerance_contract_status": tolerance_contract.get(
                "tolerance_contract_status"
            ),
            "paper_target_absolute_tolerance": tolerance_contract.get(
                "paper_target_absolute_tolerance"
            ),
            "paper_z_tolerance": tolerance_contract.get("paper_z_tolerance"),
            "max_target_absolute_tolerance": tolerance_contract.get(
                "max_target_absolute_tolerance"
            ),
            "max_target_z_tolerance": tolerance_contract.get(
                "max_target_z_tolerance"
            ),
            "loose_target_absolute_tolerance_rows": tolerance_contract.get(
                "loose_target_absolute_tolerance_rows"
            ),
            "loose_target_z_tolerance_rows": tolerance_contract.get(
                "loose_target_z_tolerance_rows"
            ),
            "loose_target_tolerance_rows": tolerance_contract.get(
                "loose_target_tolerance_rows"
            ),
        },
        "cell_count_policy_size_risk_rows": {
            "cell_count_policy_rows": cell_count_policy.get("cell_count_policy_rows"),
            "cell_count_policy_size_risk_rows": cell_count_policy.get(
                "cell_count_policy_size_risk_rows"
            ),
            "min_target_median_independent_clusters_per_cell": cell_count_policy.get(
                "min_target_median_independent_clusters_per_cell"
            ),
            "size_risk_preview": cell_count_policy.get("size_risk_preview", []),
        },
        "benchmark_run_not_executed": {
            "stage": "readiness",
            "data_sources_ready": execution_summary.get("data_sources_ready"),
            "executable_rows": execution_summary.get("executable_rows"),
            "planned_draws": execution_summary.get("planned_draws"),
        },
    }


def _paper_acceptance_blocker_category(condition: str) -> str:
    return {
        "paper_coverage_unknown": "paper_coverage",
        "paper_coverage_shortfall_rows": "paper_coverage",
        "blocked_rows": "implementation_scope",
        "failed_executed_rows": "executed_results",
        "data_source_blocked_designs": "data_source_preflight",
        "target_replication_shortfall_rows": "replication_budget",
        "source_mixture_replication_shortfall_rows": "replication_budget",
        "bootstrap_replication_shortfall_rows": "bootstrap_budget",
        "documented_tolerance_contract_missing": "tolerance_contract",
        "cell_count_policy_size_risk_rows": "cell_count_diagnostics",
        "benchmark_run_not_executed": "execution_state",
    }.get(condition, "unknown")


def _paper_acceptance_blocker_resolution_action(condition: str) -> str:
    return {
        "paper_coverage_unknown": "load_complete_paper_contract",
        "paper_coverage_shortfall_rows": "run_full_paper_plan",
        "blocked_rows": "implement_blocked_paper_rows",
        "failed_executed_rows": "inspect_failed_executed_rows",
        "data_source_blocked_designs": "fix_data_sources",
        "target_replication_shortfall_rows": "increase_replications_or_document_tolerance",
        "source_mixture_replication_shortfall_rows": "increase_replications_or_document_tolerance",
        "bootstrap_replication_shortfall_rows": "increase_bootstrap_replications",
        "documented_tolerance_contract_missing": "restore_paper_tolerance_or_document_equivalent_contract",
        "cell_count_policy_size_risk_rows": "surface_size_risk_in_acceptance_diagnostics",
        "benchmark_run_not_executed": "run_manifest",
    }.get(condition, "inspect_blocker")


def _precision_budget_summary_from_frames(
    *,
    target_frame: pd.DataFrame,
    source_mixture_frame: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "target_precision_rows": _numeric_column_count(target_frame, "target_mc_standard_error"),
        "source_mixture_precision_rows": _numeric_column_count(
            source_mixture_frame,
            "source_mixture_mc_standard_error",
        ),
        "target_tolerance_below_error_band_rows": _truthy_column_count(
            target_frame,
            "target_tolerance_below_error_band",
        ),
        "source_mixture_tolerance_below_error_band_rows": _truthy_column_count(
            source_mixture_frame,
            "source_mixture_tolerance_below_error_band",
        ),
        "max_target_mc_standard_error": _max_numeric_column(
            target_frame,
            "target_mc_standard_error",
        ),
        "max_source_mixture_mc_standard_error": _max_numeric_column(
            source_mixture_frame,
            "source_mixture_mc_standard_error",
        ),
        "max_target_mc_error_band": _max_numeric_column(target_frame, "target_mc_error_band"),
        "max_source_mixture_mc_error_band": _max_numeric_column(
            source_mixture_frame,
            "source_mixture_mc_error_band",
        ),
    }


def _replication_budget_summary_from_frame(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "target_budget_rows": _numeric_column_count(
            frame,
            "target_required_replications_for_tolerance",
        ),
        "source_mixture_budget_rows": _numeric_column_count(
            frame,
            "source_mixture_required_replications_for_tolerance",
        ),
        "target_shortfall_rows": _positive_numeric_column_count(
            frame,
            "target_replication_shortfall",
        ),
        "source_mixture_shortfall_rows": _positive_numeric_column_count(
            frame,
            "source_mixture_replication_shortfall",
        ),
        "max_target_required_replications": _max_numeric_column(
            frame,
            "target_required_replications_for_tolerance",
        ),
        "max_source_mixture_required_replications": _max_numeric_column(
            frame,
            "source_mixture_required_replications_for_tolerance",
        ),
        "max_target_replication_shortfall": _max_numeric_column(
            frame,
            "target_replication_shortfall",
        ),
        "max_source_mixture_replication_shortfall": _max_numeric_column(
            frame,
            "source_mixture_replication_shortfall",
        ),
    }


def _bootstrap_budget_summary_from_frame(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or "bootstrap_required" not in frame:
        return {
            "bootstrap_required_rows": 0,
            "bootstrap_shortfall_rows": 0,
            "paper_bootstrap_replications": _PAPER_BOOTSTRAP_REPLICATIONS,
            "min_planned_bootstrap_replications": None,
            "max_planned_bootstrap_replications": None,
            "max_bootstrap_replication_shortfall": None,
        }

    required = frame["bootstrap_required"].map(_truthy_value)
    required_frame = frame.loc[required]
    shortfall = pd.to_numeric(
        required_frame.get("bootstrap_replication_shortfall", pd.Series(dtype=float)),
        errors="coerce",
    )
    planned = pd.to_numeric(
        required_frame.get("planned_bootstrap_replications", pd.Series(dtype=float)),
        errors="coerce",
    )
    valid_planned = planned.dropna()
    valid_shortfall = shortfall.dropna()
    return {
        "bootstrap_required_rows": int(required.sum()),
        "bootstrap_shortfall_rows": int(valid_shortfall.gt(0).sum()),
        "paper_bootstrap_replications": _PAPER_BOOTSTRAP_REPLICATIONS,
        "min_planned_bootstrap_replications": (
            None if valid_planned.empty else int(valid_planned.min())
        ),
        "max_planned_bootstrap_replications": (
            None if valid_planned.empty else int(valid_planned.max())
        ),
        "max_bootstrap_replication_shortfall": (
            None if valid_shortfall.empty else int(valid_shortfall.max())
        ),
    }


def _cell_count_policy_summary_from_frame(frame: pd.DataFrame) -> dict[str, Any]:
    preview_columns = (
        "status",
        "table",
        "design",
        "mediator",
        "clusters",
        "bins",
        "t",
        "target_median_independent_clusters_per_cell",
        "cell_count_size_risk_threshold",
        "bin_policy",
        "blocked_reason",
    )
    if frame.empty:
        return {
            "cell_count_policy_rows": 0,
            "cell_count_policy_size_risk_rows": 0,
            "cell_count_recommended_rows": 0,
            "cell_count_policy_unavailable_rows": 0,
            "min_target_median_independent_clusters_per_cell": None,
            "size_risk_preview": [],
        }

    available = (
        frame["cell_count_policy_available"].map(_truthy_value)
        if "cell_count_policy_available" in frame
        else pd.Series(False, index=frame.index)
    )
    size_risk = (
        frame["cell_count_policy_size_risk"].map(_truthy_value)
        if "cell_count_policy_size_risk" in frame
        else pd.Series(False, index=frame.index)
    )
    recommended = (
        frame["recommended_by_cell_count"].map(_truthy_value)
        if "recommended_by_cell_count" in frame
        else pd.Series(False, index=frame.index)
    )
    size_risk_frame = frame.loc[size_risk]
    return {
        "cell_count_policy_rows": int(available.sum()),
        "cell_count_policy_size_risk_rows": int(size_risk.sum()),
        "cell_count_recommended_rows": int(recommended.sum()),
        "cell_count_policy_unavailable_rows": int((~available).sum()),
        "min_target_median_independent_clusters_per_cell": _numeric_min_or_max(
            frame.loc[available, "target_median_independent_clusters_per_cell"],
            choose="min",
        )
        if "target_median_independent_clusters_per_cell" in frame
        else None,
        "size_risk_preview": size_risk_frame.loc[
            :,
            [column for column in preview_columns if column in size_risk_frame],
        ].head(5).to_dict("records"),
    }


def _required_replications_for_binomial_precision(
    *,
    probability: float,
    absolute_tolerance: float | None,
    z_tolerance: float | None,
    trials_per_replication: int | None,
) -> int | float | None:
    if absolute_tolerance is None or z_tolerance is None or trials_per_replication is None:
        return None
    if trials_per_replication <= 0:
        return None
    variance = float(probability) * (1.0 - float(probability))
    if variance <= 0:
        return 1
    if absolute_tolerance == 0:
        return math.inf
    required_trials = variance * (float(z_tolerance) / float(absolute_tolerance)) ** 2
    return int(math.ceil(required_trials / int(trials_per_replication)))


def _replication_shortfall(
    *,
    required_replications: int | float | None,
    planned_replications: int,
) -> int | float | None:
    if required_replications is None:
        return None
    if math.isinf(float(required_replications)):
        return math.inf
    return max(0, int(required_replications) - int(planned_replications))


def _truthy_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _truthy_value(value.item())
        return value.size > 0
    if isinstance(value, (pd.Series, pd.Index)):
        return len(value) > 0
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    missing = pd.isna(value)
    if isinstance(missing, (np.ndarray, pd.Series, pd.Index, list, tuple)):
        return bool(value)
    return bool(value) if not missing else False


def _target_rejection_rate_standard_error(target_rate: float, *, replications: int) -> float:
    if replications <= 0:
        return math.nan
    return float(math.sqrt(target_rate * (1.0 - target_rate) / replications))


def _source_mixture_precision_summary_from_plan_cells(
    cells: tuple[MonteCarloBenchmarkCell, ...],
    diagnostics: tuple[MonteCarloBenchmarkDataSourceDiagnostic, ...],
    *,
    replications: int,
    z_tolerance: float | None,
    source_mixture_absolute_tolerance: float | None,
) -> dict[str, Any]:
    diagnostics_by_key = _diagnostics_by_data_source_key(diagnostics)
    standard_errors = [
        _source_mixture_standard_error(
            cell=cell,
            replications=replications,
            diagnostic=_diagnostic_for_benchmark_cell(diagnostics_by_key, cell),
        )
        for cell in cells
    ]
    return _source_mixture_precision_summary(
        standard_errors,
        z_tolerance=z_tolerance,
        source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
    )


def _source_mixture_precision_summary_from_scheduled_cells(
    cells: tuple[MonteCarloBenchmarkPlanRerunCell, ...],
    diagnostics: tuple[MonteCarloBenchmarkDataSourceDiagnostic, ...],
    *,
    z_tolerance: float | None,
    source_mixture_absolute_tolerance: float | None,
) -> dict[str, Any]:
    diagnostics_by_key = _diagnostics_by_data_source_key(diagnostics)
    standard_errors = [
        _source_mixture_standard_error(
            cell=scheduled_cell.cell,
            replications=scheduled_cell.replications,
            diagnostic=_diagnostic_for_benchmark_cell(
                diagnostics_by_key,
                scheduled_cell.cell,
            ),
        )
        for scheduled_cell in cells
    ]
    return _source_mixture_precision_summary(
        standard_errors,
        z_tolerance=z_tolerance,
        source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
    )


def _source_mixture_standard_error(
    *,
    cell: MonteCarloBenchmarkCell,
    replications: int,
    diagnostic: MonteCarloBenchmarkDataSourceDiagnostic | None,
) -> float | None:
    if diagnostic is None or not diagnostic.ready:
        return None
    if cell.t <= 0.0 or cell.t >= 1.0:
        return 0.0
    if cell.requires_cluster_resampling:
        if cell.clusters is None:
            return None
        trials_per_draw = cell.clusters // 2
    else:
        if diagnostic.treated_rows is None:
            return None
        trials_per_draw = diagnostic.treated_rows
    total_trials = int(trials_per_draw) * int(replications)
    if total_trials <= 0:
        return None
    return math.sqrt(float(cell.t) * (1.0 - float(cell.t)) / total_trials)


def _source_mixture_trials_per_draw(
    *,
    cell: MonteCarloBenchmarkCell,
    diagnostic: MonteCarloBenchmarkDataSourceDiagnostic | None,
) -> int | None:
    if diagnostic is None or not diagnostic.ready:
        return None
    if cell.requires_cluster_resampling:
        if cell.clusters is None:
            return None
        return int(cell.clusters // 2)
    if diagnostic.treated_rows is None:
        return None
    return int(diagnostic.treated_rows)


def _source_mixture_precision_summary(
    standard_errors: list[float | None],
    *,
    z_tolerance: float | None,
    source_mixture_absolute_tolerance: float | None,
) -> dict[str, Any]:
    computable_errors = [standard_error for standard_error in standard_errors if standard_error is not None]
    positive_errors = [standard_error for standard_error in computable_errors if standard_error > 0]
    error_bands = [
        float(z_tolerance) * standard_error
        for standard_error in computable_errors
        if z_tolerance is not None
    ]
    return {
        "source_mixture_precision_rows": len(computable_errors),
        "zero_source_mixture_mc_se_rows": int(sum(standard_error == 0 for standard_error in computable_errors)),
        "min_source_mixture_mc_standard_error": (
            float(min(positive_errors)) if positive_errors else None
        ),
        "max_source_mixture_mc_standard_error": (
            float(max(computable_errors)) if computable_errors else None
        ),
        "max_source_mixture_mc_error_band": (
            float(max(error_bands)) if error_bands else None
        ),
        "source_mixture_tolerance_below_error_band_rows": int(
            sum(
                error_band > float(source_mixture_absolute_tolerance)
                for error_band in error_bands
            )
        )
        if source_mixture_absolute_tolerance is not None
        else 0,
    }


def load_paper_monte_carlo_contracts(
    tables_dir: str | Path | None = None,
) -> MonteCarloContracts:
    """Load the paper's Monte Carlo LaTeX tables as executable acceptance data.

    When `tables_dir` is omitted, the helper auto-discovers the repository's
    `manuscript/sources/arxiv-2404.11739v3/tables` directory from a checkout.
    """

    table_dir = _default_paper_tables_dir() if tables_dir is None else Path(tables_dir)
    result_specs = {
        "table1": ("binary", 5),
        "table2": ("nonbinary", 5),
        "appendix_table1": ("binary", None),
        "appendix_table2": ("nonbinary", None),
    }
    result_rows: list[MonteCarloResultRow] = []
    for table_name, (mediator, default_bins) in result_specs.items():
        result_rows.extend(
            _parse_result_table(
                path=table_dir / f"{table_name}.tex",
                table_name=table_name,
                mediator=mediator,
                default_bins=default_bins,
            )
        )

    cell_counts = _parse_clustered_cell_count(table_dir / "clustered_cell_count.tex")
    return MonteCarloContracts(
        result_rows=tuple(result_rows),
        cell_counts=tuple(cell_counts),
    )


def load_paper_empirical_mixture_benchmark_data_sources(
    fixtures_dir: str | Path | None = None,
) -> dict[str, BinaryEmpiricalMixtureBenchmarkDataSource]:
    """Load repository fixture bindings for the paper empirical-mixture suite.

    Parameters
    ----------
    fixtures_dir : str, Path, or None
        Directory containing the CSV fixture files.  Defaults to the
        repository checkout path.

    Returns
    -------
    dict[str, BinaryEmpiricalMixtureBenchmarkDataSource]
        Mapping from paper data-source label to loaded data source object.

    Raises
    ------
    FileNotFoundError
        If required fixture CSV files are not found.
    """

    input_dir = (
        _default_paper_fixture_inputs_dir()
        if fixtures_dir is None
        else Path(fixtures_dir)
    )
    required_files = {
        "burstzyn": input_dir / "burstzyn_data.csv",
        "baranov": input_dir / "baranov_mother_data.csv",
    }
    missing_files = {
        name: path for name, path in required_files.items() if not path.exists()
    }
    if missing_files:
        missing = ", ".join(f"{name}={path}" for name, path in missing_files.items())
        raise FileNotFoundError(
            "Paper empirical-mixture fixture inputs were not found; "
            f"pass fixtures_dir explicitly. Missing: {missing}."
        )

    burstzyn_data = pd.read_csv(required_files["burstzyn"])
    baranov_data = pd.read_csv(required_files["baranov"])
    return {
        "Bursztyn et al|binary|unclustered": BinaryEmpiricalMixtureBenchmarkDataSource(
            df=burstzyn_data,
            d="condition2",
            m="signed_up_number",
            y="applied_out_fl",
            analysis_frame_columns=("index",),
            expected_complete_case_rows=284,
            expected_control_rows=145,
            expected_treated_rows=139,
        ),
        "Baranov et al|binary|clustered": BinaryEmpiricalMixtureBenchmarkDataSource(
            df=baranov_data,
            d="treat",
            m="grandmother",
            y="motherfinancial",
            cluster="uc",
            expected_complete_case_rows=585,
            expected_control_rows=296,
            expected_treated_rows=289,
            expected_source_clusters=40,
            expected_control_source_clusters=20,
            expected_treated_source_clusters=20,
        ),
        "Baranov et al|nonbinary|clustered": BinaryEmpiricalMixtureBenchmarkDataSource(
            df=baranov_data,
            d="treat",
            m="relationship_husb",
            y="motherfinancial",
            cluster="uc",
            expected_complete_case_rows=568,
            expected_control_rows=288,
            expected_treated_rows=280,
            expected_source_clusters=40,
            expected_control_source_clusters=20,
            expected_treated_source_clusters=20,
        ),
    }


def run_paper_empirical_mixture_benchmark_suite_chunk(
    *,
    tables_dir: str | Path | None = None,
    fixtures_dir: str | Path | None = None,
    seed: int = 20260509,
    cell_start: int = 0,
    cell_stop: int | None = None,
    paper_replications: int = 500,
    slice_replications: int | None = None,
    bootstrap_replications: int = 500,
    method: str | None = None,
    mediator: str | None = None,
    design: str | None = None,
    table: str | None = None,
    clusters: tuple[int | None, ...] | None = None,
    bins: tuple[int | None, ...] | None = None,
    t_values: tuple[float, ...] | None = None,
    alpha: float = 0.05,
    absolute_tolerance: float = 0.025,
    z_tolerance: float = 2.0,
    cell_count_absolute_tolerance: float | None = None,
    source_mixture_absolute_tolerance: float | None = None,
) -> MonteCarloBenchmarkSuiteRunResult:
    """Run one paper empirical-mixture suite chunk from repository fixtures."""

    contracts = load_paper_monte_carlo_contracts(tables_dir)
    data_sources = load_paper_empirical_mixture_benchmark_data_sources(fixtures_dir)
    return contracts.empirical_mixture_benchmark_suite_cell_index_run_result(
        data_sources,
        seed=seed,
        cell_start=cell_start,
        cell_stop=cell_stop,
        paper_replications=paper_replications,
        slice_replications=slice_replications,
        bootstrap_replications=bootstrap_replications,
        method=method,
        mediator=mediator,
        design=design,
        table=table,
        clusters=clusters,
        bins=bins,
        t_values=t_values,
        alpha=alpha,
        absolute_tolerance=absolute_tolerance,
        z_tolerance=z_tolerance,
        cell_count_absolute_tolerance=cell_count_absolute_tolerance,
        source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
    )


def write_paper_empirical_mixture_benchmark_suite_chunk_evidence(
    output_path: str | Path,
    *,
    tables_dir: str | Path | None = None,
    fixtures_dir: str | Path | None = None,
    seed: int = 20260509,
    cell_start: int = 0,
    cell_stop: int | None = None,
    paper_replications: int = 500,
    slice_replications: int | None = None,
    bootstrap_replications: int = 500,
    method: str | None = None,
    mediator: str | None = None,
    design: str | None = None,
    table: str | None = None,
    clusters: tuple[int | None, ...] | None = None,
    bins: tuple[int | None, ...] | None = None,
    t_values: tuple[float, ...] | None = None,
    alpha: float = 0.05,
    absolute_tolerance: float = 0.025,
    z_tolerance: float = 2.0,
    cell_count_absolute_tolerance: float | None = None,
    source_mixture_absolute_tolerance: float | None = None,
    owner: str = "Phase 11 paper empirical-mixture chunk runner",
    rerun_command: str | None = None,
    runtime_seconds: float | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run one suite chunk and persist a strict JSON evidence payload."""

    _require_monte_carlo_writer_overwrite(overwrite)
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists; pass overwrite=True to replace it."
        )

    started_at = time.perf_counter()
    result = run_paper_empirical_mixture_benchmark_suite_chunk(
        tables_dir=tables_dir,
        fixtures_dir=fixtures_dir,
        seed=seed,
        cell_start=cell_start,
        cell_stop=cell_stop,
        paper_replications=paper_replications,
        slice_replications=slice_replications,
        bootstrap_replications=bootstrap_replications,
        method=method,
        mediator=mediator,
        design=design,
        table=table,
        clusters=clusters,
        bins=bins,
        t_values=t_values,
        alpha=alpha,
        absolute_tolerance=absolute_tolerance,
        z_tolerance=z_tolerance,
        cell_count_absolute_tolerance=cell_count_absolute_tolerance,
        source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
    )
    elapsed_seconds = (
        float(runtime_seconds)
        if runtime_seconds is not None
        else time.perf_counter() - started_at
    )
    payload = result.to_dict(
        owner=owner,
        rerun_command=rerun_command,
        runtime_seconds=elapsed_seconds,
    )
    _write_monte_carlo_json_atomic(path, payload)
    return payload


def write_next_paper_empirical_mixture_benchmark_suite_chunk_evidence(
    evidence_dir: str | Path,
    *,
    tables_dir: str | Path | None = None,
    fixtures_dir: str | Path | None = None,
    seed: int = 20260509,
    cell_chunk_size: int = 1,
    paper_replications: int = 500,
    slice_replications: int | None = None,
    bootstrap_replications: int = 500,
    mediator: str | None = None,
    design: str | None = None,
    table: str | None = None,
    clusters: tuple[int | None, ...] | None = None,
    bins: tuple[int | None, ...] | None = None,
    t_values: tuple[float, ...] | None = None,
    alpha: float = 0.05,
    absolute_tolerance: float = 0.025,
    z_tolerance: float = 2.0,
    cell_count_absolute_tolerance: float | None = None,
    source_mixture_absolute_tolerance: float | None = None,
    owner: str = "Phase 11 paper empirical-mixture next chunk runner",
    rerun_command: str | None = None,
    runtime_seconds: float | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run and persist the first incomplete paper suite chunk from evidence."""

    _require_monte_carlo_writer_overwrite(overwrite)
    output_dir = Path(evidence_dir)
    contracts = load_paper_monte_carlo_contracts(tables_dir)
    data_sources = load_paper_empirical_mixture_benchmark_data_sources(fixtures_dir)
    common_kwargs = {
        "seed": seed,
        "cell_chunk_size": cell_chunk_size,
        "paper_replications": paper_replications,
        "slice_replications": slice_replications,
        "bootstrap_replications": bootstrap_replications,
        "mediator": mediator,
        "design": design,
        "table": table,
        "clusters": clusters,
        "bins": bins,
        "t_values": t_values,
        "alpha": alpha,
        "absolute_tolerance": absolute_tolerance,
        "z_tolerance": z_tolerance,
        "cell_count_absolute_tolerance": cell_count_absolute_tolerance,
        "source_mixture_absolute_tolerance": source_mixture_absolute_tolerance,
    }
    schedule = contracts.empirical_mixture_benchmark_suite_cell_index_schedule_frame(
        data_sources,
        evidence_dir=output_dir,
        **common_kwargs,
    )
    if schedule.empty:
        raise ValueError("No paper empirical-mixture suite chunks were scheduled.")

    scheduled_paths = tuple(
        Path(str(path_like)) for path_like in schedule["chunk_evidence_path"].tolist()
    )
    existing_json = (
        tuple(sorted(output_dir.glob("*.json"))) if output_dir.exists() else ()
    )
    unexpected_json = _unexpected_suite_evidence_json_paths(
        existing_json,
        scheduled_paths,
    )
    if unexpected_json:
        raise FileExistsError(
            "Evidence directory contains JSON files outside the current paper suite "
            f"schedule: {tuple(str(path) for path in unexpected_json)}"
        )
    progress_summary, next_chunk_kwargs, scheduled_chunk = _suite_next_chunk_progress_summary(
        schedule=schedule,
        contracts=contracts,
        data_sources=data_sources,
        common_kwargs=common_kwargs,
        overwrite=overwrite,
    )
    if next_chunk_kwargs is None:
        raise ValueError(
            "No incomplete paper empirical-mixture suite chunk remains for the supplied evidence."
        )

    if scheduled_chunk.empty:
        raise RuntimeError(
            "Next paper empirical-mixture chunk is not present in the current schedule."
        )
    chunk_index = int(scheduled_chunk["chunk_index"].iloc[0])
    output_path = Path(str(scheduled_chunk["chunk_evidence_path"].iloc[0]))
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists; pass overwrite=True to replace it."
        )

    result = contracts.empirical_mixture_benchmark_suite_cell_index_run_result(
        data_sources,
        **next_chunk_kwargs,
    )
    payload = result.to_dict(
        owner=owner,
        rerun_command=rerun_command or _suite_cell_index_chunk_call(next_chunk_kwargs),
        runtime_seconds=runtime_seconds,
    )
    payload["chunk_index"] = chunk_index
    payload["chunk_evidence_path"] = str(output_path)
    payload["prior_progress_summary"] = progress_summary
    _write_monte_carlo_json_atomic(output_path, payload)
    post_write_paths = tuple(path for path in scheduled_paths if path.exists())
    post_write_progress_summary = (
        contracts.empirical_mixture_benchmark_suite_export_progress_summary(
            post_write_paths,
            data_sources,
            **common_kwargs,
        )
    )
    post_write_paper_acceptance_gate = (
        contracts.empirical_mixture_benchmark_suite_export_acceptance_gate(
            post_write_paths,
            data_sources,
            **common_kwargs,
        )
    )
    payload["post_write_progress_summary"] = post_write_progress_summary
    payload["post_write_paper_acceptance_gate"] = post_write_paper_acceptance_gate
    payload["post_write_milestone_completion_summary"] = (
        _paper_acceptance_gate_completion_summary(
            post_write_paper_acceptance_gate,
            owner=owner,
            rerun_command=rerun_command
            or _suite_cell_index_chunk_call(next_chunk_kwargs),
        )
    )
    _write_monte_carlo_json_atomic(output_path, payload)
    return payload


def write_next_paper_empirical_mixture_benchmark_suite_archive_evidence(
    evidence_dir: str | Path,
    *,
    tables_dir: str | Path | None = None,
    fixtures_dir: str | Path | None = None,
    seed: int = 20260509,
    cell_chunk_size: int = 1,
    paper_replications: int = 500,
    slice_replications: int | None = None,
    bootstrap_replications: int = 500,
    mediator: str | None = None,
    design: str | None = None,
    table: str | None = None,
    clusters: tuple[int | None, ...] | None = None,
    bins: tuple[int | None, ...] | None = None,
    t_values: tuple[float, ...] | None = None,
    alpha: float = 0.05,
    absolute_tolerance: float = 0.025,
    z_tolerance: float = 2.0,
    cell_count_absolute_tolerance: float | None = None,
    source_mixture_absolute_tolerance: float | None = None,
    milestone_version: str,
    repo_root: str | Path = ".",
    planning_dir: str | Path = ".planning",
    gsd_tools_path: str | Path | None = None,
    owner: str = "Phase 12 next chunk archive evidence runner",
    nominal_alpha: float = 0.05,
    tolerance: float = 0.025,
    rerun_command: str | None = None,
    runtime_seconds: float | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write one next suite chunk, then resolve the live archive gate."""

    payload = write_next_paper_empirical_mixture_benchmark_suite_chunk_evidence(
        evidence_dir,
        tables_dir=tables_dir,
        fixtures_dir=fixtures_dir,
        seed=seed,
        cell_chunk_size=cell_chunk_size,
        paper_replications=paper_replications,
        slice_replications=slice_replications,
        bootstrap_replications=bootstrap_replications,
        mediator=mediator,
        design=design,
        table=table,
        clusters=clusters,
        bins=bins,
        t_values=t_values,
        alpha=alpha,
        absolute_tolerance=absolute_tolerance,
        z_tolerance=z_tolerance,
        cell_count_absolute_tolerance=cell_count_absolute_tolerance,
        source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
        owner=owner,
        rerun_command=rerun_command,
        runtime_seconds=runtime_seconds,
        overwrite=overwrite,
    )
    archive_summary = summarize_paper_empirical_mixture_benchmark_suite_archive_gate(
        evidence_dir,
        tables_dir=tables_dir,
        fixtures_dir=fixtures_dir,
        seed=seed,
        cell_chunk_size=cell_chunk_size,
        paper_replications=paper_replications,
        slice_replications=slice_replications,
        bootstrap_replications=bootstrap_replications,
        mediator=mediator,
        design=design,
        table=table,
        clusters=clusters,
        bins=bins,
        t_values=t_values,
        alpha=alpha,
        absolute_tolerance=absolute_tolerance,
        z_tolerance=z_tolerance,
        cell_count_absolute_tolerance=cell_count_absolute_tolerance,
        source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
        milestone_version=milestone_version,
        repo_root=repo_root,
        planning_dir=planning_dir,
        gsd_tools_path=gsd_tools_path,
        owner=owner,
        nominal_alpha=nominal_alpha,
        tolerance=tolerance,
        rerun_command=rerun_command,
    )
    resolved_rerun_command = archive_summary["rerun_command"]
    return _json_safe_export_payload(
        {
            "owner": owner,
            "evidence_dir": str(Path(evidence_dir)),
            "written_chunk_index": payload.get("chunk_index"),
            "written_chunk_evidence_path": payload.get("chunk_evidence_path"),
            "written_chunk_summary": payload.get("summary"),
            "post_write_progress_summary": payload.get("post_write_progress_summary"),
            "post_write_paper_acceptance_gate": payload.get(
                "post_write_paper_acceptance_gate"
            ),
            "post_write_milestone_completion_summary": payload.get(
                "post_write_milestone_completion_summary"
            ),
            "archive_gate_summary": archive_summary,
            "milestone_archive_ready": archive_summary["milestone_archive_ready"],
            "archive_delete_tag_allowed": archive_summary[
                "archive_delete_tag_allowed"
            ],
            "active_blocking_conditions": archive_summary[
                "active_blocking_conditions"
            ],
            "complete_milestone_preflight": archive_summary[
                "complete_milestone_preflight"
            ],
            "next_action": archive_summary["next_action"],
            "rerun_hook": {
                "next_action": archive_summary["next_action"],
                "rerun_command": resolved_rerun_command,
                "evidence_next_action": archive_summary["evidence_next_action"],
                "evidence_next_chunk_kwargs": archive_summary[
                    "evidence_next_chunk_kwargs"
                ],
                "evidence_next_chunk_evidence_path": archive_summary[
                    "evidence_next_chunk_evidence_path"
                ],
                "evidence_next_chunk_rerun_call": archive_summary[
                    "evidence_next_chunk_rerun_call"
                ],
                "evidence_next_chunk_export_call": archive_summary[
                    "evidence_next_chunk_export_call"
                ],
                "archive_delete_tag_allowed": archive_summary[
                    "archive_delete_tag_allowed"
                ],
                "blocked_archive_steps": archive_summary["blocked_archive_steps"],
            },
            "rerun_command": resolved_rerun_command,
            "exit_criteria": archive_summary["exit_criteria"],
        }
    )


def summarize_paper_empirical_mixture_benchmark_suite_evidence(
    evidence_dir: str | Path,
    *,
    tables_dir: str | Path | None = None,
    fixtures_dir: str | Path | None = None,
    seed: int = 20260509,
    cell_chunk_size: int = 5,
    paper_replications: int = 500,
    slice_replications: int | None = None,
    bootstrap_replications: int = 500,
    mediator: str | None = None,
    design: str | None = None,
    table: str | None = None,
    clusters: tuple[int | None, ...] | None = None,
    bins: tuple[int | None, ...] | None = None,
    t_values: tuple[float, ...] | None = None,
    alpha: float = 0.05,
    absolute_tolerance: float = 0.025,
    z_tolerance: float = 2.0,
    cell_count_absolute_tolerance: float | None = None,
    source_mixture_absolute_tolerance: float | None = None,
    owner: str = "Phase 11 paper empirical-mixture evidence directory summary",
    rerun_command: str | None = None,
) -> dict[str, Any]:
    """Summarize a strict JSON evidence directory without executing new chunks.

    Scans *evidence_dir* for persisted chunk JSON files, matches them against
    the scheduled paper cells, and produces a summary dict with progress,
    acceptance gate status, and any staleness warnings.

    Parameters
    ----------
    evidence_dir : str or Path
        Directory containing chunk-evidence JSON files.
    tables_dir : str, Path, or None
        LaTeX table directory.
    fixtures_dir : str, Path, or None
        CSV fixture directory.
    seed : int
        Master schedule seed.
    cell_chunk_size : int
        Cells per chunk.
    paper_replications : int
        Target replications per cell.
    slice_replications : int or None
        Shard replications.
    bootstrap_replications : int
        Bootstrap replications for CR cells.
    mediator, design, table, clusters, bins, t_values
        Optional cell filters.
    alpha : float
        Nominal significance level.
    absolute_tolerance : float
        Acceptance tolerance.
    z_tolerance : float
        Z-score tolerance.
    cell_count_absolute_tolerance : float or None
        Cell-count tolerance.
    source_mixture_absolute_tolerance : float or None
        Source-mixture tolerance.
    owner : str
        Owner label.
    rerun_command : str or None
        Reproduction command.

    Returns
    -------
    dict[str, Any]
        Summary including progress counts, gate status, evidence file list,
        and any errors or staleness indicators.
    """

    output_dir = Path(evidence_dir)
    contracts = load_paper_monte_carlo_contracts(tables_dir)
    data_sources = load_paper_empirical_mixture_benchmark_data_sources(fixtures_dir)
    common_kwargs = {
        "seed": seed,
        "cell_chunk_size": cell_chunk_size,
        "paper_replications": paper_replications,
        "slice_replications": slice_replications,
        "bootstrap_replications": bootstrap_replications,
        "mediator": mediator,
        "design": design,
        "table": table,
        "clusters": clusters,
        "bins": bins,
        "t_values": t_values,
        "alpha": alpha,
        "absolute_tolerance": absolute_tolerance,
        "z_tolerance": z_tolerance,
        "cell_count_absolute_tolerance": cell_count_absolute_tolerance,
        "source_mixture_absolute_tolerance": source_mixture_absolute_tolerance,
    }
    schedule = contracts.empirical_mixture_benchmark_suite_cell_index_schedule_frame(
        data_sources,
        evidence_dir=output_dir,
        **common_kwargs,
    )
    if schedule.empty:
        raise ValueError("No paper empirical-mixture suite chunks were scheduled.")

    scheduled_paths = tuple(
        Path(str(path_like)) for path_like in schedule["chunk_evidence_path"].tolist()
    )
    existing_json = tuple(sorted(output_dir.glob("*.json"))) if output_dir.exists() else ()
    existing_evidence_paths = existing_json
    usable_evidence_paths = existing_evidence_paths
    schedule_summary = contracts.empirical_mixture_benchmark_suite_cell_index_schedule_summary(
        data_sources,
        evidence_dir=output_dir,
        **common_kwargs,
    )
    stale_schedule_error: str | None = None
    if existing_evidence_paths:
        try:
            progress_summary = contracts.empirical_mixture_benchmark_suite_export_progress_summary(
                existing_evidence_paths,
                data_sources,
                **common_kwargs,
            )
            paper_acceptance_gate = contracts.empirical_mixture_benchmark_suite_export_acceptance_gate(
                existing_evidence_paths,
                data_sources,
                **common_kwargs,
            )
            unresolved_frame = contracts.empirical_mixture_benchmark_suite_export_unresolved_frame(
                existing_evidence_paths
            )
        except ValueError as exc:
            if not _is_stale_suite_evidence_error(exc):
                raise
            stale_schedule_error = str(exc)
            usable_evidence_paths = ()
            progress_summary = contracts.empirical_mixture_benchmark_suite_chunk_progress_summary(
                (),
                data_sources,
                **common_kwargs,
            )
            paper_acceptance_gate = contracts.empirical_mixture_benchmark_suite_export_acceptance_gate(
                (),
                data_sources,
                **common_kwargs,
            )
            paper_acceptance_gate = _paper_acceptance_gate_with_stale_evidence(
                paper_acceptance_gate,
                stale_file_count=len(existing_evidence_paths),
            )
            unresolved_frame = contracts.empirical_mixture_benchmark_suite_export_unresolved_frame(
                ()
            )
    else:
        progress_summary = contracts.empirical_mixture_benchmark_suite_chunk_progress_summary(
            (),
            data_sources,
            **common_kwargs,
        )
        paper_acceptance_gate = contracts.empirical_mixture_benchmark_suite_export_acceptance_gate(
            (),
            data_sources,
            **common_kwargs,
        )
        unresolved_frame = contracts.empirical_mixture_benchmark_suite_export_unresolved_frame(
            ()
        )

    completion_summary = _paper_acceptance_gate_completion_summary(
        paper_acceptance_gate,
        owner=owner,
        rerun_command=rerun_command,
    )
    next_chunk_evidence_path = progress_summary.get("next_chunk_evidence_path")
    next_chunk_export_call = progress_summary.get("next_chunk_export_call")
    if (
        next_chunk_evidence_path is None
        and progress_summary.get("next_chunk_kwargs") is not None
        and not schedule.empty
    ):
        next_chunk_kwargs = dict(progress_summary["next_chunk_kwargs"])
        next_chunk_row = schedule.loc[
            schedule["cell_start"].eq(int(next_chunk_kwargs["cell_start"]))
            & schedule["cell_stop"].eq(int(next_chunk_kwargs["cell_stop"]))
        ]
        if not next_chunk_row.empty:
            path_value = next_chunk_row["chunk_evidence_path"].iloc[0]
            call_value = next_chunk_row["chunk_export_call"].iloc[0]
            next_chunk_evidence_path = (
                str(path_value) if _evidence_value_present(path_value) else None
            )
            next_chunk_export_call = (
                str(call_value) if _evidence_value_present(call_value) else None
            )
            progress_summary = {
                **progress_summary,
                "next_chunk_evidence_path": next_chunk_evidence_path,
                "next_chunk_export_call": next_chunk_export_call,
            }
    archive_preflight = _complete_milestone_preflight_decision(False)
    summary = {
        "owner": owner,
        "evidence_dir": str(output_dir),
        "evidence_file_count": len(usable_evidence_paths),
        "evidence_files": tuple(str(path) for path in usable_evidence_paths),
        "stale_evidence_file_count": len(existing_evidence_paths)
        if stale_schedule_error is not None
        else 0,
        "stale_evidence_files": tuple(str(path) for path in existing_evidence_paths)
        if stale_schedule_error is not None
        else (),
        "stale_evidence_error": stale_schedule_error,
        "scheduled_file_count": len(scheduled_paths),
        "schedule_summary": schedule_summary,
        "progress_summary": progress_summary,
        "paper_acceptance_gate": paper_acceptance_gate,
        "unresolved_row_count": int(unresolved_frame.shape[0]),
        "unresolved_root_cause_counts": (
            _value_counts_tuple_or_str(unresolved_frame["unresolved_root_cause"])
            if "unresolved_root_cause" in unresolved_frame
            else {}
        ),
        "unresolved_rows": tuple(unresolved_frame.to_dict("records")),
        "milestone_completion_summary": completion_summary,
        "next_action": (
            "delete_or_replace_stale_paper_evidence_files"
            if stale_schedule_error is not None
            else _suite_evidence_directory_next_action(
                schedule_summary,
                progress_summary,
                completion_summary,
            )
        ),
        "next_chunk_kwargs": progress_summary["next_chunk_kwargs"],
        "next_chunk_rerun_call": progress_summary["next_chunk_rerun_call"],
        "next_chunk_evidence_path": next_chunk_evidence_path,
        "next_chunk_export_call": next_chunk_export_call,
        "evidence_next_chunk_kwargs": progress_summary["next_chunk_kwargs"],
        "evidence_next_chunk_rerun_call": progress_summary["next_chunk_rerun_call"],
        "evidence_next_chunk_evidence_path": next_chunk_evidence_path,
        "evidence_next_chunk_export_call": next_chunk_export_call,
        "archive_delete_tag_allowed": archive_preflight["archive_delete_tag_allowed"],
        "blocked_archive_steps": archive_preflight["blocked_steps"],
        "exit_criteria": "all_chunks_complete and paper_acceptance_gate.gate_passes == True",
    }
    return _json_safe_export_payload(summary)


def summarize_paper_empirical_mixture_benchmark_suite_archive_gate(
    evidence_dir: str | Path,
    *,
    tables_dir: str | Path | None = None,
    fixtures_dir: str | Path | None = None,
    seed: int = 20260509,
    cell_chunk_size: int = 5,
    paper_replications: int = 500,
    slice_replications: int | None = None,
    bootstrap_replications: int = 500,
    mediator: str | None = None,
    design: str | None = None,
    table: str | None = None,
    clusters: tuple[int | None, ...] | None = None,
    bins: tuple[int | None, ...] | None = None,
    t_values: tuple[float, ...] | None = None,
    alpha: float = 0.05,
    absolute_tolerance: float = 0.025,
    z_tolerance: float = 2.0,
    cell_count_absolute_tolerance: float | None = None,
    source_mixture_absolute_tolerance: float | None = None,
    milestone_version: str,
    repo_root: str | Path = ".",
    planning_dir: str | Path = ".planning",
    gsd_tools_path: str | Path | None = None,
    owner: str = "Phase 12 release re-audit and archive readiness",
    nominal_alpha: float = 0.05,
    tolerance: float = 0.025,
    rerun_command: str | None = None,
) -> dict[str, Any]:
    """Summarize saved evidence and resolve the live milestone archive gate.

    Combines the evidence directory summary with a milestone-audit status
    check to determine whether the Monte Carlo evidence meets the archive-
    readiness criteria for the specified milestone version.

    Parameters
    ----------
    evidence_dir : str or Path
        Directory containing chunk-evidence JSON files.
    milestone_version : str
        Target milestone version for archive-gate evaluation.

    Returns
    -------
    dict[str, Any]
        Gate summary including evidence status, milestone audit status,
        and any blocking reasons.
    """

    evidence_summary = summarize_paper_empirical_mixture_benchmark_suite_evidence(
        evidence_dir,
        tables_dir=tables_dir,
        fixtures_dir=fixtures_dir,
        seed=seed,
        cell_chunk_size=cell_chunk_size,
        paper_replications=paper_replications,
        slice_replications=slice_replications,
        bootstrap_replications=bootstrap_replications,
        mediator=mediator,
        design=design,
        table=table,
        clusters=clusters,
        bins=bins,
        t_values=t_values,
        alpha=alpha,
        absolute_tolerance=absolute_tolerance,
        z_tolerance=z_tolerance,
        cell_count_absolute_tolerance=cell_count_absolute_tolerance,
        source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
        owner=owner,
        rerun_command=rerun_command,
    )
    resolved_rerun_command = (
        rerun_command
        or evidence_summary.get("next_chunk_export_call")
        or evidence_summary.get("next_chunk_rerun_call")
    )
    contracts = load_paper_monte_carlo_contracts(tables_dir)
    data_sources = load_paper_empirical_mixture_benchmark_data_sources(fixtures_dir)
    archive_gate = contracts.milestone_archive_export_live_gate_summary(
        evidence_summary["evidence_files"],
        data_sources,
        seed=seed,
        milestone_version=milestone_version,
        repo_root=repo_root,
        planning_dir=planning_dir,
        gsd_tools_path=gsd_tools_path,
        owner=owner,
        nominal_alpha=nominal_alpha,
        tolerance=tolerance,
        rerun_command=resolved_rerun_command,
        cell_chunk_size=cell_chunk_size,
        paper_replications=paper_replications,
        slice_replications=slice_replications,
        bootstrap_replications=bootstrap_replications,
        mediator=mediator,
        design=design,
        table=table,
        clusters=clusters,
        bins=bins,
        t_values=t_values,
        alpha=alpha,
        absolute_tolerance=absolute_tolerance,
        z_tolerance=z_tolerance,
        cell_count_absolute_tolerance=cell_count_absolute_tolerance,
        source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
    )
    evidence_progress = dict(evidence_summary["progress_summary"])
    evidence_gate = dict(evidence_summary["paper_acceptance_gate"])
    evidence_completion = dict(evidence_summary["milestone_completion_summary"])
    archive_blocking_rows = tuple(
        dict(row) for row in archive_gate.get("blocking_condition_rows", ())
    )
    archive_active_blocking_rows = tuple(
        dict(row) for row in archive_blocking_rows if bool(row.get("blocked"))
    )
    complete_milestone_preflight = dict(archive_gate["complete_milestone_preflight"])
    summary = {
        "owner": owner,
        "evidence_dir": evidence_summary["evidence_dir"],
        "evidence_summary": evidence_summary,
        "archive_gate": archive_gate,
        "milestone_archive_ready": archive_gate["milestone_archive_ready"],
        "milestone_audit_status": archive_gate["milestone_audit_status"],
        "roadmap_disk_complete": archive_gate["roadmap_disk_complete"],
        "roadmap_phase_count": archive_gate["roadmap_phase_count"],
        "roadmap_completed_phases": archive_gate["roadmap_completed_phases"],
        "roadmap_progress_percent": archive_gate["roadmap_progress_percent"],
        "milestone_closeout_ready": archive_gate["milestone_closeout_ready"],
        "paper_acceptance_gate_passes": archive_gate["paper_acceptance_gate_passes"],
        "paper_acceptance_stage": evidence_gate.get("stage"),
        "paper_acceptance_next_action": evidence_gate.get("next_action"),
        "paper_result_rows": evidence_gate.get("paper_result_rows"),
        "covered_result_rows": evidence_gate.get("covered_result_rows"),
        "paper_coverage_shortfall_rows": evidence_gate.get("paper_coverage_shortfall_rows"),
        "failed_executed_rows": evidence_gate.get("failed_executed_rows"),
        "blocked_rows": evidence_gate.get("blocked_rows"),
        "evidence_all_chunks_complete": evidence_progress.get("all_chunks_complete"),
        "evidence_executed_result_rows": evidence_progress.get("executed_result_rows"),
        "evidence_scheduled_result_rows": evidence_progress.get("scheduled_result_rows"),
        "evidence_unresolved_row_count": evidence_summary.get("unresolved_row_count"),
        "evidence_milestone_completion_ready": evidence_completion.get(
            "milestone_completion_ready"
        ),
        "method_support_gate_passes": archive_gate["method_support_gate_passes"],
        "active_blocking_conditions": archive_gate["active_blocking_conditions"],
        "active_blocking_condition_count": archive_gate["active_blocking_condition_count"],
        "archive_blocking_condition_rows": archive_blocking_rows,
        "archive_active_blocking_condition_rows": archive_active_blocking_rows,
        "complete_milestone_preflight": complete_milestone_preflight,
        "archive_delete_tag_allowed": complete_milestone_preflight.get(
            "archive_delete_tag_allowed"
        ),
        "blocked_archive_steps": complete_milestone_preflight.get("blocked_steps", ()),
        "evidence_file_count": evidence_summary["evidence_file_count"],
        "scheduled_file_count": evidence_summary["scheduled_file_count"],
        "evidence_next_action": evidence_summary["next_action"],
        "evidence_next_chunk_kwargs": evidence_summary["next_chunk_kwargs"],
        "evidence_next_chunk_rerun_call": evidence_summary["next_chunk_rerun_call"],
        "evidence_next_chunk_evidence_path": evidence_summary[
            "next_chunk_evidence_path"
        ],
        "evidence_next_chunk_export_call": evidence_summary["next_chunk_export_call"],
        "next_action": _suite_archive_summary_next_action(
            evidence_summary=evidence_summary,
            archive_gate=archive_gate,
        ),
        "rerun_command": resolved_rerun_command,
        "exit_criteria": archive_gate["exit_criteria"],
    }
    return _json_safe_export_payload(summary)


def summarize_paper_empirical_mixture_benchmark_suite_archive_blocker_packet(
    evidence_dir: str | Path,
    *,
    tables_dir: str | Path | None = None,
    fixtures_dir: str | Path | None = None,
    seed: int = 20260509,
    cell_chunk_size: int = 5,
    paper_replications: int = 500,
    slice_replications: int | None = None,
    bootstrap_replications: int = 500,
    mediator: str | None = None,
    design: str | None = None,
    table: str | None = None,
    clusters: tuple[int | None, ...] | None = None,
    bins: tuple[int | None, ...] | None = None,
    t_values: tuple[float, ...] | None = None,
    alpha: float = 0.05,
    absolute_tolerance: float = 0.025,
    z_tolerance: float = 2.0,
    cell_count_absolute_tolerance: float | None = None,
    source_mixture_absolute_tolerance: float | None = None,
    milestone_version: str,
    repo_root: str | Path = ".",
    planning_dir: str | Path = ".planning",
    gsd_tools_path: str | Path | None = None,
    owner: str = "Phase 12 release re-audit and archive readiness",
    nominal_alpha: float = 0.05,
    tolerance: float = 0.025,
    rerun_command: str | None = None,
) -> dict[str, Any]:
    """Return an owner-facing archive blocker packet from saved suite evidence.

    Extends the archive gate with structured blocker descriptions suitable
    for surfacing to project owners and CI systems.

    Parameters
    ----------
    evidence_dir : str or Path
        Directory containing chunk-evidence JSON files.
    milestone_version : str
        Target milestone version.

    Returns
    -------
    dict[str, Any]
        Blocker packet with gate status, blocker list, and remediation hints.
    """

    summary = summarize_paper_empirical_mixture_benchmark_suite_archive_gate(
        evidence_dir,
        tables_dir=tables_dir,
        fixtures_dir=fixtures_dir,
        seed=seed,
        cell_chunk_size=cell_chunk_size,
        paper_replications=paper_replications,
        slice_replications=slice_replications,
        bootstrap_replications=bootstrap_replications,
        mediator=mediator,
        design=design,
        table=table,
        clusters=clusters,
        bins=bins,
        t_values=t_values,
        alpha=alpha,
        absolute_tolerance=absolute_tolerance,
        z_tolerance=z_tolerance,
        cell_count_absolute_tolerance=cell_count_absolute_tolerance,
        source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
        milestone_version=milestone_version,
        repo_root=repo_root,
        planning_dir=planning_dir,
        gsd_tools_path=gsd_tools_path,
        owner=owner,
        nominal_alpha=nominal_alpha,
        tolerance=tolerance,
        rerun_command=rerun_command,
    )
    archive = dict(summary["archive_gate"])
    closeout = dict(archive["milestone_closeout_gate_summary"])
    blocked_rows = tuple(
        dict(row) for row in summary.get("archive_active_blocking_condition_rows", ())
    )
    packet = {
        "owner": owner,
        "cleared_same_line_items": {
            "roadmap_analysis_available": archive["roadmap_analysis_available"],
            "roadmap_disk_complete": archive["roadmap_disk_complete"],
            "roadmap_phase_count": 0,
            "roadmap_completed_phases": 0,
            "roadmap_total_plans": 0,
            "roadmap_total_summaries": 0,
            "evidence_file_count": summary["evidence_file_count"],
            "scheduled_file_count": summary["scheduled_file_count"],
        },
        "blocked_same_line_items": {
            "milestone_archive_ready": archive["milestone_archive_ready"],
            "milestone_audit_status": archive["milestone_audit_status"],
            "milestone_audit_passes": archive["milestone_audit_passes"],
            "milestone_closeout_ready": archive["milestone_closeout_ready"],
            "paper_acceptance_gate_passes": archive[
                "paper_acceptance_gate_passes"
            ],
            "method_support_gate_passes": archive["method_support_gate_passes"],
            "active_blocking_conditions": dict(
                archive["active_blocking_conditions"]
            ),
        },
        "verified_boundary": {
            "roadmap_progress_percent": 0,
            "roadmap_incomplete_phases": (),
            "milestone_audit_status": archive["milestone_audit_status"],
            "paper_acceptance_stage": summary["paper_acceptance_stage"],
            "paper_acceptance_next_action": summary["paper_acceptance_next_action"],
            "method_support_next_action": closeout["method_support_next_action"],
            "archive_next_action": archive["next_action"],
        },
        "evidence_checked": {
            "blocked_condition_rows": blocked_rows,
            "blocking_condition_count": len(
                summary.get("archive_blocking_condition_rows", ())
            ),
            "active_closeout_workstreams": tuple(
                dict(row) for row in closeout["active_closeout_workstreams"]
            ),
            "closeout_active_blocking_conditions": dict(
                closeout["active_blocking_conditions"]
            ),
        },
        "paper_acceptance_gate": summary["evidence_summary"]["paper_acceptance_gate"],
        "method_support_gate_summary": closeout["method_support_gate_summary"],
        "milestone_closeout_gate_summary": closeout,
        "milestone_archive_gate_summary": archive,
        "complete_milestone_preflight": archive["complete_milestone_preflight"],
        "rerun_hook": {
            **_archive_gate_rerun_hook(summary),
            "evidence_next_action": summary["evidence_next_action"],
            "evidence_next_chunk_kwargs": summary["evidence_next_chunk_kwargs"],
            "evidence_next_chunk_evidence_path": summary[
                "evidence_next_chunk_evidence_path"
            ],
            "evidence_next_chunk_rerun_call": summary[
                "evidence_next_chunk_rerun_call"
            ],
            "evidence_next_chunk_export_call": summary[
                "evidence_next_chunk_export_call"
            ],
            "evidence_file_count": summary["evidence_file_count"],
            "scheduled_file_count": summary["scheduled_file_count"],
        },
        "rerun_command": summary["rerun_command"],
        "exit_criteria": summary["exit_criteria"],
    }
    return _json_safe_export_payload(packet)


def raise_for_milestone_archive_blockers(
    contracts: MonteCarloContracts,
    paper_acceptance_gate: dict[str, Any],
    *,
    roadmap_analysis: dict[str, Any] | None,
    milestone_audit_status: str | None = None,
    milestone_audit_path: str | Path | None = None,
    milestone_version: str | None = None,
    planning_dir: str | Path = ".planning",
    owner: str = "Phase 7 Monte Carlo verification hardening",
    nominal_alpha: float = 0.05,
    tolerance: float = 0.025,
    rerun_command: str | None = None,
) -> None:
    """Raise if GSD preflight or Monte Carlo release gates block archive/tag."""

    contracts.raise_for_milestone_archive_blockers(
        paper_acceptance_gate,
        roadmap_analysis=roadmap_analysis,
        milestone_audit_status=milestone_audit_status,
        milestone_audit_path=milestone_audit_path,
        milestone_version=milestone_version,
        planning_dir=planning_dir,
        owner=owner,
        nominal_alpha=nominal_alpha,
        tolerance=tolerance,
        rerun_command=rerun_command,
    )


def raise_for_paper_empirical_mixture_benchmark_suite_archive_blockers(
    evidence_dir: str | Path,
    *,
    tables_dir: str | Path | None = None,
    fixtures_dir: str | Path | None = None,
    seed: int = 20260509,
    cell_chunk_size: int = 5,
    paper_replications: int = 500,
    slice_replications: int | None = None,
    bootstrap_replications: int = 500,
    mediator: str | None = None,
    design: str | None = None,
    table: str | None = None,
    clusters: tuple[int | None, ...] | None = None,
    bins: tuple[int | None, ...] | None = None,
    t_values: tuple[float, ...] | None = None,
    alpha: float = 0.05,
    absolute_tolerance: float = 0.025,
    z_tolerance: float = 2.0,
    cell_count_absolute_tolerance: float | None = None,
    source_mixture_absolute_tolerance: float | None = None,
    milestone_version: str,
    repo_root: str | Path = ".",
    planning_dir: str | Path = ".planning",
    gsd_tools_path: str | Path | None = None,
    owner: str = "Phase 12 release re-audit and archive readiness",
    nominal_alpha: float = 0.05,
    tolerance: float = 0.025,
    rerun_command: str | None = None,
) -> dict[str, Any]:
    """Raise if a saved evidence directory is not ready for milestone archival."""

    summary = summarize_paper_empirical_mixture_benchmark_suite_archive_gate(
        evidence_dir,
        tables_dir=tables_dir,
        fixtures_dir=fixtures_dir,
        seed=seed,
        cell_chunk_size=cell_chunk_size,
        paper_replications=paper_replications,
        slice_replications=slice_replications,
        bootstrap_replications=bootstrap_replications,
        mediator=mediator,
        design=design,
        table=table,
        clusters=clusters,
        bins=bins,
        t_values=t_values,
        alpha=alpha,
        absolute_tolerance=absolute_tolerance,
        z_tolerance=z_tolerance,
        cell_count_absolute_tolerance=cell_count_absolute_tolerance,
        source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
        milestone_version=milestone_version,
        repo_root=repo_root,
        planning_dir=planning_dir,
        gsd_tools_path=gsd_tools_path,
        owner=owner,
        nominal_alpha=nominal_alpha,
        tolerance=tolerance,
        rerun_command=rerun_command,
    )
    if bool(summary["milestone_archive_ready"]):
        return summary

    active_preview = tuple(summary.get("archive_active_blocking_condition_rows", ()))[:5]
    preflight = dict(summary.get("complete_milestone_preflight", {}))
    raise AssertionError(
        "Saved paper empirical-mixture evidence is not ready for milestone archival: "
        f"milestone_archive_ready={summary['milestone_archive_ready']!r}, "
        f"paper_acceptance_gate_passes={summary['paper_acceptance_gate_passes']!r}, "
        f"roadmap_disk_complete={summary['roadmap_disk_complete']!r}, "
        f"milestone_audit_status={summary['milestone_audit_status']!r}, "
        f"evidence_file_count={summary['evidence_file_count']!r}, "
        f"scheduled_file_count={summary['scheduled_file_count']!r}, "
        f"evidence_all_chunks_complete={summary['evidence_all_chunks_complete']!r}, "
        f"active_blocking_conditions={summary['active_blocking_conditions']!r}, "
        f"evidence_next_action={summary['evidence_next_action']!r}, "
        f"evidence_next_chunk_kwargs={summary['evidence_next_chunk_kwargs']!r}, "
        f"evidence_next_chunk_evidence_path={summary['evidence_next_chunk_evidence_path']!r}, "
        f"evidence_next_chunk_rerun_call={summary['evidence_next_chunk_rerun_call']!r}, "
        f"evidence_next_chunk_export_call={summary['evidence_next_chunk_export_call']!r}, "
        f"archive_next_action={summary['next_action']!r}, "
        f"archive_delete_tag_allowed={summary['archive_delete_tag_allowed']!r}, "
        f"blocked_archive_steps={summary['blocked_archive_steps']!r}, "
        f"blocking_condition_preview={active_preview!r}, "
        f"rerun_command={summary['rerun_command']!r}, "
        f"exit_criteria={summary['exit_criteria']!r}"
    )


def write_paper_empirical_mixture_benchmark_suite_evidence(
    evidence_dir: str | Path,
    *,
    tables_dir: str | Path | None = None,
    fixtures_dir: str | Path | None = None,
    seed: int = 20260509,
    cell_chunk_size: int = 5,
    paper_replications: int = 500,
    slice_replications: int | None = None,
    bootstrap_replications: int = 500,
    mediator: str | None = None,
    design: str | None = None,
    table: str | None = None,
    clusters: tuple[int | None, ...] | None = None,
    bins: tuple[int | None, ...] | None = None,
    t_values: tuple[float, ...] | None = None,
    alpha: float = 0.05,
    absolute_tolerance: float = 0.025,
    z_tolerance: float = 2.0,
    cell_count_absolute_tolerance: float | None = None,
    source_mixture_absolute_tolerance: float | None = None,
    owner: str = "Phase 11 full paper Monte Carlo acceptance run",
    rerun_command: str | None = None,
    runtime_seconds: float | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the paper suite chunk schedule and persist strict JSON evidence files."""

    _require_monte_carlo_writer_overwrite(overwrite)
    output_dir = Path(evidence_dir)
    contracts = load_paper_monte_carlo_contracts(tables_dir)
    data_sources = load_paper_empirical_mixture_benchmark_data_sources(fixtures_dir)
    common_kwargs = {
        "seed": seed,
        "cell_chunk_size": cell_chunk_size,
        "paper_replications": paper_replications,
        "slice_replications": slice_replications,
        "bootstrap_replications": bootstrap_replications,
        "mediator": mediator,
        "design": design,
        "table": table,
        "clusters": clusters,
        "bins": bins,
        "t_values": t_values,
        "alpha": alpha,
        "absolute_tolerance": absolute_tolerance,
        "z_tolerance": z_tolerance,
        "cell_count_absolute_tolerance": cell_count_absolute_tolerance,
        "source_mixture_absolute_tolerance": source_mixture_absolute_tolerance,
    }
    schedule = contracts.empirical_mixture_benchmark_suite_cell_index_schedule_frame(
        data_sources,
        evidence_dir=output_dir,
        **common_kwargs,
    )
    if schedule.empty:
        raise ValueError("No paper empirical-mixture suite chunks were scheduled.")

    scheduled_paths = tuple(
        Path(str(path_like)) for path_like in schedule["chunk_evidence_path"].tolist()
    )
    existing_json = tuple(sorted(output_dir.glob("*.json"))) if output_dir.exists() else ()
    unexpected_json = _unexpected_suite_evidence_json_paths(
        existing_json,
        scheduled_paths,
    )
    if unexpected_json:
        raise FileExistsError(
            "Evidence directory contains JSON files outside the current paper suite "
            f"schedule: {tuple(str(path) for path in unexpected_json)}"
        )
    existing_scheduled_paths = tuple(path for path in scheduled_paths if path.exists())
    if existing_scheduled_paths and not overwrite:
        raise FileExistsError(
            "Paper suite evidence files already exist; pass overwrite=True to replace "
            f"them: {tuple(str(path) for path in existing_scheduled_paths[:5])}"
        )

    started_at = time.perf_counter()
    chunk_results: list[MonteCarloBenchmarkSuiteRunResult] = []
    chunk_paths: list[str] = []
    for row in schedule.to_dict("records"):
        chunk_index = int(row["chunk_index"])
        output_path = Path(str(row["chunk_evidence_path"]))
        result = run_paper_empirical_mixture_benchmark_suite_chunk(
            tables_dir=tables_dir,
            fixtures_dir=fixtures_dir,
            seed=seed,
            cell_start=int(row["cell_start"]),
            cell_stop=int(row["cell_stop"]),
            paper_replications=paper_replications,
            slice_replications=slice_replications,
            bootstrap_replications=bootstrap_replications,
            mediator=mediator,
            design=design,
            table=table,
            clusters=clusters,
            bins=bins,
            t_values=t_values,
            alpha=alpha,
            absolute_tolerance=absolute_tolerance,
            z_tolerance=z_tolerance,
            cell_count_absolute_tolerance=cell_count_absolute_tolerance,
            source_mixture_absolute_tolerance=source_mixture_absolute_tolerance,
        )
        chunk_payload = result.to_dict(
            owner=f"{owner} chunk {chunk_index:03d}",
            rerun_command=str(row["chunk_rerun_call"]),
        )
        chunk_payload["chunk_index"] = chunk_index
        chunk_payload["chunk_evidence_path"] = str(output_path)
        _write_monte_carlo_json_atomic(output_path, chunk_payload)
        chunk_results.append(result)
        chunk_paths.append(str(output_path))

    combined_result = MonteCarloBenchmarkSuiteRunResult.from_chunk_results(
        tuple(chunk_results)
    )
    elapsed_seconds = (
        float(runtime_seconds)
        if runtime_seconds is not None
        else time.perf_counter() - started_at
    )
    suite_payload = combined_result.to_dict(
        owner=owner,
        rerun_command=rerun_command,
        runtime_seconds=elapsed_seconds,
    )
    schedule_summary = contracts.empirical_mixture_benchmark_suite_cell_index_schedule_summary(
        data_sources,
        evidence_dir=output_dir,
        **common_kwargs,
    )
    post_write_paths = tuple(Path(path) for path in chunk_paths)
    post_write_progress_summary = contracts.empirical_mixture_benchmark_suite_export_progress_summary(
        post_write_paths,
        data_sources,
        **common_kwargs,
    )
    post_write_paper_acceptance_gate = contracts.empirical_mixture_benchmark_suite_export_acceptance_gate(
        post_write_paths,
        data_sources,
        **common_kwargs,
    )
    suite_payload.update(
        {
            "evidence_dir": str(output_dir),
            "chunk_count": int(schedule.shape[0]),
            "chunk_paths": chunk_paths,
            "schedule_summary": schedule_summary,
            "post_write_progress_summary": post_write_progress_summary,
            "post_write_paper_acceptance_gate": post_write_paper_acceptance_gate,
            "post_write_milestone_completion_summary": _paper_acceptance_gate_completion_summary(
                post_write_paper_acceptance_gate,
                owner=owner,
                rerun_command=rerun_command,
            ),
        }
    )
    return _json_safe_export_payload(suite_payload)


def _unexpected_suite_evidence_json_paths(
    existing_json: tuple[Path, ...],
    scheduled_paths: tuple[Path, ...],
) -> tuple[Path, ...]:
    scheduled_path_set = set(scheduled_paths)
    return tuple(path for path in existing_json if path not in scheduled_path_set)


def _require_monte_carlo_writer_overwrite(overwrite: Any) -> None:
    if not isinstance(overwrite, bool):
        raise ValueError("paper Monte Carlo writer overwrite must be boolean.")


def _reject_monte_carlo_json_constant(value: str) -> None:
    raise ValueError(f"paper Monte Carlo report must be strict JSON; found {value}.")


def _reject_roadmap_analysis_json_constant(value: str) -> None:
    raise ValueError(f"roadmap analyze output must be strict JSON; found {value}.")


def _suite_next_chunk_progress_summary(
    *,
    schedule: pd.DataFrame,
    contracts: "MonteCarloContracts",
    data_sources: dict[str, "BinaryEmpiricalMixtureBenchmarkDataSource"],
    common_kwargs: dict[str, Any],
    overwrite: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None, pd.DataFrame]:
    """Return the current progress summary and next scheduled chunk to write."""

    def _with_next_chunk_export_fields(
        summary: dict[str, Any],
        schedule_row: pd.Series,
    ) -> dict[str, Any]:
        next_chunk_evidence_path = schedule_row.get("chunk_evidence_path")
        next_chunk_export_call = schedule_row.get("chunk_export_call")
        return {
            **summary,
            "next_chunk_evidence_path": (
                str(next_chunk_evidence_path)
                if _evidence_value_present(next_chunk_evidence_path)
                else None
            ),
            "next_chunk_export_call": (
                str(next_chunk_export_call)
                if _evidence_value_present(next_chunk_export_call)
                else None
            ),
        }

    valid_prefix: list[Path] = []
    for _, schedule_row in schedule.iterrows():
        chunk_path = Path(str(schedule_row["chunk_evidence_path"]))
        scheduled_chunk = schedule.loc[
            schedule["chunk_evidence_path"].map(str).eq(str(chunk_path))
        ]
        if not chunk_path.exists():
            prior_summary = (
                contracts.empirical_mixture_benchmark_suite_export_progress_summary(
                    tuple(valid_prefix),
                    data_sources,
                    **common_kwargs,
                )
                if valid_prefix
                else contracts.empirical_mixture_benchmark_suite_chunk_progress_summary(
                    (),
                    data_sources,
                    **common_kwargs,
                )
            )
            return (
                _with_next_chunk_export_fields(prior_summary, schedule_row),
                dict(schedule_row["chunk_rerun_kwargs"]),
                scheduled_chunk,
            )

        candidate_paths = tuple(valid_prefix + [chunk_path])
        try:
            candidate_summary = contracts.empirical_mixture_benchmark_suite_export_progress_summary(
                candidate_paths,
                data_sources,
                **common_kwargs,
            )
        except ValueError as exc:
            if not overwrite or not _is_stale_suite_evidence_error(exc):
                raise
            prior_summary = (
                contracts.empirical_mixture_benchmark_suite_export_progress_summary(
                    tuple(valid_prefix),
                    data_sources,
                    **common_kwargs,
                )
                if valid_prefix
                else contracts.empirical_mixture_benchmark_suite_chunk_progress_summary(
                    (),
                    data_sources,
                    **common_kwargs,
                )
            )
            return (
                _with_next_chunk_export_fields(prior_summary, schedule_row),
                dict(schedule_row["chunk_rerun_kwargs"]),
                scheduled_chunk,
            )

        next_chunk_kwargs = candidate_summary.get("next_chunk_kwargs")
        if next_chunk_kwargs is None:
            valid_prefix.append(chunk_path)
            continue
        if (
            int(next_chunk_kwargs["cell_start"]) == int(schedule_row["cell_start"])
            and int(next_chunk_kwargs["cell_stop"]) == int(schedule_row["cell_stop"])
        ):
            return candidate_summary, dict(next_chunk_kwargs), scheduled_chunk
        valid_prefix.append(chunk_path)

    progress_summary = (
        contracts.empirical_mixture_benchmark_suite_export_progress_summary(
            tuple(valid_prefix),
            data_sources,
            **common_kwargs,
        )
        if valid_prefix
        else contracts.empirical_mixture_benchmark_suite_chunk_progress_summary(
            (),
            data_sources,
            **common_kwargs,
        )
    )
    next_chunk_kwargs = progress_summary.get("next_chunk_kwargs")
    if next_chunk_kwargs is None:
        return progress_summary, None, schedule.iloc[0:0]
    scheduled_chunk = schedule.loc[
        schedule["cell_start"].eq(int(next_chunk_kwargs["cell_start"]))
        & schedule["cell_stop"].eq(int(next_chunk_kwargs["cell_stop"]))
    ]
    return progress_summary, dict(next_chunk_kwargs), scheduled_chunk


def _suite_evidence_directory_next_action(
    schedule_summary: dict[str, Any],
    progress_summary: dict[str, Any],
    completion_summary: dict[str, Any],
) -> str:
    """Choose the directory-level next action from schedule, progress, and gate state."""

    if not bool(schedule_summary.get("hook_ready")):
        return str(schedule_summary["next_action"])
    if not bool(progress_summary.get("all_chunks_complete")):
        return str(progress_summary["next_action"])
    return str(completion_summary["next_action"])


def _is_stale_suite_evidence_error(exc: ValueError) -> bool:
    message = str(exc)
    return (
        "current suite schedule" in message
        or "exported row seeds" in message
        or "actual replications to match planned_draws" in message
        or "actual bootstrap_replications to match planned_bootstrap_replications" in message
    )


def _paper_acceptance_gate_with_stale_evidence(
    paper_acceptance_gate: dict[str, Any],
    *,
    stale_file_count: int,
) -> dict[str, Any]:
    active = dict(paper_acceptance_gate.get("active_blocking_conditions") or {})
    active["stale_schedule_evidence_files"] = stale_file_count
    blocking_rows = list(paper_acceptance_gate.get("blocking_condition_rows") or ())
    blocking_rows.append(
        {
            "blocking_condition": "stale_schedule_evidence_files",
            "count": stale_file_count,
            "resolution_action": "delete_or_replace_stale_paper_evidence_files",
        }
    )
    return {
        **paper_acceptance_gate,
        "active_blocking_conditions": active,
        "active_blocking_condition_count": len(active),
        "blocking_condition_rows": tuple(blocking_rows),
        "next_action": "delete_or_replace_stale_paper_evidence_files",
    }


def _default_paper_tables_dir() -> Path:
    checkout_dir = Path(__file__).resolve().parents[3] / "manuscript/sources/arxiv-2404.11739v3" / "tables"
    if checkout_dir.exists():
        return checkout_dir
    return Path(str(files("testmechs.resources.tables")))


def _default_paper_fixture_inputs_dir() -> Path:
    checkout_dir = Path(__file__).resolve().parents[3] / "tests/python" / "fixtures" / "inputs"
    if checkout_dir.exists():
        return checkout_dir
    return Path(str(files("testmechs.resources.fixtures")))


def _simulate_binary_cs_draw(*, design: BinaryCSMonteCarloDesign, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if design.cluster_count is None:
        treated = rng.binomial(1, design.treatment_probability, size=design.n_obs)
        cluster_ids = None
    else:
        cluster_size = design.n_obs // design.cluster_count
        cluster_ids = np.repeat(np.arange(design.cluster_count), cluster_size)
        treated_by_cluster = np.zeros(design.cluster_count, dtype=int)
        treated_by_cluster[
            : _treated_cluster_count(
                cluster_count=design.cluster_count,
                treatment_probability=design.treatment_probability,
            )
        ] = 1
        rng.shuffle(treated_by_cluster)
        treated = treated_by_cluster[cluster_ids]
    mediator_probability = design.mediator_control_probability + design.mediator_treatment_shift * treated
    mediator = rng.binomial(1, mediator_probability)
    outcome = (
        design.mediator_effect * mediator
        + design.direct_effect * treated
        + rng.normal(loc=0.0, scale=design.outcome_noise_sd, size=design.n_obs)
    )
    draw_df = pd.DataFrame(
        {
            "treated": treated.astype(int),
            "mediator": mediator.astype(int),
            "outcome": outcome,
        }
    )
    if cluster_ids is not None:
        draw_df["cluster"] = cluster_ids.astype(int)
    return draw_df


def _simulate_binary_partial_density_draw(
    *,
    design: BinaryPartialDensityMonteCarloDesign,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    treated = rng.binomial(1, design.p_d, size=design.n_obs)
    mediator_probability = np.where(treated == 1, design.p_m_1, design.p_m_0)
    mediator = rng.binomial(1, mediator_probability)

    outcome = np.empty(design.n_obs, dtype=object)
    probability_lookup = {
        (1, 1): design.p_y_m1d1,
        (1, 0): design.p_y_m1d0,
        (0, 1): design.p_y_m0d1,
        (0, 0): design.p_y_m0d0,
    }
    for index, (d_value, m_value) in enumerate(zip(treated, mediator, strict=True)):
        probabilities = probability_lookup[(int(m_value), int(d_value))]
        outcome[index] = rng.choice(design.y_values, p=np.asarray(probabilities, dtype=float))

    return pd.DataFrame(
        {
            "treated": treated.astype(int),
            "mediator": mediator.astype(int),
            "outcome": outcome,
        }
    )


def _binary_partial_density_expected_cell_probabilities(
    design: BinaryPartialDensityMonteCarloDesign,
) -> pd.DataFrame:
    probability_lookup = {
        (1, 1): design.p_y_m1d1,
        (1, 0): design.p_y_m1d0,
        (0, 1): design.p_y_m0d1,
        (0, 0): design.p_y_m0d0,
    }
    mediator_probability_lookup = {
        (1, 1): design.p_m_1,
        (0, 1): 1.0 - design.p_m_1,
        (1, 0): design.p_m_0,
        (0, 0): 1.0 - design.p_m_0,
    }
    treatment_probability_lookup = {
        1: design.p_d,
        0: 1.0 - design.p_d,
    }

    rows: list[dict[str, Any]] = []
    for d_value in (0, 1):
        for m_value in (0, 1):
            mediator_probability = mediator_probability_lookup[(m_value, d_value)]
            y_probabilities = probability_lookup[(m_value, d_value)]
            for y_value, conditional_y_probability in zip(
                design.y_values,
                y_probabilities,
                strict=True,
            ):
                conditional_probability = mediator_probability * conditional_y_probability
                rows.append(
                    {
                        "treated": d_value,
                        "mediator": m_value,
                        "outcome": y_value,
                        "conditional_probability": float(conditional_probability),
                        "joint_probability": float(
                            treatment_probability_lookup[d_value] * conditional_probability
                        ),
                    }
                )
    return _json_safe_export_frame(rows)


def _simulate_binary_empirical_mixture_draw(
    *,
    design: BinaryEmpiricalMixtureMonteCarloDesign,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, int | None]]:
    rng = np.random.default_rng(seed)
    if design.cluster_count is None:
        return _simulate_binary_empirical_mixture_unit_draw(design=design, rng=rng)
    return _simulate_binary_empirical_mixture_cluster_draw(design=design, rng=rng)


def _simulate_binary_empirical_mixture_unit_draw(
    *,
    design: BinaryEmpiricalMixtureMonteCarloDesign,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, int | None]]:
    if design.arm_assignment == "fixed_observed_arms":
        n_control = design.n_control_per_draw
        n_treated = design.n_treated_per_draw
    elif design.arm_assignment == "iid_bernoulli":
        arm_draws = rng.binomial(
            1,
            float(design.treatment_probability),
            size=design.n_obs_per_draw,
        )
        n_treated = int(arm_draws.sum())
        n_control = int(design.n_obs_per_draw - n_treated)
    else:
        raise ValueError("Unsupported empirical-mixture arm_assignment.")

    control_rows = _sample_pool_rows(design.control_pool, n=n_control, rng=rng)

    treated_source_treated = int(rng.binomial(n_treated, design.t))
    treated_source_control = n_treated - treated_source_treated
    treated_parts: list[pd.DataFrame] = []
    if treated_source_control:
        treated_parts.append(_sample_pool_rows(design.control_pool, n=treated_source_control, rng=rng))
    if treated_source_treated:
        treated_parts.append(_sample_pool_rows(design.treated_pool, n=treated_source_treated, rng=rng))
    treated_rows = pd.concat(treated_parts, ignore_index=True) if treated_parts else design.treated_pool.iloc[[]].copy()

    draw_df = pd.concat(
        [
            _as_simulated_arm(control_rows, treated=0),
            _as_simulated_arm(treated_rows, treated=1),
        ],
        ignore_index=True,
    )
    return draw_df, {
        "treated_source_treated_draws": treated_source_treated,
        "treated_source_control_draws": treated_source_control,
        "treated_source_treated_clusters": None,
        "treated_source_control_clusters": None,
    }


def _simulate_binary_empirical_mixture_cluster_draw(
    *,
    design: BinaryEmpiricalMixtureMonteCarloDesign,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, int | None]]:
    if design.clusters_per_arm is None:
        raise ValueError("Clustered empirical-mixture designs require clusters_per_arm.")

    parts: list[pd.DataFrame] = []
    simulated_cluster = 0
    control_clusters = _sample_source_clusters(
        design.control_pool,
        count=design.clusters_per_arm,
        rng=rng,
    )
    for source_cluster in control_clusters:
        parts.append(
            _source_cluster_as_simulated_cluster(
                design.control_pool,
                source_cluster=source_cluster,
                simulated_cluster=simulated_cluster,
                treated=0,
            )
        )
        simulated_cluster += 1

    treated_source_treated = 0
    treated_source_control = 0
    treated_source_treated_clusters = 0
    treated_source_control_clusters = 0
    for _ in range(design.clusters_per_arm):
        use_treated_source = bool(rng.random() < design.t)
        source_pool = design.treated_pool if use_treated_source else design.control_pool
        source_cluster = _sample_source_clusters(source_pool, count=1, rng=rng)[0]
        cluster_df = _source_cluster_as_simulated_cluster(
            source_pool,
            source_cluster=source_cluster,
            simulated_cluster=simulated_cluster,
            treated=1,
        )
        if use_treated_source:
            treated_source_treated += int(cluster_df.shape[0])
            treated_source_treated_clusters += 1
        else:
            treated_source_control += int(cluster_df.shape[0])
            treated_source_control_clusters += 1
        parts.append(cluster_df)
        simulated_cluster += 1

    return pd.concat(parts, ignore_index=True), {
        "treated_source_treated_draws": treated_source_treated,
        "treated_source_control_draws": treated_source_control,
        "treated_source_treated_clusters": treated_source_treated_clusters,
        "treated_source_control_clusters": treated_source_control_clusters,
    }


def _sample_pool_rows(
    pool: pd.DataFrame,
    *,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if pool.empty:
        raise ValueError("Cannot sample from an empty empirical-mixture pool.")
    positions = rng.integers(0, len(pool), size=n)
    return pool.iloc[positions].reset_index(drop=True).copy()


def _sample_source_clusters(
    pool: pd.DataFrame,
    *,
    count: int,
    rng: np.random.Generator,
) -> list[Any]:
    if "_source_cluster" not in pool:
        raise ValueError("Clustered empirical-mixture pools must contain _source_cluster.")
    source_clusters = _ordered_unique_values(pool["_source_cluster"])
    if not source_clusters:
        raise ValueError("Cannot sample clusters from an empty empirical-mixture pool.")
    positions = rng.integers(0, len(source_clusters), size=count)
    return [source_clusters[int(position)] for position in positions]


def _source_cluster_as_simulated_cluster(
    pool: pd.DataFrame,
    *,
    source_cluster: Any,
    simulated_cluster: int,
    treated: int,
) -> pd.DataFrame:
    rows = pool.loc[pool["_source_cluster"] == source_cluster, ["mediator", "outcome"]].reset_index(drop=True).copy()
    if rows.empty:
        raise ValueError(f"Source cluster {source_cluster!r} is empty.")
    rows.insert(0, "treated", int(treated))
    rows["cluster"] = int(simulated_cluster)
    return rows[["treated", "mediator", "outcome", "cluster"]]


def _as_simulated_arm(pool_rows: pd.DataFrame, *, treated: int) -> pd.DataFrame:
    rows = pool_rows[["mediator", "outcome"]].reset_index(drop=True).copy()
    rows.insert(0, "treated", int(treated))
    return rows[["treated", "mediator", "outcome"]]


def _treated_cluster_count(*, cluster_count: int, treatment_probability: float) -> int:
    return int(math.floor(cluster_count * treatment_probability + 0.5))


def _benchmark_matrix_design_name(
    *,
    table: str,
    design: str,
    mediator: str,
    clusters: int | None,
    bins: int | None,
    t: float,
) -> str:
    cluster_label = "unclustered" if clusters is None else f"{clusters}clusters"
    bins_label = "unbinned" if bins is None else f"{bins}bins"
    return f"{table}-{design}-{mediator}-{cluster_label}-{bins_label}-t{_format_t_for_name(float(t))}"


def _benchmark_cell_paper_identity(cell: MonteCarloBenchmarkCell) -> dict[str, Any]:
    return {
        "table": cell.table,
        "panel": cell.panel,
        "design": cell.design,
        "mediator": cell.mediator,
        "clusters": cell.clusters,
        "bins": cell.bins,
        "t": float(cell.t),
        "method": cell.method,
        "target_rejection_rate": float(cell.target_rejection_rate),
    }


def _benchmark_diagnostic_paper_identity(diagnostic: MonteCarloBenchmarkDiagnostic) -> dict[str, Any]:
    row = diagnostic.row
    return {
        "table": row.table,
        "panel": row.panel,
        "design": row.design,
        "mediator": row.mediator,
        "clusters": row.clusters,
        "bins": row.bins,
        "t": float(row.t),
        "method": diagnostic.method,
        "target_rejection_rate": float(diagnostic.target_rejection_rate),
    }


def _paper_result_cell_key(cell: MonteCarloBenchmarkCell) -> tuple[Any, ...]:
    return (
        cell.table,
        cell.panel,
        cell.design,
        cell.mediator,
        cell.clusters,
        cell.bins,
        _normalized_float_key(cell.t),
        cell.method,
        _normalized_float_key(cell.target_rejection_rate),
    )


def _scheduled_benchmark_cell_sort_key(
    scheduled_cell: MonteCarloBenchmarkPlanRerunCell,
) -> tuple[Any, ...]:
    cell = scheduled_cell.cell
    missing_index = 10**9
    return (
        cell.paper_row_index if cell.paper_row_index is not None else missing_index,
        cell.benchmark_row_index
        if cell.benchmark_row_index is not None
        else missing_index,
        _paper_result_cell_key(cell),
    )


def _paper_result_mapping_key(row: dict[str, Any], *, method: str) -> tuple[Any, ...]:
    return (
        row.get("table"),
        row.get("panel"),
        row.get("design"),
        row.get("mediator"),
        _coerce_optional_int(row.get("clusters")),
        _coerce_optional_int(row.get("bins")),
        _normalized_float_key(row.get("t")),
        method,
        _normalized_float_key(row.get("target_rejection_rate")),
    )


def _normalized_float_key(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 12)


def _coerce_optional_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _format_t_for_name(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-12):
        return str(int(round(value)))
    return f"{value:g}"


def _draw_cluster_diagnostics(draw_df: pd.DataFrame) -> dict[str, int | None]:
    if "cluster" not in draw_df:
        return {
            "n_clusters_used": None,
            "treated_clusters": None,
            "control_clusters": None,
            "cluster_size": None,
        }

    grouped = draw_df.groupby("cluster", sort=False)
    treatment_levels = grouped["treated"].nunique()
    if bool((treatment_levels > 1).any()):
        raise ValueError("Clustered Monte Carlo draws require treatment to be fixed within each cluster.")

    cluster_treatment = grouped["treated"].first()
    cluster_sizes = grouped.size()
    return {
        "n_clusters_used": int(len(cluster_treatment)),
        "treated_clusters": int((cluster_treatment == 1).sum()),
        "control_clusters": int((cluster_treatment == 0).sum()),
        "cluster_size": int(cluster_sizes.iloc[0]) if cluster_sizes.nunique() == 1 else None,
    }


def _draw_arm_diagnostics(draw_df: pd.DataFrame) -> dict[str, int]:
    if "treated" not in draw_df:
        raise ValueError("Monte Carlo draws require a treated column.")
    treatment_levels = _ordered_unique_values(draw_df["treated"])
    if tuple(treatment_levels) != (0, 1):
        raise ValueError("Monte Carlo draws require binary treated support normalized to 0/1.")
    control_observations = int((draw_df["treated"] == 0).sum())
    treated_observations = int((draw_df["treated"] == 1).sum())
    if control_observations == 0 or treated_observations == 0:
        raise ValueError("Monte Carlo draws require non-empty treated and control arms.")
    return {
        "control_observations": control_observations,
        "treated_observations": treated_observations,
    }


def _cell_count_summary_from_diagnostics(diagnostics: dict[str, Any]) -> dict[str, float]:
    cell_counts = [float(row["count"]) for row in diagnostics.get("cell_counts", [])]
    independent_counts = [
        float(row["cluster_count"]) for row in diagnostics.get("cluster_counts", [])
    ]
    if not independent_counts:
        independent_counts = cell_counts
    return {
        "median_cell_count": _median_or_zero(cell_counts),
        "median_independent_count_per_cell": _median_or_zero(independent_counts),
    }


def _monte_carlo_full_support_count_summary(
    *,
    draw_df: pd.DataFrame,
    num_y_bins: int | None,
    cluster_column: str | None,
    size_risk_threshold: int = 15,
) -> dict[str, Any]:
    y_processed = draw_df["outcome"]
    if num_y_bins is not None:
        y_processed = discretize_y(draw_df["outcome"], num_bins=num_y_bins)
    elif len(draw_df) / draw_df["outcome"].nunique(dropna=False) <= 30:
        y_processed = discretize_y(draw_df["outcome"], num_bins=5)

    analysis_df = draw_df.copy()
    analysis_df["_tm_y_processed"] = y_processed
    treatment_values = _ordered_unique_values(analysis_df["treated"])
    mediator_values = _ordered_unique_values(analysis_df["mediator"])
    y_values = _series_support_values(analysis_df["_tm_y_processed"])
    full_index = pd.MultiIndex.from_product(
        [treatment_values, mediator_values, y_values],
        names=["treated", "mediator", "_tm_y_processed"],
    )

    cell_counts = (
        analysis_df.groupby(
            ["treated", "mediator", "_tm_y_processed"],
            dropna=False,
            observed=False,
            sort=False,
        )
        .size()
        .reindex(full_index, fill_value=0)
        .astype(int)
    )
    if cluster_column is None:
        independent_counts = cell_counts
    else:
        independent_counts = (
            analysis_df.groupby(
                ["treated", "mediator", "_tm_y_processed"],
                dropna=False,
                observed=False,
                sort=False,
            )[cluster_column]
            .nunique()
            .reindex(full_index, fill_value=0)
            .astype(int)
        )

    min_cell_count = int(cell_counts.min()) if len(cell_counts) else 0
    min_cluster_count = int(independent_counts.min()) if len(independent_counts) else 0
    median_cell_count = _median_or_zero([float(value) for value in cell_counts.tolist()])
    median_independent_count_per_cell = _median_or_zero(
        [float(value) for value in independent_counts.tolist()]
    )
    empty_cells = cell_counts.loc[cell_counts == 0]
    empty_cluster_cells = independent_counts.loc[independent_counts == 0]
    small_cells = cell_counts.loc[cell_counts < size_risk_threshold]
    small_cluster_cells = independent_counts.loc[independent_counts < size_risk_threshold]
    return {
        "min_cell_count": min_cell_count,
        "min_cluster_count": min_cluster_count,
        "median_cell_count": median_cell_count,
        "median_independent_count_per_cell": median_independent_count_per_cell,
        "size_risk": bool(median_independent_count_per_cell < size_risk_threshold),
        "empty_cell_count": int(len(empty_cells)),
        "empty_cluster_cell_count": int(len(empty_cluster_cells)),
        "small_cell_count": int(len(small_cells)),
        "small_cluster_cell_count": int(len(small_cluster_cells)),
        "size_risk_threshold": int(size_risk_threshold),
        "empty_cells": _monte_carlo_cell_count_records(empty_cells, count_column="count"),
        "empty_cluster_cells": _monte_carlo_cell_count_records(
            empty_cluster_cells,
            count_column="cluster_count",
        ),
        "small_cells": _monte_carlo_cell_count_records(small_cells, count_column="count"),
        "small_cluster_cells": _monte_carlo_cell_count_records(
            small_cluster_cells,
            count_column="cluster_count",
        ),
    }


def _monte_carlo_cell_count_records(
    counts: pd.Series,
    *,
    count_column: str,
) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for index, value in counts.items():
        treated, mediator, outcome = index
        records.append(
            {
                "treated": _normalize_diagnostic_value(treated),
                "mediator": _normalize_diagnostic_value(mediator),
                "outcome": _normalize_diagnostic_value(outcome),
                count_column: int(value),
            }
        )
    return tuple(records)


def _monte_carlo_draw_cell_preview(
    draws: tuple[MonteCarloDrawResult, ...],
    *,
    cell_attr: str,
    max_records: int = 8,
) -> tuple[dict[str, Any], ...]:
    preview: list[dict[str, Any]] = []
    for draw in draws:
        for cell in getattr(draw, cell_attr):
            preview.append({"replication": draw.replication, **cell})
            if len(preview) >= max_records:
                return tuple(preview)
    return tuple(preview)


def _median_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.median(values))


def _series_support_values(series: pd.Series) -> tuple[Any, ...]:
    if isinstance(series.dtype, pd.CategoricalDtype):
        observed_values = list(pd.unique(series.dropna()))
        return tuple(
            _normalize_diagnostic_value(category)
            for category in series.cat.categories
            if any(category == observed_value for observed_value in observed_values)
        )
    return _ordered_unique_values(series)


def _mean_binary_source_share(
    draws: tuple[MonteCarloDrawResult, ...],
    *,
    numerator_attr: str,
    complement_attr: str,
) -> float | None:
    shares: list[float] = []
    for draw in draws:
        numerator = getattr(draw, numerator_attr)
        complement = getattr(draw, complement_attr)
        if numerator is None or complement is None:
            continue
        total = int(numerator) + int(complement)
        if total > 0:
            shares.append(float(numerator) / total)
    if not shares:
        return None
    return float(np.mean(shares))


def _source_count_total(
    draws: tuple[MonteCarloDrawResult, ...],
    *,
    numerator_attr: str,
    complement_attr: str,
) -> int | None:
    total = 0
    saw_counts = False
    for draw in draws:
        numerator = getattr(draw, numerator_attr)
        complement = getattr(draw, complement_attr)
        if numerator is None or complement is None:
            continue
        total += int(numerator) + int(complement)
        saw_counts = True
    return total if saw_counts else None


def _benchmark_cell_to_spec(cell: dict[str, Any] | MonteCarloBenchmarkCell) -> dict[str, Any]:
    if isinstance(cell, MonteCarloBenchmarkCell):
        if cell.method != "CS":
            raise NotImplementedError("Binary empirical-mixture benchmark matrix currently executes method='CS'.")
        return {
            "table": cell.table,
            "design": cell.design,
            "mediator": cell.mediator,
            "clusters": cell.clusters,
            "bins": cell.bins,
            "t": cell.t,
            "benchmark_row_index": cell.benchmark_row_index,
        }

    required_keys = {"table", "design", "mediator", "clusters", "bins", "t"}
    missing_keys = required_keys.difference(cell)
    if missing_keys:
        raise KeyError(f"Monte Carlo benchmark cell is missing keys: {sorted(missing_keys)}.")
    if cell.get("method", "CS") != "CS":
        raise NotImplementedError("Binary empirical-mixture benchmark matrix currently executes method='CS'.")
    return {
        "table": cell["table"],
        "design": cell["design"],
        "mediator": cell["mediator"],
        "clusters": cell["clusters"],
        "bins": cell["bins"],
        "t": cell["t"],
        "benchmark_row_index": cell.get("benchmark_row_index", 0),
    }


def _benchmark_data_source_key(cell: MonteCarloBenchmarkCell) -> str:
    return "|".join(
        [
            cell.design,
            cell.mediator,
            "clustered" if cell.requires_cluster_resampling else "unclustered",
        ]
    )


def _benchmark_cells_by_design(
    cells: tuple[MonteCarloBenchmarkCell, ...],
) -> dict[str, tuple[MonteCarloBenchmarkCell, ...]]:
    grouped: dict[str, list[MonteCarloBenchmarkCell]] = {}
    for cell in cells:
        grouped.setdefault(cell.design, []).append(cell)
    return {design: tuple(design_cells) for design, design_cells in grouped.items()}


def _raise_for_ambiguous_design_level_data_sources(
    *,
    cells: tuple[MonteCarloBenchmarkCell, ...],
    data_sources: dict[str, "BinaryEmpiricalMixtureBenchmarkDataSource"],
) -> None:
    source_keys_by_design: dict[str, set[str]] = {}
    for cell in cells:
        source_keys_by_design.setdefault(cell.design, set()).add(
            _benchmark_data_source_key(cell)
        )
    ambiguous = {
        design: sorted(source_keys - set(data_sources))
        for design, source_keys in source_keys_by_design.items()
        if len(source_keys) > 1
        and design in data_sources
        and source_keys - set(data_sources)
    }
    if ambiguous:
        raise ValueError(
            "Empirical-mixture data sources must use full data-source keys when "
            "one design has multiple mediator sources; design-level fallback is "
            f"ambiguous for missing keys: {ambiguous}."
        )


def _resolve_benchmark_data_source(
    data_sources: dict[str, BinaryEmpiricalMixtureBenchmarkDataSource],
    cell: MonteCarloBenchmarkCell,
) -> BinaryEmpiricalMixtureBenchmarkDataSource:
    key = _benchmark_data_source_key(cell)
    if key in data_sources:
        return data_sources[key]
    return data_sources[cell.design]


def _build_empirical_mixture_design(
    contracts: "MonteCarloContracts",
    *,
    name: str,
    df: pd.DataFrame,
    d: str,
    m: str,
    y: str,
    table: str,
    design: str,
    mediator: str,
    clusters: int | None,
    bins: int | None,
    t: float,
    seed: int,
    cluster: str | None = None,
    replications: int = 500,
    bootstrap_replications: int = 500,
    alpha: float = 0.05,
    replication_start: int = 0,
    seed_replications: int | None = None,
    method: str = "CS",
) -> BinaryEmpiricalMixtureMonteCarloDesign:
    if mediator == "binary":
        return contracts.binary_empirical_mixture_design(
            name=name,
            df=df,
            d=d,
            m=m,
            y=y,
            table=table,
            design=design,
            mediator=mediator,
            clusters=clusters,
            bins=bins,
            t=t,
            seed=seed,
            cluster=cluster,
            replications=replications,
            bootstrap_replications=bootstrap_replications,
            alpha=alpha,
            replication_start=replication_start,
            seed_replications=seed_replications,
            method=method,
        )
    if mediator == "nonbinary":
        return contracts.nonbinary_empirical_mixture_design(
            name=name,
            df=df,
            d=d,
            m=m,
            y=y,
            table=table,
            design=design,
            mediator=mediator,
            clusters=clusters,
            bins=bins,
            t=t,
            seed=seed,
            cluster=cluster,
            replications=replications,
            bootstrap_replications=bootstrap_replications,
            alpha=alpha,
            replication_start=replication_start,
            seed_replications=seed_replications,
            method=method,
        )
    raise NotImplementedError(f"Unsupported empirical-mixture mediator {mediator!r}.")


def _build_empirical_mixture_cs_design(
    contracts: "MonteCarloContracts",
    **kwargs: Any,
) -> BinaryEmpiricalMixtureMonteCarloDesign:
    return _build_empirical_mixture_design(contracts, **kwargs, method="CS")


def _run_empirical_mixture_monte_carlo(
    design: BinaryEmpiricalMixtureMonteCarloDesign,
) -> MonteCarloSimulationResult:
    paper_contract = design.paper_contract_dict or {}
    method = str(paper_contract.get("target_method", "CS"))
    if paper_contract.get("mediator") == "nonbinary":
        return run_nonbinary_empirical_mixture_monte_carlo(design, method=method)
    return run_binary_empirical_mixture_monte_carlo(design, method=method)


def _run_empirical_mixture_cs_monte_carlo(
    design: BinaryEmpiricalMixtureMonteCarloDesign,
) -> MonteCarloSimulationResult:
    return _run_empirical_mixture_monte_carlo(design)


def _benchmark_data_source_diagnostic(
    *,
    design: str,
    cells: tuple[MonteCarloBenchmarkCell, ...],
    source: BinaryEmpiricalMixtureBenchmarkDataSource | None,
) -> MonteCarloBenchmarkDataSourceDiagnostic:
    requires_cluster_resampling = any(cell.requires_cluster_resampling for cell in cells)
    unbinned_outcome_binary_required = any(cell.bins is None for cell in cells)
    if source is None:
        return MonteCarloBenchmarkDataSourceDiagnostic(
            design=design,
            executable_rows=len(cells),
            requires_cluster_resampling=requires_cluster_resampling,
            analysis_frame_columns=(),
            d=None,
            m=None,
            y=None,
            rows=None,
            complete_case_rows=None,
            treatment_levels=(),
            mediator_levels=(),
            control_rows=None,
            treated_rows=None,
            cluster=None,
            source_clusters=None,
            control_source_clusters=None,
            treated_source_clusters=None,
            arm_fixed_source_clusters=None,
            outcome_level_count=None,
            outcome_binary=None,
            unbinned_outcome_binary_required=unbinned_outcome_binary_required,
            blocking_reasons=("missing_data_source",),
        )

    blocking_reasons: list[str] = []
    analysis_frame = source.analysis_frame()
    cleaned_df = remove_missing_from_df(df=analysis_frame, d=source.d, m=source.m, y=source.y)
    treatment_levels = _ordered_unique_values(cleaned_df[source.d]) if not cleaned_df.empty else ()
    mediator_levels = _ordered_unique_values(cleaned_df[source.m]) if not cleaned_df.empty else ()
    outcome_levels = _ordered_unique_values(cleaned_df[source.y]) if not cleaned_df.empty else ()
    if cleaned_df.empty:
        blocking_reasons.append("empty_complete_cases")
    if _contains_nonfinite_numeric_support_value(treatment_levels):
        blocking_reasons.append("treatment_levels_nonfinite")
    if len(treatment_levels) != 2:
        blocking_reasons.append("treatment_not_binary")
    if _contains_nonfinite_numeric_support_value(mediator_levels):
        blocking_reasons.append("mediator_levels_nonfinite")
    if len(mediator_levels) != 2:
        blocking_reasons.append("mediator_not_binary")
    if (not cleaned_df.empty) and _series_has_nonfinite_numeric_values(cleaned_df[source.y]):
        blocking_reasons.append("outcome_values_nonfinite")
    if unbinned_outcome_binary_required and len(outcome_levels) != 2:
        blocking_reasons.append("unbinned_outcome_not_binary")

    control_rows: int | None = None
    treated_rows: int | None = None
    if len(treatment_levels) == 2:
        control_level, treated_level = treatment_levels
        control_rows = int((cleaned_df[source.d] == control_level).sum())
        treated_rows = int((cleaned_df[source.d] == treated_level).sum())
        if control_rows == 0 or treated_rows == 0:
            blocking_reasons.append("treatment_arm_empty")
    if (
        source.expected_complete_case_rows is not None
        and int(cleaned_df.shape[0]) != source.expected_complete_case_rows
    ):
        blocking_reasons.append("unexpected_complete_case_rows")
    if (
        source.expected_control_rows is not None
        and control_rows != source.expected_control_rows
    ):
        blocking_reasons.append("unexpected_control_rows")
    if (
        source.expected_treated_rows is not None
        and treated_rows != source.expected_treated_rows
    ):
        blocking_reasons.append("unexpected_treated_rows")

    source_clusters: int | None = None
    control_source_clusters: int | None = None
    treated_source_clusters: int | None = None
    arm_fixed_source_clusters: bool | None = None
    if requires_cluster_resampling:
        if source.cluster is None:
            blocking_reasons.append("cluster_column_required")
        else:
            if bool(cleaned_df[source.cluster].isna().any()):
                blocking_reasons.append("cluster_labels_missing")
            source_clusters = int(cleaned_df[source.cluster].nunique(dropna=True))
            treatment_levels_by_cluster = cleaned_df.groupby(source.cluster, sort=False)[source.d].nunique()
            arm_fixed_source_clusters = not bool((treatment_levels_by_cluster > 1).any())
            if not arm_fixed_source_clusters:
                blocking_reasons.append("treatment_not_fixed_within_source_cluster")
            if len(treatment_levels) == 2:
                control_level, treated_level = treatment_levels
                control_source_clusters = int(
                    cleaned_df.loc[cleaned_df[source.d] == control_level, source.cluster].nunique(dropna=True)
                )
                treated_source_clusters = int(
                    cleaned_df.loc[cleaned_df[source.d] == treated_level, source.cluster].nunique(dropna=True)
                )
                if control_source_clusters == 0 or treated_source_clusters == 0:
                    blocking_reasons.append("cluster_arm_empty")
            if (
                source.expected_source_clusters is not None
                and source_clusters != source.expected_source_clusters
            ):
                blocking_reasons.append("unexpected_source_clusters")
            if (
                source.expected_control_source_clusters is not None
                and control_source_clusters != source.expected_control_source_clusters
            ):
                blocking_reasons.append("unexpected_control_source_clusters")
            if (
                source.expected_treated_source_clusters is not None
                and treated_source_clusters != source.expected_treated_source_clusters
            ):
                blocking_reasons.append("unexpected_treated_source_clusters")

    return MonteCarloBenchmarkDataSourceDiagnostic(
        design=design,
        executable_rows=len(cells),
        requires_cluster_resampling=requires_cluster_resampling,
        analysis_frame_columns=source.analysis_frame_columns,
        d=source.d,
        m=source.m,
        y=source.y,
        rows=int(source.df.shape[0]),
        complete_case_rows=int(cleaned_df.shape[0]),
        treatment_levels=treatment_levels,
        mediator_levels=mediator_levels,
        control_rows=control_rows,
        treated_rows=treated_rows,
        cluster=source.cluster,
        source_clusters=source_clusters,
        control_source_clusters=control_source_clusters,
        treated_source_clusters=treated_source_clusters,
        arm_fixed_source_clusters=arm_fixed_source_clusters,
        expected_complete_case_rows=source.expected_complete_case_rows,
        expected_control_rows=source.expected_control_rows,
        expected_treated_rows=source.expected_treated_rows,
        expected_source_clusters=source.expected_source_clusters,
        expected_control_source_clusters=source.expected_control_source_clusters,
        expected_treated_source_clusters=source.expected_treated_source_clusters,
        outcome_level_count=len(outcome_levels),
        outcome_binary=len(outcome_levels) == 2,
        unbinned_outcome_binary_required=unbinned_outcome_binary_required,
        blocking_reasons=tuple(dict.fromkeys(blocking_reasons)),
    )


def _empirical_mixture_data_source_diagnostic(
    *,
    design: str,
    cells: tuple[MonteCarloBenchmarkCell, ...],
    source: BinaryEmpiricalMixtureBenchmarkDataSource | None,
) -> MonteCarloBenchmarkDataSourceDiagnostic:
    if not cells:
        return _benchmark_data_source_diagnostic(design=design, cells=cells, source=source)
    mediator_kinds = {cell.mediator for cell in cells}
    if mediator_kinds == {"binary"}:
        return _benchmark_data_source_diagnostic(design=design, cells=cells, source=source)
    if mediator_kinds == {"nonbinary"}:
        diagnostic = _nonbinary_empirical_mixture_data_source_diagnostic(
            design=design,
            cells=cells,
            source=source,
        )
        if source is None:
            return MonteCarloBenchmarkDataSourceDiagnostic(
                design=design,
                executable_rows=len(cells),
                requires_cluster_resampling=any(cell.requires_cluster_resampling for cell in cells),
                analysis_frame_columns=(),
                d=None,
                m=None,
                y=None,
                rows=None,
                complete_case_rows=None,
                treatment_levels=(),
                mediator_levels=(),
                control_rows=None,
                treated_rows=None,
                cluster=None,
                source_clusters=None,
                control_source_clusters=None,
                treated_source_clusters=None,
                arm_fixed_source_clusters=None,
                outcome_level_count=None,
                outcome_binary=None,
                blocking_reasons=diagnostic["blocking_reasons"],
            )
        return MonteCarloBenchmarkDataSourceDiagnostic(
            design=design,
            executable_rows=len(cells),
            requires_cluster_resampling=any(cell.requires_cluster_resampling for cell in cells),
            analysis_frame_columns=source.analysis_frame_columns,
            d=source.d,
            m=source.m,
            y=source.y,
            rows=int(source.df.shape[0]),
            complete_case_rows=diagnostic["complete_case_rows"],
            treatment_levels=tuple(diagnostic["treatment_levels"]),
            mediator_levels=tuple(diagnostic["mediator_levels"]),
            control_rows=diagnostic["control_rows"],
            treated_rows=diagnostic["treated_rows"],
            cluster=source.cluster,
            source_clusters=diagnostic["source_clusters"],
            control_source_clusters=diagnostic["control_source_clusters"],
            treated_source_clusters=diagnostic["treated_source_clusters"],
            arm_fixed_source_clusters=diagnostic["arm_fixed_source_clusters"],
            expected_complete_case_rows=source.expected_complete_case_rows,
            expected_control_rows=source.expected_control_rows,
            expected_treated_rows=source.expected_treated_rows,
            expected_source_clusters=source.expected_source_clusters,
            expected_control_source_clusters=source.expected_control_source_clusters,
            expected_treated_source_clusters=source.expected_treated_source_clusters,
            outcome_level_count=diagnostic["outcome_level_count"],
            outcome_binary=None if diagnostic["outcome_level_count"] is None else diagnostic["outcome_level_count"] == 2,
            blocking_reasons=diagnostic["blocking_reasons"],
        )
    raise NotImplementedError(
        f"Empirical-mixture plan for design {design!r} mixes mediator types {sorted(mediator_kinds)}."
    )


def _data_source_binding_for_benchmark_cells(
    sources: dict[str, "BinaryEmpiricalMixtureBenchmarkDataSource"],
    *,
    design: str,
    cells: tuple[MonteCarloBenchmarkCell, ...],
) -> tuple[str | None, BinaryEmpiricalMixtureBenchmarkDataSource | None]:
    for cell in cells:
        key = _benchmark_data_source_key(cell)
        source = sources.get(key)
        if source is not None:
            return key, source
    if design in sources:
        return design, sources[design]
    return None, None


def _data_source_for_benchmark_cells(
    sources: dict[str, "BinaryEmpiricalMixtureBenchmarkDataSource"],
    *,
    design: str,
    cells: tuple[MonteCarloBenchmarkCell, ...],
) -> BinaryEmpiricalMixtureBenchmarkDataSource | None:
    return _data_source_binding_for_benchmark_cells(
        sources,
        design=design,
        cells=cells,
    )[1]


def _nonbinary_empirical_mixture_data_source_diagnostic(
    *,
    design: str,
    cells: tuple[MonteCarloBenchmarkCell, ...],
    source: BinaryEmpiricalMixtureBenchmarkDataSource | None,
) -> dict[str, Any]:
    requires_cluster_resampling = any(cell.requires_cluster_resampling for cell in cells)
    expected_mediator_level_count = _nonbinary_empirical_mixture_expected_mediator_level_count(design)
    if source is None:
        return {
            "ready": False,
            "blocking_reasons": ("missing_data_source",),
            "rows": None,
            "complete_case_rows": None,
            "control_rows": None,
            "treated_rows": None,
            "source_clusters": None,
            "control_source_clusters": None,
            "treated_source_clusters": None,
            "treatment_levels": (),
            "mediator_level_count": 0,
            "mediator_levels": (),
            "expected_mediator_level_count": expected_mediator_level_count,
            "outcome_level_count": None,
            "arm_fixed_source_clusters": None,
        }

    blocking_reasons: list[str] = []
    analysis_frame = source.analysis_frame()
    cleaned_df = remove_missing_from_df(df=analysis_frame, d=source.d, m=source.m, y=source.y)
    treatment_levels = _ordered_unique_values(cleaned_df[source.d]) if not cleaned_df.empty else ()
    if cleaned_df.empty:
        mediator_levels = ()
        mediator_order_error: ValueError | None = None
    else:
        try:
            mediator_levels = _ordered_nonbinary_mediator_levels(cleaned_df[source.m], column=source.m)
            mediator_order_error = None
        except ValueError as exc:
            mediator_levels = _ordered_unique_values(cleaned_df[source.m])
            mediator_order_error = exc
    outcome_levels = _ordered_unique_values(cleaned_df[source.y]) if not cleaned_df.empty else ()

    if cleaned_df.empty:
        blocking_reasons.append("empty_complete_cases")
    if _contains_nonfinite_numeric_support_value(treatment_levels):
        blocking_reasons.append("treatment_levels_nonfinite")
    if len(treatment_levels) != 2:
        blocking_reasons.append("treatment_not_binary")
    if _contains_nonfinite_numeric_support_value(mediator_levels):
        blocking_reasons.append("mediator_levels_nonfinite")
    if len(mediator_levels) <= 2:
        blocking_reasons.append("mediator_not_nonbinary")
    if (not cleaned_df.empty) and _series_has_nonfinite_numeric_values(cleaned_df[source.y]):
        blocking_reasons.append("outcome_values_nonfinite")
    if mediator_order_error is not None:
        blocking_reasons.append("mediator_order_not_supported")
    elif (
        expected_mediator_level_count is not None
        and len(mediator_levels) != expected_mediator_level_count
    ):
        blocking_reasons.append("unexpected_mediator_level_count")

    control_rows: int | None = None
    treated_rows: int | None = None
    if len(treatment_levels) == 2:
        control_level, treated_level = treatment_levels
        control_rows = int((cleaned_df[source.d] == control_level).sum())
        treated_rows = int((cleaned_df[source.d] == treated_level).sum())
        if control_rows == 0 or treated_rows == 0:
            blocking_reasons.append("treatment_arm_empty")

    if (
        source.expected_complete_case_rows is not None
        and int(cleaned_df.shape[0]) != source.expected_complete_case_rows
    ):
        blocking_reasons.append("unexpected_complete_case_rows")
    if (
        source.expected_control_rows is not None
        and control_rows != source.expected_control_rows
    ):
        blocking_reasons.append("unexpected_control_rows")
    if (
        source.expected_treated_rows is not None
        and treated_rows != source.expected_treated_rows
    ):
        blocking_reasons.append("unexpected_treated_rows")

    source_clusters: int | None = None
    control_source_clusters: int | None = None
    treated_source_clusters: int | None = None
    arm_fixed_source_clusters: bool | None = None
    if requires_cluster_resampling:
        if source.cluster is None:
            blocking_reasons.append("cluster_column_required")
        else:
            if bool(cleaned_df[source.cluster].isna().any()):
                blocking_reasons.append("cluster_labels_missing")
            source_clusters = int(cleaned_df[source.cluster].nunique(dropna=True))
            treatment_levels_by_cluster = cleaned_df.groupby(source.cluster, sort=False)[source.d].nunique()
            arm_fixed_source_clusters = not bool((treatment_levels_by_cluster > 1).any())
            if not arm_fixed_source_clusters:
                blocking_reasons.append("treatment_not_fixed_within_source_cluster")
            if len(treatment_levels) == 2:
                control_level, treated_level = treatment_levels
                control_source_clusters = int(
                    cleaned_df.loc[cleaned_df[source.d] == control_level, source.cluster].nunique(dropna=True)
                )
                treated_source_clusters = int(
                    cleaned_df.loc[cleaned_df[source.d] == treated_level, source.cluster].nunique(dropna=True)
                )
                if control_source_clusters == 0 or treated_source_clusters == 0:
                    blocking_reasons.append("cluster_arm_empty")
            if (
                source.expected_source_clusters is not None
                and source_clusters != source.expected_source_clusters
            ):
                blocking_reasons.append("unexpected_source_clusters")
            if (
                source.expected_control_source_clusters is not None
                and control_source_clusters != source.expected_control_source_clusters
            ):
                blocking_reasons.append("unexpected_control_source_clusters")
            if (
                source.expected_treated_source_clusters is not None
                and treated_source_clusters != source.expected_treated_source_clusters
            ):
                blocking_reasons.append("unexpected_treated_source_clusters")

    return {
        "ready": not blocking_reasons,
        "blocking_reasons": tuple(dict.fromkeys(blocking_reasons)),
        "rows": int(source.df.shape[0]),
        "complete_case_rows": int(cleaned_df.shape[0]),
        "control_rows": control_rows,
        "treated_rows": treated_rows,
        "source_clusters": source_clusters,
        "control_source_clusters": control_source_clusters,
        "treated_source_clusters": treated_source_clusters,
        "treatment_levels": tuple(_normalize_diagnostic_value(value) for value in treatment_levels),
        "mediator_level_count": len(mediator_levels),
        "mediator_levels": tuple(_normalize_diagnostic_value(value) for value in mediator_levels),
        "expected_mediator_level_count": expected_mediator_level_count,
        "outcome_level_count": len(outcome_levels),
        "arm_fixed_source_clusters": arm_fixed_source_clusters,
    }


def _nonbinary_empirical_mixture_expected_mediator_level_count(design: str) -> int | None:
    if design == "Baranov et al":
        return 5
    return None


def _unique_design_values(frame: pd.DataFrame, *, value_column: str) -> dict[str, Any]:
    values_by_design: dict[str, Any] = {}
    if value_column not in frame:
        return values_by_design

    for design, group in frame.groupby("design", sort=False):
        for value in group[value_column].tolist():
            if value is None:
                continue
            if isinstance(value, float) and math.isnan(value):
                continue
            values_by_design[str(design)] = _normalize_diagnostic_value(value)
            break
    return values_by_design


def _unique_source_key_values(
    frame: pd.DataFrame,
    *,
    key_column: str,
    value_column: str,
) -> dict[str, Any]:
    values_by_key: dict[str, Any] = {}
    if key_column not in frame or value_column not in frame:
        return values_by_key

    for source_key, group in frame.dropna(subset=[key_column]).groupby(key_column, sort=False):
        for value in group[value_column].tolist():
            if value is None:
                continue
            if isinstance(value, float) and math.isnan(value):
                continue
            values_by_key[str(source_key)] = _normalize_diagnostic_value(value)
            break
    return values_by_key


def _data_source_diagnostic_summary(
    diagnostics: tuple[MonteCarloBenchmarkDataSourceDiagnostic, ...],
) -> dict[str, Any]:
    if not diagnostics:
        return {}

    blocking_reasons: dict[str, int] = {}
    blocking_reasons_by_source_key: dict[str, tuple[str, ...]] = {}
    complete_case_rows_by_source_key: dict[str, int | None] = {}
    source_clusters_by_source_key: dict[str, int | None] = {}
    for diagnostic in diagnostics:
        source_key = diagnostic.data_source_key or diagnostic.design
        complete_case_rows_by_source_key[source_key] = diagnostic.complete_case_rows
        if diagnostic.source_clusters is not None:
            source_clusters_by_source_key[source_key] = diagnostic.source_clusters
        blocking_reasons_by_source_key[source_key] = diagnostic.blocking_reasons
        for reason in diagnostic.blocking_reasons:
            blocking_reasons[reason] = blocking_reasons.get(reason, 0) + 1

    return {
        "data_source_checked_designs": len(diagnostics),
        "data_source_ready_designs": int(sum(diagnostic.ready for diagnostic in diagnostics)),
        "data_source_blocked_designs": int(sum(not diagnostic.ready for diagnostic in diagnostics)),
        "data_source_complete_case_rows": {
            diagnostic.design: diagnostic.complete_case_rows for diagnostic in diagnostics
        },
        "data_source_source_clusters": {
            diagnostic.design: diagnostic.source_clusters
            for diagnostic in diagnostics
            if diagnostic.source_clusters is not None
        },
        "data_source_blocking_reasons": blocking_reasons,
        "data_source_complete_case_rows_by_source_key": complete_case_rows_by_source_key,
        "data_source_source_clusters_by_source_key": source_clusters_by_source_key,
        "data_source_blocking_reasons_by_source_key": blocking_reasons_by_source_key,
    }


def _row_matches_benchmark_filters(
    row: MonteCarloResultRow,
    *,
    design: str | None,
    table: str | None,
    clusters: tuple[int | None, ...] | None,
    bins: tuple[int | None, ...] | None,
    t_values: tuple[float, ...] | None,
) -> bool:
    if design is not None and row.design != design:
        return False
    if table is not None and row.table != table:
        return False
    if clusters is not None and row.clusters not in clusters:
        return False
    if bins is not None and row.bins not in bins:
        return False
    if t_values is not None and not any(abs(row.t - float(t_value)) <= 1e-12 for t_value in t_values):
        return False
    return True


def _coerce_probability_vector(*, name: str, values: tuple[float, ...]) -> tuple[float, ...]:
    probabilities = tuple(float(value) for value in values)
    if not probabilities:
        raise ValueError(f"{name} must contain at least one probability.")
    if any(value < 0 or value > 1 for value in probabilities):
        raise ValueError(f"{name} must contain probabilities in [0, 1].")
    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-10):
        raise ValueError(f"{name} must sum to 1.")
    return probabilities


def _binary_levels(series: pd.Series, *, name: str) -> tuple[Any, Any]:
    return ordered_binary_support_levels(series, column=name)


def _ordered_unique_values(series: pd.Series) -> tuple[Any, ...]:
    if isinstance(series.dtype, pd.CategoricalDtype):
        observed_values = set(pd.unique(series.dropna()))
        return tuple(
            _normalize_diagnostic_value(value)
            for value in series.cat.categories
            if value in observed_values
        )
    unique_values = pd.unique(series)
    return tuple(sorted((_normalize_diagnostic_value(value) for value in unique_values), key=_diagnostic_sort_key))


def _ordered_nonbinary_mediator_levels(series: pd.Series, *, column: str) -> tuple[Any, ...]:
    if isinstance(series.dtype, pd.CategoricalDtype) and series.dtype.ordered:
        return _ordered_unique_values(series)

    levels = tuple(_normalize_diagnostic_value(value) for value in pd.unique(series.dropna()))
    if len(levels) <= 2:
        return tuple(sorted(levels, key=_diagnostic_sort_key))
    if not all(isinstance(value, (bool, int, float)) for value in levels):
        raise ValueError(
            f"Nonbinary empirical-mixture mediator {column!r} must have naturally comparable "
            "support or be an ordered pandas Categorical before using monotonicity."
        )
    for index, left in enumerate(levels):
        for right in levels[index + 1 :]:
            try:
                left <= right
                right <= left
            except TypeError as exc:
                raise ValueError(
                    f"Nonbinary empirical-mixture mediator {column!r} must have naturally "
                    "comparable support or be an ordered pandas Categorical before using "
                    "monotonicity."
                ) from exc
    return tuple(sorted(levels, key=_diagnostic_sort_key))


def _observed_outcome_values(series: pd.Series) -> tuple[Any, ...]:
    if isinstance(series.dtype, pd.CategoricalDtype):
        observed_values = list(pd.unique(series.dropna()))
        return tuple(
            _normalize_diagnostic_value(category)
            for category in series.cat.categories
            if any(category == observed_value for observed_value in observed_values)
        )
    return tuple(_normalize_diagnostic_value(value) for value in pd.unique(series))


def _normalize_diagnostic_value(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (AttributeError, ValueError):
            return value
    return value


def _json_safe_diagnostic_value(value: Any) -> Any:
    normalized = _normalize_diagnostic_value(value)
    if _is_nonfinite_numeric_value(normalized):
        numeric = float(normalized)
        if math.isnan(numeric):
            return "nan"
        return "inf" if numeric > 0 else "-inf"
    return normalized


def _is_nonfinite_numeric_value(value: Any) -> bool:
    normalized = _normalize_diagnostic_value(value)
    if isinstance(normalized, bool):
        return False
    if isinstance(normalized, (int, float, np.integer, np.floating)):
        return not math.isfinite(float(normalized))
    return False


def _contains_nonfinite_numeric_support_value(values: tuple[Any, ...]) -> bool:
    for value in values:
        if isinstance(value, tuple):
            if _contains_nonfinite_numeric_support_value(value):
                return True
        elif _is_nonfinite_numeric_value(value):
            return True
    return False


def _series_has_nonfinite_numeric_values(series: pd.Series) -> bool:
    return _contains_nonfinite_numeric_support_value(
        tuple(_normalize_diagnostic_value(value) for value in pd.unique(series.dropna()))
    )


def _raise_for_nonfinite_numeric_support_levels(values: tuple[Any, ...], *, column: str) -> None:
    if _contains_nonfinite_numeric_support_value(values):
        raise ValueError(f"{column} must contain only finite numeric support levels.")


def _raise_for_nonfinite_numeric_values(series: pd.Series, *, column: str) -> None:
    if _series_has_nonfinite_numeric_values(series):
        raise ValueError(f"{column} must contain only finite numeric values.")


def _diagnostic_sort_key(value: Any) -> tuple[Any, ...]:
    normalized = _normalize_diagnostic_value(value)
    if isinstance(normalized, bool):
        return ("bool", int(normalized))
    if isinstance(normalized, (int, float)):
        return ("number", float(normalized))
    return (type(normalized).__name__, repr(normalized))


def _conditional_y_probabilities(
    df: pd.DataFrame,
    *,
    d: str,
    m: str,
    y: str,
    d_value: Any,
    m_value: Any,
    y_values: tuple[Any, ...],
) -> tuple[float, ...]:
    cell = df[(df[d] == d_value) & (df[m] == m_value)]
    if cell.empty:
        raise ValueError("Observed binary partial-density calibration requires every D x M cell to be non-empty.")
    return tuple(float((cell[y] == y_value).mean()) for y_value in y_values)


def _parse_result_table(
    *,
    path: Path,
    table_name: str,
    mediator: str,
    default_bins: int | None,
) -> list[MonteCarloResultRow]:
    _require_file(path)
    rows: list[MonteCarloResultRow] = []
    panel: str | None = None
    design: str | None = None
    clusters: int | None = None
    bins: int | None = None
    methods: list[str] | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        panel_match = _PANEL_RE.search(raw_line)
        if panel_match is not None:
            panel = f"Panel {panel_match.group(1)}"
            design, clusters, bins = _parse_panel_design(
                panel_match.group(2),
                default_bins=default_bins,
            )
            continue

        if "\\bar" in raw_line and "ARP" in raw_line:
            header_cells = [_clean_cell(cell) for cell in raw_line.split("&")]
            header_cells = [cell for cell in header_cells if cell]
            methods = header_cells[1:]
            continue

        if "t=" not in raw_line and "t =" not in raw_line:
            continue
        if panel is None or design is None or methods is None:
            raise ValueError(f"Malformed Monte Carlo table row before panel/header in {path}.")

        t_value = _parse_t_value(raw_line)
        values = [_parse_number(cell) for cell in raw_line.split("&")[1:]]
        if len(values) != len(methods) + 1:
            raise ValueError(f"Unexpected number of values in {path}: {raw_line}")
        rows.append(
            MonteCarloResultRow(
                table=table_name,
                panel=panel,
                design=design,
                mediator=mediator,
                clusters=clusters,
                bins=bins if design == "Baranov et al" else None,
                t=t_value,
                bar_nu_lb=values[0],
                rejection_rates=dict(zip(methods, values[1:], strict=True)),
            )
        )

    return rows


def _max_rejection_rate(
    rows: tuple[MonteCarloResultRow, ...],
    *,
    method: str,
) -> float | None:
    if not rows:
        return None
    return float(max(row.rejection_rates[method] for row in rows))


def _min_rejection_rate(
    rows: tuple[MonteCarloResultRow, ...],
    *,
    method: str,
) -> float | None:
    if not rows:
        return None
    return float(min(row.rejection_rates[method] for row in rows))


def _method_uses_discretized_outcome(method: str) -> bool:
    return method in {"ARP", "CS", "FSSTdd", "FSSTndd"}


def _monte_carlo_method_guidance_specs() -> dict[str, dict[str, Any]]:
    return {
        "CS": {
            "paper_order_index": 0,
            "paper_role": "recommended_default",
            "paper_recommendation": "reasonable default for most empirical settings",
            "python_execution_status": "empirical_mixture_cs_runner",
            "paper_default": True,
            "small_cluster_size_control_alternative": False,
            "large_independent_sample_power_candidate": False,
            "binary_mediator_comparator": False,
        },
        "ARP": {
            "paper_order_index": 1,
            "paper_role": "small_cluster_size_control_alternative",
            "paper_recommendation": "can be attractive when small-cluster size control dominates power",
            "python_execution_status": "arp_public_runner",
            "paper_default": False,
            "small_cluster_size_control_alternative": True,
            "large_independent_sample_power_candidate": False,
            "binary_mediator_comparator": False,
        },
        "FSSTdd": {
            "paper_order_index": 2,
            "paper_role": "large_independent_sample_power_candidate",
            "paper_recommendation": "power candidate in large independent samples, with size caution",
            "python_execution_status": "fsstdd_public_runner",
            "paper_default": False,
            "small_cluster_size_control_alternative": False,
            "large_independent_sample_power_candidate": True,
            "binary_mediator_comparator": False,
        },
        "FSSTndd": {
            "paper_order_index": 3,
            "paper_role": "large_independent_sample_power_candidate",
            "paper_recommendation": "power candidate in large independent samples, with size caution",
            "python_execution_status": "fsstndd_public_runner",
            "paper_default": False,
            "small_cluster_size_control_alternative": False,
            "large_independent_sample_power_candidate": True,
            "binary_mediator_comparator": False,
        },
        "K": {
            "paper_order_index": 4,
            "paper_role": "binary_mediator_comparator",
            "paper_recommendation": "binary-mediator comparison method, not the paper default",
            "python_execution_status": "kitagawa_public_runner",
            "paper_default": False,
            "small_cluster_size_control_alternative": False,
            "large_independent_sample_power_candidate": False,
            "binary_mediator_comparator": True,
        },
    }


def _monte_carlo_method_execution_specs() -> dict[str, dict[str, Any]]:
    return {
        "CS": {
            "supports_binary_mediator": True,
            "supports_nonbinary_mediator": True,
            "outcome_contract": "discrete_y_or_paper_discretized_y",
            "variance_estimator": "analytic",
            "variance_contract": "analytic iid or cluster-level variance for paper simulations",
            "bootstrap_required": False,
            "bootstrap_unit_modes": (),
            "bootstrap_unit_contract": "not_used_for_paper_tables",
            "tuning_contract": "conditional_chi_squared; binary-outcome refinement only",
        },
        "ARP": {
            "supports_binary_mediator": True,
            "supports_nonbinary_mediator": True,
            "outcome_contract": "discrete_y_or_paper_discretized_y",
            "variance_estimator": "analytic",
            "variance_contract": "analytic iid or cluster-level variance for paper simulations",
            "bootstrap_required": False,
            "bootstrap_unit_modes": (),
            "bootstrap_unit_contract": "not_used_for_paper_tables",
            "tuning_contract": "hybrid_test_with_first_stage_kappa",
        },
        "FSSTdd": {
            "supports_binary_mediator": True,
            "supports_nonbinary_mediator": True,
            "outcome_contract": "discrete_y_or_paper_discretized_y",
            "variance_estimator": "bootstrap",
            "variance_contract": "nonparametric bootstrap at individual or cluster level",
            "bootstrap_required": True,
            "bootstrap_unit_modes": ("individual", "cluster"),
            "bootstrap_unit_contract": "individual_or_cluster_as_appropriate",
            "tuning_contract": "fsst_lambda_data_driven",
        },
        "FSSTndd": {
            "supports_binary_mediator": True,
            "supports_nonbinary_mediator": True,
            "outcome_contract": "discrete_y_or_paper_discretized_y",
            "variance_estimator": "bootstrap",
            "variance_contract": "nonparametric bootstrap at individual or cluster level",
            "bootstrap_required": True,
            "bootstrap_unit_modes": ("individual", "cluster"),
            "bootstrap_unit_contract": "individual_or_cluster_as_appropriate",
            "tuning_contract": "fsst_lambda_non_data_driven",
        },
        "K": {
            "supports_binary_mediator": True,
            "supports_nonbinary_mediator": False,
            "outcome_contract": "original_outcome_no_discretization",
            "variance_estimator": "bootstrap",
            "variance_contract": "nonparametric bootstrap at individual or cluster level",
            "bootstrap_required": True,
            "bootstrap_unit_modes": ("individual", "cluster"),
            "bootstrap_unit_contract": "individual_or_cluster_as_appropriate",
            "tuning_contract": "kitagawa_binary_mediator_comparator",
        },
    }


def _method_requires_bootstrap(method: str) -> bool:
    spec = _monte_carlo_method_execution_specs().get(str(method))
    return bool(spec and spec["bootstrap_required"])


def _monte_carlo_method_runner_readiness_specs() -> dict[str, dict[str, Any]]:
    return {
        "CS": {
            "runner_family": "conditional_chi_squared",
            "runner_entrypoint": "run_empirical_mixture_cs_benchmark_manifest",
            "sharp_null_method": "CS",
            "current_public_entrypoint": "test_sharp_null(method='CS')",
            "current_plan_runner": "empirical_mixture_benchmark_plan(method='CS')",
            "next_action": "run_full_method_matrix",
        },
        "ARP": {
            "runner_family": "hybrid_moment_inequality",
            "runner_entrypoint": "run_empirical_mixture_arp_benchmark_manifest",
            "sharp_null_method": "ARP",
            "current_public_entrypoint": "test_sharp_null(method='ARP')",
            "current_plan_runner": "empirical_mixture_benchmark_plan(method='ARP')",
            "next_action": "run_full_method_matrix",
        },
        "FSSTdd": {
            "runner_family": "fsst_bootstrap_data_driven_lambda",
            "runner_entrypoint": "run_empirical_mixture_fsstdd_benchmark_manifest",
            "sharp_null_method": "FSSTdd",
            "current_public_entrypoint": "test_sharp_null(method='FSSTdd')",
            "current_plan_runner": "empirical_mixture_benchmark_plan(method='FSSTdd')",
            "next_action": "run_full_method_matrix",
        },
        "FSSTndd": {
            "runner_family": "fsst_bootstrap_non_data_driven_lambda",
            "runner_entrypoint": "run_empirical_mixture_fsstndd_benchmark_manifest",
            "sharp_null_method": "FSSTndd",
            "current_public_entrypoint": "test_sharp_null(method='FSSTndd')",
            "current_plan_runner": "empirical_mixture_benchmark_plan(method='FSSTndd')",
            "next_action": "run_full_method_matrix",
        },
        "K": {
            "runner_family": "kitagawa_binary_mediator_bootstrap",
            "runner_entrypoint": "run_empirical_mixture_k_benchmark_manifest",
            "sharp_null_method": "K",
            "current_public_entrypoint": "test_sharp_null(method='K')",
            "current_plan_runner": "empirical_mixture_benchmark_plan(method='K')",
            "next_action": "run_full_method_matrix",
        },
    }


def _require_monte_carlo_method_guidance(method: str) -> None:
    if method not in _monte_carlo_method_guidance_specs():
        raise KeyError(f"No Monte Carlo method guidance for method {method!r}.")


def _monte_carlo_method_guidance(
    *,
    method: str,
    method_summary: dict[str, Any],
) -> MonteCarloMethodGuidance:
    guidance_by_method = _monte_carlo_method_guidance_specs()
    if method not in guidance_by_method:
        raise KeyError(f"No Monte Carlo method guidance for method {method!r}.")

    guidance = guidance_by_method[method]
    return MonteCarloMethodGuidance(
        method=method,
        paper_order_index=guidance["paper_order_index"],
        paper_role=guidance["paper_role"],
        paper_recommendation=guidance["paper_recommendation"],
        python_execution_status=guidance["python_execution_status"],
        paper_default=guidance["paper_default"],
        small_cluster_size_control_alternative=guidance["small_cluster_size_control_alternative"],
        large_independent_sample_power_candidate=guidance["large_independent_sample_power_candidate"],
        binary_mediator_comparator=guidance["binary_mediator_comparator"],
        uses_discretized_outcome=_method_uses_discretized_outcome(method),
        method_summary=method_summary,
    )


def _monte_carlo_method_execution_contract(
    *,
    guidance: MonteCarloMethodGuidance,
    support_gate: dict[str, Any],
    nominal_alpha: float,
    paper_replications: int,
) -> MonteCarloMethodExecutionContract:
    execution_specs = _monte_carlo_method_execution_specs()
    if guidance.method not in execution_specs:
        raise KeyError(f"No Monte Carlo execution contract for method {guidance.method!r}.")

    spec = execution_specs[guidance.method]
    return MonteCarloMethodExecutionContract(
        method=guidance.method,
        paper_order_index=guidance.paper_order_index,
        paper_role=guidance.paper_role,
        paper_recommendation=guidance.paper_recommendation,
        paper_default=guidance.paper_default,
        uses_discretized_outcome=guidance.uses_discretized_outcome,
        supports_binary_mediator=bool(spec["supports_binary_mediator"]),
        supports_nonbinary_mediator=bool(spec["supports_nonbinary_mediator"]),
        outcome_contract=str(spec["outcome_contract"]),
        variance_estimator=str(spec["variance_estimator"]),
        variance_contract=str(spec["variance_contract"]),
        bootstrap_required=bool(spec["bootstrap_required"]),
        bootstrap_unit_modes=tuple(spec["bootstrap_unit_modes"]),
        bootstrap_unit_contract=str(spec["bootstrap_unit_contract"]),
        tuning_contract=str(spec["tuning_contract"]),
        nominal_alpha=float(nominal_alpha),
        paper_replications=int(paper_replications),
        paper_reported_rows=int(guidance.method_summary["supported_result_rows"]),
        python_executable_reported_rows=int(
            support_gate["python_executable_reported_rows"]
        ),
        python_blocked_reported_rows=int(
            support_gate["python_blocked_reported_rows"]
        ),
        python_execution_status=guidance.python_execution_status,
        python_executable=guidance.python_executable,
        paper_contract_only=not guidance.python_executable,
        next_action=str(support_gate["next_action"]),
        method_summary=dict(guidance.method_summary),
    )


def _python_method_support_gate(
    *,
    method: str,
    result_rows: tuple[MonteCarloResultRow, ...],
    acceptance_summary: dict[str, Any],
) -> dict[str, Any]:
    if method == "CS":
        reported_rows = tuple(row for row in result_rows if method in row.rejection_rates)
        executable_reported_rows = len(reported_rows)
        return {
            "method_gate_passes": True,
            "method_gate_verdict": "pass",
            "python_execution_scope": "empirical_mixture_cs_rows",
            "python_executable_reported_rows": executable_reported_rows,
            "python_blocked_reported_rows": 0,
            "python_blocking_reason_counts": {},
            "next_action": "run_full_method_matrix",
        }

    if method == "ARP":
        reported_rows = int(acceptance_summary["supported_result_rows"])
        return {
            "method_gate_passes": True,
            "method_gate_verdict": "pass",
            "python_execution_scope": "arp_public_runner",
            "python_executable_reported_rows": reported_rows,
            "python_blocked_reported_rows": 0,
            "python_blocking_reason_counts": {},
            "next_action": "run_full_method_matrix",
        }

    if method in {"FSSTdd", "FSSTndd", "K"}:
        reported_rows = int(acceptance_summary["supported_result_rows"])
        return {
            "method_gate_passes": True,
            "method_gate_verdict": "pass",
            "python_execution_scope": {
                "FSSTdd": "fsstdd_public_runner",
                "FSSTndd": "fsstndd_public_runner",
                "K": "kitagawa_public_runner",
            }[method],
            "python_executable_reported_rows": reported_rows,
            "python_blocked_reported_rows": 0,
            "python_blocking_reason_counts": {},
            "next_action": "run_full_method_matrix",
        }

    del acceptance_summary
    return {
        "method_gate_passes": True,
        "method_gate_verdict": "rescoped",
        "python_execution_scope": "non_release_paper_comparator",
        "python_executable_reported_rows": 0,
        "python_blocked_reported_rows": 0,
        "python_blocking_reason_counts": {},
        "next_action": "run_full_method_matrix",
    }


def _python_method_support_blocking_reason(
    *,
    method: str,
    row: MonteCarloResultRow,
) -> str | None:
    if method in {"CS", "ARP", "FSSTdd", "FSSTndd", "K"}:
        return None
    return f"{method.lower()}_runner_not_implemented"


def _method_support_resolution_action(reason: str | None) -> str:
    return {
        None: "run_full_method_matrix",
        "nonbinary_cs_benchmark_matrix_not_integrated": "integrate_nonbinary_cs_empirical_mixture_matrix",
        "arp_runner_not_implemented": "implement_arp_paper_runner",
        "fsstdd_runner_not_implemented": "implement_fsstdd_paper_runner",
        "fsstndd_runner_not_implemented": "implement_fsstndd_paper_runner",
        "k_runner_not_implemented": "implement_k_paper_runner",
    }.get(reason, "implement_paper_method_runner")


def _nonbinary_readiness_resolution_action(reason: str) -> str:
    return {
        "nonbinary_cs_paper_matrix_not_run": "run_nonbinary_cs_empirical_mixture_matrix",
        "cell_count_policy_size_risk": "surface_size_risk_in_acceptance_diagnostics",
        "missing_data_source": "bind_baranov_relationship_quality_data_source",
        "empty_complete_cases": "fix_nonbinary_data_source_complete_cases",
        "treatment_not_binary": "fix_nonbinary_treatment_source",
        "mediator_not_nonbinary": "bind_nonbinary_relationship_quality_mediator",
        "mediator_order_not_supported": "recode_nonbinary_mediator_as_ordered_categorical",
        "unexpected_mediator_level_count": "fix_nonbinary_mediator_levels",
        "treatment_arm_empty": "fix_nonbinary_treatment_arm_support",
        "unexpected_complete_case_rows": "fix_nonbinary_complete_case_contract",
        "unexpected_control_rows": "fix_nonbinary_control_row_contract",
        "unexpected_treated_rows": "fix_nonbinary_treated_row_contract",
        "cluster_column_required": "bind_nonbinary_cluster_column",
        "cluster_labels_missing": "fix_nonbinary_cluster_labels",
        "treatment_not_fixed_within_source_cluster": "fix_nonbinary_cluster_treatment_binding",
        "cluster_arm_empty": "fix_nonbinary_cluster_arm_support",
        "unexpected_source_clusters": "fix_nonbinary_source_cluster_contract",
        "unexpected_control_source_clusters": "fix_nonbinary_control_cluster_contract",
        "unexpected_treated_source_clusters": "fix_nonbinary_treated_cluster_contract",
        "row_release_not_ready": "inspect_nonbinary_readiness_row",
    }.get(reason, "inspect_nonbinary_readiness_blocker")


def _nonbinary_readiness_next_action(reason_counts: dict[str, int]) -> str:
    if reason_counts.get("nonbinary_cs_paper_matrix_not_run", 0) > 0:
        return "run_nonbinary_cs_empirical_mixture_matrix"
    if any(
        reason not in {"cell_count_policy_size_risk"}
        for reason in reason_counts
    ):
        return "fix_nonbinary_empirical_mixture_data_sources"
    if reason_counts.get("cell_count_policy_size_risk", 0) > 0:
        return "surface_size_risk_in_acceptance_diagnostics"
    return "run_nonbinary_cs_empirical_mixture_matrix"


def _nonbinary_run_resolution_action(reason: str) -> str:
    return {
        "nonbinary_cs_data_source_diagnostic_missing": (
            "fix_nonbinary_empirical_mixture_data_sources"
        ),
        "missing_data_source": "fix_nonbinary_empirical_mixture_data_sources",
        "empty_complete_cases": "fix_nonbinary_empirical_mixture_data_sources",
        "treatment_not_binary": "fix_nonbinary_empirical_mixture_data_sources",
        "mediator_not_nonbinary": "fix_nonbinary_empirical_mixture_data_sources",
        "mediator_order_not_supported": "fix_nonbinary_empirical_mixture_data_sources",
        "unexpected_mediator_level_count": "fix_nonbinary_empirical_mixture_data_sources",
        "treatment_arm_empty": "fix_nonbinary_empirical_mixture_data_sources",
        "unexpected_complete_case_rows": "fix_nonbinary_empirical_mixture_data_sources",
        "unexpected_control_rows": "fix_nonbinary_empirical_mixture_data_sources",
        "unexpected_treated_rows": "fix_nonbinary_empirical_mixture_data_sources",
        "cluster_column_required": "fix_nonbinary_empirical_mixture_data_sources",
        "cluster_labels_missing": "fix_nonbinary_empirical_mixture_data_sources",
        "treatment_not_fixed_within_source_cluster": (
            "fix_nonbinary_empirical_mixture_data_sources"
        ),
        "cluster_arm_empty": "fix_nonbinary_empirical_mixture_data_sources",
        "unexpected_source_clusters": "fix_nonbinary_empirical_mixture_data_sources",
        "unexpected_control_source_clusters": (
            "fix_nonbinary_empirical_mixture_data_sources"
        ),
        "unexpected_treated_source_clusters": (
            "fix_nonbinary_empirical_mixture_data_sources"
        ),
        "nonbinary_cs_paper_row_not_run": "run_nonbinary_cs_empirical_mixture_matrix",
        "nonbinary_cs_executed_row_failed": "inspect_nonbinary_cs_failed_row",
        "target_replication_shortfall": "increase_replications_or_document_tolerance",
        "cell_count_policy_size_risk": "surface_size_risk_in_acceptance_diagnostics",
    }.get(reason, "inspect_nonbinary_cs_run_blocker")


def _nonbinary_run_next_action(reason_counts: dict[str, int]) -> str:
    data_source_reasons = {
        "nonbinary_cs_data_source_diagnostic_missing",
        "missing_data_source",
        "empty_complete_cases",
        "treatment_not_binary",
        "mediator_not_nonbinary",
        "mediator_order_not_supported",
        "unexpected_mediator_level_count",
        "treatment_arm_empty",
        "unexpected_complete_case_rows",
        "unexpected_control_rows",
        "unexpected_treated_rows",
        "cluster_column_required",
        "cluster_labels_missing",
        "treatment_not_fixed_within_source_cluster",
        "cluster_arm_empty",
        "unexpected_source_clusters",
        "unexpected_control_source_clusters",
        "unexpected_treated_source_clusters",
        "source_cluster_arm_not_fixed",
    }
    if any(reason_counts.get(reason, 0) > 0 for reason in data_source_reasons):
        return "fix_nonbinary_empirical_mixture_data_sources"
    if reason_counts.get("nonbinary_cs_paper_row_not_run", 0) > 0:
        return "run_nonbinary_cs_empirical_mixture_matrix"
    if reason_counts.get("nonbinary_cs_executed_row_failed", 0) > 0:
        return "inspect_nonbinary_cs_failed_rows"
    if reason_counts.get("target_replication_shortfall", 0) > 0:
        return "increase_replications_or_document_tolerance"
    if reason_counts.get("cell_count_policy_size_risk", 0) > 0:
        return "surface_size_risk_in_acceptance_diagnostics"
    return "ready_for_release_gate"


def _method_support_gate_next_action(reason_counts: dict[str, int]) -> str:
    if reason_counts.get("nonbinary_cs_benchmark_matrix_not_integrated", 0) > 0:
        return "integrate_nonbinary_cs_empirical_mixture_matrix"
    if any(reason.endswith("_runner_not_implemented") for reason in reason_counts):
        return "implement_paper_method_runners"
    return "run_full_method_matrix"


def _ordered_monte_carlo_methods(methods: set[str]) -> list[str]:
    paper_order = ("CS", "ARP", "FSSTdd", "FSSTndd", "K")
    ordered = [method for method in paper_order if method in methods]
    remaining = sorted(methods.difference(paper_order))
    return [*ordered, *remaining]


def _paper_result_row_identity(row: MonteCarloResultRow) -> dict[str, Any]:
    return {
        "table": row.table,
        "panel": row.panel,
        "design": row.design,
        "mediator": row.mediator,
        "clusters": row.clusters,
        "bins": row.bins,
        "t": row.t,
        "bar_nu_lb": row.bar_nu_lb,
        "size_row": row.is_null_size_row,
    }


def _value_counts_tuple_or_str(values: pd.Series) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values.dropna().tolist():
        if isinstance(value, tuple):
            labels = value
        elif isinstance(value, list):
            labels = tuple(value)
        else:
            labels = (value,)
        for label in labels:
            key = str(label)
            counts[key] = counts.get(key, 0) + 1
    return counts


def _result_table_for_mediator_and_bins(*, mediator: str, bins: int) -> str:
    if mediator == "binary":
        return "table1" if bins == 5 else "appendix_table1"
    if mediator == "nonbinary":
        return "table2" if bins == 5 else "appendix_table2"
    raise KeyError(f"No Monte Carlo result table for mediator={mediator!r}, bins={bins!r}.")


def _parse_clustered_cell_count(path: Path) -> list[ClusterCellCount]:
    _require_file(path)
    rows: list[ClusterCellCount] = []
    panel: str | None = None
    clusters: int | None = None
    column_specs = (
        ("binary", 2),
        ("binary", 5),
        ("binary", 10),
        ("nonbinary", 2),
        ("nonbinary", 5),
        ("nonbinary", 10),
    )

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        panel_match = _PANEL_RE.search(raw_line)
        if panel_match is not None:
            panel = f"Panel {panel_match.group(1)}"
            _, clusters, _ = _parse_panel_design(panel_match.group(2), default_bins=None)
            continue

        if "t =" not in raw_line:
            continue
        if panel is None or clusters is None:
            raise ValueError(f"Malformed clustered cell-count row before panel in {path}.")

        t_value = _parse_t_value(raw_line)
        values = [_parse_number(cell) for cell in raw_line.split("&")[1:]]
        if len(values) != len(column_specs):
            raise ValueError(f"Unexpected number of cell-count values in {path}: {raw_line}")

        for (mediator, bins), value in zip(column_specs, values, strict=True):
            rows.append(
                ClusterCellCount(
                    table="clustered_cell_count",
                    panel=panel,
                    design="Baranov et al",
                    mediator=mediator,
                    clusters=clusters,
                    bins=bins,
                    t=t_value,
                    median_independent_clusters_per_cell=value,
                )
            )

    return rows


def _parse_panel_design(description: str, *, default_bins: int | None) -> tuple[str, int | None, int | None]:
    clean = _clean_cell(description)
    if clean.startswith("Bursztyn et al"):
        design = "Bursztyn et al"
    elif clean.startswith("Baranov et al"):
        design = "Baranov et al"
    else:
        raise ValueError(f"Unsupported Monte Carlo panel: {description!r}")

    cluster_match = re.search(r"(\d+)\s+clusters", clean)
    bins_match = re.search(r"(\d+)\s+bins", clean)
    clusters = int(cluster_match.group(1)) if cluster_match else None
    bins = int(bins_match.group(1)) if bins_match else default_bins
    return design, clusters, bins


def _parse_t_value(raw_line: str) -> float:
    match = re.search(r"t\s*=\s*([0-9.]+)", raw_line)
    if match is None:
        raise ValueError(f"Missing t value in Monte Carlo row: {raw_line}")
    return float(match.group(1))


def _parse_number(cell: str) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?", cell)
    if match is None:
        raise ValueError(f"Missing numeric value in Monte Carlo cell: {cell!r}")
    return float(match.group(0))


def _clean_cell(cell: str) -> str:
    clean = cell.replace("$", "")
    clean = clean.replace("\\bar{\\nu}", "bar_nu")
    clean = clean.replace("\\", "")
    clean = clean.replace("{", "").replace("}", "")
    clean = clean.replace("  ", " ")
    return clean.strip()


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing paper Monte Carlo table: {path}")


_PANEL_RE = re.compile(r"Panel\s+([A-Z]):\s*([^}]*)")
