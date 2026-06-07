"""Regression-adjusted probability estimation for Testing Mechanisms.

This module implements the adjusted finite-support probability grid used by
the bounds estimator and partial-density helpers.  It supports OLS with
controls, one-way absorbed fixed effects, and minimal IV / IV+FE designs
via a formula-string interface.

Key public functions:

- ``parse_reg_formula`` -- parse and validate a regression-formula string.
- ``compute_adjusted_probabilities`` -- estimate the adjusted joint
  P(Y=y, M=m | D=d) grid and mediator masses.
- ``compute_adjusted_probability_influences`` -- same as above plus
  observation-level OLS influence functions for sharp-null inference.
- ``compute_adjusted_mediator_masses`` -- estimate adjusted P(M=m | D=d)
  for continuous partial-density scaling.

All public functions validate inputs eagerly and raise ``ValueError`` or
``NotImplementedError`` on violation.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .preprocess import normalize_binary_support
from .results import _json_safe_float


_FORMULA_VARIABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


@dataclass(frozen=True)
class RegressionFormulaSpec:
    """Parsed representation of a regression-formula string.

    Attributes
    ----------
    raw : str
        The original formula string (whitespace-stripped).
    formula_kind : str
        One of ``'trivial'``, ``'controls'``, ``'fixed_effects'``, ``'iv'``,
        or ``'iv_fixed_effects'``.
    treatment : str
        The treatment variable name.
    controls : tuple of str
        Control variable names (may be empty).
    fixed_effects : tuple of str
        One-way fixed-effect variable names (may be empty).
    endogenous : str or None
        The endogenous variable in IV designs (always the treatment if set).
    instruments : tuple of str
        Instrumental variable names (may be empty).

    See Also
    --------
    parse_reg_formula : Factory function that produces this dataclass.
    """

    raw: str
    formula_kind: str
    treatment: str
    controls: tuple[str, ...]
    fixed_effects: tuple[str, ...]
    endogenous: str | None
    instruments: tuple[str, ...]

    @property
    def variables(self) -> tuple[str, ...]:
        terms = [self.treatment, *self.controls, *self.fixed_effects, *self.instruments]
        return tuple(dict.fromkeys(_term_to_variable(term) for term in terms if term != "1"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "formula_kind": self.formula_kind,
            "treatment": self.treatment,
            "controls": list(self.controls),
            "fixed_effects": list(self.fixed_effects),
            "endogenous": self.endogenous,
            "instruments": list(self.instruments),
            "variables": list(self.variables),
        }


@dataclass(frozen=True)
class AdjustedProbabilityResult:
    """Result container for the adjusted joint probability grid.

    Attributes
    ----------
    p_ym_d0 : dict
        Mapping (y_value, m_value) -> P(Y=y, M=m | D=0).
    p_ym_d1 : dict
        Mapping (y_value, m_value) -> P(Y=y, M=m | D=1).
    p_m_d0 : dict
        Mediator marginal masses P(M=m | D=0).
    p_m_d1 : dict
        Mediator marginal masses P(M=m | D=1).
    y_values : list
        Ordered outcome support levels.
    m_values : list
        Ordered mediator support levels.
    diagnostics : dict
        Estimation diagnostics including grid-contract checks.
    """

    p_ym_d0: dict[tuple[object, object], float]
    p_ym_d1: dict[tuple[object, object], float]
    p_m_d0: dict[object, float]
    p_m_d1: dict[object, float]
    y_values: list[object]
    m_values: list[object]
    diagnostics: dict[str, Any]

    @property
    def probability_row_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for treatment_value, probabilities in ((0, self.p_ym_d0), (1, self.p_ym_d1)):
            for record in _adjusted_probability_records(probabilities):
                records.append(
                    {
                        "treatment": treatment_value,
                        "y": record["y"],
                        "m": record["m"],
                        "probability": record["probability"],
                        "probability_is_finite": record["probability_is_finite"],
                        "probability_nonfinite": record["probability_nonfinite"],
                    }
                )
        return records

    @property
    def mediator_mass_row_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for treatment_value, masses in ((0, self.p_m_d0), (1, self.p_m_d1)):
            for record in _adjusted_mediator_mass_records(masses):
                records.append(
                    {
                        "treatment": treatment_value,
                        "m": record["m"],
                        "mass": record["mass"],
                        "mass_is_finite": record["mass_is_finite"],
                        "mass_nonfinite": record["mass_nonfinite"],
                    }
                )
        return records

    def to_dict(self) -> dict[str, Any]:
        return {
            "p_ym_d0_records": _adjusted_probability_records(self.p_ym_d0),
            "p_ym_d1_records": _adjusted_probability_records(self.p_ym_d1),
            "probability_row_records": self.probability_row_records,
            "p_m_d0_records": _adjusted_mediator_mass_records(self.p_m_d0),
            "p_m_d1_records": _adjusted_mediator_mass_records(self.p_m_d1),
            "mediator_mass_row_records": self.mediator_mass_row_records,
            "y_values": [_json_safe_label(value) for value in self.y_values],
            "m_values": [_json_safe_label(value) for value in self.m_values],
            "diagnostics": _json_safe_payload(self.diagnostics),
        }


@dataclass(frozen=True)
class AdjustedProbabilityInfluenceResult:
    """Result container for adjusted probabilities with influence functions.

    Extends ``AdjustedProbabilityResult`` by storing per-observation influence
    function values for each (y, m) cell, enabling variance estimation under
    the sharp null.

    Attributes
    ----------
    probabilities : AdjustedProbabilityResult
        The underlying probability grid.
    p_ym_d0_influence : dict
        Mapping (y, m) -> 1-D influence array for D=0.
    p_ym_d1_influence : dict
        Mapping (y, m) -> 1-D influence array for D=1.
    row_index : pd.Index
        Row index of the complete-case sample.
    diagnostics : dict
        Estimation and influence-function diagnostics.
    """

    probabilities: AdjustedProbabilityResult
    p_ym_d0_influence: dict[tuple[object, object], np.ndarray]
    p_ym_d1_influence: dict[tuple[object, object], np.ndarray]
    row_index: pd.Index
    diagnostics: dict[str, Any]

    @property
    def influence_row_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for treatment_value, influences in (
            (0, self.p_ym_d0_influence),
            (1, self.p_ym_d1_influence),
        ):
            for record in _adjusted_influence_records(influences):
                records.append(
                    {
                        "treatment": treatment_value,
                        "y": record["y"],
                        "m": record["m"],
                        "influence": record["influence"],
                        "influence_is_finite": record["influence_is_finite"],
                        "influence_nonfinite": record["influence_nonfinite"],
                    }
                )
        return records

    def to_dict(self) -> dict[str, Any]:
        return {
            "probabilities": self.probabilities.to_dict(),
            "p_ym_d0_influence_records": _adjusted_influence_records(
                self.p_ym_d0_influence
            ),
            "p_ym_d1_influence_records": _adjusted_influence_records(
                self.p_ym_d1_influence
            ),
            "influence_row_records": self.influence_row_records,
            "row_index": [_json_safe_label(value) for value in self.row_index.tolist()],
            "diagnostics": _json_safe_payload(self.diagnostics),
        }


@dataclass(frozen=True)
class AdjustedMediatorMassResult:
    """Result container for adjusted mediator marginal masses.

    Attributes
    ----------
    p_m_d0 : dict
        Mapping m_value -> P(M=m | D=0).
    p_m_d1 : dict
        Mapping m_value -> P(M=m | D=1).
    m_values : list
        Ordered mediator support levels.
    diagnostics : dict
        Estimation diagnostics including mass-contract checks.
    """

    p_m_d0: dict[object, float]
    p_m_d1: dict[object, float]
    m_values: list[object]
    diagnostics: dict[str, Any]

    @property
    def mediator_mass_row_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for treatment_value, masses in ((0, self.p_m_d0), (1, self.p_m_d1)):
            for record in _adjusted_mediator_mass_records(masses):
                records.append(
                    {
                        "treatment": treatment_value,
                        "m": record["m"],
                        "mass": record["mass"],
                        "mass_is_finite": record["mass_is_finite"],
                        "mass_nonfinite": record["mass_nonfinite"],
                    }
                )
        return records

    def to_dict(self) -> dict[str, Any]:
        return {
            "p_m_d0_records": _adjusted_mediator_mass_records(self.p_m_d0),
            "p_m_d1_records": _adjusted_mediator_mass_records(self.p_m_d1),
            "mediator_mass_row_records": self.mediator_mass_row_records,
            "m_values": [_json_safe_label(value) for value in self.m_values],
            "diagnostics": _json_safe_payload(self.diagnostics),
        }


def _adjusted_probability_records(
    probabilities: dict[tuple[object, object], float],
) -> list[dict[str, Any]]:
    """Convert a probability grid dict to JSON-safe row records."""
    records: list[dict[str, Any]] = []
    for (y_value, m_value), probability in probabilities.items():
        probability_value, is_finite, nonfinite = _json_safe_float(probability)
        records.append(
            {
                "y": _json_safe_label(y_value),
                "m": _json_safe_label(m_value),
                "probability": probability_value,
                "probability_is_finite": is_finite,
                "probability_nonfinite": nonfinite,
            }
        )
    return records


def _adjusted_mediator_mass_records(
    masses: dict[object, float],
) -> list[dict[str, Any]]:
    """Convert mediator mass dict to JSON-safe row records."""
    records: list[dict[str, Any]] = []
    for m_value, mass in masses.items():
        mass_value, is_finite, nonfinite = _json_safe_float(mass)
        records.append(
            {
                "m": _json_safe_label(m_value),
                "mass": mass_value,
                "mass_is_finite": is_finite,
                "mass_nonfinite": nonfinite,
            }
        )
    return records


def _adjusted_influence_records(
    influences: dict[tuple[object, object], np.ndarray],
) -> list[dict[str, Any]]:
    """Convert influence-function arrays to JSON-safe row records."""
    records: list[dict[str, Any]] = []
    for (y_value, m_value), influence in influences.items():
        values = []
        finite_flags = []
        nonfinite_markers = []
        for raw_value in np.asarray(influence, dtype=float).tolist():
            value, is_finite, nonfinite = _json_safe_float(raw_value)
            values.append(value)
            finite_flags.append(is_finite)
            nonfinite_markers.append(nonfinite)
        records.append(
            {
                "y": _json_safe_label(y_value),
                "m": _json_safe_label(m_value),
                "influence": values,
                "influence_is_finite": finite_flags,
                "influence_nonfinite": nonfinite_markers,
            }
        )
    return records


def _json_safe_label(value: Any) -> Any:
    """Coerce a support label to a JSON-serializable value."""
    if isinstance(value, tuple):
        return [_json_safe_label(item) for item in value]
    if isinstance(value, list):
        return [_json_safe_label(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe_label(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe_label(value.item())
    if isinstance(value, pd.Interval):
        return str(value)
    if isinstance(value, float):
        safe_value, is_finite, nonfinite = _json_safe_float(value)
        return safe_value if is_finite else nonfinite
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _json_safe_payload(value: Any) -> Any:
    """Recursively coerce a nested payload to JSON-safe types."""
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
    if isinstance(value, pd.Interval):
        return str(value)
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
    return value


def _adjusted_probability_grid_contract_diagnostics(
    *,
    p_ym_d0: dict[tuple[object, object], float],
    p_ym_d1: dict[tuple[object, object], float],
    p_m_d0: dict[object, float],
    p_m_d1: dict[object, float],
    y_values: Sequence[object],
    m_values: Sequence[object],
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Check that the probability grid satisfies the valid-for-bounds contract."""
    arm_diagnostics: dict[str, Any] = {}
    reconstruction_rows: list[dict[str, Any]] = []
    max_joint_total_gap = 0.0
    max_mediator_total_gap = 0.0
    max_reconstruction_gap = 0.0
    total_joint_nonfinite = 0
    total_joint_negative = 0
    total_joint_above_one = 0
    total_mediator_nonfinite = 0
    total_mediator_negative = 0
    total_mediator_above_one = 0

    for treatment_value, probabilities, mediator_masses in (
        (0, p_ym_d0, p_m_d0),
        (1, p_ym_d1, p_m_d1),
    ):
        joint_values = [
            float(probabilities[(y_value, m_value)])
            for y_value in y_values
            for m_value in m_values
        ]
        mediator_values = [float(mediator_masses[m_value]) for m_value in m_values]
        joint_nonfinite = sum(not np.isfinite(value) for value in joint_values)
        joint_negative = sum(np.isfinite(value) and value < 0.0 for value in joint_values)
        joint_above_one = sum(np.isfinite(value) and value > 1.0 for value in joint_values)
        mediator_nonfinite = sum(not np.isfinite(value) for value in mediator_values)
        mediator_negative = sum(np.isfinite(value) and value < 0.0 for value in mediator_values)
        mediator_above_one = sum(np.isfinite(value) and value > 1.0 for value in mediator_values)
        total_joint_nonfinite += int(joint_nonfinite)
        total_joint_negative += int(joint_negative)
        total_joint_above_one += int(joint_above_one)
        total_mediator_nonfinite += int(mediator_nonfinite)
        total_mediator_negative += int(mediator_negative)
        total_mediator_above_one += int(mediator_above_one)

        joint_total_mass = float(sum(joint_values))
        mediator_total_mass = float(sum(mediator_values))
        joint_total_gap = abs(joint_total_mass - 1.0) if np.isfinite(joint_total_mass) else float("inf")
        mediator_total_gap = (
            abs(mediator_total_mass - 1.0)
            if np.isfinite(mediator_total_mass)
            else float("inf")
        )
        max_joint_total_gap = max(max_joint_total_gap, joint_total_gap)
        max_mediator_total_gap = max(max_mediator_total_gap, mediator_total_gap)

        arm_reconstruction_gaps = []
        for m_value in m_values:
            joint_mass = float(sum(float(probabilities[(y_value, m_value)]) for y_value in y_values))
            mediator_mass = float(mediator_masses[m_value])
            gap = (
                abs(joint_mass - mediator_mass)
                if np.isfinite(joint_mass) and np.isfinite(mediator_mass)
                else float("inf")
            )
            max_reconstruction_gap = max(max_reconstruction_gap, gap)
            arm_reconstruction_gaps.append(gap)
            reconstruction_rows.append(
                {
                    "treatment": treatment_value,
                    "m": _json_safe_label(m_value),
                    "joint_mass": joint_mass,
                    "mediator_mass": mediator_mass,
                    "absolute_gap": gap,
                }
            )

        arm_valid = (
            joint_nonfinite == 0
            and joint_negative == 0
            and joint_above_one == 0
            and mediator_nonfinite == 0
            and mediator_negative == 0
            and mediator_above_one == 0
            and joint_total_gap <= tolerance
            and mediator_total_gap <= tolerance
            and all(gap <= tolerance for gap in arm_reconstruction_gaps)
        )
        arm_diagnostics[f"d{treatment_value}"] = {
            "joint_total_mass": joint_total_mass,
            "mediator_total_mass": mediator_total_mass,
            "joint_total_gap": joint_total_gap,
            "mediator_total_gap": mediator_total_gap,
            "joint_nonfinite_count": int(joint_nonfinite),
            "joint_negative_count": int(joint_negative),
            "joint_above_one_count": int(joint_above_one),
            "mediator_nonfinite_count": int(mediator_nonfinite),
            "mediator_negative_count": int(mediator_negative),
            "mediator_above_one_count": int(mediator_above_one),
            "valid_for_bounds": bool(arm_valid),
        }

    valid_for_bounds = (
        total_joint_nonfinite == 0
        and total_joint_negative == 0
        and total_joint_above_one == 0
        and total_mediator_nonfinite == 0
        and total_mediator_negative == 0
        and total_mediator_above_one == 0
        and max_joint_total_gap <= tolerance
        and max_mediator_total_gap <= tolerance
        and max_reconstruction_gap <= tolerance
    )
    return {
        "valid_for_bounds": bool(valid_for_bounds),
        "tolerance": tolerance,
        "arms": arm_diagnostics,
        "joint_nonfinite_count": total_joint_nonfinite,
        "joint_negative_count": total_joint_negative,
        "joint_above_one_count": total_joint_above_one,
        "mediator_nonfinite_count": total_mediator_nonfinite,
        "mediator_negative_count": total_mediator_negative,
        "mediator_above_one_count": total_mediator_above_one,
        "max_joint_total_gap": max_joint_total_gap,
        "max_mediator_total_gap": max_mediator_total_gap,
        "max_mediator_mass_reconstruction_gap": max_reconstruction_gap,
        "mediator_mass_reconstruction_rows": reconstruction_rows,
    }


def _adjusted_mediator_mass_grid_contract_diagnostics(
    *,
    p_m_d0: dict[object, float],
    p_m_d1: dict[object, float],
    m_values: Sequence[object],
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Check that mediator masses satisfy the valid-for-partial-density contract."""
    arm_diagnostics: dict[str, Any] = {}
    total_nonfinite = 0
    total_negative = 0
    total_above_one = 0
    max_total_gap = 0.0

    for treatment_value, mediator_masses in ((0, p_m_d0), (1, p_m_d1)):
        mass_values = [float(mediator_masses[m_value]) for m_value in m_values]
        nonfinite = sum(not np.isfinite(value) for value in mass_values)
        negative = sum(np.isfinite(value) and value < 0.0 for value in mass_values)
        above_one = sum(np.isfinite(value) and value > 1.0 for value in mass_values)
        total_nonfinite += int(nonfinite)
        total_negative += int(negative)
        total_above_one += int(above_one)

        total_mass = float(sum(mass_values))
        total_gap = abs(total_mass - 1.0) if np.isfinite(total_mass) else float("inf")
        max_total_gap = max(max_total_gap, total_gap)
        arm_valid = (
            nonfinite == 0
            and negative == 0
            and above_one == 0
            and total_gap <= tolerance
        )
        arm_diagnostics[f"d{treatment_value}"] = {
            "total_mass": total_mass,
            "total_gap": total_gap,
            "nonfinite_count": int(nonfinite),
            "negative_count": int(negative),
            "above_one_count": int(above_one),
            "valid_for_partial_density": bool(arm_valid),
        }

    valid_for_partial_density = (
        total_nonfinite == 0
        and total_negative == 0
        and total_above_one == 0
        and max_total_gap <= tolerance
    )
    return {
        "valid_for_partial_density": bool(valid_for_partial_density),
        "tolerance": tolerance,
        "arms": arm_diagnostics,
        "nonfinite_count": total_nonfinite,
        "negative_count": total_negative,
        "above_one_count": total_above_one,
        "max_total_gap": max_total_gap,
    }


def parse_reg_formula(reg_formula: str, *, d: str) -> RegressionFormulaSpec:
    """Parse and validate a regression-formula string.

    The supported formula subset uses ``~`` as the response separator, ``+``
    for additive terms, ``|`` for fixed-effect and IV partitions, and
    ``factor(var)`` for explicit categorical dummies.

    Supported formula kinds:

    - Trivial: ``~ d`` (intercept + treatment only).
    - Controls: ``~ d + x1 + x2``.
    - Fixed effects: ``~ d + x1 | fe``.
    - IV: ``~ x1 | d ~ z1``.
    - IV + FE: ``~ x1 | fe | d ~ z1``.

    Parameters
    ----------
    reg_formula : str
        Formula string beginning with ``~``.
    d : str
        The treatment variable name that must appear in the formula.

    Returns
    -------
    RegressionFormulaSpec
        Parsed frozen dataclass recording formula kind, treatment, controls,
        fixed effects, endogenous variable, and instruments.

    Raises
    ------
    ValueError
        If the formula is syntactically invalid, missing the treatment
        variable, contains repeated variables, or uses unsupported operators.

    Examples
    --------
    >>> from testmechs.regression import parse_reg_formula
    >>> spec = parse_reg_formula("~ D + age + factor(region)", d="D")
    >>> spec.formula_kind
    'controls'
    >>> spec.controls
    ('age', 'factor(region)')

    Notes
    -----
    The parser enforces that the treatment variable is not reused as a
    control, fixed effect, or instrument.  Interactions, polynomials, and
    nested expressions are not supported.

    See Also
    --------
    compute_adjusted_probabilities : Consumes the parsed formula.
    RegressionFormulaSpec : The returned dataclass.
    """
    _validate_scalar_column_name(d, argument="d")
    if not isinstance(reg_formula, str):
        raise ValueError("reg_formula must be a string in the supported reg_formula subset.")
    raw = reg_formula.strip()
    if not raw.startswith("~") or raw.startswith("~") is False or raw.count("~") > 2:
        raise ValueError("Unsupported reg_formula; use the supported reg_formula subset.")
    if raw.split("~", maxsplit=1)[0].strip():
        raise ValueError("Unsupported reg_formula; use the supported reg_formula subset.")

    body = raw[1:].strip()
    parts = [part.strip() for part in body.split("|")]
    if len(parts) not in {1, 2, 3} or any(part == "" for part in parts):
        raise ValueError("Unsupported reg_formula; use the supported reg_formula subset.")

    if len(parts) == 1:
        rhs_terms = _parse_terms(parts[0])
        if d not in rhs_terms:
            raise ValueError("The treatment variable must appear in the supported reg_formula subset.")
        _reject_treatment_terms((term for term in rhs_terms if term != d), d=d)
        _reject_repeated_formula_variables(rhs_terms)
        controls = tuple(term for term in rhs_terms if term != d)
        return RegressionFormulaSpec(
            raw=raw,
            formula_kind="trivial" if not controls else "controls",
            treatment=d,
            controls=controls,
            fixed_effects=(),
            endogenous=None,
            instruments=(),
        )

    if len(parts) == 2 and "~" not in parts[1]:
        rhs_terms = _parse_terms(parts[0])
        if d not in rhs_terms:
            raise ValueError("The treatment variable must appear in the supported reg_formula subset.")
        _reject_treatment_terms((term for term in rhs_terms if term != d), d=d)
        fixed_effects = tuple(_parse_terms(parts[1]))
        _reject_treatment_terms(fixed_effects, d=d)
        _reject_repeated_formula_variables((*rhs_terms, *fixed_effects))
        return RegressionFormulaSpec(
            raw=raw,
            formula_kind="fixed_effects",
            treatment=d,
            controls=tuple(term for term in rhs_terms if term != d),
            fixed_effects=fixed_effects,
            endogenous=None,
            instruments=(),
        )

    if len(parts) == 2:
        endogenous, instruments = _parse_iv_part(parts[1], d=d)
        controls = tuple(_parse_terms(parts[0]))
        _reject_treatment_terms(controls, d=d)
        _reject_repeated_formula_variables((*controls, *instruments))
        return RegressionFormulaSpec(
            raw=raw,
            formula_kind="iv",
            treatment=d,
            controls=controls,
            fixed_effects=(),
            endogenous=endogenous,
            instruments=tuple(instruments),
        )

    endogenous, instruments = _parse_iv_part(parts[2], d=d)
    controls = tuple(_parse_terms(parts[0]))
    fixed_effects = tuple(_parse_terms(parts[1]))
    _reject_treatment_terms(controls, d=d)
    _reject_treatment_terms(fixed_effects, d=d)
    _reject_repeated_formula_variables((*controls, *fixed_effects, *instruments))
    return RegressionFormulaSpec(
        raw=raw,
        formula_kind="iv_fixed_effects",
        treatment=d,
        controls=controls,
        fixed_effects=fixed_effects,
        endogenous=endogenous,
        instruments=tuple(instruments),
    )


def compute_adjusted_probabilities(
    *,
    df: pd.DataFrame,
    d: str,
    m: str | Sequence[str],
    y: str,
    reg_formula: str,
) -> AdjustedProbabilityResult:
    """Estimate an adjusted finite-support probability grid.

    Runs cell-indicator OLS regressions on the complete-case sample using the
    supplied formula to produce the adjusted joint distribution
    P(Y=y, M=m | D=d) for each treatment arm, together with implied mediator
    marginal masses.

    ``reg_formula`` supports the documented controls, one-way fixed effects,
    and minimal IV / IV+FE designs.  The result contains the adjusted joint
    probability grid, mediator masses, complete-case diagnostics, and
    strict-JSON row-record exports used by adjusted bounds and partial-density
    helpers.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis dataframe.
    d : str
        Treatment column name (must be binary after normalization).
    m : str or sequence of str
        Mediator column name(s).  Multiple mediators are collapsed into a
        single vector-valued mediator.
    y : str
        Outcome column name.
    reg_formula : str
        Formula string in the supported subset.

    Returns
    -------
    AdjustedProbabilityResult
        Frozen dataclass with probability grids, mediator masses, and
        diagnostics.

    Raises
    ------
    ValueError
        If required columns are missing, role constraints are violated,
        no complete observations remain, or the design matrix lacks
        treatment variation.

    Examples
    --------
    >>> import pandas as pd
    >>> from testmechs.regression import compute_adjusted_probabilities
    >>> df = pd.DataFrame({
    ...     "D": [0,0,0,1,1,1], "M": [0,1,0,1,0,1],
    ...     "Y": ["a","b","a","b","a","b"], "x": [1,2,3,4,5,6],
    ... })
    >>> result = compute_adjusted_probabilities(df=df, d="D", m="M", y="Y", reg_formula="~ D + x")
    >>> sorted(result.p_ym_d1.keys())  # doctest: +SKIP
    [('a', 0), ('a', 1), ('b', 0), ('b', 1)]

    Notes
    -----
    Each cell probability is the OLS treatment coefficient from a regression
    of D * 1{Y=y, M=m} (for D=1) or (D-1) * 1{Y=y, M=m} (for D=0) on the
    design matrix.

    See Also
    --------
    parse_reg_formula : Formula parsing and validation.
    compute_adjusted_probability_influences : Adds influence functions.
    compute_adjusted_mediator_masses : Mediator-only masses for continuous Y.
    """

    _validate_scalar_column_name(d, argument="d")
    _validate_scalar_column_name(y, argument="y")
    _reject_treatment_outcome_role_overlap(d=d, y=y)
    spec = parse_reg_formula(reg_formula, d=d)
    mediator_columns = _mediator_columns(m)
    _reject_mediator_role_overlap(mediator_columns, d=d, y=y)
    _reject_outcome_grid_formula_role_reuse(spec, y=y, mediator_columns=mediator_columns)
    data, treatment_diagnostics = _complete_case_regression_data(
        df=df,
        d=d,
        required_columns=[d, y, *mediator_columns, *spec.variables],
    )
    data, mediator_column = _collapse_mediator_columns(data=data, mediator_columns=mediator_columns)

    y_values = _ordered_support_values(data[y])
    m_values = _ordered_support_values(data[mediator_column])
    p_ym_d0: dict[tuple[object, object], float] = {}
    p_ym_d1: dict[tuple[object, object], float] = {}

    design = _build_design(data=data, spec=spec)
    for y_value in y_values:
        for m_value in m_values:
            indicator = ((data[y] == y_value) & (data[mediator_column] == m_value)).astype(float).to_numpy()
            p_ym_d0[(y_value, m_value)] = _estimate_treatment_coefficient(
                lhs=(data[d].to_numpy(dtype=float) - 1.0) * indicator,
                design=design,
            )
            p_ym_d1[(y_value, m_value)] = _estimate_treatment_coefficient(
                lhs=data[d].to_numpy(dtype=float) * indicator,
                design=design,
            )

    p_m_d0 = {
        m_value: float(sum(p_ym_d0[(y_value, m_value)] for y_value in y_values))
        for m_value in m_values
    }
    p_m_d1 = {
        m_value: float(sum(p_ym_d1[(y_value, m_value)] for y_value in y_values))
        for m_value in m_values
    }
    probabilities = [*p_ym_d0.values(), *p_ym_d1.values()]
    probability_grid_contract = _adjusted_probability_grid_contract_diagnostics(
        p_ym_d0=p_ym_d0,
        p_ym_d1=p_ym_d1,
        p_m_d0=p_m_d0,
        p_m_d1=p_m_d1,
        y_values=y_values,
        m_values=m_values,
    )

    return AdjustedProbabilityResult(
        p_ym_d0=p_ym_d0,
        p_ym_d1=p_ym_d1,
        p_m_d0=p_m_d0,
        p_m_d1=p_m_d1,
        y_values=y_values,
        m_values=m_values,
        diagnostics={
            "formula_kind": spec.formula_kind,
            "design_columns": design.columns,
            "probability_range": {
                "min": float(min(probabilities)),
                "max": float(max(probabilities)),
            },
            "probability_grid_contract": probability_grid_contract,
            "n_obs_used": int(len(data)),
            **treatment_diagnostics,
            "mediator_columns": list(mediator_columns),
            "mediator_dimension": len(mediator_columns),
            "vector_mediator": len(mediator_columns) > 1,
            "support_ordering": (
                "Y and mediator support are ordered using natural comparisons where possible "
                "and deterministic label keys otherwise"
            ),
        },
    )


def compute_adjusted_probability_influences(
    *,
    df: pd.DataFrame,
    d: str,
    m: str | Sequence[str],
    y: str,
    reg_formula: str,
) -> AdjustedProbabilityInfluenceResult:
    """Estimate adjusted probabilities with per-observation influence functions.

    This extends ``compute_adjusted_probabilities`` by also returning
    observation-level OLS influence-function values for the treatment
    coefficient in each cell regression.  These are consumed by the sharp-null
    variance estimator for clustered or heteroskedasticity-robust inference.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis dataframe.
    d : str
        Treatment column name.
    m : str or sequence of str
        Scalar mediator column name (vector mediators not yet supported).
    y : str
        Outcome column name.
    reg_formula : str
        Formula string (must be ``'trivial'``, ``'controls'``, or
        ``'fixed_effects'`` kind).

    Returns
    -------
    AdjustedProbabilityInfluenceResult
        Frozen dataclass with probability grid, per-cell influence arrays,
        row index, and diagnostics.

    Raises
    ------
    NotImplementedError
        If the formula kind is IV/IV+FE or a vector mediator is supplied.
    ValueError
        If required columns are missing or role constraints are violated.

    Notes
    -----
    Influence functions are computed as the (design * residual) @ bread
    projection, yielding one scalar per observation per cell.

    See Also
    --------
    compute_adjusted_probabilities : Without influence functions.
    """
    _validate_scalar_column_name(d, argument="d")
    _validate_scalar_column_name(y, argument="y")
    _reject_treatment_outcome_role_overlap(d=d, y=y)
    spec = parse_reg_formula(reg_formula, d=d)
    if spec.formula_kind not in {"trivial", "controls", "fixed_effects"}:
        raise NotImplementedError(
            "Adjusted probability influence functions currently support only "
            "trivial, controls, and fixed_effects formulas."
        )
    mediator_columns = _mediator_columns(m)
    _reject_mediator_role_overlap(mediator_columns, d=d, y=y)
    _reject_outcome_grid_formula_role_reuse(spec, y=y, mediator_columns=mediator_columns)
    if len(mediator_columns) > 1:
        raise NotImplementedError(
            "Adjusted probability influence functions currently support only "
            "scalar mediator sharp-null inference. Use compute_adjusted_probabilities() "
            "for vector mediator probability grids."
        )
    mediator_column = mediator_columns[0]
    data, treatment_diagnostics = _complete_case_regression_data(
        df=df,
        d=d,
        required_columns=[d, mediator_column, y, *spec.variables],
    )

    y_values = _ordered_support_values(data[y])
    m_values = _ordered_support_values(data[mediator_column])
    p_ym_d0: dict[tuple[object, object], float] = {}
    p_ym_d1: dict[tuple[object, object], float] = {}
    p_ym_d0_influence: dict[tuple[object, object], np.ndarray] = {}
    p_ym_d1_influence: dict[tuple[object, object], np.ndarray] = {}

    design = _build_design(data=data, spec=spec)
    d_values = data[d].to_numpy(dtype=float)
    for y_value in y_values:
        for m_value in m_values:
            key = (y_value, m_value)
            indicator = (
                (data[y] == y_value) & (data[mediator_column] == m_value)
            ).astype(float).to_numpy()
            p_ym_d0[key], p_ym_d0_influence[key] = _estimate_treatment_coefficient_and_influence(
                lhs=(d_values - 1.0) * indicator,
                design=design,
            )
            p_ym_d1[key], p_ym_d1_influence[key] = _estimate_treatment_coefficient_and_influence(
                lhs=d_values * indicator,
                design=design,
            )

    p_m_d0 = {
        m_value: float(sum(p_ym_d0[(y_value, m_value)] for y_value in y_values))
        for m_value in m_values
    }
    p_m_d1 = {
        m_value: float(sum(p_ym_d1[(y_value, m_value)] for y_value in y_values))
        for m_value in m_values
    }
    probabilities = [*p_ym_d0.values(), *p_ym_d1.values()]
    probability_grid_contract = _adjusted_probability_grid_contract_diagnostics(
        p_ym_d0=p_ym_d0,
        p_ym_d1=p_ym_d1,
        p_m_d0=p_m_d0,
        p_m_d1=p_m_d1,
        y_values=y_values,
        m_values=m_values,
    )
    diagnostics = {
        "formula_kind": spec.formula_kind,
        "design_columns": design.columns,
        "probability_range": {
            "min": float(min(probabilities)),
            "max": float(max(probabilities)),
        },
        "probability_grid_contract": probability_grid_contract,
        "n_obs_used": int(len(data)),
        **treatment_diagnostics,
        "mediator_columns": list(mediator_columns),
        "mediator_dimension": len(mediator_columns),
        "vector_mediator": False,
        "support_ordering": (
            "Y and mediator support are ordered using natural comparisons where possible "
            "and deterministic label keys otherwise"
        ),
    }
    return AdjustedProbabilityInfluenceResult(
        probabilities=AdjustedProbabilityResult(
            p_ym_d0=p_ym_d0,
            p_ym_d1=p_ym_d1,
            p_m_d0=p_m_d0,
            p_m_d1=p_m_d1,
            y_values=y_values,
            m_values=m_values,
            diagnostics=diagnostics,
        ),
        p_ym_d0_influence=p_ym_d0_influence,
        p_ym_d1_influence=p_ym_d1_influence,
        row_index=data.index,
        diagnostics={
            **diagnostics,
            "influence_function_contract": (
                "OLS treatment-coefficient influence functions from the "
                "adjusted joint probability regressions."
            ),
        },
    )


def compute_adjusted_mediator_masses(
    *,
    df: pd.DataFrame,
    d: str,
    m: str | Sequence[str],
    reg_formula: str,
) -> AdjustedMediatorMassResult:
    """Estimate adjusted mediator marginal masses P(M=m | D=d).

    Used by the continuous partial-density pathway where only mediator masses
    (not the full joint grid) are needed for scaling kernel-density shapes.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis dataframe.
    d : str
        Treatment column name.
    m : str or sequence of str
        Mediator column name(s).
    reg_formula : str
        Formula string in the supported subset.

    Returns
    -------
    AdjustedMediatorMassResult
        Frozen dataclass with mediator masses per arm and diagnostics.

    Raises
    ------
    ValueError
        If required columns are missing, role constraints are violated, or
        the design matrix lacks treatment variation.

    See Also
    --------
    compute_adjusted_probabilities : Full joint grid estimation.
    """
    _validate_scalar_column_name(d, argument="d")
    spec = parse_reg_formula(reg_formula, d=d)
    mediator_columns = _mediator_columns(m)
    _reject_mediator_role_overlap(mediator_columns, d=d)
    _reject_mediator_mass_formula_role_reuse(spec, mediator_columns=mediator_columns)
    data, treatment_diagnostics = _complete_case_regression_data(
        df=df,
        d=d,
        required_columns=[d, *mediator_columns, *spec.variables],
    )
    data, mediator_column = _collapse_mediator_columns(data=data, mediator_columns=mediator_columns)

    m_values = _ordered_support_values(data[mediator_column])
    design = _build_design(data=data, spec=spec)
    p_m_d0: dict[object, float] = {}
    p_m_d1: dict[object, float] = {}
    for m_value in m_values:
        indicator = (data[mediator_column] == m_value).astype(float).to_numpy()
        p_m_d0[m_value] = _estimate_treatment_coefficient(
            lhs=(data[d].to_numpy(dtype=float) - 1.0) * indicator,
            design=design,
        )
        p_m_d1[m_value] = _estimate_treatment_coefficient(
            lhs=data[d].to_numpy(dtype=float) * indicator,
            design=design,
        )
    masses = [*p_m_d0.values(), *p_m_d1.values()]
    mediator_mass_grid_contract = _adjusted_mediator_mass_grid_contract_diagnostics(
        p_m_d0=p_m_d0,
        p_m_d1=p_m_d1,
        m_values=m_values,
    )

    return AdjustedMediatorMassResult(
        p_m_d0=p_m_d0,
        p_m_d1=p_m_d1,
        m_values=m_values,
        diagnostics={
            "formula_kind": spec.formula_kind,
            "design_columns": design.columns,
            "mass_range": {
                "min": float(min(masses)),
                "max": float(max(masses)),
            },
            "mediator_mass_grid_contract": mediator_mass_grid_contract,
            "n_obs_used": int(len(data)),
            **treatment_diagnostics,
            "target": "mediator_mass",
            "mediator_columns": list(mediator_columns),
            "mediator_dimension": len(mediator_columns),
            "vector_mediator": len(mediator_columns) > 1,
            "support_ordering": (
                "mediator support is ordered using natural comparisons where possible "
                "and deterministic label keys otherwise"
            ),
        },
    )


@dataclass(frozen=True)
class _RegressionDesign:
    matrix: np.ndarray
    treatment_index: int
    columns: list[str]


def _complete_case_regression_data(
    *,
    df: pd.DataFrame,
    d: str,
    required_columns: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop incomplete rows and normalize the treatment to {0, 1}."""
    required_columns = list(dict.fromkeys(required_columns))
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        formatted = ", ".join(repr(column) for column in missing_columns)
        raise ValueError(f"df is missing required columns: {formatted}.")
    data = df.dropna(subset=required_columns).copy()
    if data.empty:
        raise ValueError("No complete observations remain after applying reg_formula.")
    treatment_support = normalize_binary_support(data[d], column=d)
    data[d] = treatment_support.transform(data[d])
    return data, treatment_support.diagnostics(
        original_key="original_treatment_levels",
        normalized_key="normalized_treatment_levels",
    )


def _validate_scalar_column_name(value: object, *, argument: str) -> None:
    """Raise ValueError if *value* is not a non-empty string."""
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{argument} must be a non-empty string column name.")


def _parse_terms(text: str) -> list[str]:
    """Split an additive formula segment into validated term strings."""
    raw_terms = [term.strip() for term in text.split("+")]
    if any(term == "" for term in raw_terms):
        raise ValueError("Unsupported reg_formula; use the supported reg_formula subset.")
    terms = [term for term in raw_terms if term != "1"]
    for term in terms:
        _validate_formula_term(term)
    return terms


def _validate_formula_term(term: str) -> None:
    """Raise if *term* is not a valid formula variable or factor() call."""
    variable = _term_to_variable(term)
    if not variable or not _FORMULA_VARIABLE_RE.match(variable):
        raise ValueError("Unsupported reg_formula; use the supported reg_formula subset.")
    if _is_factor_term(term):
        return
    if term != variable:
        raise ValueError("Unsupported reg_formula; use the supported reg_formula subset.")
    if any(operator in term for operator in ("*", ":", "^", "/", "(", ")")):
        raise ValueError("Unsupported reg_formula; use the supported reg_formula subset.")


def _reject_treatment_terms(terms: Iterable[str], *, d: str) -> None:
    """Raise if any term references the treatment variable."""
    if any(_term_to_variable(term) == d for term in terms):
        raise ValueError("The treatment variable cannot also be used as a control, fixed effect, or instrument.")


def _reject_repeated_formula_variables(terms: Iterable[str]) -> None:
    """Raise if any variable appears more than once."""
    variables = [_term_to_variable(term) for term in terms if term != "1"]
    if len(set(variables)) != len(variables):
        raise ValueError("Formula variables cannot be repeated across the supported reg_formula subset.")


def _ordered_support_values(series: pd.Series) -> list[object]:
    """Return ordered support values for the probability-grid axes."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        observed_values = list(pd.unique(series.dropna()))
        return [
            _normalize_support_value(category)
            for category in series.cat.categories
            if any(category == observed_value for observed_value in observed_values)
        ]
    return sorted(
        [_normalize_support_value(value) for value in pd.unique(series)],
        key=_support_sort_key,
    )


def _support_sort_key(value: object) -> tuple[object, ...]:
    """Deterministic sort key for support values."""
    if isinstance(value, tuple):
        return ("tuple", *(_support_sort_key(item) for item in value))
    normalized = _normalize_support_value(value)
    if isinstance(normalized, bool):
        return ("bool", int(normalized))
    if isinstance(normalized, (int, float)):
        return ("number", float(normalized))
    return (type(normalized).__name__, repr(normalized))


def _normalize_support_value(value: object) -> object:
    """Convert NumPy scalars to plain Python types."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except (AttributeError, ValueError):
            return value
    return value


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
    y: str | None = None,
) -> None:
    """Raise if mediator columns duplicate treatment or outcome."""
    reserved_columns = {d}
    if y is not None:
        reserved_columns.add(y)
    if any(column in reserved_columns for column in mediator_columns):
        raise ValueError("m must not include treatment or outcome columns.")


def _reject_treatment_outcome_role_overlap(*, d: str, y: str) -> None:
    """Raise if treatment and outcome reference the same column."""
    if d == y:
        raise ValueError("d and y must name distinct treatment and outcome columns.")


def _reject_outcome_grid_formula_role_reuse(
    spec: RegressionFormulaSpec,
    *,
    y: str,
    mediator_columns: tuple[str, ...],
) -> None:
    """Raise if formula variables reuse outcome or mediator columns."""
    reserved_columns = {y, *mediator_columns}
    if any(variable in reserved_columns for variable in spec.variables if variable != spec.treatment):
        raise ValueError("reg_formula variables must not reuse outcome or mediator columns.")


def _reject_mediator_mass_formula_role_reuse(
    spec: RegressionFormulaSpec,
    *,
    mediator_columns: tuple[str, ...],
) -> None:
    """Raise if formula variables reuse mediator columns."""
    reserved_columns = set(mediator_columns)
    if any(variable in reserved_columns for variable in spec.variables if variable != spec.treatment):
        raise ValueError("reg_formula variables must not reuse mediator columns.")


def _collapse_mediator_columns(
    *, data: pd.DataFrame, mediator_columns: tuple[str, ...]
) -> tuple[pd.DataFrame, str]:
    """Collapse multiple mediator columns into a single tuple-valued column."""
    if len(mediator_columns) == 1:
        return data, mediator_columns[0]
    mediator_column = "_tm_adjusted_mediator"
    suffix = 1
    while mediator_column in data.columns:
        suffix += 1
        mediator_column = f"_tm_adjusted_mediator_{suffix}"
    collapsed = data.copy()
    collapsed[mediator_column] = [
        tuple(_normalize_support_value(value) for value in row)
        for row in collapsed.loc[:, list(mediator_columns)].itertuples(index=False, name=None)
    ]
    return collapsed, mediator_column


def _parse_iv_part(text: str, *, d: str) -> tuple[str, list[str]]:
    """Parse the IV partition of a formula: endogenous ~ instruments."""
    sides = [side.strip() for side in text.split("~")]
    if len(sides) != 2:
        raise ValueError("Unsupported reg_formula; use the supported reg_formula subset.")
    endogenous = sides[0]
    if endogenous != d:
        raise ValueError("Only the treatment variable can be endogenous in the supported reg_formula subset.")
    instruments = _parse_terms(sides[1])
    if not instruments:
        raise ValueError("IV reg_formula requires at least one instrument.")
    _reject_treatment_terms(instruments, d=d)
    return endogenous, instruments


def _build_design(*, data: pd.DataFrame, spec: RegressionFormulaSpec) -> _RegressionDesign:
    """Construct the design matrix for a parsed formula spec."""
    base_columns = [_intercept_column(data)]
    base_names = ["Intercept"]
    base_columns.extend(_term_columns(data, spec.controls))
    base_names.extend(_term_names(data, spec.controls))
    base_columns.extend(_term_columns(data, spec.fixed_effects))
    base_names.extend(_term_names(data, spec.fixed_effects))

    if spec.formula_kind in {"iv", "iv_fixed_effects"}:
        first_stage_columns = [*base_columns, *_term_columns(data, spec.instruments)]
        first_stage = np.column_stack(first_stage_columns)
        base_matrix = np.column_stack(base_columns)
        if np.linalg.matrix_rank(first_stage) <= np.linalg.matrix_rank(base_matrix):
            raise ValueError(
                "IV reg_formula requires instruments that add variation beyond controls and fixed effects."
            )
        fitted_treatment = first_stage @ np.linalg.lstsq(
            first_stage,
            data[spec.treatment].to_numpy(dtype=float),
            rcond=None,
        )[0]
        matrix = np.column_stack([*base_columns, fitted_treatment])
        if np.linalg.matrix_rank(matrix) <= np.linalg.matrix_rank(base_matrix):
            raise ValueError(
                "IV reg_formula requires a relevant first stage for the treatment variable."
            )
        columns = [*base_names, f"fit_{spec.treatment}"]
        return _RegressionDesign(matrix=matrix, treatment_index=len(columns) - 1, columns=columns)

    treatment = data[spec.treatment].to_numpy(dtype=float)
    matrix = np.column_stack([*base_columns, treatment])
    base_matrix = np.column_stack(base_columns)
    if np.linalg.matrix_rank(matrix) <= np.linalg.matrix_rank(base_matrix):
        raise ValueError(
            "reg_formula requires treatment variation beyond controls and fixed effects."
        )
    columns = [*base_names, spec.treatment]
    return _RegressionDesign(matrix=matrix, treatment_index=len(columns) - 1, columns=columns)


def _estimate_treatment_coefficient(*, lhs: np.ndarray, design: _RegressionDesign) -> float:
    """OLS estimate of the treatment coefficient for a given LHS vector."""
    if np.var(lhs) == 0.0:
        return 0.0
    beta = np.linalg.lstsq(design.matrix, lhs, rcond=None)[0]
    return float(beta[design.treatment_index])


def _estimate_treatment_coefficient_and_influence(
    *,
    lhs: np.ndarray,
    design: _RegressionDesign,
) -> tuple[float, np.ndarray]:
    """OLS treatment coefficient plus per-observation influence function."""
    if np.var(lhs) == 0.0:
        return 0.0, np.zeros(design.matrix.shape[0], dtype=float)
    beta = np.linalg.lstsq(design.matrix, lhs, rcond=None)[0]
    residual = lhs - design.matrix @ beta
    bread = len(lhs) * np.linalg.pinv(design.matrix.T @ design.matrix)
    influence = (design.matrix * residual[:, None]) @ bread.T
    return float(beta[design.treatment_index]), influence[:, design.treatment_index]


def _intercept_column(data: pd.DataFrame) -> np.ndarray:
    """Return an all-ones intercept column."""
    return np.ones(len(data), dtype=float)


def _term_columns(data: pd.DataFrame, terms: Iterable[str]) -> list[np.ndarray]:
    """Build design-matrix columns from formula terms."""
    columns: list[np.ndarray] = []
    for term in terms:
        variable = _term_to_variable(term)
        series = data[variable]
        if _is_factor_term(term) or not pd.api.types.is_numeric_dtype(series):
            dummies = _dummy_columns(series, prefix=variable)
            columns.extend(dummies[column].to_numpy(dtype=float) for column in dummies.columns)
        else:
            values = series.to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(
                    f"{variable} must contain only finite numeric values for reg_formula design."
                )
            columns.append(values)
    return columns


def _term_names(data: pd.DataFrame, terms: Iterable[str]) -> list[str]:
    """Generate human-readable column names for design-matrix terms."""
    names: list[str] = []
    for term in terms:
        variable = _term_to_variable(term)
        series = data[variable]
        if _is_factor_term(term) or not pd.api.types.is_numeric_dtype(series):
            dummies = _dummy_columns(series, prefix=variable)
            names.extend(str(column) for column in dummies.columns)
        else:
            names.append(variable)
    return names


def _dummy_columns(series: pd.Series, *, prefix: str) -> pd.DataFrame:
    """Return drop-first dummy columns for a categorical-like series."""
    dummy_series = series
    if isinstance(dummy_series.dtype, pd.CategoricalDtype):
        dummy_series = dummy_series.cat.remove_unused_categories()
    return pd.get_dummies(dummy_series, prefix=prefix, drop_first=True, dtype=float)


def _is_factor_term(term: str) -> bool:
    """Check if *term* is a factor() wrapper."""
    return term.startswith("factor(") and term.endswith(")")


def _term_to_variable(term: str) -> str:
    """Extract the underlying variable name from a formula term."""
    if _is_factor_term(term):
        return term[len("factor(") : -1].strip()
    return term.strip()
