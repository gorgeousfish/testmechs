# Regression Adjustment

This module provides regression-adjusted probability estimation for use with
the Testing Mechanisms bounds and partial-density helpers. It supports
controls, one-way fixed effects, and minimal IV / IV+FE designs.

## `compute_adjusted_probabilities()`

```python
testmechs.compute_adjusted_probabilities(
    *,
    df: pd.DataFrame,
    d: str,
    m: str | Sequence[str],
    y: str,
    reg_formula: str,
) -> AdjustedProbabilityResult
```

### Description

Estimates an adjusted finite-support probability grid by running cell-indicator
OLS regressions on the complete-case sample. Produces the adjusted joint
distribution P(Y=y, M=m | D=d) for each treatment arm, together with implied
mediator marginal masses.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `df` | `pd.DataFrame` | — | Analysis data frame. |
| `d` | `str` | — | Treatment column (must be binary after normalization). |
| `m` | `str \| Sequence[str]` | — | Mediator column(s). Multiple collapsed into vector. |
| `y` | `str` | — | Outcome column. |
| `reg_formula` | `str` | — | Regression formula (see [Formula Syntax](#formula-syntax)). |

### Returns

`AdjustedProbabilityResult` with attributes:

- `p_ym_d0: dict` — Mapping (y, m) → P(Y=y, M=m | D=0)
- `p_ym_d1: dict` — Mapping (y, m) → P(Y=y, M=m | D=1)
- `p_m_d0: dict` — Mediator marginal P(M=m | D=0)
- `p_m_d1: dict` — Mediator marginal P(M=m | D=1)
- `y_values: list` — Ordered outcome support
- `m_values: list` — Ordered mediator support
- `diagnostics: dict` — Grid-contract checks, complete-case diagnostics

Additional properties:

- `probability_row_records` — Long-form records with explicit `treatment` field
- `mediator_mass_row_records` — Mediator mass records
- `to_dict()` → strict-JSON-safe payload

### Example

```python
import testmechs

result = testmechs.compute_adjusted_probabilities(
    df=df, d="treat", m="mediator", y="outcome",
    reg_formula="~ treat + age + income"
)
print(result.p_ym_d1)  # Adjusted joint probabilities under D=1
print(result.p_m_d0)   # Adjusted mediator masses under D=0
```

---

## `parse_reg_formula()`

```python
testmechs.parse_reg_formula(reg_formula: str, *, d: str) -> RegressionFormulaSpec
```

### Description

Parses and validates a regression-formula string. Returns a frozen dataclass
recording the formula kind, treatment, controls, fixed effects, endogenous
variable, and instruments.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `reg_formula` | `str` | — | Formula string beginning with `~`. |
| `d` | `str` | — | Treatment variable name that must appear in the formula. |

### Returns

`RegressionFormulaSpec` with attributes:

- `raw: str` — Original formula string
- `formula_kind: str` — One of `'trivial'`, `'controls'`, `'fixed_effects'`, `'iv'`, `'iv_fixed_effects'`
- `treatment: str` — Treatment variable name
- `controls: tuple[str, ...]` — Control variables
- `fixed_effects: tuple[str, ...]` — Fixed-effect variables
- `endogenous: str | None` — Endogenous variable (IV designs)
- `instruments: tuple[str, ...]` — Instruments (IV designs)

### Raises

- `ValueError` if `reg_formula` is not a string or doesn't begin with `~`.
- `ValueError` for missing formula columns, repeated variables, treatment reuse.
- `ValueError` if non-IV designs lack treatment variation.
- `ValueError` if IV designs lack a relevant first stage.

(formula-syntax)=
### Formula Syntax

The parser supports the following formula forms:

| Kind | Syntax | Example |
| --- | --- | --- |
| Trivial | `~ d` | `"~ treat"` |
| Controls | `~ d + x1 + x2` | `"~ treat + age + income"` |
| Fixed effects | `~ d + x1 \| fe` | `"~ treat + age \| district"` |
| IV | `~ x1 \| d ~ z1` | `"~ age \| treat ~ distance"` |
| IV + FE | `~ x1 \| fe \| d ~ z1` | `"~ age \| district \| treat ~ distance"` |

**Supported term forms:**

- Variable names: `x1`, `age_baseline`
- Factor wrappers: `factor(variable)` for explicit categorical dummies

### Example

```python
import testmechs

spec = testmechs.parse_reg_formula("~ treat + age + factor(region)", d="treat")
print(spec.formula_kind)  # 'controls'
print(spec.controls)      # ('age', 'region')
print(spec.treatment)     # 'treat'

spec_fe = testmechs.parse_reg_formula("~ treat + age | district", d="treat")
print(spec_fe.formula_kind)   # 'fixed_effects'
print(spec_fe.fixed_effects)  # ('district',)
```

---

## `compute_adjusted_probability_influences()`

```python
testmechs.compute_adjusted_probability_influences(
    *,
    df: pd.DataFrame,
    d: str,
    m: str | Sequence[str],
    y: str,
    reg_formula: str,
) -> AdjustedProbabilityInfluenceResult
```

### Description

Extends `compute_adjusted_probabilities()` by also computing per-observation
influence function values for each (y, m) cell, enabling variance estimation
under the sharp null for adjusted tests.

### Returns

`AdjustedProbabilityInfluenceResult` with:

- `probabilities: AdjustedProbabilityResult` — The underlying probability grid
- `p_ym_d0_influence: dict` — (y, m) → 1-D influence array for D=0
- `p_ym_d1_influence: dict` — (y, m) → 1-D influence array for D=1
- `row_index: pd.Index` — Row index of the complete-case sample
- `diagnostics: dict` — Estimation and influence diagnostics

---

## `compute_adjusted_mediator_masses()`

```python
testmechs.compute_adjusted_mediator_masses(
    *,
    df: pd.DataFrame,
    d: str,
    m: str | Sequence[str],
    reg_formula: str,
) -> AdjustedMediatorMassResult
```

### Description

Computes adjusted mediator marginal masses P(M=m | D=d) using the same
regression framework, without requiring the outcome column.

### Returns

`AdjustedMediatorMassResult` with:

- `p_m_d0: dict` — Mapping m → P(M=m | D=0)
- `p_m_d1: dict` — Mapping m → P(M=m | D=1)
- `m_values: list` — Ordered mediator support
- `diagnostics: dict` — Mass-contract checks

---

## Adjusted Probability Workflow

A typical workflow combining regression adjustment with bounds:

```python
import pandas as pd
from importlib.resources import files
import testmechs

fixtures = files("testmechs.resources.fixtures")
df = pd.read_csv(fixtures / "baranov_mother_data.csv")

# 1. Adjusted lower bound
bound = testmechs.lb_frac_affected(
    df=df, d="treat", m="grandmother", y="motherfinancial",
    reg_formula="~ treat + age_baseline"
)
print(f"Adjusted lower bound: {bound.lower_bound:.4f}")

# 2. Adjusted ADE bounds
ade = testmechs.bounds_ade_ats(
    df=df, d="treat", m="grandmother", y="motherfinancial",
    at_group=1, reg_formula="~ treat + age_baseline"
)
print(f"Adjusted ADE: [{ade.lower_bound:.4f}, {ade.upper_bound:.4f}]")

# 3. Adjusted partial density
fig = testmechs.partial_density_plot(
    df=df, d="treat", m="grandmother", y="motherfinancial",
    reg_formula="~ treat + age_baseline"
)
fig.savefig("adjusted_partial_density.pdf")
```
