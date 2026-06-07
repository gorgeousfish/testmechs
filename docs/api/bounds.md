# Bound Estimators

This module implements lower bounds on the fraction of always-takers affected,
ADE (average direct effect) partial-identification bounds, and breakdown-point
analysis for defier-share relaxations.

## `lb_frac_affected()`

```python
testmechs.lb_frac_affected(
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
) -> LowerBoundResult
```

### Description

Computes a lower bound on the fraction of always-takers whose outcome is
affected by treatment outside the mediator channel. This equals a lower bound
on the total variation distance between Y(1,m) and Y(0,m) for always-takers
with M(1) = M(0) = m.

When `at_group` is `None`, returns a population-weighted average across all
always-taker groups.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `df` | `pd.DataFrame \| None` | `None` | Analysis data frame. Exactly one of `df` or `data_path`. |
| `data_path` | `str \| Path \| None` | `None` | Path to CSV file. |
| `d` | `str` | — | Binary treatment column. |
| `m` | `str \| Sequence[str]` | — | Mediator column(s). Sequence for vector mediator with elementwise monotonicity. |
| `y` | `str` | — | Outcome column. |
| `at_group` | `object \| None` | `None` | Target always-taker group. `None` for pooled. |
| `num_y_bins` | `int \| None` | `None` | Discretize Y into quantile bins. |
| `max_defiers_share` | `float` | `0.0` | Upper bound on defier proportion. 0.0 = strict monotonicity. |
| `allow_min_defiers` | `bool` | `False` | Use exact minimum compatible defier share. |
| `return_min_defiers` | `bool` | `False` | Include minimum compatible cap in diagnostics. |
| `reg_formula` | `str \| None` | `None` | Regression formula for adjusted bounds. |

### Returns

`LowerBoundResult` with attributes:

- `lower_bound: float` — Estimated lower bound (may be `inf` if no bite)
- `estimand: str` — Human-readable estimand label
- `at_group: object | None` — Target group
- `restriction: str` — Active monotonicity restriction (e.g. "ordered")
- `diagnostics: dict` — Solver and support diagnostics
- `to_frame()` → one-row summary DataFrame
- `to_dict()` → strict-JSON-safe payload

### Example

```python
import testmechs

bound = testmechs.lb_frac_affected(
    df=df, d="treat", m="mediator", y="outcome", at_group=1
)
print(f"Lower bound: {bound.lower_bound:.4f}")
print(bound.to_frame()[["result", "lower_bound", "lower_bound_status"]])

# With relaxed defier cap
bound_relaxed = testmechs.lb_frac_affected(
    df=df, d="treat", m="mediator", y="outcome",
    max_defiers_share=0.01
)

# Use exact minimum compatible defier share
bound_min = testmechs.lb_frac_affected(
    df=df, d="treat", m="mediator", y="outcome",
    allow_min_defiers=True
)
```

---

## `bounds_ade_ats()`

```python
testmechs.bounds_ade_ats(
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
) -> ADEBoundsResult
```

### Description

Computes sharp bounds on E[Y(1,k) - Y(0,k) | G = kk], the average direct
effect of treatment on the outcome for k-always-takers. Uses Lee-style
trimming on the conditional outcome distribution within the identified
always-taker subpopulation.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `df` | `pd.DataFrame \| None` | `None` | Analysis data frame. |
| `data_path` | `str \| Path \| None` | `None` | Path to CSV file. |
| `d` | `str` | — | Binary treatment column. |
| `m` | `str \| Sequence[str]` | — | Mediator column(s). |
| `y` | `str` | — | Outcome column. |
| `at_group` | `object` | `1` | Target always-taker group value. |
| `max_defiers_share` | `float` | `0.0` | Upper bound on defier proportion. |
| `allow_min_defiers` | `bool` | `False` | Use exact minimum compatible defier cap. |
| `reg_formula` | `str \| None` | `None` | Regression formula for adjusted bounds. |

### Returns

`ADEBoundsResult` with attributes:

- `lower_bound: float | None` — Lower ADE bound (None if no bite)
- `upper_bound: float | None` — Upper ADE bound (None if no bite)
- `at_group: object` — Target group
- `restriction: str` — Active monotonicity restriction
- `diagnostics: dict` — Theta, trimming quantiles, treatment-arm masses
- `to_frame()` → summary DataFrame
- `to_dict()` → strict-JSON-safe payload

### Example

```python
ade = testmechs.bounds_ade_ats(
    df=df, d="treat", m="mediator", y="outcome", at_group=1
)
print(f"ADE bounds: [{ade.lower_bound:.4f}, {ade.upper_bound:.4f}]")

# With regression adjustment
ade_adj = testmechs.bounds_ade_ats(
    df=df, d="treat", m="mediator", y="outcome",
    at_group=1, reg_formula="~ treat + age_baseline"
)
```

### Notes

- When `theta_kk_min` is zero, the result reports explicit no-bite diagnostics
  rather than returning an uninterpretable numeric interval.
- Vector mediators use tuple support normalization and Lee-style trimmed expectations.
- ADE trimming uses raw finite numeric outcome support; does not accept `num_y_bins`.

---

## `breakdown_defier_share()`

```python
testmechs.breakdown_defier_share(
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
) -> LowerBoundResult
```

### Description

Finds the defier-share breakdown point: the minimum `max_defiers_share`
value at which `lb_frac_affected()` returns a lower bound of zero. This is
the smallest relaxation of monotonicity that eliminates evidence against the
sharp null of full mediation.

Uses binary search between the minimum compatible defier share and 1.0.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `df` | `pd.DataFrame \| None` | `None` | Analysis data frame. |
| `data_path` | `str \| Path \| None` | `None` | Path to CSV file. |
| `d` | `str` | — | Binary treatment column. |
| `m` | `str \| Sequence[str]` | — | Mediator column(s). |
| `y` | `str` | — | Outcome column. |
| `at_group` | `object \| None` | `None` | Target always-taker group. |
| `num_y_bins` | `int \| None` | `None` | Outcome discretization. |
| `reg_formula` | `str \| None` | `None` | Regression formula for adjusted bounds. |
| `tol` | `float` | `1e-4` | Bisection convergence tolerance. |
| `max_iterations` | `int` | `80` | Maximum bisection iterations. |

### Returns

`LowerBoundResult` where `lower_bound` is the breakdown defier-share cap,
with bracket precision diagnostics.

### Example

```python
breakdown = testmechs.breakdown_defier_share(
    df=df, d="treat", m="mediator", y="outcome"
)
print(f"Breakdown cap: {breakdown.lower_bound:.4f}")
```

---

## `ci_TV()`

See {ref}`Sharp Null Tests: ci_TV() <ci-tv>` for the TV confidence
interval function, which is also related to bounds estimation.

---

## `theta_kk_min_ordered_monotone()`

```python
testmechs.theta_kk_min_ordered_monotone(
    *,
    p_m_given_d0: Mapping[object, float] | Sequence[float],
    p_m_given_d1: Mapping[object, float] | Sequence[float],
    mediator_order: Sequence[object] | None = None,
) -> dict[object, float]
```

### Description

Computes the minimum always-taker shares theta_{kk} under ordered monotonicity
for each mediator level. A utility function for understanding the identified
always-taker subpopulation before running bounds.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `p_m_given_d0` | `Mapping \| Sequence` | — | P(M\|D=0) distribution. |
| `p_m_given_d1` | `Mapping \| Sequence` | — | P(M\|D=1) distribution. |
| `mediator_order` | `Sequence \| None` | `None` | Explicit mediator ordering. Required if levels are not naturally orderable. |

### Returns

`dict[object, float]` — Mapping from mediator level to minimum theta_{kk}.

---

## R Package Correspondence

| Python | R | Notes |
| --- | --- | --- |
| `lb_frac_affected()` | `lb_frac_affected()` | Same interface; Python uses SciPy LP |
| `bounds_ade_ats()` | `bounds_ade_ats()` | Same interface; Python requires explicit `allow_min_defiers=True` |
| `breakdown_defier_share()` | `breakdown_defier_share()` | Same binary-search logic |
| `theta_kk_min_ordered_monotone()` | Internal R helper | Exposed as public API in Python |

### Parameter Name Mapping

| Python parameter | R parameter | Notes |
| --- | --- | --- |
| `df` | `data` | Python also accepts `data_path` for CSV loading |
| `d` | `d` | Identical |
| `m` | `m` | Python accepts `Sequence[str]` for vector mediators |
| `y` | `y` | Identical |
| `at_group` | `at_group` | Identical |
| `num_y_bins` | `num_Ybins` | Naming convention difference |
| `max_defiers_share` | `max_defiers_share` | Identical |
| `allow_min_defiers` | (implicit +1e-6 relaxation in R) | Python requires explicit opt-in |
| `reg_formula` | `reg_formula` | Same formula syntax |
