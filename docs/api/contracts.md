# Request and Result Contracts

This module defines typed request descriptors and result dataclasses used
throughout the testmechs package. Request objects describe *what* to compute
before computation begins; result objects encapsulate computed estimates with
diagnostics and export views.

## Request Objects

Request objects are descriptors, not estimators. They record input data
sources and estimator options, and expose a `comparison_view()` method for
reproducible audit trails.

### `SharedCSVInput`

```python
@dataclass(frozen=True)
class SharedCSVInput:
    data_path: Path
    treatment: str
    mediators: tuple[str, ...]
    outcome: str
    cluster: str | None = None
    id_column: str | None = None
```

Common data-source specification shared across estimation requests.

| Attribute | Type | Description |
| --- | --- | --- |
| `data_path` | `Path` | Absolute resolved path to input CSV. |
| `treatment` | `str` | Binary treatment column name. |
| `mediators` | `tuple[str, ...]` | Mediator column name(s). |
| `outcome` | `str` | Outcome column name. |
| `cluster` | `str \| None` | Optional cluster column. |
| `id_column` | `str \| None` | Optional row identifier column. |

**Example:**

```python
from pathlib import Path
import testmechs

dataset = testmechs.SharedCSVInput(
    data_path=Path("data.csv"),
    treatment="treat",
    mediators=("mediator",),
    outcome="outcome",
)
```

---

### `SharpNullRequest`

```python
@dataclass(frozen=True)
class SharpNullRequest:
    dataset: SharedCSVInput
    method: str = "CS"
    num_y_bins: int | None = None
    alpha: float = 0.05
    reg_formula: str | None = None
    frac_ats_affected: float | None = None
```

| Attribute | Type | Description |
| --- | --- | --- |
| `dataset` | `SharedCSVInput` | Data source with column roles. |
| `method` | `str` | Statistical method (`"CS"`, `"ARP"`, `"FSSTdd"`, `"FSSTndd"`, `"K"`). |
| `num_y_bins` | `int \| None` | Outcome discretization. |
| `alpha` | `float` | Significance level. |
| `reg_formula` | `str \| None` | Regression formula for adjustment. |
| `frac_ats_affected` | `float \| None` | Relaxed null fraction. |

**Methods:** `comparison_view()` → strict-JSON comparison payload.

---

### `LowerBoundRequest`

```python
@dataclass(frozen=True)
class LowerBoundRequest:
    dataset: SharedCSVInput
    at_group: object | None = None
    num_y_bins: int | None = None
    reg_formula: str | None = None
    max_defiers_share: float = 0.0
    allow_min_defiers: bool = False
    return_min_defiers: bool = False
```

| Attribute | Type | Description |
| --- | --- | --- |
| `dataset` | `SharedCSVInput` | Data source. |
| `at_group` | `object \| None` | Target always-taker group. |
| `num_y_bins` | `int \| None` | Outcome discretization. |
| `reg_formula` | `str \| None` | Regression formula. |
| `max_defiers_share` | `float` | Maximum defier cap. |
| `allow_min_defiers` | `bool` | Use exact minimum compatible cap. |
| `return_min_defiers` | `bool` | Report minimum cap in diagnostics. |

---

### `BreakdownDefierShareRequest`

```python
@dataclass(frozen=True)
class BreakdownDefierShareRequest:
    dataset: SharedCSVInput
    at_group: object | None = None
    num_y_bins: int | None = None
    reg_formula: str | None = None
    tol: float = 1e-4
    max_iterations: int = 80
```

---

### `ADEBoundsRequest`

```python
@dataclass(frozen=True)
class ADEBoundsRequest:
    dataset: SharedCSVInput
    at_group: object = 1
    reg_formula: str | None = None
    max_defiers_share: float = 0.0
    allow_min_defiers: bool = False
```

---

### `PartialDensityRequest`

```python
@dataclass(frozen=True)
class PartialDensityRequest:
    dataset: SharedCSVInput
    num_y_bins: int | None = None
    plot_nts: bool = False
    continuous_y: bool = False
    num_grid_points: int = 10000
    reg_formula: str | None = None
```

---

## Result Objects

### `SharpNullResult`

```python
@dataclass(frozen=True)
class SharpNullResult:
    method: str
    null_hypothesis: str
    reject: bool
    test_stat: float
    critical_value: float
    p_value: float
    beta_observed: list[float]
    approximation: str
    diagnostics: dict[str, Any]
```

| Attribute | Type | Description |
| --- | --- | --- |
| `method` | `str` | Method identifier (`"CS"`, `"ARP"`, etc.). |
| `null_hypothesis` | `str` | Human-readable null statement. |
| `reject` | `bool` | Whether the null is rejected. |
| `test_stat` | `float` | Observed test statistic. |
| `critical_value` | `float` | Critical value at alpha. |
| `p_value` | `float` | p-value of the test. |
| `beta_observed` | `list[float]` | Observed moment-inequality beta vector. |
| `approximation` | `str` | Approximation method label. |
| `diagnostics` | `dict` | Method-specific solver metadata. |

**Methods:**

- `to_frame()` → one-row `pd.DataFrame` with columns: method, reject, p_value, test_stat, critical_value, etc.
- `to_dict()` → strict-JSON-safe dictionary (NaN/Inf replaced with status fields)
- `_repr_html_()` → notebook HTML summary card

---

### `LowerBoundResult`

```python
@dataclass(frozen=True)
class LowerBoundResult:
    lower_bound: float
    estimand: str
    at_group: object | None
    restriction: str
    diagnostics: dict[str, Any]
```

| Attribute | Type | Description |
| --- | --- | --- |
| `lower_bound` | `float` | Estimated lower bound (may be `inf`). |
| `estimand` | `str` | Human-readable estimand label. |
| `at_group` | `object \| None` | Target group or `None` for pooled. |
| `restriction` | `str` | Monotonicity restriction label. |
| `diagnostics` | `dict` | Solver and support diagnostics. |

**Methods:** `to_frame()`, `to_dict()`, `_repr_html_()`

---

### `ADEBoundsResult`

```python
@dataclass(frozen=True)
class ADEBoundsResult:
    lower_bound: float | None
    upper_bound: float | None
    at_group: object
    restriction: str
    diagnostics: dict[str, Any]
```

| Attribute | Type | Description |
| --- | --- | --- |
| `lower_bound` | `float \| None` | Lower ADE bound (None if no bite). |
| `upper_bound` | `float \| None` | Upper ADE bound (None if no bite). |
| `at_group` | `object` | Target always-taker group. |
| `restriction` | `str` | Monotonicity restriction label. |
| `diagnostics` | `dict` | Theta, trimming, mass diagnostics. |

**Methods:** `to_frame()`, `to_dict()`, `_repr_html_()`

---

### `TVConfidenceIntervalResult`

```python
@dataclass(frozen=True)
class TVConfidenceIntervalResult:
    at_group: object
    alpha: float
    method: str
    accepted_grid: list[float]
    lower: float | None
    upper: float | None
    diagnostics: dict[str, Any]
```

| Attribute | Type | Description |
| --- | --- | --- |
| `at_group` | `object` | Target always-taker group. |
| `alpha` | `float` | Significance level. |
| `method` | `str` | Method used. |
| `accepted_grid` | `list[float]` | Non-rejected grid points. |
| `lower` | `float \| None` | CI lower endpoint. |
| `upper` | `float \| None` | CI upper endpoint. |
| `diagnostics` | `dict` | Grid/bisection metadata. |

**Methods:** `to_frame()`, `to_dict()`, `_repr_html_()`

---

### `PartialDensityDataResult`

```python
@dataclass(frozen=True)
class PartialDensityDataResult:
    # Row-level partial-density records
    # Positive-part diagnostics
    # Support metadata
```

**Key properties:**

- `partial_density_row_records` → long-form record list
- `to_dict()` → strict-JSON payload with nonfinite markers

---

## Support Objects

Support views answer: what can this result report, and which diagnostic fields
should be inspected?

| Function | Returns | Purpose |
| --- | --- | --- |
| `bounds_support_frame()` | `pd.DataFrame` | Bounds-specific support contract |
| `regression_adjustment_support_frame()` | `pd.DataFrame` | Regression support metadata |
| `partial_density_support_frame()` | `pd.DataFrame` | Partial-density support contract |
| `cell_count_diagnostics_support_frame()` | `pd.DataFrame` | Cell-count diagnostic details |
| `sharp_null_diagnostic_schema_frame()` | `pd.DataFrame` | Sharp-null diagnostic surface |
| `bounds_support_contract()` | `dict` | Machine-readable bounds contract |
| `cell_count_diagnostics_support_contract()` | `dict` | Cell-count contract |
| `partial_density_support_contract()` | `dict` | Partial-density contract |
| `regression_adjustment_support_contract()` | `dict` | Regression contract |
| `sharp_null_diagnostics_support_contract()` | `dict` | Sharp-null contract |

---

## Common Method: `comparison_view()`

All request objects expose `comparison_view()` which returns a strict-JSON
payload comparing what was requested:

```python
request = testmechs.SharpNullRequest(dataset=dataset, method="CS")
view = request.comparison_view()
print(view)  # JSON-safe dict with data source, method, parameters
```
