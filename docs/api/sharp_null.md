# Sharp Null Hypothesis Testing

This module provides hypothesis tests for the sharp null of full mediation:

$$H_0: Y(1,m) = Y(0,m) \quad \text{for all } m$$

Under this null, treatment $D$ affects outcome $Y$ **only** through mediator $M$.
Rejection, interpreted under the maintained assumptions, provides evidence
against the recorded mediator as a complete explanation of the treatment effect.

The approach exploits connections to the instrument validity literature: under the
sharp null plus independence and monotonicity, $D$ is a valid instrument for the
LATE of $M$ on $Y$. Testable implications of instrument validity (Kitagawa 2015,
Balke and Pearl 1997) then provide tests of the sharp null.

## `test_sharp_null()`

```python
testmechs.test_sharp_null(
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
) -> SharpNullResult
```

### Description

Tests the sharp-null hypothesis of full mediation using moment-inequality or
combined-Z approaches. Several inference procedures are available through
the `method` parameter.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `data_path` | `str \| Path \| None` | `None` | Path to CSV file. Exactly one of `data_path` or `df` must be provided. |
| `df` | `pd.DataFrame \| None` | `None` | Analysis data frame. |
| `d` | `str` | — | Binary treatment column name. |
| `m` | `str \| Sequence[str]` | — | Mediator column name(s). Single string for scalar; sequence for vector mediator. |
| `y` | `str` | — | Discrete outcome column name. |
| `method` | `str` | `"CS"` | Inference method (see table below). |
| `num_y_bins` | `int \| None` | `None` | Discretize Y into this many quantile bins. Ignored if Y already has fewer unique values. |
| `alpha` | `float` | `0.05` | Significance level. |
| `cluster` | `str \| None` | `None` | Cluster variable for cluster-robust inference. |
| `reg_formula` | `str \| None` | `None` | Regression formula for adjusted tests. Currently release-scoped for binary-mediator CS with controls or one-way fixed effects. |
| `bootstrap_replications` | `int` | `500` | Bootstrap replications for FSST and K methods. |
| `random_state` | `int \| None` | `None` | Random seed for bootstrap reproducibility. |
| `kitagawa_xi` | `float` | `0.07` | Tuning parameter for Kitagawa test scaling. |
| `frac_ats_affected` | `float \| None` | `None` | Relaxed null: fraction of always-takers affected. `None` = sharp null (zero affected). |
| `max_defiers_share` | `float` | `0.0` | Upper bound on defier proportion. 0.0 = strict monotonicity. |

### Supported Methods

| Method | Full Name | Requirements | Notes |
| --- | --- | --- | --- |
| `"CS"` | Cox and Shi (2023) | Any mediator support | Recommended default; conditional inference approach |
| `"ARP"` | Andrews, Roth, Pakes (2023) | Any mediator support | Hybrid conditional/least-favorable test |
| `"FSSTdd"` | Fang, Santos, Shaikh, Torgovitsky (2023) data-driven | Scalar ordered mediator | Data-driven moment selection |
| `"FSSTndd"` | Fang, Santos, Shaikh, Torgovitsky (2023) non-data-driven | Scalar ordered mediator | Fixed moment selection |
| `"K"` | Kitagawa (2015) | Binary mediator only | Combined Z-test exploiting LATE testable implications |

### Returns

`SharpNullResult` with attributes:

- `reject: bool` — Whether the null is rejected at level alpha
- `p_value: float` — p-value of the test
- `method: str` — Method label
- `test_stat: float` — Observed test statistic
- `critical_value: float` — Critical value at requested alpha
- `diagnostics: dict` — Solver and support diagnostics
- `to_frame()` → one-row DataFrame
- `to_dict()` → strict-JSON-safe payload

### Example

```python
import pandas as pd
import testmechs
from importlib.resources import files

# Load Bursztyn et al. (2020) data
df = pd.read_csv(files("testmechs.resources.fixtures") / "burstzyn_data.csv")

# Test: does service sign-up account for the displayed treatment effect?
result = testmechs.test_sharp_null(
    df=df, d="condition2", m="signed_up_number", y="applied_out_fl", method="CS"
)
result.p_value
#> 0.01883
result.reject
#> True
# Interpretation: The sharp-null result rejects for this displayed call.

# With cluster-robust inference (Baranov et al. 2020)
df2 = pd.read_csv(files("testmechs.resources.fixtures") / "baranov_mother_data.csv")
result2 = testmechs.test_sharp_null(
    df=df2, d="treat", m="grandmother", y="motherfinancial",
    method="CS", num_y_bins=5, cluster="uc"
)
result2.p_value
#> 0.02284
```

### Notes

- Binary-mediator adjusted `test_sharp_null(..., method="CS", reg_formula=...)` is
  supported for `~ treatment`, controls, and one-way fixed effects only.
- IV / IV+FE and nonbinary/vector adjusted sharp-null inference remain out of scope.
- Unadjusted `max_defiers_share` uses the ordered-nuisance shape constraint.

---

## `test_sharp_null_cr()`

```python
testmechs.test_sharp_null_cr(
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
) -> SharpNullResult
```

### Description

Tests the sharp null via CR confidence-set inversion using linear programming
feasibility. Rejects when the confidence set is empty at level `alpha`.

Uses SciPy's HiGHS LP solver (replacing the R package's Gurobi dependency),
making the test freely reproducible without a commercial license.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `data_path` | `str \| Path \| None` | `None` | Path to CSV file. Exactly one of `data_path` or `df`. |
| `df` | `pd.DataFrame \| None` | `None` | Analysis data frame. |
| `d` | `str` | — | Binary treatment column. |
| `m` | `str` | — | Scalar mediator column. |
| `y` | `str` | — | Discrete outcome column. |
| `ordering` | `Mapping \| None` | `None` | Explicit mediator ordering. Maps each level to a sequence of levels it dominates. |
| `B` | `int` | `500` | Bootstrap replications for confidence-set boundary. |
| `eps_bar` | `float` | `1e-3` | LP feasibility tolerance. |
| `alpha` | `float` | `0.05` | Significance level. |
| `num_Ybins` | `int \| None` | `None` | Outcome discretization. |
| `random_state` | `int \| None` | `None` | Random seed for reproducibility. |

### Returns

`SharpNullResult` with CR confidence-set interval fields, mediator ordering,
SciPy LP backend diagnostics, support checks, and strict-JSON payload.

### Example

```python
result = testmechs.test_sharp_null_cr(
    df=df, d="treat", m="mediator", y="outcome", alpha=0.05
)
print(f"CR reject: {result.reject}")
```

---

(ci-tv)=
## `ci_TV()`

```python
testmechs.ci_TV(
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
) -> TVConfidenceIntervalResult
```

### Description

Constructs a confidence interval for the total-variation causal effect by
inverting the FSST moment-inequality test over a grid. Quantifies how much the
outcome distribution shifts when treatment changes for always-takers at
mediator level `at_group`.

TV_k = sum_y |P(Y(1,k)=y) - P(Y(0,k)=y)| / 2

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `data_path` | `str \| Path \| None` | `None` | Path to CSV file. |
| `df` | `pd.DataFrame \| None` | `None` | Analysis data frame. |
| `d` | `str` | — | Binary treatment column. |
| `m` | `str` | — | Scalar mediator column. |
| `y` | `str` | — | Discrete outcome column. |
| `at_group` | `object` | — | Target always-taker mediator group. |
| `ordering` | `Mapping \| None` | `None` | Explicit mediator ordering. |
| `alpha` | `float` | `0.05` | Significance level. |
| `bootstrap_replications` | `int` | `500` | Bootstrap replications per grid point. |
| `grid_step` | `float` | `0.02` | Step size for the TV grid on [0, 1]. |
| `bisec` | `bool` | `False` | Refine endpoints with bisection. |
| `eps` | `float \| None` | `None` | Bisection tolerance. |
| `max_bisec_iterations` | `int` | `25` | Maximum bisection iterations. |
| `weight_matrix` | `str` | `"diag"` | Weight matrix: `"diag"` or `"identity"`. `"avar"` is not supported. |
| `method` | `str` | `"FSSTdd"` | FSST variant: `"FSSTdd"` or `"FSSTndd"`. |
| `random_state` | `int \| None` | `None` | Random seed. |

### Returns

`TVConfidenceIntervalResult` with attributes:

- `at_group: object` — Target group
- `alpha: float` — Significance level
- `method: str` — Method used
- `accepted_grid: list[float]` — Grid points not rejected
- `lower: float | None` — Lower CI endpoint (None if empty)
- `upper: float | None` — Upper CI endpoint (None if empty)
- `to_frame()` → summary DataFrame
- `to_dict()` → strict-JSON payload

### Example

```python
ci = testmechs.ci_TV(
    df=df, d="treat", m="mediator", y="outcome",
    at_group=1, method="FSSTdd", alpha=0.05
)
print(f"TV CI: [{ci.lower:.3f}, {ci.upper:.3f}]")
```

### Notes

- Only supports scalar ordered mediators and discrete outcomes.
- `weight_matrix="avar"` raises a clear error (source-scoped out due to
  singular-matrix failure in the R reference).
