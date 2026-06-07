# R-Python Function Mapping

Complete reference for translating between the R `TestMechs` package and the
Python `testmechs` package. Covers function names, parameter names, return
value structures, and solver differences.

## Function Correspondence Table

| R Function | Python Function | Status | Notes |
| --- | --- | --- | --- |
| `test_sharp_null()` | `testmechs.test_sharp_null()` | Complete (release-scoped) | All 5 methods supported |
| `test_sharp_null_cr()` | `testmechs.test_sharp_null_cr()` | Complete | HiGHS replaces Gurobi |
| `ci_TV()` | `testmechs.ci_TV()` | Complete | `weight.matrix="avar"` excluded |
| `lb_frac_affected()` | `testmechs.lb_frac_affected()` | Complete | Same interface |
| `bounds_ade_ats()` | `testmechs.bounds_ade_ats()` | Complete | Requires explicit `allow_min_defiers` |
| `breakdown_defier_share()` | `testmechs.breakdown_defier_share()` | Complete | Same binary-search logic |
| `partial_density_plot()` | `testmechs.partial_density_plot()` | Complete (stronger) | Adds data payloads, metadata |
| `remove_missing_from_df()` | `testmechs.remove_missing_from_df()` | Complete (stricter) | Louder failure semantics |
| `%>%` (magrittr pipe) | Not applicable | — | Python uses method chaining |

### Python-Only Public Functions (No R Equivalent)

| Python Function | Purpose |
| --- | --- |
| `testmechs.partial_density_data()` | Separate data from rendering |
| `testmechs.discretize_y()` | Public outcome discretization |
| `testmechs.normalize_binary_support()` | Binary support detection |
| `testmechs.compute_adjusted_probabilities()` | Public adjusted grid |
| `testmechs.compute_adjusted_probability_influences()` | Influence functions |
| `testmechs.compute_adjusted_mediator_masses()` | Adjusted mediator masses |
| `testmechs.parse_reg_formula()` | Public formula parser |
| `testmechs.theta_kk_min_ordered_monotone()` | Public theta computation |
| `testmechs.build_cell_count_diagnostics()` | Cell-count diagnostics |

---

## Parameter Name Mapping

### Common Parameters (All Estimators)

| Python | R | Type (Python) | Notes |
| --- | --- | --- | --- |
| `df` | `data` | `pd.DataFrame` | Python also accepts `data_path` for CSV |
| `data_path` | — | `str \| Path` | Python-only: load CSV directly |
| `d` | `d` | `str` | Identical — treatment column |
| `m` | `m` | `str \| Sequence[str]` | Python accepts sequence for vector mediator |
| `y` | `y` | `str` | Identical — outcome column |
| `cluster` | `cluster` | `str \| None` | Identical |
| `reg_formula` | `reg_formula` | `str \| None` | Same formula syntax |

### Sharp Null Parameters

| Python | R | Default | Notes |
| --- | --- | --- | --- |
| `method` | `method` | `"CS"` | Same values: CS, ARP, FSSTdd, FSSTndd, K |
| `num_y_bins` | `num_Ybins` | `None` | Naming convention: snake_case vs camelCase |
| `alpha` | `alpha` | `0.05` | Identical |
| `bootstrap_replications` | `B` | `500` | Different name, same meaning |
| `random_state` | `seed` | `None` | Different name |
| `kitagawa_xi` | `xi` | `0.07` | More descriptive Python name |
| `frac_ats_affected` | `frac_ats_affected` | `None` | Identical |
| `max_defiers_share` | `max_defiers_share` | `0.0` | Identical |

### CR Test Parameters

| Python | R | Default | Notes |
| --- | --- | --- | --- |
| `ordering` | `ordering` | `None` | Identical |
| `B` | `B` | `500` | Kept as `B` for historical compatibility |
| `eps_bar` | `eps_bar` | `1e-3` | Identical |
| `num_Ybins` | `num_Ybins` | `None` | Kept as `num_Ybins` for compatibility |
| `random_state` | `seed` | `None` | Different name |

### ci_TV Parameters

| Python | R | Default | Notes |
| --- | --- | --- | --- |
| `at_group` | `at_group` | — | Identical |
| `ordering` | `ordering` | `None` | Identical |
| `bootstrap_replications` | `B` | `500` | Different name |
| `grid_step` | `grid_step` | `0.02` | Identical |
| `bisec` | `bisec` | `False` | Identical |
| `eps` | `eps` | `None` | Identical |
| `max_bisec_iterations` | `max_bisec_iterations` | `25` | Identical |
| `weight_matrix` | `weight.matrix` | `"diag"` | Python: underscore; R: dot. `"avar"` excluded in Python |
| `method` | `method` | `"FSSTdd"` | Only FSSTdd/FSSTndd supported |
| `random_state` | `seed` | `None` | Different name |

### Bounds Parameters

| Python | R | Default | Notes |
| --- | --- | --- | --- |
| `at_group` | `at_group` | `None`/`1` | Identical |
| `num_y_bins` | `num_Ybins` | `None` | Naming convention |
| `max_defiers_share` | `max_defiers_share` | `0.0` | Identical |
| `allow_min_defiers` | — | `False` | Python-only; R uses implicit +1e-6 relaxation |
| `return_min_defiers` | — | `False` | Python-only diagnostic option |
| `tol` | `tol` | `1e-4` | Identical (breakdown_defier_share) |
| `max_iterations` | `max_iterations` | `80` | Identical |

### Partial Density Parameters

| Python | R | Default | Notes |
| --- | --- | --- | --- |
| `plot_nts` | `plot_nts` | `False` | Identical |
| `continuous_y` | `continuous_y` | `False` | Identical |
| `num_grid_points` | `num_grid_points` | `10000` | Identical |
| `density_1_label` | `density_1_label` | `"f(Y,M=1\|D=1)"` | Identical |
| `density_0_label` | `density_0_label` | `"f(Y,M=1\|D=0)"` | Identical |
| `num_y_bins` | `num_Ybins` | `None` | Naming convention |
| `caption` | — | `None` | Python-only |

---

## Return Value Differences

### Sharp Null Result

| Aspect | R | Python |
| --- | --- | --- |
| Return type | Named list | Frozen `SharpNullResult` dataclass |
| Access pattern | `result$reject` | `result.reject` |
| Frame export | Manual construction | `result.to_frame()` |
| JSON export | Manual `jsonlite` | `result.to_dict()` (strict-JSON-safe) |
| NaN handling | Native R `NA`/`NaN` | Explicit `test_stat_nonfinite` status fields |
| Notebook display | Print method | `_repr_html_()` card |

### Lower Bound Result

| Aspect | R | Python |
| --- | --- | --- |
| Return type | Named list | Frozen `LowerBoundResult` dataclass |
| Pooled target | `at_group = NULL` | `at_group = None` |
| No-bite state | Numeric `Inf` | `float('inf')` + `lower_bound_status` field |
| Defier cap | Implicit +1e-6 | Requires `allow_min_defiers=True` |

### ADE Bounds Result

| Aspect | R | Python |
| --- | --- | --- |
| Return type | Named list | Frozen `ADEBoundsResult` dataclass |
| No-bite | Numeric NA | `None` + explicit no-bite diagnostics |
| Access | `result$lower` | `result.lower_bound` |

### Partial Density

| Aspect | R | Python |
| --- | --- | --- |
| Plot return | ggplot2 object | Matplotlib `Figure` with metadata |
| Data access | Embedded in plot | Separate `partial_density_data()` |
| Metadata | None | `fig.testmechs_partial_density_contract` |

---

## Solver Differences

| Component | R | Python | Impact |
| --- | --- | --- | --- |
| Sharp null LP | Gurobi (commercial) | SciPy HiGHS (free) | No license needed |
| CR confidence set | Gurobi LP | SciPy `linprog(highs)` | Same results, free solver |
| Fractional LP | Rglpk | SciPy `linprog` | Same logic, different backend |
| OSQP (CS/ARP) | — | OSQP | Same as R's quadratic programs |
| `ci_TV` avar weight | Available but singular | Explicitly rejected | Python fails clearly |

---

## Formula Syntax (Identical in Both)

```
~ d                          # Trivial (intercept + treatment only)
~ d + x1 + x2               # Controls
~ d + x1 | fe               # One-way fixed effects
~ x1 | d ~ z1               # IV
~ x1 | fe | d ~ z1          # IV + FE
factor(var)                  # Explicit categorical dummy
```

---

## Workflow Translation Example

### R

```r
library(TestMechs)

data <- read.csv("data.csv")
result <- test_sharp_null(data, d = "treat", m = "med", y = "out", method = "CS")
cat("Reject:", result$reject, "p:", result$p_value, "\n")

lb <- lb_frac_affected(data, d = "treat", m = "med", y = "out")
cat("Lower bound:", lb$lower_bound, "\n")

partial_density_plot(data, d = "treat", m = "med", y = "out")
```

### Python

```python
import pandas as pd
import testmechs

df = pd.read_csv("data.csv")
result = testmechs.test_sharp_null(df=df, d="treat", m="med", y="out", method="CS")
print(f"Reject: {result.reject}, p: {result.p_value:.4f}")

lb = testmechs.lb_frac_affected(df=df, d="treat", m="med", y="out")
print(f"Lower bound: {lb.lower_bound:.4f}")

fig = testmechs.partial_density_plot(df=df, d="treat", m="med", y="out")
fig.savefig("partial_density.pdf")
```

---

## Key Behavioral Differences

1. **Keyword-only arguments**: All Python estimator functions use keyword-only
   arguments (`*` separator). R uses positional-or-named.

2. **Data input**: Python accepts both `df=` (DataFrame) and `data_path=`
   (CSV path). R accepts only a data frame.

3. **Strict JSON**: Python result `to_dict()` replaces NaN/Inf with explicit
   status fields. R relies on native NA representation.

4. **Defier relaxation**: Python requires explicit `allow_min_defiers=True`;
   R silently applies a +1e-6 cap relaxation.

5. **Error handling**: Python fails loudly before computation when inputs are
   invalid. R may produce warnings or partial results.

6. **Solver freedom**: Python uses free solvers (SciPy HiGHS, OSQP).
   R requires Gurobi for CR tests.

7. **Plotting backend**: Python uses Matplotlib; R uses ggplot2. Python
   separates data (`partial_density_data`) from rendering.
