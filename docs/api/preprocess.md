# Data Preprocessing

This module provides data-preparation utilities for cleaning analysis data
frames, discretizing continuous outcomes, and normalizing binary support levels.

## `remove_missing_from_df()`

```python
testmechs.remove_missing_from_df(
    *,
    df: pd.DataFrame,
    d: str,
    m: str | Sequence[str],
    y: str,
    w: str | None = None,
    reg_formula: str | None = None,
) -> pd.DataFrame
```

### Description

Drops rows with missing values in analysis-relevant columns. Identifies the
minimal set of required columns from the treatment, mediator, outcome, optional
weight, and optional regression-formula specification, validates their presence
in `df`, then returns a complete-case copy.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `df` | `pd.DataFrame` | — | Input analysis data frame. |
| `d` | `str` | — | Binary treatment column name. |
| `m` | `str \| Sequence[str]` | — | Mediator column name(s). |
| `y` | `str` | — | Outcome column name. |
| `w` | `str \| None` | `None` | Optional weight column. |
| `reg_formula` | `str \| None` | `None` | Formula whose RHS variables are also included in the complete-case set. |

### Returns

`pd.DataFrame` — A copy with rows dropped where any analysis column is missing.

### Raises

- `ValueError` if `d`, `y`, or `w` do not name exactly one non-empty DataFrame column.
- `ValueError` if `d` and `y` are the same column.
- `ValueError` if mediator columns reuse treatment or outcome columns.
- `ValueError` if formula RHS variables reuse the target outcome or mediator columns.

### Example

```python
import pandas as pd
import testmechs

df = pd.DataFrame({
    "treat": [0, 1, None, 1, 0],
    "med": [1, 0, 1, 1, 0],
    "out": [3, 4, 5, None, 2],
})

clean = testmechs.remove_missing_from_df(
    df=df, d="treat", m="med", y="out"
)
print(clean)  # Rows with None removed
```

---

## `discretize_y()`

```python
testmechs.discretize_y(yvec: pd.Series, num_bins: int) -> pd.Series
```

### Description

Discretizes an outcome vector into quantile-based bins. If `yvec` already has
`num_bins` or fewer unique values, it is returned as a categorical without
modification. Otherwise, quantile cutpoints are computed using R-compatible
type-7 interpolation.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `yvec` | `pd.Series` | — | Outcome values. Must be non-empty with no missing values. |
| `num_bins` | `int` | — | Target number of bins (positive integer). |

### Returns

`pd.Series` — A categorical series with at most `num_bins` levels.

### Raises

- `ValueError` if `num_bins` is not a positive integer.
- `ValueError` if `yvec` is empty, contains missing values, or has non-finite numeric values.
- `ValueError` if quantile discretization is not feasible for non-numeric outcomes.

### Notes

- Uses R-compatible type-7 quantile interpolation.
- Non-numeric outcomes are allowed only when the observed support already has
  `num_bins` or fewer levels.
- When duplicate cutpoints arise from point masses, adjacent bins are merged.
- Unobserved categorical levels are dropped.

### Example

```python
import pandas as pd
import testmechs

y = pd.Series([1.2, 3.4, 5.6, 7.8, 2.3, 4.5, 6.7, 8.9])
y_binned = testmechs.discretize_y(y, num_bins=4)
print(y_binned.cat.categories)
```

---

## `normalize_binary_support()`

```python
testmechs.normalize_binary_support(
    series: pd.Series, *, column: str
) -> BinarySupportNormalization
```

### Description

Detects binary support levels in a series and builds a normalization mapping to
{0, 1}. Inspects for exactly two unique non-missing values, orders them, and
returns a frozen normalization object.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `series` | `pd.Series` | — | Series with exactly two distinct non-missing values. |
| `column` | `str` | — | Descriptive name for error messages. |

### Returns

`BinarySupportNormalization` — A frozen dataclass with:

- `original_levels` — The two detected levels
- `transform(s)` method — Maps a conforming series to {0, 1}

### Raises

- `ValueError` if the series does not contain exactly two finite support levels.

### Example

```python
import pandas as pd
from testmechs.preprocess import normalize_binary_support

norm = normalize_binary_support(
    pd.Series(["control", "treated", "control"]), column="D"
)
print(norm.original_levels)  # ('control', 'treated')
transformed = norm.transform(pd.Series(["treated", "control"]))
print(transformed.tolist())  # [1, 0]
```
