# Partial Density Data and Plotting

This module provides functions for computing and visualizing partial-density
or partial-PMF records that show how mediator-outcome mass changes across
treatment arms.

The partial density illustrates the testable implications of the sharp null
visually: under the null $P(Y \in A, M=m_k \mid D=0) \geq P(Y \in A, M=m_k \mid D=1)$
for never-takers, so violations appear as regions where the treated distribution
exceeds the control distribution.

## `partial_density_data()`

```python
testmechs.partial_density_data(
    *,
    df: pd.DataFrame | None = None,
    data_path: str | Path | None = None,
    d: str,
    m: str,
    y: str,
    num_y_bins: int | None = None,
    plot_nts: bool = False,
    continuous_y: bool = False,
    num_grid_points: int = 10000,
    reg_formula: str | None = None,
) -> PartialDensityDataResult
```

### Description

Returns plot-ready partial-density or partial-PMF records. The mediator must be
a scalar binary column. With `continuous_y=False`, returns finite-support
partial-PMF records (optionally after outcome binning). With
`continuous_y=True`, evaluates a Gaussian kernel-density grid.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `df` | `pd.DataFrame \| None` | `None` | Analysis data frame. Exactly one of `df` or `data_path`. |
| `data_path` | `str \| Path \| None` | `None` | Path to CSV file. |
| `d` | `str` | — | Binary treatment column. |
| `m` | `str` | — | Scalar binary mediator column. |
| `y` | `str` | — | Outcome column (discrete or continuous). |
| `num_y_bins` | `int \| None` | `None` | Discretize Y into quantile bins (discrete mode). |
| `plot_nts` | `bool` | `False` | Plot never-taker orientation (flips D=0,M=0 vs D=1,M=0). |
| `continuous_y` | `bool` | `False` | Use Gaussian kernel-density estimation. |
| `num_grid_points` | `int` | `10000` | Grid points for continuous density estimation. |
| `reg_formula` | `str \| None` | `None` | Regression formula for adjusted partial density. |

### Returns

`PartialDensityDataResult` with:

- Row-level partial-density/PMF records
- Positive-part diagnostics (`positive_part_partial_pmf_diff`)
- Support metadata and outcome column diagnostics
- `to_dict()` → strict-JSON-safe payload with nonfinite markers
- `partial_density_row_records` property → long-form payload

### Example

```python
import testmechs
from importlib.resources import files
import pandas as pd

df = pd.read_csv(files("testmechs.resources.fixtures") / "baranov_mother_data.csv")

# Discrete partial-PMF records for the grandmother mechanism
data_result = testmechs.partial_density_data(
    df=df, d="treat", m="grandmother", y="motherfinancial", num_y_bins=5
)

# Access as DataFrame
frame = data_result.to_frame()
print(frame.head())

# Continuous outcome (Bursztyn example uses binary Y, so use Baranov)
data_cont = testmechs.partial_density_data(
    df=df, d="treat", m="grandmother", y="motherfinancial",
    continuous_y=True, num_grid_points=5000
)
```

---

## `partial_density_plot()`

```python
testmechs.partial_density_plot(
    *,
    df: pd.DataFrame | None = None,
    data_path: str | Path | None = None,
    d: str,
    m: str,
    y: str,
    num_grid_points: int = 10000,
    plot_nts: bool = False,
    density_1_label: str = "f(Y,M=1|D=1)",
    density_0_label: str = "f(Y,M=1|D=0)",
    num_y_bins: int | None = None,
    reg_formula: str | None = None,
    continuous_y: bool = False,
    caption: str | None = None,
) -> matplotlib.figure.Figure
```

### Description

Renders partial-density records as a Matplotlib figure. Computes data via
`partial_density_data()`, then renders discrete bar charts or continuous line
plots with publication-oriented styling.

**Requires the `[plot]` extra:** install the supplied source checkout or review
artifact with the `plot` extra, for example `python -m pip install -e ".[plot]"`.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `df` | `pd.DataFrame \| None` | `None` | Analysis data frame. |
| `data_path` | `str \| Path \| None` | `None` | Path to CSV file. |
| `d` | `str` | — | Binary treatment column. |
| `m` | `str` | — | Scalar binary mediator column. |
| `y` | `str` | — | Outcome column. |
| `num_grid_points` | `int` | `10000` | Grid points for continuous density. |
| `plot_nts` | `bool` | `False` | Never-taker orientation. |
| `density_1_label` | `str` | `"f(Y,M=1\|D=1)"` | Legend label for treated partial density. |
| `density_0_label` | `str` | `"f(Y,M=1\|D=0)"` | Legend label for control partial density. |
| `num_y_bins` | `int \| None` | `None` | Outcome discretization for discrete mode. |
| `reg_formula` | `str \| None` | `None` | Regression formula for adjusted density. |
| `continuous_y` | `bool` | `False` | Use continuous kernel-density plots. |
| `caption` | `str \| None` | `None` | Optional publication caption below plot. |

### Returns

`matplotlib.figure.Figure` with attached metadata:

- `fig.testmechs_partial_density_contract` — strict-JSON data contract
- `fig.testmechs_partial_density_render_metadata` — render metadata including:
  - `positive_part_annotation`
  - `positive_part_shading`
  - `legend_label_line_counts`
  - `caption_line_count`

### Plot Styling

The plot uses publication-oriented defaults:

- Colorblind-safe two-series colors
- Titled legend placed below the plotting area
- Long labels wrapped; discrete outcome ticks capped to 4 lines
- Positive-part edge emphasis for discrete bars
- Positive-part shading for continuous plots
- Bar-height labels on compact discrete plots (≤6 outcome levels)
- Optional caption with reserved bottom margin

### Example

```python
import testmechs
from importlib.resources import files
import pandas as pd

df = pd.read_csv(files("testmechs.resources.fixtures") / "baranov_mother_data.csv")

# Discrete partial-PMF plot for the grandmother mechanism
fig = testmechs.partial_density_plot(
    df=df, d="treat", m="grandmother", y="motherfinancial",
    num_y_bins=5,
    density_1_label="P(Y,M=1|D=1)",
    density_0_label="P(Y,M=1|D=0)",
    caption="Partial density: grandmother mechanism"
)
fig.savefig("partial_density_grandmother.pdf")

# Continuous kernel-density plot
fig = testmechs.partial_density_plot(
    df=df, d="treat", m="grandmother", y="motherfinancial",
    continuous_y=True,
    caption="Partial density of financial empowerment for always-takers"
)
fig.savefig("partial_density_continuous.pdf")
```
