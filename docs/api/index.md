# testmechs API Reference

`testmechs` provides Python tools for finite-support Testing Mechanisms
calculations. It implements a Python reporting layer for selected calculations
from Kwon and Roth's Testing Mechanisms framework.

## Installation

`testmechs` is not yet available from the public Python Package Index. Install
the review materials from the supplied source checkout or reviewer artifact:

```bash
# From source checkout
cd packages/python/testmechs-py
python -m pip install -e .
python -m pip install -e ".[plot]"

# From review bundle artifact
python -m pip install "dist/testmechs-0.1.0.tar.gz[plot]"
python -m pip install "dist/testmechs-0.1.0-py3-none-any.whl[plot]"
```

**Dependencies:**

- Core: NumPy, pandas, SciPy, OSQP
- Optional `[plot]`: Matplotlib

## Module Overview

| Module | Description | Key Functions |
| --- | --- | --- |
| [Sharp Null Tests](sharp_null.md) | Sharp-null hypothesis testing | `test_sharp_null()`, `test_sharp_null_cr()`, `ci_TV()` |
| [Bounds](bounds.md) | Lower-bound and ADE estimates | `lb_frac_affected()`, `bounds_ade_ats()`, `breakdown_defier_share()` |
| [Partial Density](partial_density.md) | Partial-density data and plotting | `partial_density_data()`, `partial_density_plot()` |
| [Preprocessing](preprocess.md) | Data cleaning and discretization | `remove_missing_from_df()`, `discretize_y()` |
| [Regression](regression.md) | Adjusted probability estimation | `compute_adjusted_probabilities()`, `parse_reg_formula()` |
| [Contracts](contracts.md) | Request/result descriptors | `SharedCSVInput`, `SharpNullRequest`, result classes |
| [Monte Carlo](monte_carlo.md) | Optional simulation helpers, not current article evidence | `run_binary_cs_monte_carlo()` |
| [R-Python Mapping](r_python_mapping.md) | Cross-language reference | Function/parameter correspondence table |

## Quick Start

```python
import pandas as pd
import testmechs

# Prepare data
df = pd.DataFrame({
    "treat": [0, 0, 0, 0, 1, 1, 1, 1],
    "mediator": [0, 0, 1, 1, 1, 1, 1, 1],
    "outcome": [0, 1, 0, 1, 0, 1, 1, 1],
})

# Sharp-null test
result = testmechs.test_sharp_null(
    df=df, d="treat", m="mediator", y="outcome", method="CS"
)
print(result.reject)        # True/False
print(result.p_value)       # p-value
print(result.to_frame())    # one-row summary DataFrame

# Lower bound on fraction affected
bound = testmechs.lb_frac_affected(
    df=df, d="treat", m="mediator", y="outcome"
)
print(bound.lower_bound)
print(bound.to_frame())

# ADE bounds
ade = testmechs.bounds_ade_ats(
    df=df, d="treat", m="mediator", y="outcome", at_group=1
)
print(ade.lower_bound, ade.upper_bound)

# Request descriptor for reproducible comparison
from pathlib import Path
dataset = testmechs.SharedCSVInput(
    data_path=Path("data.csv"),
    treatment="treat",
    mediators=("mediator",),
    outcome="outcome",
)
request = testmechs.SharpNullRequest(dataset=dataset, method="CS")
print(request.comparison_view())
```

## Main Calls and Returned Objects

| Public call | Returned object | Key attachments |
| --- | --- | --- |
| `test_sharp_null()` | `SharpNullResult` | method, reject, p_value, diagnostics, `to_frame()`, `to_dict()` |
| `test_sharp_null_cr()` | `SharpNullResult` | CR confidence-set interval, SciPy LP backend diagnostics |
| `ci_TV()` | `TVConfidenceIntervalResult` | Grid/bisection settings, p-values, interval endpoints |
| `lb_frac_affected()` | `LowerBoundResult` | lower_bound, estimand, at_group, restriction, diagnostics |
| `bounds_ade_ats()` | `ADEBoundsResult` | lower_bound, upper_bound, at_group, trimming diagnostics |
| `breakdown_defier_share()` | `LowerBoundResult` | Breakdown defier-share cap, bracket precision |
| `partial_density_data()` | `PartialDensityDataResult` | Row-level records, positive-part diagnostics |
| `partial_density_plot()` | `matplotlib.Figure` | Rendered figure with attached metadata |

## Result Object Methods

The main statistical result objects provide:

- **`to_dict()`** — Strict-JSON-safe dictionary payload (replaces NaN/Inf with status fields)
- **`to_frame()`** — One-row pandas DataFrame summary
- **`_repr_html_()`** — Notebook-friendly HTML display

## Version

```python
import testmechs
print(testmechs.__version__)
```

## License

`testmechs` is distributed under `AGPL-3.0-or-later`.
