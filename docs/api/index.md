# testmechs API Reference

`testmechs` implements the Testing Mechanisms framework (Kwon and Roth, 2024)
for testing whether treatment effects operate entirely through a specified
mediator. The package provides sharp-null tests, lower bounds on the fraction
affected, ADE bounds, breakdown-point analysis, and partial-density displays.

## Installation

```bash
pip install testmechs
```

From source:

```bash
pip install -e ".[plot]"
```

**Dependencies**: NumPy, pandas, SciPy, OSQP.
Optional `[plot]` extra adds Matplotlib.

## Module Overview

| Module | Description | Key Functions |
| --- | --- | --- |
| [Sharp Null Tests](sharp_null.md) | Sharp-null hypothesis testing | `test_sharp_null()`, `test_sharp_null_cr()`, `ci_TV()` |
| [Bounds](bounds.md) | Lower-bound and ADE estimates | `lb_frac_affected()`, `bounds_ade_ats()`, `breakdown_defier_share()` |
| [Partial Density](partial_density.md) | Partial-density data and plotting | `partial_density_data()`, `partial_density_plot()` |
| [Preprocessing](preprocess.md) | Data cleaning and discretization | `remove_missing_from_df()`, `discretize_y()` |
| [Regression](regression.md) | Adjusted probability estimation | `compute_adjusted_probabilities()`, `parse_reg_formula()` |
| [Contracts](contracts.md) | Request/result descriptors | `SharedCSVInput`, `SharpNullRequest`, result classes |
| [Monte Carlo](monte_carlo.md) | Optional simulation helpers | `run_binary_cs_monte_carlo()` |
| [R-Python Mapping](r_python_mapping.md) | Cross-language reference | Function/parameter correspondence table |

## Quick Start

```python
import pandas as pd
import testmechs
from importlib.resources import files

# Load bundled Bursztyn et al. (2020) data
df = pd.read_csv(files("testmechs.resources.fixtures") / "burstzyn_data.csv")

# Sharp-null test: does information affect job applications entirely through sign-up?
result = testmechs.test_sharp_null(
    df=df, d="condition2", m="signed_up_number", y="applied_out_fl", method="CS"
)
result.p_value
#> 0.01883
result.reject
#> True

# Lower bound on fraction of never-takers (M=0 under both) affected
bound = testmechs.lb_frac_affected(
    df=df, d="condition2", m="signed_up_number", y="applied_out_fl",
    num_y_bins=2, at_group=0
)
bound.lower_bound
#> 0.10654

# Breakdown-point: minimum defier share to eliminate the bound
bd = testmechs.breakdown_defier_share(
    df=df, d="condition2", m="signed_up_number", y="applied_out_fl", at_group=0
)
bd.lower_bound
#> 0.06647

# Lee-style ADE bounds for always-takers
ade = testmechs.bounds_ade_ats(
    df=df, d="condition2", m="signed_up_number", y="applied_out_fl"
)
ade.lower_bound, ade.upper_bound
#> (-0.05714, 0.24478)

# Request descriptor for reproducible comparison
from pathlib import Path
dataset = testmechs.SharedCSVInput(
    data_path=Path("burstzyn_data.csv"),
    treatment="condition2",
    mediators=("signed_up_number",),
    outcome="applied_out_fl",
)
request = testmechs.SharpNullRequest(dataset=dataset, method="CS")
```

## Main Calls and Returned Objects

| Public call | Returned object | Key attributes |
| --- | --- | --- |
| `test_sharp_null()` | `SharpNullResult` | `reject`, `p_value`, `method`, `test_stat`, `critical_value`, `diagnostics` |
| `test_sharp_null_cr()` | `SharpNullResult` | CR confidence-set interval, SciPy LP backend diagnostics |
| `ci_TV()` | `TVConfidenceIntervalResult` | `lower`, `upper`, `accepted_grid`, `at_group` |
| `lb_frac_affected()` | `LowerBoundResult` | `lower_bound`, `estimand`, `at_group`, `restriction` |
| `bounds_ade_ats()` | `ADEBoundsResult` | `lower_bound`, `upper_bound`, `at_group`, trimming diagnostics |
| `breakdown_defier_share()` | `LowerBoundResult` | Breakdown defier-share cap, bracket precision |
| `partial_density_data()` | `PartialDensityDataResult` | Row-level records, positive-part diagnostics |
| `partial_density_plot()` | `matplotlib.Figure` | Rendered figure with publication styling |

## Result Object Methods

All statistical result objects provide:

- **`to_dict()`** — Strict-JSON-safe dictionary (replaces NaN/Inf with status fields)
- **`to_frame()`** — One-row pandas DataFrame summary
- **`_repr_html_()`** — Notebook-friendly HTML display

## Bundled Datasets

| Dataset | Source | Observations |
| --- | --- | --- |
| `burstzyn_data.csv` | Bursztyn, González, & Yanagizawa-Drott (2020, AER) | 375 |
| `baranov_mother_data.csv` | Baranov et al. (2020, AER) | 903 |
| `kerwin_data.csv` | Kerwin (2018) | 945 |

Access via:

```python
from importlib.resources import files
path = files("testmechs.resources.fixtures") / "burstzyn_data.csv"
```

## Version

```python
import testmechs
print(testmechs.__version__)
#> 0.1.0
```

## License

`testmechs` is distributed under `AGPL-3.0-or-later`.
