# testmechs Documentation

**testmechs** is a Python package implementing selected finite-support Testing
Mechanisms calculations from Kwon and Roth (2026). It provides sharp-null
hypothesis tests, lower-bound estimators, average direct effect bounds,
breakdown-point analysis, and partial-density displays for causal mediation
analysis.

## Methodology

The package tests the **sharp null hypothesis of full mediation**:

$$H_0: Y(1, m) = Y(0, m) \quad \text{for all } m$$

Under this null, treatment $D$ affects outcome $Y$ **only** through its effect
on mediator $M$. Rejection, interpreted under the maintained assumptions,
provides evidence against the recorded mediator as a complete explanation of
the treatment effect.

The approach connects mediation analysis to the instrument validity literature:
under the sharp null plus independence and monotonicity, the treatment $D$ is a
valid instrument for the LATE of $M$ on $Y$. Testable implications of instrument
validity then provide tests of the sharp null.

## Features

- **Sharp-null tests** — CS (Cox and Shi 2023), ARP (Andrews, Roth, Pakes 2023),
  FSST (Fang, Santos, Shaikh, Torgovitsky 2023), and Kitagawa (2015) procedures
- **Lower bounds** on the fraction of always-takers affected outside the recorded mediator
- **Breakdown-point analysis** — minimum defier-share relaxation that sets the lower bound to zero
- **ADE bounds** — Lee-style partial-identification of the average direct effect
- **Partial-density displays** — visualize how mediator-outcome mass shifts across treatment arms
- **Cluster-robust inference** for designs with clustered randomization
- **Vector mediators** with elementwise monotonicity
- **Regression adjustment** for controls, fixed effects, and IV designs
- **Article-facing exports** — main sharp-null, bound, ADE, confidence-interval, and partial-density result objects provide strict-JSON, table, and notebook views

## Installation

Requires Python 3.12 or later.

```bash
pip install testmechs
```

For visualization support:

```bash
pip install "testmechs[plot]"
```

**Dependencies**: NumPy, pandas, SciPy, OSQP.
Optional `[plot]` extra adds Matplotlib for `partial_density_plot()`.

## Quick Start

```python
import pandas as pd
import testmechs
from importlib.resources import files

# Load bundled empirical dataset (Bursztyn et al. 2020)
df = pd.read_csv(files("testmechs.resources.fixtures") / "burstzyn_data.csv")

# The article target table uses the restricted analysis frame with non-missing
# `index`; that row reports 0.10678 and displays as 10.7%.

# Test the sharp null of full mediation
result = testmechs.test_sharp_null(
    df=df, d="condition2", m="signed_up_number", y="applied_out_fl", method="CS"
)
result.p_value
#> 0.01883
result.reject
#> True

# Lower bound on fraction of never-takers affected
bound = testmechs.lb_frac_affected(
    df=df, d="condition2", m="signed_up_number", y="applied_out_fl",
    num_y_bins=2, at_group=0
)
bound.lower_bound
#> 0.10654
```

## Guides

```{toctree}
:maxdepth: 2
:caption: User Guides

guides/interpretation
```

## API Reference

```{toctree}
:maxdepth: 2
:caption: API Documentation

api/index
api/sharp_null
api/bounds
api/partial_density
api/preprocess
api/regression
api/contracts
api/r_python_mapping
api/monte_carlo
```

## Auto-generated API

```{toctree}
:maxdepth: 2
:caption: Source Reference

autoapi/modules
```
