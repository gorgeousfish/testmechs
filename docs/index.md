# testmechs Documentation

**testmechs** is a Python package implementing the Testing Mechanisms framework
from Kwon and Roth (2024). It provides finite-support sharp-null hypothesis
tests, lower-bound estimators, average direct effect bounds, breakdown-point
analysis, and partial-density displays for causal mediation analysis.

## Methodology

The package tests the **sharp null hypothesis of full mediation**:

$$H_0: Y(1, m) = Y(0, m) \quad \text{for all } m$$

Under this null, treatment $D$ affects outcome $Y$ **only** through its effect
on mediator $M$. If rejected, there must exist alternative mechanisms — direct
effects of $D$ on $Y$ not operating through $M$.

The approach connects mediation analysis to the instrument validity literature:
under the sharp null plus independence and monotonicity, the treatment $D$ is a
valid instrument for the LATE of $M$ on $Y$. Testable implications of instrument
validity then provide tests of the sharp null.

## Features

- **Sharp-null tests** — CS (Cox and Shi 2023), ARP (Andrews, Roth, Pakes 2023),
  FSST (Fang, Santos, Shaikh, Torgovitsky 2023), and Kitagawa (2015) procedures
- **Lower bounds** on the fraction of always-takers affected through alternative channels
- **Breakdown-point analysis** — minimum defier share to eliminate evidence
- **ADE bounds** — Lee-style partial-identification of the average direct effect
- **Partial-density displays** — visualize how mediator-outcome mass shifts across treatment arms
- **Cluster-robust inference** for designs with clustered randomization
- **Vector mediators** with elementwise monotonicity
- **Regression adjustment** for controls, fixed effects, and IV designs
- **Strict-JSON exports** — all result objects provide `to_dict()`, `to_frame()`, and notebook HTML views

## Installation

```bash
pip install testmechs
```

From source:

```bash
cd packages/python/testmechs-py
pip install -e ".[plot]"
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
api/monte_carlo
api/r_python_mapping
```

## Auto-generated API

```{toctree}
:maxdepth: 2
:caption: Source Reference

autoapi/modules
```
