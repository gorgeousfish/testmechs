# testmechs

Python implementation of the Testing Mechanisms framework (Kwon and Roth, 2024)
for testing whether treatment effects operate entirely through a specified mediator.

## Overview

`testmechs` implements the finite-support Testing Mechanisms calculations from
Kwon and Roth (2024). Given a binary treatment $D$, a discrete mediator $M$,
and a discrete outcome $Y$, the package tests the sharp null hypothesis:

$$H_0: Y(1, m) = Y(0, m) \quad \text{for all } m$$

If rejected, the treatment must affect the outcome through channels beyond $M$.
The package provides sharp-null tests, lower bounds on the fraction affected,
breakdown-point analysis for monotonicity violations, Lee-style ADE bounds,
and partial-density displays.

## Installation

Requires Python 3.12 or later.

```bash
pip install testmechs
```

From source:

```bash
git clone https://github.com/gorgeousfish/testmechs.git
cd testmechs/packages/python/testmechs-py
pip install -e ".[plot]"
```

**Dependencies**: NumPy, pandas, SciPy, OSQP.
Optional `[plot]` extra adds Matplotlib for `partial_density_plot()`.

## Main Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `test_sharp_null()` | Sharp-null test (CS, ARP, FSST, Kitagawa) | `SharpNullResult` |
| `lb_frac_affected()` | Lower bound on fraction of always-takers affected | `LowerBoundResult` |
| `breakdown_defier_share()` | Minimum defier share to eliminate the bound | `LowerBoundResult` |
| `bounds_ade_ats()` | Lee-style average direct effect bounds | `ADEBoundsResult` |
| `partial_density_data()` | Partial-density/PMF records for plotting | `PartialDensityDataResult` |
| `partial_density_plot()` | Rendered partial-density figure | Matplotlib `Figure` |
| `ci_TV()` | Total-variation confidence interval (FSST inversion) | `TVConfidenceIntervalResult` |

## Example 1: Bursztyn et al. (2020) — Binary Mediator

Bursztyn, González, and Yanagizawa-Drott (2020) study a field experiment in
Saudi Arabia where married men received information about other men's support
for female labor-force participation.

- **D** (treatment): received information (`condition2`)
- **M** (mediator): signed up for a job-matching service for wife (`signed_up_number`)
- **Y** (outcome): wife applied for a job outside the home (`applied_out_fl`)

**Question**: Does the information treatment affect job applications *entirely*
through service sign-up, or are there alternative channels?

```python
import pandas as pd
import testmechs
from importlib.resources import files

df = pd.read_csv(files("testmechs.resources.fixtures") / "burstzyn_data.csv")
```

### Step 1: Test the sharp null of full mediation

```python
result = testmechs.test_sharp_null(
    df=df, d="condition2", m="signed_up_number", y="applied_out_fl", method="CS"
)
result.p_value
#> 0.01883
result.reject
#> True
```

**Interpretation**: The sharp null is rejected at the 5% level (p = 0.019).
The information treatment affects job applications through channels *beyond*
service sign-up. There must exist "never-takers" (men who would not sign up
regardless of treatment) whose wives' job application behavior is changed
by the treatment.

### Step 2: Lower bound on fraction affected

```python
lb = testmechs.lb_frac_affected(
    df=df, d="condition2", m="signed_up_number", y="applied_out_fl",
    num_y_bins=2, at_group=0
)
lb.lower_bound
#> 0.10654
```

**Interpretation**: At least 10.7% of never-takers are affected by the
treatment through alternative mechanisms. The paper reports ≥11% (rounded).

### Step 3: Breakdown defier share (robustness)

```python
bd = testmechs.breakdown_defier_share(
    df=df, d="condition2", m="signed_up_number", y="applied_out_fl", at_group=0
)
bd.lower_bound
#> 0.06647
```

**Interpretation**: Even if up to 6.6% of the population are "defiers"
(treatment *decreases* their mediator), the lower bound remains positive.
The evidence is robust to substantial monotonicity violations.
The paper reports 7% (rounded).

### Step 4: Average direct effect bounds

```python
ade = testmechs.bounds_ade_ats(
    df=df, d="condition2", m="signed_up_number", y="applied_out_fl"
)
ade.lower_bound, ade.upper_bound
#> (-0.05714, 0.24478)
```

**Interpretation**: The Lee-style bounds on the average direct effect for
always-takers are [−0.057, 0.245]. The interval includes zero but its
upper bound indicates a potentially large direct effect.

## Example 2: Baranov et al. (2020) — Clustered Design

Baranov et al. (2020) study a randomized CBT (cognitive behavioral therapy)
intervention for perinatally depressed women in Pakistan. At 7-year follow-up,
the program improved financial empowerment.

- **D** (treatment): assigned to CBT program (`treat`)
- **M** (mediator): grandmother present in household (`grandmother`)
- **Y** (outcome): mother's financial empowerment index (`motherfinancial`)
- **Cluster**: Union Council (`uc`, the randomization unit)

**Question**: Does CBT affect financial empowerment entirely through
grandmother presence, or are there alternative channels?

```python
df = pd.read_csv(files("testmechs.resources.fixtures") / "baranov_mother_data.csv")
```

### Sharp null test with cluster-robust inference

```python
result = testmechs.test_sharp_null(
    df=df, d="treat", m="grandmother", y="motherfinancial",
    method="CS", num_y_bins=5, cluster="uc"
)
result.p_value
#> 0.02284
result.reject
#> True
```

**Interpretation**: The sharp null is rejected (p = 0.023). CBT improves
financial empowerment through channels *beyond* its effect on grandmother
presence. The paper reports p = 0.02.

### Lower bound

```python
lb = testmechs.lb_frac_affected(
    df=df, d="treat", m="grandmother", y="motherfinancial",
    num_y_bins=5, at_group=0
)
lb.lower_bound
#> 0.18589
```

**Interpretation**: At least 18.6% of never-takers (women without a grandmother
regardless of treatment) have their financial empowerment affected by CBT
through alternative channels. The paper reports ≥19%.

### Breakdown defier share

```python
bd = testmechs.breakdown_defier_share(
    df=df, d="treat", m="grandmother", y="motherfinancial",
    num_y_bins=5, at_group=0
)
bd.lower_bound
#> 0.10803
```

**Interpretation**: The evidence survives up to 10.8% defiers — strong robustness.
The paper reports 11%.

## Example 3: Multi-valued and Vector Mediators

The package supports ordered multi-valued mediators and vector mediators
with elementwise monotonicity.

### Relationship quality (1–5 scale)

```python
lb = testmechs.lb_frac_affected(
    df=df, d="treat", m="relationship_husb", y="motherfinancial",
    num_y_bins=5, allow_min_defiers=True
)
lb.lower_bound
#> 0.10022
```

**Interpretation**: Pooling across all always-taker groups, at least 10.0% are
affected by CBT through channels beyond relationship quality.
The paper reports 10%.

### Combined vector mediator (grandmother + relationship)

```python
lb = testmechs.lb_frac_affected(
    df=df, d="treat", m=["grandmother", "relationship_husb"], y="motherfinancial",
    num_y_bins=5, allow_min_defiers=True
)
lb.lower_bound
#> 0.07252
```

**Interpretation**: Even when both mechanisms are considered jointly, at least
7.3% of always-takers are still affected through other channels. The paper
reports 7%. This is the tightest bound — the combined mechanisms explain most,
but not all, of the treatment effect.

## Automated Reproduction Verification

The package includes a built-in reproduction report that compares all Python
results against paper-published statistics:

```python
from testmechs.empirical import paper_empirical_reproduction_report

report = paper_empirical_reproduction_report()
report.summary["passed_target_rows"], report.summary["target_row_count"]
#> (4, 4)
report.summary["max_absolute_difference"]
#> 0.00411
```

All 4 empirical targets pass within the 0.005 tolerance on the proportion
scale. To regenerate all manuscript artifacts:

```bash
python3 manuscript/replication/run_replication.py --overwrite
```

## Bundled Datasets

| Dataset | Source | Observations |
|---------|--------|-------------|
| `burstzyn_data.csv` | Bursztyn, González, & Yanagizawa-Drott (2020, AER) | 375 |
| `baranov_mother_data.csv` | Baranov et al. (2020, AER) | 903 |
| `kerwin_data.csv` | Kerwin (2018) | 945 |

Access via:

```python
from importlib.resources import files
path = files("testmechs.resources.fixtures") / "burstzyn_data.csv"
```

## Citation

```bibtex
@article{kwon2024testing,
  title   = {Testing Mechanisms},
  author  = {Kwon, Soonwoo and Roth, Jonathan},
  journal = {Review of Economic Studies},
  year    = {2024},
  doi     = {10.1093/restud/rdag028}
}
```

## Package Authors

- Xuanyu Cai, City University of Macau (xuanyuCAI@outlook.com)
- Wenli Xu, City University of Macau (wlxu@cityu.edu.mo)

## Methodology Authors

- Soonwoo Kwon (Brown University)
- Jonathan Roth (Brown University)

## License

AGPL-3.0-or-later
