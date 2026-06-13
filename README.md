# testmechs

Python implementation of selected finite-support Testing Mechanisms calculations
from Kwon and Roth (2026), for testing whether treatment effects operate entirely
through a specified mediator.

## Overview

`testmechs` implements selected finite-support Testing Mechanisms calculations
from Kwon and Roth (2026). Given a binary treatment $D$, a discrete mediator
$M$, and a discrete outcome $Y$, the main sharp-null interface tests the
following hypothesis:

$$H_0: Y(1, m) = Y(0, m) \quad \text{for all } m$$

When rejected under the maintained assumptions, the result is evidence that the
recorded mediator does not account for the full treatment effect.
The package provides sharp-null tests, lower bounds on the fraction affected,
breakdown-point analysis for monotonicity violations, Lee-style ADE bounds,
and partial-density displays.

## Installation

Requires Python 3.12 or later.

The review-bundle version is not yet available from the public Python Package
Index; use the supplied source tree, wheel, or source archive when reproducing
the accompanying article.

From the supplied source tree:

```bash
git clone https://github.com/gorgeousfish/testmechs.git
cd testmechs/packages/python/testmechs-py
pip install -e ".[plot]"
```

After a public package-index release, the runtime package can be installed with:

```bash
pip install testmechs
pip install "testmechs[plot]"
```

**Dependencies**: NumPy, pandas, SciPy, OSQP.
Optional `[plot]` extra adds Matplotlib for `partial_density_plot()`.

## Main calls and returned objects

| Call | Returns | Reported object |
|------|---------|-----------------|
| `test_sharp_null()` | `SharpNullResult` | Sharp-null decision, p value, method, support, and diagnostics |
| `test_sharp_null_cr()` | `SharpNullResult` | Cluster-aware sharp-null decision with the same result contract |
| `ci_TV()` | `TVConfidenceIntervalResult` | Total-variation confidence interval from FSST inversion |
| `lb_frac_affected()` | `LowerBoundResult` | Lower bound, retained sample, support, restriction, and diagnostics |
| `breakdown_defier_share()` | `LowerBoundResult` | Minimum defier-share relaxation for the bound |
| `bounds_ade_ats()` | `ADEBoundsResult` | ADE-bound endpoints, target group, trimming quantities, and diagnostics |
| `partial_density_data()` | `PartialDensityDataResult` | Plot-ready partial-density or PMF records and metadata |
| `partial_density_plot()` | Matplotlib `Figure` | Rendered partial-density figure from returned records |

Request objects describe a calculation before it is run. They are descriptors,
not estimators. A request object's `comparison_view()` method returns a
strict-JSON view for comparing what was requested with the returned result
object. Support views answer a complementary question: what can this result
report, and which diagnostic fields should be inspected. For example,
`partial_density_support_frame()` and `cell_count_diagnostics_support_frame()`
summarize support normalization, solver status, finite-status labels, or
cell-count diagnostics for generated displays.

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

The quick-start calls below use the full bundled fixture. The accompanying
article's target table uses the method-paper restricted analysis frame, which
also requires non-missing `index`; that regenerated target row reports 0.10678,
displays as 10.7%, and is compared with the rounded method-paper target of at
least 11%.

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

**Interpretation**: The sharp-null result rejects at the 5% level (p = 0.019).
For this binary-mediator example, the fitted object reports evidence against
service sign-up as a complete explanation under the maintained assumptions.

### Step 2: Lower bound on fraction affected

```python
lb = testmechs.lb_frac_affected(
    df=df, d="condition2", m="signed_up_number", y="applied_out_fl",
    num_y_bins=2, at_group=0
)
lb.lower_bound
#> 0.10654
```

**Interpretation**: The lower-bound object reports 0.10654 for the never-taker
target group in this full-fixture quick-start. The article's restricted-frame
target row reports 0.10678 and uses the same returned-object interpretation.

### Step 3: Defier-share breakdown point

```python
bd = testmechs.breakdown_defier_share(
    df=df, d="condition2", m="signed_up_number", y="applied_out_fl", at_group=0
)
bd.lower_bound
#> 0.06647
```

**Interpretation**: The breakdown object reports a defier-share cap of 0.06647,
the relaxation at which the corresponding lower-bound calculation reaches zero
within the package tolerance. The article compares this with the rounded
method-paper target of 7%.

### Step 4: Average direct effect bounds

```python
ade = testmechs.bounds_ade_ats(
    df=df, d="condition2", m="signed_up_number", y="applied_out_fl"
)
ade.lower_bound, ade.upper_bound
#> (-0.05714, 0.24478)
```

**Interpretation**: The Lee-style ADE-bound object reports endpoint fields
[-0.057, 0.245] for the always-taker target group, with diagnostics available
through the returned result.

## Example 2: Baranov et al. (2020) — Clustered Design

Baranov et al. (2020) study a randomized CBT (cognitive behavioral therapy)
intervention for perinatally depressed women in Pakistan. The original article
reports financial empowerment among its follow-up outcome families.

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

**Interpretation**: The cluster-aware sharp-null result rejects at p = 0.023.
In the article, this is displayed as a package-output comparison with the
rounded method-paper value p = 0.02.

### Lower bound

```python
lb = testmechs.lb_frac_affected(
    df=df, d="treat", m="grandmother", y="motherfinancial",
    num_y_bins=5, at_group=0
)
lb.lower_bound
#> 0.18589
```

**Interpretation**: The lower-bound object reports 0.18589 for the never-taker
target group under the displayed binning and restriction. The article displays
this as 18.6% and compares it with the rounded method-paper target of at least
19%.

### Breakdown defier share

```python
bd = testmechs.breakdown_defier_share(
    df=df, d="treat", m="grandmother", y="motherfinancial",
    num_y_bins=5, at_group=0
)
bd.lower_bound
#> 0.10803
```

**Interpretation**: The breakdown object reports a defier-share cap of 0.10803
for this displayed calculation. The article compares the rounded value with the
method-paper target of 11%.

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

**Interpretation**: The pooled lower-bound object reports 0.10022 under the
minimum-compatible-defiers option. The article uses this row to show how the
same result object retains the complete-case sample, five-level mediator
support, restriction, and feasibility diagnostics behind the compact bound.

### Combined vector mediator (grandmother + relationship)

```python
lb = testmechs.lb_frac_affected(
    df=df, d="treat", m=["grandmother", "relationship_husb"], y="motherfinancial",
    num_y_bins=5, allow_min_defiers=True
)
lb.lower_bound
#> 0.07252
```

**Interpretation**: The vector-mediator lower-bound object reports 0.07252 when
grandmother presence and relationship quality are entered jointly. The article
displays the rounded 7.3% value as a support-and-diagnostics example for a
two-column mediator and compares it with the rounded method-paper target of 7%.

## Article reproduction

The accompanying article uses the package as a reporting workflow rather than as
a hidden analysis script. Its displayed empirical evidence is four lower-bound
comparisons regenerated from packaged empirical inputs and rounded method-paper
targets:

```python
from testmechs.empirical import paper_empirical_reproduction_report

report = paper_empirical_reproduction_report()
report.summary["passed_target_rows"], report.summary["target_row_count"]
#> (4, 4)
report.summary["max_absolute_difference"]
#> 0.00411
```

All 4 displayed empirical targets pass within the 0.005 tolerance on the
proportion scale. From a full source checkout, regenerate the displayed article
calculations from the repository root:

```bash
python3 manuscript/replication/run_replication.py --overwrite
```

From an unpacked reviewer bundle, install the bundled package source and use the
bundle-local replication entry point:

```bash
python3 -m venv .venv-testmechs-review
. .venv-testmechs-review/bin/activate
python3 -m pip install -e "package/source[plot]"
python3 replication/run_replication.py --overwrite
```

The replication entry point regenerates the empirical target table, request
view, sharp-null, lower-bound, ADE-bound, and partial-density example fragments.
The supplied wheel and source archive include the AGPL license text, paper
reproduction fixture CSVs, empirical statistic resources, and table resources;
the license text is the package's `LICENSE.md`. The packaged article
reproduction helpers without the source tree regenerate the empirical target
table, request view, sharp-null, lower-bound, ADE-bound, and partial-density
example fragments from those resources. In an installed wheel or source archive,
they fall back to packaged resources, letting the article examples be rerun
without the source tree. The supplied archives support package-resource checks
and reproduction of displayed article calculations; they do not establish
public package-index availability, performance, Monte Carlo operating
characteristics, or full method-paper reproduction.

The packaged resource manifest is available through
`paper_reproduction_resource_manifest()` and
`paper_reproduction_resource_manifest_packet()`. The packet records the package version
that produced the manifest, per-resource SHA-256 hashes, and the overall
manifest SHA-256. Use
`write_paper_reproduction_resource_manifest_json()` to write the strict-JSON
manifest, and `load_paper_reproduction_resource_manifest_json()` or
`load_paper_reproduction_resource_manifest_packet_json()` to reload and
validate it. The resource-manifest writer refuses to replace existing files by
default, requires `overwrite` to be a real boolean, and only replaces an
existing manifest when `overwrite=True`.

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

## Version

```python
import testmechs
print(testmechs.__version__)
#> 0.1.0
```

## Citation

```bibtex
@article{kwon2026testing,
  title   = {Testing Mechanisms},
  author  = {Kwon, Soonwoo and Roth, Jonathan},
  journal = {The Review of Economic Studies},
  year    = {2026},
  pages   = {rdag028},
  doi     = {10.1093/restud/rdag028}
}
```

## Contacts and Attribution

Package metadata distinguishes authors from maintainers. The methodology paper
and accompanying software article list Soonwoo Kwon and Jonathan Roth as
authors; the Python review bundle lists Xuanyu Cai and Wenli Xu as maintainer
contacts.

- Authors: Soonwoo Kwon, Brown University (soonwoo_kwon@brown.edu); Jonathan
  Roth, Brown University (jonathan_roth@brown.edu)
- Maintainers: Xuanyu Cai, City University of Macau (xuanyuCAI@outlook.com);
  Wenli Xu, City University of Macau (wlxu@cityu.edu.mo)

## License

The AGPL license text from `LICENSE.md` is included with the package source and
distribution artifacts.

AGPL-3.0-or-later
