# testmechs

**Testing Mechanisms: Sharp-Null Tests, Lower Bounds, and Partial Density for Finite-Support Mediation Analysis**

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE.md)
[![Version: 0.1.0](https://img.shields.io/badge/Version-0.1.0-green.svg)](https://github.com/caicxy/testmechs)

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

```bash
pip install testmechs
```

For visualization support:

```bash
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
calculations from the repository root after installing the package in the active
Python environment:

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

Use this bundle-local route for a fresh reviewer bundle. It creates its own
virtual environment and does not require source-checkout paths.

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

If you use this package in academic work, please cite both the methodology paper
and the software:

**Methodology paper (APA)**:

> Kwon, S., & Roth, J. (2026). Testing Mechanisms. *arXiv preprint arXiv:2404.11739*. https://arxiv.org/abs/2404.11739

**Software (APA)**:

> Cai, X., & Xu, W. (2026). *testmechs: Testing Mechanisms in Python* (Version 0.1.0) [Computer software]. GitHub. https://github.com/caicxy/testmechs

### BibTeX

```bibtex
@misc{kwon2026testingmechanisms,
      title={Testing Mechanisms}, 
      author={Soonwoo Kwon and Jonathan Roth},
      year={2026},
      eprint={2404.11739},
      archivePrefix={arXiv},
      primaryClass={econ.EM},
      url={https://arxiv.org/abs/2404.11739}, 
}
```

```bibtex
@software{cai2026testmechs,
  author = {Xuanyu Cai and Wenli Xu},
  title = {testmechs: Testing Mechanisms in Python},
  year = {2026},
  version = {0.1.0},
  url = {https://github.com/caicxy/testmechs},
  note = {Python package implementing the testing mechanisms framework of Kwon and Roth (2026)}
}
```

## References

Kwon, S., & Roth, J. (2026). Testing Mechanisms. *The Review of Economic Studies*, rdag028. https://doi.org/10.1093/restud/rdag028

Bursztyn, L., González, A. L., & Yanagizawa-Drott, D. (2020). Misperceived Social Norms: Women Working Outside the Home in Saudi Arabia. *American Economic Review*, 110(10), 2997–3029.

Baranov, V., Bhalotra, S., Biroli, P., & Maselko, J. (2020). Maternal Depression, Women's Empowerment, and Parental Investment: Evidence from a Randomized Controlled Trial. *American Economic Review*, 110(3), 824–859.

### Package Authors

**Python Implementation**

- **Xuanyu Cai**, City University of Macau
  Email: [xuanyuCAI@outlook.com](mailto:xuanyuCAI@outlook.com)
- **Wenli Xu**, City University of Macau
  Email: [wlxu@cityu.edu.mo](mailto:wlxu@cityu.edu.mo)

**Methodology**

- **Soonwoo Kwon**, Brown University
- **Jonathan Roth**, Brown University

## License

AGPL-3.0-or-later. See [LICENSE.md](LICENSE.md) for details.
