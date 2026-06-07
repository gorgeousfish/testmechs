# testmechs

`testmechs` provides Python tools for selected finite-support Testing
Mechanisms calculations. It is designed for empirical researchers who work with
`pandas` data frames and want sharp-null tests, lower-bound calculations,
average-direct-effect bounds, partial-density displays, diagnostics, and
strict-JSON outputs in one Python interface.

The package implements a Python reporting layer for selected calculations from
Kwon and Roth's Testing Mechanisms framework. It does not replace the method
paper as the source of the estimands or identification assumptions.

The main user-facing functions are:

- `test_sharp_null()` for CS, ARP, FSST, and Kitagawa-style sharp-null tests.
- `test_sharp_null_cr()` for the scalar default-order CR confidence-set path.
- `ci_TV()` for source-anchored FSST grid inversion of the total-variation
  target in scalar ordered-mediator, discrete-outcome designs.
- `lb_frac_affected()`, `breakdown_defier_share()`, and `bounds_ade_ats()` for
  lower-bound, defier-share, and ADE-bound calculations.
- `partial_density_data()` and `partial_density_plot()` for inspecting how
  mediator-outcome mass changes across treatment arms.
- Article reproduction helpers for rebuilding the displayed empirical examples,
  figure data, resource listings, and JSON reports used by the accompanying
  manuscript.

Beyond the main estimators, the package exposes typed request objects, result
objects, diagnostic tables, support tables, JSON readers and writers, and
article reproduction helpers.

## Installation

`testmechs` is not yet available from the public Python Package Index. The
manuscript replication materials therefore treat source-checkout installs and
review-bundle artifact installs as the supported installation routes.

From the current source checkout, install the package directory:

```bash
cd packages/python/testmechs-py
python -m pip install -e .
python -m pip install -e ".[plot]"
```

From an unpacked reviewer submission bundle, use the bundle-local package path:

```bash
python -m pip install -e "package/source[plot]"
```

For a built source archive or wheel supplied with a review bundle, install the
artifact directly:

```bash
python -m pip install "dist/testmechs-0.1.0.tar.gz[plot]"
# or
python -m pip install "dist/testmechs-0.1.0-py3-none-any.whl[plot]"
```

The core package depends on NumPy, pandas, SciPy, and OSQP. The optional `plot`
extra installs Matplotlib. Non-plotting APIs such as `partial_density_data()`,
`test_sharp_null()`, `lb_frac_affected()`, `bounds_ade_ats()`, `ci_TV()`, and the
reproduction report helpers do not require Matplotlib.

## Quick Start

```python
from pathlib import Path

import pandas as pd
import testmechs

df = pd.DataFrame({
    "treat": [0, 0, 0, 0, 1, 1, 1, 1],
    "mediator": [0, 0, 1, 1, 1, 1, 1, 1],
    "outcome": [0, 1, 0, 1, 0, 1, 1, 1],
})

sharp_null = testmechs.test_sharp_null(
    df=df,
    d="treat",
    m="mediator",
    y="outcome",
    method="CS",
)
bound = testmechs.lb_frac_affected(
    df=df,
    d="treat",
    m="mediator",
    y="outcome",
)

print(sharp_null.to_frame()[["method", "reject", "p_value"]])
print(bound.to_frame()[["result", "lower_bound", "lower_bound_status"]])

toy_path = Path("toy-mechanism-data.csv")
df.to_csv(toy_path, index=False)
dataset = testmechs.SharedCSVInput(
    data_path=toy_path,
    treatment="treat",
    mediators=("mediator",),
    outcome="outcome",
)
request = testmechs.SharpNullRequest(dataset=dataset, method="CS")
print(request.comparison_view())
```

The estimator calls return typed result objects with estimates, diagnostics,
display rows, and strict-JSON payloads. The request view records the same data
source, column roles, and method when a script needs a reproducible comparison
view before the statistic is computed.

## Main calls and returned objects

| Public call | Returned object | What stays attached |
| --- | --- | --- |
| `test_sharp_null()` | `SharpNullResult` | Method label, test decision, p-value, support diagnostics, `to_frame()` row, and strict-JSON `to_dict()` payload |
| `test_sharp_null_cr()` | `SharpNullResult` | CR confidence-set interval fields, mediator ordering, SciPy LP backend diagnostics, support checks, and strict-JSON payload |
| `ci_TV()` | `TVConfidenceIntervalResult` | Grid or bisection settings, tested TV nulls, p-values, interval endpoints, weighting choice, source-boundary diagnostics, and strict-JSON payload |
| `lb_frac_affected()` | `LowerBoundResult` | Estimand, always-taker group where applicable, active restriction, support and solver diagnostics, display row, and JSON payload |
| `breakdown_defier_share()` | `LowerBoundResult` | Defier-share sensitivity target, active restriction, support diagnostics, display row, and JSON payload |
| `bounds_ade_ats()` | `ADEBoundsResult` | Target group, endpoint status, trimming diagnostics, display row, and explicit non-finite status fields |
| `partial_density_data()` | `PartialDensityDataResult` | Grid or support records, target role, positive-part calculation, long-form rows, and plotting metadata when rendered |
| `partial_density_plot()` | Matplotlib `Figure` | Rendered partial-density display when the optional plotting extra is installed |

## Supported workflows

The Python package is organized around the analysis steps that appear in the
article and in the original method examples.

- **Run sharp-null tests.** `test_sharp_null()` supports CS, ARP, FSST, and
  Kitagawa-style procedures from pandas data frames by naming the treatment,
  mediator, outcome, method, and optional cluster column. The public runner
  validates column roles, support, tuning parameters, bootstrap settings, random
  seeds, and cluster inputs before it dispatches to a method runner.
- **Run the CR confidence-set path.** `test_sharp_null_cr()` exposes the scalar
  CR confidence-set path with explicit mediator ordering using SciPy linear
  programming instead of the historical R/Gurobi backend.
- **Invert TV nulls.** `ci_TV()` inverts the FSST moment-inequality test over a
  TV grid for one always-taker mediator group. The public contract covers
  scalar ordered mediators, explicit mediator orderings, discrete outcomes,
  FSSTdd/FSSTndd tuning, diag or identity weighting, and explicit grid or
  bisection output with p-values.
- **Compute lower bounds.** `lb_frac_affected()`,
  `breakdown_defier_share()`, and `bounds_ade_ats()` implement the lower-bound,
  defier-share, and ADE-bound calculations used in the article examples. Result
  objects keep the estimand, support, no-bite state, finite-status markers, and
  diagnostic payload together.
- **Inspect partial densities.** `partial_density_data()` returns the row-level
  partial-density or partial-PMF records behind a display. `partial_density_plot()`
  uses those records to produce Matplotlib figures when the optional plotting
  extra is installed.

Sharp-null and bounds results expose `to_dict()`, `to_frame()`, terminal
strings, and notebook HTML summaries. Their JSON payloads replace non-standard
`NaN` and infinite values with explicit finite-status fields, so reports can be
written with strict JSON.

The public checks are deliberately explicit. Functions fail before computation
when required columns are missing, analysis samples are empty, treatment or
mediator supports cannot be normalized, requested tuning values are outside the
supported range, or JSON/export payloads would otherwise contain non-standard
numeric values. Support and diagnostic frames expose the details needed to audit
reported values without relying on private state.

## Request and support objects

`SharedCSVInput`, `SharpNullRequest`, `LowerBoundRequest`,
`BreakdownDefierShareRequest`, `ADEBoundsRequest`, and `PartialDensityRequest`
record the input data source and estimator options before a statistic is
computed. They are descriptors, not estimators. Their public
`comparison_view()` method returns a strict-JSON view for comparing what was
requested with the returned result object.

The request payloads keep the main analysis choices explicit: column roles,
method names, tuning choices, target groups, defier-cap options, and any
supported `reg_formula` adjustment. Support views answer a complementary question:
what can this result report, and which diagnostic fields should be inspected?
The main reader-facing helpers include `bounds_support_frame()`,
`regression_adjustment_support_frame()`, `partial_density_support_frame()`,
`cell_count_diagnostics_support_frame()`, and
`sharp_null_diagnostic_schema_frame()`.

Reproduction display helpers format generated reports without turning them into
new estimators. `paper_monte_carlo_reproduction_display_frame()` returns a
compact reader-oriented Monte Carlo table with labels such as
`clusters=unclustered`, `outcome_bins=as observed`, and publication-friendly
nonfinite labels (`NA`, `+Inf`, `-Inf`). This display helper is reserved for
accepted Monte Carlo evidence; it does not support Monte Carlo
operating-characteristic claims in the current article.

These helpers are documentation surfaces rather than new estimators. They are
useful when a reported value needs to be checked against support normalization,
solver status, finite-status labels, or cell-count diagnostics. Release-scoped
adjustment support is documented separately in `docs/api/`.

## Supported designs and boundaries

The package is intentionally release-scoped. It implements the selected
finite-support Testing Mechanisms calculations used by the article and keeps
invalid states from flowing silently into reported output.

- Unadjusted sharp-null tests support the CS, ARP, FSST, and Kitagawa-style
  procedures exposed by `test_sharp_null()`, with release-scoped diagnostics for
  support normalization, approximation choices, and cell counts.
- `test_sharp_null_cr()` exposes the scalar CR confidence-set path with explicit
  mediator ordering and SciPy linear-programming diagnostics.
- `ci_TV()` provides source-anchored FSST grid or bisection inversion for scalar
  ordered mediators, explicit mediator orderings, discrete outcomes, and `diag`
  or `identity` weighting.
- `lb_frac_affected()`, `breakdown_defier_share()`, and `bounds_ade_ats()`
  cover the lower-bound, defier-share, and ADE-bound workflows used in the
  article examples, including scalar and vector mediator support where the
  documented restrictions apply.
- `partial_density_data()` returns row-level records behind partial-density or
  partial-PMF displays, while `partial_density_plot()` renders those records
  when the optional plotting extra is installed.
- Adjusted probability helpers support the documented `reg_formula` surface for
  selected lower-bound and ADE-bound calculations. Adjusted nonbinary/vector/IV
  sharp-null inference remains outside the current public contract.

Inputs fail early when required columns are missing, samples are empty, support
cannot be normalized, non-finite numeric values would enter finite-support
calculations, or tuning values fall outside the supported range. For boundary
details, see `docs/api/` and the manuscript evidence maps.

## Article examples and reproduction

The accompanying article uses the package as a reporting workflow rather than
as a hidden analysis script. Its displayed outputs include:

- four lower-bound comparisons regenerated from packaged empirical inputs and
  rounded method-paper targets, using a `0.005` tolerance on the proportion
  scale;
- the Baranov relationship-quality diagnostic table regenerated from the same
  selected lower-bound result object;
- a deterministic sharp-null example that shows both `to_frame()` and
  `to_dict()` views of the same fitted object;
- a deterministic ADE-bound example that exposes endpoint, target-group,
  trimming, and no-bite diagnostics; and
- a Kerwin partial-density display regenerated from plot-ready package records
  and figure metadata.

From a checkout, run:

```bash
python manuscript/replication/run_replication.py --overwrite
```

The replication entry point regenerates the empirical target table, request
view, sharp-null, lower-bound, ADE-bound, and partial-density example
fragments, the Baranov relationship-quality diagnostic table, the partial-density
figure, and the resource manifest. When a checkout is available, reproduction
helpers prefer the local fixture and statistics files. In an installed wheel or
source archive, they fall back to packaged resources, which lets a reviewer run
the packaged article reproduction helpers without the source tree.

The package exposes `testmechs.__version__` so generated reports can record the
installed version. The distribution check builds a wheel and source archive,
confirms that they include the AGPL license text and packaged reproduction
resources, and installs both artifacts in temporary environments for the
selected article reproduction path. The package metadata declares
`AGPL-3.0-or-later`, and the built artifacts include the AGPL license text from
`LICENSE.md`, paper reproduction fixture CSVs, empirical statistic resources,
and table resources.

`paper_reproduction_resource_manifest()`,
`paper_reproduction_resource_manifest_packet()`,
`write_paper_reproduction_resource_manifest_json()`,
`load_paper_reproduction_resource_manifest_json()`, and
`load_paper_reproduction_resource_manifest_packet_json()` expose and persist a
hash-stamped listing of packaged fixtures, statistics, and table resources for
review scripts. The manifest records the package version that produced the manifest,
an overall manifest SHA-256, and per-resource SHA-256 hashes. The
resource-manifest writer refuses to replace existing files by default, requires `overwrite` to be a real boolean,
and only replaces an existing manifest when `overwrite=True`. The article
reproduction route uses these helpers only to rebuild displayed article objects.

From an unpacked reviewer bundle, install the bundled package source and rerun
the bundle-local entry point:

```bash
python -m pip install -e "package/source[plot]"
python replication/run_replication.py --overwrite
```

The distribution check supports package-resource and article-output
reproduction claims. It is not a public package-index installation test and
does not establish Monte Carlo operating characteristics, performance, or full
method-paper reproduction.

## Documentation and release boundary

Detailed API contracts and extended design boundaries are documented in
`docs/api/`. The README keeps only the package-facing release boundary: public
calls fail clearly when required analysis columns are missing, support cannot be
normalized, non-finite numeric inputs would enter finite-support calculations,
or requested tuning values fall outside the supported range. Result objects
preserve compact display rows and strict-JSON payloads so reported values remain
attached to support, restriction, sample, and diagnostic fields.
Support helpers expose support normalization, solver status, finite-status
labels, or cell-count diagnostics without requiring private state.

This package implements the Python API described above, not the full historical
R package API and not the full set of method-paper simulations. The current
software article does not make Monte Carlo operating-characteristic,
performance, public package-index availability, or new empirical-substantive
claims. The repository-level functional equivalence matrix records the current
R-to-Python boundary before any package-level parity claim is made.

## License and citation

`testmechs` is distributed under `AGPL-3.0-or-later`. Please cite the method
paper when using the Testing Mechanisms estimands and cite the accompanying
software article when using this Python implementation.

## Authors

- Soonwoo Kwon (Brown University)
- Jonathan Roth (Brown University)

## Implementation

- Xuanyu Cai, City University of Macau (xuanyuCAI@outlook.com)
- Wenli Xu, City University of Macau (wlxu@cityu.edu.mo)
