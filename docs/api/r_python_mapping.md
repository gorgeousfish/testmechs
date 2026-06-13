# R-to-Python Mapping

This page maps the historical R `TestMechs` public surface to the Python
`testmechs` review-bundle API. It is a release-scoped guide for readers who know
the R package or the Kwon and Roth (2026) replication materials. It is not a
claim that the current article reproduces every method-paper table, every Monte
Carlo operating-characteristic result, or every future extension path.

The mapping follows the article's evidence boundary: paper mathematics defines
the estimands and assumptions, supported R behavior gives reference context
where available, and the Python package supplies the reportable object used by
the accompanying article.

## Public-Surface Mapping

| Historical R surface | Python public surface | Current review-bundle boundary |
| --- | --- | --- |
| `test_sharp_null()` | `testmechs.test_sharp_null()` | Release-scoped support for CS, ARP, FSST, and Kitagawa sharp-null paths used by the package examples, with result objects exposing decisions, method labels, support information, and diagnostics. Full paper-budget Monte Carlo acceptance and adjusted nonbinary/vector/IV sharp-null paths remain outside the current article claim. |
| `test_sharp_null_cr()` | `testmechs.test_sharp_null_cr()` | Scalar CR confidence-set inversion with explicit mediator ordering and SciPy HiGHS LP diagnostics. The historical R Gurobi dependency is replaced by a freely reproducible SciPy backend. |
| `lb_frac_affected()` | `testmechs.lb_frac_affected()` | Lower-bound calculations for the displayed binary, multivalued, vector-mediator, adjusted, and minimum-compatible-defier examples, returning `LowerBoundResult` objects with compact rows and strict-JSON payloads. |
| `breakdown_defier_share()` | `testmechs.breakdown_defier_share()` | Defier-share breakdown calculation that returns the relaxation threshold and bracket diagnostics in the lower-bound result contract. |
| `bounds_ade_ats()` | `testmechs.bounds_ade_ats()` | ADE-bound intervals with endpoint fields, target group, trimming quantities, and diagnostics. Python requires explicit `allow_min_defiers=True` before using the exact minimum-compatible defier cap. |
| `partial_density_plot()` | `testmechs.partial_density_data()` and `testmechs.partial_density_plot()` | Python separates plot-ready records from rendering. The data result records row-level partial-density or PMF values, positive-part diagnostics, and metadata before optional Matplotlib rendering. |
| `remove_missing_from_df()` | `testmechs.remove_missing_from_df()` | Complete-case preprocessing with stricter failure semantics for missing columns, overlapping roles, and empty complete-case samples. |
| `ci_TV()` | `testmechs.ci_TV()` | Source-anchored FSST grid and bisection inversion for scalar ordered mediators, explicit mediator ordering, discrete outcomes, and `diag` or `identity` weighting. R `weight.matrix="avar"` is source-scoped out because the R reference does not provide a stable oracle for that configuration. |

The R `%>%` import is not a statistical surface. Python uses ordinary function
calls, pandas data frames, and result-object methods rather than an infix pipe.

## Stronger Python Reporting Contracts

Several Python surfaces deliberately expose more reportable state than the
historical R call shape:

- Result objects provide `to_frame()` for compact article rows and `to_dict()`
  for strict-JSON payloads.
- Request and support descriptors make declared data roles, method choices, and
  support scope inspectable before or after computation.
- Partial-density plotting is split into returned records and optional figure
  rendering, so the displayed figure can be checked against metadata.
- Installed wheel and source-archive smoke tests use packaged resources to
  rerun the displayed article calculations outside the source checkout.

These are software-reporting contracts. They do not validate identifying
assumptions, prove package-wide behavior, or establish performance claims.

## Current Non-Claims

Do not read this mapping as evidence for unrestricted R/Python equivalence. The
current safe claim is narrower: Python implements the release-scoped,
paper-first API needed for the displayed sharp-null, lower-bound, ADE-bound,
partial-density, preprocessing, TV-inversion, and manuscript-reproduction paths.

The current article does not claim full method-paper reproduction, method-paper
Supplementary Table S1 reproduction, broad Monte Carlo size or power behavior,
performance gains, public package-index availability, or new substantive
conclusions about the source empirical studies.
