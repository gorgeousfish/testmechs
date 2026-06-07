# Monte Carlo Helpers

This module contains optional simulation helpers for studying sharp-null test
behavior under finite-support data-generating processes. These helpers are not part of the current JSS manuscript evidence. The current article does not claim Monte Carlo size, power, performance, method-paper Supplementary Table S1, or full method-paper table reproduction.

Use this page as API documentation for future simulation work, not as evidence that the accompanying article has established operating characteristics.

## User-Facing Simulation Helpers

### `BinaryCSMonteCarloDesign`

Design specification for a binary-mediator CS sharp-null simulation.

### `BinaryEmpiricalMixtureMonteCarloDesign`

Design specification for empirical-mixture simulations with binary mediators.

### `BinaryPartialDensityMonteCarloDesign`

Design specification for partial-density simulations with binary mediators.

### `BinaryEmpiricalMixtureBenchmarkDataSource`

Data-source descriptor for empirical-mixture simulations. It records the source
dataset, complete-case rows, treatment, mediator, outcome columns, and whether
the data-generating process satisfies the null.

### `run_binary_cs_monte_carlo()`

```python
from testmechs.monte_carlo import BinaryCSMonteCarloDesign, run_binary_cs_monte_carlo

design = BinaryCSMonteCarloDesign(...)
result = run_binary_cs_monte_carlo(
    design,
    replications=1000,
    bootstrap_replications=500,
    random_state=20260509,
)
print(result.rejection_rate)
```

Runs a CS sharp-null simulation under a specified binary-mediator design and
returns rejection-rate and diagnostic summaries.

### Other Simulation Runners

The module also exposes runners for empirical-mixture and partial-density
simulation designs:

| Function | Scope |
| --- | --- |
| `run_binary_empirical_mixture_cs_monte_carlo()` | CS simulations using binary empirical-mixture data-generating processes |
| `run_binary_partial_density_cs_monte_carlo()` | CS simulations under binary partial-density data-generating processes |
| `run_nonbinary_empirical_mixture_cs_monte_carlo()` | CS simulations using nonbinary empirical-mixture data-generating processes |

## Result And Diagnostic Objects

| Object | What it records |
| --- | --- |
| `MonteCarloSimulationResult` | Overall rejection rate, per-replication results, and timing |
| `MonteCarloResultRow` | One replication result with reject decision, p-value, and test statistic |
| `MonteCarloDrawResult` | Generated data draw and data-generating-process parameters |
| `MonteCarloBenchmarkCell` | One planned simulation cell with rejection rate, replications, and diagnostics |
| `MonteCarloBenchmarkDiagnostic` | Timing, failure, and warning metadata for a cell |
| `MonteCarloBenchmarkPlan` | Precomputed simulation-cell plan |
| `MonteCarloBenchmarkSuiteRunResult` | Combined result for a planned suite |

## Developer Evidence Helpers

Several functions support long-running manuscript-evidence builds, including
benchmark planning, chunked execution, archive continuation, and JSON report
writing. They are intentionally treated as future/developer evidence routes, not
as part of the current reader-facing article workflow. They should not be cited as current article evidence unless a complete fixed-seed suite has been run, accepted by the Monte Carlo preflight, and summarized in the manuscript replication outputs.

Representative helper families include:

| Helper family | Use |
| --- | --- |
| `run_empirical_mixture_*benchmark*()` | Plan or run empirical-mixture simulation cells |
| `write_*benchmark_suite*_evidence()` | Write JSON evidence for planned cells |
| `write_next_*benchmark_suite*()` | Continue scheduled long-running evidence builds |
| `summarize_*benchmark_suite*()` | Summarize completed evidence files |
| `load_*monte_carlo*_json()` and `write_*monte_carlo*_json()` | Load, merge, and write strict-JSON simulation reports |

The module-level JSON helpers include
`write_paper_monte_carlo_reproduction_report_json()` for writing a strict-JSON
summary of saved evidence directories and
`write_paper_empirical_mixture_benchmark_suite_chunk_evidence()` for executing
and writing one planned suite chunk in developer scripts. All writers refuse to replace existing files by default; pass `overwrite=True` explicitly when a script is intentionally refreshing a generated report or chunk artifact.

Paper Monte Carlo reproduction report payloads and frame records preserve NumPy
boolean diagnostics and infinite boundary markers. Non-finite values are written
as strict-JSON-safe markers such as `positive_infinity`, `negative_infinity`, or
`null`, while arrays are converted to ordinary JSON lists. integer-valued numeric count strings such as `10.0` are normalized for display. Long reader-facing Monte Carlo method, paper-case, case-id labels, and structured display values are middle-truncated only in the display table; strict rows and frame records retain the complete labels and structured payload values.

Reader-facing Monte Carlo rerun command fields are truncated only in the display `next_action` cell; strict rows and frame records retain the complete executable command strings. Boolean, negative, non-integer, or nonnumeric count values fail before display formatting. Rate, precision, and evidence-count cells remain compact strict-JSON strings rather than Python or NumPy object representations. Very long JSON display cells are capped while strict payloads stay complete. `PaperMonteCarloReproductionRequest` records the evidence directory, table and fixture inputs, schedule seed, cell chunk size, replication budgets, filters, tolerance settings, owner, and rerun command used to build a report.

The current review bundle deliberately excludes partial Monte Carlo preflight
files and chunk evidence directories. Those files remain development evidence
until they satisfy the full paper-coverage contract.

The current manuscript does not use Monte Carlo operating-characteristic results.

## Current Manuscript Boundary

- The current JSS article uses deterministic sharp-null, ADE-bound, lower-bound,
  and partial-density examples plus four selected empirical lower-bound target
  comparisons.
- It does not use Monte Carlo operating-characteristic results.
- A future manuscript revision may add simulation claims only after fixed seeds,
  replication counts, bootstrap counts, table-generation scripts, and accepted
  preflight coverage are recorded under `manuscript/replication/`.
