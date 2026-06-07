"""R-Python parity verification utilities for TestMechs.

This module drives the cross-language equivalence test harness by invoking
R scripts via ``Rscript`` subprocesses and comparing their JSON output with
corresponding Python results.  Functions here build Python-side fixture
summaries, call R helper scripts for each estimation routine, and diff the
resulting structures to detect divergence.
"""

from __future__ import annotations

from collections.abc import Sequence
import json
import subprocess
from pathlib import Path

import pandas as pd

from .contracts import SharedCSVInput


def _json_loads_strict(text: str) -> dict[str, object]:
    """Parse *text* as strict JSON (rejects NaN/Infinity) into a dict."""
    payload = json.loads(text, parse_constant=_reject_parity_json_constant)
    if not isinstance(payload, dict):
        raise ValueError("parity subprocess output must be a JSON object.")
    return payload


def _reject_parity_json_constant(value: str) -> None:
    """Callback that raises on non-finite JSON constants."""
    raise ValueError(f"parity subprocess output must be strict JSON; found {value}.")


def build_python_fixture_summary(dataset: SharedCSVInput) -> dict[str, object]:
    """Build a Python-side fixture summary for parity comparison.

    Reads the CSV dataset and computes column lists, treatment/mediator/
    outcome levels, per-column missing counts, cluster levels, and joint
    treatment-by-mediator cell counts.

    Parameters
    ----------
    dataset : SharedCSVInput
        Shared dataset contract containing the data path and variable names.

    Returns
    -------
    dict[str, object]
        Summary dictionary with keys ``n_rows``, ``columns``,
        ``treatment_levels``, ``mediator_levels``, ``outcome_levels``,
        ``missing_by_column``, ``cluster_levels``, and ``joint_counts``.
    """
    dataframe = pd.read_csv(dataset.data_path)
    columns = dataframe.columns.tolist()

    missing_by_column = {column: int(dataframe[column].isna().sum()) for column in columns}
    cluster_levels = (
        _series_levels(dataframe[dataset.cluster]) if dataset.cluster else []
    )

    joint_frame = (
        dataframe[[dataset.treatment, dataset.primary_mediator]]
        .dropna()
        .value_counts(sort=False)
        .reset_index(name="count")
    )

    summary = {
        "n_rows": int(len(dataframe)),
        "columns": columns,
        "treatment_levels": _series_levels(dataframe[dataset.treatment]),
        "mediator_levels": {
            mediator: _series_levels(dataframe[mediator]) for mediator in dataset.mediators
        },
        "outcome_levels": _series_levels(dataframe[dataset.outcome]),
        "missing_by_column": missing_by_column,
        "cluster_levels": cluster_levels,
        "joint_counts": [
            {
                "d": _stringify_summary_scalar(row[dataset.treatment]),
                "m": _stringify_summary_scalar(row[dataset.primary_mediator]),
                "count": int(row["count"]),
            }
            for row in joint_frame.to_dict(orient="records")
        ],
    }
    return summary


def run_r_fixture_summary(dataset: SharedCSVInput) -> dict[str, object]:
    """Run the R fixture-summary script and return parsed JSON output.

    Parameters
    ----------
    dataset : SharedCSVInput
        Shared dataset contract with data path and variable names.

    Returns
    -------
    dict[str, object]
        R-computed fixture summary matching the structure of
        :func:`build_python_fixture_summary`.

    Raises
    ------
    subprocess.CalledProcessError
        If the Rscript invocation exits with a non-zero code.
    """
    repo_root = Path(__file__).resolve().parents[5]
    script_path = repo_root / "tests" / "python" / "parity" / "r_fixture_summary.R"
    command = [
        "Rscript",
        str(script_path),
        str(dataset.data_path),
        dataset.treatment,
        dataset.primary_mediator,
        dataset.outcome,
        dataset.cluster or "",
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return _json_loads_strict(completed.stdout)


def run_r_sharp_null_cs_binary_m(
    *,
    data_path: str | Path,
    d: str,
    m: str,
    y: str,
    cluster: str | None,
    num_y_bins: int | None,
    reg_formula: str | None = None,
    frac_ats_affected: float | None = None,
    max_defiers_share: float = 0.0,
) -> dict[str, object]:
    """Run R ``test_sharp_null`` CS test for a binary mediator.

    Delegates to :func:`run_r_sharp_null_cs_case` with identical parameters.

    Parameters
    ----------
    data_path : str or Path
        Path to the input CSV fixture.
    d : str
        Treatment column name.
    m : str
        Binary mediator column name.
    y : str
        Outcome column name.
    cluster : str or None
        Cluster column name, or ``None`` for unclustered data.
    num_y_bins : int or None
        Number of outcome bins (``None`` uses R default).
    reg_formula : str or None
        Optional regression formula for probability estimation.
    frac_ats_affected : float or None
        Fraction of always-takers affected (``None`` for default).
    max_defiers_share : float
        Maximum share of defiers allowed.

    Returns
    -------
    dict[str, object]
        R test results including ``reject``, ``p_value``, and ``test_statistic``.
    """
    return run_r_sharp_null_cs_case(
        data_path=data_path,
        d=d,
        m=m,
        y=y,
        cluster=cluster,
        num_y_bins=num_y_bins,
        reg_formula=reg_formula,
        frac_ats_affected=frac_ats_affected,
        max_defiers_share=max_defiers_share,
    )


def run_r_sharp_null_arp_case(
    *,
    data_path: str | Path,
    d: str,
    m: str,
    y: str,
    cluster: str | None,
    num_y_bins: int | None,
) -> dict[str, object]:
    """Run the R ``test_sharp_null`` ARP (Andrews-Roth-Pakes) test.

    Parameters
    ----------
    data_path : str or Path
        Path to the input CSV fixture.
    d : str
        Treatment column name.
    m : str
        Binary mediator column name.
    y : str
        Outcome column name.
    cluster : str or None
        Cluster column name, or ``None`` for unclustered data.
    num_y_bins : int or None
        Number of outcome bins (``None`` uses R default).

    Returns
    -------
    dict[str, object]
        R ARP test results including ``reject``, ``p_value``, and diagnostics.

    Raises
    ------
    subprocess.CalledProcessError
        If the Rscript invocation fails.
    """
    repo_root = Path(__file__).resolve().parents[5]
    script_path = repo_root / "tests" / "python" / "parity" / "r_test_sharp_null_arp_binary_m.R"
    command = [
        "Rscript",
        str(script_path),
        str(Path(data_path).resolve()),
        d,
        m,
        y,
        cluster or "",
        "NULL" if num_y_bins is None else str(num_y_bins),
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return _json_loads_strict(completed.stdout)


def run_r_sharp_null_cs_case(
    *,
    data_path: str | Path,
    d: str,
    m: str,
    y: str,
    cluster: str | None,
    num_y_bins: int | None,
    reg_formula: str | None = None,
    frac_ats_affected: float | None = None,
    max_defiers_share: float = 0.0,
) -> dict[str, object]:
    """Run the R ``test_sharp_null`` CS (conditional-shift) test.

    Parameters
    ----------
    data_path : str or Path
        Path to the input CSV fixture.
    d : str
        Treatment column name.
    m : str
        Binary mediator column name.
    y : str
        Outcome column name.
    cluster : str or None
        Cluster column name, or ``None`` for unclustered data.
    num_y_bins : int or None
        Number of outcome bins (``None`` uses R default).
    reg_formula : str or None
        Optional regression formula for probability estimation.
    frac_ats_affected : float or None
        Fraction of always-takers affected (``None`` for default).
    max_defiers_share : float
        Maximum share of defiers allowed.

    Returns
    -------
    dict[str, object]
        R CS test results including ``reject``, ``p_value``, and diagnostics.

    Raises
    ------
    subprocess.CalledProcessError
        If the Rscript invocation fails.
    """
    repo_root = Path(__file__).resolve().parents[5]
    script_path = repo_root / "tests" / "python" / "parity" / "r_test_sharp_null_cs_binary_m.R"
    command = [
        "Rscript",
        str(script_path),
        str(Path(data_path).resolve()),
        d,
        m,
        y,
        cluster or "",
        "NULL" if num_y_bins is None else str(num_y_bins),
        "NULL" if reg_formula is None else reg_formula,
        "NULL" if frac_ats_affected is None else str(frac_ats_affected),
        str(max_defiers_share),
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return _json_loads_strict(completed.stdout)


def run_r_sharp_null_cr_case(
    *,
    data_path: str | Path,
    d: str,
    m: str,
    y: str,
    B: int = 100,
    eps_bar: float = 1e-3,
    alpha: float = 0.05,
    seed: int = 42,
    num_Ybins: int | None = None,
) -> dict[str, object]:
    """Run the R test_sharp_null_cr (Gurobi backend) and return parsed JSON output.

    Returns a dict with keys including 'gurobi_available', 'reject',
    'confidence_interval_lower', 'confidence_interval_upper',
    'lp_min_unperturbed', 'lp_max_unperturbed', and 'beta_obs'.
    """
    repo_root = Path(__file__).resolve().parents[5]
    script_path = repo_root / "tests" / "python" / "parity" / "r_test_sharp_null_cr.R"
    command = [
        "Rscript",
        str(script_path),
        str(Path(data_path).resolve()),
        d,
        m,
        y,
        str(B),
        str(eps_bar),
        str(alpha),
        str(seed),
        "NULL" if num_Ybins is None else str(num_Ybins),
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return _json_loads_strict(completed.stdout)


def run_r_lb_frac_affected_case(
    *,
    data_path: str | Path,
    d: str,
    m: str | Sequence[str],
    y: str,
    num_y_bins: int | None,
    at_group: object | None,
    reg_formula: str | None = None,
    allow_min_defiers: bool = False,
) -> dict[str, object]:
    """Run the R ``lb_frac_affected`` lower-bound estimator.

    Parameters
    ----------
    data_path : str or Path
        Path to the input CSV fixture.
    d : str
        Treatment column name.
    m : str or Sequence[str]
        Mediator column name(s).  Multiple mediators are comma-joined for R.
    y : str
        Outcome column name.
    num_y_bins : int or None
        Number of outcome bins (``None`` uses R default).
    at_group : object or None
        Always-taker group specification (``None`` for pooled).
    reg_formula : str or None
        Optional regression formula for probability estimation.
    allow_min_defiers : bool
        Whether to use the minimum-defiers relaxation.

    Returns
    -------
    dict[str, object]
        R lower-bound result including ``lower_bound`` and diagnostics.

    Raises
    ------
    subprocess.CalledProcessError
        If the Rscript invocation fails.
    """
    repo_root = Path(__file__).resolve().parents[5]
    script_path = repo_root / "tests" / "python" / "parity" / "r_lb_frac_affected.R"
    command = [
        "Rscript",
        str(script_path),
        str(Path(data_path).resolve()),
        d,
        m if isinstance(m, str) else ",".join(m),
        y,
        "NULL" if num_y_bins is None else str(num_y_bins),
        "NULL" if at_group is None else str(at_group),
        "NULL" if reg_formula is None else reg_formula,
        "TRUE" if allow_min_defiers else "FALSE",
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return _json_loads_strict(completed.stdout)


def run_r_bounds_ade_ats_case(
    *,
    data_path: str | Path,
    d: str,
    m: str | Sequence[str],
    y: str,
    at_group: object,
    num_gridpoints: int = 100,
    max_defiers_share: float = 0.0,
    reg_formula: str | None = None,
    expect_error: bool = False,
) -> dict[str, object]:
    """Run the R ``bounds_ade_ats`` direct-effect bounds estimator.

    Parameters
    ----------
    data_path : str or Path
        Path to the input CSV fixture.
    d : str
        Treatment column name.
    m : str or Sequence[str]
        Mediator column name(s).
    y : str
        Outcome column name.
    at_group : object
        Always-taker group level(s).
    num_gridpoints : int
        Number of grid points for the bounds calculation.
    max_defiers_share : float
        Maximum share of defiers allowed.
    reg_formula : str or None
        Optional regression formula for probability estimation.
    expect_error : bool
        If ``True``, do not raise on R-side errors; return the error payload.

    Returns
    -------
    dict[str, object]
        R bounds result including ``lower_bound``, ``upper_bound``, and
        diagnostics.

    Raises
    ------
    RuntimeError
        If R reports an error and *expect_error* is ``False``.
    subprocess.CalledProcessError
        If the Rscript invocation fails.
    """
    repo_root = Path(__file__).resolve().parents[5]
    script_path = repo_root / "tests" / "python" / "parity" / "r_bounds_ade_ats.R"
    command = [
        "Rscript",
        str(script_path),
        str(Path(data_path).resolve()),
        d,
        m if isinstance(m, str) else ",".join(m),
        y,
        _r_argument_for_level(at_group),
        str(num_gridpoints),
        str(max_defiers_share),
        "NULL" if reg_formula is None else reg_formula,
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    result = _json_loads_strict(completed.stdout)
    if result.get("error") and not expect_error:
        raise RuntimeError(str(result["error"]))
    return result


def _r_argument_for_level(value: object) -> str:
    """Format a Python level value as an R command-line argument string."""
    if value is None:
        return "NULL"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return ",".join(str(item) for item in value)
    return str(value)


def run_r_builtin_dataset_extraction(*, output_dir: str | Path) -> dict[str, object]:
    """Extract built-in R package datasets to *output_dir* as CSVs.

    Parameters
    ----------
    output_dir : str or Path
        Directory where extracted CSV files are written.

    Returns
    -------
    dict[str, object]
        Summary including extracted dataset names and row counts.

    Raises
    ------
    subprocess.CalledProcessError
        If the Rscript invocation fails.
    """
    repo_root = Path(__file__).resolve().parents[5]
    script_path = repo_root / "tests" / "python" / "parity" / "r_extract_builtin_datasets.R"
    command = [
        "Rscript",
        str(script_path),
        str(Path(output_dir).resolve()),
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return _json_loads_strict(completed.stdout)


def run_r_readme_empirical_case(
    *,
    case_name: str,
    data_path: str | Path,
) -> dict[str, object]:
    """Run a single R README empirical case by name.

    Parameters
    ----------
    case_name : str
        Name of the empirical case to execute.
    data_path : str or Path
        Path to the input CSV fixture.

    Returns
    -------
    dict[str, object]
        R empirical result for the specified case.
    """
    return run_r_readme_empirical_cases(
        case_names=[case_name],
        data_path=data_path,
    )[case_name]


def run_r_readme_empirical_cases(
    *,
    case_names: list[str],
    data_path: str | Path,
) -> dict[str, dict[str, object]]:
    """Run multiple R README empirical cases and return results keyed by name.

    Parameters
    ----------
    case_names : list[str]
        Names of the empirical cases to execute.
    data_path : str or Path
        Path to the input CSV fixture.

    Returns
    -------
    dict[str, dict[str, object]]
        Mapping from case name to R empirical result dictionary.

    Raises
    ------
    subprocess.CalledProcessError
        If the Rscript invocation fails.
    """
    repo_root = Path(__file__).resolve().parents[5]
    script_path = repo_root / "tests" / "python" / "parity" / "r_readme_empirical_cases.R"
    command = [
        "Rscript",
        str(script_path),
        str(Path(data_path).resolve()),
        *case_names,
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return _json_loads_strict(completed.stdout)


def run_r_regression_probability_case(
    *,
    data_path: str | Path,
    d: str,
    m: str,
    y: str,
    reg_formula: str,
) -> dict[str, object]:
    """Run the R regression-probability helper and return parsed JSON.

    Parameters
    ----------
    data_path : str or Path
        Path to the input CSV fixture.
    d : str
        Treatment column name.
    m : str
        Mediator column name.
    y : str
        Outcome column name.
    reg_formula : str
        R-style regression formula.

    Returns
    -------
    dict[str, object]
        R regression-probability result.

    Raises
    ------
    subprocess.CalledProcessError
        If the Rscript invocation fails.
    """
    repo_root = Path(__file__).resolve().parents[5]
    script_path = repo_root / "tests" / "python" / "parity" / "r_regression_probabilities.R"
    command = [
        "Rscript",
        str(script_path),
        str(Path(data_path).resolve()),
        d,
        m,
        y,
        reg_formula,
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return _json_loads_strict(completed.stdout)


def run_r_simulate_data_binary_m_contract(
    *,
    n: int,
    seed: int,
) -> dict[str, object]:
    """Run R ``simulate_data`` with a binary mediator contract.

    Parameters
    ----------
    n : int
        Number of observations to simulate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict[str, object]
        R simulation result including data summary and contract metadata.

    Raises
    ------
    subprocess.CalledProcessError
        If the Rscript invocation fails.
    """
    repo_root = Path(__file__).resolve().parents[5]
    script_path = repo_root / "tests" / "python" / "parity" / "r_simulate_data_binaryM_contract.R"
    command = [
        "Rscript",
        str(script_path),
        str(n),
        str(seed),
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return _json_loads_strict(completed.stdout)


def diff_fixture_summaries(
    python_summary: dict[str, object],
    r_summary: dict[str, object],
) -> list[str]:
    """Compare Python and R fixture summaries and return differences.

    Both summaries are normalized (sorted levels, integer counts) before
    comparison so that harmless ordering or type differences do not cause
    false divergences.

    Parameters
    ----------
    python_summary : dict[str, object]
        Summary produced by :func:`build_python_fixture_summary`.
    r_summary : dict[str, object]
        Summary produced by :func:`run_r_fixture_summary`.

    Returns
    -------
    list[str]
        List of human-readable difference descriptions, one per divergent
        key.  Empty if the summaries are equivalent.
    """
    normalized_python = _normalize_summary(python_summary)
    normalized_r = _normalize_summary(r_summary)
    if normalized_python == normalized_r:
        return []

    differences: list[str] = []
    all_keys = sorted(set(normalized_python) | set(normalized_r))
    for key in all_keys:
        if normalized_python.get(key) != normalized_r.get(key):
            differences.append(
                f"{key}: python={normalized_python.get(key)!r} r={normalized_r.get(key)!r}"
            )
    return differences


def _normalize_summary(summary: dict[str, object]) -> dict[str, object]:
    """Normalize a fixture summary for stable cross-language comparison."""
    normalized = dict(summary)
    normalized["columns"] = list(summary.get("columns", []))
    normalized["treatment_levels"] = list(summary.get("treatment_levels", []))
    normalized["outcome_levels"] = list(summary.get("outcome_levels", []))
    normalized["cluster_levels"] = list(summary.get("cluster_levels", []))
    normalized["missing_by_column"] = {
        key: int(value) for key, value in dict(summary.get("missing_by_column", {})).items()
    }
    normalized["mediator_levels"] = {
        key: list(value) for key, value in dict(summary.get("mediator_levels", {})).items()
    }
    normalized["joint_counts"] = sorted(
        [
            {
                "d": str(item["d"]),
                "m": str(item["m"]),
                "count": int(item["count"]),
            }
            for item in list(summary.get("joint_counts", []))
        ],
        key=lambda item: (item["d"], item["m"]),
    )
    normalized["n_rows"] = int(summary["n_rows"])
    return normalized


def _series_levels(series: pd.Series) -> list[str]:
    """Extract sorted unique non-missing levels as strings."""
    return sorted(_stringify_summary_scalar(value) for value in series.dropna().unique().tolist())


def _stringify_summary_scalar(value: object) -> str:
    """Stringify a scalar value for level comparison (rejects NA)."""
    if pd.isna(value):
        raise ValueError("Summary scalars must be non-missing before stringification.")
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (AttributeError, ValueError):
            pass
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)
