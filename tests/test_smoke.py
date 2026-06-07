"""Post-install smoke test for testmechs package."""


def test_import():
    """Verify the package imports correctly."""
    import testmechs

    assert hasattr(testmechs, "__version__")
    assert hasattr(testmechs, "test_sharp_null")
    assert hasattr(testmechs, "test_sharp_null_cr")
    assert hasattr(testmechs, "lb_frac_affected")
    assert hasattr(testmechs, "bounds_ade_ats")
    assert hasattr(testmechs, "partial_density_data")
    assert hasattr(testmechs, "partial_density_plot")


def test_version_string():
    """Verify version is a valid string."""
    import testmechs

    assert isinstance(testmechs.__version__, str)
    parts = testmechs.__version__.split(".")
    assert len(parts) >= 2


def test_basic_computation():
    """Verify a minimal computation completes without error."""
    import pandas as pd
    import testmechs

    df = pd.DataFrame(
        {
            "treat": [0, 0, 0, 0, 1, 1, 1, 1],
            "mediator": [0, 0, 1, 1, 1, 1, 1, 1],
            "outcome": [0, 1, 0, 1, 0, 1, 1, 1],
        }
    )
    result = testmechs.test_sharp_null(
        df=df, d="treat", m="mediator", y="outcome", method="CS"
    )
    assert hasattr(result, "reject")
    assert hasattr(result, "p_value")
    assert isinstance(result.reject, bool)
