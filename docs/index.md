# testmechs Documentation

**testmechs** is a Python package for selected finite-support Testing
Mechanisms calculations. It implements a reporting layer for tests, bounds,
diagnostics, exports, and article reproduction objects from Kwon and Roth's
Testing Mechanisms framework.

## Features

- Sharp-null hypothesis tests (CS and CR methods)
- Lower bounds on fraction affected
- ADE/ATS bounds estimation
- Partial-density data and visualization
- Regression-adjusted probability estimation
- Reproducible request/result contracts
- JSON, DataFrame, and HTML views for the main statistical result objects

## Getting Started

`testmechs` is not yet available from the public Python Package Index. For the
current review bundle, install the supplied source checkout or reviewer artifact.

```bash
cd packages/python/testmechs-py
python -m pip install -e ".[plot]"  # includes matplotlib
```

From an unpacked reviewer submission bundle, use the bundle-local package path:

```bash
python -m pip install -e "package/source[plot]"
```

```python
import pandas as pd
import testmechs

df = pd.DataFrame({
    "treat": [0, 0, 0, 0, 1, 1, 1, 1],
    "mediator": [0, 0, 1, 1, 1, 1, 1, 1],
    "outcome": [0, 1, 0, 1, 0, 1, 1, 1],
})

result = testmechs.test_sharp_null(
    df=df, d="treat", m="mediator", y="outcome", method="CS"
)
print(result.reject, result.p_value)
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
