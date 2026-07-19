# maintkit

Reliability analysis and maintenance optimization toolkit.

`maintkit` collects failure-time modelling, recurrent-event (Poisson process)
models, imperfect-maintenance models, degradation (Wiener) models, maintenance
optimization, and nonparametric estimators / probability plotting into a single
installable Python package. It builds on `scipy.stats`, adding censoring-aware
maximum-likelihood fitting and reliability-specific methods (reliability,
hazard, conditional reliability, confidence bounds).

## Installation

```bash
pip install -e .
```

Requires Python >= 3.9. Dependencies (`numpy`, `scipy`, `matplotlib`,
`numdifftools`) are installed automatically.

## Quick start

```python
import numpy as np
import maintkit as mk

# Weibull MLE with right-censoring
ti = np.array([12., 45., 88., 90., 110., 130.])
observed = np.array([1, 1, 1, 0, 1, 0])          # 0 = right-censored
p_hat, p_ci, p_cov = mk.weibull().fit(ti, p0=[80., 1.5], observed=observed)
eta_hat, beta_hat = p_hat

# Nonparametric CDF estimate (Kaplan-Meier)
t, F, lb, ub = mk.kaplan_meier(ti, observed, plot=False)

# Recurrent events: fit a power-law NHPP across several assets
model = mk.power_law_nhpp(a=0.02, b=1.5)
event_times = [[10., 40., 95.], [22., 60.]]       # one list per asset
p_hat, p_ci, p_cov = model.fit(event_times, truncation_times=[100., 100.])
```

## Package layout

```
src/maintkit/
    distributions.py            # expdist, weibull, reliability_from_hazard + frozen wrappers
    poisson_process.py          # poisson_process, power_law_nhpp
    imperfect_maintenance.py    # imperfect_pm_minimal_cm (proportional age reduction)
    maintenance_optimization.py # interval_replacement
    probability_plotting.py     # ecdf, kaplan_meier, empirical_mean_cumulative_function, weibull plots
    wiener.py                   # Wiener, RBM (regulated Brownian motion) degradation models
    utilities.py                # parameter transforms, helpers
tests/                          # pytest regression + core tests
examples/                       # Jupyter notebooks
legacy/matlab/                  # original MATLAB implementation (deprecated)
```

## Tests

```bash
pip install -e ".[test]"
pytest
```

## Development setup

Notebook outputs are stripped automatically on commit, so `examples/*.ipynb`
produce clean diffs. Enable the hooks once per clone:

```bash
pip install -e ".[dev]"
pre-commit install
```

To strip outputs from all notebooks right now (one-time cleanup):

```bash
pre-commit run nbstripout --all-files
```

Outputs are removed only from what git records; your working copy is rewritten
in place, so re-run the notebook if you want to keep viewing results locally.

## Notes

- The public API is re-exported from the package root, so `mk.weibull`,
  `mk.kaplan_meier`, etc. all work directly.
- The previous `weiner` (misspelled) class is kept as an alias of `Wiener`
  for backward compatibility.
- The MATLAB code under `legacy/matlab/` is retained for reference only and is
  no longer maintained.
