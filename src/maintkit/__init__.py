"""maintkit: reliability analysis and maintenance optimization toolkit.

Public API groups:

* Failure-time distributions with censoring-aware MLE
  (``expdist``, ``weibull``, ``reliability_from_hazard``).
* Recurrent-event (Poisson process) models
  (``poisson_process``, ``power_law_nhpp``).
* Imperfect maintenance models (``imperfect_pm_minimal_cm``).
* Maintenance optimization (``interval_replacement``).
* Nonparametric estimators and probability plotting
  (``ecdf``, ``kaplan_meier``, ``empirical_mean_cumulative_function``,
  ``weibull_probability_plot``).
* Degradation models (``Wiener``, ``RBM``).
"""

from maintkit.distributions import (
    reliability_distribution,
    reliability_distribution_frozen,
    reliability_from_hazard,
    expdist,
    weibull,
)
from maintkit.poisson_process import poisson_process, power_law_nhpp
from maintkit.imperfect_maintenance import imperfect_pm_minimal_cm
from maintkit.maintenance_optimization import interval_replacement
from maintkit.probability_plotting import (
    ecdf,
    kaplan_meier,
    empirical_mean_cumulative_function,
    weibull_probability_plot,
    weibull_reliability_confidence_interval,
)
from maintkit.wiener import Wiener, RBM, weiner
from maintkit.inference import FitResult, fit_mle, result_at, hessian_at
from maintkit.transforms import Transform, Identity, Log, Logit, Composite

__version__ = "0.1.0"

__all__ = [
    "reliability_distribution",
    "reliability_distribution_frozen",
    "reliability_from_hazard",
    "expdist",
    "weibull",
    "poisson_process",
    "power_law_nhpp",
    "imperfect_pm_minimal_cm",
    "interval_replacement",
    "ecdf",
    "kaplan_meier",
    "empirical_mean_cumulative_function",
    "weibull_probability_plot",
    "weibull_reliability_confidence_interval",
    "Wiener",
    "RBM",
    "weiner",
    "FitResult",
    "fit_mle",
    "result_at",
    "hessian_at",
    "Transform",
    "Identity",
    "Log",
    "Logit",
    "Composite",
    "__version__",
]
