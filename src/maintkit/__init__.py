"""maintkit: reliability analysis and maintenance optimization toolkit.

Public API groups:

* Failure-time distributions with censoring-aware MLE
  (``Exponential``, ``Weibull``, ``ReliabilityFromHazard``).
* Recurrent-event (Poisson process) models
  (``PoissonProcess``, ``PowerLawNHPP``).
* Imperfect maintenance models (``ProportionalAgeReduction``).
* Maintenance optimization (``IntervalReplacement``).
* Nonparametric estimators and probability plotting
  (``ecdf``, ``kaplan_meier``, ``empirical_mean_cumulative_function``,
  ``weibull_probability_plot``).
* Degradation models (``Wiener``, ``RegulatedBrownianMotion``).
"""

from maintkit.distributions import (
    ReliabilityDistribution,
    ReliabilityDistributionFrozen,
    ReliabilityFromHazard,
    Exponential,
    Weibull,
)
from maintkit.poisson_process import PoissonProcess, PowerLawNHPP
from maintkit.imperfect_maintenance import ProportionalAgeReduction
from maintkit.maintenance_optimization import IntervalReplacement
from maintkit.probability_plotting import (
    ecdf,
    kaplan_meier,
    empirical_mean_cumulative_function,
    weibull_probability_plot,
    weibull_reliability_confidence_interval,
)
from maintkit.wiener import Wiener, RegulatedBrownianMotion
from maintkit.inference import (
    FitResult,
    ConvergenceWarning,
    fit_mle,
    result_at,
    result_from_covariance,
    hessian_at,
)
from maintkit.transforms import Transform, Identity, Log, Logit, Composite

__version__ = "0.2.0"

__all__ = [
    "ReliabilityDistribution",
    "ReliabilityDistributionFrozen",
    "ReliabilityFromHazard",
    "Exponential",
    "Weibull",
    "PoissonProcess",
    "PowerLawNHPP",
    "ProportionalAgeReduction",
    "IntervalReplacement",
    "ecdf",
    "kaplan_meier",
    "empirical_mean_cumulative_function",
    "weibull_probability_plot",
    "weibull_reliability_confidence_interval",
    "Wiener",
    "RegulatedBrownianMotion",
    "FitResult",
    "ConvergenceWarning",
    "fit_mle",
    "result_at",
    "result_from_covariance",
    "hessian_at",
    "Transform",
    "Identity",
    "Log",
    "Logit",
    "Composite",
    "__version__",
]
