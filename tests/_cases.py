"""Registry of characterisation cases.

Single source of truth shared by ``reference/generate_reference.py`` (which records
current behaviour) and ``test_characterization.py`` (which checks it has not
drifted). Adding a fitter here automatically covers it in both.

Each case is a zero-argument callable returning whatever the fitter returns.
The harness records the returned tuple *elementwise* and makes no assumption
about what the elements mean -- deliberately, since the nine fitters currently
disagree about arity and ordering. Pinning the raw output is what makes the
P2-P5 migrations verifiable.
"""
from __future__ import annotations

import numpy as np

from tests import _datasets as ds


def _weibull_fit():
    from maintkit.distributions import Weibull
    ti, observed = ds.censored_lifetimes()
    return Weibull().fit(ti, p0=[80.0, 1.5], observed=observed)


def _weibull_fit_interval():
    from maintkit.distributions import Weibull
    ti, ins, observed = ds.interval_lifetimes()
    return Weibull().fit_interval(ti, ins, p0=[80.0, 1.5], observed=observed)


def _expdist_fit():
    from maintkit.distributions import Exponential
    ti = ds.exponential_lifetimes()
    return Exponential().fit(ti, observed="all")


def _expdist_fit_censored():
    from maintkit.distributions import Exponential
    ti, observed = ds.censored_lifetimes()
    return Exponential().fit(ti, observed=observed)


def _poisson_process_fit():
    from maintkit.poisson_process import PowerLawNHPP
    events, truncation = ds.nhpp_events()
    model = PowerLawNHPP(0.02, 1.5)
    # exercise the *base class* generic fitter, not the analytic override
    from maintkit.poisson_process import PoissonProcess
    return PoissonProcess.fit(
        model, events, p0=[0.01, 1.2], truncation_times=truncation
    )


def _power_law_nhpp_fit():
    from maintkit.poisson_process import PowerLawNHPP
    events, truncation = ds.nhpp_events()
    return PowerLawNHPP(0.02, 1.5).fit(events, truncation_times=truncation)


def _power_law_nhpp_fit_interval():
    from maintkit.poisson_process import PowerLawNHPP
    counts, inspections = ds.nhpp_interval_counts()
    # estimate_ci removed: ci and cov are always computed now
    return PowerLawNHPP(0.02, 1.5).fit_interval(
        counts, inspections, p0=[0.01, 1.2]
    )


def _imperfect_maintenance_fit():
    from maintkit.imperfect_maintenance import ProportionalAgeReduction
    failures, pm_times, truncation_time = ds.imperfect_maintenance_data()
    return ProportionalAgeReduction(0.02, 1.5, 0.4).fit(
        failures, pm_times, truncation_time
    )


def _wiener_estimate():
    from maintkit.wiener import Wiener
    t, x = ds.wiener_path()
    return Wiener(mu=0.5, sigma=1.0).fit(t, x)


def _rbm_estimate():
    from maintkit.wiener import RegulatedBrownianMotion
    # reflected_path, not wiener_path: regulated Brownian motion cannot produce a
    # wiener_path contains several.
    t, x = ds.reflected_path()
    return RegulatedBrownianMotion(0.5, 1.0).fit(t, x)


CASES = {
    "Weibull.fit": _weibull_fit,
    "Weibull.fit_interval": _weibull_fit_interval,
    "Exponential.fit": _expdist_fit,
    "Exponential.fit_censored": _expdist_fit_censored,
    "PoissonProcess.fit": _poisson_process_fit,
    "PowerLawNHPP.fit": _power_law_nhpp_fit,
    "PowerLawNHPP.fit_interval": _power_law_nhpp_fit_interval,
    "ProportionalAgeReduction.fit": _imperfect_maintenance_fit,
    "Wiener.fit": _wiener_estimate,
    "RegulatedBrownianMotion.fit": _rbm_estimate,
}


def run_case(name):
    """Run one case, returning a JSON-serialisable record of what happened.

    Failures are recorded rather than raised. A fitter that currently blows up
    is still characterised -- and a change in *how* it blows up is still a
    behavioural change worth catching.
    """
    from maintkit.inference import FitResult

    try:
        out = CASES[name]()
    except Exception as exc:  # noqa: BLE001 - characterising current behaviour
        return {"status": "error", "type": type(exc).__name__, "message": str(exc)}

    if isinstance(out, FitResult):
        # Record the same three quantities the pre-migration fitters returned,
        # so recorded values stay comparable across the migration.
        out = (out.params, out.ci, out.cov)
    elif not isinstance(out, tuple):
        out = (out,)

    record = {"status": "ok", "n_outputs": len(out), "outputs": []}
    for element in out:
        if element is None:
            record["outputs"].append(None)
        else:
            record["outputs"].append(np.asarray(element, dtype=float).tolist())
    return record
