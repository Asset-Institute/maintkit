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
    from maintkit.distributions import weibull
    ti, observed = ds.censored_lifetimes()
    return weibull().fit(ti, p0=[80.0, 1.5], observed=observed)


def _weibull_fit_interval():
    from maintkit.distributions import weibull
    ti, ins, observed = ds.interval_lifetimes()
    return weibull().fit_interval(ti, ins, p0=[80.0, 1.5], observed=observed)


def _expdist_fit():
    from maintkit.distributions import expdist
    ti = ds.exponential_lifetimes()
    return expdist().fit(ti, observed="all")


def _expdist_fit_censored():
    from maintkit.distributions import expdist
    ti, observed = ds.censored_lifetimes()
    return expdist().fit(ti, observed=observed)


def _poisson_process_fit():
    from maintkit.poisson_process import power_law_nhpp
    events, truncation = ds.nhpp_events()
    model = power_law_nhpp(0.02, 1.5)
    # exercise the *base class* generic fitter, not the analytic override
    from maintkit.poisson_process import poisson_process
    return poisson_process.fit(
        model, events, p0=[0.01, 1.2], truncation_times=truncation
    )


def _power_law_nhpp_fit():
    from maintkit.poisson_process import power_law_nhpp
    events, truncation = ds.nhpp_events()
    return power_law_nhpp(0.02, 1.5).fit(events, truncation_times=truncation)


def _power_law_nhpp_fit_interval():
    from maintkit.poisson_process import power_law_nhpp
    counts, inspections = ds.nhpp_interval_counts()
    # estimate_ci removed: ci and cov are always computed now
    return power_law_nhpp(0.02, 1.5).fit_interval(
        counts, inspections, p0=[0.01, 1.2]
    )


def _imperfect_maintenance_fit():
    from maintkit.imperfect_maintenance import imperfect_pm_minimal_cm
    failures, pm_times = ds.imperfect_maintenance_data()
    return imperfect_pm_minimal_cm(0.02, 1.5, 0.4).fit(failures, pm_times)


def _wiener_estimate():
    from maintkit.wiener import Wiener
    t, x = ds.wiener_path()
    return Wiener(mu=0.5, sigma=1.0).estimate_parameters(t, x)


def _rbm_estimate():
    from maintkit.wiener import RBM
    t, x = ds.wiener_path()
    return RBM(0.5, 1.0).estimate_parameters(t, x)


CASES = {
    "weibull.fit": _weibull_fit,
    "weibull.fit_interval": _weibull_fit_interval,
    "expdist.fit": _expdist_fit,
    "expdist.fit_censored": _expdist_fit_censored,
    "poisson_process.fit": _poisson_process_fit,
    "power_law_nhpp.fit": _power_law_nhpp_fit,
    "power_law_nhpp.fit_interval": _power_law_nhpp_fit_interval,
    "imperfect_pm_minimal_cm.fit": _imperfect_maintenance_fit,
    "Wiener.estimate_parameters": _wiener_estimate,
    "RBM.estimate_parameters": _rbm_estimate,
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
