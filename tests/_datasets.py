"""Deterministic datasets shared by the characterisation harness.

Built with plain numpy and fixed seeds rather than with maintkit's own
simulators, so the data stays identical even if those simulators change. That
matters: a characterisation test is only meaningful if the *input* is frozen.
"""
from __future__ import annotations

import numpy as np


def censored_lifetimes(n=200, beta=2.0, eta=100.0, censor_every=5, seed=101):
    """Right-censored Weibull sample. Returns (ti, observed)."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(size=n)
    ti = eta * (-np.log(1.0 - u)) ** (1.0 / beta)
    observed = np.ones(n)
    observed[::censor_every] = 0
    return ti, observed


def interval_lifetimes(n=150, beta=2.0, eta=100.0, width=20.0, seed=202):
    """Interval-censored Weibull sample. Returns (ti, ins, observed).

    ``ins`` is the lower bound of the interval, ``ti`` the upper bound.
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(size=n)
    exact = eta * (-np.log(1.0 - u)) ** (1.0 / beta)
    ins = np.floor(exact / width) * width
    ti = ins + width
    observed = np.ones(n)
    observed[::7] = 0
    return ti, ins, observed


def exponential_lifetimes(n=300, scale=25.0, seed=303):
    rng = np.random.default_rng(seed)
    return rng.exponential(scale=scale, size=n)


def nhpp_events(n_assets=25, a=0.02, b=1.5, horizon=200.0, seed=404):
    """Power-law NHPP arrivals by inversion. Returns (event_times, truncation)."""
    rng = np.random.default_rng(seed)
    events = []
    for _ in range(n_assets):
        t, times = 0.0, []
        while True:
            u = rng.uniform()
            t = (-np.log(1.0 - u) / a + t**b) ** (1.0 / b)
            if t >= horizon:
                break
            times.append(float(t))
        events.append(times)
    return events, [float(horizon)] * n_assets


def nhpp_events_unequal_horizons(a=0.02, b=1.5, horizons=None, seed=808):
    """Power-law NHPP arrivals with a DIFFERENT horizon per asset.

    The equal-horizon dataset above cannot detect the Crow closed form's
    failure mode, because that estimator is exact when all truncation times
    agree. Staggered horizons are also the realistic case: assets in a fleet
    enter service at different dates.
    """
    if horizons is None:
        horizons = [50.0, 100.0, 150.0, 200.0, 400.0] * 5
    rng = np.random.default_rng(seed)
    events = []
    for horizon in horizons:
        t, times = 0.0, []
        while True:
            u = rng.uniform()
            t = (-np.log(1.0 - u) / a + t**b) ** (1.0 / b)
            if t >= horizon:
                break
            times.append(float(t))
        events.append(times)
    return events, [float(h) for h in horizons]


def nhpp_interval_counts(n_assets=15, a=0.02, b=1.5, horizon=200.0,
                         n_inspections=8, seed=505):
    """Counts of arrivals between inspections. Returns (counts, inspections)."""
    events, _ = nhpp_events(n_assets=n_assets, a=a, b=b, horizon=horizon, seed=seed)
    inspections = list(np.linspace(0.0, horizon, n_inspections + 1))
    counts = []
    for times in events:
        arr = np.asarray(times)
        counts.append([
            int(((arr > lo) & (arr <= hi)).sum())
            for lo, hi in zip(inspections[:-1], inspections[1:])
        ])
    return counts, [list(inspections) for _ in range(n_assets)]


def imperfect_maintenance_data(a=0.02, b=1.5, rho=0.4, n_pm=6,
                               pm_spacing=50.0, seed=606):
    """Failure times under proportional age reduction.

    Returns ``(failures, pm_times, truncation_time)``. ``pm_times`` holds only
    the maintenance actions -- observation starts at 0 and ends at
    ``truncation_time``, neither of which is a PM.
    """
    rng = np.random.default_rng(seed)
    edges = [float(pm_spacing * k) for k in range(n_pm + 1)]
    pm_times = edges[1:-1]                 # drop the origin and the horizon
    truncation_time = edges[-1]
    failures = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        virtual_start = lo - rho * lo
        t = virtual_start
        while True:
            u = rng.uniform()
            t = (-np.log(1.0 - u) / a + t**b) ** (1.0 / b)
            real = t + rho * lo
            if real >= hi:
                break
            failures.append(float(real))
    return np.array(failures), np.array(pm_times), truncation_time


def wiener_path(n_steps=400, horizon=50.0, mu=0.5, sigma=1.0, seed=707):
    """Single Wiener path. Returns (t, x) as lists-of-lists (one run)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, horizon, n_steps)
    dt = np.diff(t)
    dx = rng.normal(loc=mu * dt, scale=sigma * np.sqrt(dt))
    x = np.concatenate([[0.0], np.cumsum(dx)])
    return [t.tolist()], [x.tolist()]
