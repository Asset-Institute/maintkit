"""Gamma distribution and gamma process.

The distribution models a *lifetime*; the process models *degradation* that
accumulates over time. Example 7 connects them: a degradation process plus a
failure threshold induces a lifetime distribution.

Run this file directly, or with ``python examples/gamma_examples.py``.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np

from maintkit.distributions import Gamma
from maintkit.gamma_process import Gamma_Process
from maintkit.inference import ConvergenceWarning

# These fits use finite-difference gradients, so BFGS stops on "precision loss"
# even where the estimates are good. Quietened here to keep the output readable;
# check FitResult.success in your own work.
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


# ============================================================ the distribution

def example_1_reliability():
    """A bearing has Gamma lifetime with shape 2.5 and scale 100 h.
    What fraction survives 150 h, and how does the risk change with age?
    """    
    bearing = Gamma()(2.5, 100.0)          # freeze at (beta, eta)
    print("\n--- 1. Reliability, hazard and conditional reliability of Bearing ~ Gamma(2.5,100)---")

    print(f"R(150)      = {bearing.reliability(150.0):.3f}")
    print(f"MTTF        = {2.5*100.0:.0f} h")
    for t in (50.0, 150.0, 300.0):
        print(f"hazard({t:5.0f}) = {bearing.hazard(t):.5f} /h")
    # A unit already 150 h old: does another 100 h look as safe as the first 100?
    print(f"R(100)      = {bearing.reliability(100.0):.3f}   (new unit)")
    print(f"R(100|150)  = {bearing.conditional_reliability(100.0, 150.0):.3f}   (used unit)")


def example_2_fit_complete():
    """Estimate the parameters from observed failure times."""
    print("\n--- 2. Maximum likelihood, complete data ---")
    eta, beta = 100.0, 2.5
    print(f"truth: eta = {eta}, beta = {beta}")
    failures = Gamma().rvs(beta, eta, size=300, random_state=rng)
    result = Gamma().fit(failures, p0=[80.0, 2.0])
    print(result.summary())    


def example_3_fit_censored():
    """The study ended at 400 h; units still running are right-censored."""
    print("\n--- 3. Right-censored data ---")
    eta, beta = 100.0, 2.5
    print(f"truth: eta = {eta}, beta = {beta}")
    latent = Gamma().rvs(beta, eta, size=300, random_state=rng)
    limit = 400.0
    times = np.minimum(latent, limit)
    observed = (latent <= limit).astype(float)   # 1 = failed, 0 = still running

    print(f"{int(observed.size - observed.sum())} of {observed.size} units censored")
    result = Gamma().fit(times, p0=[80.0, 2.0], observed=observed)
    print(result.summary())


def example_4_fit_interval():
    """Units are inspected every 50 h, so failures fall between inspections."""
    print("\n--- 4. Interval-censored data ---")
    eta, beta = 100.0, 2.5
    failures = Gamma().rvs(beta, eta, size=300, random_state=rng)
    width = 50.0
    lower = np.floor(failures/width)*width
    upper = lower + width

    result = Gamma().fit_interval(upper, lower, p0=[80.0, 2.0])
    print(result.summary())


# ================================================================= the process

def example_5_simulate():
    """Wear accumulates on 20 units. Increments are Gamma, so paths never
    go down -- the point of using this process for degradation.
    """
    print("\n--- 5. Simulating degradation ---")
    wear = Gamma_Process(alpha=0.8, beta=2.0)     # f(t) = alpha*t
    times = np.linspace(0.0, 50.0, 51)
    paths = wear.simulate(times.tolist(), num_samples=20)

    print(f"mean wear at t=50:  simulated {paths[:, -1].mean():.1f}, "
          f"theory {wear.beta*wear.shape(50.0):.1f}")

    plt.figure(figsize=(7, 4))
    plt.plot(times, paths.T, color="0.7", lw=0.8)
    plt.plot(times, wear.beta*wear.shape(times), "r-", lw=2, label=r"$\beta f(t)$")
    plt.xlabel("time (h)"); plt.ylabel("wear")
    plt.title("Gamma process: 20 units")
    plt.legend(); plt.tight_layout()
    return times, paths


def example_6_fit_process(times, paths):
    """Recover the parameters from the inspection records."""
    print("\n--- 6. Fitting the process ---")
    t = [times.tolist()]*paths.shape[0]
    x = [p.tolist() for p in paths]

    result = Gamma_Process().fit(t, x)
    print(result.summary())
    print("truth: alpha = 0.8, beta = 2.0")


def example_7_accelerating():
    """Wear that speeds up with age needs a non-linear shape. Compare
    f(t) = a*t**b against the linear f(t) = alpha*t by likelihood ratio.
    """
    print("\n--- 7. Accelerating wear ---")

    def power(t, a, b):
        return a*t**b

    truth = Gamma_Process(alpha=[0.5, 1.6], beta=1.5)
    truth.shape_function(power)
    times = np.linspace(0.0, 20.0, 41)
    paths = truth.simulate(times.tolist(), num_samples=40)
    t = [times.tolist()]*40
    x = [p.tolist() for p in paths]

    model = Gamma_Process()
    model.shape_function(power)
    fitted = model.fit(t, x)
    print(fitted.summary())
    print("truth: a = 0.5, b = 1.6, beta = 1.5")

    linear = Gamma_Process().fit(t, x)
    print(f"\nlikelihood ratio vs the linear shape: {2*(linear.nnlf - fitted.nnlf):.1f}")
    print("large values favour the power law (1 extra parameter)")


def example_8_threshold():
    """A unit fails when wear crosses 60. The process therefore implies a
    lifetime distribution, which is what Examples 1-4 would be fitting.
    """
    print("\n--- 8. From degradation to a failure time ---")
    wear = Gamma_Process(alpha=0.8, beta=2.0)
    threshold = 60.0
    times = np.linspace(1.0, 80.0, 200)

    # P(failed by t) = P(X(t) > L) = 1 - F(L)
    failed = np.array([1.0 - wear.transition_distribution(threshold, ti, 0.0)
                       for ti in times])

    median = times[np.searchsorted(failed, 0.5)]
    print(f"threshold {threshold:.0f}: median life {median:.0f} h, "
          f"P(fail by 40 h) = {failed[np.searchsorted(times, 40.0)]:.3f}")

    plt.figure(figsize=(7, 4))
    plt.plot(times, failed, lw=2)
    plt.axhline(0.5, color="0.7", ls=":")
    plt.axvline(median, color="0.7", ls=":")
    plt.xlabel("time (h)"); plt.ylabel(f"P(wear > {threshold:.0f})")
    plt.title("Failure time induced by a wear threshold")
    plt.tight_layout()


def main():
    example_1_reliability()
    example_2_fit_complete()
    example_3_fit_censored()
    example_4_fit_interval()
    times, paths = example_5_simulate()
    example_6_fit_process(times, paths)
    example_7_accelerating()
    example_8_threshold()
    plt.show()


if __name__ == "__main__":
    main()
