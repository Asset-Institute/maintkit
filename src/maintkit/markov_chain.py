"""Discrete-time, discrete-state Markov chains.

Conventions
-----------
* Transition matrices are row-stochastic: ``P_t[i, j] = Pr(X_{t+1} = j | X_t = i)``.
* State distributions are row vectors, propagated as ``pi_{t+1} = pi_t @ P_t``.
* The step from ``t`` to ``t + 1`` uses ``P_t``. A horizon of ``N`` means ``N``
  transitions and states ``X_t0, ..., X_{t0+N}``.
* The transitions may be constant, periodic or finite (see
  :class:`maintkit.schedule.Schedule`). A plain ``(S, S)`` array is a constant
  chain; a stack of matrices has to be wrapped to say whether it repeats.
"""
from __future__ import annotations

import numpy as np

from maintkit.schedule import Schedule, as_schedule, check_horizon

__all__ = ["MarkovChain"]


def _validate_stochastic(steps, atol):
    """Check every step is a square row-stochastic matrix.

    Returns the steps, with entries in ``[-atol, 0)`` set to exactly zero (they
    arise from floating-point arithmetic such as ``1 - 0.9 - 0.1``). Raises
    with the offending step and row otherwise.
    """
    if steps.ndim != 3 or steps.shape[1] != steps.shape[2]:
        raise ValueError(
            f"transition matrices must be square, got steps of shape {steps.shape[1:]}"
        )
    if steps.shape[1] == 0:
        raise ValueError("a Markov chain needs at least one state")
    if not np.issubdtype(steps.dtype, np.number) or np.iscomplexobj(steps):
        raise TypeError(f"transition matrices must be real-valued, got dtype {steps.dtype}")
    steps = steps.astype(float)

    if not np.all(np.isfinite(steps)):
        k, i, _ = np.argwhere(~np.isfinite(steps))[0]
        raise ValueError(f"transition matrix at step {k} has a non-finite entry in row {i}")

    if np.any(steps < -atol):
        k, i, j = np.argwhere(steps < -atol)[0]
        raise ValueError(
            f"transition matrix at step {k} has a negative probability "
            f"P[{i}, {j}] = {steps[k, i, j]:g}"
        )
    if np.any(steps < 0):
        steps = np.where(steps < 0, 0.0, steps)

    row_sums = steps.sum(axis=2)
    bad = np.abs(row_sums - 1.0) > atol
    if np.any(bad):
        k, i = np.argwhere(bad)[0]
        raise ValueError(
            f"row {i} of the transition matrix at step {k} sums to "
            f"{row_sums[k, i]:.12g}, not 1 (atol={atol:g})"
        )
    return steps


class MarkovChain:
    """A discrete-time Markov chain on a finite state space.

    Parameters
    ----------
    transitions : array_like (S, S) or Schedule
        A single transition matrix (a constant chain) or a
        :class:`~maintkit.schedule.Schedule` of them (periodic or finite).
    states : sequence, optional
        Labels for the states, in matrix order. Defaults to ``0, ..., S-1``.
        Methods accept a label wherever they take a single state; results are
        always indexed by position.
    atol : float
        Tolerance on row sums, and on negative entries treated as rounding
        error.

    Examples
    --------
    >>> P = np.array([[0.9, 0.1], [0.0, 1.0]])
    >>> chain = MarkovChain(P, states=["working", "failed"])
    >>> chain.distribution("working", horizon=3)[-1]
    array([0.729, 0.271])
    """

    def __init__(self, transitions, states=None, atol=1e-10):
        schedule = as_schedule(transitions, 2, name="transitions")
        steps = _validate_stochastic(schedule.steps, atol)
        self.transitions = Schedule(steps, repeat=schedule.repeat)
        self.atol = float(atol)

        n = steps.shape[1]
        if states is None:
            states = range(n)
        states = tuple(states)
        if len(states) != n:
            raise ValueError(f"got {len(states)} state labels for {n} states")
        if len(set(states)) != n:
            raise ValueError("state labels must be unique")
        self.states = states
        self._index = {s: i for i, s in enumerate(states)}

    # ---- basic properties ---------------------------------------------------

    @property
    def n_states(self):
        return self.transitions.shape[0]

    @property
    def period(self):
        """Period of the transition schedule (1 if constant), None if finite."""
        return self.transitions.period

    @property
    def length(self):
        """Number of steps the transitions are defined for, None if unbounded."""
        return self.transitions.length

    @property
    def is_homogeneous(self):
        """True when the same transition matrix applies at every step."""
        return self.transitions.is_constant

    def state_index(self, state):
        """Position of ``state`` (a label) in the transition matrices."""
        try:
            return self._index[state]
        except (KeyError, TypeError):
            raise KeyError(f"unknown state {state!r}; states are {list(self.states)}") from None

    def _initial_distribution(self, pi0):
        """A state label becomes a point mass; a length-S vector is checked."""
        try:
            index = self.state_index(pi0)
        except KeyError:
            pass
        else:
            pi = np.zeros(self.n_states)
            pi[index] = 1.0
            return pi

        pi = np.asarray(pi0, dtype=float)
        if pi.shape != (self.n_states,):
            raise ValueError(
                f"initial state must be a state label or a distribution of length "
                f"{self.n_states}, got {pi0!r}"
            )
        if np.any(pi < -self.atol) or abs(pi.sum() - 1.0) > self.atol:
            raise ValueError(
                f"initial distribution must be non-negative and sum to 1, got sum "
                f"{pi.sum():.12g}"
            )
        return np.clip(pi, 0.0, None)

    # ---- propagation --------------------------------------------------------

    def distribution(self, pi0, horizon, t0=0):
        """State distributions over ``horizon`` steps from time ``t0``.

        Parameters
        ----------
        pi0 : state label or array_like (S,)
            Initial state, or a distribution over states, at time ``t0``.
        horizon : int
            Number of transitions.
        t0 : int
            Start time, which selects where in the schedule the chain starts.

        Returns
        -------
        ndarray (horizon + 1, S)
            Row ``k`` is the distribution of ``X_{t0+k}``; row 0 is ``pi0``.
        """
        horizon, t0 = self._check_finite_horizon(horizon, t0)
        out = np.empty((horizon + 1, self.n_states))
        out[0] = self._initial_distribution(pi0)
        for k in range(horizon):
            out[k + 1] = out[k] @ self.transitions[t0 + k]
        return out

    def transition_matrix(self, t0, t1):
        r"""Multi-step transition matrix :math:`\Phi(t_0, t_1) = P_{t_0} P_{t_0+1} \cdots P_{t_1-1}`.

        ``Phi[i, j] = Pr(X_{t1} = j | X_{t0} = i)``. The identity when
        ``t1 == t0``.
        """
        if t1 < t0:
            raise ValueError(f"t1 must be >= t0, got t0={t0}, t1={t1}")
        horizon, t0 = self._check_finite_horizon(t1 - t0, t0)
        phi = np.eye(self.n_states)
        for k in range(horizon):
            phi = phi @ self.transitions[t0 + k]
        return phi

    def simulate(self, x0, horizon, n_paths=1, t0=0, rng=None):
        """Simulate sample paths.

        Parameters
        ----------
        x0 : state label or array_like (S,)
            Initial state, or a distribution to draw the initial state from.
        horizon : int
            Number of transitions per path.
        n_paths : int
            Number of independent paths.
        t0 : int
            Start time.
        rng : numpy.random.Generator, int or None
            Random source, or a seed for one.

        Returns
        -------
        ndarray of int (n_paths, horizon + 1)
            State indices (positions, not labels); column 0 is the initial state.
            Map to labels with ``np.asarray(chain.states)[paths]``.
        """
        horizon, t0 = self._check_finite_horizon(horizon, t0)
        if int(n_paths) != n_paths or n_paths < 1:
            raise ValueError(f"n_paths must be a positive integer, got {n_paths!r}")
        n_paths = int(n_paths)
        rng = np.random.default_rng(rng)

        paths = np.empty((n_paths, horizon + 1), dtype=int)
        pi0 = self._initial_distribution(x0)
        paths[:, 0] = _sample_rows(np.cumsum(pi0)[np.newaxis], np.zeros(n_paths, dtype=int), rng)

        # Cumulative rows are computed once per stored step, not once per time
        # step, so a long horizon on a short period costs no extra cumsums.
        cumulative = {}
        for k in range(horizon):
            t = t0 + k
            key = t % self.period if self.period is not None else t
            if key not in cumulative:
                cumulative[key] = np.cumsum(self.transitions[t], axis=1)
            paths[:, k + 1] = _sample_rows(cumulative[key], paths[:, k], rng)
        return paths

    # ---- helpers ------------------------------------------------------------

    def _check_finite_horizon(self, horizon, t0):
        if horizon is None:
            raise ValueError("a finite horizon (number of steps) is required here")
        return check_horizon(self.transitions, horizon, t0, name="the transition schedule")

    def __repr__(self):
        return f"MarkovChain(n_states={self.n_states}, transitions={self.transitions!r})"


def _sample_rows(cumulative, rows, rng):
    """Draw one index per element of ``rows`` from the matching cumulative row.

    Index ``j`` is drawn when ``c[j-1] <= u < c[j]``, so a zero-probability
    state (equal consecutive cumulative values) can never be chosen. Each row
    is rescaled to end at exactly 1, so rounding in the cumulative sum cannot
    push a draw past the last state.
    """
    c = cumulative[rows]
    c = c / c[:, -1:]
    u = rng.random(len(rows))
    return np.sum(u[:, np.newaxis] >= c, axis=1)
