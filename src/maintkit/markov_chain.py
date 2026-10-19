"""Discrete-time, discrete-state Markov chains.

Conventions
-----------
* Transition matrices are row-stochastic: ``P_t[i, j] = Pr(X_{t+1} = j | X_t = i)``.
* State distributions are row vectors, propagated as ``pi_{t+1} = pi_t @ P_t``.
* The step from ``t`` to ``t + 1`` uses ``P_t``. A horizon of ``N`` means ``N``
  transitions and states ``X_t0, ..., X_{t0+N}``.
* The transitions may be constant, periodic or finite (see
  :class:`maintkit.time_indexed.TimeIndexed`). A plain ``(S, S)`` array is a constant
  chain; a stack of matrices has to be wrapped to say whether it repeats.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse.csgraph import breadth_first_order, connected_components

from maintkit.time_indexed import TimeIndexed, as_time_indexed, check_horizon

__all__ = ["MarkovChain", "AbsorptionResult"]


@dataclass(frozen=True)
class AbsorptionResult:
    """Absorption analysis of a constant chain into its closed classes.

    Vectors are indexed by state (length ``S``); recurrent states have zero
    expected steps and absorb into their own class with probability 1.

    Attributes
    ----------
    transient : ndarray of int
        Transient states, in the order used by ``fundamental``.
    closed_classes : list of ndarray of int
        The closed (recurrent) classes, in the order of the columns of
        ``class_probabilities``.
    fundamental : ndarray (n_transient, n_transient)
        :math:`N = (I - Q)^{-1}`; ``N[a, b]`` is the expected number of visits
        to ``transient[b]`` starting from ``transient[a]`` (counting the start).
    expected_steps : ndarray (S,)
        Expected number of transitions until the chain enters a closed class.
        With a single absorbing "failed" state this is the MTTF in steps.
    variance_steps : ndarray (S,)
        Variance of that number of transitions.
    class_probabilities : ndarray (S, n_closed)
        Probability of ending up in each closed class.
    """

    transient: np.ndarray
    closed_classes: list
    fundamental: np.ndarray
    expected_steps: np.ndarray
    variance_steps: np.ndarray
    class_probabilities: np.ndarray


def _gth_stationary(P):
    """Stationary distribution of an irreducible stochastic matrix by GTH.

    Grassmann-Taksar-Heyman state reduction uses no subtractions, so it keeps
    full relative accuracy for nearly decomposable chains (rare transitions
    between groups of states), where solving ``pi (I - P) = 0`` directly loses
    digits to cancellation.
    """
    A = np.array(P, dtype=float)
    n = A.shape[0]
    for k in range(n - 1, 0, -1):
        s = A[k, :k].sum()           # = 1 - A[k, k] for the censored chain, without the subtraction
        A[:k, k] /= s
        A[:k, :k] += np.outer(A[:k, k], A[k, :k])
    pi = np.zeros(n)
    pi[0] = 1.0
    for k in range(1, n):
        pi[k] = pi[:k] @ A[:k, k]
    return pi / pi.sum()


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
    transitions : array_like (S, S) or TimeIndexed
        A single transition matrix (a constant chain) or a
        :class:`~maintkit.time_indexed.TimeIndexed` of them (periodic or finite).
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
        seq = as_time_indexed(transitions, 2, name="transitions")
        steps = _validate_stochastic(seq.steps, atol)
        self.transitions = TimeIndexed(steps, repeat=seq.repeat)
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
        """Period of the transitions (1 if constant), None if finite."""
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
            Start time, which selects where in a time-varying chain it starts.

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

    # ---- structure (constant chains) ---------------------------------------
    #
    # Everything below needs a single transition matrix. The structural
    # results (classes, periods) depend only on which entries are non-zero:
    # any positive entry, however small, is a possible transition.

    def _matrix(self, what):
        if not self.is_homogeneous:
            raise ValueError(
                f"{what} is defined here for a constant chain only; this chain's "
                f"transitions are {self.transitions!r}"
            )
        return self.transitions[0]

    def _states(self, states):
        """A single state label, or an iterable of them, as sorted unique indices."""
        try:
            return np.array([self.state_index(states)])
        except KeyError:
            pass
        try:
            items = list(states)
        except TypeError:
            raise KeyError(f"unknown state {states!r}; states are {list(self.states)}") from None
        if not items:
            raise ValueError("at least one state is required")
        return np.unique([self.state_index(s) for s in items])

    def _classes(self, P):
        n_classes, labels = connected_components(P > 0, directed=True, connection="strong")
        classes = [np.flatnonzero(labels == c) for c in range(n_classes)]
        classes.sort(key=lambda c: c[0])
        return classes

    def communicating_classes(self):
        """Communicating classes, each an array of state indices, ordered by smallest member."""
        return self._classes(self._matrix("communicating_classes"))

    def closed_classes(self):
        """The closed (recurrent) classes: those the chain can never leave."""
        P = self._matrix("closed_classes")
        out = []
        for c in self._classes(P):
            outside = np.ones(self.n_states, dtype=bool)
            outside[c] = False
            if not np.any(P[np.ix_(c, outside)] > 0):
                out.append(c)
        return out

    def transient_states(self):
        """States that are not in any closed class."""
        closed = self.closed_classes()
        recurrent = np.concatenate(closed)
        return np.setdiff1d(np.arange(self.n_states), recurrent)

    def absorbing_states(self):
        """States the chain never leaves once entered (``P[i, i] == 1``)."""
        P = self._matrix("absorbing_states")
        off_diagonal = P - np.diag(np.diag(P))
        return np.flatnonzero(~np.any(off_diagonal > 0, axis=1))

    def is_irreducible(self):
        return len(self.communicating_classes()) == 1

    def state_period(self, state=None):
        """Period of a state: the gcd of the lengths of its return paths.

        All states in a communicating class share a period. ``state`` may be
        omitted for an irreducible chain. Returns None for a state that can
        never return to itself (a transient state with no cycle through it).
        """
        P = self._matrix("state_period")
        classes = self._classes(P)
        if state is None:
            if len(classes) != 1:
                raise ValueError("the chain is reducible; say which state's period you want")
            members = classes[0]
        else:
            i = self.state_index(state)
            members = next(c for c in classes if i in c)

        sub = P[np.ix_(members, members)] > 0
        if not sub.any():
            return None
        # Breadth-first levels from one member; every edge u -> v inside the
        # class closes a cycle whose length is congruent to level[u] + 1 - level[v].
        order, _ = breadth_first_order(sub, 0, directed=True)
        level = np.full(len(members), -1)
        level[0] = 0
        for u in order:
            for v in np.flatnonzero(sub[u]):
                if level[v] < 0:
                    level[v] = level[u] + 1
        d = 0
        for u, v in np.argwhere(sub):
            d = math.gcd(d, int(level[u] + 1 - level[v]))
        return abs(d)

    # ---- long-run behaviour (constant chains) -------------------------------

    def stationary_distributions(self):
        """One stationary distribution per closed class, as rows of an ``(n_closed, S)`` array.

        Row ``r`` is supported on ``closed_classes()[r]``. Every stationary
        distribution of the chain is a convex combination of these rows.
        """
        P = self._matrix("stationary_distributions")
        closed = self.closed_classes()
        out = np.zeros((len(closed), self.n_states))
        for r, c in enumerate(closed):
            out[r, c] = _gth_stationary(P[np.ix_(c, c)])
        return out

    def stationary_distribution(self):
        """The stationary distribution, when it is unique (exactly one closed class).

        It is the long-run fraction of time spent in each state. The
        distribution of ``X_t`` converges to it only if that class is also
        aperiodic (see :meth:`state_period`).
        """
        pis = self.stationary_distributions()
        if pis.shape[0] != 1:
            raise ValueError(
                f"the chain has {pis.shape[0]} closed classes, so its stationary "
                "distribution is not unique; use stationary_distributions()"
            )
        return pis[0]

    def mean_first_passage_times(self):
        """Matrix ``M[i, j]`` of expected steps to first reach ``j`` from ``i``.

        For an irreducible chain. The diagonal is the mean recurrence time
        ``1 / pi_j`` (steps to *return*); off the diagonal it matches
        ``expected_hitting_times(j)[i]``.
        """
        P = self._matrix("mean_first_passage_times")
        if not self.is_irreducible():
            raise ValueError("mean first passage times need an irreducible chain")
        pi = _gth_stationary(P)
        n = self.n_states
        Z = np.linalg.inv(np.eye(n) - P + np.outer(np.ones(n), pi))
        return (np.diag(Z)[np.newaxis, :] - Z) / pi[np.newaxis, :] + np.diag(1.0 / pi)

    # ---- absorption and hitting (constant chains) ---------------------------

    def absorption(self):
        """Absorption of the transient states into the closed classes.

        See :class:`AbsorptionResult`. For a degradation chain whose only
        closed class is an absorbing failed state, ``expected_steps`` is the
        MTTF in steps from each state and ``variance_steps`` its variance.
        """
        P = self._matrix("absorption")
        closed = self.closed_classes()
        transient = self.transient_states()
        n, m = self.n_states, len(transient)

        Q = P[np.ix_(transient, transient)]
        N = np.linalg.inv(np.eye(m) - Q) if m else np.zeros((0, 0))
        t = N @ np.ones(m)

        expected = np.zeros(n)
        variance = np.zeros(n)
        expected[transient] = t
        variance[transient] = (2 * N - np.eye(m)) @ t - t * t

        probs = np.zeros((n, len(closed)))
        for r, c in enumerate(closed):
            probs[c, r] = 1.0
            probs[transient, r] = N @ P[np.ix_(transient, c)].sum(axis=1)

        return AbsorptionResult(transient, closed, N, expected, variance, probs)

    def _can_reach(self, P, target_mask):
        """Mask of states with a path (of length >= 0) into the target set."""
        reach = target_mask.copy()
        edges = P > 0
        while True:
            new = reach | np.any(edges[:, reach], axis=1)
            if np.array_equal(new, reach):
                return reach
            reach = new

    def hitting_probabilities(self, target):
        """Probability of ever entering ``target`` (a state or a set of states).

        1 on the target itself; 0 from states with no path to it.
        """
        P = self._matrix("hitting_probabilities")
        in_target = np.zeros(self.n_states, dtype=bool)
        in_target[self._states(target)] = True

        h = in_target.astype(float)
        solve = self._can_reach(P, in_target) & ~in_target
        if solve.any():
            A = np.eye(solve.sum()) - P[np.ix_(solve, solve)]
            h[solve] = np.linalg.solve(A, P[np.ix_(solve, in_target)].sum(axis=1))
        return h

    def expected_hitting_times(self, target):
        """Expected number of steps to first enter ``target`` (a state or a set of states).

        0 on the target itself, so this is a time to *reach*, not to return.
        ``inf`` from any state where reaching the target is not certain. With a
        repairable "failed" state as the target, this is the MTTF from each
        state, whatever happens after failure.
        """
        P = self._matrix("expected_hitting_times")
        in_target = np.zeros(self.n_states, dtype=bool)
        in_target[self._states(target)] = True

        # Reaching the target is certain exactly from the states that cannot
        # reach a state from which the target is unreachable.
        lost = ~self._can_reach(P, in_target)
        stopped = P.copy()
        stopped[in_target] = np.eye(self.n_states)[in_target]   # paths end on entering the target
        certain = ~self._can_reach(stopped, lost)

        k = np.full(self.n_states, np.inf)
        k[in_target] = 0.0
        solve = certain & ~in_target
        if solve.any():
            A = np.eye(solve.sum()) - P[np.ix_(solve, solve)]
            k[solve] = np.linalg.solve(A, np.ones(solve.sum()))
        return k

    # ---- helpers ------------------------------------------------------------

    def _check_finite_horizon(self, horizon, t0):
        if horizon is None:
            raise ValueError("a finite horizon (number of steps) is required here")
        return check_horizon(self.transitions, horizon, t0, name="the transitions")

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
