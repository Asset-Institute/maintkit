"""Finite-state, finite-action Markov decision processes.

Conventions
-----------
* Transitions have shape ``(A, S, S)``: ``P[a, i, j] = Pr(X_{t+1} = j | X_t = i, action a)``.
  Costs have shape ``(S, A)``: the expected cost of taking action ``a`` in state
  ``i``. Costs of shape ``(A, S, S)``, a cost per transition, are converted to
  expected costs using the transition probabilities.
* Either may be constant, periodic or finite in time (see
  :class:`maintkit.time_indexed.TimeIndexed`); a plain array is constant.
* The default is to *minimise* expected total (discounted) cost; pass
  ``sense="max"`` to maximise rewards instead.
* A horizon of ``N`` from ``t0`` means decisions at ``t0, ..., t0 + N - 1`` and a
  terminal cost on the state at ``t0 + N``. Solutions and policies are indexed
  by step ``k`` relative to ``t0``: step ``k`` is time ``t0 + k``.
* A policy is a :class:`~maintkit.time_indexed.TimeIndexed` of decision rules, each an
  ``(S,)`` integer array giving the action index in every state. A plain
  ``(S,)`` array is a stationary policy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from maintkit.markov_chain import MarkovChain
from maintkit.time_indexed import TimeIndexed, _alignment, as_time_indexed, check_horizon

__all__ = ["MDP", "MDPSolution"]


@dataclass(frozen=True)
class MDPSolution:
    """Result of :meth:`MDP.solve`.

    Attributes
    ----------
    values : ndarray (N + 1, S)
        Optimal expected cost-to-go. ``values[k]`` is from step ``k`` (time
        ``t0 + k``); ``values[N]`` is the terminal cost.
    policy : TimeIndexed or None
        Finite TimeIndexed of ``N`` decision rules, ``policy[k]`` an ``(S,)`` array
        of action indices for step ``k``. None when ``N = 0`` (no decisions).
    q_values : ndarray (N, S, A)
        Cost of taking each action at each step and then acting optimally.
        Inadmissible actions are ``+inf`` when minimising (``-inf`` when
        maximising).
    horizon, t0, discount
        The problem that was solved.
    """

    values: np.ndarray
    policy: TimeIndexed
    q_values: np.ndarray
    horizon: int
    t0: int
    discount: float


class MDP:
    """A Markov decision process with finitely many states and actions.

    Parameters
    ----------
    transitions : array_like (A, S, S) or TimeIndexed
        Transition matrices, one per action.
    costs : array_like (S, A) or (A, S, S), or TimeIndexed
        Expected cost per state and action, or a cost per transition
        ``c[a, i, j]``.
    admissible : array_like of bool (S, A), or TimeIndexed, optional
        Which actions may be taken in which states. Default: all of them.
        Transition rows and costs of inadmissible pairs are ignored, so they may
        be left as zeros (or ``nan``/``inf`` for costs).
    sense : {"min", "max"}
        Minimise costs (default) or maximise rewards.
    states, actions : sequence, optional
        Labels, in array order.
    atol : float
        Tolerance on transition row sums.
    """

    def __init__(self, transitions, costs, admissible=None, *, sense="min",
                 states=None, actions=None, atol=1e-10):
        if sense not in ("min", "max"):
            raise ValueError(f"sense must be 'min' or 'max', got {sense!r}")
        self.sense = sense
        self.atol = float(atol)

        P = as_time_indexed(transitions, 3, name="transitions")
        if P.shape[1] != P.shape[2]:
            raise ValueError(f"transitions must have shape (A, S, S), got steps of shape {P.shape}")
        n_actions, n_states = P.shape[0], P.shape[1]
        if n_actions == 0 or n_states == 0:
            raise ValueError("an MDP needs at least one state and one action")
        if not np.all(np.isfinite(P.steps)) or np.any(P.steps < -self.atol):
            raise ValueError("transition probabilities must be finite and non-negative")
        self.transitions = TimeIndexed(np.clip(P.steps.astype(float), 0.0, None), repeat=P.repeat)

        self.costs = self._expected_costs(costs, n_states, n_actions)

        if admissible is None:
            admissible = np.ones((n_states, n_actions), dtype=bool)
        M = as_time_indexed(admissible, 2, name="admissible")
        if M.shape != (n_states, n_actions):
            raise ValueError(f"admissible must have shape {(n_states, n_actions)}, got {M.shape}")
        if M.dtype != bool:
            raise TypeError(f"admissible must be boolean, got dtype {M.dtype}")
        self.admissible = M

        self._validate()
        self.states = self._labels(states, n_states, "state")
        self.actions = self._labels(actions, n_actions, "action")

    # ---- construction helpers -----------------------------------------------

    def _expected_costs(self, costs, n_states, n_actions):
        if isinstance(costs, TimeIndexed):
            ndim = len(costs.shape)
        else:
            ndim = np.ndim(costs)
        if ndim == 2:
            C = as_time_indexed(costs, 2, name="costs")
            if C.shape != (n_states, n_actions):
                raise ValueError(f"costs must have shape {(n_states, n_actions)}, got {C.shape}")
            return TimeIndexed(C.steps.astype(float), repeat=C.repeat)
        if ndim == 3:
            C = as_time_indexed(costs, 3, name="costs")
            if C.shape != (n_actions, n_states, n_states):
                raise ValueError(
                    f"per-transition costs must have shape {(n_actions, n_states, n_states)}, "
                    f"got {C.shape}"
                )
            # E[c | i, a] = sum_j P[a, i, j] c[a, i, j]. Zero-probability
            # transitions contribute nothing even if their cost is inf or nan.
            def expected(P, c):
                with np.errstate(invalid="ignore"):
                    return np.where(P > 0, P * c, 0.0).sum(axis=2).T
            return TimeIndexed.map(expected, self.transitions, C)
        raise ValueError(
            f"costs must be (S, A) expected costs or (A, S, S) per-transition costs, got {ndim} dimensions"
        )

    def _validate(self):
        """Admissible rows must be stochastic with finite costs, and no state may be left without an action."""
        n, _ = _alignment([self.transitions, self.costs, self.admissible])
        for t in range(n):
            P, C, M = self.transitions[t], self.costs[t], self.admissible[t]
            empty = ~M.any(axis=1)
            if empty.any():
                raise ValueError(f"state {np.flatnonzero(empty)[0]} has no admissible action at step {t}")
            for a in range(P.shape[0]):
                rows = M[:, a]
                sums = P[a].sum(axis=1)
                bad = rows & (np.abs(sums - 1.0) > self.atol)
                if bad.any():
                    i = np.flatnonzero(bad)[0]
                    raise ValueError(
                        f"action {a}, state {i} at step {t}: transition row sums to "
                        f"{sums[i]:.12g}, not 1 (atol={self.atol:g})"
                    )
                bad = rows & ~np.isfinite(C[:, a])
                if bad.any():
                    raise ValueError(
                        f"cost of action {a} in state {np.flatnonzero(bad)[0]} at step {t} is not "
                        "finite, but the action is admissible there"
                    )

    @staticmethod
    def _labels(labels, n, what):
        if labels is None:
            return tuple(range(n))
        labels = tuple(labels)
        if len(labels) != n:
            raise ValueError(f"got {len(labels)} {what} labels for {n} {what}s")
        if len(set(labels)) != n:
            raise ValueError(f"{what} labels must be unique")
        return labels

    # ---- properties ---------------------------------------------------------

    @property
    def n_states(self):
        return self.transitions.shape[1]

    @property
    def n_actions(self):
        return self.transitions.shape[0]

    @property
    def length(self):
        """Number of time steps the model is defined for, None if unbounded."""
        n, repeat = _alignment([self.transitions, self.costs, self.admissible])
        return None if repeat else n

    @property
    def period(self):
        """Joint period of transitions, costs and admissibility; None if finite."""
        n, repeat = _alignment([self.transitions, self.costs, self.admissible])
        return n if repeat else None

    def _check(self, horizon, t0):
        if horizon is None:
            raise ValueError(
                "a finite horizon (number of decision steps) is required; "
                "infinite-horizon solving is not implemented yet"
            )
        for sched, name in ((self.transitions, "transitions"), (self.costs, "costs"),
                            (self.admissible, "admissible")):
            horizon, t0 = check_horizon(sched, horizon, t0, name=name)
        return horizon, t0

    def _terminal(self, terminal):
        if terminal is None:
            return np.zeros(self.n_states)
        terminal = np.asarray(terminal, dtype=float)
        if terminal.shape != (self.n_states,):
            raise ValueError(f"terminal must have shape ({self.n_states},), got {terminal.shape}")
        if not np.all(np.isfinite(terminal)):
            raise ValueError("terminal costs must be finite")
        return terminal

    @staticmethod
    def _discount(discount):
        discount = float(discount)
        if not 0.0 <= discount <= 1.0:
            raise ValueError(f"discount must be in [0, 1], got {discount}")
        return discount

    # ---- solving ------------------------------------------------------------

    def solve(self, horizon, discount=1.0, terminal=None, t0=0):
        """Optimal policy over a finite horizon, by backward induction.

        Parameters
        ----------
        horizon : int
            Number of decision steps ``N``.
        discount : float
            Discount factor in ``[0, 1]``; 1 gives the undiscounted total.
        terminal : array_like (S,), optional
            Cost (reward, when maximising) of the final state. Default zero.
        t0 : int
            Time of the first decision, which selects the phase of a periodic model.

        Returns
        -------
        MDPSolution

        Ties between actions with exactly equal value go to the lowest action index.
        """
        horizon, t0 = self._check(horizon, t0)
        discount = self._discount(discount)
        S, A = self.n_states, self.n_actions
        worst = np.inf if self.sense == "min" else -np.inf
        pick = np.argmin if self.sense == "min" else np.argmax

        values = np.empty((horizon + 1, S))
        values[horizon] = self._terminal(terminal)
        q_values = np.empty((horizon, S, A))
        rules = np.empty((horizon, S), dtype=int)
        rows = np.arange(S)

        for k in range(horizon - 1, -1, -1):
            t = t0 + k
            M = self.admissible[t]
            future = self.transitions[t] @ values[k + 1]          # (A, S)
            Q = np.where(M, self.costs[t] + discount * future.T, worst)
            rules[k] = pick(Q, axis=1)
            values[k] = Q[rows, rules[k]]
            q_values[k] = Q

        policy = TimeIndexed.finite(rules) if horizon > 0 else None
        return MDPSolution(values, policy, q_values, horizon, t0, discount)

    # ---- policies -----------------------------------------------------------

    def _policy(self, policy):
        sched = as_time_indexed(policy, 1, name="policy")
        if sched.shape != (self.n_states,):
            raise ValueError(f"decision rules must have shape ({self.n_states},), got {sched.shape}")
        if not np.issubdtype(sched.dtype, np.integer):
            raise TypeError(f"decision rules must hold integer action indices, got dtype {sched.dtype}")
        if np.any(sched.steps < 0) or np.any(sched.steps >= self.n_actions):
            raise ValueError(f"action indices must be in 0..{self.n_actions - 1}")
        return sched

    def _check_admissible(self, policy, n_steps, t0):
        """Every action the policy takes in steps ``0..n_steps-1`` must be admissible."""
        rows = np.arange(self.n_states)
        for k in range(n_steps):
            d = policy[k]
            ok = self.admissible[t0 + k][rows, d]
            if not ok.all():
                i = np.flatnonzero(~ok)[0]
                raise ValueError(
                    f"the policy takes action {d[i]} in state {i} at step {k} "
                    f"(time {t0 + k}), where it is not admissible"
                )

    def evaluate(self, policy, horizon, discount=1.0, terminal=None, t0=0):
        """Expected cost-to-go of a given policy over a finite horizon.

        ``policy[k]`` is the decision rule at step ``k`` (time ``t0 + k``).
        Returns an ``(N + 1, S)`` array laid out like ``MDPSolution.values``.
        """
        horizon, t0 = self._check(horizon, t0)
        policy = self._policy(policy)
        check_horizon(policy, horizon, 0, name="policy")
        self._check_admissible(policy, horizon, t0)
        discount = self._discount(discount)

        rows = np.arange(self.n_states)
        values = np.empty((horizon + 1, self.n_states))
        values[horizon] = self._terminal(terminal)
        for k in range(horizon - 1, -1, -1):
            t, d = t0 + k, policy[k]
            P_d = self.transitions[t][d, rows, :]
            c_d = self.costs[t][rows, d]
            values[k] = c_d + discount * P_d @ values[k + 1]
        return values

    def induced_chain(self, policy, t0=0):
        """The Markov chain obtained by following ``policy`` from time ``t0``.

        Row ``i`` of the chain's transition matrix at its step ``k`` is row
        ``i`` of ``P[t0 + k][policy[k][i]]``. The chain's time 0 is the model's
        time ``t0``. Periods and lengths combine as in
        :meth:`TimeIndexed.map`, so a finite policy gives a finite chain.
        """
        policy = self._policy(policy)
        P = self.transitions.shift(t0)
        n, _ = _alignment([P, self.admissible.shift(t0), policy])
        self._check_admissible(policy, n, t0)

        rows = np.arange(self.n_states)
        select_rows = lambda Pt, d: Pt[d, rows, :]       # row i of the matrix for action d[i]
        chain_P = TimeIndexed.map(select_rows, P, policy)
        return MarkovChain(chain_P, states=self.states, atol=self.atol)

    def policy_costs(self, policy, t0=0):
        """TimeIndexed of ``(S,)`` expected one-step costs under ``policy`` from ``t0``."""
        policy = self._policy(policy)
        rows = np.arange(self.n_states)
        select_costs = lambda C, d: C[rows, d]            # C[i, d[i]] for each state i
        return TimeIndexed.map(select_costs, self.costs.shift(t0), policy)

    def __repr__(self):
        return (f"MDP(n_states={self.n_states}, n_actions={self.n_actions}, sense={self.sense!r}, "
                f"transitions={self.transitions!r})")
