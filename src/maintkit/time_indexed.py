"""Time-indexed sequences for discrete-time models.

A :class:`TimeIndexed` holds a quantity that may change from one time step to the
next -- a transition matrix, a cost table, a decision rule -- and answers
``x[t]`` for integer ``t >= 0``. It covers three cases with one stack of
arrays and one flag:

* constant: one step that applies at every ``t``;
* periodic: ``K`` steps repeated, ``x[t] = steps[t % K]``;
* finite: ``K`` steps defined for ``t < K`` only, an error after that.

A sequence records only what the quantity is at each step. How many steps a
problem runs for -- its horizon -- belongs to the solver or analysis call, and
:func:`check_horizon` is the single place that reconciles the two.

Code that consumes these sequences should go through ``x[t]``,
:meth:`TimeIndexed.window`, :attr:`TimeIndexed.length` and :attr:`TimeIndexed.period`
rather than reading ``steps`` directly, so the representation can be extended
later (for example with a non-repeating prefix) without changing callers.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

import numpy as np

__all__ = ["TimeIndexed", "as_time_indexed", "check_horizon"]


def _check_time(t, name="t"):
    """Return ``t`` as a non-negative Python int, or raise."""
    if isinstance(t, (bool, np.bool_)) or not isinstance(t, numbers.Integral):
        raise TypeError(f"{name} must be an integer time step, got {type(t).__name__}")
    t = int(t)
    if t < 0:
        raise IndexError(f"{name} must be >= 0, got {t}; time-indexed sequences start at t = 0")
    return t


@dataclass(frozen=True, eq=False)
class TimeIndexed:
    """A quantity indexed by integer time step.

    Parameters
    ----------
    steps : array_like, shape (K, ...)
        The ``K`` per-step values, stacked along the first axis. Each element
        ``steps[k]`` has the same shape, :attr:`shape`.
    repeat : bool
        True repeats the steps forever (``x[t] = steps[t % K]``), False
        defines the sequence for ``t < K`` only.

    The constructors :meth:`constant`, :meth:`periodic` and :meth:`finite`
    are the usual way to build one. The steps are copied and made read-only,
    so it cannot change after it is built.

    Examples
    --------
    >>> P = TimeIndexed.periodic([P_summer, P_winter])   # doctest: +SKIP
    >>> P[0] is P[2]                                   # doctest: +SKIP
    """

    steps: np.ndarray
    repeat: bool

    def __post_init__(self):
        steps = np.array(self.steps)          # copies, so later edits to the input cannot leak in
        if steps.ndim == 0 or steps.shape[0] == 0:
            raise ValueError(
                "steps must have at least one step along the first axis, "
                f"got an array of shape {steps.shape}"
            )
        steps.setflags(write=False)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "repeat", bool(self.repeat))

    # ---- constructors -------------------------------------------------------

    @classmethod
    def constant(cls, value):
        """The same ``value`` at every time step."""
        return cls(np.asarray(value)[np.newaxis], repeat=True)

    @classmethod
    def periodic(cls, values):
        """``values[0], values[1], ..., values[K-1]``, then repeat from the start."""
        return cls(_stack(values), repeat=True)

    @classmethod
    def finite(cls, values):
        """``values[t]`` for ``t < len(values)``; undefined after that."""
        return cls(_stack(values), repeat=False)

    # ---- properties ---------------------------------------------------------

    @property
    def n_steps(self):
        """Number of stored steps, ``K``."""
        return self.steps.shape[0]

    @property
    def length(self):
        """Number of time steps the sequence is defined for, or None if unbounded."""
        return None if self.repeat else self.n_steps

    @property
    def period(self):
        """Repetition period ``K``, or None for a finite sequence."""
        return self.n_steps if self.repeat else None

    @property
    def is_constant(self):
        """True when one value applies at every time step."""
        return self.repeat and self.n_steps == 1

    @property
    def shape(self):
        """Shape of the value at a single time step."""
        return self.steps.shape[1:]

    @property
    def dtype(self):
        return self.steps.dtype

    # ---- access -------------------------------------------------------------

    def _index(self, t):
        if self.repeat:
            return t % self.n_steps
        if t >= self.n_steps:
            raise IndexError(
                f"t = {t} is past the end of a finite sequence, defined for "
                f"t < {self.n_steps}"
            )
        return t

    def __getitem__(self, t):
        """Value at time step ``t`` (a read-only view)."""
        if isinstance(t, slice):
            raise TypeError(
                "time-indexed sequences are indexed by a single time step; use "
                "window(t0, n) for a range of steps"
            )
        return self.steps[self._index(_check_time(t))]

    def window(self, t0, n):
        """Values for time steps ``t0, ..., t0 + n - 1`` as an ``(n, *shape)`` array.

        Raises an error if any of those steps is past the end of a finite sequence.
        """
        t0 = _check_time(t0, "t0")
        n = _check_time(n, "n")
        if not self.repeat and t0 + n > self.n_steps:
            raise IndexError(
                f"steps {t0}..{t0 + n - 1} run past the end of a finite sequence, "
                f"defined for t < {self.n_steps}"
            )
        idx = np.arange(t0, t0 + n)
        if self.repeat:
            idx %= self.n_steps
        return self.steps[idx]

    def shift(self, t0):
        """The sequence seen from ``t0``: ``shift(t0)[t] == self[t0 + t]``.

        A periodic sequence keeps its period with its steps rotated; a finite
        one loses its first ``t0`` steps.
        """
        t0 = _check_time(t0, "t0")
        if self.repeat:
            return TimeIndexed(np.roll(self.steps, -(t0 % self.n_steps), axis=0), repeat=True)
        if t0 >= self.n_steps:
            raise IndexError(
                f"cannot shift a finite sequence of length {self.n_steps} by {t0}: "
                "nothing would be left"
            )
        return TimeIndexed(self.steps[t0:], repeat=False)

    # ---- composition --------------------------------------------------------

    @staticmethod
    def combine(f, *seqs):
        """Apply ``f`` step by step: ``out[t] = f(s1[t], s2[t], ...)``.

        Arguments that are not TimeIndexed are treated as constants. The result
        repeats if every input repeats, with period the least common multiple
        of the input periods. If any input is finite, the result is finite with
        the length of the shortest finite input: it is defined exactly where
        every input is.
        """
        if not seqs:
            raise TypeError("combine needs at least one sequence")
        seqs = [s if isinstance(s, TimeIndexed) else TimeIndexed.constant(s) for s in seqs]
        n, repeat = _alignment(seqs)
        return TimeIndexed(np.stack([np.asarray(f(*(s[t] for s in seqs))) for t in range(n)]),
                        repeat=repeat)

    # ---- display ------------------------------------------------------------

    def __repr__(self):
        if self.is_constant:
            kind = "constant"
        elif self.repeat:
            kind = f"periodic, period={self.n_steps}"
        else:
            kind = f"finite, length={self.n_steps}"
        return f"TimeIndexed({kind}, shape={self.shape}, dtype={self.dtype})"


def _alignment(seqs):
    """Steps needed to cover every distinct combination of ``seqs``, and whether it repeats.

    All repeating: the least common multiple of the periods, repeating. Any
    finite: the shortest finite length, not repeating.
    """
    finite_lengths = [s.length for s in seqs if not s.repeat]
    if finite_lengths:
        return min(finite_lengths), False
    n = 1
    for s in seqs:
        n = n * s.period // math.gcd(n, s.period)
    return n, True


def _stack(values):
    if isinstance(values, TimeIndexed):
        raise TypeError("already a TimeIndexed; pass it directly instead of re-wrapping it")
    if isinstance(values, np.ndarray):
        return values
    try:
        return np.stack([np.asarray(v) for v in values])
    except ValueError as exc:
        raise ValueError(f"every step must have the same shape: {exc}") from exc


def as_time_indexed(value, element_ndim, *, name="value"):
    """Coerce ``value`` to a :class:`TimeIndexed` whose steps have ``element_ndim`` dims.

    A TimeIndexed is returned unchanged (after checking its step dimension). A
    plain array with ``element_ndim`` dimensions becomes a constant sequence.
    Anything with an extra leading axis is refused: a stack of ``K`` arrays
    could be one period or a whole finite horizon, so the caller has to say
    which with :meth:`TimeIndexed.periodic` or :meth:`TimeIndexed.finite`.
    """
    if isinstance(value, TimeIndexed):
        if len(value.shape) != element_ndim:
            raise ValueError(
                f"{name} must have {element_ndim}-dimensional steps, got steps "
                f"of shape {value.shape}"
            )
        return value
    arr = np.asarray(value)
    if arr.ndim == element_ndim:
        return TimeIndexed.constant(arr)
    if arr.ndim == element_ndim + 1:
        raise TypeError(
            f"{name} has shape {arr.shape}, a stack of {arr.shape[0]} steps. Wrap "
            "it as TimeIndexed.periodic(...) or TimeIndexed.finite(...) to say whether "
            "it repeats or ends."
        )
    raise ValueError(
        f"{name} must be a {element_ndim}-dimensional array or a TimeIndexed, "
        f"got shape {arr.shape}"
    )


def check_horizon(seq, horizon, t0=0, *, name="sequence"):
    """Check that ``seq`` is defined for every step a problem will use.

    ``horizon`` is a number of steps, using ``seq[t0], ...,
    seq[t0 + horizon - 1]``, or None for an infinite horizon. Returns the
    horizon and start time as ints (horizon may be None).
    """
    t0 = _check_time(t0, "t0")
    if horizon is not None:
        horizon = _check_time(horizon, "horizon")

    length = seq.length
    if length is None:
        return horizon, t0
    if horizon is None:
        raise ValueError(
            f"an infinite horizon was requested, but {name} is finite and only "
            f"defined for t < {length}"
        )
    if t0 + horizon > length:
        raise ValueError(
            f"a horizon of {horizon} from t0 = {t0} needs {name} up to "
            f"t = {t0 + horizon - 1}, but it is only defined for t < {length}"
        )
    return horizon, t0
