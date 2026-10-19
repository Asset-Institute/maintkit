"""Tests for maintkit.schedule."""
import numpy as np
import pytest

from maintkit.schedule import Schedule, as_schedule, check_horizon


def _mats(k, s=2):
    """k distinct s-by-s arrays, step k filled with the value k."""
    return [np.full((s, s), float(i)) for i in range(k)]


# ---- construction ----------------------------------------------------------

def test_constant_has_period_one_and_no_length():
    sch = Schedule.constant(np.eye(3))
    assert sch.is_constant
    assert sch.period == 1
    assert sch.length is None
    assert sch.shape == (3, 3)


def test_periodic_and_finite_report_period_and_length():
    p = Schedule.periodic(_mats(3))
    f = Schedule.finite(_mats(3))
    assert (p.period, p.length) == (3, None)
    assert (f.period, f.length) == (None, 3)
    assert not p.is_constant and not f.is_constant


def test_periodic_of_one_step_is_constant():
    assert Schedule.periodic(_mats(1)).is_constant


def test_steps_are_copied_and_read_only():
    a = np.zeros((2, 2))
    sch = Schedule.constant(a)
    a[0, 0] = 99.0
    assert sch[0][0, 0] == 0.0
    with pytest.raises(ValueError):
        sch[0][0, 0] = 1.0


def test_empty_steps_rejected():
    with pytest.raises(ValueError, match="at least one step"):
        Schedule.finite(np.zeros((0, 2, 2)))


def test_mismatched_step_shapes_rejected():
    with pytest.raises(ValueError, match="same shape"):
        Schedule.periodic([np.eye(2), np.eye(3)])


def test_rewrapping_a_schedule_rejected():
    with pytest.raises(TypeError, match="already a Schedule"):
        Schedule.periodic(Schedule.constant(np.eye(2)))


# ---- indexing --------------------------------------------------------------

def test_periodic_indexing_wraps():
    sch = Schedule.periodic(_mats(3))
    for t in range(10):
        assert sch[t][0, 0] == t % 3


def test_finite_indexing_up_to_last_step_then_raises():
    sch = Schedule.finite(_mats(3))
    assert sch[2][0, 0] == 2
    with pytest.raises(IndexError, match="past the end"):
        sch[3]


def test_negative_time_rejected():
    with pytest.raises(IndexError, match=">= 0"):
        Schedule.periodic(_mats(3))[-1]


@pytest.mark.parametrize("bad", [1.0, True, "0"])
def test_non_integer_time_rejected(bad):
    with pytest.raises(TypeError):
        Schedule.constant(np.eye(2))[bad]


def test_numpy_integer_time_accepted():
    assert Schedule.periodic(_mats(3))[np.int64(4)][0, 0] == 1


def test_slice_rejected_with_pointer_to_window():
    with pytest.raises(TypeError, match="window"):
        Schedule.periodic(_mats(3))[0:2]


# ---- window and shift ------------------------------------------------------

def test_window_periodic_wraps_and_has_right_shape():
    w = Schedule.periodic(_mats(3)).window(2, 5)
    assert w.shape == (5, 2, 2)
    np.testing.assert_array_equal(w[:, 0, 0], [2, 0, 1, 2, 0])


def test_window_finite_exact_fit_and_overrun():
    sch = Schedule.finite(_mats(4))
    np.testing.assert_array_equal(sch.window(1, 3)[:, 0, 0], [1, 2, 3])
    with pytest.raises(IndexError, match="past the end"):
        sch.window(2, 3)


def test_window_of_zero_steps_is_empty():
    assert Schedule.finite(_mats(2)).window(2, 0).shape == (0, 2, 2)


@pytest.mark.parametrize("make", [Schedule.periodic, Schedule.finite])
def test_shift_matches_offset_indexing(make):
    sch = make(_mats(5))
    shifted = sch.shift(2)
    for t in range(3):
        np.testing.assert_array_equal(shifted[t], sch[t + 2])


def test_shift_periodic_beyond_one_period_and_keeps_period():
    sch = Schedule.periodic(_mats(3))
    shifted = sch.shift(7)
    assert shifted.period == 3
    for t in range(6):
        np.testing.assert_array_equal(shifted[t], sch[t + 7])


def test_shift_finite_past_end_rejected():
    with pytest.raises(IndexError, match="nothing would be left"):
        Schedule.finite(_mats(3)).shift(3)


# ---- combine ---------------------------------------------------------------

def test_combine_periodic_uses_lcm_and_aligns_steps():
    a = Schedule.periodic([np.array(i) for i in range(2)])
    b = Schedule.periodic([np.array(10 * i) for i in range(3)])
    c = Schedule.combine(lambda x, y: x + y, a, b)
    assert c.period == 6
    for t in range(12):
        assert c[t] == (t % 2) + 10 * (t % 3)


def test_combine_with_finite_takes_shortest_finite_length():
    per = Schedule.periodic([np.array(i) for i in range(2)])
    f5 = Schedule.finite([np.array(i) for i in range(5)])
    f3 = Schedule.finite([np.array(i) for i in range(3)])
    c = Schedule.combine(lambda x, y, z: x + y + z, per, f5, f3)
    assert c.length == 3
    np.testing.assert_array_equal(c.window(0, 3), [0, 3, 4])


def test_combine_treats_plain_arrays_as_constant():
    sch = Schedule.periodic(_mats(2))
    c = Schedule.combine(lambda x, y: x @ y, sch, np.eye(2))
    assert c.period == 2
    np.testing.assert_array_equal(c[1], sch[1])


def test_combine_constants_stays_constant():
    c = Schedule.combine(np.add, Schedule.constant(1.0), Schedule.constant(2.0))
    assert c.is_constant and c[5] == 3.0


def test_combine_needs_a_schedule():
    with pytest.raises(TypeError):
        Schedule.combine(np.add)


# ---- as_schedule -----------------------------------------------------------

def test_as_schedule_wraps_array_of_element_dim_as_constant():
    sch = as_schedule(np.eye(3), 2)
    assert sch.is_constant and sch.shape == (3, 3)


def test_as_schedule_passes_schedule_through():
    sch = Schedule.periodic(_mats(2))
    assert as_schedule(sch, 2) is sch


def test_as_schedule_refuses_ambiguous_stack():
    with pytest.raises(TypeError, match="periodic.*finite"):
        as_schedule(np.zeros((4, 3, 3)), 2, name="P")


def test_as_schedule_checks_schedule_step_dims():
    with pytest.raises(ValueError, match="2-dimensional"):
        as_schedule(Schedule.constant(np.zeros(3)), 2)


def test_as_schedule_rejects_wrong_ndim():
    with pytest.raises(ValueError):
        as_schedule(np.zeros(3), 2)


# ---- check_horizon ---------------------------------------------------------

@pytest.mark.parametrize("sch", [Schedule.constant(np.eye(2)), Schedule.periodic(_mats(3))])
def test_repeating_schedules_accept_any_horizon(sch):
    assert check_horizon(sch, None) == (None, 0)
    assert check_horizon(sch, 1000, t0=7) == (1000, 7)


def test_finite_schedule_refuses_infinite_horizon():
    with pytest.raises(ValueError, match="infinite horizon"):
        check_horizon(Schedule.finite(_mats(3)), None)


def test_finite_schedule_horizon_bound_including_t0():
    sch = Schedule.finite(_mats(5))
    assert check_horizon(sch, 5) == (5, 0)
    assert check_horizon(sch, 3, t0=2) == (3, 2)
    with pytest.raises(ValueError, match="only defined for t < 5"):
        check_horizon(sch, 4, t0=2)


def test_zero_horizon_allowed():
    assert check_horizon(Schedule.finite(_mats(2)), 0, t0=2) == (0, 2)


@pytest.mark.parametrize("horizon, t0, exc", [(-1, 0, IndexError), (2.0, 0, TypeError),
                                             (2, -1, IndexError)])
def test_check_horizon_rejects_bad_arguments(horizon, t0, exc):
    with pytest.raises(exc):
        check_horizon(Schedule.constant(np.eye(2)), horizon, t0)


def test_repr_names_the_kind():
    assert "constant" in repr(Schedule.constant(np.eye(2)))
    assert "period=3" in repr(Schedule.periodic(_mats(3)))
    assert "length=3" in repr(Schedule.finite(_mats(3)))
