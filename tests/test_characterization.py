"""P0 characterisation tests.

These assert nothing about whether the current behaviour is *correct* -- only
that it has not changed. They are the safety net for the P2-P5 migrations in
REFACTOR_PLAN.md, which move every fitter onto the shared machinery in
``maintkit.inference``.

Bootstrapping: run ``python tests/reference/generate_reference.py`` once on the
current code and commit the resulting JSON. Until that file exists these tests
skip with an explanatory message rather than failing.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from tests._cases import CASES, run_case

REFERENCE_PATH = pathlib.Path(__file__).parent / "reference" / "reference_values.json"

# Numerical Hessians vary slightly across BLAS/scipy builds, so compare with a
# tolerance rather than exactly. Tight enough to catch a real change in method,
# loose enough to survive a library upgrade.
RTOL = 1e-6
ATOL = 1e-9


def _load_reference():
    if not REFERENCE_PATH.exists():
        pytest.skip(
            f"No reference values at {REFERENCE_PATH}. Bootstrap them with:\n"
            "    python tests/reference/generate_reference.py"
        )
    return json.loads(REFERENCE_PATH.read_text())


@pytest.fixture(scope="module")
def reference():
    return _load_reference()


@pytest.mark.parametrize("name", sorted(CASES))
def test_fitter_output_unchanged(name, reference):
    expected = reference["cases"].get(name)
    if expected is None:
        pytest.fail(
            f"Case {name!r} has no reference entry. Re-record with:\n"
            "    python tests/reference/generate_reference.py --force"
        )

    actual = run_case(name)

    assert actual["status"] == expected["status"], (
        f"{name}: status changed {expected['status']!r} -> {actual['status']!r}. "
        f"now: {actual.get('type', '')} {actual.get('message', '')}"
    )

    if expected["status"] == "error":
        assert actual["type"] == expected["type"], (
            f"{name}: exception type changed "
            f"{expected['type']} -> {actual['type']}"
        )
        return

    assert actual["n_outputs"] == expected["n_outputs"], (
        f"{name}: fitter arity changed "
        f"{expected['n_outputs']} -> {actual['n_outputs']}. If intentional "
        "(e.g. now returning FitResult), re-record the reference values."
    )

    for i, (got, want) in enumerate(zip(actual["outputs"], expected["outputs"])):
        if want is None:
            assert got is None, f"{name}: output {i} was None, now {got!r}"
            continue
        got_a, want_a = np.asarray(got, float), np.asarray(want, float)
        assert got_a.shape == want_a.shape, (
            f"{name}: output {i} shape changed {want_a.shape} -> {got_a.shape}"
        )
        np.testing.assert_allclose(
            got_a, want_a, rtol=RTOL, atol=ATOL,
            err_msg=f"{name}: output {i} changed numerically",
        )


def test_reference_file_covers_every_case(reference):
    """A new fitter must be recorded before it can drift unnoticed."""
    missing = set(CASES) - set(reference["cases"])
    assert not missing, (
        f"Cases with no reference values: {sorted(missing)}. "
        "Re-record with: python tests/reference/generate_reference.py --force"
    )


def test_reference_metadata_present(reference):
    assert "_meta" in reference
    for key in ("maintkit", "numpy", "scipy", "python"):
        assert key in reference["_meta"]
