"""Record current fitter behaviour as reference values.

Run this ONCE, on the current code, before starting the P2-P5 migrations:

    python tests/reference/generate_reference.py

It writes ``tests/reference/reference_values.json``. Commit that file. From then on
``test_characterization.py`` compares every fitter against it, so any migration
that changes a number is caught immediately.

Re-run with --force only when a change is intended. Review the resulting diff
carefully; that diff *is* the behavioural change.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

# make the repo root importable so `tests` is a package
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests._cases import CASES, run_case  # noqa: E402

REFERENCE_PATH = pathlib.Path(__file__).with_name("reference_values.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite an existing reference_values.json",
    )
    args = parser.parse_args()

    if REFERENCE_PATH.exists() and not args.force:
        print(f"{REFERENCE_PATH} already exists. Re-run with --force to overwrite.")
        return 1

    import numpy, scipy, maintkit

    record = {
        "_meta": {
            "maintkit": getattr(maintkit, "__version__", "unknown"),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "python": sys.version.split()[0],
        },
        "cases": {},
    }

    width = max(len(k) for k in CASES)
    for name in CASES:
        result = run_case(name)
        record["cases"][name] = result
        if result["status"] == "ok":
            first = result["outputs"][0]
            preview = numpy.asarray(first).ravel()[:3] if first is not None else "None"
            print(f"  {name:<{width}}  ok     {preview}")
        else:
            print(f"  {name:<{width}}  ERROR  {result['type']}: {result['message'][:60]}")

    REFERENCE_PATH.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    failed = [k for k, v in record["cases"].items() if v["status"] != "ok"]
    n_ok = len(CASES) - len(failed)
    print(f"\nWrote {REFERENCE_PATH} ({n_ok}/{len(CASES)} cases fitted cleanly).")

    if failed:
        print()
        print("!" * 70)
        print("NOT A CLEAN RECORDING -- exiting non-zero.")
        print(f"Errored and recorded as failures: {failed}")
        print()
        print("Recording an error is only correct if the fitter is genuinely")
        print("expected to raise on this input. Otherwise the harness is calling")
        print("it wrongly, and the baseline now pins a broken call rather than")
        print("the behaviour you meant to capture. Fix and re-run.")
        print("!" * 70)
        return 1

    print("Commit this file alongside the change that made it necessary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
