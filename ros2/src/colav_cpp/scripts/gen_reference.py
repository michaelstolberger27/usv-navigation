#!/usr/bin/env python3
"""
Generate the cross-check reference vectors for the C++ control port.

Python (colav_automaton) is the source of truth. This writes a CSV of
random inputs and the Python outputs; the C++ gtest recomputes each row
and asserts it matches (see test/test_control.cpp). Regenerate after any
change to the Python control law:

    PYTHONPATH=<repo>/src python3 gen_reference.py > test/reference_control.csv
"""

import math
import os
import random
import sys

# Import the Python reference. PYTHONPATH=<repo>/src, or set COLAV_REPO.
repo = os.environ.get("COLAV_REPO")
if repo:
    sys.path.insert(0, os.path.join(repo, "src"))
from colav_automaton.controllers.prescribed_time import (  # noqa: E402
    compute_prescribed_time_control,
)
from colav_automaton.guards.conditions import L1_check, L2_check  # noqa: E402

COLUMNS = [
    "t", "x", "y", "psi", "xw", "yw", "a", "v", "eta", "tp",
    "v1x", "v1y", "delta", "u_py", "l1_py", "l2_py",
]


def main():
    rng = random.Random(20240612)
    out = sys.stdout
    out.write(",".join(COLUMNS) + "\n")
    for _ in range(1000):
        t = rng.uniform(0.0, 5.0)
        x = rng.uniform(-200.0, 200.0)
        y = rng.uniform(-200.0, 200.0)
        psi = rng.uniform(-math.pi, math.pi)
        xw = rng.uniform(-200.0, 200.0)
        yw = rng.uniform(-200.0, 200.0)
        a = rng.uniform(0.5, 3.0)
        v = rng.uniform(1.0, 15.0)
        eta = rng.uniform(1.5, 5.0)
        tp = rng.uniform(0.5, 60.0)
        v1x = rng.uniform(-200.0, 200.0)
        v1y = rng.uniform(-200.0, 200.0)
        delta = rng.uniform(0.5, 50.0)

        u = compute_prescribed_time_control(
            t, x, y, psi, xw, yw, a=a, v=v, eta=eta, tp=tp)
        l1 = L1_check(x, y, v1x, v1y, delta)
        l2 = L2_check(x, y, psi, v1x, v1y)

        vals = [t, x, y, psi, xw, yw, a, v, eta, tp, v1x, v1y, delta]
        # repr() round-trips a double exactly, so the C++ side reads the
        # identical bit pattern it was generated from.
        out.write(",".join(repr(c) for c in vals)
                  + f",{u!r},{int(l1)},{int(l2)}\n")


if __name__ == "__main__":
    main()
