#!/usr/bin/env python3
"""
Generate the cross-check reference vectors for the C++ control/risk ports.

Python (colav_automaton) is the source of truth. This writes CSVs of random
inputs and the Python outputs into test/; each C++ gtest recomputes every
row and asserts it matches. Regenerate after any change to the ported Python
functions:

    PYTHONPATH=<repo>/src python3 gen_reference.py     # writes ../test/*.csv

repr() round-trips a double exactly, so the C++ side reads the identical bit
pattern each value was generated from.
"""

import math
import os
import random
import sys

repo = os.environ.get("COLAV_REPO")
if repo:
    sys.path.insert(0, os.path.join(repo, "src"))
from colav_automaton.controllers.prescribed_time import (  # noqa: E402
    compute_prescribed_time_control,
)
from colav_automaton.guards.conditions import (  # noqa: E402
    L1_check, L2_check, compute_risk_index, G22_check,
)

TEST_DIR = os.path.join(os.path.dirname(__file__), "..", "test")

# Python defaults of compute_risk_index, fixed so the C++ RiskBetas default
# is exercised directly.
BETAS = dict(dcpa_beta1=463.0, dcpa_beta2=926.0,
             tcpa_beta1=120.0, tcpa_beta2=240.0,
             dist_beta1=148.0, dist_beta2=463.0)


def _row(vals):
    return ",".join(repr(c) for c in vals)


def gen_control(path):
    rng = random.Random(20240612)
    cols = ["t", "x", "y", "psi", "xw", "yw", "a", "v", "eta", "tp",
            "v1x", "v1y", "delta", "u_py", "l1_py", "l2_py"]
    with open(path, "w") as out:
        out.write(",".join(cols) + "\n")
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
            out.write(_row([t, x, y, psi, xw, yw, a, v, eta, tp, v1x, v1y,
                            delta]) + f",{u!r},{int(l1)},{int(l2)}\n")


def gen_risk(path):
    rng = random.Random(99887766)
    cols = ["px", "py", "psi", "v", "ox", "oy", "ov", "opsi",
            "K", "ri_py", "g22_py"]
    with open(path, "w") as out:
        out.write(",".join(cols) + "\n")
        for _ in range(1000):
            px = rng.uniform(-500.0, 500.0)
            py = rng.uniform(-500.0, 500.0)
            psi = rng.uniform(-math.pi, math.pi)
            v = rng.uniform(1.0, 15.0)
            # Obstacles spread across approaching / receding / near geometry
            # so DCPA/TCPA/F exercise every branch.
            ox = px + rng.uniform(-800.0, 800.0)
            oy = py + rng.uniform(-800.0, 800.0)
            ov = rng.uniform(0.0, 12.0)
            opsi = rng.uniform(-math.pi, math.pi)
            K = rng.uniform(0.1, 0.6)
            obstacles = [(ox, oy, ov, opsi)]
            ri = compute_risk_index(px, py, psi, obstacles, v, 300.0, **BETAS)
            g22 = G22_check(px, py, psi, obstacles, v, 300.0, K=K, **BETAS)
            out.write(_row([px, py, psi, v, ox, oy, ov, opsi, K])
                      + f",{ri!r},{int(g22)}\n")


def main():
    os.makedirs(TEST_DIR, exist_ok=True)
    gen_control(os.path.join(TEST_DIR, "reference_control.csv"))
    gen_risk(os.path.join(TEST_DIR, "reference_risk.csv"))
    print("wrote reference_control.csv, reference_risk.csv to", TEST_DIR)


if __name__ == "__main__":
    main()
