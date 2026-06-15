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
    L1_check, L2_check, compute_risk_index, G22_check, G11_check, G23_check,
)
from colav_automaton.controllers import (  # noqa: E402
    compute_v1, get_unsafe_set_vertices, default_vertex_provider,
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


def gen_geom(path):
    rng = random.Random(13572468)
    cols = ["px", "py", "psi", "xw", "yw", "v", "tp", "Cs",
            "ox", "oy", "ov", "opsi", "g11_py", "g23_py"]
    with open(path, "w") as out:
        out.write(",".join(cols) + "\n")
        for _ in range(1500):
            # CommonOcean-ish scale; obstacle scattered near the LOS so the
            # guard booleans come out a balanced mix of True/False.
            px = rng.uniform(-100.0, 100.0)
            py = rng.uniform(-100.0, 100.0)
            psi = rng.uniform(-math.pi, math.pi)
            reach = rng.uniform(1500.0, 4000.0)
            xw = px + reach * math.cos(psi)
            yw = py + reach * math.sin(psi)
            v = rng.uniform(4.0, 12.0)
            tp = rng.uniform(1.0, 3.0)
            Cs = rng.uniform(150.0, 400.0)
            # place the obstacle somewhere between ego and waypoint, with
            # lateral offset on the order of a few Cs (sometimes blocking).
            frac = rng.uniform(0.1, 0.9)
            lat = rng.uniform(-3.0, 3.0) * Cs
            mx = px + frac * (xw - px)
            my = py + frac * (yw - py)
            ox = mx - lat * math.sin(psi)
            oy = my + lat * math.cos(psi)
            ov = rng.uniform(0.0, 10.0)
            opsi = rng.uniform(-math.pi, math.pi)

            obstacles = [(ox, oy, ov, opsi)]
            g11 = G11_check(px, py, psi, xw, yw, v, tp, obstacles, Cs)
            g23 = G23_check(px, py, psi, xw, yw, v, tp, obstacles, Cs)
            out.write(_row([px, py, psi, xw, yw, v, tp, Cs, ox, oy, ov, opsi])
                      + f",{int(g11)},{int(g23)}\n")


def gen_v1(path):
    rng = random.Random(24681357)
    cols = ["px", "py", "psi", "v", "tp", "Cs",
            "ox", "oy", "ov", "opsi", "has_v1", "v1x", "v1y"]
    with open(path, "w") as out:
        out.write(",".join(cols) + "\n")
        for _ in range(1500):
            px = rng.uniform(-100.0, 100.0)
            py = rng.uniform(-100.0, 100.0)
            psi = rng.uniform(-math.pi, math.pi)
            v = rng.uniform(4.0, 12.0)
            tp = rng.uniform(1.0, 3.0)
            Cs = rng.uniform(150.0, 400.0)
            # obstacle ahead-ish so a V1 vertex exists most of the time
            ahead = rng.uniform(400.0, 2500.0)
            lat = rng.uniform(-1.5, 1.5) * Cs
            ox = px + ahead * math.cos(psi) - lat * math.sin(psi)
            oy = py + ahead * math.sin(psi) + lat * math.cos(psi)
            ov = rng.uniform(0.0, 10.0)
            opsi = rng.uniform(-math.pi, math.pi)

            # Exactly as reset_enter_avoidance constructs it.
            v1_Cs = Cs + 0.25 * Cs
            dsafe = Cs + v * tp
            max_horizon = max(60.0, 3.0 * dsafe / v)
            obstacles = [(ox, oy, ov, opsi)]

            def vertex_provider(qx, qy, obs, cs, heading,
                                _dsafe=dsafe, _v=v, _mh=max_horizon, _c=v1_Cs):
                verts = get_unsafe_set_vertices(
                    qx, qy, obs, _c, dsf=_dsafe, ship_psi=heading, ship_v=_v,
                    use_swept_region=True, max_horizon=_mh)
                if verts is not None:
                    return verts
                return default_vertex_provider(qx, qy, obs, _c, heading)

            v1 = compute_v1(px, py, psi, obstacles, v1_Cs,
                            vertex_provider, 0.0, v=v)
            if v1 is None:
                out.write(_row([px, py, psi, v, tp, Cs, ox, oy, ov, opsi])
                          + ",0,0.0,0.0\n")
            else:
                out.write(_row([px, py, psi, v, tp, Cs, ox, oy, ov, opsi])
                          + f",1,{v1[0]!r},{v1[1]!r}\n")


def main():
    os.makedirs(TEST_DIR, exist_ok=True)
    gen_control(os.path.join(TEST_DIR, "reference_control.csv"))
    gen_risk(os.path.join(TEST_DIR, "reference_risk.csv"))
    gen_geom(os.path.join(TEST_DIR, "reference_geom.csv"))
    gen_v1(os.path.join(TEST_DIR, "reference_v1.csv"))
    print("wrote reference_{control,risk,geom,v1}.csv to", TEST_DIR)


if __name__ == "__main__":
    main()
