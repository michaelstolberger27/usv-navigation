"""
Runtime compatibility normalization for the colav_unsafe_set dependency.

Builds of colav_unsafe_set in circulation differ in a sign convention
inside calc_cpa: whether the closest-point-of-approach vector advances
the relative position by tcpa (giving DCPA = 0 on a collision course)
or applies the opposite sign (giving DCPA of roughly twice the range in
the same geometry). Everything validated in this repository — the
CommonOcean batch results, the behavioural regression suite, and the
C++ cross-checks — assumes the advancing convention, and the risk-index
thresholds are tuned to it; under the other convention DCPA values are
inflated and avoidance decisions shift substantially.

On import this module probes the installed build with a known closing
geometry and, only if it follows the other convention, rebinds calc_cpa
in every upstream namespace that holds a reference to the version
matching the validated behaviour. Against a build that already matches,
the probe passes and nothing is touched.
"""

from typing import Tuple

import numpy as np

import colav_unsafe_set.risk_assessment as _ra
import colav_unsafe_set.risk_assessment.obstacle_metric_calculator as _omc
import colav_unsafe_set.risk_assessment.risk_assessment as _ra_impl
from colav_unsafe_set.objects import Agent, DynamicObstacle

# Re-use the package's own helpers so the normalized function tracks its
# conventions (and stays bit-identical to a matching build).
_quaternion_to_heading = _ra_impl.quaternion_to_heading
_normalize_angle = _ra_impl.normalize_angle


def _reference_calc_cpa(agent_object: Agent,
                        target_object: DynamicObstacle) -> Tuple[float, float]:
    """calc_cpa in the validated convention: p_rel advanced by tcpa."""
    p1 = np.array(agent_object.position[:2])
    p2 = np.array(target_object.position[:2])

    theta1 = _normalize_angle(_quaternion_to_heading(*agent_object.orientation))
    theta2 = _normalize_angle(_quaternion_to_heading(*target_object.orientation))

    v1 = agent_object.velocity * np.array([np.cos(theta1), np.sin(theta1)])
    v2 = target_object.velocity * np.array([np.cos(theta2), np.sin(theta2)])

    p_rel = p1 - p2
    v_rel = v1 - v2
    v_rel_norm_sq = np.dot(v_rel, v_rel)

    if v_rel_norm_sq < 1e-6:
        if np.allclose(p_rel, [0, 0]):
            dcpa = float('nan')
            tcpa = float('inf')
        else:
            distance = np.linalg.norm(p_rel)
            speed = np.linalg.norm(v1)
            dcpa = distance
            tcpa = distance / speed if speed > 0 else float('inf')
    else:
        tcpa = -np.dot(p_rel, v_rel) / v_rel_norm_sq
        if tcpa > 0:
            cpa_vector = p_rel + tcpa * v_rel
            dcpa = np.linalg.norm(cpa_vector)
        else:
            dcpa = float('nan')
            tcpa = float('nan')

    return dcpa, tcpa


def _installed_convention_differs() -> bool:
    """
    Probe: agent eastbound at the origin, obstacle 10 m ahead heading
    straight at it — a collision course. DCPA = 0 in the validated
    convention, ~20 m in the other.
    """
    agent = Agent(position=(0.0, 0.0, 0.0),
                  orientation=(0.0, 0.0, 0.0, 1.0),          # psi = 0
                  velocity=1.0, yaw_rate=0.0, safety_radius=1.0)
    obstacle = DynamicObstacle(tag="probe",
                               position=(10.0, 0.0, 0.0),
                               orientation=(0.0, 0.0, 1.0, 0.0),  # psi = pi
                               velocity=1.0, yaw_rate=0.0, safety_radius=1.0)
    dcpa, _ = _ra_impl.calc_cpa(agent, obstacle)
    return bool(dcpa > 1.0)


def apply() -> bool:
    """Rebind calc_cpa wherever the package holds a reference. Idempotent."""
    if not _installed_convention_differs():
        return False
    _ra_impl.calc_cpa = _reference_calc_cpa
    _omc.calc_cpa = _reference_calc_cpa      # bound at import by `from .. import`
    _ra.calc_cpa = _reference_calc_cpa       # package-level re-export
    return True


PATCHED = apply()
