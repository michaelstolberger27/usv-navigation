// C++ port of the COLAV controller's pure-math core.
//
// Each function mirrors a specific Python reference in colav_automaton and
// is verified bit-identical against it (see test/test_control.cpp). Python
// stays the source of truth; this is the deployable C++ twin that the rclcpp
// node will link, and the cross-check is the verification result.
#pragma once

namespace colav {

// Prescribed-time heading control law.
// Mirrors colav_automaton.controllers.prescribed_time.
// compute_prescribed_time_control (paper eq, with the 1%-of-tp gain floor).
double prescribed_time_control(double t, double x, double y, double psi,
                               double xw, double yw,
                               double a, double v, double eta, double tp);

// L1: ||p - V1|| > delta  (paper eq 15). Mirrors L1_check.
// Precondition: delta > 0 (the Python version raises otherwise).
bool l1_check(double px, double py, double v1x, double v1y, double delta);

// L2: V1 is ahead of the ship, within +/- pi/2 of heading (paper eq 16).
// Mirrors L2_check.
bool l2_check(double px, double py, double psi, double v1x, double v1y);

}  // namespace colav
