#include "colav_cpp/control.hpp"

#include <cmath>

namespace colav {

double prescribed_time_control(double t, double x, double y, double psi,
                               double xw, double yw,
                               double a, double v, double eta, double tp) {
  // Operation order matches the Python reference exactly so the result is
  // bit-identical (same IEEE-754 doubles, same libm).
  const double psi_dg = std::atan2(yw - y, xw - x);

  const double dx = xw - x;
  const double dy = yw - y;
  const double d_squared = dx * dx + dy * dy;

  double psi_dg_dot;
  if (d_squared < 1e-6) {
    psi_dg_dot = 0.0;
  } else {
    psi_dg_dot = (-v * dx * std::sin(psi) + v * dy * std::cos(psi)) / d_squared;
  }

  const double e = std::atan2(std::sin(psi - psi_dg), std::cos(psi - psi_dg));

  double u;
  if (t < tp) {
    const double tau = std::max(tp - t, 0.01 * tp);
    u = (1.0 / a) * psi_dg_dot + psi - eta * e / (a * tau);
  } else {
    u = (1.0 / a) * psi_dg_dot + psi;
  }
  return u;
}

bool l1_check(double px, double py, double v1x, double v1y, double delta) {
  const double dist =
      std::sqrt((px - v1x) * (px - v1x) + (py - v1y) * (py - v1y));
  return dist > delta;
}

bool l2_check(double px, double py, double psi, double v1x, double v1y) {
  const double angle_to_v1 = std::atan2(v1y - py, v1x - px);
  const double rel = std::atan2(std::sin(angle_to_v1 - psi),
                                std::cos(angle_to_v1 - psi));
  const double kAhead = M_PI / 2.0;  // V1_AHEAD_THRESHOLD
  return -kAhead < rel && rel < kAhead;
}

}  // namespace colav
