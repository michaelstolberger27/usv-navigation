#include "colav_cpp/risk.hpp"

#include <cmath>
#include <limits>

namespace colav {

namespace {

constexpr double kPi = M_PI;
const double kInf = std::numeric_limits<double>::infinity();
const double kNaN = std::numeric_limits<double>::quiet_NaN();

// Python float modulo: result takes the sign of the (positive) divisor.
double py_mod(double a, double m) {
  double r = std::fmod(a, m);
  if (r != 0.0 && r < 0.0) r += m;
  return r;
}

// normalize_angle: (angle + pi) % (2*pi) - pi  (matches the Python helper).
double normalize_angle(double angle) {
  return py_mod(angle + kPi, 2.0 * kPi) - kPi;
}

// Heading recovered through the same quaternion round-trip the Python path
// takes: _create_agent stores (0, 0, sin(psi/2), cos(psi/2)), then
// quaternion_to_heading normalises and converts back. Replicated for
// bit-identity (the round-trip perturbs the heading in the last ULPs).
double heading_via_quaternion(double psi) {
  double qz = std::sin(psi / 2.0);
  double qw = std::cos(psi / 2.0);
  double norm = std::sqrt(qz * qz + qw * qw);  // qx = qy = 0
  qz /= norm;
  qw /= norm;
  double siny_cosp = 2.0 * (qw * qz);
  double cosy_cosp = 1.0 - 2.0 * (qz * qz);
  return normalize_angle(std::atan2(siny_cosp, cosy_cosp));
}

}  // namespace

double risk_F(double z, double beta1, double beta2) {
  if (z <= beta1) return 1.0;
  const double mid = (beta1 + beta2) / 2.0;
  if (z <= mid) {
    const double q = (z - beta1) / (beta2 - beta1);
    return 1.0 - 2.0 * (q * q);
  }
  if (z <= beta2) {
    const double q = (z - beta2) / (beta2 - beta1);
    return 2.0 * (q * q);
  }
  return 0.0;
}

Cpa calc_cpa(double px, double py, double psi, double v,
             double ox, double oy, double ov, double opsi) {
  const double theta1 = heading_via_quaternion(psi);
  const double theta2 = heading_via_quaternion(opsi);

  const double v1x = v * std::cos(theta1);
  const double v1y = v * std::sin(theta1);
  const double v2x = ov * std::cos(theta2);
  const double v2y = ov * std::sin(theta2);

  const double prx = px - ox;
  const double pry = py - oy;
  const double vrx = v1x - v2x;
  const double vry = v1y - v2y;
  const double vrel2 = vrx * vrx + vry * vry;

  double dcpa, tcpa;
  if (vrel2 < 1e-6) {
    // np.allclose(p_rel, [0,0]) with default atol=1e-8, rtol=1e-5 -> |p|<=1e-8
    if (std::fabs(prx) <= 1e-8 && std::fabs(pry) <= 1e-8) {
      dcpa = kNaN;
      tcpa = kInf;
    } else {
      const double distance = std::sqrt(prx * prx + pry * pry);
      const double speed = std::sqrt(v1x * v1x + v1y * v1y);
      dcpa = distance;
      tcpa = speed > 0.0 ? distance / speed : kInf;
    }
  } else {
    tcpa = -(prx * vrx + pry * vry) / vrel2;
    if (tcpa > 0.0) {
      const double cx = prx + tcpa * vrx;
      const double cy = pry + tcpa * vry;
      dcpa = std::sqrt(cx * cx + cy * cy);
    } else {
      dcpa = kNaN;
      tcpa = kNaN;
    }
  }
  return {dcpa, tcpa};
}

double compute_risk_index(double px, double py, double psi,
                          const std::vector<Obstacle>& obstacles,
                          double v, const RiskBetas& b) {
  double max_ri = 0.0;
  for (const auto& o : obstacles) {
    const double ox = o[0], oy = o[1], ov = o[2], opsi = o[3];
    const Cpa c = calc_cpa(px, py, psi, v, ox, oy, ov, opsi);
    const double d_s = std::hypot(ox - px, oy - py);

    // tcpa < 0 -> receding, skip. NaN < 0 is false, so NaN/inf rows fall
    // through and contribute F(NaN)=F(inf)=0 — matches the Python path.
    if (c.tcpa < 0.0) continue;

    const double f_dcpa = risk_F(c.dcpa, b.dcpa1, b.dcpa2);
    const double f_tcpa = risk_F(c.tcpa, b.tcpa1, b.tcpa2);
    const double f_dist = risk_F(d_s, b.dist1, b.dist2);
    const double ri = (f_dcpa + f_tcpa + f_dist) / 3.0;
    if (ri > max_ri) max_ri = ri;
  }
  return max_ri;
}

bool g22_check(double px, double py, double psi,
               const std::vector<Obstacle>& obstacles,
               double v, double K, const RiskBetas& b) {
  return compute_risk_index(px, py, psi, obstacles, v, b) >= K;
}

}  // namespace colav
