// C++ port of the G22 risk-index path, verified bit-identical against the
// Python reference (colav_automaton.guards.conditions.compute_risk_index,
// which in turn uses colav_unsafe_set's calc_cpa). See test/test_risk.cpp.
#pragma once

#include <array>
#include <vector>

namespace colav {

// An obstacle as the automaton sees it: x, y, velocity, heading (rad).
using Obstacle = std::array<double, 4>;

// Default risk betas (metres / seconds), scaled from the paper's nautical
// values for the CommonOcean evaluation. Match the Python defaults.
struct RiskBetas {
  double dcpa1 = 463.0, dcpa2 = 926.0;
  double tcpa1 = 120.0, tcpa2 = 240.0;
  double dist1 = 148.0, dist2 = 463.0;
};

// Piecewise risk function F(z) (paper eq 20). Mirrors conditions._F.
double risk_F(double z, double beta1, double beta2);

// DCPA / TCPA between the agent and one obstacle, replicating
// colav_unsafe_set.risk_assessment.calc_cpa exactly (quaternion heading
// round-trip, the v_rel ~ 0 branch, and the NaN/inf results).
struct Cpa { double dcpa; double tcpa; };
Cpa calc_cpa(double px, double py, double psi, double v,
             double ox, double oy, double ov, double opsi);

// Max risk index over the approaching obstacles. Mirrors compute_risk_index.
double compute_risk_index(double px, double py, double psi,
                          const std::vector<Obstacle>& obstacles,
                          double v, const RiskBetas& b = {});

// G22: risk index >= K. Mirrors G22_check.
bool g22_check(double px, double py, double psi,
               const std::vector<Obstacle>& obstacles,
               double v, double K, const RiskBetas& b = {});

}  // namespace colav
