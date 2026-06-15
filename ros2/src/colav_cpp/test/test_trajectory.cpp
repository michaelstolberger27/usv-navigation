// End-to-end cross-check: run the C++ SyncRuntime through the same head-on
// scenario as the Python SyncColavRuntime and compare the whole trajectory
// tick-by-tick. Same bit-exact control law + same flow + same guard
// decisions => the paths should coincide. This is the capstone verification:
// the deployable C++ controller IS the validated Python one.
#include "colav_cpp/sync_runtime.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

std::vector<std::string> split(const std::string& line) {
  std::vector<std::string> f;
  std::stringstream ss(line);
  std::string cell;
  while (std::getline(ss, cell, ',')) f.push_back(cell);
  return f;
}

}  // namespace

TEST(TrajectoryCrossCheck, MatchesPythonReference) {
  // Scenario constants — kept in sync with gen_reference.py TRAJ.
  const double goalx = 5000.0, goaly = 0.0;
  const double ox0 = 2500.0, oy0 = 0.0, ov = 5.0, opsi = M_PI;
  const double dt = 1.0;
  colav::SyncRuntime::Params p;
  p.Cs = 300.0; p.v = 6.0; p.tp = 3.0;
  p.K = 0.35; p.K_off = 0.25; p.tp_control = 60.0;
  colav::SyncRuntime rt({goalx, goaly}, {}, 0.0, 0.0, 0.0, p);

  std::ifstream f(REFERENCE_TRAJECTORY);
  ASSERT_TRUE(f.is_open()) << "cannot open " << REFERENCE_TRAJECTORY;
  std::string line;
  std::getline(f, line);  // header

  int rows = 0, mode_match = 0, trans_match = 0, first_div = -1;
  double max_pos = 0.0, max_psi = 0.0;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    const auto c = split(line);            // k,x,y,psi,mode,transition
    const int k = std::stoi(c[0]);
    const double rx = std::stod(c[1]), ry = std::stod(c[2]), rp = std::stod(c[3]);
    const std::string rmode = c[4];
    const std::string rtrans = c.size() > 5 ? c[5] : "";

    const double sim_t = k * dt;
    const double ox = ox0 + ov * std::cos(opsi) * sim_t;
    const double oy = oy0 + ov * std::sin(opsi) * sim_t;
    const std::vector<colav::Obstacle> obs = {{ox, oy, ov, opsi}};
    const auto r = rt.step(dt, &obs);

    max_pos = std::max(max_pos, std::hypot(r.x - rx, r.y - ry));
    max_psi = std::max(max_psi, std::fabs(r.psi - rp));
    if (colav::mode_name(r.mode) == rmode) ++mode_match;
    else if (first_div < 0) first_div = k;
    if (r.transition == rtrans) ++trans_match;

    EXPECT_NEAR(r.x, rx, 1e-6) << "x at k=" << k;
    EXPECT_NEAR(r.y, ry, 1e-6) << "y at k=" << k;
    EXPECT_NEAR(r.psi, rp, 1e-6) << "psi at k=" << k;
    EXPECT_EQ(colav::mode_name(r.mode), rmode) << "mode at k=" << k;
    EXPECT_EQ(r.transition, rtrans) << "transition at k=" << k;
    ++rows;
  }

  EXPECT_GT(rows, 0);
  std::cout << "[trajectory cross-check] rows=" << rows
            << "  max_pos_diff=" << max_pos
            << "  max_psi_diff=" << max_psi
            << "  mode_match=" << mode_match << "/" << rows
            << "  trans_match=" << trans_match << "/" << rows
            << "  first_div_k=" << first_div << std::endl;
}
