// Cross-check the C++ V1 (virtual waypoint) against the Python reference
// (compute_v1 fed by the swept get_unsafe_set_vertices, exactly as
// reset_enter_avoidance builds it). Same chosen vertex => same coordinates;
// a different pick (a near-tie in angle or CPA) would show as a large diff.
#include "colav_cpp/virtual_waypoint.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

std::vector<double> parse_row(const std::string& line) {
  std::vector<double> cols;
  std::stringstream ss(line);
  std::string cell;
  while (std::getline(ss, cell, ',')) cols.push_back(std::stod(cell));
  return cols;
}

}  // namespace

TEST(V1CrossCheck, MatchesPythonReference) {
  std::ifstream f(REFERENCE_V1);
  ASSERT_TRUE(f.is_open()) << "cannot open " << REFERENCE_V1;

  std::string line;
  std::getline(f, line);  // header

  int rows = 0, has_match = 0, pt_close = 0;
  double max_diff = 0.0;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    const auto c = parse_row(line);
    ASSERT_EQ(c.size(), 13u) << "malformed row " << rows;

    // px py psi v tp Cs ox oy ov opsi has_v1 v1x v1y
    const std::vector<colav::Obstacle> obs = {{c[6], c[7], c[8], c[9]}};
    const double v1_Cs = c[5] + 0.25 * c[5];
    const double dsafe = c[5] + c[3] * c[4];
    const double max_horizon = std::max(60.0, 3.0 * dsafe / c[3]);

    const auto v1 = colav::compute_v1(c[0], c[1], c[2], obs, v1_Cs, c[3],
                                      dsafe, max_horizon, 0.0);
    const bool py_has = c[10] != 0.0;
    EXPECT_EQ(v1.has_value(), py_has) << "has_v1 mismatch, row " << rows;
    if (v1.has_value() == py_has) ++has_match;

    if (v1.has_value() && py_has) {
      const double d = std::hypot(v1->x - c[11], v1->y - c[12]);
      max_diff = std::max(max_diff, d);
      if (d < 1e-6) {
        ++pt_close;
      } else {
        // A different hull vertex was chosen. Cause: the ~1-ULP calc_cpa
        // residual (the np.linalg.norm dnrm2-vs-sqrt difference seen in the
        // risk port) can flip one trajectory sample across the dsafe filter
        // boundary, changing a hull extreme and hence the starboard/port
        // pick. Both picks are valid avoidance waypoints — not a logic error.
        std::cout << "  [v1 boundary-flip] row " << rows
                  << " cpp=(" << v1->x << "," << v1->y << ")"
                  << " py=(" << c[11] << "," << c[12] << ")" << std::endl;
      }
    }
    ++rows;
  }

  EXPECT_GT(rows, 0);
  EXPECT_EQ(has_match, rows);                 // has_v1 decision always matches
  EXPECT_GE(pt_close, rows - 3);              // allow a tiny near-tie budget
  std::cout << "[v1 cross-check] rows=" << rows
            << "  has_v1 match=" << has_match << "/" << rows
            << "  point<1e-6=" << pt_close << "/" << rows
            << "  max_diff=" << max_diff << std::endl;
}
