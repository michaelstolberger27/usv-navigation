// Cross-check the C++ geometry guards G11/G23 against the Python reference
// (G11_check / G23_check, which build the hull via colav_unsafe_set + scipy
// ConvexHull and test intersection with shapely). The bar is boolean
// agreement: a correct hull + correct polygon intersection must reach the
// same decision, even though the hull vertices are not bit-identical.
#include "colav_cpp/geometry.hpp"

#include <gtest/gtest.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<double> parse_row(const std::string& line) {
  std::vector<double> cols;
  std::stringstream ss(line);
  std::string cell;
  while (std::getline(ss, cell, ',')) cols.push_back(std::stod(cell));
  return cols;
}

}  // namespace

TEST(GeometryCrossCheck, MatchesPythonReference) {
  std::ifstream f(REFERENCE_GEOM);
  ASSERT_TRUE(f.is_open()) << "cannot open " << REFERENCE_GEOM;

  std::string line;
  std::getline(f, line);  // header

  int rows = 0, g11_match = 0, g23_match = 0;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    const auto c = parse_row(line);
    ASSERT_EQ(c.size(), 14u) << "malformed row " << rows;

    // px py psi xw yw v tp Cs ox oy ov opsi g11_py g23_py
    const std::vector<colav::Obstacle> obs = {{c[8], c[9], c[10], c[11]}};
    const bool g11 = colav::g11_check(c[0], c[1], c[2], c[3], c[4], c[5],
                                      c[6], obs, c[7]);
    const bool g23 = colav::g23_check(c[0], c[1], c[2], c[3], c[4], c[5],
                                      c[6], obs, c[7]);

    EXPECT_EQ(g11, c[12] != 0.0) << "G11 mismatch, row " << rows;
    EXPECT_EQ(g23, c[13] != 0.0) << "G23 mismatch, row " << rows;
    if (g11 == (c[12] != 0.0)) ++g11_match;
    if (g23 == (c[13] != 0.0)) ++g23_match;
    ++rows;
  }

  EXPECT_GT(rows, 0);
  std::cout << "[geom cross-check] rows=" << rows
            << "  G11 match=" << g11_match << "/" << rows
            << "  G23 match=" << g23_match << "/" << rows << std::endl;
}
