// C++ port of the virtual-waypoint (V1) computation, cross-checked against
// the Python reference (controllers.virtual_waypoint.compute_v1 fed by the
// swept get_unsafe_set_vertices, as resets.reset_enter_avoidance calls it).
#pragma once

#include <optional>
#include <vector>

#include "colav_cpp/geometry.hpp"  // Point
#include "colav_cpp/risk.hpp"      // Obstacle

namespace colav {

// Swept unsafe-set hull vertices for V1: convex hull of v1_Cs-circles at each
// obstacle's current position, its TCPA-predicted position, and trajectory
// samples up to min(tcpa, max_horizon). Mirrors get_unsafe_set_vertices with
// use_swept_region=true.
std::vector<Point> swept_unsafe_hull(double px, double py, double psi,
                                     const std::vector<Obstacle>& obstacles,
                                     double v1_Cs, double ship_v,
                                     double dsafe, double max_horizon);

// V1 selection (compute_v1): the unsafe-set vertex ahead of the ship that
// yields the largest predicted CPA, with a COLREGs starboard preference
// (port only chosen when its CPA is >10% better). nullopt if none ahead.
std::optional<Point> compute_v1(double px, double py, double psi,
                                const std::vector<Obstacle>& obstacles,
                                double v1_Cs, double ship_v, double dsafe,
                                double max_horizon,
                                double buffer_distance = 0.0);

}  // namespace colav
