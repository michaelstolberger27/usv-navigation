// C++ port of the geometry guards G11 / G23.
//
// Both reduce to: does the line-of-sight cone (a triangle) intersect the
// unsafe region (convex hull of Cs-circles around the obstacles, current
// and TCPA-predicted)? The Python path builds the hull via colav_unsafe_set
// (scipy ConvexHull) and tests intersection with shapely. Here the hull is a
// monotone-chain convex hull and intersection is the separating-axis test.
//
// The verification bar is the *boolean* (G11/G23), not bit-identical hull
// vertices: floating-point hull differences cannot change whether two
// polygons overlap except in measure-zero tangent cases. test_geometry.cpp
// cross-checks the booleans against the Python reference.
#pragma once

#include <vector>

#include "colav_cpp/risk.hpp"  // colav::Obstacle

namespace colav {

struct Point { double x; double y; };

// 2D convex hull (monotone chain), counter-clockwise, no repeated last point.
std::vector<Point> convex_hull(std::vector<Point> pts);

// Do two convex polygons overlap (closed sets: boundary contact counts)?
// Separating-axis test over both polygons' edge normals.
bool convex_polygons_intersect(const std::vector<Point>& a,
                               const std::vector<Point>& b);

// LOS cone triangle conv(B2(p, v*tp), waypoint). Mirrors create_los_cone.
std::vector<Point> los_cone(double px, double py, double xw, double yw,
                            double v, double tp);

// Unsafe-region hull (empty if no obstacles). static_only=true zeroes the
// obstacle velocities (the G23 resume check); otherwise the swept region
// adds each obstacle's TCPA-predicted Cs-circle (G11).
std::vector<Point> unsafe_region_hull(double px, double py, double psi,
                                      const std::vector<Obstacle>& obstacles,
                                      double Cs, double ship_v,
                                      bool static_only);

// G11: LOS-to-waypoint cone intersects the swept unsafe region (paper eq 13).
bool g11_check(double px, double py, double psi, double xw, double yw,
               double v, double tp, const std::vector<Obstacle>& obstacles,
               double Cs);

// G23: LOS cone intersects the static unsafe region (resume check, eq 27).
bool g23_check(double px, double py, double psi, double xw, double yw,
               double v, double tp, const std::vector<Obstacle>& obstacles,
               double Cs);

}  // namespace colav
