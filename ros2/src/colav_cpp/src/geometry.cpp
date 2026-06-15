#include "colav_cpp/geometry.hpp"

#include <algorithm>
#include <cmath>

namespace colav {

namespace {

constexpr double kPi = M_PI;

// 10 vertices approximating a Cs-circle: theta = linspace(0, 2pi, 10,
// endpoint=False), matching _generate_circle_vertices.
void add_circle(std::vector<Point>& out, double xc, double yc, double r) {
  for (int i = 0; i < 10; ++i) {
    const double theta = i * (2.0 * kPi / 10.0);
    out.push_back({xc + r * std::cos(theta), yc + r * std::sin(theta)});
  }
}

double cross(const Point& o, const Point& a, const Point& b) {
  return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

}  // namespace

std::vector<Point> convex_hull(std::vector<Point> pts) {
  std::sort(pts.begin(), pts.end(), [](const Point& a, const Point& b) {
    return a.x < b.x || (a.x == b.x && a.y < b.y);
  });
  pts.erase(std::unique(pts.begin(), pts.end(),
                        [](const Point& a, const Point& b) {
                          return a.x == b.x && a.y == b.y;
                        }),
            pts.end());
  const int n = static_cast<int>(pts.size());
  if (n < 3) return pts;

  std::vector<Point> hull(2 * n);
  int k = 0;
  for (int i = 0; i < n; ++i) {  // lower hull
    while (k >= 2 && cross(hull[k - 2], hull[k - 1], pts[i]) <= 0) --k;
    hull[k++] = pts[i];
  }
  const int lower = k + 1;
  for (int i = n - 2; i >= 0; --i) {  // upper hull
    while (k >= lower && cross(hull[k - 2], hull[k - 1], pts[i]) <= 0) --k;
    hull[k++] = pts[i];
  }
  hull.resize(k - 1);  // drop the repeated start point
  return hull;
}

bool convex_polygons_intersect(const std::vector<Point>& a,
                               const std::vector<Point>& b) {
  if (a.size() < 3 || b.size() < 3) return false;

  // Separating-axis test: a separating axis exists iff the polygons are
  // disjoint. Test every edge normal of both polygons; strict gap = separated
  // (so boundary contact counts as intersecting, matching shapely).
  const std::vector<const std::vector<Point>*> polys = {&a, &b};
  for (const auto* poly : polys) {
    const auto& p = *poly;
    const int m = static_cast<int>(p.size());
    for (int i = 0; i < m; ++i) {
      const Point& p0 = p[i];
      const Point& p1 = p[(i + 1) % m];
      const double ax = -(p1.y - p0.y);  // edge normal
      const double ay = (p1.x - p0.x);

      double amin = 1e300, amax = -1e300, bmin = 1e300, bmax = -1e300;
      for (const auto& q : a) {
        const double d = q.x * ax + q.y * ay;
        amin = std::min(amin, d);
        amax = std::max(amax, d);
      }
      for (const auto& q : b) {
        const double d = q.x * ax + q.y * ay;
        bmin = std::min(bmin, d);
        bmax = std::max(bmax, d);
      }
      if (amax < bmin || bmax < amin) return false;  // separated
    }
  }
  return true;
}

std::vector<Point> los_cone(double px, double py, double xw, double yw,
                            double v, double tp) {
  const double sx = xw - px;
  const double sy = yw - py;
  const double dist = std::sqrt(sx * sx + sy * sy);
  if (dist < 1e-6) {
    // Degenerate (waypoint at the ego): a tiny triangle around the point.
    return {{px - 0.01, py}, {px + 0.01, py}, {px, py + 0.01}};
  }
  const double ux = sx / dist;
  const double uy = sy / dist;
  const double perpx = -uy;
  const double perpy = ux;
  const double radius = v * tp;
  return {{px + radius * perpx, py + radius * perpy},
          {px - radius * perpx, py - radius * perpy},
          {xw, yw}};
}

std::vector<Point> unsafe_region_hull(double px, double py, double psi,
                                      const std::vector<Obstacle>& obstacles,
                                      double Cs, double ship_v,
                                      bool static_only) {
  if (obstacles.empty()) return {};

  std::vector<Point> verts;
  for (const auto& o : obstacles) {
    const double ox = o[0], oy = o[1], opsi = o[3];
    const double vel = static_only ? 0.0 : o[2];

    add_circle(verts, ox, oy, Cs);

    // Swept circle at the TCPA-predicted position (yaw_rate is 0, so the
    // prediction is a plain constant-velocity step along the heading).
    const Cpa c = calc_cpa(px, py, psi, ship_v, ox, oy, vel, opsi);
    if (c.tcpa > 0.0 && !std::isnan(c.tcpa)) {
      const double fx = ox + vel * std::cos(opsi) * c.tcpa;
      const double fy = oy + vel * std::sin(opsi) * c.tcpa;
      add_circle(verts, fx, fy, Cs);
    }
  }
  return convex_hull(std::move(verts));
}

namespace {

bool guard_intersects(double px, double py, double psi, double xw, double yw,
                      double v, double Cs,
                      const std::vector<Obstacle>& obstacles,
                      bool static_only) {
  if (obstacles.empty()) return false;
  if (v <= 0.0 || Cs <= 0.0) return false;
  const auto hull = unsafe_region_hull(px, py, psi, obstacles, Cs, v,
                                       static_only);
  if (hull.size() < 3) return false;
  const auto cone = los_cone(px, py, xw, yw, v, Cs / v);  // effective_tp=Cs/v
  return convex_polygons_intersect(hull, cone);
}

}  // namespace

bool g11_check(double px, double py, double psi, double xw, double yw,
               double v, double tp, const std::vector<Obstacle>& obstacles,
               double Cs) {
  (void)tp;  // cone uses effective_tp = Cs/v; tp only gates validity
  if (tp <= 0.0) return false;
  return guard_intersects(px, py, psi, xw, yw, v, Cs, obstacles,
                          /*static_only=*/false);
}

bool g23_check(double px, double py, double psi, double xw, double yw,
               double v, double tp, const std::vector<Obstacle>& obstacles,
               double Cs) {
  if (tp <= 0.0) return false;
  return guard_intersects(px, py, psi, xw, yw, v, Cs, obstacles,
                          /*static_only=*/true);
}

}  // namespace colav
