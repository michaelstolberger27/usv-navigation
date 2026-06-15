#include "colav_cpp/virtual_waypoint.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace colav {

namespace {

constexpr double kPi = M_PI;
const double kInf = std::numeric_limits<double>::infinity();

double normalize_angle(double a) {
  return std::atan2(std::sin(a), std::cos(a));
}

void add_circle(std::vector<Point>& out, double xc, double yc, double r,
                int n = 10) {
  for (int i = 0; i < n; ++i) {
    const double theta = i * (2.0 * kPi / n);
    out.push_back({xc + r * std::cos(theta), yc + r * std::sin(theta)});
  }
}

// 8 vertices per obstacle at v1_Cs — the default_vertex_provider fallback.
std::vector<Point> default_vertices(const std::vector<Obstacle>& obstacles,
                                    double Cs) {
  std::vector<Point> v;
  for (const auto& o : obstacles) {
    for (int i = 0; i < 8; ++i) {
      const double a = i * (kPi / 4.0);
      v.push_back({o[0] + Cs * std::cos(a), o[1] + Cs * std::sin(a)});
    }
  }
  return v;
}

// Predicted CPA if the ship heads toward (vx, vy), mirroring _predicted_cpa /
// _min_cpa_for_vertex.
double predicted_cpa(double px, double py, double v, double heading,
                     double ox, double oy, double ovx, double ovy) {
  const double new_vx = v * std::cos(heading);
  const double new_vy = v * std::sin(heading);
  const double dx = ox - px;
  const double dy = oy - py;
  const double dvx = ovx - new_vx;
  const double dvy = ovy - new_vy;
  const double dv_sq = dvx * dvx + dvy * dvy;
  if (dv_sq < 1e-6) return std::hypot(dx, dy);
  const double t = std::max(-(dx * dvx + dy * dvy) / dv_sq, 0.0);
  const double cx = px + new_vx * t;
  const double cy = py + new_vy * t;
  return std::hypot(cx - (ox + ovx * t), cy - (oy + ovy * t));
}

double min_cpa_for_vertex(double px, double py, double v, double vx, double vy,
                          const std::vector<Obstacle>& obstacles) {
  const double heading = std::atan2(vy - py, vx - px);
  double min_cpa = kInf;
  for (const auto& o : obstacles) {
    const double ovx = o[2] * std::cos(o[3]);
    const double ovy = o[2] * std::sin(o[3]);
    min_cpa = std::min(min_cpa,
                       predicted_cpa(px, py, v, heading, o[0], o[1], ovx, ovy));
  }
  return min_cpa;
}

}  // namespace

namespace {

// One entry of the augmented (swept) obstacle list create_unsafe_set sees.
struct Aug { double x, y, vel, heading; };

}  // namespace

std::vector<Point> swept_unsafe_hull(double px, double py, double psi,
                                     const std::vector<Obstacle>& obstacles,
                                     double v1_Cs, double ship_v,
                                     double dsafe, double max_horizon) {
  if (obstacles.empty()) return {};

  // _compute_swept_obstacles: original obstacle (its velocity) plus
  // zero-velocity trajectory samples up to min(tcpa, max_horizon).
  std::vector<Aug> aug;
  for (const auto& o : obstacles) {
    const double ox = o[0], oy = o[1], ov = o[2], opsi = o[3];
    aug.push_back({ox, oy, ov, opsi});
    const Cpa c = calc_cpa(px, py, psi, ship_v, ox, oy, ov, opsi);
    if (ov > 0.1 && c.tcpa > 0.0) {
      const double sweep = std::min(c.tcpa, max_horizon);
      int n = static_cast<int>(sweep / 10.0);
      n = std::max(3, std::min(20, n));
      for (int i = 1; i <= n; ++i) {
        const double dt = sweep * i / n;
        aug.push_back({ox + ov * std::cos(opsi) * dt,
                       oy + ov * std::sin(opsi) * dt, 0.0, opsi});
      }
    }
  }

  // create_unsafe_set on the augmented list with dsf = dsafe. Unlike the G11
  // path (large distance_safety -> all obstacles), here the index filters
  // genuinely prune distant samples. Agent and obstacle safety radius are
  // both v1_Cs, so every distance test subtracts 2*v1_Cs.
  const int k = static_cast<int>(aug.size());
  std::vector<Cpa> m(k);
  std::vector<bool> i1(k), i3(k);
  for (int i = 0; i < k; ++i) {
    m[i] = calc_cpa(px, py, psi, ship_v, aug[i].x, aug[i].y, aug[i].vel,
                    aug[i].heading);
    const double d_agent = std::hypot(aug[i].x - px, aug[i].y - py);
    i1[i] = (d_agent - 2.0 * v1_Cs) <= dsafe;
    i3[i] = (m[i].dcpa <= dsafe) && (m[i].tcpa <= 15.0);
  }
  // I2: obstacle in I1 with another obstacle within dsafe.
  std::vector<bool> i2(k, false);
  for (int i = 0; i < k; ++i) {
    if (!i1[i]) continue;
    for (int j = 0; j < k; ++j) {
      if (j == i) continue;
      const double d = std::hypot(aug[i].x - aug[j].x, aug[i].y - aug[j].y);
      if ((d - 2.0 * v1_Cs) <= dsafe) { i2[i] = true; break; }
    }
  }

  std::vector<Point> verts;
  for (int i = 0; i < k; ++i) {
    if (!(i1[i] || i2[i] || i3[i])) continue;  // uIoI membership
    add_circle(verts, aug[i].x, aug[i].y, v1_Cs);
    if (m[i].tcpa > 0.0 && !std::isnan(m[i].tcpa)) {
      add_circle(verts, aug[i].x + aug[i].vel * std::cos(aug[i].heading) * m[i].tcpa,
                 aug[i].y + aug[i].vel * std::sin(aug[i].heading) * m[i].tcpa, v1_Cs);
    }
  }
  return convex_hull(std::move(verts));
}

std::optional<Point> compute_v1(double px, double py, double psi,
                                const std::vector<Obstacle>& obstacles,
                                double v1_Cs, double ship_v, double dsafe,
                                double max_horizon, double buffer_distance) {
  if (obstacles.empty()) return std::nullopt;

  std::vector<Point> verts = swept_unsafe_hull(px, py, psi, obstacles, v1_Cs,
                                               ship_v, dsafe, max_horizon);
  if (verts.empty()) verts = default_vertices(obstacles, v1_Cs);
  if (verts.empty()) return std::nullopt;

  // Starboard-most (most negative) and port-most (most positive) vertex that
  // lies ahead (within +/- pi/2 of heading).
  const Point* stbd = nullptr;
  double stbd_angle = kInf;
  const Point* port = nullptr;
  double port_angle = -kInf;
  for (const auto& vtx : verts) {
    const double rel = normalize_angle(std::atan2(vtx.y - py, vtx.x - px) - psi);
    if (-kPi / 2.0 < rel && rel < kPi / 2.0) {
      if (rel < stbd_angle) { stbd_angle = rel; stbd = &vtx; }
      if (rel > port_angle) { port_angle = rel; port = &vtx; }
    }
  }
  if (stbd == nullptr && port == nullptr) return std::nullopt;

  const double stbd_cpa =
      stbd ? min_cpa_for_vertex(px, py, ship_v, stbd->x, stbd->y, obstacles)
           : -1.0;
  const double port_cpa =
      port ? min_cpa_for_vertex(px, py, ship_v, port->x, port->y, obstacles)
           : -1.0;

  // COLREGs starboard preference: port only when clearly (>10%) better.
  Point best;
  if (stbd != nullptr && (port == nullptr || port_cpa <= stbd_cpa * 1.1)) {
    best = *stbd;
  } else {
    best = *port;
  }

  (void)buffer_distance;  // buffer (>0) not exercised here; CommonOcean uses 0
  return best;
}

}  // namespace colav
