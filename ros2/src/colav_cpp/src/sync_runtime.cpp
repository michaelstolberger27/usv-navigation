#include "colav_cpp/sync_runtime.hpp"

#include <algorithm>
#include <cmath>

#include "colav_cpp/control.hpp"
#include "colav_cpp/virtual_waypoint.hpp"

namespace colav {

const char* mode_name(Mode m) {
  switch (m) {
    case Mode::S1_WaypointReaching: return "WAYPOINT_REACHING";
    case Mode::S2_CollisionAvoidance: return "COLLISION_AVOIDANCE";
    case Mode::S3_ConstantControl: return "CONSTANT_CONTROL";
  }
  return "?";
}

namespace {

// Arrival tolerance delta (paper eq 9), mirroring _compute_delta.
double compute_delta(double v, double a, double eta, double tp, double m) {
  const double denom_inner = m - M_PI - eta * M_PI / (a * std::pow(tp, eta));
  if (denom_inner <= 0.0) return std::max(5.0, v * tp * 0.5);
  return std::max(2.0 * v / (a * denom_inner), 1.0);
}

}  // namespace

SyncRuntime::SyncRuntime(Point goal, std::vector<Obstacle> obstacles,
                         double x0, double y0, double psi0, const Params& p)
    : p_(p),
      tp_control_(p.tp_control > 0.0 ? p.tp_control : p.tp),
      delta_(compute_delta(p.v, p.a, p.eta, p.tp, p.m)),
      dsafe_(p.Cs + p.v * p.tp),
      v1_Cs_(p.Cs + 0.25 * p.Cs),
      max_horizon_(std::max(60.0, 3.0 * (p.Cs + p.v * p.tp) / p.v)),
      waypoints_{goal},
      obstacles_(std::move(obstacles)),
      x_(x0), y_(y0), psi_(psi0), u_(psi0),
      t_(0.0), t_last_transition_(0.0),
      mode_(Mode::S1_WaypointReaching) {}

bool SyncRuntime::goal_reached(double radius) const {
  return std::hypot(waypoints_.front().x - x_, waypoints_.front().y - y_) <
         radius;
}

StepResult SyncRuntime::step(double dt, const std::vector<Obstacle>* obstacles) {
  if (obstacles != nullptr) obstacles_ = *obstacles;

  // 1. Control toward the active waypoint (goal in S1/S3, V1 in S2), with
  //    the prescribed-time clock measured in sim time since the transition.
  const Point& target = waypoints_.back();
  u_ = prescribed_time_control(t_ - t_last_transition_, x_, y_, psi_,
                               target.x, target.y, p_.a, p_.v, p_.eta,
                               tp_control_);

  // 2. Continuous flow (Euler), read from the pre-integration state + u.
  double dx, dy, dpsi;
  if (mode_ == Mode::S3_ConstantControl) {
    dx = p_.v * std::cos(psi_);
    dy = p_.v * std::sin(psi_);
    dpsi = 0.0;
  } else {
    dx = p_.v * std::cos(psi_);
    dy = p_.v * std::sin(psi_);
    dpsi = -p_.a * psi_ + p_.a * u_;
  }
  x_ += dx * dt;
  y_ += dy * dt;
  psi_ += dpsi * dt;
  t_ += dt;

  // 3. Guards — at most one transition, evaluated on the new state.
  const std::string fired = evaluate_guards();
  return {t_, x_, y_, psi_, mode_, u_, fired};
}

StepResult SyncRuntime::step_external(double dt, double x, double y, double psi,
                                      const std::vector<Obstacle>* obstacles) {
  if (obstacles != nullptr) obstacles_ = *obstacles;
  x_ = x;
  y_ = y;
  psi_ = psi;

  const Point& target = waypoints_.back();
  u_ = prescribed_time_control(t_ - t_last_transition_, x_, y_, psi_,
                               target.x, target.y, p_.a, p_.v, p_.eta,
                               tp_control_);
  t_ += dt;  // host owns integration; we only evaluate

  const std::string fired = evaluate_guards();
  return {t_, x_, y_, psi_, mode_, u_, fired};
}

std::string SyncRuntime::evaluate_guards() {
  const Point goal = waypoints_.front();
  if (mode_ == Mode::S1_WaypointReaching) {
    if (g11_check(x_, y_, psi_, goal.x, goal.y, p_.v, p_.tp, obstacles_, p_.Cs) &&
        g22_check(x_, y_, psi_, obstacles_, p_.v, p_.K, p_.betas)) {
      // reset_enter_avoidance: compute and push V1.
      const auto v1 = compute_v1(x_, y_, psi_, obstacles_, v1_Cs_, p_.v,
                                 dsafe_, max_horizon_, p_.v1_buffer);
      if (v1) waypoints_.push_back(*v1);
      mode_ = Mode::S2_CollisionAvoidance;
      t_last_transition_ = t_;
      return "avoid";
    }
  } else if (mode_ == Mode::S2_CollisionAvoidance) {
    if (waypoints_.size() >= 2) {
      const Point& v1 = waypoints_.back();
      const bool l1 = l1_check(x_, y_, v1.x, v1.y, delta_);
      const bool l2 = l2_check(x_, y_, psi_, v1.x, v1.y);
      if (!l1 || !l2) {
        waypoints_.pop_back();  // reset_reach_V1
        mode_ = Mode::S3_ConstantControl;
        t_last_transition_ = t_;
        return "hold";
      }
    }
  } else {  // S3
    if (!g23_check(x_, y_, psi_, goal.x, goal.y, p_.v, p_.tp, obstacles_,
                   p_.Cs)) {
      if (compute_risk_index(x_, y_, psi_, obstacles_, p_.v, p_.betas) <
          p_.K_off) {
        u_ = psi_;  // reset_exit_avoidance
        mode_ = Mode::S1_WaypointReaching;
        t_last_transition_ = t_;
        return "resume";
      }
    }
  }
  return "";
}

}  // namespace colav
