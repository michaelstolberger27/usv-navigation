// C++ twin of SyncColavRuntime: the deterministic tick-synchronous executive
// that orchestrates the (separately verified) control law, guards, risk index
// and V1 selection. One step() computes control, integrates the flow, and
// fires at most one transition — the same order as the Python runtime. The
// rclcpp node steps this; the trajectory cross-check verifies a full run
// against the Python reference.
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "colav_cpp/geometry.hpp"  // Point
#include "colav_cpp/risk.hpp"      // Obstacle

namespace colav {

enum class Mode { S1_WaypointReaching, S2_CollisionAvoidance, S3_ConstantControl };

const char* mode_name(Mode m);

struct StepResult {
  double t;
  double x, y, psi;
  Mode mode;
  double u;
  std::string transition;  // "avoid"/"hold"/"resume", or "" if none
};

class SyncRuntime {
 public:
  struct Params {
    double Cs = 2.0, a = 1.67, v = 12.0, eta = 3.5, tp = 1.0;
    double K = 0.35, K_off = 0.25;
    double v1_buffer = 0.0, m = 3.0;
    // Prescribed-time control horizon; defaults to tp when unset (<=0).
    double tp_control = 0.0;
    RiskBetas betas;
  };

  SyncRuntime(Point goal, std::vector<Obstacle> obstacles,
              double x0, double y0, double psi0, const Params& p);

  // Advance one tick of sim time (self-integrating). Pass the current
  // obstacle list to update it, or leave empty to keep the previous one.
  StepResult step(double dt, const std::vector<Obstacle>* obstacles = nullptr);

  Mode mode() const { return mode_; }
  bool goal_reached(double radius) const;

 private:
  Params p_;
  double tp_control_;
  double delta_;
  double dsafe_;
  double v1_Cs_;
  double max_horizon_;

  std::vector<Point> waypoints_;  // [goal] or [goal, V1]
  std::vector<Obstacle> obstacles_;
  double x_, y_, psi_, u_;
  double t_, t_last_transition_;
  Mode mode_;
};

}  // namespace colav
