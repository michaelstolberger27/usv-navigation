// C++ rclcpp twin of colav_ros/colav_node.py. Same time-triggered pattern:
// subscriptions cache the latest ego pose and obstacle list, a fixed-rate
// timer steps the verified SyncRuntime (colav_core) once per tick and
// publishes a Twist command + an S1/S2/S3 state label. It never integrates
// the vessel; the world on the other side (fake_world or VRX) does. Speaks
// the same topics as the Python node, so it is a drop-in replacement.
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/string.hpp>

#include <colav_interfaces/msg/obstacle_array.hpp>

#include "colav_cpp/sync_runtime.hpp"

namespace {

double yaw_from_quaternion(const geometry_msgs::msg::Quaternion& q) {
  const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
  const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  return std::atan2(siny_cosp, cosy_cosp);
}

}  // namespace

class ColavNode : public rclcpp::Node {
 public:
  ColavNode() : rclcpp::Node("colav_node") {
    // Demo defaults match the Python node / example scenario 3.
    const double goal_x = declare_parameter("goal_x", 100.0);
    const double goal_y = declare_parameter("goal_y", 0.0);
    colav::SyncRuntime::Params p;
    p.Cs = declare_parameter("Cs", 5.0);
    p.v = declare_parameter("v", 12.0);
    p.tp = declare_parameter("tp", 1.0);
    p.a = declare_parameter("a", 1.67);
    p.eta = declare_parameter("eta", 3.5);
    p.K_off = declare_parameter("K_off", 1.0);
    p.tp_control = declare_parameter("tp_control", 2.0);
    dt_ = declare_parameter("dt", 0.05);
    arrival_radius_ = declare_parameter("arrival_radius", 1.5);
    goal_ = {goal_x, goal_y};

    rt_ = std::make_unique<colav::SyncRuntime>(
        colav::Point{goal_x, goal_y}, std::vector<colav::Obstacle>{}, 0.0, 0.0,
        0.0, p);
    a_ = p.a;
    v_ = p.v;

    // Odometry is a high-rate sensor stream: best-effort QoS (drop,
    // don't queue) — the tick only ever reads the latest sample.
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        "ego_odom", rclcpp::SensorDataQoS(),
        [this](const nav_msgs::msg::Odometry::SharedPtr m) { on_odom(m); });
    obs_sub_ = create_subscription<colav_interfaces::msg::ObstacleArray>(
        "obstacles", 10,
        [this](const colav_interfaces::msg::ObstacleArray::SharedPtr m) {
          on_obstacles(m);
        });
    cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>("cmd", 10);
    state_pub_ = create_publisher<std_msgs::msg::String>("colav_state", 10);

    // Node-clock timer: with use_sim_time this ticks on /clock (e.g.
    // under Gazebo), otherwise on wall time.
    timer_ = rclcpp::create_timer(
        this, get_clock(), rclcpp::Duration::from_seconds(dt_),
        [this]() { tick(); });
    RCLCPP_INFO(get_logger(), "colav_node (C++) up: goal=(%.1f,%.1f) dt=%.3fs",
                goal_.x, goal_.y, dt_);
  }

 private:
  void on_odom(const nav_msgs::msg::Odometry::SharedPtr m) {
    ego_x_ = m->pose.pose.position.x;
    ego_y_ = m->pose.pose.position.y;
    ego_psi_ = yaw_from_quaternion(m->pose.pose.orientation);
  }

  void on_obstacles(const colav_interfaces::msg::ObstacleArray::SharedPtr m) {
    obstacles_.clear();
    for (const auto& o : m->obstacles) {
      obstacles_.push_back({o.x, o.y, o.velocity, o.heading});
    }
  }

  void tick() {
    if (arrived_) return;
    const auto r =
        rt_->step_external(dt_, ego_x_, ego_y_, ego_psi_, &obstacles_);

    geometry_msgs::msg::Twist cmd;
    cmd.linear.x = v_;
    cmd.angular.z = -a_ * ego_psi_ + a_ * r.u;  // yaw_rate = -a*psi + a*u
    cmd_pub_->publish(cmd);

    std_msgs::msg::String state;
    state.data = colav::mode_name(r.mode);
    state_pub_->publish(state);

    if (!r.transition.empty() || colav::mode_name(r.mode) != last_mode_) {
      RCLCPP_INFO(get_logger(), "t=%6.2fs  %s%s", r.t, colav::mode_name(r.mode),
                  r.transition.empty() ? "" : ("  [" + r.transition + "]").c_str());
      last_mode_ = colav::mode_name(r.mode);
    }
    if (std::hypot(goal_.x - ego_x_, goal_.y - ego_y_) < arrival_radius_) {
      arrived_ = true;
      cmd_pub_->publish(geometry_msgs::msg::Twist());  // stop
      RCLCPP_INFO(get_logger(), "goal reached at t=%.2fs", r.t);
    }
  }

  std::unique_ptr<colav::SyncRuntime> rt_;
  colav::Point goal_;
  double dt_, arrival_radius_, a_, v_;
  double ego_x_ = 0.0, ego_y_ = 0.0, ego_psi_ = 0.0;
  std::vector<colav::Obstacle> obstacles_;
  bool arrived_ = false;
  std::string last_mode_;

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<colav_interfaces::msg::ObstacleArray>::SharedPtr obs_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr state_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ColavNode>());
  rclcpp::shutdown();
  return 0;
}
