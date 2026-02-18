"""
Metrics extraction from COLAV controller tracker data.

Used by the batch evaluation runner to compute per-scenario metrics
after a commonocean-sim simulation completes.
"""

import numpy as np


def extract_metrics(controller, scenario, goal_pos, goal_rect, scenario_id, dt):
    """
    Extract all evaluation metrics from a completed simulation run.

    Args:
        controller: HybridAutomatonController with populated tracker data.
        scenario: CommonOcean Scenario object (for dynamic obstacles).
        goal_pos: np.ndarray [x, y] center of goal region.
        goal_rect: dict with 'length', 'width', 'orientation' of goal rectangle.
        scenario_id: str identifier for this scenario.
        dt: float simulation timestep.

    Returns:
        dict of metric name → value (one row of the results CSV).
    """
    positions = controller.position_tracker
    states = controller.state_tracker
    signals = controller.signal_tracker
    total_steps = controller.stepped

    # CPA against the first dynamic obstacle
    dyn_obs = scenario.dynamic_obstacles[0] if scenario.dynamic_obstacles else None
    if dyn_obs is not None:
        cpa_dist, cpa_step = compute_cpa(positions, dyn_obs, total_steps)
        traffic_init = dyn_obs.initial_state
        traffic_init_x = traffic_init.position[0]
        traffic_init_y = traffic_init.position[1]
        traffic_init_orientation = traffic_init.orientation
    else:
        cpa_dist, cpa_step = float('inf'), 0
        traffic_init_x = traffic_init_y = traffic_init_orientation = float('nan')

    # Goal reached
    goal_reached, goal_step = check_goal_reached(
        positions, goal_pos,
        goal_rect['length'], goal_rect['width'], goal_rect['orientation'],
    )

    # Path length
    path_length = 0.0
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i - 1][0]
        dy = positions[i][1] - positions[i - 1][1]
        path_length += np.sqrt(dx * dx + dy * dy)

    # Final distance to goal
    if positions:
        final_pos = np.array(positions[-1][:2])
        final_dist = np.linalg.norm(final_pos - goal_pos)
    else:
        final_dist = float('nan')

    # State time distribution
    n = max(len(states), 1)
    time_s1 = sum(1 for s in states if s == 'WAYPOINT_REACHING') / n
    time_s2 = sum(1 for s in states if s == 'COLLISION_AVOIDANCE') / n
    time_s3 = sum(1 for s in states if s == 'CONSTANT_CONTROL') / n

    # Avoidance activations (S1 → S2 transitions)
    activations = 0
    for i in range(1, len(states)):
        if states[i] == 'COLLISION_AVOIDANCE' and states[i - 1] == 'WAYPOINT_REACHING':
            activations += 1

    # Max yaw rate
    max_yaw = 0.0
    for sig in signals:
        yr = abs(sig[1])
        if yr > max_yaw:
            max_yaw = yr

    # Encounter type classification
    if dyn_obs is not None and positions:
        ego_pos = np.array(positions[0][:2])
        ego_heading = positions[0][2]
        obs_pos = np.array([traffic_init_x, traffic_init_y])
        encounter = classify_encounter(
            ego_pos, ego_heading, obs_pos, traffic_init_orientation,
        )
    else:
        encounter = 'unknown'

    return {
        'scenario_id': scenario_id,
        'total_steps': total_steps,
        'goal_reached': goal_reached,
        'goal_reached_step': goal_step,
        'cpa_distance': cpa_dist,
        'cpa_step': cpa_step,
        'path_length': path_length,
        'final_dist_to_goal': final_dist,
        'time_s1_pct': time_s1,
        'time_s2_pct': time_s2,
        'time_s3_pct': time_s3,
        'avoidance_activations': activations,
        'max_yaw_rate': max_yaw,
        'traffic_init_x': traffic_init_x,
        'traffic_init_y': traffic_init_y,
        'traffic_init_orientation': traffic_init_orientation,
        'encounter_type': encounter,
    }


def compute_cpa(position_tracker, dynamic_obstacle, max_steps):
    """
    Compute closest point of approach between ego and a dynamic obstacle.

    Returns:
        (min_distance, step_of_cpa)
    """
    min_dist = float('inf')
    cpa_step = 0
    for t in range(min(len(position_tracker), max_steps)):
        obs_state = dynamic_obstacle.state_at_time(t)
        if obs_state is None:
            break
        ego_pos = np.array(position_tracker[t][:2])
        obs_pos = np.array(obs_state.position)
        dist = np.linalg.norm(ego_pos - obs_pos)
        if dist < min_dist:
            min_dist = dist
            cpa_step = t
    return min_dist, cpa_step


def check_goal_reached(position_tracker, goal_center, goal_length, goal_width, goal_orientation):
    """
    Check if any position in the trajectory falls inside the goal rectangle.

    Returns:
        (reached: bool, first_step_inside: int or None)
    """
    c = np.cos(-goal_orientation)
    s = np.sin(-goal_orientation)
    half_l = goal_length / 2.0
    half_w = goal_width / 2.0

    for t, pos in enumerate(position_tracker):
        dx = pos[0] - goal_center[0]
        dy = pos[1] - goal_center[1]
        local_x = c * dx - s * dy
        local_y = s * dx + c * dy
        if abs(local_x) <= half_l and abs(local_y) <= half_w:
            return True, t
    return False, None


def classify_encounter(ego_pos, ego_heading, traffic_pos, traffic_heading):
    """
    Classify encounter geometry based on initial relative positions and headings.

    Returns one of: 'head_on', 'crossing_from_port', 'crossing_from_starboard', 'overtaking'
    """
    heading_diff = (traffic_heading - ego_heading + np.pi) % (2 * np.pi) - np.pi

    if abs(heading_diff) > 2.8:  # ~160 deg — roughly opposing
        return 'head_on'
    if abs(heading_diff) < 0.5:  # ~30 deg — same direction
        return 'overtaking'

    bearing = np.arctan2(
        traffic_pos[1] - ego_pos[1],
        traffic_pos[0] - ego_pos[0],
    )
    relative_bearing = (bearing - ego_heading + np.pi) % (2 * np.pi) - np.pi

    if relative_bearing > 0:
        return 'crossing_from_port'
    return 'crossing_from_starboard'
