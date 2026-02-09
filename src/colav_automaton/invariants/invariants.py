from hybrid_automaton.definition import invariant
from hybrid_automaton import RuntimeContext


@invariant
def is_goal_waypoint_invariant(ctx: RuntimeContext) -> bool:
    """True when only the goal waypoint remains on the stack (no active V1s)."""
    return len(ctx.configuration["waypoints"]) == 1
