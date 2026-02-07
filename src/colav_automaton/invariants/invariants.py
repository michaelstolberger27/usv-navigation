from hybrid_automaton import invariant
from hybrid_automaton.automaton_runtime_context import Context


@invariant
def is_goal_waypoint_invariant(ctx: Context) -> bool:
    """True when only the goal waypoint remains on the stack (no active V1s)."""
    return len(ctx.cfg["waypoints"]) == 1
    