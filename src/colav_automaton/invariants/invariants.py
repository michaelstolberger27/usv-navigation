from hybrid_automaton import invariant
from hybrid_automaton.automaton_runtime_context import Context


@invariant
def is_goal_waypoint_invariant(ctx: Context) -> bool:
    if ctx.cfg["waypoints"] == 1:
        return True
    return False
    