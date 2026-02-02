"""
Unsafe set computation and line-of-sight geometry utilities.

Handles convex hull generation, LOS cone creation, and multi-obstacle
collision region computation using the colav_unsafe_set API.
"""

from colav_automaton.unsafe_sets.unsafe_sets import (
    generate_dynamic_unsafe_set_from_vertices,
    get_unsafe_set_vertices,
    create_los_cone,
    compute_unified_unsafe_region,
)

__all__ = [
    'generate_dynamic_unsafe_set_from_vertices',
    'get_unsafe_set_vertices',
    'create_los_cone',
    'compute_unified_unsafe_region',
]
