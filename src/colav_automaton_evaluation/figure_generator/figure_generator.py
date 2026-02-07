import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

def plot_xy_position_over_time(x, aux_x_vertices, waypoints=None):
    timestamps = np.array([continuous_state[0] for continuous_state in x])
    continuous_state_values = np.array([continuous_state[1] for continuous_state in x])
    
    # Extract x and y positions (indices 0 and 1 of the state vector)
    x_positions = continuous_state_values[:, 0]
    y_positions = continuous_state_values[:, 1]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # --- Vehicle trajectory with time-based coloring ---
    # Create line segments
    points = np.array([x_positions, y_positions]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Create a LineCollection with colors based on timestamps
    norm = Normalize(vmin=timestamps.min(), vmax=timestamps.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm, linewidth=2)
    lc.set_array(timestamps[:-1])  # Color each segment by its start time
    
    line = ax.add_collection(lc)
    
    # Add colorbar
    cbar = fig.colorbar(line, ax=ax, label='Time (s)')
    
    # --- Mark starting position ---
    ax.plot(x_positions[0], y_positions[0], 
            marker='o', markersize=12, color='green', 
            markeredgecolor='darkgreen', markeredgewidth=2,
            label='Start Position', zorder=5)
    
    # --- Mark ending position ---
    ax.plot(x_positions[-1], y_positions[-1], 
            marker='s', markersize=12, color='red', 
            markeredgecolor='darkred', markeredgewidth=2,
            label='End Position', zorder=5)
    
    # --- Plot waypoints ---
    if waypoints is not None and len(waypoints) > 0:
        # Plot each waypoint individually
        for i, wp in enumerate(waypoints):
            if i == 0:
                # First waypoint gets the label for legend
                ax.plot(wp[0], wp[1],
                       marker='*', markersize=15, color='gold',
                       markeredgecolor='darkorange', markeredgewidth=2,
                       label='Goal Waypoints', zorder=4)
            else:
                ax.plot(wp[0], wp[1],
                       marker='*', markersize=15, color='gold',
                       markeredgecolor='darkorange', markeredgewidth=2,
                       zorder=4)
            
            # Number each waypoint
            ax.annotate(f'{i+1}', xy=(wp[0], wp[1]), 
                       xytext=(8, 8), textcoords='offset points',
                       fontsize=10, color='darkblue', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                                edgecolor='darkblue', alpha=0.8))
        
        # Draw connecting lines between waypoints
        waypoints_array = np.array([wp for wp in waypoints])
        ax.plot(waypoints_array[:, 0], waypoints_array[:, 1],
                linestyle='--', linewidth=1.5, alpha=0.4, color='orange',
                zorder=3)
    
    # --- Unsafe region (polygon exterior) ---
    verts = np.array(aux_x_vertices)
    if verts is not None and verts > 0: 
        closed_verts = np.vstack([verts, verts[0]])   # Close polygon
        unsafe_line = ax.plot(
            closed_verts[:, 0],
            closed_verts[:, 1],
            linewidth=2,
            linestyle='--',
            color='orange',
            label='Unsafe Region Boundary'
        )[0]
    
    # Formatting
    ax.set_title('Hybrid Automaton <v0.0.4> - XY Position & Unsafe Region', fontsize=16, y=1.02)
    ax.set_xlabel("X Position (m)", fontsize=12)
    ax.set_ylabel("Y Position (m)", fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Auto-scale to fit all data
    ax.autoscale()
    
    fig.tight_layout()
    return fig