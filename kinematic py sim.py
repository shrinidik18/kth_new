import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
N_AGENTS = 12
RING_RADIUS = 5
OMEGA_0 = 1.0       # Nominal rotation speed
ALPHA_FIXED = 0.75  # Braking intensity (how much they slow down)
SIGMA = 0.6         # Spatial width of the defensive crowd
K_SYNC = 0.5        # Spring-like spacing strength

# Fixed Attacker Position (Outside the ring at 0 degrees)
ATT_X, ATT_Y = 7.0, 0.0
phi_att = np.arctan2(ATT_Y, ATT_X)

# Initialize agents with random spacing to show convergence
angles = np.sort(np.random.rand(N_AGENTS) * 2 * np.pi)

def get_angular_dist(a, b):
    return np.angle(np.exp(1j * (a - b)))

def update(frame):
    global angles
    dt = 0.05
    new_angles = np.zeros_like(angles)
    
    for i in range(N_AGENTS):
        # 1. Phase-Lag (Slowing down at the attacker's angle)
        d_theta = get_angular_dist(angles[i], phi_att)
        # The 'scaling' is 1.0 everywhere except near phi_att
        speed_scaling = 1 - ALPHA_FIXED * np.exp(-(d_theta**2) / (2 * SIGMA**2))
        
        # 2. Decentralized Spacing (Consensus)
        # Prev and Next neighbors in the array
        prev_idx = (i - 1) % N_AGENTS
        next_idx = (i + 1) % N_AGENTS
        
        ideal_gap = 2 * np.pi / N_AGENTS
        
        # Force agents to maintain the ideal gap relative to neighbors
        err_prev = get_angular_dist(angles[i], angles[prev_idx] + ideal_gap)
        err_next = get_angular_dist(angles[i], angles[next_idx] - ideal_gap)
        sync_term = -K_SYNC * (err_prev + err_next)

        # 3. Final Dynamics
        omega_i = OMEGA_0 * speed_scaling + sync_term
        new_angles[i] = (angles[i] + omega_i * dt) % (2 * np.pi)
        
    angles = np.sort(new_angles) # Keep indices ordered for neighbor logic
    
    # Update Plot
    x = RING_RADIUS * np.cos(angles)
    y = RING_RADIUS * np.sin(angles)
    pts.set_data(x, y)
    return pts,

# Visualization Setup
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(-8, 8); ax.set_ylim(-8, 8)
ax.set_aspect('equal')
ax.add_artist(plt.Circle((0, 0), RING_RADIUS, color='black', fill=False, ls=':', alpha=0.3))

# Plot elements
pts, = ax.plot([], [], 'go', ms=8, label='AUVs')
ax.plot(ATT_X, ATT_Y, 'rx', ms=12, mew=3, label='Fixed Threat')

ax.set_title("Steady-State Shielding (Stationary Attacker)")
ax.legend()

ani = FuncAnimation(fig, update, frames=200, interval=40, blit=True)
plt.show()

# --- Plot Spatial Density Plot ---
# Calculate gaps to next neighbor
gaps = np.zeros(N_AGENTS)
for i in range(N_AGENTS):
    next_idx = (i + 1) % N_AGENTS
    # Angular distance to the next neighbor
    gap = (angles[next_idx] - angles[i]) % (2 * np.pi)
    gaps[i] = gap

# Local Density is 1 / distance to neighbor
density = 1.0 / gaps

# Shift angles to [-pi, pi] to center the attacker at 0
shifted_angles = (angles + np.pi) % (2 * np.pi) - np.pi

# Sort for plotting
sort_idx = np.argsort(shifted_angles)
sorted_angles = shifted_angles[sort_idx]
sorted_density = density[sort_idx]

plt.figure(figsize=(8, 5))
plt.plot(sorted_angles, sorted_density, 'b-o', lw=2)
plt.axvline(phi_att, color='r', linestyle='--', label='Attacker Location')
plt.xlabel("Angular Position (θ)")
plt.ylabel("Local Density (1 / distance to neighbor)")
plt.title("Spatial Density Plot at Steady-State")
plt.legend()
plt.grid(True)
plt.show()