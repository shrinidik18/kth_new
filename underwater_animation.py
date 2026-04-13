import numpy as np
#import matplotlib.subplots as subplots
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
N_AGENTS = 12
RING_RADIUS = 5.0
OMEGA_0 = 0.5       # Nominal rotation speed (rad/s)
ALPHA_FIXED = 0.75  # Braking intensity
SIGMA = 0.6         # Spatial width of crowd
K_SYNC = 0.5        # Synchronization gain
SURGE_LIMIT = 1.5   # m/s max velocity
TAU_SYSTEM = 2.0    # Time constant for thruster lag (added mass)

# Fixed Attacker Position
ATT_X, ATT_Y = 7.0, 0.0
phi_att = np.arctan2(ATT_Y, ATT_X)

np.random.seed(42)
angles = np.sort(np.random.rand(N_AGENTS) * 2 * np.pi)
u_actual = np.ones(N_AGENTS) * (OMEGA_0 * RING_RADIUS)

def get_angular_dist(a, b):
    return np.angle(np.exp(1j * (a - b)))

def update(frame):
    global angles, u_actual
    dt = 0.05
    new_angles = np.zeros_like(angles)
    new_u = np.zeros_like(u_actual)
    
    for i in range(N_AGENTS):
        # 1. Phase-Lag 
        d_theta = get_angular_dist(angles[i], phi_att)
        speed_scaling = 1 - ALPHA_FIXED * np.exp(-(d_theta**2) / (2 * SIGMA**2))
        
        # 2. Decentralized Spacing
        prev_idx = (i - 1) % N_AGENTS
        next_idx = (i + 1) % N_AGENTS
        ideal_gap = 2 * np.pi / N_AGENTS
        
        err_prev = get_angular_dist(angles[i], angles[prev_idx] + ideal_gap)
        err_next = get_angular_dist(angles[i], angles[next_idx] - ideal_gap)
        sync_term = -K_SYNC * (err_prev + err_next)
        
        # 3. Dynamic Mapping
        omega_cmd = OMEGA_0 * speed_scaling + sync_term
        u_cmd = RING_RADIUS * omega_cmd
        
        # Saturation Limit (Thrust limits)
        u_cmd = np.clip(u_cmd, -SURGE_LIMIT, SURGE_LIMIT)
        
        # Thruster Lag / Added Mass (First order dynamics)
        du = (u_cmd - u_actual[i]) / TAU_SYSTEM
        new_u[i] = u_actual[i] + du * dt
        
        # Final angle update using actual velocity
        actual_omega = new_u[i] / RING_RADIUS
        new_angles[i] = (angles[i] + actual_omega * dt) % (2 * np.pi)
        
    angles = np.sort(new_angles)
    u_actual = new_u
    
    x = RING_RADIUS * np.cos(angles)
    y = RING_RADIUS * np.sin(angles)
    pts.set_data(x, y)
    return pts,

# Visualization Setup
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(-8, 8); ax.set_ylim(-8, 8)
ax.set_aspect('equal')
ax.add_artist(plt.Circle((0, 0), RING_RADIUS, color='black', fill=False, ls=':', alpha=0.3))

pts, = ax.plot([], [], 'go', ms=8, label='SAM AUVs (Hydrodynamic)')
ax.plot(ATT_X, ATT_Y, 'rx', ms=12, mew=3, label='Fixed Threat')
ax.set_title("Underwater Simulation (Added Mass & Drag)")
ax.legend()

ani = FuncAnimation(fig, update, frames=400, interval=40, blit=True)
plt.show()