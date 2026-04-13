import numpy as np
import matplotlib.pyplot as plt

# Parameters
N_AGENTS = 12
RING_RADIUS = 5.0
OMEGA_0 = 0.2       # Nominal rotation speed (rad/s) -> nominal u = 1.0 m/s
ALPHA_FIXED = 0.75  # Braking intensity
SIGMA = 0.6         # Spatial width of crowd
K_SYNC = 0.5        # Synchronization gain
SURGE_LIMIT = 1.5   # m/s max velocity
TAU_SYSTEM = 2.0    # Time constant for thruster lag and added mass (seconds)

DT = 0.05
T_MAX = 60.0
STEPS = int(T_MAX / DT)

# Attacker
ATT_X, ATT_Y = 7.0, 0.0
phi_att = np.arctan2(ATT_Y, ATT_X)

def get_angular_dist(a, b):
    return np.angle(np.exp(1j * (a - b)))

def simulate_case(mode="ideal"):
    # mode: 'ideal', 'hydro', 'hydro_ff'
    
    np.random.seed(42) # Consistent random starting positions
    angles = np.sort(np.random.rand(N_AGENTS) * 2 * np.pi)
    
    # Velocity state for hydro cases
    u_actual = np.ones(N_AGENTS) * (OMEGA_0 * RING_RADIUS) 
    
    history_density = []
    history_time = []
    
    for step in range(STEPS):
        new_angles = np.zeros_like(angles)
        new_u = np.zeros_like(u_actual)
        
        for i in range(N_AGENTS):
            # Phase-Lag calculation
            d_theta = get_angular_dist(angles[i], phi_att)
            speed_scaling = 1 - ALPHA_FIXED * np.exp(-(d_theta**2) / (2 * SIGMA**2))
            
            # Synchronization calculation
            prev_idx = (i - 1) % N_AGENTS
            next_idx = (i + 1) % N_AGENTS
            ideal_gap = 2 * np.pi / N_AGENTS
            
            err_prev = get_angular_dist(angles[i], angles[prev_idx] + ideal_gap)
            err_next = get_angular_dist(angles[i], angles[next_idx] - ideal_gap)
            sync_term = -K_SYNC * (err_prev + err_next)
            
            # The nominal angular velocity commanded
            omega_cmd = OMEGA_0 * speed_scaling + sync_term
            
            if mode == "ideal":
                # Instantaneous kinematic update
                new_angles[i] = (angles[i] + omega_cmd * DT) % (2 * np.pi)
            else:
                # Dynamic mapping to surge
                u_cmd = RING_RADIUS * omega_cmd
                
                # Feed-forward term
                u_ff = 0.0
                if mode == "hydro_ff":
                    # Simple feed-forward proportional to spatial derivative of command
                    # to "pre-brake" before reaching the attacker
                    d_theta_ff = get_angular_dist(angles[i] + 0.5, phi_att)
                    scaling_ff = 1 - ALPHA_FIXED * np.exp(-(d_theta_ff**2) / (2 * SIGMA**2))
                    u_ff = RING_RADIUS * OMEGA_0 * (scaling_ff - speed_scaling) * 2.0
                
                u_cmd_total = u_cmd + u_ff
                
                # Actuator Saturation
                u_cmd_total = np.clip(u_cmd_total, -SURGE_LIMIT, SURGE_LIMIT)
                
                # Thruster Lag / Added Mass / Drag as a first-order system
                du = (u_cmd_total - u_actual[i]) / TAU_SYSTEM
                new_u[i] = u_actual[i] + du * DT
                
                # Update angle based on actual surge speed
                actual_omega = new_u[i] / RING_RADIUS
                new_angles[i] = (angles[i] + actual_omega * DT) % (2 * np.pi)
                
        angles = np.sort(new_angles)
        if mode != "ideal":
            u_actual = new_u
            
        # Metric calculation (Density Settling)
        # Calculate Peak Density
        gaps = np.zeros(N_AGENTS)
        for i in range(N_AGENTS):
            next_idx = (i + 1) % N_AGENTS
            gaps[i] = (angles[next_idx] - angles[i]) % (2 * np.pi)
            
        local_density = 1.0 / gaps
        peak_density = np.max(local_density)
        
        history_density.append(peak_density)
        history_time.append(step * DT)
        
    return history_time, history_density

# Run all cases
t_ideal, d_ideal = simulate_case("ideal")
t_hydro, d_hydro = simulate_case("hydro")
t_hydro_ff, d_hydro_ff = simulate_case("hydro_ff")

# Plot Sim-to-Real Gap Analysis
plt.figure(figsize=(10, 6))
plt.plot(t_ideal, d_ideal, 'g--', lw=2, label="Kinematic (Ideal)")
plt.plot(t_hydro, d_hydro, 'r-', lw=2, label="Hydrodynamic (Added Mass & Saturation)")
plt.plot(t_hydro_ff, d_hydro_ff, 'b-', lw=2, label="Hydrodynamic + Feed-Forward")

plt.axhline(np.max(d_ideal[-100:]), color='gray', linestyle=':', label='Target Steady-State Density')

plt.title("Sim-to-Real Gap Analysis: Density Settling Time")
plt.xlabel("Wait Time (seconds)")
plt.ylabel("Peak Crowd Density (1/rad)")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("sim_to_real_gap.png")
plt.show()
