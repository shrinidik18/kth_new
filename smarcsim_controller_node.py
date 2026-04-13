#!/usr/bin/env python
import rospy
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import tf.transformations

class SMaRCSwarmController:
    """
    ROS Node for SMaRCSim: Bridges the Phase-Lag Kinematic law to high-fidelity 
    SAM AUV dynamics via a Hierarchical Controller (Velocity Governor + LOS Path Following).
    """
    def __init__(self):
        rospy.init_node('smarc_swarm_controller')
        
        # Swarm Parameters
        self.n_agents = 12
        self.radius = 5.0
        self.omega_0 = 0.5
        self.alpha = 0.75
        self.sigma = 0.6
        self.k_sync = 0.5
        self.surge_limit = 1.5
        
        # Threat Position
        self.att_x = 7.0
        self.att_y = 0.0
        self.phi_att = np.arctan2(self.att_y, self.att_x)
        
        # State tracking
        self.positions = {}
        self.yaws = {}
        self.pubs = []
        
        # Setup Pubs/Subs for SAM AUVs
        for i in range(self.n_agents):
            # Using standard SMaRCSim namespaces (e.g., /sam_0, /sam_1...)
            rospy.Subscriber(f'/sam_{i}/odom', Odometry, self.odom_cb, callback_args=i)
            pub = rospy.Publisher(f'/sam_{i}/cmd_vel', Twist, queue_size=10)
            self.pubs.append(pub)
            self.positions[i] = (self.radius, 0.0) # Init guess
            self.yaws[i] = 0.0
            
        self.rate = rospy.Rate(20) # 20 Hz control loop
        rospy.loginfo("SMaRCSim Swarm Controller Initialized.")
        
    def odom_cb(self, msg, agent_id):
        # Update ground-truth position and heading from SMaRCSim
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.positions[agent_id] = (x, y)
        
        q = msg.pose.pose.orientation
        euler = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.yaws[agent_id] = euler[2] # Yaw
        
    def get_angular_dist(self, a, b):
        return np.angle(np.exp(1j * (a - b)))
        
    def run(self):
        while not rospy.is_shutdown():
            angles = np.zeros(self.n_agents)
            radii = np.zeros(self.n_agents)
            
            # 1. Update Spatial Map (Sensing)
            for i in range(self.n_agents):
                x, y = self.positions[i]
                angles[i] = np.arctan2(y, x)
                radii[i] = np.hypot(x, y)
                
            # Sort to establish topological neighbors dynamically
            sort_idx = np.argsort(angles)
            sorted_angles = angles[sort_idx]
            inv_map = {orig_idx: sorted_idx for sorted_idx, orig_idx in enumerate(sort_idx)}
            
            for i in range(self.n_agents):
                curr_angle = angles[i]
                
                # --- UPPER LEVEL: Velocity Governor (The Phase-Lag Law) ---
                d_theta = self.get_angular_dist(curr_angle, self.phi_att)
                speed_scaling = 1 - self.alpha * np.exp(-(d_theta**2)/(2*self.sigma**2))
                
                topo_idx = inv_map[i]
                prev_idx = (topo_idx - 1) % self.n_agents
                next_idx = (topo_idx + 1) % self.n_agents
                
                ideal_gap = 2 * np.pi / self.n_agents
                err_prev = self.get_angular_dist(curr_angle, sorted_angles[prev_idx] + ideal_gap)
                err_next = self.get_angular_dist(curr_angle, sorted_angles[next_idx] - ideal_gap)
                
                sync_term = -self.k_sync * (err_prev + err_next)
                omega_cmd = self.omega_0 * speed_scaling + sync_term
                
                # Surge Command Configuration
                u_cmd = self.radius * omega_cmd
                u_cmd = np.clip(u_cmd, -self.surge_limit, self.surge_limit)
                
                # --- LOWER LEVEL: Line of Sight (LOS) Path Following ---
                # Target heading is tangent to the circle
                psi_target = curr_angle + np.pi/2
                heading_err = self.get_angular_dist(psi_target, self.yaws[i])
                
                # Cross-track error correction (pulls sub back to the exact radius)
                r_err = self.radius - radii[i]
                heading_correction = 0.5 * r_err 
                
                # Yaw Command Configuration
                r_cmd = 1.0 * heading_err + heading_correction
                r_cmd = np.clip(r_cmd, -0.6, 0.6) # Actuator cap on turning rate
                
                # Publish Dynamics to SMaRCSim
                msg = Twist()
                msg.linear.x = u_cmd
                msg.angular.z = r_cmd
                self.pubs[i].publish(msg)
                
            self.rate.sleep()

if __name__ == '__main__':
    try:
        ctrl = SMaRCSwarmController()
        ctrl.run()
    except rospy.ROSInterruptException:
        pass
