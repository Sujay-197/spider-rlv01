import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import numpy as np
import math
from envs import HexapodStabilizeEnv

class HexapodRestabilizeEnv(HexapodStabilizeEnv):
    """
    Restabilization Environment:
    - Spawns robot in random orientations (including upside-down)
    - Goal: Learn to return to upright equilibrium pose
    - Self-righting behavior
    """
    def __init__(self, render_mode=None, max_steps=1000):
        super().__init__(render_mode, max_steps)
        
        # Target equilibrium state
        self.target_height = 0.15
        self.target_roll = 0.0
        self.target_pitch = 0.0
        
    def reset(self, seed=None, options=None):
        super(HexapodStabilizeEnv, self).reset(seed=seed)
        self.current_step_count = 0
        
        # Random orientation - full range to include upside-down
        start_z = np.random.uniform(0.1, 0.3)
        start_roll = np.random.uniform(-np.pi, np.pi)
        start_pitch = np.random.uniform(-np.pi, np.pi)
        start_yaw = np.random.uniform(-np.pi, np.pi)
        start_rpy = [start_roll, start_pitch, start_yaw]
        start_orn = p.getQuaternionFromEuler(start_rpy)
        
        p.resetBasePositionAndOrientation(self.robotId, [0, 0, start_z], start_orn)
        
        # Random initial velocities to make task harder
        lin_vel = np.random.uniform(-0.1, 0.1, size=3)
        ang_vel = np.random.uniform(-0.2, 0.2, size=3)
        p.resetBaseVelocity(self.robotId, lin_vel, ang_vel)
        
        # Random joint positions
        for j_idx in self.joint_indices:
            p.resetJointState(self.robotId, j_idx, np.random.uniform(-0.3, 0.3))
            
        return self._get_obs(), {}
    
    def _compute_reward(self):
        pos, orn = p.getBasePositionAndOrientation(self.robotId)
        lin_vel, ang_vel = p.getBaseVelocity(self.robotId)
        euler = p.getEulerFromQuaternion(orn)
        
        # --- 1. Upright Orientation Reward (Primary) ---
        # Strong reward for being upright (roll and pitch near 0)
        roll_error = abs(euler[0])
        pitch_error = abs(euler[1])
        r_upright = math.exp(-10.0 * (roll_error + pitch_error))
        
        # --- 2. Height Reward ---
        # Reward for being at target height
        height_error = abs(pos[2] - self.target_height)
        r_height = math.exp(-20.0 * height_error**2)
        
        # --- 3. Stability Reward ---
        # Penalize high angular velocity (encourage settling)
        ang_vel_magnitude = np.linalg.norm(ang_vel)
        r_stability = math.exp(-5.0 * ang_vel_magnitude)
        
        # --- 4. Energy Penalty ---
        # Penalize excessive joint movements
        joint_states = p.getJointStates(self.robotId, self.joint_indices)
        joint_vels = [s[1] for s in joint_states]
        p_energy = np.mean(np.square(joint_vels)) * 0.05
        
        # --- 5. Survival Bonus ---
        r_alive = 0.1
        
        # --- 6. Bonus for being very upright ---
        # Extra reward when nearly perfect
        if roll_error < 0.1 and pitch_error < 0.1:
            r_bonus = 2.0
        else:
            r_bonus = 0.0
        
        total_reward = r_upright + r_height + r_stability + r_alive + r_bonus - p_energy
        
        return total_reward
