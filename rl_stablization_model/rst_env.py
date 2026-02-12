import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import numpy as np
import math
import time
from envs import HexapodStabilizeEnv

class HexapodReStabilizeEnv(HexapodStabilizeEnv):
    """
    Restabilization Environment (v2):
    - Spawns robot in random orientations (including upside-down)
    - Goal: Learn to return to upright equilibrium pose
    - Self-righting behavior
    - NO premature termination for being upside down or on the ground
    """
    def __init__(self, render_mode=None, max_steps=1000):
        super().__init__(render_mode, max_steps)
        
        # Target equilibrium state
        self.target_height = 0.15
        self.target_roll = 0.0
        self.target_pitch = 0.0
        
    def reset(self, seed=None, options=None):
        # We invoke super().reset() to handle seed/basic setup if needed, 
        # but we largely overwrite the physics reset part.
        # However, HexapodStabilizeEnv.reset() calls _get_obs(), so we need to be careful.
        # Let's just call super to ensure member vars are set, then override physics.
        super(HexapodStabilizeEnv, self).reset(seed=seed)
        self.current_step_count = 0
        
        # Random orientation - full range to include upside-down
        start_z = np.random.uniform(0.1, 0.4) # Slightly higher to allow fall
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
            p.resetJointState(self.robotId, j_idx, np.random.uniform(-0.5, 0.5))
            
        return self._get_obs(), {}
    
    def step(self, action):
        self.current_step_count += 1
        
        # 1. Action Processing
        action = np.clip(action, -1, 1)
        target_positions = self.joint_limits_lower + (action + 1) * 0.5 * (self.joint_limits_upper - self.joint_limits_lower)
        
        # 2. Physics Step
        p.setJointMotorControlArray(
            self.robotId, 
            self.joint_indices, 
            p.POSITION_CONTROL, 
            targetPositions=target_positions, 
            forces=[1.5] * self.n_joints 
        )
        
        p.stepSimulation()
        if self.gui: time.sleep(self.dt)
            
        # 3. Observation & Reward
        obs = self._get_obs()
        reward = self._compute_reward()
        
        # 4. Termination Logic (RELAXED)
        terminated = False
        pos, orn = p.getBasePositionAndOrientation(self.robotId)
        
        # Only terminate if we fall into the abyss (glitch check)
        # We DO NOT terminate for low z (on ground) or high tilt (upside down)
        # because recovering from those states is the whole point.
        if pos[2] < -0.5: 
            terminated = True
        
        # 5. Truncation
        truncated = False
        if self.current_step_count >= self.max_steps:
            truncated = True
            
        return obs, reward, terminated, truncated, {}

    def _compute_reward(self):
        pos, orn = p.getBasePositionAndOrientation(self.robotId)
        lin_vel, ang_vel = p.getBaseVelocity(self.robotId)
        euler = p.getEulerFromQuaternion(orn)
        
        # --- 1. Upright Orientation Reward (Primary) ---
        # Strong reward for being upright (roll and pitch near 0)
        roll_error = abs(euler[0])
        pitch_error = abs(euler[1])
        r_upright = math.exp(-2.0 * (roll_error + pitch_error)) # Slower decay allowing some error initially
        
        # --- 2. Height Reward ---
        # Reward for being at target height
        height_error = abs(pos[2] - self.target_height)
        r_height = math.exp(-20.0 * height_error**2)
        
        # --- 3. Stability Reward ---
        # Penalize high angular velocity (encourage settling)
        ang_vel_magnitude = np.linalg.norm(ang_vel)
        r_stability = math.exp(-1.0 * ang_vel_magnitude)
        
        # --- 4. Energy Penalty ---
        # Penalize excessive joint movements
        joint_states = p.getJointStates(self.robotId, self.joint_indices)
        joint_vels = [s[1] for s in joint_states]
        p_energy = np.mean(np.square(joint_vels)) * 0.01
        
        # --- 5. Survival Bonus ---
        r_alive = 0.5
        
        # --- 6. Bonus for being very upright ---
        # Extra reward when nearly perfect to encourage fine adjustments
        r_bonus = 0.0
        if roll_error < 0.2 and pitch_error < 0.2:
            r_bonus += 1.0
            if pos[2] > 0.1: # Also actually up
                r_bonus += 1.0
        
        total_reward = r_upright + r_height + r_stability + r_alive + r_bonus - p_energy
        
        return total_reward
