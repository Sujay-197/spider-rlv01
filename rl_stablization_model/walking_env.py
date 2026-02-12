import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import os
import time
import math
import sys
from envs import HexapodStabilizeEnv, SuppressStdout

class HexapodWalkEnv(HexapodStabilizeEnv):
    def __init__(self, render_mode=None, max_steps=1000):
        # We need to initialize the base class but override observation space
        # So we repeat some init logic or modify after super().__init__
        super().__init__(render_mode, max_steps)
        
        # Target Velocity
        self.target_vel_x = 0.35
        self.target_height = 0.25
        
        # Add Previous Action to observation (smoothing)
        # Old obs_dim = 49
        # New obs_dim = 49 + 18 (prev_action) = 67
        self.obs_dim = 49 + self.n_joints
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        
        self.last_action = np.zeros(self.n_joints)
        
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed, options)
        self.last_action = np.zeros(self.n_joints)
        
        # Append initial last_action (zeros) to obs
        # Note: super().reset() calls self._get_obs() which we override below, 
        # so it should already have the right shape if we do it right.
        return obs, info
        
    def step(self, action):
        self.current_step_count += 1
        
        # Action Smoothing / Clipping
        action = np.clip(action, -1, 1)
        
        # Update last action
        self.last_action = action
        
        # Apply Motor Control
        target_positions = self.joint_limits_lower + (action + 1) * 0.5 * (self.joint_limits_upper - self.joint_limits_lower)
        p.setJointMotorControlArray(
            self.robotId, 
            self.joint_indices, 
            p.POSITION_CONTROL, 
            targetPositions=target_positions, 
            forces=[1.5] * self.n_joints 
        )
        
        p.stepSimulation()
        if self.gui: time.sleep(self.dt)
            
        obs = self._get_obs()
        reward = self._compute_reward(action)
        
        terminated = False
        pos, orn = p.getBasePositionAndOrientation(self.robotId)
        
        # Fall detection (too low) or Extreme tilt
        if pos[2] < 0.05: terminated = True
        euler = p.getEulerFromQuaternion(orn)
        if abs(euler[0]) > 0.8 or abs(euler[1]) > 0.8: terminated = True
        
        truncated = False
        if self.current_step_count >= self.max_steps:
            truncated = True
            
        return obs, reward, terminated, truncated, {}
        
    def _get_obs(self):
        # Get base obs
        base_obs = super()._get_obs() # 49 dims
        # Append last action
        obs = np.concatenate([base_obs, self.last_action])
        return obs.astype(np.float32)
        
    def _compute_reward(self, action):
        pos, orn = p.getBasePositionAndOrientation(self.robotId)
        lin_vel, ang_vel = p.getBaseVelocity(self.robotId)
        euler = p.getEulerFromQuaternion(orn)
        
        # --- 1. Velocity Tracking Reward ---
        # Reward tracking target speed, Gaussian kernel
        v_x = lin_vel[0]
        r_tracking = math.exp(-2.0 * (v_x - self.target_vel_x)**2)
        
        # --- 2. Upright / Orientation Reward ---
        # Strongly penalize Roll and Pitch
        r_upright = math.exp(-5.0 * (abs(euler[0]) + abs(euler[1])))
        
        # --- 3. Height Reward --- 
        # Encourage Torso to stay at target height
        r_height = math.exp(-50.0 * (pos[2] - self.target_height)**2)
        
        # --- 4. Penalties ---
        # Drift (Y-axis velocity)
        p_drift = abs(lin_vel[1]) * 1.0
        
        # Energy / Torque (Action magnitude)
        p_energy = np.mean(np.square(action)) * 0.1
        
        # Torso Contact Penalty (Anti-Cheating)
        # Check if thorax link is in contact with anything
        # Link Index -1 is base_link (thorax usually)
        contact_points = p.getContactPoints(bodyA=self.robotId, linkIndexA=-1)
        # Filter for contacts with ground (bodyB != robotId)
        ground_contacts = [c for c in contact_points if c[2] != self.robotId]
        
        p_contact = 0.0
        if len(ground_contacts) > 0:
            p_contact = 2.0 # Huge penalty if body drags
            
        # --- 4. Survival Bonus ---
        r_alive = 0.5
        
        # --- 5. Gait / Joint Movement Incentive ---
        # Encourage non-zero joint velocity (prevent freezing in good pose)
        # But allow efficiency. This is tricky. 
        # Usually Tracking + Energy penalty is enough.
        # If we really want to enforce legged motion, p_contact is key.
        
        # Usually Tracking + Energy penalty is enough.
        # If we really want to enforce legged motion, p_contact is key.
        
        total_reward = r_tracking + r_upright + r_alive + r_height - p_drift - p_contact - p_energy
        
        return total_reward
