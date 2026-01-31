import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import os
import time
import math

class HexapodStabilizeEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, render_mode=None):
        super(HexapodStabilizeEnv, self).__init__()
        self.render_mode = render_mode
        self.gui = (render_mode == "human")
        
        # Connect to PyBullet
        if self.gui:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        
        # 18 joints (revolute), normalized action -1 to 1
        self.n_joints = 18
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_joints,), dtype=np.float32)
        
        # Observation: 
        # Base quaternion (4), Base Ang Vel (3), Base Lin Vel (3), 
        # Joint Pos (18), Joint Vel (18)
        # Total = 4 + 3 + 3 + 18 + 18 = 46
        # Adding Base Position Z (height) might be useful, let's include pos (3)
        # Total = 49
        
        self.obs_dim = 3 + 4 + 3 + 3 + self.n_joints + self.n_joints
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        
        self.dt = 1./240.
        self.robotId = None
        self.joint_indices = []
        
        self._load_urdf()
        
    def _load_urdf(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir) # spiderbotv2
        urdf_path = os.path.join(project_root, "Hexapod-ROS", "crab_description", "hexapod_pybullet.urdf")
        
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF not found at: {urdf_path}")
            
        startPos = [0, 0, 0.2] # Start safely above ground
        startOrn = p.getQuaternionFromEuler([0, 0, 0])
        self.robotId = p.loadURDF(urdf_path, startPos, startOrn, useFixedBase=False)
        
        # Get Joint Indices
        self.joint_indices = []
        self.joint_limits_lower = []
        self.joint_limits_upper = []
        
        for i in range(p.getNumJoints(self.robotId)):
            info = p.getJointInfo(self.robotId, i)
            joint_type = info[2]
            if joint_type == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)
                self.joint_limits_lower.append(info[8])
                self.joint_limits_upper.append(info[9])
                
        if len(self.joint_indices) != 18:
            print(f"Warning: Expected 18 joints, found {len(self.joint_indices)}")
            
        self.joint_limits_lower = np.array(self.joint_limits_lower)
        self.joint_limits_upper = np.array(self.joint_limits_upper)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        
        # Reload robot to fully reset state
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        urdf_path = os.path.join(project_root, "Hexapod-ROS", "crab_description", "hexapod_pybullet.urdf")
        
        # Randomize start slightly
        start_z = 0.2 + np.random.uniform(-0.01, 0.01)
        start_rpy = np.random.uniform(-0.05, 0.05, size=3)
        start_orn = p.getQuaternionFromEuler(start_rpy)
        
        self.robotId = p.loadURDF(urdf_path, [0, 0, start_z], start_orn, useFixedBase=False)
        
        # Reset joints
        for i, j_idx in enumerate(self.joint_indices):
            # Start near 0 but with small noise
            p.resetJointState(self.robotId, j_idx, np.random.uniform(-0.05, 0.05))
            
        # Run a few steps to let physics settle
        for _ in range(10):
            p.stepSimulation()
            
        return self._get_obs(), {}
        
    def step(self, action):
        # Scale action from [-1, 1] to [limit_lower, limit_upper]
        # Or simple scaling factor if limits are wide. 
        # Using direct interpolation for now.
        
        # Clip action
        action = np.clip(action, -1, 1)
        
        # Convert -1..1 to real joint angles
        # target_pos = lower + (action + 1) * 0.5 * (upper - lower)
        # However, to learn stability, maybe better to give target angles relative to current or 0
        # Let's use direct position control mapping to limits
        
        target_positions = self.joint_limits_lower + (action + 1) * 0.5 * (self.joint_limits_upper - self.joint_limits_lower)
        
        # Apply Motor Control
        p.setJointMotorControlArray(
            self.robotId, 
            self.joint_indices, 
            p.POSITION_CONTROL, 
            targetPositions=target_positions, 
            forces=[1.5] * self.n_joints 
        )
        
        p.stepSimulation()
        if self.gui:
            time.sleep(self.dt)
            
        obs = self._get_obs()
        reward = self._compute_reward()
        
        # Termination conditions
        terminated = False
        pos, orn = p.getBasePositionAndOrientation(self.robotId)
        
        # Fall detection: Body too low
        if pos[2] < 0.08:
            terminated = True
            
        # Flip detection: Orientation too extreme
        euler = p.getEulerFromQuaternion(orn)
        if abs(euler[0]) > 0.7 or abs(euler[1]) > 0.7: # ~40 degrees
            terminated = True
            
        truncated = False # Can handle max steps in Gym wrapper
        
        return obs, reward, terminated, truncated, {}
        
    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.robotId)
        lin_vel, ang_vel = p.getBaseVelocity(self.robotId)
        
        joint_states = p.getJointStates(self.robotId, self.joint_indices)
        joint_pos = [s[0] for s in joint_states]
        joint_vel = [s[1] for s in joint_states]
        
        obs = np.concatenate([
            pos, orn, lin_vel, ang_vel, joint_pos, joint_vel
        ])
        return obs.astype(np.float32)
        
    def _compute_reward(self):
        pos, orn = p.getBasePositionAndOrientation(self.robotId)
        lin_vel, ang_vel = p.getBaseVelocity(self.robotId)
        euler = p.getEulerFromQuaternion(orn) # Roll, Pitch, Yaw
        
        # 1. Survival Bonus
        reward = 1.0
        
        # 2. Stability Penalties (Minimize Roll and Pitch)
        reward -= abs(euler[0]) * 10.0 # Roll
        reward -= abs(euler[1]) * 10.0 # Pitch
        
        # 3. Height Target (e.g., target 0.15m - 0.2m)
        target_h = 0.15
        reward -= abs(pos[2] - target_h) * 5.0
        
        # 4. Energy/Smoothness (Minimize Joint Velocities and Torques implied)
        # Use Angular Velocity of base as proxy for jitter
        reward -= np.linalg.norm(ang_vel) * 0.1
        
        # 5. Drift Penalty (Stay near [0,0])
        # reward -= (pos[0]**2 + pos[1]**2) * 0.5
        
        return reward
        
    def close(self):
        p.disconnect()
