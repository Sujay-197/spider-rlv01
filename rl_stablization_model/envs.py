import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import os
import time
import math
import sys

# Context manager to suppress PyBullet URDF loading spam (FD level)
class SuppressStdout:
    def __enter__(self):
        # Open a null file
        self.null_fd = os.open(os.devnull, os.O_RDWR)
        
        # Save the original stdout/stderr file descriptors
        self.save_stdout = os.dup(1)
        self.save_stderr = os.dup(2)
        
        # Redirect stdout/stderr to null
        os.dup2(self.null_fd, 1)
        os.dup2(self.null_fd, 2)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original stdout/stderr
        os.dup2(self.save_stdout, 1)
        os.dup2(self.save_stderr, 2)
        
        # Close the duplicated descriptors and null handle
        os.close(self.save_stdout)
        os.close(self.save_stderr)
        os.close(self.null_fd)

class HexapodStabilizeEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, render_mode=None, max_steps=1000):
        super(HexapodStabilizeEnv, self).__init__()
        self.render_mode = render_mode
        self.gui = (render_mode == "human")
        self.max_steps = max_steps
        self.current_step_count = 0
        
        if self.gui:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        self.n_joints = 18
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_joints,), dtype=np.float32)
        self.obs_dim = 3 + 4 + 3 + 3 + self.n_joints + self.n_joints
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        
        self.dt = 1./240.
        
        # Load once here
        with SuppressStdout():
            self._setup_simulation()
        
    def _setup_simulation(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        urdf_path = os.path.join(project_root, "Hexapod-ROS", "crab_description", "hexapod_pybullet.urdf")
        
        self.robotId = p.loadURDF(urdf_path, [0, 0, 0.2], useFixedBase=False)
        
        self.joint_indices = []
        self.joint_limits_lower = []
        self.joint_limits_upper = []
        for i in range(p.getNumJoints(self.robotId)):
            info = p.getJointInfo(self.robotId, i)
            if info[2] == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)
                self.joint_limits_lower.append(info[8])
                self.joint_limits_upper.append(info[9])
        
        self.joint_limits_lower = np.array(self.joint_limits_lower)
        self.joint_limits_upper = np.array(self.joint_limits_upper)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step_count = 0
        
        # Reset position and orientation without reloading URDF
        start_z = 0.2 + np.random.uniform(-0.01, 0.01)
        start_rpy = np.random.uniform(-0.05, 0.05, size=3)
        start_orn = p.getQuaternionFromEuler(start_rpy)
        
        p.resetBasePositionAndOrientation(self.robotId, [0, 0, start_z], start_orn)
        p.resetBaseVelocity(self.robotId, [0,0,0], [0,0,0])
        
        for j_idx in self.joint_indices:
            p.resetJointState(self.robotId, j_idx, np.random.uniform(-0.05, 0.05))
            
        return self._get_obs(), {}

    def step(self, action):
        self.current_step_count += 1
        action = np.clip(action, -1, 1)
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
        reward = self._compute_reward()
        
        terminated = False
        pos, orn = p.getBasePositionAndOrientation(self.robotId)
        if pos[2] < 0.08: terminated = True
        euler = p.getEulerFromQuaternion(orn)
        if abs(euler[0]) > 0.7 or abs(euler[1]) > 0.7: terminated = True
        
        truncated = False
        if self.current_step_count >= self.max_steps:
            truncated = True
            
        return obs, reward, terminated, truncated, {}
        
    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.robotId)
        lin_vel, ang_vel = p.getBaseVelocity(self.robotId)
        joint_states = p.getJointStates(self.robotId, self.joint_indices)
        joint_pos = [s[0] for s in joint_states]
        joint_vel = [s[1] for s in joint_states]
        return np.concatenate([pos, orn, lin_vel, ang_vel, joint_pos, joint_vel]).astype(np.float32)
        
    def _compute_reward(self):
        pos, orn = p.getBasePositionAndOrientation(self.robotId)
        ang_vel = p.getBaseVelocity(self.robotId)[1]
        euler = p.getEulerFromQuaternion(orn)
        reward = 1.0 - (abs(euler[0]) * 5.0) - (abs(euler[1]) * 5.0) - (abs(pos[2] - 0.15) * 5.0) - (np.linalg.norm(ang_vel) * 0.1)
        return reward
        
    def close(self):
        p.disconnect()