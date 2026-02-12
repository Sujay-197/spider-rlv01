import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import numpy as np
import math
import sys
import os

# Add parent directory to path to import envs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs import HexapodStabilizeEnv

class LegLiftEnv(HexapodStabilizeEnv):
    """
    Atomic Task 1: Lift a specific leg or legs while maintaining stability.
    Goal: Lift Target Leg Tip > 0.05m while keeping others < 0.02m.
    """
    def __init__(self, render_mode=None, max_steps=1000, target_leg_index=None):
        """
        target_leg_index: If set, fixed target. If None, random per episode.
        """
        super().__init__(render_mode, max_steps)
        self.target_leg_index_arg = target_leg_index
        self.current_target_leg = 0
        
        # Extend Observation Space: Base (49) + One-Hot Target (6) = 55
        # Base obs is: pos(3)+orn(4)+lin(3)+ang(3)+j_pos(18)+j_vel(18) = 49
        self.obs_dim = 49 + 6
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        
        # Identify Leg Tips
        self.leg_tips = self._find_leg_tips()
        print(f"LegLiftEnv: Identified Leg Tips at Links: {self.leg_tips}")
        if len(self.leg_tips) != 6:
            print("WARNING: Did not find exactly 6 leg tips. Check URDF logic.")

    def _find_leg_tips(self):
        """
        Dynamically find the link indices for the tips (foot) of the legs.
        Heuristic: Look for 'tibia_foot' or similar in link names.
        """
        tips = []
        num_joints = p.getNumJoints(self.robotId)
        # We want to ensure consistent order: R1, R2, R3, L1, L2, L3 or similar.
        # The URDF order seems to be R1, R2, R3, L1, L2, L3 based on reading.
        # collected names to sort/verify
        found_links = []
        
        for i in range(num_joints):
            info = p.getJointInfo(self.robotId, i)
            link_name = info[12].decode('utf-8')
            # URDF names: tibia_foot_r1, tibia_foot_l2, etc.
            if "tibia_foot" in link_name or "foot" in link_name:
                found_links.append((i, link_name))
        
        # Sort based on name to ensure consistence: R1, R2, R3, L1, L2, L3?
        # Usually standard is FL, FR, ML, MR, RL, RR or just matching index.
        # Let's just trust the URDF order if it looks grouped, or sort by name.
        # Let's sort by name for determinism.
        # r1, r2, r3, l1, l2, l3 -> l1, l2, l3, r1, r2, r3 if sorted alpha.
        # Let's try to keep standard counter-clockwise or similar if possible, 
        # but for "Pick a leg", index 0..5 just needs to be consistent.
        # Alpha sort: coxa_l1...
        # Let's sort alpha.
        found_links.sort(key=lambda x: x[1])
        tips = [x[0] for x in found_links]
        return tips

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed, options)
        
        # Choose target leg
        if self.target_leg_index_arg is not None:
            self.current_target_leg = self.target_leg_index_arg
        else:
            self.current_target_leg = np.random.randint(0, 6)
            
        # Add target info to obs
        return self._get_obs(), info

    def _get_obs(self):
        base_obs = super()._get_obs()
        
        # Create one-hot encoding for target leg
        one_hot = np.zeros(6, dtype=np.float32)
        one_hot[self.current_target_leg] = 1.0
        
        return np.concatenate([base_obs, one_hot])

    def step(self, action):
        return super().step(action)
        # Wait, super().step() calls _compute_reward() which uses the base class reward (stabilize).
        # We need to override _compute_reward logic. 
        # But super().step() logic is: apply action -> step sim -> get obs -> get reward.
        # So providing we override _compute_reward, it works.
        
    def _compute_reward(self):
        # 1. Get States
        pos, orn = p.getBasePositionAndOrientation(self.robotId)
        lin_vel, ang_vel = p.getBaseVelocity(self.robotId)
        euler = p.getEulerFromQuaternion(orn)
        
        # Get Tip Positions
        tip_states = p.getLinkStates(self.robotId, self.leg_tips)
        # tip_states[i][0] is Cartesian world pos (x,y,z)
        tip_positions = [s[0] for s in tip_states]
        
        target_tip_z = tip_positions[self.current_target_leg][2]
        
        # 2. Define Goals
        TARGET_LIFT_HEIGHT = 0.15 # Target Z for the lifting leg
        GROUND_THRESHOLD = 0.05   # Z below which is considered "down"
        
        # 3. Reward Components
        
        # A. Lift Reward (Target Leg)
        # Encourage getting close to target height
        lift_error = abs(target_tip_z - TARGET_LIFT_HEIGHT)
        r_lift = math.exp(-10.0 * lift_error**2)
        
        # B. Grounded Reward (Other Legs)
        # Encourage others to stay on ground
        ground_penalty = 0.0
        for i, tip_pos in enumerate(tip_positions):
            if i != self.current_target_leg:
                z = tip_pos[2]
                if z > GROUND_THRESHOLD:
                    ground_penalty += (z - GROUND_THRESHOLD) * 10.0
                    
        # C. Stability Reward (Base)
        r_upright = math.exp(-5.0 * (abs(euler[0]) + abs(euler[1])))
        r_height = math.exp(-50.0 * (pos[2] - 0.15)**2) # Keep body at consistent height
        
        # D. Energy/Drift Use
        p_drift = np.linalg.norm(lin_vel[:2]) * 1.0
        
        # E. Survival
        r_alive = 0.5
        
        # Total
        return reward

class MultiLegLiftEnv(LegLiftEnv):
    """
    Atomic Task 2/3: Lift multiple legs (Pairs or Tripods).
    Target: A binary mask of legs to lift.
    """
    def __init__(self, render_mode=None, max_steps=1000, target_mask=None):
        super().__init__(render_mode, max_steps)
        self.target_mask_arg = target_mask
        self.current_target_mask = np.zeros(6)
        
        # Obs: Base(49) + Mask(6) = 55
        # We can reuse the same obs shape/size as LegLiftEnv (55).
        
    def reset(self, seed=None, options=None):
        # We define masks for standard gaits
        # Tripod A: 0, 3, 4 (R1, L1, L2? Need to verify mapping)
        # Let's assume indices: 
        # R1(0), R2(1), R3(2)
        # L1(3), L2(4), L3(5)
        # Tripod 1: R1, R3, L2 -> 0, 2, 4
        # Tripod 2: R2, L1, L3 -> 1, 3, 5
        # Front Pair: R1, L1 -> 0, 3
        
        self.preset_masks = [
            np.array([1, 0, 1, 0, 1, 0]), # Tripod 1
            np.array([0, 1, 0, 1, 0, 1]), # Tripod 2
            np.array([1, 0, 0, 1, 0, 0]), # Front Pair
            np.array([0, 1, 0, 0, 1, 0]), # Mid Pair
            np.array([0, 0, 1, 0, 0, 1]), # Hind Pair
        ]
        
        obs, info = super(LegLiftEnv, self).reset(seed=seed) # Call Grandparent reset to skip LegLiftEnv's single leg logic
        
        if self.target_mask_arg is not None:
            self.current_target_mask = np.array(self.target_mask_arg)
        else:
            # Randomly select a preset
            idx = np.random.randint(0, len(self.preset_masks))
            self.current_target_mask = self.preset_masks[idx]
            
        return self._get_obs(), info

    def _get_obs(self):
        base_obs = super(LegLiftEnv, self)._get_obs()
        return np.concatenate([base_obs, self.current_target_mask])

    def _compute_reward(self):
        pos, orn = p.getBasePositionAndOrientation(self.robotId)
        lin_vel, ang_vel = p.getBaseVelocity(self.robotId)
        euler = p.getEulerFromQuaternion(orn)
        
        tip_states = p.getLinkStates(self.robotId, self.leg_tips)
        tip_z = [s[0][2] for s in tip_states]
        
        TARGET_HEIGHT = 0.15
        GROUND_THRESHOLD = 0.05
        
        lift_reward = 0.0
        ground_reward = 0.0
        
        for i in range(6):
            if self.current_target_mask[i] > 0.5:
                # Should Lift
                err = abs(tip_z[i] - TARGET_HEIGHT)
                lift_reward += math.exp(-10.0 * err**2)
            else:
                # Should Ground
                if tip_z[i] > GROUND_THRESHOLD:
                    ground_reward -= (tip_z[i] - GROUND_THRESHOLD) * 5.0
                    
        # Normalize lift reward by number of lifted legs
        num_lift = np.sum(self.current_target_mask)
        if num_lift > 0:
            lift_reward /= num_lift
            
        # Stability
        r_upright = math.exp(-5.0 * (abs(euler[0]) + abs(euler[1])))
        r_height = math.exp(-50.0 * (pos[2] - 0.15)**2)
        
        reward = (lift_reward * 3.0) + ground_reward + r_upright + r_height + 0.5
        return reward

class SequentialGaitEnv(MultiLegLiftEnv):
    """
    Atomic Task 4: Execute a static gait sequence.
    Phases:
    0: Lift Front Pair (R1, L1)
    1: Plant Front Pair
    2: Lift Mid Pair (R2, L2)
    3: Plant Mid Pair
    4: Lift Hind Pair (R3, L3)
    5: Plant Hind Pair
    """
    def __init__(self, render_mode=None, max_steps=2000):
        super().__init__(render_mode, max_steps)
        self.phase = 0
        self.phase_duration = 50 # steps per phase
        self.current_phase_step = 0
        
        # Mask definitions (R1=0, R2=1, R3=2, L1=3, L2=4, L3=5 assumption)
        # Update: We need to verify indices.
        # If atomic_envs sorted by name: coxa_l1, l2, l3, r1, r2, r3?
        # Let's assume indices 0-2 are R, 3-5 are L for now or vice versa.
        # Actually in `find_leg_tips` we sorted by name.
        # names: tibia_foot_l1, tibia_foot_l2, tibia_foot_l3, tibia_foot_r1...
        # So: 
        # 0: L1
        # 1: L2
        # 2: L3
        # 3: R1
        # 4: R2
        # 5: R3
        
        # Front Pair: L1(0) + R1(3)
        self.mask_front = np.array([1, 0, 0, 1, 0, 0])
        # Mid Pair: L2(1) + R2(4)
        self.mask_mid =   np.array([0, 1, 0, 0, 1, 0])
        # Hind Pair: L3(2) + R3(5)
        self.mask_hind =  np.array([0, 0, 1, 0, 0, 1])
        self.mask_none =  np.zeros(6)
        
    def reset(self, seed=None, options=None):
        self.phase = 0
        self.current_phase_step = 0
        return super().reset(seed, options)
        
    def step(self, action):
        # Update Phase
        self.current_phase_step += 1
        if self.current_phase_step >= self.phase_duration:
            self.phase = (self.phase + 1) % 6
            self.current_phase_step = 0
            
        # Set Target Mask based on phase
        if self.phase == 0:
            self.current_target_mask = self.mask_front
        elif self.phase == 2:
            self.current_target_mask = self.mask_mid
        elif self.phase == 4:
            self.current_target_mask = self.mask_hind
        else:
            self.current_target_mask = self.mask_none
            
        obs, reward, terminated, truncated, info = super(LegLiftEnv, self).step(action) # Call Grandparent step logic
        
        # Add Phase Reward?
        # The reward is computed by MultiLegLiftEnv._compute_reward which uses current_target_mask.
        # So we just need to ensure the mask is correct.
        
        # Add Bonus for Phase Completion / Progress?
        # Maybe just surviving and following the guide is enough.
        
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        # Base obs include mask. Maybe add phase index?
        # For simplicity, let's keep it same as MultiLegLift (Mask is the command).
        # The agent sees the mask changing, so it knows what to do.
        return super()._get_obs()
