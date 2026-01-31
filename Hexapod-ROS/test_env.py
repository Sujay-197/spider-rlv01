import pybullet as p
import pybullet_data
import time
import math

def main():
    # 1. Connect to PyBullet
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # 2. Setup World
    p.setGravity(0, 0, -9.81)
    planeId = p.loadURDF("plane.urdf")
    
    # 3. Load Hexapod
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(script_dir, "crab_description", "hexapod_pybullet.urdf")
    startPos = [0, 0, 0.2] # Drop from a small height
    startId = p.loadURDF(urdf_path, startPos, useFixedBase=False)
    
    # 4. Analyze Joints for RL
    print("\nXXX Joint Info XXX")
    num_joints = p.getNumJoints(startId)
    joint_indices = []
    
    # Dictionary to store debug parameter IDs
    debug_params = {}

    for i in range(num_joints):
        info = p.getJointInfo(startId, i)
        joint_name = info[1].decode("utf-8")
        joint_type = info[2]
        
        # Revolute joints (type 0) are actuated
        if joint_type == p.JOINT_REVOLUTE:
            print(f"Index: {i}, Name: {joint_name}, Limits: [{info[8]}, {info[9]}]")
            joint_indices.append(i)
            # Add slider for this joint
            debug_params[i] = p.addUserDebugParameter(joint_name, info[8], info[9], 0)
            
            # Enable motor control for this joint (position control)
            p.setJointMotorControl2(startId, i, p.POSITION_CONTROL, force=100)

    print(f"Total Actuated Joints: {len(joint_indices)}")

    # 5. Simulation Loop
    print("\nStarting Simulation... Press Ctrl+C to stop.")
    try:
        while p.isConnected():
            # Read sliders and apply positions
            for joint_idx, param_id in debug_params.items():
                target_pos = p.readUserDebugParameter(param_id)
                p.setJointMotorControl2(startId, joint_idx, p.POSITION_CONTROL, targetPosition=target_pos)
            
            p.stepSimulation()
            time.sleep(1./240.)
            
            # Optional: Camera follow
            # pos, orn = p.getBasePositionAndOrientation(startId)
            # p.resetDebugVisualizerCamera(1.5, 50, -35, pos)
            
    except KeyboardInterrupt:
        p.disconnect()

if __name__ == "__main__":
    main()
