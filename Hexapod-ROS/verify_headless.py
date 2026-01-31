import pybullet as p
import pybullet_data
import time
import numpy as np

def test_headless():
    # DIRECT mode for headless
    physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    planeId = p.loadURDF("plane.urdf")
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(script_dir, "crab_description", "hexapod_pybullet.urdf")
    try:
        startId = p.loadURDF(urdf_path, [0, 0, 0.2], useFixedBase=False)
        print("URDF Loaded Successfully")
    except Exception as e:
        print(f"Failed to load URDF: {e}")
        return

    # Check number of joints
    num_joints = p.getNumJoints(startId)
    print(f"Number of joints: {num_joints}")
    
    # Run simulation for 2 seconds (480 steps)
    print("Stepping simulation...")
    for i in range(480):
        p.stepSimulation()
        
    # Check final position
    pos, orn = p.getBasePositionAndOrientation(startId)
    print(f"Final Base Position: {pos}")
    
    # Check if exploded (z > 1.0 or z < -0.1 or x,y extremely large)
    if pos[2] > 1.0 or pos[2] < 0.0: # Should be resting on ground approx 0.05-0.1m
        print("STABILITY CHECK FAILED: Robot is not at expected height.")
        print(f"Height: {pos[2]}")
    elif np.abs(pos[0]) > 1.0 or np.abs(pos[1]) > 1.0:
        print("STABILITY CHECK FAILED: Robot drifted too far.")
    else:
        print("STABILITY CHECK PASSED: Robot is stable.")

    # List joints for RL map
    print("\nXXX Joint Map XXX")
    actuated_count = 0
    for i in range(num_joints):
        info = p.getJointInfo(startId, i)
        if info[2] == p.JOINT_REVOLUTE:
            print(f"{i}: {info[1].decode('utf-8')}")
            actuated_count += 1
    print(f"Total Actuated: {actuated_count}")

    p.disconnect()

if __name__ == "__main__":
    test_headless()
