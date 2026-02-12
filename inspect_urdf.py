import pybullet as p
import pybullet_data
import os

def inspect_robot():
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = current_dir # Running from project root ideally
    urdf_path = os.path.join(project_root, "Hexapod-ROS", "crab_description", "hexapod_pybullet.urdf")
    
    print(f"Loading URDF from: {urdf_path}")
    try:
        robotId = p.loadURDF(urdf_path, [0, 0, 0.2])
    except Exception as e:
        print(f"Failed to load URDF: {e}")
        return

    num_joints = p.getNumJoints(robotId)
    print(f"Number of Joints: {num_joints}")
    
    print("\n--- Joint / Link Info ---")
    print(f"{'Idx':<5} | {'Joint Name':<30} | {'Link Name':<30}")
    print("-" * 70)
    
    leg_tips = []
    
    for i in range(num_joints):
        info = p.getJointInfo(robotId, i)
        joint_name = info[1].decode('utf-8')
        link_name = info[12].decode('utf-8')
        
        print(f"{i:<5} | {joint_name:<30} | {link_name:<30}")
        
        # Heuristic to find tips: usually tibias or foot or has 'tip' or 'foot' in name
        # Or it's the 3rd link in a leg chain.
        if "tibia" in link_name or "foot" in link_name or "tip" in link_name:
            leg_tips.append((i, link_name))

    print("\n--- Detected Potential Leg Tips ---")
    for idx, name in leg_tips:
        print(f"Link Index: {idx}, Name: {name}")

    p.disconnect()

if __name__ == "__main__":
    inspect_robot()
