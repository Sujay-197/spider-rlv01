import argparse
import os
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from atomic_envs import LegLiftEnv 

# Add parent to path for imports if needed
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TableLoggingCallback(BaseCallback):
    def __init__(self, verbose=0, log_freq=1000):
        super(TableLoggingCallback, self).__init__(verbose)
        self.log_freq = log_freq
        self.header_printed = False
        self.start_time = time.time()
        self.episode_rewards = []

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])

        if self.n_calls % self.log_freq == 0:
            if not self.header_printed:
                print(f"{'Step':>10} | {'Reward':>10} | {'Time (s)':>10}")
                print("-" * 36)
                self.header_printed = True
            
            avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
            elapsed_time = time.time() - self.start_time
            step_str = f"{self.n_calls/1000:.1f}k" if self.n_calls >= 1000 else str(self.n_calls)
            
            print(f"{step_str:>10} | {avg_reward:>10.2f} | {elapsed_time:>10.1f}")
            self.episode_rewards = [] 
        return True

def main():
    parser = argparse.ArgumentParser(description="Atomic Walking Tasks PPO")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "enjoy", "check"], help="Mode")
    parser.add_argument("--task", type=str, default="lift", choices=["lift", "multi", "gait"], help="Atomic Task")
    parser.add_argument("--leg", type=int, default=None, help="Target leg index (0-5) for enjoy mode or specific training")
    parser.add_argument("--timesteps", type=int, default=100000, help="Timesteps")
    parser.add_argument("--model", type=str, default=None, help="Model name")
    
    args = parser.parse_args()
    
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    model_name = args.model if args.model else f"atomic_{args.task}"
    model_path = os.path.join(models_dir, model_name)
    
    # Environment Setup
    if args.task == "lift":
        env = LegLiftEnv(render_mode="human" if args.mode == "enjoy" else None, 
                         target_leg_index=args.leg)
    elif args.task == "multi":
        from atomic_envs import MultiLegLiftEnv
        env = MultiLegLiftEnv(render_mode="human" if args.mode == "enjoy" else None)
    elif args.task == "gait":
        from atomic_envs import SequentialGaitEnv
        env = SequentialGaitEnv(render_mode="human" if args.mode == "enjoy" else None)
    else:
        raise ValueError(f"Unknown task: {args.task}")
    
    if args.mode == "check":
        from stable_baselines3.common.env_checker import check_env
        print("Checking environment...")
        check_env(env)
        print("Check passed!")
        
    elif args.mode == "train":
        print(f"Training {args.task} for {args.timesteps} steps...")
        from stable_baselines3.common.monitor import Monitor
        env = Monitor(env)
        
        model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=logs_dir)
        callback = TableLoggingCallback(log_freq=5000)
        
        try:
            model.learn(total_timesteps=args.timesteps, callback=callback)
            model.save(model_path)
            print(f"Saved to {model_path}")
        except KeyboardInterrupt:
            model.save(model_path)
            print("Interrupted. Saved.")
            
    elif args.mode == "enjoy":
        print(f"Loading {model_path}...")
        if not os.path.exists(model_path + ".zip"):
            print("Model not found.")
            return

        model = PPO.load(model_path)
        obs, _ = env.reset()
        
        print(f"Enjoying Task: {args.task}. Target Leg: {env.current_target_leg}")
        print("Press Ctrl+C to stop.")
        try:
            while True:
                # Allow dynamic leg switching in enjoy loop if possible? 
                # Not easily without modifying loop to read keyboard. 
                # For now just run.
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    obs, _ = env.reset()
                    if args.leg is None:
                        print(f"New Target Leg: {env.current_target_leg}")
                    time.sleep(1.0)
                time.sleep(1./240.)
        except KeyboardInterrupt:
            pass
            
    env.close()

if __name__ == "__main__":
    main()
