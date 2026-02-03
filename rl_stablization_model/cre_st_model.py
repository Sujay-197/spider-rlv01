import argparse
import os
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from envs import HexapodStabilizeEnv
from walking_env import HexapodWalkEnv

'''
python rl_stablization_model/cre_st_model.py --mode train --timesteps k
python rl_stablization_model/cre_st_model.py --mode enjoy --model_name str
python rl_stablization_model/cre_st_model.py --mode check
'''

class TableLoggingCallback(BaseCallback):
    def __init__(self, verbose=0, log_freq=1000):
        super(TableLoggingCallback, self).__init__(verbose)
        self.log_freq = log_freq
        self.header_printed = False
        self.start_time = time.time()
        self.episode_rewards = [] # Buffer to store rewards

    def _on_step(self) -> bool:
        # Collect rewards continuously
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])

        if self.n_calls % self.log_freq == 0:
            if not self.header_printed:
                print(f"{'Step':>10} | {'Reward':>10} | {'Loss':>10} | {'Time (s)':>10}")
                print("-" * 50)
                self.header_printed = True
            
            avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
            
            elapsed_time = time.time() - self.start_time
            print(f"{self.n_calls:>10} | {avg_reward:>10.2f} | {'N/A':>10} | {elapsed_time:>10.1f}")
            
            # Clear buffer after logging (optional: keep windowed average)
            # For this request "epoch batch returns table", clearing makes sense to show progress specific to that batch.
            self.episode_rewards = [] 
            
        return True

def main():
    parser = argparse.ArgumentParser(description="Hexapod PPO RL")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "enjoy", "check"], help="train, enjoy, or check env")
    parser.add_argument("--task", type=str, default="stabilize", choices=["stabilize", "walk"], help="Task to perform: stabilize or walk")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps for training")
    parser.add_argument("--model_name", type=str, default=None, help="Name of model to save/load (optional override)")
    
    args = parser.parse_args()
    
    # Determine model name based on task if not provided
    if args.model_name is None:
        args.model_name = f"hexapod_ppo_{args.task}"
    
    models_dir = "models"
    log_dir = "logs"
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    model_path = os.path.join(models_dir, args.model_name)
    
    # Select Environment Class
    if args.task == "walk":
        EnvClass = HexapodWalkEnv
    else:
        EnvClass = HexapodStabilizeEnv
    
    if args.mode == "check":
        print(f"Checking environment for task: {args.task}...")
        env = EnvClass(render_mode=None)
        check_env(env)
        print("Environment check passed!")
        
    elif args.mode == "train":
        print(f"Starting training for {args.task} with {args.timesteps} timesteps...")
        # Wrap environment with Monitor to record episode statistics for reward logging
        from stable_baselines3.common.monitor import Monitor
        env = EnvClass(render_mode=None)
        env = Monitor(env)
        
        # Load existing model if available to continue training (optional, but good for phase 2 starting from scratch or stabilized?)
        # For now, we start fresh for walking or maybe load stabilized weights?
        # Typically PPO struggles to adapt a standing policy to walking without careful curriculum. 
        # Let's start fresh for walking as per plan.
        
        model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=log_dir)
        
        # Add start_time for logging
        model.start_time = time.time()
        callback = TableLoggingCallback(log_freq=5000)
        
        try:
            model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=False)
            model.save(model_path)
            print(f"\nModel saved to {model_path}")
        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving current model...")
            model.save(model_path)
            print("Model saved.")
            
    elif args.mode == "enjoy":
        print(f"Loading model from {model_path}...")
        if not os.path.exists(model_path + ".zip"):
            print(f"Error: Model not found at {model_path}.zip. Train first!")
            return
            
        # Increase max_steps for enjoy mode to avoid frequent resets
        env = EnvClass(render_mode="human", max_steps=50000)
        model = PPO.load(model_path)
        
        obs, info = env.reset()
        print("Starting Enjoy Mode (Ctrl+C to stop)...")
        try:
            while True:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    obs, info = env.reset()
                    time.sleep(1.0)
                    
        except KeyboardInterrupt:
            print("Stopping...")
            
    if 'env' in locals():
        env.close()

if __name__ == "__main__":
    main()
