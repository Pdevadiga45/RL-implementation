#saves and loads existing model and logs


import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import RecordVideo
import torch
import os
from datetime import datetime

class CustomCallback(BaseCallback):
    def __init__(self, log_dir, eval_freq=10000, verbose=1):
        super(CustomCallback, self).__init__(verbose)
        self.eval_freq = eval_freq
        self.log_dir = log_dir
        self.rewards = []

    def _on_step(self) -> bool:
        # Log the reward for each step
        self.rewards.append(self.locals['rewards'])
        return True

    def _on_rollout_end(self) -> None:
        # Save the rewards to a file for later analysis
        np.save(self.log_dir + '/rewards.npy', self.rewards)

# Create a custom video recorder wrapper
class CustomRecordVideo(RecordVideo):
    def __init__(self, env, video_folder, name_prefix="rl-video"):
        super().__init__(env, video_folder, episode_trigger=self.trigger_video_recording, name_prefix=name_prefix)
        self.current_episode = 0

    def trigger_video_recording(self, episode_id):
        return episode_id % 100 == 0

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.current_episode = getattr(model, '_episode_num', self.current_episode)
        if self.video_recorder:
            self.video_recorder.path = os.path.join(self.video_folder, f"{self.name_prefix}-episode-{self.current_episode}.mp4")
        return observations

# Create the environment
env = gym.make('Humanoid-v4', render_mode="rgb_array")
env = CustomRecordVideo(env, video_folder='./video')

# Initialize the SAC model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_kwargs = dict(net_arch=[400, 300])

# Check for existing model and logs
model_dir = './models'
log_dir = './logs'
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

existing_models = [f for f in os.listdir(model_dir) if f.endswith('.zip')]
if existing_models:
    latest_model = max(existing_models, key=lambda f: os.path.getmtime(os.path.join(model_dir, f)))
    model_path = os.path.join(model_dir, latest_model)
    print(f"Loading existing model: {model_path}")
    model = SAC.load(model_path, env=env, device=device)
    
    # Load additional training info
    info_file = f"training_info_{latest_model.split('_')[-1].split('.')[0]}.npy"
    if os.path.exists(os.path.join(model_dir, info_file)):
        training_info = np.load(os.path.join(model_dir, info_file), allow_pickle=True).item()
        model.num_timesteps = training_info['total_timesteps']
        model._episode_num = training_info['episodes']
        print(f"Resuming from: Episodes: {model._episode_num}, Timesteps: {model.num_timesteps}")
    
    # Ensure the replay buffer is updated
    model.replay_buffer.pos = model.num_timesteps % model.replay_buffer.buffer_size
    model.replay_buffer.full = model.num_timesteps >= model.replay_buffer.buffer_size
    
    # Find the corresponding log directory
    model_timestamp = datetime.fromtimestamp(os.path.getmtime(model_path)).strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"SAC_{model_timestamp}")
else:
    print("No existing model found. Creating a new model.")
    model = SAC('MlpPolicy', env, verbose=1, learning_rate=3e-4, batch_size=256, 
                buffer_size=1000000, tau=0.005, ent_coef='auto_0.5',  # Note: float, not string
                policy_kwargs=policy_kwargs, device=device)
    
    # Create a new log directory
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"SAC_{current_time}")

# Configure TensorBoard logging
from stable_baselines3.common.logger import configure
configure(log_path, ["stdout", "tensorboard"])

# Create custom callback to log rewards
custom_callback = CustomCallback(log_dir=log_path)

# Train the model
try:
    model.learn(total_timesteps=1000000, callback=custom_callback, reset_num_timesteps=False)
except KeyboardInterrupt:
    print("Training interrupted. Saving model and logs...")
finally:
    # Save the model and logs
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(model_dir, f"humanoid_walk_model_{current_time}.zip")
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Save additional training info
    training_info = {
        'total_timesteps': model.num_timesteps,
        'episodes': model._episode_num
    }
    np.save(os.path.join(model_dir, f"training_info_{current_time}.npy"), training_info)

# Test the trained model and render video
obs, _ = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, _ = env.reset()

env.close()

# Load and plot the training metrics
rewards = np.load(os.path.join(log_path, 'rewards.npy'))
plt.plot(np.cumsum(rewards))
plt.xlabel('Timesteps')
plt.ylabel('Cumulative Reward')
plt.title('Training Progress')
plt.show()
