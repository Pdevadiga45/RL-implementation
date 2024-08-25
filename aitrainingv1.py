import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import RecordVideo
import torch

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

# Create the environment
env = gym.make('Humanoid-v4', render_mode="rgb_array")

# Wrap the environment with RecordVideo
env = RecordVideo(env, video_folder='./video2', episode_trigger=lambda episode_id: episode_id % 100 == 0)

# Initialize the SAC model
device = torch.device("cuda")
policy_kwargs = dict(net_arch=[400, 300])
model = SAC('MlpPolicy', env, verbose=1, learning_rate=3e-4, batch_size=256, 
            buffer_size=1000000, tau=0.005, learning_starts=10000, ent_coef='auto_0.5', 
            policy_kwargs=policy_kwargs, device=device)

# Configure TensorBoard logging
from stable_baselines3.common.logger import configure
log_path = './logs2/tensorboard'
configure(log_path, ["stdout", "tensorboard"])

# Create custom callback to log rewards
custom_callback = CustomCallback(log_dir='./logs2')

# Train the model
model.learn(total_timesteps=500000, callback=custom_callback)  # 1 million timesteps

# Save the trained model
model.save('humanoid_walk_model')

# Load the trained model
model = SAC.load('humanoid_walk_model')

# Test the trained model and render video
obs, _ = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()  # Render the environment to visualize the agent's behavior
    if terminated or truncated:
        obs, _ = env.reset()

env.close()  # Close the environment and save the video

# Load and plot the training metrics
rewards = np.load('./logs2/rewards.npy')
plt.plot(np.cumsum(rewards))
plt.xlabel('Timesteps')
plt.ylabel('Cumulative Reward')
plt.title('Training Progress')
plt.show()
