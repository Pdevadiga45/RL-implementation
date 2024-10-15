import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, TD3, DDPG
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import RecordVideo
import torch
import os
from datetime import datetime
from stable_baselines3.common.logger import configure
import json  # Import json for saving completed steps


class CustomCallback(BaseCallback):
    def __init__(self, log_dir, eval_freq=10000, verbose=1):
        super(CustomCallback, self).__init__(verbose)
        self.eval_freq = eval_freq
        self.log_dir = log_dir
        self.rewards = []
        self.episode_lengths = []
        self.losses = []

    def _on_step(self) -> bool:
        self.rewards.append(self.locals.get('rewards', 0))
        n_steps = self.locals.get('n_steps', 0)
        self.episode_lengths.append(n_steps)

        if isinstance(self.model, (SAC, TD3, DDPG)):
            actor_loss = self.model.logger.name_to_value.get('train/actor_loss', 0)
            critic_loss = self.model.logger.name_to_value.get('train/critic_loss', 0)
            total_loss = actor_loss + critic_loss
            self.losses.append(total_loss)

        return True

    def _on_rollout_end(self) -> None:
        np.save(os.path.join(self.log_dir, 'rewards.npy'), self.rewards)
        np.save(os.path.join(self.log_dir, 'episode_lengths.npy'), self.episode_lengths)
        np.save(os.path.join(self.log_dir, 'losses.npy'), self.losses)


class CustomHumanoidEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.step_count = 0
        self.episode_count = 0

    def step(self, action):
        self.step_count += 1
        observation, reward, terminated, truncated, info = self.env.step(action)

        forward_reward = info.get('reward_forward', 0)
        ctrl_cost = info.get('reward_ctrl', 0)
        contact_cost = info.get('reward_contact', 0)
        healthy_reward = info.get('reward_alive', 0)

        modified_reward = 2 * forward_reward - 0.1 * ctrl_cost - 0.5 * contact_cost + 0.1 * healthy_reward
        upright_reward = 0.1 * (observation[0] - 1.0)**2
        modified_reward += upright_reward

        return observation, modified_reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.step_count = 0
        self.episode_count += 1
        return self.env.reset(**kwargs)


class CustomRecordVideo(RecordVideo):
    def __init__(self, env, video_folder, name_prefix="rl-video"):
        super().__init__(env, video_folder, episode_trigger=self.trigger_video_recording, name_prefix=name_prefix)
        self.current_episode = 0

    def trigger_video_recording(self, episode_id):
        # Change the condition if you want a different frequency for recording videos
        return episode_id % 100 == 0

    def reset(self, **kwargs):
        # Call the reset of the parent class
        observations = super().reset(**kwargs)

        # Increment the episode count and update the video recorder path
        self.current_episode += 1
        if self.video_recorder:
            self.video_recorder.path = os.path.join(self.video_folder, f"{self.name_prefix}-episode-{self.current_episode}.mp4")
        return observations


def make_env(video_folder, algo_name):
    env = gym.make('Humanoid-v4', render_mode="rgb_array")
    env = CustomHumanoidEnv(env)
    env = CustomRecordVideo(env, video_folder=os.path.join(video_folder, algo_name))
    return env


def save_model_and_info(model, model_dir, current_time, algo_name):
    model_save_path = os.path.join(model_dir, f"{algo_name}_model_{current_time}.zip")
    model.save(model_save_path)
    print(f"{algo_name} model saved to {model_save_path}")
    training_info = {
        'total_timesteps': model.num_timesteps,
        'episodes': model._episode_num
    }
    np.save(os.path.join(model_dir, f"training_info_{algo_name}_{current_time}.npy"), training_info)


def train_model(algo_name, algo_class, env, log_dir, model_dir, total_steps=10000):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{algo_name}_{current_time}")  
    os.makedirs(log_path, exist_ok=True)

    configure(log_path, ["stdout", "tensorboard"])
    custom_callback = CustomCallback(log_dir=log_path)

    model_save_path = os.path.join(model_dir, f"{algo_name}_model.zip")
    steps_completed = 0

    if os.path.exists(model_save_path):
        model = algo_class.load(model_save_path, env=env)
        steps_completed = model.num_timesteps
        print(f"Resuming {algo_name} training from {steps_completed} steps...")
    else:
        model = algo_class('MlpPolicy', env, verbose=1, device=device)

    try:
        if steps_completed < total_steps:
            model.learn(total_timesteps=total_steps - steps_completed, callback=custom_callback, reset_num_timesteps=False)
            steps_completed = model.num_timesteps
        save_model_and_info(model, model_dir, current_time, algo_name)
    except KeyboardInterrupt:
        print("Training interrupted. Saving model and logs...")
        model.save(model_save_path)
    finally:
        env.close()

    return log_path, steps_completed


# Loop through each algorithm
log_paths = []
desired_steps = 10000

algorithms = {'SAC': SAC, 'TD3': TD3, 'DDPG': DDPG}

model_dir = './models'
log_dir = './logs'
video_folder = './videos'
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(video_folder, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dictionary to track completed steps for each algorithm
completed_steps = {
    'SAC': 0,
    'TD3': 0,
    'DDPG': 0
}
# Loop through each algorithm
for algo_name, AlgoClass in algorithms.items():
    print(f"Training with {algo_name}...")

    # Check if the previous algorithm has completed the desired steps
    if algo_name == 'TD3' and completed_steps['SAC'] < desired_steps:
        print(f"SAC has not completed {desired_steps} steps yet. Current steps: {completed_steps['SAC']}.")
        break  # Break the loop if SAC has not completed

    if algo_name == 'DDPG' and completed_steps['TD3'] < desired_steps:
        print(f"TD3 has not completed {desired_steps} steps yet. Current steps: {completed_steps['TD3']}.")
        break  # Break the loop if TD3 has not completed

    # Create environment
    env = make_env(video_folder, algo_name)

    # Initialize model
    model = AlgoClass('MlpPolicy', env, verbose=1, device=device)

    # Train the model and get the number of completed steps
    log_path, completed_steps[algo_name] = train_model(algo_name, AlgoClass, env, log_dir, model_dir, total_steps=desired_steps)

    # Check if completed steps meet the desired amount
    if completed_steps[algo_name] < desired_steps:
        print(f"{algo_name} training completed with {completed_steps[algo_name]}/{desired_steps} steps. Moving to the next algorithm.")
        continue  # Continue to the next algorithm

    log_paths.append(log_path)

    # Close the environment after training
    env.close()

# After training all models, load logs and plot comparison graphs
rewards_data = []
episode_lengths_data = []
losses_data = []

# Only process log paths for algorithms that completed the desired steps
for log_path in log_paths:
    rewards = np.load(os.path.join(log_path, 'rewards.npy'))
    episode_lengths = np.load(os.path.join(log_path, 'episode_lengths.npy'))
    losses = np.load(os.path.join(log_path, 'losses.npy'))
    rewards_data.append(rewards)
    episode_lengths_data.append(episode_lengths)
    losses_data.append(losses)

# Check if rewards_data is not empty before plotting rewards and losses
if rewards_data:
    # Plot rewards and loss comparison on one graph
    plt.figure(figsize=(12, 6))

    for i, algo_name in enumerate(algorithms.keys()):
        if i < len(rewards_data):  # Ensure we don't go out of bounds
            normalized_rewards = (np.cumsum(rewards_data[i]) - np.min(np.cumsum(rewards_data[i]))) / (np.max(np.cumsum(rewards_data[i])) - np.min(np.cumsum(rewards_data[i])))
            plt.plot(normalized_rewards, label=f'{algo_name} Rewards', linestyle='--')
            plt.plot(np.arange(len(losses_data[i])), losses_data[i], label=f'{algo_name} Loss')

    plt.xlabel('Training Steps')
    plt.ylabel('Rewards/Loss')
    plt.title('Rewards and Loss Comparison Across Algorithms')
    plt.legend()
    plt.show()

# Check if episode_lengths_data is not empty before plotting episode lengths
if episode_lengths_data:
    # Plot episode length comparison in a separate graph
    plt.figure(figsize=(12, 6))

    for i, algo_name in enumerate(algorithms.keys()):
        if i < len(episode_lengths_data):  # Ensure we don't go out of bounds
            plt.plot(np.cumsum(episode_lengths_data[i]), label=f'{algo_name} Episode Length')

    plt.xlabel('Training Steps')
    plt.ylabel('Episode Length')
    plt.title('Episode Length Comparison Across Algorithms')
    plt.legend()
    plt.show()

else:
    print("No episode lengths data to plot.")
