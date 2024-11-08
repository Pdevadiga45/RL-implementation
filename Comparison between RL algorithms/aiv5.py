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
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils

class CustomCallback(BaseCallback):
    def __init__(self, log_dir, eval_freq=5000, verbose=1):
        super(CustomCallback, self).__init__(verbose)
        self.eval_freq = eval_freq
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.current_episode_length = 0
        self.current_episode_reward = 0
        self.current_episode_losses = []

    def _on_step(self) -> bool:
        reward = self.locals.get('rewards', [0])[0]
        self.current_episode_reward += reward
        terminated = self.locals.get('dones', [False])[0]
        self.current_episode_length += 1

        if isinstance(self.model, (SAC, TD3, DDPG)):
            actor_loss = self.model.logger.name_to_value.get('train/actor_loss', 0)
            critic_loss = self.model.logger.name_to_value.get('train/critic_loss', 0)
            total_loss = actor_loss + critic_loss
            self.current_episode_losses.append(total_loss)

        if terminated:
            self.episode_lengths.append(self.current_episode_length)
            self.episode_rewards.append(self.current_episode_reward)
            self.losses.append(np.mean(self.current_episode_losses))  # Store average loss for the episode
            self.current_episode_length = 0
            self.current_episode_reward = 0
            self.current_episode_losses = []  # Reset for the next episode

        return True

    def _on_rollout_end(self) -> None:
        np.save(os.path.join(self.log_dir, 'rewards.npy'), self.episode_rewards)
        np.save(os.path.join(self.log_dir, 'episode_rewards.npy'), self.episode_rewards)
        np.save(os.path.join(self.log_dir, 'episode_lengths.npy'), self.episode_lengths)
        np.save(os.path.join(self.log_dir, 'losses.npy'), self.losses)

class CustomRecordVideo(RecordVideo):
    def __init__(self, env, video_folder, run_number, name_prefix="rl-video"):
        super().__init__(env, video_folder, episode_trigger=self.trigger_video_recording, name_prefix=name_prefix)
        self.current_episode = 0
        self.run_number = run_number

    def trigger_video_recording(self, episode_id):
        return episode_id % 100 == 0

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.current_episode += 1
        if self.video_recorder:
            self.video_recorder.path = os.path.join(
                self.video_folder,
                f"{self.name_prefix}-run-{self.run_number}-episode-{self.current_episode}.mp4"
            )
        return observations

def make_env(video_folder, algo_name, run_number):
    env = gym.make('Humanoid-v4', render_mode="rgb_array")
    env = CustomRecordVideo(env, video_folder=os.path.join(video_folder, algo_name), run_number=run_number)
    return env

def save_model_and_info(model, model_dir, current_time, algo_name):
    model_save_path = os.path.join(model_dir, f"{algo_name}_model_{current_time}.zip")
    model.save(model_save_path)

def train_model(algo_name, algo_class, env, log_dir, model_dir, total_steps=100000):
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
    else:
        model = algo_class('MlpPolicy', env, verbose=1, device=device, buffer_size=50000)

    try:
        if steps_completed < total_steps:
            model.learn(total_timesteps=total_steps - steps_completed, callback=custom_callback, reset_num_timesteps=False)
            steps_completed = model.num_timesteps
        save_model_and_info(model, model_dir, current_time, algo_name)
    except KeyboardInterrupt:
        model.save(model_save_path)
    finally:
        env.close()

    return log_path

# Loop through each algorithm
log_paths = []
desired_steps = 100000
num_runs = 5

algorithms = {'SAC': SAC, 'TD3': TD3, 'DDPG': DDPG}

model_dir = './models'
log_dir = './logs'
video_folder = './videos'
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(video_folder, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dictionary to store rewards and losses from each run for analysis
results = {algo: [] for algo in algorithms.keys()}
loss_results = {algo: [] for algo in algorithms.keys()}

for algo_name, AlgoClass in algorithms.items():
    for run in range(num_runs):
        print(f"Training {algo_name} - Run {run + 1}...")
        env = make_env(video_folder, algo_name, run + 1)
        log_path = train_model(algo_name, AlgoClass, env, log_dir, model_dir, total_steps=desired_steps)

        # Load episode rewards and losses for analysis
        try:
            episode_rewards = np.load(os.path.join(log_path, 'episode_rewards.npy'))
            episode_losses = np.load(os.path.join(log_path, 'losses.npy'))
            results[algo_name].append(episode_rewards)
            loss_results[algo_name].append(episode_losses)
        except FileNotFoundError as e:
            print(f"Warning: {e}. Skipping log path: {log_path}")

        env.close()

# Truncate episode rewards to the length of the shortest run for each algorithm
for algo_name in results.keys():
    # Find the minimum length of episode rewards across all runs
    min_length = min(len(rewards) for rewards in results[algo_name])
    
    # Truncate all runs to the minimum length
    results[algo_name] = [rewards[:min_length] for rewards in results[algo_name]]

# Sample Efficiency Curve
plt.figure(figsize=(12, 8))

for algo_name, rewards in results.items():
    mean_rewards = np.mean(np.array(rewards), axis=0)
    plt.plot(mean_rewards, label=algo_name)

plt.title("Sample Efficiency Curve")
plt.xlabel("Timesteps")
plt.ylabel("Average Reward")
plt.legend()
plt.grid()
plt.show()

# Aggregate Metrics with 95% Stratified Bootstrap CIs
def aggregate_func(x):
    return np.array([
        metrics.aggregate_median(x),
        metrics.aggregate_iqm(x),
        metrics.aggregate_mean(x)
    ])

# Prepare data for aggregation
rewards_data = {algo: np.array(rewards) for algo, rewards in results.items()}

# Compute aggregate scores and confidence intervals
aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
    rewards_data, aggregate_func, reps=50000
)

# Generate Plot with Consistent Labeling
fig, axes = plot_utils.plot_interval_estimates(
    aggregate_scores, aggregate_score_cis,
    metric_names=['Median', 'IQM', 'Mean'],
    algorithms=list(rewards_data.keys()),
    xlabel='Human Normalized Score'
)

# Ensure Title and Layout are Set Properly
fig.suptitle("Aggregate Metrics with 95% Bootstrap CIs", fontsize=16)

# Adjust the layout to provide more space for the x-label
plt.subplots_adjust(bottom=0.15)  # Increase bottom margin

plt.show()

# Performance Profiles
performance_profiles = {algo: np.cumsum(np.array(rewards)) / np.sum(np.array(rewards)) for algo, rewards in results.items()}

plt.figure(figsize=(10, 6))
for algo, profile in performance_profiles.items():
    plt.plot(profile, label=algo)

plt.title("Performance Profiles")
plt.xlabel("Episodes")
plt.ylabel("Cumulative Reward Proportion")
plt.legend()
plt.grid()
plt.show()

# Pairwise Probability of Improvement Calculation
pairwise_probabilities = {
    ('SAC', 'TD3'): 0,
    ('SAC', 'DDPG'): 0,
    ('TD3', 'DDPG'): 0
}

# Calculate pairwise probabilities
for (algo1, algo2) in pairwise_probabilities.keys():
    # Find the minimum length of rewards across all runs for both algorithms
    min_length_algo1 = min(len(rewards) for rewards in results[algo1])
    min_length_algo2 = min(len(rewards) for rewards in results[algo2])
    
    # Truncate rewards to the minimum length for comparison
    min_length = min(min_length_algo1, min_length_algo2)  # Find the overall minimum length
    rewards_algo1 = np.array([rewards[:min_length] for rewards in results[algo1]])
    rewards_algo2 = np.array([rewards[:min_length] for rewards in results[algo2]])

    # Calculate the probability that algo1 > algo2 across runs
    improvement_count = 0
    for i in range(len(rewards_algo1)):  # Iterate through each run
        improvement_count += np.mean(rewards_algo1[i] > rewards_algo2[i])
    
    # Store the probability
    pairwise_probabilities[(algo1, algo2)] = improvement_count / len(rewards_algo1)

# Plot Pairwise Probability of Improvement
fig, ax = plt.subplots(figsize=(8, 6))
comparison_labels = [f"{algo1} > {algo2}" for (algo1, algo2) in pairwise_probabilities.keys()]
probabilities = list(pairwise_probabilities.values())
ax.bar(comparison_labels, probabilities, color=['blue', 'orange', 'green'])

# Labels and formatting
ax.set_ylabel("Probability of Improvement")
ax.set_title("Pairwise Probability of Improvement among Algorithms")
ax.set_ylim(0, 1)
plt.grid(axis='y')

plt.tight_layout()
plt.show()
