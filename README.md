# Reinforcement Learning Implementation

This project implements reinforcement learning using the Soft Actor-Critic (SAC) algorithm to train a model in the "Humanoid-v4" gym environment. Key features include:

Model Saving & Loading: The code checks for pre-trained models and resumes training from the last checkpoint, along with replay buffer restoration and training metrics tracking.
Custom Callbacks & Logging: It uses a custom callback to log rewards and saves them to disk for later analysis. Training progress is logged using TensorBoard.
Video Recording: A custom video recorder is used to capture agent behavior every 100 episodes.
Training Visualization: The training process is visualized with a plot of cumulative rewards using matplotlib.
Libraries used:

gymnasium : for environment simulation
stable-baselines3 : for the SAC algorithm
torch : for GPU acceleration
matplotlib : for plotting training metrics
numpy : for data manipulation
TensorBoard : for real-time logging
