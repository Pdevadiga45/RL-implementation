# Reinforcement Learning Implementation

***This project implements reinforcement learning using the Soft Actor-Critic (SAC) algorithm to train a model in the "Humanoid-v4" gym environment. Key features include:***

***Model Saving & Loading:*** The code checks for pre-trained models and resumes training from the last checkpoint, along with replay buffer restoration and training metrics tracking.

***Custom Callbacks & Logging:*** It uses a custom callback to log rewards and saves them to disk for later analysis. Training progress is logged using TensorBoard.

***Video Recording:*** A custom video recorder is used to capture agent behavior every 100 episodes.

***Training Visualization:*** The training process is visualized with a plot of cumulative rewards using matplotlib.

***Libraries used:***

gymnasium : for environment simulation

stable-baselines3 : for the SAC algorithm

torch : for GPU acceleration

matplotlib : for plotting training metrics

numpy : for data manipulation

TensorBoard : for real-time logging

# Comparative Analysis of Reinforcement Learning Algorithms in Gymnasium Environments

This project implements and evaluates three popular reinforcement learning algorithms — SAC, TD3, and DDPG — on the `Humanoid-v4` environment. By leveraging `gymnasium`, `stable-baselines3`, and custom callback functions, this project tracks and visualizes performance metrics to compare the sample efficiency and overall effectiveness of each algorithm.

## Project Setup

### Requirements
- `gymnasium`
- `stable-baselines3`
- `torch`
- `rliable`
- `matplotlib`

### Usage
1. Clone the repository.
2. Install the requirements: `pip install -r requirements.txt`
3. Run the training script: `python train_compare.py`
4. View logs and performance results in the `logs/` and `videos/` directories.

## Key Components

### Environment and Training
- **Custom Video Recording**: Automates video recording every 100 episodes for model performance review.
- **Custom Callback**: Logs rewards, episode lengths, and losses, saving them after each episode.

### Performance Evaluation
- **Sample Efficiency Curve**: Visualizes average rewards over timesteps.
- **Aggregate Metrics with Bootstrapped CIs**: Measures median, IQM, and mean performance.
- **Performance Profiles**: Tracks cumulative reward proportion across episodes.
- **Pairwise Probability of Improvement**: Compares each pair of algorithms to show improvement probabilities.

## Results
The results are saved as plots, comparing performance across the algorithms with metrics like sample efficiency and pairwise improvement probabilities.

## Acknowledgments
This project utilizes [rliable](https://github.com/google-research/rliable) for robust RL evaluations and is inspired by stable-baselines3 documentation and Gymnasium environments.
