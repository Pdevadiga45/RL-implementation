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

#  Project Overview: Reinforcement Learning Algorithm Comparison on Humanoid-v4 Model

***Objective***
The goal of this project is to evaluate the performance of three reinforcement learning (RL) algorithms—Soft Actor-Critic (SAC), Proximal Policy Optimization (PPO), and Deep Deterministic Policy Gradient (DDPG)—by training a Humanoid-v4 model in a Gymnasium environment. The aim is to understand each algorithm's efficiency and stability when applied to complex, high-dimensional tasks.

***Project Structure***
The project consists of:

**Algorithm Implementations**: Code for each RL algorithm, leveraging established libraries and custom callbacks.
Environment Setup: Details of the Humanoid-v4 Gymnasium environment, configured for tracking training metrics and episode lengths.
**Data Visualization**: Graphical analysis of key performance metrics, including reward trends, loss patterns, and training stability.

***Approach***

***Algorithm Selection***:
**SAC**: A stochastic off-policy actor-critic method focused on exploration and stability.
**PPO**: An on-policy, simpler yet robust algorithm for handling complex control problems.
**DDPG**: A deterministic off-policy algorithm known for continuous action spaces but more sensitive to hyperparameters.

***Training Pipeline***:
**Environment Setup**: Configured Humanoid-v4 environment to track episodic data and overall reward progression.
**Custom Callback Integration**: Used a custom callback to save metrics, ensuring consistency across episodes.
**Evaluation Metrics**: Performance is measured through reward patterns, loss trends, and episode length consistency.

***Visualization and Analysis***:

**Reward Plots**: Compared total rewards achieved across episodes for each algorithm.
**Loss Plots**: Focused on training stability through loss patterns.
**Episode Length Tracking**: Monitored the episode lengths to gauge convergence.

***Results***

**Each algorithm showed distinct characteristics**:

**SAC**: Demonstrated stability and steady reward improvement, effective at handling exploration-exploitation balance.
**PPO**: Displayed robustness with relatively smooth convergence, though slower than SAC in reaching peak performance.
**DDPG**: Encountered issues with instability and required fine-tuning, making it less consistent.

***Key Insights***
SAC’s effectiveness at balancing exploration and exploitation proved advantageous for the complex Humanoid-v4 task.
PPO’s robustness highlighted its value as a baseline algorithm, providing reliable, albeit slower, performance.
DDPG’s sensitivity revealed its limitations for environments requiring high adaptability without substantial tuning.
